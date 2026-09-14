#include "chat_tools.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <re2/re2.h>
#include <cstring>
#include <set>
#include <sstream>
#include <utility>

namespace sparkinfer_server {
namespace {

using json = nlohmann::json;

// Must match runtime/src/models/qwen35.cpp's Impl::kMaxLogitBiasEntries -- that's the real
// scratch-buffer cap a scatter can hold; this is where the request-facing 400 for exceeding it
// lives, so the client learns about the limit instead of silently having extra entries dropped.
constexpr int kMaxLogitBiasEntries = 1024;

// sparkinfer's own bound on `n` (OpenAI's API documents no fixed upper limit) -- each unit of n
// costs a full redundant prefill+decode session against the shared ContinuousBatchEngine queue
// (see ContinuousBatchEngine::submit_locked / max_queue_depth in sparkinfer_server.cpp); this
// caps how much queue footprint one HTTP request can claim via n-fanout.
constexpr int kMaxN = 8;
// OpenAI's documented ceiling for top_logprobs, and the same bound Qwen35Model::last_token_logprobs
// clamps to on the device side.
constexpr int kMaxTopLogprobs = 20;

constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kVisionStart = "<|vision_start|>";
constexpr const char* kImagePad    = "<|image_pad|>";
constexpr const char* kVisionEnd   = "<|vision_end|>";
constexpr const char* kVideoPad    = "<|video_pad|>";
constexpr const char* kImEnd = "<|im_end|>";
constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkClose = "</think>";
constexpr const char* kToolCallOpen = "<tool_call>";
constexpr const char* kToolCallClose = "</tool_call>";
constexpr const char* kFunctionOpen = "<function=";
constexpr const char* kFunctionClose = "</function>";
constexpr const char* kParameterOpen = "<parameter=";
constexpr const char* kParameterClose = "</parameter>";
constexpr const char* kToolResponseOpen = "<tool_response>";
constexpr const char* kToolResponseClose = "</tool_response>";

const char* kToolInstructions = R"(# Tools

You have access to the following functions:

<tools>)";

const char* kToolInstructionsTail = R"(
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)";

const char* kJsonObjectInstructions = R"(# Response Format

Your entire reply MUST be a single valid JSON value and NOTHING else -- no prose before or
after, no markdown code fences, no explanations. Output only the raw JSON.)";

const char* kJsonSchemaInstructionsHead = R"(# Response Format

Your entire reply MUST be a single valid JSON object that conforms EXACTLY to the JSON Schema
below (named ")";
const char* kJsonSchemaInstructionsMid = R"("). Output ONLY the raw JSON object -- no prose
before or after, no markdown code fences, no explanations, no additional keys unless the
schema allows them.

<response_schema>
)";
const char* kJsonSchemaInstructionsTail = "\n</response_schema>";

bool set_error(std::string& err, const std::string& message) {
    err = message;
    return false;
}

re2::RE2::Options safe_regex_options() {
    re2::RE2::Options options;
    options.set_log_errors(false);
    return options;
}

bool parse_strict_json(const std::string& text, json& value, std::string& err,
                       const std::string& where) {
    // nlohmann's ordinary DOM parser accepts duplicate object keys (last value wins). That is
    // dangerous for tool requests: an auditor, prompt renderer, and executor could otherwise
    // see different meanings. Its public parser callback reports every object/key boundary,
    // which lets us reject duplicates without depending on version-specific detail classes.
    std::vector<std::set<std::string>> object_keys;
    std::string duplicate_key;
    const json::parser_callback_t callback =
        [&](int, json::parse_event_t event, json& parsed) -> bool {
            if (event == json::parse_event_t::object_start) {
                object_keys.emplace_back();
            } else if (event == json::parse_event_t::key) {
                const std::string key = parsed.get<std::string>();
                if (object_keys.empty() || !object_keys.back().insert(key).second) {
                    if (duplicate_key.empty()) duplicate_key = key;
                }
            } else if (event == json::parse_event_t::object_end && !object_keys.empty()) {
                object_keys.pop_back();
            }
            return true;
        };
    try {
        value = json::parse(text, callback, true, false);
    } catch (const json::exception& ex) {
        return set_error(err, where + ": " + ex.what());
    }
    if (!duplicate_key.empty())
        return set_error(err, where + ": duplicate JSON object key " + duplicate_key);
    return true;
}

bool safe_protocol_name(const std::string& value) {
    if (value.empty()) return false;
    // Function and top-level argument names are injected into unquoted Qwen protocol tags.
    // Keep the accepted alphabet deliberately narrower than JSON object keys so no control,
    // whitespace, quoting, or tag-delimiter byte can change the rendered structure.
    for (const unsigned char c : value) {
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.' && c != ':') return false;
    }
    return true;
}

std::string htmlsafe_json_string(const std::string& value) {
    std::string encoded = json(value).dump(-1, ' ', true);
    std::string safe;
    safe.reserve(encoded.size());
    for (const char c : encoded) {
        switch (c) {
            case '<': safe += "\\u003c"; break;
            case '>': safe += "\\u003e"; break;
            case '&': safe += "\\u0026"; break;
            case '\'': safe += "\\u0027"; break;
            default: safe.push_back(c); break;
        }
    }
    return safe;
}

// Jinja's `tojson` filter uses htmlsafe Python json.dumps: sorted object keys, ASCII escapes,
// and a space after commas/colons. Match that byte-for-byte for the JSON shapes used by the
// pinned Qwen3.6 template so tools cannot inject template tags through descriptions/schemas.
std::string qwen_template_json(const json& value) {
    if (value.is_string()) return htmlsafe_json_string(value.get_ref<const std::string&>());
    if (value.is_array()) {
        std::ostringstream out;
        out << '[';
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) out << ", ";
            out << qwen_template_json(value[i]);
        }
        out << ']';
        return out.str();
    }
    if (value.is_object()) {
        std::ostringstream out;
        out << '{';
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first) out << ", ";
            first = false;
            out << htmlsafe_json_string(item.key()) << ": "
                << qwen_template_json(item.value());
        }
        out << '}';
        return out.str();
    }
    return value.dump(-1, ' ', true);
}

bool is_allowed_key(const json& object, const std::set<std::string>& allowed,
                    const std::string& where, std::string& err,
                    bool allow_vendor_extensions = false) {
    for (const auto& item : object.items()) {
        if (allowed.count(item.key())) continue;
        // Vendor extensions. JSON Schema reserves no "x-" prefix itself, but OpenAPI-derived
        // tooling uses it universally and MCP servers emit it (x-mcp-header and friends). Like
        // $schema/$comment these carry no constraint, so accepting and dropping them is faithful
        // rather than permissive -- there is nothing for constrained decoding to enforce, so
        // ignoring them cannot weaken a guarantee (#981).
        //
        // Opt-in per call site: this is right for a tool's JSON Schema, and wrong for the request
        // envelope, where an unexpected x- key more likely means a client mistake worth surfacing.
        if (allow_vendor_extensions && item.key().rfind("x-", 0) == 0) continue;
        return set_error(err, where + " contains unsupported field " + item.key());
    }
    return true;
}

bool parse_content(const json& value, std::string& content, bool& is_null,
                   const std::string& where, std::string& err,
                   std::vector<std::string>* images, const std::string& role,
                   std::vector<std::string>* videos) {
    content.clear();
    is_null = value.is_null();
    if (is_null) return true;
    if (value.is_string()) {
        content = value.get<std::string>();
        return true;
    }
    if (!value.is_array())
        return set_error(err, where + ".content must be a string, null, or array of text parts");
    for (size_t i = 0; i < value.size(); ++i) {
        const json& part = value[i];
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string())
            return set_error(err, where + ".content[" + std::to_string(i) + "] must have a string type");
        if (part["type"] == "image_url" && images) {
            // Mirrors the template: an image part renders as this marker, and a system/developer
            // message containing one is refused outright rather than producing a prompt shape the
            // model was never trained to read.
            if (role == "system")
                return set_error(err, where + ".content[" + std::to_string(i) +
                                      "]: system/developer messages cannot contain images");
            const json& iu = part.contains("image_url") ? part["image_url"] : json();
            if (!iu.is_object() || !iu.contains("url") || !iu["url"].is_string())
                return set_error(err, where + ".content[" + std::to_string(i) +
                                      "].image_url.url must be a string");
            images->push_back(iu["url"].get<std::string>());
            content += kVisionStart;
            content += kImagePad;
            content += kVisionEnd;
            continue;
        }
        if (part["type"] == "video_url" && videos) {
            // Same shape as image_url above, and refused in system/developer messages for the
            // same reason -- the template raises on it outright.
            if (role == "system")
                return set_error(err, where + ".content[" + std::to_string(i) +
                                      "]: system/developer messages cannot contain videos");
            const json& vu = part.contains("video_url") ? part["video_url"] : json();
            if (!vu.is_object() || !vu.contains("url") || !vu["url"].is_string())
                return set_error(err, where + ".content[" + std::to_string(i) +
                                      "].video_url.url must be a string");
            videos->push_back(vu["url"].get<std::string>());
            // ONE <|video_pad|> here, exactly as the template emits. The per-frame timestamped
            // spans are expanded later, in token space, once the clip's grid is known -- the
            // template cannot do it because it does not know how many frames were sampled.
            content += kVisionStart;
            content += kVideoPad;
            content += kVisionEnd;
            continue;
        }
        // Remaining non-text parts (audio, and image_url when the caller passes no image sink)
        // are ignored rather than rejected: a client sending a mixed multimodal payload -- common
        // even against text-only backends, since agent frameworks build one payload shape for
        // every provider -- should still get an answer from the text that IS present.
        if (part["type"] != "text") continue;
        if (!part.contains("text") || !part["text"].is_string())
            return set_error(err, where + ".content[" + std::to_string(i) + "].text must be a string");
        content += part["text"].get<std::string>();
    }
    return true;
}

bool parse_arguments(const json& value, std::string& compact, const std::string& where,
                     std::string& err) {
    json arguments;
    if (value.is_object()) {
        arguments = value;
    } else if (value.is_string()) {
        if (!parse_strict_json(value.get_ref<const std::string&>(), arguments, err,
                               where + " is not valid JSON")) return false;
    } else {
        return set_error(err, where + " must be a JSON object or an object encoded as a string");
    }
    if (!arguments.is_object()) return set_error(err, where + " must encode a JSON object");
    compact = arguments.dump();
    return true;
}

bool parse_tool_call(const json& value, ToolCall& call, const std::string& where,
                     std::string& err) {
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value, {"id", "type", "function"}, where, err)) return false;
    if (!value.contains("id") || !value["id"].is_string() || value["id"].get_ref<const std::string&>().empty())
        return set_error(err, where + ".id must be a non-empty string");
    if (value.contains("type") && (!value["type"].is_string() || value["type"] != "function"))
        return set_error(err, where + ".type must be function");
    if (!value.contains("function") || !value["function"].is_object())
        return set_error(err, where + ".function must be an object");
    const json& function = value["function"];
    if (!is_allowed_key(function, {"name", "arguments"}, where + ".function", err)) return false;
    if (!function.contains("name") || !function["name"].is_string() ||
        !safe_protocol_name(function["name"].get_ref<const std::string&>()))
        return set_error(err, where + ".function.name must be a non-empty string");
    if (!function.contains("arguments"))
        return set_error(err, where + ".function.arguments is required");
    call.id = value["id"].get<std::string>();
    call.name = function["name"].get<std::string>();
    return parse_arguments(function["arguments"], call.arguments, where + ".function.arguments", err);
}

bool is_nonnegative_integer(const json& value) {
    return value.is_number_unsigned() ||
           (value.is_number_integer() && value.get<json::number_integer_t>() >= 0);
}

int compare_integer_to_float(const json& integer, double floating) {
    if (integer.is_number_unsigned()) {
        const uint64_t value = integer.get<json::number_unsigned_t>();
        if (floating < 0.0) return 1;
        // 2^64 is exactly representable as double; every uint64_t is below it.
        if (floating >= 18446744073709551616.0) return -1;
        const uint64_t truncated = static_cast<uint64_t>(floating);
        if (value < truncated) return -1;
        if (value > truncated) return 1;
        if (floating > static_cast<double>(truncated)) return -1;
        return 0;
    }
    const int64_t value = integer.get<json::number_integer_t>();
    // Both powers of two are exact doubles and bound every int64_t conversion.
    if (floating < -9223372036854775808.0) return 1;
    if (floating >= 9223372036854775808.0) return -1;
    const int64_t truncated = static_cast<int64_t>(floating);
    if (value < truncated) return -1;
    if (value > truncated) return 1;
    const double truncated_float = static_cast<double>(truncated);
    if (floating > truncated_float) return -1;
    if (floating < truncated_float) return 1;
    return 0;
}

int compare_json_numbers(const json& lhs, const json& rhs) {
    const bool lhs_float = lhs.is_number_float();
    const bool rhs_float = rhs.is_number_float();
    if (lhs_float && rhs_float) {
        const double left = lhs.get<json::number_float_t>();
        const double right = rhs.get<json::number_float_t>();
        return left < right ? -1 : (left > right ? 1 : 0);
    }
    if (!lhs_float && rhs_float)
        return compare_integer_to_float(lhs, rhs.get<json::number_float_t>());
    if (lhs_float && !rhs_float)
        return -compare_integer_to_float(rhs, lhs.get<json::number_float_t>());
    if (lhs.is_number_unsigned() && rhs.is_number_unsigned()) {
        const uint64_t left = lhs.get<json::number_unsigned_t>();
        const uint64_t right = rhs.get<json::number_unsigned_t>();
        return left < right ? -1 : (left > right ? 1 : 0);
    }
    if (!lhs.is_number_unsigned() && !rhs.is_number_unsigned()) {
        const int64_t left = lhs.get<json::number_integer_t>();
        const int64_t right = rhs.get<json::number_integer_t>();
        return left < right ? -1 : (left > right ? 1 : 0);
    }
    if (lhs.is_number_unsigned()) {
        const int64_t right = rhs.get<json::number_integer_t>();
        if (right < 0) return 1;
        const uint64_t left = lhs.get<json::number_unsigned_t>();
        const uint64_t converted = static_cast<uint64_t>(right);
        return left < converted ? -1 : (left > converted ? 1 : 0);
    }
    const int64_t left = lhs.get<json::number_integer_t>();
    if (left < 0) return -1;
    const uint64_t converted = static_cast<uint64_t>(left);
    const uint64_t right = rhs.get<json::number_unsigned_t>();
    return converted < right ? -1 : (converted > right ? 1 : 0);
}

bool declared_type_allows(const json& schema, const char* wanted) {
    if (!schema.contains("type")) return true;
    const json& type = schema["type"];
    if (type.is_string()) return type == wanted;
    if (type.is_array())
        return std::find(type.begin(), type.end(), json(wanted)) != type.end();
    return false;
}

bool require_keyword_type(const json& schema, const std::string& where,
                          std::initializer_list<const char*> keywords,
                          const char* required_type, std::string& err) {
    if (declared_type_allows(schema, required_type)) return true;
    for (const char* keyword : keywords) {
        if (schema.contains(keyword))
            return set_error(err, where + "." + keyword + " requires type " + required_type);
    }
    return true;
}

// Defined below, next to validate_value(); declared here because schema validation resolves
// $ref at parse time so an unresolvable pointer is refused when the tool is submitted.
const json* resolve_local_ref(const json& root, const std::string& ref);

bool valid_schema_node(const json& schema, const std::string& where, bool top_level,
                       std::string& err, const json* root = nullptr) {
    // The top-level schema is the root: that is where $defs lives, and every $ref below resolves
    // against it. Threading it lets an unresolvable $ref be refused at PARSE time rather than
    // surfacing later as an argument-validation failure -- a schema whose $ref points nowhere is
    // broken as written, and saying so when it is submitted is far more useful to the caller than
    // a 400 on the first tool call that happens to exercise that branch.
    if (!root) root = &schema;
    if (!schema.is_object()) return set_error(err, where + " must be an object");
    // "$schema" and "$comment" carry no constraints -- they are annotations a validator is
    // required to ignore -- so accepting and dropping them is faithful, not permissive. Clients
    // built on @ai-sdk/openai-compatible emit "$schema" on every tool schema, and rejecting it
    // failed the whole request. Structural "$" keywords are different: $ref and $defs are accepted
    // only because validate_value() resolves them (local pointers only), and $id stays refused --
    // silently ignoring a reference would validate the arguments against nothing.
    if (!is_allowed_key(schema,
                        {"$schema", "$comment",
                         "type", "description", "default", "title", "properties",
                         "required", "additionalProperties", "items", "enum", "minimum",
                         "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems",
                         "maxItems", "minLength", "maxLength", "pattern",
                         // Every keyword below is ENFORCED in validate_value(). Nothing is
                         // whitelisted that the validator ignores: this backend has no
                         // constrained decoding, so a keyword accepted but unchecked would turn
                         // an honest 400 into a model silently violating the constraint.
                         "const", "anyOf", "oneOf", "allOf", "multipleOf", "prefixItems",
                         // $ref is resolved against $defs/definitions in validate_value(), local
                         // pointers only -- an external ref is refused rather than fetched.
                         "$ref", "$defs", "definitions",
                         // "format" is the one exception, and it is not an exception to that
                         // rule: JSON Schema defines format as an ANNOTATION by default, not an
                         // assertion, so ignoring it enforces exactly what the spec requires.
                         "format"},
                        where, err, /*allow_vendor_extensions=*/true)) return false;
    for (const char* annotation : {"$schema", "$comment"}) {
        if (schema.contains(annotation) && !schema[annotation].is_string())
            return set_error(err, where + "." + annotation + " must be a string");
    }
    for (const char* annotation : {"description", "title"}) {
        if (schema.contains(annotation) && !schema[annotation].is_string())
            return set_error(err, where + "." + annotation + " must be a string");
    }
    const std::set<std::string> allowed_types = {
        "array", "boolean", "integer", "null", "number", "object", "string"};
    if (schema.contains("type")) {
        const json& type = schema["type"];
        if (top_level) {
            if (!type.is_string() || type != "object")
                return set_error(err, where + ".type must be object");
        } else if (type.is_string()) {
            if (!allowed_types.count(type.get<std::string>()))
                return set_error(err, where + ".type is unsupported");
        } else if (type.is_array() && !type.empty()) {
            std::set<std::string> seen;
            for (const auto& item : type) {
                if (!item.is_string() || !allowed_types.count(item.get<std::string>()))
                    return set_error(err, where + ".type contains an unsupported type");
                if (!seen.insert(item.get<std::string>()).second)
                    return set_error(err, where + ".type contains a duplicate type");
            }
        } else {
            return set_error(err, where + ".type must be a string or non-empty array of strings");
        }
    }
    if (!require_keyword_type(schema, where,
                              {"properties", "required", "additionalProperties"},
                              "object", err) ||
        !require_keyword_type(schema, where, {"items", "minItems", "maxItems"},
                              "array", err) ||
        !require_keyword_type(schema, where, {"minLength", "maxLength", "pattern"},
                              "string", err)) return false;
    if (!declared_type_allows(schema, "number") &&
        !declared_type_allows(schema, "integer")) {
        for (const char* keyword :
             {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
            if (schema.contains(keyword))
                return set_error(err, where + "." + keyword +
                                      " requires type number or integer");
        }
    }
    // Composition branches are schemas too, and MUST be validated as such. Whitelisting anyOf at
    // the parent while never descending into its branches would let a caller hide a rejected
    // keyword inside one -- {"anyOf": [{"$ref": "..."}]} -- where validate_value() then ignores
    // the unknown key, constrains nothing, and the branch matches anything. The anyOf would pass
    // trivially while the caller believes a constraint is in force: exactly the silent weakening
    // that refusing $ref at the top level exists to prevent.
    // A $ref node is validated against its target, and draft-07 ignores sibling keywords -- so a
    // node carrying $ref plus constraints would silently drop those constraints. Refuse it instead
    // of accepting a schema whose visible text does not describe what is enforced.
    if (schema.contains("$ref")) {
        if (!schema["$ref"].is_string())
            return set_error(err, where + ".$ref must be a string");
        for (const auto& item : schema.items())
            if (item.key() != "$ref" && item.key() != "$comment" && item.key() != "description" &&
                item.key().rfind("x-", 0) != 0)
                return set_error(err, where + " combines $ref with " + item.key() +
                                      ", which would be silently ignored");
        if (!resolve_local_ref(*root, schema["$ref"].get<std::string>()))
            return set_error(err, where + " has an unresolvable or non-local $ref: " +
                                  schema["$ref"].get<std::string>());
    }
    // $defs / definitions hold schemas, so they are validated as schemas. Without this a rejected
    // keyword could hide in a definition and reach validate_value through a $ref.
    for (const char* defs : {"$defs", "definitions"}) {
        if (!schema.contains(defs)) continue;
        if (!schema[defs].is_object())
            return set_error(err, std::string(where) + "." + defs + " must be an object");
        for (const auto& def : schema[defs].items())
            if (!valid_schema_node(def.value(), where + "." + defs + "." + def.key(), false, err, root))
                return false;
    }
    for (const char* composition : {"anyOf", "oneOf", "allOf"}) {
        if (!schema.contains(composition)) continue;
        const json& branches = schema[composition];
        if (!branches.is_array() || branches.empty())
            return set_error(err, where + "." + composition + " must be a non-empty array");
        for (size_t i = 0; i < branches.size(); ++i) {
            if (!valid_schema_node(branches[i],
                                   where + "." + composition + "[" + std::to_string(i) + "]",
                                   false, err, root)) return false;
        }
    }
    if (schema.contains("properties")) {
        if (!schema["properties"].is_object())
            return set_error(err, where + ".properties must be an object");
        for (const auto& property : schema["properties"].items()) {
            if (!valid_schema_node(property.value(), where + ".properties." + property.key(),
                                   false, err, root)) return false;
        }
    }
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) return set_error(err, where + ".required must be an array");
        std::set<std::string> seen;
        for (const auto& item : schema["required"]) {
            if (!item.is_string() || item.get_ref<const std::string&>().empty())
                return set_error(err, where + ".required entries must be non-empty strings");
            const std::string name = item.get<std::string>();
            if (!seen.insert(name).second) return set_error(err, where + ".required contains duplicate " + name);
            if (schema.contains("properties") && !schema["properties"].contains(name) &&
                schema.contains("additionalProperties") &&
                schema["additionalProperties"].is_boolean() &&
                !schema["additionalProperties"].get<bool>())
                return set_error(err, where + ".required names a forbidden property " + name);
        }
    }
    if (schema.contains("additionalProperties") && !schema["additionalProperties"].is_boolean() &&
        !schema["additionalProperties"].is_object())
        return set_error(err, where + ".additionalProperties must be boolean or an object schema");
    if (schema.contains("additionalProperties") && schema["additionalProperties"].is_object() &&
        !valid_schema_node(schema["additionalProperties"], where + ".additionalProperties",
                           false, err, root)) return false;
    if (schema.contains("prefixItems")) {
        if (!schema["prefixItems"].is_array() || schema["prefixItems"].empty())
            return set_error(err, where + ".prefixItems must be a non-empty array");
        for (size_t i = 0; i < schema["prefixItems"].size(); ++i)
            if (!valid_schema_node(schema["prefixItems"][i],
                                   where + ".prefixItems[" + std::to_string(i) + "]", false, err,
                                   root)) return false;
    }
    if (schema.contains("items")) {
        if (!schema["items"].is_object())
            return set_error(err, where + ".items must be an object schema");
        if (!valid_schema_node(schema["items"], where + ".items", false, err, root)) return false;
    }
    if (schema.contains("enum") &&
        (!schema["enum"].is_array() || schema["enum"].empty()))
        return set_error(err, where + ".enum must be a non-empty array");
    for (const char* keyword : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
        if (schema.contains(keyword) && !schema[keyword].is_number())
            return set_error(err, where + "." + keyword + " must be a number");
    }
    if (schema.contains("minimum") && schema.contains("maximum") &&
        compare_json_numbers(schema["minimum"], schema["maximum"]) > 0)
        return set_error(err, where + ".minimum must not exceed maximum");
    for (const char* keyword : {"minItems", "maxItems", "minLength", "maxLength"}) {
        if (schema.contains(keyword) && !is_nonnegative_integer(schema[keyword]))
            return set_error(err, where + "." + keyword + " must be a non-negative integer");
    }
    if (schema.contains("minItems") && schema.contains("maxItems") &&
        schema["minItems"].get<std::size_t>() > schema["maxItems"].get<std::size_t>())
        return set_error(err, where + ".minItems must not exceed maxItems");
    if (schema.contains("minLength") && schema.contains("maxLength") &&
        schema["minLength"].get<std::size_t>() > schema["maxLength"].get<std::size_t>())
        return set_error(err, where + ".minLength must not exceed maxLength");
    if (schema.contains("pattern")) {
        if (!schema["pattern"].is_string())
            return set_error(err, where + ".pattern must be a string");
        const re2::RE2 pattern(schema["pattern"].get_ref<const std::string&>(),
                               safe_regex_options());
        if (!pattern.ok())
            return set_error(err, where + ".pattern is not a supported safe regular expression");
    }
    return true;
}

bool valid_schema(const json& schema, const std::string& where, std::string& err) {
    return valid_schema_node(schema, where, true, err);
}

bool parse_tool_definition(const json& value, ToolDefinition& tool, const std::string& where,
                           std::string& err) {
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value, {"type", "function"}, where, err)) return false;
    if (!value.contains("type") || !value["type"].is_string() || value["type"] != "function")
        return set_error(err, where + ".type must be function");
    if (!value.contains("function") || !value["function"].is_object())
        return set_error(err, where + ".function must be an object");
    const json& function = value["function"];
    if (!is_allowed_key(function, {"name", "description", "parameters", "strict"},
                        where + ".function", err)) return false;
    if (!function.contains("name") || !function["name"].is_string() ||
        !safe_protocol_name(function["name"].get_ref<const std::string&>()))
        return set_error(err, where + ".function.name is not safe for the Qwen tool protocol");
    if (function.contains("description") && !function["description"].is_string())
        return set_error(err, where + ".function.description must be a string");
    if (!function.contains("parameters"))
        return set_error(err, where + ".function.parameters is required");
    if (!valid_schema(function["parameters"], where + ".function.parameters", err)) return false;
    if (function["parameters"].contains("properties")) {
        for (const auto& property : function["parameters"]["properties"].items()) {
            if (!safe_protocol_name(property.key()))
                return set_error(err, where + ".function.parameters property " + property.key() +
                                      " is not safe for the Qwen tool protocol");
        }
    }
    if (function["parameters"].contains("required")) {
        for (const auto& property : function["parameters"]["required"]) {
            if (!safe_protocol_name(property.get_ref<const std::string&>()))
                return set_error(err, where + ".function.parameters required property " +
                                      property.get<std::string>() +
                                      " is not safe for the Qwen tool protocol");
        }
    }
    if (function.contains("strict") && !function["strict"].is_boolean())
        return set_error(err, where + ".function.strict must be boolean");
    tool.name = function["name"].get<std::string>();
    tool.spec = value;
    return true;
}

std::string trim_copy(std::string value) {
    auto ws = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && ws(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && ws(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

void trim_leading_ws(std::string& value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
}

void trim_trailing_ws(std::string& value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
}

void strip_trailing_im_end(std::string& value) {
    const std::string marker = kImEnd;
    const size_t pos = value.rfind(marker);
    if (pos != std::string::npos && value.substr(pos + marker.size()).find_first_not_of(" \t\r\n") == std::string::npos)
        value.resize(pos);
    trim_trailing_ws(value);
}

bool parse_message(const json& value, ChatMessage& message, size_t index, std::string& err) {
    const std::string where = "messages[" + std::to_string(index) + "]";
    if (!value.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(value,
                        {"role", "content", "reasoning_content", "name", "tool_call_id", "tool_calls"},
                        where, err)) return false;
    if (!value.contains("role") || !value["role"].is_string())
        return set_error(err, where + ".role must be a string");
    message.role = value["role"].get<std::string>();
    // "developer" is OpenAI's newer replacement for "system" (some current client SDKs send
    // it by default) -- treat it as an alias rather than rejecting a request this backend can
    // otherwise serve perfectly well.
    if (message.role == "developer") message.role = "system";
    if (message.role != "system" && message.role != "user" && message.role != "assistant" &&
        message.role != "tool")
        return set_error(err, where + ".role is unsupported");
    if (value.contains("content")) {
        if (!parse_content(value["content"], message.content, message.content_is_null, where, err,
                           &message.images, message.role, &message.videos)) return false;
    } else {
        message.content_is_null = true;
    }
    if (value.contains("reasoning_content")) {
        if (!value["reasoning_content"].is_string())
            return set_error(err, where + ".reasoning_content must be a string");
        message.reasoning_content = value["reasoning_content"].get<std::string>();
    }
    if (value.contains("name")) {
        if (!value["name"].is_string()) return set_error(err, where + ".name must be a string");
        message.name = value["name"].get<std::string>();
    }
    if (value.contains("tool_call_id")) {
        if (!value["tool_call_id"].is_string() || value["tool_call_id"].get_ref<const std::string&>().empty())
            return set_error(err, where + ".tool_call_id must be a non-empty string");
        message.tool_call_id = value["tool_call_id"].get<std::string>();
    }
    if (value.contains("tool_calls")) {
        if (message.role != "assistant") return set_error(err, where + ".tool_calls is only valid for assistant");
        if (!value["tool_calls"].is_array() || value["tool_calls"].empty())
            return set_error(err, where + ".tool_calls must be a non-empty array");
        std::set<std::string> ids;
        for (size_t i = 0; i < value["tool_calls"].size(); ++i) {
            ToolCall call;
            if (!parse_tool_call(value["tool_calls"][i], call,
                                 where + ".tool_calls[" + std::to_string(i) + "]", err)) return false;
            if (!ids.insert(call.id).second) return set_error(err, where + ".tool_calls contains duplicate id " + call.id);
            message.tool_calls.push_back(std::move(call));
        }
    }
    if (message.role == "system" && index != 0)
        return set_error(err, "system message must be first");
    if (message.role == "tool") {
        if (message.content_is_null) return set_error(err, where + ".content is required for tool messages");
        if (message.tool_call_id.empty()) return set_error(err, where + ".tool_call_id is required for tool messages");
    } else if (!message.tool_call_id.empty()) {
        return set_error(err, where + ".tool_call_id is only valid for tool messages");
    }
    if (message.role != "assistant" && message.content_is_null)
        return set_error(err, where + ".content must not be null for role " + message.role);
    if (message.role == "assistant" && message.content_is_null && message.tool_calls.empty())
        return set_error(err, where + " must contain content or tool_calls");
    return true;
}

bool resolve_history(const ChatRequest& request, std::string& err) {
    std::map<std::string, std::string> pending;
    for (size_t i = 0; i < request.messages.size(); ++i) {
        const ChatMessage& message = request.messages[i];
        if (message.role == "assistant" && !message.tool_calls.empty()) {
            if (!pending.empty())
                return set_error(err, "messages[" + std::to_string(i) + "] starts new tool calls before all prior tool results");
            for (const ToolCall& call : message.tool_calls) pending.emplace(call.id, call.name);
        } else if (message.role == "tool") {
            auto it = pending.find(message.tool_call_id);
            if (it == pending.end())
                return set_error(err, "messages[" + std::to_string(i) + "].tool_call_id does not reference a pending call");
            if (!message.name.empty() && message.name != it->second)
                return set_error(err, "messages[" + std::to_string(i) + "].name disagrees with its tool call");
            pending.erase(it);
        } else if (!pending.empty()) {
            return set_error(err, "tool results must immediately follow their assistant tool calls");
        }
    }
    if (!pending.empty()) return set_error(err, "assistant tool calls are missing tool result messages");
    return true;
}

std::string json_value_for_parameter(const json& value) {
    return value.is_string() ? value.get<std::string>() : qwen_template_json(value);
}

std::string render_tool_call(const ToolCall& call) {
    // call.arguments is always produced internally by parse_arguments's own .dump(), so this
    // should be unreachable -- but every other JSON parse in this file goes through the
    // exception-safe parse_strict_json rather than the throwing overload, to avoid an uncaught
    // exception surfacing as an unstructured 500 from inside the HTTP handler.
    json arguments;
    std::string parse_err;
    if (!parse_strict_json(call.arguments, arguments, parse_err, "tool call arguments"))
        arguments = json::object();
    std::ostringstream out;
    out << kToolCallOpen << '\n' << kFunctionOpen << call.name << ">\n";
    for (const auto& item : arguments.items()) {
        out << kParameterOpen << item.key() << ">\n" << json_value_for_parameter(item.value())
            << '\n' << kParameterClose << '\n';
    }
    out << kFunctionClose << '\n' << kToolCallClose;
    return out.str();
}

bool has_protocol_markup(const std::string& text) {
    // Match incomplete/misspelled openings too. A partial protocol tag must never be returned as
    // assistant content merely because it failed to reach the exact full-marker search below.
    return text.find("<tool") != std::string::npos || text.find("</tool") != std::string::npos ||
           text.find("<function") != std::string::npos || text.find("</function") != std::string::npos ||
           text.find("<parameter") != std::string::npos || text.find("</parameter") != std::string::npos ||
           text.find("<think") != std::string::npos || text.find("</think") != std::string::npos ||
           text.find("<tool_response") != std::string::npos ||
           text.find("</tool_response") != std::string::npos ||
           text.find("<|im_") != std::string::npos;
}

ParsedToolOutput fail_tool_output(ParsedToolOutput out, const std::string& error) {
    out.reasoning_content.clear();
    out.content.clear();
    out.tool_calls.clear();
    out.error = error;
    return out;
}

// tool_choice demanded a call and the output has no call to an offered function. Still a failure,
// but one the server can act on: the reasoning is kept so the model can continue from it into a
// forced call.
ParsedToolOutput fail_missing_call(ParsedToolOutput out, const std::string& error) {
    std::string reasoning = std::move(out.reasoning_content);
    out = fail_tool_output(std::move(out), error);
    out.reasoning_content = std::move(reasoning);
    out.missing_required_call = true;
    return out;
}

bool parse_scalar_from_text(const std::string& value, json& parsed) {
    std::string ignored;
    return parse_strict_json(value, parsed, ignored, "parameter value");
}

bool schema_allows_type(const json& schema, const std::string& wanted) {
    if (schema.contains("type")) {
        const json& type = schema["type"];
        if (type.is_string()) return type == wanted;
        if (type.is_array())
            return std::find(type.begin(), type.end(), json(wanted)) != type.end();
        return false;
    }
    if (!schema.contains("enum")) return true;
    for (const auto& item : schema["enum"]) {
        if ((wanted == "string" && item.is_string()) ||
            (wanted == "object" && item.is_object()) ||
            (wanted == "array" && item.is_array()) ||
            (wanted == "integer" && (item.is_number_integer() || item.is_number_unsigned())) ||
            (wanted == "number" && item.is_number()) ||
            (wanted == "boolean" && item.is_boolean()) ||
            (wanted == "null" && item.is_null())) return true;
    }
    return false;
}

bool schema_allows_non_string(const json& schema) {
    for (const char* type : {"object", "array", "integer", "number", "boolean", "null"})
        if (schema_allows_type(schema, type)) return true;
    return false;
}

bool utf8_code_point_count(const std::string& value, std::size_t& count) {
    count = 0;
    for (std::size_t i = 0; i < value.size();) {
        const unsigned char lead = static_cast<unsigned char>(value[i]);
        std::size_t length = 0;
        uint32_t code_point = 0;
        if (lead <= 0x7f) {
            length = 1;
            code_point = lead;
        } else if (lead >= 0xc2 && lead <= 0xdf) {
            length = 2;
            code_point = lead & 0x1f;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            length = 3;
            code_point = lead & 0x0f;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            length = 4;
            code_point = lead & 0x07;
        } else {
            return false;
        }
        if (i + length > value.size()) return false;
        for (std::size_t j = 1; j < length; ++j) {
            const unsigned char continuation = static_cast<unsigned char>(value[i + j]);
            if ((continuation & 0xc0) != 0x80) return false;
            code_point = (code_point << 6) | (continuation & 0x3f);
        }
        if ((length == 3 && code_point >= 0xd800 && code_point <= 0xdfff) ||
            (length == 3 && code_point < 0x800) ||
            (length == 4 && (code_point < 0x10000 || code_point > 0x10ffff))) return false;
        ++count;
        i += length;
    }
    return true;
}

// Resolves a local JSON pointer of the form "#/$defs/Name" or "#/definitions/Name" against the
// root schema. LOCAL ONLY: an external "$ref" (http://, file://, or any other document) is refused
// rather than fetched -- an inference server that dereferences caller-supplied URLs is the same
// SSRF primitive parse_image_url already refuses, and a file:// ref would read local paths.
bool validate_value(const json& value, const json& schema, const std::string& path,
                    std::string& err, const json* root = nullptr, int depth = 0);

const json* resolve_local_ref(const json& root, const std::string& ref) {
    if (ref.rfind("#/", 0) != 0) return nullptr;      // "#" alone, or an external document
    const json* node = &root;
    size_t pos = 2;
    while (pos <= ref.size()) {
        const size_t next = ref.find('/', pos);
        std::string token = ref.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        // RFC 6901 escaping: ~1 is '/', ~0 is '~', and in that order.
        for (size_t i = 0; (i = token.find("~1", i)) != std::string::npos; ) token.replace(i, 2, "/");
        for (size_t i = 0; (i = token.find("~0", i)) != std::string::npos; ) token.replace(i, 2, "~");
        if (!node->is_object() || !node->contains(token)) return nullptr;
        node = &(*node)[token];
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return node->is_object() ? node : nullptr;
}

// Nesting bound for validation. $ref makes a schema a graph rather than a tree, so a self- or
// mutually-referential definition can recurse forever on a value that never shrinks. Bounding the
// depth turns that into a clean rejection instead of a stack overflow; 64 is far past any real
// tool schema and well inside the stack.
constexpr int kMaxSchemaDepth = 64;

bool validate_value(const json& value, const json& schema, const std::string& path, std::string& err,
                    const json* root, int depth) {
    if (!schema.is_object()) return true;
    if (depth > kMaxSchemaDepth)
        return set_error(err, path + " exceeds the maximum schema nesting depth");
    // The top-level schema IS the root when no other is supplied -- that is where $defs lives.
    if (!root) root = &schema;

    if (schema.contains("$ref")) {
        if (!schema["$ref"].is_string())
            return set_error(err, path + " has a non-string $ref");
        const std::string ref = schema["$ref"].get<std::string>();
        const json* target = resolve_local_ref(*root, ref);
        if (!target) return set_error(err, path + " has an unresolvable or non-local $ref: " + ref);
        // A node carrying $ref is validated against its target. Sibling keywords are ignored, per
        // draft-07 semantics -- accepting them silently would be the usual weakening, so
        // valid_schema_node refuses a $ref node that carries anything else.
        return validate_value(value, *target, path, err, root, depth + 1);
    }

    // --- composition and single-literal keywords (#981) -------------------------------------
    // These are ENFORCED here, not merely whitelisted. This backend does no constrained decoding
    // -- arguments are validated after the fact -- so accepting a keyword the validator ignores
    // would replace an honest 400 with a model silently violating the constraint, which is worse
    // than rejecting it. Every keyword added to the schema whitelist alongside this must appear
    // below.
    //
    // anyOf is the one that actually blocks MCP tools today: `anyOf: [T, null]` is how every
    // optional/nullable field is expressed.
    if (schema.contains("const")) {
        if (value != schema["const"])
            return set_error(err, path + " does not equal the value required by const");
    }
    if (schema.contains("allOf")) {
        // Trivial in a post-hoc validator: the value must satisfy every branch. The report scoped
        // this as "schema intersection, genuinely hard" -- true when BUILDING a grammar, where the
        // branches must be merged into one production. Validating an existing value needs no
        // merge at all.
        if (!schema["allOf"].is_array() || schema["allOf"].empty())
            return set_error(err, path + " has invalid allOf schema");
        for (const json& sub : schema["allOf"])
            if (!validate_value(value, sub, path, err, root, depth + 1)) return false;
    }
    if (schema.contains("anyOf")) {
        if (!schema["anyOf"].is_array() || schema["anyOf"].empty())
            return set_error(err, path + " has invalid anyOf schema");
        bool any = false;
        for (const json& sub : schema["anyOf"]) {
            std::string ignored;   // a failing branch is not an error; only all-failing is
            if (validate_value(value, sub, path, ignored, root, depth + 1)) { any = true; break; }
        }
        if (!any) return set_error(err, path + " does not match any anyOf branch");
    }
    if (schema.contains("oneOf")) {
        if (!schema["oneOf"].is_array() || schema["oneOf"].empty())
            return set_error(err, path + " has invalid oneOf schema");
        int matched = 0;
        for (const json& sub : schema["oneOf"]) {
            std::string ignored;
            if (validate_value(value, sub, path, ignored, root, depth + 1)) matched++;
        }
        // EXACTLY one, per the spec. Matching several is as much a failure as matching none --
        // an overlapping oneOf means the caller's schema is ambiguous, and silently accepting the
        // first hit would hide that.
        if (matched != 1)
            return set_error(err, path + " matches " + std::to_string(matched) +
                                  " oneOf branches, expected exactly 1");
    }
    if (schema.contains("multipleOf") && schema["multipleOf"].is_number() && value.is_number()) {
        const double step = schema["multipleOf"].get<double>();
        if (step > 0.0) {
            const double v = value.get<double>();
            const double rem = std::fabs(v - step * std::round(v / step));
            // Relative epsilon: 0.1 is not representable in binary, so an exact fmod test rejects
            // legitimately-conforming values like 0.3 against multipleOf 0.1.
            if (rem > 1e-9 * std::max(1.0, std::fabs(v)))
                return set_error(err, path + " is not a multiple of the required step");
        }
    }
    // "format" is deliberately absent: JSON Schema defines it as an ANNOTATION by default, not an
    // assertion, so accepting and ignoring it is spec-correct rather than a silent weakening.

    if (schema.contains("type")) {
        auto matches = [&](const std::string& type) {
            if (type == "object") return value.is_object();
            if (type == "array") return value.is_array();
            if (type == "string") return value.is_string();
            if (type == "integer") return value.is_number_integer() || value.is_number_unsigned();
            if (type == "number") return value.is_number();
            if (type == "boolean") return value.is_boolean();
            if (type == "null") return value.is_null();
            return false;
        };
        bool ok = false;
        if (schema["type"].is_string()) ok = matches(schema["type"].get<std::string>());
        else if (schema["type"].is_array())
            for (const auto& type : schema["type"])
                if (type.is_string() && matches(type.get<std::string>())) ok = true;
        if (!ok) return set_error(err, path + " has the wrong JSON type");
    }
    if (schema.contains("enum")) {
        if (!schema["enum"].is_array()) return set_error(err, path + " has invalid enum schema");
        if (std::find(schema["enum"].begin(), schema["enum"].end(), value) == schema["enum"].end())
            return set_error(err, path + " is not one of the allowed enum values");
    }
    if (value.is_number()) {
        if (schema.contains("minimum") && schema["minimum"].is_number() &&
            compare_json_numbers(value, schema["minimum"]) < 0)
            return set_error(err, path + " is below minimum");
        if (schema.contains("maximum") && schema["maximum"].is_number() &&
            compare_json_numbers(value, schema["maximum"]) > 0)
            return set_error(err, path + " is above maximum");
        if (schema.contains("exclusiveMinimum") && schema["exclusiveMinimum"].is_number() &&
            compare_json_numbers(value, schema["exclusiveMinimum"]) <= 0)
            return set_error(err, path + " is not above exclusiveMinimum");
        if (schema.contains("exclusiveMaximum") && schema["exclusiveMaximum"].is_number() &&
            compare_json_numbers(value, schema["exclusiveMaximum"]) >= 0)
            return set_error(err, path + " is not below exclusiveMaximum");
    }
    if (value.is_string()) {
        const std::string& string_value = value.get_ref<const std::string&>();
        std::size_t string_length = 0;
        if (!utf8_code_point_count(string_value, string_length))
            return set_error(err, path + " is not valid UTF-8");
        if (schema.contains("minLength") &&
            string_length < schema["minLength"].get<std::size_t>())
            return set_error(err, path + " is shorter than minLength");
        if (schema.contains("maxLength") &&
            string_length > schema["maxLength"].get<std::size_t>())
            return set_error(err, path + " is longer than maxLength");
        if (schema.contains("pattern")) {
            const re2::RE2 pattern(schema["pattern"].get_ref<const std::string&>(),
                                   safe_regex_options());
            if (!pattern.ok())
                return set_error(err, path + " has an invalid pattern schema");
            if (!re2::RE2::PartialMatch(string_value, pattern))
                return set_error(err, path + " does not match pattern");
        }
    }
    if (value.is_object()) {
        const json properties = schema.value("properties", json::object());
        if (schema.contains("required")) {
            for (const auto& name : schema["required"])
                if (!value.contains(name.get<std::string>()))
                    return set_error(err, path + " is missing required parameter " + name.get<std::string>());
        }
        // JSON Schema defaults additionalProperties to true. Only an explicit false closes
        // the object; this matters for permissive nested schemas such as {"type":"object"}.
        const bool allow_unknown = !schema.contains("additionalProperties") ||
                                   schema["additionalProperties"] != json(false);
        for (const auto& item : value.items()) {
            if (properties.contains(item.key())) {
                if (!validate_value(item.value(), properties[item.key()], path + "." + item.key(), err,
                                    root, depth + 1)) return false;
            } else if (!allow_unknown) {
                return set_error(err, path + " contains unknown parameter " + item.key());
            } else if (schema.contains("additionalProperties") &&
                       schema["additionalProperties"].is_object() &&
                       !validate_value(item.value(), schema["additionalProperties"],
                                       path + "." + item.key(), err, root, depth + 1)) {
                return false;
            }
        }
    } else if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<std::size_t>())
            return set_error(err, path + " has fewer than minItems elements");
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<std::size_t>())
            return set_error(err, path + " has more than maxItems elements");
        // prefixItems constrains element i by prefixItems[i]; "items" then applies to whatever
        // remains past the prefix (2020-12 semantics). With no prefixItems this is the previous
        // behaviour exactly -- prefix_n is 0 and every element goes through "items".
        const size_t prefix_n = schema.contains("prefixItems") && schema["prefixItems"].is_array()
                                    ? schema["prefixItems"].size() : 0;
        for (size_t i = 0; i < prefix_n && i < value.size(); ++i)
            if (!validate_value(value[i], schema["prefixItems"][i],
                                path + "[" + std::to_string(i) + "]", err, root, depth + 1))
                return false;
        if (schema.contains("items")) {
            for (size_t i = prefix_n; i < value.size(); ++i)
                if (!validate_value(value[i], schema["items"],
                                    path + "[" + std::to_string(i) + "]", err,
                                    root, depth + 1)) return false;
        }
    }
    return true;
}

// root is the tool's whole parameters schema, where $defs live: a property that is a $ref (or
// reaches one through anyOf/oneOf) resolves against it, never against the property itself.
bool parse_parameter_value(const std::string& value, const json& schema, json& parsed,
                           const json* root) {
    const bool allows_string = schema_allows_type(schema, "string");
    const bool allows_non_string = schema_allows_non_string(schema);
    if (allows_string && !allows_non_string) {
        parsed = value;
        return true;
    }
    if (allows_non_string) {
        json candidate;
        if (parse_scalar_from_text(value, candidate)) {
            std::string validation_error;
            if (validate_value(candidate, schema, "parameter value", validation_error, root)) {
                parsed = std::move(candidate);
                return true;
            }
        }
    }
    if (allows_string) {
        parsed = value;
        return true;
    }
    return false;
}

const json* property_schema_for_key(const json& object_schema, const json& properties,
                                    const std::string& key) {
    if (properties.contains(key)) return &properties[key];
    if (object_schema.contains("additionalProperties")) {
        const json& additional = object_schema["additionalProperties"];
        if (additional.is_boolean() && !additional.get<bool>()) return nullptr;
        if (additional.is_object()) return &additional;
    }
    static const json unconstrained = json::object();
    return &unconstrained;
}

// Exact match first, then ASCII case-insensitive as a fallback.
//
// Qwen3.8 capitalises function names in its XML tool calls often enough to matter -- it emits
// <function=Read> for an offered `read` -- and a strict compare turns that into "model called
// unoffered function Read", failing an agent loop over a spelling difference the model chose.
// SGLang and vLLM both normalise (#981).
//
// EXACT WINS. The fallback only runs when nothing matched exactly, so a caller that offers both
// `read` and `Read` keeps the strict behaviour for both, and adding a second casing can never
// change which tool an existing exact name resolves to.
//
// AMBIGUITY KEEPS THE STRICT BEHAVIOUR. If two offered tools differ only by case, a
// case-insensitive hit is not well defined, so this returns nullptr and the caller reports the
// name as unoffered -- guessing between them would silently invoke the wrong function.
//
// ASCII only, deliberately: a locale-aware or Unicode fold would make the set of matched names
// depend on the server's locale, and tool names crossing that boundary should fail loudly.
const ToolDefinition* offered_tool(const ChatRequest& request, const std::string& name) {
    for (const ToolDefinition& tool : request.tools)
        if (tool.name == name) return &tool;

    auto ascii_lower = [](std::string v) {
        for (char& c : v)
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        return v;
    };
    const std::string want = ascii_lower(name);
    const ToolDefinition* hit = nullptr;
    for (const ToolDefinition& tool : request.tools) {
        if (ascii_lower(tool.name) != want) continue;
        if (hit) return nullptr;   // ambiguous: two offered tools differ only by case
        hit = &tool;
    }
    return hit;
}

// `unoffered` (optional) is set when the call is well-formed but names no offered function.
bool parse_one_xml_call(const std::string& block, const ChatRequest& request, ToolCall& call,
                        std::string& err, bool* unoffered = nullptr) {
    size_t pos = 0;
    while (pos < block.size() && std::isspace(static_cast<unsigned char>(block[pos]))) ++pos;
    if (block.compare(pos, std::char_traits<char>::length(kFunctionOpen), kFunctionOpen) != 0)
        return set_error(err, "tool call is missing <function=...>");
    const size_t name_start = pos + std::char_traits<char>::length(kFunctionOpen);
    const size_t name_end = block.find('>', name_start);
    if (name_end == std::string::npos) return set_error(err, "tool call has an unterminated function name");
    call.name = block.substr(name_start, name_end - name_start);
    if (!safe_protocol_name(call.name))
        return set_error(err, "tool call has an invalid function name");
    const ToolDefinition* tool = offered_tool(request, call.name);
    if (!tool) {
        if (unoffered) *unoffered = true;
        return set_error(err, "model called unoffered function " + call.name);
    }
    // Echo the name back exactly as the CLIENT offered it, not as the model spelled it. The client
    // dispatches on its own spelling -- returning the model's "Read" for an offered "read" would
    // resolve here and then miss in the caller's own handler table, moving the failure somewhere
    // harder to diagnose. No-op when the match was exact (#981).
    call.name = tool->name;

    const json& schema = tool->spec["function"]["parameters"];
    const json properties = schema.value("properties", json::object());
    json arguments = json::object();
    pos = name_end + 1;
    while (true) {
        while (pos < block.size() && std::isspace(static_cast<unsigned char>(block[pos]))) ++pos;
        if (block.compare(pos, std::char_traits<char>::length(kFunctionClose), kFunctionClose) == 0) {
            pos += std::char_traits<char>::length(kFunctionClose);
            break;
        }
        if (block.compare(pos, std::char_traits<char>::length(kParameterOpen), kParameterOpen) != 0)
            return set_error(err, "tool call contains malformed text outside parameter tags");
        const size_t key_start = pos + std::char_traits<char>::length(kParameterOpen);
        const size_t key_end = block.find('>', key_start);
        if (key_end == std::string::npos) return set_error(err, "tool call has an unterminated parameter name");
        const std::string key = block.substr(key_start, key_end - key_start);
        if (!safe_protocol_name(key))
            return set_error(err, "tool call has an invalid parameter name");
        if (arguments.contains(key)) return set_error(err, "tool call contains duplicate parameter " + key);
        const json* property_schema = property_schema_for_key(schema, properties, key);
        if (!property_schema)
            return set_error(err, "tool call contains unknown parameter " + key);
        const size_t value_start = key_end + 1;
        const size_t value_end = block.find(kParameterClose, value_start);
        if (value_end == std::string::npos) return set_error(err, "tool call has an unterminated parameter " + key);
        std::string value = block.substr(value_start, value_end - value_start);
        // The official template puts one newline on each side of the value. They are protocol
        // delimiters, not part of a string argument; preserve every other byte exactly.
        if (!value.empty() && value.front() == '\n') value.erase(value.begin());
        if (!value.empty() && value.back() == '\n') value.pop_back();
        if (has_protocol_markup(value))
            return set_error(err, "parameter " + key + " contains reserved protocol markup");
        json parsed;
        if (!parse_parameter_value(value, *property_schema, parsed, &schema))
            return set_error(err, "parameter " + key + " is not valid for its schema type");
        if (!validate_value(parsed, *property_schema, "parameter " + key, err, &schema)) return false;
        arguments[key] = std::move(parsed);
        pos = value_end + std::char_traits<char>::length(kParameterClose);
    }
    if (block.substr(pos).find_first_not_of(" \t\r\n") != std::string::npos)
        return set_error(err, "tool call contains text after </function>");
    if (!validate_value(arguments, schema, "arguments for " + call.name, err)) return false;
    call.arguments = arguments.dump();
    call.id.clear();
    return true;
}

}  // namespace

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err) {
    ChatRequest parsed;
    err.clear();
    json root;
    if (!parse_strict_json(body, root, err, "invalid JSON")) return false;
    if (!root.is_object()) return set_error(err, "request body must be a JSON object");
    if (!root.contains("messages") || !root["messages"].is_array() || root["messages"].empty())
        return set_error(err, "messages must be a non-empty array");
    for (size_t i = 0; i < root["messages"].size(); ++i) {
        ChatMessage message;
        if (!parse_message(root["messages"][i], message, i, err)) return false;
        parsed.messages.push_back(std::move(message));
    }
    if (root.contains("tools")) {
        if (!root["tools"].is_array()) return set_error(err, "tools must be an array");
        std::set<std::string> names;
        for (size_t i = 0; i < root["tools"].size(); ++i) {
            ToolDefinition tool;
            if (!parse_tool_definition(root["tools"][i], tool,
                                       "tools[" + std::to_string(i) + "]", err)) return false;
            if (!names.insert(tool.name).second) return set_error(err, "tools contains duplicate function " + tool.name);
            parsed.tools.push_back(std::move(tool));
        }
    }
    if (root.contains("tool_choice")) {
        const json& choice = root["tool_choice"];
        if (choice.is_string()) {
            const std::string value = choice.get<std::string>();
            if (value == "auto") parsed.tool_choice = ToolChoiceMode::kAuto;
            else if (value == "none") parsed.tool_choice = ToolChoiceMode::kNone;
            else if (value == "required") parsed.tool_choice = ToolChoiceMode::kRequired;
            else return set_error(err, "unsupported tool_choice " + value);
        } else if (choice.is_object()) {
            if (!is_allowed_key(choice, {"type", "function"}, "tool_choice", err)) return false;
            if (choice.value("type", "") != "function" || !choice.contains("function") ||
                !choice["function"].is_object() ||
                !is_allowed_key(choice["function"], {"name"}, "tool_choice.function", err) ||
                !choice["function"].contains("name") || !choice["function"]["name"].is_string() ||
                choice["function"]["name"].get_ref<const std::string&>().empty())
                return set_error(err, "tool_choice must name a function");
            parsed.tool_choice = ToolChoiceMode::kNamed;
            parsed.required_tool_name = choice["function"]["name"].get<std::string>();
        } else {
            return set_error(err, "tool_choice must be auto, none, required, or a named function");
        }
    }
    if (root.contains("parallel_tool_calls")) {
        if (!root["parallel_tool_calls"].is_boolean())
            return set_error(err, "parallel_tool_calls must be a boolean");
        parsed.parallel_tool_calls = root["parallel_tool_calls"].get<bool>();
    }
    if (root.contains("response_format")) {
        const json& rf = root["response_format"];
        if (!rf.is_object()) return set_error(err, "response_format must be an object");
        if (!is_allowed_key(rf, {"type", "json_schema"}, "response_format", err)) return false;
        if (!rf.contains("type") || !rf["type"].is_string())
            return set_error(err, "response_format.type must be a string");
        const std::string type = rf["type"].get<std::string>();
        if (type == "text") {
            parsed.response_format.type = ResponseFormatType::kText;
        } else if (type == "json_object") {
            parsed.response_format.type = ResponseFormatType::kJsonObject;
        } else if (type == "json_schema") {
            parsed.response_format.type = ResponseFormatType::kJsonSchema;
            if (!rf.contains("json_schema") || !rf["json_schema"].is_object())
                return set_error(err, "response_format.json_schema is required for type json_schema");
            const json& js = rf["json_schema"];
            if (!is_allowed_key(js, {"name", "schema", "strict", "description"},
                                "response_format.json_schema", err)) return false;
            if (!js.contains("name") || !js["name"].is_string() ||
                js["name"].get_ref<const std::string&>().empty())
                return set_error(err, "response_format.json_schema.name must be a non-empty string");
            if (!js.contains("schema") || !js["schema"].is_object())
                return set_error(err, "response_format.json_schema.schema is required and must be an object");
            // Reject a malformed schema at request time, before ever calling the model -- the
            // same sanity check tool `parameters` schemas already get.
            if (!valid_schema(js["schema"], "response_format.json_schema.schema", err)) return false;
            if (js.contains("strict") && !js["strict"].is_boolean())
                return set_error(err, "response_format.json_schema.strict must be boolean");
            parsed.response_format.schema_name = js["name"].get<std::string>();
            parsed.response_format.schema = js["schema"];
            parsed.response_format.strict = js.value("strict", false);
        } else {
            return set_error(err, "unsupported response_format.type " + type);
        }
    }
    // Both features are supported independently. Combining them needs a separate output grammar
    // (a tool call is not itself the requested JSON response), so reject it instead of silently
    // bypassing response_format validation.
    if (!parsed.tools.empty() && parsed.response_format.type != ResponseFormatType::kText)
        return set_error(err, "response_format is not supported together with tools");
    if (parsed.tools.empty() && parsed.tool_choice != ToolChoiceMode::kNone)
        parsed.tool_choice = ToolChoiceMode::kNone;
    std::set<std::string> offered;
    for (const ToolDefinition& tool : parsed.tools) offered.insert(tool.name);
    if (parsed.tool_choice == ToolChoiceMode::kNamed &&
        !offered.count(parsed.required_tool_name))
        return set_error(err, "tool_choice names an unoffered function " + parsed.required_tool_name);

    auto set_effort = [&](const json& value, const char* where) -> bool {
        if (!value.is_string()) return set_error(err, std::string(where) + " must be a string");
        const std::string effort = value.get<std::string>();
        if (effort != "none" && effort != "minimal" && effort != "low" &&
            effort != "medium" && effort != "high" && effort != "xhigh" && effort != "max")
            return set_error(err, std::string(where) +
                " must be none, minimal, low, medium, high, xhigh, or max");
        parsed.reasoning_effort = effort == "max" ? "xhigh" : effort;
        return true;
    };
    if (root.contains("reasoning_effort") && !root["reasoning_effort"].is_null() &&
        !set_effort(root["reasoning_effort"], "reasoning_effort")) return false;
    if (root.contains("reasoning") && !root["reasoning"].is_null()) {
        const json& reasoning = root["reasoning"];
        if (!reasoning.is_object()) return set_error(err, "reasoning must be an object");
        if (reasoning.contains("enabled")) {
            if (!reasoning["enabled"].is_boolean())
                return set_error(err, "reasoning.enabled must be a boolean");
            if (!reasoning["enabled"].get<bool>()) parsed.reasoning_effort = "none";
            else if (parsed.reasoning_effort.empty()) parsed.reasoning_effort = "medium";
        }
        if (reasoning.contains("effort") && !reasoning["effort"].is_null() &&
            !set_effort(reasoning["effort"], "reasoning.effort")) return false;
        if (reasoning.contains("max_tokens") && !reasoning["max_tokens"].is_null()) {
            if (!reasoning["max_tokens"].is_number_integer() ||
                reasoning["max_tokens"].get<long long>() <= 0)
                return set_error(err, "reasoning.max_tokens must be a positive integer");
            // OpenRouter normally converts budgets to effort for effort-only providers. Accept a
            // direct budget as medium rather than rejecting an otherwise portable request.
            if (parsed.reasoning_effort.empty()) parsed.reasoning_effort = "medium";
        }
        if (reasoning.contains("exclude")) {
            if (!reasoning["exclude"].is_boolean())
                return set_error(err, "reasoning.exclude must be a boolean");
            parsed.reasoning_exclude = reasoning["exclude"].get<bool>();
        }
    }
    for (size_t i = 0; i < parsed.messages.size(); ++i) {
        for (const ToolCall& call : parsed.messages[i].tool_calls) {
            if (!offered.count(call.name))
                return set_error(err, "messages[" + std::to_string(i) + "] references unoffered function " + call.name);
            const ToolDefinition* tool = offered_tool(parsed, call.name);
            json arguments;
            if (!parse_strict_json(call.arguments, arguments, err,
                                   "messages[" + std::to_string(i) +
                                       "].tool_calls arguments")) return false;
            for (const auto& argument : arguments.items()) {
                if (!safe_protocol_name(argument.key()))
                    return set_error(err, "messages[" + std::to_string(i) +
                                              "].tool_calls contains an unsafe parameter name");
            }
            if (!validate_value(arguments, tool->spec["function"]["parameters"],
                                "messages[" + std::to_string(i) + "].tool_calls arguments", err)) return false;
        }
    }
    if (!resolve_history(parsed, err)) return false;
    request = std::move(parsed);
    return true;
}

bool validate_response_format(const std::string& content, const ResponseFormat& format,
                              std::string& err) {
    if (format.type == ResponseFormatType::kText) return true;
    json parsed;
    if (!parse_strict_json(content, parsed, err, "response")) return false;
    if (format.type == ResponseFormatType::kJsonObject) return true;
    return validate_value(parsed, format.schema, "response", err);
}

bool parse_request_controls(const std::string& body, RequestControls& out, std::string& err,
                            int vocab, bool legacy_logprobs) {
    const auto root = json::parse(body, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        err = "request body must be a JSON object";
        return false;
    }
    if (root.contains("stream")) {
        if (!root["stream"].is_boolean()) {
            err = "stream must be a boolean";
            return false;
        }
        out.stream = root["stream"].get<bool>();
    }
    const char* max_key = root.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    if (root.contains(max_key)) {
        const auto& value = root[max_key];
        if (!value.is_number_integer() && !value.is_number_unsigned()) {
            err = std::string(max_key) + " must be a positive integer";
            return false;
        }
        try {
            const auto requested = value.get<long long>();
            if (requested <= 0 || requested > std::numeric_limits<int>::max()) {
                err = std::string(max_key) + " is outside the supported range";
                return false;
            }
            out.max_tokens = static_cast<int>(requested);
        } catch (const json::exception&) {
            err = std::string(max_key) + " is outside the supported range";
            return false;
        }
    }
    if (root.contains("stop") && !root["stop"].is_null()) {
        const auto& value = root["stop"];
        std::vector<std::string> stops;
        if (value.is_string()) {
            stops.push_back(value.get<std::string>());
        } else if (value.is_array()) {
            for (const auto& item : value) {
                if (!item.is_string()) {
                    err = "stop entries must be strings";
                    return false;
                }
                stops.push_back(item.get<std::string>());
            }
        } else {
            err = "stop must be a string or an array of strings";
            return false;
        }
        if (stops.size() > 4) {
            err = "stop supports at most 4 strings";
            return false;
        }
        for (const auto& s : stops) {
            if (s.empty()) {
                err = "stop entries must be non-empty strings";
                return false;
            }
        }
        out.stop = std::move(stops);
    }
    if (root.contains("stream_options")) {
        if (!root["stream_options"].is_object()) {
            err = "stream_options must be an object";
            return false;
        }
        const auto& opts = root["stream_options"];
        if (opts.contains("include_usage")) {
            if (!opts["include_usage"].is_boolean()) {
                err = "stream_options.include_usage must be a boolean";
                return false;
            }
            out.include_usage = opts["include_usage"].get<bool>();
        }
    }
    if (root.contains("temperature") && !root["temperature"].is_null()) {
        const auto& value = root["temperature"];
        if (!value.is_number()) {
            err = "temperature must be a number";
            return false;
        }
        const double t = value.get<double>();
        if (!(t >= 0.0) || !(t <= 2.0)) {  // NaN-safe: comparisons against NaN are false either way
            err = "temperature must be between 0.0 and 2.0";
            return false;
        }
        out.temperature = static_cast<float>(t);
    }
    if (root.contains("seed") && !root["seed"].is_null()) {
        const auto& value = root["seed"];
        if (!value.is_number_integer() && !value.is_number_unsigned()) {
            err = "seed must be an integer";
            return false;
        }
        try {
            const auto requested = value.get<long long>();
            if (requested < 0) {
                err = "seed must be non-negative";
                return false;
            }
            out.seed = static_cast<uint64_t>(requested);
            out.seed_set = true;
        } catch (const json::exception&) {
            err = "seed is outside the supported range";
            return false;
        }
    }
    if (root.contains("top_p") && !root["top_p"].is_null()) {
        const auto& value = root["top_p"];
        if (!value.is_number()) {
            err = "top_p must be a number";
            return false;
        }
        const double p = value.get<double>();
        if (!(p >= 0.0) || !(p <= 1.0)) {  // NaN-safe: comparisons against NaN are false either way
            err = "top_p must be between 0.0 and 1.0";
            return false;
        }
        out.top_p = static_cast<float>(p);
    }
    if (root.contains("top_k") && !root["top_k"].is_null()) {
        const auto& value = root["top_k"];
        if (!value.is_number_integer()) {
            err = "top_k must be an integer";
            return false;
        }
        const long long k = value.get<long long>();
        if (k < 0) {
            err = "top_k must be non-negative";
            return false;
        }
        out.top_k = static_cast<int>(std::min<long long>(k, std::numeric_limits<int>::max()));
    }
    if (root.contains("presence_penalty") && !root["presence_penalty"].is_null()) {
        const auto& value = root["presence_penalty"];
        if (!value.is_number()) {
            err = "presence_penalty must be a number";
            return false;
        }
        const double p = value.get<double>();
        if (!(p >= -2.0) || !(p <= 2.0)) {  // NaN-safe: comparisons against NaN are false either way
            err = "presence_penalty must be between -2.0 and 2.0";
            return false;
        }
        out.presence_penalty = static_cast<float>(p);
    }
    if (root.contains("frequency_penalty") && !root["frequency_penalty"].is_null()) {
        const auto& value = root["frequency_penalty"];
        if (!value.is_number()) {
            err = "frequency_penalty must be a number";
            return false;
        }
        const double f = value.get<double>();
        if (!(f >= -2.0) || !(f <= 2.0)) {
            err = "frequency_penalty must be between -2.0 and 2.0";
            return false;
        }
        out.frequency_penalty = static_cast<float>(f);
    }
    if (root.contains("logprobs") && !root["logprobs"].is_null()) {
        if (legacy_logprobs) {
            // Legacy /v1/completions: logprobs itself is an integer ("how many top logprobs per
            // token"), not a boolean -- there is no separate top_logprobs field in this mode.
            if (!root["logprobs"].is_number_integer()) {
                err = "logprobs must be an integer";
                return false;
            }
            const long long lp = root["logprobs"].get<long long>();
            if (lp < 0 || lp > 20) {
                err = "logprobs must be between 0 and 20";
                return false;
            }
            out.logprobs = lp > 0;
            out.top_logprobs = static_cast<int>(lp);
        } else {
            if (!root["logprobs"].is_boolean()) {
                err = "logprobs must be a boolean";
                return false;
            }
            out.logprobs = root["logprobs"].get<bool>();
        }
    }
    if (!legacy_logprobs && root.contains("top_logprobs") && !root["top_logprobs"].is_null()) {
        const auto& value = root["top_logprobs"];
        if (!value.is_number_integer()) {
            err = "top_logprobs must be an integer";
            return false;
        }
        const long long n = value.get<long long>();
        if (n < 0 || n > 20) {
            err = "top_logprobs must be between 0 and 20";
            return false;
        }
        if (!out.logprobs) {
            err = "top_logprobs requires logprobs to be true";
            return false;
        }
        out.top_logprobs = static_cast<int>(n);
    }
    if (root.contains("logit_bias") && !root["logit_bias"].is_null()) {
        const auto& value = root["logit_bias"];
        if (!value.is_object()) {
            err = "logit_bias must be an object";
            return false;
        }
        if (value.size() > (size_t)kMaxLogitBiasEntries) {
            err = "logit_bias supports at most " + std::to_string(kMaxLogitBiasEntries) + " entries";
            return false;
        }
        std::vector<std::pair<int, float>> bias;
        bias.reserve(value.size());
        for (const auto& entry : value.items()) {
            const std::string& key = entry.key();
            const auto& v = entry.value();
            if (!v.is_number()) {
                err = "logit_bias values must be numbers";
                return false;
            }
            const double b = v.get<double>();
            if (!(b >= -100.0) || !(b <= 100.0)) {  // NaN-safe: comparisons against NaN are false either way
                err = "logit_bias values must be between -100 and 100";
                return false;
            }
            // Strict parse: the ENTIRE key must be a base-10 non-negative integer -- from_chars
            // reports how far it got via `ptr`, so "12a"/"1.5"/" 12"/"" all fail the full-consumption
            // check below (from_chars also rejects a leading '+' and whitespace on its own).
            int id = 0;
            const auto res = std::from_chars(key.data(), key.data() + key.size(), id);
            if (res.ec != std::errc() || res.ptr != key.data() + key.size() || id < 0) {
                err = "logit_bias keys must be non-negative integer token ids";
                return false;
            }
            if (vocab > 0 && id >= vocab) {
                err = "logit_bias key " + key + " is outside the model's vocabulary";
                return false;
            }
            bias.emplace_back(id, static_cast<float>(b));
        }
        out.logit_bias = std::move(bias);
    }
    if (root.contains("n") && !root["n"].is_null()) {
        const auto& value = root["n"];
        if (!value.is_number_integer()) {
            err = "n must be an integer";
            return false;
        }
        const long long n = value.get<long long>();
        if (n < 1 || n > (long long)kMaxN) {
            err = "n must be between 1 and " + std::to_string(kMaxN);
            return false;
        }
        out.n = (int)n;
    }
    return true;
}

bool parse_legacy_completion_request(const std::string& body, std::string& prompt_out,
                                     bool& echo_out, std::string& err) {
    prompt_out.clear();
    echo_out = false;
    const auto root = json::parse(body, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        err = "request body must be a JSON object";
        return false;
    }
    if (!root.contains("prompt") || root["prompt"].is_null()) {
        err = "prompt is required";
        return false;
    }
    if (!root["prompt"].is_string()) {
        // OpenAI's own API also accepts an array of strings (batched independent prompts) or
        // pre-tokenized integer arrays -- both out of scope for v1, rejected explicitly rather
        // than silently only handling the first entry.
        err = "prompt must be a string";
        return false;
    }
    const std::string prompt = root["prompt"].get<std::string>();
    if (prompt.empty()) {
        err = "prompt must not be empty";
        return false;
    }
    if (root.contains("echo") && !root["echo"].is_null()) {
        if (!root["echo"].is_boolean()) {
            err = "echo must be a boolean";
            return false;
        }
        echo_out = root["echo"].get<bool>();
    }
    if (root.contains("suffix") && !root["suffix"].is_null()) {
        err = "suffix is not supported";
        return false;
    }
    if (root.contains("best_of") && !root["best_of"].is_null()) {
        if (!root["best_of"].is_number_integer()) {
            err = "best_of must be an integer";
            return false;
        }
        const long long bo = root["best_of"].get<long long>();
        // OpenAI defines best_of as a positive integer (default 1); 0/negative are never valid
        // regardless of the >1-is-unsupported restriction below.
        if (bo < 1) {
            err = "best_of must be a positive integer";
            return false;
        }
        if (bo > 1) {
            err = "best_of > 1 is not supported";
            return false;
        }
    }
    prompt_out = prompt;
    return true;
}

bool parse_score_request(const std::string& body, ScoreRequest& out, std::string& err, int vocab) {
    out = ScoreRequest{};
    json root;
    try {
        root = json::parse(body);
    } catch (const std::exception&) {
        err = "invalid JSON body";
        return false;
    }
    if (!root.is_object()) {
        err = "body must be a JSON object";
        return false;
    }

    const bool has_messages = root.contains("messages") && !root["messages"].is_null();
    const bool has_prompt = root.contains("prompt") && !root["prompt"].is_null();
    if (has_messages == has_prompt) {
        err = "provide exactly one of `messages` or `prompt`";
        return false;
    }
    out.use_messages = has_messages;
    if (has_prompt) {
        if (!root["prompt"].is_string()) {
            err = "prompt must be a string";
            return false;
        }
        out.prompt = root["prompt"].get<std::string>();
        if (out.prompt.empty()) {
            err = "prompt is empty";
            return false;
        }
    }

    const bool has_ids = root.contains("completion_token_ids") && !root["completion_token_ids"].is_null();
    const bool has_text = root.contains("completion") && !root["completion"].is_null();
    if (has_ids == has_text) {
        err = "provide exactly one of `completion` or `completion_token_ids`";
        return false;
    }
    if (has_ids) {
        if (!root["completion_token_ids"].is_array()) {
            err = "completion_token_ids must be an array of integers";
            return false;
        }
        const auto& arr = root["completion_token_ids"];
        if (arr.empty()) {
            err = "completion_token_ids is empty (no tokens to score)";
            return false;
        }
        out.completion_token_ids.reserve(arr.size());
        for (const auto& v : arr) {
            if (!v.is_number_integer() && !v.is_number_unsigned()) {
                err = "completion_token_ids must contain only integers";
                return false;
            }
            const long long t = v.get<long long>();
            if (t < 0 || (vocab > 0 && t >= (long long)vocab)) {
                err = "completion_token_ids entry out of range: " + std::to_string(t);
                return false;
            }
            out.completion_token_ids.push_back((int)t);
        }
        out.completion_is_ids = true;
    } else {
        if (!root["completion"].is_string()) {
            err = "completion must be a string";
            return false;
        }
        out.completion = root["completion"].get<std::string>();
        if (out.completion.empty()) {
            err = "completion is empty (no tokens to score)";
            return false;
        }
    }

    if (root.contains("top_logprobs") && !root["top_logprobs"].is_null()) {
        if (!root["top_logprobs"].is_number_integer()) {
            err = "top_logprobs must be an integer";
            return false;
        }
        const long long n = root["top_logprobs"].get<long long>();
        if (n < 0 || n > kMaxTopLogprobs) {
            err = "top_logprobs must be between 0 and " + std::to_string(kMaxTopLogprobs);
            return false;
        }
        out.top_logprobs = (int)n;
    }
    return true;
}

bool should_reject_dflash_temperature(bool dflash_env_on, float temperature) {
    return dflash_env_on && temperature > 0.f;
}

bool should_reject_dflash_penalty(bool dflash_env_on, float presence_penalty, float frequency_penalty) {
    return dflash_env_on && (presence_penalty != 0.f || frequency_penalty != 0.f);
}

bool should_reject_dflash_logit_bias(bool dflash_env_on, bool has_logit_bias) {
    return dflash_env_on && has_logit_bias;
}

std::string apply_qwen36_tools_template(const ChatRequest& request, bool enable_thinking,
                                        bool inject_reasoning_effort) {
    std::ostringstream out;
    size_t first_message = 0;
    const bool tools_active = !request.tools.empty() && request.tool_choice != ToolChoiceMode::kNone;
    const bool json_mode = request.response_format.type != ResponseFormatType::kText;
    const std::string effort = request.reasoning_effort.empty() ? "xhigh" : request.reasoning_effort;
    const std::string reasoning_instructions = (inject_reasoning_effort && enable_thinking)
        ? "Reasoning effort is set to " + effort + ". Please think carefully through the task, validate "
          "key assumptions, consider plausible alternatives, and prioritize correctness, "
          "consistency, and clarity in the final answer."
        : std::string();
    const bool has_leading_system = !request.messages.empty() && request.messages[0].role == "system";
    // Request-time validation (parse_chat_request_json) rejects tools + response_format
    // together, so tools_active and json_mode are never both true -- written as two independent
    // segments anyway to keep this function's shape uniform rather than forking it in two.
    if (tools_active || json_mode) {
        out << kImStart << "system\n";
        if (!reasoning_instructions.empty()) out << reasoning_instructions << "\n\n";
        if (tools_active) {
            out << kToolInstructions;
            for (const ToolDefinition& tool : request.tools)
                out << '\n' << qwen_template_json(tool.spec);
            out << kToolInstructionsTail;
            if (request.tool_choice == ToolChoiceMode::kRequired)
                out << "\n\nYou MUST call at least one of the offered functions.";
            else if (request.tool_choice == ToolChoiceMode::kNamed)
                out << "\n\nYou MUST call the function named " << request.required_tool_name << ".";
            if (!request.parallel_tool_calls)
                out << "\nCall at most one function.";
        }
        if (json_mode) {
            if (tools_active) out << "\n\n";
            if (request.response_format.type == ResponseFormatType::kJsonObject) {
                out << kJsonObjectInstructions;
            } else {
                out << kJsonSchemaInstructionsHead << request.response_format.schema_name
                    << kJsonSchemaInstructionsMid << qwen_template_json(request.response_format.schema)
                    << kJsonSchemaInstructionsTail;
            }
        }
        if (has_leading_system) {
            const std::string system_content = trim_copy(request.messages[0].content);
            if (!system_content.empty()) out << "\n\n" << system_content;
        }
        out << kImEnd << '\n';
        if (has_leading_system) first_message = 1;
    } else if (inject_reasoning_effort && (!reasoning_instructions.empty() || has_leading_system)) {
        // Plain case (no tools, no json_mode): the pinned template still merges leading system
        // message(s) with the reasoning-effort instructions rather than letting them fall
        // through the generic per-message loop below.
        const std::string system_content = has_leading_system ? trim_copy(request.messages[0].content)
                                                               : std::string();
        if (!system_content.empty() || !reasoning_instructions.empty()) {
            out << kImStart << "system\n";
            if (!reasoning_instructions.empty()) {
                out << reasoning_instructions;
                if (!system_content.empty()) out << "\n\n" << system_content;
            } else {
                out << system_content;
            }
            out << kImEnd << '\n';
        }
        if (has_leading_system) first_message = 1;
    }
    size_t last_user = request.messages.size();
    for (size_t i = request.messages.size(); i > 0; --i) {
        if (request.messages[i - 1].role == "user") {
            last_user = i - 1;
            break;
        }
    }
    for (size_t i = first_message; i < request.messages.size(); ++i) {
        const ChatMessage& message = request.messages[i];
        if (message.role == "tool") {
            out << kImStart << "user";
            do {
                out << '\n' << kToolResponseOpen << '\n' << trim_copy(request.messages[i].content)
                    << '\n' << kToolResponseClose;
                ++i;
            } while (i < request.messages.size() && request.messages[i].role == "tool");
            out << kImEnd << '\n';
            --i;
            continue;
        }
        out << kImStart << message.role << '\n';
        if (message.role == "assistant") {
            std::string content = trim_copy(message.content);
            std::string reasoning = trim_copy(message.reasoning_content);
            // Match the pinned tokenizer template's compatibility path for clients which put
            // a previous turn's reasoning and answer together in content instead of sending
            // reasoning_content separately.
            if (reasoning.empty()) {
                const size_t first_close = content.find(kThinkClose);
                if (first_close != std::string::npos) {
                    std::string embedded_reasoning = content.substr(0, first_close);
                    const size_t last_open = embedded_reasoning.rfind(kThinkOpen);
                    if (last_open != std::string::npos)
                        embedded_reasoning.erase(0, last_open +
                                                        std::char_traits<char>::length(kThinkOpen));
                    reasoning = trim_copy(std::move(embedded_reasoning));
                    const size_t last_close = content.rfind(kThinkClose);
                    content = trim_copy(content.substr(last_close +
                                                       std::char_traits<char>::length(kThinkClose)));
                }
            }
            // The pinned template preserves reasoning only for assistant/tool steps after the
            // latest real user query. Older chain-of-thought is intentionally not replayed.
            if (i > last_user)
                out << kThinkOpen << '\n' << reasoning << '\n' << kThinkClose << "\n\n";
            if (!message.content_is_null) out << content;
            for (size_t j = 0; j < message.tool_calls.size(); ++j) {
                if (j == 0 && !content.empty()) out << "\n\n";
                else if (j > 0) out << '\n';
                out << render_tool_call(message.tool_calls[j]);
            }
        } else {
            out << trim_copy(message.content);
        }
        out << kImEnd << '\n';
    }
    out << kImStart << "assistant\n";
    if (enable_thinking) out << kThinkOpen << '\n';
    else out << kThinkOpen << "\n\n" << kThinkClose << "\n\n";
    out << request.assistant_prefix;
    return out.str();
}

namespace {

// Structural-tag JSON builders. Kept next to the parser on purpose: every string below is a
// delimiter parse_qwen36_tool_output splits on, and the two must change together.
json st_const(const std::string& value) { return {{"type", "const_string"}, {"value", value}}; }

// Free text. Excludes exactly what has_protocol_markup rejects, so no free-text region -- reasoning,
// content, a string argument -- can produce output the parser refuses.
json st_free_text(int max_chars = -1) {
    json out = {{"type", "any_text"},
                {"excludes", {"<tool", "</tool", "<function", "</function", "<parameter", "</parameter",
                              "<think", "</think", "<|im_"}}};
    if (max_chars >= 0) out["max_chars"] = max_chars;
    return out;
}

json st_tag(const std::string& begin, json content, const std::string& end) {
    return {{"type", "tag"}, {"begin", begin}, {"content", std::move(content)}, {"end", end}};
}

json st_sequence(json elements) { return {{"type", "sequence"}, {"elements", std::move(elements)}}; }

json st_regex(const std::string& pattern) { return {{"type", "regex"}, {"pattern", pattern}}; }

// ---- Schema normalization for the grammar -------------------------------------------------------
//
// xgrammar enforces most of JSON Schema, but not all of what validate_value enforces: allOf with more
// than one branch becomes "anything", multipleOf combined with a range is dropped, and an oneOf whose
// branches may overlap is treated as anyOf. Rewrite those into forms it does enforce, and record an
// approximation wherever that is not possible, so ToolCallGrammar::exact stays truthful.

json resolve_refs_shallow(const json& root, const json& schema) {
    const json* s = &schema;
    for (int depth = 0; depth < 64 && s->is_object() && s->contains("$ref") && (*s)["$ref"].is_string(); ++depth) {
        const json* next = resolve_local_ref(root, (*s)["$ref"].get<std::string>());
        if (!next) break;
        s = next;
    }
    return *s;
}

std::set<std::string> schema_type_set(const json& s) {
    std::set<std::string> out;
    if (!s.is_object() || !s.contains("type")) return {"array", "boolean", "integer", "null", "number", "object", "string"};
    const json& type = s["type"];
    if (type.is_string()) out.insert(type.get<std::string>());
    else if (type.is_array())
        for (const auto& t : type)
            if (t.is_string()) out.insert(t.get<std::string>());
    if (out.count("number")) out.insert("integer");
    return out;
}

bool approximate(ToolCallGrammar& grammar, const std::string& why);

json merge_all_of(const json& root, json a, const json& b_in, ToolCallGrammar& grammar, const std::string& where);

// xml_framing: the value travels inside the Qwen XML tool-call protocol, where a '<' it carries
// verbatim can break the framing. False for response_format output, which has no framing.
json normalize_for_grammar(const json& root, const json& schema, ToolCallGrammar& grammar, const std::string& where,
                           int depth = 0, bool xml_framing = true) {
    if (!schema.is_object() || depth > 32) return schema;
    json s = schema;
    if (s.contains("allOf") && s["allOf"].is_array()) {
        json branches = s["allOf"];
        s.erase("allOf");
        for (const auto& branch : branches)
            s = merge_all_of(root, std::move(s), normalize_for_grammar(root, resolve_refs_shallow(root, branch), grammar, where, depth + 1, xml_framing), grammar, where);
    }
    for (const char* key : {"items", "additionalProperties"})
        if (s.contains(key) && s[key].is_object()) s[key] = normalize_for_grammar(root, s[key], grammar, where, depth + 1, xml_framing);
    for (const char* key : {"anyOf", "oneOf", "prefixItems"})
        if (s.contains(key) && s[key].is_array())
            for (auto& item : s[key]) item = normalize_for_grammar(root, item, grammar, where, depth + 1, xml_framing);
    if (s.contains("properties") && s["properties"].is_object())
        for (auto& item : s["properties"].items()) item.value() = normalize_for_grammar(root, item.value(), grammar, where, depth + 1, xml_framing);
    // oneOf is anyOf when no value can satisfy two branches; types that cannot overlap prove it.
    if (s.contains("oneOf") && s["oneOf"].is_array()) {
        std::set<std::string> seen;
        bool disjoint = true;
        for (const auto& branch : s["oneOf"]) {
            std::set<std::string> types = schema_type_set(resolve_refs_shallow(root, branch));
            for (const auto& t : types)
                if (!seen.insert(t).second) disjoint = false;
        }
        if (!disjoint) approximate(grammar, where + ": oneOf branches may overlap");
    }
    // An integer multipleOf inside a finite range is a finite set.
    if (s.contains("multipleOf")) {
        const json& m = s["multipleOf"];
        const std::set<std::string> types = schema_type_set(s);
        const bool integer_only = types.size() == 1 && types.count("integer") == 1 &&
                                  s.contains("type");
        long lo = LONG_MIN, hi = LONG_MAX;
        if (s.contains("minimum") && s["minimum"].is_number()) lo = (long)std::ceil(s["minimum"].get<double>());
        if (s.contains("exclusiveMinimum") && s["exclusiveMinimum"].is_number()) lo = std::max(lo, (long)std::floor(s["exclusiveMinimum"].get<double>()) + 1);
        if (s.contains("maximum") && s["maximum"].is_number()) hi = (long)std::floor(s["maximum"].get<double>());
        if (s.contains("exclusiveMaximum") && s["exclusiveMaximum"].is_number()) hi = std::min(hi, (long)std::ceil(s["exclusiveMaximum"].get<double>()) - 1);
        if (integer_only && m.is_number_integer() && m.get<long>() > 0 && lo != LONG_MIN && hi != LONG_MAX &&
            hi >= lo && (hi - lo) / m.get<long>() <= 1024) {
            const long step = m.get<long>();
            long first = lo % step == 0 ? lo : lo + ((step - lo % step) % step);
            if (lo < 0 && lo % step != 0) first = lo - (lo % step);
            json values = json::array();
            for (long v = first; v <= hi; v += step)
                if (v >= lo && v % step == 0) values.push_back(v);
            if (s.contains("enum") || s.contains("const")) approximate(grammar, where + ": multipleOf with enum or const");
            else s = json{{"enum", std::move(values)}};
        } else {
            approximate(grammar, where + ": multipleOf the grammar cannot enumerate");
        }
    }
    // validate_value reads a JSON integer only when it fits 64 bits (a longer one parses as a double
    // and fails "type": "integer"); xgrammar's integer rule has unbounded digits. Bound it where the
    // schema does not.
    if (s.contains("type") && !s.contains("enum") && !s.contains("const")) {
        const std::set<std::string> types = schema_type_set(s);
        const json& type = s["type"];
        const bool integer_only = type.is_string() && type == "integer";
        if (integer_only || (types.count("integer") && !(type.is_array() && std::find(type.begin(), type.end(), json("number")) != type.end()))) {
            if (!s.contains("minimum") && !s.contains("exclusiveMinimum")) s["minimum"] = std::numeric_limits<int64_t>::min();
            if (!s.contains("maximum") && !s.contains("exclusiveMaximum")) s["maximum"] = std::numeric_limits<int64_t>::max();
        }
    }
    // Strings a JSON value would carry verbatim: markup there breaks the protocol framing.
    for (const char* key : {"enum", "const"}) {
        if (!xml_framing || !s.contains(key)) continue;
        const json values = std::string(key) == "enum" ? s[key] : json::array({s[key]});
        for (const auto& v : values)
            if (v.is_string() && v.get<std::string>().find('<') != std::string::npos)
                approximate(grammar, where + ": an enum or const string contains '<'");
    }
    if (s.contains("pattern")) approximate(grammar, where + ": pattern inside a JSON value");
    // An object that declares no properties takes any keys. xgrammar's strict mode would narrow it to
    // {}, so allow them explicitly -- but no grammar can stop a key from repeating, which the strict
    // JSON reader refuses.
    if (schema_type_set(s).count("object") && s.contains("type") && !s.contains("properties") &&
        !s.contains("additionalProperties")) {
        s["additionalProperties"] = true;
        approximate(grammar, where + ": a free-form object can repeat a key");
    } else if (s.contains("additionalProperties") && !(s["additionalProperties"].is_boolean() && !s["additionalProperties"].get<bool>())) {
        approximate(grammar, where + ": additional properties can repeat a key");
    }
    return s;
}

// allOf as one schema: the conjunction of each keyword. Where a conjunction has no single-keyword form
// (two different patterns, two prefixItems) or changes meaning (additionalProperties, which each branch
// applies to its own properties), keep the first and record the approximation.
json merge_all_of(const json& root, json a, const json& b, ToolCallGrammar& grammar, const std::string& where) {
    (void)root;
    if (!b.is_object()) return a;
    for (const auto& item : b.items()) {
        const std::string& key = item.key();
        const json& bv = item.value();
        if (!a.contains(key)) {
            a[key] = bv;
            continue;
        }
        json& av = a[key];
        if (av == bv) continue;
        if (key == "type") {
            std::set<std::string> ta = schema_type_set(json{{"type", av}}), tb = schema_type_set(json{{"type", bv}}), both;
            for (const auto& t : ta)
                if (tb.count(t)) both.insert(t);
            if (both.count("number") && !(ta.count("number") && tb.count("number"))) both.erase("number");
            if (both.empty()) {
                approximate(grammar, where + ": allOf types do not intersect");
                continue;
            }
            av = both.size() == 1 ? json(*both.begin()) : json(std::vector<std::string>(both.begin(), both.end()));
        } else if (key == "minimum" || key == "exclusiveMinimum" || key == "minLength" || key == "minItems") {
            av = std::max(av.get<double>(), bv.get<double>());
            if (key == "minLength" || key == "minItems") av = (long)av.get<double>();
        } else if (key == "maximum" || key == "exclusiveMaximum" || key == "maxLength" || key == "maxItems") {
            av = std::min(av.get<double>(), bv.get<double>());
            if (key == "maxLength" || key == "maxItems") av = (long)av.get<double>();
        } else if (key == "required") {
            std::set<std::string> keys;
            for (const auto& k : av) keys.insert(k.get<std::string>());
            for (const auto& k : bv) keys.insert(k.get<std::string>());
            av = std::vector<std::string>(keys.begin(), keys.end());
        } else if (key == "properties") {
            for (const auto& prop : bv.items())
                av[prop.key()] = av.contains(prop.key())
                    ? merge_all_of(root, av[prop.key()], prop.value(), grammar, where)
                    : prop.value();
        } else if (key == "items") {
            av = merge_all_of(root, av, bv, grammar, where);
        } else if (key == "enum") {
            json kept = json::array();
            for (const auto& v : av)
                if (std::find(bv.begin(), bv.end(), v) != bv.end()) kept.push_back(v);
            av = kept;
        } else if (key == "description" || key == "title" || key == "default" || key == "examples" ||
                   key == "format" || key == "$comment" || key.rfind("x-", 0) == 0) {
            // annotations: keep the first
        } else if (key == "multipleOf" && av.is_number_integer() && bv.is_number_integer()) {
            long x = av.get<long>(), y = bv.get<long>();
            long g = x, h = y;
            while (h) { long r = g % h; g = h; h = r; }
            av = x / g * y;
        } else {
            approximate(grammar, where + ": allOf cannot combine two different " + key);
        }
    }
    return a;
}

// validate_value applies a JSON Schema pattern with RE2::PartialMatch -- it may match anywhere in the
// value -- while an xgrammar regex must match all of it. Pad each unanchored side with text that
// cannot start protocol markup. False when the rewrite would not be exact: an anchor anywhere but the
// two ends, or a top-level alternation mixed with anchors.
//
// One dialect difference is rewritten rather than refused: RE2's '.' does not match a newline and
// xgrammar's does, so '.' outside a character class becomes [^\n].
bool whole_value_regex(const std::string& pattern, std::string& out) {
    int depth = 0;
    bool in_class = false, top_level_alternation = false, inner_anchor = false;
    std::string rewritten;
    for (size_t i = 0; i < pattern.size(); ++i) {
        const char c = pattern[i];
        if (c == '\\') {
            if (i + 1 < pattern.size() && std::strchr("AzZbB", pattern[i + 1])) inner_anchor = true;
            rewritten.append(pattern, i, 2);
            ++i;
            continue;
        }
        if (in_class) {
            if (c == ']') in_class = false;
            rewritten.push_back(c);
            continue;
        }
        if (c == '.') {
            rewritten += "[^\\n]";
            continue;
        }
        rewritten.push_back(c);
        if (c == '[') in_class = true;
        else if (c == '(') ++depth;
        else if (c == ')') --depth;
        else if (c == '|' && depth == 0) top_level_alternation = true;
        else if ((c == '^' && i != 0) || (c == '$' && i + 1 != pattern.size())) inner_anchor = true;
    }
    const bool anchored_start = !pattern.empty() && pattern.front() == '^';
    bool anchored_end = pattern.size() > 1 && pattern.back() == '$';
    if (anchored_end) {   // "\$" is a literal dollar, not an anchor
        size_t slashes = 0;
        for (size_t i = pattern.size() - 1; i > 0 && pattern[i - 1] == '\\'; --i) ++slashes;
        if (slashes % 2) anchored_end = false;
    }
    if (inner_anchor || (top_level_alternation && (anchored_start || anchored_end))) return false;
    const std::string core = rewritten.substr(anchored_start ? 1 : 0,
                                              rewritten.size() - (anchored_start ? 1 : 0) - (anchored_end ? 1 : 0));
    out = (anchored_start ? "" : "[^<]*") + std::string("(?:") + core + ")" + (anchored_end ? "" : "[^<]*");
    return true;
}

bool approximate(ToolCallGrammar& grammar, const std::string& why) {
    if (grammar.exact) grammar.approximation = why;
    grammar.exact = false;
    return true;
}

// The value between "<parameter=KEY>\n" and "\n</parameter>\n", as parse_parameter_value reads it.
bool parameter_value_format(const json& root, const json& property, const std::string& where,
                            ToolCallGrammar& grammar, json& out, std::string& err) {
    const json* resolved = &property;
    for (int depth = 0; resolved->is_object() && resolved->contains("$ref"); ++depth) {
        const json& ref = (*resolved)["$ref"];
        const json* next = depth < 64 && ref.is_string() ? resolve_local_ref(root, ref.get<std::string>()) : nullptr;
        if (!next) return set_error(err, where + " has an unresolvable $ref");
        resolved = next;
    }
    const json& s = *resolved;
    if (!(schema_allows_type(s, "string") && !schema_allows_non_string(s))) {
        // Anything that may be a non-string is read as JSON first, and JSON that fails the schema is
        // never a valid string either unless the schema also allows strings -- in which case the
        // JSON-quoted form still parses to that string. So the JSON grammar of the schema is exact.
        json sub = normalize_for_grammar(root, property, grammar, where);
        for (const char* defs : {"$defs", "definitions"}) {
            if (!root.contains(defs) || sub.contains(defs)) continue;
            json normalized = root[defs];
            if (normalized.is_object())
                for (auto& item : normalized.items())
                    item.value() = normalize_for_grammar(root, item.value(), grammar, where + " " + defs + "/" + item.key());
            sub[defs] = std::move(normalized);
        }
        out = {{"type", "json_schema"}, {"json_schema", std::move(sub)}};
        return true;
    }
    // A string-only parameter is its raw text.
    if (s.contains("const") || s.contains("enum")) {
        json values = s.contains("const") ? json::array({s["const"]}) : s["enum"];
        json choices = json::array();
        for (const auto& v : values)
            if (v.is_string() && !has_protocol_markup(v.get<std::string>()))
                choices.push_back(st_const(v.get<std::string>()));
        if (choices.empty()) return set_error(err, where + " has no value the tool-call protocol can carry");
        out = choices.size() == 1 ? choices[0] : json{{"type", "or"}, {"elements", std::move(choices)}};
        return true;
    }
    const long min_length = s.contains("minLength") ? s["minLength"].get<long>() : 0;
    const long max_length = s.contains("maxLength") ? s["maxLength"].get<long>() : -1;
    if (s.contains("pattern")) {
        std::string regex;
        if (!whole_value_regex(s["pattern"].get<std::string>(), regex)) {
            approximate(grammar, where + ": pattern cannot be matched against the whole value exactly");
            out = st_free_text();
            return true;
        }
        if (min_length > 0 || max_length >= 0)
            approximate(grammar, where + ": pattern combined with length bounds");
        out = st_regex(regex);
        return true;
    }
    if (min_length == 0 && max_length < 0) {
        out = st_free_text();
        return true;
    }
    // Length bounds count code points, as validate_value does. A regex character class counts whole
    // code points; any_text's max_chars does not -- it spends the budget on a lead byte and then
    // refuses the continuation byte, a dead end once invalid UTF-8 is masked. Excluding '<' keeps the
    // value free of markup, so a length-limited string cannot carry a '<'.
    if (min_length <= 4096 && max_length <= 65536) {
        out = st_regex("[^<]{" + std::to_string(min_length) + "," +
                       (max_length >= 0 ? std::to_string(max_length) : std::string()) + "}");
        return true;
    }
    approximate(grammar, where + ": minLength too large to expand");
    out = st_free_text();
    return true;
}

bool tool_call_format(const ToolDefinition& tool, ToolCallGrammar& grammar, json& out, std::string& err) {
    const json& schema = tool.spec["function"]["parameters"];
    const json properties = schema.value("properties", json::object());
    std::set<std::string> required;
    if (schema.contains("required"))
        for (const auto& key : schema["required"]) required.insert(key.get<std::string>());
    std::set<std::string> keys = required;
    for (const auto& item : properties.items()) keys.insert(item.key());
    json elements = json::array();
    // Declared parameters in sorted order -- the order the tool schema is rendered in the prompt.
    for (const std::string& key : keys) {
        const json* property = property_schema_for_key(schema, properties, key);
        if (!property) return set_error(err, "function " + tool.name + " requires undeclared parameter " + key);
        json value;
        if (!parameter_value_format(schema, *property, "function " + tool.name + " parameter " + key,
                                    grammar, value, err))
            return false;
        // "<parameter=KEY>\n" VALUE "\n" "</parameter>\n": the template's framing, one newline on each
        // side of the value, which parse_one_xml_call strips. The newline before the closing tag is
        // part of the content rather than of the end string on purpose: xgrammar enforces a
        // free-text exclusion only against an end string that begins with the excluded markup, and
        // with "\n</parameter>\n" as the end it let a value run on past "\n</parameter>".
        json parameter = st_tag(std::string(kParameterOpen) + key + ">\n",
                                st_sequence({std::move(value), st_const("\n")}),
                                std::string(kParameterClose) + "\n");
        elements.push_back(required.count(key) ? std::move(parameter)
                                               : json{{"type", "optional"}, {"content", std::move(parameter)}});
    }
    out = st_tag(std::string(kToolCallOpen) + "\n" + kFunctionOpen + tool.name + ">\n",
                 elements.empty() ? st_const("") : st_sequence(std::move(elements)),
                 std::string(kFunctionClose) + "\n" + kToolCallClose);
    return true;
}

}  // namespace

bool build_tool_call_grammar(const ChatRequest& request, bool enable_thinking, ToolCallGrammar& out,
                             std::string& err) {
    out = ToolCallGrammar{};
    if (request.tools.empty() || request.tool_choice == ToolChoiceMode::kNone)
        return set_error(err, "the request has no tool calls to constrain");
    json calls = json::array();
    for (const ToolDefinition& tool : request.tools) {
        if (request.tool_choice == ToolChoiceMode::kNamed && tool.name != request.required_tool_name) continue;
        json call;
        if (!tool_call_format(tool, out, call, err)) return false;
        calls.push_back(std::move(call));
    }
    if (calls.empty()) return set_error(err, "tool_choice names no offered function");
    const bool demand = request.tool_choice == ToolChoiceMode::kRequired ||
                        request.tool_choice == ToolChoiceMode::kNamed;
    json call_list = {{"type", "tags_with_separator"}, {"tags", std::move(calls)}, {"separator", "\n"},
                      {"at_least_one", true}, {"stop_after_first", !request.parallel_tool_calls}};
    // Required and named: nothing but calls. Auto: content first, then optionally calls, and nothing
    // after them -- the parser rejects text that follows a call.
    json body = demand ? std::move(call_list)
                       : st_sequence({st_free_text(), {{"type", "optional"}, {"content", std::move(call_list)}}});
    // Thinking on: the prompt ends inside <think>; the reasoning closes as the template renders it.
    if (enable_thinking) body = st_sequence({st_tag("", st_free_text(), std::string(kThinkClose) + "\n\n"), std::move(body)});
    out.structural_tag = json{{"type", "structural_tag"}, {"format", std::move(body)}}.dump();
    return true;
}

namespace {

// parse_assistant_output's non-tool helpers, byte for byte.
void plain_trim_leading(std::string& s) {
    while (!s.empty() && (s[0] == '\n' || s[0] == '\r' || s[0] == ' ' || s[0] == '\t')) s.erase(0, 1);
}

void plain_trim_trailing(std::string& s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
        s.pop_back();
}

void plain_strip_trailing_im_end(std::string& s) {
    static const std::string kEnd = "<|im_end|>";
    if (s.size() >= kEnd.size() && s.compare(s.size() - kEnd.size(), kEnd.size(), kEnd) == 0)
        s.resize(s.size() - kEnd.size());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
}

std::string plain_strip_think_markers(std::string s) {
    const size_t open_len = std::char_traits<char>::length(kThinkOpen);
    const size_t close_len = std::char_traits<char>::length(kThinkClose);
    for (;;) {
        const size_t o = s.find(kThinkOpen);
        if (o == std::string::npos) break;
        const size_t c = s.find(kThinkClose, o + open_len);
        if (c == std::string::npos) {
            s.erase(o, open_len);
            continue;
        }
        s.erase(o, c + close_len - o);
    }
    for (;;) {
        const size_t c = s.find(kThinkClose);
        if (c == std::string::npos) break;
        s.erase(c, close_len);
    }
    return s;
}

}  // namespace

PlainAssistantOutput parse_plain_assistant_output(const std::string& raw, bool enable_thinking) {
    PlainAssistantOutput out;
    if (!enable_thinking) {
        out.content = raw;
        plain_strip_trailing_im_end(out.content);
        return out;
    }
    // The official Qwen3.6 generation prompt already ends in "<think>\n" when thinking is
    // enabled, so generated text normally starts inside that block and contains only the
    // closing marker. Accept a repeated opening marker defensively, but do not require one.
    const size_t open = raw.find(kThinkOpen);
    const size_t body_start = open == std::string::npos ? 0 : open + std::char_traits<char>::length(kThinkOpen);
    const size_t close = raw.find(kThinkClose, body_start);
    if (close != std::string::npos) {
        out.reasoning_content = raw.substr(body_start, close - body_start);
        out.content = raw.substr(close + std::char_traits<char>::length(kThinkClose));
    } else {
        out.reasoning_content = raw.substr(body_start);
    }
    plain_trim_leading(out.reasoning_content);
    plain_trim_trailing(out.reasoning_content);
    plain_trim_leading(out.content);
    out.content = plain_strip_think_markers(std::move(out.content));
    plain_strip_trailing_im_end(out.content);
    return out;
}

bool build_response_format_grammar(const ChatRequest& request, bool enable_thinking, ToolCallGrammar& out,
                                   std::string& err) {
    out = ToolCallGrammar{};
    const ResponseFormat& format = request.response_format;
    if (format.type == ResponseFormatType::kText) return set_error(err, "response_format is text");
    json value;
    if (format.type == ResponseFormatType::kJsonObject) {
        // Any object: json_object promises an object, and strict JSON is all validate_response_format checks.
        value = {{"type", "json_schema"}, {"json_schema", {{"type", "object"}}},
                 {"sparkinfer_json_mode", true}, {"sparkinfer_strict", false}, {"sparkinfer_forbid_think", enable_thinking}};
        approximate(out, "response_format json_object: no grammar can stop an object key from repeating");
    } else {
        const json& root = format.schema;
        json sub = normalize_for_grammar(root, root, out, "response_format", 0, /*xml_framing=*/false);
        for (const char* defs : {"$defs", "definitions"}) {
            if (!sub.contains(defs) || !sub[defs].is_object()) continue;
            for (auto& item : sub[defs].items())
                item.value() = normalize_for_grammar(root, item.value(), out,
                                                     std::string("response_format ") + defs + "/" + item.key(),
                                                     0, /*xml_framing=*/false);
        }
        value = {{"type", "json_schema"}, {"json_schema", std::move(sub)}, {"sparkinfer_json_mode", true},
                 {"sparkinfer_forbid_think", enable_thinking}};
    }
    // Thinking on: reasoning closes as the template renders it, then the JSON value. Reasoning may not
    // spell a think marker -- the content split keys on the first </think>.
    json body = enable_thinking
        ? st_sequence({st_tag("", json{{"type", "any_text"}, {"excludes", {"<think", "</think"}}},
                              std::string(kThinkClose) + "\n\n"),
                       std::move(value)})
        : std::move(value);
    out.structural_tag = json{{"type", "structural_tag"}, {"format", std::move(body)}}.dump();
    return true;
}

std::string forced_tool_call_prefix(const ChatRequest& request) {
    if (request.tools.empty()) return {};
    const std::string open = std::string(kToolCallOpen) + "\n" + kFunctionOpen;
    if (request.tool_choice == ToolChoiceMode::kNamed) return open + request.required_tool_name + ">\n";
    if (request.tool_choice != ToolChoiceMode::kRequired) return {};
    // One offered function: required can only mean that one, and naming it leaves the model
    // nothing to invent.
    if (request.tools.size() == 1) return open + request.tools[0].name + ">\n";
    return open;
}

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw, bool enable_thinking,
                                          const ChatRequest& request) {
    ParsedToolOutput out;
    std::string remaining = raw;
    if (enable_thinking) {
        const size_t open = remaining.find(kThinkOpen);
        if (open != std::string::npos) {
            if (remaining.substr(0, open).find_first_not_of(" \t\r\n") != std::string::npos) {
                return fail_tool_output(std::move(out), "unexpected text before <think>");
            }
            const size_t close = remaining.find(kThinkClose, open + std::char_traits<char>::length(kThinkOpen));
            if (close == std::string::npos) {
                return fail_tool_output(std::move(out), "unterminated <think> block");
            }
            out.reasoning_content = remaining.substr(open + std::char_traits<char>::length(kThinkOpen),
                                                     close - open - std::char_traits<char>::length(kThinkOpen));
            trim_leading_ws(out.reasoning_content);
            trim_trailing_ws(out.reasoning_content);
            remaining.erase(0, close + std::char_traits<char>::length(kThinkClose));
            trim_leading_ws(remaining);
        } else {
            // With enable_thinking=true the generation prompt itself ends in "<think>\n", so the
            // generated suffix normally begins with reasoning text and only emits </think>.
            const size_t close = remaining.find(kThinkClose);
            if (close != std::string::npos) {
                out.reasoning_content = remaining.substr(0, close);
                trim_leading_ws(out.reasoning_content);
                trim_trailing_ws(out.reasoning_content);
                remaining.erase(0, close + std::char_traits<char>::length(kThinkClose));
                trim_leading_ws(remaining);
            } else {
                return fail_tool_output(std::move(out), "unterminated implicit <think> block");
            }
        }
    } else {
        const size_t close = remaining.find(kThinkClose);
        if (close != std::string::npos) {
            const size_t open = remaining.rfind(kThinkOpen, close);
            if (open == std::string::npos) {
                return fail_tool_output(std::move(out), "unmatched </think> marker");
            }
            remaining.erase(open, close + std::char_traits<char>::length(kThinkClose) - open);
            trim_leading_ws(remaining);
        }
    }

    if (has_protocol_markup(out.reasoning_content)) {
        return fail_tool_output(std::move(out), "reasoning contains reserved protocol markup");
    }

    const size_t first_call = remaining.find(kToolCallOpen);
    if (first_call == std::string::npos) {
        out.content = remaining;
        strip_trailing_im_end(out.content);
        if (has_protocol_markup(out.content)) {
            return fail_tool_output(std::move(out), "malformed tool-call markup");
        }
        // tool_choice=required / a named function is a CONTRACT, not a hint: the caller is told at
        // least one call (or that specific call) will come back, and typically branches on
        // tool_calls without checking. The template already instructs the model that it MUST call
        // -- but instructing is not enforcing, and a model that answers in prose anyway was, until
        // now, passed straight through as an ordinary assistant message. The caller then sees a
        // successful completion with no tool_calls, which is exactly the case it was promised
        // could not happen.
        //
        // The failure is flagged (missing_required_call) rather than final: the server continues the
        // model's own reasoning into a forced call (forced_tool_call_prefix), and only a call that
        // still does not come back reaches the client as a 502.
        if (!request.tools.empty()) {
            if (request.tool_choice == ToolChoiceMode::kRequired) {
                return fail_missing_call(std::move(out),
                                         "tool_choice=required but the model returned no tool call");
            }
            if (request.tool_choice == ToolChoiceMode::kNamed) {
                return fail_missing_call(std::move(out),
                                         "tool_choice named the function \"" +
                                         request.required_tool_name +
                                         "\" but the model returned no tool call");
            }
        }
        return out;
    }
    if (request.tools.empty() || request.tool_choice == ToolChoiceMode::kNone) {
        return fail_tool_output(std::move(out), "model emitted a tool call when tools were unavailable");
    }
    out.content = remaining.substr(0, first_call);
    trim_trailing_ws(out.content);
    if (has_protocol_markup(out.content)) {
        return fail_tool_output(std::move(out),
                                "assistant content contains reserved protocol markup");
    }
    size_t pos = first_call;
    while (pos < remaining.size()) {
        while (pos < remaining.size() && std::isspace(static_cast<unsigned char>(remaining[pos]))) ++pos;
        // Trailing whitespace with nothing after it (e.g. a lone newline the model emitted
        // right before hitting EOS/token-limit, with no literal <|im_end|> text -- decode()
        // skips special-token text entirely, so that's the common case, not an edge case) is a
        // clean end of output, not "malformed markup after a tool call".
        if (pos == remaining.size()) break;
        if (remaining.compare(pos, std::char_traits<char>::length(kImEnd), kImEnd) == 0) {
            pos += std::char_traits<char>::length(kImEnd);
            while (pos < remaining.size() && std::isspace(static_cast<unsigned char>(remaining[pos]))) ++pos;
            if (pos != remaining.size()) {
                return fail_tool_output(std::move(out), "text follows terminal <|im_end|>");
            }
            break;
        }
        if (remaining.compare(pos, std::char_traits<char>::length(kToolCallOpen), kToolCallOpen) != 0) {
            return fail_tool_output(std::move(out), "text or malformed markup appears after a tool call");
        }
        const size_t body_start = pos + std::char_traits<char>::length(kToolCallOpen);
        const size_t end = remaining.find(kToolCallClose, body_start);
        if (end == std::string::npos) {
            return fail_tool_output(std::move(out), "unterminated <tool_call> block");
        }
        ToolCall call;
        bool unoffered = false;
        if (!parse_one_xml_call(remaining.substr(body_start, end - body_start), request, call, out.error,
                                &unoffered)) {
            const std::string error = out.error;
            // Under required or a named function, an invented name is a missing call, not a
            // malformed one: the server picks an offered function and forces it. Under auto the
            // model chose to call something that does not exist, and that stays its error.
            if (unoffered && (request.tool_choice == ToolChoiceMode::kRequired ||
                              request.tool_choice == ToolChoiceMode::kNamed))
                return fail_missing_call(std::move(out), error);
            return fail_tool_output(std::move(out), error);
        }
        out.tool_calls.push_back(std::move(call));
        pos = end + std::char_traits<char>::length(kToolCallClose);
    }
    // parallel_tool_calls=false promises at most one call. The first is complete and schema-valid
    // by now, so keep it and drop the rest instead of failing a usable response.
    if (!request.parallel_tool_calls && out.tool_calls.size() > 1) out.tool_calls.resize(1);
    if (request.tool_choice == ToolChoiceMode::kRequired && out.tool_calls.empty())
        return fail_missing_call(std::move(out), "model did not call a required tool");
    if (request.tool_choice == ToolChoiceMode::kNamed) {
        // A named tool_choice forces that function (OpenAI semantics), so calls to any other
        // function are dropped; with none left it is a missing call like no call at all.
        std::vector<ToolCall> named;
        for (ToolCall& call : out.tool_calls)
            if (call.name == request.required_tool_name) named.push_back(std::move(call));
        out.tool_calls = std::move(named);
        if (out.tool_calls.empty())
            return fail_missing_call(std::move(out), "model did not call the required function");
    }
    return out;
}

}  // namespace sparkinfer_server
