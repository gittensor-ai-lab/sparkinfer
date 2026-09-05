#include "chat_tools.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <re2/re2.h>
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
                    const std::string& where, std::string& err) {
    for (const auto& item : object.items()) {
        if (!allowed.count(item.key()))
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

bool valid_schema_node(const json& schema, const std::string& where, bool top_level,
                       std::string& err) {
    if (!schema.is_object()) return set_error(err, where + " must be an object");
    if (!is_allowed_key(schema,
                        {"type", "description", "default", "title", "properties",
                         "required", "additionalProperties", "items", "enum", "minimum",
                         "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems",
                         "maxItems", "minLength", "maxLength", "pattern"},
                        where, err)) return false;
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
    if (schema.contains("properties")) {
        if (!schema["properties"].is_object())
            return set_error(err, where + ".properties must be an object");
        for (const auto& property : schema["properties"].items()) {
            if (!valid_schema_node(property.value(), where + ".properties." + property.key(),
                                   false, err)) return false;
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
                           false, err)) return false;
    if (schema.contains("items")) {
        if (!schema["items"].is_object())
            return set_error(err, where + ".items must be an object schema");
        if (!valid_schema_node(schema["items"], where + ".items", false, err)) return false;
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

bool validate_value(const json& value, const json& schema, const std::string& path, std::string& err) {
    if (!schema.is_object()) return true;
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
                if (!validate_value(item.value(), properties[item.key()], path + "." + item.key(), err)) return false;
            } else if (!allow_unknown) {
                return set_error(err, path + " contains unknown parameter " + item.key());
            } else if (schema.contains("additionalProperties") &&
                       schema["additionalProperties"].is_object() &&
                       !validate_value(item.value(), schema["additionalProperties"],
                                       path + "." + item.key(), err)) {
                return false;
            }
        }
    } else if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<std::size_t>())
            return set_error(err, path + " has fewer than minItems elements");
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<std::size_t>())
            return set_error(err, path + " has more than maxItems elements");
        if (schema.contains("items")) {
            for (size_t i = 0; i < value.size(); ++i)
                if (!validate_value(value[i], schema["items"],
                                    path + "[" + std::to_string(i) + "]", err)) return false;
        }
    }
    return true;
}

bool parse_parameter_value(const std::string& value, const json& schema, json& parsed) {
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
            if (validate_value(candidate, schema, "parameter value", validation_error)) {
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

const ToolDefinition* offered_tool(const ChatRequest& request, const std::string& name) {
    for (const ToolDefinition& tool : request.tools)
        if (tool.name == name) return &tool;
    return nullptr;
}

bool parse_one_xml_call(const std::string& block, const ChatRequest& request, ToolCall& call,
                        std::string& err) {
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
    if (!tool) return set_error(err, "model called unoffered function " + call.name);

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
        if (!parse_parameter_value(value, *property_schema, parsed))
            return set_error(err, "parameter " + key + " is not valid for its schema type");
        if (!validate_value(parsed, *property_schema, "parameter " + key, err)) return false;
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
    return out.str();
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
        // Failing here routes into the same invalid_tool_output path as malformed markup, which is
        // bounded (one retry, then a 502) rather than looping.
        if (!request.tools.empty()) {
            if (request.tool_choice == ToolChoiceMode::kRequired) {
                return fail_tool_output(std::move(out),
                                        "tool_choice=required but the model returned no tool call");
            }
            if (request.tool_choice == ToolChoiceMode::kNamed) {
                return fail_tool_output(std::move(out),
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
        if (!parse_one_xml_call(remaining.substr(body_start, end - body_start), request, call, out.error)) {
            const std::string error = out.error;
            return fail_tool_output(std::move(out), error);
        }
        out.tool_calls.push_back(std::move(call));
        pos = end + std::char_traits<char>::length(kToolCallClose);
    }
    if (!request.parallel_tool_calls && out.tool_calls.size() > 1) {
        return fail_tool_output(std::move(out),
                                "model emitted parallel tool calls when they were disabled");
    }
    if (request.tool_choice == ToolChoiceMode::kRequired && out.tool_calls.empty())
        return fail_tool_output(std::move(out), "model did not call a required tool");
    if (request.tool_choice == ToolChoiceMode::kNamed) {
        if (out.tool_calls.empty())
            return fail_tool_output(std::move(out), "model did not call the required function");
        for (const ToolCall& call : out.tool_calls)
            if (call.name != request.required_tool_name)
                return fail_tool_output(std::move(out), "model called a function other than the required function");
    }
    return out;
}

}  // namespace sparkinfer_server
