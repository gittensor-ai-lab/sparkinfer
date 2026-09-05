#include "chat_tools.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

namespace {

using nlohmann::json;
using sparkinfer_server::ChatRequest;
using sparkinfer_server::ParsedToolOutput;
using sparkinfer_server::ResponseFormat;
using sparkinfer_server::ResponseFormatType;
using sparkinfer_server::ToolChoiceMode;
using sparkinfer_server::ToolCall;
using sparkinfer_server::RequestControls;
using sparkinfer_server::apply_qwen36_tools_template;
using sparkinfer_server::parse_chat_request_json;
using sparkinfer_server::parse_legacy_completion_request;
using sparkinfer_server::parse_qwen36_tool_output;
using sparkinfer_server::parse_request_controls;
using sparkinfer_server::parse_score_request;
using sparkinfer_server::ScoreRequest;
using sparkinfer_server::should_reject_dflash_logit_bias;
using sparkinfer_server::should_reject_dflash_penalty;
using sparkinfer_server::should_reject_dflash_temperature;
using sparkinfer_server::validate_response_format;

#define CHECK(expr)                                                                            \
    do {                                                                                       \
        if (!(expr)) {                                                                         \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr);             \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool parse_request(const std::string& body, ChatRequest& request) {
    std::string error;
    if (!parse_chat_request_json(body, request, error)) {
        std::fprintf(stderr, "unexpected request parse failure: %s\n", error.c_str());
        return false;
    }
    if (!error.empty()) {
        std::fprintf(stderr, "successful request parse left an error: %s\n", error.c_str());
        return false;
    }
    return true;
}

json arguments(const ToolCall& call) {
    return json::parse(call.arguments);
}

const std::string& hermes_request() {
    static const std::string body = [] {
        std::ifstream in(std::string(SPARKINFER_SERVER_TEST_DIR) +
                         "/fixtures/hermes_tool_request.json", std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    }();
    return body;
}

bool test_hermes_request_and_template() {
    ChatRequest request;
    CHECK(!hermes_request().empty());
    CHECK(parse_request(hermes_request(), request));
    CHECK(request.messages.size() == 2);
    CHECK(request.messages[0].role == "system");
    CHECK(request.messages[1].role == "user");
    CHECK(request.tools.size() == 2);
    CHECK(request.tools[0].name == "lookup_catalog");
    CHECK(request.tools[1].name == "summarize_items");
    CHECK(request.tool_choice == ToolChoiceMode::kAuto);
    CHECK(request.parallel_tool_calls);

    // The schema must remain structured and retain nested object, array, integer,
    // enum, bounds, required, and additionalProperties constraints.
    const std::string catalog_schema = request.tools[0].spec.dump();
    CHECK(contains(catalog_schema, "\"query\""));
    CHECK(contains(catalog_schema, "\"object\""));
    CHECK(contains(catalog_schema, "\"tags\""));
    CHECK(contains(catalog_schema, "\"array\""));
    CHECK(contains(catalog_schema, "\"integer\""));
    CHECK(contains(catalog_schema, "\"minimum\""));
    CHECK(contains(catalog_schema, "\"maximum\""));
    CHECK(contains(catalog_schema, "\"additionalProperties\""));
    const std::string summarize_schema = request.tools[1].spec.dump();
    CHECK(contains(summarize_schema, "\"enum\""));
    CHECK(contains(summarize_schema, "\"brief\""));
    CHECK(contains(summarize_schema, "\"detailed\""));

    const std::string prompt = apply_qwen36_tools_template(request, false);
    CHECK(contains(prompt, "<|im_start|>system\n"));
    CHECK(contains(prompt, "# Tools"));
    CHECK(contains(prompt, "<tools>"));
    CHECK(contains(prompt, "</tools>"));
    CHECK(contains(prompt, "lookup_catalog"));
    CHECK(contains(prompt, "summarize_items"));
    CHECK(contains(prompt, "additionalProperties"));
    CHECK(contains(prompt,
        R"JSON({"function": {"description": "Find catalog entries matching structured search criteria.", "name": "lookup_catalog", "parameters": {)JSON"));
    CHECK(contains(prompt, "You are a test agent."));
    CHECK(contains(prompt, "<|im_start|>user\nFind two classic books"));
    CHECK(contains(prompt, "<tool_call>")); // tool-call format instructions
    CHECK(prompt.size() >= std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n").size());
    CHECK(prompt.compare(prompt.size() - std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n").size(),
                         std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n").size(),
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0);

    const std::string escaped_body = R"JSON({
      "messages":[{"role":"user","content":"test"}],
      "tools":[{"type":"function","function":{
        "name":"escape_test","description":"<tag>&' café",
        "parameters":{"type":"object","properties":{}}
      }}]
    })JSON";
    ChatRequest escaped_request;
    CHECK(parse_request(escaped_body, escaped_request));
    const std::string escaped_prompt = apply_qwen36_tools_template(escaped_request, false);
    CHECK(contains(escaped_prompt, "\\u003ctag\\u003e\\u0026\\u0027 caf\\u00e9"));
    CHECK(!contains(escaped_prompt, "<tag>"));
    return true;
}

bool test_tool_history_round_trip() {
    const std::string body = R"JSON(
{
  "messages": [
    {"role":"user","content":"Find two classic books."},
    {
      "role":"assistant",
      "content":null,
      "reasoning_content":"I should query the catalog.",
      "tool_calls":[{
        "id":"call_catalog_1",
        "type":"function",
        "function":{
          "name":"lookup_catalog",
          "arguments":"{\"query\":{\"category\":\"books\",\"tags\":[\"classic\"]},\"limit\":2}"
        }
      }]
    },
    {
      "role":"tool",
      "tool_call_id":"call_catalog_1",
      "name":"lookup_catalog",
      "content":"{\"titles\":[\"Pride and Prejudice\",\"Moby-Dick\"]}"
    },
    {"role":"user","content":"Now summarize them."}
  ],
  "tools":[{
    "type":"function",
    "function":{
      "name":"lookup_catalog",
      "description":"Find books.",
      "parameters":{
        "type":"object",
        "properties":{
          "query":{"type":"object"},
          "limit":{"type":"integer"}
        },
        "required":["query","limit"]
      }
    }
  }]
}
)JSON";

    ChatRequest request;
    CHECK(parse_request(body, request));
    CHECK(request.messages.size() == 4);
    const auto& assistant = request.messages[1];
    CHECK(assistant.role == "assistant");
    CHECK(assistant.content_is_null);
    CHECK(assistant.reasoning_content == "I should query the catalog.");
    CHECK(assistant.tool_calls.size() == 1);
    CHECK(assistant.tool_calls[0].id == "call_catalog_1");
    CHECK(assistant.tool_calls[0].name == "lookup_catalog");
    CHECK(arguments(assistant.tool_calls[0])["limit"] == 2);
    const auto& result = request.messages[2];
    CHECK(result.role == "tool");
    CHECK(result.tool_call_id == "call_catalog_1");
    CHECK(result.name == "lookup_catalog");

    const std::string prompt = apply_qwen36_tools_template(request, false);
    CHECK(contains(prompt, "<|im_start|>assistant\n"));
    CHECK(contains(prompt, "<function=lookup_catalog>"));
    CHECK(contains(prompt, "<parameter=query>"));
    CHECK(contains(prompt, "<parameter=limit>\n2\n</parameter>"));
    CHECK(contains(prompt, "<tool_response>"));
    CHECK(contains(prompt, "Pride and Prejudice"));
    CHECK(contains(prompt, "</tool_response>"));
    CHECK(contains(prompt, "<|im_start|>user\nNow summarize them."));
    return true;
}

bool test_tool_choice_modes() {
    const std::string prefix = R"JSON({
      "messages":[{"role":"user","content":"Use lookup if allowed."}],
      "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
      "tool_choice":)JSON";

    ChatRequest request;
    CHECK(parse_request(prefix + "\"auto\"}", request));
    CHECK(request.tool_choice == ToolChoiceMode::kAuto);
    CHECK(contains(apply_qwen36_tools_template(request, false), "<tools>"));

    CHECK(parse_request(prefix + "\"none\"}", request));
    CHECK(request.tool_choice == ToolChoiceMode::kNone);
    const std::string no_tools_prompt = apply_qwen36_tools_template(request, false);
    CHECK(!contains(no_tools_prompt, "<tools>"));
    CHECK(!contains(no_tools_prompt, "<function=lookup>"));
    CHECK(contains(no_tools_prompt, "<|im_start|>user\nUse lookup if allowed."));

    CHECK(parse_request(prefix + "\"required\"}", request));
    CHECK(request.tool_choice == ToolChoiceMode::kRequired);
    CHECK(contains(apply_qwen36_tools_template(request, false), "MUST call at least one"));
    CHECK(!parse_qwen36_tool_output("No call", false, request).error.empty());

    CHECK(parse_request(prefix + R"JSON({"type":"function","function":{"name":"lookup"}})JSON" + "}", request));
    CHECK(request.tool_choice == ToolChoiceMode::kNamed);
    CHECK(request.required_tool_name == "lookup");
    CHECK(contains(apply_qwen36_tools_template(request, false), "MUST call the function named lookup"));

    for (const std::string& bad_choice : {
             std::string("\"sometimes\""),
             std::string("7"),
             std::string(R"JSON({"type":"function","function":{"name":"missing"}})JSON"),
             std::string(R"JSON({"type":"function","function":{}})JSON"),
             std::string(R"JSON({"type":"other","function":{"name":"lookup"}})JSON")}) {
        std::string error;
        ChatRequest invalid;
        CHECK(!parse_chat_request_json(prefix + bad_choice + "}", invalid, error));
        CHECK(!error.empty());
    }
    return true;
}

bool test_tool_choice_none_preserves_history() {
    const std::string body = R"JSON({
      "messages":[
        {"role":"system","content":"Do not call new tools."},
        {"role":"user","content":"Look up the record."},
        {
          "role":"assistant",
          "content":null,
          "reasoning_content":"The earlier request required a lookup.",
          "tool_calls":[{
            "id":"call_prior",
            "type":"function",
            "function":{"name":"lookup","arguments":"{\"key\":\"alpha\"}"}
          }]
        },
        {
          "role":"tool",
          "tool_call_id":"call_prior",
          "name":"lookup",
          "content":"{\"value\":42}"
        }
      ],
      "tools":[{"type":"function","function":{
        "name":"lookup",
        "description":"Look up a record.",
        "parameters":{
          "type":"object",
          "properties":{"key":{"type":"string"}},
          "required":["key"],
          "additionalProperties":false
        }
      }}],
      "tool_choice":"none"
    })JSON";

    ChatRequest request;
    CHECK(parse_request(body, request));
    CHECK(request.tool_choice == ToolChoiceMode::kNone);
    CHECK(request.tools.size() == 1); // none disables new calls; it does not erase history metadata
    CHECK(request.tools[0].name == "lookup");
    CHECK(request.messages[2].tool_calls.size() == 1);

    const std::string prompt = apply_qwen36_tools_template(request, false);
    CHECK(!contains(prompt, "# Tools"));
    CHECK(!contains(prompt, "<tools>"));
    CHECK(contains(prompt, "<function=lookup>"));
    CHECK(contains(prompt, "<parameter=key>\nalpha\n</parameter>"));
    CHECK(contains(prompt, "<tool_response>\n{\"value\":42}\n</tool_response>"));
    CHECK(contains(prompt, "The earlier request required a lookup."));
    return true;
}

bool test_one_tool_call_and_schema_types() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    const std::string raw =
        "<think>\nI need the catalog.\n</think>\n\n"
        "<tool_call>\n"
        "<function=lookup_catalog>\n"
        "<parameter=query>\n"
        "{\"category\":\"books\",\"tags\":[\"classic\"]}\n"
        "</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const ParsedToolOutput output = parse_qwen36_tool_output(raw, true, request);
    CHECK(output.error.empty());
    CHECK(output.reasoning_content == "I need the catalog.");
    CHECK(output.content.empty());
    CHECK(output.tool_calls.size() == 1);
    CHECK(output.tool_calls[0].id.empty()); // response transport owns call IDs
    CHECK(output.tool_calls[0].name == "lookup_catalog");
    const json args = arguments(output.tool_calls[0]);
    CHECK(args.is_object());
    CHECK(args["query"].is_object());
    CHECK(args["query"]["category"].is_string());
    CHECK(args["query"]["tags"].is_array());
    CHECK(args["query"]["tags"][0] == "classic");
    CHECK(args["limit"].is_number_integer());
    CHECK(args["limit"] == 2);
    return true;
}

bool test_implicit_thinking_prefill_suffix() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    // With thinking enabled, the prompt already ends in "<think>\n". The generated suffix
    // therefore starts with reasoning and closes the inherited block without another opener.
    const std::string raw =
        "I should query the catalog before answering.\n</think>\n\n"
        "<tool_call>\n<function=lookup_catalog>\n"
        "<parameter=query>\n{\"category\":\"books\",\"tags\":[]}\n</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n</tool_call>";
    const ParsedToolOutput output = parse_qwen36_tool_output(raw, true, request);
    CHECK(output.error.empty());
    CHECK(output.reasoning_content == "I should query the catalog before answering.");
    CHECK(output.content.empty());
    CHECK(output.tool_calls.size() == 1);
    CHECK(output.tool_calls[0].name == "lookup_catalog");
    CHECK(arguments(output.tool_calls[0])["limit"] == 2);
    return true;
}

bool test_all_json_schema_value_types() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Send typed values."}],
      "tools":[{"type":"function","function":{
        "name":"typed_values",
        "parameters":{
          "type":"object",
          "properties":{
            "text":{"type":"string"},
            "count":{"type":"integer"},
            "ratio":{"type":"number"},
            "enabled":{"type":"boolean"},
            "items":{"type":"array"},
            "config":{"type":"object"},
            "nothing":{"type":"null"}
          },
          "required":["text","count","ratio","enabled","items","config","nothing"],
          "additionalProperties":false
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    const std::string raw =
        "<tool_call>\n<function=typed_values>\n"
        "<parameter=text>\nhello\n</parameter>\n"
        "<parameter=count>\n7\n</parameter>\n"
        "<parameter=ratio>\n1.25\n</parameter>\n"
        "<parameter=enabled>\ntrue\n</parameter>\n"
        "<parameter=items>\n[\"a\",2]\n</parameter>\n"
        "<parameter=config>\n{\"x\":1}\n</parameter>\n"
        "<parameter=nothing>\nnull\n</parameter>\n"
        "</function>\n</tool_call>";
    const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
    CHECK(output.error.empty());
    CHECK(output.tool_calls.size() == 1);
    const json args = arguments(output.tool_calls[0]);
    CHECK(args["text"].is_string() && args["text"] == "hello");
    CHECK(args["count"].is_number_integer() && args["count"] == 7);
    CHECK(args["ratio"].is_number_float() && args["ratio"] == 1.25);
    CHECK(args["enabled"].is_boolean() && args["enabled"] == true);
    CHECK(args["items"].is_array() && args["items"].size() == 2);
    CHECK(args["config"].is_object() && args["config"]["x"] == 1);
    CHECK(args["nothing"].is_null());
    return true;
}

bool test_schema_constraints() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Send constrained values."}],
      "tools":[{"type":"function","function":{
        "name":"constrained",
        "parameters":{
          "type":"object",
          "properties":{
            "count":{"type":"integer","minimum":1,"maximum":3},
            "tags":{
              "type":"array",
              "minItems":1,
              "maxItems":2,
              "items":{"type":"string"}
            },
            "code":{"type":"string","pattern":"^[A-Z]{2}[0-9]{2}$"}
          },
          "required":["count","tags","code"],
          "additionalProperties":false
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    auto call = [](const std::string& count, const std::string& tags,
                   const std::string& code) {
        return "<tool_call>\n<function=constrained>\n"
               "<parameter=count>\n" + count + "\n</parameter>\n"
               "<parameter=tags>\n" + tags + "\n</parameter>\n"
               "<parameter=code>\n" + code + "\n</parameter>\n"
               "</function>\n</tool_call>";
    };

    const ParsedToolOutput valid = parse_qwen36_tool_output(
        call("2", "[\"classic\",\"fiction\"]", "AB12"), false, request);
    CHECK(valid.error.empty());
    CHECK(valid.tool_calls.size() == 1);

    for (const auto& invalid : {
             std::pair<const char*, std::string>{"minimum", call("0", "[\"classic\"]", "AB12")},
             std::pair<const char*, std::string>{"maximum", call("4", "[\"classic\"]", "AB12")},
             std::pair<const char*, std::string>{"minItems", call("2", "[]", "AB12")},
             std::pair<const char*, std::string>{"maxItems", call("2", "[\"a\",\"b\",\"c\"]", "AB12")},
             std::pair<const char*, std::string>{"pattern", call("2", "[\"classic\"]", "bad-code")}}) {
        const ParsedToolOutput output = parse_qwen36_tool_output(invalid.second, false, request);
        if (output.error.empty()) {
            std::fprintf(stderr, "FAIL: schema constraint %s was not enforced\n", invalid.first);
            return false;
        }
        CHECK(output.reasoning_content.empty());
        CHECK(output.content.empty());
        CHECK(output.tool_calls.empty());
    }
    return true;
}

bool test_schema_pattern_is_linear_time() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Check a value."}],
      "tools":[{"type":"function","function":{
        "name":"check_value",
        "parameters":{
          "type":"object",
          "properties":{"value":{"type":"string","pattern":"^(a+)+$"}},
          "required":["value"],
          "additionalProperties":false
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    const std::string raw =
        "<tool_call>\n<function=check_value>\n<parameter=value>\n" +
        std::string(30, 'a') + "!\n</parameter>\n</function>\n</tool_call>";
    const auto start = std::chrono::steady_clock::now();
    const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(!output.error.empty());
    CHECK(elapsed < std::chrono::seconds(1));
    return true;
}

bool test_schema_unions_and_dynamic_properties() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Send dynamic values."}],
      "tools":[{"type":"function","function":{
        "name":"dynamic_values",
        "parameters":{
          "type":"object",
          "properties":{
            "nullable_text":{"type":["string","null"]},
            "nullable_null":{"type":["string","null"]},
            "json_like_text":{"type":["string","null"]},
            "mixed_value":{"type":["string","integer"]},
            "inferred_text":{"enum":["red","blue"]},
            "inferred_number":{"enum":[1,2]}
          },
          "required":["dynamic_count"],
          "additionalProperties":{"type":"integer"}
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    const std::string raw =
        "<tool_call>\n<function=dynamic_values>\n"
        "<parameter=nullable_text>\nhello\n</parameter>\n"
        "<parameter=nullable_null>\nnull\n</parameter>\n"
        "<parameter=json_like_text>\n123\n</parameter>\n"
        "<parameter=mixed_value>\n123\n</parameter>\n"
        "<parameter=inferred_text>\nred\n</parameter>\n"
        "<parameter=inferred_number>\n2\n</parameter>\n"
        "<parameter=dynamic_count>\n3\n</parameter>\n"
        "</function>\n</tool_call>";
    const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
    CHECK(output.error.empty());
    CHECK(output.tool_calls.size() == 1);
    const json args = arguments(output.tool_calls[0]);
    CHECK(args["nullable_text"] == "hello");
    CHECK(args["nullable_null"].is_null());
    CHECK(args["json_like_text"].is_string() && args["json_like_text"] == "123");
    CHECK(args["mixed_value"].is_number_integer() && args["mixed_value"] == 123);
    CHECK(args["inferred_text"] == "red");
    CHECK(args["inferred_number"].is_number_integer() && args["inferred_number"] == 2);
    CHECK(args["dynamic_count"] == 3);

    const std::string bad_dynamic =
        "<tool_call>\n<function=dynamic_values>\n"
        "<parameter=dynamic_count>\nnot-an-integer\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(!parse_qwen36_tool_output(bad_dynamic, false, request).error.empty());

    json history = json::parse(body);
    history["messages"] = json::array({
        {{"role", "assistant"},
         {"content", nullptr},
         {"tool_calls", json::array({
             {{"id", "call_dynamic"},
              {"type", "function"},
              {"function", {{"name", "dynamic_values"},
                            {"arguments", "{\"dynamic_count\":3,\"nullable_text\":\"hello\"}"}}}}
         })}},
        {{"role", "tool"}, {"tool_call_id", "call_dynamic"}, {"content", "ok"}},
        {{"role", "user"}, {"content", "Continue."}}
    });
    ChatRequest history_request;
    CHECK(parse_request(history.dump(), history_request));
    const std::string prompt = apply_qwen36_tools_template(history_request, false);
    CHECK(contains(prompt, "<parameter=nullable_text>\nhello\n</parameter>"));
    CHECK(contains(prompt, "<parameter=dynamic_count>\n3\n</parameter>"));

    for (const json& additional : {json(), json(true)}) {
        json permissive = json::parse(body);
        json& schema = permissive["tools"][0]["function"]["parameters"];
        schema.erase("additionalProperties");
        if (!additional.is_null()) schema["additionalProperties"] = additional;
        ChatRequest permissive_request;
        CHECK(parse_request(permissive.dump(), permissive_request));
        const std::string freeform =
            "<tool_call>\n<function=dynamic_values>\n"
            "<parameter=dynamic_count>\n3\n</parameter>\n"
            "<parameter=freeform>\n{\"nested\":true}\n</parameter>\n"
            "</function>\n</tool_call>";
        const ParsedToolOutput accepted =
            parse_qwen36_tool_output(freeform, false, permissive_request);
        CHECK(accepted.error.empty());
        CHECK(arguments(accepted.tool_calls[0])["freeform"]["nested"] == true);
    }

    json closed = json::parse(body);
    closed["tools"][0]["function"]["parameters"]["additionalProperties"] = false;
    closed["tools"][0]["function"]["parameters"].erase("required");
    ChatRequest closed_request;
    CHECK(parse_request(closed.dump(), closed_request));
    CHECK(!parse_qwen36_tool_output(raw, false, closed_request).error.empty());

    history["messages"][0]["tool_calls"][0]["function"]["arguments"] =
        "{\"dynamic_count\":\"wrong\"}";
    std::string history_error;
    CHECK(!parse_chat_request_json(history.dump(), history_request, history_error));
    return true;
}

bool test_unicode_string_lengths() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Send one character."}],
      "tools":[{"type":"function","function":{
        "name":"one_character",
        "parameters":{
          "type":"object",
          "properties":{"value":{"type":"string","minLength":1,"maxLength":1}},
          "required":["value"],
          "additionalProperties":false
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    auto call = [](const std::string& value) {
        return "<tool_call>\n<function=one_character>\n<parameter=value>\n" + value +
               "\n</parameter>\n</function>\n</tool_call>";
    };
    CHECK(parse_qwen36_tool_output(call("é"), false, request).error.empty());
    CHECK(parse_qwen36_tool_output(call("😀"), false, request).error.empty());
    CHECK(!parse_qwen36_tool_output(call("ab"), false, request).error.empty());
    return true;
}

bool test_large_integer_bounds_are_exact() {
    const std::string body = R"JSON({
      "messages":[{"role":"user","content":"Send a large integer."}],
      "tools":[{"type":"function","function":{
        "name":"large_integer",
        "parameters":{
          "type":"object",
          "properties":{"value":{"type":"integer","minimum":9007199254740993}},
          "required":["value"],
          "additionalProperties":false
        }
      }}]
    })JSON";
    ChatRequest request;
    CHECK(parse_request(body, request));
    auto call = [](const char* value) {
        return std::string("<tool_call>\n<function=large_integer>\n<parameter=value>\n") +
               value + "\n</parameter>\n</function>\n</tool_call>";
    };
    CHECK(!parse_qwen36_tool_output(call("9007199254740992"), false, request).error.empty());
    CHECK(parse_qwen36_tool_output(call("9007199254740993"), false, request).error.empty());
    return true;
}

bool test_schema_keyword_type_applicability() {
    for (const auto& invalid : {
             std::pair<const char*, json>{"required", {{"type", "string"},
                                                        {"required", {"x"}}}},
             std::pair<const char*, json>{"items", {{"type", "object"},
                                                     {"items", json::object()}}},
             std::pair<const char*, json>{"maxLength", {{"type", "integer"},
                                                         {"maxLength", 1}}},
             std::pair<const char*, json>{"minimum", {{"type", "string"},
                                                       {"minimum", 1}}}}) {
        json body = {
            {"messages", {{{"role", "user"}, {"content", "test"}}}},
            {"tools", json::array({{
                {"type", "function"},
                {"function", {
                    {"name", "schema_test"},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {{"value", invalid.second}}}
                    }}
                }}
            }})}
        };
        ChatRequest request;
        std::string error;
        CHECK(!parse_chat_request_json(body.dump(), request, error));
        CHECK(contains(error, invalid.first));
    }
    return true;
}

bool test_unsupported_schema_keywords_are_rejected() {
    for (const char* unsupported : {"const", "multipleOf", "oneOf"}) {
        json body = {
            {"messages", {{{"role", "user"}, {"content", "test"}}}},
            {"tools", json::array({{
                {"type", "function"},
                {"function", {
                    {"name", "schema_test"},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {{"value", {{"type", "integer"},
                                                     {unsupported, unsupported == std::string("oneOf")
                                                         ? json::array({json{{"type", "integer"}}})
                                                         : json(2)}}}}}
                    }}
                }}
            }})}
        };
        ChatRequest request;
        std::string error;
        CHECK(!parse_chat_request_json(body.dump(), request, error));
        CHECK(contains(error, "unsupported field"));
    }
    return true;
}

bool test_parallel_tool_calls() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    const std::string raw =
        "<tool_call>\n"
        "<function=lookup_catalog>\n"
        "<parameter=query>\n{\"category\":\"books\",\"tags\":[]}\n</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=summarize_items>\n"
        "<parameter=titles>\n[\"Pride and Prejudice\",\"Moby-Dick\"]\n</parameter>\n"
        "<parameter=style>\nbrief\n</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
    CHECK(output.error.empty());
    CHECK(output.tool_calls.size() == 2);
    CHECK(output.tool_calls[0].name == "lookup_catalog");
    CHECK(output.tool_calls[1].name == "summarize_items");
    CHECK(output.tool_calls[0].id.empty());
    CHECK(output.tool_calls[1].id.empty());
    const json first = arguments(output.tool_calls[0]);
    const json second = arguments(output.tool_calls[1]);
    CHECK(first["limit"] == 2);
    CHECK(second["titles"].is_array());
    CHECK(second["titles"].size() == 2);
    CHECK(second["style"].is_string());
    CHECK(second["style"] == "brief");

    request.parallel_tool_calls = false;
    const ParsedToolOutput disallowed = parse_qwen36_tool_output(raw, false, request);
    CHECK(!disallowed.error.empty());
    CHECK(disallowed.tool_calls.empty());

    json single_request = json::parse(hermes_request());
    single_request["parallel_tool_calls"] = false;
    CHECK(parse_request(single_request.dump(), request));
    CHECK(!request.parallel_tool_calls);
    return true;
}

bool test_reasoning_effort_controls() {
    ChatRequest request;
    CHECK(parse_request(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"medium"}})",
        request));
    CHECK(request.reasoning_effort == "medium");
    // Third argument is inject_reasoning_effort, which DEFAULTS TO FALSE -- only Qwen3.8's
    // template prepends the reasoning-effort system message; Qwen3.6 and Muse Glimmer never do.
    // Calling with two arguments enables thinking but not injection, so this asserted a string the
    // call could not produce. Broken since the assertion was written (3881e0c), which added it
    // against the three-parameter signature 8dbfcdb had already introduced.
    CHECK(contains(apply_qwen36_tools_template(request, true, true),
                   "Reasoning effort is set to medium"));
    // And the default really is off, or the assertion above would pass for the wrong reason.
    CHECK(!contains(apply_qwen36_tools_template(request, true), "Reasoning effort is set to"));

    CHECK(parse_request(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low"})",
        request));
    CHECK(request.reasoning_effort == "low");
    CHECK(parse_request(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"max","exclude":true}})",
        request));
    CHECK(request.reasoning_effort == "xhigh");
    CHECK(request.reasoning_exclude);
    CHECK(parse_request(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"none"}})",
        request));
    CHECK(request.reasoning_effort == "none");
    std::string error;
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"reasoning":{"effort":"turbo"}})",
        request, error));
    return true;
}

bool test_nontext_content_parts_are_ignored_not_rejected() {
    // A mixed multimodal payload (common even against text-only backends -- agent frameworks
    // often build one request shape for every provider) must still succeed using whatever text
    // parts are present, not hard-reject the whole request.
    //
    // This used to cover image_url too. It no longer can: images are honored now, so they get
    // their own tests below. Anything still genuinely unsupported -- audio here -- must keep
    // being skipped rather than rejected.
    json body = {
        {"model", "spark"},
        {"messages", json::array({json::object({
             {"role", "user"},
             {"content", json::array({
                  json::object({{"type", "text"}, {"text", "describe this: "}}),
                  json::object({{"type", "input_audio"}, {"input_audio", json::object({{"data", "AAA"}})}}),
                  json::object({{"type", "text"}, {"text", "please"}}),
              })},
         })})},
    };
    ChatRequest request;
    CHECK(parse_request(body.dump(), request));
    CHECK(request.messages.size() == 1);
    CHECK(request.messages[0].content == "describe this: please");
    CHECK(request.messages[0].images.empty());
    return true;
}

bool test_image_parts_render_a_marker_and_collect_urls() {
    // Each image_url renders the marker the template emits, AT the position the part occupied,
    // and its URL is collected in the same order. The two must stay in lockstep: the processor
    // pairs the Nth placeholder with the Nth image to expand it to that image's token count, so
    // an ordering or count skew here silently splices one image's embeddings over another's span.
    json body = {
        {"model", "spark"},
        {"messages", json::array({json::object({
             {"role", "user"},
             {"content", json::array({
                  json::object({{"type", "text"}, {"text", "A:"}}),
                  json::object({{"type", "image_url"},
                                {"image_url", json::object({{"url", "data:image/png;base64,AAA"}})}}),
                  json::object({{"type", "text"}, {"text", " B:"}}),
                  json::object({{"type", "image_url"},
                                {"image_url", json::object({{"url", "data:image/png;base64,BBB"}})}}),
              })},
         })})},
    };
    ChatRequest request;
    CHECK(parse_request(body.dump(), request));
    CHECK(request.messages[0].images.size() == 2);
    CHECK(request.messages[0].images[0] == "data:image/png;base64,AAA");
    CHECK(request.messages[0].images[1] == "data:image/png;base64,BBB");
    CHECK(request.messages[0].content ==
          "A:<|vision_start|><|image_pad|><|vision_end|>"
          " B:<|vision_start|><|image_pad|><|vision_end|>");
    return true;
}

bool test_image_in_system_message_is_rejected() {
    // The chat template raises on this outright; parsing mirrors it rather than building a
    // prompt shape the model was never trained to read. "developer" is aliased to system, so it
    // has to be refused by the SAME rule and not slip past on the role name.
    for (const char* role : {"system", "developer"}) {
        json body = {
            {"model", "spark"},
            {"messages", json::array({json::object({
                 {"role", role},
                 {"content", json::array({json::object({
                      {"type", "image_url"},
                      {"image_url", json::object({{"url", "data:image/png;base64,AAA"}})}})})},
             })})},
        };
        ChatRequest request;
        CHECK(!parse_request(body.dump(), request));
    }
    return true;
}

bool test_malformed_image_url_is_rejected() {
    // A part that says it is an image but carries no usable url is a broken request, not
    // something to skip: skipping it would drop an image the caller believes was sent and answer
    // confidently about what is left.
    const char* bad[] = {
        R"([{"type":"image_url"}])",
        R"([{"type":"image_url","image_url":{}}])",
        R"([{"type":"image_url","image_url":{"url":123}}])",
        R"([{"type":"image_url","image_url":"data:image/png;base64,AAA"}])",
    };
    for (const char* parts : bad) {
        json body = {
            {"model", "spark"},
            {"messages", json::array({json::object({
                 {"role", "user"}, {"content", json::parse(parts)}})})},
        };
        ChatRequest request;
        CHECK(!parse_request(body.dump(), request));
    }
    return true;
}

bool test_developer_role_is_accepted_as_system_alias() {
    json body = {
        {"model", "spark"},
        {"messages", json::array({json::object({
             {"role", "developer"}, {"content", "be terse"},
         })})},
    };
    ChatRequest request;
    CHECK(parse_request(body.dump(), request));
    CHECK(request.messages.size() == 1);
    CHECK(request.messages[0].role == "system");
    return true;
}

bool test_response_format_type_validation() {
    ChatRequest request;
    std::string error;
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"response_format":"json_object"})",
        request, error));
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"bogus"}})",
        request, error));
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema"}})",
        request, error));  // missing json_schema object entirely

    json body = {{"messages", json::array({json::object({{"role", "user"}, {"content", "hi"}})})},
                 {"response_format", json::object({{"type", "text"}})}};
    CHECK(parse_request(body.dump(), request));
    CHECK(request.response_format.type == ResponseFormatType::kText);
    return true;
}

bool test_response_format_json_object_parses() {
    json body = {{"messages", json::array({json::object({{"role", "user"}, {"content", "hi"}})})},
                 {"response_format", json::object({{"type", "json_object"}})}};
    ChatRequest request;
    CHECK(parse_request(body.dump(), request));
    CHECK(request.response_format.type == ResponseFormatType::kJsonObject);
    return true;
}

bool test_response_format_json_schema_validation() {
    ChatRequest request;
    std::string error;
    // Missing "name"
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}})",
        request, error));
    // Malformed schema at request time: minimum > maximum -- reused valid_schema check.
    CHECK(!parse_chat_request_json(
        R"({"messages":[{"role":"user","content":"hi"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"n",
            "schema":{"type":"integer","minimum":10,"maximum":5}}}})",
        request, error));

    json body = {
        {"messages", json::array({json::object({{"role", "user"}, {"content", "hi"}})})},
        {"response_format", json::object({
             {"type", "json_schema"},
             {"json_schema", json::object({
                  {"name", "Answer"},
                  {"strict", true},
                  {"schema", json::object({
                       {"type", "object"},
                       {"properties", json::object({{"value", json::object({{"type", "integer"}})}})},
                       {"required", json::array({"value"})},
                       {"additionalProperties", false},
                   })},
              })},
         })},
    };
    CHECK(parse_request(body.dump(), request));
    CHECK(request.response_format.type == ResponseFormatType::kJsonSchema);
    CHECK(request.response_format.schema_name == "Answer");
    CHECK(request.response_format.strict == true);
    CHECK(request.response_format.schema["type"] == "object");
    return true;
}

bool test_response_format_strict_does_not_change_parse_outcome() {
    auto make_body = [](bool strict_value, bool include_strict) {
        json rf = json::object({{"type", "json_schema"},
                                {"json_schema", json::object({
                                     {"name", "N"},
                                     {"schema", json::object({{"type", "object"}})},
                                 })}});
        if (include_strict) rf["json_schema"]["strict"] = strict_value;
        json body = {{"messages", json::array({json::object({{"role", "user"}, {"content", "hi"}})})},
                     {"response_format", rf}};
        return body.dump();
    };
    ChatRequest a, b, c;
    CHECK(parse_request(make_body(true, true), a));
    CHECK(parse_request(make_body(false, true), b));
    CHECK(parse_request(make_body(false, false), c));
    CHECK(a.response_format.type == b.response_format.type);
    CHECK(b.response_format.type == c.response_format.type);
    CHECK(a.response_format.schema == b.response_format.schema);
    return true;
}

bool test_tools_and_response_format_rejected_together() {
    json body = {
        {"messages", json::array({json::object({{"role", "user"}, {"content", "hi"}})})},
        {"tools", json::array({json::object({
             {"type", "function"},
             {"function", json::object({{"name", "f"}, {"parameters", json::object({{"type", "object"}})}})},
         })})},
        {"response_format", json::object({{"type", "json_object"}})},
    };
    ChatRequest request;
    std::string error;
    CHECK(!parse_chat_request_json(body.dump(), request, error));
    return true;
}

bool test_response_format_steers_prompt_json_object() {
    ChatRequest request;
    request.messages.push_back({});
    request.messages.back().role = "user";
    request.messages.back().content = "give me data";
    request.response_format.type = ResponseFormatType::kJsonObject;
    const std::string prompt = apply_qwen36_tools_template(request, false);
    CHECK(contains(prompt, "# Response Format"));
    CHECK(contains(prompt, "single valid JSON value"));
    return true;
}

bool test_response_format_steers_prompt_json_schema_escaped() {
    ChatRequest request;
    request.messages.push_back({});
    request.messages.back().role = "user";
    request.messages.back().content = "give me data";
    request.response_format.type = ResponseFormatType::kJsonSchema;
    request.response_format.schema_name = "Answer";
    request.response_format.schema = json::object({
        {"type", "object"},
        {"properties", json::object({
             {"note", json::object({{"type", "string"}, {"description", "<script>alert(1)</script>"}})},
         })},
    });
    const std::string prompt = apply_qwen36_tools_template(request, false);
    CHECK(contains(prompt, "Answer"));
    CHECK(contains(prompt, "<response_schema>"));
    CHECK(contains(prompt, "</response_schema>"));
    // The htmlsafe renderer must escape a literal <script> tag inside a schema description --
    // same template-injection-safety property tool schemas already get.
    CHECK(!contains(prompt, "<script>alert(1)</script>"));
    return true;
}

bool test_response_format_text_is_a_prompt_noop() {
    ChatRequest with_default, without_field;
    with_default.messages.push_back({});
    with_default.messages.back().role = "user";
    with_default.messages.back().content = "hi";
    without_field = with_default;
    with_default.response_format.type = ResponseFormatType::kText;
    CHECK(apply_qwen36_tools_template(with_default, false) ==
          apply_qwen36_tools_template(without_field, false));
    return true;
}

bool test_validate_response_format_json_object() {
    ResponseFormat format;
    format.type = ResponseFormatType::kJsonObject;
    std::string err;
    CHECK(validate_response_format(R"({"a":1})", format, err));
    CHECK(!validate_response_format("not json", format, err));
    CHECK(!validate_response_format(R"({"a":1,"a":2})", format, err));  // duplicate key
    return true;
}

bool test_validate_response_format_json_schema() {
    ResponseFormat format;
    format.type = ResponseFormatType::kJsonSchema;
    format.schema = json::object({
        {"type", "object"},
        {"properties", json::object({{"n", json::object({{"type", "integer"}, {"minimum", 0}})}})},
        {"required", json::array({"n"})},
        {"additionalProperties", false},
    });
    std::string err;
    CHECK(validate_response_format(R"({"n":5})", format, err));
    CHECK(!validate_response_format(R"({"n":"not a number"})", format, err));
    CHECK(!validate_response_format(R"({"n":-1})", format, err));
    CHECK(!validate_response_format(R"({})", format, err));           // missing required
    CHECK(!validate_response_format(R"({"n":1,"extra":true})", format, err));  // additionalProperties
    return true;
}

bool test_request_controls_temperature_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.temperature == 0.f);
    CHECK(parse_request_controls(R"({"temperature":0})", controls, err));
    CHECK(controls.temperature == 0.f);
    CHECK(parse_request_controls(R"({"temperature":0.7})", controls, err));
    CHECK(controls.temperature > 0.69f && controls.temperature < 0.71f);
    CHECK(parse_request_controls(R"({"temperature":2.0})", controls, err));
    CHECK(controls.temperature == 2.0f);
    CHECK(!parse_request_controls(R"({"temperature":-0.1})", controls, err));
    CHECK(!parse_request_controls(R"({"temperature":2.1})", controls, err));
    CHECK(!parse_request_controls(R"({"temperature":"high"})", controls, err));
    CHECK(!parse_request_controls(R"({"temperature":null,"stream":"not a bool"})", controls, err));
    return true;
}

bool test_request_controls_seed_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(!controls.seed_set);
    CHECK(parse_request_controls(R"({"seed":0})", controls, err));
    CHECK(controls.seed_set && controls.seed == 0);
    CHECK(parse_request_controls(R"({"seed":42})", controls, err));
    CHECK(controls.seed_set && controls.seed == 42);
    CHECK(parse_request_controls(R"({"seed":9007199254740993})", controls, err));  // > 2^53
    CHECK(controls.seed_set && controls.seed == 9007199254740993ULL);
    CHECK(!parse_request_controls(R"({"seed":-1})", controls, err));
    CHECK(!parse_request_controls(R"({"seed":1.5})", controls, err));
    return true;
}

bool test_request_controls_top_p_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.top_p == 1.0f);
    CHECK(parse_request_controls(R"({"top_p":1.0})", controls, err));
    CHECK(controls.top_p == 1.0f);
    CHECK(parse_request_controls(R"({"top_p":0.9})", controls, err));
    CHECK(controls.top_p > 0.89f && controls.top_p < 0.91f);
    CHECK(parse_request_controls(R"({"top_p":0.0001})", controls, err));
    CHECK(controls.top_p > 0.f && controls.top_p < 0.001f);
    CHECK(parse_request_controls(R"({"top_p":0})", controls, err));
    CHECK(controls.top_p == 0.f);
    CHECK(!parse_request_controls(R"({"top_p":-0.1})", controls, err));
    CHECK(!parse_request_controls(R"({"top_p":1.1})", controls, err));
    CHECK(!parse_request_controls(R"({"top_p":"high"})", controls, err));
    // Does not require temperature to be set.
    CHECK(parse_request_controls(R"({"top_p":0.5})", controls, err));
    CHECK(controls.temperature == 0.f);
    return true;
}

bool test_request_controls_top_k_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.top_k == 0);
    CHECK(parse_request_controls(R"({"top_k":0})", controls, err));
    CHECK(controls.top_k == 0);
    CHECK(parse_request_controls(R"({"top_k":40})", controls, err));
    CHECK(controls.top_k == 40);
    CHECK(parse_request_controls(R"({"top_k":1000000})", controls, err));  // no upper-bound rejection
    CHECK(controls.top_k == 1000000);
    CHECK(!parse_request_controls(R"({"top_k":-1})", controls, err));
    CHECK(!parse_request_controls(R"({"top_k":1.5})", controls, err));
    CHECK(!parse_request_controls(R"({"top_k":"high"})", controls, err));
    // Does not require temperature to be set.
    CHECK(parse_request_controls(R"({"top_k":5})", controls, err));
    CHECK(controls.temperature == 0.f);
    return true;
}

bool test_request_controls_logprobs_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(!controls.logprobs && controls.top_logprobs == 0);
    CHECK(parse_request_controls(R"({"logprobs":true})", controls, err));
    CHECK(controls.logprobs && controls.top_logprobs == 0);
    CHECK(parse_request_controls(R"({"logprobs":true,"top_logprobs":5})", controls, err));
    CHECK(controls.top_logprobs == 5);
    CHECK(parse_request_controls(R"({"logprobs":true,"top_logprobs":20})", controls, err));
    CHECK(controls.top_logprobs == 20);
    CHECK(parse_request_controls(R"({"logprobs":true,"top_logprobs":0})", controls, err));
    CHECK(controls.top_logprobs == 0);
    CHECK(!parse_request_controls(R"({"logprobs":true,"top_logprobs":21})", controls, err));
    CHECK(!parse_request_controls(R"({"logprobs":true,"top_logprobs":-1})", controls, err));
    CHECK(!parse_request_controls(R"({"logprobs":"yes"})", controls, err));
    CHECK(!parse_request_controls(R"({"logprobs":true,"top_logprobs":1.5})", controls, err));
    // Cross-field rule: fresh controls each time, since `controls` above already has
    // logprobs=true persisted from earlier calls in this test (parse_request_controls only sets
    // fields present in the JSON, it never resets) -- reusing it here would trivially pass.
    RequestControls fresh1;
    CHECK(!parse_request_controls(R"({"top_logprobs":5})", fresh1, err));           // no logprobs:true
    RequestControls fresh2;
    CHECK(!parse_request_controls(R"({"logprobs":false,"top_logprobs":5})", fresh2, err));
    return true;
}

bool test_request_controls_presence_penalty_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.presence_penalty == 0.f);
    CHECK(parse_request_controls(R"({"presence_penalty":0})", controls, err));
    CHECK(controls.presence_penalty == 0.f);
    CHECK(parse_request_controls(R"({"presence_penalty":1.5})", controls, err));
    CHECK(controls.presence_penalty > 1.49f && controls.presence_penalty < 1.51f);
    CHECK(parse_request_controls(R"({"presence_penalty":-2.0})", controls, err));
    CHECK(controls.presence_penalty == -2.0f);
    CHECK(parse_request_controls(R"({"presence_penalty":2.0})", controls, err));
    CHECK(controls.presence_penalty == 2.0f);
    CHECK(!parse_request_controls(R"({"presence_penalty":-2.1})", controls, err));
    CHECK(!parse_request_controls(R"({"presence_penalty":2.1})", controls, err));
    CHECK(!parse_request_controls(R"({"presence_penalty":"high"})", controls, err));
    // Does not require temperature to be set.
    CHECK(parse_request_controls(R"({"presence_penalty":1.0})", controls, err));
    CHECK(controls.temperature == 0.f);
    return true;
}

bool test_request_controls_frequency_penalty_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.frequency_penalty == 0.f);
    CHECK(parse_request_controls(R"({"frequency_penalty":0})", controls, err));
    CHECK(controls.frequency_penalty == 0.f);
    CHECK(parse_request_controls(R"({"frequency_penalty":1.5})", controls, err));
    CHECK(controls.frequency_penalty > 1.49f && controls.frequency_penalty < 1.51f);
    CHECK(parse_request_controls(R"({"frequency_penalty":-2.0})", controls, err));
    CHECK(controls.frequency_penalty == -2.0f);
    CHECK(parse_request_controls(R"({"frequency_penalty":2.0})", controls, err));
    CHECK(controls.frequency_penalty == 2.0f);
    CHECK(!parse_request_controls(R"({"frequency_penalty":-2.1})", controls, err));
    CHECK(!parse_request_controls(R"({"frequency_penalty":2.1})", controls, err));
    CHECK(!parse_request_controls(R"({"frequency_penalty":"high"})", controls, err));
    // Does not require temperature to be set.
    CHECK(parse_request_controls(R"({"frequency_penalty":1.0})", controls, err));
    CHECK(controls.temperature == 0.f);
    return true;
}

bool test_should_reject_dflash_temperature() {
    CHECK(!should_reject_dflash_temperature(/*dflash_env_on=*/false, /*temperature=*/0.7f));
    CHECK(!should_reject_dflash_temperature(/*dflash_env_on=*/true, /*temperature=*/0.f));
    CHECK(should_reject_dflash_temperature(/*dflash_env_on=*/true, /*temperature=*/0.7f));
    return true;
}

bool test_should_reject_dflash_penalty() {
    CHECK(!should_reject_dflash_penalty(/*dflash_env_on=*/false, /*presence=*/1.0f, /*frequency=*/1.0f));
    CHECK(!should_reject_dflash_penalty(/*dflash_env_on=*/true, /*presence=*/0.f, /*frequency=*/0.f));
    CHECK(should_reject_dflash_penalty(/*dflash_env_on=*/true, /*presence=*/1.0f, /*frequency=*/0.f));
    CHECK(should_reject_dflash_penalty(/*dflash_env_on=*/true, /*presence=*/0.f, /*frequency=*/1.0f));
    CHECK(should_reject_dflash_penalty(/*dflash_env_on=*/true, /*presence=*/-1.0f, /*frequency=*/0.f));
    return true;
}

bool test_request_controls_logit_bias_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.logit_bias.empty());

    CHECK(parse_request_controls(R"({"logit_bias":{"1234":-100,"5678":5.5}})", controls, err));
    CHECK(controls.logit_bias.size() == 2);
    auto find = [&](int id) -> const float* {
        for (const auto& p : controls.logit_bias) if (p.first == id) return &p.second;
        return nullptr;
    };
    CHECK(find(1234) && *find(1234) == -100.f);
    CHECK(find(5678) && *find(5678) > 5.49f && *find(5678) < 5.51f);

    // Boundary values accepted, one past the boundary rejected.
    CHECK(parse_request_controls(R"({"logit_bias":{"0":-100,"1":100}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"0":-100.1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"0":100.1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"0":"high"}})", controls, err));

    // Not an object.
    CHECK(!parse_request_controls(R"({"logit_bias":[1,2,3]})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":"nope"})", controls, err));

    // Malformed / non-integer / negative keys, all rejected.
    CHECK(!parse_request_controls(R"({"logit_bias":{"12a":1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"1.5":1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"":1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{" 12":1}})", controls, err));
    CHECK(!parse_request_controls(R"({"logit_bias":{"-1":1}})", controls, err));

    // Entry-count cap.
    {
        std::string body = "{\"logit_bias\":{";
        for (int i = 0; i < 1025; i++) body += (i ? "," : "") + std::string("\"") + std::to_string(i) + "\":1";
        body += "}}";
        CHECK(!parse_request_controls(body, controls, err));
    }

    // vocab == 0 (default): upper bound not checked -- an absurdly large id is still accepted.
    CHECK(parse_request_controls(R"({"logit_bias":{"999999999":1}})", controls, err));
    // With an explicit vocab: in-range accepted, >= vocab rejected.
    CHECK(parse_request_controls(R"({"logit_bias":{"151935":1}})", controls, err, /*vocab=*/151936));
    CHECK(!parse_request_controls(R"({"logit_bias":{"151936":1}})", controls, err, /*vocab=*/151936));

    // Does not require temperature to be set.
    CHECK(parse_request_controls(R"({"logit_bias":{"0":1}})", controls, err));
    CHECK(controls.temperature == 0.f);
    return true;
}

bool test_should_reject_dflash_logit_bias() {
    CHECK(!should_reject_dflash_logit_bias(/*dflash_env_on=*/false, /*has_logit_bias=*/true));
    CHECK(!should_reject_dflash_logit_bias(/*dflash_env_on=*/true, /*has_logit_bias=*/false));
    CHECK(should_reject_dflash_logit_bias(/*dflash_env_on=*/true, /*has_logit_bias=*/true));
    return true;
}

bool test_request_controls_n_validation() {
    RequestControls controls;
    std::string err;
    CHECK(parse_request_controls(R"({})", controls, err));
    CHECK(controls.n == 1);
    CHECK(parse_request_controls(R"({"n":1})", controls, err));
    CHECK(controls.n == 1);
    CHECK(parse_request_controls(R"({"n":8})", controls, err));
    CHECK(controls.n == 8);
    CHECK(!parse_request_controls(R"({"n":0})", controls, err));
    CHECK(!parse_request_controls(R"({"n":-1})", controls, err));
    CHECK(!parse_request_controls(R"({"n":9})", controls, err));
    CHECK(!parse_request_controls(R"({"n":1.5})", controls, err));
    CHECK(!parse_request_controls(R"({"n":"3"})", controls, err));
    // Does not require temperature/seed to be set.
    CHECK(parse_request_controls(R"({"n":3})", controls, err));
    CHECK(controls.temperature == 0.f);
    CHECK(!controls.seed_set);
    return true;
}

bool test_parse_legacy_completion_request() {
    std::string prompt, err;
    bool echo = false;

    CHECK(parse_legacy_completion_request(R"({"prompt":"Say hi."})", prompt, echo, err));
    CHECK(prompt == "Say hi.");
    CHECK(!echo);

    CHECK(parse_legacy_completion_request(R"({"prompt":"Say hi.","echo":true})", prompt, echo, err));
    CHECK(echo);
    CHECK(parse_legacy_completion_request(R"({"prompt":"Say hi.","echo":false})", prompt, echo, err));
    CHECK(!echo);
    CHECK(!parse_legacy_completion_request(R"({"prompt":"Say hi.","echo":"yes"})", prompt, echo, err));

    // prompt required, must be a non-empty string -- arrays/ints/missing all rejected (v1 scope).
    CHECK(!parse_legacy_completion_request(R"({})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":null})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":123})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":["a","b"]})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":[1,2,3]})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":""})", prompt, echo, err));

    // suffix (fill-in-the-middle) unsupported, always rejected when set.
    CHECK(!parse_legacy_completion_request(R"({"prompt":"x","suffix":"y"})", prompt, echo, err));
    CHECK(parse_legacy_completion_request(R"({"prompt":"x","suffix":null})", prompt, echo, err));

    // best_of: default/1 accepted (no-op), >1 rejected (unsupported), <1 rejected (never valid).
    CHECK(parse_legacy_completion_request(R"({"prompt":"x","best_of":1})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":"x","best_of":2})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":"x","best_of":1.5})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":"x","best_of":0})", prompt, echo, err));
    CHECK(!parse_legacy_completion_request(R"({"prompt":"x","best_of":-1})", prompt, echo, err));
    return true;
}

bool test_request_controls_legacy_logprobs_validation() {
    RequestControls controls;
    std::string err;

    // Legacy mode: logprobs is an integer (how many top logprobs), no separate top_logprobs field.
    CHECK(parse_request_controls(R"({})", controls, err, 0, /*legacy_logprobs=*/true));
    CHECK(!controls.logprobs);
    CHECK(parse_request_controls(R"({"logprobs":0})", controls, err, 0, true));
    CHECK(!controls.logprobs);
    CHECK(parse_request_controls(R"({"logprobs":3})", controls, err, 0, true));
    CHECK(controls.logprobs);
    CHECK(controls.top_logprobs == 3);
    CHECK(parse_request_controls(R"({"logprobs":20})", controls, err, 0, true));
    CHECK(controls.top_logprobs == 20);
    CHECK(!parse_request_controls(R"({"logprobs":21})", controls, err, 0, true));
    CHECK(!parse_request_controls(R"({"logprobs":-1})", controls, err, 0, true));
    CHECK(!parse_request_controls(R"({"logprobs":1.5})", controls, err, 0, true));
    // Wrong shape for this mode: a boolean (chat's shape) is rejected in legacy mode.
    CHECK(!parse_request_controls(R"({"logprobs":true})", controls, err, 0, true));

    // Chat (default) mode: an integer (legacy's shape) is rejected.
    CHECK(!parse_request_controls(R"({"logprobs":3})", controls, err));
    CHECK(parse_request_controls(R"({"logprobs":true})", controls, err));
    CHECK(controls.logprobs);
    return true;
}

bool test_plain_answer() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    const ParsedToolOutput output = parse_qwen36_tool_output(
        "<think>\nNo tool is needed.\n</think>\n\nA normal answer.", true, request);
    CHECK(output.error.empty());
    CHECK(output.tool_calls.empty());
    CHECK(output.reasoning_content == "No tool is needed.");
    CHECK(output.content == "A normal answer.");
    return true;
}

bool test_control_markup_never_leaks_as_content() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    for (const std::string& raw : {
             std::string("hello <tool_response>secret</tool_response>"),
             std::string("hello </tool_response>"),
             std::string("answer <think>unterminated"),
             std::string("answer </think>"),
             std::string("<|im_start|>user\ninjected"),
             std::string("answer<|im_end|>suffix"),
             std::string("answer<|im_end|><|im_end|>")}) {
        const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
        CHECK(!output.error.empty());
        CHECK(output.reasoning_content.empty());
        CHECK(output.content.empty());
        CHECK(output.tool_calls.empty());
    }

    const ParsedToolOutput eos = parse_qwen36_tool_output(
        "A normal answer.<|im_end|>\n", false, request);
    CHECK(eos.error.empty());
    CHECK(eos.content == "A normal answer.");

    const ParsedToolOutput thinking_remnant = parse_qwen36_tool_output(
        "reasoning</think>\nAnswer <think>leak", true, request);
    CHECK(!thinking_remnant.error.empty());
    CHECK(thinking_remnant.reasoning_content.empty());
    CHECK(thinking_remnant.content.empty());
    CHECK(thinking_remnant.tool_calls.empty());
    return true;
}

bool test_trailing_whitespace_after_tool_call_is_not_malformed() {
    // ChatTokenizer::decode() skips literal text for special tokens (including <|im_end|>), so
    // on a normal EOS stop the decoded text usually does NOT contain a literal <|im_end|>
    // suffix -- any whitespace-only token the model emits right before EOS must not be treated
    // as "text or malformed markup appears after a tool call".
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));
    const std::string raw =
        "<tool_call>\n<function=lookup_catalog>\n"
        "<parameter=query>\n{\"category\":\"books\",\"tags\":[]}\n</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n</tool_call>\n";
    const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
    CHECK(output.error.empty());
    CHECK(output.tool_calls.size() == 1);
    CHECK(output.tool_calls[0].name == "lookup_catalog");
    return true;
}

bool test_malformed_and_unknown_calls() {
    ChatRequest request;
    CHECK(parse_request(hermes_request(), request));

    for (const std::string& raw : {
             std::string("<tool_call>\n<function=lookup_catalog>\n<parameter=limit>\n2\n</parameter>"),
             std::string("<tool_call>\n<function=does_not_exist>\n</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=lookup_catalog>\n"
                         "<parameter=query>\nnot-an-object\n</parameter>\n"
                         "<parameter=limit>\n2\n</parameter>\n"
                         "</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=lookup_catalog>\n"
                         "<parameter=query>\n{}\n</parameter>\n"
                         "<parameter=limit>\ntwo\n</parameter>\n"
                         "</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=lookup_catalog>\n"
                         "<parameter=query>\n{}\n</parameter>\n"
                         "<parameter=query>\n{}\n</parameter>\n"
                         "<parameter=limit>\n2\n</parameter>\n"
                         "</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=lookup_catalog>\n"
                         "<parameter=query>\n{}\n</parameter>\n"
                         "<parameter=limit>\n2\n</parameter>\n"
                         "<parameter=surprise>\ntrue\n</parameter>\n"
                         "</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=summarize_items>\n"
                         "<parameter=titles>\n[\"Moby-Dick\"]\n</parameter>\n"
                         "<parameter=style>\nverbose\n</parameter>\n"
                         "</function>\n</tool_call>"),
             std::string("<tool_call>\n<function=lookup_catalog>\n"
                         "<parameter=query>\n{}\n</parameter>\n"
                         "</function>\n</tool_call>")}) {
        const ParsedToolOutput output = parse_qwen36_tool_output(raw, false, request);
        CHECK(!output.error.empty());
        CHECK(output.tool_calls.empty());
    }

    ChatRequest none_request;
    std::string none_body = hermes_request();
    const size_t final_brace = none_body.find_last_of('}');
    CHECK(final_brace != std::string::npos);
    none_body.insert(final_brace, ",\"tool_choice\":\"none\"");
    CHECK(parse_request(none_body, none_request));
    const std::string valid_call =
        "<tool_call>\n<function=lookup_catalog>\n"
        "<parameter=query>\n{}\n</parameter>\n"
        "<parameter=limit>\n2\n</parameter>\n"
        "</function>\n</tool_call>";
    const ParsedToolOutput forbidden = parse_qwen36_tool_output(valid_call, false, none_request);
    CHECK(!forbidden.error.empty());
    CHECK(forbidden.tool_calls.empty());
    return true;
}

bool test_invalid_requests_and_duplicate_tools() {
    for (const std::string& body : {
             std::string("{"),
             std::string("[]"),
             std::string(R"JSON({"messages":[]})JSON"),
             std::string(R"JSON({
               "messages":[{"role":"user","content":"hi"}],
               "tools":[
                 {"type":"function","function":{"name":"same","parameters":{"type":"object"}}},
                 {"type":"function","function":{"name":"same","parameters":{"type":"object"}}}
               ]
             })JSON")}) {
        ChatRequest request;
        std::string error;
        CHECK(!parse_chat_request_json(body, request, error));
        CHECK(!error.empty());
    }
    return true;
}

bool test_unsafe_protocol_names() {
    auto request_with_names = [](const std::string& function_name,
                                 const std::string& parameter_name) {
        json body = {
            {"messages", {{{"role", "user"}, {"content", "test"}}}},
            {"tools", {{
                {"type", "function"},
                {"function", {
                    {"name", function_name},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {{parameter_name, {{"type", "string"}}}}}
                    }}
                }}
            }}}
        };
        return body.dump();
    };

    ChatRequest valid;
    CHECK(parse_request(request_with_names("safe_name-1", "safe_parameter-1"), valid));
    for (const std::string& name : {
             std::string("bad name"), std::string("bad\tname"), std::string("bad\nname"),
             std::string("bad<name"), std::string("bad>name"), std::string("bad&name"),
             std::string("bad\"name"), std::string("bad'name")}) {
        ChatRequest request;
        std::string error;
        CHECK(!parse_chat_request_json(request_with_names(name, "safe"), request, error));
        CHECK(!error.empty());
    }
    for (const std::string& name : {
             std::string("bad parameter"), std::string("bad\tparameter"),
             std::string("bad\nparameter"), std::string("bad<parameter"),
             std::string("bad>parameter"), std::string("bad&parameter"),
             std::string("bad\"parameter"), std::string("bad'parameter")}) {
        ChatRequest request;
        std::string error;
        CHECK(!parse_chat_request_json(request_with_names("safe", name), request, error));
        CHECK(!error.empty());
    }
    return true;
}

} // namespace


// ---- POST /v1/score (teacher-forced scoring) -------------------------------------------------
bool test_parse_score_request() {
    ScoreRequest r;
    std::string err;

    // messages + completion text: the ordinary audit shape.
    CHECK(parse_score_request(
        R"({"model":"m","messages":[{"role":"user","content":"hi"}],"completion":"there"})", r, err));
    CHECK(r.use_messages);
    CHECK(r.prompt.empty());
    CHECK(!r.completion_is_ids);
    CHECK(r.completion == "there");
    CHECK(r.top_logprobs == 0);

    // prompt + token ids: the drift-free shape a verifier should prefer.
    r = ScoreRequest{};
    CHECK(parse_score_request(R"({"prompt":"raw text","completion_token_ids":[1,2,3]})", r, err));
    CHECK(!r.use_messages);
    CHECK(r.prompt == "raw text");
    CHECK(r.completion_is_ids);
    CHECK(r.completion_token_ids == std::vector<int>({1, 2, 3}));

    // top_logprobs bounds.
    r = ScoreRequest{};
    CHECK(parse_score_request(R"({"prompt":"p","completion":"c","top_logprobs":20})", r, err));
    CHECK(r.top_logprobs == 20);
    CHECK(parse_score_request(R"({"prompt":"p","completion":"c","top_logprobs":0})", r, err));
    CHECK(!parse_score_request(R"({"prompt":"p","completion":"c","top_logprobs":21})", r, err));
    CHECK(contains(err, "between 0 and 20"));
    CHECK(!parse_score_request(R"({"prompt":"p","completion":"c","top_logprobs":-1})", r, err));
    CHECK(!parse_score_request(R"({"prompt":"p","completion":"c","top_logprobs":1.5})", r, err));
    CHECK(contains(err, "must be an integer"));

    // Exactly one prompt form. Both and neither are errors, NOT a silent precedence rule: a
    // verifier that believes it pinned one and silently got the other compares the wrong thing.
    CHECK(!parse_score_request(
        R"({"messages":[{"role":"user","content":"hi"}],"prompt":"p","completion":"c"})", r, err));
    CHECK(contains(err, "exactly one of `messages` or `prompt`"));
    CHECK(!parse_score_request(R"({"completion":"c"})", r, err));
    CHECK(contains(err, "exactly one of `messages` or `prompt`"));

    // Exactly one completion form, same reasoning.
    CHECK(!parse_score_request(R"({"prompt":"p","completion":"c","completion_token_ids":[1]})", r, err));
    CHECK(contains(err, "exactly one of `completion` or `completion_token_ids`"));
    CHECK(!parse_score_request(R"({"prompt":"p"})", r, err));
    CHECK(contains(err, "exactly one of `completion` or `completion_token_ids`"));

    // Nothing to score is an error, not a zero-length success -- a caller that gets back an empty
    // logprobs array would read it as "scored, and it agreed".
    CHECK(!parse_score_request(R"({"prompt":"p","completion":""})", r, err));
    CHECK(contains(err, "no tokens to score"));
    CHECK(!parse_score_request(R"({"prompt":"p","completion_token_ids":[]})", r, err));
    CHECK(contains(err, "no tokens to score"));
    CHECK(!parse_score_request(R"({"prompt":"","completion":"c"})", r, err));
    CHECK(contains(err, "prompt is empty"));

    // Token id typing and range. vocab=0 skips only the UPPER bound; negatives are always rejected.
    CHECK(!parse_score_request(R"({"prompt":"p","completion_token_ids":[1,"2"]})", r, err));
    CHECK(contains(err, "only integers"));
    CHECK(!parse_score_request(R"({"prompt":"p","completion_token_ids":[1,-3]})", r, err));
    CHECK(contains(err, "out of range"));
    CHECK(!parse_score_request(R"({"prompt":"p","completion_token_ids":[1,999]})", r, err, /*vocab=*/100));
    CHECK(contains(err, "out of range"));
    CHECK(parse_score_request(R"({"prompt":"p","completion_token_ids":[1,99]})", r, err, /*vocab=*/100));
    CHECK(parse_score_request(R"({"prompt":"p","completion_token_ids":[999999]})", r, err, /*vocab=*/0));
    CHECK(!parse_score_request(R"({"prompt":"p","completion_token_ids":{"a":1}})", r, err));
    CHECK(contains(err, "array of integers"));

    // Wrong scalar types.
    CHECK(!parse_score_request(R"({"prompt":123,"completion":"c"})", r, err));
    CHECK(contains(err, "prompt must be a string"));
    CHECK(!parse_score_request(R"({"prompt":"p","completion":123})", r, err));
    CHECK(contains(err, "completion must be a string"));

    // Malformed bodies.
    CHECK(!parse_score_request("not json", r, err));
    CHECK(contains(err, "invalid JSON"));
    CHECK(!parse_score_request("[1,2,3]", r, err));
    CHECK(contains(err, "must be a JSON object"));

    // A completion that happens to contain an end marker is still just text to score -- nothing
    // here may treat it as a stop condition (the engine's forced path suppresses EOS for the same
    // reason: the caller is owed a logprob for every token they asked about).
    r = ScoreRequest{};
    CHECK(parse_score_request(R"({"prompt":"p","completion":"a<|im_end|>b"})", r, err));
    CHECK(r.completion == "a<|im_end|>b");
    return true;
}

int main() {
    if (!test_hermes_request_and_template()) return 1;
    if (!test_tool_history_round_trip()) return 1;
    if (!test_tool_choice_modes()) return 1;
    if (!test_tool_choice_none_preserves_history()) return 1;
    if (!test_one_tool_call_and_schema_types()) return 1;
    if (!test_implicit_thinking_prefill_suffix()) return 1;
    if (!test_all_json_schema_value_types()) return 1;
    if (!test_schema_constraints()) return 1;
    if (!test_schema_pattern_is_linear_time()) return 1;
    if (!test_schema_unions_and_dynamic_properties()) return 1;
    if (!test_unicode_string_lengths()) return 1;
    if (!test_large_integer_bounds_are_exact()) return 1;
    if (!test_schema_keyword_type_applicability()) return 1;
    if (!test_unsupported_schema_keywords_are_rejected()) return 1;
    if (!test_parallel_tool_calls()) return 1;
    if (!test_reasoning_effort_controls()) return 1;
    if (!test_plain_answer()) return 1;
    if (!test_control_markup_never_leaks_as_content()) return 1;
    if (!test_trailing_whitespace_after_tool_call_is_not_malformed()) return 1;
    if (!test_nontext_content_parts_are_ignored_not_rejected()) return 1;
    if (!test_image_parts_render_a_marker_and_collect_urls()) return 1;
    if (!test_image_in_system_message_is_rejected()) return 1;
    if (!test_malformed_image_url_is_rejected()) return 1;
    if (!test_developer_role_is_accepted_as_system_alias()) return 1;
    if (!test_response_format_type_validation()) return 1;
    if (!test_response_format_json_object_parses()) return 1;
    if (!test_response_format_json_schema_validation()) return 1;
    if (!test_response_format_strict_does_not_change_parse_outcome()) return 1;
    if (!test_tools_and_response_format_rejected_together()) return 1;
    if (!test_response_format_steers_prompt_json_object()) return 1;
    if (!test_response_format_steers_prompt_json_schema_escaped()) return 1;
    if (!test_response_format_text_is_a_prompt_noop()) return 1;
    if (!test_validate_response_format_json_object()) return 1;
    if (!test_validate_response_format_json_schema()) return 1;
    if (!test_request_controls_temperature_validation()) return 1;
    if (!test_request_controls_seed_validation()) return 1;
    if (!test_request_controls_top_p_validation()) return 1;
    if (!test_request_controls_top_k_validation()) return 1;
    if (!test_request_controls_logprobs_validation()) return 1;
    if (!test_request_controls_presence_penalty_validation()) return 1;
    if (!test_request_controls_frequency_penalty_validation()) return 1;
    if (!test_should_reject_dflash_temperature()) return 1;
    if (!test_should_reject_dflash_penalty()) return 1;
    if (!test_request_controls_logit_bias_validation()) return 1;
    if (!test_should_reject_dflash_logit_bias()) return 1;
    if (!test_request_controls_n_validation()) return 1;
    if (!test_parse_legacy_completion_request()) return 1;
    if (!test_parse_score_request()) return 1;
    if (!test_request_controls_legacy_logprobs_validation()) return 1;
    if (!test_malformed_and_unknown_calls()) return 1;
    if (!test_invalid_requests_and_duplicate_tools()) return 1;
    if (!test_unsafe_protocol_names()) return 1;
    std::printf("chat_tools_test: OK\n");
    return 0;
}
