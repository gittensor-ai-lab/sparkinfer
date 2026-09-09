// Unit tests for Ollama /api/* request and response translation. Pure JSON, no engine or GPU.
//
// The assertions are about exact field names, units and value vocabularies. An Ollama client
// reads `message.content`, switches on `done`, and divides durations assuming NANOSECONDS -- each
// of those is a way to be wrong while still producing valid JSON.

#include "ollama_api.hpp"

#include <cstdio>

using namespace sparkinfer_server::ollama;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

int main() {
    // ---- model naming: Ollama addresses models as "<name>:<tag>" ----
    CHECK(with_latest_tag("spark-x2.5-4b") == "spark-x2.5-4b:latest");
    CHECK(with_latest_tag("spark-x2.5-4b:q8") == "spark-x2.5-4b:q8");
    CHECK(model_name_matches("spark-x2.5-4b", "spark-x2.5-4b"));
    CHECK(model_name_matches("spark-x2.5-4b:latest", "spark-x2.5-4b"));
    CHECK(model_name_matches("spark-x2.5-4b:anything", "spark-x2.5-4b"));
    CHECK(model_name_matches("", "spark-x2.5-4b"));          // absent == the loaded model
    CHECK(!model_name_matches("llama3.2", "spark-x2.5-4b"));

    // ---- /api/tags ----
    {
        ModelEntry m;
        m.name = "spark-x2.5-4b:latest"; m.model = m.name;
        m.modified_at = "2026-09-09T20:00:00Z"; m.size = 4380000000LL; m.digest = "abc";
        m.details.family = "spark2_5"; m.details.families = {"spark2_5"};
        m.details.parameter_size = "4.1B"; m.details.quantization_level = "Q8_0";
        const nlohmann::json j = tags_list({m});
        CHECK(j.contains("models") && j["models"].is_array() && j["models"].size() == 1);
        const auto& e = j["models"][0];
        for (const char* k : {"name","model","modified_at","size","digest","details"}) CHECK(e.contains(k));
        for (const char* k : {"parent_model","format","family","families","parameter_size","quantization_level"})
            CHECK(e["details"].contains(k));
        CHECK(e["details"]["format"] == "gguf");
        CHECK(ps_list({m})["models"].size() == 1);
        CHECK(tags_list({})["models"].is_array() && tags_list({})["models"].empty());
    }

    // ---- request translation: options -> OpenAI top level, stream forced false ----
    {
        const nlohmann::json in = {
            {"model","spark-x2.5-4b"},
            {"messages",{{{"role","user"},{"content","hi"}}}},
            {"stream", true},                                  // must NOT survive
            {"options", {{"temperature",0.2},{"top_p",0.9},{"num_predict",64},{"seed",7}}},
        };
        const nlohmann::json o = chat_request_to_openai(in);
        CHECK(o["stream"] == false);
        CHECK(o["max_tokens"] == 64);
        CHECK(o["temperature"] == 0.2);
        CHECK(o["top_p"] == 0.9);
        CHECK(o["seed"] == 7);
        CHECK(o["messages"].size() == 1);
    }
    // num_predict = -1 ("until context is full") has no OpenAI equivalent and must be DROPPED,
    // not turned into a finite cap the caller never asked for.
    {
        const nlohmann::json in = {{"model","m"},{"messages",nlohmann::json::array()},
                                   {"options",{{"num_predict",-1}}}};
        CHECK(!chat_request_to_openai(in).contains("max_tokens"));
    }
    // format:"json" -> response_format
    {
        const nlohmann::json in = {{"model","m"},{"messages",nlohmann::json::array()},{"format","json"}};
        CHECK(chat_request_to_openai(in)["response_format"]["type"] == "json_object");
    }
    {
        const nlohmann::json in = {{"model","m"},{"messages",nlohmann::json::array()},
                                   {"format",{{"type","object"}}}};
        CHECK(chat_request_to_openai(in)["response_format"]["type"] == "json_schema");
    }
    // generate
    {
        const nlohmann::json in = {{"model","m"},{"prompt","hello"},{"options",{{"num_predict",8}}}};
        const nlohmann::json o = generate_request_to_openai(in);
        CHECK(o["prompt"] == "hello" && o["stream"] == false && o["max_tokens"] == 8);
    }
    // An EMPTY suffix must not be forwarded: the ollama CLI sends "suffix":"" on an ordinary
    // `ollama run`, and passing it through made /v1/completions 400 the whole request.
    {
        const nlohmann::json in = {{"model","m"},{"prompt","hi"},{"suffix",""}};
        CHECK(!generate_request_to_openai(in).contains("suffix"));
    }
    { // a real suffix still gets through
        const nlohmann::json in = {{"model","m"},{"prompt","hi"},{"suffix","tail"}};
        CHECK(generate_request_to_openai(in)["suffix"] == "tail");
    }

    // ---- /api/generate is TEMPLATE-AWARE by default, raw only on request ----
    // Ollama applies the model's template to `prompt`; OpenAI's /v1/completions does not. Mapping
    // one onto the other made `ollama run "Reply with exactly: OK"` answer "OK: OK: OK: OK...".
    CHECK(!generate_wants_raw({{"model","m"},{"prompt","hi"}}));            // default = templated
    CHECK(generate_wants_raw({{"model","m"},{"prompt","hi"},{"raw",true}}));
    {
        const nlohmann::json o = generate_request_to_chat({{"model","m"},{"prompt","hi"}});
        CHECK(o.contains("messages") && !o.contains("prompt"));
        CHECK(o["messages"].size() == 1);
        CHECK(o["messages"][0]["role"] == "user" && o["messages"][0]["content"] == "hi");
        CHECK(o["stream"] == false);
    }
    { // a system field becomes a system turn ahead of the user turn
        const nlohmann::json o = generate_request_to_chat(
            {{"model","m"},{"prompt","hi"},{"system","be terse"}});
        CHECK(o["messages"].size() == 2);
        CHECK(o["messages"][0]["role"] == "system" && o["messages"][0]["content"] == "be terse");
        CHECK(o["messages"][1]["role"] == "user");
    }
    { // options still map through on the chat path
        const nlohmann::json o = generate_request_to_chat(
            {{"model","m"},{"prompt","hi"},{"options",{{"num_predict",12},{"temperature",0}}}});
        CHECK(o["max_tokens"] == 12 && o["temperature"] == 0);
    }
    // the generate response reads EITHER upstream shape
    {
        const nlohmann::json from_chat = {{"choices",{{{"finish_reason","stop"},
            {"message",{{"role","assistant"},{"content","Paris"}}}}}},{"usage",nlohmann::json::object()}};
        CHECK(openai_to_generate_response(from_chat,"m","t")["response"] == "Paris");
        const nlohmann::json from_text = {{"choices",{{{"finish_reason","stop"},{"text","Paris"}}}},
                                          {"usage",nlohmann::json::object()}};
        CHECK(openai_to_generate_response(from_text,"m","t")["response"] == "Paris");
    }

    // ---- durations are NANOSECONDS ----
    CHECK(ms_to_ns(1.0) == 1000000LL);
    CHECK(ms_to_ns(85.982577) == 85982577LL);
    CHECK(ms_to_ns(0.0) == 0);
    CHECK(ms_to_ns(-5.0) == 0);        // never emit a negative duration

    // ---- chat response translation ----
    {
        const nlohmann::json oai = {
            {"choices",{{{"index",0},{"finish_reason","stop"},
                         {"message",{{"role","assistant"},{"content","OK"}}}}}},
            {"usage",{{"prompt_tokens",21},{"completion_tokens",2},
                      {"ttft_ms",82.11786},{"generation_ms",85.707905}}},
        };
        const nlohmann::json j = openai_to_chat_response(oai, "spark-x2.5-4b:latest", "2026-09-09T20:00:00Z");
        CHECK(j["model"] == "spark-x2.5-4b:latest");
        CHECK(j["message"]["role"] == "assistant");
        CHECK(j["message"]["content"] == "OK");       // NOT choices[0].message.content
        CHECK(j["done"] == true);
        CHECK(j["done_reason"] == "stop");
        CHECK(j["prompt_eval_count"] == 21);
        CHECK(j["eval_count"] == 2);
        CHECK(j["prompt_eval_duration"] == 82117860LL);
        CHECK(j["eval_duration"] == 85707905LL);
        CHECK(j["total_duration"] == ms_to_ns(82.11786 + 85.707905));
        CHECK(j["load_duration"] == 0);               // model was loaded at startup, not per-request
        CHECK(!j.contains("choices") && !j.contains("usage"));   // OpenAI shape must not leak
    }
    // finish_reason "length" -> done_reason "length"
    {
        const nlohmann::json oai = {{"choices",{{{"finish_reason","length"},
                                     {"message",{{"content","x"}}}}}},{"usage",nlohmann::json::object()}};
        CHECK(openai_to_chat_response(oai,"m","t")["done_reason"] == "length");
    }
    // an empty/degenerate OpenAI body must still produce a well-formed Ollama response
    {
        const nlohmann::json j = openai_to_chat_response(nlohmann::json::object(), "m", "t");
        CHECK(j["done"] == true && j["message"]["content"] == "" && j["eval_count"] == 0);
    }

    // ---- generate response translation ----
    {
        const nlohmann::json oai = {
            {"choices",{{{"index",0},{"finish_reason","length"},{"text"," Paris."}}}},
            {"usage",{{"prompt_tokens",5},{"completion_tokens",3},
                      {"ttft_ms",10.0},{"generation_ms",20.0}}}};
        const nlohmann::json j = openai_to_generate_response(oai, "m:latest", "t");
        CHECK(j["response"] == " Paris.");            // NOT choices[0].text
        CHECK(j["done"] == true && j["done_reason"] == "length");
        CHECK(j["eval_count"] == 3);
        // "context" is opaque conversation state this server does not keep; fabricating one would
        // invite the client to send it back as if it meant something.
        CHECK(!j.contains("context"));
    }

    // ---- digest: exactly 64 lowercase hex, stable, and NEVER empty ----
    // `ollama list`/`ps` render digest[:12] with no length check, so an empty or short value
    // panics the official CLI. This is the regression test for that crash.
    {
        const std::string d = synthetic_digest("/m/x.gguf|123|2026-01-01T00:00:00Z");
        CHECK(d.size() == 64);
        for (char c : d) CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        CHECK(d.size() >= 12);                                   // the exact CLI slice
        CHECK(synthetic_digest("/m/x.gguf|123|2026-01-01T00:00:00Z") == d);   // stable
        CHECK(synthetic_digest("/m/y.gguf|123|2026-01-01T00:00:00Z") != d);   // path matters
        CHECK(synthetic_digest("/m/x.gguf|999|2026-01-01T00:00:00Z") != d);   // size matters
        CHECK(synthetic_digest("").size() == 64);                // even a degenerate seed is safe
    }

    // ---- created_at is RFC3339 UTC ----
    {
        const std::string ts = rfc3339_now();
        CHECK(ts.size() == 20);
        CHECK(ts[4]=='-' && ts[7]=='-' && ts[10]=='T' && ts[13]==':' && ts[16]==':' && ts[19]=='Z');
    }

    std::printf("ollama_api_test: OK\n");
    return 0;
}
