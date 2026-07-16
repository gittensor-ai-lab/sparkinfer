#include "chat_tokenizer.hpp"
#include "model_engine.hpp"

// Do not define CPPHTTPLIB_OPENSSL_SUPPORT — even `= 0` enables OpenSSL in httplib.
#include "../third_party/httplib.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxOutputTokens = 4096;
constexpr int kMaxInputContext = 32768;

std::string g_api_key;
std::string g_model_name = "qwen3.6-35b-a3b";
sparkinfer_server::ChatTokenizer g_tokenizer;

std::string repo_root() {
    const char* env = getenv("SPARKINFER_ROOT");
    if (env && *env) return env;
    return ".";
}

// Minimal JSON helpers (avoid extra deps on this branch).
std::string json_get_string(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return {};
    p = body.find(':', p);
    if (p == std::string::npos) return {};
    p = body.find('"', p);
    if (p == std::string::npos) return {};
    size_t e = body.find('"', p + 1);
    if (e == std::string::npos) return {};
    return body.substr(p + 1, e - p - 1);
}

bool json_get_bool(const std::string& body, const std::string& key, bool def) {
    const std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return def;
    p = body.find(':', p);
    if (p == std::string::npos) return def;
    if (body.find("true", p) != std::string::npos && body.find("true", p) < body.find(',', p))
        return true;
    if (body.find("false", p) != std::string::npos && body.find("false", p) < body.find(',', p))
        return false;
    return def;
}

int json_get_int(const std::string& body, const std::string& key, int def) {
    const std::string needle = "\"" + key + "\"";
    size_t p = body.find(needle);
    if (p == std::string::npos) return def;
    p = body.find(':', p);
    if (p == std::string::npos) return def;
    return atoi(body.c_str() + p + 1);
}

std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << c;
        }
    }
    return o.str();
}

std::string random_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << dist(rng);
    return ss.str();
}

std::string usage_json(int prompt_tokens, int completion_tokens) {
    std::ostringstream o;
    const int total = prompt_tokens + completion_tokens;
    o << "\"usage\":{\"prompt_tokens\":" << prompt_tokens << ",\"completion_tokens\":" << completion_tokens
      << ",\"total_tokens\":" << total << "}";
    return o.str();
}

bool auth_ok(const httplib::Request& req) {
    if (g_api_key.empty()) return true;
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) return false;
    const std::string prefix = "Bearer ";
    return it->second.size() > prefix.size() &&
           it->second.compare(0, prefix.size(), prefix) == 0 &&
           it->second.substr(prefix.size()) == g_api_key;
}

bool encode_messages(const std::string& body, std::vector<int>& ids, bool enable_thinking, std::string& err) {
    return g_tokenizer.encode_chat_request(body, ids, enable_thinking, err);
}

void write_stream_delta(httplib::DataSink& sink, const std::string& cid, long long created, const std::string& field,
                        const std::string& piece) {
    if (piece.empty()) return;
    std::ostringstream chunk;
    chunk << "data: {\"id\":\"" << cid << "\",\"object\":\"chat.completion.chunk\","
          << "\"created\":" << created << ",\"model\":\"" << g_model_name << "\","
          << "\"choices\":[{\"index\":0,\"delta\":{\"" << field << "\":\"" << json_escape(piece)
          << "\"},\"finish_reason\":null}]}\n\n";
    sink.write(chunk.str().c_str(), (size_t)chunk.str().size());
}

bool decode_ids(const std::vector<int>& ids, std::string& text, std::string& err) {
    text = g_tokenizer.decode(ids);
    if (text.empty() && !ids.empty()) {
        err = "detokenize returned empty text";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string tokenizer_json;
    int ctx = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* flag) { return a == flag && i + 1 < argc; };
        if (need("-m") || need("--model")) model_path = argv[++i];
        else if (need("--host")) host = argv[++i];
        else if (need("--port")) port = atoi(argv[++i]);
        else if (need("--ctx")) ctx = atoi(argv[++i]);
        else if (need("--api-key")) g_api_key = argv[++i];
        else if (need("--tokenizer")) tokenizer_json = argv[++i];
        else if (need("--model-name")) g_model_name = argv[++i];
        else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                    "usage: %s -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx N] "
                    "[--tokenizer path/to/tokenizer.json] [--model-name ID] [--api-key KEY]\n",
                    argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "error: -m model.gguf is required\n");
        return 2;
    }

    const std::string root = repo_root();
    std::string tok_path = tokenizer_json.empty() ? root + "/models/tokenizer.json" : tokenizer_json;
    std::string tok_err;
    if (!g_tokenizer.load(tok_path, tok_err)) {
        fprintf(stderr, "[sparkinfer-server] %s\n", tok_err.c_str());
        return 1;
    }

    sparkinfer_server::ModelEngine engine;
    if (!engine.load(model_path, ctx > 0 ? ctx : 0)) return 1;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        body << "{\"object\":\"list\",\"data\":[{\"id\":\"" << g_model_name
             << "\",\"object\":\"model\",\"owned_by\":\"sparkinfer\"}]}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/v1/info", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"model\":\"" << g_model_name << "\",\"max_context\":" << kMaxInputContext
             << ",\"max_output_tokens\":" << kMaxOutputTokens << "}";
        res.set_content(body.str(), "application/json");
    });

    svr.Post("/v1/tokenize", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
        std::vector<int> ids;
        std::string err;
        if (!encode_messages(req.body, ids, enable_thinking, err)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"tokens\":" << ids.size() << ",\"max_context\":" << kMaxInputContext
             << ",\"max_output_tokens\":" << kMaxOutputTokens << ",\"model\":\"" << g_model_name << "\"}";
        res.set_content(body.str(), "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&engine](const httplib::Request& req, httplib::Response& res) {
                 if (!auth_ok(req)) {
                     res.status = 401;
                     res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
                     return;
                 }
                 if (!engine.loaded()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
                     return;
                 }

                 const bool stream = json_get_bool(req.body, "stream", false);
                 const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, false);
                 int max_tokens = json_get_int(req.body, "max_tokens", 256);
                 if (max_tokens <= 0) max_tokens = 256;
                 if (max_tokens > 4096) max_tokens = 4096;

                 std::vector<int> prompt_ids;
                 std::string err;
                 if (!encode_messages(req.body, prompt_ids, enable_thinking, err)) {
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 if ((int)prompt_ids.size() + max_tokens > engine.max_seq()) {
                     res.status = 400;
                     res.set_content(
                         "{\"error\":{\"message\":\"context overflow: prompt=" +
                         std::to_string(prompt_ids.size()) + " max_tokens=" + std::to_string(max_tokens) +
                         " exceeds server ctx=" + std::to_string(engine.max_seq()) + "\"}}",
                         "application/json");
                     return;
                 }

                 const std::string cid = random_id();
                 const auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();

                 if (stream) {
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [&engine, prompt_ids, max_tokens, cid, created, enable_thinking](size_t offset,
                                                                                          httplib::DataSink& sink) {
                             if (offset > 0) {
                                 sink.done();
                                 return true;
                             }
                             std::vector<int> stream_ids;
                             stream_ids.reserve((size_t)max_tokens);
                             sparkinfer_server::ThinkingStreamSplitter splitter(enable_thinking);
                             auto on_tok = [&](int tid) {
                                 std::string piece = g_tokenizer.decode_delta(stream_ids, tid);
                                 const auto delta = splitter.feed(piece);
                                 write_stream_delta(sink, cid, created, "reasoning_content", delta.reasoning_content);
                                 write_stream_delta(sink, cid, created, "content", delta.content);
                             };
                             engine.complete_streaming(prompt_ids, max_tokens, on_tok);
                             sparkinfer_server::ThinkingStreamSplitter::Delta flush;
                             splitter.finish(flush);
                             write_stream_delta(sink, cid, created, "reasoning_content", flush.reasoning_content);
                             write_stream_delta(sink, cid, created, "content", flush.content);
                             if (!engine.last_error().empty()) {
                                 std::ostringstream err_chunk;
                                 err_chunk << "data: {\"error\":{\"message\":\"" << json_escape(engine.last_error())
                                           << "\"}}\n\n";
                                 sink.write(err_chunk.str().c_str(), (size_t)err_chunk.str().size());
                             }
                             const int prompt_tokens = (int)prompt_ids.size();
                             const int completion_tokens = (int)stream_ids.size();
                             std::ostringstream usage_chunk;
                             usage_chunk << "data: {\"id\":\"" << cid << "\",\"object\":\"chat.completion.chunk\","
                                         << "\"created\":" << created << ",\"model\":\"" << g_model_name << "\","
                                         << "\"choices\":[],"
                                         << usage_json(prompt_tokens, completion_tokens) << "}\n\n";
                             sink.write(usage_chunk.str().c_str(), (size_t)usage_chunk.str().size());
                             std::string tail =
                                 "data: {\"id\":\"" + cid +
                                 "\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,"
                                 "\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                                 "data: [DONE]\n\n";
                             sink.write(tail.c_str(), tail.size());
                             sink.done();
                             return true;
                         });
                     return;
                 }

                 std::vector<int> gen = engine.complete(prompt_ids, max_tokens);
                 std::string text;
                 if (!engine.last_error().empty()) {
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(engine.last_error()) + "\"}}",
                                     "application/json");
                     return;
                 }
                 if (!decode_ids(gen, text, err)) {
                     res.status = 500;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }

                 const auto parsed = sparkinfer_server::parse_assistant_output(text, enable_thinking);

                 std::ostringstream body;
                 body << "{\"id\":\"" << cid << "\",\"object\":\"chat.completion\",\"created\":" << created
                      << ",\"model\":\"" << g_model_name << "\",\"choices\":[{\"index\":0,\"message\":{"
                      << "\"role\":\"assistant\"";
                 if (!parsed.reasoning_content.empty())
                     body << ",\"reasoning_content\":\"" << json_escape(parsed.reasoning_content) << "\"";
                 body << ",\"content\":\"" << json_escape(parsed.content) << "\""
                      << "},\"finish_reason\":\"stop\"}],"
                      << usage_json((int)prompt_ids.size(), (int)gen.size()) << "}";
                 res.set_content(body.str(), "application/json");
             });

    fprintf(stderr,
            "[sparkinfer-server] OpenAI-compatible API on http://%s:%d\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  GET  /v1/info\n"
            "  POST /v1/tokenize\n"
            "  POST /v1/chat/completions\n",
            host.c_str(), port);

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "[sparkinfer-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
