#include "chat_tokenizer.hpp"
#include "model_engine.hpp"
#include "sparkinfer/kernels/deterministic.h"

#include <nlohmann/json.hpp>

// Do not define CPPHTTPLIB_OPENSSL_SUPPORT — even `= 0` enables OpenSSL in httplib.
#include "../third_party/httplib.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Per-request output cap. Independent of context length (checked separately against the
// live engine max_seq below) -- this just bounds how long a single request may run.
int max_output_tokens() {
    static int v = []{
        const char* e = getenv("SPARKINFER_MAX_OUTPUT_TOKENS");
        int n = e ? atoi(e) : 4096;
        return n > 0 ? n : 4096;
    }();
    return v;
}

int sse_keepalive_seconds() {
    static int v = [] {
        const char* e = getenv("SPARKINFER_SSE_KEEPALIVE_SECONDS");
        if (!e || !*e) return 15;
        return std::max(0, std::min(atoi(e), 300));
    }();
    return v;
}

bool always_stream_usage() {
    static bool v = [] {
        const char* e = getenv("SPARKINFER_ALWAYS_STREAM_USAGE");
        return !e || e[0] != '0';
    }();
    return v;
}

bool openrouter_provider_mode() {
    static bool v = [] {
        const char* e = getenv("SPARKINFER_OPENROUTER_PROVIDER");
        return e && e[0] == '1';
    }();
    return v;
}

std::string env_string(const char* name, const std::string& fallback = {}) {
    const char* value = getenv(name);
    return value && *value ? value : fallback;
}

// Positive integer deployment metadata used by OpenRouter's provider schema. Invalid or absent
// values are omitted rather than publishing a made-up limit. `created` is the one exception: the
// schema requires it, so an unconfigured deployment reports the Unix epoch and the OpenRouter
// launcher below requires operators to provide the real model timestamp.
long long env_positive_integer(const char* name, long long fallback = 0) {
    const char* value = getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    const long long parsed = strtoll(value, &end, 10);
    return errno == 0 && end && *end == '\0' && parsed > 0 ? parsed : fallback;
}

std::string g_api_key;
// Advertised model id: every completion response, /v1/models, /v1/info and /metrics carry it,
// and clients route, log and bill on it. This default is a Qwen3.6 id for historical reasons, so
// it is only correct when a Qwen3.6 checkpoint is what actually got loaded -- main() corrects it
// after load() for the architectures the engine can identify, unless --model-name was given.
std::string g_model_name = "qwen3.6-35b-a3b";
sparkinfer_server::ChatTokenizer g_tokenizer;
const auto g_start_time = std::chrono::steady_clock::now();

// Request/error metrics (GET /metrics). Counters only -- no per-request content is retained.
std::atomic<uint64_t> g_requests_total{0};
std::atomic<uint64_t> g_requests_streaming{0};
std::atomic<uint64_t> g_requests_ok{0};
std::atomic<uint64_t> g_requests_client_error{0};   // 4xx
std::atomic<uint64_t> g_requests_overloaded{0};     // 429
std::atomic<uint64_t> g_requests_alloc_failed{0};   // 503 -- real device OOM, not capacity (#779)
std::atomic<uint64_t> g_requests_timeout{0};
std::atomic<uint64_t> g_requests_cancelled{0};
std::atomic<uint64_t> g_requests_server_error{0};   // 5xx
// 502 -- the model produced tool-call output that failed schema/markup validation for a reason
// other than truncation. Distinct from server_error so monitoring doesn't conflate model-output
// quality variance with a real infrastructure fault -- the exact conflation #779 already
// documented once for overloaded (429) vs alloc_failed (503).
std::atomic<uint64_t> g_requests_invalid_tool_output{0};
// 502 -- same rationale as g_requests_invalid_tool_output above, for response_format: the model
// produced output that failed JSON/schema validation on both the original and the corrective
// retry attempt. Not a server_error -- this is a model-output-quality signal, not an infra fault.
std::atomic<uint64_t> g_requests_invalid_json_output{0};
std::atomic<uint64_t> g_prompt_tokens_total{0};
std::atomic<uint64_t> g_completion_tokens_total{0};

std::atomic<bool> g_shutdown_requested{false};
void on_shutdown_signal(int) { g_shutdown_requested = true; }

std::string repo_root() {
    const char* env = getenv("SPARKINFER_ROOT");
    if (env && *env) return env;
    return ".";
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

std::string random_id(const char* prefix = "chatcmpl-") {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream ss;
    ss << prefix << std::hex << dist(rng);
    return ss.str();
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

bool encode_messages(const std::string& body, std::vector<int>& ids, bool enable_thinking,
                     std::string& err, sparkinfer_server::ChatRequest* request = nullptr) {
    return g_tokenizer.encode_chat_request(body, ids, enable_thinking, err, request);
}

// The image placeholder's token id, resolved from the loaded tokenizer rather than hardcoded --
// it differs between checkpoints and a wrong constant would splice embeddings over ordinary text.
// Negative when this tokenizer has no such token, which is how a text-only model reports that it
// cannot take images. Resolved once: it cannot change while a model is loaded.
int image_pad_token_id() {
    static const int id = [] {
        const std::vector<int> ids = g_tokenizer.encode_raw("<|image_pad|>");
        return ids.size() == 1 ? ids[0] : -1;
    }();
    return id;
}

// The video placeholder's token id, same contract as image_pad_token_id above.
int video_pad_token_id() {
    static const int id = [] {
        const std::vector<int> ids = g_tokenizer.encode_raw("<|video_pad|>");
        return ids.size() == 1 ? ids[0] : -1;
    }();
    return id;
}

int vision_start_token_id() {
    static const int id = [] {
        const std::vector<int> ids = g_tokenizer.encode_raw("<|vision_start|>");
        return ids.size() == 1 ? ids[0] : -1;
    }();
    return id;
}

int vision_end_token_id() {
    static const int id = [] {
        const std::vector<int> ids = g_tokenizer.encode_raw("<|vision_end|>");
        return ids.size() == 1 ? ids[0] : -1;
    }();
    return id;
}

// Collects every video_url in the request, in message order, alongside collect_image_urls.
std::vector<std::string> collect_video_urls(const sparkinfer_server::ChatRequest& r) {
    std::vector<std::string> urls;
    for (const auto& m : r.messages) urls.insert(urls.end(), m.videos.begin(), m.videos.end());
    return urls;
}

// Collects every image_url in the request, in message order, matching the order the rendered
// prompt's placeholders appear in.
std::vector<std::string> collect_image_urls(const sparkinfer_server::ChatRequest& r) {
    std::vector<std::string> urls;
    for (const auto& m : r.messages) urls.insert(urls.end(), m.images.begin(), m.images.end());
    return urls;
}

// RequestControls / parse_request_controls / should_reject_dflash_temperature now live in
// chat_tools.hpp/.cpp (moved so `stop`/`temperature`/`seed` validation has a unit-test seam via
// chat_tools_test, same as ChatRequest/parse_chat_request_json already had).

// One-shot scan over a complete decoded string: is there a stop match anywhere, and where.
// Unlike StopSequenceFilter, this needs no holdback machinery -- it's only ever called once
// generation has already stopped and the full text is available in hand.
bool find_stop_match(const std::string& text, const std::vector<std::string>& stops, size_t& pos) {
    bool found = false;
    for (const auto& s : stops) {
        const size_t p = text.find(s);
        if (p != std::string::npos && (!found || p < pos)) {
            pos = p;
            found = true;
        }
    }
    return found;
}

nlohmann::json stream_chunk_base(const std::string& cid, long long created,
                                 const char* object = "chat.completion.chunk") {
    return {{"id", cid}, {"object", object}, {"created", created},
            {"model", g_model_name}};
}

// httplib::DataSink has no thread-safety for concurrent write() calls (confirmed against
// third_party/httplib.h -- nothing there documents or provides one). n>1 streaming fans out n
// branches onto separate std::threads (see run_streaming_branch below), all writing SSE chunks
// into the SAME one DataSink for this one HTTP response -- every write must go through this one
// mutex. Reads (sink.is_writable(), polled inside on_tok to detect a disconnected client) need no
// lock; only concurrent writes are unsafe.
struct GuardedSink {
    httplib::DataSink& sink;
    std::mutex& mu;
};

// SSE comments are ignored by OpenAI clients and prevent proxy idle timeouts during long prefill.
class SseHeartbeat {
public:
    explicit SseHeartbeat(GuardedSink& sink) : sink_(sink), interval_(sse_keepalive_seconds()) {
        if (interval_ > 0) worker_ = std::thread([this] { run(); });
    }
    ~SseHeartbeat() { stop(); }
    SseHeartbeat(const SseHeartbeat&) = delete;
    SseHeartbeat& operator=(const SseHeartbeat&) = delete;

    void stop() {
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            stopped_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    void run() {
        std::unique_lock<std::mutex> state_lock(state_mu_);
        while (!cv_.wait_for(state_lock, std::chrono::seconds(interval_), [this] { return stopped_; })) {
            state_lock.unlock();
            static constexpr char event[] = ": keep-alive\n\n";
            {
                std::lock_guard<std::mutex> write_lock(sink_.mu);
                if (!sink_.sink.is_writable() || !sink_.sink.write(event, sizeof(event) - 1)) {
                    state_lock.lock();
                    stopped_ = true;
                    return;
                }
            }
            state_lock.lock();
        }
    }

    GuardedSink& sink_;
    int interval_;
    std::mutex state_mu_;
    std::condition_variable cv_;
    bool stopped_ = false;
    std::thread worker_;
};

bool write_sse_json(GuardedSink& gs, const nlohmann::json& value) {
    const std::string event = "data: " + value.dump() + "\n\n";
    std::lock_guard<std::mutex> lock(gs.mu);
    return gs.sink.write(event.c_str(), event.size());
}

// choice_index identifies which of the n completions this chunk belongs to (OpenAI's own
// per-choice "index" field) -- distinct from write_stream_tool_call's own `index` param below,
// which is the tool call's position WITHIN one choice's tool_calls array.
bool write_stream_role(GuardedSink& gs, const std::string& cid, long long created, int choice_index) {
    auto chunk = stream_chunk_base(cid, created);
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
                                                {"delta", {{"role", "assistant"},
                                                           {"content", nullptr}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", nullptr}}});
    return write_sse_json(gs, chunk);
}

// Emits a delta carrying ONLY logprobs (empty content). write_stream_delta drops empty pieces, so
// without this any entries still pending when generation ends -- the last tokens produced no text
// of their own, e.g. a stop-sequence holdback or a trailing partial UTF-8 sequence -- would be
// silently dropped, and the concatenated stream would carry fewer entries than completion_tokens.
// A content delta with an empty string is well-formed and OpenAI clients tolerate it.
bool write_stream_logprobs_only(GuardedSink& gs, const std::string& cid, long long created,
                                int choice_index, const nlohmann::json& logprobs,
                                const char* field = "content",
                                const char* object = "chat.completion.chunk") {
    auto chunk = stream_chunk_base(cid, created, object);
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
                                                {"delta", {{field, ""}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", logprobs}}});
    return write_sse_json(gs, chunk);
}

bool write_stream_delta(GuardedSink& gs, const std::string& cid, long long created, const std::string& field,
                        const std::string& piece, int choice_index, const nlohmann::json& logprobs = nullptr,
                        const char* object = "chat.completion.chunk") {
    if (piece.empty()) return true;
    auto chunk = stream_chunk_base(cid, created, object);
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
                                                {"delta", {{field, piece}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", logprobs}}});
    return write_sse_json(gs, chunk);
}

bool write_stream_reasoning_delta(GuardedSink& gs, const std::string& cid, long long created,
                                  const std::string& piece, int choice_index,
                                  const nlohmann::json& logprobs = nullptr) {
    if (piece.empty()) return true;
    auto chunk = stream_chunk_base(cid, created);
    // OpenRouter standardizes on `reasoning`; `reasoning_content` remains as the widely-used
    // OpenAI-compatible alias and preserves SparkInfer's existing wire contract.
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
        {"delta", {{"reasoning", piece}, {"reasoning_content", piece}}},
        {"finish_reason", nullptr}, {"logprobs", logprobs}}});
    return write_sse_json(gs, chunk);
}

bool write_stream_tool_call(GuardedSink& gs, const std::string& cid, long long created, int choice_index,
                            size_t index, const sparkinfer_server::ToolCall& call) {
    auto chunk = stream_chunk_base(cid, created);
    nlohmann::json delta_call = {{"index", index}, {"id", call.id}, {"type", "function"},
                                {"function", {{"name", call.name},
                                              {"arguments", call.arguments}}}};
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
                                                {"delta", {{"tool_calls",
                                                            nlohmann::json::array({delta_call})}}},
                                                {"finish_reason", nullptr},
                                                {"logprobs", nullptr}}});
    return write_sse_json(gs, chunk);
}

bool write_stream_finish(GuardedSink& gs, const std::string& cid, long long created,
                         int choice_index, const std::string& reason,
                         const char* object = "chat.completion.chunk") {
    auto chunk = stream_chunk_base(cid, created, object);
    chunk["choices"] = nlohmann::json::array({{{"index", choice_index},
                                                {"delta", nlohmann::json::object()},
                                                {"finish_reason", reason},
                                                {"logprobs", nullptr}}});
    return write_sse_json(gs, chunk);
}

// Usage chunks have no per-choice shape in OpenAI's own convention (empty "choices", one
// aggregate "usage" object) -- no choice_index needed. For n>1 this is emitted ONCE, after every
// branch has joined, with prompt_tokens reported once (not xn) and completion_tokens summed
// across choices -- see the n-fanout aggregation code below.
bool write_stream_usage(GuardedSink& gs, const std::string& cid, long long created,
                        int prompt_tokens, int completion_tokens, double ttft_ms,
                        double generation_ms, double decode_tps,
                        const char* object = "chat.completion.chunk") {
    auto chunk = stream_chunk_base(cid, created, object);
    chunk["choices"] = nlohmann::json::array();
    nlohmann::json usage = {{"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}};
    if (ttft_ms >= 0.0) usage["ttft_ms"] = ttft_ms;
    if (generation_ms >= 0.0) usage["generation_ms"] = generation_ms;
    if (decode_tps >= 0.0) usage["decode_tps"] = decode_tps;
    chunk["usage"] = std::move(usage);
    return write_sse_json(gs, chunk);
}

nlohmann::json token_logprob_entry_json(int token_id, float logprob) {
    const auto piece = g_tokenizer.id_to_raw_piece(token_id);
    nlohmann::json bytes = nlohmann::json::array();
    for (uint8_t b : piece.bytes) bytes.push_back((int)b);
    return {{"token", piece.display}, {"token_id", token_id}, {"logprob", logprob}, {"bytes", bytes}};
}

// Builds the OpenAI-shaped logprobs.content array from a flat, generation-ordered list of
// per-token entries. Each entry's own top_alternatives list is truncated to top_logprobs_n
// (the request's requested top_logprobs value) regardless of how many were captured on-device.
// OpenAI omits the stop token from logprobs.content: the assistant text a client re-tokenises
// never contains <|im_end|>, so including its entry makes logprobs.content exactly one longer
// than the text and breaks anyone zipping the two together (reported as 41 entries vs 40 tokens).
// Dropped at the source rather than trimmed afterwards, because the streaming path flushes
// entries incrementally and has no "this was the last one" moment to trim at.
nlohmann::json build_logprobs_content_json(const std::vector<sparkinfer_server::TokenLogprob>& entries,
                                           int top_logprobs_n) {
    nlohmann::json content = nlohmann::json::array();
    for (const auto& e : entries) {
        nlohmann::json item = token_logprob_entry_json(e.token_id, e.logprob);
        nlohmann::json alts = nlohmann::json::array();
        const int n = std::min((int)e.top_alternatives.size(), top_logprobs_n);
        for (int i = 0; i < n; i++)
            alts.push_back(token_logprob_entry_json(e.top_alternatives[i].first, e.top_alternatives[i].second));
        item["top_logprobs"] = alts;
        content.push_back(item);
    }
    return content;
}

// Legacy /v1/completions logprobs shape: parallel arrays, not chat completions' array-of-objects
// (build_logprobs_content_json above). text_offset is a running byte offset into the OWN choice's
// text, starting at text_offset_base (0 normally; past the echoed prompt's length when echo=true
// -- this runtime doesn't compute logprobs for echoed prompt tokens, only the generated
// continuation, see the parse_legacy_completion_request doc comment).
nlohmann::json build_legacy_logprobs_json(const std::vector<sparkinfer_server::TokenLogprob>& entries,
                                          int top_logprobs_n, size_t text_offset_base) {
    nlohmann::json tokens = nlohmann::json::array(), token_logprobs = nlohmann::json::array(),
                   top_logprobs = nlohmann::json::array(), text_offset = nlohmann::json::array();
    size_t offset = text_offset_base;
    for (const auto& e : entries) {
        const auto piece = g_tokenizer.id_to_raw_piece(e.token_id);
        tokens.push_back(piece.display);
        token_logprobs.push_back(e.logprob);
        text_offset.push_back(offset);
        offset += piece.display.size();
        nlohmann::json alts = nlohmann::json::object();
        const int k = std::min((int)e.top_alternatives.size(), top_logprobs_n);
        for (int i = 0; i < k; i++) {
            const auto ap = g_tokenizer.id_to_raw_piece(e.top_alternatives[i].first);
            alts[ap.display] = e.top_alternatives[i].second;
        }
        top_logprobs.push_back(alts);
    }
    return {{"tokens", tokens}, {"token_logprobs", token_logprobs},
            {"top_logprobs", top_logprobs}, {"text_offset", text_offset}};
}

// Only ever called once, single-threaded, after every branch has joined -- still routed through
// the mutex for type consistency with every other writer (uncontended lock/unlock is negligible).
bool write_stream_done(GuardedSink& gs) {
    static const std::string done = "data: [DONE]\n\n";
    std::lock_guard<std::mutex> lock(gs.mu);
    return gs.sink.write(done.c_str(), done.size());
}

bool decode_ids(const std::vector<int>& ids, std::string& text, std::string& err) {
    text = g_tokenizer.decode(ids);
    if (text.empty() && !ids.empty()) {
        err = "detokenize returned empty text";
        return false;
    }
    return true;
}

// Builds the attempt-2 prompt for a response_format validation failure: original conversation +
// the model's own (invalid) attempt-1 output as an assistant turn, followed by a corrective user
// turn describing what was wrong. This runtime's decode is fully deterministic greedy argmax
// with no RNG anywhere -- resubmitting attempt 1's identical prompt_ids would just reproduce the
// exact same invalid output. The retry only has a chance of succeeding with a materially
// different prompt.
sparkinfer_server::ChatRequest build_retry_request(const sparkinfer_server::ChatRequest& original,
                                                    const std::string& failed_content,
                                                    const std::string& validation_error) {
    sparkinfer_server::ChatRequest retry = original;
    sparkinfer_server::ChatMessage assistant_turn;
    assistant_turn.role = "assistant";
    assistant_turn.content = failed_content.empty() ? "(no output)" : failed_content;
    retry.messages.push_back(std::move(assistant_turn));
    sparkinfer_server::ChatMessage correction;
    correction.role = "user";
    correction.content = "Your previous reply did not satisfy the required response_format: " +
                         validation_error +
                         "\nRespond again with ONLY the corrected JSON, matching the required "
                         "format exactly.";
    retry.messages.push_back(std::move(correction));
    return retry;
}

std::vector<int> load_prefix_token_ids() {
    std::vector<int> out;
    if (const char* csv = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_IDS")) {
        const char* p = csv;
        while (*p) {
            char* end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            out.push_back((int)v);
            p = end;
            while (*p == ',' || *p == ' ') p++;
        }
        return out;
    }
    const char* path = getenv("SPARKINFER_SERVER_PREFIX_TOKEN_FILE");
    if (!path || !*path) return out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "[sparkinfer-server] WARN: cannot open prefix token file %s\n", path);
        return out;
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (size_t i = 0; i < s.size();) {
        i = s.find_first_of("0123456789", i);
        if (i == std::string::npos) break;
        out.push_back(atoi(s.c_str() + i));
        i = s.find_first_not_of("0123456789", i);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string tokenizer_json;
    bool model_name_explicit = false;   // --model-name given: never second-guess the operator
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
        else if (need("--model-name")) { g_model_name = argv[++i]; model_name_explicit = true; }
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

    // Name what was actually loaded. Without this the server reports "qwen3.6-35b-a3b" whatever
    // it is serving -- a client asking Qwen3.8 a question is told it spoke to Qwen3.6, and every
    // log line and usage record inherits that.
    //
    // Deliberately narrow: only architectures the engine can positively identify are renamed, and
    // Qwen3.6 keeps the historical id byte-for-byte. Deriving the name from the checkpoint path
    // instead would be more general and would also change what existing Qwen3.6 deployments
    // advertise, which is what OpenRouter routes on.
    if (!model_name_explicit) {
        if (engine.is_qwen38()) g_model_name = "qwen3.8-27b";
        else if (engine.is_museglimmer()) g_model_name = "muse-glimmer-30b";
        if (g_model_name != "qwen3.6-35b-a3b")
            fprintf(stderr, "[sparkinfer-server] advertising model id: %s "
                            "(override with --model-name)\n", g_model_name.c_str());
    }
    g_tokenizer.set_museglimmer(engine.is_museglimmer());
    g_tokenizer.set_qwen38(engine.is_qwen38());

    const std::vector<int> prefix_ids = load_prefix_token_ids();
    if (!prefix_ids.empty()) {
        engine.set_prefix_tokens(prefix_ids);
        fprintf(stderr, "[sparkinfer-server] prefix cache: %zu tokens (batched prefill per request)\n",
                prefix_ids.size());
    }

    httplib::Server svr;

    // Reports the DEVICE, not just the process. A lost CUDA context leaves the HTTP thread
    // perfectly able to answer -- which is exactly the failure that hid a dead server behind a
    // green health check while every real request 503'd. 503 here so an orchestrator replaces
    // the process; it cannot recover without a restart.
    svr.Get("/health", [&engine](const httplib::Request&, httplib::Response& res) {
        if (!engine.device_healthy()) {
            res.status = 503;
            res.set_content("{\"status\":\"unhealthy\",\"reason\":\"cuda context lost "
                            "(unrecoverable device error) -- restart required\"}",
                            "application/json");
            return;
        }
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&engine](const httplib::Request&, httplib::Response& res) {
        using json = nlohmann::json;
        const int context = engine.max_seq();
        json supported = {
            {"temperature", {{"type", "range"}, {"min", 0}, {"max", 2}}},
            {"top_p", {{"type", "range"}, {"min", 0}, {"max", 1}}},
            {"max_tokens", {{"type", "integer"}, {"min", 1}, {"max", max_output_tokens()}, {"unit", "token"}}},
            {"stop", {{"type", "array"}, {"max_items", 4}}},
            {"seed", {{"type", "unknown"}}},
            {"presence_penalty", {{"type", "range"}, {"min", -2}, {"max", 2}}},
            {"frequency_penalty", {{"type", "range"}, {"min", -2}, {"max", 2}}},
            {"logprobs", {{"type", "boolean"}}},
            {"top_logprobs", {{"type", "integer"}, {"min", 0}, {"max", 20}}},
            {"logit_bias", {{"type", "unknown"}}},
            {"n", {{"type", "integer"}, {"min", 1}, {"max", 8}}}
        };
        if (!engine.is_museglimmer()) {
            supported["tools"] = {{"type", "boolean"}};
            supported["structured_outputs"] = {{"type", "boolean"}};
        }
        if (engine.is_qwen38()) supported["reasoning"] = {{"type", "boolean"}};
        json input = {{"type", "text"},
            {"supported_inputs", {{"max_context_length", {{"value", context}, {"unit", "token"}}}}}};
        json output = {{"type", "text"}, {"streaming", true},
            {"max_length", {{"value", max_output_tokens()}, {"unit", "token"}}},
            {"supported_parameters", std::move(supported)}};
        const std::string prompt_price = env_string("SPARKINFER_PROMPT_PRICE_USD");
        const std::string completion_price = env_string("SPARKINFER_COMPLETION_PRICE_USD");
        if (!prompt_price.empty())
            input["pricing"] = json::array({{{"type", "prompt"}, {"unit", "token"},
                                               {"cost_usd", prompt_price}}});
        if (!completion_price.empty())
            output["pricing"] = json::array({{{"type", "completion"}, {"unit", "token"},
                                                {"cost_usd", completion_price}}});

        const long long prompt_tpm = env_positive_integer("SPARKINFER_PROMPT_TOKENS_PER_MINUTE");
        if (prompt_tpm > 0)
            input["capacity"] = json::array({{{"type", "prompt"}, {"unit", "token"},
                                                {"per", "minute"}, {"value", prompt_tpm}}});
        const long long completion_tpm =
            env_positive_integer("SPARKINFER_COMPLETION_TOKENS_PER_MINUTE");
        if (completion_tpm > 0)
            output["capacity"] = json::array({{{"type", "completion"}, {"unit", "token"},
                                                 {"per", "minute"}, {"value", completion_tpm}}});
        // Advertise image input when a vision tower actually loaded. Claiming text-only while
        // accepting images makes a router refuse to send work this server can do; claiming images
        // on a text-only checkpoint is worse, so it keys off has_vision() rather than a build flag.
        //
        // CAVEAT: OpenRouter's v2.4 schema is closed (additionalProperties: false) and is not
        // vendored here, so this entry's exact shape is unverified against it. It only appears for
        // a vision-capable checkpoint, and the registered provider deployment is Qwen3.6 (GGUF, no
        // tower), so a listing in use today cannot be affected -- but verify against the model
        // monitor before registering a vision model.
        json input_modalities = json::array({std::move(input)});
        if (engine.has_vision())
            input_modalities.push_back({{"type", "image"}});

        json model = {
            {"schema_version", "2.4"}, {"id", g_model_name}, {"name", g_model_name},
            {"created", env_positive_integer("SPARKINFER_MODEL_CREATED")},
            {"hugging_face_id", env_string("SPARKINFER_HF_MODEL_ID")},
            {"quantization", nullptr},
            {"input_modalities", std::move(input_modalities)},
            {"output_modalities", json::array({std::move(output)})},
            {"is_ready", engine.loaded() && engine.device_healthy() && !g_shutdown_requested.load()}
        };
        // OpenRouter's v2.4 schema is closed (`additionalProperties: false`), while OpenAI clients
        // conventionally expect these legacy list-model fields. Provider mode must omit them;
        // ordinary deployments retain the old response shape for compatibility.
        if (!openrouter_provider_mode()) {
            model["object"] = "model";
            model["owned_by"] = "sparkinfer";
            model["context_length"] = context;
        }
        const long long rpm = env_positive_integer("SPARKINFER_REQUESTS_PER_MINUTE");
        if (rpm > 0)
            model["capacity"] = json::array({{{"type", "request"}, {"unit", "request"},
                                                {"per", "minute"}, {"value", rpm}}});
        const int concurrency = engine.max_queue_depth();
        if (concurrency > 0) {
            if (!model.contains("capacity")) model["capacity"] = json::array();
            model["capacity"].push_back({{"type", "concurrency"}, {"unit", "request"},
                                           {"value", concurrency}});
        }
        const std::string quantization = env_string("SPARKINFER_QUANTIZATION");
        if (!quantization.empty()) model["quantization"] = quantization;
        res.set_content(json({{"object", "list"}, {"data", json::array({std::move(model)})}}).dump(),
                        "application/json");
    });

    svr.Get("/v1/info", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"model\":\"" << g_model_name << "\",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << "}";
        res.set_content(body.str(), "application/json");
    });

    // Live occupancy -- lets an orchestrator (or a human) see whether this worker has room
    // before routing a request to it, and is what a fleet-level capacity/load-balancing layer
    // would poll. Single-process only: this reports this server's own queue, not fleet-wide
    // capacity across other nodes.
    svr.Get("/v1/capacity", [&engine](const httplib::Request&, httplib::Response& res) {
        std::ostringstream body;
        const int cap = engine.max_queue_depth();
        body << "{\"active_requests\":" << engine.active_requests()
             << ",\"free_kv_blocks\":" << engine.free_kv_blocks()
             << ",\"max_queue_depth\":" << cap
             << ",\"accepting_requests\":" << (g_shutdown_requested.load() ? "false" : "true") << "}";
        res.set_content(body.str(), "application/json");
    });

    svr.Get("/metrics", [&engine](const httplib::Request&, httplib::Response& res) {
        const double uptime_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - g_start_time).count();
        std::ostringstream body;
        body << "# HELP sparkinfer_uptime_seconds Process uptime\n"
                "# TYPE sparkinfer_uptime_seconds gauge\n"
             << "sparkinfer_uptime_seconds " << uptime_s << "\n"
                "# HELP sparkinfer_requests_total Chat completion requests received\n"
                "# TYPE sparkinfer_requests_total counter\n"
             << "sparkinfer_requests_total " << g_requests_total.load() << "\n"
             << "sparkinfer_requests_streaming_total " << g_requests_streaming.load() << "\n"
                "# HELP sparkinfer_requests_by_outcome_total Requests by terminal outcome\n"
                "# TYPE sparkinfer_requests_by_outcome_total counter\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"ok\"} " << g_requests_ok.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"client_error\"} "
             << g_requests_client_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"overloaded\"} "
             << g_requests_overloaded.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"alloc_failed\"} "
             << g_requests_alloc_failed.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"timeout\"} "
             << g_requests_timeout.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"cancelled\"} "
             << g_requests_cancelled.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"server_error\"} "
             << g_requests_server_error.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"invalid_tool_output\"} "
             << g_requests_invalid_tool_output.load() << "\n"
             << "sparkinfer_requests_by_outcome_total{outcome=\"invalid_json_output\"} "
             << g_requests_invalid_json_output.load() << "\n"
                "# HELP sparkinfer_tokens_total Tokens processed\n"
                "# TYPE sparkinfer_tokens_total counter\n"
             << "sparkinfer_tokens_total{kind=\"prompt\"} " << g_prompt_tokens_total.load() << "\n"
             << "sparkinfer_tokens_total{kind=\"completion\"} " << g_completion_tokens_total.load() << "\n"
                "# HELP sparkinfer_active_requests In-flight requests\n"
                "# TYPE sparkinfer_active_requests gauge\n"
             << "sparkinfer_active_requests " << engine.active_requests() << "\n"
                "# HELP sparkinfer_free_kv_blocks Free KV cache blocks\n"
                "# TYPE sparkinfer_free_kv_blocks gauge\n"
             << "sparkinfer_free_kv_blocks " << engine.free_kv_blocks() << "\n";
        // Only emitted when the LMCache bridge is actually enabled (docs/lmcache_bridge_protocol.md)
        // -- omitting the metric entirely when disabled, rather than always emitting zeros, makes
        // "is this feature even on" visible from /metrics itself, not just server startup logs.
        const auto lmc = engine.lmcache_stats();
        if (lmc.enabled) {
            body << "# HELP sparkinfer_lmcache_lookup_hits_total External KV cache tier lookups "
                    "that restored a matched prefix\n"
                    "# TYPE sparkinfer_lmcache_lookup_hits_total counter\n"
                 << "sparkinfer_lmcache_lookup_hits_total " << lmc.lookup_hits << "\n"
                    "# HELP sparkinfer_lmcache_lookup_misses_total External KV cache tier lookups "
                    "that found nothing usable (includes a genuine miss, a timed-out sidecar, and "
                    "the sidecar being unreachable -- see docs/lmcache_bridge_protocol.md's "
                    "degradation invariant, all three fall back to the same recompute path)\n"
                    "# TYPE sparkinfer_lmcache_lookup_misses_total counter\n"
                 << "sparkinfer_lmcache_lookup_misses_total " << lmc.lookup_misses << "\n";
        }
        res.set_content(body.str(), "text/plain; version=0.0.4");
    });

    svr.Post("/v1/tokenize", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, engine.is_qwen38());
        std::vector<int> ids;
        std::string err;
        // Images would render a bare <|image_pad|> into these ids: the count each image really
        // needs depends on its resized grid, which only the preprocessor knows. Reporting the
        // unexpanded number would understate the prompt by hundreds or thousands of tokens, so
        // this says so rather than returning a wrong count.
        {
            sparkinfer_server::ChatRequest probe;
            std::string perr;
            if (encode_messages(req.body, ids, enable_thinking, perr, &probe) &&
                (!collect_image_urls(probe).empty() || !collect_video_urls(probe).empty())) {
                g_requests_client_error++;
                res.status = 400;
                res.set_content("{\"error\":{\"message\":\"/v1/tokenize does not support image or "
                                "video content parts; token counts for them depend on the resized "
                                "grid, and for video on how many frames were sampled\"}}",
                                "application/json");
                return;
            }
            ids.clear();
        }
        if (!encode_messages(req.body, ids, enable_thinking, err)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}", "application/json");
            return;
        }
        std::ostringstream body;
        body << "{\"tokens\":" << ids.size() << ",\"max_context\":" << engine.max_seq()
             << ",\"max_output_tokens\":" << max_output_tokens() << ",\"model\":\"" << g_model_name << "\"}";
        res.set_content(body.str(), "application/json");
    });

    // ---- POST /v1/score : teacher-forced scoring -------------------------------------------
    //
    // Given a prompt and a SUPPLIED continuation, return the per-token logprob of each
    // continuation token under the model, without generating anything. The audit primitive an
    // external verifier needs: it can score any text (another server's answer, a perturbed
    // reference, mirrored organic traffic) rather than being limited to comparing its own
    // sampled output, so there is no finite answer set to memorise.
    //
    // Why a dedicated endpoint rather than OpenAI's legacy `/v1/completions` with
    // `echo: true, logprobs: 1, max_tokens: 0`: echo's contract is to report logprobs for the
    // PROMPT tokens, and this runtime cannot produce those -- batched prefill runs the LM head
    // only at the final prompt position, so per-prompt-token logits do not exist without a
    // second, batched LM-head pass over the whole prompt. Rather than half-implement echo and
    // return a shape that silently omits most of what the field promises, /v1/score states
    // exactly what it does. (`/v1/completions` keeps its existing echo behaviour: text only.)
    //
    // Request:
    //   {"model": "...",
    //    "messages": [...],            // chat-templated, exactly as /v1/chat/completions
    //     OR "prompt": "raw text",     // no chat template applied
    //    "completion": "text to score",
    //     OR "completion_token_ids": [1,2,3],   // bypasses tokenisation entirely
    //    "top_logprobs": 0-20,         // optional, per-position alternatives
    //    "enable_thinking": bool}      // optional, same meaning as chat completions
    //
    // Response: token/token_id/logprob triples in generation order, plus sum_logprob and the
    // same `usage` block (including the ttft_ms/generation_ms/decode_tps timing fields) every
    // other endpoint reports.
    svr.Post("/v1/score", [&engine](const httplib::Request& req, httplib::Response& res) {
        if (!auth_ok(req)) {
            res.status = 401;
            res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
            return;
        }
        auto fail = [&res](int status, const std::string& msg) {
            res.status = status;
            res.set_content("{\"error\":{\"message\":\"" + json_escape(msg) + "\"}}",
                            "application/json");
        };

        sparkinfer_server::ScoreRequest sreq;
        std::string perr;
        if (!sparkinfer_server::parse_score_request(req.body, sreq, perr, engine.vocab())) {
            fail(400, perr);
            return;
        }

        // Prompt side. The messages parse itself belongs to the chat tokenizer, which already
        // owns it -- parse_score_request only decided WHICH form was supplied.
        std::vector<int> prompt_ids;
        if (sreq.use_messages) {
            const bool enable_thinking =
                sparkinfer_server::parse_enable_thinking(req.body, engine.is_qwen38());
            std::string err;
            // Scoring an image prompt would need the tower run and the placeholders expanded.
            // Silently dropping the image and returning logprobs for the text alone is the worst
            // option for an endpoint that exists to be compared against another copy of itself.
            {
                sparkinfer_server::ChatRequest probe;
                std::string perr;
                if (encode_messages(req.body, prompt_ids, enable_thinking, perr, &probe) &&
                    (!collect_image_urls(probe).empty() || !collect_video_urls(probe).empty())) {
                    fail(400, "/v1/score does not support image or video content parts");
                    return;
                }
                prompt_ids.clear();
            }
            if (!encode_messages(req.body, prompt_ids, enable_thinking, err)) {
                fail(400, err);
                return;
            }
        } else {
            prompt_ids = g_tokenizer.encode_raw(sreq.prompt);
        }
        if (prompt_ids.empty()) { fail(400, "prompt encoded to zero tokens"); return; }

        std::vector<int> forced = sreq.completion_is_ids ? sreq.completion_token_ids
                                                         : g_tokenizer.encode_raw(sreq.completion);
        if (forced.empty()) { fail(400, "completion encoded to zero tokens"); return; }
        const int top_logprobs = sreq.top_logprobs;

        // Bounded by the same per-request output cap generation uses, and by the live context.
        if ((int)forced.size() > max_output_tokens()) {
            fail(400, "completion has " + std::to_string(forced.size()) +
                          " tokens, exceeds max_output_tokens=" + std::to_string(max_output_tokens()));
            return;
        }
        if ((int)prompt_ids.size() + (int)forced.size() > engine.max_seq()) {
            fail(400, "prompt + completion = " +
                          std::to_string(prompt_ids.size() + forced.size()) +
                          " tokens exceeds server ctx=" + std::to_string(engine.max_seq()));
            return;
        }

        std::vector<sparkinfer_server::TokenLogprob> entries;
        entries.reserve(forced.size());
        auto outcome = engine.complete_streaming(
            prompt_ids, (int)forced.size(), [](int) { return true; },
            /*temperature=*/0.f, /*seed=*/0, /*top_k=*/0, /*top_p=*/1.0f,
            /*presence_penalty=*/0.f, /*frequency_penalty=*/0.f, /*logit_bias=*/{},
            /*logprobs=*/true, top_logprobs,
            [&entries](const sparkinfer_server::TokenLogprob& tl) { entries.push_back(tl); },
            forced);

        if (!outcome.error.empty()) {
            g_requests_total++;
            if (outcome.overloaded) {
                g_requests_overloaded++;
                fail(429, outcome.error);
            } else if (outcome.alloc_failed) {
                g_requests_server_error++;
                fail(503, outcome.error);
            } else if (outcome.timed_out) {
                g_requests_timeout++;
                fail(504, outcome.error);
            } else {
                g_requests_client_error++;
                fail(400, outcome.error);
            }
            return;
        }

        // The engine emits exactly the forced tokens, so a length mismatch here would mean the
        // forced path silently diverged -- report it rather than returning a misaligned array
        // that a verifier would compare position-by-position against its own reference.
        if (outcome.tokens.size() != forced.size() || entries.size() != forced.size()) {
            g_requests_total++;
            g_requests_server_error++;
            fail(500, "scoring returned " + std::to_string(entries.size()) + " logprobs for " +
                          std::to_string(forced.size()) + " tokens");
            return;
        }

        nlohmann::json tokens = nlohmann::json::array();
        nlohmann::json token_ids = nlohmann::json::array();
        nlohmann::json token_bytes = nlohmann::json::array();
        nlohmann::json logprobs = nlohmann::json::array();
        nlohmann::json alts = nlohmann::json::array();
        double sum_logprob = 0.0;
        for (const auto& e : entries) {
            const auto piece = g_tokenizer.id_to_raw_piece(e.token_id);
            tokens.push_back(piece.display);
            token_ids.push_back(e.token_id);
            nlohmann::json bytes = nlohmann::json::array();
            for (uint8_t b : piece.bytes) bytes.push_back((int)b);
            token_bytes.push_back(std::move(bytes));
            logprobs.push_back(e.logprob);
            sum_logprob += (double)e.logprob;
            if (top_logprobs > 0) {
                nlohmann::json per = nlohmann::json::array();
                const int n = std::min((int)e.top_alternatives.size(), top_logprobs);
                for (int i = 0; i < n; i++)
                    per.push_back(token_logprob_entry_json(e.top_alternatives[i].first,
                                                           e.top_alternatives[i].second));
                alts.push_back(per);
            }
        }

        nlohmann::json usage = {{"prompt_tokens", (int)prompt_ids.size()},
                                {"completion_tokens", (int)forced.size()},
                                {"total_tokens", (int)(prompt_ids.size() + forced.size())}};
        if (outcome.ttft_ms >= 0) usage["ttft_ms"] = outcome.ttft_ms;
        if (outcome.generation_ms >= 0) usage["generation_ms"] = outcome.generation_ms;
        if (outcome.decode_tps >= 0) usage["decode_tps"] = outcome.decode_tps;

        nlohmann::json body = {{"id", random_id()},
                               {"object", "score"},
                               {"model", g_model_name},
                               {"tokens", tokens},
                               {"token_ids", token_ids},
                               {"bytes", token_bytes},
                               {"logprobs", logprobs},
                               {"sum_logprob", sum_logprob},
                               {"usage", usage}};
        if (top_logprobs > 0) body["top_logprobs"] = alts;

        g_requests_total++;
        g_requests_ok++;
        g_prompt_tokens_total += prompt_ids.size();
        g_completion_tokens_total += forced.size();
        res.set_content(body.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&engine](const httplib::Request& req, httplib::Response& res) {
                 if (!auth_ok(req)) {
                     res.status = 401;
                     res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
                     return;
                 }
                 if (g_shutdown_requested.load()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"server is shutting down\"}}",
                                     "application/json");
                     return;
                 }
                 if (!engine.loaded()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
                     return;
                 }

                 g_requests_total++;
                 sparkinfer_server::RequestControls controls;
                 std::string err;
                 if (!sparkinfer_server::parse_request_controls(req.body, controls, err, engine.vocab())) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 // The HTTP server runs ContinuousBatchEngine, whose request path is currently
                 // autoregressive even when the standalone DFlash benchmark env var is present.
                 // Do not reject valid OpenAI sampling controls based on an unrelated process env.
                 if (!controls.seed_set) {
                     static thread_local std::random_device rd;
                     controls.seed = ((uint64_t)rd() << 32) | rd();
                 }
                 const bool stream = controls.stream;
                 if (stream) g_requests_streaming++;
                 const bool enable_thinking = sparkinfer_server::parse_enable_thinking(req.body, engine.is_qwen38());
                 int max_tokens = controls.max_tokens;
                 if (max_tokens <= 0) max_tokens = 256;
                 if (max_tokens > max_output_tokens()) max_tokens = max_output_tokens();

                 std::vector<int> prompt_ids;
                 sparkinfer_server::ChatRequest chat_request;
                 sparkinfer_server::PreparedImages prepared;
                 if (!encode_messages(req.body, prompt_ids, enable_thinking, err, &chat_request)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 // Images: decode, preprocess, and expand each placeholder to the token count
                 // its grid needs. Deliberately BEFORE the max_seq check below -- a 1024x1536
                 // image expands one placeholder into 1536 tokens, so checking the unexpanded
                 // prompt would admit a request that cannot fit and fail it deep in prefill.
                 {
                     const std::vector<std::string> urls =
                         collect_image_urls(chat_request);
                     const std::vector<std::string> video_urls =
                         collect_video_urls(chat_request);
                     if (!urls.empty() || !video_urls.empty()) {
                         const int pad_id = image_pad_token_id();
                         const int vpad_id = video_pad_token_id();
                         const int vstart = vision_start_token_id();
                         const int vend = vision_end_token_id();
                         std::string ierr;
                         if (!urls.empty() && pad_id < 0) {
                             ierr = "this model's tokenizer has no image placeholder token; "
                                    "image input is not supported";
                         } else if (!video_urls.empty() && (vpad_id < 0 || vstart < 0 || vend < 0)) {
                             ierr = "this model's tokenizer has no video placeholder tokens; "
                                    "video input is not supported";
                         } else if (!video_urls.empty()) {
                             // Timestamp markers are tokenized here because the engine has no
                             // tokenizer; encode_raw so "<1.5 seconds>" is not treated as a
                             // special token lookup.
                             sparkinfer_server::ModelEngine::VideoSampling sampling;
                             if (!engine.prepare_vision(
                                     urls, video_urls, pad_id, vpad_id, vstart, vend, sampling,
                                     [](const std::string& t) { return g_tokenizer.encode_raw(t); },
                                     prompt_ids, prepared, ierr)) {
                                 // ierr already names which clip and why.
                             }
                         } else if (!engine.prepare_images(urls, pad_id, prompt_ids, prepared, ierr)) {
                             // ierr already names which image and why.
                         }
                         if (!ierr.empty()) {
                             g_requests_client_error++;
                             res.status = 400;
                             res.set_content("{\"error\":{\"message\":\"" + json_escape(ierr) + "\"}}",
                                             "application/json");
                             return;
                         }
                     }
                 }
                 // Presence of a tool protocol requires strict, buffered parsing even when
                 // tool_choice=none. The latter disables execution, not output validation:
                 // native tool markup must never leak through as ordinary assistant content.
                 const bool tool_protocol = !chat_request.tools.empty();
                 // Mutually exclusive with tool_protocol by construction -- parse_chat_request_json
                 // rejects tools + response_format together at request time.
                 const bool json_mode_active =
                     chat_request.response_format.type != sparkinfer_server::ResponseFormatType::kText;
                 if ((int)prompt_ids.size() + max_tokens > engine.max_seq()) {
                     g_requests_client_error++;
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

                 // Maps an engine outcome to the HTTP status a non-2xx response should use.
                 // Overloaded -> 429 (retry elsewhere / later, not a bad request). Alloc failed ->
                 // 503 (real device OOM -- permanent until restart, never imply "retry shortly"
                 // like 429 does; #779, where this used to fall through to the same 429 as a full
                 // queue and sent operators chasing SPARKINFER_MAX_QUEUE_DEPTH for nothing while
                 // the actual queue sat empty). Timed out -> 504. Pure (no g_requests_* side
                 // effects) -- for n>1, exactly ONE HTTP-level counter increment happens, at the
                 // post-join aggregation point (first-hard-failure-wins), not once per branch.
                 auto status_for_outcome = [](const sparkinfer_server::CompletionResult& o) -> int {
                     if (o.overloaded)   return 429;
                     if (o.alloc_failed) return 503;
                     if (o.timed_out)    return 504;
                     return 400;
                 };

                 if (stream) {
                     res.set_header("Cache-Control", "no-cache");
                     res.set_header("X-Accel-Buffering", "no");
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [&engine, prompt_ids, max_tokens, cid, created, enable_thinking,
                          // BY VALUE, like prompt_ids beside it: this provider runs after the
                          // handler returns, so a reference would dangle. Cheap -- PreparedImages
                          // shares its pixel buffers rather than owning them.
                          prepared,
                          chat_request, tool_protocol, json_mode_active,
                          include_usage = controls.include_usage || always_stream_usage(), stop = controls.stop,
                          temperature = controls.temperature, seed = controls.seed,
                          top_k = controls.top_k, top_p = controls.top_p,
                          presence_penalty = controls.presence_penalty,
                          frequency_penalty = controls.frequency_penalty,
                          logit_bias = controls.logit_bias,
                          logprobs = controls.logprobs, top_logprobs = controls.top_logprobs,
                          n = controls.n]
                         (size_t offset, httplib::DataSink& sink) {
                             if (offset > 0) {
                                 sink.done();
                                 return true;
                             }

                             std::mutex sink_mu;
                             GuardedSink gs{sink, sink_mu};
                             SseHeartbeat heartbeat(gs);

                             for (int ci = 0; ci < n; ci++) {
                                 if (!write_stream_role(gs, cid, created, ci)) {
                                     g_requests_cancelled++;
                                     heartbeat.stop();
                                     sink.done();
                                     return true;
                                 }
                             }

                             // Per-branch outcome, filled in by run_json_mode_branch/
                             // run_plain_or_tool_branch below and aggregated once every branch
                             // has finished (n==1: inline; n>1: joined threads; deduped branches:
                             // copied, see can_dedup below).
                             struct BranchOutcome {
                                 bool ok = false;
                                 enum class Fail { none, cancelled, overloaded, alloc_failed, timeout,
                                                    server_error, invalid_json_output,
                                                    invalid_tool_output } fail = Fail::none;
                                 std::string fail_message;
                                 long long prompt_tokens = 0, completion_tokens = 0;
                                 double ttft_ms = -1.0, generation_ms = -1.0, decode_tps = -1.0;
                                 // Only populated on the json_mode_active/tool_protocol sub-paths
                                 // -- used to replay an already-buffered result to a deduped index
                                 // (see can_dedup below). The plain sub-path emits incrementally,
                                 // live, and never populates these.
                                 sparkinfer_server::ParsedAssistantOutput parsed;
                                 std::string finish_reason;
                             };

                             // Emits an already-computed json_mode/tool_protocol result to choice
                             // index ci -- used for a real branch's own first emission AND to
                             // replay a deduped branch's output to another index. Takes `parsed`
                             // by value so reassigning tool-call ids (fresh per emitted choice,
                             // even on replay) never mutates the caller's stored copy.
                             auto emit_buffered_output = [&](int ci,
                                                             sparkinfer_server::ParsedAssistantOutput parsed,
                                                             const std::string& finish_reason) {
                                 if (!chat_request.reasoning_exclude)
                                     write_stream_reasoning_delta(gs, cid, created,
                                                                  parsed.reasoning_content, ci);
                                 write_stream_delta(gs, cid, created, "content", parsed.content, ci);
                                 for (size_t i = 0; i < parsed.tool_calls.size(); ++i) {
                                     parsed.tool_calls[i].id = random_id("call_");
                                     write_stream_tool_call(gs, cid, created, ci, i, parsed.tool_calls[i]);
                                 }
                                 write_stream_finish(gs, cid, created, ci, finish_reason);
                             };

                             // response_format needs the complete, validated output before
                             // anything is emitted -- a client can't un-receive already-streamed
                             // bytes if a retry becomes necessary. Buffer fully internally (like
                             // tool_protocol below), then emit via emit_buffered_output once a
                             // valid attempt is confirmed.
                             auto run_json_mode_branch = [&](int ci, uint64_t branch_seed, BranchOutcome* out) {
                                 std::vector<int> cur_prompt_ids = prompt_ids;
                                 // Per-branch copy: the continuation below re-renders the prompt,
                                 // which moves every placeholder, and branches run concurrently.
                                 // The copy is cheap -- the pixel buffers are shared, not cloned.
                                 sparkinfer_server::PreparedImages cur_images = prepared;
                                 sparkinfer_server::ChatRequest cur_request = chat_request;
                                 sparkinfer_server::CompletionResult outcome;
                                 bool ok = false;
                                 std::string validation_err;
                                 for (int attempt = 1; attempt <= 2; ++attempt) {
                                     if ((int)cur_prompt_ids.size() + max_tokens > engine.max_seq()) {
                                         validation_err = "retry prompt exceeds server context";
                                         break;
                                     }
                                     std::vector<int> ids;
                                     std::string stop_text;
                                     bool stopped_by_sequence = false;
                                     auto on_tok = [&](int tid) -> bool {
                                         if (stop.empty()) {
                                             ids.push_back(tid);
                                             return sink.is_writable();
                                         }
                                         stop_text += g_tokenizer.decode_delta(ids, tid);
                                         size_t pos;
                                         if (find_stop_match(stop_text, stop, pos)) {
                                             stopped_by_sequence = true;
                                             return false;
                                         }
                                         return sink.is_writable();
                                     };
                                     outcome = engine.complete_streaming(cur_prompt_ids, max_tokens, on_tok,
                                         temperature, branch_seed, top_k, top_p, presence_penalty,
                                         frequency_penalty, logit_bias, false, 0, nullptr, {},
                                         &cur_images);
                                     out->prompt_tokens += (long long)cur_prompt_ids.size();
                                     out->completion_tokens += (long long)ids.size();
                                     if (outcome.cancelled && !stopped_by_sequence) {
                                         out->fail = BranchOutcome::Fail::cancelled;
                                         return;
                                     }
                                     if (!outcome.error.empty()) {
                                         out->fail = outcome.overloaded ? BranchOutcome::Fail::overloaded
                                                   : outcome.alloc_failed ? BranchOutcome::Fail::alloc_failed
                                                   : outcome.timed_out ? BranchOutcome::Fail::timeout
                                                                        : BranchOutcome::Fail::server_error;
                                         out->fail_message = outcome.error;
                                         return;
                                     }
                                     std::string text;
                                     std::string decode_err;
                                     if (!decode_ids(ids, text, decode_err)) {
                                         out->fail = BranchOutcome::Fail::server_error;
                                         out->fail_message = decode_err;
                                         return;
                                     }
                                     if (stopped_by_sequence) {
                                         size_t pos;
                                         if (find_stop_match(text, stop, pos)) text.resize(pos);
                                     }
                                     out->parsed = sparkinfer_server::parse_assistant_output(
                                         text, enable_thinking, engine.is_museglimmer(), nullptr);
                                     const bool truncated = outcome.reached_token_limit || stopped_by_sequence;
                                     if (truncated) {
                                         validation_err = outcome.reached_token_limit
                                             ? "truncated: hit max_tokens before producing valid output"
                                             : "truncated: hit a stop sequence before producing valid output";
                                     } else if (!sparkinfer_server::validate_response_format(
                                                    out->parsed.content, cur_request.response_format, validation_err)) {
                                         // validation_err already set
                                     } else {
                                         ok = true;
                                         break;
                                     }
                                     if (attempt == 1) {
                                         cur_request = build_retry_request(chat_request, out->parsed.content, validation_err);
                                         cur_prompt_ids = g_tokenizer.encode_augmented(cur_request, enable_thinking);
                                         // Same re-expansion as the non-streaming branch: the
                                         // rebuilt prompt's placeholders moved.
                                         if (!cur_images.images.empty()) {
                                             std::string rerr;
                                             if (!engine.reexpand_images(image_pad_token_id(),
                                                                         cur_prompt_ids, cur_images, rerr)) {
                                                 out->fail = BranchOutcome::Fail::server_error;
                                                 out->fail_message = "image re-expansion failed: " + rerr;
                                                 return;
                                             }
                                         }
                                     }
                                 }
                                 if (!ok) {
                                     out->fail = BranchOutcome::Fail::invalid_json_output;
                                     out->fail_message = "model output did not satisfy response_format after "
                                                          "retry: " + validation_err;
                                     return;
                                 }
                                 out->finish_reason = "stop";
                                 out->ttft_ms = outcome.ttft_ms;
                                 out->generation_ms = outcome.generation_ms;
                                 out->decode_tps = outcome.decode_tps;
                                 out->ok = true;
                                 emit_buffered_output(ci, out->parsed, out->finish_reason);
                             };

                             auto run_plain_or_tool_branch = [&](int ci, uint64_t branch_seed, BranchOutcome* out) {
                                 std::vector<int> stream_ids;
                                 stream_ids.reserve((size_t)max_tokens);
                                 sparkinfer_server::ThinkingStreamSplitter splitter(enable_thinking, engine.is_museglimmer());
                                 sparkinfer_server::StopSequenceFilter stop_filter(stop);
                                 std::string tool_stop_text;  // raw accumulator, tool_protocol branch only
                                 bool stopped_by_sequence = false;
                                 // Scoped out of tool-calling responses (buffered, no live content
                                 // emission at all -- see the DFlash-check comment above for the
                                 // parallel "logprobs never touches sampling" note).
                                 const bool want_logprobs = logprobs && !tool_protocol;
                                 std::vector<sparkinfer_server::TokenLogprob> pending_logprobs;
                                 auto on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                                     if (engine.is_stop_token(tl.token_id)) return;
                                     pending_logprobs.push_back(tl);
                                 };
                                 // Hand off every entry accumulated since the last delta and clear
                                 // the buffer, so no entry is emitted twice or left behind. Returns
                                 // null (not an empty array) when there is nothing to report or the
                                 // request never asked -- which is what `"logprobs": null` means.
                                 auto take_pending = [&]() -> nlohmann::json {
                                     if (!want_logprobs || pending_logprobs.empty()) return nullptr;
                                     nlohmann::json lp{{"content",
                                         build_logprobs_content_json(pending_logprobs, top_logprobs)}};
                                     pending_logprobs.clear();
                                     return lp;
                                 };
                                 auto on_tok = [&](int tid) -> bool {
                                     // Tool-capable responses are intentionally buffered until the
                                     // native Qwen XML is complete and schema-valid.
                                     if (tool_protocol) {
                                         if (stop.empty()) {
                                             stream_ids.push_back(tid);
                                             return sink.is_writable();
                                         }
                                         tool_stop_text += g_tokenizer.decode_delta(stream_ids, tid);
                                         size_t pos;
                                         if (find_stop_match(tool_stop_text, stop, pos)) {
                                             stopped_by_sequence = true;
                                             return false;
                                         }
                                         return sink.is_writable();
                                     }
                                     std::string piece = g_tokenizer.decode_delta(stream_ids, tid);
                                     std::string safe = stop_filter.feed(piece);
                                     if (stop_filter.matched()) {
                                         if (!safe.empty()) {
                                             const auto delta = splitter.feed(safe);
                                             if (!chat_request.reasoning_exclude && !delta.reasoning_content.empty())
                                                 write_stream_reasoning_delta(gs, cid, created,
                                                                              delta.reasoning_content, ci,
                                                                              take_pending());
                                             if (!delta.content.empty())
                                                 write_stream_delta(gs, cid, created, "content", delta.content,
                                                                    ci, take_pending());
                                         }
                                         stopped_by_sequence = true;
                                         return false;
                                     }
                                     const auto delta = splitter.feed(safe);
                                     bool ok = true;
                                     // take_pending() empties the buffer, so whichever delta is
                                     // written first carries the entries for the tokens that
                                     // produced it. Reasoning deltas take them too: a token routed
                                     // to reasoning_content used to leave its entry behind to be
                                     // misattributed to the next CONTENT chunk, which then reported
                                     // logprobs for tokens whose text it does not contain.
                                     // Concatenating every chunk's entries now yields exactly one
                                     // per generated token, matching the non-streaming response.
                                     if (!chat_request.reasoning_exclude && !delta.reasoning_content.empty())
                                         ok = write_stream_reasoning_delta(gs, cid, created,
                                                                            delta.reasoning_content, ci,
                                                                            take_pending()) && ok;
                                     if (!delta.content.empty())
                                         ok = write_stream_delta(gs, cid, created, "content", delta.content,
                                                                 ci, take_pending()) && ok;
                                     return ok && sink.is_writable();
                                 };
                                 const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_on_tok_logprob =
                                     want_logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(on_tok_logprob)
                                                  : nullptr;
                                 const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, on_tok,
                                     temperature, branch_seed, top_k, top_p, presence_penalty, frequency_penalty,
                                     logit_bias, logprobs, top_logprobs, maybe_on_tok_logprob,
                                     {}, &prepared);
                                 out->prompt_tokens = (long long)prompt_ids.size();
                                 out->completion_tokens = (long long)stream_ids.size();
                                 if (outcome.cancelled && !stopped_by_sequence) {
                                     out->fail = BranchOutcome::Fail::cancelled;
                                     return;
                                 }
                                 if (!outcome.error.empty()) {
                                     out->fail = outcome.overloaded ? BranchOutcome::Fail::overloaded
                                               : outcome.alloc_failed ? BranchOutcome::Fail::alloc_failed
                                               : outcome.timed_out ? BranchOutcome::Fail::timeout
                                                                    : BranchOutcome::Fail::server_error;
                                     out->fail_message = outcome.error;
                                     return;
                                 }

                                 out->finish_reason = outcome.reached_token_limit ? "length" : "stop";
                                 if (tool_protocol) {
                                     std::string text;
                                     std::string decode_err;
                                     if (!decode_ids(stream_ids, text, decode_err)) {
                                         out->fail = BranchOutcome::Fail::server_error;
                                         out->fail_message = decode_err;
                                         return;
                                     }
                                     if (stopped_by_sequence) {
                                         size_t pos;
                                         if (find_stop_match(text, stop, pos)) text.resize(pos);
                                     }
                                     out->parsed = sparkinfer_server::parse_assistant_output(
                                         text, enable_thinking, engine.is_museglimmer(), &chat_request);
                                     if (!out->parsed.error.empty()) {
                                         if (outcome.reached_token_limit || stopped_by_sequence) {
                                             // A truncated native call is not an executable result,
                                             // but token exhaustion (or a stop sequence landing mid
                                             // tool-call XML) is still a normal completion.
                                             out->parsed = {};
                                         } else {
                                             out->fail = BranchOutcome::Fail::invalid_tool_output;
                                             out->fail_message = "invalid model tool call: " + out->parsed.error;
                                             return;
                                         }
                                     }
                                     if (!out->parsed.tool_calls.empty()) out->finish_reason = "tool_calls";
                                     out->ttft_ms = outcome.ttft_ms;
                                     out->generation_ms = outcome.generation_ms;
                                     out->decode_tps = outcome.decode_tps;
                                     out->ok = true;
                                     emit_buffered_output(ci, out->parsed, out->finish_reason);
                                     return;
                                 }
                                 if (!stopped_by_sequence) {
                                     // On a stop-match exit, on_tok already wrote whatever safe text
                                     // stop_filter cleared before returning false. Deliberately skip
                                     // finish() here: it exists to flush the splitter's OWN unrelated
                                     // holdback on a legitimate end of generation, not bytes stop_filter
                                     // never vetted -- generation ends at the match point regardless.
                                     sparkinfer_server::ThinkingStreamSplitter::Delta flush;
                                     splitter.finish(flush);
                                     // Guarded on non-empty exactly like the on_tok path: arguments
                                     // are evaluated before the call, and write_stream_delta drops
                                     // empty pieces, so an unguarded take_pending() here would
                                     // consume the entries and then throw them away with the
                                     // delta -- and the safety net below could not see them,
                                     // because the buffer would already be clear.
                                     if (!chat_request.reasoning_exclude && !flush.reasoning_content.empty())
                                         write_stream_reasoning_delta(gs, cid, created,
                                                                      flush.reasoning_content, ci,
                                                                      take_pending());
                                     if (!flush.content.empty())
                                         write_stream_delta(gs, cid, created, "content", flush.content,
                                                            ci, take_pending());
                                 }
                                 // Neither flush piece may have had text to hang them on. Emit
                                 // them rather than lose them.
                                 if (want_logprobs && !pending_logprobs.empty())
                                     write_stream_logprobs_only(gs, cid, created, ci, take_pending(),
                                                                "content");
                                 out->ttft_ms = outcome.ttft_ms;
                                 out->generation_ms = outcome.generation_ms;
                                 out->decode_tps = outcome.decode_tps;
                                 out->ok = true;
                                 write_stream_finish(gs, cid, created, ci, out->finish_reason);
                             };

                             // Wrapped in try/catch: for n>1 this body runs on a spawned
                             // std::thread (see the fan-out below), not inside httplib's own
                             // request-dispatch call stack -- an exception escaping a std::thread
                             // callable is outside httplib's exception handling and calls
                             // std::terminate(), crashing the WHOLE server for every concurrent
                             // client rather than failing just this one request. Converting any
                             // exception into the existing server_error path keeps that blast
                             // radius scoped to one branch.
                             auto run_one = [&](int ci, uint64_t branch_seed, BranchOutcome* out) {
                                 try {
                                     if (json_mode_active) run_json_mode_branch(ci, branch_seed, out);
                                     else run_plain_or_tool_branch(ci, branch_seed, out);
                                 } catch (const std::exception& e) {
                                     // ok may already be true if the throw happened inside the
                                     // final emit call (after the branch's own generation/parsing
                                     // succeeded) -- force it back to false so the post-join
                                     // aggregation's `!results[ci].ok` scan actually catches this.
                                     out->ok = false;
                                     out->fail = BranchOutcome::Fail::server_error;
                                     out->fail_message = e.what();
                                 } catch (...) {
                                     out->ok = false;
                                     out->fail = BranchOutcome::Fail::server_error;
                                     out->fail_message = "unknown exception";
                                 }
                             };

                             std::vector<BranchOutcome> results(n);
                             std::vector<uint64_t> branch_seeds(n);
                             for (int i = 0; i < n; i++) branch_seeds[i] = seed + (uint64_t)i;

                             // temperature<=0 dedup: json_mode_active/tool_protocol fully buffer
                             // their output before emitting it, so a duplicate choice can just
                             // replay branch 0's already-computed result instead of re-running
                             // generation. The plain streaming sub-path emits incrementally, live,
                             // as tokens generate -- deduping THAT needs recording and replaying
                             // the whole delta sequence, deferred to a follow-up; at
                             // temperature<=0 with n>1 it still works correctly, it just genuinely
                             // reruns n identical decodes (documented v1 scope limit).
                             const bool can_dedup = temperature <= 0.f && (json_mode_active || tool_protocol);

                             if (n == 1) {
                                 run_one(0, branch_seeds[0], &results[0]);
                             } else if (can_dedup) {
                                 run_one(0, branch_seeds[0], &results[0]);
                                 if (results[0].ok) {
                                     for (int ci = 1; ci < n; ci++) {
                                         results[ci] = results[0];
                                         // No real prefill/decode happened for this replayed
                                         // choice -- don't double-count GPU cost.
                                         results[ci].prompt_tokens = 0;
                                         results[ci].completion_tokens = 0;
                                         emit_buffered_output(ci, results[0].parsed, results[0].finish_reason);
                                     }
                                 }
                                 // else: branch 0 failed -- the whole request already fails on
                                 // this (first-hard-failure-wins, see aggregation below), so
                                 // branches 1..n-1 are left un-run rather than wasting GPU time on
                                 // completions the response will discard.
                             } else {
                                 std::vector<std::thread> threads;
                                 threads.reserve(n);
                                 for (int ci = 0; ci < n; ci++)
                                     threads.emplace_back(run_one, ci, branch_seeds[ci], &results[ci]);
                                 for (auto& t : threads) t.join();
                             }

                             // Single-threaded again: first-hard-failure-wins for both the
                             // g_requests_* counters and the HTTP-visible outcome.
                             int first_fail = -1;
                             for (int ci = 0; ci < n; ci++) {
                                 if (!results[ci].ok) { first_fail = ci; break; }
                             }
                             long long agg_prompt = 0, agg_completion = 0;
                             for (const auto& r : results) {
                                 agg_prompt += r.prompt_tokens;
                                 agg_completion += r.completion_tokens;
                             }
                             g_prompt_tokens_total += (uint64_t)agg_prompt;
                             g_completion_tokens_total += (uint64_t)agg_completion;

                             if (first_fail >= 0) {
                                 const auto& f = results[first_fail];
                                 switch (f.fail) {
                                     case BranchOutcome::Fail::cancelled:           g_requests_cancelled++; break;
                                     case BranchOutcome::Fail::overloaded:          g_requests_overloaded++; break;
                                     case BranchOutcome::Fail::alloc_failed:        g_requests_alloc_failed++; break;
                                     case BranchOutcome::Fail::timeout:             g_requests_timeout++; break;
                                     case BranchOutcome::Fail::invalid_json_output: g_requests_invalid_json_output++; break;
                                     case BranchOutcome::Fail::invalid_tool_output: g_requests_invalid_tool_output++; break;
                                     default:                                       g_requests_server_error++; break;
                                 }
                                 if (f.fail == BranchOutcome::Fail::cancelled) {
                                     // Client is already gone -- nothing left to write to.
                                     heartbeat.stop();
                                     sink.done();
                                     return true;
                                 }
                                 write_sse_json(gs, {{"error", {{"message", f.fail_message}}}});
                                 // A client that explicitly asked for usage still deserves the
                                 // token counts even though generation faulted -- ttft/generation
                                 // timing isn't meaningful for a hard engine error, so omit those.
                                 if (include_usage)
                                     write_stream_usage(gs, cid, created, (int)results[0].prompt_tokens,
                                                        (int)agg_completion, -1.0, -1.0, -1.0);
                                 heartbeat.stop();
                                 write_stream_done(gs);
                                 sink.done();
                                 return true;
                             }

                             g_requests_ok++;
                             double ttft_min = -1.0, gen_max = -1.0;
                             for (const auto& r : results) {
                                 if (r.ttft_ms >= 0.0 && (ttft_min < 0.0 || r.ttft_ms < ttft_min)) ttft_min = r.ttft_ms;
                                 if (r.generation_ms >= 0.0 && r.generation_ms > gen_max) gen_max = r.generation_ms;
                             }
                             const double decode_tps_agg =
                                 gen_max > 0.0 ? (double)agg_completion / (gen_max / 1000.0) : -1.0;
                             if (include_usage)
                                 // prompt_tokens reported ONCE (the shared prompt, not xn) --
                                 // intentionally diverges from g_prompt_tokens_total above, which
                                 // sums real per-branch prefill cost; see plan for why.
                                 write_stream_usage(gs, cid, created, (int)results[0].prompt_tokens,
                                                    (int)agg_completion, ttft_min, gen_max, decode_tps_agg);
                             heartbeat.stop();
                             write_stream_done(gs);
                             sink.done();
                             return true;
                         });
                     return;
                 }

                 // engine.complete() is complete_streaming(..., nullptr); pass a real callback
                 // whenever stop was requested so non-streaming requests can halt early too --
                 // previously stop-checking (and the compute savings of stopping early) were
                 // unavailable here entirely.
                 //
                 // One independent completion per branch_seed -- run_nonstream_branch is called
                 // once for n==1 (today's only behavior), n times fanned across threads for n>1
                 // (or once for real + (n-1) cheap replays under the temperature<=0 dedup
                 // optimization, see below). No shared mutable state between branches: each
                 // thread writes only its own results[i] slot, so unlike the streaming path this
                 // needs no mutex.
                 struct NonStreamBranchOutcome {
                     bool ok = false;
                     enum class Fail { none, overloaded, alloc_failed, timeout, server_error,
                                        invalid_json_output, invalid_tool_output } fail = Fail::none;
                     int http_status = 200;
                     std::string http_error;
                     nlohmann::json message;
                     std::string finish_reason;
                     nlohmann::json logprobs_json = nullptr;
                     long long prompt_tokens = 0, completion_tokens = 0;
                     double ttft_ms = -1.0, generation_ms = -1.0, decode_tps = -1.0;
                 };

                 // Wrapped in try/catch: for n>1 this body runs on a spawned std::thread (see
                 // the fan-out below), outside httplib's own exception handling -- an escaping
                 // exception would call std::terminate() and crash the whole server, not just
                 // fail this one request. Converting any exception into the existing
                 // server_error path keeps the blast radius scoped to one branch.
                 auto run_nonstream_branch = [&](uint64_t branch_seed) -> NonStreamBranchOutcome {
                   try {
                     NonStreamBranchOutcome out;
                     sparkinfer_server::CompletionResult outcome;
                     sparkinfer_server::ParsedAssistantOutput parsed;
                     std::string finish_reason;
                     // Populated only in the plain (non-json_mode, non-tool_protocol) branch below
                     // -- see the DFlash-check comment above for why tool-calling/response_format
                     // responses are scoped out of logprobs entirely for v1.
                     std::vector<sparkinfer_server::TokenLogprob> logprob_entries;
                     const bool want_logprobs = controls.logprobs && !tool_protocol;

                     if (json_mode_active) {
                         // response_format validation needs the complete output; a truncated or
                         // schema-violating attempt is retried once with a corrective follow-up
                         // prompt (this runtime's decode is deterministic greedy argmax with no RNG
                         // anywhere -- resubmitting the identical prompt would just reproduce the
                         // same invalid output) before giving up.
                         std::vector<int> cur_prompt_ids = prompt_ids;
                         sparkinfer_server::PreparedImages cur_images = prepared;
                         sparkinfer_server::ChatRequest cur_request = chat_request;
                         bool ok = false;
                         std::string validation_err;
                         for (int attempt = 1; attempt <= 2; ++attempt) {
                             if ((int)cur_prompt_ids.size() + max_tokens > engine.max_seq()) {
                                 validation_err = "retry prompt exceeds server context";
                                 break;
                             }
                             std::vector<int> ids;
                             std::string stop_text;
                             bool stopped_by_sequence = false;
                             auto on_tok = [&](int tid) -> bool {
                                 if (controls.stop.empty()) {
                                     ids.push_back(tid);
                                     return true;
                                 }
                                 stop_text += g_tokenizer.decode_delta(ids, tid);
                                 size_t pos;
                                 if (find_stop_match(stop_text, controls.stop, pos)) {
                                     stopped_by_sequence = true;
                                     return false;
                                 }
                                 return true;
                             };
                             outcome = engine.complete_streaming(cur_prompt_ids, max_tokens, on_tok,
                                 controls.temperature, branch_seed, controls.top_k, controls.top_p,
                                 controls.presence_penalty, controls.frequency_penalty, controls.logit_bias,
                                 false, 0, nullptr, {}, &cur_images);
                             out.prompt_tokens += (long long)cur_prompt_ids.size();
                             out.completion_tokens += (long long)ids.size();
                             if (!outcome.error.empty()) {
                                 // A hard engine fault (overloaded/alloc_failed/timed_out) is not a
                                 // validation failure -- never retry it, propagate immediately.
                                 out.http_status = status_for_outcome(outcome);
                                 out.fail = outcome.overloaded ? NonStreamBranchOutcome::Fail::overloaded
                                          : outcome.alloc_failed ? NonStreamBranchOutcome::Fail::alloc_failed
                                          : outcome.timed_out ? NonStreamBranchOutcome::Fail::timeout
                                                               : NonStreamBranchOutcome::Fail::server_error;
                                 out.http_error = outcome.error;
                                 return out;
                             }
                             std::string text;
                             std::string decode_err;
                             if (!decode_ids(ids, text, decode_err)) {
                                 out.http_status = 500;
                                 out.fail = NonStreamBranchOutcome::Fail::server_error;
                                 out.http_error = decode_err;
                                 return out;
                             }
                             if (stopped_by_sequence) {
                                 size_t pos;
                                 if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                             }
                             parsed = sparkinfer_server::parse_assistant_output(
                                 text, enable_thinking, engine.is_museglimmer(), nullptr);
                             // Unlike tool_protocol, truncation is NOT a free pass here -- a
                             // stop/length-truncated response is essentially always invalid JSON, so
                             // it's a normal validation failure subject to the same retry policy.
                             const bool truncated = outcome.reached_token_limit || stopped_by_sequence;
                             if (truncated) {
                                 validation_err = outcome.reached_token_limit
                                     ? "truncated: hit max_tokens before producing valid output"
                                     : "truncated: hit a stop sequence before producing valid output";
                             } else if (!sparkinfer_server::validate_response_format(
                                            parsed.content, cur_request.response_format, validation_err)) {
                                 // validation_err already set by validate_response_format
                             } else {
                                 ok = true;
                                 break;
                             }
                             if (attempt == 1) {
                                 cur_request = build_retry_request(chat_request, parsed.content, validation_err);
                                 cur_prompt_ids = g_tokenizer.encode_augmented(cur_request, enable_thinking);
                                 // The rebuilt prompt carries one bare placeholder per image
                                 // again, at new offsets -- re-expand, or the splice lands on the
                                 // wrong tokens.
                                 if (!cur_images.images.empty()) {
                                     std::string rerr;
                                     if (!engine.reexpand_images(image_pad_token_id(), cur_prompt_ids,
                                                                 cur_images, rerr)) {
                                         out.http_status = 500;
                                         out.fail = NonStreamBranchOutcome::Fail::server_error;
                                         out.http_error = "image re-expansion failed: " + rerr;
                                         return out;
                                     }
                                 }
                             }
                         }
                         if (!ok) {
                             out.http_status = 502;
                             out.fail = NonStreamBranchOutcome::Fail::invalid_json_output;
                             out.http_error = "model output did not satisfy response_format after "
                                               "retry: " + validation_err;
                             return out;
                         }
                         finish_reason = "stop";  // ok only true on a non-truncated, valid attempt
                     } else {
                         std::vector<int> nonstream_ids;
                         std::string nonstream_stop_text;
                         bool stopped_by_sequence = false;
                         auto nonstream_on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                             if (engine.is_stop_token(tl.token_id)) return;
                             logprob_entries.push_back(tl);
                         };
                         auto nonstream_on_tok = [&](int tid) -> bool {
                             if (controls.stop.empty()) {
                                 nonstream_ids.push_back(tid);
                                 return true;
                             }
                             nonstream_stop_text += g_tokenizer.decode_delta(nonstream_ids, tid);
                             size_t pos;
                             if (find_stop_match(nonstream_stop_text, controls.stop, pos)) {
                                 stopped_by_sequence = true;
                                 // The match-triggering token's text got truncated below (a stop
                                 // match can land mid-token) -- drop its logprobs entry too, no
                                 // partial-credit modeling for v1. Fires after nonstream_on_tok_logprob
                                 // (delivered before on_token for the same token, see step_job()'s
                                 // ordering contract), so this correctly removes THIS token's entry.
                                 if (want_logprobs && !logprob_entries.empty()) logprob_entries.pop_back();
                                 return false;
                             }
                             return true;
                         };
                         const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_nonstream_on_tok_logprob =
                             want_logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(nonstream_on_tok_logprob)
                                          : nullptr;
                         outcome = engine.complete_streaming(prompt_ids, max_tokens, nonstream_on_tok,
                             controls.temperature, branch_seed, controls.top_k, controls.top_p,
                             controls.presence_penalty, controls.frequency_penalty, controls.logit_bias,
                             controls.logprobs, controls.top_logprobs, maybe_nonstream_on_tok_logprob,
                             {}, &prepared);
                         // Defensive clamp -- should already hold, cheap insurance against any
                         // subtle off-by-one between the two accumulation paths above.
                         if (logprob_entries.size() > outcome.tokens.size())
                             logprob_entries.resize(outcome.tokens.size());
                         std::string text;
                         if (!outcome.error.empty()) {
                             out.http_status = status_for_outcome(outcome);
                             out.fail = outcome.overloaded ? NonStreamBranchOutcome::Fail::overloaded
                                      : outcome.alloc_failed ? NonStreamBranchOutcome::Fail::alloc_failed
                                      : outcome.timed_out ? NonStreamBranchOutcome::Fail::timeout
                                                           : NonStreamBranchOutcome::Fail::server_error;
                             out.http_error = outcome.error;
                             return out;
                         }
                         std::string decode_err;
                         if (!decode_ids(outcome.tokens, text, decode_err)) {
                             out.http_status = 500;
                             out.fail = NonStreamBranchOutcome::Fail::server_error;
                             out.http_error = decode_err;
                             return out;
                         }
                         out.prompt_tokens = (long long)prompt_ids.size();
                         out.completion_tokens = (long long)outcome.tokens.size();
                         if (stopped_by_sequence) {
                             size_t pos;
                             if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                         }

                         parsed = sparkinfer_server::parse_assistant_output(
                             text, enable_thinking, engine.is_museglimmer(),
                             tool_protocol ? &chat_request : nullptr);
                         if (!parsed.error.empty()) {
                             if (outcome.reached_token_limit || stopped_by_sequence) {
                                 // Never expose a truncated native tag sequence. A length/stop end
                                 // is a valid completion, so return an empty assistant result
                                 // instead of a hard failure.
                                 parsed = {};
                             } else {
                                 out.http_status = 502;
                                 out.fail = NonStreamBranchOutcome::Fail::invalid_tool_output;
                                 out.http_error = "invalid model tool call: " + parsed.error;
                                 return out;
                             }
                         }
                         finish_reason = outcome.reached_token_limit ? "length" : "stop";
                     }

                     nlohmann::json message = {{"role", "assistant"}};
                     if (!chat_request.reasoning_exclude && !parsed.reasoning_content.empty()) {
                         message["reasoning"] = parsed.reasoning_content;
                         message["reasoning_content"] = parsed.reasoning_content;
                     }
                     // parsed.tool_calls is only non-empty here for a complete, successfully-
                     // parsed call.
                     if (!parsed.tool_calls.empty()) {
                         message["content"] = parsed.content.empty()
                             ? nlohmann::json(nullptr) : nlohmann::json(parsed.content);
                         message["tool_calls"] = nlohmann::json::array();
                         for (auto& call : parsed.tool_calls) {
                             call.id = random_id("call_");
                             message["tool_calls"].push_back({
                                 {"id", call.id}, {"type", "function"},
                                 {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
                         }
                         finish_reason = "tool_calls";
                     } else {
                         message["content"] = parsed.content;
                     }

                     out.message = std::move(message);
                     out.finish_reason = finish_reason;
                     out.logprobs_json = want_logprobs
                         ? nlohmann::json{{"content", build_logprobs_content_json(logprob_entries, controls.top_logprobs)}}
                         : nlohmann::json(nullptr);
                     out.ttft_ms = outcome.ttft_ms;
                     out.generation_ms = outcome.generation_ms;
                     out.decode_tps = outcome.decode_tps;
                     out.ok = true;
                     return out;
                   } catch (const std::exception& e) {
                     NonStreamBranchOutcome out;
                     out.http_status = 500;
                     out.fail = NonStreamBranchOutcome::Fail::server_error;
                     out.http_error = e.what();
                     return out;
                   } catch (...) {
                     NonStreamBranchOutcome out;
                     out.http_status = 500;
                     out.fail = NonStreamBranchOutcome::Fail::server_error;
                     out.http_error = "unknown exception";
                     return out;
                   }
                 };

                 std::vector<uint64_t> branch_seeds(controls.n);
                 for (int i = 0; i < controls.n; i++) branch_seeds[i] = controls.seed + (uint64_t)i;
                 std::vector<NonStreamBranchOutcome> results(controls.n);

                 // temperature<=0 dedup: every non-streaming sub-path (json_mode_active,
                 // tool_protocol, plain) already fully buffers its output before this point, so a
                 // duplicate choice can just reuse branch 0's already-built result instead of
                 // re-running generation -- unlike the streaming path, there is no incrementally-
                 // emitted sub-case here to carve out.
                 const bool can_dedup = controls.temperature <= 0.f;
                 if (controls.n == 1) {
                     results[0] = run_nonstream_branch(branch_seeds[0]);
                 } else if (can_dedup) {
                     results[0] = run_nonstream_branch(branch_seeds[0]);
                     if (results[0].ok) {
                         for (int i = 1; i < controls.n; i++) {
                             results[i] = results[0];
                             // No real prefill/decode happened for this replayed choice -- don't
                             // double-count GPU cost.
                             results[i].prompt_tokens = 0;
                             results[i].completion_tokens = 0;
                         }
                     }
                     // else: branch 0 failed -- the whole request already fails on this
                     // (first-hard-failure-wins, below), so branches 1..n-1 are left un-run.
                 } else {
                     std::vector<std::thread> threads;
                     threads.reserve(controls.n);
                     for (int i = 0; i < controls.n; i++)
                         threads.emplace_back([&, i] { results[i] = run_nonstream_branch(branch_seeds[i]); });
                     for (auto& t : threads) t.join();
                 }

                 // Single-threaded again: first-hard-failure-wins for both the g_requests_*
                 // counters and the HTTP-visible outcome -- if any branch failed, the whole
                 // request fails with that branch's status/error, matching the streaming path's
                 // policy and generalizing today's n=1 behavior.
                 int first_fail = -1;
                 for (int i = 0; i < controls.n; i++) {
                     if (!results[i].ok) { first_fail = i; break; }
                 }
                 long long agg_prompt = 0, agg_completion = 0;
                 for (const auto& r : results) {
                     agg_prompt += r.prompt_tokens;
                     agg_completion += r.completion_tokens;
                 }
                 g_prompt_tokens_total += (uint64_t)agg_prompt;
                 g_completion_tokens_total += (uint64_t)agg_completion;

                 if (first_fail >= 0) {
                     const auto& f = results[first_fail];
                     switch (f.fail) {
                         case NonStreamBranchOutcome::Fail::overloaded:          g_requests_overloaded++; break;
                         case NonStreamBranchOutcome::Fail::alloc_failed:        g_requests_alloc_failed++; break;
                         case NonStreamBranchOutcome::Fail::timeout:             g_requests_timeout++; break;
                         case NonStreamBranchOutcome::Fail::invalid_json_output: g_requests_invalid_json_output++; break;
                         case NonStreamBranchOutcome::Fail::invalid_tool_output: g_requests_invalid_tool_output++; break;
                         default:                                               g_requests_server_error++; break;
                     }
                     res.status = f.http_status;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(f.http_error) + "\"}}",
                                     "application/json");
                     return;
                 }

                 double ttft_min = -1.0, gen_max = -1.0;
                 for (const auto& r : results) {
                     if (r.ttft_ms >= 0.0 && (ttft_min < 0.0 || r.ttft_ms < ttft_min)) ttft_min = r.ttft_ms;
                     if (r.generation_ms >= 0.0 && r.generation_ms > gen_max) gen_max = r.generation_ms;
                 }
                 const double decode_tps_agg =
                     gen_max > 0.0 ? (double)agg_completion / (gen_max / 1000.0) : -1.0;

                 // prompt_tokens reported ONCE (the shared prompt, not xn) -- intentionally
                 // diverges from g_prompt_tokens_total above, which sums real per-branch prefill
                 // cost; see plan for why.
                 nlohmann::json usage = {
                     {"prompt_tokens", (int)results[0].prompt_tokens},
                     {"completion_tokens", (int)agg_completion},
                     {"total_tokens", (int)(results[0].prompt_tokens + agg_completion)}};
                 if (ttft_min >= 0.0) usage["ttft_ms"] = ttft_min;
                 if (gen_max >= 0.0) usage["generation_ms"] = gen_max;
                 if (decode_tps_agg >= 0.0) usage["decode_tps"] = decode_tps_agg;

                 nlohmann::json choices = nlohmann::json::array();
                 for (int i = 0; i < controls.n; i++) {
                     choices.push_back({{"index", i},
                                        {"message", results[i].message},
                                        {"logprobs", results[i].logprobs_json},
                                        {"finish_reason", results[i].finish_reason}});
                 }
                 const nlohmann::json body = {
                     {"id", cid}, {"object", "chat.completion"}, {"created", created},
                     {"model", g_model_name}, {"choices", choices}, {"usage", usage}};
                 g_requests_ok++;
                 res.set_content(body.dump(), "application/json");
             });

    // Legacy pre-chat API: raw prompt string in, plain text out. No messages array, no chat
    // template, no tool-calling, no response_format, no reasoning split -- RequestControls/
    // parse_request_controls (temperature/top_p/top_k/penalties/logit_bias/n/seed/stop/stream/
    // logprobs) and the GuardedSink/write_stream_* SSE helpers are fully generic and reused as-is
    // from /v1/chat/completions above; ThinkingStreamSplitter/tool_protocol/json_mode_active have
    // no equivalent here and are deliberately not dragged in -- there is exactly ONE per-branch
    // shape, not a dispatcher between two.
    svr.Post("/v1/completions",
             [&engine](const httplib::Request& req, httplib::Response& res) {
                 if (!auth_ok(req)) {
                     res.status = 401;
                     res.set_content("{\"error\":{\"message\":\"unauthorized\"}}", "application/json");
                     return;
                 }
                 if (g_shutdown_requested.load()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"server is shutting down\"}}",
                                     "application/json");
                     return;
                 }
                 if (!engine.loaded()) {
                     res.status = 503;
                     res.set_content("{\"error\":{\"message\":\"model not loaded\"}}", "application/json");
                     return;
                 }

                 g_requests_total++;
                 std::string prompt;
                 bool echo = false;
                 std::string err;
                 if (!sparkinfer_server::parse_legacy_completion_request(req.body, prompt, echo, err)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 sparkinfer_server::RequestControls controls;
                 if (!sparkinfer_server::parse_request_controls(req.body, controls, err, engine.vocab(),
                                                                 /*legacy_logprobs=*/true)) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(err) + "\"}}",
                                     "application/json");
                     return;
                 }
                 if (!controls.seed_set) {
                     static thread_local std::random_device rd;
                     controls.seed = ((uint64_t)rd() << 32) | rd();
                 }
                 const bool stream = controls.stream;
                 if (stream) g_requests_streaming++;
                 int max_tokens = controls.max_tokens;
                 if (max_tokens <= 0) max_tokens = 256;
                 if (max_tokens > max_output_tokens()) max_tokens = max_output_tokens();

                 const std::vector<int> prompt_ids = g_tokenizer.encode_raw(prompt);
                 if (prompt_ids.empty()) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content("{\"error\":{\"message\":\"tokenize returned no ids\"}}",
                                     "application/json");
                     return;
                 }
                 if ((int)prompt_ids.size() + max_tokens > engine.max_seq()) {
                     g_requests_client_error++;
                     res.status = 400;
                     res.set_content(
                         "{\"error\":{\"message\":\"context overflow: prompt=" +
                         std::to_string(prompt_ids.size()) + " max_tokens=" + std::to_string(max_tokens) +
                         " exceeds server ctx=" + std::to_string(engine.max_seq()) + "\"}}",
                         "application/json");
                     return;
                 }

                 const std::string cid = random_id("cmpl-");
                 const auto created = (long long)std::chrono::duration_cast<std::chrono::seconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();
                 auto status_for_outcome = [](const sparkinfer_server::CompletionResult& o) -> int {
                     if (o.overloaded)   return 429;
                     if (o.alloc_failed) return 503;
                     if (o.timed_out)    return 504;
                     return 400;
                 };

                 if (stream) {
                     res.set_header("Cache-Control", "no-cache");
                     res.set_header("X-Accel-Buffering", "no");
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [&engine, prompt_ids, prompt, echo, max_tokens, cid, created,
                          include_usage = controls.include_usage || always_stream_usage(), stop = controls.stop,
                          temperature = controls.temperature, seed = controls.seed,
                          top_k = controls.top_k, top_p = controls.top_p,
                          presence_penalty = controls.presence_penalty,
                          frequency_penalty = controls.frequency_penalty,
                          logit_bias = controls.logit_bias,
                          logprobs = controls.logprobs, top_logprobs = controls.top_logprobs,
                          n = controls.n]
                         (size_t offset, httplib::DataSink& sink) {
                             if (offset > 0) {
                                 sink.done();
                                 return true;
                             }

                             std::mutex sink_mu;
                             GuardedSink gs{sink, sink_mu};
                             SseHeartbeat heartbeat(gs);

                             struct BranchOutcome {
                                 bool ok = false;
                                 enum class Fail { none, cancelled, overloaded, alloc_failed, timeout,
                                                    server_error } fail = Fail::none;
                                 std::string fail_message;
                                 long long prompt_tokens = 0, completion_tokens = 0;
                                 double ttft_ms = -1.0, generation_ms = -1.0, decode_tps = -1.0;
                             };

                             // Never deduped at temperature<=0, unlike the non-streaming path
                             // below -- this emits incrementally, live, as tokens generate;
                             // deduping it would need recording and replaying the whole delta
                             // sequence (same carve-out as chat completions' plain streaming
                             // path), deferred to a follow-up.
                             // Wrapped in try/catch: for n>1 this body runs on a spawned
                             // std::thread (see the fan-out below), outside httplib's own
                             // exception handling -- an escaping exception would call
                             // std::terminate() and crash the whole server, not just fail this
                             // one request. Converting any exception into the existing
                             // server_error path keeps the blast radius scoped to one branch.
                             auto run_branch = [&](int ci, uint64_t branch_seed, BranchOutcome* out) {
                               try {
                                 sparkinfer_server::StopSequenceFilter stop_filter(stop);
                                 std::vector<int> stream_ids;
                                 stream_ids.reserve((size_t)max_tokens);
                                 bool stopped_by_sequence = false;
                                 size_t offset_so_far = echo ? prompt.size() : 0;
                                 std::vector<sparkinfer_server::TokenLogprob> pending_logprobs;
                                 auto on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                                     if (engine.is_stop_token(tl.token_id)) return;
                                     pending_logprobs.push_back(tl);
                                 };
                                 if (echo) write_stream_delta(gs, cid, created, "text", prompt, ci, nullptr,
                                                              "text_completion");
                                 auto on_tok = [&](int tid) -> bool {
                                     std::string piece = g_tokenizer.decode_delta(stream_ids, tid);
                                     std::string safe = stop_filter.feed(piece);
                                     if (stop_filter.matched()) {
                                         if (!safe.empty()) {
                                             nlohmann::json lp = nullptr;
                                             if (logprobs) {
                                                 lp = build_legacy_logprobs_json(pending_logprobs, top_logprobs,
                                                                                 offset_so_far);
                                                 offset_so_far += safe.size();
                                                 pending_logprobs.clear();
                                             }
                                             write_stream_delta(gs, cid, created, "text", safe, ci, lp,
                                                                "text_completion");
                                         }
                                         stopped_by_sequence = true;
                                         return false;
                                     }
                                     nlohmann::json lp = nullptr;
                                     if (logprobs && !safe.empty()) {
                                         lp = build_legacy_logprobs_json(pending_logprobs, top_logprobs, offset_so_far);
                                         offset_so_far += safe.size();
                                         pending_logprobs.clear();
                                     }
                                     bool ok = write_stream_delta(gs, cid, created, "text", safe, ci, lp,
                                                                  "text_completion");
                                     return ok && sink.is_writable();
                                 };
                                 const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_on_tok_logprob =
                                     logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(on_tok_logprob)
                                              : nullptr;
                                 const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, on_tok,
                                     temperature, branch_seed, top_k, top_p, presence_penalty, frequency_penalty,
                                     logit_bias, logprobs, top_logprobs, maybe_on_tok_logprob);
                                 out->prompt_tokens = (long long)prompt_ids.size();
                                 out->completion_tokens = (long long)stream_ids.size();
                                 if (outcome.cancelled && !stopped_by_sequence) {
                                     out->fail = BranchOutcome::Fail::cancelled;
                                     return;
                                 }
                                 if (!outcome.error.empty()) {
                                     out->fail = outcome.overloaded ? BranchOutcome::Fail::overloaded
                                               : outcome.alloc_failed ? BranchOutcome::Fail::alloc_failed
                                               : outcome.timed_out ? BranchOutcome::Fail::timeout
                                                                    : BranchOutcome::Fail::server_error;
                                     out->fail_message = outcome.error;
                                     return;
                                 }
                                 // Entries whose tokens decoded to no text of their own (a
                                 // stop-sequence holdback, a partial UTF-8 sequence) are still
                                 // pending here and would otherwise be dropped, leaving the
                                 // concatenated stream short of usage.completion_tokens -- the
                                 // same defect the chat path had at its tail flush.
                                 if (logprobs && !pending_logprobs.empty()) {
                                     write_stream_logprobs_only(
                                         gs, cid, created, ci,
                                         build_legacy_logprobs_json(pending_logprobs, top_logprobs,
                                                                    offset_so_far),
                                         "text", "text_completion");
                                     pending_logprobs.clear();
                                 }
                                 out->ttft_ms = outcome.ttft_ms;
                                 out->generation_ms = outcome.generation_ms;
                                 out->decode_tps = outcome.decode_tps;
                                 out->ok = true;
                                 write_stream_finish(gs, cid, created, ci,
                                                     outcome.reached_token_limit ? "length" : "stop",
                                                     "text_completion");
                               } catch (const std::exception& e) {
                                 // ok may already be true if the throw happened inside the final
                                 // write_stream_finish call -- force it back to false so the
                                 // post-join aggregation's `!results[ci].ok` scan catches this.
                                 out->ok = false;
                                 out->fail = BranchOutcome::Fail::server_error;
                                 out->fail_message = e.what();
                               } catch (...) {
                                 out->ok = false;
                                 out->fail = BranchOutcome::Fail::server_error;
                                 out->fail_message = "unknown exception";
                               }
                             };

                             std::vector<BranchOutcome> results(n);
                             std::vector<uint64_t> branch_seeds(n);
                             for (int i = 0; i < n; i++) branch_seeds[i] = seed + (uint64_t)i;

                             if (n == 1) {
                                 run_branch(0, branch_seeds[0], &results[0]);
                             } else {
                                 std::vector<std::thread> threads;
                                 threads.reserve(n);
                                 for (int ci = 0; ci < n; ci++)
                                     threads.emplace_back(run_branch, ci, branch_seeds[ci], &results[ci]);
                                 for (auto& t : threads) t.join();
                             }

                             int first_fail = -1;
                             for (int ci = 0; ci < n; ci++) {
                                 if (!results[ci].ok) { first_fail = ci; break; }
                             }
                             long long agg_prompt = 0, agg_completion = 0;
                             for (const auto& r : results) {
                                 agg_prompt += r.prompt_tokens;
                                 agg_completion += r.completion_tokens;
                             }
                             g_prompt_tokens_total += (uint64_t)agg_prompt;
                             g_completion_tokens_total += (uint64_t)agg_completion;

                             if (first_fail >= 0) {
                                 const auto& f = results[first_fail];
                                 switch (f.fail) {
                                     case BranchOutcome::Fail::cancelled:    g_requests_cancelled++; break;
                                     case BranchOutcome::Fail::overloaded:   g_requests_overloaded++; break;
                                     case BranchOutcome::Fail::alloc_failed: g_requests_alloc_failed++; break;
                                     case BranchOutcome::Fail::timeout:      g_requests_timeout++; break;
                                     default:                                g_requests_server_error++; break;
                                 }
                                 if (f.fail == BranchOutcome::Fail::cancelled) {
                                     heartbeat.stop();
                                     sink.done();
                                     return true;
                                 }
                                 write_sse_json(gs, {{"error", {{"message", f.fail_message}}}});
                                 if (include_usage)
                                     write_stream_usage(gs, cid, created, (int)results[0].prompt_tokens,
                                                        (int)agg_completion, -1.0, -1.0, -1.0,
                                                        "text_completion");
                                 heartbeat.stop();
                                 write_stream_done(gs);
                                 sink.done();
                                 return true;
                             }

                             g_requests_ok++;
                             double ttft_min = -1.0, gen_max = -1.0;
                             for (const auto& r : results) {
                                 if (r.ttft_ms >= 0.0 && (ttft_min < 0.0 || r.ttft_ms < ttft_min)) ttft_min = r.ttft_ms;
                                 if (r.generation_ms >= 0.0 && r.generation_ms > gen_max) gen_max = r.generation_ms;
                             }
                             const double decode_tps_agg =
                                 gen_max > 0.0 ? (double)agg_completion / (gen_max / 1000.0) : -1.0;
                             if (include_usage)
                                 write_stream_usage(gs, cid, created, (int)results[0].prompt_tokens,
                                                    (int)agg_completion, ttft_min, gen_max, decode_tps_agg,
                                                    "text_completion");
                             heartbeat.stop();
                             write_stream_done(gs);
                             sink.done();
                             return true;
                         });
                     return;
                 }

                 // Non-streaming: unlike the streaming path above, every sub-case already fully
                 // buffers its output before this function returns, so a duplicate choice can
                 // just reuse branch 0's already-built result at temperature<=0 instead of
                 // re-running generation -- no incrementally-emitted carve-out needed here.
                 struct NonStreamLegacyResult {
                     bool ok = false;
                     enum class Fail { none, overloaded, alloc_failed, timeout, server_error } fail = Fail::none;
                     int http_status = 200;
                     std::string http_error;
                     std::string text;
                     std::string finish_reason;
                     nlohmann::json logprobs_json = nullptr;
                     long long prompt_tokens = 0, completion_tokens = 0;
                     double ttft_ms = -1.0, generation_ms = -1.0, decode_tps = -1.0;
                 };

                 // Wrapped in try/catch: for n>1 this body runs on a spawned std::thread (see
                 // the fan-out below), outside httplib's own exception handling -- an escaping
                 // exception would call std::terminate() and crash the whole server, not just
                 // fail this one request. Converting any exception into the existing
                 // server_error path keeps the blast radius scoped to one branch.
                 auto run_nonstream_branch = [&](uint64_t branch_seed) -> NonStreamLegacyResult {
                   try {
                     NonStreamLegacyResult out;
                     std::vector<int> ids;
                     std::string stop_text;
                     bool stopped_by_sequence = false;
                     std::vector<sparkinfer_server::TokenLogprob> logprob_entries;
                     auto on_tok_logprob = [&](const sparkinfer_server::TokenLogprob& tl) {
                         if (engine.is_stop_token(tl.token_id)) return;
                         logprob_entries.push_back(tl);
                     };
                     auto on_tok = [&](int tid) -> bool {
                         if (controls.stop.empty()) {
                             ids.push_back(tid);
                             return true;
                         }
                         stop_text += g_tokenizer.decode_delta(ids, tid);
                         size_t pos;
                         if (find_stop_match(stop_text, controls.stop, pos)) {
                             stopped_by_sequence = true;
                             if (controls.logprobs && !logprob_entries.empty()) logprob_entries.pop_back();
                             return false;
                         }
                         return true;
                     };
                     const std::function<void(const sparkinfer_server::TokenLogprob&)> maybe_on_tok_logprob =
                         controls.logprobs ? std::function<void(const sparkinfer_server::TokenLogprob&)>(on_tok_logprob)
                                          : nullptr;
                     const auto outcome = engine.complete_streaming(prompt_ids, max_tokens, on_tok,
                         controls.temperature, branch_seed, controls.top_k, controls.top_p,
                         controls.presence_penalty, controls.frequency_penalty, controls.logit_bias,
                         controls.logprobs, controls.top_logprobs, maybe_on_tok_logprob);
                     if (logprob_entries.size() > outcome.tokens.size())
                         logprob_entries.resize(outcome.tokens.size());
                     out.prompt_tokens = (long long)prompt_ids.size();
                     out.completion_tokens = (long long)outcome.tokens.size();
                     if (!outcome.error.empty()) {
                         out.http_status = status_for_outcome(outcome);
                         out.fail = outcome.overloaded ? NonStreamLegacyResult::Fail::overloaded
                                  : outcome.alloc_failed ? NonStreamLegacyResult::Fail::alloc_failed
                                  : outcome.timed_out ? NonStreamLegacyResult::Fail::timeout
                                                       : NonStreamLegacyResult::Fail::server_error;
                         out.http_error = outcome.error;
                         return out;
                     }
                     std::string text;
                     std::string decode_err;
                     if (!decode_ids(outcome.tokens, text, decode_err)) {
                         out.http_status = 500;
                         out.fail = NonStreamLegacyResult::Fail::server_error;
                         out.http_error = decode_err;
                         return out;
                     }
                     if (stopped_by_sequence) {
                         size_t pos;
                         if (find_stop_match(text, controls.stop, pos)) text.resize(pos);
                     }
                     out.text = echo ? (prompt + text) : text;
                     out.finish_reason = outcome.reached_token_limit ? "length" : "stop";
                     out.logprobs_json = controls.logprobs
                         ? build_legacy_logprobs_json(logprob_entries, controls.top_logprobs,
                                                      echo ? prompt.size() : 0)
                         : nlohmann::json(nullptr);
                     out.ttft_ms = outcome.ttft_ms;
                     out.generation_ms = outcome.generation_ms;
                     out.decode_tps = outcome.decode_tps;
                     out.ok = true;
                     return out;
                   } catch (const std::exception& e) {
                     NonStreamLegacyResult out;
                     out.http_status = 500;
                     out.fail = NonStreamLegacyResult::Fail::server_error;
                     out.http_error = e.what();
                     return out;
                   } catch (...) {
                     NonStreamLegacyResult out;
                     out.http_status = 500;
                     out.fail = NonStreamLegacyResult::Fail::server_error;
                     out.http_error = "unknown exception";
                     return out;
                   }
                 };

                 std::vector<uint64_t> branch_seeds(controls.n);
                 for (int i = 0; i < controls.n; i++) branch_seeds[i] = controls.seed + (uint64_t)i;
                 std::vector<NonStreamLegacyResult> results(controls.n);
                 const bool can_dedup = controls.temperature <= 0.f;
                 if (controls.n == 1) {
                     results[0] = run_nonstream_branch(branch_seeds[0]);
                 } else if (can_dedup) {
                     results[0] = run_nonstream_branch(branch_seeds[0]);
                     if (results[0].ok) {
                         for (int i = 1; i < controls.n; i++) {
                             results[i] = results[0];
                             results[i].prompt_tokens = 0;
                             results[i].completion_tokens = 0;
                         }
                     }
                 } else {
                     std::vector<std::thread> threads;
                     threads.reserve(controls.n);
                     for (int i = 0; i < controls.n; i++)
                         threads.emplace_back([&, i] { results[i] = run_nonstream_branch(branch_seeds[i]); });
                     for (auto& t : threads) t.join();
                 }

                 int first_fail = -1;
                 for (int i = 0; i < controls.n; i++) {
                     if (!results[i].ok) { first_fail = i; break; }
                 }
                 long long agg_prompt = 0, agg_completion = 0;
                 for (const auto& r : results) {
                     agg_prompt += r.prompt_tokens;
                     agg_completion += r.completion_tokens;
                 }
                 g_prompt_tokens_total += (uint64_t)agg_prompt;
                 g_completion_tokens_total += (uint64_t)agg_completion;

                 if (first_fail >= 0) {
                     const auto& f = results[first_fail];
                     switch (f.fail) {
                         case NonStreamLegacyResult::Fail::overloaded:   g_requests_overloaded++; break;
                         case NonStreamLegacyResult::Fail::alloc_failed: g_requests_alloc_failed++; break;
                         case NonStreamLegacyResult::Fail::timeout:      g_requests_timeout++; break;
                         default:                                       g_requests_server_error++; break;
                     }
                     res.status = f.http_status;
                     res.set_content("{\"error\":{\"message\":\"" + json_escape(f.http_error) + "\"}}",
                                     "application/json");
                     return;
                 }

                 double ttft_min = -1.0, gen_max = -1.0;
                 for (const auto& r : results) {
                     if (r.ttft_ms >= 0.0 && (ttft_min < 0.0 || r.ttft_ms < ttft_min)) ttft_min = r.ttft_ms;
                     if (r.generation_ms >= 0.0 && r.generation_ms > gen_max) gen_max = r.generation_ms;
                 }
                 const double decode_tps_agg =
                     gen_max > 0.0 ? (double)agg_completion / (gen_max / 1000.0) : -1.0;

                 nlohmann::json usage = {
                     {"prompt_tokens", (int)results[0].prompt_tokens},
                     {"completion_tokens", (int)agg_completion},
                     {"total_tokens", (int)(results[0].prompt_tokens + agg_completion)}};
                 if (ttft_min >= 0.0) usage["ttft_ms"] = ttft_min;
                 if (gen_max >= 0.0) usage["generation_ms"] = gen_max;
                 if (decode_tps_agg >= 0.0) usage["decode_tps"] = decode_tps_agg;

                 nlohmann::json choices = nlohmann::json::array();
                 for (int i = 0; i < controls.n; i++) {
                     choices.push_back({{"text", results[i].text}, {"index", i},
                                        {"logprobs", results[i].logprobs_json},
                                        {"finish_reason", results[i].finish_reason}});
                 }
                 const nlohmann::json body = {
                     {"id", cid}, {"object", "text_completion"}, {"created", created},
                     {"model", g_model_name}, {"choices", choices}, {"usage", usage}};
                 g_requests_ok++;
                 res.set_content(body.dump(), "application/json");
             });

    // Transport-level deadlines. Defaults are generous, not aggressive: a cold 32k-context
    // prefill has been measured taking ~90s of TTFT alone (see eval/pr_dflash_bot.py's 32k
    // sweep), so a short default here would misfire on legitimate long-context requests.
    // The read timeout resets on each byte received, so a slow streaming response keeps
    // extending it as it goes -- this only fires on a genuinely stalled connection.
    const long read_timeout_s = getenv("SPARKINFER_READ_TIMEOUT_S") ? atol(getenv("SPARKINFER_READ_TIMEOUT_S")) : 300;
    const long write_timeout_s = getenv("SPARKINFER_WRITE_TIMEOUT_S") ? atol(getenv("SPARKINFER_WRITE_TIMEOUT_S")) : 300;
    svr.set_read_timeout(read_timeout_s, 0);
    svr.set_write_timeout(write_timeout_s, 0);

    std::signal(SIGTERM, on_shutdown_signal);
    std::signal(SIGINT, on_shutdown_signal);
    // Signal handlers must stay async-signal-safe (just set the atomic flag above); the actual
    // drain-and-stop happens here, on an ordinary thread. /v1/chat/completions and /v1/capacity
    // already check g_shutdown_requested to refuse new work as soon as the flag flips, so the
    // gap between "signal received" and "svr.stop() called" (at most one poll interval) only
    // means a few new requests might land right before shutdown starts, not that the drain window
    // is unbounded.
    // Bind BEFORE the shutdown watcher exists. listen() returning early is ambiguous -- it means
    // either "we were asked to stop" or "we could never start" -- and the old code treated both as
    // a shutdown: a port clash printed "shutdown signal received, draining in-flight requests...",
    // sat through the full 30s grace period, and only then said what had actually gone wrong.
    // Under a supervisor that restart-loops, that reads as a server that starts and mysteriously
    // stops. Splitting the bind out reports the real cause immediately, on the first line.
    if (!svr.bind_to_port(host.c_str(), port)) {
        fprintf(stderr,
                "[sparkinfer-server] FATAL: cannot bind %s:%d (in use, or not permitted).\n"
                "  Another process is probably already listening there -- check with:\n"
                "    ss -tlnp | grep :%d\n"
                "  Then free it or start with a different --port / PORT=<n>.\n",
                host.c_str(), port, port);
        return 1;
    }

    std::thread shutdown_watcher([&svr] {
        while (!g_shutdown_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        fprintf(stderr, "[sparkinfer-server] shutdown signal received, draining in-flight requests...\n");
        svr.stop();
        // svr.stop() only closes the LISTENING socket -- it does not force-close connections
        // already accepted. A client that vanishes without a clean TCP close (RST, dead network,
        // a hard `kill` on a curl process) can leave an httplib worker thread blocked in a read()
        // for up to the read timeout (SPARKINFER_READ_TIMEOUT_S, default 300s) -- measured: this
        // alone can make listen() take minutes to return, defeating the point of a "graceful"
        // shutdown. Bound the total drain time instead: same SIGTERM -> grace period -> force-kill
        // shape as Kubernetes' terminationGracePeriodSeconds.
        const long grace_s = getenv("SPARKINFER_SHUTDOWN_GRACE_S")
                                 ? atol(getenv("SPARKINFER_SHUTDOWN_GRACE_S")) : 30;
        std::this_thread::sleep_for(std::chrono::seconds(grace_s));
        fprintf(stderr, "[sparkinfer-server] shutdown grace period (%lds) elapsed with requests "
                        "still draining -- forcing exit\n", grace_s);
        _exit(0);  // not exit(): other threads may still be mid-flight; skip atexit/static dtors
    });

    const std::string queue_depth_label =
        engine.max_queue_depth() > 0 ? std::to_string(engine.max_queue_depth()) : std::string("unlimited");
    fprintf(stderr,
            "[sparkinfer-server] OpenAI-compatible API on http://%s:%d\n"
            "  GET  /health\n"
            "  GET  /v1/models\n"
            "  GET  /v1/info\n"
            "  GET  /v1/capacity\n"
            "  GET  /metrics\n"
            "  POST /v1/tokenize\n"
            "  POST /v1/chat/completions\n"
            "  POST /v1/completions\n"
            "  POST /v1/score\n"
            "  read_timeout=%lds write_timeout=%lds max_output_tokens=%d max_queue_depth=%s%s\n",
            host.c_str(), port, read_timeout_s, write_timeout_s, max_output_tokens(),
            queue_depth_label.c_str(),
            sparkinfer::deterministic_mode() ? "  DETERMINISTIC=1 (bit-reproducible)" : "");

    svr.listen_after_bind();   // bind already succeeded above; this only returns on stop()
    g_shutdown_requested = true;  // unblock the watcher thread if listen() returned on its own
    shutdown_watcher.join();
    return 0;
}
