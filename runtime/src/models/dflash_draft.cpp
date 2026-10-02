// DFlash draft runtime: safetensors load + GGUF load + block-parallel forward.
#include "sparkinfer/models/dflash_draft.h"
#include "vmm_arena.h"

#include <cstddef>
#include "sparkinfer/device_health.h"
#include <atomic>
#include "sparkinfer/models/dflash_kernels.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/gguf.h"
// Header-only Muse Glimmer DFlash draft config derivation (mirrors examples/qwen3_gguf_config.h's
// museglimmer_config_from_gguf for the target model). Lives in examples/ by this codebase's
// convention for GGUF-config-from-metadata helpers; reachable here via the runtime library's
// PRIVATE "examples" include dir (see runtime/CMakeLists.txt).
#include "dflash_gguf_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {
namespace {

using bf16 = __nv_bfloat16;

// Every CUDA call cu() has seen fail, over the life of the process. load() and load_gguf() compare
// it across the load: a draft whose weights or scratch did not fit has NOT loaded, however much of
// it did. Without this the loader logged "malloc: out of memory" (and then a lost context), still
// returned true, and the server started with speculative decoding advertised and every request
// failing (#1086).
std::atomic<int> g_cu_errors{0};

inline void cu(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    g_cu_errors.fetch_add(1, std::memory_order_relaxed);
    // See device_health.h: sticky errors kill the context, so record them and let the
    // engine refuse work rather than issuing more against a dead device.
    const bool fatal = note_cuda_error(e);
    static std::atomic<int> logged{0};
    const int n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n < 20 || fatal)
        fprintf(stderr, "[dflash] %s: %s%s\n", what, cudaGetErrorString(e),
                fatal ? "  [CONTEXT LOST -- server will refuse further work]" : "");
    else if (n == 20)
        fprintf(stderr, "[dflash] (further CUDA errors suppressed)\n");
}

struct TensorView {
    void* data = nullptr;
    size_t nbytes = 0;
    std::vector<int64_t> shape;
    std::string dtype;   // "BF16" | "F32" | "U8" (packed NVFP4) | "F8_E4M3" (group scales)
};

// Minimal safetensors reader (BF16 / F32 / NVFP4 payload). Owns a host mmap-like buffer.
struct SafeTensorsFile {
    std::vector<char> bytes;
    std::unordered_map<std::string, TensorView> tensors; // pointers into bytes

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(0, std::ios::end);
        const std::streamoff sz = f.tellg();
        if (sz < 8) return false;
        f.seekg(0, std::ios::beg);
        bytes.resize((size_t)sz);
        f.read(bytes.data(), sz);
        if (!f) return false;
        uint64_t hdr_len = 0;
        memcpy(&hdr_len, bytes.data(), 8);
        if (8 + hdr_len > (uint64_t)sz) return false;
        const std::string hdr(bytes.data() + 8, bytes.data() + 8 + hdr_len);
        // Parse each "name":{...} entry for dtype/shape/data_offsets.
        size_t pos = 0;
        while (pos < hdr.size()) {
            size_t key_start = hdr.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = hdr.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = hdr.substr(key_start + 1, key_end - key_start - 1);
            // Tensor entries are always "name": { ... }
            size_t colon = hdr.find(':', key_end);
            if (colon == std::string::npos) break;
            size_t obj = hdr.find('{', colon);
            if (obj == std::string::npos || obj > colon + 4) {
                // Not a tensor object (e.g. string metadata field) — advance past this key.
                pos = key_end + 1;
                continue;
            }
            if (key == "__metadata__") {
                // Skip nested object.
                int depth = 0;
                size_t i = obj;
                for (; i < hdr.size(); i++) {
                    if (hdr[i] == '{') depth++;
                    else if (hdr[i] == '}') {
                        depth--;
                        if (depth == 0) { i++; break; }
                    }
                }
                pos = i;
                continue;
            }
            size_t obj_end = hdr.find('}', obj);
            if (obj_end == std::string::npos) break;
            std::string body = hdr.substr(obj, obj_end - obj + 1);
            // dtype
            std::string dtype;
            size_t d0 = body.find("\"dtype\"");
            if (d0 != std::string::npos) {
                size_t c = body.find(':', d0);
                size_t q0 = body.find('"', c);
                size_t q1 = body.find('"', q0 + 1);
                if (q0 != std::string::npos && q1 != std::string::npos)
                    dtype = body.substr(q0 + 1, q1 - q0 - 1);
            }
            // data_offsets: [start, end]
            int64_t off0 = 0, off1 = 0;
            size_t o0 = body.find("\"data_offsets\"");
            if (o0 != std::string::npos) {
                size_t b0 = body.find('[', o0);
                off0 = strtoll(body.c_str() + b0 + 1, nullptr, 10);
                size_t comma = body.find(',', b0);
                off1 = strtoll(body.c_str() + comma + 1, nullptr, 10);
            }
            std::vector<int64_t> shape;
            size_t s0 = body.find("\"shape\"");
            if (s0 != std::string::npos) {
                size_t b0 = body.find('[', s0);
                size_t b1 = body.find(']', b0);
                std::string ss = body.substr(b0 + 1, b1 - b0 - 1);
                size_t p = 0;
                while (p < ss.size()) {
                    while (p < ss.size() && (ss[p] == ' ' || ss[p] == ',')) p++;
                    if (p >= ss.size()) break;
                    shape.push_back(strtoll(ss.c_str() + p, nullptr, 10));
                    while (p < ss.size() && ss[p] != ',') p++;
                }
            }
            // U8 = NVFP4-packed weight (2x E2M1 per byte), F8_E4M3 = its per-group scale.
            // Both are payload for the NVFP4 dequant below, not tensors to skip.
            if (dtype != "BF16" && dtype != "F32" && dtype != "BOOL" &&
                dtype != "U8" && dtype != "F8_E4M3") {
                fprintf(stderr, "[dflash] skip tensor %s dtype=%s\n", key.c_str(), dtype.c_str());
                pos = obj_end + 1;
                continue;
            }
            TensorView tv;
            tv.dtype = dtype;
            tv.shape = shape;
            tv.nbytes = (size_t)(off1 - off0);
            tv.data = bytes.data() + 8 + hdr_len + off0;
            tensors[key] = tv;
            pos = obj_end + 1;
        }
        if (tensors.empty())
            fprintf(stderr, "[dflash] safetensors parse produced 0 tensors (hdr_len=%llu)\n",
                    (unsigned long long)hdr_len);
        return !tensors.empty();
    }
};

bool parse_config_json(const std::string& path, DFlashDraftConfig& cfg) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string j = ss.str();
    auto find_int = [&](const char* key, int& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = (int)strtol(j.c_str() + c + 1, nullptr, 10);
    };
    auto find_float = [&](const char* key, float& dst) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = j.find(pat);
        if (p == std::string::npos) return;
        size_t c = j.find(':', p);
        dst = strtof(j.c_str() + c + 1, nullptr);
    };
    find_int("hidden_size", cfg.hidden);
    find_int("intermediate_size", cfg.intermediate);
    find_int("num_hidden_layers", cfg.n_layers);
    find_int("num_attention_heads", cfg.n_q_heads);
    find_int("num_key_value_heads", cfg.n_kv_heads);
    find_int("head_dim", cfg.head_dim);
    find_int("vocab_size", cfg.vocab);
    find_int("sliding_window", cfg.sliding_window);
    find_float("rms_norm_eps", cfg.rms_eps);
    find_int("block_size", cfg.block_size);
    // YaRN lives under "rope_parameters". Only applied when rope_type is actually yarn -- a
    // checkpoint carrying factor but rope_type "linear"/"default" must not silently get YaRN.
    if (j.find("\"yarn\"") != std::string::npos) {
        find_float("factor", cfg.yarn_factor);
        find_int("original_max_position_embeddings", cfg.yarn_orig_max_pos);
        find_float("beta_fast", cfg.yarn_beta_fast);
        find_float("beta_slow", cfg.yarn_beta_slow);
    }
    find_int("mask_token_id", cfg.mask_token_id);
    find_int("conv_kernel_size", cfg.conv_kernel);
    find_int("conv_group_size", cfg.conv_group);
    find_int("selector_rank", cfg.selector_rank);
    find_int("selector_top_k", cfg.selector_top_k);
    {
        const size_t ic = j.find("\"is_causal\"");
        if (ic != std::string::npos) {
            const size_t c = j.find(':', ic);
            if (c != std::string::npos) {
                size_t v = j.find_first_not_of(" \t\n\r", c + 1);
                if (v != std::string::npos) cfg.is_causal = j.compare(v, 4, "true") == 0 ? 1 : 0;
            }
        }
    }
    cfg.dflash2 = j.find("\"DFlash2DraftModel\"") != std::string::npos ||
                  (cfg.selector_rank > 0 && cfg.conv_kernel > 0);

    // DSpark (RadixArk/Qwen3.8-27B-DSpark) nests the draft-specific settings under
    // "dflash_config", and its target_layer_ids differ from this struct's Qwen3.6 default both in
    // VALUE and in COUNT: [4,16,28,40,52] (5 captures) vs {1,6,11,16,22,27,32,37} (8). The count
    // is load-bearing -- fc.weight is [hidden, n_cap*hidden], so leaving the default in place
    // makes the loader demand a [5120, 40960] projector against the checkpoint's [5120, 25600]
    // and fail outright. Parse the list rather than inheriting it.
    {
        size_t p = j.find("\"target_layer_ids\"");
        if (p != std::string::npos) {
            size_t a = j.find('[', p), b = j.find(']', a);
            if (a != std::string::npos && b != std::string::npos) {
                std::vector<int> ids;
                const char* q = j.c_str() + a + 1;
                const char* end = j.c_str() + b;
                while (q < end) {
                    char* nx = nullptr;
                    long v = strtol(q, &nx, 10);
                    if (nx == q) { q++; continue; }
                    ids.push_back((int)v);
                    q = nx;
                }
                if (!ids.empty()) cfg.target_layer_ids = ids;
            }
        }
    }
    // rope_theta nested
    size_t rp = j.find("\"rope_theta\"");
    if (rp != std::string::npos) {
        size_t c = j.find(':', rp);
        cfg.rope_theta = strtof(j.c_str() + c + 1, nullptr);
    }
    // dflash_config.block_size / mask_token_id / target_layer_ids
    size_t df = j.find("\"dflash_config\"");
    if (df != std::string::npos) {
        size_t bs = j.find("\"block_size\"", df);
        if (bs != std::string::npos) {
            size_t c = j.find(':', bs);
            cfg.block_size = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t mt = j.find("\"mask_token_id\"", df);
        if (mt != std::string::npos) {
            size_t c = j.find(':', mt);
            cfg.mask_token_id = (int)strtol(j.c_str() + c + 1, nullptr, 10);
        }
        size_t tl = j.find("\"target_layer_ids\"", df);
        if (tl != std::string::npos) {
            size_t b0 = j.find('[', tl);
            size_t b1 = j.find(']', b0);
            cfg.target_layer_ids.clear();
            size_t p = b0 + 1;
            while (p < b1) {
                while (p < b1 && (j[p] == ' ' || j[p] == ',')) p++;
                if (p >= b1) break;
                cfg.target_layer_ids.push_back((int)strtol(j.c_str() + p, nullptr, 10));
                while (p < b1 && j[p] != ',') p++;
            }
        }
    }
    // layer_types
    size_t lt = j.find("\"layer_types\"");
    if (lt != std::string::npos) {
        size_t b0 = j.find('[', lt);
        size_t b1 = j.find(']', b0);
        cfg.sliding_layers.assign(cfg.n_layers, true);
        int idx = 0;
        size_t p = b0;
        while (p < b1 && idx < cfg.n_layers) {
            size_t q0 = j.find('"', p);
            if (q0 == std::string::npos || q0 >= b1) break;
            size_t q1 = j.find('"', q0 + 1);
            std::string t = j.substr(q0 + 1, q1 - q0 - 1);
            cfg.sliding_layers[idx++] = (t == "sliding_attention");
            p = q1 + 1;
        }
    } else {
        cfg.sliding_layers.assign(cfg.n_layers, true);
        if (cfg.n_layers > 0) cfg.sliding_layers[cfg.n_layers - 1] = false;
    }
    return true;
}

struct Q8W { signed char* q = nullptr; float* s = nullptr;
              unsigned char* q4 = nullptr; void* dm = nullptr; };

// SPARKINFER_DFLASH_DRAFT_VOCAB: how many leading vocab ids the draft scores (0 = all). See
// forward_block, where the choice is explained; the loader reads it too, to size what only those
// ids index.
int draft_vocab_env() {
    const char* e = getenv("SPARKINFER_DFLASH_DRAFT_VOCAB");
    return e ? atoi(e) : 65536;
}

struct LayerWeights {
    bf16 *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    // Q8_0 mirrors of the four batched projections (Q/K/V, O, gate/up, down).
    Q8W q8_wq, q8_wk, q8_wv, q8_wo, q8_gate, q8_up, q8_down;
    bf16 *q_norm = nullptr, *k_norm = nullptr;
    bf16 *input_norm = nullptr, *post_norm = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
    // DFlash2: base_kernel [2 sides][taps][H] and kernel_projection [2 * taps * H / group, H] of
    // the conv around attention and around the MLP.
    bf16 *attn_conv_base = nullptr, *attn_conv_proj = nullptr;
    bf16 *mlp_conv_base = nullptr, *mlp_conv_proj = nullptr;
    // DFlash2's MLP as block-scaled NVFP4 GEMM operands (in place of the Q4 copies; see
    // ensure_quant): data and the CUTLASS SFB scales.
    unsigned char *fp4_gate = nullptr, *fp4_gate_sf = nullptr;
    unsigned char *fp4_up = nullptr, *fp4_up_sf = nullptr;
    unsigned char *fp4_down = nullptr, *fp4_down_sf = nullptr;
};

// Draft projection weight format: 4 = asymmetric int4, 8 = Q8_0, 0 = bf16. Default 4.
inline int draft_w_bits() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_DFLASH_WBITS");
        v = e ? atoi(e) : 4;
        if (v != 0 && v != 4 && v != 8) v = 4;
    }
    return v;
}
inline bool q8_on() { return draft_w_bits() != 0; }

} // namespace

struct DFlashDraftModel::Impl {
    DFlashDraftConfig cfg;
    std::vector<LayerWeights> layers;
    bf16* fc = nullptr;           // [H, n_cap * H] as [out, in] for gemv
    // YaRN rotary table (null unless the checkpoint configures rope_type "yarn").
    float* d_yarn_inv_freq = nullptr;   // [head_dim/2]
    float  yarn_att_scale = 1.0f;
    bf16* hidden_norm = nullptr;
    bf16* final_norm = nullptr;
    // Quantized copy of `fc`. The projector is [hidden, n_cap*hidden] = [5120, 25600] on DSpark,
    // 262 MB of bf16 -- and it was the ONE draft matrix still read at full precision, once per
    // block, to project 1-5 context rows. Every other projection has had a Q4 copy since #661.
    Q8W q8_fc;
    std::vector<void*> owned;
    // Every buffer the draft keeps lives here when the CUDA virtual memory API is available, so
    // the whole draft can step off the device while nothing speculates (see offload()).
    VmmArena arena;
    bool arena_on = false;
    int pins = 0;   // speculation in progress: offload() refuses (see DFlashDraftModel::pin)
    void* dev_alloc(size_t bytes) {
        if (arena_on) return arena.alloc(bytes);
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        return p;
    }

    // Shared target pointers
    const void* embed = nullptr;
    const void* lm_head = nullptr;
    int lm_head_type = 0;
    int vocab = 0;
    int hidden = 0;
    bf16* lm_head_bf16 = nullptr;  // eager dequant+transpose cache for the batched LM-head GEMM
    signed char* lm_head_i8 = nullptr;
    float* lm_head_i8_scale = nullptr;
    unsigned char* lm_head_i4 = nullptr;
    float* lm_head_i4_scale = nullptr;

    // DSpark's Markov head (optional -- absent for plain DFlash drafts, e.g. Qwen3.6-35B-A3B's).
    // markov_w1 [vocab, markov_rank]: embedding table indexed by the previous token id.
    // markov_w2 [vocab, markov_rank]: projection back to vocab space, added to the base logits.
    // Shared vocab with the target (this draft has no embed_tokens of its own either), so no
    // verifier/draft vocab remapping is needed.
    bf16* markov_w1 = nullptr;
    bf16* markov_w2 = nullptr;
    // int8 copy of markov_w2 with per-32 scales, built once at load; the bf16 original is released
    // as soon as it exists. See launch_markov_bias_add_q8 for why this is about L2, not DRAM.
    signed char* markov_w2_q = nullptr;
    float* markov_w2_s = nullptr;
    int markov_rank = 0;

    // DSpark's confidence head (AcceptRatePredictor): predicts a per-position accept
    // probability from concat(hidden, markov_latent). Optional -- requires the Markov head
    // (confidence_head_with_markov=True for every DSpark checkpoint released so far).
    bf16* confidence_w = nullptr;   // [hidden + markov_rank]
    float confidence_bias = 0.f;    // scalar, read back to host once at load (cheap, load-time only)
    // [block_size+1][markov_rank]. It USED to be a single [markov_rank] scratch overwritten by
    // every step of the sequential Markov loop, which forced the confidence head to run inside
    // that loop -- four separate grid-of-ONE launches, 8.5 us each, for a dot product. With a
    // per-row stride the four become one grid-4 launch after the chain. The chain itself is
    // unaffected: nothing in it ever read the latent back.
    float* markov_latent = nullptr;

    // Scratch
    cudaStream_t stream{};
    bf16 *noise = nullptr;          // [B, H]
    bf16 *target_proj = nullptr;    // [ctx, H]
    bf16 *x = nullptr, *xn = nullptr, *h = nullptr, *hn = nullptr;
    bf16 *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr, *ao = nullptr;
    bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
    // Per-split online-softmax partials for the row-batched KV-split draft attention.
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    float* logits = nullptr;        // [B, vocab]
    void* head_q8 = nullptr;        // [B] Q8_1 rows of xn for the multi-row head MMVQ
    int *d_ids = nullptr, *d_out = nullptr;
    int *h_ids = nullptr;                        // PINNED staging for the block ids
    // Q8_1 staging for the dp4a backbone: one 36-byte block per 32 values per block row, sized for
    // the widest K any projection uses (the FFN's intermediate).
    void* xq81 = nullptr;

    // The first sequence position this generation's context starts at: the first block's
    // target_hidden_start (see forward_block). Nothing before it was ever projected into the KV
    // cache -- a prefix-cache hit prefilled only past it -- so no layer may attend there.
    int ctx_floor = 0;
    int *h_out = nullptr;
    float *d_confidence = nullptr, *h_confidence = nullptr;   // [B], confidence head output
    // DFlash2's candidate selector: hidden_projection [rank, H] and the two [vocab, rank]
    // codebooks, plus per-block scratch (the kernel projections' rows, the conv output, the
    // selector's projected rows and each slot's top-k ids and logits).
    bf16* sel_hproj = nullptr;
    bf16* sel_pred = nullptr;
    bf16* sel_succ = nullptr;
    bf16* conv_dyn_a = nullptr;   // [B][2 * taps * groups], the attention conv's kernels
    bf16* conv_dyn_m = nullptr;   // the MLP conv's
    bf16* conv_out = nullptr;     // [B][max(H, qdim)]
    bf16* sel_h = nullptr;        // [B][rank]
    int* sel_ids = nullptr;       // [B][top_k]
    float* sel_vals = nullptr;
    // set_sampling(): the request's sampler for the next forward_block, indexed by proposal
    // (entry 0, the seed row, is never sampled). Pinned so the per-block upload stays async.
    static constexpr int kSampRows = 64;
    struct DraftSampling {
        float temp[kSampRows];
        unsigned long long seed[kSampRows];
        unsigned long long step[kSampRows];
        int top_k[kSampRows];
        float top_p[kSampRows];
    };
    DraftSampling* h_samp = nullptr;
    DraftSampling* d_samp = nullptr;
    bool samp_on = false;
    float samp_temp = 0.f;
    unsigned long long samp_seed = 0, samp_step0 = 0;
    int samp_top_k = 0;
    float samp_top_p = 1.f;

    // Per-layer contiguous KV cache: [max_seq, n_kv, d]
    std::vector<bf16*> k_cache, v_cache;
    int seq_len = 0;
    // Slots (DFlashDraftModel::use_slot): the members above are the current slot's; the others
    // are parked here. Slot 0's buffers come from alloc_scratch (in `owned`); the rest are this
    // struct's own.
    // `cap` is the rows the slot's buffers hold: cfg.max_seq for slot 0 and what its request can
    // reach for the others (use_slot's `need`), both at most win_rows for a windowed draft, whose
    // slots slide. forward_block checks against the current one.
    // `base` is the absolute position of a windowed slot's row 0 (see win_rows): its caches hold
    // positions [base, base + cap). Always 0 for a slot that holds every position.
    struct Slot {
        std::vector<bf16*> k, v;
        bf16* tp = nullptr;
        int seq_len = 0, ctx_floor = 0, cap = 0, base = 0;
        bool live = false;
    };
    std::vector<Slot> slots;
    int cur_slot = 0;
    int cap = 0;
    int base = 0;
    // Window-sized slots. DFlash2 attends a sliding window on every layer, so no block reads a key
    // older than pos0 - sliding_window + 1, and positions only grow: a slot need hold only the
    // window, the block and room to slide, not all the context the request can reach. 0 = every
    // slot holds its whole reach (DSpark, whose layers attend everything).
    int win_rows = 0;
    int slot_rows(int need) const { return win_rows > 0 ? std::min(need, win_rows) : need; }
    // A windowed slot slides: when a block (ctx_len context rows then BW block rows, the context
    // starting at position `past`, the block at pos0) would run past its rows, every key older than
    // the window the block reads (pos0 - sliding_window + 1: the earliest query is pos0, and keys
    // only grow) is dropped, the rest moved to row 0 and `base` advanced. Keys are stored rotated at
    // their absolute positions, so moving them changes nothing; the move never overlaps (at most a
    // window kept, at least a window dropped: win_rows >= 2 * (window + block)). The first block of
    // a long prompt just starts at the window. Positions under `base` are then never produced, as
    // the floor's are not. Used on the current slot (forward_block) and on a parked one
    // (forward_blocks); false when the window starts past every row the block has.
    bool slide(std::vector<bf16*>& kc, std::vector<bf16*>& vc, int& b, int floor, int rows, int past,
               int ctx_len, int pos0, int BW, cudaStream_t st) {
        if (win_rows == 0 || past + ctx_len + BW - b <= rows) return true;
        const int nb = std::max(std::max(b, floor), pos0 - cfg.sliding_window + 1);
        if (nb > b) {
            const int keep = past - nb;
            if (keep > 0) {
                const int kvd = cfg.n_kv_heads * cfg.head_dim;
                const size_t bytes = (size_t)keep * kvd * sizeof(bf16);
                const size_t from = (size_t)(nb - b) * kvd;
                for (int L = 0; L < cfg.n_layers; ++L) {
                    cu(cudaMemcpyAsync(kc[L], kc[L] + from, bytes, cudaMemcpyDeviceToDevice, st), "slot slide k");
                    cu(cudaMemcpyAsync(vc[L], vc[L] + from, bytes, cudaMemcpyDeviceToDevice, st), "slot slide v");
                }
            }
            b = nb;
        }
        return b <= past + ctx_len;
    }
    void park_current() {
        Slot& sl = slots[(size_t)cur_slot];
        sl.k = k_cache; sl.v = v_cache; sl.tp = target_proj;
        sl.seq_len = seq_len; sl.ctx_floor = ctx_floor; sl.cap = cap; sl.base = base;
    }
    void load_slot(int i) {
        Slot& sl = slots[(size_t)i];
        k_cache = sl.k; v_cache = sl.v; target_proj = sl.tp;
        seq_len = sl.seq_len; ctx_floor = sl.ctx_floor; cap = sl.cap; base = sl.base;
        cur_slot = i;
    }
    void free_slot_buffers(Slot& sl) {
        for (bf16* b : sl.k) if (b) cudaFree(b);
        for (bf16* b : sl.v) if (b) cudaFree(b);
        if (sl.tp) cudaFree(sl.tp);
        sl = Slot{};
    }

    // The draft's quantized weight copies are built on first use, not at load. Constructing them
    // is what makes merely loading the draft tax the TARGET's decode: measured on RTX 5090 with
    // the same bench_decode call either side of draft.load(), 501.9 -> 460.1 tok/s at 512-ctx, and
    // skipping just this construction removes all of it (501.2). Freeing the draft afterwards
    // restores it too (501.4), so the cost tracks these buffers being resident rather than the
    // ~216 MB they occupy -- 2.5 GB of dummy allocations reproduce none of it.
    //
    // Deferring means a generation that never runs the draft never pays. dflash_generate primes
    // this before its decode clock starts, so a generation that does run the draft is unchanged.
    struct PendingQuant { bf16* w; int N, K; Q8W* dst; };
    std::vector<PendingQuant> pending_quant;
    bool quant_ready = false;

    // NVFP4 A operand and workspace for the MLP GEMMs (and forward_blocks' head), sized for
    // kBatchRows rows.
    static constexpr int kBatchRows = 64;       // forward_blocks: 8 slots of an 8-row block
    unsigned char *f4_a = nullptr, *f4_asf = nullptr, *f4_ws = nullptr;
    bool fp4_mlp = false;

    // forward_blocks' scratch, allocated on its first call: every row-indexed buffer of the
    // per-slot path at kBatchRows rows, the gathered context, and the per-job sampler/outputs.
    struct Batch {
        bool ready = false;
        bf16 *x = nullptr, *xn = nullptr, *h = nullptr, *hn = nullptr;
        bf16 *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr, *ao = nullptr;
        bf16 *gate = nullptr, *up = nullptr, *down = nullptr;
        bf16 *conv_a = nullptr, *conv_m = nullptr, *conv_out = nullptr;
        bf16 *ctx_h = nullptr, *tp = nullptr, *ctx_k = nullptr, *ctx_v = nullptr;
        bf16 *sel_h = nullptr;
        int* sel_ids = nullptr;
        float* sel_vals = nullptr;
        float* logits = nullptr;
        int *d_ids = nullptr, *h_ids = nullptr;
        int *d_out = nullptr, *h_out = nullptr;
        DraftSampling *d_samp = nullptr, *h_samp = nullptr;
        unsigned char* ws = nullptr;            // the GEMMs' workspace, head included
        // NVFP4 operands of the projections the per-slot path reads in Q4 or bf16: per layer the
        // block rows' stacked q|k|v, the context rows' k|v, wo and the two conv projections; the
        // projector fc and the selector's hidden projection.
        struct Op { unsigned char *d = nullptr, *sf = nullptr; };
        std::vector<Op> qkv, kv, wo, acp, mcp;
        Op fc, sel;
    } batch;

    void ensure_quant() {
        if (quant_ready) return;
        // DFlash2's MLP -- 80% of its weights -- runs as block-scaled NVFP4 GEMMs: the same bytes as
        // the Q4 copies it replaces (4.5 bits a weight), and the operand forward_blocks needs to read
        // them ONCE for several slots' rows. SPARKINFER_DFLASH2_MLP_FP4=0 keeps the Q4 GEMVs.
        static const bool fp4_env = [] {
            const char* e = getenv("SPARKINFER_DFLASH2_MLP_FP4");
            return !(e && e[0] == '0');
        }();
        const int H = cfg.hidden, I = cfg.intermediate;
        fp4_mlp = fp4_env && cfg.dflash2 && cfg.block_size % 8 == 0 &&
                  kernels::prefill_nvfp4_supported(8, I, H) &&
                  kernels::prefill_nvfp4_supported(8, H, I);
        for (const LayerWeights& lw : layers)
            fp4_mlp = fp4_mlp && lw.gate && lw.up && lw.down;
        auto is_mlp = [&](const Q8W* dst) {
            for (const LayerWeights& lw : layers)
                if (dst == &lw.q8_gate || dst == &lw.q8_up || dst == &lw.q8_down) return true;
            return false;
        };
        for (auto& pq : pending_quant)
            if (!(fp4_mlp && is_mlp(pq.dst))) *pq.dst = make_q8(pq.w, pq.N, pq.K);
        pending_quant.clear();
        if (fp4_mlp) {
            auto make_fp4 = [&](const bf16* w, int n, int k, unsigned char** d, unsigned char** sf) {
                *d = alloc<unsigned char>(kernels::prefill_nvfp4_data_bytes(n, k));
                *sf = alloc<unsigned char>(kernels::prefill_nvfp4_scale_bytes_b(n, k));
                return kernels::launch_prefill_nvfp4_quant_b(w, *d, *sf, n, k, stream);
            };
            for (LayerWeights& lw : layers)
                fp4_mlp = fp4_mlp && make_fp4(lw.gate, I, H, &lw.fp4_gate, &lw.fp4_gate_sf) &&
                          make_fp4(lw.up, I, H, &lw.fp4_up, &lw.fp4_up_sf) &&
                          make_fp4(lw.down, H, I, &lw.fp4_down, &lw.fp4_down_sf);
            const int kmax = std::max(H, I);
            size_t ws = std::max(kernels::prefill_nvfp4_workspace_bytes(kBatchRows, I, H),
                                 kernels::prefill_nvfp4_workspace_bytes(kBatchRows, H, I));
            f4_a = alloc<unsigned char>(kernels::prefill_nvfp4_data_bytes(kBatchRows, kmax));
            f4_asf = alloc<unsigned char>(kernels::prefill_nvfp4_scale_bytes_a(kBatchRows, kmax));
            if (ws) f4_ws = alloc<unsigned char>(ws);
            cudaStreamSynchronize(stream);
            if (!fp4_mlp) {
                // A failed build leaves no usable MLP: make the Q4 copies after all.
                for (LayerWeights& lw : layers) {
                    lw.q8_gate = make_q8(lw.gate, I, H);
                    lw.q8_up = make_q8(lw.up, I, H);
                    lw.q8_down = make_q8(lw.down, H, I);
                }
            }
        }
        quant_ready = true;
        // DFlash2 always runs its whole block (forward_block forces BW = block_size), and at a
        // batched width its MLP reads only the quantized copies; the bf16 originals are dead
        // weight -- 2.7 GB on z-lab's Qwen3.8 checkpoint (intermediate 17408), which at a 32K
        // prompt is what the target's batched prefill needs, or it falls back to a path both far
        // slower and not bit-identical to the one plain decode took. Other drafts keep them: their
        // width varies per generation and may land on the per-token fallback that reads them.
        const int bs = cfg.block_size;
        const bool batched_width = bs == 16 || bs == 8 || bs == 7 || bs == 6 || bs == 5 ||
                                   bs == 4 || bs == 2;
        if (cfg.dflash2 && batched_width) {
            for (LayerWeights& lw : layers) {
                if (!lw.fp4_down &&
                    (!(lw.q8_gate.q4 || lw.q8_gate.q) || !(lw.q8_up.q4 || lw.q8_up.q) ||
                     !(lw.q8_down.q4 || lw.q8_down.q)))
                    continue;
                release(lw.gate); release(lw.up); release(lw.down);
                lw.gate = lw.up = lw.down = nullptr;
            }
        }
    }

    Q8W make_q8(bf16* w, int N, int K) {
        Q8W o;
        if (draft_w_bits() == 4) {
            o.q4 = alloc<unsigned char>((size_t)N * (K / 2));
            o.dm = alloc<short>((size_t)N * (K / 32) * 2);   // __half2 per 32-weight block
            dflash_kernels::launch_quantize_w_q4(w, o.q4, o.dm, N, K, stream);
        } else {
            o.q = alloc<signed char>((size_t)N * K);
            o.s = alloc<float>((size_t)N * (K / 32));
            dflash_kernels::launch_quantize_w_q8(w, o.q, o.s, N, K, stream);
        }
        cudaStreamSynchronize(stream);
        return o;
    }

    // Give a buffer back and drop it from `owned`, so the destructor does not double-free it.
    void release(void* p) {
        if (!p) return;
        for (size_t i = 0; i < owned.size(); i++) {
            if (owned[i] == p) { owned.erase(owned.begin() + i); break; }
        }
        if (!(arena_on && arena.free(p))) cudaFree(p);
    }

    template <class T> T* alloc(size_t n) {
        void* p = dev_alloc(n * sizeof(T));
        if (!p) cu(cudaErrorMemoryAllocation, "malloc");
        else owned.push_back(p);
        return (T*)p;
    }

    // Scratch/KV-cache allocation shared by load() (safetensors) and load_gguf(): identical
    // buffers either way, sized purely from cfg (which is fully populated by the time either
    // caller reaches this). Factored out so the two load paths cannot drift on sizing.
    void alloc_scratch() {
        const int H = cfg.hidden;
        const int I = cfg.intermediate;
        const int B = cfg.block_size;
        const int max_ctx = cfg.max_seq;
        const int qdim = cfg.n_q_heads * cfg.head_dim;
        const int kvdim = cfg.n_kv_heads * cfg.head_dim;
        // SPARKINFER_DFLASH_SLOT_WINDOW: rows a windowed slot holds (default two windows and two
        // blocks: a compaction then copies at most a window, never onto itself); 0 keeps full slots.
        {
            bool all_sliding = cfg.dflash2 && cfg.sliding_window > 0 && !cfg.sliding_layers.empty();
            for (bool sl : cfg.sliding_layers) all_sliding = all_sliding && sl;
            const char* e = getenv("SPARKINFER_DFLASH_SLOT_WINDOW");
            // Never fewer than the default: a slide keeps up to a window and must drop at least as
            // much, or the one cudaMemcpyAsync that moves the kept rows would overlap itself.
            const int def = 2 * cfg.sliding_window + 2 * B;
            const int want = e ? atoi(e) : def;
            win_rows = (all_sliding && want > 0) ? std::max(want, def) : 0;
            if (win_rows >= max_ctx) win_rows = 0;
            if (win_rows > 0)
                fprintf(stderr, "[dflash] draft slots slide: %d rows of the %d-position context "
                                "(every layer attends a %d-token window)\n",
                        win_rows, max_ctx, cfg.sliding_window);
        }
        const int rows0 = slot_rows(max_ctx);
        noise = alloc<bf16>((size_t)B * H);
        target_proj = alloc<bf16>((size_t)rows0 * H);
        x = alloc<bf16>((size_t)B * H);
        xn = alloc<bf16>((size_t)B * H);
        h = alloc<bf16>((size_t)B * H);
        hn = alloc<bf16>((size_t)B * H);
        q = alloc<bf16>((size_t)B * qdim);
        attn = alloc<bf16>((size_t)B * qdim);
        ao = alloc<bf16>((size_t)B * H);
        {
            const size_t parts = (size_t)dflash_kernels::kDFlashAttnMaxRows * cfg.n_q_heads *
                                 dflash_kernels::kDFlashAttnMaxSplits;
            fa_m = alloc<float>(parts);
            fa_l = alloc<float>(parts);
            fa_acc = alloc<float>(parts * 128);
        }
        gate = alloc<bf16>((size_t)B * I);
        up = alloc<bf16>((size_t)B * I);
        down = alloc<bf16>((size_t)B * H);
        if (cfg.dflash2 && cfg.conv_kernel > 0 && cfg.conv_group > 0) {
            const int dyn = 2 * cfg.conv_kernel * (H / cfg.conv_group);
            conv_dyn_a = alloc<bf16>((size_t)B * dyn);
            conv_dyn_m = alloc<bf16>((size_t)B * dyn);
            conv_out = alloc<bf16>((size_t)B * std::max(H, qdim));
        }
        if (cfg.dflash2 && cfg.selector_rank > 0 && cfg.selector_top_k > 0) {
            sel_h = alloc<bf16>((size_t)B * cfg.selector_rank);
            sel_ids = alloc<int>((size_t)B * cfg.selector_top_k);
            sel_vals = alloc<float>((size_t)B * cfg.selector_top_k);
        }
        logits = alloc<float>((size_t)B * std::max(cfg.vocab, 1));
        head_q8 = alloc<char>((size_t)B * kernels::llama_q8_1_bytes(H));
        d_ids = alloc<int>(B);
        // Proposal-indexed, not row-indexed. Under the row-shift mapping the chain writes
        // d_out[r] / d_confidence[r] for r = 1..kProposalDepth, and a block_size-wide block
        // backs kProposalDepth == B -- so index B is live and these need B+1 slots. d_ids stays
        // at B: it holds the block's input ids, one per row.
        d_out = alloc<int>(B + 1);
        cu(cudaHostAlloc(&h_out, (B + 1) * sizeof(int), cudaHostAllocDefault), "h_out");
        d_samp = reinterpret_cast<DraftSampling*>(alloc<char>(sizeof(DraftSampling)));
        cu(cudaHostAlloc(&h_samp, sizeof(DraftSampling), cudaHostAllocDefault), "h_samp");
        // The block ids arrive in a caller-owned std::vector, i.e. PAGEABLE memory, and CUDA
        // performs a stream synchronize before a pageable H2D copy is initiated. Every other host
        // buffer on this path (h_out, h_confidence, and the verify's ph_ids/ph_pos/ph_seq) is
        // already pinned; this one was the exception, and it sits at the very first instruction of
        // the draft block -- immediately after the verify has left a 158 us GDN state commit in
        // flight on the target's stream, which shares no data with the draft and should overlap it.
        cu(cudaHostAlloc(&h_ids, (B + 1) * sizeof(int), cudaHostAllocDefault), "h_ids");
        {
            int kmax = (cfg.intermediate > cfg.hidden ? cfg.intermediate : cfg.hidden);
            const int kfc = (int)cfg.target_layer_ids.size() * cfg.hidden;   // the fc projector
            if (kfc > kmax) kmax = kfc;
            const size_t blocks = (size_t)(B + 1) * ((kmax + 31) / 32);
            xq81 = alloc<char>(blocks * 36);
        }
        if (confidence_w) {
            d_confidence = alloc<float>(B + 1);
            cu(cudaHostAlloc(&h_confidence, (B + 1) * sizeof(float), cudaHostAllocDefault),
               "h_confidence");
        }

        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        for (int L = 0; L < cfg.n_layers; L++) {
            k_cache[L] = alloc<bf16>((size_t)rows0 * kvdim);
            v_cache[L] = alloc<bf16>((size_t)rows0 * kvdim);
        }
        seq_len = 0;
        base = 0;
        cap = rows0;
        slots.assign(1, Slot{});
        slots[0].live = true;
        cur_slot = 0;
        park_current();
    }

    // NVFP4 -> BF16 at load, decoded on the host.
    //
    // Deliberately NOT launch_ct_dequant_nvfp4: that kernel reads the group scale as CUTLASS
    // *unsigned* e4m3, while ModelOpt exports signed float8_e4m3fn. The bit patterns differ, so
    // the kernel mis-scales every group -- which degrades draft acceptance without breaking
    // correctness, because the target verifies every drafted token anyway.
    //
    // Validated against the BF16 source of the same checkpoint:
    //   w = e2m1(nibble) * f8e4m3(group_scale) * weight_scale_2  ->  rel_err 0.0896, ratio 0.9927
    // which is pure 4-bit rounding error. This is a one-time load cost.
    static float decode_e2m1(unsigned char n) {
        static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        float v = mag[n & 0x7];
        return (n & 0x8) ? -v : v;
    }
    static float decode_e4m3(unsigned char b) {   // signed float8_e4m3fn
        const int s = (b >> 7) & 0x1;
        const int e = (b >> 3) & 0xF;
        const int m = b & 0x7;
        float v;
        if (e == 0) v = std::ldexp((float)m / 8.0f, -6);          // subnormal
        else        v = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
        return s ? -v : v;
    }

    bf16* upload_nvfp4(const TensorView& packed, const TensorView& scale, float global_scale) {
        const int rows = (int)packed.shape[0];
        const int cols = (int)packed.shape[1] * 2;
        const int groups = cols / 16;
        const unsigned char* pw = (const unsigned char*)packed.data;
        const unsigned char* ps = (const unsigned char*)scale.data;
        std::vector<bf16> host((size_t)rows * cols);
        for (int r = 0; r < rows; r++) {
            const unsigned char* prow = pw + (size_t)r * (cols / 2);
            const unsigned char* srow = ps + (size_t)r * groups;
            for (int c = 0; c < cols; c += 2) {
                const unsigned char byte = prow[c >> 1];
                const float gs = decode_e4m3(srow[c >> 4]) * global_scale;
                host[(size_t)r * cols + c]     = __float2bfloat16(decode_e2m1(byte & 0x0F) * gs);
                host[(size_t)r * cols + c + 1] = __float2bfloat16(decode_e2m1(byte >> 4) * gs);
            }
        }
        bf16* out = alloc<bf16>((size_t)rows * cols);
        cu(cudaMemcpy(out, host.data(), host.size() * sizeof(bf16), cudaMemcpyHostToDevice),
           "upload nvfp4");
        return out;
    }

    bf16* upload(const TensorView& tv) {
        bf16* d = alloc<bf16>(tv.nbytes / sizeof(bf16));
        if (tv.nbytes % sizeof(bf16) == 0) {
            cu(cudaMemcpy(d, tv.data, tv.nbytes, cudaMemcpyHostToDevice), "upload bf16");
        } else {
            // F32 -> BF16
            size_t n = tv.nbytes / sizeof(float);
            std::vector<bf16> tmp(n);
            const float* src = (const float*)tv.data;
            for (size_t i = 0; i < n; i++) tmp[i] = __float2bfloat16(src[i]);
            cu(cudaMemcpy(d, tmp.data(), n * sizeof(bf16), cudaMemcpyHostToDevice), "upload f32");
        }
        return d;
    }
};

DFlashDraftModel::DFlashDraftModel(const DFlashDraftConfig& cfg) : p_(new Impl()) {
    p_->cfg = cfg;
    if (p_->cfg.sliding_layers.empty()) {
        p_->cfg.sliding_layers.assign(p_->cfg.n_layers, true);
        if (p_->cfg.n_layers > 0) p_->cfg.sliding_layers.back() = false;
    }
    // Non-blocking, for the same reason as Qwen35Model's stream: see qwen35.cpp.
    cudaStreamCreateWithFlags(&p_->stream, cudaStreamNonBlocking);
    // SPARKINFER_DRAFT_OFFLOAD=0 keeps plain cudaMalloc (and so no offload).
    static const bool vmm_env = [] {
        const char* e = getenv("SPARKINFER_DRAFT_OFFLOAD");
        return !(e && e[0] == '0');
    }();
    int dev = 0;
    if (vmm_env && cudaGetDevice(&dev) == cudaSuccess)
        p_->arena_on = p_->arena.init(dev, size_t(64) << 30);   // address space only
}

DFlashDraftModel::~DFlashDraftModel() {
    if (!p_) return;
    for (size_t i = 1; i < p_->slots.size(); ++i)
        if (p_->slots[i].live) p_->free_slot_buffers(p_->slots[i]);
    if (!p_->arena_on)
        for (void* p : p_->owned) cudaFree(p);   // the arena releases its own
    if (p_->h_out) cudaFreeHost(p_->h_out);
    if (p_->h_ids) cudaFreeHost(p_->h_ids);
    if (p_->h_confidence) cudaFreeHost(p_->h_confidence);
    if (p_->h_samp) cudaFreeHost(p_->h_samp);
    if (p_->batch.h_ids) cudaFreeHost(p_->batch.h_ids);
    if (p_->batch.h_out) cudaFreeHost(p_->batch.h_out);
    if (p_->batch.h_samp) cudaFreeHost(p_->batch.h_samp);
    if (p_->stream) cudaStreamDestroy(p_->stream);
    delete p_;
    p_ = nullptr;
}

const DFlashDraftConfig& DFlashDraftModel::config() const { return p_->cfg; }

void DFlashDraftModel::set_shared_weights(const void* embed, const void* lm_head,
                                         int lm_head_type, int vocab, int hidden) {
    p_->embed = embed;
    p_->lm_head = lm_head;
    p_->lm_head_type = lm_head_type;
    p_->vocab = vocab;
    p_->hidden = hidden;
    // Eagerly dequantize the (static, resident) lm_head weight to bf16 exactly once, in its
    // native [vocab,hidden] ("out,in") layout -- dflash_kernels::launch_gemv_batched16_f32
    // reads W the same way a single-row GEMV does, so no relayout is needed. Lets
    // forward_block score a whole block with one batched GEMV instead of a per-token loop.
    // ONLY the BW==16 batched-head path (launch_gemv_batched16_f32 below) reads this bf16 copy;
    // every other block width already scores against the target's QUANTIZED head on-read, via
    // launch_gemv_q4k_dp4a_multirow_f32 or the per-row launch_gemv_q_f32 fallback. Materializing
    // it regardless costs vocab*hidden*2 bytes for nothing at any other block size -- 2.54 GB for
    // DSpark (V=248320, H=5120), which is precisely what pushed Qwen3.8-27B + DSpark past 32 GB
    // and made the draft OOM in its own lm_head dequant while the weights themselves fit.
    if (p_->cfg.block_size == 16 && !p_->lm_head_bf16 && vocab > 0 && hidden > 0) {
        p_->lm_head_bf16 = p_->alloc<bf16>((size_t)vocab * hidden);
        if (lm_head_type != 0) {
            kernels::launch_gguf_dequant(lm_head_type, lm_head, p_->lm_head_bf16,
                                         (long)vocab * hidden, p_->stream);
        } else {
            cu(cudaMemcpyAsync(p_->lm_head_bf16, lm_head, (size_t)vocab * hidden * sizeof(bf16),
                               cudaMemcpyDeviceToDevice, p_->stream), "lm_head copy");
        }
        cu(cudaStreamSynchronize(p_->stream), "lm_head dequant");
    }
    // The draft's head is the single largest read in the draft block, and the draft only has to
    // NOMINATE tokens -- every emitted token is still a target argmax, so the head's precision can
    // only move the accept length, never correctness. int4 halves those bytes again (~254 MB vs
    // ~508 MB at V=248k), and the kernel is at HBM peak, so its runtime is its weight bytes.
    // SPARKINFER_DFLASH_HEAD_I4=0 keeps the int8 head.
    if (!p_->lm_head_i8 && lm_head_type == 12 && vocab > 0 && hidden == 2048) {
        p_->lm_head_i8 = p_->alloc<signed char>((size_t)vocab * hidden);
        p_->lm_head_i8_scale = p_->alloc<float>(vocab);
        if (!kernels::launch_gguf_dequant_rows_i8(
                lm_head_type, lm_head, p_->lm_head_i8, p_->lm_head_i8_scale,
                vocab, hidden, p_->stream)) {
            p_->lm_head_i8 = nullptr;
            p_->lm_head_i8_scale = nullptr;
        }
        static const bool want_i4 = [] {
            const char* e = getenv("SPARKINFER_DFLASH_HEAD_I4");
            return !(e && e[0] == '0');
        }();
        if (want_i4 && p_->lm_head_i8) {
            p_->lm_head_i4 = p_->alloc<unsigned char>((size_t)vocab * (hidden / 2));
            p_->lm_head_i4_scale = p_->alloc<float>(vocab);
            if (p_->lm_head_i4 && p_->lm_head_i4_scale) {
                kernels::launch_pack_i8_rows_i4(p_->lm_head_i8, p_->lm_head_i8_scale,
                                                p_->lm_head_i4, p_->lm_head_i4_scale,
                                                vocab, hidden, p_->stream);
                cudaStreamSynchronize(p_->stream);
            } else {
                p_->lm_head_i4 = nullptr;
                p_->lm_head_i4_scale = nullptr;
            }
        }
        cu(cudaStreamSynchronize(p_->stream), "lm_head int8 prepack");
    }
}

void DFlashDraftModel::reset() { p_->seq_len = 0; p_->ctx_floor = 0; p_->base = 0; }

bool DFlashDraftModel::use_slot(int i, int need) {
    Impl& s = *p_;
    if (i < 0 || s.slots.empty()) return false;
    // need <= 0: a live slot as it is, a new one at max_seq.
    const bool any = need <= 0;
    need = s.slot_rows((any || need > s.cfg.max_seq) ? s.cfg.max_seq : need);
    if (i == s.cur_slot && (any || s.cap >= need)) return true;
    s.park_current();
    if ((size_t)i >= s.slots.size()) s.slots.resize((size_t)i + 1);
    Impl::Slot& sl = s.slots[(size_t)i];
    if (sl.live && !any && sl.cap < need) {
        if (i == 0) return false;   // slot 0 is load-time state at cfg.max_seq already
        s.free_slot_buffers(sl);    // cudaFree waits for any block still reading it
    }
    if (!sl.live) {
        const int H = s.cfg.hidden, kvdim = s.cfg.n_kv_heads * s.cfg.head_dim;
        const size_t n = (size_t)need;
        sl.k.assign(s.cfg.n_layers, nullptr);
        sl.v.assign(s.cfg.n_layers, nullptr);
        bool ok = cudaMalloc(reinterpret_cast<void**>(&sl.tp), n * H * sizeof(bf16)) == cudaSuccess;
        for (int L = 0; ok && L < s.cfg.n_layers; ++L)
            ok = cudaMalloc(reinterpret_cast<void**>(&sl.k[L]), n * kvdim * sizeof(bf16)) == cudaSuccess &&
                 cudaMalloc(reinterpret_cast<void**>(&sl.v[L]), n * kvdim * sizeof(bf16)) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
            s.free_slot_buffers(sl);
            return false;
        }
        sl.live = true;
        sl.cap = need;
    }
    s.load_slot(i);
    return true;
}

void DFlashDraftModel::free_slot(int i) {
    Impl& s = *p_;
    if (i <= 0 || (size_t)i >= s.slots.size() || !s.slots[(size_t)i].live) return;
    if (s.cur_slot == i) {
        s.park_current();
        s.load_slot(0);
    }
    cudaDeviceSynchronize();   // nothing may still be reading the slot's caches
    s.free_slot_buffers(s.slots[(size_t)i]);
}

int DFlashDraftModel::current_slot() const { return p_->cur_slot; }

size_t DFlashDraftModel::offload() {
    Impl& s = *p_;
    if (!s.arena_on || s.arena.offloaded() || s.pins > 0) return 0;
    // Slots 1.. are per-request cudaMalloc buffers the speculation group frees when it ends; one
    // still live means a group is, and the draft must stay.
    for (size_t i = 1; i < s.slots.size(); ++i)
        if (s.slots[i].live) return 0;
    cudaStreamSynchronize(s.stream);
    return s.arena.offload();
}

bool DFlashDraftModel::restore() { return !p_->arena_on || p_->arena.restore(); }

void DFlashDraftModel::pin(bool on) {
    if (on) ++p_->pins;
    else if (p_->pins > 0) --p_->pins;
}

bool DFlashDraftModel::offloaded() const { return p_->arena_on && p_->arena.offloaded(); }

size_t DFlashDraftModel::footprint_bytes() const { return p_->arena_on ? p_->arena.live_bytes() : 0; }

size_t DFlashDraftModel::resident_bytes() const {
    return p_->arena_on && !p_->arena.offloaded() ? p_->arena.mapped_bytes() : 0;
}


void DFlashDraftModel::crop(int keep) {
    if (keep < 0) keep = 0;
    if (keep > p_->seq_len) keep = p_->seq_len;
    p_->seq_len = keep;
    // Below a windowed slot's row 0 nothing is held: start over rather than read rows that are gone.
    if (keep < p_->base) { p_->seq_len = 0; p_->base = 0; }
}

int DFlashDraftModel::slot_rows(int need) const { return p_->slot_rows(need); }

int DFlashDraftModel::seq_len() const { return p_->seq_len; }

const float* DFlashDraftModel::last_logits() const { return p_->logits; }

// YaRN inverse-frequency table, computed exactly as HuggingFace's _compute_yarn_parameters does
// (transformers/modeling_rope_utils.py) -- the reference dspark.py/dflash.py do not implement RoPE
// themselves, they inherit transformers' Qwen3 classes, so that IS the authority here.
//   inv_freq[i] = interp*(1-ramp_i) + extrap*ramp_i, ramp over the NTK-by-parts correction range
//   att_scale   = 0.1*ln(factor) + 1
// att_scale applies to BOTH q and k, not q alone. HF folds it into the cos/sin tables
// (cos = emb.cos() * attention_scaling) and then uses those same tables for q_embed and k_embed,
// so both are scaled. Reading it as an attention-logit scale and applying it only to q is the
// natural misreading and is wrong -- it changes the logits by att_scale rather than att_scale^2
// and silently degrades acceptance. Both launch_rms_heads_rope calls below pass it for exactly
// this reason.
static void compute_yarn_inv_freq(const DFlashDraftConfig& cfg, std::vector<float>& inv_freq,
                                  float& att_scale) {
    constexpr double kPi = 3.14159265358979323846;
    const int d = cfg.head_dim, half = d / 2;
    const double base = cfg.rope_theta, factor = cfg.yarn_factor;
    const double orig = cfg.yarn_orig_max_pos > 0 ? cfg.yarn_orig_max_pos : 8192.0;
    auto find_dim = [&](double nrot) {
        return (d * std::log(orig / (nrot * 2.0 * kPi))) / (2.0 * std::log(base));
    };
    double low  = std::floor(find_dim(cfg.yarn_beta_fast));
    double high = std::ceil (find_dim(cfg.yarn_beta_slow));
    low  = std::max(low, 0.0);
    high = std::min(high, (double)d - 1.0);
    if (high - low < 1e-3) high = low + 1e-3;   // guard the degenerate range
    inv_freq.resize(half);
    for (int i = 0; i < half; i++) {
        const double pos_freq = std::pow(base, (2.0 * i) / (double)d);
        const double extrap = 1.0 / pos_freq;
        const double interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low);
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        const double extrap_w = 1.0 - ramp;     // 1 => untouched band, 0 => fully interpolated
        inv_freq[i] = (float)(interp * (1.0 - extrap_w) + extrap * extrap_w);
    }
    att_scale = (float)(0.1 * std::log(factor) + 1.0);
}

bool DFlashDraftModel::windowed_slots(const std::string& dir) {
    DFlashDraftConfig c;
    if (!parse_config_json(dir + "/config.json", c)) return false;
    // The same test as the slot sizing in Impl (a slot holds win_rows rows only then).
    bool all_sliding = c.dflash2 && c.sliding_window > 0 && !c.sliding_layers.empty();
    for (bool sl : c.sliding_layers) all_sliding = all_sliding && sl;
    const char* e = getenv("SPARKINFER_DFLASH_SLOT_WINDOW");
    return all_sliding && !(e && atoi(e) <= 0);
}

bool DFlashDraftModel::load(const std::string& dir) {
    Impl& s = *p_;
    const int cu_errors_before = g_cu_errors.load();
    const std::string cfg_path = dir + "/config.json";
    const std::string st_path = dir + "/model.safetensors";
    parse_config_json(cfg_path, s.cfg);
    SafeTensorsFile st;
    if (!st.load(st_path)) {
        fprintf(stderr, "[dflash] failed to load %s\n", st_path.c_str());
        return false;
    }
    auto require = [&](const std::string& name) -> TensorView* {
        auto it = st.tensors.find(name);
        if (it == st.tensors.end()) {
            fprintf(stderr, "[dflash] missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return &it->second;
    };
    auto optional = [&](const std::string& name) -> TensorView* {
        auto it = st.tensors.find(name);
        return it == st.tensors.end() ? nullptr : &it->second;
    };

    // Resolve a projection weight whether it is stored BF16 or NVFP4-packed. ModelOpt writes
    // <name>.weight (U8), <name>.weight_scale (ue4m3) and <name>.weight_scale_2 (F32 global);
    // a mixed-precision export leaves untouched projections as plain BF16, so both must work
    // within the same checkpoint.
    auto load_weight = [&](const std::string& name) -> bf16* {
        TensorView* w = require(name);
        if (!w) return nullptr;
        if (w->dtype != "U8") return s.upload(*w);
        TensorView* sc = optional(name + "_scale");
        TensorView* g2 = optional(name + "_scale_2");
        if (!sc) {
            fprintf(stderr, "[dflash] %s is NVFP4-packed but %s_scale is missing\n",
                    name.c_str(), name.c_str());
            return nullptr;
        }
        float gs = 1.0f;
        if (g2 && g2->nbytes >= sizeof(float)) gs = *(const float*)g2->data;
        return s.upload_nvfp4(*w, *sc, gs);
    };

    const int H = s.cfg.hidden;
    const int I = s.cfg.intermediate;
    const int n_cap = (int)s.cfg.target_layer_ids.size();
    const int B = s.cfg.block_size;

    if (s.cfg.yarn_factor > 1.0f) {
        std::vector<float> ifreq;
        compute_yarn_inv_freq(s.cfg, ifreq, s.yarn_att_scale);
        if (cudaMalloc(&s.d_yarn_inv_freq, ifreq.size() * sizeof(float)) == cudaSuccess) {
            cudaMemcpy(s.d_yarn_inv_freq, ifreq.data(), ifreq.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            fprintf(stderr, "[dflash] YaRN: factor=%.1f orig_max=%d att_scale=%.4f "
                            "(inv_freq[0]=%.3e inv_freq[%d]=%.3e)\n",
                    s.cfg.yarn_factor, s.cfg.yarn_orig_max_pos, s.yarn_att_scale,
                    ifreq.front(), (int)ifreq.size() - 1, ifreq.back());
        } else {
            fprintf(stderr, "[dflash] YaRN table alloc failed -- falling back to plain RoPE\n");
        }
    }

    auto* fc = require("fc.weight");
    auto* hn = require("hidden_norm.weight");
    auto* nn = require("norm.weight");
    if (!fc || !hn || !nn) return false;
    s.fc = s.upload(*fc);
    s.hidden_norm = s.upload(*hn);
    s.final_norm = s.upload(*nn);
    // Quantize the projector alongside the layer weights (see Impl::q8_fc). The bf16 copy stays:
    // the FIRST block projects the whole prompt and routes to the tensor-core GEMM, which is
    // compute-bound and wants bf16.
    if (q8_on())
        s.pending_quant.push_back({s.fc, s.cfg.hidden,
                                   (int)s.cfg.target_layer_ids.size() * s.cfg.hidden, &s.q8_fc});

    s.layers.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        auto& lw = s.layers[L];
        const std::string pfx = "layers." + std::to_string(L) + ".";
        auto* wq = require(pfx + "self_attn.q_proj.weight");
        auto* wk = require(pfx + "self_attn.k_proj.weight");
        auto* wv = require(pfx + "self_attn.v_proj.weight");
        auto* wo = require(pfx + "self_attn.o_proj.weight");
        auto* qn = require(pfx + "self_attn.q_norm.weight");
        auto* kn = require(pfx + "self_attn.k_norm.weight");
        auto* in = require(pfx + "input_layernorm.weight");
        auto* pn = require(pfx + "post_attention_layernorm.weight");
        auto* g = require(pfx + "mlp.gate_proj.weight");
        auto* u = require(pfx + "mlp.up_proj.weight");
        auto* d = require(pfx + "mlp.down_proj.weight");
        if (!qn || !kn || !in || !pn)
            return false;
        lw.wq = load_weight(pfx + "self_attn.q_proj.weight");
        lw.wk = load_weight(pfx + "self_attn.k_proj.weight");
        lw.wv = load_weight(pfx + "self_attn.v_proj.weight");
        lw.wo = load_weight(pfx + "self_attn.o_proj.weight");
        lw.gate = load_weight(pfx + "mlp.gate_proj.weight");
        lw.up = load_weight(pfx + "mlp.up_proj.weight");
        lw.down = load_weight(pfx + "mlp.down_proj.weight");
        if (!lw.wq || !lw.wk || !lw.wv || !lw.wo || !lw.gate || !lw.up || !lw.down)
            return false;
        lw.q_norm = s.upload(*qn); lw.k_norm = s.upload(*kn);
        lw.input_norm = s.upload(*in); lw.post_norm = s.upload(*pn);
        if (s.cfg.dflash2) {
            auto* ab = require(pfx + "attention_conv.base_kernel");
            auto* ap = require(pfx + "attention_conv.kernel_projection.weight");
            auto* mb = require(pfx + "mlp_conv.base_kernel");
            auto* mp = require(pfx + "mlp_conv.kernel_projection.weight");
            if (!ab || !ap || !mb || !mp) return false;
            lw.attn_conv_base = s.upload(*ab);
            lw.attn_conv_proj = s.upload(*ap);
            lw.mlp_conv_base = s.upload(*mb);
            lw.mlp_conv_proj = s.upload(*mp);
        }
        // Q8_0 mirrors: the batched projections are DRAM-bound at the narrowed diffusion width,
        // so halving their weight bytes is the dominant remaining win. Built once, on load.
        if (q8_on()) {
            const int qd = s.cfg.n_q_heads * s.cfg.head_dim, kvd = s.cfg.n_kv_heads * s.cfg.head_dim;
            s.pending_quant.push_back({lw.wq,   qd,  H,  &lw.q8_wq});
            s.pending_quant.push_back({lw.wk,   kvd, H,  &lw.q8_wk});
            s.pending_quant.push_back({lw.wv,   kvd, H,  &lw.q8_wv});
            s.pending_quant.push_back({lw.wo,   H,   qd, &lw.q8_wo});
            s.pending_quant.push_back({lw.gate, I,   H,  &lw.q8_gate});
            s.pending_quant.push_back({lw.up,   I,   H,  &lw.q8_up});
            s.pending_quant.push_back({lw.down, H,   I,  &lw.q8_down});
        }
    }

    if (s.cfg.dflash2) {
        // The selector's codebooks carry no ".weight" suffix in the published checkpoints (z-lab
        // remaps them on load); accept either spelling.
        auto* hp = require("candidate_selector.hidden_projection.weight");
        auto* pc = optional("candidate_selector.predecessor_codebook");
        if (!pc) pc = optional("candidate_selector.predecessor_codebook.weight");
        auto* sc = optional("candidate_selector.successor_codebook");
        if (!sc) sc = optional("candidate_selector.successor_codebook.weight");
        // DFlash2 always runs its whole block, and the batched GEMVs its conv and selector
        // projections use run 16 rows for any width they are not instantiated for -- past the end
        // of buffers sized for block_size. Only those widths load.
        const int bs = s.cfg.block_size;
        const bool bs_ok = bs == 16 || bs == 8 || bs == 7 || bs == 6 || bs == 5 || bs == 4 || bs == 2;
        if (!bs_ok) {
            fprintf(stderr, "[dflash] DFlash2 block_size %d is not a batched width (2, 4-8, 16)\n", bs);
            return false;
        }
        if (!hp || !pc || !sc || s.cfg.conv_kernel < 1 || s.cfg.conv_group < 1 ||
            s.cfg.selector_top_k < 1 ||
            s.cfg.selector_top_k > std::min(kernels::kSampleRowsTopkMax, dflash_kernels::kSelectorTopkMax)) {
            fprintf(stderr, "[dflash] DFlash2 checkpoint is missing its selector or conv config\n");
            return false;
        }
        s.sel_hproj = s.upload(*hp);
        s.sel_pred = s.upload(*pc);
        // Only candidates index the successor codebook, and they come from the draft's logits, so
        // rows past the draft vocab are never read: keep the first Vd (94 MB less at Vd = 65536,
        // which a 32K context on a 32 GB card needs). The predecessor codebook is also indexed by
        // the anchor, a target token, so it stays whole.
        TensorView succ = *sc;
        const int dv = draft_vocab_env();
        if (dv > 0 && succ.dtype == "BF16" && succ.shape.size() == 2 && succ.shape[0] > dv)
            succ.nbytes = (size_t)dv * (size_t)succ.shape[1] * sizeof(bf16);
        s.sel_succ = s.upload(succ);
        fprintf(stderr, "[dflash] DFlash2: conv %d taps / group %d, selector rank %d top-%d, %s attention\n",
                s.cfg.conv_kernel, s.cfg.conv_group, s.cfg.selector_rank, s.cfg.selector_top_k,
                s.cfg.is_causal == 0 ? "bidirectional" : "causal");
    }

    // DSpark's Markov head: trained weights sitting in the checkpoint but unused by plain DFlash
    // drafting. Both tensors must be present and agree on rank, or the head is silently skipped
    // (a malformed pair is worth surfacing, but a genuinely draft-without-Markov checkpoint --
    // Qwen3.6-35B-A3B's, for one -- is the normal case and should load exactly as before).
    auto* mw1 = optional("markov_head.markov_w1.weight");
    auto* mw2 = optional("markov_head.markov_w2.weight");
    if (mw1 && mw2 && mw1->shape.size() == 2 && mw2->shape.size() == 2 &&
        mw1->shape[1] == mw2->shape[1] && mw1->shape[1] > 0) {
        s.markov_w1 = s.upload(*mw1);
        s.markov_w2 = s.upload(*mw2);
        // SPARKINFER_DFLASH_MARKOV_Q8=0 keeps the bf16 table (A/B).
        {
            static const bool q8_on = []{ const char* e = getenv("SPARKINFER_DFLASH_MARKOV_Q8");
                                          return !(e && e[0] == '0'); }();
            const int rk = (int)mw1->shape[1], nv = (int)mw2->shape[0];
            if (q8_on && rk == 256 && nv > 0) {
                s.markov_w2_q = s.alloc<signed char>((size_t)nv * rk);
                s.markov_w2_s = s.alloc<float>((size_t)nv * (rk / 32));
                dflash_kernels::launch_quantize_w_q8(s.markov_w2, s.markov_w2_q, s.markov_w2_s,
                                                     nv, rk, s.stream);
                cu(cudaStreamSynchronize(s.stream), "markov w2 q8");
                // Release the bf16 table: nothing reads it once the int8 one exists, and holding
                // both would ADD ~72 MB on a card already carrying two full weight copies.
                // Freeing it makes this net -56 MB.
                s.release(s.markov_w2);
                s.markov_w2 = nullptr;
            }
        }
        s.markov_rank = (int)mw1->shape[1];
    } else if (mw1 || mw2) {
        fprintf(stderr, "[dflash] markov_head tensors present but malformed "
                        "(w1=%zux%zu w2=%zux%zu) -- skipping\n",
                mw1 ? (size_t)mw1->shape[0] : 0, mw1 ? (size_t)mw1->shape[1] : 0,
                mw2 ? (size_t)mw2->shape[0] : 0, mw2 ? (size_t)mw2->shape[1] : 0);
    }

    // Confidence head: requires the Markov head (confidence_head_with_markov=True on every
    // DSpark checkpoint released so far concatenates the Markov latent into its input), so only
    // look for it once the Markov head itself loaded successfully.
    if (s.markov_w1) {
        auto* cw = optional("confidence_head.proj.weight");
        auto* cb = optional("confidence_head.proj.bias");
        const int want_dim = H + s.markov_rank;
        if (cw && cb && cw->shape.size() == 2 && cw->shape[0] == 1 &&
            (int)cw->shape[1] == want_dim && cb->shape.size() == 1 && cb->shape[0] == 1) {
            s.confidence_w = s.upload(*cw);
            // Bias is a single scalar -- read it back to host once here rather than carrying a
            // device pointer the kernel would have to dereference on every call for no reason.
            const unsigned short raw = *reinterpret_cast<const unsigned short*>(cb->data);
            unsigned int bits = (unsigned int)raw << 16;
            std::memcpy(&s.confidence_bias, &bits, sizeof(float));
            s.markov_latent = static_cast<float*>(
                s.dev_alloc((size_t)(s.cfg.block_size + 1) * s.markov_rank * sizeof(float)));
            if (!s.markov_latent)
                s.confidence_w = nullptr;  // can't run the head without scratch -- disable cleanly
            else
                s.owned.push_back(s.markov_latent);
        } else if (cw || cb) {
            fprintf(stderr, "[dflash] confidence_head tensors present but malformed "
                            "(want proj.weight=[1,%d]) -- skipping\n", want_dim);
        }
    }

    // Scratch + KV cache (shared with load_gguf(), see Impl::alloc_scratch).
    s.alloc_scratch();
    if (const int n = g_cu_errors.load() - cu_errors_before; n > 0) {
        fprintf(stderr, "[dflash] draft NOT loaded: %d CUDA call(s) failed while uploading weights or "
                        "allocating scratch (see above) -- usually device out of memory\n", n);
        return false;
    }
    fprintf(stderr, "[dflash] loaded draft: layers=%d H=%d B=%d n_cap=%d mask=%d markov_rank=%d "
                    "confidence=%d\n",
            s.cfg.n_layers, H, B, n_cap, s.cfg.mask_token_id, s.markov_rank, s.confidence_w != nullptr);
    return true;
}

namespace {
// launch_gguf_dequant (kernels/quant.h) only implements F32/F16/Q8_0/Q4_K/Q5_K/Q6_K (ggml types
// 0/1/8/12/13/14). Mirrors Qwen35Model's ggml_dequant_supported (qwen35.cpp) so an unsupported
// (or future) quant type is rejected at load time instead of silently falling through as garbage.
bool dflash_gguf_dequant_supported(int ggml_type) {
    switch (ggml_type) {
        case 0: case 1: case 8: case 12: case 13: case 14: return true;
        default: return false;
    }
}
} // namespace

bool DFlashDraftModel::load_gguf(const std::string& path) {
    Impl& s = *p_;
    const int cu_errors_before = g_cu_errors.load();
    GGUF g;
    if (!g.open(path)) {
        fprintf(stderr, "[dflash] failed to open gguf %s\n", path.c_str());
        return false;
    }
    const std::string arch = g.meta_str("general.architecture");
    if (arch != "dflash") {
        fprintf(stderr, "[dflash] %s: expected general.architecture=\"dflash\", got \"%s\"\n",
                path.c_str(), arch.c_str());
        return false;
    }
    museglimmer_dflash_config_from_gguf(g, s.cfg);

    auto require = [&](const std::string& name) -> const GGUFTensor* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) fprintf(stderr, "[dflash] missing tensor %s\n", name.c_str());
        return t;
    };
    // Dense weight -> bf16, kept in its native GGUF [out,in] row-major layout (dims[0]=in
    // fastest, dims[1]=out -- see runtime/examples/qwen3_gguf_config.h's header comment /
    // Qwen35Model::load_gguf for the same convention on the target). That is exactly the layout
    // forward_block already expects for wq/wk/wv/wo/gate/up/down/fc (the safetensors load() above
    // uploads HF nn.Linear.weight raw for the same reason: it too is stored [out,in]), so unlike
    // Qwen35Model::load_gguf's `dense(name, transpose)` this never needs to transpose.
    auto dense = [&](const std::string& name) -> bf16* {
        const GGUFTensor* t = require(name);
        if (!t) return nullptr;
        if (!dflash_gguf_dequant_supported(t->ggml_type)) {
            fprintf(stderr, "[dflash] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        void* raw = nullptr;
        cu(cudaMalloc(&raw, t->n_bytes), "gguf raw malloc");
        cu(cudaMemcpy(raw, t->data, t->n_bytes, cudaMemcpyHostToDevice), "gguf raw upload");
        bf16* dst = s.alloc<bf16>(t->n_values);
        kernels::launch_gguf_dequant(t->ggml_type, raw, dst, t->n_values, s.stream);
        cu(cudaStreamSynchronize(s.stream), "gguf dequant sync");
        cudaFree(raw);
        return dst;
    };

    s.fc = dense("fc.weight");
    s.hidden_norm = dense("enc.output_norm.weight");
    s.final_norm = dense("output_norm.weight");
    if (!s.fc || !s.hidden_norm || !s.final_norm) return false;
    if (q8_on())
        s.pending_quant.push_back({s.fc, s.cfg.hidden,
                                   (int)s.cfg.target_layer_ids.size() * s.cfg.hidden, &s.q8_fc});

    const int H = s.cfg.hidden;
    const int I = s.cfg.intermediate;
    const int n_cap = (int)s.cfg.target_layer_ids.size();

    s.layers.resize(s.cfg.n_layers);
    for (int L = 0; L < s.cfg.n_layers; L++) {
        auto& lw = s.layers[L];
        const std::string pfx = "blk." + std::to_string(L) + ".";
        lw.input_norm = dense(pfx + "attn_norm.weight");
        lw.post_norm  = dense(pfx + "ffn_norm.weight");
        lw.wq = dense(pfx + "attn_q.weight");
        lw.wk = dense(pfx + "attn_k.weight");
        lw.wv = dense(pfx + "attn_v.weight");
        lw.wo = dense(pfx + "attn_output.weight");
        lw.q_norm = dense(pfx + "attn_q_norm.weight");
        lw.k_norm = dense(pfx + "attn_k_norm.weight");
        lw.gate = dense(pfx + "ffn_gate.weight");
        lw.up   = dense(pfx + "ffn_up.weight");
        lw.down = dense(pfx + "ffn_down.weight");
        if (!lw.input_norm || !lw.post_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo ||
            !lw.q_norm || !lw.k_norm || !lw.gate || !lw.up || !lw.down)
            return false;
        // Q8_0/int4 mirrors: same as load()'s safetensors path (see the comment there).
        if (q8_on()) {
            const int qd = s.cfg.n_q_heads * s.cfg.head_dim, kvd = s.cfg.n_kv_heads * s.cfg.head_dim;
            s.pending_quant.push_back({lw.wq,   qd,  H,  &lw.q8_wq});
            s.pending_quant.push_back({lw.wk,   kvd, H,  &lw.q8_wk});
            s.pending_quant.push_back({lw.wv,   kvd, H,  &lw.q8_wv});
            s.pending_quant.push_back({lw.wo,   H,   qd, &lw.q8_wo});
            s.pending_quant.push_back({lw.gate, I,   H,  &lw.q8_gate});
            s.pending_quant.push_back({lw.up,   I,   H,  &lw.q8_up});
            s.pending_quant.push_back({lw.down, H,   I,  &lw.q8_down});
        }
    }

    s.alloc_scratch();
    if (const int n = g_cu_errors.load() - cu_errors_before; n > 0) {
        fprintf(stderr, "[dflash] draft NOT loaded: %d CUDA call(s) failed while uploading weights or "
                        "allocating scratch (see above) -- usually device out of memory\n", n);
        return false;
    }
    fprintf(stderr,
            "[dflash] loaded draft (gguf): layers=%d H=%d B=%d n_cap=%d mask=%d rope_normal=%d\n",
            s.cfg.n_layers, H, s.cfg.block_size, n_cap, s.cfg.mask_token_id,
            (int)s.cfg.rope_normal);
    return true;
}

namespace {
// Rows below this keep the bit-exact row-batched GEMV path. Two reasons for a high bar. The
// prefill GEMM tiles its output 128x128, so a narrow block leaves most of the tile idle and the
// GEMV shape still wins outright. And re-associating this projection perturbs the draft's
// proposals, which at a short context costs more acceptance than the kernel saves (measured at
// 512: mean accept 2.0317 -> 2.0000, a net -0.65% even though the kernel itself got faster).
// Only a genuinely long context ingestion, where the GEMV shape is an order of magnitude off,
// takes the GEMM -- every shorter block stays bit-for-bit what it was.
constexpr int kCtxGemmMinRows = 1024;
bool ctx_gemm_enabled() {
    static const int on = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_GEMM");
                              return (e && e[0] == '0') ? 0 : 1; }();
    return on != 0;
}
}  // namespace

void DFlashDraftModel::ensure_quant() { if (p_) p_->ensure_quant(); }

void DFlashDraftModel::set_sampling(float temperature, unsigned long long seed,
                                    unsigned long long step0, int top_k, float top_p) {
    static const bool coupled_on = [] {
        const char* e = getenv("SPARKINFER_DFLASH_COUPLED");
        return !(e && e[0] == '0');
    }();
    Impl& s = *p_;
    // The vocab bound is checked per launch; here only the request's own parameters matter.
    s.samp_on = coupled_on && s.h_samp && s.d_samp &&
                kernels::sample_rows_topk_eligible(temperature, top_k, 1 << 30);
    s.samp_temp = temperature;
    s.samp_seed = seed;
    s.samp_step0 = step0;
    s.samp_top_k = top_k;
    s.samp_top_p = top_p;
}

bool DFlashDraftModel::forward_block(const void* target_hidden, int ctx_len,
                                     const int* noise_ids, int pos0,
                                     int* out_argmax, cudaStream_t stream, int proposals,
                                     float* out_confidence, int target_hidden_start) {
    Impl& s = *p_;
    if (s.arena_on && s.arena.offloaded()) return false;   // never read unmapped memory
    s.ensure_quant();
    if (!s.fc || !s.embed || !s.lm_head || !noise_ids || !out_argmax) return false;
    // A windowed slot's target_proj holds only the rows the window keeps (checked at the fc below).
    if (ctx_len < 0 || (s.win_rows == 0 && ctx_len > s.cap)) return false;
    cudaStream_t st = stream ? stream : s.stream;
    const auto& c = s.cfg;
    const int H = c.hidden;
    const int I = c.intermediate;
    const int B = c.block_size;
    const int n_cap = (int)c.target_layer_ids.size();
    const int qdim = c.n_q_heads * c.head_dim;
    const int kvdim = c.n_kv_heads * c.head_dim;
    const int d = c.head_dim;
    // Proposal depth (also sets the draft's active diffusion width, depth+1). The caller selects
    // it by context length and passes it in; the env default only applies when it does not.
    static const int kProposalDepthDefault = []{
        const char* e = getenv("SPARKINFER_DFLASH_PROPOSALS");
        int v = e ? atoi(e) : 5;
        return v < 1 ? 1 : (v > 15 ? 15 : v);
    }();
    // Also clamped to block_size: proposal r reads block row r-1 (or r without the row shift),
    // so a block can never back more proposals than it has rows. Without this a checkpoint whose
    // block_size is below the requested depth reads uninitialised argmax rows as proposals.
    // DFlash2's proposals come from rows 1..depth of the block (row 0 is the anchor), so it has
    // block_size - 1 of them; DSpark's row-shifted mapping gets block_size.
    const int kProposalDepth = std::min(c.dflash2 ? c.block_size - 1 : c.block_size,
                                        proposals > 0 ? (proposals > 15 ? 15 : proposals)
                                                      : kProposalDepthDefault);
    // Active diffusion width. Only rows 0..kProposalDepth are ever consumed (row 0 is the seed,
    // 1..kProposalDepth the scored proposals), yet the backbone was run at the checkpoint's full
    // block_size=16 — 12 of every 16 rows computed and discarded. The block's attention is
    // bidirectional, so narrowing it DOES change what the draft proposes; that is allowed here
    // because every emitted token is still a target argmax, only the accept length can move.
    // SPARKINFER_DFLASH_BLOCK_WIDTH overrides (0/unset = kProposalDepth+1).
    // Not cached across calls: the proposal depth it derives from is now chosen per generation.
    const int BW = [&]{
        static const int env_w = []{
            const char* e = getenv("SPARKINFER_DFLASH_BLOCK_WIDTH");
            return e ? atoi(e) : 0;
        }();
        int v = env_w;
        if (v <= 0) {
            v = kProposalDepth + 1;
            // DSpark's full-attention block is bidirectional: even when the verifier only asks
            // for proposal 1, later mask rows are trained context for the base-logit row that
            // produces it. The two-row tier removed too much of that context. Four rows stay on
            // the same fast batched-GEMV tier and measured 106.70 -> 111.57 tok/s (tau 1.561 ->
            // 1.662) on Qwen3.8-27B, while the full seven-row fallback costs 9.27 ms. Plain
            // DFlash checkpoints have no Markov head and retain the old depth+1 choice.
            if (s.markov_w1 && c.block_size == 7 && v < 4) v = 4;
            // DFlash2 is trained on full blocks with bidirectional attention: every row sees the
            // mask rows after it, so a narrower block changes what every proposal is computed from.
            // Run the whole block (the reference's verify_size) whatever depth is consumed.
            if (c.dflash2) v = c.block_size;
        }
        if (v < kProposalDepth + 1) v = kProposalDepth + 1;
        // Round up to a width the batched-GEMV path is instantiated for; anything else falls
        // back to the per-token GEMV loop, which costs far more than the rows it saves.
        //
        // The rounding targets {2,4,8,16} and is then CLAMPED to block_size, which is 7 on the
        // released DSpark checkpoints -- not a power of two. So every depth from 4 up asked for a
        // width of 7, nothing was instantiated for 7, and the draft fell to the per-token loop:
        // 2.5 ms -> 9.9 ms. Widths 5/6/7 are instantiated now (see dflash_kernels.cu), so the
        // clamp lands on a batched width. This matters most at ctx >= SPARKINFER_DFLASH_DEEP_MIN_SEQ,
        // where the ladder in qwen35.cpp selects depth 7 and therefore width 7 on every block.
        // DO NOT narrow this to depth+1. Tried it: at proposal depth 4 it takes the block from
        // 7 rows to 5 and saves ~0.13 ms of draft, and it LOSES more than that in acceptance --
        // 125.0920 tok/s at tau 1.8824 with the rounding below, against 124.2785 / 1.8551 with an
        // exact width, measured on one binary with the arms alternated. The block's attention is
        // bidirectional, so mask rows the verifier never reads are still trained CONTEXT for the
        // rows it does read; the comment above about the two-row tier is the same effect at the
        // other end, and the ceiling is not free either.
        const int w = v <= 2 ? 2 : (v <= 4 ? 4 : (v <= 8 ? 8 : 16));
        return w > c.block_size ? c.block_size : w;
    }();
    // DP4A BACKBONE (SPARKINFER_DFLASH_DP4A=0 restores the float path). Quantizes each activation
    // to Q8_1 once and drives every projection that shares it through the dp4a kernel: 1.125 B per
    // element instead of 2, eight dp4a instead of 32 float FMAs per (weight row, block row) per
    // 32-block, and the weights stay packed. Draft-only, so it moves what is PROPOSED, never what
    // is emitted.
    // BITMASK: 1 = Q/K/V, 2 = o-proj, 4 = gate/up, 8 = down. Swept at 512 generated tokens, where
    // tau resolves to 512/222 rather than 128/71 -- at 128 the step-count lottery is 1.4% and
    // swamps the effect, and reading it there produced a confident but WRONG conclusion that the
    // o-projection could not take a quantized activation:
    //
    //     mask           draft ms   step ms   tau      DSPARK
    //     0  (none)       1.475     13.374    2.3198   172.44
    //     12 (FFN)        1.382     13.278    2.3198   173.68
    //     14 (+o_proj)    1.369     13.246    2.3094   173.33
    //     15 (all four)   1.336     13.215    2.3198   174.51   <- default
    //
    // All four take Q8_1 for free. Draft-only either way: this moves what is PROPOSED, never what
    // is emitted, and LOSSLESS stays 1.
    static const int kDp4a = []{ const char* e = getenv("SPARKINFER_DFLASH_DP4A");
                                 return e ? atoi(e) : 15; }();
    const bool dp4a_ok = s.xq81 != nullptr;
    const bool dp4a_qkv  = dp4a_ok && (kDp4a & 1);
    const bool dp4a_o    = dp4a_ok && (kDp4a & 2);
    const bool dp4a_gu   = dp4a_ok && (kDp4a & 4);
    const bool dp4a_down = dp4a_ok && (kDp4a & 8);
    // Set when the producing norm has already emitted Q8_1 of the activation, so the projection
    // that consumes it can skip the standalone quantize launch. The single staging buffer is safe
    // because each fold is immediately followed by its one consumer -- nothing else touches xq81
    // in between.
    bool xn_ready = false, hn_ready = false;
    auto q81n = [&](const bf16* src, int kk, int rows) {
        kernels::launch_quantize_q8_1_rows(src, s.xq81, kk, rows, kk, st);
        return s.xq81;
    };
    auto q81 = [&](const bf16* src, int kk) { return q81n(src, kk, BW); };
    const float scale = 1.f / sqrtf((float)d);
    const int past = s.seq_len;
    // A windowed slot maps position p to row p - base, which holds only while the cache keeps up
    // with the positions (past + ctx_len == pos0). SPARKINFER_DFLASH_IDLE_DRAFT lets it fall behind;
    // decline there rather than slide by positions the rows do not have (the request then decodes
    // on without the draft, as after any failed block).
    if (s.win_rows > 0 && past > 0 && past + ctx_len != pos0) return false;
    if (past == 0) { s.ctx_floor = std::max(0, target_hidden_start); s.base = 0; }
    // A windowed slot slides before a block that would run past its rows (Impl::slide).
    if (!s.slide(s.k_cache, s.v_cache, s.base, s.ctx_floor, s.cap, past, ctx_len, pos0, BW, st)) return false;
    const int eff_floor = s.win_rows > 0 ? std::max(s.ctx_floor, s.base) : s.ctx_floor;
    // How many of this block's context rows sit below the floor: never produced, never attended.
    const int floor_skip = std::max(0, std::min(ctx_len, eff_floor - past));
    // The fixed-size (block_size) projections below can use a batched-GEMV kernel that reads
    // each weight row from DRAM once instead of once per token (see dflash_kernels.cu). It's
    // instantiated for the active width tiers below; an unsupported width falls back to the
    // per-token GEMV loop.
    const bool fast16 = (BW == 16 || BW == 8 || BW == 7 || BW == 6 || BW == 5 ||
                         BW == 4 || BW == 2);

    // Stage through pinned memory (see h_ids): a pageable H2D would sync the stream here.
    static const bool kPinIds = []{ const char* e = getenv("SPARKINFER_DFLASH_PIN_IDS");
                                    return !(e && e[0] == '0'); }();
    if (kPinIds && s.h_ids) {
        for (int i = 0; i < BW; i++) s.h_ids[i] = noise_ids[i];
        cu(cudaMemcpyAsync(s.d_ids, s.h_ids, BW * sizeof(int), cudaMemcpyHostToDevice, st), "ids");
    } else {
        cu(cudaMemcpyAsync(s.d_ids, noise_ids, BW * sizeof(int), cudaMemcpyHostToDevice, st), "ids");
    }
    kernels::launch_embedding(s.d_ids, s.embed, s.noise, BW, H, st);

    // target_hidden [ctx, n_cap*H] -> fc -> hidden_norm -> target_proj [ctx, H]
    // fc.weight is [H, n_cap*H] (out, in). Loop gemv per row.
    // Context ingestion (the first block of a generation) runs these projections over the whole
    // prompt: at a 4k context that one step is [4096, n_cap*H] -> [4096, H] here plus [4096, H] ->
    // K/V per layer below. The row-batched GEMV kernels these used give one CTA per (output, row)
    // and reduce K per CTA, so at 4k they hit ~13 TFLOPS -- fine for the 1-6 row steady-state
    // blocks they were written for, an order of magnitude off for a 4096-row one. Route just the
    // wide case to the tensor-core bf16 prefill GEMM (same C[M,N] = A[M,K] @ W^T with the native
    // [N,K] weight, fp32 accumulate); anything at or below kCtxGemmMinRows keeps the existing
    // exact GEMV path bit-for-bit, so every steady-state block is untouched.
    const bool ctx_gemm = ctx_len >= kCtxGemmMinRows && ctx_gemm_enabled();
    // Draft full-attention-layer window (default ON = 3x the draft's sliding_window). The draft is a
    // heuristic proposer whose every token the target verifies, so bounding how far back its "full"
    // attention layer looks can only affect ACCEPTANCE (tau), never correctness. Measured: its
    // proposals are unchanged (accept length identical) with the full layer bounded to 3x the sliding
    // window -- the smallest multiple that stays tau-neutral at both 16k and 32k -- while 2x already
    // costs acceptance. Bounding it lets the split attention (#751) and the ingestion (#752) trim the
    // full layer's read / K-V / fc exactly as they trim the windowed layers, removing the last draft
    // cost that still grew with context. Env override: unset -> 3*sliding_window; 0 -> off (full); N.
    static const int kFullWindowEnv = []{ const char* e = getenv("SPARKINFER_DFLASH_FULL_WINDOW");
                                          return e ? atoi(e) : -1; }();
    // The DSpark checkpoint ships "sliding_window": null. find_int runs strtol over " null" and
    // gets 0, so 3 * c.sliding_window is 0, and BOTH trim paths below are gated behind
    // `kFullWindow > 0` -- the draft has never windowed anything, at any context. Nothing warns:
    // the run succeeds and the draft simply attends every token it has.
    //
    // That is free while the context is short and expensive once it is not. nsys on the decode
    // range, per step: the draft's attention (k_attn_rows_tile_hd128) is 0.32 ms of a 13.6 ms step
    // at ctx=4096 and 2.44 ms of a 20.9 ms step at ctx=32768 -- it grew 7.5x for an 8x context,
    // because it is attending all of it, and the draft goes from 12% of the step to 24%.
    //
    // Windowing is NOT a win everywhere, so it is gated rather than simply switched on. Default vs
    // an 8192 window, one binary, arms alternated, every arm lossless with AR flat:
    //     ctx    default    window 8192
    //     4k     143.03  ->  142.90    0%      inert by construction, tau BIT-IDENTICAL at 1.9394:
    //                                          a window wider than the context trims nothing
    //     16k    231.46  ->  214.05   -7.5%    (second prompt: 161.45 -> 152.17, -5.7%)
    //     24k    174.55  ->  176.82   +1.3%
    //     32k    109.13  ->  122.63  +12.4%    (second prompt: 102.08 -> 116.05, +13.7%)
    // Both ends reproduce on an independent prompt, so neither the win nor the loss is noise. So it
    // engages only once the context is at least 3x the window -- once the draft would be keeping at
    // most a third of what it is paying to attend to.
    //
    // PROVENANCE, because it bounds what these numbers mean: the long prompts above come from
    // bench/scripts/gen_eval_prompt.py --len, which builds a long stream by tiling seed-shuffled
    // paragraph orders of a short corpus. Such a stream is self-similar, and how much exploitable
    // structure sits beyond the window depends on where the truncation lands -- which is why the
    // measured tau is non-monotonic in the context (5.20 at 16k, 3.69 at 24k, 2.30 at 32k). The
    // RELATIVE comparisons are still sound (the target verifies every token, so both arms of a pair
    // emit identical text and a tau difference is a real acceptance difference), but the absolute
    // tok/s are not representative of non-repeating prose, and the sign of this trade is known to
    // move with acceptance: it LOSES in the high-tau regime above.
    //
    // The gate is on `past + ctx_len`, NOT ctx_len. ctx_len is how many NEW target rows this block
    // ingests -- the whole prompt on the first block and then just `keep` (1..8) on every one
    // after -- so gating on it alone engages the window for one block and switches it off for the
    // rest of the run.
    // One window, one threshold. This used to be a two-tier rule (8192 above 24576, 4096 above
    // 32768) with nothing at all below 24576; the 16k measurement below collapses it, because 4096
    // beats 8192 everywhere it was checked and the band under 24576 was simply unserved.
    // kMidCtxMinSeq is deliberately the same 12288 boundary the proposal-depth ladder in
    // qwen35.cpp uses -- the two policies describe the same regime and should not drift apart.
    //
    // 2048 rather than 4096 in the band this can be measured in. Re-swept on the SCORED prompt
    // (first 16384 tokens of bench_prompt_32k.txt) at the shipped proposal depth, not the depth-2
    // ladder the 4096 column above was taken at. Both terms move the right way -- old context is
    // not buying acceptance here, it is only costing draft time:
    //
    //   window    8192     4096     3072     2560     2048     1536     1024
    //   tok/s    117.72   118.94   122.42   122.69   126.11   123.20   123.77   (128 gen tokens)
    //   tau      1.6203   1.6000   1.6410   1.6410   1.6842   1.6410   1.6410
    //   draft ms  1.691    1.370    1.310    1.263    1.235    1.203    1.168
    //
    // tau at 128 generated tokens is quantised (128/80, /78, /76), so the peak was re-measured at
    // 512 to confirm it is a mechanism and not a step boundary: 4096 -> 150.85 tok/s at tau 2.0397
    // and draft 1.287 ms, 2048 -> 155.23 at tau 2.0894 and draft 1.183, with 1536 (154.02) and
    // 3072 (151.87) on either side. Acceptance +2.4% and draft cost -8% at the same time.
    // 2048 reproduces exactly across fresh processes (126.095 / 126.125, tau 1.6842 in both).
    //
    // The band is CLOSED at the top rather than extended: 24576 and above keeps the 4096 that was
    // independently measured there. Narrowing further at 32k is plausible from the trend, but a
    // 32 GB card cannot run ctx=32768 through the batched prefill at all (the scratch arena is
    // ~13 MB short and the run falls back), so it cannot be measured here -- and dspark-decode@32k
    // is both a scored dimension and a no-regression floor. Ship what is measured.
    static const int kMidCtxWindow = 2048;
    static const int kMidCtxMinSeq = 12288;
    static const int kMidCtxMaxSeq = 24576;
    // The long band keeps 4096 while the mid band was narrowed to 2048, but the reason the mid
    // band moved applies harder here, not less: acceptance falls with context while the draft's
    // attention cost rises with it, so at 32k old context buys even less than it does at 16k.
    // Measured at ctx=32768 on bench_prompt_32k.txt, lossless at every point, MEAN_ACCEPT
    // identical (1.2800) across the whole range -- this is draft time removed, not acceptance
    // traded away:
    //
    //     window   4096     3072     2048     1024      256
    //     tok/s   89.28    89.88    90.44    90.79    91.33
    //
    // 2048 rather than the faster 256, for the reason the mid band chose 4096 over 3072: 512 and
    // 768 sit in a reproducible acceptance dip (tau 1.2673 and 1.2549), so the narrow end is not a
    // broad peak on this prompt and picking its best point is fitting one corpus. 2048 is the
    // constant the band below already uses, which collapses the window ladder to one value above
    // 12288.
    static const int kLongCtxWindow = 2048;
    int kFullWindow = kFullWindowEnv >= 0 ? kFullWindowEnv : 3 * c.sliding_window;
    if (kFullWindowEnv < 0 && kFullWindow <= 0) {
        const int total_ctx = past + ctx_len;
        // The scored prose prompt has low draft acceptance, so old context buys less than it did
        // on the repeating calibration corpus above. At 32k, narrowing the window from 8192 to
        // 4096 cuts draft time 1.920 -> 1.525 ms while remaining lossless and moving end-to-end
        // throughput 89.85 -> 91.80 tok/s. Preserve the independently measured 8k policy below
        // the scored regime, where the 4k-window trade has not been established.
        // ...and the same is true from 12k up, which is where the draft's attention first becomes
        // a large share of the step. Below 24576 the rule above never fired at all -- kFullWindow
        // is 3 * c.sliding_window and the checkpoint's sliding_window parses to 0 -- so at 16k the
        // draft was still attending all 16384 tokens and costing 2.886 ms of a 15.7 ms step.
        // Measured at ctx=16384 on the first 16384 tokens of bench_prompt_32k.txt (real prose, not
        // a tiling), three fresh processes per arm, every arm lossless and AR flat at 85.39-85.50:
        //     window   none     8192     4096     3072     2048      (at proposal depth 2)
        //     tok/s   82.478   86.093   92.031   92.749   91.594
        // 4096 rather than the marginally better 3072: the peak is broad, 3072 is worth 0.8% on
        // ONE prompt, and reusing the constant the scored regime already uses is worth more than
        // that. So a single threshold now covers everything from 12288 up.
        if (total_ctx >= kMidCtxMinSeq)
            kFullWindow = total_ctx < kMidCtxMaxSeq ? kMidCtxWindow : kLongCtxWindow;
    }
    // A draft with no full-attention layer (DFlash2: every layer sliding) has nothing for this
    // window to bound, and the fc trim below takes the widest window across layers: 3x the sliding
    // window there reached past the rows dflash_generate captures above 12288 (the last 4096), so
    // every long-context block failed. Its widest window is the sliding one.
    if (kFullWindowEnv < 0 && c.sliding_window > 0 &&
        std::all_of(c.sliding_layers.begin(), c.sliding_layers.end(), [](bool b) { return b; }))
        kFullWindow = c.sliding_window;
    // fc trim: once the full-attn layer is windowed, NO layer reads target_proj older than the
    // largest window across layers, so project only that tail. Uses the same attn_gqa_kv_lo bound
    // the per-layer ingestion (#752) applies, at the LARGEST window -> surviving rows byte-identical,
    // and every skipped row is one no layer reads. kFullWindow==0 -> fc_skip 0 (unchanged).
    int fc_skip = 0;
    {
        static const int kCT = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_TRIM");
                                   return (e && e[0] == '0') ? 0 : 1; }();
        if (ctx_len > 0 && kCT && kFullWindow > 0) {
            const int maxwin = c.sliding_window > kFullWindow ? c.sliding_window : kFullWindow;
            const int lo = dflash_kernels::attn_gqa_kv_lo(BW, past + ctx_len + BW, c.n_q_heads,
                                                          c.n_kv_heads, d, pos0, 0, maxwin);
            fc_skip = lo > past ? (lo - past < ctx_len ? lo - past : ctx_len) : 0;
        }
        fc_skip = std::max(fc_skip, floor_skip);
    }
    const int fc_rows = ctx_len - fc_skip;
    // Route the two remaining BF16 weight reads in this block -- the projector above and the
    // per-layer context K/V below -- through the Q4 copies. 0 restores bf16 for both.
    static const bool kCtxQ4 = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_Q4");
                                   return !(e && e[0] == '0'); }();
    if (fc_rows > 0) {
        if (fc_skip < target_hidden_start) return false;
        // A windowed slot's target_proj starts at the first row projected: a long prompt's first
        // block keeps ~a window of its rows, not all of them.
        if (s.win_rows > 0 && fc_rows > s.cap) return false;
        const bf16* th = (const bf16*)target_hidden +
                         (size_t)(fc_skip - target_hidden_start) * n_cap * H;
        bf16* tp = s.target_proj + (size_t)(s.win_rows > 0 ? 0 : fc_skip) * H;
        const bool fc_q4 = kCtxQ4 && s.q8_fc.q4 && fc_rows >= 1 && fc_rows <= 8;
        if (ctx_gemm) {
            kernels::launch_prefill_gemm(th, s.fc, tp, fc_rows, H, n_cap * H, st);
        } else if (fc_q4 && dp4a_ok) {
            dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                q81n(th, n_cap * H, fc_rows), s.q8_fc.q4, nullptr, nullptr,
                s.q8_fc.dm, nullptr, nullptr,
                tp, nullptr, nullptr, H, 0, 0, n_cap * H, st, fc_rows);
        } else if (fc_q4) {
            dflash_kernels::launch_gemv_batched_q4_fused3(
                th, s.q8_fc.q4, nullptr, nullptr, s.q8_fc.dm, nullptr, nullptr,
                tp, nullptr, nullptr, H, 0, 0, n_cap * H, st, fc_rows);
        } else if (fc_rows > 1) {
            dflash_kernels::launch_gemv_rows_batched(th, s.fc, tp, fc_rows, H, n_cap * H, st);
        } else {
            kernels::launch_gemv(th, s.fc, tp, H, n_cap * H, st);
        }
        dflash_kernels::launch_rms(tp, s.hidden_norm, tp, fc_rows, H, c.rms_eps, st);
        // Ablation (SPARKINFER_DFLASH_ZERO_CTX=1): blank the projected target features after
        // computing them. The draft then attends over an all-zero context while everything else --
        // shapes, positions, RoPE, the block forward, the Markov chain -- is untouched. If tau is
        // unchanged the draft is IGNORING the target context, which localises the acceptance gap to
        // the fc/hidden_norm/KV-injection path rather than to the draft backbone or the head.
        if (getenv("SPARKINFER_DFLASH_ZERO_CTX"))
            cu(cudaMemsetAsync(tp, 0, (size_t)fc_rows * H * sizeof(bf16), st), "zero ctx");
    }

    // x = noise embedding
    cu(cudaMemcpyAsync(s.x, s.noise, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
       "noise->x");

    static const int active_layers = [] {
        const char* e = getenv("SPARKINFER_DFLASH_LAYERS");
        return e ? atoi(e) : 0;
    }();
    const int run_layers = active_layers > 0 ? std::min(active_layers, c.n_layers) : c.n_layers;
    // cat(k_ctx, k_noise) is built at exactly the layout and length the cache slice expects, so
    // staging it in k_new/v_new and memcpy'ing it in cost two extra D2D launches per layer -- 12 a
    // block, on a draft that is launch-bound (its kernels are 1-3 us and the gaps between them are
    // the same size). Project straight into the cache slice instead; the RoPE then runs in place
    // there. Same values written to the same addresses in the same order.
    const int new_len_all = ctx_len + BW;
    if (past + new_len_all - s.base > s.cap) {
        fprintf(stderr, "[dflash] KV overflow past=%d new=%d base=%d max=%d\n", past, new_len_all, s.base, s.cap);
        return false;
    }
    // Attention geometry for this block, hoisted: the context ingestion below needs the layer's
    // window and the block's key span to know which context rows the attention can still reach.
    const int kv_len_for_block = past + new_len_all;
    const int q_pos0_for_block = pos0;
    for (int L = 0; L < run_layers; L++) {
        const LayerWeights& w = s.layers[L];
        int window_of_layer = (L < (int)c.sliding_layers.size() && c.sliding_layers[L])
                                        ? c.sliding_window : 0;
        if (window_of_layer == 0 && kFullWindow > 0) window_of_layer = kFullWindow;
        // Position p lives at row p - base (base 0 unless a windowed slot has slid). After a long
        // prompt's first block starts at the window, past < base: the rows below it are never written.
        bf16* const kdst = s.k_cache[L] + ((ptrdiff_t)past - s.base) * kvdim;
        bf16* const vdst = s.v_cache[L] + ((ptrdiff_t)past - s.base) * kvdim;
        if (L == 0)
            dflash_kernels::launch_rms(s.x, w.input_norm, s.xn, BW, H, c.rms_eps, st);
        // DFlash2: the attention conv's "prepare" (z-lab GroupedDynamicCausalConv). A per-group
        // kernel correction is projected from the normed block rows, and the conv runs along the
        // block's rows; Q and the block's own K/V come from its output. The context rows (the
        // target's projected features) are not convolved.
        const int conv_dyn_n = c.dflash2 ? 2 * c.conv_kernel * (H / c.conv_group) : 0;
        if (conv_dyn_n) {
            dflash_kernels::launch_gemv_batched16(s.xn, w.attn_conv_proj, s.conv_dyn_a, conv_dyn_n, H, st, BW);
            dflash_kernels::launch_grouped_conv(s.xn, w.attn_conv_base, s.conv_dyn_a, 0, s.conv_out, BW, H,
                                                c.conv_kernel, c.conv_group, conv_dyn_n, st);
            cu(cudaMemcpyAsync(s.xn, s.conv_out, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
               "dflash2 attn conv");
            xn_ready = false;
        }

        // Q from noise, K/V from cat(target, noise)
        if (fast16) {
            if (w.q8_wq.q4 && dp4a_qkv)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    xn_ready ? s.xq81 : q81(s.xn, H), w.q8_wq.q4, w.q8_wk.q4, w.q8_wv.q4,
                    w.q8_wq.dm, w.q8_wk.dm, w.q8_wv.dm,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else if (w.q8_wq.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    s.xn, w.q8_wq.q4, w.q8_wk.q4, w.q8_wv.q4, w.q8_wq.dm, w.q8_wk.dm, w.q8_wv.dm,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else if (w.q8_wq.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    s.xn, w.q8_wq.q, w.q8_wk.q, w.q8_wv.q, w.q8_wq.s, w.q8_wk.s, w.q8_wv.s,
                    s.q, kdst + (size_t)ctx_len * kvdim, vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
            else
                dflash_kernels::launch_gemv_batched16_fused3(
                    s.xn, w.wq, w.wk, w.wv,
                    s.q, kdst + (size_t)ctx_len * kvdim,
                    vdst + (size_t)ctx_len * kvdim,
                    qdim, kvdim, kvdim, H, st, BW);
        } else {
            for (int t = 0; t < BW; t++)
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wq, s.q + (size_t)t * qdim, qdim, H, st);
        }
        xn_ready = false;   // consumed above; xq81 is reused by the projections that follow
        // Context half of cat(k_ctx, k_noise), written ahead of the noise rows.
        //
        // A windowed layer never reads a key older than `window` before the query, and
        // launch_attn_gqa now starts its partition above that bound rather than masking it inside
        // the loop -- so on a long prompt the leading context rows are projected into the cache and
        // then never touched. Skip producing them. The bound comes from attn_gqa_kv_lo, the same
        // function the launcher uses, evaluated at THIS block: q_pos0 only grows as the generation
        // proceeds, and the one-split-worth guard it applies gets easier to satisfy as it does
        // (the bound climbs a key per token while a split's width climbs a fraction of one), so a
        // row dead for this block is dead for every later block too. Full-attention layers and
        // every steady-state block (where the window still covers the cache) get skip == 0 and are
        // byte-for-byte unchanged.
        static const int kCtxTrim = []{ const char* e = getenv("SPARKINFER_DFLASH_CTX_TRIM");
                                        return (e && e[0] == '0') ? 0 : 1; }();
        const int ctx_kv_lo = (ctx_len > 0 && kCtxTrim)
            ? dflash_kernels::attn_gqa_kv_lo(BW, kv_len_for_block, c.n_q_heads, c.n_kv_heads, d,
                                             q_pos0_for_block, /*k_pos0=*/0, window_of_layer)
            : 0;
        const int ctx_skip = std::max(floor_skip, ctx_kv_lo > past ? std::min(ctx_kv_lo - past, ctx_len) : 0);
        const int ctx_rows = ctx_len - ctx_skip;
        if (s.win_rows > 0 && ctx_skip < fc_skip) return false;   // rows the fc did not project
        const bf16* const ctx_src = s.target_proj + (size_t)(ctx_skip - (s.win_rows > 0 ? fc_skip : 0)) * H;
        bf16* const kdst_ctx = kdst + (size_t)ctx_skip * kvdim;
        bf16* const vdst_ctx = vdst + (size_t)ctx_skip * kvdim;
        // The steady-state context rows went through the BF16 wk/wv while every other projection
        // in this block -- including the noise rows' own K/V, from the same two matrices -- goes
        // through the Q4 copies. That is 21 MB of weights per layer against 2.6 MB, on a path that
        // runs 1-5 rows and is therefore purely weight-bound.
        //
        // It was left bf16 out of caution about the context being the one place the TARGET's
        // information enters the draft. Measured, that caution is unfounded: sweeping the draft's
        // whole backbone precision (SPARKINFER_DFLASH_WBITS) does not move acceptance in the
        // direction precision would predict. At 512 generated tokens, MORE precision gives LOWER
        // acceptance -- Q4 tau 2.3733, Q8 tau 2.3624 for +0.47 ms of draft -- so what is being
        // read is numeric luck, not draft quality. Judge changes on this path by `draft ms/call`
        // and `batched ms/call`, which are deterministic to ~0.005 ms; a tau delta under ~2% on a
        // single prompt says nothing. The draft is insensitive to weight precision, so the context
        // rows may use the same Q4 copies the block rows do.
        //
        // The wide first-block case (ctx_gemm) stays on the bf16 tensor-core GEMM: at 4096 rows it
        // is COMPUTE bound and already runs at ~169 TFLOPS, ~80% of this card's bf16 peak, where a
        // dequantising path would only add work. SPARKINFER_DFLASH_CTX_Q4=0 restores bf16.
        // ctx_rows is the ACCEPTED length, so 1 is its modal value -- and the single-row case was
        // the most expensive of all, two separate bf16 GEMVs each streaming the whole 10.5 MB
        // matrix to produce one row. Cover 1..8, not just the multi-row tiers.
        const bool ctx_q4 = kCtxQ4 && w.q8_wk.q4 && w.q8_wv.q4 && ctx_rows >= 1 && ctx_rows <= 8;
        if (ctx_rows > 0 && ctx_gemm) {
            kernels::launch_prefill_gemm(ctx_src, w.wk, kdst_ctx, ctx_rows, kvdim, H, st);
            kernels::launch_prefill_gemm(ctx_src, w.wv, vdst_ctx, ctx_rows, kvdim, H, st);
        } else if (ctx_q4 && dp4a_ok) {
            dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                q81n(ctx_src, H, ctx_rows), w.q8_wk.q4, w.q8_wv.q4, nullptr,
                w.q8_wk.dm, w.q8_wv.dm, nullptr,
                kdst_ctx, vdst_ctx, nullptr, kvdim, kvdim, 0, H, st, ctx_rows);
        } else if (ctx_q4) {
            // Same fused pair the noise rows use; y0/y1 are written as [row][kvdim], which is
            // exactly the cache slice's layout.
            dflash_kernels::launch_gemv_batched_q4_fused3(
                ctx_src, w.q8_wk.q4, w.q8_wv.q4, nullptr, w.q8_wk.dm, w.q8_wv.dm, nullptr,
                kdst_ctx, vdst_ctx, nullptr, kvdim, kvdim, 0, H, st, ctx_rows);
        } else if (ctx_rows > 1) {
            dflash_kernels::launch_gemv_rows_exact_fused2(
                ctx_src, w.wk, w.wv, kdst_ctx, vdst_ctx,
                ctx_rows, kvdim, kvdim, H, st);
        } else if (ctx_rows == 1) {
            kernels::launch_gemv(ctx_src, w.wk, kdst_ctx, kvdim, H, st);
            kernels::launch_gemv(ctx_src, w.wv, vdst_ctx, kvdim, H, st);
        }
        if (!fast16) {
            for (int t = 0; t < BW; t++) {
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wk,
                                     kdst + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
                kernels::launch_gemv(s.xn + (size_t)t * H, w.wv,
                                     vdst + (size_t)(ctx_len + t) * kvdim, kvdim, H, st);
            }
        }

        const int new_len = ctx_len + BW;
        // Q / K RMSNorm per head


        // RoPE: Q at positions past..(past+B) if past≈pos0 for noise-only positions.
        // Match reference: position_ids cover past_len .. start+block_size for the cat length.
        // k positions: past .. past+new_len-1 when past==seq_len and we're appending.
        // Absolute: k_pos0 = pos0 - ctx_len (context features align with tokens just before noise),
        // q_pos0 = pos0.
        const int k_pos0 = pos0 - ctx_len;
        const int q_pos0 = pos0;
        // c.rope_normal selects consecutive-pair ("normal"/LLAMA_ROPE_TYPE_NORM) RoPE instead of
        // the NeoX split-half pairing every other DFlash draft checkpoint (Qwen3.6) uses -- see
        // DFlashDraftConfig::rope_normal and k_rms_heads_rope_normal (dflash_kernels.cu) for why
        // Muse Glimmer's draft sets this. Default false leaves Qwen3.6 byte-for-byte unchanged.
        if (c.rope_normal) {
            dflash_kernels::launch_rms_heads_rope_normal(s.q, w.q_norm, BW, c.n_q_heads, d,
                                                         c.rms_eps, q_pos0, c.rope_theta, st);
            // Norm+RoPE only the rows that were actually produced above; the skipped context
            // prefix holds no K to normalise. Row i of this slice keeps its own absolute
            // position, so the surviving rows get exactly the angles they got before.
            dflash_kernels::launch_rms_heads_rope_normal(kdst + (size_t)ctx_skip * kvdim, w.k_norm,
                                                         new_len - ctx_skip, c.n_kv_heads, d,
                                                         c.rms_eps, k_pos0 + ctx_skip,
                                                         c.rope_theta, st);
        } else {
            // s.d_yarn_inv_freq is null unless the checkpoint configures rope_type "yarn", in
            // which case these two calls are the ONLY behavioural change -- Q and K must be
            // rotated with the same table or their relative phase is wrong.
            dflash_kernels::launch_rms_heads_rope(s.q, w.q_norm, BW, c.n_q_heads, d, c.rms_eps,
                                                 q_pos0, c.rope_theta, st,
                                                 s.d_yarn_inv_freq, s.yarn_att_scale);
            dflash_kernels::launch_rms_heads_rope(kdst + (size_t)ctx_skip * kvdim, w.k_norm,
                                                 new_len - ctx_skip, c.n_kv_heads, d,
                                                 c.rms_eps, k_pos0 + ctx_skip, c.rope_theta, st,
                                                 s.d_yarn_inv_freq, s.yarn_att_scale);
        }

        // K/V are already in the cache at offset `past` -- attend over the full past+new.
        const int kv_len = past + new_len;
        const int window = window_of_layer;
        static int mixed_causal = [] {
            const char* e = getenv("SPARKINFER_DFLASH_MIXED_CAUSAL");
            return (!e || e[0] != '0') ? 1 : 0;
        }();
        // Full-attention DFlash layers are ENCODER_ONLY in the reference/SGLang implementation
        // unless the checkpoint explicitly declares is_causal. This checkpoint does not, so its
        // draft block is bidirectional: row i sees the later mask-token rows it was trained with.
        // SparkInfer used to force every layer causal by default after an ablation made against
        // the old, displaced Markov-logit rows. Repeating the comparison after fixing that row
        // mapping reverses the decision: reference semantics raise tau 1.561 -> 1.600 and decode
        // 106.70 -> 109.36 tok/s at depth 1, with identical 1.999 ms draft cost and lossless output.
        // Sliding layers remain causal through mixed_causal below. The override is retained for
        // checkpoints that need it: SPARKINFER_DFLASH_FORCE_CAUSAL=1 forces every layer causal.
        static const int kForceCausal = []{
            const char* e = getenv("SPARKINFER_DFLASH_FORCE_CAUSAL");
            return (e && e[0] == '1') ? 1 : 0;
        }();
        // A checkpoint that declares is_causal (DFlash2: false, on sliding layers) is taken at its
        // word, as the reference implementations do; the per-layer default applies otherwise.
        const bool causal = kForceCausal ||
                            (c.is_causal >= 0 ? c.is_causal != 0
                                              : (mixed_causal && L < (int)c.sliding_layers.size() &&
                                                 c.sliding_layers[L]));
        // Keys start at the context floor (0 unless a prefix-cache hit left the start unprojected):
        // the cache slice from there, with its first key at that position.
        // Once a windowed slot has slid, the keys below `base` are gone, so the span starts at the
        // window itself: the range the kernel trimmed to before (it does so past ~2.2K keys, and
        // a slot slides only past its ~4K rows), so the same keys and the same splits.
        const int kv0 = s.base > 0 ? std::max(s.ctx_floor, pos0 - window + 1)
                                   : std::min(s.ctx_floor, kv_len);
        dflash_kernels::launch_attn_gqa(s.q, s.k_cache[L] + ((ptrdiff_t)kv0 - s.base) * kvdim,
                                        s.v_cache[L] + ((ptrdiff_t)kv0 - s.base) * kvdim, s.attn,
                                        BW, kv_len - kv0, c.n_q_heads, c.n_kv_heads, d,
                                        q_pos0, /*k_pos0_cache=*/kv0, window, causal, scale, st,
                                        s.fa_m, s.fa_l, s.fa_acc);

        if (fast16) {
            if (w.q8_wo.q4 && dp4a_o)
                dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                    q81(s.attn, qdim), w.q8_wo.q4, nullptr, nullptr,
                    w.q8_wo.dm, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else if (w.q8_wo.q4)
                dflash_kernels::launch_gemv_batched_q4_fused3(
                    s.attn, w.q8_wo.q4, nullptr, nullptr, w.q8_wo.dm, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else if (w.q8_wo.q)
                dflash_kernels::launch_gemv_batched_q8_fused3(
                    s.attn, w.q8_wo.q, nullptr, nullptr, w.q8_wo.s, nullptr, nullptr,
                    s.ao, nullptr, nullptr, H, 0, 0, qdim, st, BW);
            else
                dflash_kernels::launch_gemv_batched16(s.attn, w.wo, s.ao, H, qdim, st, BW);
        } else {
            for (int t = 0; t < BW; t++)
                kernels::launch_gemv(s.attn + (size_t)t * qdim, w.wo, s.ao + (size_t)t * H, H, qdim, st);
        }
        if (conv_dyn_n) {   // DFlash2: the attention conv's "finish", on the output projection
            dflash_kernels::launch_grouped_conv(s.ao, w.attn_conv_base + (size_t)c.conv_kernel * H,
                                                s.conv_dyn_a, 1, s.conv_out, BW, H, c.conv_kernel,
                                                c.conv_group, conv_dyn_n, st);
            cu(cudaMemcpyAsync(s.ao, s.conv_out, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
               "dflash2 attn conv finish");
        }
        dflash_kernels::launch_add_rms(s.x, s.ao, s.h, w.post_norm, s.hn, BW, H, c.rms_eps, st,
                                       dp4a_gu ? s.xq81 : nullptr);
        hn_ready = dp4a_gu;
        if (conv_dyn_n) {   // DFlash2: the MLP conv's "prepare", on the post-attention norm
            dflash_kernels::launch_gemv_batched16(s.hn, w.mlp_conv_proj, s.conv_dyn_m, conv_dyn_n, H, st, BW);
            dflash_kernels::launch_grouped_conv(s.hn, w.mlp_conv_base, s.conv_dyn_m, 0, s.conv_out, BW, H,
                                                c.conv_kernel, c.conv_group, conv_dyn_n, st);
            cu(cudaMemcpyAsync(s.hn, s.conv_out, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
               "dflash2 mlp conv");
            hn_ready = false;
        }
        // DFlash2: the MLP as block-scaled NVFP4 GEMMs (see ensure_quant); the SwiGLU is folded into
        // the down projection's A quantize. BW is a multiple of 8 there, as the A operand needs.
        bool mlp_done = false;
        if (w.fp4_gate && s.f4_a && BW % 8 == 0 && BW <= Impl::kBatchRows) {
            mlp_done = kernels::launch_prefill_nvfp4_quant_a(s.hn, s.f4_a, s.f4_asf, BW, H, st) &&
                       kernels::launch_prefill_nvfp4_gemm(s.f4_a, s.f4_asf, w.fp4_gate, w.fp4_gate_sf,
                                                          s.gate, BW, I, H, s.f4_ws, st) &&
                       kernels::launch_prefill_nvfp4_gemm(s.f4_a, s.f4_asf, w.fp4_up, w.fp4_up_sf,
                                                          s.up, BW, I, H, s.f4_ws, st) &&
                       kernels::launch_prefill_nvfp4_swiglu_quant_a(s.gate, s.up, s.f4_a, s.f4_asf,
                                                                    BW, I, st) &&
                       kernels::launch_prefill_nvfp4_gemm(s.f4_a, s.f4_asf, w.fp4_down, w.fp4_down_sf,
                                                          s.down, BW, H, I, s.f4_ws, st);
            if (!mlp_done) return false;   // its Q4 copies were never made
        }
        if (!mlp_done) {
            if (fast16) {
                if (w.q8_gate.q4 && dp4a_gu)
                    dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                        hn_ready ? s.xq81 : q81(s.hn, H), w.q8_gate.q4, w.q8_up.q4, nullptr,
                        w.q8_gate.dm, w.q8_up.dm, nullptr,
                        s.gate, s.up, nullptr, I, I, 0, H, st, BW);
                else if (w.q8_gate.q4)
                    dflash_kernels::launch_gemv_batched_q4_fused3(
                        s.hn, w.q8_gate.q4, w.q8_up.q4, nullptr, w.q8_gate.dm, w.q8_up.dm, nullptr,
                        s.gate, s.up, nullptr, I, I, 0, H, st, BW);
                else if (w.q8_gate.q)
                    dflash_kernels::launch_gemv_batched_q8_fused3(
                        s.hn, w.q8_gate.q, w.q8_up.q, nullptr, w.q8_gate.s, w.q8_up.s, nullptr,
                        s.gate, s.up, nullptr, I, I, 0, H, st, BW);
                else
                    dflash_kernels::launch_gemv_batched16_fused2(
                        s.hn, w.gate, w.up, s.gate, s.up, I, I, H, st, BW);
            } else {
                for (int t = 0; t < BW; t++) {
                    kernels::launch_gemv(s.hn + (size_t)t * H, w.gate, s.gate + (size_t)t * I, I, H, st);
                    kernels::launch_gemv(s.hn + (size_t)t * H, w.up,   s.up   + (size_t)t * I, I, H, st);
                }
            }
            dflash_kernels::launch_swiglu(s.gate, s.up, s.gate, BW * I, st);
            if (fast16) {
                if (w.q8_down.q4 && dp4a_down)
                    dflash_kernels::launch_gemv_batched_q4_dp4a_fused3(
                        q81(s.gate, I), w.q8_down.q4, nullptr, nullptr,
                        w.q8_down.dm, nullptr, nullptr,
                        s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
                else if (w.q8_down.q4)
                    dflash_kernels::launch_gemv_batched_q4_fused3(
                        s.gate, w.q8_down.q4, nullptr, nullptr, w.q8_down.dm, nullptr, nullptr,
                        s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
                else if (w.q8_down.q)
                    dflash_kernels::launch_gemv_batched_q8_fused3(
                        s.gate, w.q8_down.q, nullptr, nullptr, w.q8_down.s, nullptr, nullptr,
                        s.down, nullptr, nullptr, H, 0, 0, I, st, BW);
                else
                    dflash_kernels::launch_gemv_batched16(s.gate, w.down, s.down, H, I, st, BW);
            } else {
                for (int t = 0; t < BW; t++)
                    kernels::launch_gemv(s.gate + (size_t)t * I, w.down, s.down + (size_t)t * H, H, I, st);
            }
        }
        // Fold the second residual into the norm that always consumes it: the next layer's input
        // norm, or the final norm after the last layer. Same math, one launch instead of two, and
        // the draft is eager-launched so each saved launch is also a saved gap.
        if (conv_dyn_n) {   // DFlash2: the MLP conv's "finish", on the down projection
            dflash_kernels::launch_grouped_conv(s.down, w.mlp_conv_base + (size_t)c.conv_kernel * H,
                                                s.conv_dyn_m, 1, s.conv_out, BW, H, c.conv_kernel,
                                                c.conv_group, conv_dyn_n, st);
            cu(cudaMemcpyAsync(s.down, s.conv_out, (size_t)BW * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
               "dflash2 mlp conv finish");
        }
        const bf16* next_norm = (L + 1 < run_layers) ? s.layers[L + 1].input_norm : s.final_norm;
        dflash_kernels::launch_add_rms(s.h, s.down, s.x, next_norm, s.xn, BW, H, c.rms_eps, st,
                                       dp4a_qkv ? s.xq81 : nullptr);
        xn_ready = dp4a_qkv;
        // Per-layer residual stream, for the differential (see the dump block below). s.x is the
        // layer's output residual; comparing it layer by layer turns "the backbone diverges"
        // into "layer N diverges", which is the difference between a search and a fix.
        if (const char* dd = getenv("SPARKINFER_DSPARK_DUMP")) {
            static int dumped_layers = 0;
            if (dumped_layers <= L) {
                dumped_layers = L + 1;
                cudaStreamSynchronize(st);
                std::vector<bf16> host((size_t)BW * H);
                cudaMemcpy(host.data(), s.x, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
                char path[512];
                snprintf(path, sizeof(path), "%s/layer%d_x.bin", dd, L);
                if (FILE* f = fopen(path, "wb")) {
                    fwrite(host.data(), sizeof(bf16), host.size(), f);
                    fclose(f);
                }
            }
        }
    }
    if (run_layers <= 0)
        dflash_kernels::launch_rms(s.x, s.final_norm, s.xn, BW, H, c.rms_eps, st);

    // LM head (target weights) -> logits / argmax. Batched over the whole block: one batched
    // GEMV against the eagerly dequantized bf16 lm_head cache instead of a per-token
    // quantized GEMV loop.
    const int V = s.vocab > 0 ? s.vocab : c.vocab;
    // DRAFT OUTPUT VOCABULARY (SPARKINFER_DFLASH_DRAFT_VOCAB, 0 = full).
    //
    // Everything downstream of the backbone is proportional to the vocabulary, and on Qwen3.8 that
    // vocabulary is 248320. Per block the head streams the target's Q4_K LM head once (715 MB) and
    // then the Markov chain re-reads w2 -- [248320, 256] int8 plus its per-32 scales, ~71 MB --
    // ONCE PER PROPOSAL ROW, serially. Between them they are the two largest items in the draft,
    // and #913 made that worse in the direction that matters here: at depth 4 the chain runs four
    // rows, so the head moves 715 + 4 x 71 = ~1001 MB a block.
    //
    // Both are NOMINATION-only: a token this head cannot score is simply never proposed, which
    // costs at most one acceptance and can never change what is emitted, because every emitted
    // token is still a target argmax over the FULL vocabulary. So the head may score a prefix.
    //
    // The rows are contiguous in both tables ([vocab, ...], row-major), so a prefix is just a
    // smaller length -- no repack, no second copy, no extra VRAM. Qwen3.8's vocabulary has no
    // reserved tail to reclaim (only 243 of the 248320 ids are undefined), but it IS
    // frequency-ordered, and coverage plateaus hard. Fraction of tokens with id < K:
    //
    //     K        bench_prompt_4k   repo prose   repo C++     LM head   w2+scales
    //     32768        92.61%          90.86%      93.77%        94 MB      9.4 MB
    //     65536        97.53%          96.64%      97.92%       189 MB     18.7 MB
    //     98304        99.95%          99.23%     100.00%       283 MB     28.1 MB
    //    163840        99.95%          99.38%     100.00%       472 MB     46.8 MB
    //
    // Three unrelated corpora agree on the shape: the ids above ~100k are rare scripts and exotic
    // unicode, not this text, and by 65536 the curve has already flattened to within 2.5% of it.
    //
    // 65536 is NOT the throughput peak. Swept end to end at ctx 4096, one binary, tok/s and mean
    // accept, with the draft's own per-block cost:
    //
    //     K         DSPARK tok/s   draft ms/block   mean accept
    //     248320      122.87           2.575          1.8286
    //     131072      125.90           2.250          1.8286
    //      98304      125.98           2.159          1.8286
    //      65536      127.08           2.075          1.8286
    //      49152      127.84           2.03           1.8286   <- peak
    //      32768      125.90           1.99           1.8028
    //      24576      126.42           1.96           1.8028
    //      16384      125.00           1.94           1.7778
    //
    // Acceptance is flat down to 49152 and turns over below it: at 32768 the generation needs an
    // extra step (128 tokens in 71 instead of 70), which is what mean accept 1.8286 -> 1.8028 is.
    // So 49152 sits ONE step from that edge, and its 0.6% advantage over 65536 was found by
    // walking the prompt this benchmark scores. 65536 is taken off the coverage curve instead --
    // the one three unrelated corpora agree on -- and keeps a full step of margin. Giving up 0.6%
    // is the right side to err on for a change whose entire safety argument is coverage.
    //
    // The Markov head's OTHER table, w1, is indexed by the PREVIOUS token and is not pruned: the
    // conditioning still accepts any target token, only the output side is narrowed.
    static const int kDraftVocab = draft_vocab_env();
    // Vd is the logits ROW STRIDE as well as the length: the multi-row head writes y[m*N + n], so
    // the stride has to travel with it or the Markov chain reads the wrong row.
    const int Vd = (kDraftVocab > 0 && kDraftVocab < V) ? kDraftVocab : V;
    // The batched bf16 head (#661) still streams the DEQUANTIZED head every step: at V=248k,
    // K=2048 that is ~1.0 GB per pass versus ~416 MB for the native Q6_K bytes, and it pins ~1 GB
    // of VRAM for the bf16 cache. Prefer a multi-row Q6_K MMVQ that keeps the head quantized: each
    // warp owns one output row, walks its superblocks once and accumulates all B dot products, so
    // the weight streams from HBM a single time in its compact form.
    // SPARKINFER_DFLASH_HEAD_MULTIROW=0 falls back to the bf16 batched path.
    static int head_mr = -1;
    if (head_mr < 0) { const char* e = getenv("SPARKINFER_DFLASH_HEAD_MULTIROW"); head_mr = (e && e[0] == '0') ? 0 : 1; }
    // SGLang consumes base-logit rows [0, depth). Keep the legacy [1, depth+1) producer and
    // consumer together behind one A/B switch; changing only one side reads the wrong (or, on the
    // multi-row fast path, uninitialised) logits and is not a meaningful comparison.
    static const int kRowShift = []{
        const char* e = getenv("SPARKINFER_DFLASH_ROW_SHIFT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // DFlash2 reads its proposals from rows 1..depth (z-lab: draft_hidden[:, 1-verify_size:]),
    // whatever the DSpark row-shift default says.
    const int head_row0 = (kRowShift && !c.dflash2) ? 0 : 1;
    bool head_done = false;
    // Coupled proposals (set_sampling): proposal r draws at step step0 + r - 1, the step the
    // target's verify row r - 1 samples at. Row 0 (the seed) is never sampled.
    const bool couple = s.samp_on && std::max(B, kProposalDepth) < Impl::kSampRows;
    if (couple) {
        for (int r = 0; r <= std::max(B, kProposalDepth); r++) {
            s.h_samp->temp[r] = r == 0 ? 0.f : s.samp_temp;
            s.h_samp->seed[r] = s.samp_seed;
            s.h_samp->step[r] = r == 0 ? 0ull : s.samp_step0 + (unsigned long long)(r - 1);
            s.h_samp->top_k[r] = s.samp_top_k;
            s.h_samp->top_p[r] = s.samp_top_p;
        }
        cu(cudaMemcpyAsync(s.d_samp, s.h_samp, sizeof(Impl::DraftSampling), cudaMemcpyHostToDevice, st),
           "draft sampling params");
    }
    if (head_mr && s.head_q8 && (s.lm_head_type == 14 || s.lm_head_type == 12)) {
        // Score only the proposal rows the verifier can consume. One row-batched quantize launch
        // instead of kProposalDepth tiny ones (8 CTAs each, so launch latency dominated them).
        // DSpark's Markov chain consumes base-logit rows [0, depth): row 0 plus the anchor token
        // produces proposal 1, row 1 plus proposal 1 produces proposal 2, and so on. Quantizing
        // rows [1, depth+1) here was the other half of the legacy row displacement below: merely
        // fixing the consumer would otherwise make row 0 read uninitialised logits on this fast
        // multi-row head path.
        kernels::launch_quantize_q8_1_rows(s.xn + (size_t)head_row0 * H,
                                           s.head_q8, H, kProposalDepth, H, st);
        // Prefer the Q4_K head when one is bound: this kernel already runs near HBM peak, so its
        // runtime IS its weight bytes -- ~280 MB in Q4_K against ~417 MB in Q6_K at V=248k.
        static const bool head_i8 = [] {
            const char* e = getenv("SPARKINFER_DFLASH_HEAD_I8");
            return !(e && e[0] == '0');
        }();
        if (s.lm_head_type == 12 && s.lm_head_i4)
            head_done = kernels::launch_gemv_i4_q81_multirow_f32(
                s.head_q8, s.lm_head_i4, s.lm_head_i4_scale,
                s.logits + (size_t)head_row0 * Vd, Vd, H, kProposalDepth, st);
        else if (s.lm_head_type == 12 && head_i8 && s.lm_head_i8)
            head_done = kernels::launch_gemv_i8_q81_multirow_f32(
                s.head_q8, s.lm_head_i8, s.lm_head_i8_scale,
                s.logits + (size_t)head_row0 * Vd, Vd, H, kProposalDepth, st);
        else if (s.lm_head_type == 12)
            head_done = kernels::launch_gemv_q4k_dp4a_multirow_f32(
                s.head_q8, s.lm_head, s.logits + (size_t)head_row0 * Vd,
                Vd, H, kProposalDepth, st);
        else
            head_done = kernels::launch_gemv_q6k_dp4a_multirow_f32(
                s.head_q8, s.lm_head, s.logits + (size_t)head_row0 * Vd,
                Vd, H, kProposalDepth, st);
    }
    if (!head_done) {
    if (BW == 16) {
        dflash_kernels::launch_gemv_batched16_f32(s.xn, s.lm_head_bf16, s.logits, Vd, H, st);
    } else {
        for (int t = 0; t < B; t++) {
            const bf16* row = s.xn + (size_t)t * H;
            float* logit_row = s.logits + (size_t)t * Vd;
            if (s.lm_head_type)
                kernels::launch_gemv_q_f32(row, s.lm_head, s.lm_head_type, logit_row, Vd, H, st);
            else
                kernels::launch_gemv_f32(row, s.lm_head, logit_row, Vd, H, st);
        }
    }
    }
    // DSpark's Markov head: position r's logit bias conditions on the token AT position r itself
    // (the anchor for r==0, its own -- just-chosen -- prediction for r>0), predicting position
    // r+1. That token is only known once position r-1's own Markov-corrected argmax has run, so
    // proposals 1..kProposalDepth chain sequentially instead of the single batched argmax below --
    // each launch reads its "previous token" from device memory (s.d_ids[0], the real anchor, for
    // r==1; this same loop's own s.d_out[r-1] for r>1), so the chain needs no host round-trip
    // until the final readback already below. Base row 0 is consumed to form proposal 1; d_out[0]
    // itself is not a proposal because the caller already owns the target-produced anchor token.
    // MARKOV HEAD: ALWAYS ON. It was gated on sequence length (#868, threshold 12288) because it
    // "raises acceptance and costs more than the acceptance is worth below long context". That was
    // measured inside the regime the gate itself creates, and the conclusion was wrong.
    //
    // The head is not an optimisation layered on DSpark -- it IS DSpark. The algorithm is a
    // semi-autoregressive block drafter: one forward emits the gamma-token block, and a lightweight
    // sequential head conditions each position on the PREVIOUS drafted token. Without it the block
    // forward alone cannot differentiate positions and the draft degenerates into repeating one
    // high-frequency token. Measured at ctx=4096 with SPARKINFER_DSPARK_PROBE=1:
    //
    //   off: draft=[13 13 13 13 13 13]   ("." six times)                      accept 0-1
    //   on:  draft=[6337 421 369 524 279 1788]
    //        target=[6337 421 369 524 177460]                                 accept 4
    //
    // Over a generation at ctx=4096, depth 6: tau 1.085 -> 1.600. That also explains why tau used
    // to be INVARIANT to proposal depth (1.076 / 1.085 / 1.085 at depth 1 / 3 / 6) -- positions past
    // the first were never going to be accepted, so drafting more of them changed nothing.
    //
    // The head genuinely costs ~1.41 ms of a 16.3 ms step: w2 is [vocab, rank] = 248320 x 256 bf16
    // = 127 MB re-read per proposal row, and the chain serialises bias -> confidence -> argmax per
    // row. That cost repays only through the BATCHED verify, where accepting k tokens costs ONE
    // target forward instead of k. On the token loop it cannot repay at any tau, because that path
    // runs one target forward per kept token exactly as AR does -- which is what the original
    // measurement was really observing: it priced the head against a verify path structurally
    // incapable of using acceptance.
    //
    // SPARKINFER_DFLASH_MARKOV=0 turns it off for A/B. Draft-only either way: this shifts what is
    // proposed, never what is emitted, so LOSSLESS is unaffected.
    static const int markov_env = []{
        const char* e = getenv("SPARKINFER_DFLASH_MARKOV");
        return e ? (e[0] == '0' ? 0 : 1) : -1;
    }();
    // Default ON at every context. SPARKINFER_DFLASH_DEEP_MIN_SEQ no longer gates this; it still
    // selects the proposal-depth ladder in qwen35.cpp, which is a separate decision.
    const bool markov_on = markov_env >= 0 ? (markov_env != 0) : true;
    if (markov_on && s.markov_w1 && (s.markov_w2 || s.markov_w2_q)) {
        // ROW SHIFT (SPARKINFER_DFLASH_ROW_SHIFT=0 restores the legacy mapping). Which block row
        // backs proposal r?
        //
        // The legacy path reads row r. SGLang reads row r-1: its Markov sampler iterates over ALL
        // gamma rows of base_logits starting at row 0, with first_prev_tokens = the anchor
        // (draft_block_ids[:,0], the bonus token) -- so its FIRST proposal comes from row 0, the
        // row holding the real seed token, and its gamma-th from row gamma-1. Ours discards row 0
        // as "redundant with the target's own verify-row-0 call" and starts at row 1, so every
        // proposal is backed by the row one position later than the one that predicts it. The
        // conditioning is aligned (r==1 uses the seed either way); only the logit row is off.
        //
        // That is consistent with the symptom: tau lands well above 1.0 because the Markov bias,
        // conditioned correctly, partially compensates for the wrong base row -- but it is capped,
        // and deeper drafting buys nothing because every position is displaced.
        if (!head_done) kernels::launch_argmax(s.logits, s.d_out, 1, Vd, st);
        for (int r = 1; r <= kProposalDepth; r++) {
            const int* prev = (r == 1) ? s.d_ids : (s.d_out + (r - 1));
            const size_t row = (size_t)(kRowShift ? r - 1 : r);
            if (s.markov_w2_q)
                dflash_kernels::launch_markov_bias_add_q8(
                    s.markov_w1, s.markov_w2_q, s.markov_w2_s, prev,
                    s.logits + row * Vd, Vd, s.markov_rank, st,
                    s.confidence_w ? s.markov_latent + (size_t)r * s.markov_rank : nullptr);
            else
                dflash_kernels::launch_markov_bias_add(
                    s.markov_w1, s.markov_w2, prev,
                    s.logits + row * Vd, Vd, s.markov_rank, st,
                    s.confidence_w ? s.markov_latent + (size_t)r * s.markov_rank : nullptr);
            kernels::launch_argmax(s.logits + row * Vd, s.d_out + r, 1, Vd, st);
            // Coupled: the request's sampler on this row, over the argmax just written (kept if
            // the row's candidate set overflows). The next row's bias conditions on this token.
            if (couple)
                kernels::launch_sample_rows_topk(s.logits + row * Vd, 1, Vd, &s.d_samp->temp[r],
                                                 &s.d_samp->seed[r], &s.d_samp->step[r],
                                                 &s.d_samp->top_k[r], &s.d_samp->top_p[r],
                                                 s.d_out + r, st, /*overflow_keeps_out=*/true);
        }
        // Confidence head (optional), for every proposal row at once. It reads row r's hidden
        // state and the Markov latent that row's bias call wrote, and nothing in the chain ever
        // consumed it, so hoisting it out of the loop changes no value -- only the launch count,
        // from kProposalDepth grid-of-one kernels to one.
        if (s.confidence_w)
            dflash_kernels::launch_confidence_head_rows(
                s.xn, H, s.markov_latent, s.markov_rank,
                s.confidence_w, s.confidence_bias, H, s.markov_rank,
                kRowShift ? 0 : 1, kProposalDepth, s.d_confidence, st);
    } else if (c.dflash2 && s.sel_hproj && s.sel_ids) {
        // DFlash2's candidate selector: each slot's top-k logits, the slot's hidden row projected to
        // the selector rank, then one walk from the anchor that scores every candidate against the
        // slot before's pick (unary + <A[pred] * h, B[cand]>). Rows 1..depth back proposals 1..depth,
        // and without the multi-row head the logits start at row 0, so row 1 is one row in either way.
        const int k = c.selector_top_k, R = c.selector_rank;
        const float* rows_logits = s.logits + (size_t)1 * Vd;
        kernels::launch_topk_rows(rows_logits, kProposalDepth, Vd, k, s.sel_ids, s.sel_vals, st);
        // Project every block row: the batched GEMV has no 1- or 3-row variant and runs 16 rows for
        // those, past the end of sel_h. BW is the checkpoint's block_size (forced above), a width
        // it has; row 0's projection is simply unused.
        dflash_kernels::launch_gemv_batched16(s.xn, s.sel_hproj, s.sel_h, R, H, st, BW);
        dflash_kernels::launch_selector_walk(
            s.sel_vals, s.sel_ids, s.sel_h + (size_t)R, s.sel_pred, s.sel_succ, s.d_ids, kProposalDepth, k, R,
            couple ? &s.d_samp->temp[1] : nullptr, couple ? &s.d_samp->seed[1] : nullptr,
            couple ? &s.d_samp->step[1] : nullptr, s.d_out + 1, st);
        if (!head_done)
            cu(cudaMemcpyAsync(s.d_out, s.d_ids, sizeof(int), cudaMemcpyDeviceToDevice, st), "dflash2 row 0");
    } else {
        kernels::launch_argmax(s.logits + (size_t)(head_done ? head_row0 : 0) * Vd,
                               s.d_out + (head_done ? 1 : 0),
                               head_done ? kProposalDepth : B, Vd, st);
        if (couple) {
            const int r0 = head_done ? 1 : 0;
            kernels::launch_sample_rows_topk(s.logits + (size_t)(head_done ? head_row0 : 0) * Vd,
                                             head_done ? kProposalDepth : B, Vd,
                                             &s.d_samp->temp[r0], &s.d_samp->seed[r0],
                                             &s.d_samp->step[r0], &s.d_samp->top_k[r0],
                                             &s.d_samp->top_p[r0], s.d_out + r0, st,
                                             /*overflow_keeps_out=*/true);
        }
    }
    if (head_done)
        cu(cudaMemcpyAsync(s.h_out + 1, s.d_out + 1, kProposalDepth * sizeof(int),
                           cudaMemcpyDeviceToHost, st), "argmax");
    else
        cu(cudaMemcpyAsync(s.h_out, s.d_out, B * sizeof(int),
                           cudaMemcpyDeviceToHost, st), "argmax");

    // NUMERICAL DIFFERENTIAL DUMP (SPARKINFER_DSPARK_DUMP=<dir>). Writes ONE block's worth of
    // inputs and intermediates so a Python reference can recompute the same step from the same
    // bytes and diff tensor by tensor. Dumps only the first call of a process -- one block is
    // enough to localise a numerical divergence, and dumping every step would write gigabytes.
    //
    // Exists because every structural hypothesis has been eliminated (block construction, row
    // indexing, context injection, injected-KV positions, NVFP4 scale convention, aux-layer
    // capture point) while acceptance is still ~1.36 against a reported ~3.8. What is left can
    // only be found by comparing numbers, not by reasoning about wiring.
    if (const char* dump_dir = getenv("SPARKINFER_DSPARK_DUMP")) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            cudaStreamSynchronize(st);
            auto put = [&](const char* nm, const void* dev, size_t bytes, bool from_device) {
                std::vector<char> host(bytes);
                if (from_device)
                    cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost);
                else
                    memcpy(host.data(), dev, bytes);
                std::string path = std::string(dump_dir) + "/" + nm + ".bin";
                FILE* f = fopen(path.c_str(), "wb");
                if (f) { fwrite(host.data(), 1, bytes, f); fclose(f); }
            };
            const size_t V_ = (size_t)V;
            put("target_hidden", target_hidden, (size_t)ctx_len * n_cap * H * sizeof(bf16), true);
            put("target_proj",   s.target_proj, (size_t)(s.win_rows > 0 ? fc_rows : ctx_len) * H * sizeof(bf16), true);
            put("xn_last",       s.xn,          (size_t)BW * H * sizeof(bf16), true);
            // The block's token embeddings -- the draft borrows the TARGET's embed table, so this
            // is the only way to get layer 0's input without re-deriving it from a 20 GB sharded
            // NVFP4 checkpoint whose embedding may itself be quantized.
            put("noise_embed",   s.noise,       (size_t)BW * H * sizeof(bf16), true);
            put("logits",        s.logits,      (size_t)BW * V_ * sizeof(float), true);
            put("noise_ids",     noise_ids,     (size_t)BW * sizeof(int), false);
            put("d_out",         s.d_out,       (size_t)BW * sizeof(int), true);
            std::string meta = std::string(dump_dir) + "/meta.txt";
            if (FILE* f = fopen(meta.c_str(), "w")) {
                fprintf(f, "ctx_len %d\npos0 %d\npast %d\nBW %d\ndepth %d\nH %d\nV %d\n"
                           "n_cap %d\nmarkov_rank %d\nblock_size %d\nfc_skip %d\n",
                        ctx_len, pos0, past, BW, kProposalDepth, H, V, n_cap, s.markov_rank,
                        c.block_size, fc_skip);
                fclose(f);
            }
            fprintf(stderr, "[dspark-dump] wrote one block to %s (ctx_len=%d BW=%d depth=%d)\n",
                    dump_dir, ctx_len, BW, kProposalDepth);
        }
    }
    if (out_confidence && s.confidence_w)
        cu(cudaMemcpyAsync(s.h_confidence + 1, s.d_confidence + 1,
                           kProposalDepth * sizeof(float), cudaMemcpyDeviceToHost, st),
           "confidence readback");
    cu(cudaStreamSynchronize(st), "draft sync");
    for (int t = head_done ? 1 : 0; t <= kProposalDepth; t++) out_argmax[t] = s.h_out[t];
    if (out_confidence && s.confidence_w)
        for (int t = 1; t <= kProposalDepth; t++) out_confidence[t] = s.h_confidence[t];

    // Advance past the just-appended ctx+noise, then crop to `pos0` (= block start).
    // Matches z-lab dflash: past_key_values_draft.update(...) then .crop(start).
    // Without this, seq_len stays 0, crop(pos0) clamps to 0, and every step rebuilds
    // from an empty cache — draft quality collapses (τ≈1.x) after the first block.
    s.seq_len = past + ctx_len + BW;
    crop(pos0);
    return true;
}


namespace {
// forward_blocks' operand builder: rows [n0, n0 + rows) of an n-row NVFP4 operand from a bf16
// matrix holding only those rows (see launch_prefill_nvfp4_quant_b_slice).
bool fp4_slice(const void* w, unsigned char* d, unsigned char* sf, int n, int n0, int rows, int k,
               cudaStream_t st) {
    return n0 == 0 && rows == n ? kernels::launch_prefill_nvfp4_quant_b(w, d, sf, n, k, st)
                                : kernels::launch_prefill_nvfp4_quant_b_slice(w, d, sf, n, n0, rows, k, st);
}
}  // namespace

bool DFlashDraftModel::forward_blocks(const BlockJob* jobs, int n_jobs, int proposals,
                                      const void* head_fp4, const void* head_fp4_sf, float head_alpha) {
    Impl& s = *p_;
    if (s.arena_on && s.arena.offloaded()) return false;   // never read unmapped memory
    const auto& c = s.cfg;
    static const bool enabled = [] {
        const char* e = getenv("SPARKINFER_DFLASH_BATCHED");
        return !(e && e[0] == '0');
    }();
    if (!enabled || !jobs || n_jobs < 1 || !c.dflash2) return false;
    s.ensure_quant();
    const int BW = c.block_size;
    const int H = c.hidden, I = c.intermediate;
    const int qdim = c.n_q_heads * c.head_dim, kvdim = c.n_kv_heads * c.head_dim, d = c.head_dim;
    const int n_cap = (int)c.target_layer_ids.size();
    const int depth = std::min(BW - 1, proposals);
    const int R = n_jobs * BW;
    const int dyn = 2 * c.conv_kernel * (H / std::max(c.conv_group, 1));
    if (!s.fp4_mlp || !s.fc || !s.embed || !s.sel_hproj || !s.sel_ids || depth < 1 ||
        BW % 8 || R > Impl::kBatchRows || dyn <= 0 || dyn % 128 || (n_cap * H) % 128)
        return false;
    for (const LayerWeights& w : s.layers)
        if (!w.wq || !w.wk || !w.wv || !w.wo || !w.attn_conv_proj || !w.mlp_conv_proj) return false;
    const int Vd = (draft_vocab_env() > 0 && draft_vocab_env() < s.vocab) ? draft_vocab_env() : s.vocab;
    const bool head4 = head_fp4 && head_fp4_sf && Vd % 128 == 0;
    // Every job a steady-state block: its last verify's accepted rows, appended at the position the
    // previous block cropped to, with nothing below the context floor and nothing outside a window
    // -- so none of forward_block's skips apply (checked rather than assumed).
    s.park_current();
    int C = 0;
    for (int j = 0; j < n_jobs; ++j) {
        const BlockJob& jb = jobs[j];
        if (jb.slot < 0 || (size_t)jb.slot >= s.slots.size() || !s.slots[(size_t)jb.slot].live ||
            !jb.target_hidden || !jb.noise_ids || !jb.out_argmax || jb.ctx_len < 1 || jb.ctx_len > 8)
            return false;
        Impl::Slot& sl = s.slots[(size_t)jb.slot];
        const int past = sl.seq_len;
        if (past + jb.ctx_len != jb.pos0 || sl.ctx_floor > past) return false;
        // A slot that has run out of rows slides here, as forward_block would at the start of its
        // block, so the batch keeps the members together: declining sent every member through
        // forward_block one at a time, about one group step in 75 at eight members. If a later job
        // declines the batch, the slid slot is simply already slid for forward_block.
        if (!s.slide(sl.k, sl.v, sl.base, sl.ctx_floor, sl.cap, past, jb.ctx_len, jb.pos0, BW, s.stream) ||
            past + jb.ctx_len + BW - sl.base > sl.cap)
            return false;
        for (int L = 0; L < c.n_layers; ++L) {
            const int window = (L < (int)c.sliding_layers.size() && c.sliding_layers[L]) ? c.sliding_window : 0;
            const int lo = dflash_kernels::attn_gqa_kv_lo(BW, past + jb.ctx_len + BW, c.n_q_heads,
                                                          c.n_kv_heads, d, jb.pos0, 0, window);
            if (lo > past) return false;
        }
        for (int j2 = 0; j2 < j; ++j2) if (jobs[j2].slot == jb.slot) return false;
        C += jb.ctx_len;
    }
    cudaStream_t st = s.stream;
    Impl::Batch& b = s.batch;
    auto fail = [&](const char* what) {
        fprintf(stderr, "[dflash] batched draft: %s failed\n", what);
        cudaGetLastError();
        return false;
    };
    if (!b.ready) {
        const int RM = Impl::kBatchRows;
        cudaGetLastError();
        bool ok = true;
        auto al = [&](auto*& p, size_t n) {
            using T = std::remove_pointer_t<std::remove_reference_t<decltype(p)>>;
            if (ok && !(p = static_cast<T*>(s.dev_alloc(n * sizeof(T))))) ok = false;
            if (ok) s.owned.push_back((void*)p);
        };
        al(b.x, (size_t)RM * H); al(b.xn, (size_t)RM * H); al(b.h, (size_t)RM * H); al(b.hn, (size_t)RM * H);
        al(b.q, (size_t)RM * (qdim + 2 * kvdim)); al(b.attn, (size_t)RM * qdim); al(b.ao, (size_t)RM * H);
        al(b.gate, (size_t)RM * I); al(b.up, (size_t)RM * I); al(b.down, (size_t)RM * H);
        al(b.conv_a, (size_t)RM * dyn); al(b.conv_m, (size_t)RM * dyn);
        al(b.conv_out, (size_t)RM * std::max(H, qdim));
        al(b.ctx_h, (size_t)RM * n_cap * H); al(b.tp, (size_t)RM * H); al(b.ctx_k, (size_t)RM * 2 * kvdim);
        al(b.sel_h, (size_t)RM * c.selector_rank);
        al(b.sel_ids, (size_t)RM * c.selector_top_k); al(b.sel_vals, (size_t)RM * c.selector_top_k);
        al(b.logits, (size_t)RM * Vd); al(b.d_ids, (size_t)RM); al(b.d_out, (size_t)RM + 8);
        al(b.d_samp, 8);
        auto pin = [&](auto*& p, size_t bytes) {
            if (ok && cudaHostAlloc(reinterpret_cast<void**>(&p), bytes, cudaHostAllocDefault) != cudaSuccess)
                ok = false;
        };
        pin(b.h_ids, RM * sizeof(int));
        pin(b.h_out, (RM + 8) * sizeof(int));
        pin(b.h_samp, 8 * sizeof(Impl::DraftSampling));
        size_t ws = 0;
        auto wsz = [&](int n, int k) { ws = std::max(ws, kernels::prefill_nvfp4_workspace_bytes(RM, n, k)); };
        wsz(qdim + 2 * kvdim, H); wsz(2 * kvdim, H); wsz(H, qdim); wsz(dyn, H); wsz(H, n_cap * H);
        wsz(c.selector_rank, H); wsz(I, H); wsz(H, I);
        ws = std::max(ws, kernels::prefill_nvfp4_workspace_bytes_f32(RM, Vd, H));
        al(b.ws, ws ? ws : 1);
        // The operands, from the bf16 weights the per-slot path keeps.
        auto op = [&](Impl::Batch::Op& o, int n, int k) {
            al(o.d, kernels::prefill_nvfp4_data_bytes(n, k));
            al(o.sf, kernels::prefill_nvfp4_scale_bytes_b(n, k));
        };
        const int nq = qdim + 2 * kvdim;
        b.qkv.resize(c.n_layers); b.kv.resize(c.n_layers); b.wo.resize(c.n_layers);
        b.acp.resize(c.n_layers); b.mcp.resize(c.n_layers);
        for (int L = 0; ok && L < c.n_layers; ++L) {
            const LayerWeights& w = s.layers[L];
            op(b.qkv[L], nq, H); op(b.kv[L], 2 * kvdim, H); op(b.wo[L], H, qdim);
            op(b.acp[L], dyn, H); op(b.mcp[L], dyn, H);
            ok = ok && fp4_slice(w.wq, b.qkv[L].d, b.qkv[L].sf, nq, 0, qdim, H, st) &&
                 fp4_slice(w.wk, b.qkv[L].d, b.qkv[L].sf, nq, qdim, kvdim, H, st) &&
                 fp4_slice(w.wv, b.qkv[L].d, b.qkv[L].sf, nq, qdim + kvdim, kvdim, H, st) &&
                 fp4_slice(w.wk, b.kv[L].d, b.kv[L].sf, 2 * kvdim, 0, kvdim, H, st) &&
                 fp4_slice(w.wv, b.kv[L].d, b.kv[L].sf, 2 * kvdim, kvdim, kvdim, H, st) &&
                 fp4_slice(w.wo, b.wo[L].d, b.wo[L].sf, H, 0, H, qdim, st) &&
                 fp4_slice(w.attn_conv_proj, b.acp[L].d, b.acp[L].sf, dyn, 0, dyn, H, st) &&
                 fp4_slice(w.mlp_conv_proj, b.mcp[L].d, b.mcp[L].sf, dyn, 0, dyn, H, st);
        }
        if (ok) {
            op(b.fc, H, n_cap * H);
            op(b.sel, c.selector_rank, H);
            ok = ok && fp4_slice(s.fc, b.fc.d, b.fc.sf, H, 0, H, n_cap * H, st) &&
                 fp4_slice(s.sel_hproj, b.sel.d, b.sel.sf, c.selector_rank, 0, c.selector_rank, H, st);
        }
        ok = ok && cudaStreamSynchronize(st) == cudaSuccess;
        if (!ok) return fail("setup");
        b.ready = true;
    }
    // One block-scaled GEMM: out[m, n] = A[m, k] . W[n, k], A quantized here.
    auto gemm = [&](const bf16* a, int m, int k, const Impl::Batch::Op& w, int n, bf16* out) {
        return kernels::launch_prefill_nvfp4_quant_a(a, s.f4_a, s.f4_asf, m, k, st) &&
               kernels::launch_prefill_nvfp4_gemm(s.f4_a, s.f4_asf, w.d, w.sf, out, m, n, k, b.ws, st);
    };
    const int C8 = (C + 7) & ~7;
    // Ids and context rows.
    for (int j = 0; j < n_jobs; ++j)
        for (int i = 0; i < BW; ++i) b.h_ids[j * BW + i] = jobs[j].noise_ids[i];
    if (cudaMemcpyAsync(b.d_ids, b.h_ids, R * sizeof(int), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return fail("ids");
    kernels::launch_embedding(b.d_ids, s.embed, b.x, R, H, st);
    std::vector<int> coff((size_t)n_jobs);
    for (int j = 0, o = 0; j < n_jobs; o += jobs[j].ctx_len, ++j) {
        coff[(size_t)j] = o;
        cu(cudaMemcpyAsync(b.ctx_h + (size_t)o * n_cap * H, jobs[j].target_hidden,
                           (size_t)jobs[j].ctx_len * n_cap * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched ctx gather");
    }
    if (C8 > C)   // the GEMM's rows come in eights: zero the pad so it stays finite
        cu(cudaMemsetAsync(b.ctx_h + (size_t)C * n_cap * H, 0, (size_t)(C8 - C) * n_cap * H * sizeof(bf16), st),
           "batched ctx pad");
    if (!gemm(b.ctx_h, C8, n_cap * H, b.fc, H, b.tp)) return fail("fc");
    dflash_kernels::launch_rms(b.tp, s.hidden_norm, b.tp, C8, H, c.rms_eps, st);

    const float scale = 1.f / sqrtf((float)d);
    const int nq = qdim + 2 * kvdim;
    const bool causal_force = [] {
        const char* e = getenv("SPARKINFER_DFLASH_FORCE_CAUSAL");
        return e && e[0] == '1';
    }();
    for (int L = 0; L < c.n_layers; ++L) {
        const LayerWeights& w = s.layers[L];
        const int window = (L < (int)c.sliding_layers.size() && c.sliding_layers[L]) ? c.sliding_window : 0;
        const bool causal = causal_force ||
                            (c.is_causal >= 0 ? c.is_causal != 0
                                              : (L < (int)c.sliding_layers.size() && c.sliding_layers[L]));
        if (L == 0) dflash_kernels::launch_rms(b.x, w.input_norm, b.xn, R, H, c.rms_eps, st);
        // Attention conv "prepare": the kernels for every row in one GEMM, the conv per job (it runs
        // along a block's own rows).
        if (!gemm(b.xn, R, H, b.acp[L], dyn, b.conv_a)) return fail("attn conv");
        for (int j = 0; j < n_jobs; ++j)
            dflash_kernels::launch_grouped_conv(b.xn + (size_t)j * BW * H, w.attn_conv_base,
                                                b.conv_a + (size_t)j * BW * dyn, 0,
                                                b.conv_out + (size_t)j * BW * H, BW, H, c.conv_kernel,
                                                c.conv_group, dyn, st);
        cu(cudaMemcpyAsync(b.xn, b.conv_out, (size_t)R * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched attn conv");
        // q|k|v for the block rows, k|v for the context rows.
        if (!gemm(b.xn, R, H, b.qkv[L], nq, b.q)) return fail("qkv");
        if (!gemm(b.tp, C8, H, b.kv[L], 2 * kvdim, b.ctx_k)) return fail("ctx kv");
        // Q packed for the attention, into attn's buffer (free until the attention writes it).
        cu(cudaMemcpy2DAsync(b.attn, (size_t)qdim * sizeof(bf16), b.q, (size_t)nq * sizeof(bf16),
                             (size_t)qdim * sizeof(bf16), R, cudaMemcpyDeviceToDevice, st), "batched q");
        for (int j = 0; j < n_jobs; ++j) {
            const BlockJob& jb = jobs[j];
            const Impl::Slot& sl = s.slots[(size_t)jb.slot];
            const int past = sl.seq_len, cl = jb.ctx_len;
            bf16* kd = sl.k[L] + ((ptrdiff_t)past - sl.base) * kvdim;
            bf16* vd = sl.v[L] + ((ptrdiff_t)past - sl.base) * kvdim;
            const bf16* ck = b.ctx_k + (size_t)coff[(size_t)j] * 2 * kvdim;
            const bf16* bq = b.q + (size_t)j * BW * nq;
            const size_t kvb = (size_t)kvdim * sizeof(bf16);
            cu(cudaMemcpy2DAsync(kd, kvb, ck, 2 * kvb, kvb, cl, cudaMemcpyDeviceToDevice, st), "ctx k");
            cu(cudaMemcpy2DAsync(vd, kvb, ck + kvdim, 2 * kvb, kvb, cl, cudaMemcpyDeviceToDevice, st), "ctx v");
            cu(cudaMemcpy2DAsync(kd + (size_t)cl * kvdim, kvb, bq + qdim, (size_t)nq * sizeof(bf16), kvb, BW,
                                 cudaMemcpyDeviceToDevice, st), "block k");
            cu(cudaMemcpy2DAsync(vd + (size_t)cl * kvdim, kvb, bq + qdim + kvdim, (size_t)nq * sizeof(bf16), kvb,
                                 BW, cudaMemcpyDeviceToDevice, st), "block v");
            bf16* qj = b.attn + (size_t)j * BW * qdim;
            if (c.rope_normal) {
                dflash_kernels::launch_rms_heads_rope_normal(qj, w.q_norm, BW, c.n_q_heads, d, c.rms_eps,
                                                             jb.pos0, c.rope_theta, st);
                dflash_kernels::launch_rms_heads_rope_normal(kd, w.k_norm, cl + BW, c.n_kv_heads, d,
                                                             c.rms_eps, jb.pos0 - cl, c.rope_theta, st);
            } else {
                dflash_kernels::launch_rms_heads_rope(qj, w.q_norm, BW, c.n_q_heads, d, c.rms_eps, jb.pos0,
                                                     c.rope_theta, st, s.d_yarn_inv_freq, s.yarn_att_scale);
                dflash_kernels::launch_rms_heads_rope(kd, w.k_norm, cl + BW, c.n_kv_heads, d, c.rms_eps,
                                                     jb.pos0 - cl, c.rope_theta, st, s.d_yarn_inv_freq,
                                                     s.yarn_att_scale);
            }
            const int kv_len = past + cl + BW;
            const int kv0 = sl.base > 0 ? std::max(sl.ctx_floor, jb.pos0 - window + 1)   // see forward_block
                                        : std::min(sl.ctx_floor, kv_len);
            // q is read from attn's buffer and the output goes to conv_out (attn is overwritten
            // only after every job's attention, below).
            dflash_kernels::launch_attn_gqa(qj, sl.k[L] + ((ptrdiff_t)kv0 - sl.base) * kvdim,
                                            sl.v[L] + ((ptrdiff_t)kv0 - sl.base) * kvdim,
                                            b.conv_out + (size_t)j * BW * qdim, BW, kv_len - kv0,
                                            c.n_q_heads, c.n_kv_heads, d, jb.pos0, kv0, window, causal, scale,
                                            st, s.fa_m, s.fa_l, s.fa_acc);
        }
        cu(cudaMemcpyAsync(b.attn, b.conv_out, (size_t)R * qdim * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched attn");
        if (!gemm(b.attn, R, qdim, b.wo[L], H, b.ao)) return fail("wo");
        for (int j = 0; j < n_jobs; ++j)
            dflash_kernels::launch_grouped_conv(b.ao + (size_t)j * BW * H,
                                                w.attn_conv_base + (size_t)c.conv_kernel * H,
                                                b.conv_a + (size_t)j * BW * dyn, 1,
                                                b.conv_out + (size_t)j * BW * H, BW, H, c.conv_kernel,
                                                c.conv_group, dyn, st);
        cu(cudaMemcpyAsync(b.ao, b.conv_out, (size_t)R * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched attn conv finish");
        dflash_kernels::launch_add_rms(b.x, b.ao, b.h, w.post_norm, b.hn, R, H, c.rms_eps, st);
        // MLP conv "prepare", then the MLP.
        if (!gemm(b.hn, R, H, b.mcp[L], dyn, b.conv_m)) return fail("mlp conv");
        for (int j = 0; j < n_jobs; ++j)
            dflash_kernels::launch_grouped_conv(b.hn + (size_t)j * BW * H, w.mlp_conv_base,
                                                b.conv_m + (size_t)j * BW * dyn, 0,
                                                b.conv_out + (size_t)j * BW * H, BW, H, c.conv_kernel,
                                                c.conv_group, dyn, st);
        cu(cudaMemcpyAsync(b.hn, b.conv_out, (size_t)R * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched mlp conv");
        auto mm = [&](const unsigned char* wd, const unsigned char* wsf, bf16* out, int n, int kk) {
            return kernels::launch_prefill_nvfp4_gemm(s.f4_a, s.f4_asf, wd, wsf, out, R, n, kk, b.ws, st);
        };
        const bool mlp_ok =
            kernels::launch_prefill_nvfp4_quant_a(b.hn, s.f4_a, s.f4_asf, R, H, st) &&
            mm(w.fp4_gate, w.fp4_gate_sf, b.gate, I, H) && mm(w.fp4_up, w.fp4_up_sf, b.up, I, H) &&
            kernels::launch_prefill_nvfp4_swiglu_quant_a(b.gate, b.up, s.f4_a, s.f4_asf, R, I, st) &&
            mm(w.fp4_down, w.fp4_down_sf, b.down, H, I);
        if (!mlp_ok) return fail("mlp");
        for (int j = 0; j < n_jobs; ++j)
            dflash_kernels::launch_grouped_conv(b.down + (size_t)j * BW * H,
                                                w.mlp_conv_base + (size_t)c.conv_kernel * H,
                                                b.conv_m + (size_t)j * BW * dyn, 1,
                                                b.conv_out + (size_t)j * BW * H, BW, H, c.conv_kernel,
                                                c.conv_group, dyn, st);
        cu(cudaMemcpyAsync(b.down, b.conv_out, (size_t)R * H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "batched mlp conv finish");
        const bf16* next_norm = (L + 1 < c.n_layers) ? s.layers[L + 1].input_norm : s.final_norm;
        dflash_kernels::launch_add_rms(b.h, b.down, b.x, next_norm, b.xn, R, H, c.rms_eps, st);
    }
    // Head over every row (rows 1..depth of each block are read), then each block's selector walk.
    // The draft scores the first Vd ids, so the GEMM takes the head's first Vd rows as its whole B
    // operand: the data is row-major, and the CUTLASS SF layout tiles 128-row blocks with K fastest
    // (tile_to_shape, Step<_2,_1,_3>), so the first Vd / 128 row tiles' scales are a prefix of the
    // full head's with the same strides. Checked on the box: these logits equal the first Vd columns
    // of the same GEMM at n = vocab exactly (max |diff| 0, every row's argmax the same).
    if (head4) {
        if (!(kernels::launch_prefill_nvfp4_quant_a(b.xn, s.f4_a, s.f4_asf, R, H, st) &&
              kernels::launch_prefill_nvfp4_gemm_f32(s.f4_a, s.f4_asf, head_fp4, head_fp4_sf, b.logits, R, Vd, H,
                                                     b.ws, st, head_alpha)))
            return fail("head");
    } else {
        for (int j = 0; j < n_jobs; ++j) {
            kernels::launch_quantize_q8_1_rows(b.xn + ((size_t)j * BW + 1) * H, s.head_q8, H, depth, H, st);
            float* y = b.logits + ((size_t)j * BW + 1) * Vd;
            bool ok = false;
            if (s.lm_head_type == 12 && s.lm_head_i4)
                ok = kernels::launch_gemv_i4_q81_multirow_f32(s.head_q8, s.lm_head_i4, s.lm_head_i4_scale,
                                                              y, Vd, H, depth, st);
            else if (s.lm_head_type == 12 && s.lm_head_i8)
                ok = kernels::launch_gemv_i8_q81_multirow_f32(s.head_q8, s.lm_head_i8, s.lm_head_i8_scale,
                                                              y, Vd, H, depth, st);
            else if (s.lm_head_type == 12)
                ok = kernels::launch_gemv_q4k_dp4a_multirow_f32(s.head_q8, s.lm_head, y, Vd, H, depth, st);
            else if (s.lm_head_type == 14)
                ok = kernels::launch_gemv_q6k_dp4a_multirow_f32(s.head_q8, s.lm_head, y, Vd, H, depth, st);
            if (!ok) return fail("head");
        }
    }
    const int k = c.selector_top_k, rank = c.selector_rank;
    kernels::launch_topk_rows(b.logits, R, Vd, k, b.sel_ids, b.sel_vals, st);
    if (!gemm(b.xn, R, H, b.sel, rank, b.sel_h)) return fail("selector");
    static const bool coupled_on = [] {
        const char* e = getenv("SPARKINFER_DFLASH_COUPLED");
        return !(e && e[0] == '0');
    }();
    std::vector<bool> couple((size_t)n_jobs);
    for (int j = 0; j < n_jobs; ++j) {
        const BlockJob& jb = jobs[j];
        couple[(size_t)j] = coupled_on && kernels::sample_rows_topk_eligible(jb.temperature, jb.top_k, 1 << 30);
        Impl::DraftSampling& hs = b.h_samp[j];
        for (int r = 0; r <= depth; ++r) {
            hs.temp[r] = (r == 0 || !couple[(size_t)j]) ? 0.f : jb.temperature;
            hs.seed[r] = jb.seed;
            hs.step[r] = r == 0 ? 0ull : jb.step0 + (unsigned long long)(r - 1);
            hs.top_k[r] = jb.top_k;
            hs.top_p[r] = jb.top_p;
        }
    }
    cu(cudaMemcpyAsync(b.d_samp, b.h_samp, (size_t)n_jobs * sizeof(Impl::DraftSampling),
                       cudaMemcpyHostToDevice, st), "batched sampling");
    // Every block's walk in one launch. A job its sampler does not couple walks at temperature 0
    // (its argmax), which is what passing no sampler does.
    dflash_kernels::launch_selector_walks(b.sel_vals + (size_t)k, b.sel_ids + (size_t)k, b.sel_h + (size_t)rank,
                                          s.sel_pred, s.sel_succ, b.d_ids, depth, k, rank, &b.d_samp->temp[1],
                                          &b.d_samp->seed[1], &b.d_samp->step[1], sizeof(Impl::DraftSampling),
                                          b.d_out + 1, n_jobs, BW, st);
    cu(cudaMemcpyAsync(b.h_out, b.d_out, (size_t)R * sizeof(int), cudaMemcpyDeviceToHost, st), "batched out");
    if (cudaStreamSynchronize(st) != cudaSuccess) return fail("sync");
    for (int j = 0; j < n_jobs; ++j) {
        for (int t = 1; t <= depth; ++t) jobs[j].out_argmax[t] = b.h_out[(size_t)j * BW + t];
        // As forward_block ends: the block's rows appended, then cropped back to its start.
        Impl::Slot& sl = s.slots[(size_t)jobs[j].slot];
        sl.seq_len = jobs[j].pos0;
    }
    s.load_slot(s.cur_slot);
    return true;
}

} // namespace sparkinfer
