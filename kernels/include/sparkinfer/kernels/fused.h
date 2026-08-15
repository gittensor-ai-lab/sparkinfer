#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Fused RMSNorm:  out[r] = weight * x[r] / sqrt(mean(x[r]^2) + eps)
// Optionally adds a residual first (out and residual may alias x).
//   x / residual / out: [rows, cols] (bf16), weight: [cols] (bf16)
void launch_rmsnorm(const void* x_bf16, const void* weight_bf16, void* out_bf16,
                    int rows, int cols, float eps, cudaStream_t stream = nullptr);

void launch_add_rmsnorm(const void* x_bf16, const void* residual_bf16,
                        const void* weight_bf16, void* out_bf16,
                        int rows, int cols, float eps, cudaStream_t stream = nullptr);

// Fused residual+RMSNorm that also emits the residual sum:
//   out_sum = x + residual;  out_norm = (out_sum / rms(out_sum)) * weight
void launch_add_rmsnorm2(const void* x_bf16, const void* residual_bf16, const void* weight_bf16,
                         void* out_sum_bf16, void* out_norm_bf16,
                         int rows, int cols, float eps, cudaStream_t stream = nullptr);

// add_rmsnorm2 that additionally emits a Q8_1 quantization of out_norm (si_block_q8_1),
// so the downstream int8 GEMV skips its own quantize node. rows==1, cols % 256 == 0.
void launch_add_rmsnorm2_q8(const void* x_bf16, const void* residual_bf16, const void* weight_bf16,
                            void* out_sum_bf16, void* out_norm_bf16, void* out_q8,
                            int cols, float eps, cudaStream_t stream = nullptr);
void launch_add_rmsnorm2_q8_rows(const void* x_bf16, const void* residual_bf16,
                                 const void* weight_bf16, void* out_sum_bf16,
                                 void* out_norm_bf16, void* out_q8, int rows, int cols,
                                 float eps, cudaStream_t stream = nullptr);

// Sandwich-norm residual add (Gemma2/Muse-Glimmer style): out = residual + RMSNorm(block_out)
// * weight. Unlike add_rmsnorm2/3 above (which norm the SUM), this norms block_out ALONE --
// the sub-block's raw output, before any residual add -- and only the normalized result joins
// the residual stream. Used for both the post-attention and post-FFN sandwich-norm steps.
// RMSNorm + per-row int8 quantize in one pass (still writes the bf16). Bit-identical.
bool launch_rmsnorm_quant_i8(const void* x, const void* weight, void* out,
                             signed char* q, float* scale, int rows, int cols,
                             float eps, cudaStream_t stream,
                             signed char* qp = nullptr);

// Sandwich norm fed from the split-K int32 accumulator (skips the reduce + the bf16 round trip).
void launch_norm_then_add_acc(const void* residual_bf16, const int* acc, const float* sxr,
                              const float* row_scale, const void* weight_bf16, void* out_bf16,
                              int rows, int cols, float eps, cudaStream_t stream);

// The Muse sandwich pair as one node: out = residual + RMSNorm(block_out, w1, eps1), then
// out2 = RMSNorm(out, w2, eps2). Bit-identical to the two separate launches; false = caller
// runs them separately (odd cols, or SPARKINFER_MUSE_SANDWICH_FUSE=0).
bool launch_norm_then_add_rmsnorm(const void* residual, const void* block_out, const void* w1,
                                  void* out, const void* w2, void* out2,
                                  int rows, int cols, float eps1, float eps2,
                                  cudaStream_t stream = nullptr);
void launch_norm_then_add(const void* residual_bf16, const void* block_out_bf16,
                          const void* weight_bf16, void* out_bf16,
                          int rows, int cols, float eps, cudaStream_t stream = nullptr);

// Muse Glimmer's sandwich-norm tail in one launch instead of two:
//   out_x  = residual + RMSNorm(branch, post_w, post_eps)   (what launch_norm_then_add does)
//   out_xn = RMSNorm(out_x, next_w, eps)                    (what the following launch_rmsnorm does)
// Bit-identical to that pair: same 256-thread block, same per-stage loop order and reduction tree,
// and stage 2 re-reads the bf16-rounded out_x from memory exactly as the separate kernel would.
// Fused QK-norm + optional NORM-convention RoPE + K/V append (Muse Glimmer). One kernel in place
// of launch_rmsnorm_qk + launch_rope_kv_append_normal / launch_kv_append. Bit-identical.
void launch_muse_qknorm_rope_kv(void* q, void* k, const void* v, const void* q_w, const void* k_w,
                                void* k_pool, void* v_pool, const int* block_table,
                                const int* pos_angle, const int* pos_slot,
                                int n_q_heads, int n_kv_heads, int head_dim, float theta,
                                int block_size, float eps, bool do_rope, cudaStream_t stream = nullptr);

// Returns true if it also wrote Q8_1(out_xn) into out_q8, letting the caller skip the standalone
// quantize its next MMVQ would otherwise need. Pass out_q8 = nullptr to opt out. Never assume the
// emission happened -- it is conditional on the row shape and on the register path being taken.
bool launch_muse_sandwich_tail(const void* residual_bf16, const void* branch_bf16,
                               const void* post_w_bf16, const void* next_w_bf16,
                               void* out_x_bf16, void* out_xn_bf16, void* out_q8,
                               int rows, int cols, float post_eps, float eps,
                               cudaStream_t stream = nullptr);

// Fold residual_add(res1,res2) into add_rmsnorm2: out_sum = x + (res1 + res2).
void launch_add_rmsnorm3(const void* x_bf16, const void* res1_bf16, const void* res2_bf16,
                         const void* weight_bf16, void* out_sum_bf16, void* out_norm_bf16,
                         int rows, int cols, float eps, cudaStream_t stream = nullptr);
void launch_add_rmsnorm3_q8(const void* x_bf16, const void* res1_bf16, const void* res2_bf16,
                            const void* weight_bf16, void* out_sum_bf16, void* out_norm_bf16,
                            void* out_q8, int cols, float eps, cudaStream_t stream = nullptr);
void launch_add_rmsnorm3_q8_rows(const void* x_bf16, const void* res1_bf16,
                                 const void* res2_bf16, const void* weight_bf16,
                                 void* out_sum_bf16, void* out_norm_bf16, void* out_q8,
                                 int rows, int cols, float eps, cudaStream_t stream = nullptr);

// Fused per-head Q-norm + K-norm in one kernel (1 graph node vs 2). In-place on q/k.
void launch_rmsnorm_qk(void* q, void* k, const void* q_w, const void* k_w,
                       int n_q_heads, int n_kv_heads, int head_dim, float eps, cudaStream_t stream = nullptr);

// Token embedding gather: out[t,:] = table[ids[t],:]  (bf16).
//   ids: [n_tokens] (int32), table: [vocab, hidden], out: [n_tokens, hidden]
void launch_embedding(const int* ids, const void* table, void* out,
                      int n_tokens, int hidden, cudaStream_t stream = nullptr);

// Greedy argmax over each row of logits.  logits: [n_rows, vocab] (fp32),
// out_id: [n_rows] (int32).
void launch_argmax(const float* logits, int* out_id, int n_rows, int vocab,
                   cudaStream_t stream = nullptr);

// Gemma2/Muse-Glimmer-style final-logit softcap, applied in place before any
// argmax/sampling reads logits: logits[v] = tanh(logits[v] * scale / cap) * cap.
// logits: [n_rows, vocab] (fp32).
void launch_logit_softcap(float* logits, int n_rows, int vocab, float scale, float cap,
                          cudaStream_t stream = nullptr);

// Gumbel-max temperature sampling, mutating logits in place before argmax:
//   logits[v] = logits[v] / *temp_f32 - log(-log(curand_uniform(Philox(seed, row*vocab+v, step))))
// temp/seed/step are read from device memory on EVERY launch (not baked into a CUDA graph node
// at capture time), so one captured decode graph can safely replay this across separate requests
// with different temperature/seed values without re-capturing. *temp_f32 <= 0 is a cheap internal
// no-op (logits left untouched, byte-identical to plain greedy argmax) -- callers on a
// CUDA-graph-captured path must always launch this unconditionally rather than host-gating on
// temperature, since the graph's node topology is frozen at capture time and may be replayed for
// a later request with a different temperature. logits: [n_rows, vocab] (fp32), same buffer
// launch_argmax reads next.
void launch_temperature_sample(float* logits, int n_rows, int vocab,
                               const float* temp_f32, const unsigned long long* seed_u64,
                               const unsigned long long* step_u64, cudaStream_t stream = nullptr);

// top_k / top_p (nucleus) truncation, applied in place BEFORE launch_temperature_sample: masks
// logits outside the surviving set to -infinity via a full descending sort + cumulative-softmax
// cutoff (n_rows == 1 only -- decode-time truncation, not a batched prefill/verify operation).
// top_k hard-truncates to the k highest-logit entries first, then top_p narrows further within
// that set (cumulative probability renormalized against the top_k survivors' own mass, matching
// HuggingFace/vLLM's mask-then-renormalize convention) -- entry rank 0 (the eventual greedy
// winner) is unconditionally kept by both checks, so at least one surviving entry always exists
// and launch_argmax can never see an all -inf row.
//
// *top_k_i32 <= 0 or >= vocab disables top_k (no truncation). *top_p_f32 <= 0 or >= 1.0 disables
// top_p. Both are read from device memory on EVERY launch, same graph-replay-across-requests
// rationale as launch_temperature_sample's temp/seed/step -- ALWAYS launch this unconditionally
// on a CUDA-graph-captured decode path, never host-gate on top_k/top_p being set.
//
// scratch (vocab_iota/sorted_logits/sorted_idx/topk_exp/topk_cumsum/rank_by_id, all length `vocab`;
// sort_temp/scan_temp sized via topk_sort_temp_storage_bytes()/topk_scan_temp_storage_bytes())
// must be allocated ONCE at load time with a fixed address -- CUB's num_items argument is a host
// constant (here, `vocab`) that cannot vary per call, so this always processes the full vocab
// regardless of top_k/top_p, unlike launch_temperature_sample's near-free no-op when disabled.
// vocab_iota must be filled once via launch_vocab_iota_init() before the first call and never
// mutated afterward.
//
// rank_by_id[id] = the sorted rank of vocab entry `id` (an inverse permutation of sorted_idx,
// scattered as a side effect of the internal exp kernel -- free, no extra launch). Used by
// launch_extract_chosen_logit() below to recover the RAW (pre-mask, pre-temperature-noise) logit
// of whichever token is ultimately sampled, for logprobs/top_logprobs -- see that function's doc
// comment. Always populated regardless of whether the caller wants logprobs (same graph-safety
// discipline as everything else on this path); the cost is one extra int write inside a kernel
// that's already memory-bound and already touches every rank once.
void launch_topk_topp_mask(float* logits, int vocab,
                           const int* vocab_iota, float* sorted_logits, int* sorted_idx,
                           float* topk_exp, float* topk_cumsum,
                           void* sort_temp, size_t sort_temp_bytes,
                           void* scan_temp, size_t scan_temp_bytes,
                           const int* top_k_i32, const float* top_p_f32,
                           int* rank_by_id,
                           cudaStream_t stream = nullptr);

// Looks up the raw (pre-mask, pre-temperature-noise) logit of *out_id (whatever launch_argmax
// picked, downstream of launch_temperature_sample -- not necessarily rank 0 once real temperature
// sampling is active) via rank_by_id + sorted_logits from the launch_topk_topp_mask call that
// preceded it in the same decode step. <<<1,1>>>, negligible cost. Always launched
// unconditionally, same rationale as launch_topk_topp_mask -- the caller (qwen35.cpp) does not
// know at capture time whether THIS particular replay's request wants logprobs, since the graph
// may be replayed for a later, different request via use_prefix_session.
void launch_extract_chosen_logit(const int* out_id, const int* rank_by_id,
                                 const float* sorted_logits, float* chosen_logit,
                                 cudaStream_t stream = nullptr);

// OpenAI-style presence_penalty/frequency_penalty: logits[v] -= frequency_penalty*counts[v] +
// presence_penalty*(counts[v]>0), for every v in [0,vocab). `counts` is the CURRENT session's
// running per-vocab-id generation count (Qwen35Model::Impl::penalty_counts, already swapped in
// by activate_session() before forward_token() runs -- same "by the time any decode kernel
// launches, the correct session's buffer is already swapped in" guarantee lin_state gets).
// presence_penalty_f32/frequency_penalty_f32 are read from device memory on EVERY launch, same
// graph-replay-across-requests rationale as launch_temperature_sample's temp/seed/step -- ALWAYS
// launch this unconditionally on a CUDA-graph-captured decode path, never host-gate on the
// penalty values being nonzero. UNLIKE top_k/top_p, this has NO inertness proof at
// presence_penalty==0 && frequency_penalty==0 the way top_k/top_p do at temperature<=0 -- see
// qwen35.h's forward_token doc comment; a nonzero penalty can change the greedy-argmax winner on
// its own, which is why DFlash needs its own separate rejection check for this (see
// should_reject_dflash_penalty in chat_tools.hpp).
//
// Call BEFORE launch_topk_topp_mask so the penalized distribution flows through top_k/top_p
// truncation, temperature sampling, AND logprobs reporting -- matches real-world OpenAI/vLLM
// behavior of reporting logprobs against what was actually sampled from, not an artificially
// "raw" distribution.
void launch_presence_frequency_penalty(float* logits, const int* counts, int vocab,
                                       const float* presence_penalty_f32,
                                       const float* frequency_penalty_f32,
                                       cudaStream_t stream = nullptr);

// Records that this decode step's sampled token (*out_id) was chosen, incrementing its running
// count in the CURRENT session's counts buffer -- so the NEXT decode step's
// launch_presence_frequency_penalty call sees the update. <<<1,1>>>, negligible cost. Always
// launched unconditionally (same graph-replay-safety rationale as everything else on this path).
// Call AFTER launch_argmax, alongside launch_extract_chosen_logit.
void launch_increment_penalty_count(int* counts, const int* out_id, cudaStream_t stream = nullptr);

// logits[v] += bias[v] for every vocab entry -- always launched unconditionally on the CUDA-graph-
// captured decode path, same graph-replay-safety rationale as launch_presence_frequency_penalty.
// Unlike that kernel's counts (refreshed every decode step), `bias` is set ONCE per request by
// launch_scatter_logit_bias below (called outside this graph, from Qwen35Model::set_logit_bias)
// and stays constant for the rest of the request's decode -- no per-step host round-trip needed.
// logit_bias has no inertness proof at temperature<=0 (an arbitrary per-vocab additive bias CAN
// change the greedy-argmax winner on its own), same story as presence/frequency penalty -- see
// should_reject_dflash_logit_bias in chat_tools.hpp.
//
// Call alongside launch_presence_frequency_penalty, BEFORE launch_topk_topp_mask (order between
// the two additive kernels doesn't matter mathematically) so the biased distribution flows through
// truncation, temperature sampling, AND logprobs reporting.
void launch_logit_bias(float* logits, const float* bias, int vocab, cudaStream_t stream = nullptr);

// bias[ids[i]] = vals[i] for i in [0,k) -- a one-time host-triggered scatter, NOT part of the
// captured decode graph (call from Qwen35Model::set_logit_bias, before the request's first
// forward_token()). ids/vals are small (k <= kMaxLogitBiasEntries) transient scratch, distinct
// from the persistent per-session `bias` buffer launch_logit_bias reads every decode step. An
// out-of-range id is silently dropped (defensive backstop; real validation is
// parse_request_controls, which has the real vocab size).
void launch_scatter_logit_bias(float* bias, const int* ids, const float* vals, int k, int vocab,
                               cudaStream_t stream = nullptr);

// Load-time-only helpers (no kernel launch involved beyond a one-time iota fill) -- call once per
// model load, never on the decode hot path.
size_t topk_sort_temp_storage_bytes(int vocab);
size_t topk_scan_temp_storage_bytes(int vocab);
void launch_vocab_iota_init(int* vocab_iota, int vocab, cudaStream_t stream = nullptr);

// A CUDA 12.4+ conditional-graph-node variant of launch_topk_topp_mask (skip the sort/scan
// pipeline entirely on replay whenever top_k/top_p are disabled, recovering near-zero cost for
// the common case) was prototyped and spiked in isolation successfully, but live-verified to
// corrupt determinism when embedded in the real decode graph: requests with identical
// (seed,step,temperature,top_k,top_p) produced different output when interleaved with other
// differently-configured requests on the same reused (use_prefix_session) graph -- reproducible,
// but did NOT reproduce in an isolated minimal repro (same CUB-in-conditional-body pattern,
// alternating active/inactive replays, passed cleanly). Root cause is presumably an interaction
// between the conditional node and the surrounding ~hundreds of other nodes in the full decode
// graph, not the conditional-node mechanism itself -- not safely resolvable without a deeper CUDA
// graph engine investigation. Reverted; launch_topk_topp_mask (always-runs) is the shipped path.
// Left as a documented dead end so a future attempt doesn't have to rediscover this.

// Benchmark-only decode feedback: tok = out_id; pos/writepos/seqlen += 1.
// Capturable, so a decode CUDA graph can self-feed during throughput timing.
void launch_decode_feedback(int* scalars, const int* out_id, cudaStream_t stream = nullptr);
// Qwen3.5/Qwen3.6 hybrid Gated DeltaNet helpers.
void launch_qwen36_split_q_gate(const void* qg_bf16, void* q_bf16, void* gate_bf16,
                                int n_heads, int head_dim, cudaStream_t stream = nullptr);

void launch_qwen36_mul_sigmoid(void* x_bf16, const void* gate_bf16, int n,
                               cudaStream_t stream = nullptr);

void launch_qwen36_sigmoid_scalar(const void* x_bf16, float* out_f32,
                                  cudaStream_t stream = nullptr);
void launch_qwen36_sigmoid_rows(const void* x_bf16, float* out_f32, int rows,
                                cudaStream_t stream = nullptr);

// Shared-expert SwiGLU with folded gate scalar: out[i] = dw * SiLU(gate[i]) * up[i].
void launch_qwen36_shared_swiglu(const void* gate_bf16, const void* up_bf16,
                                 const float* dw_f32, void* out_bf16, int n,
                                 cudaStream_t stream = nullptr);
void launch_qwen36_shared_swiglu_rows(const void* gate_bf16, const void* up_bf16,
                                      const float* dw_f32, void* out_bf16,
                                      int rows, int ffn, cudaStream_t stream = nullptr);

void launch_qwen36_conv_split_l2(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream = nullptr);

// Fused conv_split + per-head l2_norm: one block per head, head_dim threads.
// Eliminates the two standalone l2_norm_heads kernel launches per GDN layer.
// SPARKINFER_GDN_FUSE=0 restores the split path for A/B.
void launch_qwen36_conv_split_l2norm_fused(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream = nullptr);

// qh_block: v-head -> q/k-head broadcast convention. false = cyclic (vh % q_heads), the
// original/validated convention for Qwythos and Qwen3.6-35B-A3B's checkpoints (v_heads/q_heads
// ratio 2). true = block (vh / (v_heads/q_heads)), which Qwen3.8-27B's checkpoint needs instead
// (ratio 3) -- confirmed empirically against a real reference implementation (llama.cpp), since
// the two checkpoints' own HF-side v-head layout conventions differ despite the shared
// architecture family. This is a property of the loaded checkpoint, not a global default.
void launch_qwen36_gdn_ar(const void* q_bf16, const void* k_bf16, const void* v_bf16,
                          const void* alpha_bf16, const void* beta_bf16,
                          const void* dt_bf16, const void* a_bf16,
                          float* state_f32, void* out_bf16,
                          int q_heads, int v_heads, int head_dim, bool qh_block,
                          cudaStream_t stream = nullptr);

void launch_qwen36_gated_norm(const void* x_bf16, const void* z_bf16,
                              const void* weight_bf16, void* out_bf16,
                              int v_heads, int head_dim, float eps,
                              cudaStream_t stream = nullptr);

// Gated norm + Q8_1 emit for ssm_out MMVQ (skips bf16 lin_norm + separate quantize).
void launch_qwen36_gated_norm_q8(const void* x_bf16, const void* z_bf16,
                                 const void* weight_bf16, void* out_q8,
                                 int v_heads, int head_dim, float eps,
                                 cudaStream_t stream = nullptr);

// DEBUG ONLY (Muse Glimmer bring-up): prints "[mgstage] step=.. layer=.. tag=.. n=..
// l2=.. v0=.. v1=.. v2=.." via device printf. Capturable (no host sync), so it can be
// dropped into a CUDA-graph-captured forward pass -- it fires once per launch/replay.
// Gated at every call site behind SPARKINFER_MG_STAGE_DEBUG; safe to leave compiled in.
void launch_mg_debug_bf16(const void* x_bf16, int n, int tag, int layer, int step,
                          cudaStream_t stream = nullptr);
void launch_mg_debug_f32(const float* x_f32, int n, int tag, int layer, int step,
                         cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
