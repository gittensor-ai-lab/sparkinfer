#pragma once
#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Kernels the vision tower needs that the text path does not have.
//
// The text model is RMSNorm + SwiGLU throughout; the vision tower is LayerNorm + GELU, and its
// attention is FULL and bidirectional over patches with no KV cache and no mask. So these are
// genuinely new rather than a re-parameterisation of existing kernels.
//
// Deliberately simple: the tower is 27 blocks at hidden=1152 over at most a few thousand patches,
// which is a rounding error next to a 27B decode step. Correctness against bench/scripts/vision_ref.py
// comes first; if the tower ever shows up in a profile it can be revisited.
// All tensors bf16, fp32 accumulation.

// out[r, :] = (x[r, :] - mean) / sqrt(var + eps) * weight + bias
// LayerNorm proper -- subtracts the mean, unlike the text path's RMSNorm. Both weight AND bias,
// which is what the checkpoint's norm1/norm2/merger.norm tensors carry.
void launch_vision_layernorm(const void* x, const void* weight, const void* bias, void* out,
                             int n_rows, int dim, float eps, cudaStream_t stream = nullptr);

// x[r, c] += bias[c]. launch_prefill_gemm has no bias term, so every vision Linear is a GEMM
// followed by this.
void launch_vision_add_bias(void* x, const void* bias, int n_rows, int dim,
                            cudaStream_t stream = nullptr);

// x[i] = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))), in place.
// The TANH approximation, matching config's hidden_act="gelu_pytorch_tanh". The erf form differs
// by ~1e-3 near the knee, which compounds over 27 blocks.
void launch_vision_gelu_tanh(void* x, long n, cudaStream_t stream = nullptr);

// acc[i] += add[i], in place -- the residual connections.
void launch_vision_residual_add(void* acc, const void* add, long n, cudaStream_t stream = nullptr);

// Full bidirectional attention over all patches, one kernel per (head, query).
// q/k/v are [n_tokens, n_heads*head_dim] as the QKV projection leaves them; out matches.
// No mask: a causal mask here is the classic silent bug -- it still produces embeddings, just
// wrong ones, and nothing downstream can tell.
// 2D rotary position embedding for the vision tower, applied IN PLACE to q and k before
// attention. Every block re-projects qkv, so this runs once per block, not once per image.
//
// cos_table/sin_table are float [n_tokens, head_dim/2], row-major over the patch grid, and hold
// only the DISTINCT half of the frequencies: the reference concatenates its width-head_dim/2
// frequency vector with itself before taking cos/sin, so the upper half repeats the lower.
//
// Omitting this entirely is what made the tower agree with a hand-written reference while both
// disagreed with transformers at cosine 0.77 -- see bench/scripts/vision_hf_check.py.
void launch_vision_rope(void* q, void* k, const void* cos_table, const void* sin_table,
                        int n_tokens, int n_heads, int head_dim, cudaStream_t stream = nullptr);

void launch_vision_attention(const void* q, const void* k, const void* v, void* out,
                             int n_tokens, int n_heads, int head_dim, float scale,
                             cudaStream_t stream = nullptr);

// Regroup row-major patches into merge x merge blocks:
//   out[b, (r*merge + c)*dim + d] = x[((by*merge + r)*grid_w + (bx*merge + c)), d]
// where b = by*(grid_w/merge) + bx. This is the transpose vision_ref.py does at the merger, and
// it is what makes a row-major patch order equivalent to HF's merge-block-major one (proved to
// 1e-15 in bench/scripts/vision_order_check.py).
void launch_vision_patch_merge(const void* x, void* out, int grid_h, int grid_w, int merge,
                               int dim, cudaStream_t stream = nullptr);

// Overwrite selected rows of the embedded prompt with vision embeddings:
//   x[positions[i], :] = emb[i, :]   for i in [0, n_img)
// x is the [n_tokens, hidden] bf16 output of launch_embedding; emb is [n_img, hidden] bf16.
// positions is device-side and must be strictly increasing (the caller derives it from the token
// stream). Nothing here validates the COUNT -- qwen_vision_splice_embeddings does that on the
// host, before any device work, because a count mismatch shifts every later token and must be an
// error rather than a partial write.
void launch_vision_splice(void* x, const int* positions, const void* emb,
                          int n_img, int hidden, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
