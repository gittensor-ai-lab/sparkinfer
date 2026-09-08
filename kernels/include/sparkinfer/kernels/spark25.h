#pragma once
#include <cuda_runtime.h>

// Spark-X2.5 (iFlytek, model_type "spark2_5") -- the two elementwise steps this architecture needs
// that no other model wired up here does. Both are tiny and memory-bound; they exist as their own
// kernels rather than as modes threaded through the fused FFN/attention kernels so that bringing up
// a new architecture cannot change a single byte of any existing model's numerics.

namespace sparkinfer { namespace kernels {

// GeGLU: out[i] = gelu(gate[i]) * up[i], with gelu the EXACT erf form
// 0.5*x*(1+erf(x/sqrt(2))) -- transformers' ACT2FN["gelu"] is GELUActivation, i.e.
// nn.functional.gelu with approximate='none', NOT the tanh approximation. Spark-X2.5's
// reference MLP refuses to build with any hidden_act other than "gelu".
//
// Shapes match launch_prefill_swiglu: flat over n elements, so the same call serves a single
// decode row (n = ffn) and a batch of prefill rows (n = rows*ffn).
void launch_spark25_geglu(const void* gate, const void* up, void* out, long n,
                          cudaStream_t stream = nullptr);

// Head-wise attention output gate: attn[t, h*head_dim + d] *= sigmoid(gate[t, h]).
//
// ONE sigmoid scalar per head, broadcast across that head's head_dim elements -- Spark-X2.5's
// g_proj is [hidden, num_heads], not the [hidden, num_heads*head_dim] that
// launch_qwen36_mul_sigmoid consumes elementwise for Qwen3.6/Qwen3.8/Muse Glimmer. The sigmoid
// is taken in fp32 before the multiply, matching the reference
// (torch.sigmoid(gate_score.float()) then .to(attn_output.dtype)).
void launch_spark25_mul_sigmoid_headwise(void* attn, const void* gate, int n_tokens,
                                         int n_heads, int head_dim,
                                         cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
