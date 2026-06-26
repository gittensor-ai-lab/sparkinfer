#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

enum class GemmLayout { ROW_MAJOR, COL_MAJOR };
enum class GemmPrecision { FP16, BF16, FP8_E4M3, INT8 };

struct GemmConfig {
    GemmPrecision precision = GemmPrecision::BF16;
    GemmLayout    layout_a  = GemmLayout::ROW_MAJOR;
    GemmLayout    layout_b  = GemmLayout::COL_MAJOR;
    bool          use_tensor_cores = true;
    int           split_k = 1;
};

// C = alpha * A @ B + beta * C
// A: [M, K], B: [K, N], C: [M, N]
void launch_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    const GemmConfig& cfg,
    cudaStream_t stream = nullptr
);

// Batched GEMM: C[i] = A[i] @ B[i]
void launch_batched_gemm(
    const void** A, const void** B, void** C,
    int batch, int M, int N, int K,
    float alpha, float beta,
    const GemmConfig& cfg,
    cudaStream_t stream = nullptr
);

// Linear with fp32 output: C[M,N] = A[M,K] @ B[K,N]  (A,B bf16; C fp32).
// Used for the LM head (hidden -> vocab logits).
void launch_linear_f32(
    const void* A, const void* B, float* C,
    int M, int N, int K, cudaStream_t stream = nullptr);

// Decode GEMV: y[N] = x[K] @ W^T, W is [N,K] row-major ([out,in], GGUF-native).
// x,W bf16. y is bf16 (launch_gemv) or fp32 (launch_gemv_f32). One warp per row.
void launch_gemv(const void* x, const void* W, void* y, int N, int K,
                 cudaStream_t stream = nullptr);
void launch_gemv_f32(const void* x, const void* W, float* y, int N, int K,
                     cudaStream_t stream = nullptr);

// Quantized on-read GEMV: same as launch_gemv but W is GGUF-native Q4_K/Q6_K
// [N,K] (wtype = ggml type id, 12=Q4_K / 14=Q6_K). Dequantizes each block in
// registers with a full-precision (fp32) activation dot — reads the quantized
// bytes (4x less than bf16) with no int8 activation, so token-match is preserved.
void launch_gemv_q(const void* x, const void* W, int wtype, void* y, int N, int K,
                   cudaStream_t stream = nullptr);
void launch_gemv_q_f32(const void* x, const void* W, int wtype, float* y, int N, int K,
                       cudaStream_t stream = nullptr);

// Pre-quantized Q8_1 activation path: quantize x[K] ONCE (q8 int8 [K], ad/as [K/32]),
// then run Q4_K dp4a GEMVs that read it — kills the per-block re-quantization that the
// in-kernel dp4a path repeats N/8 times (and per GEMV). Output is bit-exact vs that path.
void launch_quantize_q8_1(const void* x, void* q8, float* ad, float* as, int K,
                          cudaStream_t stream = nullptr);
void launch_gemv_q_dp4a_pq(const void* q8, const float* ad, const float* as, const void* W,
                           void* y, int N, int K, cudaStream_t stream = nullptr);
void launch_gemv_q_dp4a_pq_f32(const void* q8, const float* ad, const float* as, const void* W,
                               float* y, int N, int K, cudaStream_t stream = nullptr);

// Faithful llama.cpp Q4_K mul_mat_vec_q: activation in block_q8_1 (llama_q8_1_bytes(K) bytes),
// nwarps=4 cooperate per row. A/B test vs our split-K dp4a (SPARKINFER_LLAMA=1).
size_t llama_q8_1_bytes(int K);
void launch_quantize_q8_1_blocks(const void* x, void* y, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q4k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q4k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);
// Same, for Q6_K weights (attn-V upgrades + LM head). q81 = block_q8_1(activation).
void launch_mmvq_q6k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q6k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);
// 1-warp-per-row Q6_K dp4a GEMV (large-N, e.g. LM head): GEMV_WPB rows/block.
void launch_gemv_q6k_dp4a_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
