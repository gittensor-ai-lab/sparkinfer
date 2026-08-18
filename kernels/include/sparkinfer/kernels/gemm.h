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
bool launch_gemv_rows(const void* x, const void* W, void* y,
                      int M, int N, int K, cudaStream_t stream = nullptr);
bool launch_gemv_rows_f32(const void* x, const void* W, float* y,
                          int M, int N, int K, cudaStream_t stream = nullptr);

// Fused GEMV + sigmoid for the shared-expert gate scalar (N=1). Uses scratch_bf16
// for the bf16 dot (same as launch_gemv) then sigmoid_scalar. SPARKINFER_GEMV_SIGMOID=1 enables.
void launch_gemv_sigmoid(const void* x, const void* W, void* scratch_bf16, float* y, int K,
                         cudaStream_t stream = nullptr);

// Quantized on-read GEMV: same as launch_gemv but W is GGUF-native Q4_K/Q6_K/Q8_0
// [N,K] (wtype = ggml type id, 12=Q4_K / 14=Q6_K / 8=Q8_0). Dequantizes each block in
// registers with a full-precision (fp32) activation dot — reads the quantized
// bytes (2-4x less than bf16) with no int8 activation, so token-match is preserved.
void launch_gemv_q(const void* x, const void* W, int wtype, void* y, int N, int K,
                   cudaStream_t stream = nullptr);
void launch_gemv_q_f32(const void* x, const void* W, int wtype, float* y, int N, int K,
                       cudaStream_t stream = nullptr);

// Compressed-tensors FP8 (E4M3) on-read GEMV. W is the packed payload
// [bf16 scale[N] | e4m3 weight[N*K]] (SI_QTYPE_FP8). Dequant matches launch_ct_dequant_fp8
// then a bf16 GEMV: each weight is rounded to bf16(float(e4m3)*scale) before the dot.
void launch_gemv_fp8(const void* x, const void* W, void* y, int N, int K,
                     cudaStream_t stream = nullptr);

// Compressed-tensors NVFP4 on-read GEMV. W is the SI_QTYPE_NVFP4 payload
// [256 B header with f32 global_scale | ue4m3 scale[N*(K/16)] | packed u8[N*(K/2)]].
// Dequant matches launch_ct_dequant_nvfp4 then a bf16 GEMV: each weight is rounded
// to bf16(e2m1 * ue4m3 / global_scale) before the dot. K must be a multiple of 16.
// Row-batched NVFP4 GEMV: M activation rows [M,K] against one weight stream -> y [M,N].
// Each row reproduces launch_gemv_nvfp4's own dot/reduction order exactly, so results are
// bit-identical to M separate one-row calls; only the weight/scale loads are shared.
// Returns false for widths/shapes it does not cover, so callers keep their row loop as fallback.
bool launch_gemv_nvfp4_rows(const void* x, const void* W, void* y, int M, int N, int K,
                            cudaStream_t stream = nullptr);

void launch_gemv_nvfp4(const void* x, const void* W, void* y, int N, int K,
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

// Pull `bytes` of `p` into L2 (prefetch.global.L2 only -- no loads, no stores, no side effects).
// Intended to run on a side stream during a latency-bound kernel, so the next weight-streaming
// GEMV finds its leading slice already resident. Never changes a computed value.
void launch_l2_prefetch(const void* p, size_t bytes, cudaStream_t stream = nullptr);
// Row-batched Q8_1 quantize: `rows` activation rows of K values, x row stride `x_stride` elements,
// y rows contiguous (llama_q8_1_bytes(K) apart). Same per-row math as the single-row launcher.
void launch_quantize_q8_1_rows(const void* x, void* y, int K, int rows, int x_stride,
                               cudaStream_t stream = nullptr);
void launch_mmvq_q4k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream = nullptr);
// Two Q4_K matrices, one shared Q8_1 activation, one launch. Per-row results are bit-identical to
// two launch_mmvq_q4k calls. Returns false if the shape is unsupported.
bool launch_mmvq_q4k_kfixed2(const void* q81, const void* W0, const void* W1, void* y0, void* y1,
                             int N0, int N1, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q4k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);

// Exact short-row target verifier: M contiguous Q8_1 activations against one Q4_K matrix.
// Preserves launch_mmvq_q4k's four-warp dot/reduction order independently for every row while
// sharing the weight traffic within a CTA. y is row-major [M,N]. Returns false if unsupported.
bool launch_mmvq_q4k_rows(const void* q81, const void* W, void* y,
                          int M, int N, int K, cudaStream_t stream = nullptr);
// Fused GDN qkv+z Q4_K MMVQ (shared Q8_1 activation). K is hidden (2048 -> NSUPER=8, 4096 -> 16).
void launch_mmvq_gdn_qkv_z_pack2(const void* q81, const void* qkv_w, const void* z_w,
                                 void* qkv_out, void* z_out, int n_qkv, int n_z, int K,
                                 cudaStream_t stream = nullptr);
// Shared-expert gate scalar: Q4_K mmvq + sigmoid (K=2048/4096, N=1).
void launch_mmvq_q4k_sigmoid(const void* q81, const void* W, float* out, int K, cudaStream_t stream = nullptr);
// Same, for Q6_K weights (attn-V upgrades + LM head). q81 = block_q8_1(activation).
void launch_mmvq_q6k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q6k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);
// Exact short-row Q6_K target projection, matching the four-warp decode association per row.
bool launch_mmvq_q6k_rows(const void* q81, const void* W, void* y,
                          int M, int N, int K, cudaStream_t stream = nullptr);
// Q8_0 x Q8_1 dp4a mmvq (Qwen3.6 UD attention/GDN projections kept int8 on device)
void launch_mmvq_q80(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream = nullptr);
void launch_mmvq_q80_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);
// Exact short-row Q8_0 target projection, matching the four-warp decode association per row.
bool launch_mmvq_q80_rows(const void* q81, const void* W, void* y,
                          int M, int N, int K, cudaStream_t stream = nullptr);
// GGUF dispatch for exact short-row projections (Q8_0=8, Q4_K=12, Q6_K=14).
// Two bf16 weight matrices against one activation block in a single launch. Per-row results are
// bit-identical to two launch_gemv_rows calls.
bool launch_gemv_rows2(const void* x, const void* W0, const void* W1, void* y0, void* y1,
                       int M, int N0, int N1, int K, cudaStream_t stream = nullptr);

bool launch_mmvq_rows(int qtype, const void* q81, const void* W, void* y,
                      int M, int N, int K, cudaStream_t stream = nullptr);
// FP32-output counterpart for verifier logits. It preserves the serial decode reduction order.
bool launch_mmvq_rows_f32(int qtype, const void* q81, const void* W, float* y,
                          int M, int N, int K, cudaStream_t stream = nullptr);
// GDN: four Q4_K projections from one block_q8_1 activation in a single grid (K=2048).
void launch_gdn_quad_mmvq_q4k(const void* q81,
    const void* W0, const void* W1, const void* W2, const void* W3,
    void* y0, void* y1, void* y2, void* y3,
    int N0, int N1, int N2, int N3, int K, cudaStream_t stream = nullptr);
// Full-attn: Q+K+V Q4_K projections from one block_q8_1 activation in a single grid (K=2048).
void launch_attn_qkv_mmvq_q4k(const void* q81,
    const void* Wq, const void* Wk, const void* Wv,
    void* yq, void* yk, void* yv,
    int Nq, int Nk, int Nv, int K, cudaStream_t stream = nullptr);
// 1-warp-per-row Q6_K dp4a GEMV (large-N, e.g. LM head): GEMV_WPB rows/block.
void launch_gemv_q6k_dp4a_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream = nullptr);
// M activation rows vs one shared Q6_K weight (DFlash draft head: B block tokens, same lm_head).
// The weight streams from HBM once instead of M times. q81 = M contiguous llama_q8_1_bytes(K) rows,
// y = [M, N] fp32. Returns false if the shape is unsupported so the caller can fall back.
// Same, for a Q4_K weight (the LM-head copy the target keeps). ~280 MB vs ~417 MB at V=248k.
bool launch_gemv_q4k_dp4a_multirow_f32(const void* q81, const void* W, float* y,
                                       int N, int K, int M, cudaStream_t stream = nullptr);
bool launch_gemv_i8_q81_multirow_f32(const void* q81, const signed char* W,
                                     const float* sw, float* y,
                                     int N, int K, int M, cudaStream_t stream = nullptr);
void launch_pack_i8_rows_i4(const signed char* W_i8, const float* scale_i8,
                            unsigned char* W_i4, float* scale_i4,
                            int rows, int K, cudaStream_t stream = nullptr);
bool launch_gemv_i4_q81_multirow_f32(const void* q81, const unsigned char* W,
                                     const float* sw, float* y,
                                     int N, int K, int M, cudaStream_t stream = nullptr);
bool launch_gemv_q6k_dp4a_multirow_f32(const void* q81, const void* W, float* y,
                                       int N, int K, int M, cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
