#include "sparkinfer/kernels/prefill_nvfp4.h"

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

#include <cstdlib>

namespace sparkinfer::kernels {
namespace {
using namespace cute;
using E4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF = cutlass::bfloat16_t;
using Cluster = Shape<_1, _1, _1>;

// Batched prefill runs M = 128, so the grid is 1 x ceil(N / TileN) CTAs and the tile shape alone
// decides both how much of the GPU the GEMM covers and how deep its mainloop pipelines. The
// shipped 128x128x128 tile is tuned for the fat FFN gate/up projection and leaves the narrower
// projections well short. Measured cold-L2 on an RTX 5090 (170 SMs), M = 128, median of 9,
// microseconds -- every variant produces a bitwise-identical result to the shipped tile:
//
//   tile            gate_up      ffn_down          wo    q/gate/k/v
//                  N=19968       N=6656       N=6656       N=8704
//                  K=6656       K=19968       K=4096       K=6656
//   128x128x128      59.74        88.54        22.94        36.54     <- shipped
//   128x64x128       67.52        84.77        26.37        41.89
//   128x128x256    * 56.19        88.48        25.02        39.14
//   128x64x256       60.51      * 74.05      * 21.63      * 31.84
//   256x128x128      70.27       131.58       35.07         53.98
//
// The lever is the K tile, not the N tile: a 256-deep K step gives the mainloop twice the work
// per stage to hide the weight fetch behind, worth -16% on ffn_down and -13% on q/gate/k/v.
// Narrowing N on top of that pays only once the wide tile has stopped covering the machine
// (ffn_down and wo issue 52 CTAs, q/gate/k/v 68, against gate/up's 156 of 170 SMs) -- narrowing
// N for gate/up costs 8%, so the choice is made per call from the CTA count.
//
// Both instantiations read the SAME weights: the GMEM block-scale layout comes from
// Sm1xxBlockScaledConfig<SFVecSize>::tile_atom_to_shape_SFA/SFB
// (cutlass/detail/sm100_blockscaled_layout.hpp), which is a function of SFVecSize and the problem
// shape only -- the CTA tile appears solely in the smem layouts, inside the kernel. So the
// load-time conversion is unchanged and the dispatch below is free to pick per call.
template <class Tile>
struct Fp4Gemm {
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        BF, cutlass::layout::RowMajor, 8, BF, cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        E4, cutlass::layout::RowMajor, 32, E4, cutlass::layout::ColumnMajor, 32, float,
        Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideC = typename Kernel::StrideC;
    using StrideD = typename Kernel::StrideD;
    using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;

    static auto shape(int m, int n, int k) { return cute::make_shape(m, n, k, 1); }

    static typename Gemm::Arguments args(const void* a, const void* sa, const void* b,
                                         const void* sb, void* d, int m, int n, int k) {
        auto as = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
        auto bs = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
        auto cs = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
        auto ds = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
        return {cutlass::gemm::GemmUniversalMode::kGemm, shape(m, n, k),
                {static_cast<const cutlass::float_e2m1_t*>(a), as,
                 static_cast<const cutlass::float_e2m1_t*>(b), bs,
                 static_cast<const cutlass::float_ue4m3_t*>(sa),
                 ScaleConfig::tile_atom_to_shape_SFA(shape(m, n, k)),
                 static_cast<const cutlass::float_ue4m3_t*>(sb),
                 ScaleConfig::tile_atom_to_shape_SFB(shape(m, n, k))},
                {{1.f, 0.f}, static_cast<const BF*>(nullptr), cs, static_cast<BF*>(d), ds}};
    }

    static bool run(const void* a, const void* sa, const void* b, const void* sb, void* d,
                    int m, int n, int k, void* ws, cudaStream_t st) {
        Gemm gemm;
        auto ar = args(a, sa, b, sb, d, m, n, k);
        return gemm.can_implement(ar) == cutlass::Status::kSuccess &&
               gemm.initialize(ar, ws, st) == cutlass::Status::kSuccess &&
               gemm.run(st) == cutlass::Status::kSuccess;
    }

    static size_t workspace(int m, int n, int k) {
        return Gemm::get_workspace_size(args(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k));
    }
};

using WideTile = Shape<_128, _128, _256>;     // fat N (ffn gate/up)
using NarrowTile = Shape<_128, _64, _256>;    // everything that does not cover the machine
using WideGemm = Fp4Gemm<WideTile>;
using NarrowGemm = Fp4Gemm<NarrowTile>;

// The scale layouts are tile-independent (see above), so one instantiation can answer for both.
using SizeGemm = WideGemm;

int sm_count() {
    static int sms = -1;
    if (sms < 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            sms = 170;
    }
    return sms;
}

// Halve the tile width once the wide tile has stopped covering the machine: at 2 * ceil(N/128)
// <= SM count the narrow tile still fits in one wave, so the extra CTAs are free, while for
// gate/up (156 of 170 SMs already) splitting would cost a second wave.
// SPARKINFER_MUSE_NVFP4_TILE=wide|narrow pins it so both arms of an A/B come out of one binary.
bool use_narrow(int n) {
    static int mode = -1;
    if (mode < 0) {
        const char* e = getenv("SPARKINFER_MUSE_NVFP4_TILE");
        mode = (e && e[0] == 'w') ? 0 : (e && e[0] == 'n') ? 1 : 2;
    }
    if (mode != 2) return mode == 1;
    return 2 * ((n + 127) / 128) <= sm_count();
}

template <class Layout>
__global__ void quant_rows(const __nv_bfloat16* src, unsigned char* dst,
                           cutlass::float_ue4m3_t* sf, int rows, int cols, Layout layout) {
    // V is fixed by the format: one ue4m3 scale per 16 values. The mapping of lanes onto those 16
    // is not. One lane per value left half of every warp idle (lane < V), reduced 16 real values
    // across 32 lanes, and stored 8 bytes per warp per step -- 63 GB/s, 3.5% of peak. Two values
    // per lane puts 8 lanes on a group and 4 groups on a warp: every lane live, the amax reduction
    // is 3 steps inside its own 8-lane group, and the warp's 4 groups store 32 contiguous bytes.
    // Bit-identical: max is order-independent, and each value keeps the same scale and the same
    // x / float(qs) rounding it had before.
    constexpr int V = 16;
    constexpr int LPV = 8;            // lanes per scale group (2 values each)
    constexpr int GPW = 32 / LPV;     // scale groups per warp
    const int vpr = cols / V;
    const int gpb = (int)(blockDim.x >> 5) * GPW;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sub = lane & (LPV - 1);
    const int gin = lane >> 3;
    const int total = rows * vpr;
    for (int g = blockIdx.x * gpb + warp * GPW + gin; g < total; g += gridDim.x * gpb) {
        const int row = g / vpr, k0 = (g - row * vpr) * V;
        const size_t base = (size_t)row * cols + k0 + sub * 2;
        const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(src + base);
        const float x0 = __bfloat162float(v2.x), x1 = __bfloat162float(v2.y);
        float a = fmaxf(fabsf(x0), fabsf(x1));
        #pragma unroll
        for (int d = LPV >> 1; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d));
        cutlass::float_ue4m3_t qs(fmaxf(a * (1.f / 6.f), 0x1p-9f));
        cutlass::float_e2m1_t q0(x0 / float(qs));
        cutlass::float_e2m1_t q1(x1 / float(qs));
        dst[base >> 1] = (unsigned char)((q0.raw() & 15u) | ((q1.raw() & 15u) << 4));
        if (sub == 0) {
            auto scales = cute::make_tensor(sf, layout);
            scales(row, k0, 0) = qs;
        }
    }
}

// SwiGLU folded into the down projection's A-operand quantize. The unfused pair writes the
// [128, 19968] product back as bf16 (5.1 MB a layer) purely so the quantizer can read it again;
// fusing deletes that store and that load. One ue4m3 scale still covers 16 values, but the 16 are
// spread over 8 lanes at 2 values each, so every lane is live, the amax reduction runs inside its
// own 8-lane group, and each lane's two neighbours pack into the single byte it stores -- 32
// contiguous bytes per warp.
//
// The product is rounded through bf16 before quantizing, exactly as pf_swiglu_kernel stores it,
// so this emits byte-identical FP4 to launch_prefill_swiglu + launch_prefill_nvfp4_quant_a.
template <class Layout>
__global__ void swiglu_quant_rows(const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ up,
                                  unsigned char* dst, cutlass::float_ue4m3_t* sf,
                                  int rows, int cols, Layout layout) {
    constexpr int V = 16;            // values per ue4m3 scale group
    constexpr int LPG = V / 2;       // 8 lanes per group, 2 values each
    const int glane = threadIdx.x & (LPG - 1);
    const int groups = rows * (cols / V);
    const int stride = (gridDim.x * blockDim.x) / LPG;
    // The four 8-lane groups in a warp can retire on different iterations of the grid-stride
    // loop, so the amax butterfly must name only its own group, never the whole warp.
    const unsigned gmask = 0xFFu << (threadIdx.x & 24);
    for (int grp = (blockIdx.x * blockDim.x + threadIdx.x) / LPG; grp < groups; grp += stride) {
        const int row = grp / (cols / V), k0 = (grp % (cols / V)) * V;
        const size_t base = (size_t)row * cols + k0 + 2 * glane;
        const float g0 = __bfloat162float(gate[base]),     u0 = __bfloat162float(up[base]);
        const float g1 = __bfloat162float(gate[base + 1]), u1 = __bfloat162float(up[base + 1]);
        const float x0 = __bfloat162float(__float2bfloat16(g0 / (1.f + __expf(-g0)) * u0));
        const float x1 = __bfloat162float(__float2bfloat16(g1 / (1.f + __expf(-g1)) * u1));
        float a = fmaxf(fabsf(x0), fabsf(x1));
        #pragma unroll
        for (int d = LPG / 2; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(gmask, a, d));
        cutlass::float_ue4m3_t qs(fmaxf(a * (1.f / 6.f), 0x1p-9f));
        cutlass::float_e2m1_t q0(x0 / float(qs)), q1(x1 / float(qs));
        dst[base >> 1] = (unsigned char)((q0.raw() & 15u) | ((q1.raw() & 15u) << 4));
        if (glane == 0) {
            auto scales = cute::make_tensor(sf, layout);
            scales(row, k0, 0) = qs;
        }
    }
}
// Muse's attention gate folded into the o-projection's A-operand quantize: att * sigmoid(qg)
// straight to FP4. The int8 leg already does exactly this
// (launch_prefill_gate_quant_rows_i8), and the reverted #816 came unstuck precisely because its
// FP4 leg did NOT -- it quantized the raw `att` while the gate was riding along inside the int8
// quantize, dropping the gate on every layer. Doing the gate here means no path ever reads an
// un-gated `att`, and no separate pass over [128, 4096] is needed either.
template <class Layout>
__global__ void gate_quant_rows(const __nv_bfloat16* __restrict__ src,
                                const __nv_bfloat16* __restrict__ gate,
                                unsigned char* dst, cutlass::float_ue4m3_t* sf,
                                int rows, int cols, Layout layout) {
    constexpr int V = 16, LPG = V / 2;
    const int glane = threadIdx.x & (LPG - 1);
    const int groups = rows * (cols / V);
    const int stride = (gridDim.x * blockDim.x) / LPG;
    const unsigned gmask = 0xFFu << (threadIdx.x & 24);
    for (int grp = (blockIdx.x * blockDim.x + threadIdx.x) / LPG; grp < groups; grp += stride) {
        const int row = grp / (cols / V), k0 = (grp % (cols / V)) * V;
        const size_t base = (size_t)row * cols + k0 + 2 * glane;
        const float s0 = __bfloat162float(src[base]),     g0 = __bfloat162float(gate[base]);
        const float s1 = __bfloat162float(src[base + 1]), g1 = __bfloat162float(gate[base + 1]);
        // Same expression and same bf16 rounding as pf_mul_sigmoid_kernel.
        const float x0 = __bfloat162float(__float2bfloat16(s0 / (1.f + __expf(-g0))));
        const float x1 = __bfloat162float(__float2bfloat16(s1 / (1.f + __expf(-g1))));
        float a = fmaxf(fabsf(x0), fabsf(x1));
        #pragma unroll
        for (int d = LPG / 2; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(gmask, a, d));
        cutlass::float_ue4m3_t qs(fmaxf(a * (1.f / 6.f), 0x1p-9f));
        cutlass::float_e2m1_t q0(x0 / float(qs)), q1(x1 / float(qs));
        dst[base >> 1] = (unsigned char)((q0.raw() & 15u) | ((q1.raw() & 15u) << 4));
        if (glane == 0) {
            auto scales = cute::make_tensor(sf, layout);
            scales(row, k0, 0) = qs;
        }
    }
}

} // namespace

bool prefill_nvfp4_supported(int m, int n, int k) {
    int dev=0, major=0, minor=0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
           major == 12 && minor == 0 && m > 0 && !(m & 7) && !(n & 127) && !(k & 127);
}
size_t prefill_nvfp4_data_bytes(int r, int c) { return ((size_t)r*c + 1)/2; }
size_t prefill_nvfp4_scale_bytes_a(int m, int k) {
    return (size_t)cute::size(cute::filter_zeros(
        SizeGemm::ScaleConfig::tile_atom_to_shape_SFA(SizeGemm::shape(m,128,k))));
}
size_t prefill_nvfp4_scale_bytes_b(int n, int k) {
    return (size_t)cute::size(cute::filter_zeros(
        SizeGemm::ScaleConfig::tile_atom_to_shape_SFB(SizeGemm::shape(128,n,k))));
}
size_t prefill_nvfp4_workspace_bytes(int m, int n, int k) {
    const size_t w = WideGemm::workspace(m, n, k);
    const size_t nw = NarrowGemm::workspace(m, n, k);
    return w > nw ? w : nw;   // one arena serves whichever tile the dispatch picks
}
bool launch_prefill_nvfp4_quant_a(const void* s, void* d, void* sf, int m, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = SizeGemm::ScaleConfig::tile_atom_to_shape_SFA(SizeGemm::shape(m,128,k));
    int blocks = (m * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_swiglu_quant_a(const void* g, const void* u, void* d, void* sf,
                                         int m, int k, cudaStream_t st) {
    if (!g || !u || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = SizeGemm::ScaleConfig::tile_atom_to_shape_SFA(SizeGemm::shape(m,128,k));
    int blocks = (m * (k / 16) * 8 + 255) / 256; if (blocks > 4096) blocks = 4096;
    swiglu_quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,
                                            (unsigned char*)d,(cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gate_quant_a(const void* s, const void* g, void* d, void* sf,
                                       int m, int k, cudaStream_t st) {
    if (!s || !g || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = SizeGemm::ScaleConfig::tile_atom_to_shape_SFA(SizeGemm::shape(m,128,k));
    int blocks = (m * (k / 16) * 8 + 255) / 256; if (blocks > 4096) blocks = 4096;
    gate_quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(const __nv_bfloat16*)g,
                                          (unsigned char*)d,(cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_quant_b(const void* s, void* d, void* sf, int n, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(128,n,k)) return false;
    auto l = SizeGemm::ScaleConfig::tile_atom_to_shape_SFB(SizeGemm::shape(128,n,k));
    int blocks = (n * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,n,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gemm(const void* a,const void* sa,const void* b,const void* sb,
                               void* d,int m,int n,int k,void* ws,cudaStream_t st) {
    if (!a||!sa||!b||!sb||!d||!prefill_nvfp4_supported(m,n,k)) return false;
    if (use_narrow(n) && NarrowGemm::run(a, sa, b, sb, d, m, n, k, ws, st)) return true;
    return WideGemm::run(a, sa, b, sb, d, m, n, k, ws, st);
}
} // namespace sparkinfer::kernels
