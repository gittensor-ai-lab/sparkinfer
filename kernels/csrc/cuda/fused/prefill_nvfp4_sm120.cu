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

template <class TileShape>
struct Cfg {
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        BF, cutlass::layout::RowMajor, 8, BF, cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        E4, cutlass::layout::RowMajor, 32, E4, cutlass::layout::ColumnMajor, 32, float,
        TileShape, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Two N-tilings of the same GEMM. The 128-wide tile reuses each A tile across twice as many
// output columns and is the right default; the 64-wide one exists only to fill the machine when
// the wide grid cannot (see nvfp4_prefer_narrow).
using Wide = Cfg<Shape<_128, _128, _256>>;
using Narrow = Cfg<Shape<_128, _64, _256>>;

using Gemm = typename Wide::Gemm;
using Kernel = typename Wide::Kernel;
using StrideA = typename Kernel::StrideA;
using StrideB = typename Kernel::StrideB;
using StrideC = typename Kernel::StrideC;
using StrideD = typename Kernel::StrideD;
// Depends only on the FP4 scale-vector size, not on the tile, so one quantized operand feeds
// either config -- quant_a/quant_b stay tile-agnostic.
using ScaleConfig = typename Wide::Mainloop::Sm1xxBlkScaledConfig;
static_assert(cute::is_same_v<ScaleConfig, typename Narrow::Mainloop::Sm1xxBlkScaledConfig>,
              "both tilings must agree on the block-scale layout");

auto shape(int m, int n, int k) { return cute::make_shape(m, n, k, 1); }
auto sfa_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFA(shape(m,n,k)); }
auto sfb_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFB(shape(m,n,k)); }

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

// Muse's attention gate folded into the o-projection's A-operand quantize: att * sigmoid(qg)
// straight to FP4. The int8 leg already does this (launch_prefill_gate_quant_rows_i8) and
// deliberately leaves `att` UN-GATED; #816 was reverted because its FP4 o-projection quantized that
// raw `att` and dropped the gate on every layer (TOP1 235/512, KL 0.489). Doing the gate here means
// no path ever reads an un-gated `att`, and no separate pass over [128, qdim] is needed.
template <class Layout>
__global__ void gate_quant_rows(const __nv_bfloat16* __restrict__ src,
                                const __nv_bfloat16* __restrict__ gate,
                                unsigned char* dst, cutlass::float_ue4m3_t* sf,
                                int rows, int cols, Layout layout) {
    constexpr int V = 16, LPG = V / 2;
    const int glane = threadIdx.x & (LPG - 1);
    const int groups = rows * (cols / V);
    const int stride = (gridDim.x * blockDim.x) / LPG;
    // The four 8-lane groups in a warp can retire on different grid-stride iterations, so the amax
    // butterfly must name only its own group.
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
        if (glane == 0) { auto scales = cute::make_tensor(sf, layout); scales(row, k0, 0) = qs; }
    }
}

// The down projection is the only consumer of SwiGLU. Produce the exact bf16-rounded activation
// directly into its FP4 A operand instead of writing and rereading the 128 x 19968 bf16 tensor.
template <class Layout>
__global__ void swiglu_quant_rows(const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ up,
                                  unsigned char* dst, cutlass::float_ue4m3_t* sf,
                                  int rows, int cols, Layout layout) {
    // 8 values per lane instead of 2, so two lanes cover a 16-value scale group. The two operand
    // reads become one 16-byte load each instead of four 4-byte loads, the store becomes one
    // 4-byte store instead of four 1-byte ones, and the amax butterfly collapses from three
    // shuffles to one. This kernel moves 11.7 MB per layer in ~12 us -- 0.98 TB/s of a 1.79 TB/s
    // part -- so it was issue-bound on narrow accesses, not bandwidth-bound.
    // Bit-identical: max is order-independent, so each group keeps exactly the same scale, and
    // every value keeps the same SiLU-in-float, round-once-to-bf16, x / float(qs) sequence.
    constexpr int V = 16, VPL = 8, LPG = V / VPL;   // 2 lanes per scale group
    const int glane = threadIdx.x & (LPG - 1);
    const int groups = rows * (cols / V);
    const int stride = (gridDim.x * blockDim.x) / LPG;
    for (int grp = (blockIdx.x * blockDim.x + threadIdx.x) / LPG;
         grp < groups; grp += stride) {
        const int row = grp / (cols / V), k0 = (grp % (cols / V)) * V;
        const size_t base = (size_t)row * cols + k0 + VPL * glane;
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gate + base);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(up + base);
        float x[VPL];
        float a = 0.f;
        #pragma unroll
        for (int p = 0; p < VPL / 2; ++p) {
            const __nv_bfloat162 gp = g2[p], upv = u2[p];
            const float ga = __bfloat162float(gp.x), gb = __bfloat162float(gp.y);
            const float ua = __bfloat162float(upv.x), ub = __bfloat162float(upv.y);
            // Match pf_swiglu_kernel: SiLU and multiply in float, then round once to bf16.
            x[2 * p]     = __bfloat162float(__float2bfloat16(ga / (1.f + __expf(-ga)) * ua));
            x[2 * p + 1] = __bfloat162float(__float2bfloat16(gb / (1.f + __expf(-gb)) * ub));
            a = fmaxf(a, fmaxf(fabsf(x[2 * p]), fabsf(x[2 * p + 1])));
        }
        // The scale group is exactly this lane and its odd/even partner, which are adjacent lanes
        // in the same warp on every iteration, so a lane-1 xor is warp-safe without a group mask.
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        cutlass::float_ue4m3_t qs(fmaxf(a * (1.f / 6.f), 0x1p-9f));
        unsigned char packed[VPL / 2];
        #pragma unroll
        for (int p = 0; p < VPL / 2; ++p) {
            cutlass::float_e2m1_t q0(x[2 * p] / float(qs)), q1(x[2 * p + 1] / float(qs));
            packed[p] = (unsigned char)((q0.raw() & 15u) | ((q1.raw() & 15u) << 4));
        }
        *reinterpret_cast<unsigned int*>(dst + (base >> 1)) =
            *reinterpret_cast<const unsigned int*>(packed);
        if (glane == 0) { auto scales = cute::make_tensor(sf, layout); scales(row, k0, 0) = qs; }
    }
}

template <class C = Wide>
typename C::Gemm::Arguments args(const void* a, const void* sa, const void* b, const void* sb,
                                 void* d, int m, int n, int k) {
    auto as = cutlass::make_cute_packed_stride(StrideA{}, {m,k,1});
    auto bs = cutlass::make_cute_packed_stride(StrideB{}, {n,k,1});
    auto cs = cutlass::make_cute_packed_stride(StrideC{}, {m,n,1});
    auto ds = cutlass::make_cute_packed_stride(StrideD{}, {m,n,1});
    return {cutlass::gemm::GemmUniversalMode::kGemm, shape(m,n,k),
            {static_cast<const cutlass::float_e2m1_t*>(a), as, static_cast<const cutlass::float_e2m1_t*>(b), bs, static_cast<const cutlass::float_ue4m3_t*>(sa), sfa_layout(m,n,k), static_cast<const cutlass::float_ue4m3_t*>(sb), sfb_layout(m,n,k)},
            {{1.f, 0.f}, static_cast<const BF*>(nullptr), cs, static_cast<BF*>(d), ds}};
}

int sm_count() {
    static const int sms = [] {
        int dev = 0, c = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            return 0;
        return c;
    }();
    return sms;
}

// The wide tile wins whenever it can keep the GPU busy, because each of its CTAs amortizes one
// A-tile load over twice as many output columns. It cannot at m <= 128: the grid is then one CTA
// tall, so it is just ceil(n/128) blocks -- for Muse Glimmer's down/wo (n=6656) that is 52 CTAs on
// a 170-SM RTX 5090, i.e. under a third of the machine. Halving the N tile doubles the block count
// and the memory parallelism that goes with it. Measured on RTX 5090 at m=128 with DRAM-cold
// weights, wide -> narrow: down (n=6656,k=19968) 66.7 -> 53.6 us, wo (n=6656,k=4096) 18.4 -> 14.3,
// gate/up (n=19968,k=6656) 53.3 -> 51.2. From m=256 the wide tile is ahead at every one of those
// shapes (1.09x-1.56x), so the narrow tiling stays confined to the single-CTA-tall case.
bool prefer_narrow(int m, int n) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_NVFP4_NARROW_TILE");
        return !e || atoi(e) != 0;
    }();
    const int sms = sm_count();
    // Narrow only once the WIDE grid has stopped covering the machine. The old test (n/128 < sms)
    // sent gate/up (n=19968 -> 156 CTAs of 170) to the narrow tile as well, but 156 CTAs is already
    // ~92% of one wave, and halving N there just doubles the A-tile re-reads. Requiring the narrow
    // grid to still fit ONE wave (2*ceil(n/128) <= sms) keeps gate/up wide and leaves wo (52 CTAs)
    // and q|gate|k|v (68) narrow. Measured cold-L2 at m=128, us: gate/up narrow 60.51 -> wide 56.19;
    // wo wide 25.02 -> narrow 21.63; qkvg wide 39.14 -> narrow 31.84.
    return on && m <= 128 && sms > 0 && 2 * ((n + 127) / 128) <= sms;
}

template <class C>
bool run_gemm(const void* a, const void* sa, const void* b, const void* sb,
              void* d, int m, int n, int k, void* ws, cudaStream_t st) {
    typename C::Gemm gemm;
    auto ar = args<C>(a, sa, b, sb, d, m, n, k);
    return gemm.can_implement(ar) == cutlass::Status::kSuccess &&
           gemm.initialize(ar, ws, st) == cutlass::Status::kSuccess &&
           gemm.run(st) == cutlass::Status::kSuccess;
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
    return (size_t)cute::size(cute::filter_zeros(sfa_layout(m,128,k)));
}
size_t prefill_nvfp4_scale_bytes_b(int n, int k) {
    return (size_t)cute::size(cute::filter_zeros(sfb_layout(128,n,k)));
}
size_t prefill_nvfp4_workspace_bytes(int m, int n, int k) {
    // Either tiling may run for a given shape, so the caller's buffer has to cover both.
    const size_t w = Wide::Gemm::get_workspace_size(
        args<Wide>(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
    const size_t nw = Narrow::Gemm::get_workspace_size(
        args<Narrow>(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
    return w > nw ? w : nw;
}
bool launch_prefill_nvfp4_quant_a(const void* s, void* d, void* sf, int m, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = sfa_layout(m,128,k);
    int blocks = (m * (k / 16) + 31) / 32; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gate_quant_a(const void* sr, const void* g, void* d, void* sf,
                                       int m, int k, cudaStream_t st) {
    if (!sr || !g || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = sfa_layout(m,128,k);
    int blocks = (m * (k / 16) * 8 + 255) / 256; if (blocks > 4096) blocks = 4096;
    gate_quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)sr,(const __nv_bfloat16*)g,
                                          (unsigned char*)d,(cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_swiglu_quant_a(const void* g, const void* u, void* d, void* sf,
                                         int m, int k, cudaStream_t st) {
    if (!g || !u || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = sfa_layout(m,128,k);
    // 2 lanes per 16-value scale group (see swiglu_quant_rows), so one thread per 8 values.
    int blocks = (m * (k / 16) * 2 + 255) / 256; if (blocks > 4096) blocks = 4096;
    swiglu_quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)g,(const __nv_bfloat16*)u,
                                            (unsigned char*)d,(cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_quant_b(const void* s, void* d, void* sf, int n, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(128,n,k)) return false;
    auto l = sfb_layout(128,n,k);
    int blocks = (n * (k / 16) + 31) / 32; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,n,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gemm(const void* a,const void* sa,const void* b,const void* sb,
                               void* d,int m,int n,int k,void* ws,cudaStream_t st) {
    if (!a||!sa||!b||!sb||!d||!prefill_nvfp4_supported(m,n,k)) return false;
    return prefer_narrow(m,n) ? run_gemm<Narrow>(a,sa,b,sb,d,m,n,k,ws,st)
                              : run_gemm<Wide>(a,sa,b,sb,d,m,n,k,ws,st);
}
} // namespace sparkinfer::kernels
