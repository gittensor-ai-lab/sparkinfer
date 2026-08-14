#include "sparkinfer/kernels/prefill_nvfp4.h"

#include <cstdlib>
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

namespace sparkinfer::kernels {
namespace {
using namespace cute;
using E4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF = cutlass::bfloat16_t;
using Cluster = Shape<_1, _1, _1>;

// The GEMM stack, parameterized on the CTA tile. At m == 128 rows there is one M-tile, so
// CTAs = n/TILE_N: with the 128-wide tile the two hidden-output GEMMs (ffn_down, wo; n=6656)
// launch 52 CTAs on a 170-SM part and stream their weights at ~1.06 TB/s, and even gate/up's
// 156 CTAs are one incomplete wave at 1.24. The 64-wide tile doubles the CTAs in flight and
// measured faster for every shape this path serves. The scale-factor layout atoms come from
// the UMMA blockscaled format (Sm1xxBlkScaledConfig), not the CTA tile, so both variants read
// the same packed A/B scale buffers -- asserted below.
template <int TILE_N, int TILE_K = 128>
struct FP4Stack {
    using Tile = Shape<_128, Int<TILE_N>, Int<TILE_K>>;
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
    using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
};
using Wide = FP4Stack<128>;
using Skinny = FP4Stack<64>;
// 256-deep K tile: fewer mainloop trips and longer contiguous weight reads per CTA;
// measured additive on top of the 64-wide tile. Shipped default; K=128 stays for the env A/B.
using SkinnyK256 = FP4Stack<64, 256>;
using WideK256 = FP4Stack<128, 256>;
using Kernel = Wide::Kernel;
using Gemm = Wide::Gemm;
using StrideA = typename Kernel::StrideA;
using StrideB = typename Kernel::StrideB;
using StrideC = typename Kernel::StrideC;
using StrideD = typename Kernel::StrideD;
using ScaleConfig = typename Wide::Mainloop::Sm1xxBlkScaledConfig;

auto shape(int m, int n, int k) { return cute::make_shape(m, n, k, 1); }
auto sfa_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFA(shape(m,n,k)); }
auto sfb_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFB(shape(m,n,k)); }
// Both tile variants must agree on the packed scale buffer layouts, or the skinny variant
// would read scales packed for the wide one.
static_assert(std::is_same_v<decltype(Wide::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1))),
                             decltype(Skinny::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1)))> &&
              std::is_same_v<decltype(Wide::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1))),
                             decltype(SkinnyK256::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1)))> &&
              std::is_same_v<decltype(Wide::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1))),
                             decltype(WideK256::ScaleConfig::tile_atom_to_shape_SFA(
                                 cute::make_shape(1, 1, 1, 1)))>,
              "scale-factor layouts diverge across tile variants");

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

template <class S>
typename S::Gemm::Arguments args_t(const void* a, const void* sa, const void* b, const void* sb,
                                   void* d, int m, int n, int k) {
    auto as = cutlass::make_cute_packed_stride(typename S::Kernel::StrideA{}, {m,k,1});
    auto bs = cutlass::make_cute_packed_stride(typename S::Kernel::StrideB{}, {n,k,1});
    auto cs = cutlass::make_cute_packed_stride(typename S::Kernel::StrideC{}, {m,n,1});
    auto ds = cutlass::make_cute_packed_stride(typename S::Kernel::StrideD{}, {m,n,1});
    return {cutlass::gemm::GemmUniversalMode::kGemm, shape(m,n,k),
            {static_cast<const cutlass::float_e2m1_t*>(a), as, static_cast<const cutlass::float_e2m1_t*>(b), bs, static_cast<const cutlass::float_ue4m3_t*>(sa), sfa_layout(m,n,k), static_cast<const cutlass::float_ue4m3_t*>(sb), sfb_layout(m,n,k)},
            {{1.f, 0.f}, static_cast<const BF*>(nullptr), cs, static_cast<BF*>(d), ds}};
}
Gemm::Arguments args(const void* a, const void* sa, const void* b, const void* sb,
                     void* d, int m, int n, int k) {
    return args_t<Wide>(a, sa, b, sb, d, m, n, k);
}
template <class S>
bool run_gemm(const void* a, const void* sa, const void* b, const void* sb,
              void* d, int m, int n, int k, void* ws, cudaStream_t st) {
    typename S::Gemm gemm; auto ar = args_t<S>(a,sa,b,sb,d,m,n,k);
    return gemm.can_implement(ar) == cutlass::Status::kSuccess &&
           gemm.initialize(ar,ws,st) == cutlass::Status::kSuccess &&
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
    size_t r = Gemm::get_workspace_size(args(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
    const size_t cand[3] = {
        Skinny::Gemm::get_workspace_size(args_t<Skinny>(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k)),
        SkinnyK256::Gemm::get_workspace_size(args_t<SkinnyK256>(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k)),
        WideK256::Gemm::get_workspace_size(args_t<WideK256>(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k))};
    for (size_t c : cand) if (c > r) r = c;
    return r;
}
bool launch_prefill_nvfp4_quant_a(const void* s, void* d, void* sf, int m, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = sfa_layout(m,128,k);
    int blocks = (m * (k / 16) + 31) / 32; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,m,k,l);
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
    // At m = 128 rows one M-tile is all there is, so CTAs = n / TILE_N: the 128-wide tile
    // put 52 CTAs on 170 SMs for the hidden-output GEMMs (ffn_down, wo) and one incomplete
    // 156-CTA wave for gate/up. The 64-wide tile doubles both -- measured faster for every
    // shape this path serves, so it is the default; SPARKINFER_MUSE_NVFP4_SKINNY sets an
    // n threshold above which the 128-wide tile returns (0 restores it everywhere).
    static int sk = -1;
    if (sk < 0) {
        const char* e = getenv("SPARKINFER_MUSE_NVFP4_SKINNY");
        sk = e ? atoi(e) : 1 << 30;
    }
    // 256-deep K tile: fewer mainloop trips, longer contiguous weight reads per CTA --
    // measured +0.8% on top of the 64-wide tile, output bit-identical. =0 restores K=128.
    static int k256 = -1;
    if (k256 < 0) { const char* e = getenv("SPARKINFER_MUSE_NVFP4_K256"); k256 = (e && e[0]=='0') ? 0 : 1; }
    if (n <= sk) return k256 ? run_gemm<SkinnyK256>(a,sa,b,sb,d,m,n,k,ws,st)
                             : run_gemm<Skinny>(a,sa,b,sb,d,m,n,k,ws,st);
    return k256 ? run_gemm<WideK256>(a,sa,b,sb,d,m,n,k,ws,st)
                : run_gemm<Wide>(a,sa,b,sb,d,m,n,k,ws,st);
}
} // namespace sparkinfer::kernels
