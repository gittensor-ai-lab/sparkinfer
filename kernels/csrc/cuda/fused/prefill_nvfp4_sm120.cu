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
#include <algorithm>

namespace sparkinfer::kernels {
namespace {
using namespace cute;
using E4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF = cutlass::bfloat16_t;
using Cluster = Shape<_1, _1, _1>;

template<class CtaShape>
struct Nvfp4Plan {
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, CtaShape, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        BF, cutlass::layout::RowMajor, 8, BF, cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
        E4, cutlass::layout::RowMajor, 32, E4, cutlass::layout::ColumnMajor, 32, float,
        CtaShape, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue, void>;
    using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using Scale = typename Mainloop::Sm1xxBlkScaledConfig;
    static auto problem(int m, int n, int k) { return cute::make_shape(m,n,k,1); }
    static typename Device::Arguments arguments(const void* a,const void* sa,const void* b,
            const void* sb,void* d,int m,int n,int k) {
        using SA=typename Kernel::StrideA; using SB=typename Kernel::StrideB;
        using SC=typename Kernel::StrideC; using SD=typename Kernel::StrideD;
        auto p=problem(m,n,k);
        return {cutlass::gemm::GemmUniversalMode::kGemm,p,
            {static_cast<const cutlass::float_e2m1_t*>(a),cutlass::make_cute_packed_stride(SA{}, {m,k,1}),
             static_cast<const cutlass::float_e2m1_t*>(b),cutlass::make_cute_packed_stride(SB{}, {n,k,1}),
             static_cast<const cutlass::float_ue4m3_t*>(sa),Scale::tile_atom_to_shape_SFA(p),
             static_cast<const cutlass::float_ue4m3_t*>(sb),Scale::tile_atom_to_shape_SFB(p)},
            {{1.f,0.f},static_cast<const BF*>(nullptr),cutlass::make_cute_packed_stride(SC{}, {m,n,1}),
             static_cast<BF*>(d),cutlass::make_cute_packed_stride(SD{}, {m,n,1})}};
    }
    static bool launch(const void* a,const void* sa,const void* b,const void* sb,void* d,
            int m,int n,int k,void* ws,cudaStream_t st) {
        Device op; auto x=arguments(a,sa,b,sb,d,m,n,k);
        return op.can_implement(x)==cutlass::Status::kSuccess &&
               op.initialize(x,ws,st)==cutlass::Status::kSuccess && op.run(st)==cutlass::Status::kSuccess;
    }
    static size_t workspace(int m,int n,int k) {
        return Device::get_workspace_size(arguments(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
    }
};
using FullPlan = Nvfp4Plan<Shape<_128,_128,_256>>;
using SplitPlan = Nvfp4Plan<Shape<_128,_64,_256>>;
using LayoutPlan = FullPlan;

bool split_n_tile(int n) {
    static int override = -1;
    if (override < 0) { const char* e=getenv("SPARKINFER_MUSE_NVFP4_TILE");
        override=(e&&e[0]=='w')?0:(e&&e[0]=='n')?1:2; }
    if (override < 2) return override == 1;
    static int sms = 0;
    if (!sms) { int dev=0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev); }
    return 2 * ((n + 127) / 128) <= sms;
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

enum class FuseOp { SwiGLU, SigmoidGate };

// Common producer+quantizer. Each eight-lane subgroup owns one 16-value scale block and each lane
// produces two adjacent values. Keeping the producer as a template operation avoids maintaining
// two nearly identical quantizers and guarantees both paths use the same scale/store mapping.
template<FuseOp Op, class Layout>
__global__ void produce_fp4(const __nv_bfloat16* lhs, const __nv_bfloat16* rhs,
                            unsigned char* dst, cutlass::float_ue4m3_t* sf,
                            int rows, int cols, Layout layout) {
    constexpr int lanes_per_group=8;
    const int subgroup=threadIdx.x & 7;
    const int group_count=rows*(cols/16);
    const int groups_per_grid=(gridDim.x*blockDim.x)/lanes_per_group;
    const unsigned mask=0xffu << (threadIdx.x & 24);
    for (int g=(blockIdx.x*blockDim.x+threadIdx.x)/lanes_per_group;
         g<group_count; g+=groups_per_grid) {
        const int row=g/(cols/16), col=(g%(cols/16))*16+subgroup*2;
        const size_t pos=(size_t)row*cols+col;
        const __nv_bfloat162 l=*reinterpret_cast<const __nv_bfloat162*>(lhs+pos);
        const __nv_bfloat162 r=*reinterpret_cast<const __nv_bfloat162*>(rhs+pos);
        float x[2];
        if constexpr (Op == FuseOp::SwiGLU) {
            const float a0=__bfloat162float(l.x), a1=__bfloat162float(l.y);
            x[0]=__bfloat162float(__float2bfloat16(a0/(1.f+__expf(-a0))*__bfloat162float(r.x)));
            x[1]=__bfloat162float(__float2bfloat16(a1/(1.f+__expf(-a1))*__bfloat162float(r.y)));
        } else {
            x[0]=__bfloat162float(__float2bfloat16(__bfloat162float(l.x)/(1.f+__expf(-__bfloat162float(r.x)))));
            x[1]=__bfloat162float(__float2bfloat16(__bfloat162float(l.y)/(1.f+__expf(-__bfloat162float(r.y)))));
        }
        float peak=fmaxf(fabsf(x[0]),fabsf(x[1]));
#pragma unroll
        for(int d=4;d;d>>=1) peak=fmaxf(peak,__shfl_xor_sync(mask,peak,d));
        cutlass::float_ue4m3_t scale(fmaxf(peak/6.f,0x1p-9f));
        cutlass::float_e2m1_t q0(x[0]/float(scale)),q1(x[1]/float(scale));
        dst[pos>>1]=(unsigned char)((q0.raw()&15u)|((q1.raw()&15u)<<4));
        if(subgroup==0) { auto scales=cute::make_tensor(sf,layout); scales(row,col-subgroup*2,0)=scale; }
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
    return (size_t)cute::size(cute::filter_zeros(LayoutPlan::Scale::tile_atom_to_shape_SFA(LayoutPlan::problem(m,128,k))));
}
size_t prefill_nvfp4_scale_bytes_b(int n, int k) {
    return (size_t)cute::size(cute::filter_zeros(LayoutPlan::Scale::tile_atom_to_shape_SFB(LayoutPlan::problem(128,n,k))));
}
size_t prefill_nvfp4_workspace_bytes(int m, int n, int k) {
    return std::max(FullPlan::workspace(m,n,k),SplitPlan::workspace(m,n,k));
}
bool launch_prefill_nvfp4_quant_a(const void* s, void* d, void* sf, int m, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = LayoutPlan::Scale::tile_atom_to_shape_SFA(LayoutPlan::problem(m,128,k));
    int blocks = (m * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
template<FuseOp Op>
static bool launch_producer(const void* x,const void* y,void* d,void* sf,int m,int k,cudaStream_t st) {
    if(!x||!y||!d||!sf||!prefill_nvfp4_supported(m,128,k)) return false;
    auto layout=LayoutPlan::Scale::tile_atom_to_shape_SFA(LayoutPlan::problem(m,128,k));
    int blocks=(m*(k/16)*8+255)/256; if(blocks>4096) blocks=4096;
    produce_fp4<Op><<<blocks,256,0,st>>>((const __nv_bfloat16*)x,(const __nv_bfloat16*)y,
        (unsigned char*)d,(cutlass::float_ue4m3_t*)sf,m,k,layout);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool launch_prefill_nvfp4_swiglu_quant_a(const void* g,const void* u,void* d,void* sf,
        int m,int k,cudaStream_t st) {
    return launch_producer<FuseOp::SwiGLU>(g,u,d,sf,m,k,st);
}
bool launch_prefill_nvfp4_gate_quant_a(const void* x,const void* gate,void* d,void* sf,
        int m,int k,cudaStream_t st) {
    return launch_producer<FuseOp::SigmoidGate>(x,gate,d,sf,m,k,st);
}
bool launch_prefill_nvfp4_quant_b(const void* s, void* d, void* sf, int n, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(128,n,k)) return false;
    auto l = LayoutPlan::Scale::tile_atom_to_shape_SFB(LayoutPlan::problem(128,n,k));
    int blocks = (n * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,n,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gemm(const void* a,const void* sa,const void* b,const void* sb,
                               void* d,int m,int n,int k,void* ws,cudaStream_t st) {
    if (!a||!sa||!b||!sb||!d||!prefill_nvfp4_supported(m,n,k)) return false;
    if (split_n_tile(n) && SplitPlan::launch(a,sa,b,sb,d,m,n,k,ws,st)) return true;
    return FullPlan::launch(a,sa,b,sb,d,m,n,k,ws,st);
}
} // namespace sparkinfer::kernels
