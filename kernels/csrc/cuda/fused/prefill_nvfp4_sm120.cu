#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"

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

// ---------------------------------------------------------------------------------------------
// L2 EVICTION POLICY FOR THE WEIGHT STREAM.
//
// At M=128 the grid is 1 x ceil(N/TileN), so every CTA re-reads the WHOLE A operand while the B
// weight slice it owns is read exactly once and never revisited. The L2 traffic that creates is
// larger than the DRAM traffic for three of the four shapes -- down: A is 1.44 MB re-read by 104
// CTAs = 150 MB of L2 against 74.8 MB of B from DRAM; qkvg 65 vs 32.6; wo 30.7 vs 15.3 -- so the
// read-once weight stream evicting the reused A tile is a real cost.
//
// `__ldcs` is the usual fix and was worth +1.4% on the older hand-written int8 GEMM, but CUTLASS
// loads A, B, SFA and SFB entirely through TMA and TMA ignores both `__ldcs` and a
// cudaAccessPolicyWindow (measured: the window moves this kernel ~0%). The mechanism that DOES
// reach a TMA load is the per-instruction .L2::cache_hint operand, which CUTLASS already plumbs
// through Copy_Traits::with(mbar, multicast_mask, cache_hint) as a DEFAULTED argument -- the
// sm120 mainloop simply never passes it (sm120_blockscaled_mma_tma.hpp:677-681 use the 1-arg form).
//
// Rather than vendor or patch CUTLASS, derive from its collective and hide `load()`. GemmUniversal
// is templated on the mainloop type and every nested type it needs (Params, SharedStorage,
// DispatchPolicy, TiledMma) is inherited, so only this one method changes. The body below is
// CUTLASS's verbatim apart from the two hints: B and SFB are marked EVICT_FIRST so the weight
// stream surrenders its lines first, and A/SFA are left EVICT_NORMAL -- marking A EVICT_FIRST
// instead measured 61% WORSE, which is the control that proves the hint is doing what it says.
// A cache hint cannot change any value, so this is bit-identical by construction.
// Marking the reused A tile EVICT_LAST ("keep me") on top of this measured a WASH (9048 vs 9041,
// splitting the pairs), so only the one-sided policy ships.
template <class Base>
struct MmaEvictFirstB : Base {
    using Base::Base;
    using typename Base::Params;
    using typename Base::MainloopPipeline;
    using typename Base::PipelineState;
    using typename Base::TensorStorage;

    template <class TensorA, class TensorB, class TensorSFA, class TensorSFB,
              class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void
    load(Params const& params, MainloopPipeline pipeline, PipelineState smem_pipe_write,
         cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB> const& load_inputs,
         BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count,
         int thread_idx, uint32_t block_rank_in_cluster, TensorStorage& shared_tensors) {
        using cute::_0;
        using cute::Int;
        constexpr auto EF = cute::TMA::CacheHintSm90::EVICT_FIRST;
        int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
            Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                                    typename Base::SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                                    typename Base::SmemLayoutB{});
            Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()),
                                      typename Base::SmemLayoutSFA{});
            Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()),
                                      typename Base::SmemLayoutSFB{});

            auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
            auto block_tma_a = params.tma_load_a.get_slice(0);
            auto block_tma_b = params.tma_load_b.get_slice(0);
            auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
            auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
            auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

            using TS = typename Base::TileShape;
            using TSFB = typename Base::TileShapeSFB;
            auto broadcast_n = make_layout(
                make_shape(Int<cute::size<1>(TSFB{}) / cute::size<1>(TS{})>{},
                           Int<cute::numeric_limits<int>::max()>{}),
                make_stride(_0{}, cute::size<1>(TSFB{}) / cute::size<1>(TS{})));
            Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);
            Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);
            Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);
            Tensor gSFB = gSFB_nkl(_,_,broadcast_n(n_coord),_,l_coord);

            Tensor tAgA = block_tma_a.partition_S(gA);
            Tensor tAsA = block_tma_a.partition_D(sA);
            Tensor tBgB = block_tma_b.partition_S(gB);
            Tensor tBsB = block_tma_b.partition_D(sB);
            Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);
            Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);
            Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
            Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);

            CUTLASS_PRAGMA_NO_UNROLL
            for ( ; k_tile_count > 0; --k_tile_count) {
                pipeline.producer_acquire(smem_pipe_write);
                using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
                int write_stage = smem_pipe_write.index();
                    copy(params.tma_load_a.with(*tma_barrier),
                         tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
                    copy(params.tma_load_sfa.with(*tma_barrier),
                         tAgSFA(_,_,_,*k_tile_iter), tAsSFA(_,_,_,write_stage));
                copy(params.tma_load_b.with(*tma_barrier, 0, EF),
                     tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
                copy(params.tma_load_sfb.with(*tma_barrier, 0, EF),
                     tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));
                ++k_tile_iter;
                ++smem_pipe_write;
            }
        }
        __syncwarp();
    }
};

template <class TileShape, bool EvictFirstB = false>
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
    using MainloopSel = cute::conditional_t<EvictFirstB, MmaEvictFirstB<Mainloop>, Mainloop>;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, MainloopSel, Epilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Two N-tilings of the same GEMM. The 128-wide tile reuses each A tile across twice as many
// output columns and is the right default; the 64-wide one exists only to fill the machine when
// the wide grid cannot (see nvfp4_prefer_narrow).
using Wide = Cfg<Shape<_128, _128, _256>>;
using Narrow = Cfg<Shape<_128, _64, _256>>;
// Same two tilings with the weight stream marked evict-first. Kept as separate instantiations so
// the policy can be A/B'd in ONE binary (SPARKINFER_NVFP4_EVICT_FIRST), which is the only way to
// measure it without a rebuild between arms.
using WideEF = Cfg<Shape<_128, _128, _256>, true>;
using NarrowEF = Cfg<Shape<_128, _64, _256>, true>;
// Both tiles above are M=128 because they were chosen for the scored ctx=128, where the grid is
// exactly one CTA tall and only N matters. Long-context prefill runs the SAME GEMMs at m=16384
// (FFN chunks and the GDN projections), where an M=128 tile reloads each B tile once per 128 rows
// -- 128 times over the operand. A taller tile amortises that. SPARKINFER_NVFP4_BIG_TILE picks:
//   1 = 256x128, 2 = 128x256, 3 = 256x256   (0 = off, keep the M=128 tiles)
// K tile halved to 128: at 256 the MN-larger tiles need more smem per stage than
// StageCountAutoCarveout can carve two stages out of, which CUTLASS rejects outright.
// Measured at ctx=16384 against 128x128x256: 256x128x128 +6.7%, 128x256x128 -14%, 256x256x64 -76%
// (K=64 leaves the mainloop too little per stage). M is the axis that pays. 384x128 / 512x128 /
// 256x64x256 do not compile -- CUTLASS cannot carve two mainloop stages out of their shared memory
// -- and non-power-of-two tiles fail cute's stride-divisibility and the epilogue's
// MMA_TILE_M | EPI_TILE_M check, so 256x128x128 is the reachable optimum.
using BigM = Cfg<Shape<_256, _128, _128>, true>;
// M is the axis that pays (256x128 beat 128x128 by 6.7% at m=16384 while 128x256 lost 14%), so
// probe further up it. K stays >=128: at 64 the mainloop has too few elements per stage to cover
// its own latency and the GEMM collapses (measured 2494 pp, a 4x loss).
// EVICT_FIRST on B stays correct at this tile too: dropping it measured 11310 vs 11409 pp.
// Non-power-of-two tiles are not reachable: K=192 fails cute's stride-divisibility check and
// M=192 fails the epilogue's "MMA_TILE_M must divide EPI_TILE_M", so 256x128x128 stands.

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
                                 void* d, int m, int n, int k, float alpha = 1.f,
                                 const void* c = nullptr) {
    auto as = cutlass::make_cute_packed_stride(StrideA{}, {m,k,1});
    auto bs = cutlass::make_cute_packed_stride(StrideB{}, {n,k,1});
    auto cs = cutlass::make_cute_packed_stride(StrideC{}, {m,n,1});
    auto ds = cutlass::make_cute_packed_stride(StrideD{}, {m,n,1});
    // beta is 1 exactly when a source is supplied — the residual accumulate. With c == nullptr
    // beta MUST stay 0: the epilogue skips the C load entirely on beta == 0, and a non-zero beta
    // over a null pointer faults.
    const float beta = c ? 1.f : 0.f;
    return {cutlass::gemm::GemmUniversalMode::kGemm, shape(m,n,k),
            {static_cast<const cutlass::float_e2m1_t*>(a), as, static_cast<const cutlass::float_e2m1_t*>(b), bs, static_cast<const cutlass::float_ue4m3_t*>(sa), sfa_layout(m,n,k), static_cast<const cutlass::float_ue4m3_t*>(sb), sfb_layout(m,n,k)},
            {{alpha, beta}, static_cast<const BF*>(c), cs, static_cast<BF*>(d), ds}};
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
              void* d, int m, int n, int k, void* ws, cudaStream_t st, float alpha,
              const void* c) {
    typename C::Gemm gemm;
    auto ar = args<C>(a, sa, b, sb, d, m, n, k, alpha, c);
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
int nvfp4_big_tile() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_NVFP4_BIG_TILE");
        const int x = e ? atoi(e) : 1;
        return (x == 0 || x == 1) ? x : 1;
    }();
    return v;
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
                               void* d,int m,int n,int k,void* ws,cudaStream_t st, float alpha,
                               const void* c) {
    if (!a||!sa||!b||!sb||!d||!prefill_nvfp4_supported(m,n,k)) return false;
    // Default ON; SPARKINFER_NVFP4_EVICT_FIRST=0 restores CUTLASS's stock loads for A/B.
    static const bool ef = [] {
        const char* e = getenv("SPARKINFER_NVFP4_EVICT_FIRST");
        return !e || atoi(e) != 0;
    }();
    // Long-context: a taller/wider tile than the m=128-tuned pair above. Gated on a many-CTA-tall
    // grid so the scored ctx=128 shape is untouched, and it falls through if CUTLASS cannot
    // implement the shape.
    const int big = nvfp4_big_tile();
    if (big && m >= 512) {
        if (run_gemm<BigM>(a,sa,b,sb,d,m,n,k,ws,st,alpha,c)) return true;

    }
    if (ef)
        return prefer_narrow(m,n) ? run_gemm<NarrowEF>(a,sa,b,sb,d,m,n,k,ws,st,alpha,c)
                                  : run_gemm<WideEF>(a,sa,b,sb,d,m,n,k,ws,st,alpha,c);
    return prefer_narrow(m,n) ? run_gemm<Narrow>(a,sa,b,sb,d,m,n,k,ws,st,alpha,c)
                              : run_gemm<Wide>(a,sa,b,sb,d,m,n,k,ws,st,alpha,c);
}

// ---- HuggingFace "compressed-tensors" NVFP4 checkpoint dequant (load-time, one-shot) ----
// See sparkinfer/kernels/compressed_tensors.h -- converts a checkpoint's own NVFP4 encoding
// (block_size=16 UE4M3 group scale + one F32 global scale) straight to bf16, so the result feeds
// this runtime's OWN from-bf16 quantizers (quant_rows above, or the Q4_K requant path) rather than
// needing the GEMM kernels above to understand a second, foreign two-level scale scheme.
namespace {
// global_scale_dev, when non-null, supplies the global scale from device memory instead of the
// host-side scalar. Callers that hold the checkpoint payload (whose header carries the scale on
// the device) need this: reading it back to the host would mean a D2H copy, and these launches
// happen inside batched prefill, which runs under CUDA graph capture -- a synchronizing copy
// there is not merely slow, it is illegal and aborts the capture.
__global__ void ct_dequant_nvfp4_kernel(const unsigned char* __restrict__ packed,
                                        const unsigned char* __restrict__ group_scale,
                                        float global_scale,
                                        const float* __restrict__ global_scale_dev,
                                        __nv_bfloat16* __restrict__ out,
                                        int rows, int cols) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long n = (long)rows * cols;
    if (i >= n) return;
    if (global_scale_dev) global_scale = *global_scale_dev;
    const int r = (int)(i / cols);
    const int c = (int)(i - (long)r * cols);
    const unsigned char byte = packed[(size_t)r * (cols / 2) + (c >> 1)];
    const unsigned char nibble = (c & 1) ? (byte >> 4) : (byte & 0xF);
    const unsigned char sbyte = group_scale[(size_t)r * (cols / 16) + (c >> 4)];
    const float v = float(cutlass::float_e2m1_t::bitcast(nibble));
    const float s = float(cutlass::float_ue4m3_t::bitcast(sbyte));
    // weight_global_scale is stored as the quantization-time multiplier that mapped local block
    // scales (local_amax/6) UP into UE4M3's representable range before rounding to 8 bits, so
    // dequant must invert it: block_scale_fp32 = e4m3(block_scale) / global_scale. Confirmed
    // empirically against the real unsloth/Qwen3.8-27B-NVFP4 checkpoint -- multiplying (the other
    // natural reading) produces weight magnitudes in the hundreds of thousands; dividing produces
    // the expected ~0.01-0.05 std typical of a trained projection matrix.
    out[i] = __float2bfloat16(v * s / global_scale);
}

// Group-wise twin of the kernel above. Same arithmetic, same operand order, same
// __float2bfloat16 -- only the work decomposition changes, so the output is bit-identical.
//
// The per-element kernel is the dominant cost of a ModelOpt prefill: nsys puts it at 33.2 ms of a
// 52.9 ms prefill@128 (63%), moving 138 MB per GDN qkv call at ~730 GB/s while the GEMVs beside it
// run at ~1500. Per thread it paid a 64-bit `i / cols` division, re-read each packed byte from two
// threads and each scale byte from sixteen, and issued a 2-byte store.
//
// One thread owns one 16-element group instead -- exactly the granularity the scale is stored at:
//   * the row index comes from blockIdx.y, so the division disappears;
//   * the group's 8 packed bytes are one 8-byte load (the row pitch is cols/2 and cols%16==0, so
//     every group is 8-byte aligned), and its scale is one byte read once, not sixteen times;
//   * the 16 results leave as two 16-byte stores.
// A warp then reads 256 contiguous packed bytes and writes 1 KB, all coalesced.
// ALU MODES. ncu says this kernel is NOT bandwidth bound -- DRAM 43.7% against SM 80.9% at 78%
// warps active, with zero local memory. It is the per-element decode that costs, so the levers are
// the two CUTLASS conversions and the fp32 divide, not the byte count. (Fusing this kernel into the
// row-quantize that follows it cuts DRAM traffic 4.4x and measures 1.2% SLOWER, which is the
// control that proves the point -- do not re-try it.)
//
//   0  as merged: cutlass conversions, (v * s) / global_scale
//   1  the integer decodes, same divide -- bit-identical
//   2  mode 1 plus a hoisted reciprocal, which is NOT bit-identical (measured, see below)
enum : int { CT_ALU_BASE = 0, CT_ALU_INT = 1, CT_ALU_RCP = 2 };

// e2m1 magnitudes doubled are {0,1,2,3,4,6,8,12} -- all < 16, so the table is the nibbles of one
// 32-bit literal and the decode is a shift, a mask and an int->float, with no memory touched.
// Halving the result is exact in binary floating point, so 0.5f * this is exactly the value
// cutlass::float_e2m1_t::bitcast produces, sign and -0.0 included.
__device__ __forceinline__ float ct_e2m1_x2(unsigned n) {
    const unsigned mag = (0xC8643210u >> ((n & 7u) << 2)) & 15u;
    return __int_as_float(__float_as_int(__uint2float_rn(mag)) | ((n & 8u) << 28));
}
// Unsigned E4M3 -> float by assembling the bits. For e>0, (8+m) * 2^(e-10) is exactly the fp32
// with exponent field e+120 and mantissa m<<20; e==0 is the subnormal leg, an exact multiply.
__device__ __forceinline__ float ct_ue4m3(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) return (float)m * (1.f / 512.f);
    return __int_as_float((int)(((e + 120u) << 23) | (m << 20)));
}

template <int ALU>
__global__ void ct_dequant_nvfp4_g16_kernel(const unsigned char* __restrict__ packed,
                                            const unsigned char* __restrict__ group_scale,
                                            float global_scale,
                                            const float* __restrict__ global_scale_dev,
                                            __nv_bfloat16* __restrict__ out,
                                            int rows, int cols) {
    if (global_scale_dev) global_scale = *global_scale_dev;
    const float inv_gs = (ALU >= CT_ALU_RCP) ? (1.f / global_scale) : 0.f;
    const int ngroups = cols >> 4;
    const int r = blockIdx.y;
    if (r >= rows) return;
    const unsigned char* prow = packed + (size_t)r * (size_t)(cols >> 1);
    const unsigned char* srow = group_scale + (size_t)r * (size_t)ngroups;
    __nv_bfloat16* orow = out + (size_t)r * (size_t)cols;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < ngroups;
         g += gridDim.x * blockDim.x) {
        const uint2 pk = *reinterpret_cast<const uint2*>(prow + (size_t)g * 8);
        const float s = (ALU >= CT_ALU_INT) ? ct_ue4m3(srow[g])
                                            : float(cutlass::float_ue4m3_t::bitcast(srow[g]));
        // __align__(16) so the two uint4 stores below are legal; indices are compile-time constant
        // under the unroll, so this stays in registers (verified: ncu local ld/st = 0).
        __align__(16) __nv_bfloat16 o[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            const unsigned word = (j < 8) ? pk.x : pk.y;
            const unsigned byte = (word >> (8 * ((j & 7) >> 1))) & 255u;
            const unsigned char nib = (unsigned char)((j & 1) ? (byte >> 4) : (byte & 0xF));
            const float v = (ALU >= CT_ALU_INT) ? (0.5f * ct_e2m1_x2(nib))
                                                : float(cutlass::float_e2m1_t::bitcast(nib));
            o[j] = __float2bfloat16((ALU >= CT_ALU_RCP) ? (v * s * inv_gs) : (v * s / global_scale));
        }
        *reinterpret_cast<uint4*>(orow + (size_t)g * 16)     = *reinterpret_cast<const uint4*>(o);
        *reinterpret_cast<uint4*>(orow + (size_t)g * 16 + 8) = *reinterpret_cast<const uint4*>(o + 8);
    }
}

// Fused NVFP4 -> per-row int8, so the bf16 never lands in DRAM.
//
// This was tried BEFORE the integer decode above and measured 1.2% SLOWER: it cut DRAM traffic
// 4.4x (295 MB -> 85 MB per GDN qkv projection) and gained nothing, because at that point both
// this kernel and the dequant it replaces were ALU bound -- ncu had them at SM 75-81% with DRAM
// at 10-44%. Halving the per-element arithmetic changes that premise, so the traffic saving is
// worth re-testing on top of it. SPARKINFER_CT_NVFP4_ROWS_I8 selects.
//
// The value set, the reduction and the rounding are pf_quant_rows_fast_kernel's, unchanged: the
// same VEC=8 slots so a thread holds the same eight columns, the same bf16 rounding on the way in,
// the same amax over that value set, the same d = amax/127.0f and roundf, and the same
// amax==0 -> 0 rule. The row amax is a MAX, which is associative and exact in floating point, so
// no thread mapping can change it. Bit-identical to the two-kernel path by construction.
//
// Eight consecutive columns from a multiple of 8 lie inside one 16-element group, so a slot is one
// 4-byte packed load plus one scale byte.
template <int BLOCK, int VEC, int SLOTS>
__global__ __launch_bounds__(BLOCK) void ct_nvfp4_rows_i8_kernel(
        const unsigned char* __restrict__ packed, const unsigned char* __restrict__ group_scale,
        const float* __restrict__ global_scale_dev, signed char* __restrict__ q,
        float* __restrict__ scale, int rows, int cols) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    const float gs = *global_scale_dev;
    const unsigned char* prow = packed + (size_t)r * (size_t)(cols >> 1);
    const unsigned char* srow = group_scale + (size_t)r * (size_t)(cols >> 4);
    const size_t base = (size_t)r * (size_t)cols;

    __nv_bfloat16 reg[SLOTS][VEC];
    float amax = 0.f;
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            const unsigned pk = *reinterpret_cast<const unsigned*>(prow + (c >> 1));
            const float gsc = ct_ue4m3(srow[c >> 4]);
            #pragma unroll
            for (int v = 0; v < VEC; v++) {
                const unsigned byte = (pk >> (8 * (v >> 1))) & 255u;
                const unsigned char nib = (unsigned char)((v & 1) ? (byte >> 4) : (byte & 0xF));
                reg[s][v] = __float2bfloat16((0.5f * ct_e2m1_x2(nib)) * gsc / gs);
                amax = fmaxf(amax, fabsf(__bfloat162float(reg[s][v])));
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float sred[BLOCK / 32];
    if ((tid & 31) == 0) sred[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < BLOCK / 32) ? sred[tid] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (tid == 0) sred[0] = v;
    }
    __syncthreads();
    const float d = sred[0] / 127.0f;
    if (tid == 0) scale[r] = d;
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            signed char out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; v++)
                out[v] = (signed char)((sred[0] == 0.f) ? 0
                                       : (int)roundf(__bfloat162float(reg[s][v]) / d));
            *reinterpret_cast<uint2*>(&q[base + c]) = *reinterpret_cast<const uint2*>(out);
        }
    }
}
} // namespace

bool launch_ct_dequant_nvfp4_rows_i8(const void* packed_u8, const void* group_scale_ue4m3,
                                     const float* global_scale_dev, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream) {
    static const int on = [] {
        const char* e = getenv("SPARKINFER_CT_NVFP4_ROWS_I8");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    constexpr int BLOCK = 256, VEC = 8;
    if (!on || !packed_u8 || !group_scale_ue4m3 || !global_scale_dev || !q || !scale) return false;
    if (rows <= 0 || cols <= 0 || (cols & 15) != 0) return false;
    const int slots = (cols + BLOCK * VEC - 1) / (BLOCK * VEC);
    #define SI_NVFP4_ROWS_I8(S) \
        ct_nvfp4_rows_i8_kernel<BLOCK, VEC, S><<<rows, BLOCK, 0, stream>>>( \
            reinterpret_cast<const unsigned char*>(packed_u8), \
            reinterpret_cast<const unsigned char*>(group_scale_ue4m3), \
            global_scale_dev, q, scale, rows, cols)
    switch (slots) {
        case 1: SI_NVFP4_ROWS_I8(1); break;
        case 2: SI_NVFP4_ROWS_I8(2); break;
        case 3: SI_NVFP4_ROWS_I8(3); break;
        case 4: SI_NVFP4_ROWS_I8(4); break;
        default: return false;
    }
    #undef SI_NVFP4_ROWS_I8
    return cudaPeekAtLastError() == cudaSuccess;
}

static void ct_dequant_nvfp4_launch(const void* packed_u8, const void* group_scale_ue4m3,
                                    float global_scale, const float* global_scale_dev,
                                    void* out_bf16, int rows, int cols, cudaStream_t stream) {
    // Group-wise path when the shape allows it (SPARKINFER_CT_NVFP4_G16=0 restores the
    // per-element kernel for A/B). cols%16==0 makes every group's 8 packed bytes 8-byte aligned
    // and every 16-value output run 32-byte aligned; the payload pointers themselves come from
    // cudaMalloc plus a 256-byte header, so both bases are aligned too.
    static const int g16 = [] {
        const char* e = getenv("SPARKINFER_CT_NVFP4_G16");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Which ALU form the group-wise kernel uses; see the enum above. Default is the bit-identical
    // integer decode. SPARKINFER_CT_NVFP4_ALU picks 0/1/2 for A/B.
    static const int alu = [] {
        const char* e = getenv("SPARKINFER_CT_NVFP4_ALU");
        const int v = e ? atoi(e) : CT_ALU_INT;
        return (v >= CT_ALU_BASE && v <= CT_ALU_RCP) ? v : CT_ALU_INT;
    }();
    if (g16 && rows > 0 && cols > 0 && (cols & 15) == 0 &&
        ((reinterpret_cast<size_t>(packed_u8) | reinterpret_cast<size_t>(out_bf16)) & 15u) == 0) {
        const int ngroups = cols >> 4;
        const int threads = 256;
        const int bx = (ngroups + threads - 1) / threads;
        const dim3 grid(bx > 0 ? bx : 1, rows);
        auto* pk = reinterpret_cast<const unsigned char*>(packed_u8);
        auto* gsc = reinterpret_cast<const unsigned char*>(group_scale_ue4m3);
        auto* ob = reinterpret_cast<__nv_bfloat16*>(out_bf16);
        if (alu == CT_ALU_RCP)
            ct_dequant_nvfp4_g16_kernel<CT_ALU_RCP><<<grid, threads, 0, stream>>>(
                pk, gsc, global_scale, global_scale_dev, ob, rows, cols);
        else if (alu == CT_ALU_INT)
            ct_dequant_nvfp4_g16_kernel<CT_ALU_INT><<<grid, threads, 0, stream>>>(
                pk, gsc, global_scale, global_scale_dev, ob, rows, cols);
        else
            ct_dequant_nvfp4_g16_kernel<CT_ALU_BASE><<<grid, threads, 0, stream>>>(
                pk, gsc, global_scale, global_scale_dev, ob, rows, cols);
        return;
    }
    const long n = (long)rows * cols;
    const int threads = 256;
    const long blocks = (n + threads - 1) / threads;
    ct_dequant_nvfp4_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(packed_u8),
        reinterpret_cast<const unsigned char*>(group_scale_ue4m3), global_scale, global_scale_dev,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), rows, cols);
}

void launch_ct_dequant_nvfp4(const void* packed_u8, const void* group_scale_ue4m3,
                             float global_scale, void* out_bf16, int rows, int cols,
                             cudaStream_t stream) {
    ct_dequant_nvfp4_launch(packed_u8, group_scale_ue4m3, global_scale, nullptr, out_bf16,
                            rows, cols, stream);
}

void launch_ct_dequant_nvfp4_dev(const void* packed_u8, const void* group_scale_ue4m3,
                                 const float* global_scale_dev, void* out_bf16, int rows, int cols,
                                 cudaStream_t stream) {
    ct_dequant_nvfp4_launch(packed_u8, group_scale_ue4m3, 1.f, global_scale_dev, out_bf16,
                            rows, cols, stream);
}

template <class Layout>
__global__ void pack_sfb_rows(const unsigned char* __restrict__ src,
                              cutlass::float_ue4m3_t* sf, int rows, int cols,
                              Layout layout) {
    const int vpr = cols / 16;
    const int total = rows * vpr;
    for (int g = (int)(blockIdx.x * blockDim.x + threadIdx.x); g < total;
         g += (int)(gridDim.x * blockDim.x)) {
        const int row = g / vpr, k0 = (g - row * vpr) * 16;
        auto scales = cute::make_tensor(sf, layout);
        scales(row, k0, 0) = cutlass::float_ue4m3_t::bitcast(src[g]);
    }
}

bool launch_ct_nvfp4_pack_sfb(const void* scale_rowmajor, void* sfb,
                              int n, int k, cudaStream_t st) {
    if (!scale_rowmajor || !sfb || !prefill_nvfp4_supported(128, n, k)) return false;
    const size_t bytes = prefill_nvfp4_scale_bytes_b(n, k);
    if (bytes && cudaMemsetAsync(sfb, 0, bytes, st) != cudaSuccess) return false;
    auto l = sfb_layout(128, n, k);
    int blocks = (n * (k / 16) + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    pack_sfb_rows<<<blocks, 256, 0, st>>>(
        reinterpret_cast<const unsigned char*>(scale_rowmajor),
        reinterpret_cast<cutlass::float_ue4m3_t*>(sfb), n, k, l);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace sparkinfer::kernels
