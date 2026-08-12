#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {

// Decode is DRAM-bound (~86% bus utilization measured on a 5090), and the missing ~14% is time
// the bus sits idle while a latency-bound kernel runs. The worst offender is the Muse Glimmer
// sandwich-norm tail: two single-CTA reductions per layer, ~3.4 us each, during which 169 of 170
// SMs and the entire memory bus do nothing. This kernel fills that window by pulling the leading
// slice of the weight matrix the NEXT big GEMV will stream into L2, so those bytes are already
// resident when it starts. Pure `prefetch.global.L2` -- no data reaches registers, nothing is
// written, so it cannot perturb any result. The arithmetic downstream is bit-identical; only
// where a byte is served from changes.
__global__ void si_l2_prefetch_kernel(const char* __restrict__ p, size_t bytes) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) << 7;   // one 128 B line per thread
    const size_t stride = (size_t)gridDim.x * blockDim.x << 7;
    for (; i < bytes; i += stride)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
}

// `prefetch.global.L2` is only a hint and the hardware may drop it under memory pressure -- which
// is exactly the regime this runs in. This variant issues real 16 B loads instead, so the fill is
// guaranteed; the accumulator is sunk behind a condition that never holds, which keeps the loads
// from being dead-code-eliminated without ever storing anything.
__device__ unsigned si_l2_pf_sink;
__global__ void si_l2_load_kernel(const uint4* __restrict__ p, size_t n16) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (; i < n16; i += stride) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xFFFFFFFFu && n16 == 0) si_l2_pf_sink = acc;   // never taken
}

void launch_l2_prefetch(const void* p, size_t bytes, cudaStream_t stream) {
    if (!p || !bytes) return;
    static int mode = -1;
    if (mode < 0) { const char* e = getenv("SPARKINFER_MG_L2PF_MODE"); mode = e ? atoi(e) : 0; }
    if (mode == 1) {
        const size_t n16 = bytes >> 4;
        const int threads = 256;
        const int blocks = (int)((n16 + threads - 1) / threads < 512 ? (n16 + threads - 1) / threads : 512);
        si_l2_load_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(p), n16);
        return;
    }
    const size_t lines = (bytes + 127) >> 7;
    const int threads = 256;
    // Cap the grid so the prefetch never crowds out the latency-bound kernel it overlaps: 512 CTAs
    // is already ~3x what it takes to saturate the bus, and leaves the single-CTA tail its SM.
    const int blocks = (int)((lines + threads - 1) / threads < 512 ? (lines + threads - 1) / threads : 512);
    si_l2_prefetch_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(p), bytes);
}


} // namespace kernels
} // namespace sparkinfer
