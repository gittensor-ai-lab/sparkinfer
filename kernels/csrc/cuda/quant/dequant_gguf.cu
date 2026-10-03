// GGUF block dequantization (Q3_K..Q6_K, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, F16, BF16, F32) -> bf16, plus bf16
// transposes. The Q4_K/Q6_K decoders are validated byte-exact against the gguf
// python reference (.cudaverify/deqtest.cu). Used to load GGUF weights: dense
// tensors are dequantized once at load; expert stacks are kept quantized in VRAM
// and dequantized per-layer into a reused scratch buffer.
//
// Portable CUDA — runs on the architectures this tree builds: sm_89/90/100/120 (Ada, Hopper,
// datacenter Blackwell, consumer Blackwell: RTX 5090 / PRO 6000). sm_121 (RTX Spark, Jetson
// Thor) is NOT built — it needs CUDA 12.9+, see the exclusion note in CMakeLists.txt.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "sparkinfer/kernels/qtype.h"
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include "sparkinfer/kernels/dequant_gguf_fast.h"
#include "sparkinfer/kernels/dequant_rows_i8_fast.h"
#endif

namespace sparkinfer {
namespace kernels {

// ggml type ids
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13,
       GGML_Q6_K = 14, GGML_IQ4_NL = 20, GGML_IQ3_S = 21, GGML_IQ4_XS = 23, GGML_BF16 = 30 };

__device__ __forceinline__ float gg_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

__device__ __forceinline__ void gg_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}


// ---- Q3_K / IQ4_NL / IQ4_XS / IQ3_S: the other types llama.cpp's "UD" dynamic quants mix in.
// Block layouts and the two lookup tables follow ggml (ggml-common.h, ggml-quants.c
// dequantize_row_*; MIT, (c) the ggml authors). Load-time only: one thread per block.

__constant__ signed char gg_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                 1, 13, 25, 38, 53, 69, 89, 113};

__constant__ unsigned int gg_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// Q3_K: hmask[32] qs[64] scales[12] d -- 110 bytes per 256.
__global__ void deq_q3k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 110;
    const unsigned char* hm = blk;
    const unsigned char* q = blk + 32;
    const unsigned char* sc = blk + 96;
    const float d_all = gg_h2f(blk + 108);
    unsigned int aux[4];
    for (int i = 0; i < 3; ++i)
        aux[i] = (unsigned)sc[4 * i] | ((unsigned)sc[4 * i + 1] << 8) | ((unsigned)sc[4 * i + 2] << 16) |
                 ((unsigned)sc[4 * i + 3] << 24);
    const unsigned int kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const signed char* scales = reinterpret_cast<const signed char*>(aux);
    __nv_bfloat16* yy = y + b * 256;
    int is = 0;
    unsigned char m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *yy++ = __float2bfloat16(dl * ((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *yy++ = __float2bfloat16(dl * ((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// IQ4_NL: d qs[16] -- 18 bytes per 32.
__global__ void deq_iq4nl_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 18;
    const float d = gg_h2f(blk);
    const unsigned char* qs = blk + 2;
    __nv_bfloat16* yy = y + b * 32;
    for (int j = 0; j < 16; ++j) {
        yy[j] = __float2bfloat16(d * gg_kvalues_iq4nl[qs[j] & 0xf]);
        yy[j + 16] = __float2bfloat16(d * gg_kvalues_iq4nl[qs[j] >> 4]);
    }
}

// IQ4_XS: d scales_h scales_l[4] qs[128] -- 136 bytes per 256.
__global__ void deq_iq4xs_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 136;
    const float d = gg_h2f(blk);
    const unsigned int scales_h = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
    const unsigned char* scales_l = blk + 4;
    const unsigned char* qs = blk + 8;
    __nv_bfloat16* yy = y + b * 256;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            yy[j] = __float2bfloat16(dl * gg_kvalues_iq4nl[qs[j] & 0xf]);
            yy[j + 16] = __float2bfloat16(dl * gg_kvalues_iq4nl[qs[j] >> 4]);
        }
        yy += 32;
        qs += 16;
    }
}

// IQ3_S: d qs[64] qh[8] signs[32] scales[4] -- 110 bytes per 256.
__global__ void deq_iq3s_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 110;
    const float d = gg_h2f(blk);
    const unsigned char* qs = blk + 2;
    const unsigned char* qh = blk + 66;
    const unsigned char* signs = blk + 74;
    const unsigned char* scales = blk + 106;
    __nv_bfloat16* yy = y + b * 256;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
        for (int half = 0; half < 2; ++half) {
            const float db = half ? db2 : db1;
            const unsigned int h = qh[ib32 + half];   // two qh bytes per 64 values
            for (int l = 0; l < 4; ++l) {
                const unsigned int g1 = gg_iq3s_grid[qs[2 * l + 0] | ((h << (8 - 2 * l)) & 256)];
                const unsigned int g2 = gg_iq3s_grid[qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    const float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    const float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    yy[j + 0] = __float2bfloat16(db * v1 * ((signs[l] & (1u << j)) ? -1.f : 1.f));
                    yy[j + 4] = __float2bfloat16(db * v2 * ((signs[l] & (1u << (j + 4))) ? -1.f : 1.f));
                }
                yy += 8;
            }
            qs += 8;
            signs += 4;
        }
    }
}

// one thread per 256-value block
__global__ void deq_q4k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 144;
    float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* q = blk + 16;
    __nv_bfloat16* yy = y + b * 256; int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int s, m;
        gg_scale_min_k4(is,   sc, &s, &m); float d1 = d * s, m1 = dmin * m;
        gg_scale_min_k4(is+1, sc, &s, &m); float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; l++) yy[j + l]      = __float2bfloat16(d1 * (q[l] & 0xF) - m1);
        for (int l = 0; l < 32; l++) yy[j + 32 + l] = __float2bfloat16(d2 * (q[l] >> 4)  - m2);
        q += 32; is += 2;
    }
}

__device__ __forceinline__ float deq_q3a_val(const unsigned char* blk, int t) {
    const float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4;
    const unsigned char* qs = blk + 16;
    const unsigned char* qh = blk + 80;
    const int group = t >> 5, p = t & 31;
    const int j = group >> 1, m = p >> 3, r = p & 7;
    const int field = (group & 1) | ((r >> 2) << 1);
    const int lo = (qs[16 * j + 4 * m + (r & 3)] >> (2 * field)) & 3;
    const int hi = (qh[p] >> group) & 1;
    int s, mn;
    gg_scale_min_k4(group, sc, &s, &mn);
    return d * s * (lo | (hi << 2)) - dmin * mn;
}

__global__ void deq_q3a_kernel(const unsigned char* __restrict__ src,
                               __nv_bfloat16* __restrict__ y, long nblocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const unsigned char* blk = src + b * SI_QTYPE_Q3A;
    __nv_bfloat16* yy = y + b * 256;
    for (int t = 0; t < 256; ++t) yy[t] = __float2bfloat16(deq_q3a_val(blk, t));
}

// Q5_K: 176-byte super-block of 256 — d, dmin (fp16), 6-bit scales+mins (12B, like Q4_K),
// qh 1 high bit/quant (32B), qs 4 low bits/quant (128B). Byte-exact match to the ggml reference.
__global__ void deq_q5k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 176;
    float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4;    // scales + mins (6-bit packed)
    const unsigned char* qh = blk + 16;   // high bit per quant
    const unsigned char* ql = blk + 48;   // low 4 bits per quant
    __nv_bfloat16* yy = y + b * 256; int is = 0; unsigned char u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        int s, m;
        gg_scale_min_k4(is,   sc, &s, &m); float d1 = d * s, m1 = dmin * m;
        gg_scale_min_k4(is+1, sc, &s, &m); float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; l++) yy[j + l]      = __float2bfloat16(d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1);
        for (int l = 0; l < 32; l++) yy[j + 32 + l] = __float2bfloat16(d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2);
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

__global__ void deq_q6k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 210;
    const unsigned char* ql = blk; const unsigned char* qh = blk + 128;
    const signed char* sc = (const signed char*)(blk + 192); float d = gg_h2f(blk + 208);
    __nv_bfloat16* yy = y + b * 256;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            yy[l]    = __float2bfloat16(d * sc[is + 0] * q1);
            yy[l+32] = __float2bfloat16(d * sc[is + 2] * q2);
            yy[l+64] = __float2bfloat16(d * sc[is + 4] * q3);
            yy[l+96] = __float2bfloat16(d * sc[is + 6] * q4);
        }
        ql += 64; qh += 32; sc += 8; yy += 128;
    }
}

// ---------------------------------------------------------------------------
// Fused GGUF -> per-row-int8 dequant for the int8 prefill GEMM. Replaces the
// dequant-to-bf16 + pf_quantize_rows_i8 round trip (write 2B + read 2B per
// value) with one pass that decodes the superblocks twice (rows are L2-hot:
// a 12288-wide Q4_K row is ~7 KB) and writes 1B per value:
//   pass 1: block-wide amax of the exactly-dequantized row
//   pass 2: q = round(v / (amax/127)), matching pf_quantize_rows_i8
// One block (256 threads) per row; thread t decodes value t of each superblock.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float deq_q4k_val(const unsigned char* blk, int t) {
    const float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = qs[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    int s, m; gg_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * nib - dmin * m;
}
__device__ __forceinline__ float deq_q5k_val(const unsigned char* blk, int t) {
    const float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = ql[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    const int hbit = (qh[l] >> (2 * j64 + hi)) & 1;
    int s, m; gg_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * (nib + (hbit ? 16 : 0)) - dmin * m;
}
__device__ __forceinline__ float deq_q6k_val(const unsigned char* blk, int t) {
    const int half = t >> 7, r = t & 127, quad = r >> 5, l = r & 31;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const signed char* sc = (const signed char*)(blk + 192) + half * 8;
    const float d = gg_h2f(blk + 208);
    const int is = l / 16;
    int qv;
    if (quad == 0)      qv = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    else if (quad == 1) qv = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    else if (quad == 2) qv = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    else                qv = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * sc[is + 2 * quad] * qv;
}

// Single-pass Q→i8 for cols ≤ 2048 (Qwen3.6 MoE H/mffn). Values live in registers
// (templated NSB) so the int8 rows are bit-identical to the two-pass kernel — same
// dequant, same amax reduce, same roundf — while skipping the second decode pass.
// Wider rows (Qwythos attn K=4096, FFN K=12288) keep two-pass so we never reject the
// i8 path (the previous cols>4096 early-return regressed Qwythos @4k prefill ~10%).
static constexpr int kDeqRowsI8MaxNsb = 8;

template <int QT, int NSB>   // QT: 12=Q4_K, 13=Q5_K, 14=Q6_K, 112=Q3_A; NSB = cols/256
__global__ void deq_rows_i8_kernel(const unsigned char* __restrict__ src,
                                   signed char* __restrict__ q, float* __restrict__ scale,
                                   int cols) {
    constexpr int BS = (QT == 12) ? 144 : (QT == 13) ? 176 : (QT == 14) ? 210 : SI_QTYPE_Q3A;
    const int row = blockIdx.x, t = threadIdx.x;
    const unsigned char* rbase = src + (size_t)row * NSB * BS;

    float vals[NSB];
    float amax = 0.f;
    #pragma unroll
    for (int sb = 0; sb < NSB; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t)
                      : (QT == 14) ? deq_q6k_val(blk, t) : deq_q3a_val(blk, t);
        vals[sb] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float swarp[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float v = (t < 8) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (t == 0) swarp[0] = v;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) scale[row] = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    #pragma unroll
    for (int sb = 0; sb < NSB; sb++)
        qrow[sb * 256 + t] = (signed char)(int)roundf(vals[sb] * inv);
}

// Two-pass fallback for cols > 2048 (Qwythos projections). Same math as above.
template <int QT>
__global__ void deq_rows_i8_twopass_kernel(const unsigned char* __restrict__ src,
                                           signed char* __restrict__ q, float* __restrict__ scale,
                                           int cols) {
    constexpr int BS = (QT == 12) ? 144 : (QT == 13) ? 176 : (QT == 14) ? 210 : SI_QTYPE_Q3A;
    const int row = blockIdx.x, t = threadIdx.x;
    const int nsb = cols >> 8;
    const unsigned char* rbase = src + (size_t)row * nsb * BS;

    float amax = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t)
                      : (QT == 14) ? deq_q6k_val(blk, t) : deq_q3a_val(blk, t);
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float swarp[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float v = (t < 8) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (t == 0) swarp[0] = v;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) scale[row] = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t)
                      : (QT == 14) ? deq_q6k_val(blk, t) : deq_q3a_val(blk, t);
        qrow[sb * 256 + t] = (signed char)(int)roundf(v * inv);
    }
}

__global__ void deq_q8_0_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 34; float d = gg_h2f(blk);
    const signed char* q = (const signed char*)(blk + 2); __nv_bfloat16* yy = y + b * 32;
    for (int l = 0; l < 32; l++) yy[l] = __float2bfloat16(d * q[l]);
}

__global__ void deq_f16_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = __float2bfloat16(gg_h2f(src + i * 2));
}

// BF16 source, bf16 destination: the bits already line up, so this is a copy. Without it the
// dispatcher's final else would read the tensor as F32 -- twice the bytes, none of them right.
__global__ void deq_bf16_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    unsigned short bits;
    memcpy(&bits, src + i * 2, 2);
    memcpy(&y[i], &bits, 2);
}
__global__ void deq_f32_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = __float2bfloat16(src[i]);
}

__global__ void transpose2d_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int rows, int cols) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= (long)rows * cols) return;
    int r = idx / cols, c = idx % cols;
    dst[(long)c * rows + r] = src[idx];               // [rows,cols] -> [cols,rows]
}
__global__ void transpose3d_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int E, int A, int B) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= (long)E * A * B) return;
    int e = idx / ((long)A * B); int rem = idx % ((long)A * B); int a = rem / B, b = rem % B;
    dst[((long)e * B + b) * A + a] = src[idx];        // [E,A,B] -> [E,B,A]
}
// Interleave two per-head projection weights into the layout split_q_gate_kernel (qwen36.cu)
// expects: out[h][0:hd] = q, out[h][hd:2hd] = gate, per head h. q/gate are native ggml
// [out,in] storage (in_dim contiguous per output row), so this is a whole-row reorder, not a
// per-element transform -- combined row h*2*hd+d (d<hd) = q's row h*hd+d; combined row
// h*2*hd+hd+d = gate's row h*hd+d. Load-time only (once per model load, not perf-critical).
__global__ void interleave_qgate_rows_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out, int n_heads, int head_dim, int in_dim
) {
    const long total = (long)n_heads * 2 * head_dim * in_dim;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const long row = idx / in_dim;
        const long col = idx % in_dim;
        const int h = (int)(row / (2 * head_dim));
        const int r = (int)(row % (2 * head_dim));
        const bool is_gate = r >= head_dim;
        const long src_row = (long)h * head_dim + (is_gate ? r - head_dim : r);
        out[idx] = is_gate ? gate[src_row * in_dim + col] : q[src_row * in_dim + col];
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/quant.h"

void launch_gguf_dequant(int ggml_type, const void* src, void* dst_bf16, long n_values, cudaStream_t stream) {
    // Coalesced Q4_K path (warp per super-block, 16-byte stores; byte-exact). Falls through to the
    // scalar kernels below for other types or when SPARKINFER_DEQUANT_COALESCED=0.
    if (launch_gguf_dequant_fast(ggml_type, src, dst_bf16, n_values, stream)) return;
    auto* d = reinterpret_cast<__nv_bfloat16*>(dst_bf16);
    auto* s = reinterpret_cast<const unsigned char*>(src);
    const int T = 256;
    if (ggml_type == GGML_Q4_K) { long nb = n_values/256; deq_q4k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == SI_QTYPE_Q3A) { long nb = n_values/256; deq_q3a_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q5_K) { long nb = n_values/256; deq_q5k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q6_K) { long nb = n_values/256; deq_q6k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q8_0) { long nb = n_values/32;  deq_q8_0_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q3_K) { long nb = n_values/256; deq_q3k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_IQ4_NL) { long nb = n_values/32; deq_iq4nl_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_IQ4_XS) { long nb = n_values/256; deq_iq4xs_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_IQ3_S) { long nb = n_values/256; deq_iq3s_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_F16)  { deq_f16_kernel<<<(n_values+T-1)/T,T,0,stream>>>(s,d,n_values); }
    else if (ggml_type == GGML_BF16) { deq_bf16_kernel<<<(n_values+T-1)/T,T,0,stream>>>(s,d,n_values); }
    else /* F32 */                   { deq_f32_kernel<<<(n_values+T-1)/T,T,0,stream>>>(reinterpret_cast<const float*>(src),d,n_values); }
}

bool launch_gguf_dequant_rows_i8(int ggml_type, const void* src, signed char* q, float* scale,
                                 int rows, int cols, cudaStream_t stream) {
    // Vector-store path: 4 consecutive values per thread => 4-byte stores, 128 B per warp, and the
    // decoded row stays in registers. Bit-identical; =0 restores the byte-store kernel below.
    if (launch_gguf_dequant_rows_i8_fast(ggml_type, src, q, scale, rows, cols, stream)) return true;
    if ((cols & 255) != 0) return false;
    auto* s = reinterpret_cast<const unsigned char*>(src);
    const int nsb = cols >> 8;
    if (nsb > 0 && nsb <= kDeqRowsI8MaxNsb) {
        // Explicit NSB instantiations keep vals[] in registers (bit-same as two-pass).
        #define SI_LAUNCH(QT, NSB) deq_rows_i8_kernel<QT, NSB><<<rows, 256, 0, stream>>>(s, q, scale, cols)
        #define SI_BY_NSB(QT) \
            switch (nsb) { \
                case 1: SI_LAUNCH(QT, 1); break; \
                case 2: SI_LAUNCH(QT, 2); break; \
                case 3: SI_LAUNCH(QT, 3); break; \
                case 4: SI_LAUNCH(QT, 4); break; \
                case 5: SI_LAUNCH(QT, 5); break; \
                case 6: SI_LAUNCH(QT, 6); break; \
                case 7: SI_LAUNCH(QT, 7); break; \
                default: SI_LAUNCH(QT, 8); break; \
            }
        if (ggml_type == GGML_Q4_K)      { SI_BY_NSB(12); }
        else if (ggml_type == GGML_Q5_K) { SI_BY_NSB(13); }
        else if (ggml_type == GGML_Q6_K) { SI_BY_NSB(14); }
        else if (ggml_type == SI_QTYPE_Q3A) { SI_BY_NSB(SI_QTYPE_Q3A); }
        else return false;
        #undef SI_BY_NSB
        #undef SI_LAUNCH
    } else {
        // Qwythos attn/FFN: keep two-pass so i8 GEMM still engages (do not return false).
        if (ggml_type == GGML_Q4_K)      deq_rows_i8_twopass_kernel<12><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else if (ggml_type == GGML_Q5_K) deq_rows_i8_twopass_kernel<13><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else if (ggml_type == GGML_Q6_K) deq_rows_i8_twopass_kernel<14><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else if (ggml_type == SI_QTYPE_Q3A) deq_rows_i8_twopass_kernel<SI_QTYPE_Q3A><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else return false;
    }
    return true;
}

bool launch_gguf_dequant_rows_i8_pair(int ggml_type,
                                      const void* src0, signed char* q0, float* scale0,
                                      const void* src1, signed char* q1, float* scale1,
                                      int rows, int cols, cudaStream_t stream) {
    if (launch_gguf_dequant_rows_i8_fast_pair(ggml_type, src0, q0, scale0, src1, q1, scale1,
                                              rows, cols, stream))
        return true;
    // Fallback: two single launches (still correct).
    if (!launch_gguf_dequant_rows_i8(ggml_type, src0, q0, scale0, rows, cols, stream)) return false;
    if (!launch_gguf_dequant_rows_i8(ggml_type, src1, q1, scale1, rows, cols, stream)) return false;
    return true;
}

bool launch_gguf_dequant_rows_i8_gather(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream) {
    return launch_gguf_dequant_rows_i8_fast_gather(
        ggml_type, src0, q0, scale0, live_le, n_live, rows_per_expert, cols, expert_bytes, stream);
}

bool launch_gguf_dequant_rows_i8_gather_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    if (launch_gguf_dequant_rows_i8_fast_gather_pair(
            ggml_type, src0, q0, scale0, src1, q1, scale1, live_le, n_live, rows_per_expert, cols,
            expert_bytes0, expert_bytes1, stream))
        return true;
    // Fallback: two single gathers.
    if (!launch_gguf_dequant_rows_i8_gather(ggml_type, src0, q0, scale0, live_le, n_live,
                                            rows_per_expert, cols, expert_bytes0, stream))
        return false;
    if (!launch_gguf_dequant_rows_i8_gather(ggml_type, src1, q1, scale1, live_le, n_live,
                                            rows_per_expert, cols, expert_bytes1, stream))
        return false;
    return true;
}

bool launch_gguf_dequant_rows_i8_mask(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream) {
    return launch_gguf_dequant_rows_i8_fast_mask(
        ggml_type, src0, q0, scale0, counts, e_base, n_in, rows_per_expert, cols, expert_bytes,
        stream);
}

bool launch_gguf_dequant_rows_i8_mask_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    if (launch_gguf_dequant_rows_i8_fast_mask_pair(
            ggml_type, src0, q0, scale0, src1, q1, scale1, counts, e_base, n_in, rows_per_expert,
            cols, expert_bytes0, expert_bytes1, stream))
        return true;
    if (!launch_gguf_dequant_rows_i8_mask(ggml_type, src0, q0, scale0, counts, e_base, n_in,
                                          rows_per_expert, cols, expert_bytes0, stream))
        return false;
    return launch_gguf_dequant_rows_i8_mask(ggml_type, src1, q1, scale1, counts, e_base, n_in,
                                            rows_per_expert, cols, expert_bytes1, stream);
}

void launch_transpose_bf16(const void* src, void* dst, int rows, int cols, cudaStream_t stream) {
    long n = (long)rows*cols; const int T=256;
    transpose2d_kernel<<<(n+T-1)/T,T,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<__nv_bfloat16*>(dst), rows, cols);
}
void launch_transpose3d_bf16(const void* src, void* dst, int E, int A, int B, cudaStream_t stream) {
    long n = (long)E*A*B; const int T=256;
    transpose3d_kernel<<<(n+T-1)/T,T,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<__nv_bfloat16*>(dst), E, A, B);
}
void launch_interleave_qgate_rows(const void* q, const void* gate, void* out,
                                  int n_heads, int head_dim, int in_dim, cudaStream_t stream) {
    const long total = (long)n_heads * 2 * head_dim * in_dim;
    const int T = 256;
    long blocks = (total + T - 1) / T;
    if (blocks > 65535) blocks = 65535;   // kernel grid-strides, so a capped grid is fine
    interleave_qgate_rows_kernel<<<(unsigned)blocks, T, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(out), n_heads, head_dim, in_dim);
}
#endif

} // namespace kernels
} // namespace sparkinfer
