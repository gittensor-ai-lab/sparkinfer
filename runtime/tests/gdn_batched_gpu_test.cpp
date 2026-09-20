// Packed-decode GDN kernels: B independent sequences in one launch must equal B sequential
// single-sequence launches, BIT FOR BIT.
//
// Bit-exactness is the bar, not closeness. The recurrent state is carried across every decode
// step, so a last-bit difference compounds; and packed decode has to reproduce what the
// unbatched path would have emitted or it changes what users are served.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::vector<uint16_t> rand_bf16(size_t n, uint32_t seed, float scale) {
    std::vector<uint16_t> h(n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        const float f = scale * (2.f * ((x >> 8) % 10000) / 10000.f - 1.f);
        uint32_t b; memcpy(&b, &f, 4);
        h[i] = (uint16_t)(b >> 16);
    }
    return h;
}

std::vector<float> rand_f32(size_t n, uint32_t seed, float scale) {
    std::vector<float> h(n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        h[i] = scale * (2.f * ((x >> 8) % 10000) / 10000.f - 1.f);
    }
    return h;
}

template <class T>
T* upload(const std::vector<T>& h) {
    void* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return (T*)d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// `compact` runs the pair on the COMPACTED bf16 state the continuous-batch path converts a
// session to. It is a separate case rather than a variation because the slot offset changes
// units with it: the compacted state is bf16 packed from the allocation base, so `off` counts
// bf16 elements there and floats otherwise. Both launchers therefore have to be handed the base
// pointer and the offset separately -- and until they were, the unbatched one doubled every
// non-zero offset and this pair still passed, because it only ever ran fp32.
bool test_gdn_ar(int B, int q_heads, int v_heads, int hd, bool qh_block, size_t off = 0,
                 bool compact = false) {
    const size_t qdim = (size_t)q_heads * hd, vdim = (size_t)v_heads * hd;
    const size_t st_n = (size_t)v_heads * hd * hd;

    auto hq = rand_bf16(B * qdim, 11, 1.f), hk = rand_bf16(B * qdim, 22, 1.f);
    auto hv = rand_bf16(B * vdim, 33, 1.f);
    auto ha = rand_bf16(B * v_heads, 44, 1.f), hb = rand_bf16(B * v_heads, 55, 1.f);
    auto hdt = rand_bf16(v_heads, 66, 0.5f), haa = rand_bf16(v_heads, 77, -0.5f);
    // off models a per-layer slice of a session's state. Under `compact` the same allocation
    // holds bf16 values from its base, so the seed is bf16 bytes laid into a float-sized buffer.
    std::vector<float> hst;
    if (compact) {
        hst.assign(st_n + off, 0.f);
        const auto seed = rand_bf16(st_n + off, 88, 0.1f);
        memcpy(hst.data(), seed.data(), seed.size() * sizeof(uint16_t));
    } else {
        hst = rand_f32(st_n + off, 88, 0.1f);
    }

    auto* dq = upload(hq); auto* dk = upload(hk); auto* dv = upload(hv);
    auto* da = upload(ha); auto* db = upload(hb);
    auto* ddt = upload(hdt); auto* daa = upload(haa);
    if (!dq || !dk || !dv || !da || !db || !ddt || !daa) { printf("FAIL: alloc\n"); return false; }

    // reference: B sequential unbatched calls, each on its own state
    std::vector<float*> ref_states(B);
    std::vector<uint16_t> ref_out(B * vdim);
    for (int b = 0; b < B; b++) {
        ref_states[b] = upload(hst);                       // every row starts from the SAME state
        void* out = nullptr; cudaMalloc(&out, vdim * sizeof(uint16_t));
        sparkinfer::kernels::launch_qwen36_gdn_ar(
            (const char*)dq + b * qdim * 2, (const char*)dk + b * qdim * 2,
            (const char*)dv + b * vdim * 2,
            (const char*)da + b * v_heads * 2, (const char*)db + b * v_heads * 2,
            ddt, daa, ref_states[b], off, out, q_heads, v_heads, hd, qh_block, nullptr, compact);
        cudaDeviceSynchronize();
        auto o = download((uint16_t*)out, vdim);
        memcpy(&ref_out[(size_t)b * vdim], o.data(), vdim * sizeof(uint16_t));
        cudaFree(out);
    }

    // batched: one launch, per-row state pointer array
    std::vector<float*> bat_states(B);
    for (int b = 0; b < B; b++) bat_states[b] = upload(hst);
    auto* dstates = upload(bat_states);
    void* bout = nullptr; cudaMalloc(&bout, B * vdim * sizeof(uint16_t));
    if (!sparkinfer::kernels::launch_qwen36_gdn_ar_batched(
            dq, dk, dv, da, db, ddt, daa, dstates, off, bout,
            B, q_heads, v_heads, hd, qh_block, nullptr, compact)) {
        printf("FAIL: batched launcher declined hd=%d\n", hd);
        return false;
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("FAIL: cuda error\n"); return false; }

    auto got = download((uint16_t*)bout, B * vdim);
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != ref_out[i]) {
            printf("FAIL: gdn_ar out mismatch at %zu (row %zu): batched=%u ref=%u\n",
                   i, i / vdim, got[i], ref_out[i]);
            return false;
        }
    }
    // Compare the slot the launch was told to advance, in ITS units. Comparing the fp32 view of a
    // compacted state would read the wrong half of the buffer and pass on two identical wrongs.
    for (int b = 0; b < B; b++) {
        auto rs = download(ref_states[b], st_n + off), bs = download(bat_states[b], st_n + off);
        const size_t elem = compact ? sizeof(uint16_t) : sizeof(float);
        const char* rp = reinterpret_cast<const char*>(rs.data()) + off * elem;
        const char* bp = reinterpret_cast<const char*>(bs.data()) + off * elem;
        for (size_t i = 0; i < st_n; i++) {
            if (memcmp(rp + i * elem, bp + i * elem, elem) != 0) {
                printf("FAIL: gdn_ar state row %d mismatch at %zu (compact=%d)\n", b, i, (int)compact);
                return false;
            }
        }
    }
    printf("[ok] gdn_ar B=%d qh=%d vh=%d hd=%d qh_block=%d off=%zu compact=%d  out+state bit-identical\n",
           B, q_heads, v_heads, hd, (int)qh_block, off, (int)compact);
    return true;
}

bool test_conv(int B, int q_heads, int v_heads, int hd, int ck, size_t coff = 0) {
    const size_t qdim = (size_t)q_heads * hd, vdim = (size_t)v_heads * hd;
    const size_t qkv = 2 * qdim + vdim;
    const size_t cs_n = (size_t)(ck - 1) * qkv;

    auto hqkv = rand_bf16(B * qkv, 101, 1.f);
    auto hw = rand_bf16(qkv * ck, 202, 0.5f);
    auto hcs = rand_bf16(cs_n + coff, 303, 0.3f);
    auto* dqkv = upload(hqkv); auto* dw = upload(hw);

    std::vector<uint16_t*> ref_cs(B);
    std::vector<uint16_t> ref_q(B * qdim), ref_k(B * qdim), ref_v(B * vdim);
    for (int b = 0; b < B; b++) {
        ref_cs[b] = upload(hcs);
        void *oq = nullptr, *ok = nullptr, *ov = nullptr;
        cudaMalloc(&oq, qdim * 2); cudaMalloc(&ok, qdim * 2); cudaMalloc(&ov, vdim * 2);
        sparkinfer::kernels::launch_qwen36_conv_split_l2norm_fused(
            (const char*)dqkv + b * qkv * 2, dw, ref_cs[b] + coff, oq, ok, ov,
            q_heads, v_heads, hd, ck, 1e-6f, nullptr);
        cudaDeviceSynchronize();
        auto a = download((uint16_t*)oq, qdim), c = download((uint16_t*)ok, qdim),
             e = download((uint16_t*)ov, vdim);
        memcpy(&ref_q[(size_t)b * qdim], a.data(), qdim * 2);
        memcpy(&ref_k[(size_t)b * qdim], c.data(), qdim * 2);
        memcpy(&ref_v[(size_t)b * vdim], e.data(), vdim * 2);
        cudaFree(oq); cudaFree(ok); cudaFree(ov);
    }

    std::vector<uint16_t*> bat_cs(B);
    for (int b = 0; b < B; b++) bat_cs[b] = upload(hcs);
    auto* dcs = upload(bat_cs);
    void *bq = nullptr, *bk = nullptr, *bv = nullptr;
    cudaMalloc(&bq, B * qdim * 2); cudaMalloc(&bk, B * qdim * 2); cudaMalloc(&bv, B * vdim * 2);
    sparkinfer::kernels::launch_qwen36_conv_split_l2norm_fused_batched(
        dqkv, dw, (void* const*)dcs, coff, bq, bk, bv, B, q_heads, v_heads, hd, ck, 1e-6f, nullptr);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { printf("FAIL: conv cuda error\n"); return false; }

    auto gq = download((uint16_t*)bq, B * qdim), gk = download((uint16_t*)bk, B * qdim),
         gv = download((uint16_t*)bv, B * vdim);
    for (size_t i = 0; i < gq.size(); i++)
        if (gq[i] != ref_q[i] || gk[i] != ref_k[i]) { printf("FAIL: conv q/k mismatch at %zu\n", i); return false; }
    for (size_t i = 0; i < gv.size(); i++)
        if (gv[i] != ref_v[i]) { printf("FAIL: conv v mismatch at %zu\n", i); return false; }
    for (int b = 0; b < B; b++) {
        auto rs = download(ref_cs[b] + coff, cs_n), bs = download(bat_cs[b] + coff, cs_n);
        for (size_t i = 0; i < cs_n; i++)
            if (rs[i] != bs[i]) { printf("FAIL: conv state row %d at %zu\n", b, i); return false; }
    }
    printf("[ok] conv    B=%d qh=%d vh=%d hd=%d ck=%d  out+state bit-identical\n",
           B, q_heads, v_heads, hd, ck);
    return true;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { printf("[SKIP] no GPU\n"); return 0; }
    bool ok = true;
    // Qwen3.8-27B linear-attention shape: 16 q-heads, 32/48 v-heads, head_dim 128, conv kernel 4.
    for (int B : {1, 2, 4, 8}) {
        ok &= test_gdn_ar(B, 16, 32, 128, false);
        ok &= test_gdn_ar(B, 16, 48, 128, true);
        ok &= test_conv(B, 16, 32, 128, 4);
        ok &= test_conv(B, 16, 48, 128, 4);
        // non-zero offset: the packed path passes ONE array of session-base pointers and selects
        // the layer with state_off, so a wrong offset must not pass as a wrong answer.
        ok &= test_gdn_ar(B, 16, 48, 128, true, (size_t)48 * 128 * 128);
        ok &= test_conv(B, 16, 48, 128, 4, (size_t)3 * (2 * 16 * 128 + 48 * 128));
        // Same pair on the compacted bf16 state, at slot 0 AND at a later slot. A session that
        // has been through decode_packed carries this form, and the unbatched kernel is reached
        // with it every time the packed batch declines or decays to a single row -- which on a
        // 64-layer stack means 47 of the 48 GDN layers ran on a doubled offset.
        ok &= test_gdn_ar(B, 16, 48, 128, true, 0, true);
        ok &= test_gdn_ar(B, 16, 48, 128, true, (size_t)48 * 128 * 128, true);
        ok &= test_gdn_ar(B, 16, 32, 128, false, (size_t)7 * 32 * 128 * 128, true);
    }
    printf(ok ? "[PASS] gdn_batched_gpu_test\n" : "[FAIL] gdn_batched_gpu_test\n");
    return ok ? 0 : 1;
}
