#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

template <class T> T* device_copy(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <class T> bool equal_device(const T* a, const T* b, size_t n, const char* what) {
    std::vector<T> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost);
    if (ha == hb) return true;
    size_t i = 0;
    while (i < n && ha[i] == hb[i]) i++;
    std::printf("[FAIL] %s differs at %zu\n", what, i);
    return false;
}

} // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("[SKIP] no CUDA device\n");
        return 0;
    }

    constexpr int N = 4, QH = 4, VH = 4, HD = 128, CK = 4;
    constexpr int QD = QH * HD, VD = VH * HD, QKVD = 2 * QD + VD;
    const size_t conv_state_n = (CK - 1) * QKVD;
    const size_t scan_state_n = (size_t)VH * HD * HD;
    auto make_bf16 = [](size_t n, int salt, float scale) {
        std::vector<uint16_t> h(n);
        for (size_t i = 0; i < n; i++) {
            const int v = (int)((i * 1103515245u + 12345u + salt) % 2001u) - 1000;
            h[i] = bf16(scale * v / 1000.f);
        }
        return h;
    };

    auto hqkv = make_bf16((size_t)N * QKVD, 1, 0.2f);
    auto hcw = make_bf16((size_t)QKVD * CK, 2, 0.1f);
    auto hcs = make_bf16(conv_state_n, 3, 0.2f);
    uint16_t *dqkv = device_copy(hqkv), *dcw = device_copy(hcw), *dcs = device_copy(hcs);
    uint16_t *bq, *bk, *bv, *bc, *sq, *sk, *sv, *sc, *sc_live;
    cudaMalloc(&bq, (size_t)N * QD * 2); cudaMalloc(&bk, (size_t)N * QD * 2);
    cudaMalloc(&bv, (size_t)N * VD * 2); cudaMalloc(&bc, (size_t)N * conv_state_n * 2);
    cudaMalloc(&sq, (size_t)N * QD * 2); cudaMalloc(&sk, (size_t)N * QD * 2);
    cudaMalloc(&sv, (size_t)N * VD * 2); cudaMalloc(&sc, conv_state_n * 2);
    cudaMalloc(&sc_live, conv_state_n * 2); cudaMemcpy(sc_live, dcs, conv_state_n * 2, cudaMemcpyDeviceToDevice);
    sparkinfer::kernels::launch_dflash_gdn_conv(dqkv, dcw, dcs, bq, bk, bv, bc,
                                                 N, QH, VH, HD, CK, 1e-6f);
    uint16_t *cq, *ck, *cv;
    cudaMalloc(&cq, (size_t)N * QD * 2); cudaMalloc(&ck, (size_t)N * QD * 2);
    cudaMalloc(&cv, (size_t)N * VD * 2);
    sparkinfer::kernels::launch_dflash_gdn_conv_compact(
        dqkv, dcw, dcs, cq, ck, cv, N, QH, VH, HD, CK, 1e-6f);
    for (int t = 0; t < N; t++) {
        sparkinfer::kernels::launch_dflash_gdn_conv(
            dqkv + (size_t)t * QKVD, dcw, sc_live,
            sq + (size_t)t * QD, sk + (size_t)t * QD, sv + (size_t)t * VD, sc,
            1, QH, VH, HD, CK, 1e-6f);
        cudaMemcpy(sc_live, sc, conv_state_n * 2, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    bool ok = equal_device(bq, sq, (size_t)N * QD, "conv q") &&
              equal_device(bk, sk, (size_t)N * QD, "conv k") &&
              equal_device(bv, sv, (size_t)N * VD, "conv v") &&
              equal_device(bq, cq, (size_t)N * QD, "compact conv q") &&
              equal_device(bk, ck, (size_t)N * QD, "compact conv k") &&
              equal_device(bv, cv, (size_t)N * VD, "compact conv v") &&
              equal_device(bc + (size_t)(N - 1) * conv_state_n, sc_live,
                           conv_state_n, "conv checkpoint");
    for (int accept = 1; accept <= N; accept++) {
        cudaMemcpy(sc_live, dcs, conv_state_n * 2, cudaMemcpyDeviceToDevice);
        sparkinfer::kernels::launch_dflash_gdn_conv_commit(
            dqkv, sc_live, accept, QH, VH, HD, CK);
        cudaDeviceSynchronize();
        char what[64];
        std::snprintf(what, sizeof(what), "compact conv commit %d", accept);
        ok = equal_device(bc + (size_t)(accept - 1) * conv_state_n, sc_live,
                          conv_state_n, what) && ok;
    }

    auto hq = make_bf16((size_t)N * QD, 4, 0.1f);
    auto hk = make_bf16((size_t)N * QD, 5, 0.1f);
    auto hv = make_bf16((size_t)N * VD, 6, 0.1f);
    auto ha = make_bf16((size_t)N * VH, 7, 0.1f);
    auto hb = make_bf16((size_t)N * VH, 8, 0.1f);
    auto hdt = make_bf16(VH, 9, 0.05f);
    auto hdecay = make_bf16(VH, 10, -0.05f);
    std::vector<float> hstate(scan_state_n);
    for (size_t i = 0; i < hstate.size(); i++) hstate[i] = ((int)(i % 37) - 18) * 0.0001f;
    uint16_t *dq = device_copy(hq), *dk = device_copy(hk), *dv = device_copy(hv);
    uint16_t *da = device_copy(ha), *db = device_copy(hb), *ddt = device_copy(hdt), *ddecay = device_copy(hdecay);
    float* dstate = device_copy(hstate);
    uint16_t *bout, *sout; float *bcp, *scp, *sstate;
    cudaMalloc(&bout, (size_t)N * VD * 2); cudaMalloc(&sout, (size_t)N * VD * 2);
    cudaMalloc(&bcp, (size_t)N * scan_state_n * 4); cudaMalloc(&scp, scan_state_n * 4);
    cudaMalloc(&sstate, scan_state_n * 4); cudaMemcpy(sstate, dstate, scan_state_n * 4, cudaMemcpyDeviceToDevice);
    sparkinfer::kernels::launch_dflash_gdn_scan(dq, dk, dv, da, db, ddt, ddecay,
                                                 dstate, bout, bcp, N, QH, VH, HD);
    uint16_t* cout = nullptr;
    cudaMalloc(&cout, (size_t)N * VD * 2);
    sparkinfer::kernels::launch_dflash_gdn_scan_compact(
        dq, dk, dv, da, db, ddt, ddecay, dstate, cout, N, QH, VH, HD);
    for (int t = 0; t < N; t++) {
        sparkinfer::kernels::launch_dflash_gdn_scan(
            dq + (size_t)t * QD, dk + (size_t)t * QD, dv + (size_t)t * VD,
            da + (size_t)t * VH, db + (size_t)t * VH, ddt, ddecay,
            sstate, sout + (size_t)t * VD, scp, 1, QH, VH, HD);
        cudaMemcpy(sstate, scp, scan_state_n * 4, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    ok = equal_device(bout, sout, (size_t)N * VD, "scan output") &&
         equal_device(bout, cout, (size_t)N * VD, "compact scan output") &&
         equal_device(bcp + (size_t)(N - 1) * scan_state_n, sstate,
                      scan_state_n, "scan checkpoint") && ok;
    for (int accept = 1; accept <= N; accept++) {
        cudaMemcpy(sstate, dstate, scan_state_n * 4, cudaMemcpyDeviceToDevice);
        sparkinfer::kernels::launch_dflash_gdn_scan_commit(
            dk, dv, da, db, ddt, ddecay, sstate, accept, QH, VH, HD);
        cudaDeviceSynchronize();
        char what[64];
        std::snprintf(what, sizeof(what), "compact scan commit %d", accept);
        ok = equal_device(bcp + (size_t)(accept - 1) * scan_state_n, sstate,
                          scan_state_n, what) && ok;
    }

    // One-sequence short-block QK-norm/RoPE/KV append must equal N decode-shaped launches.
    constexpr int AQH = 4, AKH = 2, AHD = 128, ARD = 64, ABS = 16;
    constexpr int APOS0 = 14, APOOLTOK = 32;
    auto haqq = make_bf16((size_t)N * AQH * AHD, 21, 0.1f);
    auto hakk = make_bf16((size_t)N * AKH * AHD, 22, 0.1f);
    auto havv = make_bf16((size_t)N * AKH * AHD, 23, 0.1f);
    auto hanw = make_bf16(AHD, 24, 0.05f);
    std::vector<int> hpos(N), htable = {0, 1};
    for (int i = 0; i < N; i++) hpos[i] = APOS0 + i;
    uint16_t *aqq0 = device_copy(haqq), *aqq1 = device_copy(haqq);
    uint16_t *akk = device_copy(hakk), *avv = device_copy(havv), *anw = device_copy(hanw);
    int *apos = device_copy(hpos), *atable = device_copy(htable);
    uint16_t *akp0, *avp0, *akp1, *avp1;
    const size_t apool_n = (size_t)APOOLTOK * AKH * AHD;
    cudaMalloc(&akp0, apool_n * 2); cudaMalloc(&avp0, apool_n * 2);
    cudaMalloc(&akp1, apool_n * 2); cudaMalloc(&avp1, apool_n * 2);
    cudaMemset(akp0, 0, apool_n * 2); cudaMemset(avp0, 0, apool_n * 2);
    cudaMemset(akp1, 0, apool_n * 2); cudaMemset(avp1, 0, apool_n * 2);
    sparkinfer::kernels::launch_dflash_qknorm_rope_kv_partial(
        aqq0, akk, avv, anw, anw, akp0, avp0, atable, apos,
        N, AQH, AKH, AHD, ARD, 1000000.f, 1e-6f, ABS, 2);
    for (int i = 0; i < N; i++)
        sparkinfer::kernels::launch_qknorm_rope_kv_partial(
            aqq1 + (size_t)i * AQH * AHD,
            akk + (size_t)i * AKH * AHD, avv + (size_t)i * AKH * AHD,
            anw, anw, akp1, avp1, atable, apos + i,
            1, AQH, AKH, AHD, ARD, 1000000.f, 1e-6f, ABS, 2);
    cudaDeviceSynchronize();
    ok = equal_device(aqq0, aqq1, (size_t)N * AQH * AHD, "short-block Q RoPE") &&
         equal_device(akp0, akp1, apool_n, "short-block K append") &&
         equal_device(avp0, avp1, apool_n, "short-block V append") && ok;

    auto haqraw = make_bf16((size_t)N * AQH * AHD * 2, 28, 0.1f);
    uint16_t *aqraw = device_copy(haqraw);
    uint16_t *aiq0, *aiq1, *aig0, *aig1;
    uint16_t *aik0 = device_copy(hakk), *aik1 = device_copy(hakk);
    cudaMalloc(&aiq0, (size_t)N * AQH * AHD * 2); cudaMalloc(&aiq1, (size_t)N * AQH * AHD * 2);
    cudaMalloc(&aig0, (size_t)N * AQH * AHD * 2); cudaMalloc(&aig1, (size_t)N * AQH * AHD * 2);
    signed char *aikp0, *aivp0, *aikp1, *aivp1;
    uint16_t *aiks0, *aivs0, *aiks1, *aivs1;
    const size_t ascale_n = (size_t)APOOLTOK * AKH;
    cudaMalloc(&aikp0, apool_n); cudaMalloc(&aivp0, apool_n);
    cudaMalloc(&aikp1, apool_n); cudaMalloc(&aivp1, apool_n);
    cudaMalloc(&aiks0, ascale_n * 2); cudaMalloc(&aivs0, ascale_n * 2);
    cudaMalloc(&aiks1, ascale_n * 2); cudaMalloc(&aivs1, ascale_n * 2);
    cudaMemset(aikp0, 0, apool_n); cudaMemset(aivp0, 0, apool_n);
    cudaMemset(aikp1, 0, apool_n); cudaMemset(aivp1, 0, apool_n);
    cudaMemset(aiks0, 0, ascale_n * 2); cudaMemset(aivs0, 0, ascale_n * 2);
    cudaMemset(aiks1, 0, ascale_n * 2); cudaMemset(aivs1, 0, ascale_n * 2);
    sparkinfer::kernels::launch_dflash_qknorm_rope_kv_partial_int8_gated(
        aqraw, aiq0, aig0, aik0, avv, anw, anw, aikp0, aivp0, aiks0, aivs0, atable, apos,
        N, AQH, AKH, AHD, ARD, 1000000.f, 1e-6f, ABS, 2);
    for (int i = 0; i < N; i++)
        sparkinfer::kernels::launch_qknorm_rope_kv_partial_int8_gated(
            aqraw + (size_t)i * AQH * AHD * 2,
            aiq1 + (size_t)i * AQH * AHD, aig1 + (size_t)i * AQH * AHD,
            aik1 + (size_t)i * AKH * AHD, avv + (size_t)i * AKH * AHD,
            anw, anw, aikp1, aivp1, aiks1, aivs1, atable, apos + i,
            1, AQH, AKH, AHD, ARD, 1000000.f, 1e-6f, ABS, 2);
    cudaDeviceSynchronize();
    ok = equal_device(aiq0, aiq1, (size_t)N * AQH * AHD, "short-block int8 Q RoPE") &&
         equal_device(aig0, aig1, (size_t)N * AQH * AHD, "short-block Q gate") &&
         equal_device(aikp0, aikp1, apool_n, "short-block int8 K append") &&
         equal_device(aivp0, aivp1, apool_n, "short-block int8 V append") &&
         equal_device(aiks0, aiks1, ascale_n, "short-block K scales") &&
         equal_device(aivs0, aivs1, ascale_n, "short-block V scales") && ok;

    constexpr int SFFN = 512;
    auto hsg = make_bf16((size_t)N * SFFN, 25, 0.2f);
    auto hsu = make_bf16((size_t)N * SFFN, 26, 0.2f);
    auto hsw = make_bf16(N, 27, 0.1f);
    uint16_t *dsg = device_copy(hsg), *dsu = device_copy(hsu), *dsw = device_copy(hsw);
    float *dfw0, *dfw1; uint16_t *dsh0, *dsh1;
    cudaMalloc(&dfw0, N * sizeof(float)); cudaMalloc(&dfw1, N * sizeof(float));
    cudaMalloc(&dsh0, (size_t)N * SFFN * 2); cudaMalloc(&dsh1, (size_t)N * SFFN * 2);
    sparkinfer::kernels::launch_qwen36_sigmoid_rows(dsw, dfw0, N);
    for (int i = 0; i < N; i++)
        sparkinfer::kernels::launch_qwen36_sigmoid_scalar(dsw + i, dfw1 + i);
    sparkinfer::kernels::launch_qwen36_shared_swiglu_rows(dsg, dsu, dfw0, dsh0, N, SFFN);
    for (int i = 0; i < N; i++)
        sparkinfer::kernels::launch_qwen36_shared_swiglu(
            dsg + (size_t)i * SFFN, dsu + (size_t)i * SFFN, dfw1 + i,
            dsh1 + (size_t)i * SFFN, SFFN);
    cudaDeviceSynchronize();
    ok = equal_device(dfw0, dfw1, N, "shared gate sigmoid rows") &&
         equal_device(dsh0, dsh1, (size_t)N * SFFN, "shared SwiGLU rows") && ok;

    // Exact multi-row Q4_K projection must be byte-identical to M independent decode MMVQs.
    constexpr int MH = 2048, MN = 8192;
    auto hact = make_bf16((size_t)N * MH, 11, 0.2f);
    uint16_t* dact = device_copy(hact);
    const size_t q81_row = sparkinfer::kernels::llama_q8_1_bytes(MH);
    void* dq81 = nullptr;
    cudaMalloc(&dq81, (size_t)N * q81_row);
    sparkinfer::kernels::launch_quantize_q8_1_rows(dact, dq81, MH, N, MH);
    std::vector<unsigned char> hw((size_t)MN * 8 * 144);
    for (int row = 0; row < MN; row++) for (int sb = 0; sb < 8; sb++) {
        unsigned char* p = hw.data() + ((size_t)row * 8 + sb) * 144;
        p[0] = 0x1f; p[1] = 0x21; p[2] = 0x1f; p[3] = 0x21; // finite fp16 dm
        for (int i = 4; i < 144; i++) p[i] = (unsigned char)((row * 17 + sb * 29 + i * 13) & 255);
    }
    unsigned char* dw = device_copy(hw);
    uint16_t *ym = nullptr, *ys = nullptr;
    cudaMalloc(&ym, (size_t)N * MN * 2); cudaMalloc(&ys, (size_t)N * MN * 2);
    ok = sparkinfer::kernels::launch_mmvq_q4k_rows(dq81, dw, ym, N, MN, MH) && ok;
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q4k((const char*)dq81 + (size_t)m * q81_row,
                                              dw, ys + (size_t)m * MN, MN, MH);
    cudaDeviceSynchronize();
    ok = equal_device(ym, ys, (size_t)N * MN, "Q4_K exact rows") && ok;
    float *yfm = nullptr, *yfs = nullptr;
    cudaMalloc(&yfm, (size_t)N * MN * sizeof(float));
    cudaMalloc(&yfs, (size_t)N * MN * sizeof(float));
    ok = sparkinfer::kernels::launch_mmvq_rows_f32(12, dq81, dw, yfm, N, MN, MH) && ok;
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q4k_f32((const char*)dq81 + (size_t)m * q81_row,
                                                 dw, yfs + (size_t)m * MN, MN, MH);
    cudaDeviceSynchronize();
    ok = equal_device(yfm, yfs, (size_t)N * MN, "Q4_K exact FP32 rows") && ok;
    std::vector<unsigned char> hw6((size_t)MN * 8 * 210);
    for (int row = 0; row < MN; row++) for (int sb = 0; sb < 8; sb++) {
        unsigned char* p = hw6.data() + ((size_t)row * 8 + sb) * 210;
        for (int i = 0; i < 208; i++)
            p[i] = (unsigned char)((row * 11 + sb * 31 + i * 7) & 255);
        p[208] = 0x00; p[209] = 0x3c; // finite fp16 super-block scale (1.0)
    }
    unsigned char* dw6 = device_copy(hw6);
    ok = sparkinfer::kernels::launch_mmvq_q6k_rows(dq81, dw6, ym, N, MN, MH) && ok;
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q6k((const char*)dq81 + (size_t)m * q81_row,
                                              dw6, ys + (size_t)m * MN, MN, MH);
    cudaDeviceSynchronize();
    ok = equal_device(ym, ys, (size_t)N * MN, "Q6_K exact rows") && ok;
    std::vector<unsigned char> hw8((size_t)MN * (MH / 32) * 34);
    for (int row = 0; row < MN; row++) for (int kb = 0; kb < MH / 32; kb++) {
        unsigned char* p = hw8.data() + ((size_t)row * (MH / 32) + kb) * 34;
        p[0] = 0x00; p[1] = 0x3c; // finite fp16 block scale (1.0)
        for (int i = 2; i < 34; i++) p[i] = (unsigned char)((row * 5 + kb * 19 + i * 3) & 255);
    }
    unsigned char* dw8 = device_copy(hw8);
    ok = sparkinfer::kernels::launch_mmvq_q80_rows(dq81, dw8, ym, N, MN, MH) && ok;
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q80((const char*)dq81 + (size_t)m * q81_row,
                                              dw8, ys + (size_t)m * MN, MN, MH);
    cudaDeviceSynchronize();
    ok = equal_device(ym, ys, (size_t)N * MN, "Q8_0 exact rows") && ok;
    constexpr int K512 = 512, N512 = 2048;
    const size_t q815_row = sparkinfer::kernels::llama_q8_1_bytes(K512);
    void* dq815 = nullptr;
    cudaMalloc(&dq815, (size_t)N * q815_row);
    sparkinfer::kernels::launch_quantize_q8_1_rows(dact, dq815, K512, N, MH);
    std::vector<unsigned char> hw85((size_t)N512 * (K512 / 32) * 34);
    for (int row = 0; row < N512; row++) for (int kb = 0; kb < K512 / 32; kb++) {
        unsigned char* p = hw85.data() + ((size_t)row * (K512 / 32) + kb) * 34;
        p[0] = 0x00; p[1] = 0x3c;
        for (int i = 2; i < 34; i++) p[i] = (unsigned char)((row * 23 + kb * 3 + i * 5) & 255);
    }
    unsigned char* dw85 = device_copy(hw85);
    ok = sparkinfer::kernels::launch_mmvq_q80_rows(dq815, dw85, ym, N, N512, K512) && ok;
    for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q80((const char*)dq815 + (size_t)m * q815_row,
                                              dw85, ys + (size_t)m * N512, N512, K512);
    cudaDeviceSynchronize();
    ok = equal_device(ym, ys, (size_t)N * N512, "Q8_0 exact rows K512") && ok;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    constexpr int REPS = 40;
    cudaEventRecord(ev0);
    for (int r = 0; r < REPS; r++)
        sparkinfer::kernels::launch_mmvq_q4k_rows(dq81, dw, ym, N, MN, MH);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float batch_ms = 0.f; cudaEventElapsedTime(&batch_ms, ev0, ev1);
    cudaEventRecord(ev0);
    for (int r = 0; r < REPS; r++) for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q4k((const char*)dq81 + (size_t)m * q81_row,
                                              dw, ys + (size_t)m * MN, MN, MH);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float serial_ms = 0.f; cudaEventElapsedTime(&serial_ms, ev0, ev1);
    std::printf("Q4_K rows: batch %.3f ms, four serial %.3f ms, %.2fx\n",
                batch_ms / REPS, serial_ms / REPS, serial_ms / batch_ms);
    cudaEventRecord(ev0);
    for (int r = 0; r < REPS; r++)
        sparkinfer::kernels::launch_mmvq_rows_f32(12, dq81, dw, yfm, N, MN, MH);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float fbatch_ms = 0.f; cudaEventElapsedTime(&fbatch_ms, ev0, ev1);
    cudaEventRecord(ev0);
    for (int r = 0; r < REPS; r++) for (int m = 0; m < N; m++)
        sparkinfer::kernels::launch_mmvq_q4k_f32((const char*)dq81 + (size_t)m * q81_row,
                                                 dw, yfs + (size_t)m * MN, MN, MH);
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float fserial_ms = 0.f; cudaEventElapsedTime(&fserial_ms, ev0, ev1);
    std::printf("Q4_K FP32 rows: batch %.3f ms, four serial %.3f ms, %.2fx\n",
                fbatch_ms / REPS, fserial_ms / REPS, fserial_ms / fbatch_ms);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("[FAIL] CUDA: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf(ok ? "[PASS] DFlash GDN checkpoints and compact commits equal serial state\n"
                   : "[FAIL] DFlash GDN checkpoint/commit equivalence\n");
    return ok ? 0 : 1;
}
