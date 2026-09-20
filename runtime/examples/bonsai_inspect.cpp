// Prints what this runtime reads out of a Ternary-Bonsai-2 GGUF: the PTQ1_0 tensor inventory, the
// prism.hadamard.* rotation metadata, and one decoded block. A checkpoint is 6 GB and cannot live
// in a unit test, so this is how the readers get exercised against the real file.
//
//   bonsai_inspect <model.gguf> [tensor-name]
#include "sparkinfer/gguf.h"
#include "sparkinfer/prism_hadamard.h"
#include "sparkinfer/ternary_ptq1.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [tensor]\n", argv[0]); return 2; }
    sparkinfer::GGUF gguf;
    if (!gguf.open(argv[1])) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    std::printf("architecture     %s\n", gguf.meta_str("general.architecture").c_str());
    std::printf("file_type        %ld\n", gguf.meta_int("general.file_type", -1));
    std::printf("blocks           %ld\n", gguf.meta_int("qwen35.block_count", 0));
    std::printf("embedding        %ld\n", gguf.meta_int("qwen35.embedding_length", 0));
    std::printf("context          %ld\n", gguf.meta_int("qwen35.context_length", 0));

    sparkinfer::PrismHadamard had;
    std::string err;
    if (!had.load(gguf, err)) { std::fprintf(stderr, "hadamard metadata: %s\n", err.c_str()); return 1; }
    std::printf("hadamard         present=%d version=%ld block=%ld transform=%s\n",
                (int)had.present, had.version, had.block_size, had.transform.c_str());
    std::printf("                 rotated=%zu inverse=%zu gdn_v_grouped=%d\n",
                had.rotated.size(), had.inverse_rotated.size(), (int)had.gdn_v_grouped);
    for (const std::string& n : had.inverse_rotated)
        std::printf("                 inverse-rotated %s\n", n.c_str());
    {   // which families are rotated at all, and is the embedding among them
        std::map<std::string, int> fam;
        for (const std::string& n : had.rotated) {
            std::string f = n;
            if (f.compare(0, 4, "blk.") == 0) {
                const size_t a = f.find('.', 4);
                f = "blk.N" + f.substr(a);
            }
            fam[f]++;
        }
        for (const auto& kv : fam)
            std::printf("                 rotated %-28s x%d\n", kv.first.c_str(), kv.second);
    }
    for (const auto& kv : had.signs_by_width) {
        long plus = 0;
        for (int8_t s : kv.second) plus += (s > 0);
        std::printf("                 signs width %-6ld  +1 %ld  -1 %ld\n",
                    kv.first, plus, (long)kv.second.size() - plus);
    }

    if (argc > 2 && std::strcmp(argv[2], "--meta") == 0) {
        for (const auto& kv : gguf.meta_all())
            std::printf("  %-46s %s\n", kv.first.c_str(), kv.second.c_str());
        return 0;
    }

    if (argc > 2 && std::strcmp(argv[2], "--list") == 0) {
        std::map<int, std::pair<long, double>> by_type;   // ggml type -> (count, GB)
        std::map<std::string, std::string> shapes;        // a representative name per shape family
        for (const auto& kv : gguf.tensors()) {
            auto& e = by_type[kv.second.ggml_type];
            e.first += 1;
            e.second += (double)kv.second.n_bytes / 1e9;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "[%ld, %ld] type %d",
                          kv.second.dims[0], kv.second.dims[1], kv.second.ggml_type);
            std::string family = kv.first;
            const size_t dot = family.find('.');
            if (family.compare(0, 4, "blk.") == 0) {      // blk.31.ffn_down.weight -> blk.N.ffn_down.weight
                const size_t second = family.find('.', dot + 1);
                family = "blk.N" + family.substr(second);
            }
            shapes[family] = buf;
        }
        if (argc > 3) {   // a substring filter prints real names instead of collapsed families
            std::map<std::string, std::string> hits;
            for (const auto& kv : gguf.tensors()) {
                if (kv.first.find(argv[3]) == std::string::npos) continue;
                char buf[64];
                std::snprintf(buf, sizeof(buf), "[%ld, %ld] type %d",
                              kv.second.dims[0], kv.second.dims[1], kv.second.ggml_type);
                hits[kv.first] = buf;
            }
            for (const auto& kv : hits) std::printf("  %-40s %s\n", kv.first.c_str(), kv.second.c_str());
            return 0;
        }
        for (const auto& kv : by_type)
            std::printf("type %-4d %5ld tensors  %7.2f GB\n", kv.first, kv.second.first, kv.second.second);
        for (const auto& kv : shapes)
            std::printf("  %-34s %s\n", kv.first.c_str(), kv.second.c_str());
        return 0;
    }

    const char* want = argc > 2 ? argv[2] : "blk.0.attn_gate.weight";
    const sparkinfer::GGUFTensor* t = gguf.tensor(want);
    if (!t) { std::fprintf(stderr, "tensor %s not found\n", want); return 1; }
    std::printf("tensor %s: type %d dims [%ld, %ld] values %ld bytes %ld\n",
                want, t->ggml_type, t->dims[0], t->dims[1], t->n_values, t->n_bytes);
    if (t->ggml_type != sparkinfer::kPtq1GgmlType) return 0;

    const long want_bytes = t->n_values / sparkinfer::kPtq1BlockElems * sparkinfer::kPtq1BlockBytes;
    std::printf("sizing           %s (expected %ld bytes)\n",
                want_bytes == t->n_bytes ? "matches the block layout" : "MISMATCH", want_bytes);
    std::printf("rotated          %s\n", had.rotates(want) ? "yes (input axis)" : "no");

    float vals[sparkinfer::kPtq1BlockElems];
    sparkinfer::ptq1_dequant_block(static_cast<const uint8_t*>(t->data), vals);
    const float scale = sparkinfer::ptq1_block_scale(static_cast<const uint8_t*>(t->data));
    std::map<int, int> hist;
    long nonternary = 0;
    for (float v : vals) {
        const int trit = v > 0.5f * scale ? 1 : (v < -0.5f * scale ? -1 : 0);
        hist[trit]++;
        if (std::abs(std::abs(v) - scale) > 1e-6f && v != 0.0f) ++nonternary;
    }
    std::printf("first block      scale %.6f  -1:%d  0:%d  +1:%d  off-grid:%ld\n",
                scale, hist[-1], hist[0], hist[1], nonternary);
    std::printf("first 8 values   ");
    for (int i = 0; i < 8; ++i) std::printf("%+.5f ", vals[i]);
    std::printf("\n");

    // Transcode the whole tensor to Q4_K and compare against the direct ternary dequant. Real
    // tensors are the test the unit test cannot be: adjacent groups there have genuinely different
    // scales, which is exactly what Q4_K's shared six-bit grid has to absorb.
    const long n = t->n_values;
    if (n % sparkinfer::kQ4KBlockElems == 0) {
        std::vector<uint8_t> q4k((size_t)(n / sparkinfer::kQ4KBlockElems) * sparkinfer::kQ4KBlockBytes);
        sparkinfer::ptq1_to_q4k(static_cast<const uint8_t*>(t->data), (size_t)n, q4k.data());
        std::vector<float> ref((size_t)n);
        sparkinfer::ptq1_dequant(static_cast<const uint8_t*>(t->data), (size_t)n, ref.data());

        // Error is measured in units of the group's OWN scale: a trit step. Dividing by |want| is
        // meaningless for the third of all weights that are zero, and that -- not the packing --
        // was what an early run's "worst relative error 9.06" was measuring.
        double worst_lo = 0.0, worst_hi = 0.0, sum_sq_err = 0.0, sum_sq_ref = 0.0, worst_zero = 0.0;
        long off_grid = 0;
        for (long sb = 0; sb < n / sparkinfer::kQ4KBlockElems; ++sb) {
            const uint8_t* blk = q4k.data() + (size_t)sb * sparkinfer::kQ4KBlockBytes;
            uint16_t hd, hm;
            std::memcpy(&hd, blk, 2);
            std::memcpy(&hm, blk + 2, 2);
            auto h2f = [](uint16_t h) -> float {
                const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
                if (exp == 0) {   // subnormal: d lives down here for any group scale below ~3.8e-3
                    const float f = (float)man * 5.9604644775390625e-8f;
                    return sign ? -f : f;
                }
                uint32_t bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
                float f; std::memcpy(&f, &bits, sizeof(f)); return f;
            };
            const float d = h2f(hd), dmin = h2f(hm);
            const uint8_t* scales = blk + 4;
            const uint8_t* qs = blk + 16;
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, mn;
                if (j < 4) { sc = scales[j] & 63; mn = scales[j + 4] & 63; }
                else {
                    sc = (uint8_t)((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4));
                    mn = (uint8_t)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
                }
                for (int i = 0; i < 32; ++i) {
                    const uint8_t byte = qs[(j / 2) * 32 + i];
                    const int q = (j % 2) == 0 ? (byte & 0xF) : (byte >> 4);
                    const float got = d * (float)sc * (float)q - dmin * (float)mn;
                    const float want = ref[(size_t)sb * sparkinfer::kQ4KBlockElems + j * 32 + i];
                    const double e = std::fabs(got - want);
                    const size_t idx = (size_t)sb * sparkinfer::kQ4KBlockElems + j * 32 + i;
                    const float gs = sparkinfer::ptq1_block_scale(
                        static_cast<const uint8_t*>(t->data) +
                        (idx / sparkinfer::kPtq1BlockElems) * sparkinfer::kPtq1BlockBytes);
                    const double steps = gs > 0 ? e / gs : 0.0;   // error as a fraction of a trit
                    if (want == 0.0f) worst_zero = std::fmax(worst_zero, steps);
                    else if (j < 4) worst_lo = std::fmax(worst_lo, steps);
                    else worst_hi = std::fmax(worst_hi, steps);
                    sum_sq_err += e * e;
                    sum_sq_ref += (double)want * want;
                    if (q != 7 && q != 8 && q != 9) ++off_grid;
                }
            }
        }
        std::printf("q4k transcode    worst err (trit steps) zero %.4f  sub-block j<4 %.4f  j>=4 %.4f\n"
                    "                 rms err/rms %.6f  off-grid nibbles %ld  size %.2f GB/27B\n",
                    worst_zero, worst_lo, worst_hi,
                    std::sqrt(sum_sq_err / (sum_sq_ref > 0 ? sum_sq_ref : 1)), off_grid,
                    27e9 * sparkinfer::kQ4KBlockBytes / sparkinfer::kQ4KBlockElems / 1e9);
    }
    return 0;
}
