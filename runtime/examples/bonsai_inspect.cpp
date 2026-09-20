// Prints what this runtime reads out of a Ternary-Bonsai-2 GGUF: the PTQ1_0 tensor inventory, the
// prism.hadamard.* rotation metadata, and one decoded block. A checkpoint is 6 GB and cannot live
// in a unit test, so this is how the readers get exercised against the real file.
//
//   bonsai_inspect <model.gguf> [tensor-name]
#include "sparkinfer/gguf.h"
#include "sparkinfer/prism_hadamard.h"
#include "sparkinfer/ternary_ptq1.h"

#include <cstdio>
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
    for (const auto& kv : had.signs_by_width) {
        long plus = 0;
        for (int8_t s : kv.second) plus += (s > 0);
        std::printf("                 signs width %-6ld  +1 %ld  -1 %ld\n",
                    kv.first, plus, (long)kv.second.size() - plus);
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
    return 0;
}
