// Exercises qwen_vision_splice_embeddings: does it put the right rows in the right places, and
// does it REFUSE a count mismatch instead of writing a partial result?
//
// The refusal cases matter more than the happy path. Too few placeholders drops later image
// embeddings; too many leaves stale token embeddings inside the image span. Either way the model
// gets a prompt whose image region is quietly truncated or padded, describes it fluently, and
// nothing downstream can tell. This test exists to keep that a hard error.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "sparkinfer/models/qwen_vision.h"
#include "sparkinfer/models/qwen_vision_preprocess.h"

using namespace sparkinfer;
static int g_fail = 0;
static void check(bool ok, const char* what) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
}

int main() {
    const int H = 8, IMG = 248056, N = 10, n_img = 3;
    // positions 2, 5, 6 carry the placeholder
    std::vector<int> ids = {1, 2, IMG, 4, 5, IMG, IMG, 8, 9, 10};

    std::vector<unsigned short> host_x((size_t)N * H);
    for (size_t i = 0; i < host_x.size(); i++) host_x[i] = 0x3F80 >> 0;   // arbitrary sentinel
    void* d_x = nullptr;
    if (cudaMalloc(&d_x, host_x.size() * sizeof(unsigned short)) != cudaSuccess) {
        printf("[FAIL] cudaMalloc\n"); return 1;
    }
    cudaMemcpy(d_x, host_x.data(), host_x.size() * sizeof(unsigned short), cudaMemcpyHostToDevice);

    // Distinct, exactly-bf16-representable values per row so a misplaced row is unambiguous.
    std::vector<float> emb((size_t)n_img * H);
    for (int r = 0; r < n_img; r++)
        for (int d = 0; d < H; d++) emb[(size_t)r * H + d] = (float)(r + 1) * 16.0f + d;

    std::string err;
    check(qwen_vision_splice_embeddings(d_x, ids.data(), N, emb.data(), n_img, H, IMG, err),
          "splice with matching placeholder count succeeds");

    std::vector<unsigned short> got(host_x.size());
    cudaMemcpy(got.data(), d_x, got.size() * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    auto bf2f = [](unsigned short h) { unsigned u = (unsigned)h << 16; float f; memcpy(&f, &u, 4); return f; };

    bool placed = true, untouched = true;
    const int expect_pos[3] = {2, 5, 6};
    for (int r = 0; r < n_img; r++)
        for (int d = 0; d < H; d++)
            if (bf2f(got[(size_t)expect_pos[r] * H + d]) != emb[(size_t)r * H + d]) placed = false;
    for (int t = 0; t < N; t++) {
        if (t == 2 || t == 5 || t == 6) continue;
        for (int d = 0; d < H; d++)
            if (got[(size_t)t * H + d] != host_x[(size_t)t * H + d]) untouched = false;
    }
    check(placed, "image rows land at exactly the placeholder positions");
    check(untouched, "non-placeholder rows are left untouched");

    // Refusals. Each must fail AND leave the buffer alone.
    cudaMemcpy(d_x, host_x.data(), host_x.size() * sizeof(unsigned short), cudaMemcpyHostToDevice);
    check(!qwen_vision_splice_embeddings(d_x, ids.data(), N, emb.data(), 2, H, IMG, err),
          "refuses when embeddings < placeholders");
    check(!qwen_vision_splice_embeddings(d_x, ids.data(), N, emb.data(), 4, H, IMG, err),
          "refuses when embeddings > placeholders");
    std::vector<int> none(N, 7);
    check(!qwen_vision_splice_embeddings(d_x, none.data(), N, emb.data(), n_img, H, IMG, err),
          "refuses when the prompt has no placeholders at all");
    cudaMemcpy(got.data(), d_x, got.size() * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    check(memcmp(got.data(), host_x.data(), host_x.size() * sizeof(unsigned short)) == 0,
          "a refused splice writes NOTHING (no partial state)");
    check(qwen_vision_splice_embeddings(d_x, none.data(), N, emb.data(), 0, H, IMG, err),
          "a text-only prompt (0 images, 0 placeholders) is a no-op success");

    cudaFree(d_x);

    // ---- placeholder expansion, MULTIPLE images ----
    // Every test so far has used exactly one image, so the per-image counts vector -- the whole
    // reason expand takes a vector rather than an int -- has never actually been exercised. Two
    // images of DIFFERENT sizes is the case that catches an expansion that reuses one count, or
    // that expands in the wrong order: with equal counts both bugs are invisible.
    {
        const std::vector<int> in = {1, IMG, 2, IMG, 3};
        std::vector<int> out;
        check(qwen_vision_expand_placeholders(in, IMG, {2, 3}, out, err),
              "expand accepts two images with different token counts");
        const std::vector<int> want = {1, IMG, IMG, 2, IMG, IMG, IMG, 3};
        check(out == want, "each placeholder expands to ITS OWN count, in order");

        check(!qwen_vision_expand_placeholders(in, IMG, {2}, out, err),
              "refuses 2 placeholders but 1 image");
        check(!qwen_vision_expand_placeholders(in, IMG, {2, 3, 4}, out, err),
              "refuses 2 placeholders but 3 images");
        check(!qwen_vision_expand_placeholders(in, IMG, {2, 0}, out, err),
              "refuses a zero token count");
        check(!qwen_vision_expand_placeholders(in, IMG, {2, -1}, out, err),
              "refuses a negative token count");

        std::vector<int> out2;
        check(qwen_vision_expand_placeholders({1, 2, 3}, IMG, {}, out2, err) &&
              out2 == std::vector<int>({1, 2, 3}),
              "a text-only prompt with no images passes through unchanged");
    }

    printf(g_fail ? "[FAIL] %d check(s) failed\n" : "[OK] splice and expansion behave correctly\n", g_fail);
    return g_fail ? 1 : 0;
}
