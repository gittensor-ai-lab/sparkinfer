// Cross-checks qwen_vision_smart_resize against the Python reference over a grid of shapes.
// Prints one line per case as "h w -> H W" so bench/scripts/vision_resize_check.py can diff it
// against transformers' own smart_resize. Exact integer logic, so any disagreement is a real bug,
// not tolerance -- and an off-by-one here changes the patch grid and therefore how many image
// tokens the prompt must carry.
#include <cstdio>
#include <string>
#include <vector>
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/models/qwen_vision_config.h"
using namespace sparkinfer;

// Placeholder expansion, checked alongside smart_resize because the two together determine
// whether the prompt's image-token count matches what the tower produces -- the single condition
// the splice refuses on.
static int expand_checks() {
    QwenVisionConfig cfg;
    const int IMG = cfg.image_token_id, VS = cfg.vision_start_token_id, VE = cfg.vision_end_token_id;
    int fail = 0;
    auto want = [&](bool ok, const char* what) {
        printf("# %-56s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fail++;
    };
    std::string err; std::vector<int> out;

    // one image, template emits a single pad, expands to 4
    want(qwen_vision_expand_placeholders({1, VS, IMG, VE, 2}, IMG, {4}, out, err)
         && out == std::vector<int>({1, VS, IMG, IMG, IMG, IMG, VE, 2}),
         "single placeholder expands to N in place");
    // two images with different grids
    want(qwen_vision_expand_placeholders({VS, IMG, VE, 9, VS, IMG, VE}, IMG, {2, 3}, out, err)
         && out == std::vector<int>({VS, IMG, IMG, VE, 9, VS, IMG, IMG, IMG, VE}),
         "two images expand independently, in order");
    // text-only prompt is untouched
    want(qwen_vision_expand_placeholders({1, 2, 3}, IMG, {}, out, err)
         && out == std::vector<int>({1, 2, 3}),
         "text-only prompt passes through unchanged");
    // count mismatches must fail, not guess
    want(!qwen_vision_expand_placeholders({VS, IMG, VE}, IMG, {2, 2}, out, err),
         "more images than placeholders is an error");
    want(!qwen_vision_expand_placeholders({VS, IMG, VE, VS, IMG, VE}, IMG, {2}, out, err),
         "more placeholders than images is an error");
    want(!qwen_vision_expand_placeholders({VS, IMG, VE}, IMG, {0}, out, err),
         "a zero token count is an error");
    // grid -> token count agrees with the reference formula
    want(qwen_vision_num_tokens(48, 48, cfg) == 576, "48x48 grid -> 576 image tokens");
    want(qwen_vision_num_tokens(4, 4, cfg) == 4,      "4x4 grid -> 4 image tokens");
    want(qwen_vision_num_tokens(5, 4, cfg) == 0,      "odd grid -> 0 (caller must pad)");
    return fail;
}

int main(int argc, char** argv) {
    const int ef = expand_checks();
    if (ef) printf("# EXPAND CHECKS FAILED: %d\n", ef);
    QwenVisionConfig cfg;              // defaults match the released checkpoint
    const int factor = cfg.size_granularity();
    long minp = cfg.min_pixels, maxp = cfg.max_pixels;
    if (argc >= 3) { minp = atol(argv[1]); maxp = atol(argv[2]); }
    printf("# factor=%d min_pixels=%ld max_pixels=%ld\n", factor, minp, maxp);
    const int dims[] = {1, 7, 16, 31, 32, 33, 64, 100, 127, 224, 256, 333, 512, 768,
                        1024, 1080, 1920, 2048, 3000, 4096, 8192};
    for (int h : dims)
        for (int w : dims) {
            int oh = 0, ow = 0; std::string err;
            if (qwen_vision_smart_resize(h, w, factor, minp, maxp, &oh, &ow, err))
                printf("%d %d -> %d %d\n", h, w, oh, ow);
            else
                printf("%d %d -> ERR\n", h, w);
        }
    return 0;
}
