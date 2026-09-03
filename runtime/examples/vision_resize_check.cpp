// Cross-checks qwen_vision_smart_resize against the Python reference over a grid of shapes.
// Prints one line per case as "h w -> H W" so bench/scripts/vision_resize_check.py can diff it
// against transformers' own smart_resize. Exact integer logic, so any disagreement is a real bug,
// not tolerance -- and an off-by-one here changes the patch grid and therefore how many image
// tokens the prompt must carry.
#include <cstdio>
#include <string>
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/models/qwen_vision_config.h"
using namespace sparkinfer;

int main(int argc, char** argv) {
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
