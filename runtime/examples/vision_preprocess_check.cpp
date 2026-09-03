// Differential for image preprocessing: our resize/normalize/patchify vs. HF's, on a real image.
//
// Every other vision check so far happened to use images whose sides were already multiples of
// 32, which makes smart_resize the identity -- so the bicubic resampler, the one stage never
// verified against upstream, never actually ran. This binary dumps the preprocessed tensor for
// ANY image so bench/scripts/vision_preprocess_ref.py can judge it against PIL.
//
// CPU only: no runtime init, no device allocation, so this is safe to run while the eval bot
// has the GPU.
//
// Usage: vision_preprocess_check <ckpt_dir> <image> [--out FILE]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sparkinfer/models/qwen_vision_config.h"
#include "sparkinfer/models/qwen_vision_preprocess.h"
#include "sparkinfer/models/qwen_vision_hf_config.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_FAILURE_STRINGS_MUTABLE
#include "stb_image.h"

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <ckpt_dir> <image> [--out FILE]\n", argv[0]); return 2; }
    const std::string dir = argv[1], img_path = argv[2];
    std::string out_path = "/tmp/vision_pre.bin";
    for (int i = 3; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--out")) out_path = argv[i + 1];

    QwenVisionConfig vc; std::string err;
    if (!qwen_vision_config_from_hf_json(dir, vc, err) || !vc.present) {
        printf("[FAIL] vision config: %s\n", err.c_str()); return 1;
    }
    int w = 0, h = 0, ch = 0;
    unsigned char* rgb = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
    if (!rgb) { printf("[FAIL] cannot decode %s\n", img_path.c_str()); return 1; }
    printf("image %s  %dx%d (source channels %d)\n", img_path.c_str(), w, h, ch);

    std::vector<float> pixels; int gh = 0, gw = 0;
    if (!qwen_vision_preprocess(rgb, h, w, vc, pixels, &gh, &gw, err)) {
        printf("[FAIL] preprocess: %s\n", err.c_str()); stbi_image_free(rgb); return 1;
    }
    stbi_image_free(rgb);

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { printf("[FAIL] cannot write %s\n", out_path.c_str()); return 1; }
    fwrite(pixels.data(), sizeof(float), pixels.size(), f);
    fclose(f);
    printf("patch grid %dx%d = %d patches, %zu floats -> %s\n", gh, gw, gh * gw, pixels.size(),
           out_path.c_str());
    printf("compare with: vision_preprocess_ref.py <dir> %s --compare %s\n",
           img_path.c_str(), out_path.c_str());
    return 0;
}
