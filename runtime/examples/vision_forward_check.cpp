// Runs the CUDA vision tower on the same synthetic input bench/scripts/vision_ref.py uses, and
// writes the merged embeddings to a raw f32 file for that script to diff against.
//
// The reference is the contract. This binary deliberately does NOT decide whether the result is
// correct -- it only produces the number; vision_ref.py --compare does the judging, so the
// tolerance lives in one place and cannot drift between the two.
//
// With --pixels, the patch tensor is read from a file instead of synthesised. That is how the
// tower gets compared against transformers itself: feeding HF's OWN pixel_values in means the
// comparison tests the tower alone, with preprocessing eliminated as a variable rather than
// assumed correct. The file is raw f32, [gh*gw, C*T*P*P], row-major over the patch grid --
// callers holding HF's merge-block-major order must permute it first.
//
// Usage: vision_forward_check <checkpoint_dir> [--h 768 --w 768] [--out /tmp/vision_cuda.bin]
//        vision_forward_check <checkpoint_dir> --pixels FILE --gh GH --gw GW [--out FILE]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sparkinfer/safetensors.h"
#include "sparkinfer/models/qwen_vision.h"
#include "qwen_vision_hf_config.h"

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <checkpoint_dir> [--h H --w W] [--out FILE]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    int h = 768, w = 768;                       // 48x48 patch grid: pos-embed resample is the
    std::string out_path = "/tmp/vision_cuda.bin";   // identity there, one less variable
    std::string pix_path;
    int arg_gh = 0, arg_gw = 0;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--h")) h = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--w")) w = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--out")) out_path = argv[i + 1];
        else if (!strcmp(argv[i], "--pixels")) pix_path = argv[i + 1];
        else if (!strcmp(argv[i], "--gh")) arg_gh = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--gw")) arg_gw = atoi(argv[i + 1]);
    }

    QwenVisionConfig cfg; std::string err;
    if (!qwen_vision_config_from_hf_json(dir, cfg, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    if (!cfg.present) { printf("[FAIL] no vision_config\n"); return 1; }
    const int gran = cfg.size_granularity();
    const int P = cfg.patch_size, T = cfg.temporal_patch, C = cfg.in_channels;
    int gh, gw;
    if (!pix_path.empty()) {
        if (arg_gh <= 0 || arg_gw <= 0) { printf("[FAIL] --pixels needs --gh and --gw\n"); return 1; }
        gh = arg_gh; gw = arg_gw; h = gh * P; w = gw * P;
    } else {
        if (h % gran || w % gran) { printf("[FAIL] dims must be multiples of %d\n", gran); return 1; }
        gh = h / P; gw = w / P;
    }
    if (gh % cfg.spatial_merge || gw % cfg.spatial_merge) {
        printf("[FAIL] patch grid %dx%d not divisible by merge %d\n", gh, gw, cfg.spatial_merge);
        return 1;
    }
    const int N = gh * gw;
    const int patch_in = C * T * P * P;

    // The same deterministic ramp vision_ref.py builds, patchified the same way: row-major over
    // the grid, each patch (C, T, P, P) with the still image repeated across T.
    std::vector<float> pix((size_t)N * patch_in);
    if (!pix_path.empty()) {
        FILE* pf = fopen(pix_path.c_str(), "rb");
        if (!pf) { printf("[FAIL] cannot read %s\n", pix_path.c_str()); return 1; }
        fseek(pf, 0, SEEK_END); const long bytes = ftell(pf); fseek(pf, 0, SEEK_SET);
        if (bytes != (long)(pix.size() * sizeof(float))) {
            printf("[FAIL] %s holds %ld bytes, expected %zu for [%d,%d]\n",
                   pix_path.c_str(), bytes, pix.size() * sizeof(float), N, patch_in);
            fclose(pf); return 1;
        }
        if (fread(pix.data(), sizeof(float), pix.size(), pf) != pix.size()) {
            printf("[FAIL] short read from %s\n", pix_path.c_str()); fclose(pf); return 1;
        }
        fclose(pf);
        printf("pixels from %s: [%d, %d]\n", pix_path.c_str(), N, patch_in);
    } else
    for (int gy = 0; gy < gh; gy++)
      for (int gx = 0; gx < gw; gx++) {
        float* dst = &pix[((size_t)gy * gw + gx) * patch_in];
        for (int c = 0; c < C; c++)
          for (int t = 0; t < T; t++)
            for (int py = 0; py < P; py++)
              for (int px = 0; px < P; px++) {
                const int Y = gy * P + py, X = gx * P + px;
                float v = c == 0 ? (float)Y / (h - 1) : c == 1 ? (float)X / (w - 1)
                                                               : (float)(Y + X) / (h + w - 2);
                dst[((c * T + t) * P + py) * P + px] = v * 2.f - 1.f;
              }
      }

    SafeTensorsModel st;
    if (!st.open(dir)) { printf("[FAIL] cannot open safetensors in %s\n", dir.c_str()); return 1; }
    QwenVisionWeights vw;
    if (!load_qwen_vision_weights(st, cfg, vw, err)) { printf("[FAIL] load: %s\n", err.c_str()); return 1; }
    printf("loaded vision tower: %d blocks, hidden=%d, pos table %dx%d\n",
           (int)vw.blocks.size(), cfg.hidden, vw.pos_side, vw.pos_side);

    const int nblk = (gh / cfg.spatial_merge) * (gw / cfg.spatial_merge);
    std::vector<float> out((size_t)nblk * cfg.out_hidden);
    if (!qwen_vision_forward(vw, cfg, pix.data(), gh, gw, out.data(), err)) {
        printf("[FAIL] forward: %s\n", err.c_str()); free_qwen_vision_weights(vw); return 1;
    }
    free_qwen_vision_weights(vw);

    double mean = 0, absmax = 0;
    for (float v : out) { mean += v; absmax = std::fmax(absmax, std::fabs((double)v)); }
    mean /= out.size();
    double var = 0; for (float v : out) var += (v - mean) * (v - mean);
    printf("image %dx%d -> grid %dx%d = %d patches -> %d merged embeddings of %d\n",
           h, w, gh, gw, N, nblk, cfg.out_hidden);
    printf("out mean=%+.6f std=%.6f absmax=%.4f\n", mean, std::sqrt(var / out.size()), absmax);

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { printf("[FAIL] cannot write %s\n", out_path.c_str()); return 1; }
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    printf("wrote %zu floats -> %s\n", out.size(), out_path.c_str());
    printf("compare with: vision_ref.py <dir> --h %d --w %d --compare %s\n", h, w, out_path.c_str());
    return 0;
}
