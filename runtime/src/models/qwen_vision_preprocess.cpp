// Image preprocessing for the vision tower: smart_resize -> bicubic -> normalize -> patchify.
#include "sparkinfer/models/qwen_vision_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sparkinfer {

namespace {

// PIL/torchvision BICUBIC kernel, a = -0.5. When DOWNscaling, the filter support is widened by
// the scale factor (antialiasing) -- without that, downsampling aliases badly and the resized
// image differs visibly from the reference, not just numerically.
inline float cubic(float x, float a = -0.5f) {
    x = std::fabs(x);
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

// Separable bicubic resample of one channel plane, with antialias support scaling on downscale.
void resample_axis(const float* src, int src_len, int dst_len, int stride_src, int stride_dst,
                   int lines, int line_stride_src, int line_stride_dst, float* dst) {
    const float scale = (float)src_len / (float)dst_len;
    const float filter_scale = scale > 1.0f ? scale : 1.0f;   // widen only when shrinking
    const float support = 2.0f * filter_scale;
    for (int o = 0; o < dst_len; o++) {
        const float center = (o + 0.5f) * scale;
        const int lo = std::max(0, (int)std::floor(center - support + 0.5f));
        const int hi = std::min(src_len, (int)std::floor(center + support + 0.5f));
        float wsum = 0.0f;
        float wts[64]; int n = 0;
        for (int i = lo; i < hi && n < 64; i++, n++) {
            const float w = cubic(((i + 0.5f) - center) / filter_scale);
            wts[n] = w; wsum += w;
        }
        if (wsum == 0.0f) { wsum = 1.0f; if (n > 0) wts[0] = 1.0f; }
        for (int L = 0; L < lines; L++) {
            const float* s = src + (size_t)L * line_stride_src;
            float acc = 0.0f;
            for (int i = 0; i < n; i++) acc += wts[i] * s[(size_t)(lo + i) * stride_src];
            dst[(size_t)L * line_stride_dst + (size_t)o * stride_dst] = acc / wsum;
        }
    }
}

}  // namespace

bool qwen_vision_smart_resize(int height, int width, int factor, long min_pixels, long max_pixels,
                              int* out_h, int* out_w, std::string& err) {
    if (height <= 0 || width <= 0 || factor <= 0) { err = "smart_resize: non-positive input"; return false; }
    const int mx = std::max(height, width), mn = std::min(height, width);
    if ((double)mx / (double)mn > 200.0) {
        err = "smart_resize: absolute aspect ratio must be smaller than 200";
        return false;
    }
    // round(x / f) * f, with PYTHON's round -- banker's rounding, ties to even. std::lround
    // rounds ties away from zero and is WRONG here. This is not pedantry: at 16x2048,
    // round(16/32) = round(0.5) is 0 in Python and 1 with lround, so Python's h_bar*w_bar is 0
    // and falls into the scale-up branch, returning 2912 where lround returns 2048. Caught by
    // bench/scripts/vision_resize_check.py, which demands exact equality for exactly this reason.
    //
    // std::nearbyint honours the current rounding mode, which is round-to-nearest-EVEN by
    // default under IEEE 754 -- the same rule Python's round() applies.
    long h_bar = (long)std::nearbyint((double)height / factor) * factor;
    long w_bar = (long)std::nearbyint((double)width / factor) * factor;
    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(((double)height * width) / (double)max_pixels);
        h_bar = std::max((long)factor, (long)std::floor(height / beta / factor) * factor);
        w_bar = std::max((long)factor, (long)std::floor(width  / beta / factor) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt((double)min_pixels / ((double)height * width));
        h_bar = (long)std::ceil(height * beta / factor) * factor;
        w_bar = (long)std::ceil(width  * beta / factor) * factor;
    }
    if (h_bar <= 0 || w_bar <= 0) { err = "smart_resize: degenerate result"; return false; }
    *out_h = (int)h_bar; *out_w = (int)w_bar;
    return true;
}

int qwen_vision_num_tokens(int grid_h, int grid_w, const QwenVisionConfig& cfg) {
    const int m = cfg.spatial_merge;
    if (m <= 0 || grid_h % m || grid_w % m) return 0;
    return (grid_h / m) * (grid_w / m);
}

bool qwen_vision_preprocess(const unsigned char* rgb, int height, int width,
                            const QwenVisionConfig& cfg,
                            std::vector<float>& pixels, int* grid_h, int* grid_w,
                            std::string& err) {
    if (!rgb || height <= 0 || width <= 0) { err = "preprocess: empty image"; return false; }
    int rh = 0, rw = 0;
    if (!qwen_vision_smart_resize(height, width, cfg.size_granularity(),
                                  cfg.min_pixels, cfg.max_pixels, &rh, &rw, err)) return false;

    const int C = cfg.in_channels, P = cfg.patch_size, T = cfg.temporal_patch;
    const int gh = rh / P, gw = rw / P;

    // uint8 -> float in [0,1], planar, then resize each plane, then normalize with mean/std.
    // Order matters: the reference rescales BEFORE normalizing, and normalizes AFTER resizing.
    std::vector<float> plane((size_t)C * height * width);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                plane[((size_t)c * height + y) * width + x] =
                    rgb[((size_t)y * width + x) * C + c] * (1.0f / 255.0f);

    std::vector<float> tmp((size_t)C * height * rw), out((size_t)C * rh * rw);
    for (int c = 0; c < C; c++) {   // horizontal then vertical
        resample_axis(&plane[(size_t)c * height * width], width, rw, 1, 1,
                      height, width, rw, &tmp[(size_t)c * height * rw]);
        resample_axis(&tmp[(size_t)c * height * rw], height, rh, rw, rw,
                      rw, 1, 1, &out[(size_t)c * rh * rw]);
    }
    for (int c = 0; c < C; c++) {
        const float mu = cfg.mean[c], sd = cfg.std_[c] != 0.f ? cfg.std_[c] : 1.f;
        float* p = &out[(size_t)c * rh * rw];
        for (long i = 0; i < (long)rh * rw; i++) p[i] = (p[i] - mu) / sd;
    }

    // Patchify: row-major over the grid, each patch (C, T, P, P), still image repeated across T.
    pixels.assign((size_t)gh * gw * C * T * P * P, 0.0f);
    for (int gy = 0; gy < gh; gy++)
      for (int gx = 0; gx < gw; gx++) {
        float* dst = &pixels[((size_t)gy * gw + gx) * C * T * P * P];
        for (int c = 0; c < C; c++)
          for (int t = 0; t < T; t++)
            for (int py = 0; py < P; py++)
              for (int px = 0; px < P; px++)
                dst[((c * T + t) * P + py) * P + px] =
                    out[((size_t)c * rh + (gy * P + py)) * rw + (gx * P + px)];
      }
    *grid_h = gh; *grid_w = gw;
    return true;
}

}  // namespace sparkinfer
