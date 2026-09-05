// Image preprocessing for the vision tower: smart_resize -> bicubic -> normalize -> patchify.
#include "sparkinfer/models/qwen_vision_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sparkinfer {

namespace {

// PIL's BICUBIC kernel, a = -0.5. NOT torchvision's, which uses a = -0.75 -- the two libraries
// genuinely disagree on this constant, so "bicubic" alone does not pin the filter down. We match
// PIL, i.e. the slow Qwen2VLImageProcessor; see bench/scripts/vision_preprocess_ref.py for the
// measured cost of that choice against the fast (torchvision) processor the checkpoint declares.
//
// When DOWNscaling, the filter support is widened by the scale factor (antialiasing) -- without
// that, downsampling aliases badly and the resized image differs visibly from the reference, not
// just numerically. Verified against PIL to max|diff| 1e-4 by vision_preprocess_ref.py.
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

// Resize + rescale + normalize ONE frame into planar [C, rh, rw] float32.
//
// Split out of qwen_vision_preprocess so the video path reuses this byte-for-byte rather than
// growing a second copy of the reference-exact ordering below. That ordering is load-bearing and
// was measured, not guessed (see the comments inside): resize the uint8 VALUES, round-and-clamp
// the intermediate back to uint8 the way PIL does, and only then rescale and normalize. A video
// is just N frames through this same funnel, so any drift here would apply identically to images
// and video -- which is exactly the property that makes the image reference check cover both.
void normalize_frame(const unsigned char* rgb, int height, int width, int rh, int rw,
                     const QwenVisionConfig& cfg, std::vector<float>& out) {
    const int C = cfg.in_channels;
    std::vector<float> plane((size_t)C * height * width);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                plane[((size_t)c * height + y) * width + x] =
                    (float)rgb[((size_t)y * width + x) * C + c];

    std::vector<float> tmp((size_t)C * height * rw);
    out.assign((size_t)C * rh * rw, 0.0f);
    for (int c = 0; c < C; c++) {   // horizontal then vertical
        resample_axis(&plane[(size_t)c * height * width], width, rw, 1, 1,
                      height, width, rw, &tmp[(size_t)c * height * rw]);
        // The reference's INTERMEDIATE image is uint8 too: PIL resamples horizontally into an
        // 8-bit temp, then vertically out of it, so the round-and-clamp happens TWICE. Carrying
        // float through both passes and rounding once at the end is more accurate but drifts
        // from the reference by several LSB at edges, which is not ours to decide.
        float* t = &tmp[(size_t)c * height * rw];
        for (long i = 0; i < (long)height * rw; i++)
            t[i] = std::min(255.0f, std::max(0.0f, std::floor(t[i] + 0.5f)));
        resample_axis(t, height, rh, rw, rw, rw, 1, 1, &out[(size_t)c * rh * rw]);
    }
    for (int c = 0; c < C; c++) {
        const float mu = cfg.mean[c], sd = cfg.std_[c] != 0.f ? cfg.std_[c] : 1.f;
        float* p = &out[(size_t)c * rh * rw];
        for (long i = 0; i < (long)rh * rw; i++) {
            // The uint8 round-trip the reference performs, then rescale, then normalize.
            // floor(x+0.5), i.e. round-half-UP, because PIL's 8-bit resample accumulates in
            // fixed point and rounds with a half-LSB add -- not nearbyint's round-half-to-EVEN,
            // which disagrees on exact .5 ties. (smart_resize genuinely does want banker's
            // rounding; that is Python's round(), a different reference entirely.)
            const float q = std::min(255.0f, std::max(0.0f, std::floor(p[i] + 0.5f)));
            p[i] = (q * (1.0f / 255.0f) - mu) / sd;
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

    std::vector<float> out;
    normalize_frame(rgb, height, width, rh, rw, cfg, out);

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

bool qwen_vision_preprocess_video(const unsigned char* const* frames, int n_frames,
                                  int height, int width, const QwenVisionConfig& cfg,
                                  std::vector<float>& pixels,
                                  int* grid_t, int* grid_h, int* grid_w, std::string& err) {
    if (!frames || n_frames <= 0) { err = "preprocess video: no frames"; return false; }
    for (int i = 0; i < n_frames; i++)
        if (!frames[i]) { err = "preprocess video: null frame " + std::to_string(i); return false; }
    if (height <= 0 || width <= 0) { err = "preprocess video: empty frame"; return false; }

    const int C = cfg.in_channels, P = cfg.patch_size, T = cfg.temporal_patch;
    if (T <= 0) { err = "preprocess video: temporal_patch must be positive"; return false; }

    // ONE smart_resize for the whole clip, not per frame. The reference sizes a video from a
    // single (height, width) because every frame shares a grid -- a per-frame resize would let
    // two frames disagree on grid_h/grid_w, and the tower packs T frames into ONE patch, so a
    // mismatch there is not a quality question, it is a shape error at the Conv3d.
    int rh = 0, rw = 0;
    if (!qwen_vision_smart_resize(height, width, cfg.size_granularity(),
                                  cfg.min_pixels, cfg.max_pixels, &rh, &rw, err)) return false;
    const int gh = rh / P, gw = rw / P;

    // Pad the frame count up to a multiple of the temporal patch by REPEATING THE LAST FRAME.
    // This is the reference's own padding (Qwen3VLProcessor._calculate_timestamps does
    // `indices.extend(indices[-1] ...)`), not a convenience: a trailing partial group would
    // otherwise be dropped or zero-filled, and zero-fill is a black frame the model will happily
    // describe. Repeating the last frame is temporally meaningless but visually truthful.
    const int n_pad = (n_frames % T == 0) ? n_frames : n_frames + (T - n_frames % T);
    const int gt = n_pad / T;

    // Normalize each DISTINCT frame once. The padding repeats an already-normalized frame rather
    // than re-running the resize on it.
    std::vector<std::vector<float>> norm((size_t)n_frames);
    for (int i = 0; i < n_frames; i++)
        normalize_frame(frames[i], height, width, rh, rw, cfg, norm[(size_t)i]);

    // Patchify, group-major: [gt * gh * gw, C*T*P*P]. Within a group the layout is identical to
    // the image path's (C, T, P, P) -- the only difference is that T now carries REAL consecutive
    // frames instead of the same still repeated. That is the whole of "video" at this layer.
    pixels.assign((size_t)gt * gh * gw * C * T * P * P, 0.0f);
    for (int g = 0; g < gt; g++)
      for (int gy = 0; gy < gh; gy++)
        for (int gx = 0; gx < gw; gx++) {
          float* dst = &pixels[(((size_t)g * gh + gy) * gw + gx) * C * T * P * P];
          for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++) {
              const int fi = std::min(g * T + t, n_frames - 1);   // last-frame padding
              const float* src = norm[(size_t)fi].data();
              for (int py = 0; py < P; py++)
                for (int px = 0; px < P; px++)
                  dst[((c * T + t) * P + py) * P + px] =
                      src[((size_t)c * rh + (gy * P + py)) * rw + (gx * P + px)];
            }
        }
    *grid_t = gt; *grid_h = gh; *grid_w = gw;
    return true;
}

std::vector<float> qwen_vision_video_timestamps(const std::vector<int>& frame_indices,
                                                double fps, int temporal_patch) {
    std::vector<float> out;
    if (frame_indices.empty() || temporal_patch <= 0) return out;
    if (fps <= 0.0) fps = 24.0;   // the reference's own fallback when fps cannot be inferred

    // Pad indices to a whole number of temporal patches by repeating the last, then average
    // each patch's first and last frame time. Both steps are the reference's
    // (Qwen3VLProcessor._calculate_timestamps); the average is what makes a 2-frame patch report
    // the midpoint of the interval it actually covers rather than its leading edge.
    std::vector<int> idx = frame_indices;
    while (idx.size() % (size_t)temporal_patch) idx.push_back(idx.back());
    for (size_t i = 0; i < idx.size(); i += (size_t)temporal_patch) {
        const double a = idx[i] / fps, b = idx[i + (size_t)temporal_patch - 1] / fps;
        out.push_back((float)((a + b) / 2.0));
    }
    return out;
}

bool qwen_vision_mrope_positions(const std::vector<int>& token_ids,
                                 int image_token_id, int video_token_id,
                                 const std::vector<QwenVisionSpanGrid>& spans,
                                 int spatial_merge, int start_pos,
                                 std::vector<int>& out, std::string& err) {
    if (spatial_merge <= 0) { err = "mrope: spatial_merge must be positive"; return false; }
    out.assign(token_ids.size() * 3, 0);

    long pos = start_pos;
    size_t next_span = 0;
    size_t i = 0;
    while (i < token_ids.size()) {
        const int id = token_ids[i];
        if (id != image_token_id && id != video_token_id) {
            // Text: all three axes share one counter, which is plain 1D RoPE.
            out[i * 3 + 0] = (int)pos;
            out[i * 3 + 1] = (int)pos;
            out[i * 3 + 2] = (int)pos;
            pos++;
            i++;
            continue;
        }
        size_t j = i;
        while (j < token_ids.size() && token_ids[j] == id) j++;
        const size_t run = j - i;

        if (next_span >= spans.size()) {
            err = "mrope: prompt has more vision spans than grids were supplied";
            return false;
        }
        const QwenVisionSpanGrid& g = spans[next_span++];
        const int lt = g.grid_t;
        const int lh = g.grid_h / spatial_merge;
        const int lw = g.grid_w / spatial_merge;
        if (lt <= 0 || lh <= 0 || lw <= 0) {
            err = "mrope: vision span has a degenerate merged grid";
            return false;
        }
        if ((size_t)lt * lh * lw != run) {
            err = "mrope: vision span covers " + std::to_string(run) +
                  " tokens but its grid implies " + std::to_string((size_t)lt * lh * lw);
            return false;
        }

        // meshgrid(t, h, w) in ij order, flattened -- the reference's get_vision_position_ids.
        // The temporal index is NOT offset by start_pos twice: the reference adds start_position
        // to h and w up front and to t only after the time_interval multiply, which for
        // time_interval = 1 lands all three on the same base.
        size_t k = i;
        for (int t = 0; t < lt; t++)
            for (int h = 0; h < lh; h++)
                for (int w = 0; w < lw; w++, k++) {
                    out[k * 3 + 0] = (int)(pos + t);
                    out[k * 3 + 1] = (int)(pos + h);
                    out[k * 3 + 2] = (int)(pos + w);
                }

        // The advance that the 1D path gets wrong: max(h, w), not the token count.
        pos += (lh > lw ? lh : lw);
        i = j;
    }
    if (next_span != spans.size()) {
        err = "mrope: " + std::to_string(spans.size() - next_span) +
              " supplied grid(s) had no matching vision span in the prompt";
        return false;
    }
    return true;
}

bool qwen_vision_expand_video_placeholders(const std::vector<int>& in, int video_token_id,
                                           int vision_start_token_id, int vision_end_token_id,
                                           const std::vector<QwenVideoSpan>& videos,
                                           std::vector<int>& out, std::string& err) {
    size_t found = 0;
    for (int t : in) if (t == video_token_id) found++;
    if (found != videos.size()) {
        err = "video placeholder count mismatch: prompt has " + std::to_string(found) +
              " <|video_pad|>, caller preprocessed " + std::to_string(videos.size()) + " video(s)";
        return false;
    }
    for (size_t i = 0; i < videos.size(); i++) {
        if (videos[i].tokens_per_frame <= 0) {
            err = "video " + std::to_string(i) + ": tokens_per_frame must be positive";
            return false;
        }
        if (videos[i].timestamp_tokens.empty()) {
            err = "video " + std::to_string(i) + ": no temporal groups";
            return false;
        }
    }

    out.clear();
    out.reserve(in.size());
    size_t vi = 0;
    for (int t : in) {
        if (t != video_token_id) { out.push_back(t); continue; }
        const QwenVideoSpan& v = videos[vi++];
        for (const std::vector<int>& ts : v.timestamp_tokens) {
            out.insert(out.end(), ts.begin(), ts.end());
            out.push_back(vision_start_token_id);
            out.insert(out.end(), (size_t)v.tokens_per_frame, video_token_id);
            out.push_back(vision_end_token_id);
        }
    }
    return true;
}

bool qwen_vision_expand_placeholders(const std::vector<int>& in, int image_token_id,
                                     const std::vector<int>& counts,
                                     std::vector<int>& out, std::string& err) {
    size_t found = 0;
    for (int t : in) if (t == image_token_id) found++;
    if (found != counts.size()) {
        err = "expand: prompt has " + std::to_string(found) + " image placeholder(s) but "
            + std::to_string(counts.size()) + " image(s) were supplied";
        return false;
    }
    for (size_t i = 0; i < counts.size(); i++) {
        if (counts[i] <= 0) {
            err = "expand: image " + std::to_string(i) + " needs a positive token count";
            return false;
        }
    }
    size_t total = in.size();
    for (int c : counts) total += (size_t)c - 1;   // each placeholder becomes c tokens
    out.clear();
    out.reserve(total);
    size_t idx = 0;
    for (int t : in) {
        if (t == image_token_id) out.insert(out.end(), (size_t)counts[idx++], image_token_id);
        else out.push_back(t);
    }
    return true;
}

}  // namespace sparkinfer
