// Video preprocessing — pure C++, no GPU.
//
// The load-bearing property here is an EQUIVALENCE, not a tolerance: the tower's Conv3d has a
// time extent of temporal_patch, an image fills it by repeating one still, and a video fills it
// with real frames. So a 2-frame video whose frames are the SAME still must produce byte-identical
// pixels to that still preprocessed as an image. If that ever stops holding, the video path has
// drifted away from the image path's reference-exactness (resize kernel, uint8 round-trip, patch
// layout) and every video number quietly stops meaning what the image checks certified.
#include "sparkinfer/models/qwen_vision_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); failures++; }
    else       std::printf("  ok:   %s\n", what.c_str());
}

// Deterministic, high-contrast synthetic frame. Edges matter: the reference divergence the image
// path was fixed for showed up at high-contrast edges and nowhere else, so a flat test image
// would pass even with the resize ordering wrong.
static std::vector<unsigned char> make_frame(int h, int w, int seed) {
    std::vector<unsigned char> px((size_t)h * w * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const int v = ((x / 5 + y / 7 + seed) % 2) ? 250 : 5;
            px[((size_t)y * w + x) * 3 + 0] = (unsigned char)v;
            px[((size_t)y * w + x) * 3 + 1] = (unsigned char)((v + 60 * seed) % 256);
            px[((size_t)y * w + x) * 3 + 2] = (unsigned char)(255 - v);
        }
    return px;
}

int main() {
    QwenVisionConfig cfg;   // defaults are the released checkpoint's
    std::string err;
    const int H = 140, W = 196;

    std::printf("temporal_patch=%d spatial_merge=%d patch=%d\n",
                cfg.temporal_patch, cfg.spatial_merge, cfg.patch_size);

    // --- 1. an image IS a 2-frame video of the same still -------------------------------------
    {
        auto f = make_frame(H, W, 0);
        std::vector<float> img, vid;
        int igh = 0, igw = 0, gt = 0, vgh = 0, vgw = 0;
        const bool a = qwen_vision_preprocess(f.data(), H, W, cfg, img, &igh, &igw, err);
        check(a, "image preprocess succeeds" + (a ? "" : " -- " + err));

        const unsigned char* frames[2] = { f.data(), f.data() };
        const bool b = qwen_vision_preprocess_video(frames, 2, H, W, cfg, vid, &gt, &vgh, &vgw, err);
        check(b, "video preprocess succeeds" + (b ? "" : " -- " + err));

        check(gt == 1, "two frames at temporal_patch=2 make ONE temporal group");
        check(igh == vgh && igw == vgw, "video grid matches image grid");
        check(img.size() == vid.size(), "video pixel buffer matches image size");

        bool identical = img.size() == vid.size();
        double worst = 0.0;
        for (size_t i = 0; i < img.size() && i < vid.size(); i++)
            worst = std::max(worst, (double)std::fabs(img[i] - vid[i]));
        identical = identical && worst == 0.0;
        std::printf("  max|image - video(same still x2)| = %.17g\n", worst);
        check(identical, "duplicated-still video is BYTE-IDENTICAL to the image path");
    }

    // --- 2. distinct frames actually reach distinct slots in the time axis ---------------------
    // Guards the failure where the packing loop reads frame 0 for every t: the buffer would still
    // be the right shape and the equivalence above would still pass, so it needs its own check.
    {
        auto f0 = make_frame(H, W, 0), f1 = make_frame(H, W, 1);
        const unsigned char* frames[2] = { f0.data(), f1.data() };
        std::vector<float> vid;
        int gt = 0, gh = 0, gw = 0;
        check(qwen_vision_preprocess_video(frames, 2, H, W, cfg, vid, &gt, &gh, &gw, err),
              "two DISTINCT frames preprocess");

        const int C = cfg.in_channels, P = cfg.patch_size, T = cfg.temporal_patch;
        bool any_diff = false;
        for (int c = 0; c < C && !any_diff; c++)
            for (int py = 0; py < P && !any_diff; py++)
                for (int px = 0; px < P && !any_diff; px++) {
                    const float t0 = vid[(size_t)((c * T + 0) * P + py) * P + px];
                    const float t1 = vid[(size_t)((c * T + 1) * P + py) * P + px];
                    if (t0 != t1) any_diff = true;
                }
        check(any_diff, "distinct frames land in distinct temporal slots (t=0 != t=1)");
    }

    // --- 3. odd frame counts pad by repeating the LAST frame, not with zeros -------------------
    {
        auto f0 = make_frame(H, W, 0), f1 = make_frame(H, W, 1), f2 = make_frame(H, W, 2);
        const unsigned char* three[3] = { f0.data(), f1.data(), f2.data() };
        std::vector<float> vid;
        int gt = 0, gh = 0, gw = 0;
        check(qwen_vision_preprocess_video(three, 3, H, W, cfg, vid, &gt, &gh, &gw, err),
              "three frames preprocess");
        check(gt == 2, "3 frames at temporal_patch=2 make TWO groups (padded)");

        // Group 1 holds frame 2 at t=0 and the PADDING at t=1. Padding repeats frame 2, so the
        // two slots must be equal -- and must not be the all-zero a black frame would give.
        const int C = cfg.in_channels, P = cfg.patch_size, T = cfg.temporal_patch;
        const size_t g1 = (size_t)1 * gh * gw * C * T * P * P;
        bool pad_repeats = true, pad_nonzero = false;
        for (int c = 0; c < C; c++)
            for (int py = 0; py < P; py++)
                for (int px = 0; px < P; px++) {
                    const float t0 = vid[g1 + (size_t)((c * T + 0) * P + py) * P + px];
                    const float t1 = vid[g1 + (size_t)((c * T + 1) * P + py) * P + px];
                    if (t0 != t1) pad_repeats = false;
                    if (t1 != 0.0f) pad_nonzero = true;
                }
        check(pad_repeats, "padded slot repeats the last real frame");
        check(pad_nonzero, "padded slot is NOT a black frame");
    }

    // --- 4. timestamps: pairwise-averaged, reference formula ------------------------------------
    {
        // 4 frames sampled at original indices 0,12,24,36 from 24fps footage -> times
        // 0.0,0.5,1.0,1.5 -> groups average to 0.25 and 1.25.
        auto ts = qwen_vision_video_timestamps({0, 12, 24, 36}, 24.0, 2);
        check(ts.size() == 2, "4 frames -> 2 group timestamps");
        check(ts.size() == 2 && std::fabs(ts[0] - 0.25f) < 1e-6f, "group 0 timestamp is the midpoint 0.25s");
        check(ts.size() == 2 && std::fabs(ts[1] - 1.25f) < 1e-6f, "group 1 timestamp is the midpoint 1.25s");

        // Odd count pads by repeating the last INDEX, so the final group is a degenerate interval.
        auto odd = qwen_vision_video_timestamps({0, 24, 48}, 24.0, 2);
        check(odd.size() == 2, "3 frames -> 2 group timestamps");
        check(odd.size() == 2 && std::fabs(odd[1] - 2.0f) < 1e-6f, "padded final group repeats its own time");

        // fps <= 0 must fall back to 24 rather than dividing by zero.
        auto fb = qwen_vision_video_timestamps({0, 24}, 0.0, 2);
        check(fb.size() == 1 && std::isfinite(fb[0]), "fps=0 falls back rather than producing inf/nan");
        check(fb.size() == 1 && std::fabs(fb[0] - 0.5f) < 1e-6f, "fps=0 fallback uses 24fps");
    }

    // --- 5. rejects the inputs that would silently corrupt a prompt -----------------------------
    {
        auto f = make_frame(H, W, 0);
        const unsigned char* one[1] = { f.data() };
        std::vector<float> vid; int gt = 0, gh = 0, gw = 0;
        check(!qwen_vision_preprocess_video(nullptr, 1, H, W, cfg, vid, &gt, &gh, &gw, err),
              "null frame array rejected");
        check(!qwen_vision_preprocess_video(one, 0, H, W, cfg, vid, &gt, &gh, &gw, err),
              "zero frames rejected");
        const unsigned char* withnull[2] = { f.data(), nullptr };
        check(!qwen_vision_preprocess_video(withnull, 2, H, W, cfg, vid, &gt, &gh, &gw, err),
              "null frame pointer rejected");
    }

    // --- 6. placeholder expansion: nested spans, timestamps, exact token budget --------------
    {
        const int VS = cfg.vision_start_token_id, VE = cfg.vision_end_token_id;
        const int VP = cfg.video_token_id;
        // What the chat template produces: "Video 1: <|vision_start|><|video_pad|><|vision_end|>"
        const std::vector<int> in = { 700, 701, VS, VP, VE, 702 };

        QwenVideoSpan v;
        v.tokens_per_frame = 4;                       // a 4x4 patch grid merged 2x2
        v.timestamp_tokens = { {900, 901}, {902, 903} };   // two temporal groups
        std::vector<int> out;
        err.clear();   // else a previous section's message trails into this one's label
        check(qwen_vision_expand_video_placeholders(in, VP, VS, VE, {v}, out, err),
              "video expansion succeeds" + (err.empty() ? "" : " -- " + err));

        // Per group: 2 timestamp tokens + vision_start + 4 pads + vision_end = 8. Two groups = 16.
        // Plus the 5 non-placeholder tokens the template contributed.
        check(out.size() == 5 + 16, "expanded length is template tokens + per-group spans");

        int pads = 0, starts = 0, ends = 0;
        for (int t : out) { if (t == VP) pads++; if (t == VS) starts++; if (t == VE) ends++; }
        check(pads == 8, "one pad per merged patch per group (2 groups x 4)");
        // The template's outer pair PLUS one inner pair per group -- the nesting is intentional.
        check(starts == 3 && ends == 3, "outer template pair plus one inner pair per frame");

        const std::vector<int> want = {
            700, 701, VS,
            900, 901, VS, VP, VP, VP, VP, VE,
            902, 903, VS, VP, VP, VP, VP, VE,
            VE, 702 };
        check(out == want, "expansion matches the reference layout exactly");
    }

    // --- 7. expansion refuses the mismatches that would misalign a prompt ----------------------
    {
        const int VS = cfg.vision_start_token_id, VE = cfg.vision_end_token_id, VP = cfg.video_token_id;
        std::vector<int> out;
        QwenVideoSpan v; v.tokens_per_frame = 4; v.timestamp_tokens = { {900} };

        // Two placeholders, one video preprocessed: silently expanding one would leave a bare
        // <|video_pad|> the splice then has no embedding for.
        check(!qwen_vision_expand_video_placeholders({VS, VP, VE, VS, VP, VE}, VP, VS, VE, {v}, out, err),
              "placeholder/video count mismatch rejected");

        QwenVideoSpan bad; bad.tokens_per_frame = 0; bad.timestamp_tokens = { {900} };
        check(!qwen_vision_expand_video_placeholders({VS, VP, VE}, VP, VS, VE, {bad}, out, err),
              "zero tokens_per_frame rejected");

        QwenVideoSpan empty; empty.tokens_per_frame = 4;
        check(!qwen_vision_expand_video_placeholders({VS, VP, VE}, VP, VS, VE, {empty}, out, err),
              "video with no temporal groups rejected");
    }

    std::printf(failures ? "\nFAILED (%d)\n" : "\nPASSED\n", failures);
    return failures ? 1 : 0;
}
