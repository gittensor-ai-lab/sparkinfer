#pragma once
#include <string>
#include <vector>
#include "sparkinfer/models/qwen_vision_config.h"

namespace sparkinfer {

// Qwen2VLImageProcessor's dynamic-resolution sizing, ported exactly from
// transformers/models/qwen2_vl/image_processing_qwen2_vl.py::smart_resize.
//
// Both sides end up divisible by `factor` (= patch_size * spatial_merge, 32 here) and the total
// pixel count lands in [min_pixels, max_pixels] while keeping the aspect ratio as close as
// possible. Integer logic with round/floor/ceil at specific points -- ported verbatim rather than
// re-derived, because an off-by-one changes the patch grid and therefore how many image tokens
// the prompt must carry.
//
// Returns false (and sets err) for an aspect ratio beyond 200:1, matching the reference's own
// refusal rather than silently producing a degenerate grid.
bool qwen_vision_smart_resize(int height, int width, int factor, long min_pixels, long max_pixels,
                              int* out_h, int* out_w, std::string& err);

// Full preprocess: RGB8 -> resized, normalized, patchified float32 ready for the tower.
//
//   rgb:      [height, width, 3] uint8, row-major
//   pixels:   [grid_h*grid_w, in_ch*temporal*patch*patch] float32, row-major over the patch grid,
//             each patch laid out (C, T, P, P) with the still image repeated across T.
//
// Row-major patch order is equivalent to HF's merge-block-major -- proved to 1e-15 by
// bench/scripts/vision_order_check.py -- and launch_vision_patch_merge does the regroup.
bool qwen_vision_preprocess(const unsigned char* rgb, int height, int width,
                            const QwenVisionConfig& cfg,
                            std::vector<float>& pixels, int* grid_h, int* grid_w,
                            std::string& err);

// Full preprocess for a VIDEO: N same-sized RGB8 frames -> resized, normalized, patchified.
//
// A video differs from an image in exactly one place at this layer: the tower's Conv3d has a time
// extent of temporal_patch (2), and an image fills it by repeating the same still, whereas a video
// fills it with REAL consecutive frames. Everything else -- smart_resize, the PIL-exact bicubic,
// the uint8 round-trip, the (C, T, P, P) patch layout -- is shared with the image path via one
// helper, so the two cannot drift apart.
//
//   frames:   n_frames pointers to [height, width, 3] uint8, row-major. All frames MUST share
//             height/width: one smart_resize is computed for the whole clip, because T frames are
//             packed into a single patch and a per-frame grid would not line up.
//   pixels:   [grid_t*grid_h*grid_w, in_ch*temporal*patch*patch] float32, group-major.
//   grid_t:   number of TEMPORAL GROUPS = ceil(n_frames / temporal_patch), not the frame count.
//
// A trailing partial group is padded by repeating the LAST frame, matching the reference. Zero
// fill would be a black frame, which the model describes as if it were real content.
bool qwen_vision_preprocess_video(const unsigned char* const* frames, int n_frames,
                                  int height, int width, const QwenVisionConfig& cfg,
                                  std::vector<float>& pixels,
                                  int* grid_t, int* grid_h, int* grid_w, std::string& err);

// Per-temporal-group timestamps, in seconds, for the "<N.N seconds>" markers the prompt carries
// before each frame's vision span.
//
// frame_indices are the ORIGINAL frame numbers that were sampled (not 0..n-1) -- that is what
// makes the timestamps reflect real elapsed time when frames are subsampled. Returns one entry
// per temporal group: the mean of the group's first and last frame time. fps <= 0 falls back to
// 24, which is the reference's behaviour when a container reports no frame rate.
std::vector<float> qwen_vision_video_timestamps(const std::vector<int>& frame_indices,
                                                double fps, int temporal_patch);

// How many image placeholder tokens a given grid needs: one per MERGED patch. The prompt must
// carry exactly this many image_token_id between vision_start and vision_end, or the splice
// silently misaligns the whole sequence.
int qwen_vision_num_tokens(int grid_h, int grid_w, const QwenVisionConfig& cfg);

// One video's expansion parameters, for qwen_vision_expand_video_placeholders below.
struct QwenVideoSpan {
    // Placeholder tokens ONE temporal group needs: (grid_h/merge) * (grid_w/merge). Every group
    // of a clip has the same count -- the whole clip shares one smart_resize.
    int tokens_per_frame = 0;
    // Tokenized "<N.N seconds>" marker, one entry per temporal group. Tokenized by the CALLER:
    // the runtime has no tokenizer, and re-tokenizing the rendered prompt to insert these would
    // risk shifting unrelated tokens through merge effects at the seams -- the same reason the
    // image expansion works in token space.
    std::vector<std::vector<int>> timestamp_tokens;
};

// Expands each single video placeholder into per-frame spans carrying timestamp markers.
//
// The chat template emits ONE <|video_pad|> per video, wrapped once:
//     "<|vision_start|><|video_pad|><|vision_end|>"
// and the reference PROCESSOR replaces ONLY the <|video_pad|> token -- not the wrapper -- with
//     for each temporal group g:  "<t_g seconds>" <|vision_start|> <|video_pad|>*n <|vision_end|>
// so the result is deliberately NESTED: the template's pair wraps the whole clip and every frame
// carries its own inner pair. That looks like a bug and is not; Qwen3.5 separates video frames
// with timestamps, and get_rope_index relies on each frame being its own vision span.
//
// The timestamps are not decoration. Without MRoPE's temporal axis they are the only signal that
// orders the frames, and even with it they carry real elapsed time across subsampled frames.
bool qwen_vision_expand_video_placeholders(const std::vector<int>& in, int video_token_id,
                                           int vision_start_token_id, int vision_end_token_id,
                                           const std::vector<QwenVideoSpan>& videos,
                                           std::vector<int>& out, std::string& err);

// One vision span's grid, for qwen_vision_mrope_positions below.
struct QwenVisionSpanGrid {
    int grid_t = 1;      // temporal groups; 1 for an image, and 1 PER GROUP for video
    int grid_h = 0;      // pre-merge patch grid
    int grid_w = 0;
};

// Builds the LM's 3D (t, h, w) position ids for a prompt containing vision spans.
//
// Qwen3.5's text model uses INTERLEAVED MRoPE (rope_parameters.mrope_section = [11,11,10],
// mrope_interleaved = true), so a token's rotary position is not one number but three, and the
// frequency index selects which axis it reads. Two consequences the 1D path gets wrong:
//
//   1. Vision tokens get (t, row, col) rather than a running counter, so the tower's spatial
//      layout reaches attention instead of being flattened into sequence order.
//   2. Text AFTER a vision span resumes at start + max(grid_h, grid_w)/merge, NOT at
//      start + <number of vision tokens>. A 32x32 image contributes 256 tokens but advances the
//      position counter by only 16.
//
// For TEXT tokens all three axes are equal, which is exactly plain 1D RoPE -- that degeneracy is
// why a text-only prompt is unaffected by any of this, and why enabling MRoPE cannot move
// text-only numbers.
//
// out is filled with 3*n_tokens ints laid out [t0,h0,w0, t1,h1,w1, ...]. spans must list every
// vision span in prompt order, one entry per MAXIMAL placeholder run (so one per image, and one
// per temporal group for video, matching how the prompt is expanded).
bool qwen_vision_mrope_positions(const std::vector<int>& token_ids,
                                 int image_token_id, int video_token_id,
                                 const std::vector<QwenVisionSpanGrid>& spans,
                                 int spatial_merge, int start_pos,
                                 std::vector<int>& out, std::string& err);

// Expands each single image placeholder into the number of tokens its image actually needs.
//
// The chat template emits exactly ONE <|image_pad|> per image:
//     "<|vision_start|><|image_pad|><|vision_end|>"
// and the reference PROCESSOR (not the template) expands it, because only the processor knows the
// resized grid:
//     num_image_tokens = image_grid_thw.prod() // merge_size**2
// which for a still image is (grid_h/merge)*(grid_w/merge) -- qwen_vision_num_tokens.
//
// Done in token space rather than by editing the rendered string: the template's output has
// already been tokenized, expanding text would force a re-tokenize, and a re-tokenize can shift
// unrelated tokens through merge effects at the seams.
//
// counts must have one entry per placeholder found, in order. A mismatch is an error rather than
// a best-effort expansion: it means the caller preprocessed a different number of images than the
// prompt refers to, and guessing which is right would produce a silently misaligned prompt.
bool qwen_vision_expand_placeholders(const std::vector<int>& in, int image_token_id,
                                     const std::vector<int>& counts,
                                     std::vector<int>& out, std::string& err);

}  // namespace sparkinfer
