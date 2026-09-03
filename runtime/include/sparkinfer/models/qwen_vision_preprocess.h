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

// How many image placeholder tokens a given grid needs: one per MERGED patch. The prompt must
// carry exactly this many image_token_id between vision_start and vision_end, or the splice
// silently misaligns the whole sequence.
int qwen_vision_num_tokens(int grid_h, int grid_w, const QwenVisionConfig& cfg);

}  // namespace sparkinfer
