#pragma once
#include <string>

namespace sparkinfer {

// Vision tower geometry for Qwen3.8-27B's `qwen3_5_vision` encoder, read from config.json's
// `vision_config` block. Defaults are the values the released ModelOpt checkpoint ships, so a
// checkpoint that omits a field still lands on something correct for that model rather than zero.
//
// The tower is UNQUANTIZED bf16 (666 tensors, ~1.1 GB) even in the NVFP4 checkpoint: the
// quantization_config's `ignore` list excludes `model.visual.*`. So it loads as plain bf16 and
// needs no NVFP4/FP8 path -- see load_compressed_tensors.
struct QwenVisionConfig {
    bool  present      = false;   // true once a vision_config block was actually found
    int   depth        = 27;      // transformer blocks in the tower
    int   hidden       = 1152;
    int   n_heads      = 16;      // head_dim = hidden / n_heads = 72
    int   intermediate = 4304;    // MLP inner width (linear_fc1 -> gelu -> linear_fc2)
    int   in_channels  = 3;
    int   patch_size   = 16;
    int   temporal_patch = 2;     // Conv3d kernel's time extent; a still image is duplicated to 2
    int   spatial_merge  = 2;     // 2x2 patches merged before projection -> merger input is
                                  // hidden * spatial_merge^2 = 4608
    int   num_pos_embeddings = 2304;   // learned table, 48x48 grid
    int   out_hidden   = 5120;    // merger output == the LM's hidden size, so merged patch
                                  // embeddings drop straight into the token embedding stream
    std::string hidden_act = "gelu_pytorch_tanh";

    // Special token ids that mark where image embeddings are spliced into the prompt. The tower
    // produces one embedding per MERGED patch, and exactly that many image_token_id placeholders
    // must appear between vision_start and vision_end.
    int vision_start_token_id = 248053;
    int vision_end_token_id   = 248054;
    int image_token_id        = 248056;
    int video_token_id        = 248057;

    // Preprocessing (preprocessor_config.json). Qwen2VLImageProcessorFast is dynamic-resolution:
    // it does not resize to fixed dimensions, it scales so the pixel COUNT lands in
    // [min_pixels, max_pixels] and both sides are multiples of patch_size * spatial_merge.
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_[3] = {0.5f, 0.5f, 0.5f};
    long  min_pixels = 65536;
    long  max_pixels = 16777216;

    int merged_patch_dim() const { return hidden * spatial_merge * spatial_merge; }
    int head_dim() const { return n_heads > 0 ? hidden / n_heads : 0; }
    // Both image dimensions must be a multiple of this, so the patch grid divides evenly into
    // the 2x2 merge blocks.
    int size_granularity() const { return patch_size * spatial_merge; }
};

}  // namespace sparkinfer
