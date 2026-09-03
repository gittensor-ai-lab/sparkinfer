#pragma once
#include <string>
#include <vector>
#include "sparkinfer/models/qwen_vision_config.h"

namespace sparkinfer {

class SafeTensorsModel;

// Device-resident vision tower. All bf16: the checkpoint's quantization_config ignores
// model.visual.*, so there is no NVFP4/FP8 path to mirror here -- it loads as shipped.
struct QwenVisionBlockWeights {
    const void* norm1_w = nullptr; const void* norm1_b = nullptr;
    const void* qkv_w   = nullptr; const void* qkv_b   = nullptr;   // [3H, H], [3H]
    const void* proj_w  = nullptr; const void* proj_b  = nullptr;   // [H, H],  [H]
    const void* norm2_w = nullptr; const void* norm2_b = nullptr;
    const void* fc1_w   = nullptr; const void* fc1_b   = nullptr;   // [I, H],  [I]
    const void* fc2_w   = nullptr; const void* fc2_b   = nullptr;   // [H, I],  [H]
};

struct QwenVisionWeights {
    const void* patch_w = nullptr;   // [H, C*T*P*P] -- the Conv3d flattened to a dense matrix
    const void* patch_b = nullptr;   // [H]
    std::vector<QwenVisionBlockWeights> blocks;
    const void* merger_norm_w = nullptr; const void* merger_norm_b = nullptr;
    const void* merger_fc1_w  = nullptr; const void* merger_fc1_b  = nullptr;  // [4H_m, 4H_m]
    const void* merger_fc2_w  = nullptr; const void* merger_fc2_b  = nullptr;  // [out, 4H_m]
    // Host copy of the learned position table, [num_pos, H]. Kept on the HOST because the
    // bilinear resample to an arbitrary patch grid is done host-side and uploaded per image --
    // it is a few hundred KB and depends on the image's grid, so precomputing it there keeps the
    // device path free of an interpolation kernel and matches vision_ref.py exactly.
    std::vector<float> pos_table;
    int pos_side = 0;                // sqrt(num_pos); the table is a pos_side x pos_side grid
    std::vector<void*> owned;        // every cudaMalloc above, for teardown
};

// Loads model.visual.* from an already-open checkpoint onto the device.
bool load_qwen_vision_weights(SafeTensorsModel& st, const QwenVisionConfig& cfg,
                              QwenVisionWeights& w, std::string& err);
void free_qwen_vision_weights(QwenVisionWeights& w);

// Runs the tower on one image's patches.
//   pixels_host: [grid_h*grid_w, C*T*P*P] float32, ROW-MAJOR over the patch grid, each patch laid
//                out (C, T, P, P). Row-major is equivalent to HF's merge-block-major ordering --
//                proved to 1e-15 by bench/scripts/vision_order_check.py -- and the merge kernel
//                does the regroup that makes it so.
//   out_host:    [(grid_h/merge)*(grid_w/merge), out_hidden] float32, ready to splice into the
//                token embedding stream at image_token_id positions.
bool qwen_vision_forward(const QwenVisionWeights& w, const QwenVisionConfig& cfg,
                         const float* pixels_host, int grid_h, int grid_w,
                         float* out_host, std::string& err);

}  // namespace sparkinfer
