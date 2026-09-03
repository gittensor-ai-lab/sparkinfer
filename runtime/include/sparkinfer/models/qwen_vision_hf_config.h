#pragma once
// Reads config.json's `vision_config` block (and preprocessor_config.json) into a
// QwenVisionConfig. Companion to qwen38_hf_config.h, kept separate so a text-only caller does not
// have to care that the tower exists.
//
// Absence is NOT an error: a checkpoint with no vision_config is a text-only checkpoint, and
// `present` stays false so callers can decide. Only a MALFORMED block is an error.
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "sparkinfer/models/qwen_vision_config.h"

inline bool qwen_vision_config_from_hf_json(const std::string& model_dir,
                                            sparkinfer::QwenVisionConfig& vc, std::string& err) {
    std::ifstream cf(model_dir + "/config.json");
    if (!cf) { err = "config.json not found in " + model_dir; return false; }
    nlohmann::json root;
    try { cf >> root; }
    catch (const std::exception& e) { err = std::string("config.json parse error: ") + e.what(); return false; }

    // The block sits at the top level on this checkpoint, but tolerate it nested under
    // text_config the way some exports place it.
    const nlohmann::json* v = nullptr;
    if (root.contains("vision_config") && root["vision_config"].is_object()) v = &root["vision_config"];
    else if (root.contains("text_config") && root["text_config"].is_object()
             && root["text_config"].contains("vision_config")
             && root["text_config"]["vision_config"].is_object()) v = &root["text_config"]["vision_config"];
    if (!v) { vc.present = false; return true; }   // text-only checkpoint

    const auto& j = *v;
    auto geti = [&](const char* k, int d) {
        return j.contains(k) && j[k].is_number_integer() ? j[k].get<int>() : d; };
    vc.present            = true;
    vc.depth              = geti("depth", vc.depth);
    vc.hidden             = geti("hidden_size", vc.hidden);
    vc.n_heads            = geti("num_heads", vc.n_heads);
    vc.intermediate       = geti("intermediate_size", vc.intermediate);
    vc.in_channels        = geti("in_channels", vc.in_channels);
    vc.patch_size         = geti("patch_size", vc.patch_size);
    vc.temporal_patch     = geti("temporal_patch_size", vc.temporal_patch);
    vc.spatial_merge      = geti("spatial_merge_size", vc.spatial_merge);
    vc.num_pos_embeddings = geti("num_position_embeddings", vc.num_pos_embeddings);
    vc.out_hidden         = geti("out_hidden_size", vc.out_hidden);
    if (j.contains("hidden_act") && j["hidden_act"].is_string())
        vc.hidden_act = j["hidden_act"].get<std::string>();

    // deepstack taps intermediate tower layers into the LM. This checkpoint ships an EMPTY list,
    // so there is nothing to wire -- but a non-empty one would silently change the contract, so
    // refuse rather than quietly producing wrong embeddings.
    if (j.contains("deepstack_visual_indexes") && j["deepstack_visual_indexes"].is_array()
        && !j["deepstack_visual_indexes"].empty()) {
        err = "vision_config.deepstack_visual_indexes is non-empty; deepstack is not implemented";
        return false;
    }

    auto gettok = [&](const char* k, int d) {
        return root.contains(k) && root[k].is_number_integer() ? root[k].get<int>() : d; };
    vc.vision_start_token_id = gettok("vision_start_token_id", vc.vision_start_token_id);
    vc.vision_end_token_id   = gettok("vision_end_token_id",   vc.vision_end_token_id);
    vc.image_token_id        = gettok("image_token_id",        vc.image_token_id);
    vc.video_token_id        = gettok("video_token_id",        vc.video_token_id);

    // Preprocessing. Optional: the defaults already match this checkpoint, and a missing file
    // should not stop the tower loading.
    std::ifstream pf(model_dir + "/preprocessor_config.json");
    if (pf) {
        nlohmann::json p;
        try {
            pf >> p;
            if (p.contains("image_mean") && p["image_mean"].is_array() && p["image_mean"].size() == 3)
                for (int i = 0; i < 3; i++) vc.mean[i] = p["image_mean"][i].get<float>();
            if (p.contains("image_std") && p["image_std"].is_array() && p["image_std"].size() == 3)
                for (int i = 0; i < 3; i++) vc.std_[i] = p["image_std"][i].get<float>();
            // "size" is a pixel BUDGET here, not width/height: shortest_edge/longest_edge are
            // min/max total pixels for Qwen2VLImageProcessorFast's dynamic resolution.
            if (p.contains("size") && p["size"].is_object()) {
                const auto& sz = p["size"];
                if (sz.contains("shortest_edge") && sz["shortest_edge"].is_number_integer())
                    vc.min_pixels = sz["shortest_edge"].get<long>();
                if (sz.contains("longest_edge") && sz["longest_edge"].is_number_integer())
                    vc.max_pixels = sz["longest_edge"].get<long>();
            }
        } catch (const std::exception&) { /* defaults already correct for this checkpoint */ }
    }

    if (vc.hidden <= 0 || vc.n_heads <= 0 || vc.hidden % vc.n_heads != 0) {
        err = "vision_config: hidden_size must be a positive multiple of num_heads";
        return false;
    }
    if (vc.spatial_merge <= 0 || vc.patch_size <= 0) {
        err = "vision_config: patch_size and spatial_merge_size must be positive";
        return false;
    }
    return true;
}
