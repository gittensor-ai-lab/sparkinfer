#pragma once

// Shared "what is this -m path, and how do I configure a Qwen35Config for it" helper for the
// example/benchmark binaries.
//
// Three checkpoint shapes ship today:
//   *.gguf                     -> config from GGUF metadata      -> Qwen35Model::load_gguf
//   dir with quantization_config in config.json
//                              -> config from HF config.json     -> load_compressed_tensors
//                                 (an HF quantized checkout; the loader routes each tensor by
//                                  the bytes present, so this covers both llm-compressor's
//                                  "compressed-tensors" mixed FP8/NVFP4 -- e.g.
//                                  unsloth/Qwen3.8-27B-NVFP4 -- and NVIDIA ModelOpt's uniform
//                                  NVFP4 -- e.g. gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090)
//   dir with a legacy config.txt
//                              -> flat key=value + .bin blobs    -> load_weights
//                                 (runtime/tools/convert_qwen35.py output)
//
// This mirrors server/src/model_engine.cpp's own dispatch deliberately: the PR eval bot benchmarks
// and teacher-forced-scores the SAME checkpoint the server serves, so a benchmark that configured
// the model even slightly differently would be measuring something nobody ships. Keeping the
// detection in one header means the two cannot drift.
//
// Requires nlohmann_json, which is why this lives in runtime/examples/ (example EXECUTABLES link
// it; the sparkinfer_runtime library still does not -- see qwen38_hf_config.h's own comment).

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen_config.h"
#include "qwen3_gguf_config.h"
#include "qwen38_hf_config.h"

#include <sys/stat.h>

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

enum class QwenCheckpointKind {
    Gguf,               // *.gguf
    CompressedTensors,  // HF dir, config.json has quantization_config (NVFP4/FP8)
    LegacyWeightDir,    // dir with config.txt + .bin blobs
};

// Detects the checkpoint shape at `path` and populates `cfg`. `gguf_out` is opened only for the
// Gguf kind (the caller keeps it alive for load_gguf). Returns false with `err` set on any
// malformed input -- never guesses, since a silently-wrong config reads as a plausible-but-wrong
// benchmark number rather than a crash.
inline bool qwen_checkpoint_open(const std::string& path,
                                 sparkinfer::Qwen35Config& cfg,
                                 sparkinfer::GGUF& gguf_out,
                                 QwenCheckpointKind& kind_out,
                                 std::string& err) {
    struct stat st {};
    const bool stat_ok = stat(path.c_str(), &st) == 0;
#ifdef _WIN32
    const bool is_dir = stat_ok && (st.st_mode & _S_IFDIR) != 0;
#else
    const bool is_dir = stat_ok && S_ISDIR(st.st_mode);
#endif

    if (!is_dir) {
        kind_out = QwenCheckpointKind::Gguf;
        if (!gguf_out.open(path)) { err = "cannot open GGUF " + path; return false; }
        qwen3_config_from_gguf(gguf_out, cfg);
        return true;
    }

    // A legacy convert_qwen35.py directory is identified by config.txt; anything else with a
    // config.json is an HF checkout. Check config.txt FIRST so an HF checkout that happens to
    // carry both is still read as HF (config.json is the richer, authoritative one).
    {
        std::ifstream jf(path + "/config.json");
        if (jf) {
            nlohmann::json root;
            try {
                jf >> root;
            } catch (const std::exception& e) {
                err = path + "/config.json parse error: " + e.what();
                return false;
            }
            if (!root.contains("quantization_config")) {
                err = path + " is a plain (unquantized) safetensors checkout -- not supported by "
                             "this tool; only compressed-tensors (NVFP4/FP8) directories are";
                return false;
            }
            kind_out = QwenCheckpointKind::CompressedTensors;
            return qwen38_config_from_hf_json(path, cfg, err);
        }
    }

    std::ifstream tf(path + "/config.txt");
    if (!tf) {
        err = path + " is a directory but has neither config.json nor config.txt";
        return false;
    }
    kind_out = QwenCheckpointKind::LegacyWeightDir;
    return true;   // caller parses config.txt itself (legacy, tool-specific key set)
}

// Loads weights for a checkpoint already described by `kind`.
inline bool qwen_checkpoint_load(sparkinfer::Qwen35Model& model,
                                 const std::string& path,
                                 QwenCheckpointKind kind) {
    switch (kind) {
        case QwenCheckpointKind::Gguf:              return model.load_gguf(path);
        case QwenCheckpointKind::CompressedTensors: return model.load_compressed_tensors(path);
        case QwenCheckpointKind::LegacyWeightDir:   return model.load_weights(path);
    }
    return false;
}

inline const char* qwen_checkpoint_kind_label(QwenCheckpointKind kind) {
    switch (kind) {
        case QwenCheckpointKind::Gguf:              return "native GGUF, experts quantized";
        case QwenCheckpointKind::CompressedTensors: return "HF quantized (NVFP4/FP8)";
        case QwenCheckpointKind::LegacyWeightDir:   return "bf16";
    }
    return "?";
}
