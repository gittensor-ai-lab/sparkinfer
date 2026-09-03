#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

// safetensors dtype tags, as they appear in the format's JSON header.
enum class STDType { F32, F16, BF16, I64, I32, I8, U8, F8_E4M3, Unknown };

struct STTensor {
    STDType dtype = STDType::Unknown;
    int     n_dims = 0;
    // 5 slots, not 4: Qwen3.8's vision tower carries a 5-D Conv3d patch-embed weight
    // (model.visual.patch_embed.proj.weight, [out, in_ch, t_patch, ph, pw]). A 4-wide array made
    // that tensor uncatalogable, so the reader skipped it and image input was structurally
    // impossible regardless of what else was implemented.
    long    dims[5] = {1, 1, 1, 1, 1};   // row-major, dims[0] slowest (matches safetensors' own order)
    long    n_values = 0;
    long    n_bytes = 0;
    const void* data = nullptr;       // pointer into the owning shard's mmap
};

// One safetensors shard: mmaps the file, parses the flat JSON header (tensor name ->
// {dtype, shape, data_offsets}), resolves tensor data pointers. No nesting beyond one
// level in a safetensors header, so this is a small hand-written parser rather than a
// general JSON library -- mirrors gguf.h/gguf.cpp's dependency-free, mmap-based design
// (the runtime library links no JSON dependency; that's server-only).
class SafeTensorsShard {
public:
    ~SafeTensorsShard();
    bool open(const std::string& path);

    const STTensor* tensor(const std::string& name) const;
    const std::unordered_map<std::string, STTensor>& tensors() const { return tensors_; }

private:
#ifndef _WIN32
    int    fd_ = -1;
#else
    void*  win_file_ = (void*)(intptr_t)-1;
    void*  win_map_  = nullptr;
#endif
    void*  base_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<std::string, STTensor> tensors_;
};

// A full (possibly multi-shard) safetensors model directory: reads
// model.safetensors.index.json (or a single model.safetensors if unsharded) to learn which
// shard each tensor lives in, opens every referenced shard, and exposes a single flat
// tensor(name) lookup across all of them.
class SafeTensorsModel {
public:
    bool open(const std::string& model_dir);
    const STTensor* tensor(const std::string& name) const;

private:
    std::vector<SafeTensorsShard> shards_;
    std::unordered_map<std::string, int> shard_of_;   // tensor name -> index into shards_
};

} // namespace sparkinfer
