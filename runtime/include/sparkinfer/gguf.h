#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace sparkinfer {

struct GGUFTensor {
    int   ggml_type = 0;
    int   n_dims = 0;
    long  dims[4] = {1, 1, 1, 1};   // ggml ne order (dims[0] fastest)
    long  n_values = 0;
    long  n_bytes = 0;
    const void* data = nullptr;     // pointer into the mmap'd file
};

// Minimal read-only GGUF (v3) reader: mmaps the file, parses metadata + tensor
// table, exposes scalar metadata and tensor data pointers. Arrays (e.g. the
// tokenizer vocab) are skipped — tokenization is done in Python.
class GGUF {
public:
    ~GGUF();
    bool open(const std::string& path);

    long        meta_int(const std::string& key, long def = 0) const;
    double      meta_float(const std::string& key, double def = 0) const;
    std::string meta_str(const std::string& key, const std::string& def = "") const;
    const GGUFTensor* tensor(const std::string& name) const;

private:
#ifndef _WIN32
    int    fd_ = -1;
#else
    void*  win_file_ = (void*)(intptr_t)-1;
    void*  win_map_  = nullptr;
#endif
    void*  base_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<std::string, long>        ints_;
    std::unordered_map<std::string, double>      floats_;
    std::unordered_map<std::string, std::string> strs_;
    std::unordered_map<std::string, GGUFTensor>  tensors_;
};

} // namespace sparkinfer
