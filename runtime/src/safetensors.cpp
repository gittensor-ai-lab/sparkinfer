// Minimal safetensors reader (mmap) + a small dependency-free JSON parser scoped to the
// safetensors header shape (a flat object of tensor-name -> {dtype, shape, data_offsets},
// plus an optional __metadata__ block) and the sibling model.safetensors.index.json shape
// (weight_map: tensor-name -> shard filename). No general JSON library is linked into the
// runtime target (that's server-only, see server/CMakeLists.txt) -- mirrors gguf.cpp's
// dependency-free, mmap-based, bounds-checked design.

#include "sparkinfer/safetensors.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <variant>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace sparkinfer {

namespace {

// ---- mmap helpers (identical shape to gguf.cpp's) ----

#ifdef _WIN32
bool map_readonly(const std::string& path, void*& base, size_t& size,
                  void*& file_handle, void*& map_handle) {
    file_handle = (void*)CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[safetensors] open failed: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx((HANDLE)file_handle, &li) || li.QuadPart <= 0) {
        fprintf(stderr, "[safetensors] stat failed: %s\n", path.c_str());
        CloseHandle((HANDLE)file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    size = (size_t)li.QuadPart;
    map_handle = (void*)CreateFileMappingA((HANDLE)file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_handle) {
        fprintf(stderr, "[safetensors] CreateFileMapping failed\n");
        CloseHandle((HANDLE)file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    base = MapViewOfFile((HANDLE)map_handle, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        fprintf(stderr, "[safetensors] MapViewOfFile failed\n");
        CloseHandle((HANDLE)map_handle);
        CloseHandle((HANDLE)file_handle);
        map_handle = nullptr;
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    return true;
}

void unmap_readonly(void* base, void* file_handle, void* map_handle) {
    if (base) UnmapViewOfFile(base);
    if (map_handle) CloseHandle((HANDLE)map_handle);
    if (file_handle && file_handle != INVALID_HANDLE_VALUE) CloseHandle((HANDLE)file_handle);
}
#else
bool map_readonly(const std::string& path, void*& base, size_t& size, int& fd) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[safetensors] open failed: %s\n", path.c_str()); return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "[safetensors] stat failed: %s\n", path.c_str());
        close(fd);
        fd = -1;
        return false;
    }
    size = (size_t)st.st_size;
    base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[safetensors] mmap failed\n");
        close(fd);
        fd = -1;
        base = nullptr;
        return false;
    }
    return true;
}

void unmap_readonly(void* base, size_t size, int fd) {
    if (base && base != MAP_FAILED) munmap(base, size);
    if (fd >= 0) close(fd);
}
#endif

// ---- tiny JSON value + parser, scoped to what safetensors headers actually contain ----

struct Json;
using JsonObject = std::map<std::string, Json>;
using JsonArray  = std::vector<Json>;

struct Json {
    std::variant<std::monostate, bool, double, std::string, JsonArray, JsonObject> v;

    bool is_object() const { return std::holds_alternative<JsonObject>(v); }
    bool is_array()  const { return std::holds_alternative<JsonArray>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_number() const { return std::holds_alternative<double>(v); }

    const JsonObject& obj() const { static JsonObject e; auto p = std::get_if<JsonObject>(&v); return p ? *p : e; }
    const JsonArray&  arr() const { static JsonArray e;  auto p = std::get_if<JsonArray>(&v);  return p ? *p : e; }
    std::string str(const std::string& def = "") const { auto p = std::get_if<std::string>(&v); return p ? *p : def; }
    double num(double def = 0) const { auto p = std::get_if<double>(&v); return p ? *p : def; }
};

struct JsonCursor {
    const char* p; size_t off, size; bool ok = true;

    void skip_ws() { while (off < size && (p[off]==' '||p[off]=='\t'||p[off]=='\n'||p[off]=='\r')) off++; }
    char peek() { return off < size ? p[off] : '\0'; }
    bool expect(char c) { skip_ws(); if (off>=size || p[off]!=c) { ok=false; return false; } off++; return true; }

    std::string parse_string() {
        skip_ws();
        if (!expect('"')) return {};
        std::string s;
        while (off < size && p[off] != '"') {
            char c = p[off++];
            if (c == '\\' && off < size) {
                char e = p[off++];
                switch (e) {
                    case '"': s += '"'; break; case '\\': s += '\\'; break; case '/': s += '/'; break;
                    case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break;
                    case 'b': s += '\b'; break; case 'f': s += '\f'; break;
                    case 'u': off += 4; break;   // \uXXXX: not needed for tensor/shard names, skip
                    default: s += e; break;
                }
            } else s += c;
        }
        if (off >= size) { ok = false; return {}; }
        off++;   // closing quote
        return s;
    }

    double parse_number() {
        skip_ws();
        size_t start = off;
        if (off < size && (p[off]=='-'||p[off]=='+')) off++;
        while (off < size && (isdigit((unsigned char)p[off]) || p[off]=='.' || p[off]=='e' || p[off]=='E'
                              || p[off]=='+' || p[off]=='-')) off++;
        if (off == start) { ok = false; return 0; }
        return strtod(std::string(p + start, off - start).c_str(), nullptr);
    }

    Json parse_value() {
        skip_ws();
        Json j;
        if (off >= size) { ok = false; return j; }
        char c = p[off];
        if (c == '"') { j.v = parse_string(); return j; }
        if (c == '{') { j.v = parse_object(); return j; }
        if (c == '[') { j.v = parse_array(); return j; }
        if (c == 't') { off += 4; j.v = true; return j; }     // "true"
        if (c == 'f') { off += 5; j.v = false; return j; }    // "false"
        if (c == 'n') { off += 4; j.v = std::monostate{}; return j; }  // "null"
        j.v = parse_number();
        return j;
    }

    JsonArray parse_array() {
        JsonArray a;
        if (!expect('[')) return a;
        skip_ws();
        if (peek() == ']') { off++; return a; }
        while (ok) {
            a.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { off++; continue; }
            break;
        }
        expect(']');
        return a;
    }

    JsonObject parse_object() {
        JsonObject o;
        if (!expect('{')) return o;
        skip_ws();
        if (peek() == '}') { off++; return o; }
        while (ok) {
            std::string key = parse_string();
            if (!ok) break;
            if (!expect(':')) break;
            o[key] = parse_value();
            skip_ws();
            if (peek() == ',') { off++; continue; }
            break;
        }
        expect('}');
        return o;
    }
};

bool parse_json(const char* data, size_t len, Json& out) {
    JsonCursor c{data, 0, len};
    out = c.parse_value();
    return c.ok;
}

STDType dtype_from_str(const std::string& s) {
    if (s == "F32")  return STDType::F32;
    if (s == "F16")  return STDType::F16;
    if (s == "BF16") return STDType::BF16;
    if (s == "I64")  return STDType::I64;
    if (s == "I32")  return STDType::I32;
    if (s == "I8")   return STDType::I8;
    if (s == "U8")   return STDType::U8;
    if (s == "F8_E4M3") return STDType::F8_E4M3;
    return STDType::Unknown;
}

int dtype_size(STDType t) {
    switch (t) {
        case STDType::F32: case STDType::I32: return 4;
        case STDType::F16: case STDType::BF16: return 2;
        case STDType::I64: return 8;
        case STDType::I8: case STDType::U8: case STDType::F8_E4M3: return 1;
        default: return 0;
    }
}

// Reads a whole small file into memory (used for model.safetensors.index.json, which is
// plain text, not the mmap'd tensor data).
bool read_whole_file(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return false; }
    out.resize((size_t)n);
    size_t rd = n > 0 ? fread(&out[0], 1, (size_t)n, f) : 0;
    fclose(f);
    return rd == (size_t)n;
}

bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

} // namespace

SafeTensorsShard::~SafeTensorsShard() {
#ifdef _WIN32
    unmap_readonly(base_, win_file_, win_map_);
#else
    unmap_readonly(base_, size_, fd_);
#endif
    base_ = nullptr;
    size_ = 0;
#ifdef _WIN32
    win_file_ = (void*)(intptr_t)-1;
    win_map_ = nullptr;
#else
    fd_ = -1;
#endif
}

bool SafeTensorsShard::open(const std::string& path) {
#ifdef _WIN32
    if (!map_readonly(path, base_, size_, win_file_, win_map_)) return false;
#else
    if (!map_readonly(path, base_, size_, fd_)) return false;
#endif
    if (size_ < 8) { fprintf(stderr, "[safetensors] %s too small for a header\n", path.c_str()); return false; }

    uint64_t hlen = 0;
    memcpy(&hlen, base_, 8);
    // File-controlled header length -- bounds-check before trusting it as a JSON span.
    if (hlen > size_ - 8) {
        fprintf(stderr, "[safetensors] %s: header length %llu exceeds file size\n",
                path.c_str(), (unsigned long long)hlen);
        return false;
    }

    Json header;
    if (!parse_json((const char*)base_ + 8, (size_t)hlen, header) || !header.is_object()) {
        fprintf(stderr, "[safetensors] %s: header JSON parse failed\n", path.c_str());
        return false;
    }

    const size_t data_start = 8 + (size_t)hlen;
    for (const auto& [name, val] : header.obj()) {
        if (name == "__metadata__" || !val.is_object()) continue;   // arbitrary string map, not a tensor
        const auto& info = val.obj();
        auto dt_it = info.find("dtype");
        auto sh_it = info.find("shape");
        auto off_it = info.find("data_offsets");
        if (dt_it == info.end() || sh_it == info.end() || off_it == info.end()
            || !sh_it->second.is_array() || !off_it->second.is_array()
            || off_it->second.arr().size() != 2) {
            fprintf(stderr, "[safetensors] %s: malformed tensor entry %s\n", path.c_str(), name.c_str());
            return false;
        }

        STTensor t;
        t.dtype = dtype_from_str(dt_it->second.str());
        const auto& shape = sh_it->second.arr();
        if (shape.size() > 5) {
            // 5 dims covers everything these checkpoints ship -- the widest is the vision tower's
            // Conv3d patch embed at [out, in_ch, t_patch, ph, pw]. Anything wider is unexpected;
            // skip cataloging it rather than failing the whole checkpoint, and say so loudly
            // (this used to fire on the 5-D patch embed, silently making image input impossible).
            fprintf(stderr, "[safetensors] %s: skipping tensor %s with %zu dims (max 5)\n",
                    path.c_str(), name.c_str(), shape.size());
            continue;
        }
        t.n_dims = (int)shape.size();
        long nv = 1;
        for (int d = 0; d < t.n_dims; d++) { long e = (long)shape[d].num(); t.dims[d] = e; nv *= e; }
        if (t.n_dims == 0) nv = 1;   // scalar tensor
        t.n_values = nv;

        const uint64_t start = (uint64_t)off_it->second.arr()[0].num();
        const uint64_t end   = (uint64_t)off_it->second.arr()[1].num();
        const int esz = dtype_size(t.dtype);
        if (esz == 0 || end < start) {
            fprintf(stderr, "[safetensors] %s: tensor %s has unsupported/invalid dtype or offsets\n",
                    path.c_str(), name.c_str());
            return false;
        }
        t.n_bytes = (long)(end - start);
        // Bounds-check the resolved data region against the actual file size, same
        // discipline as gguf.cpp -- a truncated/malformed shard must fail loudly here,
        // not produce an out-of-bounds pointer a later cudaMemcpy would dereference.
        if (data_start > size_ || start > size_ - data_start || end > size_ - data_start) {
            fprintf(stderr, "[safetensors] %s: tensor %s data out of bounds\n", path.c_str(), name.c_str());
            return false;
        }
        t.data = (const uint8_t*)base_ + data_start + start;
        tensors_[name] = t;
    }
    return true;
}

const STTensor* SafeTensorsShard::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool SafeTensorsModel::open(const std::string& model_dir) {
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    const std::string single_path = model_dir + "/model.safetensors";

    std::vector<std::string> shard_files;
    std::map<std::string, std::string> weight_map;   // tensor name -> shard filename

    if (file_exists(index_path)) {
        std::string text;
        if (!read_whole_file(index_path, text)) {
            fprintf(stderr, "[safetensors] failed to read %s\n", index_path.c_str());
            return false;
        }
        Json idx;
        if (!parse_json(text.data(), text.size(), idx) || !idx.is_object()) {
            fprintf(stderr, "[safetensors] %s: JSON parse failed\n", index_path.c_str());
            return false;
        }
        auto wm_it = idx.obj().find("weight_map");
        if (wm_it == idx.obj().end() || !wm_it->second.is_object()) {
            fprintf(stderr, "[safetensors] %s: missing weight_map\n", index_path.c_str());
            return false;
        }
        for (const auto& [tname, shard] : wm_it->second.obj()) {
            weight_map[tname] = shard.str();
        }
    } else if (file_exists(single_path)) {
        // Unsharded model -- every tensor lives in the one file; weight_map is populated
        // lazily below once the shard is opened and its own tensor list is known.
        shard_files.push_back(single_path);
    } else {
        fprintf(stderr, "[safetensors] neither %s nor %s exists\n", index_path.c_str(), single_path.c_str());
        return false;
    }

    // Collect the distinct shard filenames referenced by the index (sharded case).
    if (!weight_map.empty()) {
        std::map<std::string, int> seen;
        for (const auto& [tname, shard] : weight_map) {
            if (seen.find(shard) == seen.end()) {
                seen[shard] = (int)shard_files.size();
                shard_files.push_back(shard);
            }
        }
    }

    // A shard referenced by the index may be deliberately absent (e.g. model_mtp.safetensors for
    // a checkpoint whose MTP head is out of scope and never downloaded) -- skip it rather than
    // failing the whole model, since only tensors that actually resolve to an opened shard are
    // ever looked up. Filter to existing files first, then open in place (SafeTensorsShard has a
    // custom destructor, so no temporary may be copied/moved into the vector).
    std::vector<std::string> present_shard_files;
    std::map<std::string, int> opened_shard_index;   // shard filename -> index into shards_
    for (const auto& f : shard_files) {
        const std::string path = f.find('/') != std::string::npos ? f : (model_dir + "/" + f);
        if (!file_exists(path)) {
            fprintf(stderr, "[safetensors] %s: shard %s not present -- skipping (tensors mapped "
                            "to it will be unavailable)\n", model_dir.c_str(), f.c_str());
            continue;
        }
        opened_shard_index[f] = (int)present_shard_files.size();
        present_shard_files.push_back(f);
    }

    shards_.resize(present_shard_files.size());
    for (size_t i = 0; i < present_shard_files.size(); i++) {
        const std::string path = present_shard_files[i].find('/') != std::string::npos
            ? present_shard_files[i] : (model_dir + "/" + present_shard_files[i]);
        if (!shards_[i].open(path)) return false;
    }

    if (!weight_map.empty()) {
        for (const auto& [tname, shard] : weight_map) {
            auto sit = opened_shard_index.find(shard);
            if (sit != opened_shard_index.end()) shard_of_[tname] = sit->second;
        }
    } else {
        // Unsharded: every tensor the single shard actually parsed belongs to shard 0.
        for (const auto& [tname, t] : shards_[0].tensors()) { (void)t; shard_of_[tname] = 0; }
    }
    return true;
}

const STTensor* SafeTensorsModel::tensor(const std::string& name) const {
    auto it = shard_of_.find(name);
    if (it == shard_of_.end()) return nullptr;
    return shards_[it->second].tensor(name);
}

} // namespace sparkinfer
