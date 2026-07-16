// Minimal GGUF v3 reader (mmap). Parses header, metadata KV (scalars captured,
// arrays skipped), and the tensor table; resolves tensor data pointers.

#include "sparkinfer/gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>

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
// ggml value types
enum { VT_U8=0, VT_I8=1, VT_U16=2, VT_I16=3, VT_U32=4, VT_I32=5, VT_F32=6,
       VT_BOOL=7, VT_STR=8, VT_ARR=9, VT_U64=10, VT_I64=11, VT_F64=12 };

int scalar_size(uint32_t t) {
    switch (t) { case VT_U8: case VT_I8: case VT_BOOL: return 1;
        case VT_U16: case VT_I16: return 2; case VT_U32: case VT_I32: case VT_F32: return 4;
        case VT_U64: case VT_I64: case VT_F64: return 8; default: return 0; }
}

struct Cursor {
    const uint8_t* p; size_t off, size; bool ok = true;
    template <class T> T rd() { T v{}; if (off + sizeof(T) > size) { ok=false; return v; } memcpy(&v, p+off, sizeof(T)); off += sizeof(T); return v; }
    std::string rd_str() { uint64_t n = rd<uint64_t>(); if (!ok || off+n>size) { ok=false; return {}; } std::string s((const char*)(p+off), n); off += n; return s; }
    void skip(size_t n) { off += n; if (off > size) ok = false; }
};

// block (bytes, elements) per ggml type for n_bytes computation
void block_info(int t, long& bytes, long& elems) {
    switch (t) {
        case 0:  bytes=4;   elems=1;   break;   // F32
        case 1:  bytes=2;   elems=1;   break;   // F16
        case 8:  bytes=34;  elems=32;  break;   // Q8_0
        case 12: bytes=144; elems=256; break;   // Q4_K
        case 13: bytes=176; elems=256; break;   // Q5_K
        case 14: bytes=210; elems=256; break;   // Q6_K
        default: bytes=0;   elems=1;   break;
    }
}

#ifdef _WIN32
bool map_readonly(const std::string& path, void*& base, size_t& size,
                  void*& file_handle, void*& map_handle) {
    file_handle = (void*)CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[gguf] open failed: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx((HANDLE)file_handle, &li) || li.QuadPart <= 0) {
        fprintf(stderr, "[gguf] stat failed: %s\n", path.c_str());
        CloseHandle((HANDLE)file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    size = (size_t)li.QuadPart;
    map_handle = (void*)CreateFileMappingA((HANDLE)file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!map_handle) {
        fprintf(stderr, "[gguf] CreateFileMapping failed\n");
        CloseHandle((HANDLE)file_handle);
        file_handle = INVALID_HANDLE_VALUE;
        return false;
    }
    base = MapViewOfFile((HANDLE)map_handle, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        fprintf(stderr, "[gguf] MapViewOfFile failed\n");
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
    if (fd < 0) { fprintf(stderr, "[gguf] open failed: %s\n", path.c_str()); return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "[gguf] stat failed: %s\n", path.c_str());
        close(fd);
        fd = -1;
        return false;
    }
    size = (size_t)st.st_size;
    base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[gguf] mmap failed\n");
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
} // namespace

GGUF::~GGUF() {
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

bool GGUF::open(const std::string& path) {
#ifdef _WIN32
    if (!map_readonly(path, base_, size_, win_file_, win_map_)) return false;
#else
    if (!map_readonly(path, base_, size_, fd_)) return false;
#endif

    Cursor c{ (const uint8_t*)base_, 0, size_ };
    char magic[4]; memcpy(magic, c.p, 4); c.off = 4;
    if (memcmp(magic, "GGUF", 4) != 0) { fprintf(stderr, "[gguf] bad magic\n"); return false; }
    uint32_t version = c.rd<uint32_t>();
    uint64_t n_tensors = c.rd<uint64_t>();
    uint64_t n_kv = c.rd<uint64_t>();
    (void)version;

    // metadata
    for (uint64_t i = 0; i < n_kv && c.ok; i++) {
        std::string key = c.rd_str();
        uint32_t vt = c.rd<uint32_t>();
        if (vt == VT_STR) { strs_[key] = c.rd_str(); }
        else if (vt == VT_F32) { floats_[key] = c.rd<float>(); }
        else if (vt == VT_F64) { floats_[key] = c.rd<double>(); }
        else if (vt == VT_BOOL || vt == VT_U8) { ints_[key] = c.rd<uint8_t>(); }
        else if (vt == VT_I8)  { ints_[key] = c.rd<int8_t>(); }
        else if (vt == VT_U16) { ints_[key] = c.rd<uint16_t>(); }
        else if (vt == VT_I16) { ints_[key] = c.rd<int16_t>(); }
        else if (vt == VT_U32) { ints_[key] = c.rd<uint32_t>(); }
        else if (vt == VT_I32) { ints_[key] = c.rd<int32_t>(); }
        else if (vt == VT_U64) { ints_[key] = (long)c.rd<uint64_t>(); }
        else if (vt == VT_I64) { ints_[key] = c.rd<int64_t>(); }
        else if (vt == VT_ARR) {
            uint32_t et = c.rd<uint32_t>(); uint64_t n = c.rd<uint64_t>();
            if (et == VT_STR) { for (uint64_t k = 0; k < n && c.ok; k++) c.rd_str(); }
            else {
                // Skip the scalar payload, but fail loudly on an unsupported element
                // type (scalar_size==0 -> a 0-byte skip would desync the cursor) or a
                // declared span that overflows / runs past the file.
                int es = scalar_size(et);
                if (es == 0 || n > (c.size - c.off) / (size_t)es) {
                    fprintf(stderr, "[gguf] bad metadata array (elem type %u, n=%llu) for %s\n",
                            et, (unsigned long long)n, key.c_str());
                    return false;
                }
                c.skip((size_t)n * es);
            }
        } else { fprintf(stderr, "[gguf] unknown vt %u for %s\n", vt, key.c_str()); return false; }
    }
    if (!c.ok) { fprintf(stderr, "[gguf] metadata parse error\n"); return false; }

    long alignment = ints_.count("general.alignment") ? ints_["general.alignment"] : 32;
    // general.alignment is file-controlled; it must be a positive power of two (the
    // spec default is 32). A present-but-zero value would divide-by-zero (SIGFPE) when
    // computing data_start below, and a negative value would mis-align it. Clamp any
    // invalid alignment back to the default instead of trusting it.
    if (alignment <= 0) alignment = 32;

    // tensor infos
    struct Info { std::string name; GGUFTensor t; uint64_t offset; };
    std::vector<Info> infos; infos.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors && c.ok; i++) {
        Info in; in.name = c.rd_str();
        // n_dims is file-controlled; GGUFTensor::dims is fixed at ggml's
        // GGML_MAX_DIMS (4). Reject nd > 4 before the loop so a malformed or
        // future-format tensor cannot write past dims[4] (which would clobber
        // the adjacent n_values/n_bytes/data members) and desync the cursor.
        uint32_t nd = c.rd<uint32_t>();
        if (!c.ok || nd > 4) { fprintf(stderr, "[gguf] tensor %s has invalid n_dims=%u (max 4)\n", in.name.c_str(), nd); return false; }
        in.t.n_dims = nd;
        long nv = 1;
        for (uint32_t d = 0; d < nd; d++) { long e = (long)c.rd<uint64_t>(); in.t.dims[d] = e; nv *= e; }
        in.t.ggml_type = c.rd<uint32_t>();
        in.offset = c.rd<uint64_t>();
        in.t.n_values = nv;
        long bb, be; block_info(in.t.ggml_type, bb, be);
        in.t.n_bytes = be ? (nv / be) * bb : 0;
        infos.push_back(in);
    }
    if (!c.ok) { fprintf(stderr, "[gguf] tensor table parse error\n"); return false; }

    size_t data_start = (c.off + alignment - 1) / alignment * alignment;
    for (auto& in : infos) {
        // The tensor offset/size are file-controlled. The metadata is already validated
        // against the file, but the resolved data region is not — bounds-check it against
        // size_ so a truncated/malformed GGUF fails loudly here instead of producing an
        // out-of-bounds pointer that is later dereferenced (e.g. cudaMemcpy on upload).
        // Written with subtraction so the address arithmetic itself cannot overflow.
        if (data_start > size_ ||
            in.offset > size_ - data_start ||
            (uint64_t)in.t.n_bytes > size_ - data_start - in.offset) {
            fprintf(stderr, "[gguf] tensor %s data out of bounds (offset=%llu n_bytes=%ld file=%zu)\n",
                    in.name.c_str(), (unsigned long long)in.offset, in.t.n_bytes, size_);
            return false;
        }
        in.t.data = (const uint8_t*)base_ + data_start + in.offset;
        tensors_[in.name] = in.t;
    }
    return true;
}

long GGUF::meta_int(const std::string& k, long d) const { auto it=ints_.find(k); return it==ints_.end()?d:it->second; }
double GGUF::meta_float(const std::string& k, double d) const { auto it=floats_.find(k); return it==floats_.end()?d:it->second; }
std::string GGUF::meta_str(const std::string& k, const std::string& d) const { auto it=strs_.find(k); return it==strs_.end()?d:it->second; }
const GGUFTensor* GGUF::tensor(const std::string& n) const { auto it=tensors_.find(n); return it==tensors_.end()?nullptr:&it->second; }

} // namespace sparkinfer
