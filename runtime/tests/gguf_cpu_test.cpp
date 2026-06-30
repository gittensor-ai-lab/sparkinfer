// CPU test for the GGUF reader's tensor-table parsing. Builds tiny in-memory
// GGUF files and checks that a tensor whose dimension product overflows signed
// long is rejected (regression for the silent-accept bug where the product
// wraps to a small/zero n_values that slips past the data-bounds check), while
// a well-formed tensor still loads. Pure host C++, no GPU/CUDA needed.
//
// Build: g++ -O2 -std=c++17 gguf_cpu_test.cpp ../src/gguf.cpp -I../include -o gguf_cpu_test

#include "sparkinfer/gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>

using std::vector;

static void put_u32(vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
}
static void put_u64(vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; i++) b.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
}
static void put_str(vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    for (char c : s) b.push_back((uint8_t)c);
}

static std::string write_temp(const vector<uint8_t>& bytes) {
    char path[] = "/tmp/sparkinfer_gguf_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); std::abort(); }
    ssize_t n = write(fd, bytes.data(), bytes.size());
    if (n != (ssize_t)bytes.size()) { perror("write"); std::abort(); }
    close(fd);
    return std::string(path);
}

static int failures = 0;
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; }       \
    } while (0)

int main() {
    using namespace sparkinfer;

    // 1) A tensor whose dimension product overflows signed long must be rejected.
    {
        vector<uint8_t> b;
        b.insert(b.end(), {'G', 'G', 'U', 'F'});
        put_u32(b, 3);                 // version
        put_u64(b, 1);                 // n_tensors
        put_u64(b, 0);                 // n_kv
        put_str(b, "t");               // tensor name
        put_u32(b, 2);                 // n_dims
        put_u64(b, 0x100000000ull);    // 2^32
        put_u64(b, 0x100000000ull);    // 2^32 -> product 2^64 overflows long
        // ggml_type/offset are intentionally omitted: open() must bail at the
        // dimension-product guard before it ever reads them.
        std::string p = write_temp(b);
        GGUF g;
        bool ok = g.open(p);
        unlink(p.c_str());
        CHECK(!ok, "tensor with overflowing dimension product was accepted");
    }

    // 2) A well-formed single tensor still loads (the guard is not over-eager).
    {
        vector<uint8_t> b;
        b.insert(b.end(), {'G', 'G', 'U', 'F'});
        put_u32(b, 3);                 // version
        put_u64(b, 1);                 // n_tensors
        put_u64(b, 0);                 // n_kv
        put_str(b, "t");               // name
        put_u32(b, 1);                 // n_dims
        put_u64(b, 4);                 // dims[0] = 4
        put_u32(b, 0);                 // ggml_type = F32 (4 bytes/elem) -> 16 bytes
        put_u64(b, 0);                 // data offset = 0
        // Header ends at byte 57; tensor data aligns to 32 -> starts at byte 64.
        while (b.size() < 64) b.push_back(0);
        for (int i = 0; i < 16; i++) b.push_back(0);  // 16 bytes of tensor data
        std::string p = write_temp(b);
        GGUF g;
        bool ok = g.open(p);
        const GGUFTensor* t = g.tensor("t");
        unlink(p.c_str());
        CHECK(ok, "well-formed GGUF failed to open");
        CHECK(t != nullptr && t->n_values == 4 && t->n_bytes == 16,
              "well-formed tensor parsed with wrong shape");
    }

    if (failures == 0) {
        printf("gguf_cpu_test: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "gguf_cpu_test: %d check(s) failed\n", failures);
    return 1;
}
