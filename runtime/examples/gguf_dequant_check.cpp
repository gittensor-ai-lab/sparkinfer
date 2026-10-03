// Dequantizes GGUF tensors on the GPU with launch_gguf_dequant and writes each one as raw bf16,
// so eval/gguf_dequant_check.py can compare it against gguf-py's reference dequantizer.
//   gguf_dequant_check <file.gguf> <out_dir> <tensor> [tensor ...]
// Writes <out_dir>/<tensor>.bf16 (n_values little-endian bf16, ggml element order).
#include <cstdio>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "sparkinfer/gguf.h"
#include "sparkinfer/kernels/quant.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file.gguf> <out_dir> <tensor> [tensor ...]\n", argv[0]);
        return 2;
    }
    sparkinfer::GGUF g;
    if (!g.open(argv[1])) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    const std::string out_dir = argv[2];
    int rc = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string name = argv[i];
        const sparkinfer::GGUFTensor* t = g.tensor(name);
        if (!t || t->n_bytes <= 0) { fprintf(stderr, "%s: missing or unsized\n", name.c_str()); rc = 1; continue; }
        void* src = nullptr;
        void* dst = nullptr;
        if (cudaMalloc(&src, t->n_bytes) != cudaSuccess ||
            cudaMalloc(&dst, (size_t)t->n_values * 2) != cudaSuccess) {
            fprintf(stderr, "%s: cudaMalloc failed\n", name.c_str());
            return 1;
        }
        cudaMemcpy(src, t->data, t->n_bytes, cudaMemcpyHostToDevice);
        sparkinfer::kernels::launch_gguf_dequant(t->ggml_type, src, dst, t->n_values, nullptr);
        std::vector<unsigned short> host((size_t)t->n_values);
        cudaError_t e = cudaMemcpy(host.data(), dst, host.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(src);
        cudaFree(dst);
        if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", name.c_str(), cudaGetErrorString(e)); return 1; }
        const std::string path = out_dir + "/" + name + ".bf16";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); return 1; }
        fwrite(host.data(), 2, host.size(), f);
        fclose(f);
        printf("%s type %d values %ld -> %s\n", name.c_str(), t->ggml_type, t->n_values, path.c_str());
    }
    return rc;
}
