#pragma once
#include <cuda_runtime.h>

namespace sparkinfer {

// A synchronous cudaMemcpy from pageable host memory (an mmap'd GGUF, a std::vector) may return
// once the bytes are staged for DMA, before they reach the device. Work on the default stream is
// ordered behind that DMA; work on a cudaStreamNonBlocking stream -- every stream the models
// create -- is not. A dequant/transcode launched on such a stream right after the copy could read
// the destination half-written: Qwen3.6-35B-A3B's 2 MB F32 router lost a run-dependent tail of its
// rows (zeros), so layer 0 routed to whichever experts those zeros made the top 8, and the model's
// perplexity sat near 17 instead of 6 in most launches. Every host-to-device upload in the model
// code, and the KV cache's block tables (read by those streams on every step), goes through here:
// the copy, then a wait on the stream that carried it.
inline cudaError_t si_h2d_complete(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    cudaError_t e = cudaMemcpy(dst, src, bytes, kind);
    if (e == cudaSuccess && kind == cudaMemcpyHostToDevice) e = cudaStreamSynchronize(0);
    return e;
}

}  // namespace sparkinfer
