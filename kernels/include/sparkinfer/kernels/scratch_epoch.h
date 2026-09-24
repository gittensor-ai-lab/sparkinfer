#pragma once
#include <cstdint>

// A count of how many times a grow-on-demand prefill scratch buffer has been freed and replaced.
//
// The whole-prefill CUDA graph (qwen35_prefill.cpp) bakes every pointer its nodes were captured
// with, including the kernel-level scratch below (the GDN scan workspace, the attention V plane,
// the skinny-GEMM split-K partials). Those buffers only ever grow, but growing frees the old one,
// and a pass that never captures a graph can be the one that grows them. A graph captured before
// that point would replay against freed device memory, so it records this value at capture and
// is dropped when the value has moved.
namespace sparkinfer { namespace kernels {

uint64_t prefill_scratch_epoch();
void note_prefill_scratch_moved();   // call after freeing a buffer a captured graph may reference

}}  // namespace sparkinfer::kernels
