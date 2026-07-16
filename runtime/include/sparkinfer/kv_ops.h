#pragma once
#include <cuda_runtime.h>

namespace sparkinfer {

// Append one new token's K/V into the paged cache for each sequence.
//   k_new/v_new:  [num_seqs, num_kv_heads, head_dim]   (bf16)
//   k_pool/v_pool base of this layer's sub-pool (caller adds layer offset)
//   block_table:  [num_seqs, max_blocks_per_seq]       (int32, device)
//   write_pos:    [num_seqs]  position index for the new token (= old seq_len)
void launch_kv_append(
    void* k_pool, void* v_pool,
    const void* k_new, const void* v_new,
    const int* block_table, const int* write_pos,
    int num_seqs, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

// Elementwise residual add: out[i] = a[i] + b[i]  (bf16). out may alias a.
void launch_residual_add(const void* a, const void* b, void* out, int n,
                         cudaStream_t stream = nullptr);

} // namespace sparkinfer
