#pragma once
// Ternary-Bonsai-2 announces prism.hadamard.gdn_v_grouped, and what it means is that everything
// PRODUCING a GDN v head stores its heads transposed: as [heads_per_group][groups] rather than the
// [groups][heads_per_group] the architecture reads them in. That covers attn_qkv's v rows,
// attn_gate, ssm_conv1d's v channels, ssm_a, ssm_dt.bias and the alpha/beta projections. ssm_out,
// which CONSUMES v, is stored in ordinary order.
//
// The regrouping has to run in the architecture's direction rather than the file's, even though
// the file is self-consistent in its own: the GDN kernel bakes in which q/k head drives which v
// head (v head r is driven by q/k head r / heads_per_group), so the head index is not free.
//
// Measured against the checkpoint this model was quantized from, every one of the 48 heads maps as
// 3*(h % 16) + h / 16, with 16 = ssm.group_count -- exactly a [3,16] -> [16,3] transpose.

namespace sparkinfer {

// Destination head index (architecture order) -> the head to read from the stored tensor.
constexpr long gdn_v_source_head(long dst_head, long groups, long per_group) {
    return (dst_head % per_group) * groups + (dst_head / per_group);
}

// The inverse: where a stored head belongs once regrouped. Only the tests and the comment above
// need this direction, but stating it keeps the pair honest.
constexpr long gdn_v_dest_head(long src_head, long groups, long per_group) {
    return (src_head % groups) * per_group + (src_head / groups);
}

}  // namespace sparkinfer
