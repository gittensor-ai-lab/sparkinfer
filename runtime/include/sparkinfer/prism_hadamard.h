#pragma once
// The Hadamard rotation that Ternary-Bonsai-2 carries in its GGUF metadata (prism.hadamard.*).
//
// Ternary quantisation is brutal on outlier channels, so the checkpoint's weights were rotated
// offline: each rotated weight was multiplied, along its INPUT axis, by diag(sign) followed by a
// normalized Sylvester-Walsh Hadamard applied in blocks of 1024. The transform is orthogonal and
// symmetric, so at inference the matching rotation is applied to the activation entering that
// weight -- W_rot . (H . diag(s) . x) reproduces W . x.
//
// What the checkpoint declares (read, never assumed -- see load()):
//   - 401 rotated weights: FFN gate/up/down, the linear-attention blocks' qkv/gate/ssm_out, the
//     full-attention blocks' q/k/v/output, and the LM head;
//   - token_embd.weight rotated by the INVERSE. It is a lookup, not a matmul, so it has no input
//     activation to rotate; instead its rows are STORED rotated (which is what makes quantising
//     them to trits survivable) and the runtime applies R^-1 to the row it reads. The residual
//     stream therefore stays in the ordinary basis -- it could not do otherwise, since RMSNorm's
//     per-channel weight does not commute with R;
//   - three sign vectors, one per input width the rotated weights take (5120, 6144, 17408).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sparkinfer {

class GGUF;

struct PrismHadamard {
    bool        present = false;
    long        version = 0;
    long        block_size = 0;          // 1024 in this checkpoint
    std::string transform;               // "normalized-sylvester-walsh-hadamard"
    std::string axis;                    // "input-last-dimension"
    std::string sign_mode;               // "explicit"
    bool        gdn_v_grouped = false;
    std::unordered_map<long, std::vector<int8_t>> signs_by_width;   // width -> +-1 per input index
    std::unordered_set<std::string> rotated;          // weights stored pre-rotated
    std::unordered_set<std::string> inverse_rotated;  // weights carrying the inverse rotation

    // Reads prism.hadamard.* out of an open GGUF. Returns false only when the metadata is present
    // but inconsistent (an unsupported transform, a sign vector that does not match its declared
    // width, a block size that is not a power of two); a checkpoint without the keys at all leaves
    // `present` false and returns true.
    bool load(const GGUF& gguf, std::string& err);

    const std::vector<int8_t>* signs_for(long width) const;
    bool rotates(const std::string& tensor) const { return rotated.count(tensor) != 0; }
};

// In-place normalized Walsh-Hadamard transform over each `block`-sized span of `x`. Orthogonal and
// its own inverse: applying it twice returns the input.
void hadamard_transform(float* x, long n, long block);

// x[i] *= sign[i]; sign values are +-1.
void hadamard_apply_sign(float* x, long n, const int8_t* sign);

// The activation-side rotation for a weight whose input width is `n`: sign first, then the
// blockwise Hadamard -- the order that inverts a weight rotated as W . diag(s) . H.
void hadamard_rotate_activation(float* x, long n, long block, const int8_t* sign);

// The inverse, R^-1 = R^T: the Hadamard first, then the signs. Used on a row read out of the
// inverse-rotated token_embd, which is stored in the rotated basis.
void hadamard_unrotate_activation(float* x, long n, long block, const int8_t* sign);

}  // namespace sparkinfer
