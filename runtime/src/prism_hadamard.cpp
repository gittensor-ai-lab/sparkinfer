#include "sparkinfer/prism_hadamard.h"

#include "sparkinfer/gguf.h"

#include <cmath>

namespace sparkinfer {

bool PrismHadamard::load(const GGUF& gguf, std::string& err) {
    const std::string transform_key = "prism.hadamard.transform";
    const std::string t = gguf.meta_str(transform_key);
    if (t.empty()) return true;   // not a prism checkpoint; nothing to do

    present = true;
    transform = t;
    version = gguf.meta_int("prism.hadamard.version", 0);
    block_size = gguf.meta_int("prism.hadamard.block_size", 0);
    axis = gguf.meta_str("prism.hadamard.axis");
    sign_mode = gguf.meta_str("prism.hadamard.sign_mode");
    gdn_v_grouped = gguf.meta_int("prism.hadamard.gdn_v_grouped", 0) != 0;

    if (transform != "normalized-sylvester-walsh-hadamard") {
        err = "prism.hadamard.transform is '" + transform + "', which this runtime does not implement";
        return false;
    }
    if (axis != "input-last-dimension") {
        err = "prism.hadamard.axis is '" + axis + "', expected input-last-dimension";
        return false;
    }
    if (sign_mode != "explicit") {
        err = "prism.hadamard.sign_mode is '" + sign_mode + "', expected explicit";
        return false;
    }
    if (block_size <= 0 || (block_size & (block_size - 1)) != 0) {
        err = "prism.hadamard.block_size " + std::to_string(block_size) + " is not a power of two";
        return false;
    }

    const std::vector<long> widths = gguf.meta_int_array("prism.hadamard.sign_widths");
    const std::vector<long> values = gguf.meta_int_array("prism.hadamard.sign_values");
    long total = 0;
    for (long w : widths) total += w;
    if (widths.empty() || total != (long)values.size()) {
        err = "prism.hadamard.sign_values holds " + std::to_string(values.size()) +
              " entries, but the declared widths sum to " + std::to_string(total);
        return false;
    }
    long off = 0;
    for (long w : widths) {
        std::vector<int8_t> s;
        s.reserve((size_t)w);
        for (long i = 0; i < w; ++i) {
            const long v = values[(size_t)(off + i)];
            if (v != 1 && v != -1) {
                err = "prism.hadamard.sign_values contains " + std::to_string(v) + ", expected +-1";
                return false;
            }
            s.push_back((int8_t)v);
        }
        // A width repeated in the metadata would silently drop one of the vectors.
        if (signs_by_width.count(w)) {
            err = "prism.hadamard.sign_widths repeats width " + std::to_string(w);
            return false;
        }
        signs_by_width[w] = std::move(s);
        off += w;
    }

    for (const std::string& n : gguf.meta_str_array("prism.hadamard.weight_names")) rotated.insert(n);
    for (const std::string& n : gguf.meta_str_array("prism.hadamard.inverse_weight_names"))
        inverse_rotated.insert(n);
    if (rotated.empty()) {
        err = "prism.hadamard.weight_names is empty: nothing would be rotated";
        return false;
    }
    return true;
}

const std::vector<int8_t>* PrismHadamard::signs_for(long width) const {
    auto it = signs_by_width.find(width);
    return it == signs_by_width.end() ? nullptr : &it->second;
}

void hadamard_transform(float* x, long n, long block) {
    if (block <= 1 || n % block != 0) return;
    const float norm = 1.0f / std::sqrt((float)block);
    for (long base = 0; base < n; base += block) {
        float* b = x + base;
        for (long len = 1; len < block; len <<= 1) {
            for (long i = 0; i < block; i += (len << 1)) {
                for (long j = 0; j < len; ++j) {
                    const float u = b[i + j];
                    const float v = b[i + j + len];
                    b[i + j] = u + v;
                    b[i + j + len] = u - v;
                }
            }
        }
        for (long i = 0; i < block; ++i) b[i] *= norm;
    }
}

void hadamard_apply_sign(float* x, long n, const int8_t* sign) {
    for (long i = 0; i < n; ++i) x[i] *= (float)sign[i];
}

void hadamard_unrotate_activation(float* x, long n, long block, const int8_t* sign) {
    // R = H . diag(s) with both factors orthogonal and symmetric, so R^-1 = R^T = diag(s) . H:
    // the same two steps in the other order.
    hadamard_transform(x, n, block);
    if (sign) hadamard_apply_sign(x, n, sign);
}

void hadamard_rotate_activation(float* x, long n, long block, const int8_t* sign) {
    if (sign) hadamard_apply_sign(x, n, sign);
    hadamard_transform(x, n, block);
}

}  // namespace sparkinfer
