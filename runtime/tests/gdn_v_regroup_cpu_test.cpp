// The GDN v-head regrouping Ternary-Bonsai-2 needs. This mapping took the longest of anything in
// that checkpoint to pin down, because getting it wrong leaves every weight correct and every one
// attached to the wrong head -- the model still runs, still emits plausible magnitudes, and still
// produces nothing but nonsense. These are the properties that identify it.
#include "sparkinfer/gdn_v_regroup.h"

#include <cstdio>
#include <set>

static int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using sparkinfer::gdn_v_dest_head;
using sparkinfer::gdn_v_source_head;

void test_matches_the_mapping_measured_against_the_unquantized_checkpoint() {
    // Ternary-Bonsai-2: 48 v heads, ssm.group_count 16, so three heads per group. Measured head by
    // head against Qwen3.8-27B's own in_proj_z and A_log: stored head h holds what the
    // architecture calls head 3*(h % 16) + h / 16.
    const long groups = 16, per_group = 3, heads = groups * per_group;
    for (long stored = 0; stored < heads; ++stored)
        CHECK(gdn_v_dest_head(stored, groups, per_group) == 3 * (stored % 16) + stored / 16);

    // A few the debugging actually printed, so a future refactor has something concrete to fail on.
    CHECK(gdn_v_dest_head(0, groups, per_group) == 0);
    CHECK(gdn_v_dest_head(1, groups, per_group) == 3);
    CHECK(gdn_v_dest_head(2, groups, per_group) == 6);
    CHECK(gdn_v_dest_head(16, groups, per_group) == 1);
    CHECK(gdn_v_dest_head(17, groups, per_group) == 4);
    CHECK(gdn_v_dest_head(47, groups, per_group) == 47);
}

void test_the_two_directions_invert_each_other() {
    // The loader reads with gdn_v_source_head; everything else reasons in the other direction.
    // If they ever disagree the weights land on the wrong heads and nothing else complains.
    for (long groups : {1L, 2L, 4L, 16L}) {
        for (long per_group : {1L, 2L, 3L, 8L}) {
            const long heads = groups * per_group;
            for (long h = 0; h < heads; ++h) {
                CHECK(gdn_v_source_head(gdn_v_dest_head(h, groups, per_group), groups, per_group) == h);
                CHECK(gdn_v_dest_head(gdn_v_source_head(h, groups, per_group), groups, per_group) == h);
            }
        }
    }
}

void test_it_is_a_permutation_not_a_gather() {
    // Every head must appear exactly once. A mapping that drops one and repeats another would
    // silently duplicate a head's decay across the stack.
    for (long groups : {1L, 2L, 16L}) {
        for (long per_group : {1L, 3L, 8L}) {
            const long heads = groups * per_group;
            std::set<long> seen;
            for (long h = 0; h < heads; ++h) {
                const long src = gdn_v_source_head(h, groups, per_group);
                CHECK(src >= 0 && src < heads);
                seen.insert(src);
            }
            CHECK((long)seen.size() == heads);
        }
    }
}

void test_one_head_per_group_leaves_the_order_alone() {
    // per_group == 1 means there is nothing to transpose, and a model whose v and q/k head counts
    // match must not have its heads shuffled by a regrouping that thinks it has work to do.
    for (long h = 0; h < 16; ++h) CHECK(gdn_v_source_head(h, 16, 1) == h);
    // Likewise a single group.
    for (long h = 0; h < 8; ++h) CHECK(gdn_v_source_head(h, 1, 8) == h);
}

int main() {
    test_matches_the_mapping_measured_against_the_unquantized_checkpoint();
    test_the_two_directions_invert_each_other();
    test_it_is_a_permutation_not_a_gather();
    test_one_head_per_group_leaves_the_order_alone();
    std::printf("gdn_v_regroup_cpu_test: %s\n", failures ? "FAILURES" : "OK");
    return failures ? 1 : 0;
}
