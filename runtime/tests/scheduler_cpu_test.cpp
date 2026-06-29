// CPU test for the continuous-batching scheduler — pure host policy, no GPU needed.
// Validates priority ordering, token-budget packing, and preemption victim selection.

#include "sparkinfer/scheduler.h"
#include <cstdio>
#include <vector>

using S = sparkinfer::Scheduler;
using SG = sparkinfer::SequenceGroup;
#define CHECK(x) do { if (!(x)) { printf("FAIL: %s (line %d)\n", #x, __LINE__); return 1; } } while (0)

static SG group(uint64_t id, int num_seqs, int priority) {
    return SG{id, num_seqs, /*max_new_tokens=*/128, priority};
}

int main() {
    // Empty scheduler → empty batch.
    {
        S sched;
        auto batch = sched.schedule();
        CHECK(batch.decode_seq_ids.empty());
        CHECK(batch.prefill_seq_ids.empty());
        CHECK(batch.total_tokens == 0);
    }

    // Single group fills the batch.
    {
        S sched(/*policy=*/sparkinfer::SchedulePolicy::PRIORITY, /*max_tokens=*/4);
        sched.add_sequence_group(group(10, 2, 5));
        auto batch = sched.schedule();
        CHECK(batch.total_tokens == 2);
        CHECK(batch.decode_seq_ids.size() == 2u);
        CHECK(batch.decode_seq_ids[0] == 10);
        CHECK(batch.decode_seq_ids[1] == 10);
    }

    // Higher priority groups are scheduled first.
    {
        S sched(sparkinfer::SchedulePolicy::PRIORITY, 8);
        sched.add_sequence_group(group(1, 1, 1));
        sched.add_sequence_group(group(2, 1, 10));
        sched.add_sequence_group(group(3, 1, 5));
        auto batch = sched.schedule();
        CHECK(batch.decode_seq_ids.size() == 3u);
        CHECK(batch.decode_seq_ids[0] == 2);
        CHECK(batch.decode_seq_ids[1] == 3);
        CHECK(batch.decode_seq_ids[2] == 1);
        CHECK(batch.total_tokens == 3);
    }

    // Token budget stops packing when the next whole group would exceed the limit.
    {
        S sched(sparkinfer::SchedulePolicy::PRIORITY, 5);
        sched.add_sequence_group(group(1, 2, 10));
        sched.add_sequence_group(group(2, 2, 5));
        sched.add_sequence_group(group(3, 2, 1));
        auto batch = sched.schedule();
        CHECK(batch.total_tokens == 4);
        CHECK(batch.decode_seq_ids.size() == 4u);
        // Groups 1 and 2 fit (2+2=4); group 3 would push to 6 > 5.
        CHECK(batch.decode_seq_ids[0] == 1);
        CHECK(batch.decode_seq_ids[1] == 1);
        CHECK(batch.decode_seq_ids[2] == 2);
        CHECK(batch.decode_seq_ids[3] == 2);
    }

    // remove_sequence_group drops a group from future schedules.
    {
        S sched(sparkinfer::SchedulePolicy::PRIORITY, 8);
        sched.add_sequence_group(group(1, 1, 1));
        sched.add_sequence_group(group(2, 1, 5));
        sched.remove_sequence_group(1);
        auto batch = sched.schedule();
        CHECK(batch.decode_seq_ids.size() == 1u);
        CHECK(batch.decode_seq_ids[0] == 2);
    }

    // preempt() evicts lowest-priority groups until enough tokens are freed.
    {
        S sched(sparkinfer::SchedulePolicy::PRIORITY, 8);
        sched.add_sequence_group(group(1, 2, 1));
        sched.add_sequence_group(group(2, 3, 5));
        sched.add_sequence_group(group(3, 1, 10));
        auto victims = sched.preempt(4);
        CHECK(victims.size() == 2u);
        CHECK(victims[0] == 1);  // lowest priority, 2 seqs
        CHECK(victims[1] == 2);  // next lowest, 3 seqs → 5 freed total
    }

    // preempt() returns a single group when it alone satisfies the budget.
    {
        S sched(sparkinfer::SchedulePolicy::PRIORITY, 8);
        sched.add_sequence_group(group(1, 5, 1));
        sched.add_sequence_group(group(2, 2, 10));
        auto victims = sched.preempt(3);
        CHECK(victims.size() == 1u);
        CHECK(victims[0] == 1);
    }

    // preempt() with zero need returns empty.
    {
        S sched;
        sched.add_sequence_group(group(1, 1, 1));
        auto victims = sched.preempt(0);
        CHECK(victims.empty());
    }

    printf("scheduler_cpu_test: OK\n");
    return 0;
}
