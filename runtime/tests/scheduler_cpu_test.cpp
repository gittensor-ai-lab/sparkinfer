#include "sparkinfer/scheduler.h"

#include <cassert>
#include <cstdio>

int main() {
    using namespace sparkinfer;

    // CONTINUOUS_BATCHING (vLLM V1): decode-first, then one small prefill in remaining budget.
    {
        Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 4);
        std::vector<ScheduledSequence> active;
        active.push_back({1, 10, SeqPhase::DECODE, 5, 3, 0});
        active.push_back({2, 11, SeqPhase::DECODE, 1, 1, 0});
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 128});  // small → may mix

        ScheduleBatch batch = sched.schedule(active);
        assert(batch.decode_request_ids.size() == 2);
        assert(batch.decode_request_ids[0] == 1);
        assert(batch.decode_request_ids[1] == 2);
        assert(batch.prefill_request_ids.size() == 1);
        assert(batch.prefill_request_ids[0] == 3);
        assert(batch.total_tokens == 3);

        for (auto& s : active) {
            if (s.request_id == 3) {
                s.phase = SeqPhase::DECODE;
                s.prefill_remaining = 0;
            }
        }
        batch = sched.schedule(active);
        assert(batch.prefill_request_ids.empty());
        assert(batch.decode_request_ids.size() == 3);
        assert(batch.decode_request_ids[0] == 3);
        assert(batch.total_tokens == 3);
    }

    // Large prefill is deferred while decode is in flight (avoids atomic-prefill ITL spikes).
    {
        Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 4);
        std::vector<ScheduledSequence> active;
        active.push_back({1, 10, SeqPhase::DECODE, 5, 3, 0});
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 8192});
        ScheduleBatch batch = sched.schedule(active);
        assert(batch.decode_request_ids.size() == 1);
        assert(batch.prefill_request_ids.empty());
    }

    // Large prefill runs once decode drains.
    {
        Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 4);
        std::vector<ScheduledSequence> active;
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 8192});
        ScheduleBatch batch = sched.schedule(active);
        assert(batch.prefill_request_ids.size() == 1);
        assert(batch.decode_request_ids.empty());
    }

    // Decode fills the whole budget → prefill waits one step.
    {
        Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 2);
        std::vector<ScheduledSequence> active;
        active.push_back({1, 10, SeqPhase::DECODE, 5, 3, 0});
        active.push_back({2, 11, SeqPhase::DECODE, 1, 1, 0});
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 128});
        ScheduleBatch batch = sched.schedule(active);
        assert(batch.decode_request_ids.size() == 2);
        assert(batch.prefill_request_ids.empty());
    }

    // PRIORITY: exclusive prefill (no mix).
    {
        Scheduler sched(SchedulePolicy::PRIORITY, 4);
        std::vector<ScheduledSequence> active;
        active.push_back({1, 10, SeqPhase::DECODE, 5, 3, 0});
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 128});
        ScheduleBatch batch = sched.schedule(active);
        assert(batch.prefill_request_ids.size() == 1);
        assert(batch.decode_request_ids.empty());
    }

    // Budget=1: pack at most one request total.
    {
        Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 1);
        std::vector<ScheduledSequence> active;
        active.push_back({1, 10, SeqPhase::DECODE, 5, 3, 0});
        active.push_back({2, 11, SeqPhase::DECODE, 1, 1, 0});
        active.push_back({3, 12, SeqPhase::PREFILL, 9, 0, 128});
        ScheduleBatch batch = sched.schedule(active);
        assert(batch.decode_request_ids.size() == 1);
        assert(batch.decode_request_ids[0] == 1);
        assert(batch.prefill_request_ids.empty());
    }

    printf("[PASS] scheduler_cpu_test\n");
    return 0;
}
