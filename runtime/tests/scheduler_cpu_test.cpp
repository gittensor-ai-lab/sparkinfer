#include "sparkinfer/scheduler.h"

#include <cassert>
#include <cstdio>

int main() {
    using namespace sparkinfer;

    Scheduler sched(SchedulePolicy::CONTINUOUS_BATCHING, 4);

    std::vector<ScheduledSequence> active;
    active.push_back({1, 10, SeqPhase::DECODE, 5, 3});
    active.push_back({2, 11, SeqPhase::DECODE, 1, 1});
    active.push_back({3, 12, SeqPhase::PREFILL, 9, 0});

    ScheduleBatch batch = sched.schedule(active);
    assert(batch.prefill_request_ids.size() == 1);
    assert(batch.prefill_request_ids[0] == 3);
    assert(batch.decode_request_ids.empty());

    for (auto& s : active) {
        if (s.request_id == 3) s.phase = SeqPhase::DECODE;
    }
    batch = sched.schedule(active);
    assert(batch.prefill_request_ids.empty());
    assert(batch.decode_request_ids.size() == 1);
    assert(batch.decode_request_ids[0] == 3);  // highest priority among decode jobs

    printf("[PASS] scheduler_cpu_test\n");
    return 0;
}
