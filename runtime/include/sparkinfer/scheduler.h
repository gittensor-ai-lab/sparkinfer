#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <memory>

namespace sparkinfer {

enum class SchedulePolicy {
    CONTINUOUS_BATCHING,
    CHUNKED_PREFILL,
    PRIORITY,
};

struct SequenceGroup {
    uint64_t group_id;
    int num_seqs;
    int max_new_tokens;
    int priority;   // higher = more urgent
};

struct ScheduleBatch {
    std::vector<uint64_t> prefill_seq_ids;
    std::vector<uint64_t> decode_seq_ids;
    int total_tokens;
};

class Scheduler {
public:
    explicit Scheduler(SchedulePolicy policy = SchedulePolicy::CHUNKED_PREFILL,
                       int max_tokens_per_batch = 8192);

    void add_sequence_group(SequenceGroup group);
    void remove_sequence_group(uint64_t group_id);

    ScheduleBatch schedule();

    // Preempt lowest-priority decode sequences to make room for prefill
    std::vector<uint64_t> preempt(int tokens_needed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
