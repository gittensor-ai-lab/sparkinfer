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

enum class SeqPhase {
    PREFILL,
    DECODE,
};

struct SequenceGroup {
    uint64_t group_id;
    int num_seqs;
    int max_new_tokens;
    int priority;   // higher = more urgent
};

// Per-request state tracked by the continuous-batch engine.
struct ScheduledSequence {
    uint64_t request_id = 0;
    uint64_t seq_id = 0;
    SeqPhase phase = SeqPhase::PREFILL;
    int priority = 0;
    int tokens_in_phase = 0;     // prefill progress or decode tokens emitted
    int prefill_remaining = 0;   // prompt tokens left (PREFILL only); 0 if unknown
};

struct ScheduleBatch {
    std::vector<uint64_t> prefill_request_ids;
    std::vector<uint64_t> decode_request_ids;
    int total_tokens = 0;
};

class Scheduler {
public:
    explicit Scheduler(SchedulePolicy policy = SchedulePolicy::CONTINUOUS_BATCHING,
                       int max_tokens_per_batch = 256);
    ~Scheduler();

    // Pick the next request(s) to advance one step.
    // CONTINUOUS_BATCHING / CHUNKED_PREFILL: vLLM V1 decode-first packing (up to
    // max_tokens_per_batch), then at most one prefill in remaining budget.
    // PRIORITY: exclusive prefill-first (no mix).
    ScheduleBatch schedule(const std::vector<ScheduledSequence>& active) const;

    // Legacy group API (kept for compatibility).
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
