// Scheduler — continuous-batching policy over in-flight requests.
// Host-only; decides which sequences run in the next step.

#include "sparkinfer/scheduler.h"

#include <unordered_map>
#include <algorithm>

namespace sparkinfer {

struct Scheduler::Impl {
    SchedulePolicy policy;
    int max_tokens_per_batch;
    std::unordered_map<uint64_t, SequenceGroup> groups;
};

Scheduler::Scheduler(SchedulePolicy policy, int max_tokens_per_batch)
    : impl_(new Impl{policy, max_tokens_per_batch, {}}) {}

Scheduler::~Scheduler() = default;

ScheduleBatch Scheduler::schedule(const std::vector<ScheduledSequence>& active) const {
    ScheduleBatch batch;
    if (active.empty()) return batch;

    std::vector<const ScheduledSequence*> ordered;
    ordered.reserve(active.size());
    for (const auto& s : active) ordered.push_back(&s);
    std::sort(ordered.begin(), ordered.end(),
              [](const ScheduledSequence* a, const ScheduledSequence* b) {
                  return a->priority > b->priority;
              });

    // Prefill-first: one prefill job at a time so TTFT stays predictable.
    for (const ScheduledSequence* s : ordered) {
        if (s->phase == SeqPhase::PREFILL) {
            batch.prefill_request_ids.push_back(s->request_id);
            batch.total_tokens = 1;
            return batch;
        }
    }

    // One decode request per step while the model forward path is batch=1.
    for (const ScheduledSequence* s : ordered) {
        if (s->phase != SeqPhase::DECODE) continue;
        batch.decode_request_ids.push_back(s->request_id);
        batch.total_tokens = 1;
        break;
    }
    return batch;
}

void Scheduler::add_sequence_group(SequenceGroup g) { impl_->groups[g.group_id] = g; }
void Scheduler::remove_sequence_group(uint64_t id)  { impl_->groups.erase(id); }

ScheduleBatch Scheduler::schedule() {
    ScheduleBatch batch;
    std::vector<const SequenceGroup*> ordered;
    for (auto& kv : impl_->groups) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const SequenceGroup* a, const SequenceGroup* b) { return a->priority > b->priority; });
    for (auto* g : ordered) {
        if (batch.total_tokens + g->num_seqs > impl_->max_tokens_per_batch) break;
        for (int i = 0; i < g->num_seqs; i++) batch.decode_request_ids.push_back(g->group_id);
        batch.total_tokens += g->num_seqs;
    }
    return batch;
}

std::vector<uint64_t> Scheduler::preempt(int tokens_needed) {
    std::vector<const SequenceGroup*> ordered;
    for (auto& kv : impl_->groups) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const SequenceGroup* a, const SequenceGroup* b) { return a->priority < b->priority; });
    std::vector<uint64_t> victims;
    int freed = 0;
    for (auto* g : ordered) {
        if (freed >= tokens_needed) break;
        victims.push_back(g->group_id);
        freed += g->num_seqs;
    }
    return victims;
}

} // namespace sparkinfer
