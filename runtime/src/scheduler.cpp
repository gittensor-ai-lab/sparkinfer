// Scheduler — minimal continuous-batching policy over sequence groups.
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

void Scheduler::add_sequence_group(SequenceGroup g) { impl_->groups[g.group_id] = g; }
void Scheduler::remove_sequence_group(uint64_t id)  { impl_->groups.erase(id); }

ScheduleBatch Scheduler::schedule() {
    ScheduleBatch batch; batch.total_tokens = 0;
    // Order by priority (desc); pack decode steps until the token budget is hit.
    std::vector<const SequenceGroup*> ordered;
    for (auto& kv : impl_->groups) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const SequenceGroup* a, const SequenceGroup* b) { return a->priority > b->priority; });
    for (auto* g : ordered) {
        if (batch.total_tokens + g->num_seqs > impl_->max_tokens_per_batch) continue;
        for (int i = 0; i < g->num_seqs; i++) batch.decode_seq_ids.push_back(g->group_id);
        batch.total_tokens += g->num_seqs;
    }
    return batch;
}

std::vector<uint64_t> Scheduler::preempt(int tokens_needed) {
    std::vector<const SequenceGroup*> ordered;
    for (auto& kv : impl_->groups) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(),
              [](const SequenceGroup* a, const SequenceGroup* b) { return a->priority < b->priority; });
    std::vector<uint64_t> victims; int freed = 0;
    for (auto* g : ordered) { if (freed >= tokens_needed) break; victims.push_back(g->group_id); freed += g->num_seqs; }
    return victims;
}

} // namespace sparkinfer
