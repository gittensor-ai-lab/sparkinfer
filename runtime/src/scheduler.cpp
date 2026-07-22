// Scheduler — continuous-batching policy over in-flight requests.
// Host-only; decides which sequences run in the next step.
//
// vLLM V1-style iteration-level scheduling (CONTINUOUS_BATCHING / CHUNKED_PREFILL):
//   1. Pack pending decode requests first (up to max_tokens_per_batch) — protects ITPS
//   2. Fill remaining budget with at most one prefill
// Large prefills (prefill_remaining > SPARKINFER_PREFILL_MIX_MAX, default 2048) are
// NOT mixed with decode: sparkinfer's hybrid batched prefill is atomic (one GEMM pass),
// so admitting an 8k prefill mid-decode creates a multi-hundred-ms ITL spike. Small
// prefills may still mix. PRIORITY keeps exclusive prefill-first (no mix).

#include "sparkinfer/scheduler.h"

#include <unordered_map>
#include <algorithm>
#include <cstdlib>

namespace sparkinfer {

namespace {
int prefill_mix_max_tokens() {
    static int v = [] {
        const char* e = getenv("SPARKINFER_PREFILL_MIX_MAX");
        // 0 = always allow mix; default 2048 keeps TTFT-friendly short prompts mixed
        // while deferring long atomic prefills until decode drains.
        int x = e ? atoi(e) : 2048;
        return x >= 0 ? x : 2048;
    }();
    return v;
}
}  // namespace

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

    const bool mix = impl_->policy == SchedulePolicy::CHUNKED_PREFILL ||
                     impl_->policy == SchedulePolicy::CONTINUOUS_BATCHING;
    const int budget = impl_->max_tokens_per_batch > 0 ? impl_->max_tokens_per_batch : 1;

    if (!mix) {
        // PRIORITY: exclusive prefill-first (legacy serving behavior).
        for (const ScheduledSequence* s : ordered) {
            if (s->phase != SeqPhase::PREFILL) continue;
            batch.prefill_request_ids.push_back(s->request_id);
            batch.total_tokens += 1;
            return batch;
        }
        for (const ScheduledSequence* s : ordered) {
            if (s->phase != SeqPhase::DECODE) continue;
            if ((int)batch.decode_request_ids.size() >= budget) break;
            batch.decode_request_ids.push_back(s->request_id);
            batch.total_tokens += 1;
        }
        return batch;
    }

    // vLLM V1: decode-first, then admit one prefill into remaining budget.
    for (const ScheduledSequence* s : ordered) {
        if (s->phase != SeqPhase::DECODE) continue;
        if ((int)batch.decode_request_ids.size() >= budget) break;
        batch.decode_request_ids.push_back(s->request_id);
        batch.total_tokens += 1;
    }
    if (batch.total_tokens < budget) {
        const int mix_max = prefill_mix_max_tokens();
        for (const ScheduledSequence* s : ordered) {
            if (s->phase != SeqPhase::PREFILL) continue;
            // Defer large atomic prefills while decode is in flight.
            if (!batch.decode_request_ids.empty() && mix_max > 0 &&
                s->prefill_remaining > mix_max) {
                continue;
            }
            batch.prefill_request_ids.push_back(s->request_id);
            batch.total_tokens += 1;
            break;
        }
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
