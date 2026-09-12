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
// The decode width above which a step is no longer dominated by its fixed weight read. Defaults
// to the packed-decode ceiling; below it a wider batch is very nearly free.
int packed_decode_width() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_WIDE_DECODE_ROWS");
        const int x = e ? atoi(e) : 32;
        return x < 1 ? 1 : x;
    }();
    return v;
}
// How many prefills one iteration may admit while the decode batch is still filling. 1 restores
// the pre-#994 behaviour; 2 restores #994's, for a paired A/B out of one binary.
//
// This was 2 because that is the step #994 measured: it went from admitting ONE row per
// iteration to two, and stopped there. Two is not a property of the trade -- it is where the
// measurement stopped. The cost of admitting another prefill is one more row in this iteration's
// packed forward, and #990/#992/#993 made that row nearly free up to `packed_decode_width()`
// rows: the packed decode reads the whole ~15 GB weight set once per step whatever the width, so
// every row admitted before the batch reaches that width is amortized against a read the step
// was going to do anyway. What the cap actually costs is TIME AT A NARROW WIDTH -- every
// iteration spent ramping is an iteration paying the full weight read for a fraction of the
// rows.
//
// So the bound is the width the batch is ramping TOWARD, which is the same
// `packed_decode_width()` the `deep_ramp` gate below is already written against, not a fixed 2.
// Measured on an RTX 5090 at the bot's own invocation (qwen3_gguf_cb_bench <model> C 256 256 512),
// aggregate tok/s, two interleaved repeats per arm out of ONE binary:
//
//   admit/step      2 (main)      6        8       12       16       32
//   c16              790.80    805.55   810.15   811.70   815.80        -
//   c32             1296.80         -        -        -  1370.15   1379.30
//
// Monotone in the cap at both widths, and mean ITL falls with it (c32 21.29 -> 19.74 ms), so
// this is not a latency-for-throughput trade on the mean. It IS one on the tail: a deep ramp now
// admits its whole backlog in one iteration, and max ITL over the run moves from ~103 ms to
// 94-879 ms depending on where the burst lands. #994 made exactly this trade once already
// (+29 ms max ITL for +4.3% at c32) and documented it; this takes the same trade to the end of
// the curve.
//
// Narrow concurrencies cannot see any of it: `deep_ramp` requires (pending + have) * 2 >= width,
// i.e. 16 in flight, so c1/c2/c4/c8 keep allow == 1 and the previous scheduler byte for byte.
// Measured, same binary, admit/step 2 -> 32: c4 345.3 -> 345.5, c8 538.0 -> 538.5.
int prefills_per_step() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_PREFILLS_PER_STEP");
        const int x = e ? atoi(e) : packed_decode_width();
        return x < 1 ? 1 : x;
    }();
    return v;
}
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
        // Admitting ONE prefill per iteration is right once the decode batch is wide, and wrong
        // while it is still filling. A packed decode step on this runtime costs a fixed weight
        // read plus a small per-row term -- measured on RTX 5090 / Qwen3.8-27B-NVFP4 at
        // concurrency 32, 14.85 ms fixed against 0.41 ms per row -- so a step at four rows costs
        // 97% of what a step at thirty-two costs. Ramping one row per iteration therefore pays a
        // full weight read for a handful of tokens, thirty-one times, and because the rows START
        // staggered they FINISH staggered and the same waste is paid again as a drain tail.
        //
        // So: while the decode batch is still narrower than the width a packed step serves for
        // that flat cost, admitting another prefill costs nothing it was not already paying.
        // At and above that width this reverts to exactly one, which is what protects ITL in
        // steady state -- the regime the original rule was written for.
        // How many, though, is a trade, and TWO is where it settles. Each extra prefill admitted
        // into one iteration adds almost exactly one prefill's duration to the worst inter-token
        // gap, because they run back to back ahead of the next decode step. Measured at
        // concurrency 32, decode_tokens=8200 on both arms:
        //
        //   admitted   agg tok/s   mean ITL   max ITL
        //   1 (before)     981.0     27.96      75.5
        //   2            1023.6     27.82     104.0
        //   4           ~1037       27.9      ~155
        //   8           ~1045       27.6      ~293
        //
        // Throughput saturates by two while the tail keeps climbing ~28 ms per admission, so the
        // aggressive settings buy 1% for another 150 ms of worst-case latency -- the same ITL
        // spike prefill_mix_max_tokens() above exists to prevent. A width-scaled taper (8 when
        // empty, down to 1) was tried and is strictly worse than a flat two on both axes: 1025.7
        // tok/s at 206.7 ms max ITL. Steady-state ITL is untouched either way, which is the point:
        // this shortens the ramp, it does not change the step.
        //
        // The second admission also has to EARN its ~25 ms, and it only does when the ramp is
        // deep. That cost is one prefill and is therefore flat in concurrency, while the saving
        // grows with how many rows have to be filled -- measured, before -> after, every run at
        // the full decode_tokens:
        //
        //   concurrency    2       4       8      16      32
        //   agg tok/s   +0.8%   +1.1%   +0.5%   +3.4%   +4.3%
        //   max ITL      none   +23ms   +24ms   +29ms   +29ms
        //
        // At 4 and 8 that is nearly the whole latency price for almost none of the benefit, so
        // require a deep ramp before widening. Below it this is bit-for-bit the previous
        // scheduler.
        //
        // "Deep" has to be measured against the batch this load will EVENTUALLY reach -- pending
        // plus already-decoding -- not against the pending queue alone. The queue drains as the
        // ramp proceeds, so a bare `pending` test closes the gate partway up and gives most of
        // the saving back: at concurrency 16 it shut after two admissions and the gain fell from
        // +3.4% to +0.1%. The sum is invariant across the ramp, which is the property wanted.
        int pending = 0;
        for (const ScheduledSequence* s : ordered)
            if (s->phase == SeqPhase::PREFILL) ++pending;
        const int wide = packed_decode_width();
        const int have = (int)batch.decode_request_ids.size();
        // ...and "deep" has to be measured against the batch this load will reach, not against
        // the packed-decode ceiling. Fixed at packed_decode_width() the test needs sixteen rows
        // in flight, which no concurrency below sixteen can ever produce -- so c2/c4/c8 keep
        // allow == 1 and ramp one row per iteration however deep their backlog is, which is the
        // exact regime the rule above was written to fix, unreachable by construction. The cost
        // of the extra admission is one prefill and is flat in concurrency; the saving is the
        // iterations not spent at a narrow width, and a narrow step is not cheap -- on Muse
        // Glimmer a packed step is 16.73 ms of fixed weight read against 0.125 ms per row, so a
        // two-row step costs 95% of a sixteen-row one.
        //
        // Measured on Muse Glimmer, box19, the bot's own invocation
        // (qwen3_gguf_cb_bench <model> C 256 256 512), agg_tok_s, two interleaved repeats per
        // arm out of ONE binary:
        //
        //   concurrency        2         4          8
        //   before        169.7     234.3      425.8
        //   after         170.4     236.3      437.0   (+2.6% at 8, +0.9% at 4, +0.4% at 2)
        //
        // and mean ITL FALLS at eight (17.71 -> 17.54 ms), so the step is not paying for it.
        // c16/c32 already satisfied the old test and are byte-for-byte unchanged, which is also
        // why every DSpark concurrency dim (c16/c32) is untouched.
        // SPARKINFER_CB_RAMP_DEEP_ROWS=32 restores the previous threshold exactly.
        static const int deep_rows = [] {
            const char* e = getenv("SPARKINFER_CB_RAMP_DEEP_ROWS");
            const int x = e ? atoi(e) : 2;
            return x < 1 ? 1 : x;
        }();
        const bool deep_ramp = (pending + have) * 2 >= deep_rows;
        const int allow = (have < wide && deep_ramp) ? prefills_per_step() : 1;
        int taken = 0;
        for (const ScheduledSequence* s : ordered) {
            if (s->phase != SeqPhase::PREFILL) continue;
            // Defer large atomic prefills while decode is in flight.
            if (!batch.decode_request_ids.empty() && mix_max > 0 &&
                s->prefill_remaining > mix_max) {
                continue;
            }
            batch.prefill_request_ids.push_back(s->request_id);
            batch.total_tokens += 1;
            if (++taken >= allow || batch.total_tokens >= budget) break;
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
