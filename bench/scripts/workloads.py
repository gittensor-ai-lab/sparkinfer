#!/usr/bin/env python3
"""Generate per-workload prompts at a target context length, as token ids.

Six workloads chosen because they span the range of what speculative decoding can exploit: how
PREDICTABLE the next tokens are given the last ones. That is what acceptance length (tau) measures,
so a single "average" prompt hides the thing being compared -- counting and chat differ by more
than any engine difference does.

    chat        free-form dialogue          least predictable
    code        structured source
    math        worked arithmetic
    json        rigid record structure
    repetition  near-identical lines
    counting    fully determined            most predictable

Context is padded with MORE OF THE SAME workload, never with filler: a chat prompt padded with
JSON would report chat's generation against JSON's KV, and the acceptance rate would belong to
neither. Padding is truncated to land within a few tokens of the target.

Usage: workloads.py <tokenizer.json> <workload> <ctx_tokens> [--text]
"""
import argparse, json, sys
from tokenizers import Tokenizer

TURNS = [
    ("What's a good way to structure a small CLI tool?", "Start with argument parsing separated from the work itself. Keep a thin main that resolves flags, then hand a plain data object to a function that does the job and returns a result. That keeps the logic testable without spawning a process."),
    ("How do I handle configuration?", "Layer it: built-in defaults, then a config file, then environment variables, then flags, each overriding the last. Resolve once at startup into a single struct and pass that around, rather than reading the environment from deep inside the code."),
    ("What about logging?", "Log to stderr, print results to stdout, so the tool composes in a pipeline. Use levels sparingly -- info for what a user asked for, debug for what you needed while writing it."),
    ("Any advice on errors?", "Fail with the specific thing that went wrong and what was expected. An error that says which file, which field, and which value is worth ten that say the operation failed."),
]
CODE = '''def resolve_config(defaults, file_cfg, env, flags):
    """Later layers override earlier ones; each layer is a plain dict."""
    out = dict(defaults)
    for layer in (file_cfg, env, flags):
        for key, value in layer.items():
            if value is not None:
                out[key] = value
    return out


def validate(cfg, schema):
    for key, kind in schema.items():
        if key not in cfg:
            raise KeyError(f"missing required config key: {key}")
        if not isinstance(cfg[key], kind):
            raise TypeError(f"{key}: expected {kind.__name__}, got {type(cfg[key]).__name__}")
    return cfg

'''
MATH = """Step {i}: compute {a} * {b} + {c}.
  {a} * {b} = {ab}
  {ab} + {c} = {r}
  Carry {r} into the next step.
"""
JSONREC = '  {{"id": {i}, "name": "record-{i}", "score": {s}, "active": {act}, "tags": ["alpha", "beta"]}},\n'
# Repetition means REPEATED, verbatim. The earlier version varied the request id and the latency
# on every line, so the next tokens were only mostly predictable and tau stalled around 3.6 against
# a block-size ceiling of 7. A log line emitted identically -- a health check, a heartbeat -- is
# both a real serving pattern and an actual test of the ceiling.
REPEAT = "The service returned status ok for the health check in region us-east-1 with latency 21 ms.\n"


def build(workload, min_chars):
    """Return (context_text, instruction). Context is repeated until it is long enough."""
    parts = []
    i = 0
    while sum(len(p) for p in parts) < min_chars:
        i += 1
        if workload == "chat":
            q, a = TURNS[i % len(TURNS)]
            # Plain role markers, NOT the template's own <|im_start|> tokens: those are special
            # ids, and decoding then re-encoding the trimmed body does not round-trip them, which
            # left chat ~7% short of its context target while every other workload hit it exactly.
            # Nesting real template tokens inside the user turn would also be a prompt the model
            # never sees in practice.
            parts.append(f"User: {q}\nAssistant: {a}\n\n")
        elif workload == "code":
            parts.append(CODE.replace("resolve_config", f"resolve_config_{i}").replace("validate", f"validate_{i}"))
        elif workload == "math":
            a, b, c = 12 + i, 7 + (i % 9), 3 * i
            parts.append(MATH.format(i=i, a=a, b=b, c=c, ab=a * b, r=a * b + c))
        elif workload == "json":
            parts.append(JSONREC.format(i=i, s=round((i * 37) % 100 / 10, 1), act="true" if i % 2 else "false"))
        elif workload == "repetition":
            parts.append(REPEAT)
        elif workload == "counting":
            # Cycle 1..100 rather than counting away to infinity. Counting up without bound reaches
            # four- and five-digit numbers within a 4k prompt, and each of those costs several
            # tokens whose split is not obvious -- which is why tau capped near 3.9. A bounded
            # cycle keeps every number one or two tokens and the sequence exactly determined, so
            # this measures the speculative ceiling instead of the tokenizer.
            parts.append(f"{(i - 1) % 100 + 1}, ")
        else:
            sys.exit(f"unknown workload: {workload}")
    ctx = "".join(parts)
    # Ask for enough output to fill the measured window. A prompt that is satisfied in 20 tokens
    # ends at EOS and the throughput is then computed over a fraction of a second, which is noise.
    instr = {
        "chat": "Continue this conversation with the next assistant reply, at length.",
        "code": "Continue this module with several more functions, in the same style.",
        "math": "Continue the calculation for many more steps, same format.",
        "json": "Continue this JSON array with many more records in the same shape.",
        "repetition": "Continue these log lines in exactly the same format, for at least 40 lines.",
        "counting": "Continue this sequence of numbers for at least 200 more numbers.",
    }[workload]
    return ctx, instr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer"); ap.add_argument("workload"); ap.add_argument("ctx", type=int)
    ap.add_argument("--text", action="store_true", help="print the prompt text instead of ids")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)

    # Overshoot on characters, then trim on TOKENS -- the ratio differs per workload (counting is
    # ~2 chars/token, prose ~4), so a character target alone lands far off for some of them.
    ctx_text, instr = build(a.workload, a.ctx * 6)
    head = "<|im_start|>user\n"
    # enable_thinking=False, via the empty think block the template emits for it.
    #
    # This is the single largest methodology choice here. With thinking ON, most of a 128-token
    # window is <think> reasoning prose -- unpredictable by construction -- so tau measures the
    # reasoning, not the workload: repetition stalls at 4.03 with thinking and reaches 6.667
    # (against a block-size ceiling of 7) without it. Speculative decoding is being asked how well
    # the draft predicts the TASK, so the task is what the model has to be emitting.
    tail = f"\n\n{instr}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    fixed = len(tok.encode(head + tail, add_special_tokens=False).ids)
    all_body = tok.encode(ctx_text, add_special_tokens=False).ids
    # Trim to AT MOST ctx, never one over. Decoding a token slice and re-encoding the assembled
    # prompt does not round-trip exactly, so a single budget calculation lands a token or two off.
    # That is harmless at 4k/16k but not at the top: a 32769-token prompt crosses max_seq, and
    # sparkinfer then falls back from batched to sequential prefill (9936 -> 163 pp/s) AND stops
    # being lossless (matched 19/64 instead of 64/64). Measuring that would report a broken path
    # as if it were a slow one.
    def _clean(text):
        # End the context on a natural boundary. Trimming to a token budget lands mid-item: the
        # 16k counting prompt ended '..., 72, 73, 7' -- a number cut in half -- and the model then
        # resumes from a broken token, which is not what this workload is meant to measure. It
        # cost tau 7.5 -> 4.03, reproducible to three decimals across reps, so it read as a real
        # context-scaling effect rather than a truncation artefact.
        for sep in ("\n", ", "):
            cut = text.rfind(sep)
            if cut > len(text) // 2:
                return text[:cut + len(sep)]
        return text

    budget = max(1, a.ctx - fixed)
    for _ in range(8):
        prompt = head + _clean(tok.decode(all_body[:budget])) + tail
        ids = tok.encode(prompt, add_special_tokens=False).ids
        if len(ids) <= a.ctx:
            break
        budget -= (len(ids) - a.ctx)
    if a.text:
        print(prompt)
    else:
        print(" ".join(map(str, ids)))
    print(f"# workload={a.workload} target_ctx={a.ctx} actual={len(ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
