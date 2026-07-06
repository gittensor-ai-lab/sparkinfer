#!/usr/bin/env python3
"""Fetch real benchmark datasets from HuggingFace, convert to this suite's JSONL schema,
and take a stratified ~10% dev sample (small enough to iterate, real enough to be meaningful).

  pip install datasets
  python3 fetch.py                      # all 5, ~10% each (seeded, stratified)
  python3 fetch.py --fraction 0.1 --max-items 300 --min-items 20 --seed 42
  python3 fetch.py --benchmarks gsm8k,mmlu_pro

Sources: gsm8k (openai/gsm8k), mmlu_pro (TIGER-Lab/MMLU-Pro, stratified by category),
humaneval (openai/openai_humaneval), ifeval (google/IFEval, instructions mapped to our
checks), bfcl (NousResearch/hermes-function-calling-v1 real tool calls with ground truth).
Each adapter fails independently, leaving that benchmark's existing file untouched.
"""
import argparse, json, os, random, re
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))


def _ld(*a, **k):
    from datasets import load_dataset
    try:
        return load_dataset(*a, **k)
    except Exception:
        return load_dataset(*a, trust_remote_code=True, **k)


# - adapters: return list of schema rows -

def fetch_gsm8k():
    try: ds = _ld("openai/gsm8k", "main", split="test")
    except Exception: ds = _ld("gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(ds):
        m = re.search(r"####\s*(-?[0-9,]+)", r["answer"])
        if m:
            out.append({"id": f"gsm8k-{i}", "benchmark": "gsm8k",
                        "prompt": r["question"], "target": int(m.group(1).replace(",", ""))})
    return out, None


def fetch_mmlu_pro():
    ds = _ld("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for i, r in enumerate(ds):
        out.append({"id": f"mmlupro-{i}", "benchmark": "mmlu_pro", "prompt": r["question"],
                    "choices": list(r["options"]), "answer": str(r["answer"]).strip().upper(),
                    "meta": {"category": r.get("category")}})
    return out, (lambda r: r["meta"]["category"])   # stratify by subject


def fetch_humaneval():
    ds = _ld("openai/openai_humaneval", split="test")
    out = [{"id": r["task_id"].replace("/", "-"), "benchmark": "humaneval",
            "prompt": r["prompt"], "entry_point": r["entry_point"], "test": r["test"]} for r in ds]
    return out, None


# IFEval instruction_id -> our machine-checkable constraint (None => unsupported).
def _ifeval_map(iid, kw):
    kw = kw or {}
    def words():
        n, rel = kw.get("num_words", 0), (kw.get("relation") or "")
        return {"type": "word_count_min", "count": n} if "least" in rel else {"type": "word_count_max", "count": max(0, n - 1)}
    def sents():
        n, rel = kw.get("num_sentences", 0), (kw.get("relation") or "")
        return {"type": "sentence_count_min", "count": n} if "least" in rel else {"type": "sentence_count_max", "count": max(0, n - 1)}
    table = {
        "keywords:existence":       lambda: {"type": "keywords_include", "keywords": kw.get("keywords", [])},
        "keywords:forbidden_words": lambda: {"type": "keywords_forbidden", "keywords": kw.get("forbidden_words", [])},
        "length_constraints:number_words":      words,
        "length_constraints:number_sentences":  sents,
        "length_constraints:number_paragraphs": lambda: {"type": "paragraph_count", "count": kw.get("num_paragraphs", 1)},
        "detectable_content:number_placeholders": lambda: {"type": "placeholders_min", "count": kw.get("num_placeholders", 1)},
        "detectable_content:postscript":          lambda: {"type": "postscript", "marker": kw.get("postscript_marker", "P.S.")},
        "detectable_format:number_bullet_lists":  lambda: {"type": "bullet_count", "count": kw.get("num_bullets", 1)},
        "detectable_format:number_highlighted_sections": lambda: {"type": "highlighted_min", "count": kw.get("num_highlights", 1)},
        "detectable_format:json_format": lambda: {"type": "json_format"},
        "detectable_format:title":       lambda: {"type": "title_present"},
        "detectable_format:multiple_sections": lambda: {"type": "multiple_sections", "marker": kw.get("section_spliter", "SECTION"), "count": kw.get("num_sections", 2)},
        "startend:end_checker": lambda: {"type": "ends_with", "suffix": kw.get("end_phrase", "")},
        "startend:quotation":   lambda: {"type": "quotation"},
        "change_case:english_capital":   lambda: {"type": "all_uppercase"},
        "change_case:english_lowercase": lambda: {"type": "all_lowercase"},
        "punctuation:no_comma":          lambda: {"type": "no_commas"},
    }
    fn = table.get(iid)
    return fn() if fn else None


def fetch_ifeval():
    try: ds = _ld("google/IFEval", split="train")
    except Exception: ds = _ld("HuggingFaceH4/ifeval", split="train")
    out = []
    for r in ds:
        ids, kws = r["instruction_id_list"], r.get("kwargs") or [{}] * len(r["instruction_id_list"])
        mapped = [_ifeval_map(iid, kw) for iid, kw in zip(ids, kws)]
        if mapped and all(m is not None for m in mapped):        # keep only fully-checkable rows
            out.append({"id": f"ifeval-{r['key']}", "benchmark": "ifeval",
                        "prompt": r["prompt"], "instructions": mapped})
    return out, None


def _maybe(v):
    import ast
    if not isinstance(v, str):
        return v
    try: return json.loads(v)
    except Exception:
        try: return ast.literal_eval(v)
        except Exception: return None


SENSITIVE_RE = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|auth[_ -]?token|bearer[_ -]?token|"
    r"client[_ -]?secret|connection string|connstr|oauth|password|refresh[_ -]?token|"
    r"secret|token|credential)"
)


def _has_sensitive_text(obj):
    return bool(SENSITIVE_RE.search(json.dumps(obj, sort_keys=True) if not isinstance(obj, str) else obj))


def fetch_bfcl():
    # Real tool-calling with structured ground truth. True BFCL answers aren't loadable on HF
    # and xLAM-60k is gated. Hermes embeds parseable <tool_call> ground truth and tool schemas.
    # Rows whose prompts, tools, or targets mention credential-like fields are skipped so the
    # checked-in sample avoids fake secrets and secret-scanner noise.
    ds = _ld("NousResearch/hermes-function-calling-v1", split="train")
    out = []
    for i, r in enumerate(ds):
        conv, tools = _maybe(r["conversations"]), _maybe(r["tools"])
        if not isinstance(conv, list) or not isinstance(tools, list):
            continue
        user = next((t.get("value") for t in conv if t.get("from") in ("human", "user")), None)
        calls = []
        for t in conv:
            for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", t.get("value", "") or "", re.S):
                try: calls.append(json.loads(m))
                except Exception: pass
        if not user or len(calls) != 1 or not calls[0].get("name"):
            continue
        args = calls[0].get("arguments", {})
        args = _maybe(args) if isinstance(args, str) else args
        target = {"name": calls[0]["name"], "arguments": args if isinstance(args, dict) else {}}
        if _has_sensitive_text(user) or _has_sensitive_text(tools) or _has_sensitive_text(target):
            continue
        out.append({"id": f"bfcl-{i}", "benchmark": "bfcl", "prompt": user, "tools": tools,
                    "target": target,
                    "meta": {"source": "hermes-function-calling-v1"}})
    return out, (lambda r: r["target"]["name"])      # stratify by tool


ADAPTERS = {"gsm8k": fetch_gsm8k, "mmlu_pro": fetch_mmlu_pro, "humaneval": fetch_humaneval,
            "ifeval": fetch_ifeval, "bfcl": fetch_bfcl}


# - stratified sampler -

def sample(rows, frac, seed, lo, hi, key=None):
    rnd = random.Random(seed)
    target = max(lo, min(hi, len(rows), int(round(len(rows) * frac))))
    if key is None:
        pick = rows[:]; rnd.shuffle(pick); return pick[:target]
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows: groups[key(r)].append(r)
    out = []
    for g in sorted(groups, key=str):
        items = groups[g][:]; random.Random(f"{seed}:{g}").shuffle(items)
        out += items[:max(1, round(target * len(items) / len(rows)))]
    rnd.shuffle(out)
    return out[:hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="gsm8k,mmlu_pro,humaneval,ifeval,bfcl")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--min-items", type=int, default=20)
    ap.add_argument("--max-items", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for b in [x for x in args.benchmarks.split(",") if x]:
        try:
            rows, key = ADAPTERS[b]()
            picked = sample(rows, args.fraction, args.seed, args.min_items, args.max_items, key)
            path = os.path.join(args.out, f"{b}.jsonl")
            with open(path, "w") as f:
                for r in picked: f.write(json.dumps(r) + "\n")
            print(f"  {b:10s} {len(picked):4d} / {len(rows):5d} real items  -> {path}")
        except Exception as e:
            print(f"  {b:10s} FAILED ({str(e)[:80]}) - kept existing file")


if __name__ == "__main__":
    main()
