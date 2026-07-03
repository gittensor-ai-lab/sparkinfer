#!/usr/bin/env python3
"""Programmatic scorers for the LLM-quality benchmark suite.

Each scorer takes (item, output_text) and returns a dict:
    {"score": float in [0,1], "pass": bool, "detail": str}

They are deterministic and dependency-free (stdlib only) so the same code scores
sparkinfer and llama.cpp identically - quality parity is then just a diff of scores.

Run `python3 scorers.py` to self-test the scorers on crafted cases (no model/GPU needed).
"""
import re, json, ast, sys, subprocess, tempfile, os


# - helpers -

def _code_from_output(text):
    """Pull a python code block out of a model completion (```-fenced or bare)."""
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m[0]
    return text


def _last_number(text):
    """Final numeric answer: prefer '#### x' / 'answer is x', else the last number."""
    for pat in (r"####\s*(-?\$?[0-9][0-9,]*\.?[0-9]*)",
                r"(?:answer|total|result)\s*(?:is|:|=)?\s*\$?(-?[0-9][0-9,]*\.?[0-9]*)"):
        m = re.findall(pat, text, re.I)
        if m:
            return _to_num(m[-1])
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
    return _to_num(nums[-1]) if nums else None


def _to_num(s):
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def _extract_choice(text, n):
    """Chosen MCQ letter A..(A+n-1) from a completion, most-explicit patterns first."""
    letters = "ABCDEFGHIJKLMNOP"[:n]
    for pat in (r"answer\s*(?:is|:)?\s*\(?([A-P])\)?",
                r"\bthe answer is\s*\(?([A-P])\)?",
                r"\(([A-P])\)",
                r"(?:^|\n)\s*([A-P])[\).\s]"):
        m = re.findall(pat, text, re.I)
        for c in reversed(m):
            if c.upper() in letters:
                return c.upper()
    # fallback: last standalone letter in range
    for tok in reversed(re.findall(r"[A-P]", text.upper())):
        if tok in letters:
            return tok
    return None


def _first_json(text):
    """First balanced JSON object/array in text (models often wrap it in prose/fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_c:
                    depth += 1
                elif text[i] == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(open_c, start + 1)
    return None


def _result(passed, detail="", score=None):
    return {"score": float(passed) if score is None else float(score),
            "pass": bool(passed), "detail": detail}


# - GSM8K -

def score_gsm8k(item, output):
    got = _last_number(output)
    want = _to_num(str(item["target"]))
    ok = got is not None and want is not None and abs(got - want) < 1e-6
    return _result(ok, f"got={got} want={want}")


# - MMLU-Pro -

def score_mmlu_pro(item, output):
    n = len(item["choices"])
    got = _extract_choice(output, n)
    want = item["answer"].strip().upper()
    return _result(got == want, f"chose={got} want={want}")


# - IFEval -
# Each item carries a list of machine-checkable constraints. Strict = every one holds.

def _words(t):
    return re.findall(r"\b[\w']+\b", t)

def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]

def _bullets(t):
    return [l for l in t.splitlines() if re.match(r"\s*(?:[-*]|\d+\.)\s+", l)]


_IFEVAL_CHECKS = {
    "keywords_include":   lambda o, a: all(k.lower() in o.lower() for k in a["keywords"]),
    "keywords_forbidden": lambda o, a: all(k.lower() not in o.lower() for k in a["keywords"]),
    "forbidden_letter":   lambda o, a: a["letter"].lower() not in o.lower(),
    "word_count_exact":   lambda o, a: len(_words(o)) == a["count"],
    "word_count_min":     lambda o, a: len(_words(o)) >= a["count"],
    "word_count_max":     lambda o, a: len(_words(o)) <= a["count"],
    "sentence_count":     lambda o, a: len(_sentences(o)) == a["count"],
    "sentence_count_min": lambda o, a: len(_sentences(o)) >= a["count"],
    "sentence_count_max": lambda o, a: len(_sentences(o)) <= a["count"],
    "bullet_count":       lambda o, a: len(_bullets(o)) == a["count"],
    "highlighted_min":    lambda o, a: len(re.findall(r"\*+[^*\n]+\*+", o)) >= a["count"],
    "postscript":         lambda o, a: a["marker"].lower() in o.lower(),
    "quotation":          lambda o, a: o.strip().startswith('"') and o.strip().endswith('"'),
    "multiple_sections":  lambda o, a: len([s for s in o.split(a["marker"]) if s.strip()]) >= a["count"],
    "paragraph_count":    lambda o, a: len([p for p in re.split(r"\n\s*\n", o.strip()) if p]) == a["count"],
    "starts_with":        lambda o, a: o.strip().startswith(a["prefix"]),
    "ends_with":          lambda o, a: o.strip().endswith(a["suffix"]),
    "all_uppercase":      lambda o, a: o.upper() == o and any(c.isalpha() for c in o),
    "all_lowercase":      lambda o, a: o.lower() == o and any(c.isalpha() for c in o),
    "no_commas":          lambda o, a: "," not in o,
    "json_format":        lambda o, a: _first_json(o) is not None,
    "title_present":      lambda o, a: bool(re.search(r"<<.+?>>", o)),
    "placeholders_min":   lambda o, a: len(re.findall(r"\[.+?\]", o)) >= a["count"],
}

def score_ifeval(item, output):
    checks = item["instructions"]
    results = []
    for c in checks:
        fn = _IFEVAL_CHECKS.get(c["type"])
        results.append(bool(fn(output, c)) if fn else False)
    frac = sum(results) / len(results) if results else 0.0
    strict = all(results) and bool(results)
    detail = ", ".join(f"{c['type']}={'ok' if r else 'X'}" for c, r in zip(checks, results))
    return {"score": frac, "pass": strict, "detail": detail}


# - HumanEval -

def score_humaneval(item, output, timeout=8):
    """pass@1: assemble prompt + completion + hidden tests, run in an isolated subprocess."""
    body = _code_from_output(output)
    # If the model repeated the signature, use its code as-is; else graft onto the stub.
    program = (body if item["entry_point"] in body and "def " in body
               else item["prompt"] + "\n" + body)
    program += "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    try:
        r = subprocess.run([sys.executable, "-c", program],
                           capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        return _result(ok, "passed" if ok else f"fail: {r.stderr.strip().splitlines()[-1:]}")
    except subprocess.TimeoutExpired:
        return _result(False, "timeout")
    except Exception as e:  # noqa
        return _result(False, f"harness error: {e}")


# - BFCL (tool calling) -

def _norm(v):
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v

def score_bfcl(item, output):
    call = _first_json(output)
    if call is None:
        return _result(False, "no JSON tool call found")
    if isinstance(call, list):
        call = call[0] if call else {}
    name = call.get("tool") or call.get("name") or call.get("function")
    args = call.get("arguments") or call.get("parameters") or call.get("args") or {}
    tgt = item["target"]
    name_ok = _norm(name) == _norm(tgt["name"])
    # required args must match (normalized); extra args allowed unless strict.
    want = _norm(tgt["arguments"])
    got = _norm(args if isinstance(args, dict) else {})
    args_ok = all(k in got and got[k] == v for k, v in want.items())
    ok = name_ok and args_ok
    return _result(ok, f"name={name}({'ok' if name_ok else 'X'}) args={'ok' if args_ok else 'X'}")


SCORERS = {
    "gsm8k": score_gsm8k, "mmlu_pro": score_mmlu_pro, "ifeval": score_ifeval,
    "humaneval": score_humaneval, "bfcl": score_bfcl,
}


# - self-test -

def _selftest():
    cases = [
        # (benchmark, item, output, expected_pass)
        ("gsm8k", {"target": 16}, "He has 12-5=7, then 7+9 = 16. #### 16", True),
        ("gsm8k", {"target": 16}, "The answer is 15.", False),
        ("mmlu_pro", {"choices": list("abcd"), "answer": "B"}, "It is O(n log n), so the answer is (B).", True),
        ("mmlu_pro", {"choices": list("abcd"), "answer": "B"}, "The answer is C.", False),
        ("ifeval", {"instructions": [
            {"type": "bullet_count", "count": 3},
            {"type": "forbidden_letter", "letter": "z"},
            {"type": "word_count_max", "count": 40}]},
            "- one two three\n- four five six\n- seven eight nine", True),
        ("ifeval", {"instructions": [{"type": "bullet_count", "count": 3}]},
            "- only one bullet", False),
        ("humaneval", {"entry_point": "add", "prompt": "def add(a, b):",
                       "test": "def check(f):\n    assert f(2,3)==5\n    assert f(-1,1)==0"},
            "def add(a, b):\n    return a + b", True),
        ("humaneval", {"entry_point": "add", "prompt": "def add(a, b):",
                       "test": "def check(f):\n    assert f(2,3)==5"},
            "def add(a, b):\n    return a - b", False),
        ("bfcl", {"target": {"name": "book_flight", "arguments": {"from": "Los Angeles", "to": "Tokyo"}}},
            'Sure: {"tool": "book_flight", "arguments": {"from": "Los Angeles", "to": "Tokyo"}}', True),
        ("bfcl", {"target": {"name": "book_flight", "arguments": {"from": "LA", "to": "Tokyo"}}},
            '{"tool": "get_weather", "arguments": {"city": "LA"}}', False),
    ]
    ok = 0
    for bm, item, out, want in cases:
        r = SCORERS[bm](item, out)
        good = r["pass"] == want
        ok += good
        print(f"  [{'PASS' if good else 'FAIL'}] {bm:10s} expect_pass={want!s:5s} -> {r}")
    print(f"\n{ok}/{len(cases)} scorer self-tests passed")
    return ok == len(cases)


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
