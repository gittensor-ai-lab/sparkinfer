#!/usr/bin/env python3
"""sparkinfer (DSpark) vs SGLang on the SAME checkpoint, per workload and context length.

Both engines are given the IDENTICAL token ids and asked for the same number of greedy tokens, so
the only difference measured is the engine. Prompts come from workloads.py, which pads each
workload with more of its own kind -- acceptance rate depends on how predictable the next tokens
are, so a single averaged prompt would hide the whole effect being compared.

sparkinfer is measured with dspark_tau_check, the same binary the hourly eval scores, which also
reports tau and verifies losslessness against its own AR reference in the same process. A
speculative number that is not lossless is not a speedup, so a run that loses losslessness is
reported as such rather than as a throughput win.

SGLang is driven over /generate with input_ids, bypassing its tokenizer: re-tokenizing the same
text in two engines can differ by a token or two at the seams, and at ctx=32k that is a different
prompt, not a rounding error.

ONE engine per run. The two models are ~20 and ~28 GB on a 32 GB card, so they cannot both be
resident: measure sglang with the server up, stop it, measure sparkinfer, then join the files.

Usage: compare_sglang.py --engine sglang     --out sgl.json  ... (server running)
       compare_sglang.py --engine sparkinfer --out si.json   ... (server stopped)
"""
import argparse, json, re, statistics, subprocess, sys, time, urllib.request

BIN = "/workspace/eval/bot_repo/build/runtime/dspark_tau_check"
GEN = "/workspace/eval/workloads.py"


def prompt_ids(tokenizer, workload, ctx):
    out = subprocess.run([sys.executable, GEN, tokenizer, workload, str(ctx)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"workload gen failed: {out.stderr.strip()[:200]}")
    return [int(x) for x in out.stdout.split()]


def run_sparkinfer(target, draft, ids, new):
    """Returns dict with ar_tps, dspark_tps, tau, lossless, prefill_pp."""
    r = subprocess.run([BIN, target, draft, str(new)] + [str(i) for i in ids],
                       capture_output=True, text=True, timeout=3600)
    m = dict(re.findall(r"^METRIC (\w+) ([\d.]+)$", r.stdout, re.M))
    if not m:
        return {"error": (r.stdout + r.stderr)[-300:]}
    return {
        "ar_tps": float(m.get("AR_TPS", 0)),
        "dspark_tps": float(m.get("DSPARK_TPS", 0)),
        "tau": float(m.get("MEAN_ACCEPT", 0)),
        "prefill_pp": float(m.get("DSPARK_PREFILL_PP", 0)),
        "lossless": m.get("LOSSLESS") == "1",
    }


def _sgl_once(base, ids, new):
    body = json.dumps({
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": new, "temperature": 0.0, "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(base + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.loads(resp.read())
    dt = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    return dt, (meta.get("completion_tokens") or new), meta


def run_sglang(base, ids, new):
    """Decode-only throughput, isolated by a two-point measurement.

    A single /generate call times prefill + decode together. sparkinfer's DSPARK_TPS is
    decode-only (its decode_s excludes ttft), so comparing that against an e2e number would
    understate SGLang badly -- at ctx=4k a 32-token generation measured 48.7 tok/s e2e while
    almost all of that wall time was the 4096-token prefill.

    Two runs on the SAME prompt, one generating 1 token and one generating `new`, differ only by
    the extra decode steps: rate = (new - 1) / (t_new - t_1). Prefill, queueing and HTTP overhead
    are identical in both and cancel.
    """
    t1, n1, _ = _sgl_once(base, ids, 1)
    tn, nn, meta = _sgl_once(base, ids, new)
    steps = max(1, nn - n1)
    dt = tn - t1
    return {
        "prefill_s": t1,
        "decode_tps": steps / dt if dt > 0 else 0.0,
        "e2e_tps": nn / tn if tn > 0 else 0.0,
        "gen_tokens": nn,
        "spec_accept": meta.get("spec_accept_length"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sgl", default="http://127.0.0.1:30000")
    ap.add_argument("--workloads", default="chat,code,math,json,repetition,counting")
    ap.add_argument("--ctxs", default="4096,16384,32768")
    ap.add_argument("--new", type=int, default=64); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--engine", choices=["sparkinfer", "sglang"], required=True,
                    help="ONE engine per pass -- see below")
    ap.add_argument("--out", required=True, help="JSON results file for this pass")
    a = ap.parse_args()

    # ONE engine per invocation, by necessity. Both models are ~20-28 GB and the card is 32 GB,
    # so they cannot be resident at the same time: with SGLang up, sparkinfer's target load fails
    # outright ("[FAIL] target load"). Run sglang with the server up, stop it, then run
    # sparkinfer, and join the two result files afterwards.
    rows = []
    for ctx in [int(c) for c in a.ctxs.split(",")]:
        for w in a.workloads.split(","):
            ids = prompt_ids(a.tokenizer, w, ctx)
            row = {"workload": w, "ctx": ctx, "n_prompt": len(ids)}
            if a.engine == "sparkinfer":
                runs = [run_sparkinfer(a.target, a.draft, ids, a.new) for _ in range(a.reps)]
                ok = [x for x in runs if "error" not in x]
                if not ok:
                    print(f"  {w}@{ctx}: sparkinfer FAILED: {runs[0].get('error','')[:160]}",
                          file=sys.stderr)
                    continue
                best = max(ok, key=lambda x: x["dspark_tps"])
                row.update(si_ar=best["ar_tps"], si_dspark=best["dspark_tps"], si_tau=best["tau"],
                           si_prefill=best["prefill_pp"],
                           si_lossless=all(x["lossless"] for x in ok))
            else:
                try:
                    sg = [run_sglang(a.sgl, ids, a.new) for _ in range(a.reps)]
                    row.update(sgl_decode_tps=max(x["decode_tps"] for x in sg),
                               sgl_e2e_tps=max(x["e2e_tps"] for x in sg),
                               sgl_prefill_s=min(x["prefill_s"] for x in sg),
                               sgl_accept=sg[0].get("spec_accept"))
                except Exception as e:
                    print(f"  {w}@{ctx}: sglang FAILED: {str(e)[:160]}", file=sys.stderr)
                    continue
            rows.append(row)
            print(f"  done {a.engine} {w}@{ctx}", file=sys.stderr)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {a.out} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
