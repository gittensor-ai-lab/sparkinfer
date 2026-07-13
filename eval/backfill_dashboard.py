#!/usr/bin/env python3
"""Apply a stored eval RESULT_JSON to dashboard/data.json (no git push).

Usage:
  python3 eval/backfill_dashboard.py --pr 294 --run 0294-cf2fd83
  python3 eval/backfill_dashboard.py --pr 294 --result path/to/result.json [--merge]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import pr_eval_bot as bot

LOG_BASE = "https://raw.githubusercontent.com/gittensor-ai-lab/sparkinfer-log/main/runs"


def _parse_result_log(text: str) -> dict:
    for line in text.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    raise ValueError("RESULT_JSON not found in log")


def _polaris_fields(run: str | None) -> dict:
    """Attach dashboard polaris receipt links when sparkinfer-log has a TDX receipt."""
    if not run:
        return {}
    try:
        with urllib.request.urlopen(f"{LOG_BASE}/{run}/receipt.json", timeout=60) as resp:
            receipt = json.load(resp)
    except Exception:
        return {}
    if not (receipt.get("tdx") or receipt.get("attestation_type") == "tdx-quote"):
        return {}
    rid = receipt.get("receipt_id") or ""
    return {
        "polaris_receipt_url": f"https://gittensor-ai-lab.github.io/sparkinfer-log/?run={run}",
        "polaris_receipt_hash": rid[:16] if rid else None,
    }


def _load_result(run: str | None, result_path: str | None) -> dict:
    if result_path:
        with open(result_path) as f:
            return json.load(f)
    if not run:
        raise ValueError("pass --run or --result")
    url = f"{LOG_BASE}/{run}/log.txt"
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    try:
        return _parse_result_log(text)
    except ValueError:
        with urllib.request.urlopen(f"{LOG_BASE}/{run}/result.json", timeout=60) as resp:
            return json.load(resp)


def _areas_from_title(title: str) -> list[str]:
    m = re.match(r"^(\w+)", title or "")
    prefix = (m.group(1) if m else "").lower()
    areas = set()
    if prefix in ("perf", "feat", "fix"):
        if prefix in ("perf", "feat"):
            areas.update(("kernels", "runtime"))
        else:
            areas.add("runtime")
    return sorted(areas) or ["runtime"]


def apply(pr_num: int, res: dict, *, merge: bool, proof_run: str | None, title: str | None) -> None:
    repo = "gittensor-ai-lab/sparkinfer"
    pr_title = title or res.get("title") or f"PR #{pr_num}"
    pr = {"number": pr_num, "title": pr_title}
    areas = _areas_from_title(pr_title)
    proof = None
    if proof_run:
        proof = f"https://gittensor-ai-lab.github.io/sparkinfer-log/?run={proof_run}"
    elif res.get("id"):
        proof = f"https://gittensor-ai-lab.github.io/sparkinfer-log/?run={res['id']}"

    bot.push_dash = lambda _msg: None  # noqa: ARG005 — local backfill only
    res = {**res, **{k: v for k, v in _polaris_fields(proof_run or res.get("id")).items() if v}}
    bot.update_dashboard(repo, pr, areas, res, proof_url=proof)
    if merge:
        bot.record_merge(repo, pr_num)
    print(f">> dashboard updated for PR #{pr_num} (merge={merge})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill dashboard from eval log")
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--title", help="PR title for dashboard row")
    ap.add_argument("--run", help="eval log run id, e.g. 0294-cf2fd83")
    ap.add_argument("--result", help="local result.json path")
    ap.add_argument("--merge", action="store_true", help="also run record_merge (frontier + ctx)")
    args = ap.parse_args()
    res = _load_result(args.run, args.result)
    apply(args.pr, res, merge=args.merge, proof_run=args.run or res.get("id"), title=args.title)


if __name__ == "__main__":
    main()
