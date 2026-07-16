#!/usr/bin/env bash
# End-to-end Polaris test: GPU eval on SSH box (judge.py) + TDX attest on bot host.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
[ -f .env.eval ] && set -a && source .env.eval && set +a
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/speedy}"
LOG="${POLARIS_TEST_LOG:-/tmp/sparkinfer_polaris_test.log}"

echo "[$(date -u +%FT%TZ)] Polaris pipeline test → $LOG"
python3 eval/vast_eval.py \
  --ssh "${EVAL_SSH_HOST}:${EVAL_SSH_PORT}" \
  --dual --polaris \
  --ref origin/main \
  --frontier 0 --ceiling "${CEILING:-366}" \
  --eval-mode longctx \
  --keep \
  2>&1 | tee "$LOG"

# If eval succeeded, run TDX attest on the POLARIS_ATTESTATION line (same as pr_eval_bot).
python3 - "$LOG" <<'PY'
import json, hashlib, sys, os
log = open(sys.argv[1]).read()
polaris_line = next((l for l in log.splitlines() if l.startswith("POLARIS_ATTESTATION ")), None)
result_line = next((l for l in log.splitlines() if l.startswith("RESULT_JSON ")), None)
if not polaris_line or not result_line:
    print("!! missing POLARIS_ATTESTATION or RESULT_JSON in log"); sys.exit(1)
attestation = json.loads(polaris_line[len("POLARIS_ATTESTATION "):])
res = json.loads(result_line[len("RESULT_JSON "):])
from eval.polaris.client import PolarisClient
from eval.polaris.receipt import build_polaris_receipt
pub = ""
for line in open("eval/polaris/sparkinfer_eval.pub"):
    line = line.strip()
    if line and not line.startswith("#"):
        pub = line; break
nonce_input = (
    attestation.get("code", {}).get("commit", "") +
    attestation.get("references", {}).get("model_sha256", "") +
    attestation.get("references", {}).get("eval_seed", "")
).encode()
nonce = hashlib.sha256(nonce_input).hexdigest()[:64]
client = PolarisClient(os.environ.get("POLARIS_API_KEY", ""))
polaris_resp = client.attest_scoring(attestation.get("measurements", {}), nonce, pub)
receipt = build_polaris_receipt(polaris_resp, attestation)
intel = polaris_resp.get("verification", {}).get("intel_verified")
print(f">> Polaris pipeline OK: intel_verified={intel} receipt_id={receipt.get('receipt_id','?')[:16]}")
with open("/tmp/polaris_test_receipt.json", "w") as f:
    json.dump(receipt, f, indent=2)
print(">> receipt saved to /tmp/polaris_test_receipt.json")
PY
