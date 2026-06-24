#!/usr/bin/env python3
"""Automatic evaluation on a vast.ai GPU: provision (or reuse) → build/correctness/speed → label → teardown.

Requires VAST_API_KEY (`vastai set api-key <key>`). The numeric label is computed on-box by
bench/scripts/label.py (deterministic) — this script only orchestrates.

  # reuse an existing box (started if stopped, STOPPED again after the eval — the default):
  python eval/vast_eval.py --reuse 42134865 --frontier 164 --ceiling 366 --ref main

  # evaluate then DESTROY (frees the disk too), or --keep to leave it running:
  python eval/vast_eval.py --ref <git-ref> --frontier 164 --ceiling 366 --destroy

By default the instance is STOPPED after every eval: compute billing pauses while the disk and
cached weights (/workspace/models) persist for a fast next run. --keep leaves it running.

Self-healing: on --reuse, if the box won't become SSH-ready within --reuse-timeout (default 5 min),
it is stopped and a fresh box is provisioned via the vast API automatically; the new id is saved to
~/.sparkinfer_vast_instance (VAST_INSTANCE_FILE) so the next run reuses it. --no-recreate disables this.

Env: VAST_API_KEY, SSH_KEY (default ~/.ssh/id_ed25519), LLAMACPP_DIR, EVAL_IMAGE, EVAL_REPO, VAST_INSTANCE_FILE.
"""
import argparse, json, os, subprocess, sys, time
from vastai import VastAI

REPO    = os.environ.get("EVAL_REPO",  "https://github.com/gittensor-ai-lab/sparkinfer")
IMAGE   = os.environ.get("EVAL_IMAGE", "nvidia/cuda:12.8.0-devel-ubuntu24.04")   # needs nvcc for sm_120
SSH_KEY = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_ed25519"))
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/workspace/.llamacpp")            # persists across stop/start
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))  # self-healed id

def sh(host, port, cmd, timeout=3600):
    return subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
         "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)

def info_of(v, iid):
    return next((i for i in v.show_instances() if i.get("id") == iid), None)

def endpoint(info):
    """Prefer the DIRECT endpoint (public_ipaddr + mapped :22) — the vast SSH proxy authenticates
    against account keys and is flakier; the direct port uses the instance's authorized_keys."""
    ip = info.get("public_ipaddr"); ports = info.get("ports") or {}
    m = ports.get("22/tcp")
    if ip and m:
        return ip.strip(), int(m[0]["HostPort"])
    return info.get("ssh_host"), int(info.get("ssh_port"))

def wait_ssh(host, port, tries=60):
    for _ in range(tries):
        try:
            if sh(host, port, "echo ok", timeout=15).stdout.strip().endswith("ok"): return True
        except Exception: pass
        time.sleep(10)
    return False

def save_instance(iid):
    try:
        with open(INSTANCE_FILE, "w") as f: f.write(str(iid))
    except Exception: pass

def funds():
    """Usable vast funds in USD = balance + CREDIT. Credit is spent first and is the field that
    actually matters — a $0 'balance' with positive credit can still rent. None if unreadable."""
    try:
        out = subprocess.run(["vastai", "show", "user", "--raw"], capture_output=True, text=True, timeout=30).stdout
        u = json.loads(out)
        return float(u.get("balance") or 0) + float(u.get("credit") or 0)
    except Exception:
        return None

def bring_up(v, iid, deadline_s):
    """Start the instance if needed and wait until SSH-reachable, within deadline_s.
    Returns (host, port), or None if it never comes up (treat the box as dead/stuck)."""
    info = info_of(v, iid)
    if not info:
        print(f">> instance {iid} not found"); return None
    if info.get("actual_status") != "running":
        print(f">> starting instance {iid} ...")
        try: v.start_instance(id=iid)
        except Exception as e: print("  start:", str(e)[:150])
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        info = info_of(v, iid)
        st = (info or {}).get("actual_status")
        if info and st == "running" and (info.get("public_ipaddr") or info.get("ssh_host")):
            host, port = endpoint(info)
            if wait_ssh(host, port, tries=4):     # box running; probe SSH (~40s), retry until deadline
                print(f">> instance {iid}: ssh root@{host}:{port}")
                return host, port
        else:
            print(f"  instance {iid}: status={st or '?'} — waiting ...")
        time.sleep(15)
    print(f">> instance {iid} did not become SSH-ready within {deadline_s}s")
    return None

def provision(v, args):
    """Create a fresh instance via the vast API. Returns the new instance id, or None."""
    offers = v.search_offers(query=f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 inet_down>=100",
                             order="dph_total", limit=10)
    if not offers:
        print(">> no matching offers"); return None
    off = offers[0]
    print(f">> creating instance on offer {off['id']} {off.get('gpu_name')} ${off.get('dph_total'):.3f}/hr")
    # The SDK's create_instance has no ssh/direct kwargs (those are CLI flags). Use the CLI so we
    # get the DIRECT SSH endpoint the rest of this script expects; --raw returns {success, new_contract}.
    out = subprocess.run(["vastai", "create", "instance", str(off["id"]), "--image", args.image,
                          "--disk", "120", "--ssh", "--direct", "--raw"],
                         capture_output=True, text=True, timeout=120).stdout
    try: res = json.loads(out)
    except Exception: print(">> create failed:", out[:300]); return None
    if not res.get("success"): print(">> create failed:", str(res)[:300]); return None
    return res.get("new_contract")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling",  type=float, default=0)
    ap.add_argument("--reuse", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="leave the instance running after eval (default: stop it)")
    ap.add_argument("--destroy", action="store_true", help="destroy after eval instead of stopping (also frees the disk)")
    ap.add_argument("--gpu", default="RTX_5090")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--reuse-timeout", type=int, default=300, help="seconds to wait for a reused box before recreating (default 300 = 5 min)")
    ap.add_argument("--new-timeout", type=int, default=900, help="seconds to wait for a freshly created box")
    ap.add_argument("--no-recreate", action="store_true", help="on reuse failure, error out instead of provisioning a new box")
    args = ap.parse_args()

    v = VastAI(); created = False; iid = args.reuse
    host = port = None
    bal = funds()
    if bal is not None: print(f">> vast funds: ${bal:.2f} (balance + credit)")

    # 1) Try to bring up the reused box within a bounded window (default 5 min).
    if iid:
        ep = bring_up(v, iid, args.reuse_timeout)
        if ep:
            host, port = ep
        elif args.no_recreate:
            sys.exit(f"instance {iid} never came up (--no-recreate)")
        else:
            # Stop scheduling on the dead/stuck box, then fall through to provision a fresh one.
            print(f">> reused instance {iid} is dead/stuck — stopping it and provisioning a new box")
            try: v.stop_instance(id=iid)
            except Exception as e: print("  stop:", str(e)[:150])
            iid = 0

    # 2) No working box yet → create one via the vast API and bring it up.
    if not iid:
        iid = provision(v, args)
        if not iid: sys.exit("could not provision an instance")
        created = True
        ep = bring_up(v, iid, args.new_timeout)    # fresh box: longer (provision + boot + first apt)
        if not ep:
            try: v.destroy_instance(id=iid)         # clean up a born-dead box
            except Exception: pass
            sys.exit(f"new instance {iid} never came up")
        host, port = ep

    save_instance(iid)                              # persist the working id (the bot reuses it next run)
    if args.reuse and iid != args.reuse:
        print(f"NEW_INSTANCE_ID {iid}")             # machine-readable for the bot
        print(f">> switched to fresh instance {iid} (old {args.reuse} stopped; destroy it if unneeded)")

    try:
        setup = ("export DEBIAN_FRONTEND=noninteractive; "
                 "command -v git >/dev/null || (apt-get update -q && apt-get install -y -q git curl cmake build-essential); "
                 "pip install -q --break-system-packages huggingface_hub tokenizers >/dev/null 2>&1 || true; "
                 f"if [ -d /root/sparkinfer/.git ]; then cd /root/sparkinfer && git fetch -q origin && git checkout -q {args.ref} && git pull -q origin {args.ref}; "
                 f"else git clone -q {REPO} /root/sparkinfer && cd /root/sparkinfer && git checkout -q {args.ref}; fi")
        if sh(host, port, setup, timeout=1800).returncode: print(">> setup warnings (continuing)")
        # Trust: grade with the harness from the protected default branch, not the submission's copy.
        # The build still measures the PR's kernels/runtime/moe; only bench/scripts (the scoring code,
        # incl. label.py + accuracy*) is pinned to origin/main. Fail-closed (&&): no trusted harness -> no eval.
        ev = (f"cd /root/sparkinfer && git fetch -q origin main && git checkout -q origin/main -- bench/scripts && "
              f"SI_NO_CHECKOUT=1 MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
              f"bench/scripts/evaluate.sh --ref {args.ref} --frontier {args.frontier} --ceiling {args.ceiling}")
        r = sh(host, port, ev, timeout=10800)
        sys.stdout.write(r.stdout[-4000:])
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if line:
            print("\n=== VERDICT ==="); print(json.dumps(json.loads(line[len("RESULT_JSON "):]), indent=2))
        else:
            print("\n!! no RESULT_JSON; stderr tail:\n" + r.stderr[-1500:])
    finally:
        # Default: STOP after every eval — pauses compute billing, keeps disk + weights for fast reuse.
        if args.keep:
            print(f">> leaving instance {iid} running (--keep)")
        elif args.destroy:
            print(f">> destroying instance {iid} (disk freed)");
            try: v.destroy_instance(id=iid)
            except Exception as e: print("destroy:", str(e)[:150])
        else:
            print(f">> stopping instance {iid} — disk/weights persist; resume with --reuse {iid}")
            try: v.stop_instance(id=iid)
            except Exception as e: print("stop:", str(e)[:150])

if __name__ == "__main__":
    main()
