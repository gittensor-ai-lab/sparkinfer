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

Self-healing: on --reuse, if the box won't become SSH-ready within --reuse-timeout (default 2 min),
it is stopped and a fresh box is provisioned via the vast API automatically; the new id is saved to
~/.sparkinfer_vast_instance (VAST_INSTANCE_FILE) so the next run reuses it. --no-recreate disables this.

Env: VAST_API_KEY, SSH_KEY (default ~/.ssh/id_ed25519), LLAMACPP_DIR, EVAL_IMAGE, EVAL_REPO, VAST_INSTANCE_FILE.
"""
import argparse, json, os, random, subprocess, sys, time
from vastai import VastAI

REPO    = os.environ.get("EVAL_REPO",  "https://github.com/gittensor-ai-lab/sparkinfer")
IMAGE   = os.environ.get("EVAL_IMAGE", "nvidia/cuda:12.8.0-devel-ubuntu24.04")   # needs nvcc for sm_120
# Default to the raw CUDA image + `--ssh --direct` (proven path): template 1ea6ef1d8cc4ad95e710c4c1daed378c
# provisions boxes that reach "running" but never expose direct SSH with our key, so wait_ssh always
# times out (every cron run failed all 3 attempts). Set EVAL_TEMPLATE_HASH to opt back into a template.
TEMPLATE_HASH = os.environ.get("EVAL_TEMPLATE_HASH", "")  # "" = use EVAL_IMAGE with --ssh --direct
SSH_KEY = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_ed25519"))
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/workspace/.llamacpp")            # persists across stop/start
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))  # self-healed id
# IPs of hosts that repeatedly hang on image pull or never expose direct SSH, despite high vast
# "reliability" scores (which track uptime, not image-pull / direct-SSH success). Whack-a-mole, but
# the offending set is small and recurring. Override/extend via VAST_SKIP_HOSTS (comma-separated).
_DEFAULT_SKIP = "94.177.17.69,120.238.149.205,192.3.91.246"
SKIP_HOSTS_PERMANENT = set(filter(None, os.environ.get("VAST_SKIP_HOSTS", _DEFAULT_SKIP).split(",")))

def sh(host, port, cmd, timeout=3600):
    try:
        return subprocess.run(
            ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=40",
             "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 1, stdout="", stderr=f"ssh timeout after {timeout}s")

def info_of(v, iid):
    try:
        result = v.show_instances_v1(params={"id": iid})
        instances = result if isinstance(result, list) else result.get("instances", [])
        hit = next((i for i in instances if i.get("id") == iid), None)
        if hit is not None: return hit
    except Exception: pass
    # fallback to deprecated API in case v1 paginator misses the instance
    try: return next((i for i in v.show_instances() if i.get("id") == iid), None)
    except Exception: return None

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

LOADING_TIMEOUT = 300   # bail if stuck in "loading" longer than this. The ~5GB CUDA-devel image
                        # legitimately takes 3-5 min to pull on many hosts; 180s abandoned healthy
                        # boxes mid-pull. The host blacklist (not a tight timeout) handles the
                        # persistently-hung offenders.
SSH_CONNECT_TIMEOUT = 120  # bail if "running" but SSH won't connect. A healthy box accepts SSH
                           # within one poll of going "running"; a host that reports a phantom
                           # "running" at 0s with no sshd never recovers — abandon it fast (2 min)
                           # and let the provision-retry loop try another host.

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
    loading_since = None
    running_since = None
    while time.time() < deadline:
        info = info_of(v, iid)
        st = (info or {}).get("actual_status")
        if info and st == "running" and (info.get("public_ipaddr") or info.get("ssh_host")):
            if running_since is None: running_since = time.time()
            ssh_elapsed = int(time.time() - running_since)
            loading_since = None
            host, port = endpoint(info)
            if wait_ssh(host, port, tries=2):
                print(f">> instance {iid}: ssh root@{host}:{port}")
                return host, port
            if ssh_elapsed > SSH_CONNECT_TIMEOUT:
                print(f">> instance {iid} running for >{SSH_CONNECT_TIMEOUT}s but SSH won't connect — giving up")
                return None
            print(f"  instance {iid}: running ({ssh_elapsed}s) — SSH not ready yet ...")
        else:
            running_since = None
            if st == "loading":
                if loading_since is None: loading_since = time.time()
                elapsed = int(time.time() - loading_since)
                print(f"  instance {iid}: loading ({elapsed}s) — waiting ...")
                if elapsed > LOADING_TIMEOUT:
                    print(f">> instance {iid} stuck in 'loading' for >{LOADING_TIMEOUT}s — giving up")
                    return None
            else:
                print(f"  instance {iid}: status={st or '?'} — waiting ...")
        time.sleep(15)
    print(f">> instance {iid} did not become SSH-ready within {deadline_s}s")
    return None

def provision(v, args, skip_hosts=None):
    """Create a fresh instance via the vast API. Returns the new instance id, or None."""
    offers = v.search_offers(query=f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 inet_down>=100",
                             order="dph_total", limit=10)
    if not offers:
        print(">> no matching offers"); return None
    # Exclude permanently blacklisted hosts + any session-level skip_hosts.
    all_skip = SKIP_HOSTS_PERMANENT | (skip_hosts or set())
    pool = [o for o in offers[:5] if o.get("public_ipaddr") not in all_skip]
    if not pool: pool = [o for o in offers if o.get("public_ipaddr") not in all_skip]
    if not pool: print(">> all offers are on blacklisted hosts"); return None
    off = random.choice(pool)
    print(f">> creating instance on offer {off['id']} {off.get('gpu_name')} ${off.get('dph_total'):.3f}/hr host={off.get('public_ipaddr','?')}")
    # Create via the CLI: the SDK's create_instance has no ssh/direct kwargs (those are CLI flags),
    # and --template_hash applies a preconfigured image+env. --raw returns {success, new_contract}.
    cmd = ["vastai", "create", "instance", str(off["id"]), "--disk", "120", "--ssh", "--direct", "--raw"]
    if TEMPLATE_HASH:
        cmd += ["--template_hash", TEMPLATE_HASH]; print(f">> using template {TEMPLATE_HASH}")
    else:
        cmd += ["--image", args.image]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
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
    ap.add_argument("--reuse-timeout", type=int, default=300, help="seconds to wait for a reused box before recreating (default 300 = 5 min; a cold start of a stopped cached box can take minutes — destroying it prematurely wastes the 17GB cache)")
    ap.add_argument("--new-timeout", type=int, default=480, help="seconds to wait for a freshly created box (default 480 = 8 min)")
    ap.add_argument("--no-recreate", action="store_true", help="on reuse failure, error out instead of provisioning a new box")
    ap.add_argument("--destroy-on-error", action="store_true", help="destroy (not just stop) the instance if the eval produces no result")
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
            # Destroy the stuck box (can't SSH → no value in keeping disk) and provision a fresh one.
            stuck_host = (info_of(v, iid) or {}).get("public_ipaddr")
            print(f">> reused instance {iid} is dead/stuck — destroying it and provisioning a new box")
            try: v.destroy_instance(id=iid)
            except Exception as e: print("  destroy:", str(e)[:150])
            iid = 0

    # 2) No working box yet → create one, retrying on different hosts if needed.
    if not iid:
        skip = {stuck_host} if 'stuck_host' in dir() and stuck_host else set()
        MAX_ATTEMPTS = 5   # ~half of cheap offers are phantom-running hosts; each bad one now
                           # costs only ~2 min (SSH_CONNECT_TIMEOUT), so try more before erroring
        for attempt in range(1, MAX_ATTEMPTS + 1):
            iid = provision(v, args, skip_hosts=skip)
            if not iid: sys.exit("could not provision an instance")
            created = True
            ep = bring_up(v, iid, args.new_timeout)
            if ep:
                host, port = ep; break
            bad_host = (info_of(v, iid) or {}).get("public_ipaddr")
            print(f">> instance {iid} (host {bad_host}) never came up — destroying and trying another")
            try: v.destroy_instance(id=iid)
            except Exception as e: print("  destroy:", str(e)[:150])
            if bad_host: skip.add(bad_host)
            iid = 0
            if attempt == MAX_ATTEMPTS: sys.exit(f"all {MAX_ATTEMPTS} provision attempts failed — giving up")

    save_instance(iid)                              # persist the working id (the bot reuses it next run)
    if args.reuse and iid != args.reuse:
        print(f"NEW_INSTANCE_ID {iid}")             # machine-readable for the bot
        print(f">> switched to fresh instance {iid} (old {args.reuse} stopped; destroy it if unneeded)")

    MODEL_PATH = "/workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    MODEL_READY = "/tmp/sparkinfer_model_ready"

    def wait_model(host, port, timeout=2700):
        """Poll until the model file is fully downloaded (sentinel file appears)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = sh(host, port, f"test -f '{MODEL_READY}' && echo yes || echo no", timeout=60)
            if r.returncode == 0 and r.stdout.strip() == "yes":
                return True
            elapsed = int(deadline - time.time())
            print(f"  model download in progress (~{timeout-elapsed}s elapsed) ...")
            time.sleep(30)
        return False

    try:
        # pull/N/head refs (fork PRs) aren't fetched by default — need explicit fetch + FETCH_HEAD checkout.
        if args.ref.startswith("pull/") and args.ref.endswith("/head"):
            checkout = f"git fetch -q origin '{args.ref}' && git checkout -q FETCH_HEAD"
        else:
            checkout = f"git fetch -q origin '{args.ref}' 2>/dev/null || true && git checkout -q '{args.ref}'"
        setup = ("export DEBIAN_FRONTEND=noninteractive; "
                 "(command -v git >/dev/null && command -v cmake >/dev/null && dpkg -s libisl23 >/dev/null 2>&1 && dpkg -s python3-pip >/dev/null 2>&1) "
                 "|| (apt-get update -q && apt-get install -y -q git curl cmake build-essential libisl23 python3-pip); "
                 "python3 -m pip install -q --break-system-packages huggingface_hub 'huggingface-hub[cli]' tokenizers >/dev/null 2>&1 || true; "
                 f"if [ -d /root/sparkinfer/.git ]; then cd /root/sparkinfer && {checkout}; "
                 f"else git clone -q {REPO} /root/sparkinfer && cd /root/sparkinfer && {checkout}; fi")
        sr = sh(host, port, setup, timeout=1800)
        if sr.returncode:
            print(f">> setup rc={sr.returncode} — stdout/stderr tail (continuing):")
            sys.stdout.write((sr.stdout or "")[-1500:]); sys.stdout.write((sr.stderr or "")[-1500:])

        # Pre-cache the model in a nohup background job so SSH drops don't abort the download.
        # If the file is already present (reused box), this is instant. Otherwise we poll for the
        # sentinel file created when the download completes.
        prefetch = (
            f"if [ -f '{MODEL_PATH}' ]; then touch '{MODEL_READY}' && echo cached; "
            f"elif [ -f '{MODEL_READY}' ]; then echo already_running; "
            f"else mkdir -p /workspace/models && rm -f '{MODEL_READY}'; "
            f"nohup bash -c '"
            f"  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-30B-A3B-GGUF "
            f"    Qwen3-30B-A3B-Q4_K_M.gguf --local-dir /workspace/models >>/tmp/dl.log 2>&1 "
            f"  || curl -fL -C - https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
            f"       -o {MODEL_PATH} >>/tmp/dl.log 2>&1; "
            f"  [ -f {MODEL_PATH} ] && touch {MODEL_READY}"
            f"' >/dev/null 2>&1 & echo started; fi"
        )
        pr = sh(host, port, prefetch, timeout=30)
        status = pr.stdout.strip()
        if status == "cached":
            print(">> model already cached — skipping download")
        else:
            print(f">> model download started in background ({status}) — polling for completion ...")
            if not wait_model(host, port):
                print("!! model download timed out — evaluate.sh will retry (may add time)")

        # Reap any leftover reference server / runner from a previous PR on this kept-alive box —
        # a leaked llama-server holding port 8081 would make this PR's accuracy.sh fail to bind.
        sh(host, port, "pkill -f llama-server 2>/dev/null; pkill -f qwen3_gguf 2>/dev/null; sleep 1; true", timeout=30)

        # Trust: grade with the harness from the protected default branch, not the submission's copy.
        # The build still measures the PR's kernels/runtime/moe; only bench/scripts (the scoring code,
        # incl. label.py + accuracy*) is pinned to origin/main. Fail-closed (&&): no trusted harness -> no eval.
        ev = (f"cd /root/sparkinfer && git fetch -q origin main && git checkout -q origin/main -- bench/scripts && "
              f"SI_NO_CHECKOUT=1 MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
              f"bench/scripts/evaluate.sh --ref {args.ref} --frontier {args.frontier} --ceiling {args.ceiling}")
        got_result = False
        r = sh(host, port, ev, timeout=10800)
        sys.stdout.write(r.stdout[-4000:])
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        got_result = bool(line)
        if line:
            print("\n=== VERDICT ==="); print(json.dumps(json.loads(line[len("RESULT_JSON "):]), indent=2))
        else:
            print("\n!! no RESULT_JSON; stderr tail:\n" + r.stderr[-1500:])
    finally:
        # Default: STOP after every eval — pauses compute billing, keeps disk + weights for fast reuse.
        # --destroy-on-error only destroys if the instance itself is the problem (created fresh but no
        # result AND it's a newly created box — reused boxes that survive setup are kept even on eval
        # failure, since the disk cache (model + llama.cpp) is still valuable for the next run).
        destroy = args.destroy or (args.destroy_on_error and not got_result and created)
        if args.keep:
            print(f">> leaving instance {iid} running (--keep)")
        elif destroy:
            print(f">> destroying instance {iid} (disk freed)");
            try: v.destroy_instance(id=iid)
            except Exception as e: print("destroy:", str(e)[:150])
        else:
            print(f">> stopping instance {iid} — disk/weights persist; resume with --reuse {iid}")
            try: v.stop_instance(id=iid)
            except Exception as e: print("stop:", str(e)[:150])

if __name__ == "__main__":
    main()
