#!/usr/bin/env python3
"""Automatic evaluation on a vast.ai GPU: provision (or reuse) → build/correctness/speed → label → teardown.

Requires VAST_API_KEY (`vastai set api-key <key>`). The numeric label is computed on-box by
bench/scripts/label.py (deterministic) — this script only orchestrates.

  # reuse an existing box (started if stopped, STOPPED again after the eval — the default):
  python eval/vast_eval.py --reuse 42134865 --frontier 164 --ceiling 366 --ref main

  # fixed SSH box (EVAL_TRANSPORT=ssh — does not rent from vast.ai):
  python eval/vast_eval.py --ssh 91.224.44.227:50200 --frontier 164 --ceiling 366 --ref main

By default the vast instance is STOPPED after every eval. With --ssh the fixed box is left running.

Self-healing: on --reuse, if the box won't become SSH-ready within --reuse-timeout (default 2 min),
it is stopped and a fresh box is provisioned via the vast API automatically; the new id is saved to
~/.sparkinfer_vast_instance (VAST_INSTANCE_FILE) so the next run reuses it. --no-recreate disables this.

Env: VAST_API_KEY, SSH_KEY, EVAL_TRANSPORT (vast|ssh), EVAL_SSH_HOST, EVAL_SSH_PORT, EVAL_REPO, VAST_INSTANCE_FILE.
"""
import argparse, json, os, random, shlex, shutil, subprocess, sys, time

from ssh_box import ssh_box_arg, ssh_box_enabled

# Resolve vastai CLI binary — subprocess.run doesn't always inherit the full user PATH
# when invoked from a bot/cron context. Try shutil.which first, then known locations.
_VASTAI_BIN = shutil.which("vastai") or os.path.expanduser("~/.local/bin/vastai")
if not os.path.isfile(_VASTAI_BIN):
    _VASTAI_BIN = "vastai"  # fallback: hope it's on PATH

REPO    = os.environ.get("EVAL_REPO",  "https://github.com/gittensor-ai-lab/sparkinfer")
IMAGE   = os.environ.get("EVAL_IMAGE", "nvidia/cuda:12.8.0-devel-ubuntu24.04")   # needs nvcc for sm_120
# Provision from a maintainer-vetted vast template that reliably exposes direct SSH. (The earlier
# default 1ea6ef1d8cc4ad95e710c4c1daed378c brought boxes to "running" with no working SSH; the raw
# image worked but vast's --ssh injection was flaky host-to-host. This template is the vetted fix.)
# Set EVAL_TEMPLATE_HASH="" to fall back to the raw EVAL_IMAGE + --ssh --direct path.
TEMPLATE_HASH = os.environ.get("EVAL_TEMPLATE_HASH", "7f806603ccd0de9b7370266673c0a32d")
SSH_KEY = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_ed25519"))
# Bare-metal SSH boxes often have nvcc outside default PATH (non-interactive ssh).
BOX_CUDA_ENV = "export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH; "
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/workspace/.llamacpp")            # persists across stop/start
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))  # self-healed id
# IPs of hosts that repeatedly hang on image pull or never expose direct SSH, despite high vast
# "reliability" scores (which track uptime, not image-pull / direct-SSH success). Whack-a-mole, but
# the offending set is small and recurring. Override/extend via VAST_SKIP_HOSTS (comma-separated).
_DEFAULT_SKIP = "94.177.17.69,120.238.149.205,192.3.91.246,47.253.144.202,175.121.93.64,180.70.178.129"
SKIP_HOSTS_PERMANENT = set(filter(None, os.environ.get("VAST_SKIP_HOSTS", _DEFAULT_SKIP).split(",")))

# --pinned: reuse a stable, known-good box (cached model, good download speed) as the default and
# NEVER destroy it. If it exists but can't be brought up within --reuse-timeout, provision a fresh
# box right away (default REUSE_MAX_RETRIES=0) and re-pin — stopped vast boxes that won't resume
# (host busy) would otherwise STALL a manual run, which has no scheduled retry to fall back on. Set
# VAST_REUSE_MAX_RETRIES>0 to instead wait across that-many scheduled runs before provisioning.
# Counter persists in REUSE_RETRY_FILE across runs.
REUSE_RETRY_FILE = os.path.expanduser(os.environ.get("VAST_REUSE_RETRY_FILE", "~/.sparkinfer_reuse_retries"))
REUSE_MAX_RETRIES = int(os.environ.get("VAST_REUSE_MAX_RETRIES", "0"))
PINNED_RETRY_RC = 75   # distinct exit code: "pinned box not up; retry on the next run" (not an error)
def _reuse_retries():
    try: return int(open(REUSE_RETRY_FILE).read().strip())
    except Exception: return 0
def _set_reuse_retries(n):
    try:
        with open(REUSE_RETRY_FILE, "w") as f: f.write(str(n))
    except Exception: pass

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
        out = subprocess.run([_VASTAI_BIN, "show", "user", "--raw"], capture_output=True, text=True, timeout=30).stdout
        u = json.loads(out)
        return float(u.get("balance") or 0) + float(u.get("credit") or 0)
    except Exception:
        return None

LOADING_TIMEOUT = 300   # bail if stuck in "loading" longer than this. The ~5GB CUDA-devel image
                        # legitimately takes 3-5 min to pull on many hosts; 180s abandoned healthy
                        # boxes mid-pull. The host blacklist (not a tight timeout) handles the
                        # persistently-hung offenders.
SSH_CONNECT_TIMEOUT = 180  # bail if "running" but SSH won't connect. Healthy boxes connect within
                           # a poll or two of "running"; a phantom-"running" host never does. 180s
                           # gives a slow-but-real box a little more slack than 120 before we give
                           # up and let the retry loop try another host.

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
    """Create a fresh instance via the vast API. Returns the new instance id, or None.
    Prefers higher-reliability hosts among the cheapest offers (reliability doesn't fully predict
    the phantom-"running" failure, but it screens out the genuinely flaky); the SSH timeout +
    blacklist + retry loop handle the rest."""
    base = f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 inet_down>=100"
    offers = v.search_offers(query=f"{base} reliability>0.97", order="dph_total", limit=25)
    if not offers:   # reliability filter too strict / API quirk → fall back to the unfiltered search
        offers = v.search_offers(query=base, order="dph_total", limit=25)
    if not offers:
        print(">> no matching offers"); return None
    # Exclude blacklisted + already-tried hosts, then from the cheapest dozen pick the MOST reliable.
    all_skip = SKIP_HOSTS_PERMANENT | (skip_hosts or set())
    cands = [o for o in offers if o.get("public_ipaddr") not in all_skip]
    if not cands: print(">> all offers are on blacklisted/skipped hosts"); return None
    off = max(cands[:12], key=lambda o: o.get("reliability2", 0))   # cheapest-12, best reliability
    print(f">> creating instance on offer {off['id']} {off.get('gpu_name')} ${off.get('dph_total'):.3f}/hr "
          f"host={off.get('public_ipaddr','?')} rel={off.get('reliability2','?')}")
    # Create via the CLI: the SDK's create_instance has no ssh/direct kwargs (those are CLI flags),
    # and --template_hash applies a preconfigured image+env. --raw returns {success, new_contract}.
    cmd = [_VASTAI_BIN, "create", "instance", str(off["id"]), "--disk", "120", "--ssh", "--direct", "--raw"]
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
    ap.add_argument("--guard-128-baseline", type=float, default=0,
                    help="main/origin 128-token decode tok/s used as the no-regression guard baseline")
    ap.add_argument("--guard-512-baseline", type=float, default=0,
                    help="main/origin 512-context decode tok/s used as the no-regression guard baseline")
    ap.add_argument("--guard-4k-baseline", type=float, default=0,
                    help="main/origin 4k-context decode tok/s used as the no-regression guard baseline")
    ap.add_argument("--guard-16k-baseline", type=float, default=0,
                    help="main/origin 16k-context decode tok/s used as the no-regression guard baseline")
    ap.add_argument("--guard-32k-baseline", type=float, default=0,
                    help="main/origin 32k-context decode tok/s used as the no-regression guard baseline")
    ap.add_argument("--guard-2k-baseline", type=float, default=0, help=argparse.SUPPRESS)
    # --- dual-model scoring: Qwen3.6 (primary, scored) + Qwen3-30B (no-regression guard) ---
    ap.add_argument("--dual", action="store_true",
                    help="score Qwen3.6-35B-A3B and guard Qwen3-30B-A3B against no-regression in one build "
                         "(--guard-*-baseline are the Qwen3-30B guard; --p-* are the Qwen3.6 scored target)")
    ap.add_argument("--p-guard-128-baseline", type=float, default=0, help="[--dual] Qwen3.6 main 128-token decode tok/s")
    ap.add_argument("--p-guard-512-baseline", type=float, default=0, help="[--dual] Qwen3.6 main 512-context tok/s")
    ap.add_argument("--p-guard-4k-baseline",  type=float, default=0, help="[--dual] Qwen3.6 main 4k-context tok/s")
    ap.add_argument("--p-guard-16k-baseline", type=float, default=0, help="[--dual] Qwen3.6 main 16k-context tok/s")
    ap.add_argument("--p-guard-32k-baseline", type=float, default=0, help="[--dual] Qwen3.6 main 32k-context tok/s")
    ap.add_argument("--p-llama-128-baseline", type=float, default=0, help="[--dual] Qwen3.6 llama.cpp 128-token tok/s (display + difficulty ref)")
    ap.add_argument("--p-llama-512-baseline", type=float, default=0, help="[--dual] Qwen3.6 llama.cpp 512-context tok/s")
    ap.add_argument("--p-llama-4k-baseline",  type=float, default=0, help="[--dual] Qwen3.6 llama.cpp 4k-context tok/s")
    ap.add_argument("--p-llama-16k-baseline", type=float, default=0, help="[--dual] Qwen3.6 llama.cpp 16k-context tok/s")
    ap.add_argument("--p-llama-32k-baseline", type=float, default=0, help="[--dual] Qwen3.6 llama.cpp 32k-context tok/s")
    ap.add_argument("--eval-mode", default=os.environ.get("SPARKINFER_EVAL_MODE", "longctx"),
                    choices=["longctx", "short"],
                    help="longctx scores 16k with a 128-token decode no-regression guard; short keeps legacy 128-token scoring")
    ap.add_argument("--reuse", type=int, default=0)
    ap.add_argument("--ssh", default="", metavar="HOST:PORT",
                    help="fixed SSH eval box (EVAL_TRANSPORT=ssh); vast.ai path unchanged when omitted")
    ap.add_argument("--keep", action="store_true", help="leave the instance running after eval (default: stop it)")
    ap.add_argument("--destroy", action="store_true", help="destroy after eval instead of stopping (also frees the disk)")
    ap.add_argument("--gpu", default="RTX_5090")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--reuse-timeout", type=int, default=300, help="seconds to wait for a reused box before recreating (default 300 = 5 min; a cold start of a stopped cached box can take minutes — destroying it prematurely wastes the 17GB cache)")
    ap.add_argument("--new-timeout", type=int, default=480, help="seconds to wait for a freshly created box (default 480 = 8 min)")
    ap.add_argument("--no-recreate", action="store_true", help="on reuse failure, error out instead of provisioning a new box")
    ap.add_argument("--pinned", action="store_true", help="the --reuse box is the stable default: NEVER destroy it; on bring-up failure exit PINNED_RETRY_RC for up to REUSE_MAX_RETRIES runs before provisioning a new box (pinned kept)")
    ap.add_argument("--destroy-on-error", action="store_true", help="destroy (not just stop) the instance if the eval produces no result")
    ap.add_argument("--polaris", action="store_true",
                    help="generate a Polaris verifiable receipt (unsigned attestation from eval box)")
    args = ap.parse_args()

    if not args.ssh and ssh_box_enabled():
        args.ssh = ssh_box_arg()

    bare_metal = bool(args.ssh)
    got_result = False
    v = None
    iid = args.reuse
    created = False
    host = port = None

    if bare_metal:
        ssh_host, _, ssh_port = args.ssh.partition(":")
        if not ssh_host:
            sys.exit("--ssh requires HOST:PORT")
        host = ssh_host.strip()
        port = int(ssh_port or "22")
        if not wait_ssh(host, port, tries=12):
            sys.exit(f"SSH box root@{host}:{port} not reachable")
        print(f">> bare-metal eval box: ssh root@{host}:{port}")
        args.keep = True  # never stop a fixed box
    else:
        from vastai import VastAI
        v = VastAI()
        bal = funds()
        if bal is not None:
            print(f">> vast.ai transport: funds ${bal:.2f} (balance + credit)")

    if not bare_metal:
        # --- vast.ai: reuse / provision / bring-up (unchanged) ---
        # 1) Try to bring up the reused box within a bounded window (default 5 min).
        if iid:
            ep = bring_up(v, iid, args.reuse_timeout)
            if ep:
                host, port = ep
                if args.pinned:
                    _set_reuse_retries(0)
            elif args.no_recreate:
                sys.exit(f"instance {iid} never came up (--no-recreate)")
            elif args.pinned:
                if info_of(v, iid) is None:
                    print(f">> pinned instance {iid} no longer exists (vast reclaimed it) — "
                          f"provisioning a fresh box now.")
                    _set_reuse_retries(0)
                    iid = 0
                else:
                    n = _reuse_retries() + 1
                    if n <= REUSE_MAX_RETRIES:
                        _set_reuse_retries(n)
                        print(f">> pinned instance {iid} exists but not SSH-ready within {args.reuse_timeout}s "
                              f"(miss {n}/{REUSE_MAX_RETRIES}) — leaving it intact; retry on the next scheduled run.")
                        sys.exit(PINNED_RETRY_RC)
                    _set_reuse_retries(0)
                    print(f">> pinned instance {iid} unavailable after {REUSE_MAX_RETRIES} retries — "
                          f"provisioning a NEW box (pinned {iid} kept, NOT destroyed).")
                    iid = 0
            else:
                stuck_host = (info_of(v, iid) or {}).get("public_ipaddr")
                print(f">> reused instance {iid} is dead/stuck — destroying it and provisioning a new box")
                try:
                    v.destroy_instance(id=iid)
                except Exception as e:
                    print("  destroy:", str(e)[:150])
                iid = 0

        # 2) No working box yet → create one, retrying on different hosts if needed.
        stuck_host = None
        if not iid:
            skip = set()
            MAX_ATTEMPTS = 8
            for attempt in range(1, MAX_ATTEMPTS + 1):
                iid = provision(v, args, skip_hosts=skip)
                if not iid:
                    sys.exit("could not provision an instance")
                created = True
                ep = bring_up(v, iid, args.new_timeout)
                if ep:
                    host, port = ep
                    break
                bad_host = (info_of(v, iid) or {}).get("public_ipaddr")
                print(f">> instance {iid} (host {bad_host}) never came up — destroying and trying another")
                try:
                    v.destroy_instance(id=iid)
                except Exception as e:
                    print("  destroy:", str(e)[:150])
                if bad_host:
                    skip.add(bad_host)
                iid = 0
                if attempt == MAX_ATTEMPTS:
                    sys.exit(f"all {MAX_ATTEMPTS} provision attempts failed — giving up")

        save_instance(iid)
        if args.reuse and iid != args.reuse:
            print(f"NEW_INSTANCE_ID {iid}")
            print(f">> switched to fresh instance {iid} (old {args.reuse} stopped; destroy it if unneeded)")

    MODEL_PATH = "/workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    MODEL_READY = "/tmp/sparkinfer_model_ready"
    # HuggingFace is throttled to ~KB/s from many vast hosts, so pull the GGUF from Google Drive
    # first (gdown handles the large-file confirm token), then fall back to HF/curl. Override the
    # Drive file id with MODEL_GDRIVE_ID="" to disable and use HF only.
    MODEL_GDRIVE_ID = os.environ.get("MODEL_GDRIVE_ID", "1BSLqKBs_Bo6up7YlFqwvRXuuQ4z0GcQf")

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
        # CRITICAL: force-clean the tree first. The eval step pins bench/scripts to origin/main, which
        # leaves the worktree dirty; a plain `git checkout` then FAILS ("local changes would be
        # overwritten") and silently leaves the box on the PREVIOUS PR's commit — so the next PR gets
        # evaluated against stale code. `reset --hard` + `clean -fd` + `checkout -f` guarantees the
        # working tree is exactly the requested ref. (Build dir lives under build/, model under
        # /workspace — neither is touched by clean here since build/ is rm -rf'd by evaluate.sh.)
        reset = "git reset -q --hard >/dev/null 2>&1; git clean -qfd bench >/dev/null 2>&1 || true"
        if args.ref.startswith("pull/") and args.ref.endswith("/head"):
            checkout = f"{reset}; git fetch -q origin '{args.ref}' && git checkout -qf FETCH_HEAD"
        else:
            # Branch ref (e.g. 'main' or 'origin/main'): fetch the BRANCH by name and check out
            # exactly what was fetched (FETCH_HEAD). Fetching the literal 'origin/main' fails (no such
            # ref on the remote — the branch is 'main'); the old `|| true` then silently checked out a
            # STALE local tracking ref, so on a REUSED box the same-box baseline built pre-merge code
            # (e.g. it measured main WITHOUT a just-merged PR). Strip any 'origin/' to the branch name.
            branch = args.ref.split("origin/", 1)[-1]
            # Fetch + checkout, then VERIFY the resulting HEAD matches origin/<branch>.
            # On a reused box with a stale local tree, a silent fetch failure left the box on
            # a previous PR's commit (not main) — the same-box baseline then inflated every
            # subsequent evaluation. The post-checkout guard catches this: if FETCH_HEAD ≠
            # origin/<branch>, the fetch was a no-op on a disconnected remote, and the box
            # must be re-cloned from scratch.
            checkout = (
                f"{reset}; git fetch -q origin '{branch}' && git checkout -qf FETCH_HEAD && "
                f"if [ \"$(git rev-parse HEAD)\" != \"$(git rev-parse origin/{branch})\" ]; then "
                f"echo '!! baseline checkout mismatch: HEAD != origin/{branch} — re-cloning'; "
                f"cd / && rm -rf /root/sparkinfer && "
                f"git clone -q {REPO} /root/sparkinfer && cd /root/sparkinfer && "
                f"git fetch -q origin '{branch}' && git checkout -qf FETCH_HEAD; "
                f"fi"
            )
        # g++-12: nvcc 12.8 breaks against Ubuntu 24.04's GCC 13.3 libstdc++ (cstdio /__gnu_cxx
        # errors). The build pins CMAKE_CUDA_HOST_COMPILER=g++-12, so it must be present.
        setup = ("export DEBIAN_FRONTEND=noninteractive; "
                 "git config --global --add safe.directory /root/sparkinfer 2>/dev/null || true; "
                 "(command -v git >/dev/null && command -v cmake >/dev/null && dpkg -s libisl23 >/dev/null 2>&1 && dpkg -s python3-pip >/dev/null 2>&1 && dpkg -s g++-12 >/dev/null 2>&1) "
                 "|| (apt-get update -q && apt-get install -y -q git curl cmake build-essential libisl23 python3-pip gcc-12 g++-12); "
                 "python3 -m pip install -q --break-system-packages huggingface_hub 'huggingface-hub[cli]' tokenizers >/dev/null 2>&1 || true; "
                 f"if [ -d /root/sparkinfer/.git ]; then cd /root/sparkinfer && {checkout}; "
                 f"else git clone -q {REPO} /root/sparkinfer && cd /root/sparkinfer && {checkout}; fi")
        sr = sh(host, port, setup, timeout=1800)
        if sr.returncode:
            print(f">> setup rc={sr.returncode} — stdout/stderr tail (continuing):")
            sys.stdout.write((sr.stdout or "")[-1500:]); sys.stdout.write((sr.stderr or "")[-1500:])

        # HF auth: write the token (from the local HF_TOKEN env, never committed) to the box's HF
        # token file so all hf downloads authenticate — lifts anonymous rate limits + reaches the
        # gated Qwen tokenizer repos. Sent in its own call so it never lands in a printed error tail.
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            sh(host, port, "mkdir -p ~/.cache/huggingface && "
                           f"printf %s {shlex.quote(hf_token)} > ~/.cache/huggingface/token && "
                           "chmod 600 ~/.cache/huggingface/token", timeout=30)
            print(">> HF token configured on box (authenticated model downloads)")

        # Pre-cache the model in a nohup background job so SSH drops don't abort the download.
        # If the file is already present (reused box), this is instant. Otherwise we poll for the
        # sentinel file created when the download completes.
        prefetch = (
            f"if [ -f '{MODEL_PATH}' ]; then touch '{MODEL_READY}' && echo cached; "
            f"elif [ -f '{MODEL_READY}' ]; then echo already_running; "
            f"else mkdir -p /workspace/models && rm -f '{MODEL_READY}'; "
            f"nohup bash -c '"
            f"  gid=\"{MODEL_GDRIVE_ID}\"; "
            f"  if [ -n \"$gid\" ]; then pip install -q gdown 2>>/tmp/dl.log; "
            f"    gdown --no-cookies -q \"$gid\" -O {MODEL_PATH}.part >>/tmp/dl.log 2>&1; "
            f"    sz=$(stat -c%s {MODEL_PATH}.part 2>/dev/null || echo 0); "
            f"    if [ \"$sz\" -gt 10000000000 ]; then mv -f {MODEL_PATH}.part {MODEL_PATH}; "
            f"    else echo \"gdrive failed (sz=$sz) -> HF\" >>/tmp/dl.log; rm -f {MODEL_PATH}.part; fi; "
            f"  fi; "
            f"  [ -f {MODEL_PATH} ] "
            f"  || HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-30B-A3B-GGUF "
            f"       Qwen3-30B-A3B-Q4_K_M.gguf --local-dir /workspace/models >>/tmp/dl.log 2>&1 "
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

        if args.dual:
            # Dual-model needs the Qwen3.6 GGUF too. Google Drive first (gdown handles the large-file
            # confirm token) — HF is throttled to ~KB/s from many vast hosts; HF/curl are the fallback.
            # Override the Drive id with MODEL36_GDRIVE_ID="" to disable. Separate dir from Qwen3 (the
            # two models have different tokenizers); evaluate_dual.sh's primary MODELS_DIR defaults to
            # <guard dir>36 (i.e. /workspace/models -> /workspace/models36).
            P36_DIR  = "/workspace/models36"
            P36_PATH = f"{P36_DIR}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            P36_READY = "/tmp/sparkinfer_model36_ready"
            P36_GDRIVE_ID = os.environ.get("MODEL36_GDRIVE_ID", "1Ayx_DYLnl1v5aKMiSGyO4KTmwVun2mVt")
            p36 = (
                f"if [ -f '{P36_PATH}' ]; then touch '{P36_READY}' && echo cached; "
                f"elif [ -f '{P36_READY}' ]; then echo already_running; "
                f"else mkdir -p {P36_DIR} && rm -f '{P36_READY}'; "
                f"nohup bash -c '"
                f"  gid=\"{P36_GDRIVE_ID}\"; "
                f"  if [ -n \"$gid\" ]; then pip install -q gdown 2>>/tmp/dl36.log; "
                f"    gdown --no-cookies -q \"$gid\" -O {P36_PATH}.part >>/tmp/dl36.log 2>&1; "
                f"    sz=$(stat -c%s {P36_PATH}.part 2>/dev/null || echo 0); "
                f"    if [ \"$sz\" -gt 15000000000 ]; then mv -f {P36_PATH}.part {P36_PATH}; "
                f"    else echo \"gdrive failed (sz=$sz) -> HF\" >>/tmp/dl36.log; rm -f {P36_PATH}.part; fi; "
                f"  fi; "
                f"  [ -f {P36_PATH} ] "
                f"  || HF_HUB_DISABLE_XET=1 hf download unsloth/Qwen3.6-35B-A3B-GGUF "
                f"       Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir {P36_DIR} >>/tmp/dl36.log 2>&1 "
                f"  || curl -fL -C - https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
                f"       -o {P36_PATH} >>/tmp/dl36.log 2>&1; "
                f"  [ -f {P36_PATH} ] && touch {P36_READY}"
                f"' >/dev/null 2>&1 & echo started; fi"
            )
            s36 = sh(host, port, p36, timeout=30).stdout.strip()
            if s36 == "cached":
                print(">> Qwen3.6 model already cached — skipping download")
            else:
                print(f">> Qwen3.6 model download started ({s36}) — polling ...")
                deadline = time.time() + 3000
                while time.time() < deadline:
                    r = sh(host, port, f"test -f '{P36_READY}' && echo yes || echo no", timeout=60)
                    if r.returncode == 0 and r.stdout.strip() == "yes":
                        break
                    time.sleep(20)
                else:
                    print("!! Qwen3.6 download slow — evaluate_dual.sh will retry (may add time)")

        # Reap any leftover reference server / runner from a previous PR on this kept-alive box —
        # a leaked llama-server holding port 8081 would make this PR's accuracy.sh fail to bind.
        sh(host, port, "pkill -f llama-server 2>/dev/null; pkill -f qwen3_gguf 2>/dev/null; sleep 1; true", timeout=30)

        # Trust: grade with the harness from the protected default branch, not the submission's copy.
        # The build still measures the PR's kernels/runtime/moe; only bench/scripts (the scoring code,
        # incl. label.py + accuracy*) is pinned to origin/main. Fail-closed (&&): no trusted harness -> no eval.
        # H1: a fresh, UNPREDICTABLE held-out prompt seed per eval so a PR can't overfit the in-repo
        # prompt. The seed is echoed into the verdict (eval_seed) so the prompt stays reproducible.
        eval_seed = os.urandom(8).hex()
        print(f">> held-out eval prompt seed: {eval_seed}")
        # Difficulty compensation ON (Option B): as the frontier pulls past llama.cpp each further %
        # gain is harder, so label.py scales the label tier up (raw % + significance gate unchanged).
        # Governance-tunable via SPARKINFER_DIFFICULTY_{K,REF,MAX}; applies from new evals onward.
        if args.dual:
            # Dual-model: score Qwen3.6 (primary) + guard Qwen3-30B (no-regression). The existing
            # --guard-*-baseline are the Qwen3-30B guard (G_*); --p-* carry the Qwen3.6 scored target.
            if args.polaris:
                # Polaris: run judge.py which wraps evaluate_dual.sh and produces an unsigned
                # attestation (POLARIS_ATTESTATION) alongside the normal RESULT_JSON.
                # Pin eval/polaris/ to origin/main — same trust model as bench/scripts.
                ev = (f"cd /root/sparkinfer && git fetch -q origin main && "
                      f"git checkout -q origin/main -- bench/scripts eval/polaris/ && "
                      f"SI_NO_CHECKOUT=1 SPARKINFER_EVAL_SEED={eval_seed} "
                      f"SPARKINFER_EVAL_MODE={args.eval_mode} "
                      f"SPARKINFER_G_GUARD_128_BASELINE={args.guard_128_baseline or args.guard_2k_baseline} "
                      f"SPARKINFER_G_GUARD_512_BASELINE={args.guard_512_baseline} "
                      f"SPARKINFER_G_GUARD_4K_BASELINE={args.guard_4k_baseline} "
                      f"SPARKINFER_G_GUARD_16K_BASELINE={args.guard_16k_baseline} "
                      f"SPARKINFER_G_GUARD_32K_BASELINE={args.guard_32k_baseline} "
                      f"SPARKINFER_P_GUARD_128_BASELINE={args.p_guard_128_baseline} "
                      f"SPARKINFER_P_GUARD_512_BASELINE={args.p_guard_512_baseline} "
                      f"SPARKINFER_P_GUARD_4K_BASELINE={args.p_guard_4k_baseline} "
                      f"SPARKINFER_P_GUARD_16K_BASELINE={args.p_guard_16k_baseline} "
                      f"SPARKINFER_P_GUARD_32K_BASELINE={args.p_guard_32k_baseline} "
                      f"SPARKINFER_P_LLAMA_128_BASELINE={args.p_llama_128_baseline} "
                      f"SPARKINFER_P_LLAMA_512_BASELINE={args.p_llama_512_baseline} "
                      f"SPARKINFER_P_LLAMA_4K_BASELINE={args.p_llama_4k_baseline} "
                      f"SPARKINFER_P_LLAMA_16K_BASELINE={args.p_llama_16k_baseline} "
                      f"SPARKINFER_P_LLAMA_32K_BASELINE={args.p_llama_32k_baseline} "
                      f"MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
                      f"python3 eval/polaris/judge.py --ref {args.ref} "
                      f"--ceiling {args.ceiling} --script bench/scripts/evaluate_dual.sh "
                      f"--model-file /workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf "
                      f"--guard-model-file /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf "
                      f"--build-dir /root/sparkinfer/build/runtime")
            else:
                ev = (f"cd /root/sparkinfer && git fetch -q origin main && git checkout -q origin/main -- bench/scripts && "
                      f"SI_NO_CHECKOUT=1 SPARKINFER_EVAL_SEED={eval_seed} "
                      f"SPARKINFER_EVAL_MODE={args.eval_mode} "
                      f"SPARKINFER_G_GUARD_128_BASELINE={args.guard_128_baseline or args.guard_2k_baseline} "
                      f"SPARKINFER_G_GUARD_512_BASELINE={args.guard_512_baseline} "
                      f"SPARKINFER_G_GUARD_4K_BASELINE={args.guard_4k_baseline} "
                      f"SPARKINFER_G_GUARD_16K_BASELINE={args.guard_16k_baseline} "
                      f"SPARKINFER_G_GUARD_32K_BASELINE={args.guard_32k_baseline} "
                      f"SPARKINFER_P_GUARD_128_BASELINE={args.p_guard_128_baseline} "
                      f"SPARKINFER_P_GUARD_512_BASELINE={args.p_guard_512_baseline} "
                      f"SPARKINFER_P_GUARD_4K_BASELINE={args.p_guard_4k_baseline} "
                      f"SPARKINFER_P_GUARD_16K_BASELINE={args.p_guard_16k_baseline} "
                      f"SPARKINFER_P_GUARD_32K_BASELINE={args.p_guard_32k_baseline} "
                      f"SPARKINFER_P_LLAMA_128_BASELINE={args.p_llama_128_baseline} "
                      f"SPARKINFER_P_LLAMA_512_BASELINE={args.p_llama_512_baseline} "
                      f"SPARKINFER_P_LLAMA_4K_BASELINE={args.p_llama_4k_baseline} "
                      f"SPARKINFER_P_LLAMA_16K_BASELINE={args.p_llama_16k_baseline} "
                      f"SPARKINFER_P_LLAMA_32K_BASELINE={args.p_llama_32k_baseline} "
                      f"MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
                      f"bench/scripts/evaluate_dual.sh --ref {args.ref} "
                      f"--ceiling {args.ceiling}")
        else:
            ev = (f"cd /root/sparkinfer && git fetch -q origin main && git checkout -q origin/main -- bench/scripts && "
                  f"SI_NO_CHECKOUT=1 SPARKINFER_EVAL_SEED={eval_seed} SPARKINFER_DIFFICULTY_BOOST=1 "
                  f"SPARKINFER_EVAL_MODE={args.eval_mode} "
                  f"SPARKINFER_GUARD_128_BASELINE={args.guard_128_baseline or args.guard_2k_baseline} "
                  f"SPARKINFER_GUARD_512_BASELINE={args.guard_512_baseline} "
                  f"SPARKINFER_GUARD_4K_BASELINE={args.guard_4k_baseline} "
                  f"SPARKINFER_GUARD_16K_BASELINE={args.guard_16k_baseline} "
                  f"SPARKINFER_GUARD_32K_BASELINE={args.guard_32k_baseline} "
                  f"MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
                  f"bench/scripts/evaluate.sh --ref {args.ref} --frontier {args.frontier} --ceiling {args.ceiling}")
        got_result = False
        if bare_metal:
            ev = BOX_CUDA_ENV + ev
        r = sh(host, port, ev, timeout=10800)
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        polaris_line = next((l for l in r.stdout.splitlines()
                             if l.startswith("POLARIS_ATTESTATION ")), None)
        got_result = bool(line)
        # Always emit machine-readable lines from the full SSH capture — the tail below is
        # for humans only; polaris+judge output can be >>4k and would drop these lines.
        if line:
            print(line)
        if polaris_line:
            print(polaris_line)
        sys.stdout.write(r.stdout[-4000:])
        if line:
            print("\n=== VERDICT ==="); print(json.dumps(json.loads(line[len("RESULT_JSON "):]), indent=2))
        else:
            print("\n!! no RESULT_JSON; stderr tail:\n" + r.stderr[-1500:])
    finally:
        if bare_metal:
            print(f">> bare-metal box left running (ssh root@{host}:{port})")
        else:
            destroy = args.destroy or (args.destroy_on_error and not got_result and created)
            if args.keep:
                print(f">> leaving instance {iid} running (--keep)")
            elif destroy:
                print(f">> destroying instance {iid} (disk freed)")
                try:
                    v.destroy_instance(id=iid)
                except Exception as e:
                    print("destroy:", str(e)[:150])
            else:
                print(f">> stopping instance {iid} — disk/weights persist; resume with --reuse {iid}")
                try:
                    v.stop_instance(id=iid)
                except Exception as e:
                    print("stop:", str(e)[:150])

if __name__ == "__main__":
    main()
