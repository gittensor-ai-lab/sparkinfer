"""Report-only mode for the eval bots: measure every PR, write nothing to GitHub.

SPARKINFER_REPORT_ONLY=1 (in .env.eval) makes every bot and the sync run their rounds as usual --
the box measures, verdicts are computed and printed into the bot's own log -- but nothing they
would send to GitHub leaves this machine: no comment, label, merge, close, review or push.

One choke point rather than a flag at each call site: pr_eval_bot imports this module and calls
install(), which wraps subprocess.run. Every bot imports pr_eval_bot (and through it
copycat_guard), and all of them reach GitHub only through subprocess.run(["gh", ...]) or
subprocess.run(["git", ..., "push", ...]), so a write added later is covered without anyone
remembering to guard it. Reads (pr list / view / checks / diff, GET api calls, graphql queries)
run as normal, since the bots need them to find and measure PRs.

A blocked call returns exit status 0 with no output, so the bot carries on exactly as if the write
had gone through. One line goes to stderr (the bot's log), and the full call -- comment bodies and
--body-file contents included -- is appended to SPARKINFER_REPORT_ONLY_LOG
(~/.sparkinfer_report_only.log), so the report a bot would have posted can still be read.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys

LOG = os.path.expanduser(os.environ.get("SPARKINFER_REPORT_ONLY_LOG", "~/.sparkinfer_report_only.log"))

# gh subcommands that only read. Anything not listed counts as a write: a gh subcommand added
# later is blocked until someone decides it is safe, rather than let through by default.
_GH_READS = {
    "pr": {"list", "view", "checks", "diff", "status"},
    "issue": {"list", "view", "status"},
    "label": {"list"},
    "run": {"list", "view", "watch", "download"},
    "workflow": {"list", "view"},
    "release": {"list", "view", "download"},
    "repo": {"view", "list"},
    "auth": None, "search": None, "status": None, "help": None, "version": None,
}
# git global options that take a value, so the subcommand is found past them.
_GIT_OPTS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def enabled() -> bool:
    return os.environ.get("SPARKINFER_REPORT_ONLY", "0").strip() not in ("", "0", "false", "no")


def _positionals(args):
    return [a for a in args if not a.startswith("-")]


def _gh_api_reads(args) -> bool:
    """`gh api ...`: a read is GET without a body, or a graphql query that is not a mutation."""
    endpoint = next((a for a in args[1:] if not a.startswith("-")), "")
    if endpoint == "graphql":
        return "mutation" not in " ".join(args).lower()
    method, explicit = "GET", False
    for i, a in enumerate(args):
        if a in ("-X", "--method") and i + 1 < len(args):
            method, explicit = args[i + 1].upper(), True
        elif a.startswith("--method="):
            method, explicit = a.split("=", 1)[1].upper(), True
    if method != "GET":
        return False
    # gh sends -f / -F fields as a POST body unless the method is given as GET explicitly.
    body = any(a in ("-f", "-F", "--field", "--raw-field", "--input")
               or a.startswith(("--field=", "--raw-field=", "--input=")) for a in args)
    return explicit or not body


def is_github_write(cmd) -> bool:
    """Would this argv change something on GitHub? Only gh and git are ever considered."""
    if not isinstance(cmd, (list, tuple)) or not cmd:
        return False
    prog, args = os.path.basename(str(cmd[0])), [str(a) for a in cmd[1:]]
    if prog == "gh":
        pos = _positionals(args)
        if not pos:
            return False
        if pos[0] == "api":
            return not _gh_api_reads(args)
        if pos[0] in _GH_READS:
            allowed = _GH_READS[pos[0]]
            return allowed is not None and (len(pos) < 2 or pos[1] not in allowed)
        return True
    if prog == "git":
        i = 0
        while i < len(args):
            a = args[i]
            if a in _GIT_OPTS_WITH_VALUE:
                i += 2
            elif a.startswith("-"):
                i += 1
            else:
                return a == "push"
        return False
    return False


def _record(cmd) -> None:
    args = [str(a) for a in cmd]
    line = " ".join(args)
    print(f"[report-only] not sent: {line[:200]}{'...' if len(line) > 200 else ''}", file=sys.stderr)
    parts = list(args)
    for i, a in enumerate(args[:-1]):
        if a == "--body-file" and os.path.isfile(args[i + 1]):
            try:
                with open(args[i + 1], errors="replace") as f:
                    parts.append("--- body-file contents ---\n" + f.read())
            except OSError:
                pass
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(LOG, "a") as f:
            f.write(f"\n===== {stamp}\n" + "\n".join(parts) + "\n")
    except OSError as e:
        print(f"[report-only] could not write {LOG}: {e}", file=sys.stderr)


def install() -> bool:
    """Wrap subprocess.run when SPARKINFER_REPORT_ONLY is set. Idempotent; returns whether active."""
    if not enabled():
        return False
    if getattr(subprocess.run, "_report_only", False):
        return True
    real_run = subprocess.run

    def run(*popenargs, **kwargs):
        cmd = popenargs[0] if popenargs else kwargs.get("args")
        if is_github_write(cmd):
            _record(cmd)
            text = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding")
            empty = "" if text else b""
            captured = kwargs.get("capture_output") or kwargs.get("stdout") == subprocess.PIPE
            return subprocess.CompletedProcess(cmd, 0, empty if captured else None,
                                               empty if captured else None)
        return real_run(*popenargs, **kwargs)

    run._report_only = True
    run._real_run = real_run
    subprocess.run = run
    print(">> REPORT-ONLY: measuring and logging only -- nothing is posted, labelled, merged, "
          f"closed or pushed (blocked calls: {LOG})", file=sys.stderr)
    return True
