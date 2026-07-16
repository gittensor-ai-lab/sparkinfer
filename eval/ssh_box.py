"""Eval box transport — vast.ai (default) or a fixed SSH box.

Set EVAL_TRANSPORT=ssh to use a pinned bare-metal GPU (EVAL_SSH_HOST / EVAL_SSH_PORT).
Leave unset or EVAL_TRANSPORT=vast to keep the existing vast.ai rent/reuse/stop logic.

Legacy alias: EVAL_USE_VAST=0 also selects the SSH box (requires EVAL_SSH_HOST).
"""

import os

_TRANSPORT = os.environ.get("EVAL_TRANSPORT", "").strip().lower()


def vast_enabled():
    """True when vast.ai provisioning should run (the default)."""
    if _TRANSPORT == "ssh":
        return False
    if _TRANSPORT == "vast":
        return True
    use_vast = os.environ.get("EVAL_USE_VAST", "").strip().lower()
    if use_vast in ("0", "false", "no"):
        return False
    return True


def ssh_box_endpoint():
    """Return (host, port) for the fixed SSH eval box, or None if not configured."""
    direct = os.environ.get("EVAL_SSH", "").strip()
    if direct:
        host, sep, port = direct.partition(":")
        if not host:
            return None
        return host.strip(), int(port or "22")
    host = os.environ.get("EVAL_SSH_HOST", "").strip()
    if not host:
        return None
    return host, int(os.environ.get("EVAL_SSH_PORT", "22"))


def ssh_box_arg():
    """host:port string for vast_eval.py --ssh, or empty."""
    ep = ssh_box_endpoint()
    if not ep:
        return ""
    host, port = ep
    return f"{host}:{port}"


def ssh_box_enabled():
    """Use the fixed SSH box instead of vast.ai."""
    if vast_enabled():
        return False
    return ssh_box_endpoint() is not None
