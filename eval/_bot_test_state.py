"""Keeps the bot test suites off the controller's own state.

Every path the bots keep state in -- ~/.sparkinfer_* (scores, strikes, the stale clock, the vast.ai
instance, the log and web checkouts) and the shared bot lock -- is pointed into a temp dir, one per
suite. By pattern rather than by a list of names: a list missed the Bonsai strikes file once and the
stale clock's file once, and a test run wrote both into the controller's home. isolate() runs when a
suite is imported and again in its setUpModule, so suites run in one process do not read each
other's state either (the last suite imported used to own every module's paths).
"""
import atexit
import os
import shutil
import tempfile

_PREFIX = "sparkinfer-bot-tests-"
_HOME_STATE = os.path.expanduser("~/.sparkinfer_")
_SHARED_LOCKS = {"/tmp/sparkinfer_bot.lock", os.environ.get("SPARKINFER_LOCK_FILE", "/tmp/sparkinfer_bot.lock")}


def new_state_dir():
    path = tempfile.mkdtemp(prefix=_PREFIX)
    atexit.register(shutil.rmtree, path, True)
    return path


def _is_state_path(value):
    return (value.startswith(_HOME_STATE) or value in _SHARED_LOCKS
            or value.startswith(os.path.join(tempfile.gettempdir(), _PREFIX)))


def isolate(state_dir, *mods):
    """Point every module-level path of `mods` that names controller state into `state_dir`,
    whatever its name (the web checkout's DASH and DATA_JSON are paths too)."""
    for mod in mods:
        for name in dir(mod):
            value = getattr(mod, name)
            if name.isupper() and isinstance(value, str) and _is_state_path(value):
                setattr(mod, name, os.path.join(state_dir, f"{mod.__name__}.{name}"))
        if hasattr(mod, "PINNED_INSTANCE"):
            mod.PINNED_INSTANCE = ""
