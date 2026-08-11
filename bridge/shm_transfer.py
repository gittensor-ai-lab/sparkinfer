"""POSIX shared-memory helpers for the LMCache sidecar.

Deliberately does NOT use multiprocessing.shared_memory: its naming/permission behavior isn't
guaranteed to interoperate with the C++ side's raw shm_open()/mmap() calls, and verifying that
interop was flagged as an open question in the original design. Sidestepping it entirely: POSIX
shm_open()-created objects are plain files under /dev/shm (that's the standard Linux
implementation, not an implementation detail this code is gambling on), so both sides just treat
shm as "a file under /dev/shm" via plain open()+mmap. The C++ side's shm_open("/name", ...) and
this module's open("/dev/shm/name", ...) address the exact same object.
"""
from __future__ import annotations

import mmap
import os

_SHM_DIR = "/dev/shm"


def _path_for(name: str) -> str:
    """name is a POSIX shm name (as passed on the wire, e.g. "/sparkinfer_kv_7_p") -- strip the
    leading slash sparkinfer's protocol always includes, since /dev/shm/<name> is not itself
    slash-prefixed."""
    return os.path.join(_SHM_DIR, name.lstrip("/"))


def create(name: str, size: int) -> mmap.mmap:
    """Creates a new shm region of exactly `size` bytes and returns it mapped read/write. Raises
    FileExistsError if name is already in use -- callers should pick unique names (the protocol
    doc's convention is /sparkinfer_kv_<request_id>_<side>, which request_id already makes
    unique) rather than this module silently overwriting something.
    """
    path = _path_for(name)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, size)
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)  # the mapping keeps the underlying object alive; the fd itself is not needed


def attach(name: str) -> mmap.mmap:
    """Maps an existing shm region (created by the peer) read/write, sized to whatever it
    actually is. Raises FileNotFoundError if the region doesn't exist -- callers should treat
    that as a protocol-level error (the peer's shm_name was stale or never created), not retry
    silently.
    """
    path = _path_for(name)
    fd = os.open(path, os.O_RDWR)
    try:
        size = os.fstat(fd).st_size
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)


def unlink(name: str) -> None:
    """Removes the name from /dev/shm. Safe to call even if already removed (e.g. by the peer) --
    matches the C++ side's shm_unlink() being a similarly best-effort cleanup call. Existing
    mappings stay valid after unlink (standard POSIX shm semantics); this only removes the name
    other opens would use to find it.
    """
    try:
        os.unlink(_path_for(name))
    except FileNotFoundError:
        pass
