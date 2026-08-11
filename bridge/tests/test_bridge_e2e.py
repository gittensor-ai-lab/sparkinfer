"""Integration test for bridge/lmcache_bridge.py: a real sidecar subprocess, the real `lmcache`
CPU-RAM backend, and a raw-socket test client following protocol.py (standing in for the C++
BridgeClient, which this repo has no Python bindings to exercise from here).

Skipped automatically if `lmcache` isn't installed (see bridge/requirements.txt) -- this is a
heavy, real dependency (torch included), not something every CI environment running the rest of
sparkinfer's tests is expected to have.

Slow: real engine construction (first-time torch/lmcache import + LMCacheEngineBuilder +
post_init()) takes on the order of 10 seconds. That cost is why lmcache_bridge.py builds the
engine eagerly at startup rather than lazily on first HELLO -- see BridgeServer's docstring.
"""
import importlib.util
import os
import socket
import subprocess
import sys
import time
import unittest

_BRIDGE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _BRIDGE_DIR)

import protocol as proto  # noqa: E402
import shm_transfer as shm  # noqa: E402

_HAVE_LMCACHE = importlib.util.find_spec("lmcache") is not None

NUM_LAYERS, KV_HEADS, HEAD_DIM, BLOCK_SIZE = 4, 2, 8, 16
CHUNK_SIZE = 256


def _recv_frame(conn: socket.socket) -> proto.Frame:
    return proto.decode_frame(conn.recv(proto.MAX_FRAME_BYTES))


def _hello(conn: socket.socket, req_id: int) -> int:
    payload = (
        proto.Writer()
        .u32(NUM_LAYERS).u32(KV_HEADS).u32(HEAD_DIM).u32(BLOCK_SIZE)
        .u8(0).u32(2).str("e2e-test-model").bytes()
    )
    conn.send(proto.encode_frame(proto.HELLO, req_id, payload))
    f = _recv_frame(conn)
    assert f.msg_type == proto.HELLO_ACK, f
    r = proto.Reader(f.payload)
    ok = r.u8()
    r.str()  # lmcache_version, unchecked
    chunk_size = r.u32()
    err = r.str()
    assert ok == 1, err
    return chunk_size


@unittest.skipUnless(_HAVE_LMCACHE, "lmcache not installed (see bridge/requirements.txt)")
class BridgeE2ETest(unittest.TestCase):
    def setUp(self) -> None:
        self.sock_path = f"/tmp/sparkinfer_lmcache_e2e_{os.getpid()}.sock"
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

        env = dict(os.environ)
        env["LMCACHE_TRACK_USAGE"] = "false"
        env["DO_NOT_TRACK"] = "1"
        env["SPARKINFER_LMCACHE_CHUNK_SIZE"] = str(CHUNK_SIZE)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(_BRIDGE_DIR, "lmcache_bridge.py"),
             "--socket", self.sock_path, "--instance-id", f"e2e-{os.getpid()}",
             "--log-level", "WARNING",
             "--num-layers", str(NUM_LAYERS), "--num-kv-heads", str(KV_HEADS),
             "--head-dim", str(HEAD_DIM), "--block-size", str(BLOCK_SIZE),
             "--elem-bytes", "2", "--model-name", "e2e-test-model"],
            env=env, cwd=_BRIDGE_DIR,
        )
        for _ in range(300):  # up to 30s for cold engine construction
            if os.path.exists(self.sock_path):
                break
            time.sleep(0.1)
        else:
            self.proc.kill()
            raise RuntimeError("sidecar never created its socket within 30s")
        time.sleep(0.5)  # let listen() actually start accepting

    def tearDown(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)

    def test_lookup_miss_store_lookup_hit_byte_identical(self):
        ctrl = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(ctrl.close)
        ctrl.settimeout(10)
        ctrl.connect(self.sock_path)
        self.assertEqual(_hello(ctrl, 1), CHUNK_SIZE)

        token_ids = list(range(1000, 1000 + CHUNK_SIZE))
        lookup_payload = proto.Writer().u32(len(token_ids))
        for t in token_ids:
            lookup_payload.i32(t)

        # miss before anything is stored
        ctrl.send(proto.encode_frame(proto.LOOKUP, 2, lookup_payload.bytes()))
        f = _recv_frame(ctrl)
        r = proto.Reader(f.payload)
        matched = r.i32()
        self.assertEqual(matched, 0, "expected a miss before any STORE")

        # store on a second connection (mirrors BridgeClient's separate STORE connection)
        store_conn = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(store_conn.close)
        store_conn.settimeout(10)
        store_conn.connect(self.sock_path)
        self.assertEqual(_hello(store_conn, 3), CHUNK_SIZE)

        block_bytes = BLOCK_SIZE * KV_HEADS * HEAD_DIM * 2
        n_blocks = CHUNK_SIZE // BLOCK_SIZE
        chunk_bytes = NUM_LAYERS * 2 * n_blocks * block_bytes
        fake_kv = bytearray((i * 37 + 11) % 256 for i in range(chunk_bytes))

        store_region = f"/sparkinfer_kv_{os.getpid()}_c"
        mm = shm.create(store_region, len(fake_kv))
        mm[:] = fake_kv
        mm.close()

        store_payload = proto.Writer().u32(len(token_ids))
        for t in token_ids:
            store_payload.i32(t)
        store_payload.u32(0).u32(CHUNK_SIZE).str(store_region)
        store_conn.send(proto.encode_frame(proto.STORE, 4, store_payload.bytes()))
        f = _recv_frame(store_conn)
        self.assertEqual(f.msg_type, proto.STORE_ACK)
        r = proto.Reader(f.payload)
        ok = r.u8()
        err = r.str()
        self.assertEqual(ok, 1, err)
        # STORE's shm region is created (and unlinked) by the creator, per the protocol doc --
        # this test stands in for the C++ side, which unlinks after receiving the ack.
        shm.unlink(store_region)

        # hit after storing
        ctrl.send(proto.encode_frame(proto.LOOKUP, 5, lookup_payload.bytes()))
        f = _recv_frame(ctrl)
        r = proto.Reader(f.payload)
        matched = r.i32()
        shm_name = r.str()
        n_chunks = r.u32()
        chunks = [(r.i32(), r.i32(), r.u64()) for _ in range(n_chunks)]
        self.assertEqual(matched, CHUNK_SIZE)
        self.assertEqual(chunks, [(0, CHUNK_SIZE, 0)])

        got = shm.attach(shm_name)
        got_bytes = bytes(got[chunks[0][2]: chunks[0][2] + chunk_bytes])
        got.close()
        shm.unlink(shm_name)
        self.assertEqual(got_bytes, bytes(fake_kv), "retrieved KV bytes must be byte-identical to what was stored")

        ctrl.send(proto.encode_frame(proto.PING, 6, b""))
        f = _recv_frame(ctrl)
        self.assertEqual(f.msg_type, proto.PONG)


if __name__ == "__main__":
    unittest.main()
