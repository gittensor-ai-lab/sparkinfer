#!/usr/bin/env python3
"""sparkinfer <-> LMCache sidecar. Embeds the real `lmcache` package via its documented
in-process API and speaks ../docs/lmcache_bridge_protocol.md back to sparkinfer's C++ server.

Run standalone for debugging: see bridge/README.md.
"""
from __future__ import annotations

# LMCache phones home to stats.lmcache.ai:8080 for usage telemetry by default, with no fast
# timeout observed in a restricted-network environment during development (see the Stage 1 spike
# notes in the protocol doc) -- these MUST be set before `import lmcache` anywhere below, or
# engine construction can hang.
import os

os.environ.setdefault("LMCACHE_TRACK_USAGE", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")

import argparse
import logging
import socket
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

import torch

import protocol as proto
import shm_transfer as shm

logger = logging.getLogger("lmcache_bridge")


# --- the connector: LMCache's real integration point (lmcache.v1.gpu_connector.GPUConnectorInterface) ---
#
# Confirmed via the Stage 1 spike: this is what a non-vLLM/non-SGLang caller implements. Instead
# of a real GPU, to_gpu/from_gpu read/write a POSIX shm region this sidecar creates per request.
# Addressing is deterministic from token position (chunk_index = (tok - base) // chunk_size),
# not from call order, since LMCache's internal chunk splitting order isn't part of its public
# contract -- this lets the connector stay correct regardless of what order to_gpu/from_gpu fire.


class ShmConnector:
    """One instance shared by the whole sidecar process (LMCacheEngine is a singleton per
    instance_id). begin_request()/end_request() scope chunk_size/base_tok/mmap for the duration
    of one store()/retrieve() call -- LMCacheEngine calls back into to_gpu/from_gpu synchronously
    within that call, never concurrently with another request on this connector (EngineHandle.lock
    in _handle_lookup/_handle_store serializes engine access across connections/threads).
    """

    def __init__(self, num_layers: int, kv_heads: int, head_dim: int, block_size: int,
                elem_bytes: int, int8_kv: bool, chunk_size: int) -> None:
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.elem_bytes = elem_bytes
        self.int8_kv = int8_kv
        self.chunk_size = chunk_size
        self.kvcaches = None  # required by GPUConnectorInterface's base initializer, unused here

        self._active_mmap = None
        self._active_base_tok = 0
        self._touched_chunks: list[tuple[int, int]] = []  # (start_tok, end_tok) actually copied

    # -- bytes-per-chunk layout, matching docs/lmcache_bridge_protocol.md's "shm region layout" --

    def block_bytes(self) -> int:
        return self.block_size * self.kv_heads * self.head_dim * self.elem_bytes

    def scale_bytes_per_chunk(self, n_tok: int) -> int:
        if not self.int8_kv:
            return 0
        # one fp16 (2-byte) scale per (token, kv_head), per layer -- see kv_cache.cpp's
        # scale_layer_stride convention on the C++ side.
        return self.num_layers * n_tok * self.kv_heads * 2

    def chunk_byte_size(self, n_tok: int) -> int:
        n_blocks = n_tok // self.block_size
        kv_bytes = self.num_layers * 2 * n_blocks * self.block_bytes()  # K then V
        return kv_bytes + 2 * self.scale_bytes_per_chunk(n_tok)  # K scale + V scale

    def begin_request(self, mm, base_tok: int) -> None:
        """Called by the sidecar before invoking store()/retrieve() -- `mm` is this request's
        shm mmap, `base_tok` is the token position that chunk index 0 corresponds to."""
        self._active_mmap = mm
        self._active_base_tok = base_tok
        self._touched_chunks = []

    def end_request(self) -> list[tuple[int, int]]:
        touched = self._touched_chunks
        self._active_mmap = None
        self._touched_chunks = []
        return touched

    def _offset_for(self, start_tok: int, n_tok: int) -> int:
        # Deterministic from token position, not call order -- see the module docstring.
        chunk_index = (start_tok - self._active_base_tok) // self.chunk_size
        return chunk_index * self.chunk_byte_size(self.chunk_size) if n_tok == self.chunk_size \
            else chunk_index * self.chunk_byte_size(self.chunk_size)  # only full chunks in v1

    # -- GPUConnectorInterface --

    def get_shape(self, num_tokens: Optional[int] = None):
        n = num_tokens if num_tokens is not None else self.chunk_size
        # Opaque byte layout -- shape is informational for LMCache's own bookkeeping, not used to
        # interpret the bytes (kv_dtype=torch.uint8 makes every element 1 byte, see
        # LMCacheMetadata in lmcache_bridge.py's build_engine()).
        return torch.Size([self.chunk_byte_size(n)])

    def from_gpu(self, memory_obj, start: int, end: int, **kwargs) -> None:
        """STORE path: copy from our shm (the "GPU") into memory_obj (LMCache's own storage)."""
        assert self._active_mmap is not None, "from_gpu called outside begin_request/end_request"
        n_tok = end - start
        off = self._offset_for(start, n_tok)
        size = self.chunk_byte_size(n_tok)
        dst = memory_obj.byte_array.cast("B")
        n = min(len(dst), size)
        dst[:n] = self._active_mmap[off : off + n]
        self._touched_chunks.append((start, end))

    def to_gpu(self, memory_obj, start: int, end: int, **kwargs) -> None:
        """RETRIEVE path: copy from memory_obj (LMCache's own storage) into our shm."""
        assert self._active_mmap is not None, "to_gpu called outside begin_request/end_request"
        n_tok = end - start
        off = self._offset_for(start, n_tok)
        size = self.chunk_byte_size(n_tok)
        src = bytes(memory_obj.byte_array.cast("B"))
        n = min(len(src), size)
        self._active_mmap[off : off + n] = src[:n]
        self._touched_chunks.append((start, end))

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        for mo, s, e in zip(memory_objs, starts, ends):
            self.from_gpu(mo, s, e, **kwargs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        for mo, s, e in zip(memory_objs, starts, ends):
            self.to_gpu(mo, s, e, **kwargs)


@dataclass
class EngineHandle:
    engine: object
    connector: ShmConnector
    layout: proto.HelloRequest
    chunk_size: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def build_engine(layout: proto.HelloRequest, instance_id: str) -> EngineHandle:
    # Imported here (not at module scope) so --help / argument errors don't pay LMCache's import
    # cost, and so the LMCACHE_TRACK_USAGE env vars above are guaranteed set first regardless of
    # how this module is invoked.
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

    chunk_size = int(os.environ.get("SPARKINFER_LMCACHE_CHUNK_SIZE", "256"))
    max_local_cpu_gb = float(os.environ.get("SPARKINFER_LMCACHE_MAX_CPU_GB", "4.0"))

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=max_local_cpu_gb,
        remote_url=None,
    )
    connector = ShmConnector(
        num_layers=layout.num_layers,
        kv_heads=layout.num_kv_heads,
        head_dim=layout.head_dim,
        block_size=layout.block_size,
        elem_bytes=layout.elem_bytes,
        int8_kv=layout.int8_kv,
        chunk_size=chunk_size,
    )
    # LMCacheMetadata.get_shapes(num_tokens) allocates MemoryObj as
    # [kv_shape[1], kv_shape[0], num_tokens, kv_shape[3]*kv_shape[4]] elements of kv_dtype
    # (verified by reading lmcache/v1/metadata.py -- not documented). With kv_dtype=torch.uint8
    # (1 byte/element), that total must equal ShmConnector.chunk_byte_size(chunk_size) exactly,
    # or store()/retrieve() allocate a MemoryObj smaller than what the connector actually
    # writes/reads, silently truncating (found via the Stage 1 E2E test: a naive
    # kv_shape=(layers,2,chunk_size,kv_heads,head_dim) under-sized the buffer by exactly
    # elem_bytes, corrupting exactly the back half of every retrieved chunk). Folding
    # elem_bytes and the int8 scale contribution into kv_shape[4] (with kv_shape[3]=1) makes
    # the two sides agree regardless of int8_kv -- see ShmConnector.chunk_byte_size for the
    # matching derivation.
    per_token_bytes = layout.num_kv_heads * layout.head_dim * layout.elem_bytes
    if layout.int8_kv:
        per_token_bytes += layout.num_kv_heads * 2  # one fp16 scale per (token, kv_head)
    metadata = LMCacheMetadata(
        model_name=layout.model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,  # opaque bytes -- confirmed accepted in the Stage 1 spike
        kv_shape=(layout.num_layers, 2, chunk_size, 1, per_token_bytes),
        chunk_size=chunk_size,
    )
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        connector,
        lambda t, r: t,  # single-worker: broadcast is a no-op
        lambda o, r: o,
    )
    engine.post_init()  # required -- store()/retrieve() assert on storage_manager without it
    return EngineHandle(engine=engine, connector=connector, layout=layout, chunk_size=chunk_size)


class BridgeServer:
    """Accepts multiple concurrent AF_UNIX SOCK_SEQPACKET connections (the C++ side opens two:
    one for LOOKUP/PING, one for STORE -- see the protocol doc's "Transport" section) and serves
    each on its own thread.

    The LMCacheEngine is built once, eagerly, before serve_forever() ever accepts a connection --
    NOT lazily from the first HELLO. That was the original design (avoids duplicating the KV
    layout as both CLI args and a HELLO payload) but real testing found engine construction
    (first-time `import torch`/`lmcache` plus LMCacheEngineBuilder + post_init()) takes on the
    order of 10 seconds, which blows straight through the tight timeout budget the C++ side's
    ctrl-connection handshake uses (SPARKINFER_LMCACHE_LOOKUP_TIMEOUT_MS, default 5ms) -- a lazy
    build would make the sidecar's very first real handshake fail (safely, but for tens of
    seconds after every server restart, during which the cache silently does nothing). So the KV
    layout is CLI args here, duplicating HELLO's payload fields -- HELLO becomes a cheap
    layout-match check against the already-built engine, not a trigger for building it.
    """

    def __init__(self, socket_path: str, engine: EngineHandle) -> None:
        self.socket_path = socket_path
        self._engine = engine

    def sweep_orphans(self) -> None:
        """Removes any /sparkinfer_kv_* shm regions left behind by a prior sidecar process that
        crashed or was killed before it could unlink them (e.g. a LOOKUP hit the C++ side never
        got around to reading). Only meaningful at startup, before any request could have
        created a legitimate one of these -- never called mid-run."""
        try:
            for name in os.listdir(shm._SHM_DIR):
                if name.startswith("sparkinfer_kv_"):
                    shm.unlink("/" + name)
                    logger.info("swept orphaned shm region from a prior run: %s", name)
        except OSError:
            pass  # /dev/shm not listable for some reason -- not fatal, just skip the sweep

    def serve_forever(self) -> None:
        self.sweep_orphans()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        srv.bind(self.socket_path)
        srv.listen(8)
        logger.info("listening on %s", self.socket_path)
        try:
            while True:
                conn, _ = srv.accept()
                t = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
                t.start()
        finally:
            srv.close()
            shm_path = self.socket_path
            if os.path.exists(shm_path):
                os.unlink(shm_path)

    def _serve_connection(self, conn: socket.socket) -> None:
        peer_layout: Optional[proto.HelloRequest] = None
        try:
            while True:
                raw = conn.recv(proto.MAX_FRAME_BYTES)
                if not raw:
                    return  # peer closed
                try:
                    frame = proto.decode_frame(raw)
                except proto.ProtocolError:
                    logger.warning("malformed frame, dropping connection")
                    return

                if frame.msg_type == proto.HELLO:
                    peer_layout = proto.HelloRequest.decode(frame.payload)
                    if peer_layout != self._engine.layout:
                        conn.send(proto.encode_frame(
                            proto.HELLO_ACK, frame.request_id,
                            proto.encode_hello_ack(False, "", 0, "layout mismatch: sidecar was "
                                                   "started with a different KV layout")))
                        return
                    conn.send(proto.encode_frame(
                        proto.HELLO_ACK, frame.request_id,
                        proto.encode_hello_ack(True, _lmcache_version(), self._engine.chunk_size)))
                elif frame.msg_type == proto.PING:
                    conn.send(proto.encode_frame(proto.PONG, frame.request_id, b""))
                elif frame.msg_type == proto.LOOKUP:
                    self._handle_lookup(conn, frame, peer_layout)
                elif frame.msg_type == proto.STORE:
                    self._handle_store(conn, frame, peer_layout)
                else:
                    conn.send(proto.encode_frame(
                        proto.ERROR_MSG, frame.request_id,
                        proto.encode_error(f"unexpected msg_type {frame.msg_type}")))
        except (ConnectionError, OSError):
            pass  # peer went away -- nothing to clean up beyond closing our end
        finally:
            conn.close()

    def _handle_lookup(self, conn: socket.socket, frame: proto.Frame,
                       layout: Optional[proto.HelloRequest]) -> None:
        if layout is None:
            conn.send(proto.encode_frame(proto.ERROR_MSG, frame.request_id,
                                         proto.encode_error("LOOKUP before HELLO")))
            return
        try:
            req = proto.LookupRequest.decode(frame.payload)
        except proto.ProtocolError as e:
            conn.send(proto.encode_frame(proto.ERROR_MSG, frame.request_id, proto.encode_error(str(e))))
            return

        handle = self._engine
        n_tok_aligned = (len(req.token_ids) // handle.chunk_size) * handle.chunk_size
        if n_tok_aligned == 0:
            conn.send(proto.encode_frame(proto.LOOKUP_RESP, frame.request_id,
                                         proto.encode_lookup_resp(0, "", [])))
            return

        region_name = f"/sparkinfer_kv_{frame.request_id}_p"
        n_chunks_possible = n_tok_aligned // handle.chunk_size
        region_size = n_chunks_possible * handle.connector.chunk_byte_size(handle.chunk_size)
        mm = shm.create(region_name, max(region_size, 1))
        matched_tokens = 0  # set below on success; stays 0 if an exception fires first, so the
                            # finally block below knows there is nothing for the C++ side to read
        try:
            with handle.lock:
                handle.connector.begin_request(mm, base_tok=0)
                tokens = torch.tensor(req.token_ids[:n_tok_aligned], dtype=torch.int64)
                try:
                    mask = handle.engine.retrieve(tokens=tokens, shm_name=region_name)
                finally:
                    touched = handle.connector.end_request()

            # Only leading contiguous hits count -- a gap would mean decode has to recompute
            # the missing middle anyway, so nothing after the first miss is usable as a prefix.
            leading = 0
            for i in range(0, n_tok_aligned, handle.chunk_size):
                if bool(mask[i]):
                    leading = i + handle.chunk_size
                else:
                    break
            matched_tokens = leading

            chunks = [
                proto.LookupChunk(
                    start_tok=s, len_tok=e - s,
                    shm_offset_bytes=handle.connector._offset_for(s, e - s),
                )
                for (s, e) in touched
                if s < matched_tokens
            ]
            conn.send(proto.encode_frame(
                proto.LOOKUP_RESP, frame.request_id,
                proto.encode_lookup_resp(matched_tokens, region_name if matched_tokens else "", chunks)))
        except Exception as e:  # noqa: BLE001 -- report as ERROR, never crash the connection
            logger.exception("LOOKUP failed")
            conn.send(proto.encode_frame(proto.ERROR_MSG, frame.request_id, proto.encode_error(str(e))))
        finally:
            mm.close()
            if matched_tokens == 0:
                shm.unlink(region_name)
            # else: a hit -- left for the C++ side to shm_open by name and read (protocol doc's
            # shm lifecycle). Orphaned if the C++ side crashes before consuming it; swept on the
            # next sidecar startup by sweep_orphans() rather than tracked/timed here.

    def _handle_store(self, conn: socket.socket, frame: proto.Frame,
                      layout: Optional[proto.HelloRequest]) -> None:
        if layout is None:
            conn.send(proto.encode_frame(proto.ERROR_MSG, frame.request_id,
                                         proto.encode_error("STORE before HELLO")))
            return
        try:
            req = proto.StoreRequest.decode(frame.payload)
        except proto.ProtocolError as e:
            conn.send(proto.encode_frame(proto.ERROR_MSG, frame.request_id, proto.encode_error(str(e))))
            return

        handle = self._engine
        try:
            mm = shm.attach(req.shm_name)
        except FileNotFoundError as e:
            conn.send(proto.encode_frame(proto.STORE_ACK, frame.request_id,
                                         proto.encode_store_ack(False, str(e))))
            return

        try:
            n_new = req.new_end_tok - req.new_start_tok
            if n_new <= 0 or n_new % handle.chunk_size != 0:
                conn.send(proto.encode_frame(
                    proto.STORE_ACK, frame.request_id,
                    proto.encode_store_ack(False, "store range not chunk-aligned")))
                return

            mask = torch.zeros(len(req.token_ids), dtype=torch.bool)
            mask[req.new_start_tok : req.new_end_tok] = True
            tokens = torch.tensor(req.token_ids, dtype=torch.int64)

            with handle.lock:
                handle.connector.begin_request(mm, base_tok=req.new_start_tok)
                try:
                    handle.engine.store(tokens=tokens, mask=mask)
                finally:
                    handle.connector.end_request()

            conn.send(proto.encode_frame(proto.STORE_ACK, frame.request_id, proto.encode_store_ack(True)))
        except Exception as e:  # noqa: BLE001
            logger.exception("STORE failed")
            conn.send(proto.encode_frame(proto.STORE_ACK, frame.request_id, proto.encode_store_ack(False, str(e))))
        finally:
            mm.close()
            # STORE's shm region was created by the C++ side; per the protocol doc it owns
            # unlinking (does so once it receives this STORE_ACK), not us.


def _lmcache_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("lmcache")
    except Exception:  # noqa: BLE001 -- informational only, never fatal
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="AF_UNIX SOCK_SEQPACKET path to listen on")
    parser.add_argument("--instance-id", default="sparkinfer-lmcache")
    parser.add_argument("--log-level", default="INFO")
    # KV layout, duplicating what HELLO also carries -- see BridgeServer's docstring for why this
    # is CLI args and not deferred to the first HELLO (engine construction is ~10s of first-time
    # import + setup cost that must happen before any connection is accepted, not inside one).
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--int8-kv", action="store_true")
    parser.add_argument("--elem-bytes", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    layout = proto.HelloRequest(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        int8_kv=args.int8_kv,
        elem_bytes=args.elem_bytes,
        model_name=args.model_name,
    )
    logger.info("building LMCacheEngine for layout=%s ...", layout)
    handle = build_engine(layout, args.instance_id)
    logger.info("LMCacheEngine ready (chunk_size=%d)", handle.chunk_size)

    server = BridgeServer(args.socket, handle)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            from lmcache.v1.cache_engine import LMCacheEngineBuilder

            LMCacheEngineBuilder.destroy(args.instance_id)
        except Exception:  # noqa: BLE001 -- best-effort shutdown
            pass


if __name__ == "__main__":
    sys.exit(main())
