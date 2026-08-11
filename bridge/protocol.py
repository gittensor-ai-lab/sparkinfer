"""Wire format for the sparkinfer <-> LMCache-sidecar bridge.

Python twin of runtime/src/lmcache_bridge_client.cpp's frame (de)serialization. Both sides must
track ../docs/lmcache_bridge_protocol.md by hand -- there is no IDL/codegen in this design. If you
change one side, update the doc and the other side's constants in the same change.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = 0x53494B56  # 'SIKV' little-endian on the wire
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 2 * 1024 * 1024  # see the protocol doc's "Framing note"

# msg_type values
HELLO = 1
HELLO_ACK = 2
LOOKUP = 3
LOOKUP_RESP = 4
STORE = 5
STORE_ACK = 6
PING = 7
PONG = 8
ERROR_MSG = 9

# FrameHeader: magic(u32) version(u16) msg_type(u16) payload_len(u32) request_id(u32), 16 bytes.
_HEADER_FMT = "<IHHII"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert HEADER_SIZE == 16


class ProtocolError(Exception):
    """A frame was malformed or failed a sanity check. Callers treat this exactly like a
    transport failure -- log it, tear down the connection, never crash the sidecar process."""


@dataclass
class Frame:
    msg_type: int
    request_id: int
    payload: bytes
    version: int = PROTOCOL_VERSION


def encode_frame(msg_type: int, request_id: int, payload: bytes) -> bytes:
    """One bytes object carrying header+payload, meant for a single send() call -- SOCK_SEQPACKET
    preserves message boundaries only if the whole frame goes out in one write."""
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(f"payload too large: {len(payload)} bytes")
    header = struct.pack(_HEADER_FMT, MAGIC, PROTOCOL_VERSION, msg_type, len(payload), request_id)
    return header + payload


def decode_frame(raw: bytes) -> Frame:
    """Decodes exactly one message (the result of one recv() call). Raises ProtocolError on any
    malformed input -- never silently accepts a truncated or corrupt frame."""
    if len(raw) < HEADER_SIZE:
        raise ProtocolError(f"frame too short: {len(raw)} bytes")
    magic, version, msg_type, payload_len, request_id = struct.unpack(
        _HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic:#x}")
    if HEADER_SIZE + payload_len != len(raw):
        raise ProtocolError(
            f"payload_len mismatch: header says {payload_len}, got {len(raw) - HEADER_SIZE}"
        )
    return Frame(msg_type=msg_type, request_id=request_id, payload=raw[HEADER_SIZE:], version=version)


class Writer:
    """Accumulates a payload matching BridgeClient's put_* helpers byte-for-byte."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def u8(self, v: int) -> "Writer":
        self._parts.append(struct.pack("<B", v))
        return self

    def u32(self, v: int) -> "Writer":
        self._parts.append(struct.pack("<I", v))
        return self

    def u64(self, v: int) -> "Writer":
        self._parts.append(struct.pack("<Q", v))
        return self

    def i32(self, v: int) -> "Writer":
        self._parts.append(struct.pack("<i", v))
        return self

    def str(self, s: str) -> "Writer":
        b = s.encode("utf-8")
        self._parts.append(struct.pack("<I", len(b)))
        self._parts.append(b)
        return self

    def bytes(self) -> bytes:
        return b"".join(self._parts)


@dataclass
class Reader:
    """Cursor-based reader mirroring BridgeClient's Reader: every getter raises ProtocolError on
    underflow instead of letting struct.unpack throw a raw exception, so callers get one
    consistent error type to catch."""

    data: bytes
    offset: int = field(default=0)

    def _need(self, n: int) -> None:
        if self.offset + n > len(self.data):
            raise ProtocolError(
                f"read past end: need {n} bytes at offset {self.offset}, have {len(self.data)}"
            )

    def u8(self) -> int:
        self._need(1)
        v = self.data[self.offset]
        self.offset += 1
        return v

    def u32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return v

    def u64(self) -> int:
        self._need(8)
        v = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return v

    def i32(self) -> int:
        self._need(4)
        v = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return v

    def str(self) -> str:
        n = self.u32()
        self._need(n)
        s = self.data[self.offset : self.offset + n].decode("utf-8")
        self.offset += n
        return s

    def i32_array(self, n: int) -> list[int]:
        self._need(4 * n)
        vals = list(struct.unpack_from(f"<{n}i", self.data, self.offset))
        self.offset += 4 * n
        return vals


# --- message-specific helpers ---------------------------------------------------------------
#
# These encode/decode the payload bodies described in the protocol doc. Kept here (not spread
# across lmcache_bridge.py) so the wire format has one home on the Python side, matching how the
# C++ side keeps put_*/Reader next to the frame helpers rather than in the caller.


@dataclass
class HelloRequest:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    int8_kv: bool
    elem_bytes: int
    model_name: str

    @staticmethod
    def decode(payload: bytes) -> "HelloRequest":
        r = Reader(payload)
        return HelloRequest(
            num_layers=r.u32(),
            num_kv_heads=r.u32(),
            head_dim=r.u32(),
            block_size=r.u32(),
            int8_kv=bool(r.u8()),
            elem_bytes=r.u32(),
            model_name=r.str(),
        )


def encode_hello_ack(ok: bool, lmcache_version: str, chunk_size_tokens: int, error: str = "") -> bytes:
    return (
        Writer()
        .u8(1 if ok else 0)
        .str(lmcache_version)
        .u32(chunk_size_tokens)
        .str(error)
        .bytes()
    )


@dataclass
class LookupRequest:
    token_ids: list[int]

    @staticmethod
    def decode(payload: bytes) -> "LookupRequest":
        r = Reader(payload)
        n = r.u32()
        return LookupRequest(token_ids=r.i32_array(n))


@dataclass
class LookupChunk:
    start_tok: int
    len_tok: int
    shm_offset_bytes: int


def encode_lookup_resp(matched_tokens: int, shm_name: str, chunks: list[LookupChunk]) -> bytes:
    w = Writer().i32(matched_tokens).str(shm_name).u32(len(chunks))
    for c in chunks:
        w.i32(c.start_tok).i32(c.len_tok).u64(c.shm_offset_bytes)
    return w.bytes()


@dataclass
class StoreRequest:
    token_ids: list[int]
    new_start_tok: int
    new_end_tok: int
    shm_name: str

    @staticmethod
    def decode(payload: bytes) -> "StoreRequest":
        r = Reader(payload)
        n = r.u32()
        token_ids = r.i32_array(n)
        new_start = r.u32()
        new_end = r.u32()
        shm_name = r.str()
        return StoreRequest(
            token_ids=token_ids, new_start_tok=new_start, new_end_tok=new_end, shm_name=shm_name
        )


def encode_store_ack(ok: bool, error: str = "") -> bytes:
    return Writer().u8(1 if ok else 0).str(error).bytes()


def encode_error(message: str) -> bytes:
    return Writer().str(message).bytes()
