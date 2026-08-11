# sparkinfer &lt;-&gt; LMCache bridge wire protocol

Authoritative spec for the protocol between sparkinfer's C++ server (`runtime/src/lmcache_bridge_client.cpp`)
and the Python sidecar (`bridge/lmcache_bridge.py`) that embeds the real `lmcache` package. Both sides'
constants must track this document by hand — there is no IDL/codegen in this design (deliberately, to avoid
a gRPC/protobuf dependency neither side otherwise needs). If you change one side, update this doc and the
other side's constants in the same change; the HELLO handshake's version check refuses to talk on any
mismatch rather than guessing.

See `/home/autotiny/.claude/plans/fuzzy-riding-taco.md` for the overall design this protocol serves.

## Why this exists

LMCache's only stable, documented integration surface is a Python in-process API
(`lmcache.v1.cache_engine.LMCacheEngineBuilder`/`LMCacheEngine`, driving a caller-supplied
`GPUConnectorInterface` subclass — confirmed by reading the installed package source and round-tripping
`store()`/`retrieve()` against it, see the Stage 1 spike notes below). sparkinfer is pure C++/CUDA with no
Python serving layer. Rather than embed CPython in sparkinfer's tuned single-worker-thread hot path, or
reverse-engineer LMCache's undocumented, pickle-based standalone-server wire protocol, a small sidecar
process embeds LMCache the way it's actually meant to be used, and speaks this protocol — designed and
owned by sparkinfer — back to the C++ server.

## Transport

`AF_UNIX` `SOCK_SEQPACKET` — message-based, so neither side hand-rolls stream reassembly. One socket path
per sparkinfer server instance, passed to the sidecar at spawn time (`--socket <path>`). Fall back to
`SOCK_STREAM` + the length prefix below only if `SOCK_SEQPACKET` proves awkward from Python's `socket`
module during implementation (not expected, but not load-bearing either way since the frame already carries
its own length).

**The C++ side opens two independent connections to the same socket path, each with its own HELLO
handshake**: a "control" connection used for LOOKUP + PING (and the initial HELLO), and a separate
connection used only by the dedicated STORE background thread. This is deliberate, not an accident of
implementation: LOOKUP and STORE share nothing (not even a mutex) so a slow STORE_ACK can never delay a
LOOKUP the worker thread is waiting on within its tight timeout budget. The sidecar must accept multiple
concurrent client connections (each gets its own `LMCacheEngine` interaction through the one underlying
engine instance -- the engine itself is process-wide, only the socket is per-connection).

## Frame format

Fixed 16-byte header, little-endian, followed by `payload_len` bytes of payload:

```c
struct FrameHeader {
    uint32_t magic;        // 0x53494B56 ('SIKV', ASCII "SIKV" byte-swapped by LE encoding)
    uint16_t version;      // protocol version -- see "Versioning" below
    uint16_t msg_type;     // one of the MsgType values below
    uint32_t payload_len;  // bytes following this header
    uint32_t request_id;   // caller-assigned, echoed back in the matching response
};
```

`MsgType`:

| value | name           | direction       |
|-------|----------------|-----------------|
| 1     | HELLO          | C++ -> sidecar  |
| 2     | HELLO_ACK      | sidecar -> C++  |
| 3     | LOOKUP         | C++ -> sidecar  |
| 4     | LOOKUP_RESP    | sidecar -> C++  |
| 5     | STORE          | C++ -> sidecar  |
| 6     | STORE_ACK      | sidecar -> C++  |
| 7     | PING           | C++ -> sidecar  |
| 8     | PONG           | sidecar -> C++  |
| 9     | ERROR          | sidecar -> C++  |

All multi-byte payload fields are little-endian. Strings are length-prefixed (`uint32` byte length,
followed by raw UTF-8 bytes, no NUL terminator).

**Framing note**: `SOCK_SEQPACKET` preserves message boundaries but only if each frame (header + payload)
is written with a single `send()`/`sendmsg()` call and read with a single `recv()` into a buffer large
enough for the whole message -- two separate `send()` calls for header then payload would arrive as two
separate messages, not one. Size receive buffers for the worst case: `token_ids` bounded by sparkinfer's
max context (~163840 tokens, `kMaxBlocksPerSeq * block_size` in `kv_cache.cpp`) means a LOOKUP/STORE
payload can approach ~640KB; implementations should size around 1-2MiB to have headroom.

## Versioning

`version` starts at `1`. HELLO carries the sender's version; HELLO_ACK carries the sidecar's. Any mismatch
is an ERROR + connection close — the C++ side treats this exactly like "sidecar unavailable" (see
Degradation below), never a fatal server error. Bump `version` on any wire-format change to any message
(new/removed/reordered field, changed type) and update both sides in the same change.

## Messages

### HELLO (C++ -> sidecar)

Sent once, immediately after connect, before any LOOKUP/STORE. Payload:

```
uint32 num_layers
uint32 num_kv_heads
uint32 head_dim
uint32 block_size          // sparkinfer's KVCacheManager block_size (tokens/block)
uint8  int8_kv             // 0 = bf16 KV, 1 = int8 K/V + fp16 per-(token,kv_head) scale
uint32 elem_bytes          // bytes per K or V element (2 for bf16, 1 for int8)
str    model_name          // metadata.model_name for the sidecar's LMCacheMetadata
```

### HELLO_ACK (sidecar -> C++)

```
uint8  ok                  // 0 = reject (layout unsupported / config error), 1 = accept
str    lmcache_version     // str(importlib.metadata.version("lmcache")), informational
uint32 chunk_size_tokens   // LMCache's configured chunk_size (sparkinfer default: 256)
str    error                // present (possibly empty) even when ok=1; human-readable on ok=0
```

If `ok=0`, the sidecar closes the connection after sending this frame. The C++ side must not retry the
handshake on the same connection; the reconnect/cooldown logic (see Degradation) applies as if the
connection had failed outright.

### LOOKUP (C++ -> sidecar)

Sent once per eligible prefill (see the gating rule below), before starting the token-loop/batched dispatch
for the range about to be (re)computed.

```
uint32    n_tokens
int32[n]  token_ids
```

`token_ids` is the full prompt prefix from position 0 up to (and including) the range about to be
recomputed -- not just the new range -- matching what STORE sends and what LMCache's own chunk hashing
needs (chunk hashes chain off preceding chunks, per `lmcache.v1.cache_engine.LMCacheEngine.store()`'s own
`tokens`/`mask` contract found during the Stage 1 spike). The sidecar computes matched-length and chunk
boundaries itself; sparkinfer never computes a hash.

Gating rule (enforced on the C++ side, not the sidecar): only sent when the range about to be recomputed is
`>= chunk_size_tokens` (one LMCache chunk, 256 by default) — below that, sparkinfer always recomputes
locally without asking. This bounds the IPC round trip to cases where it can plausibly pay off.

### LOOKUP_RESP (sidecar -> C++)

```
uint32 matched_tokens       // 0 = miss; otherwise the longest matched prefix length, a multiple of
                             // chunk_size_tokens
str    shm_name             // POSIX shm object name holding the matched KV bytes (empty if matched_tokens==0)
uint32 n_chunks
repeated n_chunks:
    uint32 chunk_start_tok
    uint32 chunk_len_tok
    uint64 shm_offset_bytes  // byte offset of this chunk's data within shm_name (K region; see
                              // "shm region layout" below for how V and int8 scales are located)
```

The sidecar has already copied the matched bytes into a freshly created shm region by the time this
response is sent (see Stage 3 in the plan: contiguous staging buffer, not per-block zero-copy, for v1). The
C++ side is responsible for `shm_unlink`-ing (harmless if the sidecar already did) once it has finished
`mmap`-reading, per the "one short-lived shm region per in-flight request" rule below.

### STORE (C++ -> sidecar)

Sent on `clear_prefix_cache()` (about to free an active prefix's blocks for a different one) and
`close_session()` (session ending, prompt was `>= chunk_size_tokens`) -- never on every request.

```
uint32    n_tokens          // full token-ID context needed for LMCache's chunk hashing
int32[n]  token_ids
uint32    new_start_tok     // sub-range actually being stored: [new_start_tok, new_end_tok)
uint32    new_end_tok
str       shm_name          // shm region the C++ side already staged the KV bytes into
```

### STORE_ACK (sidecar -> C++)

```
uint8 ok
str   error   // empty when ok=1
```

STORE is fire-and-forget from the worker thread's perspective (see "Threading" below) — the ack is
consumed by the dedicated background STORE thread, not the worker thread that triggered the store.

### PING / PONG

Empty payload on both. C++ sends PING on a fixed low-frequency timer (~2s), independent of request
traffic, so a hung-but-connected sidecar is detected even with no active requests. Three consecutive missed
PONGs (or one failed `send`/`recv`) marks the bridge dead.

### ERROR (sidecar -> C++)

```
str message
```

Sent for any request the sidecar cannot fulfill for a reason other than a normal cache miss (malformed
frame, internal LMCache exception, etc.). The C++ side treats this identically to a timeout for that one
request (miss / no-op) — it does not tear down the connection unless it's an ERROR sent in place of
HELLO_ACK.

## shm region layout

One short-lived POSIX shm object per in-flight LOOKUP or STORE, not a long-lived pool (simpler correctness
story for the first landable increment; a fixed ring buffer is a reasonable later optimization once traffic
volume is measured). Naming: `/sparkinfer_kv_<request_id>_<side>` where `<side>` is `c` (created by C++, for
STORE) or `p` (created by the sidecar/Python side, for LOOKUP's response) — the creator always
`shm_open(O_CREAT | O_EXCL, ...)`, `ftruncate`s to the exact needed size, `mmap`s, and `shm_unlink`s
*immediately* after mapping (safe, standard POSIX idiom — the unlink only removes the name, not the mapping,
so the peer can still open-then-immediately-fail only if it tries to open by name after the creator unlinks;
to avoid that race, the creator does not unlink until it has confirmed the peer no longer needs the name --
in practice this means: STORE's creator (C++) unlinks only after receiving STORE_ACK; LOOKUP's creator
(sidecar) unlinks only after including a `Content-Length`-equivalent (`n_chunks`/offsets) in LOOKUP_RESP,
which the C++ side uses to `shm_open` *by name* one time within the same round trip before the sidecar
would ever unlink -- the sidecar unlinks only after the C++ side's *next* frame (e.g. the following PING)
confirms liveness, giving a generous window; if this ordering proves fragile in implementation, fall back to
never unlinking proactively and instead sweeping `/dev/shm/sparkinfer_kv_*` for orphans older than a few
seconds on sidecar startup).

Layout is per-chunk: each `LOOKUP_RESP` chunk entry's `shm_offset_bytes` points to the start of
that chunk's own self-contained block below (sized from that chunk's own `chunk_len_tok`), not a
single layout shared across all `n_chunks`. A multi-chunk match is `n_chunks` independent blocks
back-to-back in the same region, each addressed by its own offset -- a reader must not assume
chunk N+1 immediately follows chunk N's computed size; always use the offset given.

Contiguous, no padding beyond natural alignment, within one chunk's block:

```
[K bytes: num_layers * chunk_blocks * block_bytes]
[V bytes: num_layers * chunk_blocks * block_bytes]
[K scale bytes: num_layers * chunk_blocks * block_size * num_kv_heads * 2  -- only when int8_kv]
[V scale bytes: num_layers * chunk_blocks * block_size * num_kv_heads * 2  -- only when int8_kv]
```

where `block_bytes = block_size * num_kv_heads * head_dim * elem_bytes` and `chunk_blocks = chunk_len_tok /
block_size`. All of this is opaque `uint8` as far as the sidecar/LMCache is concerned (see the Stage 1 spike
note on `LMCacheMetadata.kv_dtype = torch.uint8`) -- sparkinfer is the only side that interprets these bytes
as bf16 or int8+scale.

## Threading / must-not-block-worker-thread rule

sparkinfer has exactly one worker thread driving all GPU work (`ContinuousBatchEngine::worker_loop`). It
must never block on the sidecar indefinitely:

- LOOKUP is synchronous-with-timeout: `SO_RCVTIMEO` set to `SPARKINFER_LMCACHE_LOOKUP_TIMEOUT_MS`
  (default 5) on the socket. A timeout is treated as a miss and execution falls through to normal recompute
  -- never an error surfaced to the request.
- STORE is fire-and-forget from the worker thread's perspective: the worker thread hands the store
  descriptor (already-staged shm region + token range) to one dedicated background STORE thread, which owns
  the actual frame send + STORE_ACK wait + shm unlink. A slow or absent STORE_ACK never stalls `step_job`.
- On any socket error, malformed frame, or ERROR-in-place-of-HELLO_ACK: mark the bridge dead, apply an
  exponential-backoff cooldown (start 5s, cap at some reasonable ceiling e.g. 60s) before the next reconnect
  attempt, which happens lazily on the next eligible LOOKUP/STORE rather than on a separate timer thread.
  While dead: every LOOKUP silently returns a miss, every STORE silently no-ops. The model server never
  refuses to serve, never surfaces a bridge-down condition as a request-visible error.

## Degradation is always safe

Every failure mode above (sidecar never started, sidecar crashed, HELLO layout mismatch, timeout, malformed
frame) degrades to exactly today's behavior: no cross-request cache, full recompute. This is the single
invariant every implementation change to this protocol must preserve.

## Stage 1 spike notes (informing this spec)

Confirmed against `lmcache==0.5.3` (see `/home/autotiny/.claude/plans/fuzzy-riding-taco.md` Stage 1):

- `LMCacheEngineBuilder.get_or_create(instance_id, config, metadata, gpu_connector, broadcast_fn,
  broadcast_object_fn)` then an explicit `engine.post_init()` call -- easy to miss, `store()`/`retrieve()`
  raise `AssertionError` on `storage_manager` without it.
- `LMCacheMetadata.kv_dtype = torch.uint8` works -- LMCache does not require an interpretable float dtype,
  confirming the "opaque bytes" design above needs no int8-to-bf16 conversion.
- The integration point is `lmcache.v1.gpu_connector.GPUConnectorInterface`: `to_gpu`/`from_gpu`/
  `batched_to_gpu`/`batched_from_gpu`/`get_shape`, given a `MemoryObj` LMCache owns (its `.byte_array`
  gives raw byte access). `bridge/lmcache_bridge.py`'s connector subclass implements these against the
  POSIX shm regions this protocol describes, in place of a real GPU.
- `MemoryObj.byte_array` is a ctypes-backed `memoryview` reporting an explicit-byte-order format (`<B`) --
  CPython's memoryview slice-assignment rejects that outright (`NotImplementedError: unsupported format`).
  Call `.cast("B")` before any slice-assignment into it.
- LMCache phones home to `stats.lmcache.ai:8080` for usage telemetry by default, with no fast timeout in a
  restricted-network environment -- the sidecar process **must** set `LMCACHE_TRACK_USAGE=false` and
  `DO_NOT_TRACK=1` in its environment before importing `lmcache`, or a slow/blocked network path can hang
  engine construction.
- LMCache starts non-daemon background periodic threads (e.g. `PinMonitor-thread`) that keep the process
  alive after `main()` logic completes -- the sidecar's shutdown path must call
  `LMCacheEngineBuilder.destroy(instance_id)` (or find whatever the maintained equivalent is at
  implementation time) rather than relying on the process exiting on its own.
