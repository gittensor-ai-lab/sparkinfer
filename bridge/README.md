# sparkinfer LMCache bridge sidecar

A small Python process that embeds the real [LMCache](https://github.com/LMCache/LMCache)
library via its documented in-process API and speaks a sparkinfer-owned wire protocol
(`../docs/lmcache_bridge_protocol.md`) back to sparkinfer's C++ server. See that doc for the
"why" (LMCache has no stable non-Python integration surface) and the full protocol spec, and
`/home/autotiny/.claude/plans/fuzzy-riding-taco.md` for the overall design.

Not started automatically by anything in this directory -- `sparkinfer_server` spawns it when
`SPARKINFER_LMCACHE_ENABLE=1` is set (see the server's startup code). This README is for running
it standalone, e.g. to debug the sidecar in isolation from the C++ side.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` is a real ~500MB dependency, pulled in transitively by `lmcache` even for CPU-only KV
offload -- there is no way around it while using LMCache's real API (see `requirements.txt`'s
comment).

## Run standalone

```bash
python3 lmcache_bridge.py \
  --socket /tmp/sparkinfer_lmcache.sock \
  --num-layers 52 --num-kv-heads 2 --head-dim 128 --block-size 16 \
  --elem-bytes 1 --int8-kv \
  --model-name muse-glimmer-30B \
  --log-level DEBUG
```

The KV layout flags must match whatever the C++ side's `BridgeKVLayout` actually is (see
`runtime/include/sparkinfer/lmcache_bridge_client.h`) -- a mismatched HELLO from the C++ side is
rejected (see the protocol doc's HELLO_ACK semantics), not silently coerced.

Engine construction (first-time `torch`/`lmcache` import + `LMCacheEngineBuilder.get_or_create()`
+ `post_init()`) takes on the order of 10 seconds and happens *before* the socket is even bound --
this is deliberate, not a bug (see `BridgeServer`'s docstring in `lmcache_bridge.py`): it keeps
that cost out of any timeout-bound request path on the C++ side.

Useful env vars (see the protocol doc / `lmcache_bridge.py` for the full list):
- `SPARKINFER_LMCACHE_CHUNK_SIZE` (default 256) -- must be a multiple of `--block-size`.
- `SPARKINFER_LMCACHE_MAX_CPU_GB` (default 4.0) -- CPU-RAM backend size cap.
- `LMCACHE_TRACK_USAGE=false` / `DO_NOT_TRACK=1` -- set automatically by `lmcache_bridge.py`
  itself before importing `lmcache` (it phones home to `stats.lmcache.ai:8080` otherwise, found
  during development to hang indefinitely in a restricted-network sandbox). Override only if you
  specifically want telemetry back on.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_protocol.py` is pure unit tests, no dependencies beyond the standard library.
`tests/test_bridge_e2e.py` spawns a real sidecar subprocess against the real `lmcache` CPU-RAM
backend and round-trips actual bytes through it (skipped automatically if `lmcache` isn't
installed) -- slow (~15-30s, dominated by the cold engine-construction cost above), but this is
the test that actually caught a real bug during development: an early version's `LMCacheMetadata`
`kv_shape` silently under-sized LMCache's internal buffer by exactly `elem_bytes`, corrupting the
back half of every retrieved chunk. Don't skip re-running it after touching `ShmConnector` or
`build_engine()`'s shape/dtype math in `lmcache_bridge.py`.
