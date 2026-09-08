#!/usr/bin/env python3
"""Independent Spark-X2.5 (spark2_5) forward pass, straight from the GGUF's own bytes.

Built to answer one question: does sparkinfer's CUDA path compute the architecture that
XHToken/Spark-X2.5-4B's modeling_spark.py describes? Nothing here is shared with the runtime --
the GGUF is re-parsed, Q8_0 is re-dequantized, and every step follows the reference module
(Spark2_5Attention / Spark2_5MLP / Spark2_5DecoderLayer) rather than the runtime's kernels, so an
agreement is evidence and not a tautology.

Architecture, per config.json + the GGUF metadata:
  36 layers, hidden 2560, ffn 10240, 16 q heads / 4 kv heads, head_dim 256, RMS eps 1e-6
  layer_types: 3 sliding_attention then 1 full_attention, repeating (sliding_window 512)
  rope: full = theta 5e6 over 64 of 256 dims;  sliding = theta 1e4 over all 256
  GeGLU (exact erf GELU), head-wise sigmoid attention output gate, tied embeddings

usage: spark25_ref_check.py <model.gguf> [--tokens 1,2,3] [--dump-logits out.npy]
                            [--layers N] [--compare sparkinfer_logits.npy]
"""
import argparse, struct, sys
import numpy as np

# ---------------------------------------------------------------- GGUF reader

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30

class GGUF:
    def __init__(self, path):
        self.buf = np.memmap(path, dtype=np.uint8, mode='r')
        b = self.buf
        assert b[:4].tobytes() == b'GGUF', 'not a GGUF file'
        o = 4
        ver, = struct.unpack_from('<I', b, o); o += 4
        assert ver == 3, f'GGUF v{ver} unsupported'
        n_tensors, = struct.unpack_from('<Q', b, o); o += 8
        n_kv, = struct.unpack_from('<Q', b, o); o += 8
        self.kv = {}
        SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
        def rd_str(o):
            n, = struct.unpack_from('<Q', b, o); o += 8
            return b[o:o+n].tobytes().decode('utf-8', 'replace'), o + n
        for _ in range(n_kv):
            k, o = rd_str(o)
            t, = struct.unpack_from('<I', b, o); o += 4
            if t == 8:
                v, o = rd_str(o); self.kv[k] = v
            elif t == 9:
                et, = struct.unpack_from('<I', b, o); o += 4
                n, = struct.unpack_from('<Q', b, o); o += 8
                if et == 8:
                    for _ in range(n):
                        ln, = struct.unpack_from('<Q', b, o); o += 8 + ln
                    self.kv[k] = '<str array>'
                else:
                    vals = [struct.unpack_from(FMT[et], b, o + i*SZ[et])[0] for i in range(n)]
                    o += n * SZ[et]
                    self.kv[k] = vals
            else:
                self.kv[k] = struct.unpack_from(FMT[t], b, o)[0]; o += SZ[t]
        self.tensors = {}
        for _ in range(n_tensors):
            name, o = rd_str(o)
            nd, = struct.unpack_from('<I', b, o); o += 4
            dims = [struct.unpack_from('<Q', b, o + 8*d)[0] for d in range(nd)]; o += 8*nd
            dt, = struct.unpack_from('<I', b, o); o += 4
            off, = struct.unpack_from('<Q', b, o); o += 8
            self.tensors[name] = (dims, dt, off)
        align = self.kv.get('general.alignment', 32)
        self.data_start = (o + align - 1) // align * align
        self._cache = {}

    def tensor(self, name):
        """Return the tensor as float32 with shape (dims[1], dims[0]) -- i.e. (out, in), the
        orientation `y = W @ x` wants, since GGUF stores dims[0] fastest."""
        if name in self._cache:
            return self._cache[name]
        if name not in self.tensors:
            raise KeyError(name)
        dims, dt, off = self.tensors[name]
        n = int(np.prod(dims))
        base = self.data_start + off
        if dt == GGML_F32:
            a = np.frombuffer(self.buf[base:base + n*4].tobytes(), dtype=np.float32, count=n)
        elif dt == GGML_F16:
            a = np.frombuffer(self.buf[base:base + n*2].tobytes(), dtype=np.float16, count=n).astype(np.float32)
        elif dt == GGML_BF16:
            raw = np.frombuffer(self.buf[base:base + n*2].tobytes(), dtype=np.uint16, count=n)
            a = (raw.astype(np.uint32) << 16).view(np.float32)
        elif dt == GGML_Q8_0:
            # Q8_0: blocks of 32 = fp16 scale + 32 int8. value = scale * q
            nb = n // 32
            raw = np.frombuffer(self.buf[base:base + nb*34].tobytes(), dtype=np.uint8, count=nb*34)
            raw = raw.reshape(nb, 34)
            scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
            qs = raw[:, 2:].view(np.int8).astype(np.float32)
            a = (qs * scales).reshape(-1)
        else:
            raise NotImplementedError(f'ggml type {dt} for {name}')
        shape = tuple(reversed(dims))          # (dims[-1], ..., dims[0])
        out = a.reshape(shape)
        self._cache[name] = out
        return out


# ---------------------------------------------------------------- reference ops

def bf16(x):
    """Round float32 through bfloat16, the dtype the reference keeps activations in."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even on the truncated 16 low bits
    r = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return (((u + r) & np.uint32(0xFFFF0000))).view(np.float32)

def rms_norm(x, w, eps):
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (w.astype(np.float32) * (x.astype(np.float32) / np.sqrt(v + eps)))

def gelu_erf(x):
    from scipy.special import erf as _erf          # exact GELU, matches ACT2FN["gelu"]
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))

def gelu_erf_np(x):
    # scipy-free fallback: erf via the complementary error function identity is not in numpy,
    # so use the high-accuracy Abramowitz-Stegun 7.1.26 form only if scipy is unavailable.
    try:
        return gelu_erf(x)
    except ImportError:
        t = 1.0 / (1.0 + 0.3275911 * np.abs(x) / np.sqrt(2.0))
        y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*np.exp(-(x/np.sqrt(2.0))**2)
        return 0.5 * x * (1.0 + np.sign(x) * y)

def rope_cos_sin(positions, head_dim, theta, partial_rotary_factor):
    rd = int(head_dim * partial_rotary_factor)
    inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    f = np.outer(np.asarray(positions, dtype=np.float64), inv)
    f = np.concatenate([f, f], axis=-1)
    return np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)

def apply_rope(x, cos, sin):
    """x: [heads, seq, head_dim]; cos/sin: [seq, rope_dim]. Split-half (NeoX) pairing, exactly
    modeling_spark.apply_rotary_pos_emb: rotate the leading rope_dim dims, pass the rest."""
    rd = cos.shape[-1]
    xr = x[..., :rd].astype(np.float32)
    xp = x[..., rd:].astype(np.float32)
    half = rd // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    xr = xr * cos[None, :, :] + rot * sin[None, :, :]
    return np.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


def forward(g, token_ids, n_layers=None, verbose=False):
    P = 'spark2_5.'
    L      = int(g.kv[P+'block_count'])
    H      = int(g.kv[P+'embedding_length'])
    nq     = int(g.kv[P+'attention.head_count'])
    nkv    = int(g.kv[P+'attention.head_count_kv'])
    hd     = int(g.kv[P+'attention.key_length'])
    eps    = float(g.kv[P+'attention.layer_norm_rms_epsilon'])
    win    = int(g.kv[P+'attention.sliding_window'])
    th_full= float(g.kv[P+'rope.freq_base'])
    th_swa = float(g.kv[P+'rope.freq_base_swa'])
    rd_full= int(g.kv[P+'rope.dimension_count'])
    rd_swa = int(g.kv[P+'rope.dimension_count_swa'])
    pattern= [bool(v) for v in g.kv[P+'attention.sliding_window_pattern']]
    if n_layers: L = min(L, n_layers)

    qdim, kvdim = nq*hd, nkv*hd
    T = len(token_ids)
    pos = np.arange(T)

    emb = g.tensor('token_embd.weight')          # (vocab, hidden)
    x = emb[np.asarray(token_ids)].astype(np.float32).copy()   # [T, H]

    cos_f, sin_f = rope_cos_sin(pos, hd, th_full, rd_full / hd)
    cos_s, sin_s = rope_cos_sin(pos, hd, th_swa,  rd_swa  / hd)

    for i in range(L):
        b = f'blk.{i}.'
        swa = pattern[i] if i < len(pattern) else False
        residual = x
        h = rms_norm(x, g.tensor(b+'attn_norm.weight'), eps)

        qkv = h @ g.tensor(b+'attn_qkv.weight').T                 # [T, qdim+2*kvdim]
        q = qkv[:, :qdim].reshape(T, nq, hd).transpose(1, 0, 2)   # [nq, T, hd]
        k = qkv[:, qdim:qdim+kvdim].reshape(T, nkv, hd).transpose(1, 0, 2)
        v = qkv[:, qdim+kvdim:].reshape(T, nkv, hd).transpose(1, 0, 2)
        gate_score = h @ g.tensor(b+'attn_gate.weight').T         # [T, nq]

        cos, sin = (cos_s, sin_s) if swa else (cos_f, sin_f)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        rep = nq // nkv
        kk = np.repeat(k, rep, axis=0)
        vv = np.repeat(v, rep, axis=0)

        scale = 1.0 / np.sqrt(hd)
        scores = np.einsum('hqd,hkd->hqk', q, kk) * scale
        mask = np.tril(np.ones((T, T), dtype=bool))
        if swa and win > 0:
            # HF sliding_window_overlay: kv_idx > q_idx - window, i.e. exactly `window` keys
            # including self.
            idx = np.arange(T)
            mask &= (idx[None, :] > idx[:, None] - win)
        scores = np.where(mask[None, :, :], scores, -np.inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores); p /= p.sum(axis=-1, keepdims=True)
        attn = np.einsum('hqk,hkd->hqd', p, vv)                   # [nq, T, hd]

        gate = 1.0 / (1.0 + np.exp(-gate_score.astype(np.float32)))   # sigmoid, fp32
        attn = attn * gate.T[:, :, None]

        attn = attn.transpose(1, 0, 2).reshape(T, qdim)
        x = residual + attn @ g.tensor(b+'attn_output.weight').T

        residual = x
        h = rms_norm(x, g.tensor(b+'ffn_norm.weight'), eps)
        gg = h @ g.tensor(b+'ffn_gate.weight').T
        uu = h @ g.tensor(b+'ffn_up.weight').T
        x = residual + (gelu_erf_np(gg) * uu) @ g.tensor(b+'ffn_down.weight').T
        if verbose:
            print(f'  layer {i:2d} ({"swa" if swa else "full"}) |x| mean={np.abs(x).mean():.5f}',
                  file=sys.stderr)

    x = rms_norm(x, g.tensor('output_norm.weight'), eps)
    head = g.tensor('output.weight') if 'output.weight' in g.tensors else emb   # tied
    return x @ head.T                                              # [T, vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('--tokens', default='9707,11,847,829,374')
    ap.add_argument('--layers', type=int, default=0)
    ap.add_argument('--dump-logits', default='')
    ap.add_argument('--compare', default='')
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--verify-greedy', type=int, default=0,
                    help='prompt length. Treats the remaining --tokens as a greedy continuation '
                         'produced by another implementation and checks that THIS reference would '
                         'have produced each of them: argmax(logits[p]) must equal tokens[p+1] for '
                         'every p from prompt_len-1 to the end. One teacher-forced pass verifies '
                         'the whole continuation, which is what makes an otherwise slow reference '
                         'usable as a check on a full generation.')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    g = GGUF(a.gguf)
    print(f'arch={g.kv.get("general.architecture")} layers={g.kv.get("spark2_5.block_count")} '
          f'hidden={g.kv.get("spark2_5.embedding_length")} vocab={g.kv.get("spark2_5.vocab_size")}')
    ids = [int(t) for t in a.tokens.split(',') if t.strip()]
    print(f'tokens: {ids}')
    logits = forward(g, ids, n_layers=a.layers or None, verbose=a.verbose)
    last = logits[-1]
    top = np.argsort(-last)[:a.topk]
    print('top-%d for the final position:' % a.topk)
    for r, t in enumerate(top):
        print(f'  {r}: id={t} logit={last[t]:.5f}')
    if a.dump_logits:
        np.save(a.dump_logits, logits.astype(np.float32))
        print(f'wrote {a.dump_logits} shape={logits.shape}')
    if a.verify_greedy:
        n = a.verify_greedy
        if n < 1 or n >= len(ids):
            print(f'--verify-greedy {n} is not a valid prompt length for {len(ids)} tokens')
            return
        pred = logits.argmax(-1)
        bad = []
        for p in range(n - 1, len(ids) - 1):
            if int(pred[p]) != ids[p + 1]:
                bad.append((p, ids[p + 1], int(pred[p])))
        total = len(ids) - n
        print(f'greedy agreement over the {total} continuation tokens: '
              f'{total - len(bad)}/{total}')
        for p, want, got in bad[:20]:
            gap = float(logits[p][got] - logits[p][want])
            print(f'  position {p}: expected {want}, reference argmax {got} '
                  f'(logit gap {gap:.4f})')
        # Margin at each verified step, so an "agrees" result carries how close it came to not.
        margins = []
        for p in range(n - 1, len(ids) - 1):
            row = logits[p]
            top2 = np.partition(row, -2)[-2:]
            margins.append(float(abs(top2[1] - top2[0])))
        if margins:
            print(f'top1-top2 margin over those steps: min={min(margins):.4f} '
                  f'median={float(np.median(margins)):.4f}')
    if a.compare:
        other = np.load(a.compare).astype(np.float32)
        if other.shape != logits.shape:
            other = other.reshape(logits.shape)
        d = np.abs(other - logits)
        rel = d / (np.abs(logits) + 1e-6)
        print(f'compare vs {a.compare}: max_abs={d.max():.6f} mean_abs={d.mean():.6f} '
              f'max_rel={rel.max():.6f}')
        print(f'argmax agree: {int((other.argmax(-1) == logits.argmax(-1)).sum())}/{logits.shape[0]}')

if __name__ == '__main__':
    main()
