#!/usr/bin/env python3
"""DS5 oracle fixture generator (ADR-002 oracle-fixture strategy, ADR-005 formats).

Generates golden per-op tensors + end-to-end logits that gate every Zig/Metal
kernel. The oracle forward pass is implemented here as pure fp32 torch ops over
a GGUF-named weight dict and is cross-checked against HF transformers'
Qwen3MoeForCausalLM on the synthetic model, so real-model fixture generation
never has to instantiate the full HF model (weights stream lazily from
safetensors shards).

Modes:
  --synthetic            downsized 4-layer/8-expert model (contracts.SYNTH_TINY;
                         MUST stay in sync). Q8_0 quantization is BAKED INTO the
                         model weights before any fixture is computed, and a
                         matching tests/fixtures/synthetic/model.gguf is emitted,
                         so kernels/parser/e2e all see identical weight values.
  --hf DIR               real Qwen3-MoE HF checkpoint (safetensors). Per-op
                         fixtures for --layers; with --e2e also streamed greedy
                         decode + logits for the 5 prompts (hours of disk I/O on
                         a 30B; run on a machine with the download).

Usage:
  python tools/make_fixtures.py --synthetic --out tests/fixtures/synthetic
  python tools/make_fixtures.py --hf ~/models/Qwen3-30B-A3B-Instruct-2507 \
      --out tests/fixtures/qwen3-30b-a3b --layers 0,23,47 --e2e

Requires: torch, transformers, numpy, safetensors (see .venv).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# GGML dtype ids (contracts.Dtype)
# ---------------------------------------------------------------------------
F32, F16, Q8_0, I32 = 0, 1, 8, 26

DS5T_MAGIC = 0x54355344  # "DS5T"

# Default tolerances per op (ADR-005 §4).
TOL = {
    "rmsnorm": (1e-5, 1e-4),
    "rope": (1e-5, 1e-4),
    "matmul_quant": (5e-4, 2e-3),
    "attention": (1e-4, 1e-3),
    "router": (1e-5, 0.0),
    "expert_mlp": (1e-3, 5e-3),
    "layer": (2e-3, 1e-2),
    "logits": (5e-2, 5e-2),
}

# Synthetic config. MUST stay byte-identical to contracts.SYNTH_TINY.
SYNTH_CONFIG = {
    "n_layers": 4,
    "hidden_dim": 256,
    "n_q_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 64,
    "n_experts": 8,
    "top_k": 4,
    "expert_ffn_dim": 128,
    "vocab_size": 512,
    "rms_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "norm_topk_prob": True,
    "max_ctx": 512,
}


# ---------------------------------------------------------------------------
# DS5T tensor files. numpy C-order: last axis is contiguous, so GGUF-style
# ne[] (ne[0] fastest-varying) is the REVERSED numpy shape.
# ---------------------------------------------------------------------------

def write_ds5t(path: Path, data: bytes, dtype_id: int, np_shape: tuple[int, ...]):
    ne = list(reversed(np_shape)) + [1] * (4 - len(np_shape))
    hdr = struct.pack(
        "<IIII4QQQ", DS5T_MAGIC, 1, dtype_id, len(np_shape), *ne, len(data), 0
    )
    assert len(hdr) == 64
    path.write_bytes(hdr + data)


def write_f32(path: Path, t: torch.Tensor):
    a = t.detach().to(torch.float32).contiguous().numpy()
    write_ds5t(path, a.tobytes(), F32, a.shape)


def write_i32(path: Path, t: torch.Tensor):
    a = t.detach().to(torch.int32).contiguous().numpy()
    write_ds5t(path, a.tobytes(), I32, a.shape)


# ---------------------------------------------------------------------------
# Q8_0: blocks of 32 along the row (last numpy axis). Block = f16 scale + 32×i8.
# Frozen dequant semantics: value = f32(f16(d)) * q  (ADR-005 §1).
# ---------------------------------------------------------------------------

def _roundf(x: np.ndarray) -> np.ndarray:
    """C roundf: round half away from zero (numpy rounds half to even)."""
    return np.trunc(x + np.copysign(0.5, x))


def q8_0_quantize(w: np.ndarray) -> tuple[bytes, np.ndarray]:
    """w: fp32 (..., k) with k % 32 == 0. Returns (gguf block bytes, dequant fp32)."""
    assert w.dtype == np.float32 and w.shape[-1] % 32 == 0
    lead = w.shape[:-1]
    nb = w.shape[-1] // 32
    b = w.reshape(*lead, nb, 32)
    amax = np.abs(b).max(axis=-1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = _roundf(b * inv[..., None]).astype(np.int8)
    d16 = d.astype(np.float16)
    blob = np.empty((*lead, nb, 34), np.uint8)
    blob[..., 0:2] = d16[..., None].view(np.uint8)
    blob[..., 2:] = q.view(np.uint8)
    deq = (q.astype(np.float32) * d16.astype(np.float32)[..., None]).reshape(w.shape)
    return blob.tobytes(), deq


def bake_q8(t: torch.Tensor) -> tuple[bytes, torch.Tensor]:
    blob, deq = q8_0_quantize(t.detach().to(torch.float32).contiguous().numpy())
    return blob, torch.from_numpy(deq)


# ---------------------------------------------------------------------------
# Oracle forward pass: pure fp32 torch over GGUF-named weights (ADR-005 §6).
# Weight shapes are torch/HF convention: proj weights (out_features, in_features);
# expert banks (n_experts, out, in). GGUF ne[] is the reverse, which matches.
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = (x * x).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (T, n_heads, head_dim); cos/sin: (T, head_dim)
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def gqa_attention(q, k_cache, v_cache, pos: int, scale: float) -> torch.Tensor:
    """q: (Tq, Hq, hd); caches: (Hkv, Tk, hd) with Tk = pos + Tq valid.
    Causal: query token t attends to cache positions 0..pos+t. Returns (Tq, Hq*hd)."""
    tq, hq, hd = q.shape
    hkv, tk, _ = k_cache.shape
    rep = hq // hkv
    k = k_cache.repeat_interleave(rep, dim=0)  # (Hq, Tk, hd)
    v = v_cache.repeat_interleave(rep, dim=0)
    scores = torch.einsum("qhd,hkd->hqk", q, k) * scale
    mask = torch.arange(tk)[None, None, :] > (pos + torch.arange(tq))[None, :, None]
    scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores.to(torch.float32), dim=-1)
    out = torch.einsum("hqk,hkd->qhd", p, v)
    return out.reshape(tq, hq * hd)


def router_topk(h: torch.Tensor, w: torch.Tensor, top_k: int, norm: bool):
    """ADR-005 §2. h: (T, dim); w: (n_experts, dim). Returns ids (T,k) i64, weights (T,k) f32."""
    logits = h @ w.T
    p = torch.softmax(logits.to(torch.float32), dim=-1)
    top_v, top_i = torch.topk(p, top_k, dim=-1)
    if norm:
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    return top_i, top_v, p


def expert_mlp(h, ids, weights, gate_bank, up_bank, down_bank) -> torch.Tensor:
    """Accumulated MoE output (no residual). Banks: (E, out, in)."""
    out = torch.zeros_like(h)
    t_count, k = ids.shape
    for t in range(t_count):
        for j in range(k):
            e = int(ids[t, j])
            x = h[t]
            g = torch.nn.functional.silu(gate_bank[e] @ x)
            u = up_bank[e] @ x
            out[t] += weights[t, j] * (down_bank[e] @ (g * u))
    return out


class LayerTrace:
    """Intermediates captured while running one decoder layer."""

    __slots__ = (
        "x_in", "h_norm", "q_pre", "q_normed", "q_roped", "k_pre", "k_normed",
        "k_roped", "v", "attn_out", "x_mid", "h2", "ids", "gate_w", "probs",
        "moe_out", "x_out",
    )


def layer_forward(x, wp, li: int, cfg: dict, positions, trace: LayerTrace | None = None):
    """One transformer block per ADR-005 §6. x: (T, dim). wp: name -> fp32 tensor."""
    hq, hkv, hd = cfg["n_q_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    eps = cfg["rms_eps"]
    t_len = x.shape[0]
    pre = f"blk.{li}."

    h = rms_norm(x, wp(pre + "attn_norm.weight"), eps)
    q = (h @ wp(pre + "attn_q.weight").T).view(t_len, hq, hd)
    k = (h @ wp(pre + "attn_k.weight").T).view(t_len, hkv, hd)
    v = (h @ wp(pre + "attn_v.weight").T).view(t_len, hkv, hd)
    qn = rms_norm(q, wp(pre + "attn_q_norm.weight"), eps)
    kn = rms_norm(k, wp(pre + "attn_k_norm.weight"), eps)
    cos, sin = rope_cos_sin(positions, hd, cfg["rope_theta"])
    qr = apply_rope(qn, cos, sin)
    kr = apply_rope(kn, cos, sin)
    a = gqa_attention(
        qr, kr.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous(),
        pos=0, scale=1.0 / math.sqrt(hd),
    )
    x_mid = x + a @ wp(pre + "attn_output.weight").T
    h2 = rms_norm(x_mid, wp(pre + "ffn_norm.weight"), eps)
    ids, gate_w, probs = router_topk(
        h2, wp(pre + "ffn_gate_inp.weight"), cfg["top_k"], cfg["norm_topk_prob"]
    )
    moe = expert_mlp(
        h2, ids, gate_w,
        wp(pre + "ffn_gate_exps.weight"), wp(pre + "ffn_up_exps.weight"),
        wp(pre + "ffn_down_exps.weight"),
    )
    x_out = x_mid + moe

    if trace is not None:
        trace.x_in, trace.h_norm = x, h
        trace.q_pre, trace.q_normed, trace.q_roped = q, qn, qr
        trace.k_pre, trace.k_normed, trace.k_roped = k, kn, kr
        trace.v, trace.attn_out, trace.x_mid, trace.h2 = v, a, x_mid, h2
        trace.ids, trace.gate_w, trace.probs, trace.moe_out = ids, gate_w, probs, moe
        trace.x_out = x_out
    return x_out


def model_forward(tokens: list[int], wp, cfg: dict, want_traces: set[int] = frozenset()):
    """Full prefill forward. Returns (logits (T, vocab), {layer: LayerTrace})."""
    positions = torch.arange(len(tokens))
    x = wp("token_embd.weight")[torch.tensor(tokens)]
    traces = {}
    for li in range(cfg["n_layers"]):
        tr = LayerTrace() if li in want_traces else None
        x = layer_forward(x, wp, li, cfg, positions, tr)
        if tr is not None:
            traces[li] = tr
    x = rms_norm(x, wp("output_norm.weight"), cfg["rms_eps"])
    logits = x @ wp("output.weight").T
    return logits, traces


def greedy_decode(tokens: list[int], wp, cfg: dict, n_new: int) -> list[int]:
    """Greedy continuation; argmax ties -> lowest token id (np.argmax). Recomputes
    the full prefix each step (no KV cache) — oracle simplicity over speed."""
    seq = list(tokens)
    for _ in range(n_new):
        logits, _ = model_forward(seq, wp, cfg)
        seq.append(int(np.argmax(logits[-1].numpy())))
    return seq


# ---------------------------------------------------------------------------
# Synthetic model: HF Qwen3MoeForCausalLM, seeded, Q8_0 baked into weights.
# ---------------------------------------------------------------------------

def build_synthetic(seed: int):
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    c = SYNTH_CONFIG
    hf_cfg = Qwen3MoeConfig(
        vocab_size=c["vocab_size"],
        hidden_size=c["hidden_dim"],
        intermediate_size=c["expert_ffn_dim"] * 2,
        moe_intermediate_size=c["expert_ffn_dim"],
        num_hidden_layers=c["n_layers"],
        num_attention_heads=c["n_q_heads"],
        num_key_value_heads=c["n_kv_heads"],
        head_dim=c["head_dim"],
        num_experts=c["n_experts"],
        num_experts_per_tok=c["top_k"],
        norm_topk_prob=c["norm_topk_prob"],
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_theta=c["rope_theta"],
        rms_norm_eps=c["rms_eps"],
        max_position_embeddings=c["max_ctx"],
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
        attn_implementation="eager",
        use_cache=False,
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(hf_cfg).eval().float()
    # Randomize norm weights away from the trivial all-ones init so rmsnorm
    # fixtures actually exercise the weight multiply.
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1:
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
    return model


def extract_synthetic_weights(model, cfg: dict):
    """GGUF-named fp32 weight dict + Q8_0 baking. Quantized tensors are replaced
    by their dequantized values BOTH in the returned dict and inside the HF
    model, and the raw blobs are kept for the GGUF. Router + norms stay f32."""
    sd = {p: q for p, q in model.named_parameters()}
    weights: dict[str, torch.Tensor] = {}
    blobs: dict[str, bytes] = {}

    def put_q8(gguf_name: str, hf_param: torch.Tensor):
        blob, deq = bake_q8(hf_param)
        hf_param.copy_(deq)
        weights[gguf_name] = deq
        blobs[gguf_name] = blob

    def put_f32(gguf_name: str, hf_param: torch.Tensor):
        weights[gguf_name] = hf_param.detach().clone()

    put_q8("token_embd.weight", sd["model.embed_tokens.weight"])
    put_q8("output.weight", sd["lm_head.weight"])
    put_f32("output_norm.weight", sd["model.norm.weight"])
    for li in range(cfg["n_layers"]):
        h = f"model.layers.{li}."
        g = f"blk.{li}."
        put_f32(g + "attn_norm.weight", sd[h + "input_layernorm.weight"])
        put_f32(g + "ffn_norm.weight", sd[h + "post_attention_layernorm.weight"])
        put_f32(g + "attn_q_norm.weight", sd[h + "self_attn.q_norm.weight"])
        put_f32(g + "attn_k_norm.weight", sd[h + "self_attn.k_norm.weight"])
        for proj, gn in [("q_proj", "attn_q"), ("k_proj", "attn_k"),
                         ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            put_q8(g + gn + ".weight", sd[h + f"self_attn.{proj}.weight"])
        put_f32(g + "ffn_gate_inp.weight", sd[h + "mlp.gate.weight"])
        # transformers 5.x fuses experts: gate_up_proj (E, 2*ffn, hidden).
        gu = sd[h + "mlp.experts.gate_up_proj"]
        dn = sd[h + "mlp.experts.down_proj"]
        ffn = cfg["expert_ffn_dim"]
        gate_b, gate_d = bake_q8(gu[:, :ffn, :])
        up_b, up_d = bake_q8(gu[:, ffn:, :])
        dn_b, dn_d = bake_q8(dn)
        gu[:, :ffn, :] = gate_d
        gu[:, ffn:, :] = up_d
        dn.copy_(dn_d)
        weights[g + "ffn_gate_exps.weight"] = gate_d
        weights[g + "ffn_up_exps.weight"] = up_d
        weights[g + "ffn_down_exps.weight"] = dn_d
        blobs[g + "ffn_gate_exps.weight"] = gate_b
        blobs[g + "ffn_up_exps.weight"] = up_b
        blobs[g + "ffn_down_exps.weight"] = dn_b
    return weights, blobs


# ---------------------------------------------------------------------------
# Minimal GGUF v3 writer + reader (synthetic model artifact + self-test).
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_ALIGN = 32
GV_U32, GV_F32, GV_BOOL, GV_STR, GV_U64 = 4, 6, 7, 8, 10


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, cfg: dict, weights: dict[str, torch.Tensor],
               blobs: dict[str, bytes]):
    meta: list[tuple[str, int, bytes]] = [
        ("general.architecture", GV_STR, _gguf_str("qwen3moe")),
        ("general.name", GV_STR, _gguf_str("ds5-synthetic-tiny")),
        ("general.alignment", GV_U32, struct.pack("<I", GGUF_ALIGN)),
        ("qwen3moe.block_count", GV_U32, struct.pack("<I", cfg["n_layers"])),
        ("qwen3moe.embedding_length", GV_U32, struct.pack("<I", cfg["hidden_dim"])),
        ("qwen3moe.attention.head_count", GV_U32, struct.pack("<I", cfg["n_q_heads"])),
        ("qwen3moe.attention.head_count_kv", GV_U32, struct.pack("<I", cfg["n_kv_heads"])),
        ("qwen3moe.attention.key_length", GV_U32, struct.pack("<I", cfg["head_dim"])),
        ("qwen3moe.attention.value_length", GV_U32, struct.pack("<I", cfg["head_dim"])),
        ("qwen3moe.attention.layer_norm_rms_epsilon", GV_F32, struct.pack("<f", cfg["rms_eps"])),
        ("qwen3moe.rope.freq_base", GV_F32, struct.pack("<f", cfg["rope_theta"])),
        ("qwen3moe.expert_count", GV_U32, struct.pack("<I", cfg["n_experts"])),
        ("qwen3moe.expert_used_count", GV_U32, struct.pack("<I", cfg["top_k"])),
        ("qwen3moe.expert_feed_forward_length", GV_U32, struct.pack("<I", cfg["expert_ffn_dim"])),
        ("qwen3moe.context_length", GV_U32, struct.pack("<I", cfg["max_ctx"])),
        ("qwen3moe.vocab_size", GV_U32, struct.pack("<I", cfg["vocab_size"])),
    ]

    tensors: list[tuple[str, list[int], int, bytes]] = []  # name, ne, ggml_type, data
    for name in sorted(weights):
        if name in blobs:
            ne = list(reversed(weights[name].shape))
            tensors.append((name, ne, Q8_0, blobs[name]))
        else:
            a = weights[name].to(torch.float32).contiguous().numpy()
            tensors.append((name, list(reversed(a.shape)), F32, a.tobytes()))

    out = bytearray()
    out += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(meta))
    for key, vt, vb in meta:
        out += _gguf_str(key) + struct.pack("<I", vt) + vb

    infos = bytearray()
    data = bytearray()
    for name, ne, gt, blob in tensors:
        off = len(data)
        infos += _gguf_str(name)
        infos += struct.pack("<I", len(ne))
        infos += struct.pack(f"<{len(ne)}Q", *ne)
        infos += struct.pack("<IQ", gt, off)
        data += blob
        pad = (-len(data)) % GGUF_ALIGN
        data += b"\x00" * pad
    out += infos
    out += b"\x00" * ((-len(out)) % GGUF_ALIGN)
    out += data
    path.write_bytes(bytes(out))


def read_gguf(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Self-test reader: returns (metadata, name -> fp32 array (Q8_0 dequantized))."""
    buf = path.read_bytes()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return vals if len(vals) > 1 else vals[0]

    def take_str():
        nonlocal off
        n = take("<Q")
        s = buf[off:off + n].decode()
        off += n
        return s

    magic, ver, n_tensors, n_meta = take("<IIQQ")
    assert magic == GGUF_MAGIC and ver == 3
    meta = {}
    for _ in range(n_meta):
        key = take_str()
        vt = take("<I")
        if vt == GV_STR:
            meta[key] = take_str()
        elif vt == GV_U32:
            meta[key] = take("<I")
        elif vt == GV_U64:
            meta[key] = take("<Q")
        elif vt == GV_F32:
            meta[key] = take("<f")
        elif vt == GV_BOOL:
            meta[key] = bool(take("<B"))
        else:
            raise ValueError(f"unhandled gguf value type {vt}")
    infos = []
    for _ in range(n_tensors):
        name = take_str()
        nd = take("<I")
        ne = struct.unpack_from(f"<{nd}Q", buf, off)
        off += 8 * nd
        gt, doff = take("<IQ")
        infos.append((name, list(ne), gt, doff))
    data_start = (off + GGUF_ALIGN - 1) // GGUF_ALIGN * GGUF_ALIGN
    tensors = {}
    for name, ne, gt, doff in infos:
        n_elems = int(np.prod(ne))
        shape = tuple(reversed(ne))
        if gt == F32:
            a = np.frombuffer(buf, np.float32, n_elems, data_start + doff).reshape(shape)
        elif gt == Q8_0:
            nb = n_elems // 32
            raw = np.frombuffer(buf, np.uint8, nb * 34, data_start + doff).reshape(nb, 34)
            d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
            q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            a = (q * d).reshape(shape)
        else:
            raise ValueError(f"unhandled ggml type {gt}")
        tensors[name] = a
    return meta, tensors


# ---------------------------------------------------------------------------
# HF safetensors lazy weight provider (real models; no HF model instantiated).
# ---------------------------------------------------------------------------

class HfProvider:
    """wp(gguf_name) -> fp32 tensor, read lazily from safetensors shards.
    Stacks per-expert tensors into banks; caches nothing beyond an LRU of 1
    bank to keep the 24GB dev box viable."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.dir = model_dir
        self.safe_open = safe_open
        self.cfg_json = json.loads((model_dir / "config.json").read_text())
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.exists():
            self.index = json.loads(idx_path.read_text())["weight_map"]
        else:
            single = model_dir / "model.safetensors"
            with safe_open(single, framework="pt") as f:
                self.index = {k: "model.safetensors" for k in f.keys()}
        self._cache: dict[str, torch.Tensor] = {}
        self._shards: dict[str, object] = {}  # kept open across reads

    def config(self) -> dict:
        c = self.cfg_json
        return {
            "n_layers": c["num_hidden_layers"],
            "hidden_dim": c["hidden_size"],
            "n_q_heads": c["num_attention_heads"],
            "n_kv_heads": c["num_key_value_heads"],
            "head_dim": c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            "n_experts": c["num_experts"],
            "top_k": c["num_experts_per_tok"],
            "expert_ffn_dim": c["moe_intermediate_size"],
            "vocab_size": c["vocab_size"],
            "rms_eps": c["rms_norm_eps"],
            "rope_theta": float(c["rope_theta"]),
            "norm_topk_prob": c["norm_topk_prob"],
            "max_ctx": c["max_position_embeddings"],
        }

    def _read(self, hf_name: str) -> torch.Tensor:
        shard = self.index[hf_name]
        if shard not in self._shards:
            self._shards[shard] = self.safe_open(self.dir / shard, framework="pt")
        return self._shards[shard].get_tensor(hf_name).to(torch.float32)

    _MAP = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    _LAYER_MAP = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    _BANK = {"ffn_gate_exps.weight": "gate_proj", "ffn_up_exps.weight": "up_proj",
             "ffn_down_exps.weight": "down_proj"}

    def __call__(self, gguf_name: str) -> torch.Tensor:
        if gguf_name in self._MAP:
            hf = self._MAP[gguf_name]
            if hf == "lm_head.weight" and hf not in self.index:
                hf = "model.embed_tokens.weight"  # tied embeddings fallback
            return self._read(hf)
        assert gguf_name.startswith("blk.")
        _, li, rest = gguf_name.split(".", 2)
        base = f"model.layers.{li}."
        if rest in self._LAYER_MAP:
            return self._read(base + self._LAYER_MAP[rest])
        if rest in self._BANK:
            if gguf_name in self._cache:
                return self._cache[gguf_name]
            proj = self._BANK[rest]
            n_experts = self.cfg_json["num_experts"]
            bank = torch.stack(
                [self._read(base + f"mlp.experts.{e}.{proj}.weight")
                 for e in range(n_experts)]
            )
            self._cache.clear()
            self._cache[gguf_name] = bank
            return bank
        raise KeyError(gguf_name)


# ---------------------------------------------------------------------------
# Fixture case emission
# ---------------------------------------------------------------------------

class FixtureSet:
    def __init__(self, out_dir: Path, model_name: str, cfg: dict, seed: int):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cases: list[dict] = []
        self.prompts: list[dict] = []
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip() or "unknown"
        import transformers
        self.manifest = {
            "ds5_fixture_version": 1,
            "model": {"name": model_name, "config": cfg},
            "generator": {
                "tool": "make_fixtures.py",
                "git_commit": git,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": seed,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "cases": self.cases,
            "prompts": self.prompts,
        }

    def add(self, op: str, name: str, params: dict, tensors: dict[str, tuple], tol=None):
        """tensors: role -> (writer_fn, tensor)."""
        entry_t = {}
        for role, (writer, t) in tensors.items():
            fn = f"{name}.{role}.ds5t"
            writer(self.dir / fn, t)
            entry_t[role] = fn
        atol, rtol = tol or TOL[op]
        self.cases.append({"op": op, "name": name, "params": params,
                           "tensors": entry_t, "tolerance": {"atol": atol, "rtol": rtol}})

    def add_prompt(self, name: str, text: str | None, token_ids: list[int],
                   greedy: list[int], logits: torch.Tensor):
        fn = f"{name}.logits.ds5t"
        write_f32(self.dir / fn, logits)
        atol, rtol = TOL["logits"]
        self.prompts.append({
            "name": name, "text": text, "token_ids": token_ids,
            "greedy_tokens": greedy, "logits": fn,
            "tolerance": {"atol": atol, "rtol": rtol},
        })

    def save(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1))


def raw_writer(dtype_id: int, np_shape: tuple[int, ...]):
    return lambda path, blob: write_ds5t(path, blob, dtype_id, np_shape)


def assert_no_router_neartie(probs: torch.Tensor, top_k: int, tol=1e-6):
    """ADR-005 §2: fixture inputs must not depend on the tie-break rule."""
    s, _ = torch.sort(probs, dim=-1, descending=True)
    gaps = s[:, :top_k + 1].diff(dim=-1).abs()
    assert gaps.min() > tol, "router fixture has a near-tie; change seed/prompt"


def emit_layer_cases(fx: FixtureSet, wp, cfg: dict, li: int, trace: LayerTrace,
                     positions: torch.Tensor):
    t_len = int(positions.shape[0])
    hq, hkv, hd = cfg["n_q_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    dim, ffn = cfg["hidden_dim"], cfg["expert_ffn_dim"]
    eps = cfg["rms_eps"]
    n = f"l{li}"

    fx.add("rmsnorm", f"{n}_attn_norm", {"eps": eps, "n_rows": t_len, "dim": dim}, {
        "input": (write_f32, trace.x_in),
        "weight": (write_f32, wp(f"blk.{li}.attn_norm.weight")),
        "output": (write_f32, trace.h_norm),
    })
    fx.add("rmsnorm", f"{n}_q_norm",
           {"eps": eps, "n_rows": t_len * hq, "dim": hd}, {
        "input": (write_f32, trace.q_pre.reshape(t_len * hq, hd)),
        "weight": (write_f32, wp(f"blk.{li}.attn_q_norm.weight")),
        "output": (write_f32, trace.q_normed.reshape(t_len * hq, hd)),
    })

    # matmul_quant: oracle output recomputed from the dequantized Q8_0 weight so
    # quantization error is excluded (identical values to what the kernel reads).
    wq = wp(f"blk.{li}.attn_q.weight")
    blob, deq = bake_q8(wq)
    fx.add("matmul_quant", f"{n}_attn_q",
           {"m": t_len, "n": hq * hd, "k": dim, "w_dtype": "q8_0"}, {
        "input": (write_f32, trace.h_norm),
        "weight": (raw_writer(Q8_0, tuple(wq.shape)), blob),
        "output": (write_f32, trace.h_norm @ deq.T),
    })

    fx.add("rope", f"{n}_rope_q",
           {"theta": cfg["rope_theta"], "n_tokens": t_len, "n_heads": hq,
            "head_dim": hd, "freq_scale": 1.0}, {
        "input": (write_f32, trace.q_normed),
        "positions": (write_i32, positions),
        "output": (write_f32, trace.q_roped),
    })

    k_cache = trace.k_roped.transpose(0, 1).contiguous()  # (Hkv, T, hd)
    v_cache = trace.v.transpose(0, 1).contiguous()
    scale = 1.0 / math.sqrt(hd)
    fx.add("attention", f"{n}_attn_prefill",
           {"n_q_heads": hq, "n_kv_heads": hkv, "head_dim": hd, "scale": scale,
            "pos": 0, "n_tokens": t_len, "max_ctx": t_len}, {
        "q": (write_f32, trace.q_roped),
        "k_cache": (write_f32, k_cache),
        "v_cache": (write_f32, v_cache),
        "output": (write_f32, trace.attn_out),
    })
    q_last = trace.q_roped[-1:].contiguous()
    out_last = gqa_attention(q_last, k_cache, v_cache, pos=t_len - 1, scale=scale)
    fx.add("attention", f"{n}_attn_decode",
           {"n_q_heads": hq, "n_kv_heads": hkv, "head_dim": hd, "scale": scale,
            "pos": t_len - 1, "n_tokens": 1, "max_ctx": t_len}, {
        "q": (write_f32, q_last),
        "k_cache": (write_f32, k_cache),
        "v_cache": (write_f32, v_cache),
        "output": (write_f32, out_last),
    })

    assert_no_router_neartie(trace.probs, cfg["top_k"])
    fx.add("router", f"{n}_router",
           {"n_experts": cfg["n_experts"], "top_k": cfg["top_k"],
            "norm_topk_prob": cfg["norm_topk_prob"], "dim": dim,
            "n_tokens": t_len, "w_dtype": "f32"}, {
        "input": (write_f32, trace.h2),
        "weight": (write_f32, wp(f"blk.{li}.ffn_gate_inp.weight")),
        "expert_ids": (write_i32, trace.ids),
        "gate_weights": (write_f32, trace.gate_w),
    })

    banks = {}
    for role, gname in [("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"),
                        ("down", "ffn_down_exps")]:
        w = wp(f"blk.{li}.{gname}.weight")
        blob, deq = bake_q8(w)
        banks[role] = ((raw_writer(Q8_0, tuple(w.shape)), blob), deq)
    moe_q8 = expert_mlp(trace.h2, trace.ids, trace.gate_w,
                        banks["gate"][1], banks["up"][1], banks["down"][1])
    fx.add("expert_mlp", f"{n}_experts",
           {"ffn_dim": ffn, "dim": dim, "n_experts": cfg["n_experts"],
            "top_k": cfg["top_k"], "n_tokens": t_len, "w_dtype": "q8_0"}, {
        "input": (write_f32, trace.h2),
        "gate": banks["gate"][0],
        "up": banks["up"][0],
        "down": banks["down"][0],
        "expert_ids": (write_i32, trace.ids),
        "gate_weights": (write_f32, trace.gate_w),
        "output": (write_f32, moe_q8),
    })

    fx.add("layer", f"{n}_block", {"layer": li, "pos": 0, "n_tokens": t_len}, {
        "input": (write_f32, trace.x_in),
        "positions": (write_i32, positions),
        "output": (write_f32, trace.x_out),
    })


def make_prompts(cfg: dict, seed: int) -> list[tuple[str, str | None, list[int]]]:
    """5 deterministic prompts (ADR-005). Synthetic: raw token ids, no tokenizer."""
    rng = np.random.default_rng(seed)
    v = cfg["vocab_size"]
    return [
        ("p0_random17", None, rng.integers(0, v, 17).tolist()),
        ("p1_ascending", None, list(range(1, 10))),
        ("p2_repeat", None, [7] * 12),
        ("p3_boundary", None, [0, v - 1] * 4),
        ("p4_random33", None, rng.integers(0, v, 33).tolist()),
    ]


REAL_PROMPT_TEXTS = [
    ("p0_capital", "The capital of France is"),
    ("p1_count", "1, 2, 3, 4, 5,"),
    ("p2_code", "def fibonacci(n):"),
    ("p3_json", 'Respond with JSON: {"name":'),
    ("p4_reason", "If all cats are animals and Tom is a cat, then Tom is"),
]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def run_synthetic(out_dir: Path, seed: int, n_new: int):
    cfg = SYNTH_CONFIG
    model = build_synthetic(seed)
    weights, blobs = extract_synthetic_weights(model, cfg)
    wp = weights.__getitem__

    # Semantic guard: our pure-torch recipe must match HF transformers exactly
    # (same baked weights, fp32, eager attention).
    guard_tokens = make_prompts(cfg, seed)[0][2]
    ids = torch.tensor([guard_tokens])
    hf_logits = model(input_ids=ids).logits[0]
    my_logits, _ = model_forward(guard_tokens, wp, cfg)
    diff = (hf_logits - my_logits).abs().max().item()
    assert diff < 2e-4, f"oracle recipe diverges from HF transformers: {diff}"
    print(f"[guard] recipe vs HF transformers max |Δlogit| = {diff:.2e} OK")

    fx = FixtureSet(out_dir, "synthetic-tiny-qwen3moe", cfg, seed)
    fx.manifest["model"]["gguf"] = "model.gguf"

    # Per-op cases from a traced forward of p0 at layers {0, mid, last}.
    layers = sorted({0, cfg["n_layers"] // 2, cfg["n_layers"] - 1})
    positions = torch.arange(len(guard_tokens))
    _, traces = model_forward(guard_tokens, wp, cfg, want_traces=set(layers))
    for li in layers:
        emit_layer_cases(fx, wp, cfg, li, traces[li], positions)

    # Final lm_head matmul case.
    final_h = rms_norm(traces[max(layers)].x_out, wp("output_norm.weight"), cfg["rms_eps"])
    blob = blobs["output.weight"]
    fx.add("matmul_quant", "lm_head",
           {"m": final_h.shape[0], "n": cfg["vocab_size"], "k": cfg["hidden_dim"],
            "w_dtype": "q8_0"}, {
        "input": (write_f32, final_h),
        "weight": (raw_writer(Q8_0, tuple(wp("output.weight").shape)), blob),
        "output": (write_f32, final_h @ wp("output.weight").T),
    })

    # E2E prompts: greedy continuation + full-sequence logits.
    for name, text, toks in make_prompts(cfg, seed):
        seq = greedy_decode(toks, wp, cfg, n_new)
        logits, _ = model_forward(seq, wp, cfg)
        fx.add_prompt(name, text, toks, seq[len(toks):], logits)
        print(f"[e2e] {name}: prompt {len(toks)} tokens -> greedy {seq[len(toks):]}")

    # GGUF artifact + round-trip self-test.
    gguf_path = out_dir / "model.gguf"
    write_gguf(gguf_path, cfg, weights, blobs)
    meta, back = read_gguf(gguf_path)
    assert meta["qwen3moe.block_count"] == cfg["n_layers"]
    for name, t in weights.items():
        np.testing.assert_array_equal(back[name], t.numpy(), err_msg=name)
    print(f"[gguf] {gguf_path} round-trips exactly ({len(weights)} tensors)")

    fx.save()
    total = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"[done] {len(fx.cases)} cases, {len(fx.prompts)} prompts, "
          f"{total / 1e6:.1f} MB in {out_dir}")


def run_hf_blocktrace(model_dir: Path, out_dir: Path, prompt_name: str,
                       append_n: int, layers: list[int] | None):
    """T06 follow-up (M2c router-divergence localization): trace an
    ARBITRARY prompt (by REAL_PROMPT_TEXTS name), optionally extended with
    the first `append_n` tokens of its own oracle greedy continuation, and
    dump per-layer residual-stream + router state — NOT the full per-op
    battery emit_layer_cases produces (that's for kernel-exactness gating
    and is expensive per layer on a 30B model); this is a light, fast dump
    meant for one-off layer-localization diagnostics.

    Unlike run_hf's per-op trace (hardcoded to prompts[0] == p0_capital),
    this lets any prompt be traced, and `--trace-append N` extends the
    traced sequence past the prompt into the decode steps that produced
    tokens generated[0..N-1] — needed to localize divergences that first
    appear deep into generation (e.g. p3/p4 in the T06 gate), not just in
    the prefill.

    Output: {out_dir}/blocktrace_{prompt_name}/l{i}_block_out.ds5t (T,hidden),
    l{i}_router_ids.ds5t (T,top_k) i32, l{i}_router_gate_w.ds5t (T,top_k),
    l{i}_router_probs_full.ds5t (T,n_experts) — the FULL post-softmax
    distribution, not just the top-k the router kernel contract exposes —
    plus meta.json with the traced token sequence.

    If `append_n` > 0 and {out_dir}/manifest.json already has a "prompts"
    entry for this prompt with >= append_n greedy_tokens (i.e. an --e2e run
    already happened here), those EXACT tokens are reused instead of
    recomputing greedy_decode — same deterministic model/code path, and
    avoids ~append_n redundant O(seq^2) recompute-from-scratch passes.
    """
    wp = HfProvider(model_dir)
    cfg = wp.config()
    n_layers = cfg["n_layers"]
    want = set(range(n_layers)) if not layers else set(layers)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [(name, text, tok(text)["input_ids"]) for name, text in REAL_PROMPT_TEXTS]
    match = [p for p in prompts if p[0] == prompt_name]
    if not match:
        names = ", ".join(p[0] for p in prompts)
        raise SystemExit(f"unknown --trace-prompt {prompt_name!r}; choices: {names}")
    name, text, ptoks = match[0]

    seq = list(ptoks)
    if append_n > 0:
        reused = False
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            for p in existing.get("prompts", []):
                if p["name"] == name and len(p.get("greedy_tokens", [])) >= append_n:
                    seq = list(ptoks) + list(p["greedy_tokens"][:append_n])
                    reused = True
                    print(f"[blocktrace] {name}: reusing {append_n} greedy tokens "
                          f"from existing {manifest_path}")
                    break
        if not reused:
            print(f"[blocktrace] {name}: no cached greedy_tokens found, "
                  f"recomputing {append_n} steps (slow: O(n^2) recompute)")
            seq = greedy_decode(ptoks, wp, cfg, append_n)
    print(f"[blocktrace] {name}: prompt {len(ptoks)} tokens + {append_n} appended "
          f"= {len(seq)} traced positions, {len(want)} layers")

    t0 = time.time()
    _, traces = model_forward(seq, wp, cfg, want_traces=want)

    bdir = out_dir / f"blocktrace_{name}"
    bdir.mkdir(parents=True, exist_ok=True)
    for li in sorted(want):
        tr = traces[li]
        write_f32(bdir / f"l{li}_block_out.ds5t", tr.x_out)
        write_i32(bdir / f"l{li}_router_ids.ds5t", tr.ids)
        write_f32(bdir / f"l{li}_router_gate_w.ds5t", tr.gate_w)
        write_f32(bdir / f"l{li}_router_probs_full.ds5t", tr.probs)
    (bdir / "meta.json").write_text(json.dumps({
        "prompt_name": name, "prompt_text": text, "prompt_token_ids": ptoks,
        "append_n": append_n, "seq": seq, "n_layers_traced": len(want),
        "layers": sorted(want),
    }, indent=1))
    print(f"[blocktrace] {name}: wrote {len(want)} layers to {bdir} "
          f"in {time.time() - t0:.0f}s")


def run_hf(model_dir: Path, out_dir: Path, layers: list[int], seed: int,
           n_new: int, e2e: bool):
    wp = HfProvider(model_dir)
    cfg = wp.config()
    print(f"[hf] {model_dir.name}: {cfg['n_layers']} layers, "
          f"{cfg['n_experts']} experts, hidden {cfg['hidden_dim']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [(name, text, tok(text)["input_ids"]) for name, text in REAL_PROMPT_TEXTS]

    fx = FixtureSet(out_dir, model_dir.name, cfg, seed)

    # Per-op traces: run only the embedding + layers up to max(layers), tracing
    # the requested ones. Weights load lazily and are freed after each layer.
    toks = prompts[0][2]
    positions = torch.arange(len(toks))
    x = wp("token_embd.weight")[torch.tensor(toks)]
    for li in range(max(layers) + 1):
        tr = LayerTrace() if li in layers else None
        x = layer_forward(x, wp, li, cfg, positions, tr)
        if tr is not None:
            emit_layer_cases(fx, wp, cfg, li, tr, positions)
            print(f"[ops] layer {li} cases emitted")

    if e2e:
        for name, text, ptoks in prompts:
            t0 = time.time()
            seq = greedy_decode(ptoks, wp, cfg, n_new)
            logits, _ = model_forward(seq, wp, cfg)
            fx.add_prompt(name, text, ptoks, seq[len(ptoks):], logits)
            print(f"[e2e] {name}: {seq[len(ptoks):]} "
                  f"({tok.decode(seq[len(ptoks):])!r}) in {time.time() - t0:.0f}s")

    fx.save()
    print(f"[done] {len(fx.cases)} cases, {len(fx.prompts)} prompts in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--hf", type=Path, metavar="DIR")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices for per-op cases "
                         "(default: 0,mid,last)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-new", type=int, default=8, help="greedy tokens per prompt")
    ap.add_argument("--e2e", action="store_true",
                    help="(--hf) also run streamed end-to-end logits; SLOW on 30B")
    ap.add_argument("--block-trace", action="store_true",
                    help="(--hf) lightweight per-layer residual-stream + router dump "
                         "for an arbitrary prompt/depth (see --trace-prompt/--trace-append), "
                         "instead of the full per-op fixture battery")
    ap.add_argument("--trace-prompt", default="p0_capital",
                    help="(--block-trace) REAL_PROMPT_TEXTS name to trace")
    ap.add_argument("--trace-append", type=int, default=0,
                    help="(--block-trace) extend the traced sequence with this many "
                         "tokens of the prompt's own oracle greedy continuation, to "
                         "reach a later decode step instead of just the prefill")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(args.out, args.seed, args.n_new)
    elif args.block_trace:
        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        run_hf_blocktrace(args.hf, args.out, args.trace_prompt, args.trace_append, layers)
    else:
        cfg_probe = json.loads((args.hf / "config.json").read_text())
        n_layers = cfg_probe["num_hidden_layers"]
        layers = ([int(x) for x in args.layers.split(",")] if args.layers
                  else sorted({0, n_layers // 2, n_layers - 1}))
        run_hf(args.hf, args.out, layers, args.seed, args.n_new, args.e2e)


if __name__ == "__main__":
    sys.exit(main())
