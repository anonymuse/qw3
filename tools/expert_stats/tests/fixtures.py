"""Synthetic fixture builders for merge_stats.py tests.

The model shape is fixed (94 layers x 128 experts, top-8), so fixtures are
generated programmatically rather than checked in as ~1.5MB JSON blobs. A
deterministic seed keeps runs reproducible. These fixtures mimic the M1
routing-telemetry capture format and the per-quant-type sensitivity format
documented in docs/runbooks/expert-stats-capture.md.
"""

from __future__ import annotations

import random

NUM_LAYERS = 94
NUM_EXPERTS = 128
TOP_K = 8

MODEL_ID = "Qwen3-235B-A22B-Instruct-2507"
MODEL_HASH = "sha256:" + "ab" * 32
CORPUS_ID = "router-calibration-corpus-v1"
CORPUS_HASH = "sha256:" + "cd" * 32


def make_telemetry(total_tokens: int = 10_000, seed: int = 1234,
                   num_layers: int = NUM_LAYERS,
                   num_experts: int = NUM_EXPERTS) -> dict:
    """Synthetic routing-telemetry dump in the M1 capture format.

    Each token activates top-8 experts per layer; counts are drawn skewed
    (a few hot experts, a long cold tail) to resemble real routing skew.
    """
    rng = random.Random(seed)
    layers = []
    for li in range(num_layers):
        # Skewed distribution: expert weight ~ 1/(rank+1), shuffled per layer.
        weights = [1.0 / (r + 1) for r in range(num_experts)]
        rng.shuffle(weights)
        wsum = sum(weights)
        counts = []
        gate_sums = []
        for w in weights:
            # Expected activations: total_tokens * top_k * (w / wsum), clamped
            # to total_tokens (an expert fires at most once per token).
            expected = total_tokens * TOP_K * (w / wsum)
            count = min(int(expected), total_tokens)
            counts.append(count)
            # Mean gate weight in (0, 1); sum scales with count.
            gate_sums.append(count * rng.uniform(0.05, 0.35))
        layers.append({
            "layer": li,
            "activation_counts": counts,
            "gate_weight_sums": gate_sums,
        })
    return {
        "source_id": f"telemetry-synth-{seed}",
        "model_id": MODEL_ID,
        "model_hash": MODEL_HASH,
        "corpus_id": CORPUS_ID,
        "corpus_hash": CORPUS_HASH,
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts,
        "top_k": TOP_K,
        "total_tokens": total_tokens,
        "layers": layers,
    }


def make_sensitivity(quant_type: str = "Q4_K_M", seed: int = 42,
                     coverage: float = 1.0,
                     model_hash: str = MODEL_HASH,
                     num_layers: int = NUM_LAYERS,
                     num_experts: int = NUM_EXPERTS) -> dict:
    """Synthetic per-quant-type sensitivity file (imatrix/KLD-derived).

    coverage < 1.0 produces a partial pass (a subset of experts), which the
    schema explicitly allows.
    """
    rng = random.Random(seed)
    experts = []
    for li in range(num_layers):
        for ei in range(num_experts):
            if rng.random() > coverage:
                continue
            experts.append({
                "layer": li,
                "expert_index": ei,
                "kld": rng.uniform(1e-5, 0.2),
                "kld_stderr": rng.uniform(1e-7, 1e-3),
                "ppl_ratio": rng.uniform(1.0, 1.15),
                "imatrix_importance": rng.uniform(0.0, 50.0),
            })
    return {
        "source_id": f"kld-{quant_type}-synth-{seed}",
        "quant_type": quant_type,
        "model_id": MODEL_ID,
        "model_hash": model_hash,
        "llama_cpp_commit": "0000000000000000000000000000000000000000",
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts,
        "top_k": TOP_K,
        "experts": experts,
    }
