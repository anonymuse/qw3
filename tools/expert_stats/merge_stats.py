#!/usr/bin/env python3
"""merge_stats.py — merge routing telemetry and quantization-sensitivity data
into a validated expert_stats.json (WP-1, adopts amendment A5).

Inputs
------
1. Routing-telemetry dump (--telemetry): JSON produced by the M1 235B telemetry
   capture (execution plan hour-zero task 3: llama.cpp run over the
   router-calibration corpus, per-layer expert-usage distribution). Format is
   documented in docs/runbooks/expert-stats-capture.md and in
   tests/fixtures.py:make_telemetry().

2. Zero or more sensitivity files (--sensitivity, repeatable): JSON derived
   from offline llama.cpp imatrix / KLD passes, one file per candidate quant
   type. Format documented in the runbook and tests/fixtures.py:make_sensitivity().

Output
------
expert_stats.json conforming to docs/specs/schemas/expert_stats.schema.json:
one record per (layer, expert) with routing frequency, optional per-quant-type
sensitivity, and a per-field `sources` provenance block.

Refusals (exit code 2, no output written)
-----------------------------------------
- model-shape mismatch: anything other than 94 layers x 128 experts, top-8
- missing provenance: absent header fields (model/corpus ids+hashes) or
  absent source_id in any input
- model-hash mismatch between telemetry and sensitivity inputs
- unknown quant type, duplicate quant type, out-of-range indices,
  malformed counts

Frequency and sensitivity arrive from different passes; partial records
(telemetry-only, or sensitivity covering a subset of experts) are valid as
long as every present field carries provenance.

Usage:
    python tools/expert_stats/merge_stats.py \
        --telemetry telemetry.json \
        --sensitivity sens_q4km.json --sensitivity sens_iq3s.json \
        --git-commit "$(git rev-parse HEAD)" \
        --output expert_stats.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

TOOL_VERSION = "merge_stats.py/0.1.0"

# Model shape is fixed by spec section 5 (Qwen3-235B-A22B, ADR-005 constants).
NUM_LAYERS = 94
NUM_EXPERTS = 128
TOP_K = 8

# Candidate quant types tracked by the schema (ADR-002 kernel sequencing:
# Q8_0 first, Q4-class next, I-quants when 235B placement demands them).
ALLOWED_QUANT_TYPES = ("Q8_0", "Q4_K_M", "Q4_K_S", "IQ3_S", "IQ2_M")

SENSITIVITY_METRIC_FIELDS = ("kld", "kld_stderr", "ppl_ratio", "imatrix_importance")

DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs" / "specs" / "schemas" / "expert_stats.schema.json"
)


class MergeError(Exception):
    """Refusal: bad shape, missing provenance, or malformed input."""


# ---------------------------------------------------------------------------
# Input loading + validation
# ---------------------------------------------------------------------------

def _require(obj: dict, key: str, ctx: str):
    if key not in obj or obj[key] in (None, ""):
        raise MergeError(f"{ctx}: missing required field {key!r} (provenance refusal)")
    return obj[key]


def load_json(path: Path, ctx: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise MergeError(f"{ctx}: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise MergeError(f"{ctx}: not valid JSON ({exc}): {path}")
    if not isinstance(data, dict):
        raise MergeError(f"{ctx}: top level must be a JSON object: {path}")
    return data


def check_shape(obj: dict, ctx: str) -> None:
    """Refuse anything that is not the fixed 94x128 top-8 model shape."""
    layers = _require(obj, "num_layers", ctx)
    experts = _require(obj, "num_experts_per_layer", ctx)
    if layers != NUM_LAYERS or experts != NUM_EXPERTS:
        raise MergeError(
            f"{ctx}: model-shape mismatch — got {layers} layers x {experts} experts, "
            f"expected {NUM_LAYERS} x {NUM_EXPERTS} (Qwen3-235B-A22B). Refusing."
        )
    top_k = obj.get("top_k")
    if top_k is not None and top_k != TOP_K:
        raise MergeError(f"{ctx}: top_k={top_k}, expected {TOP_K}. Refusing.")


def validate_telemetry(tel: dict) -> None:
    ctx = "telemetry"
    _require(tel, "source_id", ctx)
    _require(tel, "model_id", ctx)
    _require(tel, "model_hash", ctx)
    _require(tel, "corpus_id", ctx)
    _require(tel, "corpus_hash", ctx)
    check_shape(tel, ctx)

    total_tokens = _require(tel, "total_tokens", ctx)
    if not isinstance(total_tokens, int) or total_tokens <= 0:
        raise MergeError(f"{ctx}: total_tokens must be a positive integer")

    layers = _require(tel, "layers", ctx)
    if not isinstance(layers, list) or len(layers) != NUM_LAYERS:
        got = len(layers) if isinstance(layers, list) else type(layers).__name__
        raise MergeError(
            f"{ctx}: expected exactly {NUM_LAYERS} layer records, got {got}. "
            "Model-shape refusal."
        )

    seen_layers = set()
    for entry in layers:
        if not isinstance(entry, dict):
            raise MergeError(f"{ctx}: layer entry is not an object")
        li = entry.get("layer")
        if not isinstance(li, int) or not (0 <= li < NUM_LAYERS):
            raise MergeError(f"{ctx}: layer index {li!r} out of range 0..{NUM_LAYERS - 1}")
        if li in seen_layers:
            raise MergeError(f"{ctx}: duplicate layer index {li}")
        seen_layers.add(li)

        counts = entry.get("activation_counts")
        if not isinstance(counts, list) or len(counts) != NUM_EXPERTS:
            raise MergeError(
                f"{ctx}: layer {li}: activation_counts must have exactly "
                f"{NUM_EXPERTS} entries. Model-shape refusal."
            )
        for e, c in enumerate(counts):
            if not isinstance(c, int) or c < 0:
                raise MergeError(
                    f"{ctx}: layer {li} expert {e}: activation count {c!r} "
                    "must be a non-negative integer"
                )
            if c > total_tokens:
                raise MergeError(
                    f"{ctx}: layer {li} expert {e}: activation count {c} "
                    f"exceeds total_tokens {total_tokens}"
                )

        gws = entry.get("gate_weight_sums")
        if not isinstance(gws, list) or len(gws) != NUM_EXPERTS:
            raise MergeError(
                f"{ctx}: layer {li}: gate_weight_sums must have exactly "
                f"{NUM_EXPERTS} entries. Model-shape refusal."
            )
        for e, g in enumerate(gws):
            if not isinstance(g, (int, float)):
                raise MergeError(
                    f"{ctx}: layer {li} expert {e}: gate weight sum {g!r} not numeric"
                )


def validate_sensitivity(sens: dict, telemetry: dict | None) -> None:
    ctx_base = "sensitivity"
    src = _require(sens, "source_id", ctx_base)
    ctx = f"{ctx_base}[{src}]"
    qt = _require(sens, "quant_type", ctx)
    if qt not in ALLOWED_QUANT_TYPES:
        raise MergeError(
            f"{ctx}: unknown quant_type {qt!r}; allowed: {', '.join(ALLOWED_QUANT_TYPES)}"
        )
    _require(sens, "model_id", ctx)
    _require(sens, "model_hash", ctx)
    _require(sens, "llama_cpp_commit", ctx)
    check_shape(sens, ctx)

    if telemetry is not None:
        if sens["model_hash"] != telemetry["model_hash"]:
            raise MergeError(
                f"{ctx}: model_hash {sens['model_hash']!r} does not match "
                f"telemetry model_hash {telemetry['model_hash']!r}. Refusing to "
                "merge stats captured from different artifacts."
            )

    experts = _require(sens, "experts", ctx)
    if not isinstance(experts, list) or not experts:
        raise MergeError(f"{ctx}: experts must be a non-empty list")

    seen = set()
    for rec in experts:
        if not isinstance(rec, dict):
            raise MergeError(f"{ctx}: expert record is not an object")
        li = rec.get("layer")
        ei = rec.get("expert_index")
        if not isinstance(li, int) or not (0 <= li < NUM_LAYERS):
            raise MergeError(f"{ctx}: layer index {li!r} out of range 0..{NUM_LAYERS - 1}")
        if not isinstance(ei, int) or not (0 <= ei < NUM_EXPERTS):
            raise MergeError(
                f"{ctx}: expert index {ei!r} out of range 0..{NUM_EXPERTS - 1}"
            )
        if (li, ei) in seen:
            raise MergeError(f"{ctx}: duplicate record for (layer={li}, expert={ei})")
        seen.add((li, ei))
        if not any(f in rec for f in SENSITIVITY_METRIC_FIELDS):
            raise MergeError(
                f"{ctx}: (layer={li}, expert={ei}) carries none of "
                f"{SENSITIVITY_METRIC_FIELDS}"
            )
        for f in SENSITIVITY_METRIC_FIELDS:
            if f in rec and not isinstance(rec[f], (int, float)):
                raise MergeError(
                    f"{ctx}: (layer={li}, expert={ei}) field {f}={rec[f]!r} not numeric"
                )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(telemetry: dict, sensitivities: list[dict], git_commit: str,
          capture_date: str | None = None) -> dict:
    """Build the merged expert_stats document. Inputs must be pre-validated."""
    tel_src = telemetry["source_id"]
    total_tokens = telemetry["total_tokens"]

    # index -> record, so layers may arrive out of order
    records: dict[tuple[int, int], dict] = {}
    for entry in telemetry["layers"]:
        li = entry["layer"]
        counts = entry["activation_counts"]
        gws = entry["gate_weight_sums"]
        for ei in range(NUM_EXPERTS):
            count = counts[ei]
            rec = {
                "layer": li,
                "expert_index": ei,
                "activation_count": count,
                "activation_fraction": count / total_tokens,
                "mean_gate_weight": (gws[ei] / count) if count > 0 else 0.0,
                "sources": {
                    "activation_count": tel_src,
                    "activation_fraction": tel_src,
                    "mean_gate_weight": tel_src,
                },
            }
            records[(li, ei)] = rec

    seen_quants: set[str] = set()
    sens_sources: list[str] = []
    for sens in sensitivities:
        qt = sens["quant_type"]
        if qt in seen_quants:
            raise MergeError(f"duplicate sensitivity input for quant_type {qt!r}")
        seen_quants.add(qt)
        sens_sources.append(f"{qt}:{sens['source_id']}")
        for srec in sens["experts"]:
            key = (srec["layer"], srec["expert_index"])
            rec = records[key]
            qsens = rec.setdefault("quantization_sensitivity", {})
            qsens[qt] = {
                f: srec[f] for f in SENSITIVITY_METRIC_FIELDS if f in srec
            }
            rec["sources"]["quantization_sensitivity"] = ";".join(
                s for s in sens_sources
                if s.split(":", 1)[0] in qsens
            )

    header = {
        "model_id": telemetry["model_id"],
        "model_hash": telemetry["model_hash"],
        "corpus_id": telemetry["corpus_id"],
        "corpus_hash": telemetry["corpus_hash"],
        "capture_date": capture_date
        or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "tool_versions": {
            "capture_tool": TOOL_VERSION,
        },
    }
    llama_commits = {s["llama_cpp_commit"] for s in sensitivities}
    if llama_commits:
        header["tool_versions"]["llama_cpp_commit"] = ";".join(sorted(llama_commits))

    return {
        "header": header,
        "model_shape": {
            "num_layers": NUM_LAYERS,
            "num_experts_per_layer": NUM_EXPERTS,
            "top_k": TOP_K,
        },
        "experts": [
            records[(li, ei)]
            for li in range(NUM_LAYERS)
            for ei in range(NUM_EXPERTS)
        ],
    }


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(doc: dict, schema_path: Path | None) -> None:
    """Validate the merged document. Uses `jsonschema` against the schema file
    when the package is installed; always runs the built-in structural checks
    (which encode the same constraints the schema formalizes)."""
    # Built-in structural checks — no third-party deps in the repo .venv.
    header = doc.get("header")
    if not isinstance(header, dict):
        raise MergeError("output: missing header")
    for key in ("model_id", "model_hash", "corpus_id", "corpus_hash",
                "capture_date", "git_commit"):
        if not header.get(key):
            raise MergeError(f"output: header missing {key!r}")

    shape = doc.get("model_shape", {})
    if (shape.get("num_layers"), shape.get("num_experts_per_layer"),
            shape.get("top_k")) != (NUM_LAYERS, NUM_EXPERTS, TOP_K):
        raise MergeError("output: model_shape is not 94x128 top-8")

    experts = doc.get("experts")
    if not isinstance(experts, list) or len(experts) != NUM_LAYERS * NUM_EXPERTS:
        raise MergeError(
            f"output: expected {NUM_LAYERS * NUM_EXPERTS} expert records, "
            f"got {len(experts) if isinstance(experts, list) else 'none'}"
        )
    for rec in experts:
        li, ei = rec.get("layer"), rec.get("expert_index")
        if not (isinstance(li, int) and 0 <= li < NUM_LAYERS):
            raise MergeError(f"output: bad layer index {li!r}")
        if not (isinstance(ei, int) and 0 <= ei < NUM_EXPERTS):
            raise MergeError(f"output: bad expert index {ei!r}")
        frac = rec.get("activation_fraction")
        if frac is not None and not (0.0 <= frac <= 1.0):
            raise MergeError(
                f"output: (layer={li}, expert={ei}) activation_fraction {frac} "
                "outside [0, 1]"
            )
        sources = rec.get("sources", {})
        for field in ("activation_count", "activation_fraction",
                      "mean_gate_weight", "quantization_sensitivity"):
            if field in rec and field not in sources:
                raise MergeError(
                    f"output: (layer={li}, expert={ei}) field {field!r} present "
                    "without provenance in sources"
                )
        for qt in rec.get("quantization_sensitivity", {}):
            if qt not in ALLOWED_QUANT_TYPES:
                raise MergeError(
                    f"output: (layer={li}, expert={ei}) unknown quant type {qt!r}"
                )

    # Full JSON Schema validation, when available.
    if schema_path is not None:
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as exc:  # pragma: no cover
            raise MergeError(f"output: schema validation failed: {exc.message}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="merge_stats.py",
        description=(
            "Merge M1 routing-telemetry and offline llama.cpp imatrix/KLD "
            "sensitivity data into a validated expert_stats.json "
            "(Qwen3-235B-A22B: 94 layers x 128 experts, top-8). "
            "Refuses on model-shape mismatch or missing provenance."
        ),
    )
    p.add_argument("--telemetry", type=Path, required=True,
                   help="routing-telemetry dump (JSON, M1 capture format)")
    p.add_argument("--sensitivity", type=Path, action="append", default=[],
                   metavar="FILE",
                   help="per-quant-type sensitivity JSON (repeatable, "
                        "one file per quant type)")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="path to write merged expert_stats.json")
    p.add_argument("--git-commit", default=None,
                   help="repo commit recorded in the header "
                        "(default: read from `git rev-parse HEAD`)")
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH,
                   help="expert_stats JSON Schema path (used when the "
                        "jsonschema package is installed; built-in checks "
                        "always run)")
    p.add_argument("--capture-date", default=None,
                   help="ISO 8601 capture timestamp (default: now, UTC)")
    return p


def resolve_git_commit(explicit: str | None) -> str:
    if explicit:
        return explicit
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise MergeError(
        "could not determine git commit for provenance; pass --git-commit"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        telemetry = load_json(args.telemetry, "telemetry")
        validate_telemetry(telemetry)

        sensitivities = []
        for spath in args.sensitivity:
            sens = load_json(spath, "sensitivity")
            validate_sensitivity(sens, telemetry)
            sensitivities.append(sens)

        git_commit = resolve_git_commit(args.git_commit)
        doc = merge(telemetry, sensitivities, git_commit,
                    capture_date=args.capture_date)
        schema_path = args.schema if args.schema and args.schema.exists() else None
        validate_output(doc, schema_path)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
    except MergeError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    n_sens = sum(
        1 for r in doc["experts"] if "quantization_sensitivity" in r
    )
    print(
        f"wrote {args.output}: {len(doc['experts'])} expert records "
        f"({NUM_LAYERS} layers x {NUM_EXPERTS} experts), "
        f"{n_sens} with sensitivity data "
        f"({len(sensitivities)} quant type(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
