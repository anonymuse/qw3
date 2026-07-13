"""Unit tests for merge_stats.py (stdlib unittest — the repo .venv carries no
pytest). Run from the repo root:

    source .venv/bin/activate
    python -m unittest discover tools/expert_stats/tests -v
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TOOL_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TESTS_DIR))

import fixtures  # noqa: E402
import merge_stats as ms  # noqa: E402

GIT_COMMIT = "f" * 40


def run_merge(telemetry, sensitivities=()):
    """Validate inputs and merge, as main() does, returning the document."""
    ms.validate_telemetry(telemetry)
    for s in sensitivities:
        ms.validate_sensitivity(s, telemetry)
    doc = ms.merge(telemetry, list(sensitivities), GIT_COMMIT)
    ms.validate_output(doc, None)
    return doc


class TestTelemetryOnlyMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telemetry = fixtures.make_telemetry()
        cls.doc = run_merge(cls.telemetry)

    def test_full_shape(self):
        self.assertEqual(len(self.doc["experts"]), 94 * 128)
        shape = self.doc["model_shape"]
        self.assertEqual(
            (shape["num_layers"], shape["num_experts_per_layer"], shape["top_k"]),
            (94, 128, 8),
        )

    def test_header_provenance(self):
        h = self.doc["header"]
        self.assertEqual(h["model_id"], fixtures.MODEL_ID)
        self.assertEqual(h["model_hash"], fixtures.MODEL_HASH)
        self.assertEqual(h["corpus_id"], fixtures.CORPUS_ID)
        self.assertEqual(h["corpus_hash"], fixtures.CORPUS_HASH)
        self.assertEqual(h["git_commit"], GIT_COMMIT)
        self.assertTrue(h["capture_date"])
        self.assertIn("capture_tool", h["tool_versions"])

    def test_records_ordered_and_indexed(self):
        for i, rec in enumerate(self.doc["experts"]):
            self.assertEqual(rec["layer"], i // 128)
            self.assertEqual(rec["expert_index"], i % 128)

    def test_frequency_fields(self):
        total = self.telemetry["total_tokens"]
        for rec in self.doc["experts"][:256]:
            li, ei = rec["layer"], rec["expert_index"]
            count = self.telemetry["layers"][li]["activation_counts"][ei]
            self.assertEqual(rec["activation_count"], count)
            self.assertAlmostEqual(rec["activation_fraction"], count / total)
            if count > 0:
                gsum = self.telemetry["layers"][li]["gate_weight_sums"][ei]
                self.assertAlmostEqual(rec["mean_gate_weight"], gsum / count)

    def test_partial_records_allowed_without_sensitivity(self):
        # Telemetry-only merge is a valid partial record set.
        self.assertTrue(
            all("quantization_sensitivity" not in r for r in self.doc["experts"])
        )

    def test_per_field_provenance(self):
        src = self.telemetry["source_id"]
        for rec in self.doc["experts"][:256]:
            for field in ("activation_count", "activation_fraction",
                          "mean_gate_weight"):
                self.assertEqual(rec["sources"][field], src)


class TestSensitivityMerge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telemetry = fixtures.make_telemetry()
        cls.s_q4 = fixtures.make_sensitivity("Q4_K_M", seed=7)
        cls.s_iq3 = fixtures.make_sensitivity("IQ3_S", seed=8, coverage=0.25)
        cls.doc = run_merge(cls.telemetry, [cls.s_q4, cls.s_iq3])

    def test_sensitivity_fields_merged(self):
        rec = self.doc["experts"][0]
        q4 = rec["quantization_sensitivity"]["Q4_K_M"]
        src = self.s_q4["experts"][0]
        self.assertEqual((src["layer"], src["expert_index"]), (0, 0))
        for f in ("kld", "kld_stderr", "ppl_ratio", "imatrix_importance"):
            self.assertAlmostEqual(q4[f], src[f])

    def test_partial_coverage_second_pass(self):
        n_iq3 = sum(
            1 for r in self.doc["experts"]
            if "IQ3_S" in r.get("quantization_sensitivity", {})
        )
        self.assertEqual(n_iq3, len(self.s_iq3["experts"]))
        self.assertLess(n_iq3, 94 * 128)  # genuinely partial

    def test_sensitivity_provenance(self):
        for rec in self.doc["experts"]:
            qsens = rec.get("quantization_sensitivity")
            if not qsens:
                continue
            src = rec["sources"]["quantization_sensitivity"]
            for qt in qsens:
                self.assertIn(f"{qt}:", src)
            # No provenance claimed for quant types absent from the record.
            if "IQ3_S" not in qsens:
                self.assertNotIn("IQ3_S:", src)

    def test_llama_cpp_commit_in_header(self):
        self.assertIn("llama_cpp_commit", self.doc["header"]["tool_versions"])


class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.telemetry = fixtures.make_telemetry()

    def test_wrong_layer_count(self):
        bad = fixtures.make_telemetry(num_layers=93)
        with self.assertRaisesRegex(ms.MergeError, "shape"):
            ms.validate_telemetry(bad)

    def test_wrong_expert_count(self):
        bad = fixtures.make_telemetry(num_experts=127)
        with self.assertRaisesRegex(ms.MergeError, "shape"):
            ms.validate_telemetry(bad)

    def test_declared_shape_but_missing_layer_records(self):
        bad = copy.deepcopy(self.telemetry)
        bad["layers"] = bad["layers"][:-1]  # claims 94, delivers 93
        with self.assertRaisesRegex(ms.MergeError, "94 layer records"):
            ms.validate_telemetry(bad)

    def test_wrong_top_k(self):
        bad = copy.deepcopy(self.telemetry)
        bad["top_k"] = 2
        with self.assertRaisesRegex(ms.MergeError, "top_k"):
            ms.validate_telemetry(bad)

    def test_missing_provenance_model_hash(self):
        bad = copy.deepcopy(self.telemetry)
        del bad["model_hash"]
        with self.assertRaisesRegex(ms.MergeError, "model_hash"):
            ms.validate_telemetry(bad)

    def test_missing_provenance_source_id(self):
        bad = copy.deepcopy(self.telemetry)
        bad["source_id"] = ""
        with self.assertRaisesRegex(ms.MergeError, "source_id"):
            ms.validate_telemetry(bad)

    def test_missing_corpus_hash(self):
        bad = copy.deepcopy(self.telemetry)
        del bad["corpus_hash"]
        with self.assertRaisesRegex(ms.MergeError, "corpus_hash"):
            ms.validate_telemetry(bad)

    def test_negative_activation_count(self):
        bad = copy.deepcopy(self.telemetry)
        bad["layers"][3]["activation_counts"][5] = -1
        with self.assertRaisesRegex(ms.MergeError, "non-negative"):
            ms.validate_telemetry(bad)

    def test_count_exceeds_total_tokens(self):
        bad = copy.deepcopy(self.telemetry)
        bad["layers"][0]["activation_counts"][0] = bad["total_tokens"] + 1
        with self.assertRaisesRegex(ms.MergeError, "exceeds total_tokens"):
            ms.validate_telemetry(bad)

    def test_sensitivity_model_hash_mismatch(self):
        sens = fixtures.make_sensitivity(model_hash="sha256:" + "ee" * 32)
        with self.assertRaisesRegex(ms.MergeError, "does not match"):
            ms.validate_sensitivity(sens, self.telemetry)

    def test_sensitivity_unknown_quant_type(self):
        sens = fixtures.make_sensitivity()
        sens["quant_type"] = "Q9_9"
        with self.assertRaisesRegex(ms.MergeError, "unknown quant_type"):
            ms.validate_sensitivity(sens, self.telemetry)

    def test_sensitivity_missing_llama_commit(self):
        sens = fixtures.make_sensitivity()
        del sens["llama_cpp_commit"]
        with self.assertRaisesRegex(ms.MergeError, "llama_cpp_commit"):
            ms.validate_sensitivity(sens, self.telemetry)

    def test_sensitivity_out_of_range_expert(self):
        sens = fixtures.make_sensitivity(coverage=0.01)
        sens["experts"][0]["expert_index"] = 128
        with self.assertRaisesRegex(ms.MergeError, "out of range"):
            ms.validate_sensitivity(sens, self.telemetry)

    def test_sensitivity_out_of_range_layer(self):
        sens = fixtures.make_sensitivity(coverage=0.01)
        sens["experts"][0]["layer"] = 94
        with self.assertRaisesRegex(ms.MergeError, "out of range"):
            ms.validate_sensitivity(sens, self.telemetry)

    def test_duplicate_quant_type_refused(self):
        s1 = fixtures.make_sensitivity("Q4_K_M", seed=1, coverage=0.01)
        s2 = fixtures.make_sensitivity("Q4_K_M", seed=2, coverage=0.01)
        ms.validate_telemetry(self.telemetry)
        with self.assertRaisesRegex(ms.MergeError, "duplicate sensitivity"):
            ms.merge(self.telemetry, [s1, s2], GIT_COMMIT)


class TestCLI(unittest.TestCase):
    """End-to-end runs of the CLI in a subprocess, as the acceptance
    criteria demand (`--help` works; refusal exits nonzero)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.tel_path = tmp / "telemetry.json"
        cls.sens_path = tmp / "sens_q4km.json"
        cls.out_path = tmp / "expert_stats.json"
        cls.tel_path.write_text(json.dumps(fixtures.make_telemetry()))
        cls.sens_path.write_text(json.dumps(fixtures.make_sensitivity("Q4_K_M")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(TOOL_DIR / "merge_stats.py"), *args],
            capture_output=True, text=True,
        )

    def test_help(self):
        res = self._run("--help")
        self.assertEqual(res.returncode, 0)
        self.assertIn("expert_stats", res.stdout)
        self.assertIn("--telemetry", res.stdout)

    def test_end_to_end_merge(self):
        res = self._run(
            "--telemetry", str(self.tel_path),
            "--sensitivity", str(self.sens_path),
            "--git-commit", GIT_COMMIT,
            "--output", str(self.out_path),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        doc = json.loads(self.out_path.read_text())
        self.assertEqual(len(doc["experts"]), 94 * 128)
        ms.validate_output(doc, None)  # merged output passes validation

    def test_cli_refuses_bad_shape(self):
        bad_path = Path(self.tmp.name) / "bad_telemetry.json"
        bad_path.write_text(json.dumps(fixtures.make_telemetry(num_layers=90)))
        out = Path(self.tmp.name) / "should_not_exist.json"
        res = self._run(
            "--telemetry", str(bad_path),
            "--git-commit", GIT_COMMIT,
            "--output", str(out),
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("refused", res.stderr)
        self.assertFalse(out.exists())  # no output on refusal


class TestSchemaFile(unittest.TestCase):
    """The schema document itself must exist, parse, and pin the model shape."""

    def setUp(self):
        self.schema_path = (
            TOOL_DIR.parents[1] / "docs" / "specs" / "schemas"
            / "expert_stats.schema.json"
        )

    def test_schema_parses_and_pins_shape(self):
        schema = json.loads(self.schema_path.read_text())
        shape = schema["properties"]["model_shape"]["properties"]
        self.assertEqual(shape["num_layers"]["enum"], [94])
        self.assertEqual(shape["num_experts_per_layer"]["enum"], [128])
        self.assertEqual(shape["top_k"]["enum"], [8])

    def test_merged_output_validates_against_schema_if_jsonschema_present(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema not installed in .venv; "
                          "built-in checks covered elsewhere")
        doc = run_merge(fixtures.make_telemetry(),
                        [fixtures.make_sensitivity("Q4_K_M")])
        ms.validate_output(doc, self.schema_path)


if __name__ == "__main__":
    unittest.main()
