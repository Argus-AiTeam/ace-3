"""Task-native scalar and fail-closed frozen-evidence tests; no native layers."""

import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l15_s18_margin_sensitivity_v1 as d


class MarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.reports = {
            layer: json.loads((d.INPUT / f"layer{layer}/reports.json").read_text())
            for layer in (14, 15)}
        with np.load(d.INPUT / "layer15/actual_stages.npz", allow_pickle=False) as data:
            cls.arrays = {key: data[key].copy() for key in data.files}
        cls.parent = d.retained.state_from(cls.arrays, "input", "input_hidden")
        cls.reference = np.full(896, float.fromhex("0x1.8c99188229fdcp+10"))
        # Deliberately synthetic parent reference: isolates the scalar algebra
        # from the diagnostic's separately authenticated original L14 reference.
        cls.parent_ref = np.full(896, 1584.5)
        cls.exact = d.decomposition(cls.arrays, cls.parent, cls.parent_ref,
                                   cls.reference, cls.arrays["stage17"])
        cls.analysis = d.sensitivity(cls.exact)

    def test_exact_gate(self):
        baseline = self.analysis["baseline"]
        self.assertFalse(baseline["accepted"])
        self.assertEqual(baseline["actual_fp16_bits"], "6633")
        self.assertEqual(baseline["nearest_fp16_bits"], "6632")
        self.assertEqual(baseline["excess_error"], "118614349833/549755813888")
        self.assertEqual(baseline["excess_budget"], "1/8")
        self.assertTrue(self.analysis["nearest_pass"]["accepted"])

    def test_even_cell_ties(self):
        low, high, minimum, maximum = d.margin.passing_cell(0x6632)
        self.assertEqual((low, high), (Fraction(3171, 2), Fraction(3173, 2)))
        self.assertEqual(d.rational.project(minimum, 0), 0x6632)
        self.assertEqual(d.rational.project(maximum, 0), 0x6632)
        self.assertEqual(d.rational.project(minimum - 1, 0), 0x6631)
        self.assertEqual(d.rational.project(maximum + 1, 0), 0x6633)

    def test_minimal_q24_cut(self):
        cut = self.analysis["minimal_residual_cut"]
        self.assertEqual(cut["baseline_Q24_integer"], 26617418751)
        self.assertEqual(cut["delta_Q24_units"], -365567)
        self.assertEqual(cut["delta"], "-365567/16777216")
        self.assertFalse(cut["continuous_cut_is_strict"])
        self.assertEqual(cut["one_Q24_unit_less_reduction_word"], "6633")

    def test_decomposition(self):
        v = {key: Fraction(value) for key, value in self.exact.items()}
        self.assertEqual(v["parent_I_over_2p24"], Fraction(26586391551, d.Q))
        self.assertEqual(v["scratch_I_over_2p24"], Fraction(26590712831, d.Q))
        self.assertEqual(v["S11"], Fraction(4321280, d.Q))
        self.assertEqual(v["S17_actual"], Fraction(26705920, d.Q))
        self.assertEqual(v["S18_projection_error"], Fraction(8023041, d.Q))
        self.assertEqual(v["L14_signed_Q24_drift"] + v["L15_net_increment_error"]
                         + v["S18_projection_error"], v["signed_global_error"])

    def test_component_sensitivity(self):
        rows = self.analysis["component_sensitivities"]
        for component in ("L14_drift", "L15_net_increment_error"):
            self.assertEqual(rows[component + "_single_component_boundary"]
                             ["required_delta_Q24_units"], -365567)
        row = rows["canonical_S17_on_retained_S16"]
        self.assertEqual(row["correction"], "0")
        self.assertFalse(row["scalar_gate"]["accepted"])
        self.assertTrue(rows["S18_projection"]["RNE_correct_for_frozen_Q24_input"])
        for row in (rows["remove_L14_drift"], rows["remove_L15_net_increment_error"]):
            if not row["on_Q24_grid"]:
                self.assertNotIn("scalar_gate", row)

    def test_fp16_component_cut_minimality(self):
        for name in ("S11", "S17"):
            cut = self.analysis[name + "_FP16_cut"]
            self.assertTrue(cut["scalar_accepted"])
            self.assertEqual(cut["output_word"], "6632")
            word = int(cut["word"], 16)
            other = (Fraction(self.exact["scratch_I_over_2p24"]) if name == "S17"
                     else Fraction(self.exact["parent_I_over_2p24"])
                     + Fraction(self.exact["S17_actual"]))
            closer = d.rational.fp16_value(word + 1)
            self.assertEqual(d.rational.project(d.margin.q24_units(other + closer), 0), 0x6633)

    def test_bad_decomposition_rejected(self):
        exact = dict(self.exact, L14_signed_Q24_drift="0")
        with self.assertRaisesRegex(ValueError, "identity"):
            d.sensitivity(exact)

    def test_non_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "Q24"):
            d.margin.q24_units(Fraction(1, 3))

    def test_first_failure(self):
        self.assertEqual(d.select_failure(self.document["layers"], self.reports),
                         self.document["first_failure"])

    def test_earlier_failure_rejected(self):
        reports = copy.deepcopy(self.reports)
        reports[14][17]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "earlier"):
            d.select_failure(self.document["layers"], reports)

    def test_changed_mandatory_selection_rejected(self):
        reports = copy.deepcopy(self.reports)
        reports[15][18]["binary64_v1"]["failures"] = []
        with self.assertRaisesRegex(ValueError, "mandatory"):
            d.select_failure(self.document["layers"], reports)

    def test_upstream_boundaries(self):
        d.check_upstream(self.document, self.validation)
        for key in d.upstream.FLAGS:
            changed = dict(self.document)
            changed[key] = 1 if type(changed[key]) is int else True
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_upstream(changed, self.validation)
        with self.assertRaisesRegex(ValueError, "validation"):
            d.check_upstream(self.document, dict(self.validation, executed=0))

    def test_external_binding_rejected_before_read(self):
        with patch.object(d.retained, "record") as record:
            with self.assertRaisesRegex(ValueError, "outside"):
                d.prior.BoundInputs().bind({"path": "/outside/input", "bytes": 1, "sha256": "x"})
            record.assert_not_called()
        inputs = d.prior.BoundInputs()
        signature = d.retained.record(d.SOURCE)
        inputs.bind(signature)
        annotated = dict(signature, records=896, semantic_sha256="annotation-bound-by-parent")
        self.assertEqual(d.bind_closure(inputs, annotated), d.SOURCE)
        with self.assertRaisesRegex(ValueError, "closure"):
            d.bind_closure(inputs, dict(annotated, sha256="changed"))
        with self.assertRaisesRegex(ValueError, "closure"):
            d.bind_closure(inputs, dict(annotated, bytes=signature["bytes"] + 1))

    def test_changed_digest_rejected(self):
        for name, digest in d.PINS.items():
            with self.subTest(name=name), patch.object(
                    d.retained, "record", return_value={"sha256": "wrong"}):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    d.margin.pinned(d.prior.BoundInputs(), d.INPUT / name, digest)

    def test_state_and_kv_mutations_rejected(self):
        d.check_layer(self.arrays, self.parent)
        for key in ("input_i", "input_z", "input_hidden", "scratch_i", "output_i",
                    "output_cache_k", "stage16"):
            changed = {name: value.copy() for name, value in self.arrays.items()}
            changed[key].flat[0] ^= 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_layer(changed, self.parent)
        changed = dict(self.arrays, input_cache_v=np.zeros((1, 128), dtype="<u2"))
        with self.assertRaises(ValueError):
            d.check_layer(changed, self.parent)

    def test_reference_reanchoring_rejected(self):
        item = copy.deepcopy(self.document["layers"][1]["original_reference"])
        extension = {"reference_only": False}
        with self.assertRaises(ValueError):
            d.upstream.check_reference_suffix(extension)
        previous = self.document["layers"][0]["original_reference"]
        self.assertEqual(item["input_binary64"], previous["binary64"])
        self.assertEqual(item["input_fp16"], previous["fp16"])

    def test_origins_and_contract(self):
        origins = d.origins()
        self.assertEqual(origins[d.MODULE]["path"], str(d.SOURCE))
        self.assertEqual(origins[d.TEST_MODULE]["path"],
                         str(d.ROOT / "tests" / (d.TEST_MODULE.split(".")[-1] + ".py")))
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["upstream_sha256"], d.PINS)
        self.assertEqual(contract["focused_tests"], d.EXPECTED_TESTS)
        for key, value in d.FLAGS.items():
            self.assertEqual(contract[key], value)
        with patch.object(d, "__file__", "/outside/source.py"):
            with self.assertRaisesRegex(ValueError, "origin"):
                d.origins()

    def test_no_execution_or_admission(self):
        with d.no_execution():
            for function in (d.upstream.native.candidate._stages, d.retained.save,
                             d.margin.subprocess.Popen):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function()
        self.assertEqual(self.analysis["classification"], "inconclusive_unique_cause")
        self.assertFalse(d.FLAGS["candidate_admitted"])
        self.assertFalse(d.FLAGS["successor_published"])


if __name__ == "__main__":
    unittest.main()
