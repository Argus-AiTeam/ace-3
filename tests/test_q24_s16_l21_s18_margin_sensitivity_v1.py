"""Focused frozen-evidence and exact-boundary checks; no native execution."""

import copy
from fractions import Fraction
import json
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l21_s18_margin_sensitivity_v1 as d


class MarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.command = json.loads((d.INPUT / "command.json").read_text())
        cls.reports = {layer: json.loads((d.INPUT / f"layer{layer}/reports.json").read_text())
                       for layer in d.LAYERS}
        with np.load(d.INPUT / "layer21/actual_stages.npz", allow_pickle=False) as data:
            cls.arrays = {key: data[key].copy() for key in data.files}
        cls.parent = d.retained.state_from(cls.arrays, "input", "input_hidden")
        cls.reference = Fraction(420904987619281, 4398046511104)
        cls.analysis = d.sensitivity(int(cls.arrays["output_i"][62]), cls.reference)

    def test_baseline_exact_gate(self):
        baseline = self.analysis["baseline"]
        self.assertFalse(baseline["accepted"])
        self.assertEqual(baseline["actual_fp16_bits"], "55ef")
        self.assertEqual(baseline["nearest_fp16_bits"], "55fb")
        self.assertEqual(baseline["excess_error"], "3/4")
        self.assertEqual(baseline["excess_budget"], "1/8")
        actual = Fraction(1519, 16)
        nearest = Fraction(1531, 16)
        floor = abs(nearest - self.reference)
        self.assertEqual(abs(actual - self.reference) - floor, Fraction(3, 4))

    def test_minimal_q24_adjustment(self):
        cut = self.analysis["minimal_residual_cut"]
        self.assertEqual(cut["baseline_Q24_integer"], 1592645376)
        self.assertEqual(cut["adjusted_Q24_integer"], 1602748417)
        self.assertEqual(cut["delta_Q24_units"], 10103041)
        self.assertEqual(cut["delta"], "10103041/16777216")
        self.assertEqual(cut["output_word"], "55f9")
        self.assertEqual(cut["scalar_gate"]["excess_error"], "1/8")
        self.assertTrue(cut["scalar_gate"]["accepted"])

    def test_one_unit_less_fails(self):
        cut = self.analysis["minimal_residual_cut"]
        self.assertEqual(cut["one_unit_less_Q24_integer"], 1602748416)
        self.assertEqual(cut["one_unit_less_delta_Q24_units"], 10103040)
        self.assertEqual(cut["one_unit_less_word"], "55f8")
        self.assertEqual(cut["one_unit_less_gate"]["excess_error"], "3/16")
        self.assertFalse(cut["one_unit_less_gate"]["accepted"])

    def test_odd_midpoint_ownership(self):
        cell = self.analysis["passing_FP16_interval"]
        self.assertEqual(cell["RNE_lower_midpoint"], "3057/32")
        self.assertFalse(cell["lower_inclusive"])
        self.assertFalse(cell["upper_inclusive"])
        midpoint = Fraction(3057, 32)
        self.assertEqual(midpoint * d.Q, 1602748416)
        # Independent NumPy RNE conversion confirms the excluded tie.
        words = np.array([float(midpoint), float(midpoint + Fraction(1, d.Q))],
                         dtype=np.float64).astype(np.float16).view(np.uint16)
        self.assertEqual(words.tolist(), [0x55F8, 0x55F9])

    def test_passing_interval_is_exhaustive(self):
        cell = self.analysis["passing_FP16_interval"]
        self.assertEqual((cell["first_word"], cell["last_word"]), ("55f9", "55fd"))
        floor = abs(Fraction(1531, 16) - self.reference)
        for word in range(0x55EF, 0x5600):
            value = Fraction.from_float(float(np.array([word], dtype=np.uint16)
                                             .view(np.float16)[0]))
            expected = abs(value - self.reference) - floor <= Fraction(1, 8)
            self.assertEqual(expected, 0x55F9 <= word <= 0x55FD)

    def test_full_vector_boundary_preserves_other_coordinates(self):
        item = self.document["layers"][-1]["original_reference"]
        reference = np.load(item["binary64"]["path"], allow_pickle=False)
        with np.load(item["fp16"]["path"], allow_pickle=False) as data:
            trajectory = {key: data[key].copy() for key in data.files}
        before = self.arrays["stage18"].copy()
        result = d.vector_boundary(self.arrays, trajectory, reference, self.analysis)
        self.assertEqual(result["minimum"]["failure_count"], 0)
        self.assertEqual(result["minimum"]["coordinates"], 896)
        self.assertEqual(result["one_unit_less"]["failed_indices"], [62])
        self.assertTrue(np.array_equal(self.arrays["stage18"], before))

    def test_invalid_scalar_inputs_rejected(self):
        for integer, reference in ((True, self.reference), (1592645376, Fraction(1, 3)),
                                   (1592645376, Fraction(0)), (1592645376, Fraction(65504))):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                d.sensitivity(integer, reference)

    def test_passing_baseline_rejected(self):
        with self.assertRaisesRegex(ValueError, "failing interior"):
            d.sensitivity(1602748417, self.reference)

    def test_frozen_scope_and_failure(self):
        d.check_upstream(self.document, self.validation, self.command)
        self.assertEqual(d.select_failure(self.document["layers"], self.reports),
                         self.document["first_failure"])

    def test_wrong_native_scope_rejected(self):
        for key, value in (("audit", {"native_layers": [8, 16, 17, 18, 19, 20, 21]}),
                           ("native_layer_invocations", 5), ("first_native_layer", 8)):
            document = copy.deepcopy(self.document)
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_upstream(document, self.validation, self.command)

    def test_admission_and_rtl_flags_rejected(self):
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "rtl_invocations", "native_L0_L8_invocations"):
            document = copy.deepcopy(self.document)
            document[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_upstream(document, self.validation, self.command)

    def test_frozen_validation_counts_rejected(self):
        for key, value in (("collected", 0), ("executed", 19), ("failures", 1),
                           ("errors", 1), ("skipped", 1), ("legacy_19_tests_executed", True)):
            validation = copy.deepcopy(self.validation)
            validation[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_upstream(self.document, validation, self.command)

    def test_frozen_context_rejected(self):
        for key in ("cwd", "executable", "PYTHONPATH"):
            command = copy.deepcopy(self.command)
            command[key] = "/different-checkout"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_upstream(self.document, self.validation, command)

    def test_earlier_failure_rejected(self):
        reports = copy.deepcopy(self.reports)
        reports[20][0]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "earlier mandatory"):
            d.select_failure(self.document["layers"], reports)

    def test_changed_coordinate_gate_and_kv_rejected(self):
        for key, value in (("index", 63), ("excess_error", "1/8"), ("excess_budget", "3/4")):
            reports = copy.deepcopy(self.reports)
            reports[21][-1]["binary64_v1"]["failures"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.select_failure(self.document["layers"], reports)
        entries = copy.deepcopy(self.document["layers"])
        entries[0]["prior_layer_kv_consumed"] = True
        with self.assertRaisesRegex(ValueError, "KV boundary"):
            d.select_failure(entries, self.reports)

    def test_reference_substitution_and_closure_mismatch_rejected(self):
        entry = self.document["layers"][-1]
        changed = copy.deepcopy(entry["original_reference"])
        changed["input_binary64"] = changed["binary64"]
        with self.assertRaisesRegex(ValueError, "reference substitution"):
            d.check_reference(entry, changed)
        inputs = d.margin.prior.BoundInputs()
        record = d.retained.record(d.INPUT / "result.json")
        inputs.bind(record)
        d.margin.bind_closure(inputs, {**record, "annotation": "not a binding field"})
        for key, value in (("sha256", "0" * 64), ("bytes", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.margin.bind_closure(inputs, {**record, key: value})

    def test_state_and_operand_changes_rejected(self):
        d.margin.check_layer(self.arrays, self.parent)
        for key in ("input_i", "output_i", "stage16", "output_cache_k"):
            arrays = {name: value.copy() for name, value in self.arrays.items()}
            arrays[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.margin.check_layer(arrays, self.parent)

    def test_origin_and_contract_changes_rejected(self):
        with patch.object(d, "__file__", "/different-checkout/diagnostic.py"):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        for key, value in (("excess_budget", "3/4"), ("policy_id", "replacement"),
                           ("candidate_admitted", True), ("native_invocations", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**contract, key: value})

    def test_no_execution_guard(self):
        with d.no_execution():
            for function in (d.upstream.native.candidate._stages, d.retained.save,
                             subprocess.Popen):
                with self.subTest(function=function), self.assertRaisesRegex(
                        RuntimeError, "forbidden"):
                    function()

    def test_output_preservation_and_scope(self):
        self.assertEqual(d.output_path(d.OUTPUT / "unused_focused_test_path"),
                         d.OUTPUT / "unused_focused_test_path")
        for path in (d.INPUT, d.OUTPUT / "nested" / "outside", d.ROOT / "elsewhere"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)
        with patch.object(d.Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "preserve historical"):
                d.output_path(d.OUTPUT / "existing")


if __name__ == "__main__":
    unittest.main()
