"""Exact scalar and fail-closed authentication tests; no native execution."""

import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l13_s18_margin_sensitivity_v1 as d


def exact_fixture():
    return {
        "L12_signed_Q24_drift": "530376431897/2199023255552",
        "L13_net_increment_error": "-363807407/2199023255552",
        "S11": "-1331/8192", "S17_actual": "1133/2048", "S17_local": "1133/2048",
        "S18_projection_error": "101905/262144",
        "original_L12_reference": "3481008357369575/2199023255552",
        "original_L13_reference": "1740933991535819/1099511627776",
        "output_I_over_2p24": "415134191/262144",
        "parent_I_over_2p24": "415031759/262144",
        "scratch_I_over_2p24": "414989167/262144",
        "signed_global_error": "692426861365/1099511627776",
    }


class MarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = d.sensitivity(exact_fixture())

    def test_exact_gate(self):
        baseline = self.analysis["baseline"]
        self.assertFalse(baseline["accepted"])
        self.assertEqual(baseline["excess_over_budget"], "73951570741/549755813888")
        self.assertTrue(self.analysis["nearest_pass"]["accepted"])
        self.assertEqual(baseline["excess_budget"], "1/8")

    def test_odd_midpoint_ties(self):
        lower, upper, lo, hi = d.passing_cell(0x662F)
        self.assertEqual((lower, upper), (Fraction(3165, 2), Fraction(3167, 2)))
        self.assertEqual(d.rational.project(lo - 1, 0), 0x662E)
        self.assertEqual(d.rational.project(lo, 0), 0x662F)
        self.assertEqual(d.rational.project(hi, 0), 0x662F)
        self.assertEqual(d.rational.project(hi + 1, 0), 0x6630)

    def test_even_midpoint_ties(self):
        lower, upper, lo, hi = d.passing_cell(0x6630)
        self.assertEqual(Fraction(lo, d.Q), lower)
        self.assertEqual(Fraction(hi, d.Q), upper)
        self.assertEqual(d.rational.project(lo, 0), 0x6630)
        self.assertEqual(d.rational.project(hi, 0), 0x6630)

    def test_minimal_q24_cut(self):
        cut = self.analysis["minimal_residual_cut"]
        self.assertEqual(cut["continuous_distance_to_upper_midpoint"], "29167/262144")
        self.assertEqual(cut["delta_Q24_units"], -1866689)
        self.assertEqual(cut["delta"], "-1866689/16777216")
        self.assertEqual(cut["one_Q24_unit_less_reduction_word"], "6630")

    def test_parent_versus_increment(self):
        rows = self.analysis["component_sensitivities"]
        self.assertTrue(rows["remove_L12_drift"]["inside_passing_RNE_cell"])
        self.assertFalse(rows["remove_L13_net_increment_error"]["inside_passing_RNE_cell"])
        self.assertTrue(rows["remove_both"]["inside_passing_RNE_cell"])
        self.assertFalse(rows["remove_L12_drift"]["on_Q24_grid"])
        self.assertNotIn("scalar_gate", rows["remove_L12_drift"])

    def test_s17_same_input_agreement(self):
        row = self.analysis["component_sensitivities"]["canonical_S17_on_retained_S16"]
        self.assertEqual(row["correction"], "0")
        self.assertEqual(row["output_word"], "6630")
        self.assertFalse(row["scalar_gate"]["accepted"])

    def test_minimum_fp16_component_cuts(self):
        exact = {k: Fraction(v) for k, v in exact_fixture().items()}
        for label, component, other in (
            ("S17", exact["S17_actual"], exact["scratch_I_over_2p24"]),
            ("S11", exact["S11"], exact["parent_I_over_2p24"] + exact["S17_actual"]),
        ):
            with self.subTest(component=label):
                cut = self.analysis[label + "_FP16_cut"]
                word = int(cut["word"], 16)
                value = d.rational.fp16_value(word)
                self.assertEqual(Fraction(cut["delta"]), value - component)
                self.assertEqual(cut["output_word"], "662f")
                # Closest next representable value toward the unchanged operand fails.
                closer = word + 1 if value > 0 else word - 1
                closer_output = other + d.rational.fp16_value(closer)
                self.assertEqual(d.rational.project(d.q24_units(closer_output), 0), 0x6630)

    def test_reject_non_grid(self):
        with self.assertRaisesRegex(ValueError, "Q24"):
            d.q24_units(Fraction(1, 3))

    def test_reject_bad_cell(self):
        for word in (0, -1, 0x7C00, 0x8000, True):
            with self.subTest(word=word), self.assertRaises(ValueError):
                d.passing_cell(word)

    def test_reject_changed_decomposition(self):
        exact = exact_fixture()
        exact["L12_signed_Q24_drift"] = "0"
        with self.assertRaisesRegex(ValueError, "identity"):
            d.sensitivity(exact)

    def test_reject_external_binding_before_read(self):
        with patch.object(d.retained, "record") as record:
            with self.assertRaises(ValueError):
                d.prior.BoundInputs().bind(
                    {"path": "/outside/input", "bytes": 1, "sha256": "x"})
            record.assert_not_called()

    def test_reject_changed_pin(self):
        with patch.object(d.retained, "record", return_value={"sha256": "wrong"}):
            with self.assertRaisesRegex(ValueError, "pinned"):
                d.pinned(d.prior.BoundInputs(), d.INPUT / "result.json", d.prior.RESULT_SHA)

    def test_reject_changed_attribution_boundaries(self):
        document = {
            "diagnostic_id": d.prior.ID, "status": "DIAGNOSED", "node": [13, 0, 18],
            "index": 62, "retained_status": "FAIL", **d.FLAGS,
            "baseline": {"actual_fp16_bits": "6630", "nearest_fp16_bits": "662f",
                         "excess_budget": "1/8", "accepted": False},
        }
        validation = {"collected": 16, "executed": 16, "failures": 0, "errors": 0, "skipped": 0}
        d.check_attribution(document, validation)
        for field in d.FLAGS:
            changed = copy.deepcopy(document)
            changed[field] = 1 if type(changed[field]) is int else True
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.check_attribution(changed, validation)
        validation["executed"] = 0
        with self.assertRaises(ValueError):
            d.check_attribution(document, validation)

    def test_scope_and_execution_guards(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["focused_tests"], d.EXPECTED_TESTS)
        self.assertEqual(self.analysis["classification"], "inconclusive_unique_cause")
        for key, value in d.FLAGS.items():
            self.assertEqual(contract[key], value)
        with d.no_execution():
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                d.subprocess.Popen(["not-an-executed-command"])
        with self.assertRaisesRegex(ValueError, "selected"):
            d.authenticate(Path("/not-selected"), d.ATTRIBUTION)


if __name__ == "__main__":
    unittest.main()
