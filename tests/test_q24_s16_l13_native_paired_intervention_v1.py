"""Focused native paired diagnostic checks; no accepted ancestor replay."""

from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_native_paired_intervention_v1 as d


EXPECTED_TESTS = 12


class PairedTests(unittest.TestCase):
    def test_q24_mapping_ties_even(self):
        x = np.zeros(896, dtype="<f8")
        x[:4] = np.asarray([0.5, 1.5, -0.5, -1.5]) / (1 << 24)
        parent = d.mapped_parent(x)
        np.testing.assert_array_equal(parent["i"][:4], [0, 2, 0, -2])
        self.assertEqual(int(parent["z"][2]), 1)

    def test_q24_mapping_does_not_lift_fp16(self):
        x = np.full(896, 1 + 2 ** -20, dtype="<f8")
        parent = d.mapped_parent(x)
        self.assertEqual(int(parent["i"][0]), (1 << 24) + 16)
        self.assertEqual(int(parent["h"][0]), 0x3c00)

    def test_mapping_signed_zero(self):
        parent = d.mapped_parent(np.full(896, -0.0, dtype="<f8"))
        self.assertTrue(np.all(parent["h"] == 0x8000))
        self.assertTrue(np.all(parent["z"] == 1))

    def test_mapping_rejects_invalid(self):
        for x in (np.zeros(895, dtype="<f8"), np.zeros(896, dtype="<f4"),
                  np.full(896, np.nan), np.full(896, 65505.0)):
            with self.assertRaises(ValueError):
                d.mapped_parent(x)

    def test_reproduction_rejects_bit_drift(self):
        a = {"x": np.asarray([0.0], dtype="<f8")}
        d.same_arrays(a, a)
        with self.assertRaises(ValueError):
            d.same_arrays(a, {"x": np.asarray([-0.0], dtype="<f8")})

    def test_reproduction_rejects_schema(self):
        for b in ({}, {"x": np.ones(1, dtype="<f4")}, {"x": np.ones(2, dtype="<f8")}):
            with self.assertRaises(ValueError):
                d.same_arrays({"x": np.ones(1, dtype="<f8")}, b)

    def test_interaction_exact(self):
        self.assertEqual(d.interaction(Fraction(3, 8), Fraction(1, 8),
                                       Fraction(1, 4), Fraction(1, 8)), Fraction(1, 8))
        self.assertEqual(d.interaction(3, 2, 2, 1), 0)

    def test_classification_is_conditional(self):
        result = d.classification({"accepted": False}, {"accepted": True})
        self.assertEqual(result["classification"], "incoming_parent_sensitive_under_explicit_Q24_mapping")
        self.assertTrue(result["missing_evidence"])
        self.assertFalse(d.classification({"accepted": False}, {"accepted": False})
                         ["mapped_native_index62_passes"])
        with self.assertRaises(ValueError):
            d.classification({"accepted": True}, {"accepted": True})

    def test_output_scope_and_exclusivity(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/unrelated")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError):
                d.output_path(d.ROOT / "build/q24_s16_l13_native_paired_intervention_existing")

    def test_unknown_intervention_fails(self):
        with self.assertRaises(ValueError):
            d.execute({}, {}, "rtz_policy_change", {})

    def test_signed_decomposition_and_state_check(self):
        original = {k: np.full(896, v, dtype="<f8")
                    for k, v in (("input", 1), ("s11", 2), ("s17", 3), ("s18", 6))}
        parent = d.mapped_parent(original["input"])
        arrays = {f"stage{s}": np.full(896, np.float16(v).view(np.uint16), dtype="<u2")
                  for s, v in ((11, 2), (12, 3), (17, 3), (18, 6))}
        arrays.update(scratch_i=np.full(896, 3 << 24, dtype="<i8"),
                      output_i=np.full(896, 6 << 24, dtype="<i8"))
        row = d.parts(parent, arrays, original, 62)
        self.assertEqual(row["signed_error"], "0")
        self.assertEqual(sum(map(Fraction, row["signed_parts"].values())), 0)
        self.assertTrue(row["gate"]["accepted"])
        arrays["output_i"][62] += 1
        with self.assertRaises(ValueError):
            d.parts(parent, arrays, original, 62)

    def test_contract_boundaries(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(tuple(contract["modes"]), d.MODES)
        self.assertEqual(contract["rtl_invocations"], 0)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertIn("1/8", contract["gates"])


if __name__ == "__main__":
    unittest.main()
