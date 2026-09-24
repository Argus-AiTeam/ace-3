"""Focused legacy/v2 gate-applicability regressions; no decoder RTL."""

from fractions import Fraction
import json
import struct
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import decoder_gate_policy as policy


def words(*values):
    return np.asarray(values, dtype="<f2").view("<u2")


class DecoderGatePolicyTests(unittest.TestCase):
    def evaluate(self, stage=18, actual=1579, reference=1581, selected=policy.POLICY_ID,
                 reference64=float.fromhex("0x1.8ab7c1503ecc5p+10")):
        return policy.evaluate_decoder_stage(
            stage=stage, actual=words(actual), reference=words(reference), policy=selected,
            reference_binary64=None if reference64 is None else np.asarray([reference64], dtype="<f8"))

    def test_contract_binding(self):
        contract = json.loads(policy.CONTRACT.read_text())
        self.assertEqual(contract["policy_id"], policy.POLICY_ID)
        self.assertEqual(contract["numerical_profile"], policy.binary64.PROFILE_ID)
        self.assertTrue(contract["opt_in"])

    def test_exact_conflicting_scalar_and_legacy(self):
        v2 = self.evaluate()
        self.assertEqual(v2["status"], "PASS")
        self.assertFalse(v2["fp16"]["passed"])
        self.assertEqual(v2["fp16"]["failures"][0]["absolute_error"], "2")
        self.assertEqual(v2["fp16"]["failures"][0]["relative_error"], "2/1581")
        self.assertEqual(v2["fp16"]["failures"][0]["ulp"], 2)
        self.assertIn("diagnostic", v2["fp16_role"])
        row = v2["binary64_v1"]["rows"][0]
        self.assertEqual(row["q"], "566583104315/4398046511104")
        self.assertEqual(row["excess_error"], "0")
        self.assertEqual(self.evaluate(selected=None)["status"], "FAIL")
        self.assertEqual(self.evaluate(selected=None, reference64=None)["status"], "FAIL")
        for actual, excess in ((1580, "1"), (1581, "2"), (1582, "3")):
            failed = self.evaluate(actual=actual)
            self.assertEqual(failed["status"], "FAIL")
            self.assertTrue(failed["fp16"]["passed"])
            self.assertEqual(failed["binary64_v1"]["rows"][0]["excess_error"], excess)

    def test_every_internal_gate_stays_mandatory(self):
        for stage in range(18):
            for selected in (None, policy.POLICY_ID):
                report = self.evaluate(stage=stage, selected=selected, reference64=None)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(report["fp16_role"], "mandatory")

    def test_missing_reference_blocks_only_opt_in(self):
        self.assertEqual(self.evaluate(actual=1581, reference64=None)["status"], "BLOCKED")
        self.assertEqual(self.evaluate(actual=1581, selected=None, reference64=None)["status"], "PASS")

    def test_scope_and_invalid_operands(self):
        for stage in (-1, 19, True):
            with self.assertRaises(ValueError):
                policy.fp16_is_mandatory(stage, policy.POLICY_ID)
        with self.assertRaises(ValueError):
            policy.fp16_is_mandatory(18, "native-fp16-only")
        with self.assertRaises(ValueError):
            self.evaluate(stage=17)
        for actual, reference, reference64 in (
                (float("inf"), 1, 1), (1, float("nan"), 1), (1, 1, float("inf")), (1, 1, 65505)):
            with self.assertRaises(policy.binary64.InvalidOperand):
                self.evaluate(actual=actual, reference=reference, reference64=reference64)
        with self.assertRaises(ValueError):
            policy.compare_fp16(np.asarray([1.0]), words(1))
        with self.assertRaises(ValueError):
            policy.compare_fp16(words(1, 2), words(1))

    def test_original_exact_predicate_boundaries(self):
        cases = [(0.125, 0), (0.1251220703125, 0), (1001, 1000),
                 (1025, 1024), (-1025, -1024), (1579, 1581),
                 (0, 2**-24), (-0.0, 0.0), (65504, 65472)]
        for actual, reference in cases:
            a, r = words(actual), words(reference)
            av = Fraction.from_float(struct.unpack("<e", a.tobytes())[0])
            rv = Fraction.from_float(struct.unpack("<e", r.tobytes())[0])
            def order(word):
                return 0x8000 - (word & 0x7fff) if word & 0x8000 else 0x8000 + word
            ulp = abs(order(int(a[0])) - order(int(r[0])))
            expected = abs(av - rv) <= Fraction(1, 8) or (
                abs(av - rv) / max(abs(rv), Fraction(1, 16384)) < Fraction(1, 1000) and ulp <= 1)
            self.assertEqual(policy.compare_fp16(a, r)["passed"], expected)

    def test_software_mode_cannot_launch_continuation(self):
        from ace3.model.candidates import run_single_round_residual_rtl as runner
        args = ["runner", "--out", str(runner.BUILD / "single_round_residual_rtl_execution_test"),
                "--gate-policy", policy.POLICY_ID, "--reevaluate-software", str(runner.BUILD / "old"),
                "--retained-manifest-sha256", "0" * 64, "--retained-review", "/review.json",
                "--continuation-layer", "5"]
        with patch("sys.argv", args), patch.object(runner, "continue_cone") as rtl:
            with self.assertRaises(SystemExit):
                runner.main()
            rtl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
