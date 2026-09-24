"""Regressions for the reviewed legacy layer21 lineage schema."""

import copy
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model import audit_position2_tail_lineage as audit


class RetainedLayerSchemaTests(unittest.TestCase):
    def setUp(self):
        self.layers = {}
        self.bits = {}
        for index in (21, 22, 23):
            positions = []
            for position in range(3):
                path = f"layer{index}-position{position}.hex"
                bits = [index * 3 + position] * 896
                self.bits[path] = bits
                semantic = audit.binding.head.sha256_bytes(
                    audit.binding.head.terminal_payload(bits))
                tx = {"position": position,
                      "output": {"path": path, "semantic_sha256": semantic}}
                if index > 21:
                    tx["input"] = {"sha256": self.layers[index - 1]["positions"][
                        position]["output"]["semantic_sha256"]}
                positions.append(tx)
            layer = {
                "layer_index": index, "actual_output_fed_rtl_chain": True,
                "positions": positions,
                "independent_comparisons": [
                    {"stage": stage, "failure_count": 0} for stage in range(19)],
            }
            if index != 21:
                layer["earliest_material_mismatch"] = None
            self.layers[index] = layer
        self.prepared = [tx["output"] for tx in self.layers[21]["positions"]]
        self.addCleanup(patch.stopall)
        patch.object(audit.binding, "authenticate").start()
        patch.object(audit.binding.head, "load_terminal_bits",
                     side_effect=lambda path: self.bits[str(Path(path))]).start()

    def run_audit(self, layers=None, prepared=None):
        return audit.audit_retained_layers(
            self.layers if layers is None else layers,
            self.prepared if prepared is None else prepared, 896)

    def test_legacy_absence_and_explicit_none_preserve_all_six_links(self):
        self.assertEqual(len(self.run_audit()), 6)
        self.layers[21]["earliest_material_mismatch"] = None
        self.assertEqual(len(self.run_audit()), 6)

    def test_recorded_mismatch_is_not_legacy_absence(self):
        self.layers[21]["earliest_material_mismatch"] = {"stage": 0}
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "layer21 comparison"):
            self.run_audit()

    def test_later_layers_still_require_the_field(self):
        for index in (22, 23):
            with self.subTest(layer=index):
                layers = copy.deepcopy(self.layers)
                del layers[index]["earliest_material_mismatch"]
                with self.assertRaisesRegex(audit.binding.evidence.AttemptError,
                                            f"layer{index} comparison"):
                    self.run_audit(layers)

    def test_incomplete_or_failing_legacy_evidence_is_rejected(self):
        for case in ("output", "stage", "failure", "chain"):
            with self.subTest(case=case):
                layers = copy.deepcopy(self.layers)
                layer = layers[21]
                if case == "output":
                    layer["positions"].pop()
                elif case == "stage":
                    layer["independent_comparisons"].pop()
                elif case == "failure":
                    layer["independent_comparisons"][0]["failure_count"] = 1
                else:
                    layer["actual_output_fed_rtl_chain"] = False
                with self.assertRaisesRegex(audit.binding.evidence.AttemptError,
                                            "layer21 comparison"):
                    self.run_audit(layers)

    def test_incomplete_and_nonfinite_outputs_are_rejected(self):
        path = self.layers[21]["positions"][0]["output"]["path"]
        for bits in ([0] * 895, [0x7c00] * 896, [0xfc00] * 896, [0x7e00] * 896):
            with self.subTest(first=bits[0], count=len(bits)):
                self.bits[path] = bits
                with self.assertRaisesRegex(audit.binding.evidence.AttemptError,
                                            "incomplete or nonfinite"):
                    self.run_audit()

    def test_semantic_drift_is_rejected(self):
        self.layers[21]["positions"][0]["output"]["semantic_sha256"] = "wrong"
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "semantic digest"):
            self.run_audit()

    def test_each_hidden_handoff_is_required(self):
        for index in (22, 23):
            for position in range(3):
                with self.subTest(layer=index, position=position):
                    layers = copy.deepcopy(self.layers)
                    layers[index]["positions"][position]["input"]["sha256"] = "foreign"
                    with self.assertRaisesRegex(audit.binding.evidence.AttemptError,
                                                "hidden ancestry"):
                        self.run_audit(layers)

    def test_prepared_layer22_inputs_must_match_layer21(self):
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError,
                                    "prepared layer22 input"):
            self.run_audit(prepared=[])


class RmsnormPolicyCompatibilityTests(unittest.TestCase):
    def source(self, predicate, floor="2.0**-14"):
        return f"""
ABSOLUTE_TOLERANCE = 0.125
RELATIVE_TOLERANCE = 0.001
MAX_ULP_DISTANCE = 1
def fp16_policy_comparison():
    relative = absolute / np.maximum(np.abs(expected_values), {floor})
    if {predicate}:
        failures.append((token, index))
"""

    def test_current_predicate_matches_independent_exact_controls(self):
        result = audit.probe_rmsnorm_policy(self.source(
            "absolute[token, index] > ABSOLUTE_TOLERANCE and "
            "(relative[token, index] >= RELATIVE_TOLERANCE or distance > MAX_ULP_DISTANCE)"))
        self.assertEqual(result["required_policy_disagreements"], 0)
        self.assertEqual(len(result["cases"]), 8)

    def test_retained_predicate_has_three_finite_counterexamples(self):
        result = audit.probe_rmsnorm_policy(self.source(
            "absolute[token, index] > ABSOLUTE_TOLERANCE and "
            "relative[token, index] > RELATIVE_TOLERANCE and distance > MAX_ULP_DISTANCE",
            floor="2.0**-24"))
        self.assertEqual(result["required_policy_disagreements"], 3)
        self.assertEqual([row["case"] for row in result["cases"]
                          if row["source_failure"] != row["required_policy_failure"]],
                         ["relative_equality_two_ulp", "relative_pass_two_ulp",
                          "negative_relative_equality"])

    def test_changed_threshold_and_missing_expression_are_rejected(self):
        source = self.source("absolute[token, index] > ABSOLUTE_TOLERANCE")
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "tolerance drift"):
            audit.probe_rmsnorm_policy(source.replace("0.125", "0.25"))
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "unrecognized"):
            audit.probe_rmsnorm_policy(self.source("True"))


class CurrentTailPreparationTests(unittest.TestCase):
    def test_preparation_gate_matches_exact_fraction_oracle(self):
        np = audit.binding.np
        for expected, actual in ((256, 256), (1, 1.125), (1024, 1025),
                                 (1000, 1001), (1023.5, 1022.5), (-1000, -1001),
                                 (1000, 1002), (0, 2**-24), (-0.0, 0.0)):
            with self.subTest(expected=expected, actual=actual):
                bits = np.asarray([expected, actual], dtype="<f2").view("<u2")
                error = abs(Fraction(actual) - Fraction(expected))
                ratio = error / max(abs(Fraction(expected)), Fraction(1, 2**14))
                def ordered(raw):
                    return 0x8000 - (raw & 0x7fff) if raw & 0x8000 else 0x8000 + raw
                distance = abs(ordered(int(bits[1])) - ordered(int(bits[0])))
                failure = error > Fraction(1, 8) and (ratio >= Fraction(1, 1000) or distance > 1)
                result = audit.binding.fp16_interstage_comparison(bits[1:], bits[:1])
                self.assertEqual(result["failure_count"], int(failure))
                self.assertEqual(result["max_relative_error"], float(ratio))

    def test_nonfinite_operands_rejected_on_either_side(self):
        np = audit.binding.np
        finite = np.asarray([0], dtype="<u2")
        for raw in (0x7c00, 0xfc00, 0x7e00):
            nonfinite = np.asarray([raw], dtype="<u2")
            for actual, expected in ((finite, nonfinite), (nonfinite, finite)):
                with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "nonfinite"):
                    audit.binding.fp16_interstage_comparison(actual, expected)

    def test_reference_preserves_fp32_and_both_fp16_boundaries(self):
        np = audit.binding.np
        torch = audit.binding.head.torch
        bits = np.asarray([[0x3000 + (index * 37) % 0x1800 for index in range(896)],
                           [0x8000 | (0x2400 + (index * 19) % 0x1400) for index in range(896)]],
                          dtype="<u2")
        weights = np.asarray([0x3800 + (index * 11) % 0x600 for index in range(896)], dtype="<u2")
        x = torch.from_numpy(bits.view("<f2").copy()).float()
        w = torch.from_numpy(weights.view("<f2").copy())
        normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).half()
        expected = (normalized * w).half().numpy().view("<u2")
        result = audit.binding.evidence.fp16_interstage_expected(bits, weights)
        np.testing.assert_array_equal(result, expected)
        mathematical = (bits.view("<f2").astype(np.float64) /
                        np.sqrt(np.mean(bits.view("<f2").astype(np.float64) ** 2,
                                        axis=-1, keepdims=True) + 1e-6) *
                        weights.view("<f2").astype(np.float64)).astype("<f2").view("<u2")
        self.assertTrue(np.any(result != mathematical))

    def test_reference_rejects_nonfinite_and_wrong_shape(self):
        np = audit.binding.np
        weights = np.full(896, 0x3c00, dtype="<u2")
        for bits in (np.full((1, 896), 0x7c00, dtype="<u2"),
                     np.zeros((1, 895), dtype="<u2")):
            with self.assertRaises(audit.binding.evidence.AttemptError):
                audit.binding.evidence.fp16_interstage_expected(bits, weights)

    def test_legacy_runtime_mapping_cannot_grant_current_authority(self):
        mapping = audit.binding.comparison_mapping()
        self.assertFalse(mapping["current_execution_compatible"])
        self.assertIn("2^-14", mapping["active_relative_denominator"])
        self.assertIn("1e-30", mapping["expressions"]["relative"]["expression"])

    def test_named_foreign_parent_rejected_before_any_artifact_access(self):
        contract = audit.binding.load(audit.binding.CONTRACT)
        foreign = {"parent": {"attempt_id": "model24_layer23_attempt001"}}
        with self.assertRaisesRegex(audit.binding.evidence.AttemptError, "^foreign_parent$"):
            audit.binding.admit(foreign, Path("unused"), contract)


if __name__ == "__main__":
    unittest.main()
