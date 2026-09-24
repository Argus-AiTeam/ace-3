"""Focused control boundaries; official tensors are used only by the real run."""

import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_software_canonical_control_v1 as control


class CanonicalControlTests(unittest.TestCase):
    def test_rounding_controls_sign_ties_subnormals_and_endpoints(self):
        values = np.array([0., -0., 1., -1., 1 + 2**-11, -(1 + 2**-11),
                           2**-25, -2**-25, 65504., -65504.], dtype="<f8")
        expected = {
            "rne": [0, 0x8000, 0x3c00, 0xbc00, 0x3c00, 0xbc00, 0, 0x8000, 0x7bff, 0xfbff],
            "toward_zero": [0, 0x8000, 0x3c00, 0xbc00, 0x3c00, 0xbc00, 0, 0x8000, 0x7bff, 0xfbff],
            "away_zero": [0, 0x8000, 0x3c00, 0xbc00, 0x3c01, 0xbc01, 1, 0x8001, 0x7bff, 0xfbff],
        }
        for mode, words in expected.items():
            np.testing.assert_array_equal(control.rounding_control(values, mode), words)
        upper = np.array([1 + 3 * 2**-12, -(1 + 3 * 2**-12)], dtype="<f8")
        np.testing.assert_array_equal(control.rounding_control(upper, "toward_zero"), [0x3c00, 0xbc00])
        np.testing.assert_array_equal(control.rounding_control(upper, "rne"), [0x3c01, 0xbc01])

    def test_rounding_controls_reject_invalid_range_dtype_and_modes(self):
        for values in (np.array([np.nan]), np.array([np.inf]), np.array([65505.]),
                       np.array([-65505.]), np.ones(2, dtype="<f4")):
            with self.assertRaisesRegex(ValueError, "invalid"):
                control.rounding_control(values, "rne")
        with self.assertRaisesRegex(ValueError, "unknown"):
            control.rounding_control(np.zeros(1, dtype="<f8"), "witness_patch")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            control.decompose(None, None, repair_sufficiency_only=True, weighted_backward_only=True)

    def test_counterfactual_gate_keeps_real_local_error_and_global_drift_failures(self):
        words = np.full(896, 0x3c00, dtype="<u2")
        bad = words.copy()
        bad[0] = 0x4000
        local = control.gates.evaluate_decoder_stage(
            stage=16, actual=bad, reference=words, policy=control.gates.POLICY_ID,
            local_reference=words)
        self.assertEqual(local["status"], "FAIL")
        global_result = control.gates.evaluate_decoder_stage(
            stage=18, actual=words, reference=words, policy=control.gates.POLICY_ID,
            reference_binary64=np.full(896, 1.5, dtype="<f8"))
        self.assertEqual(global_result["status"], "FAIL")
        self.assertEqual(global_result["binary64_v1"]["failure_count"], 896)

    def test_weighted_sum_exact_cancellation_and_invalid_inputs(self):
        weights = np.array([1.0, -2.0, 0.5], dtype="<f8")
        self.assertEqual(control.weighted_sum(weights, [Fraction(1, 3), Fraction(1, 6), Fraction(1, 8)]),
                         Fraction(1, 16))
        for bad in (weights[:-1], weights.astype("<f4"), np.array([1., np.nan, 2.])):
            with self.assertRaises(ValueError):
                control.weighted_sum(bad, [1, 2, 3])

    def test_ordered_secants_preserve_signed_effects_and_zero_change(self):
        effects = [Fraction(3, 8), Fraction(-1, 8), Fraction(0)]
        changes = [Fraction(1, 4), Fraction(1, 2), Fraction(0)]
        slopes = control.finite_secant(effects, changes)
        self.assertEqual(control.weighted_sum(slopes, changes), sum(effects))
        self.assertEqual(slopes[-1], 0)
        with self.assertRaisesRegex(ValueError, "unchanged operand"):
            control.finite_secant([Fraction(1)], [Fraction(0)])
        with self.assertRaisesRegex(ValueError, "coverage"):
            control.finite_secant(effects, changes[:-1])

    def test_rms_backward_includes_off_coordinate_coupling(self):
        actual = np.array([1.0, 3.0, -0.5], dtype="<f8")
        original = np.array([1.0, 2.0, -0.5], dtype="<f8")
        gamma = np.array([1.5, 0.5, -2.0], dtype="<f8")
        weights = np.array([1.0, 0.0, 0.0], dtype="<f8")
        direct, coupled = control.rms_backward(weights, actual, original, gamma)
        # Independent scalar evaluation: changing coordinate 1 changes output 0.
        def scalar(x):
            radius = (sum(float(v) ** 2 for v in x) / len(x) + 1e-6) ** 0.5
            return sum(float(w) * float(g) * float(v) / radius
                       for w, g, v in zip(weights, gamma, x))
        change = control.exact_delta(actual, original)
        self.assertEqual(control.weighted_sum(direct, change), 0)
        self.assertNotEqual(control.weighted_sum(coupled, change), 0)
        self.assertAlmostEqual(float(control.weighted_sum(direct + coupled, change)),
                               scalar(actual) - scalar(original), places=14)
        zero = np.zeros(3, dtype="<f8")
        d, c = control.rms_backward(weights, zero, zero, gamma)
        self.assertEqual(control.weighted_sum(d + c, control.exact_delta(zero, zero)), 0)
        with self.assertRaises(ValueError):
            control.rms_backward(weights, actual, original, gamma[:-1])

    def test_weighted_modes_conflict_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            control.decompose(None, None, producer_cones_only=True, weighted_backward_only=True)

    def test_producer_telescope_separates_rounding_oracle_and_operand_errors(self):
        count = control.local.SIZES[11]
        words = np.full(count, 0x3c00, dtype="<u2")
        local_words = np.full(count, 0x3c01, dtype="<u2")
        same = np.full(count, 1.0 + 2.0 ** -12, dtype="<f8")
        original = np.full(count, 0.75, dtype="<f8")
        rows = control.cone_rows(words, local_words, same, original, 2, 11, "actual")
        self.assertEqual(len(rows), 896)
        for row in rows:
            self.assertEqual(sum(Fraction(row[k]) for k in control.CONE_TERMS), Fraction(1, 4))
            self.assertEqual(row["local_fp16_delta"], "-1/1024")
            self.assertEqual(row["oracle_rounding_difference"], "1/1024")
            self.assertEqual(row["binary64_to_fp16_rounding"], "-1/4096")
            self.assertEqual(row["same_input_operator_delta"], "-1/4096")
        self.assertEqual(rows[-1]["index"], 895)
        summary = control.decomposition_totals(rows, control.CONE_TERMS)
        self.assertEqual(summary["coordinates"], 896)
        self.assertEqual(summary["classifications"], {"both": 896})
        self.assertEqual(summary["closure_failures"], 0)

    def test_producer_telescope_rejects_missing_nonfinite_wrong_shape_and_scope(self):
        words = np.zeros(896, dtype="<u2")
        values = np.zeros(896, dtype="<f8")
        for bad in (values[:-1], values.astype("<f4"), np.full(896, np.nan)):
            with self.assertRaisesRegex(ValueError, "binary64 cone"):
                control.cone_rows(words, words, bad, values, 0, 11, "actual")
        with self.assertRaisesRegex(ValueError, "scope"):
            control.cone_rows(words, words, values, values, 9, 11, "actual")
        corrupt = words.copy()
        corrupt[-1] = 0x7c00
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            control.cone_rows(corrupt, words, values, values, 0, 11, "actual")
        with self.assertRaises(KeyError):
            control.cone_original({"input_hidden": values}, values)
        with self.assertRaisesRegex(ValueError, "splice"):
            control.cone_original({"input_hidden": values}, np.ones(896, dtype="<f8"))

    def test_weighted_down_all_inputs_cancellation_and_reduction_rounding(self):
        rows = []
        for index in range(4864):
            rows.append({"stage": 16, "layer": 2, "lineage": "actual", "index": index,
                         "local_fp16_delta": "1/2", "oracle_rounding_difference": "0",
                         "binary64_to_fp16_rounding": "-1/4", "gate_operand_delta": "1/8",
                         "up_operand_delta": "-1/16", "global_operator_delta": "5/16"})
        weights = np.ones(4864, dtype="<f8")
        weights[1::2] = -1
        weights[-1] = 2
        endpoint = {"stage": 17, "layer": 2, "index": 62, "lineage": "actual",
                    "operand_trajectory_delta": "31/32", "same_input_operator_delta": "-1/8",
                    "global_operator_delta": "27/32"}
        contributions, total = control.down_weighted_rows(rows, endpoint, weights)
        self.assertEqual(len(contributions), 4864)
        self.assertEqual(total["weighted_input_delta"], "15/16")
        self.assertEqual(total["binary64_reduction_rounding_delta"], "1/32")
        self.assertEqual(contributions[-1]["input_index"], 4863)
        rows[-1]["index"] = 0
        with self.assertRaisesRegex(ValueError, "splice"):
            control.down_weighted_rows(rows, endpoint, weights)

    def residual_fixture(self, incoming=0, operator=0):
        words = np.full(896, 0x3c00, dtype="<u2")
        parent = control.root_state(words)
        parent["i"] += incoming
        parent["h"][:] = [control.rational.project(int(i), int(z))
                          for i, z in zip(parent["i"], parent["z"])]
        branch = np.full(896, operator, dtype="<u2")
        scratch = control.retained.transition_reference(parent, branch)
        output = control.retained.transition_reference(scratch, branch)
        arrays = {"input_hidden": parent["h"], "input_i": parent["i"], "input_z": parent["z"],
                  "input_cache_k": np.empty((0, 128), dtype="<u2"),
                  "input_cache_v": np.empty((0, 128), dtype="<u2"),
                  "stage11": branch, "stage17": branch,
                  "stage12": scratch["h"], "stage18": output["h"]}
        for prefix, state in (("scratch", scratch), ("output", output)):
            arrays[prefix + "_i"], arrays[prefix + "_z"] = state["i"], state["z"]
        return arrays

    def test_exact_state_operator_and_projection_split(self):
        canonical = self.residual_fixture()
        actual = self.residual_fixture(incoming=4096, operator=0x0c00)
        rows = control.residual_rows(actual, canonical, canonical, 0, 12)
        self.assertEqual(len(rows), 896)
        row = rows[-1]
        self.assertEqual(row["classification"], "both")
        self.assertEqual(row["incoming_delta"], "1/4096")
        self.assertEqual(row["operator_delta"], "1/4096")
        self.assertEqual(row["state_delta"], "1/2048")
        self.assertEqual(row["projection_delta"], "-1/2048")
        self.assertEqual(row["view_delta"], "0")
        self.assertEqual(row["operator_same_input_arithmetic_delta"], "1/4096")
        self.assertEqual(row["operator_operand_trajectory_delta"], "0")
        totals = control.decomposition_totals(rows, ("state_delta",))
        self.assertEqual(totals["terms"]["state_delta"]["signed_sum"], "7/16")

    def test_cancellation_and_operand_drift_are_not_local_error(self):
        canonical = self.residual_fixture()
        actual = self.residual_fixture(incoming=4096, operator=0x8c00)
        rows = control.residual_rows(actual, canonical, actual, 1, 12)
        self.assertTrue(all(r["state_cancellation"] for r in rows))
        self.assertEqual(rows[62]["operator_same_input_arithmetic_delta"], "0")
        self.assertEqual(rows[62]["operator_operand_trajectory_delta"], "-1/4096")
        self.assertEqual(control.drift_class(1, 0), "incoming_only")
        self.assertEqual(control.drift_class(0, 1), "operator_only")
        self.assertEqual(control.drift_class(0, 0), "neither")

    def test_decomposition_rejects_splice_nonfinite_and_scope(self):
        canonical = self.residual_fixture()
        spliced = self.residual_fixture()
        spliced["scratch_i"][-1] += 1
        with self.assertRaisesRegex(ValueError, "spliced"):
            control.residual_rows(spliced, canonical, canonical, 0, 12)
        with self.assertRaisesRegex(ValueError, "scope"):
            control.residual_rows(canonical, canonical, canonical, 9, 12)
        corrupt = dict(canonical, stage11=np.full(896, 0x7c00, dtype="<u2"))
        with self.assertRaises(ValueError):
            control.residual_rows(canonical, canonical, corrupt, 0, 12)

    def test_original_global_reference_is_not_reanchored(self):
        actual = self.residual_fixture(incoming=4096, operator=0x0c00)
        previous = np.ones(896, dtype="<f8")
        reference = np.full(896, 1.25, dtype="<f8")
        row = control.global_rows(actual, previous, reference, 0, "actual")[62]
        self.assertEqual(row["incoming_delta"], "1/4096")
        self.assertEqual(Fraction(row["net_operator_delta"]), Fraction(1, 2048) - Fraction(1, 4))
        self.assertEqual(Fraction(row["global_error"]), Fraction(1, 1024) - Fraction(1, 4))
        self.assertEqual(row["original_output_binary64_hex"], float(1.25).hex())
        reference[-1] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid original"):
            control.global_rows(actual, previous, reference, 0, "actual")

    def test_original_global_residuals_close_all_coordinates(self):
        actual = self.residual_fixture(incoming=4096, operator=0x0c00)
        original = {"input_hidden": np.ones(896, dtype="<f8"),
                    "stage11": np.full(896, 0.125, dtype="<f8"),
                    "stage12": np.full(896, 1.125, dtype="<f8"),
                    "stage17": np.full(896, -0.25, dtype="<f8"),
                    "stage18": np.full(896, 0.875, dtype="<f8")}
        for stage in (12, 18):
            rows = control.original_residual_rows(actual, original, 2, stage, "actual")
            self.assertEqual(len(rows), 896)
            for row in rows:
                self.assertEqual(Fraction(row["incoming_delta"]) + Fraction(row["operator_delta"])
                                 - Fraction(row["original_residual_rounding"])
                                 + Fraction(row["projection_error"]), Fraction(row["global_error"]))
                self.assertEqual(row["closure_error"], "0")
        self.assertEqual(rows[62]["original_operator_binary64_hex"], (-0.25).hex())
        original["stage17"][-1] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid original operand"):
            control.original_residual_rows(actual, original, 2, 18, "actual")
        with self.assertRaisesRegex(ValueError, "scope"):
            control.original_residual_rows(actual, original, 9, 18, "actual")

    def test_original_binary64_rounding_is_not_assigned_to_q24(self):
        actual = self.residual_fixture()
        original = {"input_hidden": np.ones(896, dtype="<f8"),
                    "stage11": np.full(896, 2.0 ** -53, dtype="<f8"),
                    "stage12": np.ones(896, dtype="<f8")}
        row = control.original_residual_rows(actual, original, 0, 12, "canonical")[0]
        self.assertEqual(Fraction(row["original_residual_rounding"]), -Fraction(1, 2 ** 53))
        self.assertEqual(row["global_error"], "0")
        original["stage12"][-1] = 2.0
        with self.assertRaisesRegex(ValueError, "residual identity"):
            control.original_residual_rows(actual, original, 0, 12, "canonical")

    def test_original_endpoint_authentication_includes_signed_zero(self):
        expected = np.ones(896, dtype="<f8")
        captured = {"stage18": expected.copy()}
        control.verify_global_endpoint(captured, expected)
        captured["stage18"][-1] = np.nextafter(1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "endpoint drift"):
            control.verify_global_endpoint(captured, expected)
        expected[-1], captured["stage18"][-1] = 0.0, -0.0
        with self.assertRaisesRegex(ValueError, "endpoint drift"):
            control.verify_global_endpoint(captured, expected)
        captured["stage18"][-1] = np.inf
        with self.assertRaisesRegex(ValueError, "invalid original endpoint"):
            control.verify_global_endpoint(captured, expected)

    def test_root_sign_and_exact_carry(self):
        words = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3c00], dtype="<u2"), 896)
        state = control.root_state(words)
        control.retained.verify_parent(state, state, embedding=words)
        increment = np.full(896, 0x0c00, dtype="<u2")
        state = control.retained.transition_reference(state, increment)
        self.assertEqual(int(state["i"][4]), (1 << 24) + 4096)
        self.assertEqual(int(state["h"][4]), 0x3c00)

    def test_all_coordinate_comparison_and_real_error(self):
        left = np.zeros(896, dtype="<u2")
        right = left.copy()
        right[-1] = 0x3c00
        report = control.compare(left, right)
        self.assertEqual(report["coordinates"], 896)
        self.assertEqual(report["first_bit_difference"], 895)
        self.assertEqual(report["fp16_gate"]["failure_count"], 1)

    def test_l9_and_invalid_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "L0-L2"):
            control.canonical_layer({}, 9, {})
        with self.assertRaisesRegex(ValueError, "FP16-only"):
            control.canonical_layer({}, 0, {"h": np.zeros(896, dtype="<u2")})

    def test_canonical_control_never_dispatches_candidate(self):
        tensors = {name: np.zeros(shape, dtype=dtype)
                   for name, (shape, dtype) in control.local.tensor_shapes(0).items()}
        for name, value in tensors.items():
            if name.endswith(".weight") or name.endswith(".scales"):
                value.fill(1)
        parent = control.root_state(np.full(896, 0x3c00, dtype="<u2"))
        with patch.object(control.retained.candidate, "stages", side_effect=AssertionError("candidate dispatch")), \
                patch.object(control.retained.candidate, "add", side_effect=AssertionError("candidate arithmetic")):
            arrays = control.canonical_layer(tensors, 0, parent)
        np.testing.assert_array_equal(arrays["output_i"], parent["i"])
        np.testing.assert_array_equal(arrays["stage18"], parent["h"])
        self.assertEqual(arrays["output_cache_k"].shape, (1, 128))

    def test_original_witness_still_rejects_and_cannot_be_rewritten(self):
        actual = np.full(896, 0x616e, dtype="<u2")
        reference = np.full(896, float.fromhex(control.WITNESS["reference_binary64_hex"]))
        report = control.gates.evaluate_decoder_stage(
            stage=18, actual=actual, reference=actual, policy=control.gates.POLICY_ID,
            reference_binary64=reference)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["binary64_v1"]["rows"][62]["excess_error"], "1/2")
        with self.assertRaisesRegex(ValueError, "rejection/coverage"):
            control.preserve_witness(report, {"first_failure": control.WITNESS})


if __name__ == "__main__":
    unittest.main()
