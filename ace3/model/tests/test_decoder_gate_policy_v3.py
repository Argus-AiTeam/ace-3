"""Focused reference-boundary regressions; no decoder simulator execution."""

import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import decoder_gate_policy as old
from ace3.model.candidates import decoder_gate_policy_v3 as policy
from ace3.model.candidates import local_operator_reference_v3 as local


def words(*values):
    return np.asarray(values, dtype="<f2").view("<u2")


class LocalOperatorV3Tests(unittest.TestCase):
    def test_contract_and_cross_boundary(self):
        contract = json.loads(policy.CONTRACT.read_text())
        self.assertEqual(contract["policy_id"], policy.POLICY_ID)
        self.assertEqual(contract["numerical_profile"], old.binary64.PROFILE_ID)
        arrays = {"input_hidden": np.full(896, words(1579)[0], dtype="<u2"),
                  "stage11": np.full(896, words(-0.25)[0], dtype="<u2")}
        reference = local.local_reference(12, arrays, {}, 8)
        self.assertTrue(np.all(reference == words(1579)[0]))
        trajectory = np.full(896, words(1581)[0], dtype="<u2")
        report = policy.evaluate_decoder_stage(
            stage=12, actual=reference, reference=trajectory,
            local_reference=reference, policy=policy.POLICY_ID)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["fp16"]["failure_count"], 896)
        for selected in (None, old.POLICY_ID):
            self.assertEqual(old.evaluate_decoder_stage(
                stage=12, actual=reference, reference=trajectory, policy=selected)["status"], "FAIL")
        corrupted = reference.copy()
        corrupted[62] = words(1581)[0]
        failed = policy.evaluate_decoder_stage(
            stage=12, actual=corrupted, reference=trajectory,
            local_reference=reference, policy=policy.POLICY_ID)
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(failed["local_operator_fp16"]["failures"][0]["index"], 62)

    def test_global_drift_and_missing_references(self):
        for stage in range(18):
            result = policy.evaluate_decoder_stage(
                stage=stage, actual=words(1), reference=words(1), policy=policy.POLICY_ID)
            self.assertEqual(result["status"], "BLOCKED")
        result = policy.evaluate_decoder_stage(
            stage=18, actual=words(1), reference=words(1), policy=policy.POLICY_ID)
        self.assertEqual(result["status"], "BLOCKED")
        # Every local comparison can be exact while a different original global
        # trajectory still rejects the layer output.
        self.assertEqual(policy.evaluate_decoder_stage(
            stage=17, actual=words(1581), reference=words(1581),
            local_reference=words(1581), policy=policy.POLICY_ID)["status"], "PASS")
        report = policy.evaluate_decoder_stage(
            stage=18, actual=words(1581), reference=words(1581),
            reference_binary64=np.asarray([1579.125], dtype="<f8"), policy=policy.POLICY_ID)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["binary64_v1"]["failure_count"], 1)
        for invalid in (float("inf"), float("nan"), 65505):
            with self.assertRaises(ValueError):
                policy.evaluate_decoder_stage(
                    stage=18, actual=words(1), reference=words(1),
                    reference_binary64=np.asarray([invalid], dtype="<f8"), policy=policy.POLICY_ID)

    def test_invalid_local_reference_and_scope(self):
        for reference in (words(float("nan")), words(float("inf")), words(1, 2),
                          np.asarray([1.0], dtype="<f8")):
            with self.assertRaises(ValueError):
                policy.evaluate_decoder_stage(stage=0, actual=words(1), reference=words(1),
                                              local_reference=reference, policy=policy.POLICY_ID)
        with self.assertRaises(ValueError):
            local.local_reference(12, {"stage12": words(1)}, {}, 8)
        with self.assertRaises(ValueError):
            policy.evaluate_decoder_stage(stage=18, actual=words(1), reference=words(1),
                                          local_reference=words(1), policy=policy.POLICY_ID)
        with self.assertRaises(ValueError):
            policy.evaluate_decoder_stage(stage=0, actual=words(1), reference=words(1), policy=None)

    def test_canonical_metadata_and_lane_order(self):
        tensors, records = {}, {}
        for name, (shape, dtype) in local.tensor_shapes(5).items():
            value = np.ones(shape, dtype=dtype)
            tensors[name] = value
            records[name] = {"shape": list(shape), "dtype": dtype,
                             "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        local.authenticate_tensors(tensors, records, 5)
        name = "model.layers.5.self_attn.q_proj.qzeros"
        tensors[name][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "canonical tensor data"):
            local.authenticate_tensors(tensors, records, 5)
        with self.assertRaisesRegex(ValueError, "wrong canonical tensor"):
            local.authenticate_tensors(tensors, records, 6)
        prefix = "model.layers.5.self_attn.q_proj"
        tensors[prefix + ".qweight"][:] = 0x76543210
        tensors[prefix + ".qzeros"][:] = 0
        tensors[prefix + ".scales"][:] = 1
        tensors[prefix + ".bias"][:] = 0
        actual_input = np.zeros(896, dtype="<f2")
        actual_input[0] = 1
        result = local.local_reference(1, {"stage00": actual_input.view("<u2")}, tensors, 5)
        self.assertEqual(result[:8].view("<f2").tolist(), [0, 4, 1, 5, 2, 6, 3, 7])

    def test_state_operand_and_residual_identity(self):
        arrays = {f"stage{s:02d}": np.zeros(n, dtype="<u2") for s, n in local.SIZES.items()}
        arrays["input_hidden"] = np.zeros(896, dtype="<u2")
        for k in ("k", "v"):
            arrays[f"input_cache_{k}"] = np.empty((0, 128), dtype="<u2")
            arrays[f"output_cache_{k}"] = np.zeros((1, 128), dtype="<u2")
        args = {"expected_hidden": arrays["input_hidden"].copy(), "position": 0, "history": [9707]}
        self.assertEqual(local.validate_lineage(arrays, **args)["status"], "PASS")
        for key, value in (("input_hidden", np.ones(896, dtype="<u2")),
                           ("input_cache_k", np.zeros((1, 128), dtype="<u2")),
                           ("output_cache_v", np.ones((1, 128), dtype="<u2")),
                           ("stage06", np.ones(128, dtype="<u2")),
                           ("stage18", np.ones(896, dtype="<u2"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                local.validate_lineage(dict(arrays, **{key: value}), **args)
        for position, history in ((1, [9707]), (0, [358])):
            with self.assertRaises(ValueError):
                local.validate_lineage(arrays, args["expected_hidden"], position=position, history=history)

    def test_rtl_numerics_do_not_admit_missing_runtime_state(self):
        with patch.object(policy, "evaluate_p0_transaction") as evaluate:
            result = policy.evaluate_actual_rtl_result(runtime_admission=None)
        evaluate.assert_not_called()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["numerical_status"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
