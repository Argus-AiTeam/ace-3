"""Software Q24 state/references, with no simulator or official-model execution."""

import unittest

import numpy as np

from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_software_candidate_v1 as candidate
from ace3.model.candidates import q24_software_root_v3 as experiment
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational


def words(value, size=896):
    return np.full(size, value, dtype="<f2").view("<u2")


class Q24SoftwareTests(unittest.TestCase):
    def test_root_and_fp16_only_or_invented_parent_rejection(self):
        embedding = words(1)
        state = candidate.lift(embedding)
        experiment.verify_parent(state, state, embedding=embedding)
        with self.assertRaisesRegex(ValueError, "FP16-only"):
            experiment.verify_parent({"h": embedding}, state)
        invented = {k: v.copy() for k, v in state.items()}
        invented["i"][0] += 1
        with self.assertRaisesRegex(ValueError, "invented root"):
            experiment.verify_parent(invented, invented, embedding=embedding)
        with self.assertRaisesRegex(ValueError, "invented/spliced"):
            experiment.verify_parent(invented, state)
        with self.assertRaisesRegex(ValueError, "FP16-only"):
            next(candidate.stages({}, 0, {"h": embedding}, {}))

    def test_root_zero_subnormal_and_sign(self):
        embedding = np.resize(np.asarray([0, 0x8000, 1, 0x8001, 0x3ff, 0x7bff, 0xfbff],
                                         dtype="<u2"), 896)
        state = candidate.lift(embedding)
        experiment.verify_parent(state, state, embedding=embedding)
        for integer, tag, word in zip(state["i"], state["z"], embedding, strict=True):
            self.assertEqual(rational.root(int(word)), (int(integer), int(tag)))

    def test_exact_carry_survives_hidden_rounding(self):
        parent = candidate.lift(words(1))
        for _ in range(3):
            expected = experiment.transition_reference(parent, words(1 / 4096))
            parent = candidate.add(parent, words(1 / 4096))
            experiment.verify_parent(parent, expected)
        self.assertEqual(int(parent["h"][0]), int(words(1 + 1 / 1024)[0]))
        self.assertEqual(int(parent["i"][0]), (1 << 24) + 3 * (1 << 12))
        relifted = candidate.lift(parent["h"])
        with self.assertRaisesRegex(ValueError, "invented/spliced"):
            experiment.verify_parent(relifted, parent)

    def test_general_residual_reference_and_two_transitions(self):
        parent = candidate.lift(words(1579))
        parent = candidate.add(parent, words(0.25))
        arrays = {"input_hidden": parent["h"].copy(), "input_i": parent["i"].copy(),
                  "input_z": parent["z"].copy(), "stage11": words(0.25),
                  "input_cache_k": np.empty((0, 128), dtype="<u2"),
                  "input_cache_v": np.empty((0, 128), dtype="<u2")}
        scratch = candidate.add(parent, arrays["stage11"])
        arrays.update(scratch_i=scratch["i"], scratch_z=scratch["z"], stage12=scratch["h"])
        expected = experiment.check_stage_state(12, arrays, parent)
        old_local = local.local_reference(12, {k: arrays[k] for k in local.OPERANDS[12]}, {}, 0)
        self.assertTrue(np.all(expected == words(1580)))
        self.assertTrue(np.all(old_local == words(1579)))
        arrays["stage17"] = words(-0.25)
        successor = candidate.add(scratch, arrays["stage17"])
        arrays.update(output_i=successor["i"], output_z=successor["z"], stage18=successor["h"])
        experiment.check_stage_state(18, arrays, parent)
        arrays["output_i"] = arrays["output_i"].copy()
        arrays["output_i"][0] += 1
        with self.assertRaisesRegex(ValueError, "invented/spliced"):
            experiment.check_stage_state(18, arrays, parent)

    def test_invalid_zero_overflow_and_nonfinite_rejected(self):
        for integer, tag in ((1, 1), (0, 2), (1 << 63, 0),
                             (65520 << 24, 0), (-65520 << 24, 0)):
            with self.subTest(integer=integer, tag=tag), self.assertRaises(ValueError):
                candidate.view(integer, tag)
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                candidate.lift(words(value))
        negative = candidate.lift(words(-0.0))
        added = candidate.add(negative, words(-0.0))
        self.assertTrue(np.all(added["h"] == 0x8000))
        self.assertTrue(np.all(candidate.add(negative, words(0.0))["h"] == 0))

    def test_native_gemm_lanes_no_zero_adjustment(self):
        prefix = "model.layers.0.self_attn.q_proj"
        tensors = {
            prefix + ".qweight": np.full((896, 112), 0x76543210, dtype="<i4"),
            prefix + ".qzeros": np.zeros((7, 112), dtype="<i4"),
            prefix + ".scales": np.ones((7, 896), dtype="<f2"),
            prefix + ".bias": np.zeros(896, dtype="<f2")}
        activation = words(0)
        activation[0] = words(1)[0]
        actual = candidate.project(tensors, prefix, activation, single_bias=True)
        expected = local.local_reference(1, {"stage00": activation}, tensors, 0)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[:8].view("<f2").tolist(), [0, 4, 1, 5, 2, 6, 3, 7])

    def test_local_correctness_does_not_waive_global_or_reference_requirement(self):
        common = {"actual": words(1581), "reference": words(1581), "policy": gates.POLICY_ID}
        self.assertEqual(gates.evaluate_decoder_stage(
            stage=12, local_reference=words(1581), **common)["status"], "PASS")
        self.assertEqual(gates.evaluate_decoder_stage(
            stage=12, local_reference=words(1579), **common)["status"], "FAIL")
        self.assertEqual(gates.evaluate_decoder_stage(stage=18, **common)["status"], "BLOCKED")
        self.assertEqual(gates.evaluate_decoder_stage(
            stage=18, reference_binary64=np.full(896, 1579.125, dtype="<f8"),
            **common)["status"], "FAIL")

    def test_wrong_cache_or_hidden_identity_rejected(self):
        parent = candidate.lift(words(1))
        arrays = {"input_hidden": parent["h"].copy(), "input_i": parent["i"].copy(),
                  "input_z": parent["z"].copy(), "stage05": words(1, 128),
                  "stage06": words(1, 128), "output_cache_k": words(2, 128).reshape(1, 128),
                  "input_cache_k": np.empty((0, 128), dtype="<u2"),
                  "input_cache_v": np.empty((0, 128), dtype="<u2")}
        with self.assertRaisesRegex(ValueError, "KV write/read"):
            experiment.check_stage_state(6, arrays, parent)
        arrays["output_cache_k"] = words(1, 128).reshape(1, 128)
        arrays["input_i"][0] += 1
        with self.assertRaisesRegex(ValueError, "producer-to-consumer"):
            experiment.check_stage_state(6, arrays, parent)


if __name__ == "__main__":
    unittest.main()
