"""Independent RTZ format checks and unchanged numerical rejection regressions."""

import ast
from pathlib import Path
import unittest

import numpy as np
import torch

from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as candidate


class NativeTowardZeroTests(unittest.TestCase):
    def test_rne_all_midpoints_and_adjacent_binary64_values(self):
        bits = np.arange(0x7bff, dtype="<u2")
        lower = bits.view("<f2").astype("<f8")
        upper = (bits + 1).astype("<u2").view("<f2").astype("<f8")
        midpoint = (lower + upper) / 2
        for values, expected in (
                (np.nextafter(midpoint, lower), bits),
                (midpoint, bits + (bits & 1)),
                (np.nextafter(midpoint, upper), bits + 1)):
            np.testing.assert_array_equal(candidate.rne(torch.from_numpy(values)), expected)
            np.testing.assert_array_equal(candidate.rne(torch.from_numpy(-values)), expected | 0x8000)
        exact_projection = torch.tensor([-2423259023 / 68719476736], dtype=torch.float64)
        np.testing.assert_array_equal(candidate.rne(exact_projection), [0xa883])

    def test_all_finite_encodings_and_intervals(self):
        positive = np.arange(0x7c00, dtype="<u2")
        all_words = np.concatenate((positive, positive | 0x8000))
        grid = all_words.view("<f2").astype("<f8")
        np.testing.assert_array_equal(candidate.toward_zero(grid), all_words)
        lower = positive[:-1].view("<f2").astype("<f8")
        upper = positive[1:].view("<f2").astype("<f8")
        for fraction in (0.25, 0.5, 0.75):
            interior = lower + fraction * (upper - lower)
            np.testing.assert_array_equal(candidate.toward_zero(interior), positive[:-1])
            np.testing.assert_array_equal(candidate.toward_zero(-interior), positive[:-1] | 0x8000)
        tiny = np.asarray([np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)], dtype="<f8")
        np.testing.assert_array_equal(candidate.toward_zero(tiny), [0, 0x8000])

    def test_invalid_inputs_rejected(self):
        for value in (np.nan, np.inf, -np.inf, 65505.0, -65505.0):
            with self.assertRaises(ValueError):
                candidate.toward_zero(np.asarray([value], dtype="<f8"))
        with self.assertRaises(ValueError):
            candidate.toward_zero(np.asarray([1], dtype="<f4"))
        with self.assertRaises(ValueError):
            candidate.decoded(np.asarray([0x7c00], dtype="<u2"))
        with self.assertRaises(ValueError):
            next(candidate.stages({}, 3, {}, {}))

    def test_native_never_imports_reference_implementation(self):
        tree = ast.parse(Path(candidate.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("reference" in name or "oracle" in name for name in imports))
        self.assertFalse(any("local_reference" in getattr(node, "attr", "")
                             for node in ast.walk(tree)))

    def test_s16_local_gate_is_not_replaced_by_rtz_identity(self):
        operands = {name: np.full(4864, value, dtype="<f2").view("<u2")
                    for name, value in (("stage14", 1.0), ("stage15", 1.0))}
        reference = local.local_reference(16, operands, {}, 0)
        wrong = reference.copy()
        wrong[0] = np.asarray([2.0], dtype="<f2").view("<u2")[0]
        report = gates.evaluate_decoder_stage(stage=16, actual=wrong, reference=reference,
            policy=gates.POLICY_ID, local_reference=reference)
        self.assertEqual(report["status"], "FAIL")

    def test_global_original_witness_remains_rejecting(self):
        actual = np.asarray([0x616e], dtype="<u2")
        original = np.asarray([float.fromhex("0x1.5bc1b05c9106ap+9")], dtype="<f8")
        report = gates.evaluate_decoder_stage(stage=18, actual=actual, reference=actual,
            policy=gates.POLICY_ID, reference_binary64=original)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["binary64_v1"]["rows"][0]["excess_error"], "1/2")
        with self.assertRaises(ValueError):
            gates.evaluate_decoder_stage(stage=18, actual=actual, reference=actual,
                policy=gates.POLICY_ID, local_reference=actual, reference_binary64=original)


if __name__ == "__main__":
    unittest.main()
