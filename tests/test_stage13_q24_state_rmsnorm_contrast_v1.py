"""Independent rational arithmetic and synthetic actual-entry suffix checks."""

from fractions import Fraction as F
import inspect
import math
import unittest

import numpy as np

from ace3.model.candidates import stage13_q24_state_rmsnorm_contrast_v1 as d


previous_tests = d.retained.module_from(d.base.TEST, "stage13_fp16_synthetic_fixtures")
suffix_oracle = previous_tests.suffix_oracle


def verify_norm(integers, zeros, weights, actual, details):
    mean = round(F(sum(int(x) ** 2 for x in integers), 896) + 281474977)
    root = math.isqrt(mean)
    assert details == {"mean_q48": mean, "root_q24": root}, "Q24 RMSNorm mean/root"
    for i, (value, zero, weight) in enumerate(zip(integers, zeros, weights.view("<u2"), strict=True)):
        value = int(value)
        integer = round(F(value) * suffix_oracle.scalar(weight) * (1 << 24) / root)
        sign = (bool(value < 0) if value else bool(zero)) ^ bool(int(weight) & 0x8000)
        expected = suffix_oracle.round_word(F(integer, 1 << 24), sign)
        assert int(actual[i]) == expected, f"Q24 RMSNorm coordinate {i}"


class StateContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def test_q24_signed_zero_and_subnormal_oracle(self):
        integers = np.resize(np.array([0, 0, 1, -1, (1 << 24) + 17, -(1 << 24) - 19,
                                       65504 << 24, -(65504 << 24)], dtype="<i8"), 896)
        zeros = np.resize(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype="u1"), 896)
        gamma = np.resize(np.array([1, -1, 0.5, -0.5], dtype="<f2"), 896)
        words, details = d.stage13(integers, zeros, gamma)
        verify_norm(integers, zeros, gamma, words, details)
        words[62] ^= 1
        with self.assertRaisesRegex(AssertionError, "coordinate 62"):
            verify_norm(integers, zeros, gamma, words, details)

    def test_signed_zero_product(self):
        integers, zeros = np.zeros(896, dtype="<i8"), np.zeros(896, dtype="u1")
        zeros[1::2] = 1
        gamma = np.ones(896, dtype="<f2")
        gamma[2::4] = -1
        words, details = d.stage13(integers, zeros, gamma)
        verify_norm(integers, zeros, gamma, words, details)
        np.testing.assert_array_equal(words[:4], [0, 0x8000, 0x8000, 0x8000])

    def test_full_actual_entry_without_science_or_reference_selection(self):
        raw, tensors, operands, refs, classified = previous_tests.fixture()
        raw["input_i"][0] += 4096
        raw["scratch_i"][0] += 4096
        raw["output_i"][0] += 4096
        protected = {k: d.retained.digest(v) for k, v in raw.items()}
        for value in (*raw.values(), *tensors.values(), *operands):
            value.flags.writeable = False
        oracle = type("Oracle", (), {"verify_norm": staticmethod(verify_norm),
                                     "suffix_oracle": suffix_oracle})
        audit = d.base.new_audit()
        with d.guarded(d.OUTPUT, audit):
            payload, row = d.evaluate_control(
                d.CONTROLS[0], raw, {"control": d.CONTROLS[0], "failure_count": 0},
                classified, tensors, raw, np.ones(896, dtype="<f8"), refs, operands, oracle, audit)
        self.assertEqual(row["component_oracles"]["status"], "PASS")
        self.assertEqual([r["stage"] for r in row["stages"]], list(range(10, 19)))
        self.assertTrue(all(r["status"] == "PASS" for r in row["stages"]))
        self.assertTrue(all(r["direction_pass"] for r in row["paired_margins"]))
        self.assertEqual(audit["stage_dispatches"], [[d.CONTROLS[0], 23, 0, s] for s in range(13, 19)])
        self.assertEqual(audit["forbidden_events"], [])
        self.assertEqual(audit["writes"], [])
        self.assertEqual(protected, {k: d.retained.digest(payload[k]) for k in raw})
        self.assertEqual(row["stage13_integer_oracle"]["input_fields"][:2], ["scratch_i", "scratch_z"])
        np.testing.assert_array_equal(payload["q24_state_contrast_output_i"], raw["scratch_i"])

    def test_q24_information_changes_norm_without_changing_fp16_view(self):
        integers = np.full(896, 1 << 24, dtype="<i8")
        zeros = np.zeros(896, dtype="u1")
        gamma = np.ones(896, dtype="<f2")
        gamma[0] = 1.5
        integers[0] += 8191
        self.assertEqual(d.prior.state.fp16_view(int(integers[0]), 0), 0x3c00)
        actual, detail = d.stage13(integers, zeros, gamma)
        verify_norm(integers, zeros, gamma, actual, detail)
        rounded, rounded_detail = d.base.stage13(np.full(896, 0x3c00, dtype="<u2"), gamma)
        self.assertNotEqual(detail, rounded_detail)
        self.assertTrue(np.any(actual != rounded))

    def test_fp16_representable_state_preserves_old_arithmetic(self):
        words = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00], dtype="<u2"), 896)
        decoded = [d.retained.fp16_adaptation_oracle.decode_f16_q24(int(w)) for w in words]
        integers = np.array([v[0] for v in decoded], dtype="<i8")
        zeros = np.array([v[3] if v[0] == 0 else 0 for v in decoded], dtype="u1")
        gamma = np.ones(896, dtype="<f2")
        actual, detail = d.stage13(integers, zeros, gamma)
        expected, previous_detail = d.base.stage13(words, gamma)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(detail, previous_detail)

    def test_invalid_state_and_weight_rejected(self):
        for defect in ("shape", "dtype", "zero", "nonzero_tag", "range", "weight"):
            integers = np.ones(896, dtype="<i8")
            zeros = np.zeros(896, dtype="u1")
            gamma = np.ones(896, dtype="<f2")
            if defect == "shape":
                integers = integers[:-1]
            elif defect == "dtype":
                integers = integers.astype("<f8")
            elif defect == "zero":
                zeros[0] = 2
            elif defect == "nonzero_tag":
                zeros[0] = 1
            elif defect == "range":
                integers[0] = 1 << 50
            else:
                gamma[0] = np.nan
            with self.subTest(defect=defect), self.assertRaises((ValueError, OverflowError)):
                d.stage13(integers, zeros, gamma)

    def test_reference_free_signatures(self):
        self.assertEqual(list(inspect.signature(d.stage13).parameters), ["integers", "zeros", "gamma"])
        self.assertEqual(list(inspect.signature(d.compute_suffix).parameters), ["raw", "tensors", "operands"])
        text = inspect.getsource(d.compute_suffix)
        for forbidden in ("reference", "nearest", "compare(", "margin", "classified"):
            self.assertNotIn(forbidden, text)

    def test_protected_state_and_kv_gates(self):
        for key in ("stage12", "output_i", "output_cache_v", "stage03"):
            raw, tensors, operands, _, _ = previous_tests.fixture()
            raw[key].flat[0] ^= (1 << 14) if key == "output_i" else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.compute_suffix(raw, tensors, operands)

    def test_forbidden_dispatches_and_writes(self):
        for function, args in (
                (d.base.execute, ()), (d.prior.execute, ()), (d.retained.execute, ()),
                (d.retained.selection.nearest, (None,) * 3),
                (d.native.projection, ({}, d.PREFIX + "self_attn.o_proj", None)),
                (d.prior.audit_event, ("subprocess.Popen", ())),
                (d.prior.audit_event, ("open", (str(d.SOURCE), "w", 577)))):
            audit = d.base.new_audit()
            with d.guarded(d.OUTPUT, audit):
                with self.assertRaises(ValueError):
                    function(*args)
            self.assertTrue(audit["forbidden_events"])

    def test_exact_threshold_direction_and_nine_control_classification(self):
        previous = previous_tests.IntegerContrastTests()
        previous.test_exact_s18_threshold()
        previous.test_zero_opposite_and_correct_original_reference_movement()
        previous.test_numerical_oracle_or_direction_failure_is_rejected()
        with self.assertRaises(ValueError):
            d.base.scientific_status([])

    def test_distinct_source_test_output_and_owner(self):
        self.assertNotEqual(d.SOURCE, d.base.SOURCE)
        self.assertNotEqual(d.TEST, d.base.TEST)
        self.assertNotEqual(d.OUTPUT, d.base.OUTPUT)
        self.assertIn(d.MISSION, d.OUTPUT.name)
        self.assertNotIn("782b265df4c5", str(d.OUTPUT))
        self.assertNotIn("64399a56b844", str(d.OUTPUT))


if __name__ == "__main__":
    unittest.main()
