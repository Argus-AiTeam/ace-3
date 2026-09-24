import unittest
from fractions import Fraction

from trace_lineage_a_layer06_stage12_stage17_producers import (
    INNER_ORDERS, components, denominator_accounting, lifted_cases,
    exact_norm_round, norm_boundary_measurement, project_vectors, reference_add,
    reference_norm, residual_cases, units, values,
)
import numpy as np


class NormBoundaryTests(unittest.TestCase):
    def test_exact_squared_midpoints_signs_subnormals_binades_and_ties(self):
        for bits in (0, 1, 2, 0x3ff, 0x400, 0x3bff, 0x3c00, 0x3c01, 0x7bfd):
            lo, hi = Fraction(units(bits), 2**24), Fraction(units(bits + 1), 2**24)
            midpoint = (lo + hi) / 2
            expected = bits if bits % 2 == 0 else bits + 1
            self.assertEqual(exact_norm_round(midpoint, Fraction(1)), expected)
            self.assertEqual(exact_norm_round(-midpoint, Fraction(1)), expected | 0x8000)
            self.assertEqual(exact_norm_round(lo, Fraction(1)), bits)
            self.assertEqual(exact_norm_round(midpoint * 3, Fraction(9)), expected)
        self.assertEqual(exact_norm_round(Fraction(3), Fraction(4)), 0x3e00)
        for numerator, variance in ((Fraction(1), Fraction(0)), (Fraction(65505), Fraction(1))):
            with self.assertRaises(ValueError):
                exact_norm_round(numerator, variance)

    def test_primary_recurrence_and_global_reduction_are_retained(self):
        h, w = [0x3c00, 0x4000], [0x3c00, 0x3800]
        primary, disagreements, reduction = norm_boundary_measurement(h, w)
        self.assertEqual(primary, reference_norm(h, w))
        self.assertEqual(disagreements, [])
        self.assertEqual(reduction["sum_squares_q48"], 5 * 2**48)


class Layer06ProducerTests(unittest.TestCase):
    def test_native_gemm_all_lanes_no_zero_plus_one(self):
        tensors = {"qweight": [0x76543210] * 128, "qzeros": [0x11111111],
                   "scales": [0x3800] * 8}
        bits, totals = project_vectors([0x3c00] * 128, [0x3800] * 128, tensors)
        for channel in range(8):
            coefficient = ((channel % 2) * 4 + channel // 2 - 1) / 2
            self.assertEqual(totals["A"][channel], int(128 * coefficient * 2**48))
            self.assertEqual(units(bits["A"][channel]), int(128 * coefficient * 2**24))
            self.assertEqual(totals["R"][channel] * 2, totals["A"][channel])

    def test_group_scale_and_zero_change(self):
        tensors = {"qweight": [0x22222222] * 256,
                   "qzeros": [0x11111111, 0x33333333],
                   "scales": [0x3800] * 8 + [0x3400] * 8}
        bits, totals = project_vectors([0x3c00] * 256, [0x3c00] * 256, tensors)
        self.assertEqual(bits["A"], [0x5000] * 8)
        self.assertEqual(totals["A"], [32 * 2**48] * 8)

    def test_invalid_geometry_nonfinite_and_overflow_rejected(self):
        tensors = {"qweight": [0x11111111] * 128, "qzeros": [0],
                   "scales": [0x3c00] * 8}
        for actual in ([], [0] * 127, [0x7c00] * 128, [-1] * 128):
            with self.assertRaises(ValueError):
                project_vectors(actual, actual, tensors)
        enormous = dict(tensors, scales=[0x7bff] * 8)
        with self.assertRaises(ValueError):
            project_vectors([0x7bff] * 128, [0x7bff] * 128, enormous)

    def test_residual_rne_sign_and_cancellation(self):
        self.assertEqual(reference_add([0x3c00, 0x3c01, 0x8000, 1, 0xbc00],
                                       [0x1000, 0x1000, 0x8000, 1, 0x3c00]),
                         [0x3c00, 0x3c02, 0x8000, 2, 0])

    def test_nested_orders_full_denominator_and_local_remainders(self):
        stage12 = residual_cases([0x4000, 0x3c00], [0x3c00, 0x3c00],
                                 [0x3c00, 0], [0, 0],
                                 [0x4201, 0x3c00], [0x3c00, 0x3c00])
        stage17 = {"A": [0x4001, 0], "DA": [0x4000, 0],
                   "DR": [0x3c00, 0], "R": [0x3bff, 0]}
        hidden, chains = lifted_cases(
            stage12, stage17, reference_add(stage12["A"], stage17["A"]),
            reference_add(stage12["R"], stage17["R"]))
        self.assertEqual(len(chains), 4)
        self.assertNotEqual(components(stage12, INNER_ORDERS, 0)["incoming_first"]["local_stage12"], 0)
        weight = [0x3c00] * 2
        norms = {k: reference_norm(v, weight) for k, v in hidden.items()}
        reductions = {}
        for k, h in hidden.items():
            x = values(h)
            reductions[k] = {
                "inverse_rms_binary64_hex": float(1 / np.sqrt(np.mean(x*x) + 1e-6)).hex(),
                "sum_squares_q48": sum(units(b)**2 for b in h),
            }
        self.assertEqual(hidden["A"][1], hidden["R"][1])
        self.assertNotEqual(norms["A"][1], norms["R"][1])
        for i in range(2):
            split = components(norms, chains, i)
            self.assertEqual(len({sum(p.values()) for p in split.values()}), 1)
        accounting = denominator_accounting(hidden, norms, reductions, weight, chains, 1)
        self.assertTrue(any(Fraction(t["denominator_change"]) for parts in accounting.values()
                            for t in parts.values()))
        for parts in accounting.values():
            for term in parts.values():
                self.assertEqual(Fraction(term["numerator_change"]), 0)
                self.assertEqual(sum(Fraction(term[k]) for k in (
                    "numerator_change", "denominator_change",
                    "rounding_and_binary64_operation_remainder")), Fraction(term["rounded_total"]))
