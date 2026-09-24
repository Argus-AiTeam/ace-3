"""Focused controls, exact arithmetic, and fail-closed diagnostic boundaries."""

import ast
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ace3.model.candidates import diagnose_q24_s16_l13_original_intermediate_drift_v1 as d


EXPECTED_TESTS = 10


def vector(value):
    return np.full(896, value, dtype="<f8")


class IntermediateTests(unittest.TestCase):
    def test_exact_signed_closure(self):
        original = {"input": vector(1), "s11": vector(2), "s17": vector(3),
                    "residual": vector(3), "s18": vector(6)}
        incoming = {"s18": vector(6.25)}
        row = d.decompose(1 << 24, 0x4000, 0x4200, 0x4680, original, incoming, 62)
        self.assertEqual(Fraction(row["total_signed_error"]), Fraction(1, 2))
        self.assertEqual(sum(map(Fraction, row["signed_parts"].values())), Fraction(1, 2))
        self.assertEqual(Fraction(row["same_binary64_operator_input_effect"]), Fraction(1, 4))
        self.assertEqual(Fraction(row["actual_minus_same_binary64_operator_on_Q24_parent"]), Fraction(1, 4))

    def test_binary64_roundoff_is_not_lost(self):
        original = {"input": vector(1), "s11": vector(2**-54), "s17": vector(2**-54),
                    "residual": vector(1), "s18": vector(1)}
        row = d.decompose(1 << 24, 0, 0, 0x3c00, original, {"s18": vector(1)}, 0)
        self.assertEqual(Fraction(row["total_signed_error"]), 0)
        self.assertEqual(Fraction(row["signed_parts"]["negative_original_binary64_addition_roundoff"]),
                         Fraction(1, 1 << 53))
        self.assertEqual(Fraction(row["original_residual_addition_roundoff"]), -Fraction(1, 1 << 54))

    def test_bitwise_reference_match(self):
        d.same_binary64(vector(1), vector(1))
        changed = vector(1)
        changed[62] = np.nextafter(1.0, 2.0)
        with self.assertRaises(ValueError):
            d.same_binary64(changed, vector(1))

    def test_reference_shape_dtype_finite_and_zero_sign(self):
        for invalid in (np.ones(896, dtype="<f4"), np.ones(895), vector(float("nan")), vector(-0.0)):
            with self.subTest(dtype=invalid.dtype), self.assertRaises(ValueError):
                d.same_binary64(invalid, vector(0))

    def test_bound_capture_observes_unchanged_returns(self):
        projections = {"o": object(), "down": object()}
        state = SimpleNamespace(projections=projections,
                                reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
                                reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
        namespace = {}

        def projection(activation, operand):
            return activation * (2 if operand is projections["o"] else 3)

        def layer(s, hidden, position):
            self.assertEqual(position, 0)
            output = namespace["_reference_projection"](hidden, s.projections["o"])
            residual = hidden + output
            down = namespace["_reference_projection"](residual, s.projections["down"])
            s.reference_k = s.reference_v = torch.zeros((1, 2, 64), dtype=torch.float64)
            return residual + down

        namespace.update(_reference_projection=projection, _reference_layer_step=layer)
        result = d.capture(namespace, state, vector(1))
        for name, expected in (("s11", 2), ("residual", 3), ("s17", 9), ("s18", 12)):
            np.testing.assert_array_equal(result[name], vector(expected))
        self.assertIs(namespace["_reference_projection"], projection)

    def test_capture_rejects_nonempty_cache(self):
        state = SimpleNamespace(reference_k=torch.empty((1, 2, 64)),
                                reference_v=torch.empty((1, 2, 64)))
        with self.assertRaises(ValueError):
            d.capture({"_reference_projection": object()}, state, vector(1))

    def test_capture_restores_observer_after_error(self):
        projection = object()

        def fail(*args):
            raise ValueError("missing source operand")

        namespace = {"_reference_projection": projection, "_reference_layer_step": fail}
        state = SimpleNamespace(reference_k=torch.empty((0, 2, 64)),
                                reference_v=torch.empty((0, 2, 64)))
        with self.assertRaisesRegex(ValueError, "missing source operand"):
            d.capture(namespace, state, vector(1))
        self.assertIs(namespace["_reference_projection"], projection)

    def test_binary64_control_cannot_establish_native_forcing(self):
        for accepted in (False, True):
            result = d.attribution({"accepted": False}, {"accepted": accepted})
            self.assertEqual(result["classification"], "inconclusive")
            self.assertEqual(result["incoming_drift_sufficient_in_binary64_then_RNE_control"], not accepted)
            self.assertEqual(len(result["missing_evidence"]), 2)
        with self.assertRaises(ValueError):
            d.attribution({"accepted": True}, {"accepted": False})

    def test_unchanged_exact_gate(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        row = d.prior.measure(0x6630, float.fromhex("0x1.8bd7b2092532cp+10"))
        self.assertFalse(row["accepted"])
        self.assertEqual(Fraction(row["excess_error"]), Fraction(142671047477, 549755813888))

    def test_no_execution_or_successor_publication(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["rtl_invocations"], 0)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "execute_layers", "_stages",
                                 "continuation_stages", "run_factory", "publish_parent"})


if __name__ == "__main__":
    unittest.main()
