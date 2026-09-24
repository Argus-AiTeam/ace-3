"""Synthetic L10 cuts and guarded suffix dispatch; no accepted native replay."""

import ast
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l10_coordinate62_producer_cone_v1 as d


EXPECTED_TESTS = 14


def fixture():
    parent = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
    actual = {key: np.full(896, 0x3c00, dtype="<u2") for key in ("stage11", "stage17")}
    scratch, output = d.producer.compose(parent, actual["stage11"], actual["stage17"])
    original = {key: np.full(896, value, dtype="<f8") for key, value in
                (("input", 2), ("s11", 3), ("s17", 4), ("residual", 5), ("s18", 9))}
    return {"parent": parent, "arrays": actual}, original, scratch, output


def rows(rescues=()):
    return [{"label": label, "index62": {
        "accepted": label in rescues, "actual_fp16_bits": "662f" if label in rescues else "6630",
    }} for label, _ in d.plan()]


class L10ProducerTests(unittest.TestCase):
    def test_complete_deterministic_controls(self):
        self.assertEqual(d.plan(), d.producer.plan())
        self.assertEqual(len(d.plan()), 13)
        self.assertEqual(len(dict(d.plan())), 13)
        self.assertIn("S12", d.BRANCH_DEFINITIONS)
        self.assertIn("S18", d.BRANCH_DEFINITIONS)

    def test_exact_residual_and_factorial_closure(self):
        layer, original, scratch, output = fixture()
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)
        values = {parts: int(d.producer.frozen_parent(
            layer["parent"], layer["arrays"], original, parts)[-1]["i"][62])
                  for _, parts in d.plan()[:8]}
        for parts, value in values.items():
            self.assertEqual(value - values[()], sum(values[(p,)] - values[()] for p in parts))

    def test_inherited_o_down_cuts_are_scalar_and_nonmutating(self):
        layer, original, _, _ = fixture()
        for branch, expected in (("inherited", 4), ("o", 5), ("down", 6)):
            incoming, o, down, _, output = d.producer.frozen_parent(
                layer["parent"], layer["arrays"], original, (branch,))
            self.assertEqual(output["i"][62], expected << 24)
            self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))
            if branch != "o":
                np.testing.assert_array_equal(o, layer["arrays"]["stage11"])
            if branch != "down":
                np.testing.assert_array_equal(down, layer["arrays"]["stage17"])
            if branch == "inherited":
                self.assertEqual(incoming["h"][62], 0x4000)
        self.assertTrue(np.all(layer["parent"]["i"] == 1 << 24))

    def test_scratch_and_scratch_down_keep_unselected_coordinates(self):
        layer, original, scratch, baseline = fixture()
        def state_from(arrays, prefix, stage):
            return scratch if prefix == "scratch" else baseline
        with patch.object(d.prior.retained, "state_from", side_effect=state_from):
            for label, expected in (("scratch", 6), ("scratch_down", 9)):
                operands, output = d.cut_parent(layer, original, label)
                self.assertEqual(operands["scratch_i"][62], 5 << 24)
                self.assertEqual(output["i"][62], expected << 24)
                self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))

    def test_mapped_output_controls_have_declared_width(self):
        layer, original, _, baseline = fixture()
        with patch.object(d.prior.retained, "state_from", return_value=baseline):
            _, scalar = d.cut_parent(layer, original, "mapped62")
            _, full = d.cut_parent(layer, original, "mapped_all")
        self.assertEqual(scalar["i"][62], 9 << 24)
        self.assertTrue(np.all(scalar["i"][np.arange(896) != 62] == 3 << 24))
        self.assertTrue(np.all(full["i"] == 9 << 24))

    def test_invalid_branch_and_inconsistent_state_fail(self):
        layer, original, _, _ = fixture()
        with self.assertRaises(ValueError):
            d.cut_parent(layer, original, "unknown")
        layer["parent"]["h"][62] = 0
        with self.assertRaises(ValueError):
            d.producer.compose(layer["parent"], layer["arrays"]["stage11"], layer["arrays"]["stage17"])

    def test_mapping_ties_even_and_signed_zero(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        self.assertEqual(Fraction(d.coordinate.compare_parents(mapped, mapped, values)[3]
                                 ["mapped_minus_original"]), Fraction(1, 1 << 25))

    def test_unchanged_threshold_and_retained_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(d.prior.measure(0x6630, reference)["excess_budget"]), Fraction(1, 8))

    def test_guard_blocks_all_earlier_layers_before_execution(self):
        dispatch = []
        with patch.object(d.upstream, "execute_layer") as execute:
            for layer in list(range(10)) + [14, True, 10.0]:
                with self.assertRaisesRegex(ValueError, "outside L10-L13"):
                    d.execute_layer(layer, {}, {}, dispatch)
            execute.assert_not_called()
        self.assertEqual(dispatch, [])

    def test_suffix_preserves_reference_and_parent_even_when_gates_fail(self):
        parents = [object() for _ in range(4)]
        data = {"layers": {layer: {"tensors": layer, "trajectory": object(), "reference": object()}
                           for layer in range(11, 14)}}
        def execute(tensors, layer, incoming, trajectory, reference):
            self.assertEqual(tensors, layer)
            self.assertIs(incoming, parents[layer - 11])
            self.assertIs(trajectory, data["layers"][layer]["trajectory"])
            self.assertIs(reference, data["layers"][layer]["reference"])
            return {"next": parents[layer - 10]}, {}, [{"status": "FAIL"}]
        dispatch = []
        with patch.object(d.upstream, "execute_layer", side_effect=execute), \
                patch.object(d.prior.retained, "state_from", side_effect=lambda a, *_: a["next"]):
            result = d.execute_suffix(parents[0], data, lambda l, a, v, r: r[0]["status"], dispatch)
        self.assertEqual(dispatch, [11, 12, 13])
        self.assertEqual(result, ["FAIL"] * 3)

    def test_conditional_classification_never_unique(self):
        for branch, expected in (("inherited", "conditional_inherited_L9_branch_sufficiency"),
                                 ("o", "conditional_local_L10_branch_sufficiency"),
                                 ("down", "conditional_local_L10_branch_sufficiency")):
            result = d.classify(rows(("frozen_" + branch,)))
            self.assertEqual(result["classification"], expected)
            self.assertFalse(result["unique_upstream_producer_attributed"])
            self.assertTrue(result["missing_evidence"])
        for rescues in ((), ("frozen_o_down",), ("frozen_inherited", "frozen_down")):
            self.assertEqual(d.classify(rows(rescues))["classification"],
                             "unresolved_under_tested_branch_mappings")
        for invalid in (rows()[:-1], list(reversed(rows())), rows(("actual",))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_foreign_origin_and_tampered_binding_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        item = {"path": str(d.ROOT / "build/test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_exclusive_scoped_output(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/other")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l10_coordinate62_producer_cone_test")

    def test_only_guarded_native_delegate_and_no_external_execution(self):
        tree = ast.parse(Path(d.__file__).read_text())
        delegates = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                     and n.func.value.id == "upstream" and n.func.attr == "execute_layer"]
        self.assertEqual(len(delegates), 1)
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute_layer")
        self.assertIn(delegates[0], list(ast.walk(guard)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name in ("subprocess", "runpy") for n in node.names))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("system", "popen", "run_layer", "execute_l8"))


if __name__ == "__main__":
    unittest.main()
