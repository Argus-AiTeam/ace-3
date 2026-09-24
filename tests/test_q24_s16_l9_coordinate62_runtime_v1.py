"""Focused runtime tests: synthetic arithmetic, no native accepted-layer replay."""

import ast
import copy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_runtime_v1 as d


EXPECTED_TESTS = 16


def fixture():
    parent = d.entry.paired.mapped_parent(np.ones(896, dtype="<f8"))
    arrays = {key: np.full(896, 0x3c00, dtype="<u2") for key in ("stage11", "stage17")}
    original = {key: np.full(896, value, dtype="<f8")
                for key, value in (("input", 2), ("s11", 3), ("s17", 4), ("s18", 9))}
    return parent, arrays, original


class RuntimeTests(unittest.TestCase):
    def test_every_accepted_native_layer_rejected_before_dispatch(self):
        with patch.object(d.entry.upstream, "execute_layer") as dispatch:
            for layer in range(9):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L0-L8"):
                    d.execute_layer(layer, 0, None, {})
            dispatch.assert_not_called()

    def test_other_layers_types_and_positions_rejected(self):
        with patch.object(d.entry.upstream, "execute_layer") as dispatch:
            for layer, position in ((-1, 0), (10, 0), (23, 0), (9.0, 0), (True, 0),
                                    (9, 1), (9, -1), (9, False), (9, 0.0)):
                with self.subTest(layer=layer, position=position), self.assertRaises(ValueError):
                    d.execute_layer(layer, position, None, {})
            dispatch.assert_not_called()

    def test_native_dispatch_keeps_actual_parent_and_original_references(self):
        item = {key: object() for key in ("tensors", "trajectory", "reference")}
        parent, result = object(), object()
        with patch.object(d.entry.upstream, "execute_layer", return_value=result) as dispatch:
            self.assertIs(d.execute_layer(9, 0, parent, {"layers": {9: item}}), result)
        dispatch.assert_called_once_with(
            item["tensors"], 9, parent, item["trajectory"], item["reference"])

    def test_native_errors_are_not_converted_to_success(self):
        item = dict(tensors=None, trajectory=None, reference=None)
        with patch.object(d.entry.upstream, "execute_layer", side_effect=ValueError("KV mismatch")):
            with self.assertRaisesRegex(ValueError, "KV mismatch"):
                d.execute_layer(9, 0, None, {"layers": {9: item}})

    def test_factorial_keeps_all_eight_controls_and_exact_deltas(self):
        rows = d.frozen_controls(*fixture())
        self.assertEqual([r["label"] for r in rows], [label for label, _ in d.entry.plan()[:8]])
        self.assertEqual(rows[0]["S12_Q24_units62"], 2 << 24)
        self.assertEqual(rows[0]["S18_Q24_units62"], 3 << 24)
        self.assertEqual(rows[-1]["S18_Q24_units62"], 9 << 24)
        for row in rows:
            self.assertIs(row["candidate_admitted"], False)
            self.assertIs(row["complete_native_operator_gates_evaluated"], False)

    def test_frozen_controls_never_dispatch_native_layers(self):
        with patch.object(d.entry.upstream, "execute_layer") as dispatch:
            d.frozen_controls(*fixture())
            dispatch.assert_not_called()

    def test_frozen_controls_do_not_mutate_evidence(self):
        values = fixture()
        expected = copy.deepcopy(values)
        d.frozen_controls(*values)
        for actual, before in zip(values, expected):
            for key in actual:
                np.testing.assert_array_equal(actual[key], before[key])

    def test_inconsistent_q24_projection_is_rejected(self):
        parent, arrays, original = fixture()
        parent["h"][62] = 0
        with self.assertRaises(ValueError):
            d.frozen_controls(parent, arrays, original)

    def test_exact_excess_threshold_and_historical_failure_preserved(self):
        measure = d.entry.prior.measure
        self.assertTrue(measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        failure = measure(0x6630, reference)
        self.assertFalse(failure["accepted"])
        self.assertEqual(Fraction(failure["excess_budget"]), Fraction(1, 8))
        self.assertTrue(measure(0x662f, reference)["accepted"])

    def test_source_context_binds_runtime_and_import_only_accepted_tests(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.ORIGIN_ONLY_TEST):
            self.assertEqual(Path(origins[name]["path"]),
                             d.ROOT.joinpath(*name.split(".")).with_suffix(".py"))

    def test_foreign_repository_module_is_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_implicit_pythonpath_is_rejected(self):
        with patch.dict(d.entry.os.environ, {"PYTHONPATH": ""}):
            with self.assertRaisesRegex(ValueError, "repository-bound"):
                d.source_context()

    def test_changed_bound_input_rejected_before_review_check(self):
        item = {"path": str(d.ROOT / "build/input"), "bytes": 1, "sha256": "old"}
        with patch.object(d.entry, "record", return_value={**item, "sha256": "changed"}), \
                patch.object(d.entry, "check_review") as review:
            with self.assertRaisesRegex(ValueError, "authenticated input changed"):
                d.reauthenticate({"bound": {item["path"]: item}})
            review.assert_not_called()

    def test_unchanged_inputs_recheck_pinned_independent_review(self):
        item = {"path": str(d.ROOT / "build/input"), "bytes": 1, "sha256": "unchanged"}
        binding = object()
        with patch.object(d.entry, "record", return_value=item), \
                patch.object(d.entry, "check_review") as review:
            d.reauthenticate({"bound": {item["path"]: item}, "L8_review_binding": binding})
            review.assert_called_once_with(binding)

    def test_existing_and_out_of_scope_outputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside bounded"):
            d.output_path(d.ROOT / "build/other")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l9_coordinate62_runtime_test")

    def test_runtime_only_dispatches_guarded_l9_and_never_runs_accepted_suite(self):
        tree = ast.parse(Path(d.__file__).read_text())
        diagnose = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "diagnose")
        threads = [n for n in ast.walk(diagnose) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "set_num_threads"]
        self.assertEqual(len(threads), 1)
        self.assertEqual(ast.literal_eval(threads[0].args[0]), 1)
        dispatches = [n for n in ast.walk(diagnose) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "execute_layer"]
        self.assertEqual(len(dispatches), 2)
        self.assertEqual([(ast.literal_eval(n.args[0]), ast.literal_eval(n.args[1]))
                          for n in dispatches], [(9, 0), (9, 0)])
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {"stages", "continuation_stages", "execute_suffix",
                                 "Popen", "system", "publish_parent", "authenticate_resume"})
        self.assertEqual(d.TEST_MODULE, "tests.test_q24_s16_l9_coordinate62_runtime_v1")


if __name__ == "__main__":
    unittest.main()
