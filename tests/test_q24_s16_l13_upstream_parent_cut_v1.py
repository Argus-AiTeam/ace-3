"""Focused parent-cut semantics and fail-closed source/output boundaries."""

import ast
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_upstream_parent_cut_v1 as d


EXPECTED_TESTS = 14


def rows(flips=()):
    return [{"cut_after_layer": cut,
             "actual_index62": {"actual_fp16_bits": "6630", "accepted": False},
             "mapped_index62": {"actual_fp16_bits": "662f" if cut in flips else "6630",
                                "accepted": cut in flips}}
            for cut in d.CUTS]


class ParentCutTests(unittest.TestCase):
    def test_first_cut_is_ascending_not_nearest_target(self):
        result = d.classify(rows((10, 12)))
        self.assertEqual(result["first_flipping_cut_after_layer"], 10)
        self.assertEqual(result["flipping_cuts_after_layer"], [10, 12])
        self.assertFalse(result["unique_upstream_producer_attributed"])

    def test_no_flip_stays_inconclusive_about_producer(self):
        result = d.classify(rows())
        self.assertIsNone(result["first_flipping_cut_after_layer"])
        self.assertEqual(result["classification"], "no_6630_to_662f_flip_in_tested_cuts")
        self.assertTrue(result["missing_evidence"])

    def test_missing_or_reordered_cuts_rejected(self):
        for invalid in (rows()[:-1], list(reversed(rows()))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_changed_baseline_rejected(self):
        controls = rows()
        controls[0]["actual_index62"]["accepted"] = True
        with self.assertRaises(ValueError):
            d.classify(controls)

    def test_other_word_or_rejected_coordinate_is_not_rescue(self):
        controls = rows((9, 10))
        controls[0]["mapped_index62"]["actual_fp16_bits"] = "662e"
        controls[1]["mapped_index62"]["accepted"] = False
        self.assertEqual(d.classify(controls)["flipping_cuts_after_layer"], [])

    def test_mapping_ties_even_signed_zero_and_half_grid(self):
        values = np.zeros(896, dtype="<f8")
        values[:5] = [2**-25, 3 * 2**-25, -2**-25, -0.0, 1.0]
        parent = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(parent["i"][:5], [0, 2, 0, 0, 1 << 24])
        np.testing.assert_array_equal(parent["z"][:5], [0, 0, 1, 1, 0])
        for integer, value in zip(parent["i"], values, strict=True):
            self.assertLessEqual(abs(Fraction(int(integer), 1 << 24)
                                     - Fraction.from_float(float(value))), Fraction(1, 1 << 25))

    def test_mapping_rejects_invalid_vectors(self):
        for value in (np.zeros(895), np.zeros(896, dtype="<f4"),
                      np.full(896, np.nan), np.full(896, 65505.0)):
            with self.assertRaises(ValueError):
                d.paired.mapped_parent(value)

    def test_execution_cannot_expand_layer_scope(self):
        with patch.object(d.native, "stages") as stages:
            for layer in (8, 14, True):
                with self.assertRaises(ValueError):
                    d.execute_layer({}, layer, {}, {}, np.zeros(896))
            stages.assert_not_called()

    def test_binding_rejects_external_source(self):
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.bind_input({"path": "/tmp/not-repository.py", "bytes": 0, "sha256": "bad"}, {})

    def test_binding_rejects_changed_and_conflicting_evidence(self):
        item = {"path": str(d.ROOT / "build/parent-cut-test-input"), "bytes": 1, "sha256": "a"}
        changed = {**item, "sha256": "b"}
        with patch.object(d, "record", return_value=changed):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.bind_input(item, {})
        with self.assertRaisesRegex(ValueError, "conflicting binding"):
            d.bind_input(item, {item["path"]: changed})

    def test_module_origin_rejects_other_checkout(self):
        foreign = SimpleNamespace(__file__="/tmp/ace3/model/foreign.py")
        with patch.dict(d.sys.modules, {"ace3.model.foreign": foreign}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()

    def test_output_is_exclusive_and_scope_bound(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/wrong-prefix")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l13_upstream_parent_cut_test")

    def test_unchanged_exact_threshold_and_index62(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])

    def test_contract_and_no_external_execution_or_publication(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["cuts_after_layer"], list(d.CUTS))
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertEqual(contract["rtl_invocations"], 0)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "execute_layers", "publish_parent",
                                 "continuation_stages", "run_factory"})


if __name__ == "__main__":
    unittest.main()
