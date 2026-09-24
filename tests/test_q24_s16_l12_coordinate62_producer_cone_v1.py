"""Producer cuts, exact residuals, unchanged gates and bounded provenance."""

import ast
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v1 as d


EXPECTED_TESTS = 16


def fixture():
    parent = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
    actual = {key: np.full(896, 0x3c00, dtype="<u2") for key in ("stage11", "stage17")}
    original = {key: np.full(896, value, dtype="<f8")
                for key, value in (("input", 2), ("s11", 3), ("s17", 4))}
    return parent, actual, original


def rows(rescues=()):
    return [{"label": label, "index62": {
        "accepted": label in rescues or label in ("mapped62", "mapped_all"),
        "actual_fp16_bits": "662f" if label in rescues or label in ("mapped62", "mapped_all") else "6630",
    }} for label, _ in d.plan()]


class ProducerConeTests(unittest.TestCase):
    def test_plan_covers_full_factorial_and_named_controls(self):
        plan = d.plan()
        self.assertEqual(len(plan), 13)
        self.assertEqual(len(set(label for label, _ in plan)), 13)
        self.assertEqual({frozenset(parts) for _, parts in plan[:8]},
                         {frozenset(p) for size in range(4)
                          for p in d.itertools.combinations(d.BRANCHES, size)})

    def test_baseline_exact_residual_composition(self):
        parent, actual, original = fixture()
        _, _, _, scratch, output = d.frozen_parent(parent, actual, original, ())
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)
        self.assertEqual(output["h"][62], 0x4200)

    def test_inherited_cut_changes_only_selected_complete_coordinate(self):
        parent, actual, original = fixture()
        incoming, _, _, _, output = d.frozen_parent(parent, actual, original, ("inherited",))
        self.assertEqual(incoming["i"][62], 2 << 24)
        self.assertEqual(incoming["h"][62], 0x4000)
        self.assertEqual(output["i"][62], 4 << 24)
        self.assertEqual(parent["i"][62], 1 << 24)
        self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))

    def test_o_cut_keeps_fp16_boundary_and_actual_down(self):
        parent, actual, original = fixture()
        _, o, down, scratch, output = d.frozen_parent(parent, actual, original, ("o",))
        self.assertEqual(o.dtype, np.dtype("<u2"))
        self.assertEqual(o[62], 0x4200)
        np.testing.assert_array_equal(down, actual["stage17"])
        self.assertEqual(scratch["i"][62], 4 << 24)
        self.assertEqual(output["i"][62], 5 << 24)

    def test_down_cut_keeps_scratch(self):
        parent, actual, original = fixture()
        _, _, down, scratch, output = d.frozen_parent(parent, actual, original, ("down",))
        self.assertEqual(down[62], 0x4400)
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 6 << 24)

    def test_joint_cut_is_exact_additive_and_non_mutating(self):
        parent, actual, original = fixture()
        outputs = {parts: d.frozen_parent(parent, actual, original, parts)[-1]["i"][62]
                   for _, parts in d.plan()[:8]}
        self.assertEqual(outputs[d.BRANCHES], 9 << 24)
        self.assertEqual(outputs[d.BRANCHES] - outputs[()],
                         sum(outputs[(b,)] - outputs[()] for b in d.BRANCHES))
        self.assertTrue(np.all(actual["stage11"] == 0x3c00))
        self.assertTrue(np.all(parent["i"] == 1 << 24))

    def test_invalid_frozen_cut_and_inconsistent_state_rejected(self):
        parent, actual, original = fixture()
        with self.assertRaises(ValueError):
            d.frozen_parent(parent, actual, original, ("scratch",))
        parent["h"][62] = 0
        with self.assertRaises(ValueError):
            d.compose(parent, actual["stage11"], actual["stage17"])

    def test_signed_zero_and_nearest_q24_ties_even(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(mapped["h"][:3], [0x8000, 0x8000, 0])

    def test_unchanged_exact_threshold_and_historical_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        failed, rescued = d.prior.measure(0x6630, reference), d.prior.measure(0x662f, reference)
        self.assertFalse(failed["accepted"])
        self.assertTrue(rescued["accepted"])
        self.assertEqual(Fraction(failed["excess_budget"]), Fraction(1, 8))

    def test_conditional_branch_is_not_unique_producer(self):
        result = d.classify(rows(("frozen_down",)))
        self.assertEqual(result["classification"], "conditional_single_branch_sufficiency")
        self.assertEqual(result["sufficient_frozen_single_branches"], ["down"])
        self.assertFalse(result["unique_upstream_producer_attributed"])
        self.assertTrue(result["missing_evidence"])

    def test_multiple_or_joint_only_rescues_are_unresolved(self):
        for rescues in ((), ("frozen_o", "frozen_down"), ("frozen_o_down",)):
            self.assertEqual(d.classify(rows(rescues))["classification"], "unresolved_producer_interaction")

    def test_incomplete_or_changed_endpoints_rejected(self):
        for invalid in (rows()[:-1], list(reversed(rows()))):
            with self.assertRaises(ValueError):
                d.classify(invalid)
        invalid = rows()
        invalid[0]["index62"]["accepted"] = True
        with self.assertRaises(ValueError):
            d.classify(invalid)

    def test_foreign_origin_and_tampered_binding_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {})
        item = {"path": str(d.ROOT / "build/producer-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_suffix_only_uses_native_layer13(self):
        data = {"tensors": {}, "trajectory": {}, "reference": np.zeros(896)}
        with patch.object(d.upstream, "execute_layer", return_value="sentinel") as execute:
            self.assertEqual(d.coordinate.execute({}, data), "sentinel")
            self.assertEqual(execute.call_args.args[1], 13)

    def test_output_is_exclusive_and_scope_bound(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/wrong-prefix")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l12_coordinate62_producer_cone_test")

    def test_contract_no_external_execution_or_publication(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["controls"], [label for label, _ in d.plan()])
        self.assertEqual(contract["reviewed_result_sha256"], d.REVIEWED_SHA)
        self.assertEqual(contract["reviewed_upstream_sha256"], d.coordinate.REVIEWED_SHA)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["rtl_invocations"], 0)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(ast.parse(Path(d.__file__).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers",
                                 "continuation_stages", "run_factory"})


if __name__ == "__main__":
    unittest.main()
