"""Bounded entry/producer controls, provenance and unchanged suffix gates."""

import ast
from fractions import Fraction
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1 as d


EXPECTED_TESTS = 18


def fixture():
    parent = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
    actual = {key: np.full(896, 0x3c00, dtype="<u2") for key in ("stage11", "stage17")}
    original = {key: np.full(896, value, dtype="<f8")
                for key, value in (("input", 2), ("s11", 3), ("s17", 4))}
    return parent, actual, original


def rows(rescues=()):
    return [{"label": label, "index62": {
        "accepted": label in rescues or label == "mapped_all",
        "actual_fp16_bits": "662f" if label in rescues or label == "mapped_all" else "6630",
    }} for label, _ in d.plan()]


class EntryProducerTests(unittest.TestCase):
    def test_plan_has_complete_factorial_and_bounded_controls(self):
        plan = d.plan()
        self.assertEqual(len(plan), 11)
        self.assertEqual(len(set(label for label, _ in plan)), 11)
        self.assertEqual(plan[:8], d.producer.plan()[:8])
        self.assertEqual([label for label, _ in plan[8:]], ["inherited_native", "mapped62", "mapped_all"])

    def test_actual_s12_and_s18_exact_sums(self):
        parent, actual, original = fixture()
        *_, scratch, output = d.producer.frozen_parent(parent, actual, original, ())
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)
        self.assertEqual(output["h"][62], 0x4200)

    def test_inherited_cut_preserves_complete_coordinate_and_other_branches(self):
        parent, actual, original = fixture()
        incoming, o, down, _, output = d.producer.frozen_parent(parent, actual, original, ("inherited",))
        self.assertEqual(incoming["i"][62], 2 << 24)
        self.assertEqual(incoming["h"][62], 0x4000)
        np.testing.assert_array_equal(o, actual["stage11"])
        np.testing.assert_array_equal(down, actual["stage17"])
        self.assertEqual(output["i"][62], 4 << 24)
        self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))
        self.assertTrue(np.all(parent["i"] == 1 << 24))

    def test_o_cut_uses_fp16_and_retains_actual_down(self):
        parent, actual, original = fixture()
        _, o, down, scratch, output = d.producer.frozen_parent(parent, actual, original, ("o",))
        self.assertEqual(o.dtype, np.dtype("<u2"))
        self.assertEqual(o[62], 0x4200)
        np.testing.assert_array_equal(down, actual["stage17"])
        self.assertEqual(scratch["i"][62], 4 << 24)
        self.assertEqual(output["i"][62], 5 << 24)

    def test_down_cut_retains_actual_s12(self):
        parent, actual, original = fixture()
        _, _, down, scratch, output = d.producer.frozen_parent(parent, actual, original, ("down",))
        self.assertEqual(down[62], 0x4400)
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 6 << 24)

    def test_joint_cuts_close_exact_additive_interaction(self):
        parent, actual, original = fixture()
        values = {parts: int(d.producer.frozen_parent(parent, actual, original, parts)[-1]["i"][62])
                  for _, parts in d.plan()[:8]}
        for parts, value in values.items():
            self.assertEqual(value - values[()], sum(values[(p,)] - values[()] for p in parts))
        self.assertEqual(values[d.producer.BRANCHES], 9 << 24)

    def test_inconsistent_or_invalid_state_is_rejected(self):
        parent, actual, original = fixture()
        with self.assertRaises(ValueError):
            d.producer.frozen_parent(parent, actual, original, ("unknown",))
        parent["h"][62] = 0
        with self.assertRaises(ValueError):
            d.producer.compose(parent, actual["stage11"], actual["stage17"])

    def test_mapping_ties_even_signed_zero_and_half_grid(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(mapped["h"][:3], [0x8000, 0x8000, 0])
        self.assertEqual(Fraction(d.coordinate.compare_parents(mapped, mapped, values)[3]
                                 ["mapped_minus_original"]), Fraction(1, 1 << 25))
        values[0] = np.nan
        with self.assertRaises(ValueError):
            d.paired.mapped_parent(values)

    def test_unchanged_threshold_and_retained_coordinate_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(d.prior.measure(0x6630, reference)["excess_budget"]), Fraction(1, 8))

    def test_inherited_sufficiency_does_not_attribute_unique_producer(self):
        result = d.classify(rows(("frozen_inherited",)))
        self.assertEqual(result["classification"], "conditional_inherited_L8_branch_sufficiency")
        self.assertFalse(result["unique_upstream_producer_attributed"])
        self.assertFalse(result["inherited_native_rescue"])
        self.assertTrue(result["missing_evidence"])

    def test_local_sufficiency_is_only_conditional(self):
        for branch in ("o", "down"):
            result = d.classify(rows(("frozen_" + branch,)))
            self.assertEqual(result["classification"], "conditional_local_L9_branch_sufficiency")
            self.assertEqual(result["sufficient_frozen_single_branches"], [branch])
            self.assertFalse(result["unique_upstream_producer_attributed"])

    def test_multiple_joint_only_or_no_singleton_is_unresolved(self):
        for rescues in ((), ("frozen_o_down",), ("frozen_inherited", "frozen_down")):
            self.assertEqual(d.classify(rows(rescues))["classification"],
                             "unresolved_under_tested_branch_mappings")

    def test_reordered_incomplete_and_changed_endpoints_rejected(self):
        for invalid in (rows()[:-1], list(reversed(rows()))):
            with self.assertRaises(ValueError):
                d.classify(invalid)
        invalid = rows()
        invalid[0]["index62"]["accepted"] = True
        with self.assertRaises(ValueError):
            d.classify(invalid)
        invalid = rows()
        invalid[-1]["index62"]["accepted"] = False
        with self.assertRaises(ValueError):
            d.classify(invalid)

    def test_suffix_threads_only_native_l10_l13_despite_numerical_failures(self):
        parents = [object() for _ in range(5)]
        data = {"layers": {layer: {"tensors": layer, "trajectory": object(), "reference": object()}
                           for layer in range(10, 14)}}
        calls = []

        def execute(tensors, layer, incoming, trajectory, reference):
            offset = layer - 10
            self.assertEqual(tensors, layer)
            self.assertIs(incoming, parents[offset])
            self.assertIs(reference, data["layers"][layer]["reference"])
            self.assertIs(trajectory, data["layers"][layer]["trajectory"])
            calls.append(layer)
            return {"next": parents[offset + 1]}, {}, [{"status": "FAIL"}]

        with patch.object(d.upstream, "execute_layer", side_effect=execute), \
                patch.object(d.prior.retained, "state_from", side_effect=lambda a, *_: a["next"]):
            summaries = d.execute_suffix(parents[0], data, lambda layer, a, l, r: r[0]["status"])
        self.assertEqual(calls, [10, 11, 12, 13])
        self.assertEqual(summaries, ["FAIL"] * 4)

    def test_foreign_module_and_tampered_input_are_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/foreign", "bytes": 0, "sha256": "x"}, {})
        item = {"path": str(d.ROOT / "build/test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_only_exact_pinned_external_review_metadata_is_allowed(self):
        item = {"path": str(d.L8_REVIEW), "sha256": d.L8_REVIEW_SHA, "bytes": 123}
        review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
                  "mission_id": "dc98a86e7c02", "round": 3, "review": {"status": "done"}}
        with patch.object(d, "record", return_value=item), \
                patch.object(Path, "read_text", return_value=json.dumps(review)):
            self.assertEqual(d.check_review(item), review)
            with self.assertRaisesRegex(ValueError, "review binding mismatch"):
                d.check_review({**item, "path": "/tmp/review.json"})
            with self.assertRaisesRegex(ValueError, "review binding mismatch"):
                d.check_review({**item, "sha256": "changed"})

    def test_output_must_be_fresh_and_scope_bound(self):
        name = "q24_s16_l9_coordinate62_producer_cone_test"
        with patch.object(Path, "exists", return_value=False):
            for value in (d.ROOT / "build" / name, Path("build") / name):
                with self.subTest(accepted=value):
                    self.assertEqual(d.output_path(value), d.ROOT / "build" / name)
        for value in (
            d.ROOT / "build/other",
            d.ROOT / "build/q24_s16_l9_coordinate62_entry_producer_cone_test",
            d.ROOT / "build/q24_s16_l9_coordinate62_producer_cone",
            d.ROOT / "build/q24_s16_l9_coordinate62_producer_coneX_test",
            d.ROOT / name,
            d.ROOT / "build/nested" / name,
        ):
            with self.subTest(rejected=value), \
                    patch.object(d.sys, "argv", [d.MODULE, "--out", str(value)]), \
                    patch.object(Path, "mkdir") as mkdir, \
                    patch.object(d, "validate") as validate, \
                    patch.object(d, "diagnose") as diagnose:
                with self.assertRaisesRegex(ValueError, "outside bounded build scope"):
                    d.main()
                mkdir.assert_not_called()
                validate.assert_not_called()
                diagnose.assert_not_called()
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build" / name)

    def test_contract_and_no_external_execution_or_publication(self):
        contract = json.loads(d.CONTRACT.read_text())
        command = shlex.split(contract["command"])
        self.assertEqual(command[command.index("-m") + 1], d.MODULE)
        output = Path(command[command.index("--out") + 1])
        self.assertEqual(output.parent, Path("build"))
        self.assertTrue(output.name.startswith("q24_s16_l9_coordinate62_producer_cone_"))
        with patch.object(Path, "exists", return_value=False):
            self.assertEqual(d.output_path(d.ROOT / output), d.ROOT / output)
        self.assertEqual(contract["controls"], [label for label, _ in d.plan()])
        self.assertEqual(contract["reviewed_result_sha256"], d.REVIEWED_SHA)
        self.assertEqual(contract["coordinate_result_sha256"], d.producer.REVIEWED_SHA)
        self.assertEqual(contract["upstream_result_sha256"], d.coordinate.REVIEWED_SHA)
        self.assertEqual(contract["rtl_invocations"], 0)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(ast.parse(Path(d.__file__).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers",
                                 "continuation_stages", "authenticate_resume", "run_factory"})


if __name__ == "__main__":
    unittest.main()
