"""Focused v2 arithmetic, evidence, lineage and check-only boundary tests."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l12_coordinate62_producer_cone_v2 as d


EXPECTED_TESTS = 26


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


class ProducerConeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())

    def test_full_factorial_plan_unchanged(self):
        self.assertIs(d.plan, d.legacy.plan)
        self.assertEqual(len(d.plan()), 13)
        self.assertEqual({frozenset(parts) for _, parts in d.plan()[:8]},
                         {frozenset(parts) for size in range(4)
                          for parts in d.legacy.itertools.combinations(d.legacy.BRANCHES, size)})

    def test_baseline_exact_residual_composition(self):
        scratch, output = d.frozen_parent(*fixture(), ())[-2:]
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)
        self.assertEqual(output["h"][62], 0x4200)

    def test_inherited_cut_only_complete_coordinate(self):
        parent, actual, original = fixture()
        incoming, _, _, _, output = d.frozen_parent(parent, actual, original, ("inherited",))
        self.assertEqual(incoming["i"][62], 2 << 24)
        self.assertEqual(incoming["h"][62], 0x4000)
        self.assertEqual(output["i"][62], 4 << 24)
        np.testing.assert_array_equal(incoming["i"][np.arange(896) != 62],
                                      parent["i"][np.arange(896) != 62])

    def test_o_cut_retains_fp16_and_actual_down(self):
        parent, actual, original = fixture()
        _, o, down, scratch, output = d.frozen_parent(parent, actual, original, ("o",))
        self.assertEqual(o.dtype, np.dtype("<u2"))
        self.assertEqual(o[62], 0x4200)
        np.testing.assert_array_equal(down, actual["stage17"])
        self.assertEqual(scratch["i"][62], 4 << 24)
        self.assertEqual(output["i"][62], 5 << 24)

    def test_down_cut_retains_scratch(self):
        _, _, down, scratch, output = d.frozen_parent(*fixture(), ("down",))
        self.assertEqual(down[62], 0x4400)
        self.assertEqual(scratch["i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 6 << 24)

    def test_joint_additivity_and_no_input_mutation(self):
        parent, actual, original = fixture()
        outputs = {parts: d.frozen_parent(parent, actual, original, parts)[-1]["i"][62]
                   for _, parts in d.plan()[:8]}
        self.assertEqual(outputs[d.legacy.BRANCHES], 9 << 24)
        self.assertEqual(outputs[d.legacy.BRANCHES] - outputs[()],
                         sum(outputs[(branch,)] - outputs[()] for branch in d.legacy.BRANCHES))
        self.assertTrue(np.all(parent["i"] == 1 << 24))
        self.assertTrue(np.all(actual["stage11"] == 0x3c00))

    def test_invalid_frozen_control_rejected(self):
        with self.assertRaises(ValueError):
            d.frozen_parent(*fixture(), ("scratch",))

    def test_inconsistent_state_rejected(self):
        parent, actual, _ = fixture()
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

    def test_exact_threshold_and_historical_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        failed = d.prior.measure(0x6630, reference)
        self.assertFalse(failed["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(failed["excess_budget"]), Fraction(1, 8))

    def test_single_rescue_is_not_unique_producer(self):
        result = d.classify(rows(("frozen_down",)))
        self.assertEqual(result["classification"], "conditional_single_branch_sufficiency")
        self.assertFalse(result["unique_upstream_producer_attributed"])
        self.assertTrue(result["missing_evidence"])

    def test_joint_and_multiple_rescues_unresolved(self):
        for rescues in ((), ("frozen_o", "frozen_down"), ("frozen_o_down",)):
            self.assertEqual(d.classify(rows(rescues))["classification"], "unresolved_producer_interaction")

    def test_incomplete_or_reordered_controls_rejected(self):
        for invalid in (rows()[:-1], list(reversed(rows()))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_changed_baseline_rejected(self):
        invalid = rows()
        invalid[0]["index62"]["accepted"] = True
        with self.assertRaises(ValueError):
            d.classify(invalid)

    def test_contract_matches_pinned_reviewed_v3(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["reviewed_result_sha256"],
                         "4073e5da85af77e0770a399e724c64f8326abb1a636f21554f4966b50b7299a9")
        self.assertEqual(self.contract["reviewed_task"], "1d567e59cfac")
        self.assertEqual(self.contract["controls"], [label for label, _ in d.plan()])

    def test_contract_rejects_admission_and_dispatch(self):
        for key, value in (("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True), ("rtl_invocations", 1),
                           ("native_layer_invocations", 1), ("normal_host_review", "OPTIONAL"),
                           ("interface", ["--execute"]), ("scientific_result_claim", True)):
            changed = deepcopy(self.contract)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_threshold_and_mapping_changes_rejected(self):
        for key in ("thresholds", "mapping", "branch_definitions", "reviewed_result_sha256",
                    "reviewed_upstream_sha256", "source_updates", "retained_freeze_sha256"):
            changed = deepcopy(self.contract)
            changed[key] = "substituted"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_history_is_exact_and_read_only(self):
        self.assertEqual({str(Path(item["path"]).relative_to(d.ROOT)): item["sha256"]
                          for item in d.history_records()}, d.HISTORY)
        self.assertIs(d.original_branches, d.legacy.original_branches)
        self.assertIs(d.frozen_parent, d.legacy.frozen_parent)

    def test_foreign_module_origin_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_foreign_and_tampered_bindings_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {})
        item = {"path": str(d.ROOT / "build/producer-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_cli_rejects_native_execution_and_output(self):
        for arguments in ([], ["--execute"], ["--check", "--out", "build/not-authorized"]):
            with patch.object(d.sys, "argv", [d.MODULE, *arguments]), \
                    patch.object(d.sys, "stderr", io.StringIO()), \
                    patch.object(d, "validate") as validate:
                with self.assertRaises(SystemExit) as error:
                    d.main()
                self.assertEqual(error.exception.code, 2)
                validate.assert_not_called()

    def test_surface_has_no_native_dispatch_calls(self):
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(ast.parse(Path(d.__file__).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers",
                                 "execute_layer", "stages", "diagnose", "original_branches"})

    def test_reanchored_reference_rejected(self):
        extension = {"reference_only": True, "policy_id": d.prior.gates.POLICY_ID,
                     "global_reference_policy": "actual-parent-reference"}
        with self.assertRaisesRegex(ValueError, "reference substituted"):
            d.coordinate.check_reference_chain(extension)

    def test_spliced_reference_chain_rejected(self):
        extension = {
            "reference_only": True, "policy_id": d.prior.gates.POLICY_ID,
            "global_reference_policy": "legacy-binary64-AWQ-fully-independent-propagation",
            "original_binary64_parent": "original64", "original_fp16_parent": "original16",
            "layers": {"9": {"input_binary64": "substituted", "input_fp16": "original16",
                             "prior_kv": "own empty P0"}},
        }
        with self.assertRaisesRegex(ValueError, "spliced original-input reference"):
            d.coordinate.check_reference_chain(extension)

    def test_wrong_parent_receipt_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, "parent receipt lineage mismatch"):
            d.coordinate.check_parent_receipt({"next_layer": 14}, {}, {}, {}, {})

    def test_wrong_reviewed_result_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed v3 coordinate scope mismatch"):
            d.check_reviewed({"diagnostic_id": d.legacy.coordinate.ID})


if __name__ == "__main__":
    unittest.main()
