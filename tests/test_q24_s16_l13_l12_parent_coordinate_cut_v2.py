"""Unchanged coordinate semantics and check-only versioned authentication."""

import ast
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v2 as d


EXPECTED_TESTS = 26


def controls(rescues=()):
    return [{"label": label, "kind": kind, "indices": indices,
             "index62": {"actual_fp16_bits": "662f" if label == "mapped" or label in rescues else "6630",
                         "accepted": label == "mapped" or label in rescues}}
            for label, kind, indices in d.interventions()]


class CoordinateCutTests(unittest.TestCase):
    def test_exact_coverage_and_order(self):
        plan = list(d.interventions())
        self.assertEqual(len(plan), 954)
        self.assertEqual([indices for _, kind, indices in plan if kind == "singleton"],
                         [[i] for i in range(896)])
        self.assertEqual(len({label for label, _, _ in plan}), 954)

    def test_shards_partition_and_complements(self):
        plan = list(d.interventions())[898:]
        self.assertEqual(len(plan), 56)
        covered = []
        for shard, complement in zip(plan[::2], plan[1::2], strict=True):
            self.assertEqual(len(shard[2]), 32)
            self.assertEqual(len(complement[2]), 864)
            self.assertFalse(set(shard[2]) & set(complement[2]))
            self.assertEqual(sorted(shard[2] + complement[2]), list(range(896)))
            covered.extend(shard[2])
        self.assertEqual(covered, list(range(896)))

    def test_intervention_copies_paired_coordinates_without_mutation(self):
        actual = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
        values = np.ones(896, dtype="<f8")
        values[62] = 2.0
        mapped = d.paired.mapped_parent(values)
        changed = d.intervene(actual, mapped, [62])
        for key in actual:
            self.assertEqual(changed[key][62], mapped[key][62])
            np.testing.assert_array_equal(changed[key][:62], actual[key][:62])
            np.testing.assert_array_equal(changed[key][63:], actual[key][63:])
            self.assertFalse(np.shares_memory(changed[key], actual[key]))
        self.assertEqual(actual["i"][62], 1 << 24)

    def test_empty_and_full_intervention(self):
        actual = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        mapped = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
        d.paired.same_arrays(d.intervene(actual, mapped, []), actual)
        d.paired.same_arrays(d.intervene(actual, mapped, list(range(896))), mapped)

    def test_invalid_coordinate_selection_rejected(self):
        parent = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        for indices in ([-1], [896], [True], [1, 1], [2, 1], (1,)):
            with self.assertRaises(ValueError):
                d.intervene(parent, parent, indices)

    def test_inconsistent_paired_state_rejected(self):
        parent = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        parent["h"][0] = 0x3c00
        with self.assertRaises(ValueError):
            d.intervene(parent, parent, [0])

    def test_signed_zero_and_ties_even_mapping(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        actual = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        changed = d.intervene(actual, mapped, [0, 1])
        np.testing.assert_array_equal(changed["h"][:3], [0x8000, 0x8000, 0])

    def test_exact_parent_comparison(self):
        values = np.full(896, 3 * 2**-25, dtype="<f8")
        actual = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        mapped = d.paired.mapped_parent(values)
        rows = d.compare_parents(actual, mapped, values)
        self.assertEqual(len(rows), 896)
        self.assertEqual(rows[62]["mapped_minus_actual_Q24_units"], 2)
        self.assertEqual(Fraction(rows[62]["mapped_minus_original"]), Fraction(1, 1 << 25))

    def test_single_coordinate_classification_is_not_producer_attribution(self):
        result = d.classify(controls(("single_062",)))
        self.assertEqual(result["sufficient_single_coordinates"], [62])
        self.assertTrue(result["unique_sufficient_singleton_in_tested_mapping"])
        self.assertFalse(result["unique_upstream_producer_attributed"])
        self.assertTrue(result["missing_evidence"])

    def test_multiple_singletons_and_other_accepted_word(self):
        rows = controls(("single_002", "single_062"))
        rows[4]["index62"]["actual_fp16_bits"] = "662e"
        result = d.classify(rows)
        self.assertEqual(result["sufficient_single_coordinates"], [2, 62])
        self.assertEqual(result["single_coordinate_662f_rescues"], [62])
        self.assertFalse(result["unique_sufficient_singleton_in_tested_mapping"])

    def test_shard_only_classification(self):
        result = d.classify(controls(("shard_032", "complement_064")))
        self.assertEqual(result["classification"], "bounded_shard_sufficiency_without_singleton_rescue")
        self.assertEqual(result["sufficient_single_coordinates"], [])
        self.assertEqual(result["sufficient_shard_controls"], ["shard_032", "complement_064"])

    def test_unresolved_interactions_are_not_exhaustive_claim(self):
        result = d.classify(controls())
        self.assertEqual(result["classification"], "full_parent_rescue_with_unresolved_subset_interactions")
        self.assertFalse(result["unique_upstream_producer_attributed"])

    def test_incomplete_reordered_or_changed_endpoints_rejected(self):
        for rows in (controls()[:-1], list(reversed(controls()))):
            with self.assertRaises(ValueError):
                d.classify(rows)
        for index in (0, 1):
            rows = controls()
            rows[index]["index62"]["accepted"] = index == 0
            with self.assertRaises(ValueError):
                d.classify(rows)

    def test_unchanged_exact_threshold_and_retained_coordinate(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])

    def test_scientific_helpers_are_unchanged_v1_functions(self):
        for name in ("interventions", "intervene", "classify", "compare_parents", "check_reports"):
            self.assertIs(getattr(d, name), getattr(d.legacy, name))

    def test_foreign_source_and_changed_binding_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {}, {})
        item = {"path": str(d.ROOT / "build/coordinate-test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.bind_input(item, {}, {})

    def test_cli_has_only_check_and_no_dispatch_calls(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--out", "build/not-authorized"]):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    d.main()
        self.assertEqual(error.exception.code, 2)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(ast.parse(Path(d.__file__).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"execute", "diagnose", "execute_layer", "stages", "execute_layers",
                                 "system", "Popen", "publish_parent", "continuation_stages"})

    def test_contract_preserves_scientific_fields_and_no_admission(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        self.assertEqual(contract["controls"], len(list(d.interventions())))
        for key in ("candidate_admitted", "policy_adopted", "successor_published", "scientific_result_claim"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["native_layer_invocations"], 0)
        self.assertEqual(contract["rtl_invocations"], 0)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")

    def test_each_selected_rebinding_retains_snapshot_and_current(self):
        for relative, update in d.SOURCE_UPDATES.items():
            retained, current, snapshot = d.source_records(relative, update)
            records = {current["path"]: current, snapshot["path"]: snapshot}
            bound, updates = {}, {}
            with patch.object(d.upstream, "record", side_effect=lambda p: records[str(p)]):
                path = d.bind_input(retained, bound, updates)
            self.assertEqual(path, Path(snapshot["path"]))
            self.assertEqual(bound, records)
            self.assertEqual(updates[relative]["retained_original"], retained)
            self.assertFalse(updates[relative]["historical_source_relabelled"])

    def test_arbitrary_historical_binding_rejected(self):
        for relative, update in d.SOURCE_UPDATES.items():
            retained, current, _ = d.source_records(relative, update)
            for changed in ({**retained, "sha256": "bad"}, current):
                with self.assertRaisesRegex(ValueError, "unexpected historical source binding"):
                    d.bind_input(changed, {}, {})

    def test_changed_current_source_or_snapshot_rejected(self):
        for relative, update in d.SOURCE_UPDATES.items():
            retained, current, snapshot = d.source_records(relative, update)
            for target in (current, snapshot):
                records = {current["path"]: current, snapshot["path"]: snapshot}
                records[target["path"]] = {**target, "sha256": "bad"}
                with patch.object(d.upstream, "record", side_effect=lambda p: records[str(p)]):
                    with self.assertRaisesRegex(ValueError, "binding mismatch"):
                        d.bind_input(retained, {}, {})

    def test_conflicting_source_binding_rejected(self):
        relative, update = next(iter(d.SOURCE_UPDATES.items()))
        retained, current, snapshot = d.source_records(relative, update)
        with patch.object(d.upstream, "record", return_value=snapshot):
            with self.assertRaisesRegex(ValueError, "conflicting binding"):
                d.bind_input(retained, {current["path"]: {**current, "sha256": "bad"}}, {})

    def test_threshold_gate_reference_and_scope_edits_rejected(self):
        original = json.loads(d.CONTRACT.read_text())
        for key in d.PRESERVED_FIELDS:
            contract = deepcopy(original)
            contract[key] = None
            with self.assertRaisesRegex(ValueError, "v1 scientific contract changed"):
                d.check_contract(contract)

    def test_binding_and_execution_policy_edits_rejected(self):
        original = json.loads(d.CONTRACT.read_text())
        for key in ("source_updates", "retained_freeze_sha256", "historical_blocked_sha256",
                    "native_layer_invocations", "interface"):
            contract = deepcopy(original)
            contract[key] = None
            with self.assertRaisesRegex(ValueError, "v2 check contract mismatch"):
                d.check_contract(contract)

    def test_contract_origin_outside_repository_rejected(self):
        with patch.object(d, "CONTRACT", Path("/tmp/foreign-contract.json")):
            with self.assertRaisesRegex(ValueError, "contract origin mismatch"):
                d.origins()

    def test_incomplete_or_changed_gate_reports_rejected(self):
        reports = [{"node": [13, 0, stage], "policy_id": d.prior.gates.POLICY_ID,
                    "status": "PASS", "residual_state_lineage": "PASS", "kv_lineage": "PASS",
                    "local_operator_fp16" if stage < 18 else "binary64_v1": {"passed": True}}
                   for stage in range(19)]
        with self.assertRaisesRegex(ValueError, "incomplete endpoint gate reports"):
            d.check_reports(reports[:-1], reports)
        for key in ("policy_id", "status", "residual_state_lineage", "kv_lineage", "local_operator_fp16"):
            changed = deepcopy(reports)
            changed[0][key] = "changed"
            with self.assertRaisesRegex(ValueError, "reviewed endpoint gate changed"):
                d.check_reports(changed, reports)


if __name__ == "__main__":
    unittest.main()
