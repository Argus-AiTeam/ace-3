"""Coordinate coverage, paired-state semantics and fail-closed evidence tests."""

import ast
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1 as d


EXPECTED_TESTS = 18


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

    def test_only_native_layer13_dispatch(self):
        data = {"tensors": {}, "trajectory": {}, "reference": np.zeros(896)}
        with patch.object(d.upstream, "execute_layer", return_value="sentinel") as execute:
            self.assertEqual(d.execute({}, data), "sentinel")
            self.assertEqual(execute.call_args.args[1], 13)

    def test_foreign_source_and_changed_binding_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "bad"}, {})
        item = {"path": str(d.ROOT / "build/coordinate-test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_output_is_exclusive_and_scope_bound(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/wrong-prefix")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l13_l12_parent_coordinate_cut_test")

    def test_contract_no_publication_or_external_execution(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["controls"], len(list(d.interventions())))
        self.assertEqual(contract["reviewed_result_sha256"], d.REVIEWED_SHA)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertEqual(contract["rtl_invocations"], 0)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(ast.parse(Path(d.__file__).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "publish_parent", "execute_layers",
                                 "continuation_stages", "run_factory"})


if __name__ == "__main__":
    unittest.main()
