"""Synthetic L8 cuts, provenance, unchanged gates and bounded suffix dispatch."""

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l8_coordinate62_producer_cone_v1 as d


EXPECTED_TESTS = 16


def fixture():
    parent = d.paired.mapped_parent(np.ones(896, dtype="<f8"))
    o = np.full(896, 0x3c00, dtype="<u2")
    scratch, output = d.producer.compose(parent, o, o)
    arrays = {"stage11": o.copy(), "stage17": o.copy(), "stage12": scratch["h"]}
    arrays.update({"scratch_" + k: v for k, v in scratch.items()})
    original = {key: np.full(896, value, dtype="<f8") for key, value in
                (("input", 2), ("s11", 3), ("s17", 4), ("residual", 5), ("s18", 9))}
    return {"parent": parent, "arrays": arrays, "output": output}, original


def rows(rescues=()):
    return [{"label": label, "index62": {
        "accepted": label in rescues, "actual_fp16_bits": "662f" if label in rescues else "6630",
    }} for label, _ in d.plan()]


class ProducerConeTests(unittest.TestCase):
    def test_plan_is_full_factorial_plus_disclosed_s12_output_cuts(self):
        self.assertEqual(len(d.plan()), 11)
        self.assertEqual(d.plan()[:8], d.producer.plan()[:8])
        self.assertEqual([p[0] for p in d.plan()[8:]], ["scratch", "scratch_down", "mapped62"])

    def test_actual_reconstructs_exact_accepted_state(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "actual")
        d.paired.same_arrays(output, layer["output"])
        self.assertEqual(operands["scratch_i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)

    def test_inherited_cut_preserves_other_branches_and_coordinates(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "frozen_inherited")
        self.assertEqual(operands["input_i"][62], 2 << 24)
        np.testing.assert_array_equal(operands["o"], layer["arrays"]["stage11"])
        np.testing.assert_array_equal(operands["down"], layer["arrays"]["stage17"])
        self.assertEqual(output["i"][62], 4 << 24)
        self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))
        self.assertTrue(np.all(layer["parent"]["i"] == 1 << 24))

    def test_o_and_down_use_only_selected_fp16_branch(self):
        layer, original = fixture()
        for label, value in (("frozen_o", 5), ("frozen_down", 6)):
            operands, output = d.cut_parent(layer, original, label)
            self.assertEqual(output["i"][62], value << 24)
            self.assertEqual(operands["o"].dtype, np.dtype("<u2"))
            self.assertEqual(operands["down"].dtype, np.dtype("<u2"))
        operands, _ = d.cut_parent(layer, original, "frozen_down")
        self.assertEqual(operands["scratch_i"][62], 2 << 24)

    def test_full_factorial_additive_closure(self):
        layer, original = fixture()
        values = {parts: int(d.cut_parent(layer, original, label)[1]["i"][62])
                  for label, parts in d.plan()[:8]}
        for parts, value in values.items():
            self.assertEqual(value - values[()], sum(values[(p,)] - values[()] for p in parts))

    def test_scratch_cut_and_joint_keep_explicit_state(self):
        layer, original = fixture()
        for label, value in (("scratch", 6), ("scratch_down", 9)):
            operands, output = d.cut_parent(layer, original, label)
            self.assertEqual(operands["scratch_i"][62], 5 << 24)
            self.assertEqual(output["i"][62], value << 24)
            self.assertEqual(output["i"][0], 3 << 24)

    def test_output_mapping_replaces_complete_coordinate_only(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "mapped62")
        self.assertIsNone(operands)
        expected = d.paired.mapped_parent(original["s18"])
        for key in output:
            self.assertEqual(output[key][62], expected[key][62])
            np.testing.assert_array_equal(output[key][np.arange(896) != 62],
                                          layer["output"][key][np.arange(896) != 62])

    def test_invalid_cut_or_parent_fails_explicitly(self):
        layer, original = fixture()
        with self.assertRaisesRegex(ValueError, "unknown L8 control"):
            d.cut_parent(layer, original, "other")
        layer["parent"]["h"][62] = 0
        with self.assertRaises(ValueError):
            d.cut_parent(layer, original, "frozen_o")

    def test_mapping_ties_even_signed_zero_and_half_grid(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(mapped["h"][:3], [0x8000, 0x8000, 0])
        self.assertEqual(Fraction(d.coordinate.compare_parents(mapped, mapped, values)[3]
                                 ["mapped_minus_original"]), Fraction(1, 1 << 25))

    def test_unchanged_excess_boundary_and_retained_witness(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(d.prior.measure(0x6630, reference)["excess_budget"]), Fraction(1, 8))

    def test_singleton_sufficiency_is_conditional_not_root_cause(self):
        for branch in ("inherited", "o", "down"):
            result = d.classify(rows(("frozen_" + branch,)))
            self.assertEqual(result["classification"], "conditional_inherited_L7_branch_sufficiency"
                             if branch == "inherited" else "conditional_local_L8_branch_sufficiency")
            self.assertFalse(result["unique_upstream_producer_attributed"])
            self.assertTrue(result["missing_evidence"])

    def test_multiple_joint_only_no_singleton_remain_unresolved(self):
        for rescues in ((), ("frozen_o_down",), ("frozen_inherited", "frozen_down")):
            self.assertEqual(d.classify(rows(rescues))["classification"],
                             "unresolved_under_tested_branch_mappings")
        for invalid in (rows()[:-1], list(reversed(rows())), rows(("actual",))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_suffix_is_only_l9_l13_with_independent_reference_and_immediate_parent(self):
        parents = [object() for _ in range(6)]
        data = {"layers": {layer: {"tensors": layer, "trajectory": object(), "reference": object()}
                           for layer in range(9, 14)}}
        calls = []

        def execute(tensors, layer, incoming, trajectory, reference):
            self.assertEqual(tensors, layer)
            self.assertIs(incoming, parents[layer - 9])
            self.assertIs(reference, data["layers"][layer]["reference"])
            self.assertIs(trajectory, data["layers"][layer]["trajectory"])
            calls.append(layer)
            return {"next": parents[layer - 8]}, {}, [{"status": "FAIL"}]

        with patch.object(d.upstream, "execute_layer", side_effect=execute), \
                patch.object(d.prior.retained, "state_from", side_effect=lambda a, *_: a["next"]):
            result = d.execute_suffix(parents[0], data, lambda layer, a, l, r: r[0]["status"])
        self.assertEqual(calls, [9, 10, 11, 12, 13])
        self.assertEqual(result, ["FAIL"] * 5)

    def test_foreign_module_and_tampered_evidence_are_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/foreign", "bytes": 0, "sha256": "x"}, {})
        item = {"path": str(d.ROOT / "build/test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_output_is_exclusive_and_scoped(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/other")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l8_coordinate62_producer_cone_test")

    def test_contract_binds_plan_and_non_admission(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["controls"], [label for label, _ in d.plan()])
        self.assertEqual(contract["reviewed_result_sha256"], d.REVIEWED_SHA)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertEqual(contract["rtl_invocations"], 0)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)


if __name__ == "__main__":
    unittest.main()
