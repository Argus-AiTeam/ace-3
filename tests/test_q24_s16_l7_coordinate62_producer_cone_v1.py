"""Focused synthetic branch, provenance, gate and L8-L13 dispatch checks."""

from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l7_coordinate62_producer_cone_v1 as d


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
    def test_plan_is_factorial_plus_s12_and_output_cuts(self):
        self.assertEqual(len(d.plan()), 11)
        self.assertEqual(d.plan()[:8], d.producer.plan()[:8])
        self.assertEqual([p[0] for p in d.plan()[8:]], ["scratch", "scratch_down", "mapped62"])

    def test_actual_reconstructs_accepted_state(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "actual")
        d.paired.same_arrays(output, layer["output"])
        self.assertEqual(operands["scratch_i"][62], 2 << 24)
        self.assertEqual(output["i"][62], 3 << 24)

    def test_inherited_preserves_uncut_branches_and_coordinates(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "frozen_inherited")
        self.assertEqual(operands["input_i"][62], 2 << 24)
        np.testing.assert_array_equal(operands["o"], layer["arrays"]["stage11"])
        np.testing.assert_array_equal(operands["down"], layer["arrays"]["stage17"])
        self.assertEqual(output["i"][62], 4 << 24)
        self.assertTrue(np.all(output["i"][np.arange(896) != 62] == 3 << 24))
        self.assertTrue(np.all(layer["parent"]["i"] == 1 << 24))

    def test_o_down_are_fp16_single_coordinate_cuts(self):
        layer, original = fixture()
        for label, value in (("frozen_o", 5), ("frozen_down", 6)):
            operands, output = d.cut_parent(layer, original, label)
            self.assertEqual(output["i"][62], value << 24)
            self.assertEqual(operands["o"].dtype, np.dtype("<u2"))
            self.assertEqual(operands["down"].dtype, np.dtype("<u2"))
            self.assertEqual(output["i"][0], 3 << 24)

    def test_factorial_additive_closure(self):
        layer, original = fixture()
        values = {parts: int(d.cut_parent(layer, original, label)[1]["i"][62])
                  for label, parts in d.plan()[:8]}
        for parts, value in values.items():
            self.assertEqual(value - values[()], sum(values[(p,)] - values[()] for p in parts))

    def test_s12_and_joint_cuts_keep_explicit_state(self):
        layer, original = fixture()
        for label, value in (("scratch", 6), ("scratch_down", 9)):
            operands, output = d.cut_parent(layer, original, label)
            self.assertEqual(operands["scratch_i"][62], 5 << 24)
            self.assertEqual(output["i"][62], value << 24)
            self.assertEqual(output["i"][0], 3 << 24)

    def test_output_cut_maps_all_three_state_components(self):
        layer, original = fixture()
        operands, output = d.cut_parent(layer, original, "mapped62")
        self.assertIsNone(operands)
        expected = d.paired.mapped_parent(original["s18"])
        for key in output:
            self.assertEqual(output[key][62], expected[key][62])
            np.testing.assert_array_equal(output[key][np.arange(896) != 62],
                                          layer["output"][key][np.arange(896) != 62])

    def test_invalid_cut_and_inconsistent_state_rejected(self):
        layer, original = fixture()
        with self.assertRaisesRegex(ValueError, "unknown L7 control"):
            d.cut_parent(layer, original, "other")
        layer["parent"]["h"][62] = 0
        with self.assertRaises(ValueError):
            d.cut_parent(layer, original, "frozen_o")

    def test_exact_mapping_ties_even_signed_zero_and_bound(self):
        values = np.zeros(896, dtype="<f8")
        values[:4] = [-0.0, -2**-25, 2**-25, 3 * 2**-25]
        mapped = d.paired.mapped_parent(values)
        np.testing.assert_array_equal(mapped["i"][:4], [0, 0, 0, 2])
        np.testing.assert_array_equal(mapped["z"][:4], [1, 1, 0, 0])
        np.testing.assert_array_equal(mapped["h"][:3], [0x8000, 0x8000, 0])
        self.assertEqual(Fraction(d.coordinate.compare_parents(mapped, mapped, values)[3]
                                 ["mapped_minus_original"]), Fraction(1, 1 << 25))

    def test_unchanged_exact_gate_and_retained_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(Fraction(d.prior.measure(0x6630, reference)["excess_budget"]), Fraction(1, 8))

    def test_singletons_only_establish_conditional_sufficiency(self):
        for branch in ("inherited", "o", "down"):
            result = d.classify(rows(("frozen_" + branch,)))
            self.assertEqual(result["classification"], "conditional_inherited_L6_branch_sufficiency"
                             if branch == "inherited" else "conditional_local_L7_branch_sufficiency")
            self.assertFalse(result["unique_upstream_producer_attributed"])
            self.assertTrue(result["missing_evidence"])
        for rescues in ((), ("frozen_o_down",), ("frozen_inherited", "frozen_down")):
            self.assertEqual(d.classify(rows(rescues))["classification"],
                             "unresolved_under_tested_branch_mappings")
        for invalid in (rows()[:-1], list(reversed(rows())), rows(("actual",))):
            with self.assertRaises(ValueError):
                d.classify(invalid)

    def test_suffix_is_l8_l13_and_threads_actual_parent_not_reference(self):
        parents = [object() for _ in range(7)]
        data = {"layers": {layer: {"tensors": layer, "trajectory": object(), "reference": object()}
                           for layer in range(8, 14)}}
        calls = []

        def execute(tensors, layer, incoming, trajectory, reference):
            self.assertEqual(tensors, layer)
            self.assertIs(incoming, parents[layer - 8])
            self.assertIs(reference, data["layers"][layer]["reference"])
            self.assertIs(trajectory, data["layers"][layer]["trajectory"])
            calls.append(layer)
            return {"next": parents[layer - 7]}, {}, [{"status": "FAIL"}]

        with patch.object(d, "execute_l8", side_effect=lambda t, p, f, b: execute(t, 8, p, f, b)), \
                patch.object(d.upstream, "execute_layer", side_effect=execute), \
                patch.object(d.prior.retained, "state_from", side_effect=lambda a, *_: a["next"]):
            result = d.execute_suffix(parents[0], data, lambda layer, a, l, r: r[0]["status"])
        self.assertEqual(calls, [8, 9, 10, 11, 12, 13])
        self.assertEqual(result, ["FAIL"] * 6)

    def test_l8_wrapper_requires_complete_ordered_stages(self):
        parent = d.paired.mapped_parent(np.zeros(896, dtype="<f8"))
        with patch.object(d.native.candidate, "_stages", return_value=iter(())):
            with self.assertRaisesRegex(ValueError, "incomplete native L8"):
                d.execute_l8({}, parent, {}, np.zeros(896))
        with patch.object(d.native.candidate, "_stages", return_value=iter((1,))):
            with self.assertRaisesRegex(ValueError, "out-of-order"):
                d.execute_l8({}, parent, {}, np.zeros(896))

    def test_origin_tampering_and_foreign_evidence_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/foreign", "bytes": 0, "sha256": "x"}, {})
        item = {"path": str(d.ROOT / "build/test-input"), "bytes": 1, "sha256": "a"}
        with patch.object(d.upstream, "record", return_value={**item, "sha256": "b"}):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item, {})

    def test_output_exclusive_and_scoped(self):
        with self.assertRaises(ValueError):
            d.output_path(d.ROOT / "build/other")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.ROOT / "build/q24_s16_l7_coordinate62_producer_cone_test")

    def test_contract_and_reviewed_evidence_non_admission(self):
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["controls"], [label for label, _ in d.plan()])
        self.assertEqual(contract["reviewed_result_sha256"], d.REVIEWED_SHA)
        self.assertEqual(contract["reproduced_result_sha256"], d.REPRODUCED_SHA)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertEqual(contract["rtl_invocations"], 0)
        reviewed = {
            "diagnostic_id": d.previous.ID, "status": "DIAGNOSED",
            "classification": "conditional_inherited_L7_branch_sufficiency",
            "native_retained_bitwise_reproduction": True, "original_L8_S18_bitwise_reproduction": True,
            "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
            "source_operand_state_KV_lineage_checks": "PASS", "rtl_invocations": 0,
        }
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(contract[key], False)
            reviewed[key] = False
        d.authenticate_reviewed(reviewed)
        for key, bad in (("candidate_admitted", True), ("rtl_invocations", 1),
                         ("original_global_reference_unchanged", False),
                         ("classification", "unique_root_cause")):
            with self.assertRaisesRegex(ValueError, "scientific parent mismatch"):
                d.authenticate_reviewed({**reviewed, key: bad})


if __name__ == "__main__":
    unittest.main()
