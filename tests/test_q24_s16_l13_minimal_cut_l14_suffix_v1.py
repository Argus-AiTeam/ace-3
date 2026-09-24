"""Focused cut/authentication/dispatch tests; no real native layer executes."""

import copy
from fractions import Fraction
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_minimal_cut_l14_suffix_v1 as d


def fixture():
    parent = {"i": np.zeros(896, dtype="<i8"), "z": np.zeros(896, dtype="u1"),
              "h": np.zeros(896, dtype="<u2")}
    parent["i"][62], parent["h"][62] = 26568588224, 0x6630
    cut = {"baseline_Q24_integer": 26568588224, "delta_Q24_units": -1866689,
           "delta": "-1866689/16777216", "output_word": "662f",
           "one_Q24_unit_less_reduction_word": "6630"}
    return parent, cut


def references():
    previous64, previous16 = {"r": 8}, {"f": 8}
    result = {"reference_only": True, "accepted_ancestor_oracle_replays": 0,
              "policy_id": d.gates.POLICY_ID,
              "global_reference_policy": "legacy-binary64-AWQ-fully-independent-propagation",
              "original_binary64_parent": previous64, "original_fp16_parent": previous16,
              "layers": {}}
    for layer in range(9, 24):
        item = {"input_binary64": previous64, "input_fp16": previous16,
                "binary64": {"r": layer}, "fp16": {"f": layer}, "prior_kv": "own empty P0"}
        result["layers"][str(layer)] = item
        previous64, previous16 = item["binary64"], item["fp16"]
    return result


class SuffixTests(unittest.TestCase):
    def test_exact_minimal_cut(self):
        parent, cut = fixture()
        changed = d.minimal_cut(parent, cut)
        self.assertEqual(int(changed["i"][62]), 26566721535)
        self.assertEqual(int(changed["h"][62]), 0x662F)
        self.assertEqual(Fraction(int(changed["i"][62]) - int(parent["i"][62]), 1 << 24),
                         Fraction(cut["delta"]))

    def test_no_mutation_or_other_coordinate_change(self):
        parent, cut = fixture()
        original = copy.deepcopy(parent)
        changed = d.minimal_cut(parent, cut)
        for key in parent:
            np.testing.assert_array_equal(parent[key], original[key])
            np.testing.assert_array_equal(changed[key][:62], parent[key][:62])
            np.testing.assert_array_equal(changed[key][63:], parent[key][63:])

    def test_zero_signs_preserved(self):
        parent, cut = fixture()
        parent["z"][0], parent["h"][0] = 1, 0x8000
        np.testing.assert_array_equal(d.minimal_cut(parent, cut)["z"], parent["z"])

    def test_adjacent_decrement_rejected(self):
        parent, cut = fixture()
        cut["delta_Q24_units"] += 1
        cut["delta"] = str(Fraction(cut["delta_Q24_units"], 1 << 24))
        with self.assertRaisesRegex(ValueError, "minimal"):
            d.minimal_cut(parent, cut)

    def test_larger_decrement_rejected(self):
        parent, cut = fixture()
        cut["delta_Q24_units"] -= 1
        cut["delta"] = str(Fraction(cut["delta_Q24_units"], 1 << 24))
        with self.assertRaisesRegex(ValueError, "minimal"):
            d.minimal_cut(parent, cut)

    def test_wrong_frozen_parent_rejected(self):
        parent, cut = fixture()
        cut["baseline_Q24_integer"] += 1
        with self.assertRaisesRegex(ValueError, "reviewed"):
            d.minimal_cut(parent, cut)

    def test_broken_state_view_rejected(self):
        parent, cut = fixture()
        parent["h"][0] = 1
        with self.assertRaisesRegex(ValueError, "view mismatch"):
            d.minimal_cut(parent, cut)

    def test_reference_recurrence(self):
        extension = references()
        keys = ("checkpoint", "original_binary64_parent", "original_fp16_parent",
                "original_specification", "input_freeze")
        for key in ("checkpoint", "original_specification", "input_freeze"):
            extension[key] = {"scientific": key}
        extension["generators"] = [{"scientific": "generator"}]
        extension["python"] = {"path": "/outside/runtime", "bytes": 1, "sha256": "runtime"}
        inputs = Mock()
        d.bind_reference_inputs(inputs, extension)
        self.assertEqual([call.args[0] for call in inputs.bind.call_args_list],
                         [extension[key] for key in keys] + extension["generators"])

    def test_reanchored_global_reference_rejected(self):
        extension = references()
        extension["layers"]["14"]["input_binary64"] = {"candidate": True}
        with self.assertRaisesRegex(ValueError, "recurrence"):
            d.check_reference_suffix(extension)

    def test_spliced_fp16_reference_rejected(self):
        extension = references()
        extension["layers"]["23"]["input_fp16"] = {"f": 13}
        with self.assertRaisesRegex(ValueError, "recurrence"):
            d.check_reference_suffix(extension)

    def test_cross_layer_kv_rejected(self):
        extension = references()
        extension["layers"]["14"]["prior_kv"] = "L13 KV"
        with self.assertRaisesRegex(ValueError, "KV"):
            d.check_reference_suffix(extension)

    def test_reference_replay_rejected(self):
        extension = references()
        extension["accepted_ancestor_oracle_replays"] = 1
        with self.assertRaisesRegex(ValueError, "replayed"):
            d.check_reference_suffix(extension)

    def test_native_prefix_and_scope_guard(self):
        audit = {"native_layers": []}
        with d.suffix_only(audit):
            for layer in (*range(14), 24, True):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L14-L23"):
                    next(d.native.candidate._stages({}, layer, {}, {}))
        self.assertEqual(audit["native_layers"], [])

    def test_first_dispatch_is_l14(self):
        audit = {"native_layers": []}
        with patch.object(d.native.candidate, "_stages", return_value=iter([0])) as producer:
            with d.suffix_only(audit):
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    next(d.native.candidate._stages({}, 15, {}, {}))
                self.assertEqual(list(d.native.candidate._stages({}, 14, {}, {})), [0])
            producer.assert_called_once()
        self.assertEqual(audit["native_layers"], [14])

    def test_external_execution_forbidden(self):
        with d.suffix_only({"native_layers": []}):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                d.subprocess.Popen(["never-executed"])

    def test_pinned_evidence_mismatch_rejected(self):
        with patch.object(d.retained, "record", return_value={"sha256": "wrong"}):
            with self.assertRaisesRegex(ValueError, "pinned evidence mismatch"):
                d.margin.pinned(d.prior.BoundInputs(), d.MARGIN / "result.json", d.MARGIN_SHA)

    def test_first_gate_failure_stops_producer(self):
        arrays, reports, refs, visited = {}, [], {}, []

        def producer(*args):
            for stage in range(19):
                visited.append(stage)
                arrays[f"stage{stage:02d}"] = np.zeros(896, dtype="<u2")
                yield stage

        report = {"status": "FAIL", "stage": 0,
                  "local_operator_fp16": {"passed": False, "failures": [{"index": 7}]}}
        with patch.object(d.native.candidate, "_stages", producer), \
                patch.object(d.retained, "check_stage_state", return_value=None), \
                patch.dict(d.local.OPERANDS, {0: ()}), \
                patch.object(d.local, "local_reference", return_value=np.zeros(896, dtype="<u2")), \
                patch.object(d.gates, "evaluate_decoder_stage", return_value=report):
            failure, _ = d.drive_layer({}, 14, {}, {"stage00": np.zeros(896, dtype="<u2")},
                                      np.zeros(896), arrays, refs, reports)
        self.assertEqual(visited, [0])
        self.assertEqual(failure["node"], [14, 0, 0])
        self.assertEqual(failure["index"], 7)
        self.assertEqual(len(reports), 1)

    def test_versioned_nonadmission_contract(self):
        contract = json.loads(d.CONTRACT.read_text())
        for key, value in d.FLAGS.items():
            self.assertEqual(contract[key], value)
        self.assertEqual(contract["layers"], list(range(14, 24)))
        self.assertEqual(contract["focused_tests"], 18)


if __name__ == "__main__":
    unittest.main()
