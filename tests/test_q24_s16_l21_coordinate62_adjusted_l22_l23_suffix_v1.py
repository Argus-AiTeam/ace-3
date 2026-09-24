"""Focused authentication and synthetic orchestration tests; no native execution."""

import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l21_coordinate62_adjusted_l22_l23_suffix_v1 as d


class SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.command = json.loads((d.INPUT / "command.json").read_text())
        with np.load(d.margin.INPUT / "layer21/actual_stages.npz", allow_pickle=False) as data:
            cls.arrays = {key: data[key].copy() for key in data.files}
        cls.parent = d.retained.state_from(cls.arrays, "output", "stage18")
        cls.cut = cls.document["minimal_residual_cut"]
        freeze = json.loads((d.upstream.upstream.margin.INPUT / "freeze.json").read_text())
        cls.extension = json.loads(Path(freeze["reference_extension"]["path"]).read_text())

    def test_accepted_margin(self):
        d.check_margin(self.document, self.validation, self.command)

    def test_wrong_dependency(self):
        for key, value in (("status", "BLOCKED"), ("node", [15, 0, 18]),
                           ("index", 61), ("upstream_sha256", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_margin(dict(self.document, **{key: value}), self.validation, self.command)

    def test_dependency_counts(self):
        for key in ("collected", "executed", "errors", "failures", "skipped"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "counts"):
                d.check_margin(self.document, dict(self.validation, **{key: 1}), self.command)

    def test_dependency_context_and_linkage(self):
        for key, value in (("cwd", "/outside"), ("executable", "/usr/bin/python"),
                           ("PYTHONPATH", ""), ("dont_write_bytecode", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "context"):
                d.check_margin(self.document, self.validation, dict(self.command, **{key: value}))
        changed = copy.deepcopy(self.document)
        changed["validation"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "linkage"):
            d.check_margin(changed, self.validation, self.command)

    def test_nonadmission_flags(self):
        for key in d.margin.FLAGS:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "boundary"):
                d.check_margin(dict(self.document, **{key: True}), self.validation, self.command)

    def test_pinned_inputs(self):
        for directory, pins in ((d.INPUT, d.PINS), (d.margin.INPUT, d.margin.PINS)):
            for name, digest in pins.items():
                signature = d.retained.record(directory / name)
                self.assertEqual(signature["sha256"], digest)
                if name.endswith(".json"):
                    with patch.object(d.retained, "record",
                                      return_value=dict(signature, sha256="wrong")):
                        with self.assertRaisesRegex(ValueError, "pinned"):
                            d.margin.scalar_math.pinned(
                                d.margin.margin.prior.BoundInputs(), directory / name, digest)

    def test_exact_adjustment_without_mutation(self):
        original = copy.deepcopy(self.parent)
        changed = d.minimal_cut(self.parent, self.cut)
        self.assertEqual(int(changed["i"][62]), 1602748417)
        self.assertEqual(int(changed["i"][62]) - int(self.parent["i"][62]), 10103041)
        self.assertEqual(int(changed["h"][62]), 0x55F9)
        for key in ("i", "h"):
            self.assertEqual(np.flatnonzero(changed[key] != self.parent[key]).tolist(), [62])
        self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
        for key in self.parent:
            self.assertTrue(np.array_equal(self.parent[key], original[key]))
            self.assertFalse(np.shares_memory(changed[key], self.parent[key]))

    def test_changed_cut(self):
        for key, value in (("delta_Q24_units", 10103040), ("delta_Q24_units", 10103041.0),
                           ("adjusted_Q24_integer", 1602748416), ("output_word", "55fa"),
                           ("one_unit_less_word", "55f9"), ("delta", "1/2")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "adjustment"):
                d.minimal_cut(self.parent, dict(self.cut, **{key: value}))

    def test_paired_state_required(self):
        with self.assertRaises(ValueError):
            d.minimal_cut({"h": self.parent["h"]}, self.cut)
        changed = copy.deepcopy(self.parent)
        changed["i"][61] += 1 << 24
        with self.assertRaises(ValueError):
            d.minimal_cut(changed, self.cut)

    def test_original_global_vector_and_one_less(self):
        item = self.extension["layers"]["21"]
        inputs = d.margin.margin.prior.BoundInputs()
        reference = d.upstream.upstream.global_reference(inputs, item)
        trajectory = inputs.archive(item["fp16"])
        analysis = d.margin.sensitivity(int(self.parent["i"][62]),
                                       d.Fraction.from_float(float(reference[62])))
        self.assertEqual(analysis["minimal_residual_cut"], self.cut)
        self.assertEqual(analysis["minimal_residual_cut"]["one_unit_less_gate"]["excess_error"],
                         "3/16")
        self.assertEqual(d.margin.vector_boundary(self.arrays, trajectory, reference, analysis),
                         self.document["projected_full_vector_gate"])
        changed = d.minimal_cut(self.parent, self.cut)
        report = d.gates.evaluate_decoder_stage(
            stage=18, actual=changed["h"], reference=trajectory["stage18"],
            policy=d.gates.POLICY_ID, local_reference=None, reference_binary64=reference)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["binary64_v1"]["failure_count"], 0)

    def test_reference_substitution(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        for layer in d.LAYERS:
            for key in ("input_binary64", "input_fp16", "prior_kv"):
                changed = copy.deepcopy(self.extension)
                changed["layers"][str(layer)][key] = "counterfactual reference"
                with self.subTest(layer=layer, key=key), self.assertRaises(ValueError):
                    d.check_boundary(self.parent, kv, changed)
        with self.assertRaisesRegex(ValueError, "lineage"):
            d.check_margin(dict(self.document, reference_policy="actual parent"),
                           self.validation, self.command)

    def test_own_empty_kv(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        d.check_boundary(self.parent, kv, self.extension)
        for kind in kv:
            for value in (self.arrays["output_cache_" + kind],
                          np.empty((0, 128), dtype="<f2")):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    d.check_boundary(self.parent, dict(kv, **{kind: value}), self.extension)

    def test_native_scope_guard(self):
        audit = {"native_layers": []}
        with d.suffix_only(audit):
            for layer in (*range(22), 24, True):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L22-L23"):
                    next(d.native.candidate._stages({}, layer, {}, {}))
                with self.assertRaisesRegex(ValueError, "L22-L23"):
                    d.drive_layer({}, layer, {}, {}, None, {}, {}, [])
        self.assertEqual(audit["native_layers"], [])

    def test_sequential_dispatch(self):
        audit = {"native_layers": []}
        with patch.object(d.native.candidate, "_stages", side_effect=lambda *args: iter([0])) as raw:
            with d.suffix_only(audit):
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    next(d.native.candidate._stages({}, 23, {}, {}))
                for layer in d.LAYERS:
                    self.assertEqual(list(d.native.candidate._stages({}, layer, {}, {})), [0])
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    next(d.native.candidate._stages({}, 23, {}, {}))
            self.assertEqual(raw.call_count, 2)
        self.assertEqual(audit["native_layers"], [22, 23])

    def test_external_and_alternate_dispatch_forbidden(self):
        with d.suffix_only({"native_layers": []}):
            for function in (d.subprocess.Popen, d.os.system, d.native.stages, d.native.run):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function()

    def check_stop(self, stop):
        arrays, reports, refs, visited, closed = {}, [], {}, [], []

        def producer(*args):
            try:
                for stage in range(19):
                    visited.append(stage)
                    arrays[f"stage{stage:02d}"] = np.zeros(896, dtype="<u2")
                    if stage == 16:
                        arrays["s16_unrounded_binary64"] = np.zeros(896)
                    yield stage
            finally:
                closed.append(True)

        def evaluate(**kwargs):
            stage = kwargs["stage"]
            return {"stage": stage, "status": "FAIL" if stage == stop else "PASS",
                    "local_operator_fp16": {"passed": stage != stop, "failures": [{"index": 7}]},
                    "binary64_v1": {"passed": stage != stop, "failures": [{"index": 7}]}}

        with patch.object(d.native.candidate, "_stages", producer), \
                patch.object(d.retained, "check_stage_state", return_value=np.zeros(896, dtype="<u2")), \
                patch.dict(d.local.OPERANDS, {stage: () for stage in range(19)}), \
                patch.object(d.local, "local_reference", return_value=np.zeros(896, dtype="<u2")), \
                patch.object(d.gates, "evaluate_decoder_stage", side_effect=evaluate), \
                patch.object(d.native, "first_failure_index", return_value=7):
            failure, _ = d.drive_layer({}, 22, {}, {
                f"stage{stage:02d}": np.zeros(896, dtype="<u2") for stage in range(19)},
                np.zeros(896), arrays, refs, reports)
        self.assertEqual(visited, list(range(stop + 1)))
        self.assertEqual(closed, [True])
        self.assertEqual(len(reports), stop + 1)
        self.assertEqual(failure["node"], [22, 0, stop])
        self.assertEqual(failure["gate"], "binary64_v1" if stop == 18 else "local_operator_fp16")

    def test_first_local_failure_stops_producer(self):
        self.check_stop(0)

    def test_global_failure_stops_producer(self):
        self.check_stop(18)

    def exercise_suffix(self, fail):
        layers, seen, saved = [], [], []
        inputs = MagicMock()
        inputs.bind.return_value = "/synthetic/model"

        def drive(tensors, layer, parent, trajectory, reference, arrays, refs, reports):
            expected = (d.minimal_cut(self.parent, self.cut) if layer == 22 else self.parent)
            d.retained.verify_parent(parent, expected)
            seen.append(layer)
            arrays.update(copy.deepcopy(self.arrays))
            return ({"node": [layer, 0, 0], "index": 7, "gate": "local_operator_fp16"}
                    if fail else None), {}

        def save(path, arrays):
            saved.append(str(path))
            return {"path": str(path)}

        with patch.object(d.upstream.upstream, "safe_open"), \
                patch.object(d.local, "tensor_shapes", return_value={}), \
                patch.object(d.local, "authenticate_tensors"), \
                patch.object(d.upstream.upstream, "global_reference"), \
                patch.object(d, "drive_layer", side_effect=drive), \
                patch.object(Path, "mkdir"), \
                patch.object(d.retained, "save", side_effect=save), \
                patch.object(d.retained, "write"), \
                patch.object(d.retained, "record", side_effect=lambda p: {"path": str(p)}):
            failure = d.execute_suffix(d.OUTPUT, inputs, self.extension,
                                       d.minimal_cut(self.parent, self.cut),
                                       {"native_layers": []}, layers)
        self.assertEqual(seen, [22] if fail else [22, 23])
        for entry in layers:
            self.assertEqual(entry["original_reference"],
                             self.extension["layers"][str(entry["layer"])])
            self.assertEqual(entry["prior_kv"], "own empty P0")
            self.assertIs(entry["prior_layer_kv_consumed"], False)
            self.assertEqual("output_state_evidence" in entry, not fail)
            self.assertEqual("own_kv_evidence" in entry, not fail)
        if fail:
            self.assertEqual(failure["node"], [22, 0, 0])
            self.assertFalse(any("layer23" in path or "counterfactual_state" in path
                                 for path in saved))
        else:
            self.assertIsNone(failure)
            self.assertEqual(layers[1]["input_state_evidence"], layers[0]["output_state_evidence"])

    def test_suffix_failure_has_no_successor(self):
        self.exercise_suffix(True)

    def test_suffix_immediate_parent_handoff(self):
        self.exercise_suffix(False)

    def test_repository_origins(self):
        importlib.import_module(d.upstream.upstream.prior.LEGACY_TEST)
        records = d.origins()
        self.assertEqual(records[d.MODULE]["path"], str(d.SOURCE))
        self.assertIn(d.TEST_MODULE, records)
        with patch.object(d, "__file__", "/outside/source.py"):
            with self.assertRaisesRegex(ValueError, "origin"):
                d.origins()
        with patch.object(d.margin, "__file__", "/outside/margin.py"):
            with self.assertRaisesRegex(ValueError, "origin"):
                d.origins()

    def test_outside_binding_rejected(self):
        with patch.object(d.retained, "record") as record:
            with self.assertRaisesRegex(ValueError, "outside"):
                d.margin.margin.prior.BoundInputs().bind(
                    {"path": "/outside/source.py", "sha256": "wrong", "bytes": 1})
            record.assert_not_called()

    def test_contract(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        for key, value in (("excess_budget", "1/4"), ("layers", [21, 22]),
                           ("margin_sha256", {}), ("suffix_sha256", {}),
                           ("focused_tests", 19), ("delta_Q24_units", 10103040),
                           ("reference_policy", "actual"), ("candidate_admitted", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(dict(contract, **{key: value}))

    def test_output_scope_and_preservation(self):
        for path in (d.ROOT / "build/other", d.OUTPUT / "child/grandchild"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "outside"):
                d.output_path(path)
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "resolve", return_value=d.ROOT / "elsewhere"):
            with self.assertRaisesRegex(ValueError, "outside"):
                d.output_path(d.OUTPUT)

    def test_state_failure_precedes_numerical_gate(self):
        with patch.object(d.native.candidate, "_stages", return_value=(stage for stage in (0,))), \
                patch.object(d.retained, "check_stage_state", side_effect=ValueError("state")), \
                patch.object(d.gates, "evaluate_decoder_stage") as evaluate:
            with self.assertRaisesRegex(ValueError, "state"):
                d.drive_layer({}, 22, {}, {}, None, {}, {}, [])
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
