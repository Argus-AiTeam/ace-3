"""Focused dependency, cut, no-replay and gate-stop tests; no native computation."""

import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l15_cut_l16_suffix_v1 as d


class SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.command = json.loads((d.INPUT / "command.json").read_text())
        path = cls.document["dependency"]["frozen_L15_actual"]["path"]
        with np.load(path, allow_pickle=False) as data:
            cls.arrays = {key: data[key].copy() for key in data.files}
        cls.parent = d.retained.state_from(cls.arrays, "output", "stage18")
        cls.cut = cls.document["minimal_residual_cut"]
        freeze = json.loads((d.upstream.margin.INPUT / "freeze.json").read_text())
        cls.extension = json.loads(Path(freeze["reference_extension"]["path"]).read_text())

    def test_accepted_preflight(self):
        d.check_preflight(self.document, self.validation, self.command)

    def test_wrong_dependency(self):
        for key, value in (("status", "BLOCKED"), ("node", [13, 0, 18]),
                           ("index", 61), ("prospective_next_layer", 15),
                           ("upstream_sha256", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_preflight(dict(self.document, **{key: value}),
                                  self.validation, self.command)

    def test_dependency_counts(self):
        for key in ("collected", "executed", "errors", "failures", "skipped"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "counts"):
                d.check_preflight(self.document, dict(self.validation, **{key: 1}), self.command)

    def test_dependency_context_and_linkage(self):
        for key, value in (("cwd", "/outside"), ("executable", "/usr/bin/python"),
                           ("PYTHONPATH", ""), ("dont_write_bytecode", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "context"):
                d.check_preflight(self.document, self.validation, dict(self.command, **{key: value}))
        changed = copy.deepcopy(self.document)
        changed["validation"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "linkage"):
            d.check_preflight(changed, self.validation, self.command)

    def test_nonadmission_flags(self):
        for key in d.preflight.FLAGS:
            changed = dict(self.document, **{key: True})
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_preflight(changed, self.validation, self.command)

    def test_pinned_inputs(self):
        for name, digest in d.PINS.items():
            with self.subTest(name=name):
                signature = d.retained.record(d.INPUT / name)
                self.assertEqual(signature["sha256"], digest)
                with patch.object(d.retained, "record", return_value=dict(signature, sha256="wrong")):
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        d.upstream.margin.pinned(d.upstream.prior.BoundInputs(), d.INPUT / name, digest)

    def test_exact_cut_and_no_mutation(self):
        original = copy.deepcopy(self.parent)
        changed = d.preflight.minimal_cut(self.parent, self.cut)
        self.assertEqual(int(changed["i"][62]), 26617053184)
        self.assertEqual(int(changed["h"][62]), 0x6632)
        self.assertEqual(np.flatnonzero(changed["i"] != self.parent["i"]).tolist(), [62])
        self.assertEqual(np.flatnonzero(changed["h"] != self.parent["h"]).tolist(), [62])
        self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
        for key in self.parent:
            self.assertTrue(np.array_equal(self.parent[key], original[key]))
            self.assertFalse(np.shares_memory(changed[key], self.parent[key]))
        self.assertEqual(d.preflight.rational.project(int(changed["i"][62]) + 1, 0), 0x6633)

    def test_changed_cut_or_fp16_only_parent(self):
        with self.assertRaisesRegex(ValueError, "paired"):
            d.preflight.minimal_cut({"h": self.parent["h"]}, self.cut)
        for delta in (-365566, -365568, -365567.0):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                d.preflight.minimal_cut(self.parent, dict(self.cut, delta_Q24_units=delta))

    def test_cut_full_original_global_gate(self):
        changed = d.preflight.minimal_cut(self.parent, self.cut)
        item = self.extension["layers"]["15"]
        inputs = d.upstream.prior.BoundInputs()
        reference = d.upstream.global_reference(inputs, item)
        trajectory = inputs.archive(item["fp16"])["stage18"]
        reports = [d.gates.evaluate_decoder_stage(
            stage=18, actual=parent["h"], reference=trajectory, policy=d.gates.POLICY_ID,
            local_reference=None, reference_binary64=reference)
            for parent in (self.parent, changed)]
        self.assertEqual([report["status"] for report in reports], ["FAIL", "PASS"])
        self.assertTrue(reports[1]["binary64_v1"]["passed"])
        self.assertEqual(reports[0]["binary64_v1"]["failures"][0]["index"], 62)
        self.assertEqual(reports[0]["binary64_v1"]["failures"][0]["excess_budget"], "1/8")

    def test_reference_reanchoring(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        for key in ("input_binary64", "input_fp16", "prior_kv"):
            changed = copy.deepcopy(self.extension)
            changed["layers"]["16"][key] = "cut reference"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recurrence"):
                d.preflight.check_boundary(self.parent, kv, changed)

    def test_own_empty_kv(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        d.preflight.check_boundary(self.parent, kv, self.extension)
        for kind in kv:
            for value in (self.arrays["output_cache_" + kind],
                          np.empty((0, 128), dtype="<f2")):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    d.preflight.check_boundary(self.parent, dict(kv, **{kind: value}), self.extension)

    def test_native_scope_guard(self):
        audit = {"native_layers": []}
        with d.suffix_only(audit):
            for layer in (*range(16), 24, True):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L16-L23"):
                    next(d.native.candidate._stages({}, layer, {}, {}))
                with self.assertRaisesRegex(ValueError, "L16-L23"):
                    d.drive_layer({}, layer, {}, {}, None, {}, {}, [])
        self.assertEqual(audit["native_layers"], [])

    def test_sequential_dispatch(self):
        audit = {"native_layers": []}
        with patch.object(d.native.candidate, "_stages", return_value=iter([0])) as producer:
            with d.suffix_only(audit):
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    next(d.native.candidate._stages({}, 17, {}, {}))
                self.assertEqual(list(d.native.candidate._stages({}, 16, {}, {})), [0])
            producer.assert_called_once()
        self.assertEqual(audit["native_layers"], [16])

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
            failure, _ = d.drive_layer({}, 16, {}, {
                f"stage{stage:02d}": np.zeros(896, dtype="<u2") for stage in range(19)},
                np.zeros(896), arrays, refs, reports)
        self.assertEqual(visited, list(range(stop + 1)))
        self.assertEqual(closed, [True])
        self.assertEqual(len(reports), stop + 1)
        self.assertEqual(failure["node"], [16, 0, stop])
        self.assertEqual(failure["gate"], "binary64_v1" if stop == 18 else "local_operator_fp16")

    def test_first_local_failure_stops_producer(self):
        self.check_stop(0)

    def test_global_failure_stops_producer(self):
        self.check_stop(18)

    def test_repository_origins(self):
        importlib.import_module(d.upstream.prior.LEGACY_TEST)
        records = d.origins()
        self.assertEqual(records[d.MODULE]["path"], str(d.SOURCE))
        self.assertIn(d.TEST_MODULE, records)
        with patch.object(d, "__file__", "/outside/source.py"):
            with self.assertRaisesRegex(ValueError, "origin"):
                d.origins()

    def test_outside_binding_rejected(self):
        with patch.object(d.retained, "record") as record:
            with self.assertRaisesRegex(ValueError, "outside"):
                d.upstream.prior.BoundInputs().bind(
                    {"path": "/outside/source.py", "sha256": "wrong", "bytes": 1})
            record.assert_not_called()

    def test_contract(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        for key, value in (("excess_budget", "1/4"), ("layers", [15, 16]),
                           ("preflight_sha256", {}), ("focused_tests", 19)):
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


if __name__ == "__main__":
    unittest.main()
