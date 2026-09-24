"""Focused frozen-boundary and synthetic orchestration checks; no native replay."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_l15_l21_sparse_cut_suffix_closure_v1 as d


class ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = d.base.prior.BoundInputs()
        cls.documents = d.pinned_documents(cls.inputs)
        freeze = cls.inputs.read(d.retained.record(d.base.margin.INPUT / "freeze.json"))
        cls.extension = cls.inputs.read(freeze["reference_extension"])
        cls.cuts, cls.trajectories, cls.references = {}, {}, {}
        for layer, suffix, margin in ((15, "L13_suffix", "L15_margin"),
                                      (21, "L15_suffix", "L21_margin")):
            entry = cls.documents[suffix]["layers"][-1]
            arrays = cls.inputs.archive(entry["actual_stages"])
            cls.cuts[layer] = (d.retained.state_from(arrays, "output", "stage18"),
                               cls.documents[margin]["minimal_residual_cut"])
            item = cls.extension["layers"][str(layer)]
            cls.trajectories[layer] = cls.inputs.archive(item["fp16"])
            cls.references[layer] = d.base.global_reference(cls.inputs, item)
        cls.parent = cls.inputs.archive(
            cls.documents["L13_suffix"]["layers"][0]["input_state_evidence"])

    def test_contract(self):
        d.check_contract(json.loads(d.CONTRACT.read_text()))

    def test_contract_mutations(self):
        original = json.loads(d.CONTRACT.read_text())
        for key, value in (("layers", [14]), ("focused_tests", 19),
                           ("excess_budget", "1/4"), ("reference_policy", "actual"),
                           ("cuts", {}), ("upstream", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(dict(original, **{key: value}))

    def test_flags(self):
        d.check_flags(d.FLAGS)
        for key in d.FLAGS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_flags(dict(d.FLAGS, **{key: True}))

    def test_pins(self):
        for path, digest in d.UPSTREAM.values():
            self.assertEqual(d.retained.record(d.ROOT / path)["sha256"], digest)

    def test_origin_escape(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            d.base.prior.BoundInputs().bind(
                {"path": "/outside/source.py", "bytes": 1, "sha256": "wrong"})

    def test_source_mismatch(self):
        record = d.retained.record(d.SOURCE)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            d.base.prior.BoundInputs().bind(dict(record, sha256="wrong"))

    def test_historical_source_evolution(self):
        inputs, historical = d.base.prior.BoundInputs(), {}
        for name in ("L20_endpoints", "L20_interior"):
            for record in self.documents[name]["authenticated_inputs"]:
                if Path(record["path"]) in d.HISTORICAL_L20_SOURCES:
                    d.bind_l20_record(inputs, record, historical)
        self.assertEqual(set(historical), {str(path) for path in d.HISTORICAL_L20_SOURCES})
        for path, row in historical.items():
            self.assertEqual(row["current"], d.retained.record(path))
            self.assertEqual(inputs.records[path], row["current"])
        changed = historical[str(d.interior.v1.SOURCE)]
        self.assertNotEqual(changed["historical"]["sha256"], changed["current"]["sha256"])
        self.assertEqual(changed["historical"]["sha256"],
                         "87ca9f9bc2956709fb7e7d1c2b6b129d85c4c70e6b113a216396f3e4719f2979")

    def test_historical_origin_escape(self):
        record = d.retained.record(d.interior.v1.SOURCE)
        with self.assertRaisesRegex(ValueError, "outside"):
            d.bind_l20_record(d.base.prior.BoundInputs(),
                             dict(record, path="/outside/source.py"), {})
        with patch.object(Path, "resolve", return_value=Path("/outside/source.py")):
            with self.assertRaisesRegex(ValueError, "outside"):
                d.bind_l20_record(d.base.prior.BoundInputs(), record, {})

    def test_historical_non_source_mismatch(self):
        record = d.retained.record(d.ROOT / d.UPSTREAM["L20_endpoints"][0])
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            d.bind_l20_record(d.base.prior.BoundInputs(), dict(record, sha256="wrong"), {})
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            d.bind_l20_record(d.base.prior.BoundInputs(),
                             dict(d.retained.record(d.SOURCE), sha256="wrong"), {})

    def test_historical_module_origin_mismatch(self):
        for name, record in self.documents["L20_interior"]["validation"]["origins"].items():
            d.check_historical_origin(name, record)
        with self.assertRaisesRegex(ValueError, "origin mismatch"):
            d.check_historical_origin(d.MODULE, d.retained.record(d.interior.v1.SOURCE))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            d.check_historical_origin("outside.module", d.retained.record(d.SOURCE))

    def test_current_source_binding_preserved(self):
        inputs, historical = d.base.prior.BoundInputs(), {}
        record = next(row for row in self.documents["L20_interior"]["authenticated_inputs"]
                      if row["path"] == str(d.interior.v1.SOURCE))
        d.bind_l20_record(inputs, record, historical)
        with self.assertRaisesRegex(ValueError, "conflicting binding"):
            inputs.bind(record)
        with self.assertRaisesRegex(ValueError, "conflicting historical"):
            d.bind_l20_record(inputs, dict(record, sha256="wrong"), historical)

    def test_output_scope(self):
        for path in (d.ROOT / "result.json", d.ROOT / "build/other_attempt001"):
            with self.assertRaisesRegex(ValueError, "scope"):
                d.output_path(path)
        with self.assertRaisesRegex(ValueError, "already exists"):
            with patch.object(Path, "exists", return_value=True):
                d.output_path(d.OUTPUT)

    def test_history(self):
        d.check_history(self.documents)

    def test_history_not_relabelled(self):
        changed = dict(self.documents)
        changed["historical_attempt002"] = dict(
            self.documents["historical_attempt002"], status="PASS")
        with self.assertRaisesRegex(ValueError, "relabeled"):
            d.check_history(changed)

    def test_original_reference_chain(self):
        d.base.check_reference_suffix(self.extension)
        for layer in d.LAYERS:
            changed = copy.deepcopy(self.extension)
            changed["layers"][str(layer)]["input_binary64"] = "substituted"
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                d.base.check_reference_suffix(changed)

    def test_frozen_l13_global_gate(self):
        item = self.extension["layers"]["13"]
        report = d.global_gate(13, self.parent, self.inputs.archive(item["fp16"]),
                               d.base.global_reference(self.inputs, item))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(int(self.parent["i"][62]), 26566721535)

    def boundary(self, layer, failure=None):
        return d.apply_boundary(layer, self.cuts[layer][0], failure, self.cuts,
                                self.trajectories[layer], self.references[layer])

    def test_l15_boundary(self):
        failure, parent, report = self.boundary(15)
        self.assertIsNone(failure)
        self.assertEqual(int(parent["i"][62]), 26617053184)
        self.assertEqual(int(parent["h"][62]), 0x6632)
        self.assertEqual(report["status"], "PASS")

    def test_l21_boundary(self):
        failure, parent, report = self.boundary(21)
        self.assertIsNone(failure)
        self.assertEqual(int(parent["i"][62]), 1602748417)
        self.assertEqual(int(parent["h"][62]), 0x55F9)
        self.assertEqual(report["status"], "PASS")

    def test_raw_arrays_unchanged(self):
        for layer in (15, 21):
            raw = copy.deepcopy(self.cuts[layer][0])
            _, changed, _ = self.boundary(layer)
            for key in raw:
                self.assertTrue(np.array_equal(self.cuts[layer][0][key], raw[key]))
                self.assertFalse(np.shares_memory(changed[key], self.cuts[layer][0][key]))
            self.assertTrue(np.array_equal(changed["z"], raw["z"]))
            for key in ("i", "h"):
                self.assertEqual(np.flatnonzero(changed[key] != raw[key]).tolist(), [62])

    def test_changed_cut_rejected(self):
        for layer in (15, 21):
            parent, cut = self.cuts[layer]
            cuts = dict(self.cuts)
            cuts[layer] = (parent, dict(cut, delta_Q24_units=cut["delta_Q24_units"] + 1))
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                d.apply_boundary(layer, parent, None, cuts,
                                 self.trajectories[layer], self.references[layer])

    def test_spliced_raw_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "spliced"):
            d.apply_boundary(15, self.parent, None, self.cuts,
                             self.trajectories[15], self.references[15])

    def test_expected_raw_failure_preserved(self):
        raw_failure = {"node": [15, 0, 18], "index": 62, "gate": "binary64_v1"}
        saved = copy.deepcopy(raw_failure)
        failure, _, report = self.boundary(15, raw_failure)
        self.assertIsNone(failure)
        self.assertEqual(raw_failure, saved)
        self.assertEqual(report["status"], "PASS")

    def test_unplanned_failure_stops_before_cut(self):
        for stage, index in ((17, 62), (18, 61)):
            failure = {"node": [15, 0, stage], "index": index, "gate": "binary64_v1"}
            with patch.object(d.l15.preflight, "minimal_cut") as cut:
                actual, _, report = self.boundary(15, failure)
            self.assertEqual(actual, failure)
            self.assertIsNone(report)
            cut.assert_not_called()
        failure = {"node": [20, 0, 18], "index": 62, "gate": "binary64_v1"}
        self.assertEqual(d.apply_boundary(20, self.parent, failure, self.cuts, {}, None)[0],
                         failure)

    def test_adjusted_global_failure_stops(self):
        with patch.object(d, "global_gate", return_value={"status": "FAIL"}), \
                patch.object(d.native, "first_failure_index", return_value=7):
            failure, _, _ = self.boundary(21)
        self.assertEqual(failure, {"node": [21, 0, 18], "index": 7, "gate": "binary64_v1"})

    def test_native_scope_and_sequence(self):
        audit = {"native_layers": []}
        with patch.object(d.native.candidate, "_stages", side_effect=lambda *a: iter([0])) as raw:
            with d.base.suffix_only(audit):
                for layer in (*range(14), 24, True):
                    with self.subTest(layer=layer), self.assertRaises(ValueError):
                        next(d.native.candidate._stages({}, layer, {}, {}))
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    next(d.native.candidate._stages({}, 15, {}, {}))
                for layer in d.LAYERS:
                    self.assertEqual(list(d.native.candidate._stages({}, layer, {}, {})), [0])
                with self.assertRaises(ValueError):
                    next(d.native.candidate._stages({}, 23, {}, {}))
        self.assertEqual(raw.call_count, 10)
        self.assertEqual(audit["native_layers"], list(d.LAYERS))

    def test_external_dispatch_forbidden(self):
        with d.base.suffix_only({"native_layers": []}):
            for function in (d.base.subprocess.Popen, d.base.os.system, d.native.run, d.native.stages):
                with self.assertRaises(RuntimeError):
                    function()

    def test_empty_own_kv(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        d.l15.preflight.check_boundary(self.cuts[15][0], kv, self.extension)
        for kind in kv:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                d.l15.preflight.check_boundary(
                    self.cuts[15][0], dict(kv, **{kind: np.zeros((1, 128), dtype="<u2")}),
                    self.extension)

    def test_walk_sequential_state(self):
        visited, expected = [], [self.parent]

        def step(layer, parent):
            self.assertIs(parent, expected[0])
            visited.append(layer)
            changed = {key: value.copy() for key, value in parent.items()}
            expected[0] = changed
            return None, changed

        self.assertIsNone(d.walk(self.parent, step))
        self.assertEqual(visited, list(d.LAYERS))

    def test_walk_first_failure(self):
        for stop in d.LAYERS:
            visited = []
            failure = {"node": [stop, 0, 17], "index": 7, "gate": "local_operator_fp16"}

            def step(layer, parent):
                visited.append(layer)
                return (failure, None) if layer == stop else (None, parent)

            self.assertEqual(d.walk(self.parent, step), failure)
            self.assertEqual(visited, list(range(14, stop + 1)))


if __name__ == "__main__":
    unittest.main()
