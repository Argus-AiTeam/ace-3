"""Focused cut, provenance and fail-closed boundary tests; no native execution."""

import copy
from fractions import Fraction
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l15_cut_l16_suffix_preflight_v1 as d


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.command = json.loads((d.INPUT / "command.json").read_text())
        with np.load(d.margin.INPUT / "layer15/actual_stages.npz", allow_pickle=False) as data:
            cls.arrays = {key: data[key].copy() for key in data.files}
        cls.parent = d.retained.state_from(cls.arrays, "output", "stage18")
        cls.cut = cls.document["minimal_residual_cut"]
        freeze = json.loads((d.margin.margin.INPUT / "freeze.json").read_text())
        cls.extension = json.loads(Path(freeze["reference_extension"]["path"]).read_text())

    def test_corrected_evidence(self):
        d.check_margin(self.document, self.validation, self.command)

    def test_wrong_evidence_selection(self):
        for key, value in (("status", "BLOCKED"), ("node", [13, 0, 18]),
                           ("index", 61), ("retained_status", "PASS"),
                           ("upstream_sha256", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_margin(dict(self.document, **{key: value}), self.validation, self.command)

    def test_non_admitting_flags(self):
        for key, value in d.FLAGS.items():
            changed = dict(self.document)
            changed[key] = True if type(value) is not bool else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_flags(changed)

    def test_validation_counts(self):
        for key in ("collected", "executed", "failures", "errors", "skipped"):
            changed = dict(self.validation, **{key: 1})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "counts"):
                d.check_margin(self.document, changed, self.command)

    def test_command_context(self):
        for key, value in (("cwd", "/outside"), ("executable", "/usr/bin/python"),
                           ("PYTHONPATH", ""), ("dont_write_bytecode", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "context"):
                d.check_margin(self.document, self.validation, dict(self.command, **{key: value}))

    def test_evidence_linkage(self):
        changed = copy.deepcopy(self.document)
        changed["validation"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "linkage"):
            d.check_margin(changed, self.validation, self.command)

    def test_minimal_cut(self):
        changed = d.minimal_cut(self.parent, self.cut)
        self.assertEqual(int(changed["i"][62]), 26617053184)
        self.assertEqual(int(changed["h"][62]), 0x6632)
        self.assertEqual(int(self.parent["i"][62]), 26617418751)
        self.assertEqual(int(self.parent["h"][62]), 0x6633)
        self.assertEqual(np.flatnonzero(changed["i"] != self.parent["i"]).tolist(), [62])
        self.assertEqual(np.flatnonzero(changed["h"] != self.parent["h"]).tolist(), [62])
        self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
        for key in changed:
            self.assertFalse(np.shares_memory(changed[key], self.parent[key]))

    def test_tie_and_adjacent_unit(self):
        changed = d.minimal_cut(self.parent, self.cut)
        integer = int(changed["i"][62])
        self.assertEqual(Fraction(integer, d.Q), Fraction(3173, 2))
        reference = Fraction(self.document["exact_coordinate_decomposition"]["original_L15_reference"])
        passing = d.margin.margin.scalar(d.rational.project(integer, 0), reference)
        failing = d.margin.margin.scalar(d.rational.project(integer + 1, 0), reference)
        self.assertTrue(passing["accepted"])
        self.assertFalse(failing["accepted"])
        self.assertEqual(passing["excess_budget"], "1/8")

    def test_changed_cut_rejected(self):
        for key, value in (("delta_Q24_units", -365566), ("delta_Q24_units", -365568),
                           ("delta_Q24_units", -365567.0), ("delta", "-1/3"),
                           ("output_word", "6631"), ("continuous_cut_is_strict", True),
                           ("baseline_Q24_integer", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                d.minimal_cut(self.parent, dict(self.cut, **{key: value}))

    def test_parent_shapes_and_types(self):
        for key, value in (("i", self.parent["i"][:895]),
                           ("i", self.parent["i"].astype(np.float64)),
                           ("z", self.parent["z"].astype(np.int64)),
                           ("h", self.parent["h"].astype(np.float16))):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "shape/dtype"):
                d.minimal_cut(dict(self.parent, **{key: value}), self.cut)
        with self.assertRaisesRegex(ValueError, "paired"):
            d.minimal_cut({"h": self.parent["h"]}, self.cut)

    def test_parent_projection_and_frozen_identity(self):
        changed = {key: value.copy() for key, value in self.parent.items()}
        changed["h"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "view mismatch"):
            d.minimal_cut(changed, self.cut)
        changed = {key: value.copy() for key, value in self.parent.items()}
        changed["i"][62] += 1
        with self.assertRaisesRegex(ValueError, "minimal cut"):
            d.minimal_cut(changed, self.cut)

    def test_empty_p0_boundary(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        d.check_boundary(d.minimal_cut(self.parent, self.cut), kv, self.extension)

    def test_nonempty_or_changed_kv_rejected(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        for kind in ("k", "v"):
            for bad in (self.arrays["output_cache_" + kind],
                        np.empty((0, 128), dtype="<f2"), np.empty((0, 127), dtype="<u2")):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    d.check_boundary(self.parent, dict(kv, **{kind: bad}), self.extension)

    def test_original_reference_reanchoring_rejected(self):
        kv = {kind: np.empty((0, 128), dtype="<u2") for kind in ("k", "v")}
        for key in ("input_binary64", "input_fp16", "prior_kv"):
            changed = copy.deepcopy(self.extension)
            changed["layers"]["16"][key] = "counterfactual parent"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "recurrence"):
                d.check_boundary(self.parent, kv, changed)
        changed = dict(self.extension, reference_only=False)
        with self.assertRaisesRegex(ValueError, "re-anchored"):
            d.check_boundary(self.parent, kv, changed)

    def test_external_origin_rejected_before_read(self):
        with patch.object(d.retained, "record") as record:
            with self.assertRaisesRegex(ValueError, "outside"):
                d.margin.prior.BoundInputs().bind(
                    {"path": "/outside/source.py", "bytes": 1, "sha256": "x"})
            record.assert_not_called()
        signature = d.retained.record(d.SOURCE)
        with self.assertRaisesRegex(ValueError, "module origin"):
            d.bind_origins(d.margin.prior.BoundInputs(), {"ace3.model.wrong": signature})

    def test_pins_and_closure_tampering(self):
        for name, digest in d.PINS.items():
            with self.subTest(name=name):
                record = d.retained.record(d.INPUT / name)
                self.assertEqual(record["sha256"], digest)
                with patch.object(d.retained, "record", return_value=dict(record, sha256="wrong")):
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        d.margin.margin.pinned(d.margin.prior.BoundInputs(), d.INPUT / name, digest)
        inputs = d.margin.prior.BoundInputs()
        signature = d.retained.record(d.SOURCE)
        inputs.bind(signature)
        self.assertEqual(d.margin.bind_closure(inputs, dict(signature, annotation=True)), d.SOURCE)
        with self.assertRaisesRegex(ValueError, "closure"):
            d.margin.bind_closure(inputs, dict(signature, bytes=signature["bytes"] + 1))

    def test_repository_origins(self):
        importlib.import_module(d.margin.prior.LEGACY_TEST)
        records = d.origins()
        self.assertEqual(records[d.MODULE]["path"], str(d.SOURCE))
        self.assertIn(d.TEST_MODULE, records)
        with patch.object(d, "__file__", "/outside/source.py"):
            with self.assertRaisesRegex(ValueError, "origin"):
                d.origins()

    def test_contract(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        for key, value in (("excess_budget", "1/4"), ("prospective_next_layer", 17),
                           ("upstream_sha256", {}), ("focused_tests", 19)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(dict(contract, **{key: value}))

    def test_no_execution_or_state_publication(self):
        with d.no_execution():
            for function in (d.margin.upstream.native.candidate._stages, d.retained.save,
                             d.margin.margin.subprocess.Popen, d.margin.margin.os.system):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function()

    def test_output_scope_and_preservation(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            d.output_path(d.ROOT / "build" / "other")
        with self.assertRaisesRegex(ValueError, "outside"):
            d.output_path(d.OUTPUT / "child" / "grandchild")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "resolve", return_value=d.ROOT / "elsewhere"):
            with self.assertRaisesRegex(ValueError, "outside"):
                d.output_path(d.OUTPUT / "reviewer001")


if __name__ == "__main__":
    unittest.main()
