"""Focused evidence, coordinate-cut and bounded search-control checks."""

import copy
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_to_l21_coordinate62_input_sensitivity_v1 as d


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.producer = json.loads(d.PRODUCER_INPUT.read_text())
        cls.margin = json.loads((d.suffix.INPUT / "result.json").read_text())
        cls.frozen = json.loads((d.margin.INPUT / "result.json").read_text())
        cls.suffix = json.loads((d.suffix.OUTPUT / "result.json").read_text())
        record = cls.frozen["layers"][-1]["actual_stages"]
        with np.load(record["path"], allow_pickle=False) as archive:
            cls.arrays = {key: archive[key].copy() for key in archive.files}
        cls.parent = d.retained.state_from(cls.arrays, "input", "input_hidden")

    def test_contract(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check", "--execute"])
        self.assertFalse(self.contract["scientific_result_claim"])

    def test_changed_contract(self):
        for key, value in (("excess_budget", "3/4"), ("policy_id", "other"),
                           ("native_layers", [21]), ("focused_tests", 19),
                           ("native_L0_L8_invocations", False),
                           ("normal_host_review", "OPTIONAL")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for key in d.producer.LOCAL_GATE:
            changed = copy.deepcopy(self.contract)
            changed["local_gate"][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)
        changed = copy.deepcopy(self.contract)
        changed["search"]["coordinate"] = 63
        with self.assertRaises(ValueError):
            d.check_contract(changed)

    def test_context(self):
        self.assertEqual(d.producer.context()["PYTHONPATH"], str(d.ROOT))
        with patch.dict(os.environ, {"PYTHONPATH": "/other"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.producer.context()

    def test_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_origin(self):
        module = ModuleType("ace3.model.candidates.foreign")
        module.__file__ = "/other/foreign.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with patch.object(d, "__file__", "/other/diagnostic.py"):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_pinned_evidence(self):
        inputs = d.margin.margin.prior.BoundInputs()
        with self.assertRaisesRegex(ValueError, "pinned evidence mismatch"):
            d.margin.scalar_math.pinned(inputs, d.PRODUCER_INPUT, "0" * 64)
        record = d.retained.record(d.PRODUCER_INPUT)
        self.assertEqual(record["sha256"], d.PRODUCER_SHA256)

    def test_source_binding(self):
        inputs = d.margin.margin.prior.BoundInputs()
        record = d.retained.record(d.SOURCE)
        inputs.bind(record)
        for key, value in (("sha256", "0" * 64), ("bytes", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                inputs.bind({**record, key: value})

    def test_producer_evidence(self):
        d.check_producer(self.producer, self.frozen, self.margin)
        self.assertEqual(self.producer["status"], "INCONCLUSIVE")
        for key, value in (("status", "PASS_BOUNDARY"), ("native_layers", [20] * 11),
                           ("candidate_admitted", True), ("lineage", {}),
                           ("original_reference", {}), ("parent_binding", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_producer({**self.producer, key: value}, self.frozen, self.margin)

    def test_producer_controls(self):
        for controls in (self.producer["controls"][:-1],
                         list(reversed(self.producer["controls"]))):
            with self.assertRaises(ValueError):
                d.check_producer({**self.producer, "controls": controls}, self.frozen, self.margin)
        changed = copy.deepcopy(self.producer)
        changed["controls"][1]["index62"]["excess_budget"] = "3/4"
        with self.assertRaises(ValueError):
            d.check_producer(changed, self.frozen, self.margin)

    def test_validation_counts(self):
        validation = self.producer["validation"]
        d.check_validation(validation, 24)
        for key, value in (("collected", 0), ("executed", 19), ("failures", 1),
                           ("errors", 1), ("skipped", 1), ("skipped", False),
                           ("legacy_19_tests_executed", True), ("upstream_tests_executed", True),
                           ("cwd", "/other")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_validation({**validation, key: value}, 24)

    def test_suffix_evidence(self):
        d.check_suffix(self.suffix, self.margin)
        for key, value in (("status", "PASS"), ("audit", {"native_layers": [21, 22, 23]}),
                           ("successor_published", True), ("reference_policy", "actual-derived"),
                           ("minimal_residual_cut", {}), ("frozen_lineage", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_suffix({**self.suffix, key: value}, self.margin)

    def test_state_operand_kv_gates(self):
        d.margin.margin.check_layer(self.arrays, self.parent)
        for key in ("input_i", "output_i", "output_z", "stage16",
                    "output_cache_k", "output_cache_v"):
            changed = {name: value.copy() for name, value in self.arrays.items()}
            changed[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.margin.margin.check_layer(changed, self.parent)

    def test_original_reference(self):
        entry = self.frozen["layers"][-1]
        changed = copy.deepcopy(entry["original_reference"])
        changed["input_binary64"] = changed["binary64"]
        with self.assertRaisesRegex(ValueError, "reference substitution"):
            d.margin.check_reference(entry, changed)

    def test_zero_probe(self):
        changed = d.perturb(self.parent, 0)
        for key in self.parent:
            self.assertTrue(np.array_equal(changed[key], self.parent[key]))
            self.assertFalse(np.shares_memory(changed[key], self.parent[key]))

    def test_one_coordinate_probes(self):
        before = {key: value.copy() for key, value in self.parent.items()}
        for delta in (1, -1, 1 << 24, -(1 << 24)):
            changed = d.perturb(self.parent, delta)
            self.assertEqual(int(changed["i"][62]), int(self.parent["i"][62]) + delta)
            self.assertEqual(np.flatnonzero(changed["i"] != self.parent["i"]).tolist(), [62])
            self.assertEqual(int(changed["h"][62]), d.margin.rational.project(
                int(changed["i"][62]), int(changed["z"][62])))
            self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
        for key in before:
            self.assertTrue(np.array_equal(self.parent[key], before[key]))

    def test_invalid_delta(self):
        for delta in (True, 1.0, np.int64(1), "1", (1 << 24) + 1, -(1 << 24) - 1):
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, "integer Q24"):
                d.perturb(self.parent, delta)
        changed = {key: value.copy() for key, value in self.parent.items()}
        changed["h"][62] ^= 1
        with self.assertRaises(ValueError):
            d.perturb(changed, 1)

    def test_search_definition(self):
        points = d.search_points()
        self.assertEqual(len(points), 51)
        self.assertEqual(len(set(points)), 51)
        self.assertEqual(points[0], 0)
        self.assertEqual(points[1::2], tuple(1 << k for k in range(25)))
        self.assertEqual(points[2::2], tuple(-(1 << k) for k in range(25)))
        self.assertEqual(d.SEARCH["prospective_native_layers"], [21])
        self.assertIn("no monotonicity", d.SEARCH["minimality"])
        calls = []

        def failed(delta):
            calls.append(delta)
            return {"delta_Q24_units": delta, "boundary": None}, None

        suffix = unittest.mock.Mock(return_value={"status": "PASS"})
        exhausted = d.search(failed, suffix)
        self.assertEqual(calls, list(points))
        self.assertEqual(exhausted["reason"], "SEARCH_EXHAUSTED")
        self.assertTrue(exhausted["search_exhausted"])
        suffix.assert_not_called()
        for stop in ("L20_GLOBAL_GATE_BOUNDARY", "LOCAL_GATE_BOUNDARY", "GLOBAL_GATE_BOUNDARY"):
            with self.subTest(stop=stop):
                blocked = d.search(lambda delta: ({"delta_Q24_units": delta, "boundary": stop},
                                                 None), suffix)
                self.assertEqual(len(blocked["samples"]), 1)
                self.assertEqual(blocked["reason"], stop)
                self.assertFalse(blocked["search_exhausted"])
                suffix.assert_not_called()
        for passing in (1, -1, 4, -4):
            calls.clear()

            def probe(delta):
                calls.append(delta)
                return {"delta_Q24_units": delta,
                        "boundary": "PASS_BOUNDARY" if delta == passing else None}, "parent"

            result = d.search(probe, suffix)
            self.assertEqual(result["one_unit_less"]["delta_Q24_units"],
                             passing - (1 if passing > 0 else -1))
            self.assertEqual(result["minimality_proven"], abs(passing) == 1)
            self.assertEqual(result["status"], "BOUNDED_INPUT_ADJUSTMENT_PASS"
                             if abs(passing) == 1 else "INCONCLUSIVE")
            suffix.assert_called_with("parent")
        with self.assertRaisesRegex(ValueError, "baseline unexpectedly passed"):
            d.search(lambda delta: ({"delta_Q24_units": delta, "boundary": "PASS_BOUNDARY"},
                                   "parent"), suffix)
        suffix.return_value = {"status": "FAIL"}
        result = d.search(lambda delta: ({"delta_Q24_units": delta,
                                         "boundary": "PASS_BOUNDARY" if delta == 1 else None},
                                        "parent"), suffix)
        self.assertFalse(result["minimality_proven"])
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_no_execution(self):
        with d.no_execution():
            for layer in (*range(9), 20, 21, 22, 23):
                with self.subTest(layer=layer), self.assertRaisesRegex(RuntimeError, "forbidden"):
                    d.producer.native.candidate._stages(layer=layer)
            for function, args in ((subprocess.Popen, (["verilator"],)),
                                   (os.system, ("verilator",)), (d.retained.save, ())):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    function(*args)
        for layer in (*range(21), 24, True):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L21-L23"):
                with d.native_only([], layer):
                    self.fail("forbidden native layer admitted")
        for layer in (21, 22, 23):
            audit = []
            with patch.object(d.producer.native.candidate, "_stages",
                              return_value=iter([0])) as raw:
                with d.native_only(audit, layer):
                    self.assertEqual(list(d.producer.native.candidate._stages(
                        {}, layer, {}, {})), [0])
                    with self.assertRaisesRegex(ValueError, "dispatch mismatch"):
                        next(d.producer.native.candidate._stages({}, 8, {}, {}))
                    with self.assertRaisesRegex(RuntimeError, "forbidden"):
                        subprocess.Popen(["verilator"])
                self.assertEqual(audit, [layer])
                raw.assert_called_once()
        with d.native_only([21] * 54, 21):
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                next(d.producer.native.candidate._stages({}, 21, {}, {}))

    def test_check_only_interface(self):
        for argv in ([], ["--execute", "--check"], ["--check", "--out", "existing"],
                     ["--check", "--layer", "8"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    d.main(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_blocked_is_not_success(self):
        output = io.StringIO()
        with patch.object(d, "preflight", side_effect=ValueError("binding mismatch")):
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["scientific_result_claim"])
        self.assertIn("binding mismatch", result["error"])


if __name__ == "__main__":
    unittest.main()
