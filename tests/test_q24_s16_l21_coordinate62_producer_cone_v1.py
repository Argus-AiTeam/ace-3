"""Focused preflight/control checks; never execute native ancestors or RTL."""

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

from ace3.model.candidates import diagnose_q24_s16_l21_coordinate62_producer_cone_v1 as d


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((d.INPUT / "result.json").read_text())
        cls.validation = json.loads((d.INPUT / "validation.json").read_text())
        cls.command = json.loads((d.INPUT / "command.json").read_text())
        cls.frozen = json.loads((d.margin.INPUT / "result.json").read_text())
        cls.contract = json.loads(d.CONTRACT.read_text())
        with np.load(d.margin.INPUT / "layer21/actual_stages.npz",
                     allow_pickle=False) as archive:
            cls.arrays = {key: archive[key].copy() for key in archive.files}
        cls.parent = d.retained.state_from(cls.arrays, "input", "input_hidden")

    def test_contract(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check", "--execute"])
        self.assertEqual(self.contract["native_layers"], [])

    def test_changed_contract_rejected(self):
        for key, value in (("excess_budget", "3/4"), ("policy_id", "other"),
                           ("native_layers", [9]), ("native_L0_L8_invocations", False),
                           ("normal_host_review", "OPTIONAL"), ("focused_tests", 19)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for key in d.LOCAL_GATE:
            changed = copy.deepcopy(self.contract)
            changed["local_gate"][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_repo_context(self):
        self.assertEqual(d.context()["PYTHONPATH"], str(d.ROOT))
        with patch.dict(os.environ, {"PYTHONPATH": "/different-checkout"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.context()

    def test_repository_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.margin.margin.prior.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_origins_rejected(self):
        foreign = ModuleType("ace3.model.candidates.foreign")
        foreign.__file__ = "/different-checkout/foreign.py"
        with patch.dict(sys.modules, {foreign.__name__: foreign}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()
        with patch.object(d, "__file__", "/different-checkout/diagnostic.py"):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_selected_margin_evidence(self):
        d.dependency.check_margin(self.document, self.validation, self.command)
        self.assertEqual(self.document["baseline"]["excess_error"], "3/4")
        self.assertEqual(self.document["minimal_residual_cut"]["delta_Q24_units"], 10103041)

    def test_pin_mismatch_rejected(self):
        inputs = d.margin.margin.prior.BoundInputs()
        with self.assertRaisesRegex(ValueError, "pinned evidence mismatch"):
            d.margin.scalar_math.pinned(inputs, d.INPUT / "result.json", "0" * 64)

    def test_original_reference_and_lineage_rejected(self):
        for key, value in (("reference_policy", "actual-derived"),
                           ("lineage", {**d.LINEAGE, "own_empty_P0_FP16_KV": "FAIL"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.dependency.check_margin({**self.document, key: value},
                                          self.validation, self.command)
        entry = self.frozen["layers"][-1]
        changed = copy.deepcopy(entry["original_reference"])
        changed["input_binary64"] = changed["binary64"]
        with self.assertRaisesRegex(ValueError, "reference substitution"):
            d.margin.check_reference(entry, changed)

    def test_validation_and_admission_rejected(self):
        for key, value in (("collected", 0), ("executed", 19), ("skipped", 1),
                           ("failures", 1), ("errors", 1),
                           ("legacy_19_tests_executed", True),
                           ("upstream_tests_executed", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.dependency.check_margin(self.document,
                                          {**self.validation, key: value}, self.command)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.dependency.check_margin({**self.document, key: True},
                                          self.validation, self.command)

    def test_source_binding_rejected(self):
        record = d.retained.record(d.INPUT / "result.json")
        inputs = d.margin.margin.prior.BoundInputs()
        inputs.bind(record)
        for key, value in (("bytes", 0), ("sha256", "0" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.margin.margin.bind_closure(inputs, {**record, key: value})
        with self.assertRaisesRegex(ValueError, "module origin mismatch"):
            d.margin.upstream.preflight.bind_origins(
                inputs, {d.MODULE: {**record, "path": "/different-checkout/candidate.py"}})

    def test_state_operand_and_kv_rejected(self):
        d.margin.margin.check_layer(self.arrays, self.parent)
        for key in ("input_i", "output_i", "output_z", "stage16", "output_cache_k",
                    "output_cache_v"):
            arrays = {name: value.copy() for name, value in self.arrays.items()}
            arrays[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.margin.margin.check_layer(arrays, self.parent)

    def test_reconstruction_rejected(self):
        failure = self.document["first_failure"]
        analysis = {key: self.document[key] for key in (
            "baseline", "minimal_residual_cut", "projected_full_vector_gate")}
        d.check_reconstruction(self.document, self.frozen, failure, analysis)
        with self.assertRaises(ValueError):
            d.check_reconstruction(self.document, self.frozen, {**failure, "index": 63},
                                   analysis)
        with self.assertRaises(ValueError):
            d.check_reconstruction(self.document, self.frozen, failure,
                                   {**analysis, "baseline": {}})

    def test_native_L0_L8_and_later_forbidden(self):
        with d.no_execution():
            for layer in (*range(9), 9, 21, 23):
                with self.subTest(layer=layer), self.assertRaisesRegex(RuntimeError, "forbidden"):
                    d.margin.upstream.native.candidate._stages(layer=layer)

    def test_rtl_external_and_state_publication_forbidden(self):
        with d.no_execution():
            for function, args in ((subprocess.Popen, (["verilator", "--version"],)),
                                   (os.system, ("verilator --version",)),
                                   (d.retained.save, ())):
                with self.subTest(function=function), self.assertRaisesRegex(
                        RuntimeError, "forbidden"):
                    function(*args)

    def test_exclusive_interface(self):
        for args in ([], ["--out", "existing"], ["--check", "--execute"],
                     ["--check", "--layer", "8"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    d.main(args)
                self.assertEqual(raised.exception.code, 2)

    def test_blocked_is_not_success(self):
        output = io.StringIO()
        with patch.object(d, "preflight", side_effect=ValueError("binding mismatch")):
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("binding mismatch", result["error"])
        self.assertFalse(result["scientific_result_claim"])

    def test_registered_nested_controls(self):
        controls = list(d.controls())
        self.assertEqual(len(controls), 11)
        self.assertEqual(controls[0], ("retained_native", ()))
        self.assertEqual(controls[-1][1], (0, 3, 7, 10, 11, 13, 14, 15, 16, 17))
        for previous, current in zip(controls[1:], controls[2:]):
            self.assertEqual(len(set(current[1]) - set(previous[1])), 1)
            self.assertTrue(set(previous[1]) < set(current[1]))
        with self.assertRaisesRegex(ValueError, "unregistered"):
            d.drive_control({}, {}, {}, None, (18,))

    def test_l21_dispatch_guard_and_budget(self):
        audit = {"native_layers": []}
        with patch.object(d.native.candidate, "_stages", return_value=iter(())) as raw:
            with d.l21_only(audit):
                for layer in (*range(21), 22, 23, True):
                    with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L21"):
                        list(d.native.candidate._stages({}, layer, {}, {}))
                for _ in range(11):
                    list(d.native.candidate._stages({}, 21, {}, {}))
                with self.assertRaisesRegex(ValueError, "budget"):
                    list(d.native.candidate._stages({}, 21, {}, {}))
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    subprocess.Popen(["verilator"])
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    d.retained.save()
            self.assertEqual(raw.call_count, 11)
        self.assertEqual(audit["native_layers"], [21] * 11)

    def test_canonical_s16_preserves_rtz_and_operand_identity(self):
        arrays = {key: self.arrays[key].copy() for key in ("stage14", "stage15")}
        unrounded, words = d.canonical_s16(arrays)
        self.assertEqual(words.dtype, np.dtype("<u2"))
        self.assertTrue(np.array_equal(words, d.margin.margin.prior.rtz_reference(unrounded)))
        arrays["stage16"] = np.full(4864, 0x7C00, dtype="<u2")
        self.assertTrue(np.array_equal(words, d.canonical_s16(arrays)[1]))
        arrays["stage14"][0] = 0x7C00
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            d.canonical_s16(arrays)

    def test_native_baseline_byte_mismatch_rejected(self):
        arrays = {"value": np.array([0.0], dtype="<f8")}
        d.same_arrays(arrays, {"value": arrays["value"].copy()})
        for other in ({}, {"value": np.array([-0.0], dtype="<f8")},
                      {"value": np.array([0.0], dtype="<f4")}):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "mismatch"):
                d.same_arrays(arrays, other)

    def test_first_gate_stops_before_consumer(self):
        visited = []
        def producer(tensors, layer, parent, arrays):
            for stage in range(2):
                visited.append(stage)
                arrays["input_hidden"] = np.zeros(896, dtype="<u2")
                arrays[f"stage{stage:02d}"] = np.zeros(896, dtype="<u2")
                yield stage
        with patch.object(d.native.candidate, "_stages", side_effect=producer), \
                patch.object(d.retained, "check_stage_state"), \
                patch.object(d.local, "local_reference", return_value=np.zeros(896, dtype="<u2")), \
                patch.object(d.gates, "evaluate_decoder_stage", return_value={"stage": 0, "status": "FAIL"}):
            _, reports, _, _ = d.drive_control({}, {}, {"stage00": None}, None, ())
        self.assertEqual(visited, [0])
        self.assertEqual(d.decision(reports), "LOCAL_GATE_BOUNDARY")

    def test_cut_is_gated_before_consumer(self):
        def producer(tensors, layer, parent, arrays):
            arrays.update(input_hidden=np.zeros(896, dtype="<u2"),
                          stage00=np.zeros(896, dtype="<u2"))
            yield 0
            self.fail("consumer executed after failed cut gate")
        expected = np.ones(896, dtype="<u2")
        reference = np.zeros(896, dtype="<f8")
        with patch.object(d.native.candidate, "_stages", side_effect=producer), \
                patch.object(d.retained, "check_stage_state"), \
                patch.object(d.local, "local_reference", return_value=expected), \
                patch.object(d.gates, "evaluate_decoder_stage",
                             return_value={"stage": 0, "status": "FAIL"}) as gate:
            _, _, changes, _ = d.drive_control(
                {}, {}, {"stage00": None}, reference, list(d.controls())[-1][1])
        self.assertTrue(np.array_equal(gate.call_args.kwargs["actual"], expected))
        self.assertEqual(changes[0]["changed_indices"], list(range(896)))
        self.assertIsNone(gate.call_args.kwargs["reference_binary64"])

    def test_control_boundary_classification(self):
        reports = [{"stage": stage, "status": "PASS"} for stage in range(19)]
        self.assertEqual(d.decision(reports), "PASS_BOUNDARY")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            d.decision(reports[:-1])
        reports[-1] = {"stage": 18, "status": "FAIL",
                       "binary64_v1": {"failures": [{"index": 62}]}}
        self.assertIsNone(d.decision(reports))
        reports[-1]["binary64_v1"]["failures"].append({"index": 63})
        self.assertEqual(d.decision(reports), "GLOBAL_GATE_BOUNDARY")
        reports[0]["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "continued"):
            d.decision(reports)

    def test_execute_errors_keep_measured_dispatches(self):
        def fail(audit):
            audit["native_layers"].append(21)
            raise ValueError("operand binding mismatch")
        output = io.StringIO()
        with patch.object(d, "execute", side_effect=fail), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertEqual(d.main(["--execute"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["native_layers"], [21])
        self.assertEqual(result["native_layer_invocations"], 1)
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertFalse(result["candidate_admitted"])


if __name__ == "__main__":
    unittest.main()
