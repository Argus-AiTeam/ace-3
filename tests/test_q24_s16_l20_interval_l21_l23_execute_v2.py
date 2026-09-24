"""Focused interior executor tests; native layer drivers are always synthetic."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_interval_l21_l23_execute_v2 as d


class InteriorExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in (d.interior.TEST_MODULE, d.endpoint.TEST_MODULE, d.v1.TEST_MODULE,
                     d.prepared.TEST_MODULE, d.upstream.TEST_MODULE, d.upstream.LEGACY_TEST):
            __import__(name)
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.frozen, cls.failed, cls.blocked = (
            json.loads((d.INPUT / name).read_text()) for name in d.PINS)
        cls.endpoints = json.loads((d.interior.INPUT / "diagnostic.stdout.json").read_text())
        cls.terminal = json.loads((d.interior.INPUT / "terminal.json").read_text())
        cls.parent = {"i": np.zeros(896, dtype="<i8"), "z": np.zeros(896, dtype="u1"),
                      "h": np.zeros(896, dtype="<u2")}
        cls.parent["i"][62], cls.parent["h"][62] = 26514683648, 0x662C
        cls.parent["z"][0], cls.parent["h"][0] = 1, 0x8000
        cls.trajectory = {"stage18": cls.parent["h"].copy()}
        cls.reference = np.zeros(896, dtype="<f8")
        cls.reference[62] = float.fromhex("0x1.8b12540d033edp+10")
        cls.analysis = d.prepared.analyze(cls.parent, cls.trajectory, cls.reference)

    def result(self):
        return {"status": "BLOCKED", **d.FLAGS, "native_layers": [],
                "probes": d.interior.schedule(self.analysis, self.parent)}

    def driver(self, result, calls, fail=None, mutate=None):
        def run_layer(layer, state, entry):
            calls.append((layer, d.v1.state_binding(state)))
            result["native_layers"].append(layer)
            failed = fail is not None and layer == fail[0]
            last = fail[1] if failed else 18
            for stage in range(last + 1):
                passed = not (failed and stage == last)
                entry["reports"].append({
                    "stage": stage, "node": [layer, 0, stage],
                    "policy_id": d.prepared.gates.POLICY_ID,
                    "residual_state_lineage": "PASS", "kv_lineage": "PASS",
                    "local_reference_independent": stage < 18,
                    "status": "PASS" if passed else "FAIL",
                    "local_operator_fp16" if stage < 18 else "binary64_v1":
                        {"passed": passed, "coordinates": 896},
                })
            entry.update(status="FAIL" if failed else "PASS",
                         failure={"node": [layer, 0, last], "index": 62,
                                  "gate": "binary64_v1" if last == 18 else "local_operator_fp16"}
                         if failed else None)
            output = None if failed else d.prepared.change_coordinate(state, int(state["i"][62]) + 1)
            if mutate is not None:
                mutate(state, entry)
            return output
        return run_layer

    def run_probes(self, result, driver, reference=None):
        d.execute_probes(result, self.parent, self.analysis, self.trajectory,
                         self.reference if reference is None else reference, driver)

    def test_contract_interfaces(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check", "--execute"])
        self.assertEqual(self.contract["reproduction"], d.COMMAND)
        self.assertEqual(self.contract["execution"], d.EXECUTE_COMMAND)

    def test_threshold_scope_and_reference_mutations_rejected(self):
        for key, value in [(key, "changed") for key in d.FLAGS] + [
            ("focused_tests", 19), ("reference_policy", "same-input global"),
            ("excess_budget", "3/4"), ("reproduction", "python -m foreign"),
            ("input_sha256", {}), ("interface", ["--execute"]),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for section, key, value in (("executor", "native_layers", [8, 21]),
                                    ("executor", "probe_count", 16),
                                    ("executor", "maximum_native_layer_invocations", 46),
                                    ("local_gate", "ordered_FP16_ULP", True)):
            changed = copy.deepcopy(self.contract)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_explicit_repo_bound_context(self):
        self.assertEqual(d.upstream.producer.context()["executable"], str(d.PYTHON))
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.upstream.producer.context()
        with patch.object(Path, "cwd", return_value=Path("/foreign")):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.upstream.producer.context()

    def test_required_source_and_test_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.interior.MODULE, d.interior.TEST_MODULE,
                     d.upstream.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_source_origin_rejected(self):
        module = ModuleType("ace3.model.candidates.foreign_interior_executor")
        module.__file__ = "/foreign/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_pinned_digest_checked_before_parsing(self):
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        name = next(iter(d.PINS))
        with patch.object(d.retained, "record", return_value={"sha256": "0" * 64}):
            with patch.object(inputs, "read") as read:
                with self.assertRaisesRegex(ValueError, "pinned evidence"):
                    d.v1.pinned(inputs, d.INPUT / name, d.PINS[name])
                read.assert_not_called()

    def test_frozen_schedule_reference_and_lineage_rejected_if_changed(self):
        with patch.object(d.interior, "schedule", return_value=self.frozen["probes"]):
            d.check_frozen(self.frozen, self.endpoints, self.terminal, self.parent)
            for key in ("probes", "analysis", "parent_binding", "original_L20_reference",
                        "lineage", "baseline_control", "historical_endpoint_execution"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    d.check_frozen({**self.frozen, key: None},
                                   self.endpoints, self.terminal, self.parent)

    def test_historical_failed_check_and_missing_executor_preserved(self):
        d.check_prior(self.failed, self.blocked, self.frozen)
        for key, value in (("status", "PASS"), ("native_layers", [21]), ("candidate_count", 14)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_prior(self.failed, {**self.blocked, key: value}, self.frozen)
        changed = copy.deepcopy(self.failed)
        changed["validation"]["errors"] = 0
        with self.assertRaises(ValueError):
            d.check_prior(changed, self.blocked, self.frozen)

    def test_exact_fifteen_strict_interior_probes(self):
        rows = self.result()["probes"]
        self.assertEqual([row["Q24_integer"] for row in rows],
                         list(range(26500661248, 26515341313, 1048576)))
        self.assertEqual(len({row["state_binding"]["i"]["sha256"] for row in rows}), 15)
        self.assertEqual({row["input_FP16_word"] for row in rows}, {"662c"})
        self.assertTrue({26499612672, 26516389888, 26514683648}.isdisjoint(
            row["Q24_integer"] for row in rows))

    def test_probes_copy_state_preserve_zero_sign_and_change_only_i62(self):
        before = d.v1.state_binding(self.parent)
        for row in self.result()["probes"]:
            state = d.prepared.change_coordinate(self.parent, row["Q24_integer"])
            self.assertTrue(np.array_equal(state["z"], self.parent["z"]))
            self.assertTrue(np.array_equal(state["h"], self.parent["h"]))
            for key in state:
                self.assertFalse(np.shares_memory(state[key], self.parent[key]))
                self.assertTrue(np.array_equal(np.delete(state[key], 62),
                                               np.delete(self.parent[key], 62)))
        self.assertEqual(d.v1.state_binding(self.parent), before)

    def test_invalid_schedule_and_state_rejected(self):
        for args in ((True, 100, 50), (0, 15, 7), (0, 32, 16), (0, 32, 0),
                     (0, 1 << 63, 7)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                d.interior.interior_points(*args)
        parent = {key: value.copy() for key, value in self.parent.items()}
        parent["z"][62] = 2
        with self.assertRaises(ValueError):
            d.interior.schedule(self.analysis, parent)

    def test_changed_candidate_set_rejected_before_dispatch(self):
        for mutate in (lambda rows: rows.reverse(), lambda rows: rows.pop(),
                       lambda rows: rows[0].update(Q24_integer=26514683648)):
            result = self.result()
            mutate(result["probes"])
            driver = Mock()
            with self.assertRaisesRegex(ValueError, "probe set changed"):
                self.run_probes(result, driver)
            driver.assert_not_called()

    def test_failed_original_l20_precondition_aborts_without_native(self):
        reference = self.reference.copy()
        reference[62] = 0
        result, driver = self.result(), Mock()
        self.run_probes(result, driver, reference)
        driver.assert_not_called()
        self.assertEqual(result["reason"], "L20_GLOBAL_GATE_BOUNDARY")
        self.assertEqual(result["native_layers"], [])
        self.assertFalse(result["sampled_set_complete"])
        self.assertEqual(result["probes"][1]["status"], "NOT_EXECUTED")

    def test_all_pass_has_45_calls_and_immediate_actual_parents(self):
        result, calls = self.result(), []
        self.run_probes(result, self.driver(result, calls))
        self.assertEqual(result["status"], "BOUNDED_INTERIOR_PROBES_PASS")
        self.assertEqual(result["native_layers"], [21, 22, 23] * 15)
        self.assertEqual(result["native_layer_invocations"], 45)
        for index, row in enumerate(result["probes"]):
            self.assertEqual(calls[index * 3][1], row["state_binding"])
            for previous, following in zip(row["layers"], row["layers"][1:]):
                self.assertEqual(previous["output_state_binding"], following["input_state_binding"])
            self.assertTrue(all(layer["prior_kv"] == "own empty P0"
                                and layer["prior_layer_kv_consumed"] is False for layer in row["layers"]))
        self.assertTrue(result["sampled_set_complete"])
        self.assertEqual(result["first_passing_probe_index"], 0)

    def test_l21_failure_stops_each_probe(self):
        result, calls = self.result(), []
        self.run_probes(result, self.driver(result, calls, fail=(21, 18)))
        self.assertEqual(result["native_layers"], [21] * 15)
        self.assertEqual(result["status"], "INTERIOR_GATE_FAIL")
        self.assertEqual(result["native_L22_invocations"], 0)
        self.assertIsNone(result["first_passing_probe_index"])
        self.assertTrue(all(row["first_failure"]["node"] == [21, 0, 18]
                            for row in result["probes"]))

    def test_l22_local_failure_stops_before_l23(self):
        result, calls = self.result(), []
        self.run_probes(result, self.driver(result, calls, fail=(22, 12)))
        self.assertEqual(result["native_layers"], [21, 22] * 15)
        self.assertEqual(result["native_L23_invocations"], 0)
        self.assertTrue(all(row["first_failure"]["gate"] == "local_operator_fp16"
                            for row in result["probes"]))

    def test_l23_failure_exhausts_only_prepared_samples(self):
        result, calls = self.result(), []
        self.run_probes(result, self.driver(result, calls, fail=(23, 18)))
        self.assertEqual(result["native_layer_invocations"], 45)
        self.assertEqual(result["status"], "INTERIOR_GATE_FAIL")
        self.assertTrue(result["sampled_set_complete"])
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertFalse(result["scientific_result_claim"])

    def test_failed_probe_does_not_stop_later_candidates(self):
        result, calls = self.result(), []
        passing = self.driver(result, calls)
        failing = self.driver(result, calls, fail=(21, 0))

        def mixed(layer, state, entry):
            return (failing if not calls else passing)(layer, state, entry)

        self.run_probes(result, mixed)
        self.assertEqual(result["native_layers"], [21] + [21, 22, 23] * 14)
        self.assertEqual(result["first_passing_probe_index"], 1)
        self.assertEqual(result["status"], "INTERIOR_GATE_FAIL")

    def test_blocked_driver_aborts_remaining_batch(self):
        result = self.result()

        def blocked(layer, state, entry):
            result["native_layers"].append(layer)
            raise RuntimeError("operand binding failed")

        with self.assertRaisesRegex(RuntimeError, "operand binding"):
            self.run_probes(result, blocked)
        self.assertEqual(result["native_layers"], [21])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["probes"][1]["status"], "NOT_EXECUTED")

    def test_input_mutation_rejected(self):
        result, calls = self.result(), []

        def mutate(state, entry):
            state["i"][62] += 1

        with self.assertRaisesRegex(ValueError, "mutated its input"):
            self.run_probes(result, self.driver(result, calls, mutate=mutate))
        self.assertEqual(result["native_layers"], [21])

    def test_malformed_policy_state_kv_and_global_reports_rejected(self):
        for mutate in (
            lambda report: report.update(policy_id="other"),
            lambda report: report.update(kv_lineage="FAIL"),
            lambda report: report.update(residual_state_lineage="FAIL"),
            lambda report: report.update(local_reference_independent=True),
            lambda report: report["binary64_v1"].update(coordinates=1),
        ):
            result, calls = self.result(), []
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.run_probes(result, self.driver(
                    result, calls, mutate=lambda state, entry: mutate(entry["reports"][-1])))
            self.assertEqual(result["native_layers"], [21])

    def test_dispatch_accounting_and_budget_rejected(self):
        result, calls = self.result(), []
        driver = self.driver(result, calls)

        def wrong(layer, state, entry):
            output = driver(layer, state, entry)
            result["native_layers"].append(layer)
            return output

        with self.assertRaisesRegex(ValueError, "dispatch accounting"):
            self.run_probes(result, wrong)
        for layers in ([21] * 46, [8], [True], [20], [24]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                d.accounting({"native_layers": layers})

    def test_native_guard_blocks_replay_external_and_publication(self):
        for layer in (*range(21), 24, True):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                with d.endpoint.native_only([], layer):
                    self.fail("invalid layer admitted")
        audit = []
        with patch.object(d.endpoint.native.candidate, "_stages", return_value=iter(["synthetic"])):
            with d.endpoint.native_only(audit, 21):
                self.assertEqual(list(d.endpoint.native.candidate._stages(None, 21, None, None)),
                                 ["synthetic"])
                with self.assertRaises(ValueError):
                    list(d.endpoint.native.candidate._stages(None, 21, None, None))
                with self.assertRaises(RuntimeError):
                    subprocess.Popen(["forbidden"])
                with self.assertRaises(RuntimeError):
                    np.save("forbidden", np.zeros(1))
        self.assertEqual(audit, [21])

    def test_check_only_does_not_open_model_or_evaluate_native(self):
        result = self.result()
        result["validation"] = {"origins": {}, "contract": {}}
        inputs = SimpleNamespace(records={}, bind=Mock())
        with patch.object(d, "validate"), patch.object(d, "authenticate", return_value=(
                inputs, self.frozen, self.parent, {})), patch.object(d.endpoint, "check_inputs"):
            with patch.object(d.engine, "safe_open") as model, patch.object(
                    d.endpoint, "evaluate_layer") as evaluate:
                d.checked_execution(result, False)
                model.assert_not_called()
                evaluate.assert_not_called()
        self.assertEqual(result["status"], "INTERIOR_EXECUTOR_READY")
        self.assertEqual(result["native_layers"], [])
        self.assertEqual(len(result["probes"]), 15)
        self.assertEqual(result["normal_host_review"], "REQUIRED")

    def test_test_collection_failure_error_and_skip_rejected(self):
        good = SimpleNamespace(testsRun=26, skipped=[], wasSuccessful=lambda: True)
        d.check_test_result(good, 26)
        for outcome, count in (
            (good, 19),
            (SimpleNamespace(testsRun=25, skipped=[], wasSuccessful=lambda: True), 26),
            (SimpleNamespace(testsRun=26, skipped=[1], wasSuccessful=lambda: True), 26),
            (SimpleNamespace(testsRun=26, skipped=[], wasSuccessful=lambda: False), 26),
        ):
            with self.assertRaises(ValueError):
                d.check_test_result(outcome, count)

    def test_cli_failure_is_explicit_nonzero_and_modes_exclusive(self):
        output = io.StringIO()
        with patch.object(d, "checked_execution", side_effect=ValueError("source mismatch")):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(d.main(["--check"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("source mismatch", result["error"])
        self.assertEqual(result["native_layers"], [])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--execute"])


if __name__ == "__main__":
    unittest.main()
