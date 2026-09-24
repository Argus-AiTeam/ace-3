"""Focused v2 executor tests; all stage producers here are synthetic, never native."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_endpoint_l21_l23_execute_v2 as d


class EndpointExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.frozen = json.loads((d.INPUT / "diagnostic.stdout.json").read_text())
        cls.history = json.loads((d.INPUT / "result.json").read_text())
        cls.preparation = json.loads((d.v1.INPUT / "diagnostic.stdout.json").read_text())
        cls.parent = {"i": np.zeros(896, dtype="<i8"), "z": np.zeros(896, dtype="u1"),
                      "h": np.zeros(896, dtype="<u2")}
        cls.parent["i"][62] = 26514683648
        cls.parent["h"][62] = 0x662C
        cls.parent["z"][0], cls.parent["h"][0] = 1, 0x8000
        cls.trajectory = {"stage18": cls.parent["h"].copy()}
        cls.reference = np.zeros(896, dtype="<f8")
        cls.reference[62] = float.fromhex("0x1.8b12540d033edp+10")
        cls.analysis = d.prepared.analyze(cls.parent, cls.trajectory, cls.reference)

    def frozen_check(self, document=None, history=None):
        d.check_frozen(self.frozen if document is None else document,
                       self.history if history is None else history, self.preparation,
                       self.frozen["candidate_set"], self.frozen["baseline_control"]["state_binding"])

    def result(self):
        return {"status": "BLOCKED", **d.FLAGS, "native_layers": [],
                "candidate_set": d.v1.candidates(self.analysis, self.analysis, self.parent)}

    def driver(self, result, calls, fail=None, mutate=None):
        def run_layer(layer, state, entry):
            calls.append((layer, d.v1.state_binding(state)))
            result["native_layers"].append(layer)
            last = fail[1] if fail is not None and layer == fail[0] else 18
            failed = fail is not None and layer == fail[0]
            for stage in range(last + 1):
                passed = not (failed and stage == last)
                report = {"stage": stage, "node": [layer, 0, stage],
                          "policy_id": d.prepared.gates.POLICY_ID,
                          "residual_state_lineage": "PASS", "kv_lineage": "PASS",
                          "local_reference_independent": stage < 18,
                          "status": "PASS" if passed else "FAIL",
                          "local_operator_fp16" if stage < 18 else "binary64_v1":
                              {"passed": passed, "coordinates": 896}}
                entry["reports"].append(report)
            entry.update(status="FAIL" if failed else "PASS",
                         failure={"node": [layer, 0, last], "index": 0,
                                  "gate": "binary64_v1" if last == 18 else "local_operator_fp16"}
                         if failed else None)
            output = None if failed else d.prepared.change_coordinate(
                state, int(state["i"][62]) + 1)
            if mutate is not None:
                mutate(state, entry)
            return output
        return run_layer

    def execute(self, result, driver, reference=None):
        d.execute_endpoints(result, self.parent, self.analysis, self.trajectory,
                            self.reference if reference is None else reference, driver)

    def test_contract_and_explicit_interfaces(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check", "--execute"])
        self.assertEqual(self.contract["reproduction"], d.COMMAND)
        self.assertEqual(self.contract["execution"], d.EXECUTE_COMMAND)

    def test_scope_threshold_reference_and_admission_mutations_rejected(self):
        changes = [(key, "changed") for key in d.FLAGS]
        changes.extend([("focused_tests", 19), ("excess_budget", "3/4"),
                        ("reference_policy", "local reference"), ("interface", ["--execute"]),
                        ("reproduction", "python -m foreign")])
        for key, value in changes:
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for section, key, value in (("executor", "native_layers", [8, 21]),
                                    ("executor", "candidate_endpoint_count", 3),
                                    ("local_gate", "ordered_FP16_ULP", True)):
            changed = copy.deepcopy(self.contract)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_repo_bound_context(self):
        self.assertEqual(d.upstream.producer.context()["executable"], str(d.PYTHON))
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.upstream.producer.context()
        with patch.object(Path, "cwd", return_value=Path("/foreign")):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.upstream.producer.context()

    def test_candidate_and_test_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.v1.MODULE, d.v1.TEST_MODULE, d.upstream.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_source_origin_rejected(self):
        module = ModuleType("ace3.model.candidates.foreign_v2")
        module.__file__ = "/foreign/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_pinned_input_digest_before_parse(self):
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        with patch.object(d.retained, "record", return_value={"sha256": "0" * 64}):
            with patch.object(inputs, "read") as read:
                with self.assertRaisesRegex(ValueError, "pinned evidence"):
                    d.v1.pinned(inputs, d.INPUT / "result.json", d.PINS["result.json"])
                read.assert_not_called()

    def test_v1_blocked_history_preserved(self):
        self.frozen_check()
        for key, value in (("status", "PASS"), ("reason", "NATIVE_FAILED"),
                           ("native_L21_invocations", 1), ("successor_published", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.frozen_check(history={**self.history, key: value})

    def test_frozen_reference_lineage_and_endpoint_binding(self):
        for key in ("analysis", "parent_binding", "original_L20_reference", "lineage",
                    "candidate_set", "baseline_control", "historical_sensitivity"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.frozen_check(document={**self.frozen, key: None})

    def test_frozen_v1_output_size_schema_authenticated_without_mutation(self):
        before = copy.deepcopy(self.history)
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        self.assertTrue(self.history["outputs"])
        for record in self.history["outputs"]:
            with self.subTest(path=record["path"]):
                self.assertEqual(set(record), {"path", "sha256", "size"})
                self.assertEqual(d.bind_v1_output(inputs, record), Path(record["path"]))
                self.assertEqual(inputs.records[record["path"]],
                                 {"path": record["path"], "sha256": record["sha256"],
                                  "bytes": record["size"]})
                self.assertEqual(inputs.records[record["path"]], d.retained.record(record["path"]))
                invalid = [
                    {**record, "size": record["size"] + 1},
                    {**record, "size": -1},
                    {**record, "size": True},
                    {**record, "size": float(record["size"])},
                    {**record, "sha256": "0" * 64},
                    {**record, "path": "/foreign/output.json"},
                    {**record, "path": "relative/output.json"},
                    {**record, "bytes": record["size"]},
                    {key: value for key, value in record.items() if key != "size"},
                    inputs.records[record["path"]],
                ]
                for changed in invalid:
                    fresh = d.prepared.margin.margin.prior.BoundInputs()
                    with self.subTest(record=changed), self.assertRaises(ValueError):
                        d.bind_v1_output(fresh, changed)
                    self.assertEqual(fresh.records, {})
        self.assertEqual(self.history, before)

    def test_equal_h_endpoints_are_distinct_integer_candidates(self):
        rows = d.v1.candidates(self.analysis, self.analysis, self.parent)
        self.assertEqual([row["Q24_integer"] for row in rows],
                         [3159 * (1 << 23), 3161 * (1 << 23)])
        self.assertEqual([row["delta_Q24_units"] for row in rows], [-15070976, 1706240])
        self.assertEqual(rows[0]["state_binding"]["h"], rows[1]["state_binding"]["h"])
        self.assertNotEqual(rows[0]["state_binding"]["i"], rows[1]["state_binding"]["i"])

    def test_only_i62_changes_with_preserved_zero_tags_and_no_alias(self):
        before = d.v1.state_binding(self.parent)
        for row in self.analysis["candidate_endpoints"]:
            changed = d.prepared.change_coordinate(self.parent, row["Q24_integer"])
            self.assertEqual(changed["h"][0], 0x8000)
            self.assertTrue(np.array_equal(changed["z"], self.parent["z"]))
            for key in changed:
                self.assertFalse(np.shares_memory(changed[key], self.parent[key]))
                self.assertTrue(np.array_equal(np.delete(changed[key], 62),
                                               np.delete(self.parent[key], 62)))
        self.assertEqual(d.v1.state_binding(self.parent), before)

    def test_invalid_integer_and_state_rejected(self):
        for integer in (True, 1 << 63, -(1 << 63) - 1, 1.0):
            with self.assertRaises(ValueError):
                d.prepared.change_coordinate(self.parent, integer)
        for key, value in (("z", 2), ("h", 0), ("i", 0)):
            parent = {k: v.copy() for k, v in self.parent.items()}
            parent[key][62] = value
            with self.assertRaises(ValueError):
                d.prepared.change_coordinate(parent, 26499612672)

    def test_candidate_count_order_and_deduplication_rejected(self):
        for rows in ([], self.analysis["candidate_endpoints"][:1],
                     list(reversed(self.analysis["candidate_endpoints"])),
                     self.analysis["candidate_endpoints"] * 2):
            changed = {**self.analysis, "candidate_endpoints": rows}
            with self.assertRaises(ValueError):
                d.v1.candidates(changed, self.analysis, self.parent)
        result = self.result()
        result["candidate_set"].reverse()
        with self.assertRaises(ValueError):
            self.execute(result, lambda *args: self.fail("dispatch before binding"))

    def test_both_suffixes_order_state_chain_and_nonadmission(self):
        result, calls = self.result(), []
        baseline = d.v1.state_binding(self.parent)
        self.execute(result, self.driver(result, calls))
        self.assertEqual([layer for layer, _ in calls], [21, 22, 23, 21, 22, 23])
        self.assertEqual(result["status"], "BOUNDED_ENDPOINTS_PASS")
        self.assertEqual(result["native_layer_invocations"], 6)
        for offset, row in zip((0, 3), result["candidate_set"]):
            self.assertEqual(calls[offset][1], row["state_binding"])
            for index in (1, 2):
                self.assertEqual(calls[offset + index][1],
                                 row["layers"][index - 1]["output_state_binding"])
        self.assertEqual(d.v1.state_binding(self.parent), baseline)
        for flag in ("candidate_admitted", "policy_adopted", "successor_published",
                     "scientific_result_claim"):
            self.assertIs(result[flag], False)
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertEqual(result["normal_host_review"], "REQUIRED")

    def test_local_failure_stops_each_endpoint_at_first_stage(self):
        result, calls = self.result(), []
        self.execute(result, self.driver(result, calls, (21, 0)))
        self.assertEqual([layer for layer, _ in calls], [21, 21])
        self.assertEqual(result["status"], "ENDPOINT_GATE_FAIL")
        self.assertTrue(all(len(row["layers"][0]["reports"]) == 1
                            for row in result["candidate_set"]))

    def test_full_vector_global_failure_cannot_enter_l22(self):
        result, calls = self.result(), []
        self.execute(result, self.driver(result, calls, (21, 18)))
        self.assertEqual([layer for layer, _ in calls], [21, 21])
        self.assertEqual(result["native_L22_invocations"], 0)
        self.assertEqual(result["native_L23_invocations"], 0)
        self.assertTrue(all(row["first_failure"]["gate"] == "binary64_v1"
                            for row in result["candidate_set"]))

    def test_l22_failure_cannot_enter_l23(self):
        result, calls = self.result(), []
        self.execute(result, self.driver(result, calls, (22, 7)))
        self.assertEqual([layer for layer, _ in calls], [21, 22, 21, 22])
        self.assertEqual(result["native_layer_invocations"], 4)
        self.assertEqual(result["native_L23_invocations"], 0)

    def test_l20_full_vector_precondition_before_dispatch(self):
        result = self.result()
        reference = self.reference.copy()
        reference[63] = 1.0
        self.execute(result, lambda *args: self.fail("L20 failure dispatched"), reference)
        self.assertEqual(result["reason"], "L20_GLOBAL_GATE_BOUNDARY")
        self.assertEqual(result["native_layer_invocations"], 0)
        self.assertEqual(result["candidate_set"][1]["status"], "NOT_EXECUTED")

    def test_report_state_kv_and_gate_corruption_blocks_batch(self):
        for key, value in (("node", [8, 0, 0]), ("kv_lineage", "FAIL"),
                           ("residual_state_lineage", "FAIL"), ("policy_id", "foreign"),
                           ("status", "FAIL")):
            result, calls = self.result(), []
            def mutate(state, entry):
                entry["reports"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.execute(result, self.driver(result, calls, mutate=mutate))
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["candidate_set"][1]["status"], "NOT_EXECUTED")
        result, calls = self.result(), []
        def mutate_input(state, entry):
            state["i"][1] = 1
        with self.assertRaisesRegex(ValueError, "mutated its input"):
            self.execute(result, self.driver(result, calls, mutate=mutate_input))

    def test_runtime_error_keeps_attempt_accounting_and_aborts(self):
        result = self.result()
        def broken(layer, state, entry):
            result["native_layers"].append(layer)
            raise RuntimeError("synthetic stage failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
            self.execute(result, broken)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["native_layer_invocations"], 1)
        self.assertEqual(result["candidate_set"][0]["native_layers"], [21])
        self.assertEqual(result["candidate_set"][1]["status"], "NOT_EXECUTED")

    def test_guard_allows_one_synthetic_requested_layer_and_counts_it(self):
        audit = []
        def synthetic(*args):
            yield 0
        with patch.object(d.native.candidate, "_stages", synthetic):
            with d.native_only(audit, 21):
                self.assertEqual(list(d.native.candidate._stages({}, 21, {}, {})), [0])
                with self.assertRaises(ValueError):
                    list(d.native.candidate._stages({}, 21, {}, {}))
        self.assertEqual(audit, [21])

    def test_guard_rejects_l0_l8_other_layers_and_budget_before_native(self):
        for layer in (*range(9), 9, 20, 24, True):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                with d.native_only([], layer):
                    self.fail("forbidden layer entered")
        with self.assertRaises(ValueError):
            with d.native_only([21, 22, 23, 21, 22, 23], 21):
                self.fail("budget exceeded")
        audit = []
        with d.native_only(audit, 21):
            with self.assertRaises(ValueError):
                list(d.native.candidate._stages({}, 8, {}, {}))
        self.assertEqual(audit, [])

    def test_external_execution_and_state_publication_are_blocked(self):
        with d.native_only([], 21):
            for call in (lambda: subprocess.Popen(["unused"]),
                         lambda: os.system("unused"),
                         lambda: np.save("unused", self.parent["h"]),
                         lambda: np.savez("unused", **self.parent),
                         lambda: d.retained.save(Path("unused"), self.parent)):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    call()

    def test_exact_test_count_zero_skips_and_failures_required(self):
        outcome = SimpleNamespace(testsRun=25, skipped=[], wasSuccessful=lambda: True)
        d.check_test_result(outcome, 25)
        for count in (0, 19, 24, 26, True):
            with self.assertRaises(ValueError):
                d.check_test_result(outcome, count)
        for changed in (SimpleNamespace(testsRun=24, skipped=[], wasSuccessful=lambda: True),
                        SimpleNamespace(testsRun=25, skipped=[("test", "skip")],
                                        wasSuccessful=lambda: True),
                        SimpleNamespace(testsRun=25, skipped=[], wasSuccessful=lambda: False)):
            with self.assertRaises(ValueError):
                d.check_test_result(changed, 25)

    def test_reused_stage_driver_obeys_real_local_gate_and_closes_on_failure(self):
        arrays, locals_, reports, yielded, closed = {}, {}, [], [], []
        def synthetic(tensors, layer, parent, arrays):
            try:
                arrays["stage00"] = np.full(896, 0x3C00, dtype="<u2")
                yielded.append(0)
                yield 0
                self.fail("stage driver continued after mandatory failure")
            finally:
                closed.append(True)
        zeros = np.zeros(896, dtype="<u2")
        with patch.object(d.native.candidate, "_stages", synthetic):
            with patch.object(d.retained, "check_stage_state", return_value=zeros):
                with patch.object(d.local, "local_reference", return_value=zeros):
                    with patch.dict(d.local.OPERANDS, {0: ()}):
                        with d.native_only([], 21):
                            failure, _ = d.engine.drive_layer(
                                {}, 21, self.parent, {"stage00": zeros}, self.reference,
                                arrays, locals_, reports)
        self.assertEqual(yielded, [0])
        self.assertEqual(closed, [True])
        self.assertEqual(failure["node"], [21, 0, 0])
        self.assertEqual(failure["gate"], "local_operator_fp16")
        self.assertIs(reports[0]["local_operator_fp16"]["passed"], False)


if __name__ == "__main__":
    unittest.main()
