"""Endpoint executor regressions using synthetic callbacks, never native layers."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l20_endpoint_l21_l23_executor_v1 as d
from ace3.model.candidates import diagnose_q24_s16_l20_endpoint_l21_l23_execute_v2 as runtime


class EndpointExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.document = json.loads((d.INPUT / "diagnostic.stdout.json").read_text())
        cls.history = json.loads((d.INPUT / "result.json").read_text())
        cls.parent = {
            "i": np.zeros(896, dtype="<i8"),
            "z": np.zeros(896, dtype="u1"),
            "h": np.zeros(896, dtype="<u2"),
        }
        cls.parent["i"][62] = 26514683648
        cls.parent["h"][62] = 0x662C
        cls.parent["z"][0] = 1
        cls.parent["h"][0] = 0x8000
        reference = np.zeros(896, dtype="<f8")
        reference[62] = float.fromhex("0x1.8b12540d033edp+10")
        cls.analysis = d.prepared.analyze(
            cls.parent, {"stage18": cls.parent["h"].copy()}, reference)

    def check_preparation(self, document):
        d.check_preparation(
            document, self.document["parent_binding"], self.document["original_L20_reference"],
            self.document["frozen_first_failure"], self.document["analysis"])

    def test_contract(self):
        d.check_contract(self.contract)
        self.assertEqual(self.contract["interface"], ["--check", "--execute"])
        self.assertEqual(self.contract["reproduction"], d.COMMAND)

    def test_contract_scope_and_threshold_mismatch(self):
        for key, value in (("interface", ["--execute"]), ("excess_budget", "3/4"),
                           ("reference_policy", "local reference"), ("focused_tests", 19),
                           ("normal_host_review", "OPTIONAL"), ("reproduction", "python -m x")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})
        for section, key, value in (("executor", "candidate_endpoint_count", 3),
                                    ("local_gate", "ordered_FP16_ULP", True)):
            changed = copy.deepcopy(self.contract)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_contract_execution_and_publication_flags(self):
        for key in d.FLAGS:
            changed = copy.deepcopy(self.contract)
            changed[key] = True if type(changed[key]) is int else "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_repository_context(self):
        self.assertEqual(d.prepared.upstream.producer.context()["executable"], str(d.PYTHON))
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign"}):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.prepared.upstream.producer.context()
        with patch.object(Path, "cwd", return_value=Path("/foreign")):
            with self.assertRaisesRegex(ValueError, "repo-bound"):
                d.prepared.upstream.producer.context()

    def test_repository_origins(self):
        records = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.prepared.MODULE,
                     d.prepared.upstream.LEGACY_TEST):
            self.assertEqual(records[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_origin_rejected(self):
        module = ModuleType("ace3.model.candidates.foreign")
        module.__file__ = "/foreign/source.py"
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()
        with patch.object(d, "__file__", "/foreign/source.py"):
            with self.assertRaisesRegex(ValueError, "origin mismatch"):
                d.origins()

    def test_pinned_input_before_parse(self):
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        with patch.object(d.retained, "record", return_value={"sha256": "0" * 64}):
            with patch.object(inputs, "read") as read:
                with self.assertRaisesRegex(ValueError, "pinned evidence"):
                    d.pinned(inputs, d.INPUT / "result.json", d.PINS["result.json"])
                read.assert_not_called()

    def test_input_source_digest_size_origin(self):
        inputs = d.prepared.margin.margin.prior.BoundInputs()
        record = d.retained.record(d.SOURCE)
        inputs.bind(record)
        for key, value in (("sha256", "0" * 64), ("bytes", 0), ("path", "/foreign/source.py")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                inputs.bind({**record, key: value})

    def test_two_endpoints_independent_integer_oracle(self):
        rows = d.candidates(self.analysis, self.analysis, self.parent)
        self.assertEqual([row["Q24_integer"] for row in rows],
                         [3159 * (1 << 23), 3161 * (1 << 23)])
        self.assertEqual([row["delta_Q24_units"] for row in rows], [-15070976, 1706240])
        self.assertEqual(rows[0]["input_FP16_word"], rows[1]["input_FP16_word"])
        self.assertNotEqual(rows[0]["state_binding"]["i"], rows[1]["state_binding"]["i"])
        self.assertEqual(rows[0]["state_binding"]["h"], rows[1]["state_binding"]["h"])
        self.assertTrue(all(row["status"] == "NOT_EXECUTED" for row in rows))

    def test_endpoint_count_order_duplicate_rejected(self):
        for rows in ([], self.analysis["candidate_endpoints"][:1],
                     self.analysis["candidate_endpoints"] * 2,
                     list(reversed(self.analysis["candidate_endpoints"])),
                     [self.analysis["candidate_endpoints"][0]] * 2):
            changed = copy.deepcopy(self.analysis)
            changed["candidate_endpoints"] = rows
            with self.assertRaises(ValueError):
                d.candidates(changed, self.analysis, self.parent)
        for count in (1, 3, True):
            changed = {**self.analysis, "candidate_endpoint_count": count}
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                d.candidates(changed, changed, self.parent)

    def test_endpoint_state_and_threshold_mutations(self):
        for key, value in (("Q24_integer", True), ("delta_Q24_units", 0),
                           ("input_FP16_word", "662d"), ("delta_exact", "0")):
            changed = copy.deepcopy(self.analysis)
            changed["candidate_endpoints"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.candidates(changed, self.analysis, self.parent)
        for key, value in (("excess_budget", "3/4"), ("reference_binary64_hex", "0x1p0"),
                           ("baseline_Q24_integer", 0)):
            changed = {**self.analysis, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.candidates(changed, self.analysis, self.parent)

    def test_parent_is_unchanged_including_zero_sign(self):
        before = {key: value.copy() for key, value in self.parent.items()}
        rows = d.candidates(self.analysis, self.analysis, self.parent)
        for key in before:
            self.assertTrue(np.array_equal(self.parent[key], before[key]))
        for row in rows:
            self.assertEqual(row["state_binding"]["z"], d.state_binding(self.parent)["z"])
        for row in self.analysis["candidate_endpoints"]:
            changed = d.prepared.change_coordinate(self.parent, row["Q24_integer"])
            for key in changed:
                self.assertFalse(np.shares_memory(changed[key], self.parent[key]))
                self.assertTrue(np.array_equal(np.delete(changed[key], 62),
                                               np.delete(self.parent[key], 62)))

    def test_invalid_parent_state_rejected(self):
        for key, value in (("i", 0), ("z", 2), ("h", 0)):
            parent = {name: array.copy() for name, array in self.parent.items()}
            parent[key][62] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.candidates(self.analysis, self.analysis, parent)

    def test_prepared_boundary_and_historical_failure(self):
        self.check_preparation(self.document)
        for key, value in (("status", "PASS"), ("native_L9_plus_invocations", 1),
                           ("native_L0_L8_invocations", False), ("rtl_invocations", 1),
                           ("successor_published", True), ("candidate_admitted", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_preparation({**self.document, key: value})

    def test_reference_lineage_and_gate_binding_rejected(self):
        for key in ("original_L20_reference", "parent_binding", "lineage",
                    "frozen_first_failure", "analysis", "frozen_sensitivity"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_preparation({**self.document, key: {}})
        with self.assertRaises(ValueError):
            self.check_preparation({**self.document, "reference_policy": "same-input reference"})

    def test_preserve_preparation_blocker_and_accounting(self):
        d.check_history(self.history, self.document)
        for key, value in (("status", "PASS"), ("reason", "NATIVE_FAIL"),
                           ("native_probe_accounting", []), ("candidate_set", []),
                           ("historical_sensitivity", {}), ("successor_published", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_history({**self.history, key: value}, self.document)

    def test_collection_failure_error_and_skip_rejected(self):
        result = unittest.TestResult()
        result.testsRun = d.EXPECTED_TESTS
        d.check_test_result(result, d.EXPECTED_TESTS)
        for count in (0, 19, 21, True):
            with self.assertRaises(ValueError):
                d.check_test_result(result, count)
        for field in ("failures", "errors", "skipped"):
            changed = unittest.TestResult()
            changed.testsRun = d.EXPECTED_TESTS
            getattr(changed, field).append(("test", "failure"))
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.check_test_result(changed, d.EXPECTED_TESTS)

    def test_native_dispatch_guard_all_layers(self):
        module = ModuleType("ace3.model.candidates.guard_probe")
        for attribute in ("stages", "continuation_stages", "_stages",
                          "native_layer", "run", "execute", "dispatch", "save_state"):
            setattr(module, attribute, lambda *args: self.fail("unguarded native dispatch"))
        with patch.dict(sys.modules, {module.__name__: module}), d.no_execution():
            for layer in range(24):
                for attribute in ("stages", "native_layer", "execute", "dispatch"):
                    with self.subTest(layer=layer, attribute=attribute):
                        with self.assertRaisesRegex(RuntimeError, "forbidden"):
                            getattr(module, attribute)(layer)

    def test_external_execution_and_publication_guard(self):
        with d.no_execution():
            for function, args in ((subprocess.Popen, (["true"],)), (os.system, ("true",)),
                                   (np.save, ("forbidden.npy", self.parent["h"])),
                                   (np.savez, ("forbidden.npz",)),
                                   (np.savez_compressed, ("forbidden.npz",))):
                with self.subTest(function=str(function)), self.assertRaises(RuntimeError):
                    function(*args)

    def test_exclusive_execution_and_no_publication_cli(self):
        for argv in ([], ["--check", "--execute"], ["--check", "--out", "forbidden"],
                     ["--execute", "--admit"], ["--execute", "--publish-state"],
                     ["--execute", "--layer", "8"], ["--execute", "--rtl"],
                     ["--execute", "--adopt-policy"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                d.main(argv)
            self.assertEqual(caught.exception.code, 2)
        for execute in (False, True):
            def checked(result, requested):
                self.assertIs(requested, execute)
                result["status"] = ("BOUNDED_ENDPOINTS_PASS" if execute
                                    else "ENDPOINT_EXECUTOR_PREFLIGHT_VALIDATED")
            with patch.object(d, "checked_execution", side_effect=checked), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(d.main(["--execute" if execute else "--check"]), 0)

    def test_execution_reference_bindings(self):
        records = {
            "freeze": {"path": "freeze"}, "extension": {"path": "extension"},
            "checkpoint": {"path": "checkpoint"},
        }
        layers = {}
        for layer in (20, 21, 22, 23):
            layers[str(layer)] = {}
            for key in ("fp16", "binary64"):
                record = {"path": f"{layer}-{key}"}
                records[record["path"]] = record
                layers[str(layer)][key] = record
        extension = {"layers": layers, "checkpoint": records["checkpoint"]}
        document = {"original_L20_reference": layers["20"], "parent_binding": {}}
        inputs = Mock()
        inputs.records = records.copy()
        inputs.read.side_effect = lambda record: (
            {"reference_extension": records["extension"]}
            if record == records["freeze"] else extension)
        inputs.archive.return_value = self.parent
        with patch.object(d.retained, "record", return_value=records["freeze"]):
            parent, actual = d.execution_inputs(inputs, document)
            self.assertIs(parent, self.parent)
            self.assertIs(actual, extension)
            for key in records:
                inputs.records = {name: value for name, value in records.items() if name != key}
                with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "unbound"):
                    d.execution_inputs(inputs, document)
            inputs.records = records.copy()
            with self.assertRaisesRegex(ValueError, "reference substitution"):
                d.execution_inputs(inputs, {**document, "original_L20_reference": {}})

    def test_runtime_allowlist_budget_duplicate(self):
        for layer in (*range(21), 24, True, 21.0):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                with runtime.native_only([], layer):
                    self.fail("forbidden layer entered")
        with self.assertRaisesRegex(ValueError, "budget"):
            with runtime.native_only([21] * 6, 21):
                self.fail("budget exceeded")
        audit = []
        with patch.object(runtime.native.candidate, "_stages", return_value=iter(())) as raw:
            with runtime.native_only(audit, 21):
                with self.assertRaisesRegex(ValueError, "dispatch"):
                    list(runtime.native.candidate._stages({}, 8, self.parent, {}))
                list(runtime.native.candidate._stages({}, 21, self.parent, {}))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    list(runtime.native.candidate._stages({}, 21, self.parent, {}))
                for attribute in ("execute", "dispatch", "save_state"):
                    module = ModuleType("ace3.model.candidates.runtime_guard_probe")
                    setattr(module, attribute, lambda: self.fail("forbidden dispatch"))
                    with patch.dict(sys.modules, {module.__name__: module}):
                        with runtime.native_only([], 22), self.assertRaises(RuntimeError):
                            getattr(module, attribute)()
        raw.assert_called_once()
        self.assertEqual(audit, [21])

    def endpoint_result(self):
        return {
            **d.FLAGS, "native_layers": [],
            "candidate_set": d.candidates(self.analysis, self.analysis, self.parent),
        }

    def synthetic_layer(self, result, layer, state, entry, failed_stage=None):
        result["native_layers"].append(layer)
        reports = []
        for stage in range(19 if failed_stage is None else failed_stage + 1):
            passed = stage != failed_stage
            reports.append({
                "stage": stage, "node": [layer, 0, stage],
                "policy_id": d.prepared.gates.POLICY_ID,
                "residual_state_lineage": "PASS", "kv_lineage": "PASS",
                "local_reference_independent": stage < 18,
                "status": "PASS" if passed else "FAIL",
                "local_operator_fp16" if stage < 18 else "binary64_v1":
                    {"passed": passed, "coordinates": 896},
            })
        entry.update(reports=reports, status="PASS" if failed_stage is None else "FAIL",
                     failure=None if failed_stage is None else {
                         "node": [layer, 0, failed_stage],
                         "gate": "binary64_v1" if failed_stage == 18 else "local_operator_fp16",
                     })
        return (d.prepared.change_coordinate(state, int(state["i"][62]) + 1)
                if failed_stage is None else None)

    def test_endpoint_execution_chains_distinct_states(self):
        result = self.endpoint_result()
        before = d.state_binding(self.parent)
        def run_layer(layer, state, entry):
            row = result["candidate_set"][len(result["native_layers"]) // 3]
            self.assertEqual(int(state["i"][62]), row["Q24_integer"] + layer - 21)
            self.assertEqual(entry["prior_kv"], "own empty P0")
            self.assertFalse(entry["prior_layer_kv_consumed"])
            return self.synthetic_layer(result, layer, state, entry)
        runtime.execute_endpoints(
            result, self.parent, self.analysis, {"stage18": self.parent["h"]},
            self.reference(), run_layer)
        self.assertEqual(result["status"], "BOUNDED_ENDPOINTS_PASS")
        self.assertEqual(result["native_layers"], [21, 22, 23, 21, 22, 23])
        self.assertEqual(result["native_layer_invocations"], 6)
        self.assertEqual(d.state_binding(self.parent), before)
        for key in ("native_L0_L8_invocations", "rtl_invocations",
                    "candidate_admitted", "successor_published", "policy_adopted"):
            self.assertEqual(result[key], d.FLAGS[key])

    def reference(self):
        reference = np.zeros(896, dtype="<f8")
        reference[62] = float.fromhex("0x1.8b12540d033edp+10")
        return reference

    def test_endpoint_execution_stops_each_failure(self):
        for stage in (0, 12, 16, 18):
            result = self.endpoint_result()
            runtime.execute_endpoints(
                result, self.parent, self.analysis, {"stage18": self.parent["h"]},
                self.reference(), lambda layer, state, entry: self.synthetic_layer(
                    result, layer, state, entry, stage))
            self.assertEqual(result["status"], "ENDPOINT_GATE_FAIL")
            self.assertEqual(result["native_layers"], [21, 21])
            self.assertTrue(all(row["first_failure"]["node"] == [21, 0, stage]
                                for row in result["candidate_set"]))
        result = self.endpoint_result()
        callback = Mock()
        with patch.object(runtime.upstream, "parent_gate", return_value={"status": "FAIL"}):
            runtime.execute_endpoints(
                result, self.parent, self.analysis, {}, self.reference(), callback)
        callback.assert_not_called()
        self.assertEqual(result["native_layer_invocations"], 0)
        self.assertEqual(result["reason"], "L20_GLOBAL_GATE_BOUNDARY")

    def test_endpoint_execution_blocked_accounting(self):
        result = self.endpoint_result()
        def blocked(layer, state, entry):
            result["native_layers"].append(layer)
            raise RuntimeError("synthetic dispatch failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            runtime.execute_endpoints(
                result, self.parent, self.analysis, {"stage18": self.parent["h"]},
                self.reference(), blocked)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["native_layer_invocations"], 1)
        self.assertEqual(result["candidate_set"][0]["native_layers"], [21])
        self.assertEqual(result["candidate_set"][1]["status"], "NOT_EXECUTED")

    def test_endpoint_execution_rejects_state_and_gate_mutations(self):
        for field in ("state", "policy_id", "kv_lineage", "residual_state_lineage",
                      "local_reference_independent"):
            result = self.endpoint_result()
            def changed(layer, state, entry):
                output = self.synthetic_layer(result, layer, state, entry)
                if field == "state":
                    state["i"][62] += 1
                else:
                    entry["reports"][0][field] = "changed"
                return output
            with self.subTest(field=field), self.assertRaises(ValueError):
                runtime.execute_endpoints(
                    result, self.parent, self.analysis, {"stage18": self.parent["h"]},
                    self.reference(), changed)
            self.assertEqual(result["native_layers"], [21])

    def test_layer_evaluation_preserves_reference_and_operands(self):
        inputs, model = Mock(), Mock()
        trajectory, reference, tensor = object(), object(), object()
        item = {"fp16": {"path": "trajectory"}, "binary64": {"path": "global"},
                "canonical": {"source": "frozen"}}
        inputs.archive.return_value = trajectory
        model.get_tensor.return_value = tensor
        entry = {"reports": []}
        failure = {"node": [21, 0, 18], "gate": "binary64_v1"}
        with patch.object(runtime.local, "tensor_shapes", return_value={"weight": ()}), \
                patch.object(runtime.local, "authenticate_tensors") as authenticate, \
                patch.object(runtime.engine, "global_reference", return_value=reference) as global_ref, \
                patch.object(runtime.engine, "drive_layer",
                             return_value=(failure, {})) as drive:
            self.assertIsNone(runtime.evaluate_layer(
                inputs, model, {"layers": {"21": item}}, 21, self.parent, [], entry))
        authenticate.assert_called_once_with({"weight": tensor}, item["canonical"], 21)
        global_ref.assert_called_once_with(inputs, item)
        self.assertIs(drive.call_args.args[2], self.parent)
        self.assertIs(drive.call_args.args[3], trajectory)
        self.assertIs(drive.call_args.args[4], reference)
        self.assertEqual(entry["original_reference"], item)
        self.assertEqual(entry["failure"], failure)

    def test_execute_cli_keeps_failure_accounting(self):
        def blocked(result, requested):
            self.assertTrue(requested)
            result["native_layers"].append(21)
            raise RuntimeError("synthetic runtime failure")
        stdout = io.StringIO()
        with patch.object(d, "checked_execution", side_effect=blocked), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(d.main(["--execute"]), 1)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["native_layer_invocations"], 1)
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertFalse(result["successor_published"])


if __name__ == "__main__":
    unittest.main()
