"""Read-only frozen evidence and mocked execution; never native layer replay."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_threshold_refine_v2 as d


class ThresholdRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with d.zero_dispatch():
            cls.data = d.authenticate()
        cls.binding = cls.data["v1_refinement_evidence"]["result"]
        cls.observed = json.loads(Path(cls.binding["path"]).read_text())

    def reject_observed(self, observed, message):
        with patch.object(d.base.producer, "read_result", return_value=(self.binding, observed)), \
                self.assertRaisesRegex(ValueError, message):
            d.authenticate_observation({**self.data, "bound": dict(self.data["bound"])})

    def test_01_reviewed_v1_is_frozen_not_relabelled(self):
        evidence = self.data["v1_refinement_evidence"]
        self.assertEqual(evidence["result"]["sha256"], d.OBSERVED[1])
        self.assertTrue(evidence["read_only"])
        self.assertEqual(evidence["historical_status"], "BLOCKED")
        self.assertEqual([(r["input_Q24_units62"], r["observed_accepted"])
                          for r in evidence["endpoints"]],
                         [(26501812021, True), (26501875462, False)])

    def test_02_exact_integer_seventeen_control_plan(self):
        values = [r["input_Q24_units62"] for r in d.plan()]
        self.assertEqual(values, [
            26501812021, 26501815986, 26501819951, 26501823916, 26501827881,
            26501831846, 26501835811, 26501839776, 26501843742, 26501847707,
            26501851672, 26501855637, 26501859602, 26501863567, 26501867532,
            26501871497, 26501875462])
        self.assertEqual(len(set(values)), 17)
        self.assertTrue(all(type(v) is int for v in values))
        self.assertEqual({b - a for a, b in zip(values, values[1:])}, {3965, 3966})
        self.assertLessEqual(max(b - a for a, b in zip(values, values[1:])), 4096)
        self.assertTrue(all("accepted" not in row for row in d.plan()))

    def test_03_only_coordinate62_changes_without_parent_mutation(self):
        actual = self.data["layers"][9]["parent"]
        snapshot = {k: v.copy() for k, v in actual.items()}
        for control in d.plan():
            parent = d.input_parent(actual, control)
            self.assertEqual(int(parent["i"][62]), control["input_Q24_units62"])
            self.assertEqual(int(parent["z"][62]), 0)
            self.assertEqual(int(parent["h"][62]),
                             d.entry.prior.rational.project(control["input_Q24_units62"], 0))
            for key in actual:
                self.assertTrue(d.entry.np.array_equal(
                    d.entry.np.delete(parent[key], 62), d.entry.np.delete(actual[key], 62)))
        d.entry.paired.same_arrays(actual, snapshot)

    def test_04_invalid_controls_and_parent_rejected(self):
        actual = self.data["layers"][9]["parent"]
        for control in ({}, {**d.plan()[0], "label": "foreign"},
                        {**d.plan()[0], "input_Q24_units62": float(d.PASSING)},
                        {**d.plan()[0], "input_Q24_units62": d.PASSING - 1}):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "unknown"):
                d.input_parent(actual, control)
        parent = {k: v.copy() for k, v in actual.items()}
        parent["h"][0] ^= 1
        with self.assertRaises(ValueError):
            d.input_parent(parent, d.plan()[0])

    def test_05_contract_and_threshold_changes_rejected(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        self.assertEqual(contract["thresholds"],
                         json.loads(d.base.legacy.CONTRACT.read_text())["thresholds"])
        for key in contract:
            if key not in ("validation", "refinement", "claim_boundary"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                    d.check_contract({**contract, key: None})

    def test_06_source_test_and_contract_origins_are_repository_bound(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.prior.MODULE,
                     "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            self.assertTrue(Path(origins[name]["path"]).is_relative_to(d.ROOT))
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()
        with patch.object(d, "CONTRACT", Path("/tmp/foreign.json")):
            with self.assertRaisesRegex(ValueError, "contract origin mismatch"):
                d.source_context()

    def test_07_independent_original_reference_chain_preserved(self):
        refs = d.base.validation_evidence(json.loads(d.CONTRACT.read_text()), self.data)[
            "original_input_global_reference_bindings"]
        predecessor = refs["original_binary64_parent"]
        self.assertEqual([r["layer"] for r in refs["layers"]], [9, 10, 11, 12, 13])
        for row in refs["layers"]:
            self.assertEqual(row["input_binary64"], predecessor)
            self.assertEqual(self.data["bound"][row["binary64"]["path"]], row["binary64"])
            predecessor = row["binary64"]

    def test_08_missing_reference_and_splice_rejected(self):
        bound = dict(self.data["bound"])
        del bound[self.data["extension"]["original_binary64_parent"]["path"]]
        with self.assertRaisesRegex(ValueError, "unauthenticated"):
            d.base.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                       {**self.data, "bound": bound})
        extension = copy.deepcopy(self.data["extension"])
        extension["layers"]["10"]["input_binary64"] = extension["original_binary64_parent"]
        with self.assertRaisesRegex(ValueError, "predecessor mismatch"):
            d.base.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                       {**self.data, "extension": extension})

    def test_09_observed_metadata_and_history_changes_rejected(self):
        for key, value in (("status", "DIAGNOSED"), ("policy_id", "wrong"),
                           ("native_L0_L8_invocations", 1), ("rtl_invocations", 1),
                           ("candidate_admitted", True), ("control_count", 17)):
            with self.subTest(key=key):
                self.reject_observed({**self.observed, key: value}, "metadata mismatch")

    def test_10_observed_source_changes_rejected(self):
        observed = copy.deepcopy(self.observed)
        observed["origins_after_execution"][d.prior.MODULE]["sha256"] = "0" * 64
        self.reject_observed(observed, "observed source mismatch")

    def test_11_observed_artifact_changes_rejected(self):
        observed = copy.deepcopy(self.observed)
        observed["artifacts"][0]["sha256"] = "0" * 64
        self.reject_observed(observed, "binding")

    def test_12_observed_state_KV_decision_and_bracket_changes_rejected(self):
        observed = copy.deepcopy(self.observed)
        row = next(r for r in observed["controls"] if r["input_Q24_units62"] == d.PASSING)
        row["layers"][0]["reports"][0]["kv_lineage"] = "FAIL"
        self.reject_observed(observed, "state/KV/gate mismatch")
        observed = copy.deepcopy(self.observed)
        row = next(r for r in observed["controls"] if r["input_Q24_units62"] == d.PASSING)
        row["all_L9_L13_gates_pass"] = False
        self.reject_observed(observed, "endpoint decision mismatch")
        observed = copy.deepcopy(self.observed)
        observed["L13_all_sample_transition_brackets"][0]["width_Q24_units"] = 4096
        self.reject_observed(observed, "transition bracket mismatch")

    def test_13_dispatch_and_selected_parent_linkage_changes_rejected(self):
        self.reject_observed({**self.observed, "native_execution_records": []},
                             "dispatch accounting mismatch")
        self.reject_observed({**self.observed, "reviewed_policy_bindings": []},
                             "selected-parent linkage mismatch")

    def test_14_wrong_result_pin_rejected(self):
        with patch.object(d.entry, "record", return_value={**self.binding, "sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "wrong pinned producer result"):
                d.base.producer.read_result(d.OBSERVED, {})

    def test_15_check_validates_once_without_publication_or_sweep(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"synthetic": True}) as validate, \
                patch.object(d, "execute_sweep") as sweep, \
                patch.object(d.entry, "write") as write, patch("builtins.print") as output:
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            output.assert_called_once_with(json.dumps({"synthetic": True}))
            sweep.assert_not_called()
            write.assert_not_called()

    def test_16_cli_requires_one_explicit_mode(self):
        for args in ([], ["--out"], ["--check", "--out", "build/forbidden"], ["--rtl"]):
            with self.subTest(args=args), patch.object(d.sys, "argv", [d.MODULE, *args]), \
                    patch.object(d.sys, "stderr", io.StringIO()), patch.object(d, "validate") as validate:
                with self.assertRaises(SystemExit) as error:
                    d.main()
                self.assertEqual(error.exception.code, 2)
                validate.assert_not_called()

    def test_17_exactly_eighty_five_L9_L13_P0_evaluations(self):
        self.assertEqual(d.execution_plan(), [
            {"layer": layer, "position": 0} for _ in range(17) for layer in (9, 10, 11, 12, 13)])
        self.assertEqual(len(d.execution_plan()), 85)
        self.assertEqual(d.execution_contract()["native_layer_evaluations"], 85)

    def test_18_unauthorized_routes_rejected_before_authentication_and_numerics(self):
        invalid = [{"layers": (layer, 9, 10, 11, 12, 13)} for layer in range(9)]
        invalid += [{"layers": route} for route in (
            (), (9,), (9, 10, 11, 12, 14), (10, 9, 11, 12, 13),
            (9.0, 10, 11, 12, 13), None)]
        invalid += [{"position": p} for p in (1, -1, False, 0.0)]
        invalid += [{"backend": b} for b in ("rtl", "gpu", "native", None)]
        with patch.object(d, "authenticate") as auth, patch.object(d, "source_context") as origins, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            for route in invalid:
                with self.subTest(route=route), self.assertRaisesRegex(ValueError, "execution"):
                    d.execute_sweep(**route)
            for guard in (auth, origins, native):
                guard.assert_not_called()

    def test_19_dispatch_and_stage_guards_reject_unplanned_work(self):
        for layer in (*range(9), 10, 14, 9.0):
            for surface in ("execute_layer", "stages"):
                owner = d.entry.upstream if surface == "execute_layer" else d.entry.native
                with self.subTest(layer=layer, surface=surface), \
                        patch.object(d.entry.upstream, "execute_layer") as native, \
                        patch.object(d.entry.native, "stages") as stages:
                    with self.assertRaisesRegex(ValueError, "outside planned"):
                        with d.native_execution(d.execution_plan()):
                            args = (None, layer, None, None)
                            getattr(owner, surface)(*args, *([None] if surface == "execute_layer" else []))
                    native.assert_not_called()
                    stages.assert_not_called()
        with self.assertRaisesRegex(ValueError, "accounting"):
            with d.native_execution(d.execution_plan()):
                pass
        for owner, name in ((d.subprocess, "Popen"), (d.entry.os, "system")):
            with patch.object(owner, name) as external:
                with self.assertRaisesRegex(AssertionError, "forbidden"):
                    with d.native_execution(d.execution_plan()):
                        getattr(owner, name)("forbidden")
                external.assert_not_called()

    def test_20_output_scope_is_fresh_exclusive_and_versioned(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_directory(path), path)
            for bad in (d.ROOT / "build" / d.OUTPUT_PREFIX, d.ROOT / "build/foreign",
                        d.ROOT / (d.OUTPUT_PREFIX + "outside"), path / "nested"):
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "scope"):
                    d.output_directory(bad)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_directory(path)
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_directory(path)

    def test_21_mocked_sweep_keeps_references_endpoints_and_all_failures(self):
        frozen = {}
        for integer in (d.PASSING, d.FAILING):
            row = next(r for r in self.observed["controls"] if r["input_Q24_units62"] == integer)
            for item in row["layers"]:
                saved = d.entry.native.load_state(item["state_operand_KV_local_archive"],
                    lambda binding: d.entry.upstream.bind_input(binding, self.data["bound"]))
                frozen[integer, item["layer"]] = saved, item["reports"]
        calls = []

        def fake_native(tensors, layer, parent, trajectory, reference):
            index = len(calls) // 5
            integer = d.PASSING if index % 2 == 0 and index != 16 else d.FAILING
            retained = self.data["layers"][layer]
            self.assertIs(tensors, retained["tensors"])
            self.assertIs(trajectory, retained["trajectory"])
            self.assertIs(reference, retained["reference"])
            if layer == 9:
                self.assertEqual(int(parent["i"][62]), d.plan()[index]["input_Q24_units62"])
            else:
                d.entry.prior.retained.verify_parent(parent, calls[-1])
            saved, reports = frozen[integer, layer]
            arrays = {k.removeprefix("arrays__"): v for k, v in saved.items() if k.startswith("arrays__")}
            locals_ = {k.removeprefix("local_references__"): v for k, v in saved.items()
                       if k.startswith("local_references__")}
            d.entry.native.stages(tensors, layer, parent, arrays)
            calls.append(d.entry.prior.retained.state_from(arrays, "output", "stage18"))
            return arrays, locals_, reports

        with patch.object(d, "authenticate", return_value=self.data), \
                patch.object(d.entry.upstream, "execute_layer", side_effect=fake_native) as native, \
                patch.object(d.entry.native, "stages", return_value=None) as stages, \
                patch.object(d.entry.torch, "set_num_threads"), \
                patch.object(d.base.legacy.runtime, "reauthenticate") as reauthenticate:
            result = d.execute_sweep()
        self.assertEqual(native.call_count, 85)
        self.assertEqual(stages.call_count, 85)
        reauthenticate.assert_called_once_with(self.data)
        self.assertEqual(result["native_execution_records"], d.execution_plan())
        self.assertEqual(result["native_stage_execution_records"], d.execution_plan())
        self.assertTrue(result["observed_endpoint_bitwise_reproduction"])
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertEqual(result["smallest_observed_transition_width_Q24_units"], 3965)
        self.assertEqual(len(result["L13_all_sample_transition_brackets"]), 15)
        self.assertTrue(any(not row["all_L9_L13_gates_pass"] for row in result["controls"]))
        self.assertEqual(result["artifacts"], [])
        self.assertEqual(result["original_input_global_reference_bindings"],
                         self.observed["original_input_global_reference_bindings"])
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(result[key], False)

    def test_22_source_mismatch_rejected_before_authentication_and_dispatch(self):
        with patch.object(d, "authenticate") as auth, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            with self.assertRaisesRegex(ValueError, "changed before execution"):
                d.execute_sweep(validation={"origins": {}, "contract": {}})
            auth.assert_not_called()
            native.assert_not_called()

    def test_23_execution_cli_validates_once_with_separate_publication(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        validation = {"origins": {}, "contract": {}}
        result = {"status": "DIAGNOSED", "native_L0_L8_invocations": 0,
                  "native_L9_L13_invocations": 85, "rtl_invocations": 0, "native_layer_count": 85,
                  "native_execution_records": d.execution_plan(),
                  "native_stage_execution_records": d.execution_plan()}
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(path)]), \
                patch.object(d, "output_directory", return_value=path), \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", return_value=validation) as validate, \
                patch.object(d, "execute_sweep", return_value=result) as sweep, \
                patch.object(d.entry, "write") as write, \
                patch.object(d.entry, "record", return_value={"synthetic": True}), patch("builtins.print"):
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            mkdir.assert_called_once_with(exist_ok=False)
            self.assertEqual(sweep.call_count, 1)
            self.assertIs(sweep.call_args.kwargs["validation"], validation)
            self.assertEqual([call.args[0] for call in write.call_args_list], [
                path / "preflight_validation.json", path / "validation.json", path / "result.json"])

    def test_24_failed_preflight_never_dispatches_or_publishes(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(path)]), \
                patch.object(d, "output_directory", return_value=path), \
                patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", side_effect=ValueError("preflight failed")) as validate, \
                patch.object(d, "execute_sweep") as sweep, patch.object(d.entry, "write") as write:
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                d.main()
            validate.assert_called_once_with()
            sweep.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
