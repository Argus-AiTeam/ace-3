"""Frozen-input authentication and synthetic preparation only; no native layers."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_threshold_refine_v1 as d


class ThresholdRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(d.entry.upstream, "execute_layer",
                          side_effect=AssertionError("native authentication dispatch")) as dispatch, \
                patch.object(d.entry.native, "stages",
                             side_effect=AssertionError("native authentication stages")) as stages:
            cls.data = d.authenticate()
            dispatch.assert_not_called()
            stages.assert_not_called()
        cls.binding = cls.data["observed_bracket_evidence"]["result"]
        cls.observed = json.loads(Path(cls.binding["path"]).read_text())

    def reject_observed(self, observed, message):
        with patch.object(d.prior.producer, "read_result", return_value=(self.binding, observed)), \
                self.assertRaisesRegex(ValueError, message):
            d.authenticate_observation({**self.data, "bound": dict(self.data["bound"])})

    def test_01_observed_bracket_is_read_only_bound_input(self):
        evidence = self.data["observed_bracket_evidence"]
        self.assertEqual(evidence["result"]["sha256"], d.OBSERVED[1])
        self.assertTrue(evidence["read_only"])
        self.assertEqual([(row["input_Q24_units62"], row["observed_accepted"])
                          for row in evidence["endpoints"]],
                         [(26501685138, True), (26502192670, False)])

    def test_02_exact_refinement_samples(self):
        values = [row["input_Q24_units62"] for row in d.plan()]
        self.assertEqual(values, [
            26501685138, 26501748580, 26501812021, 26501875462, 26501938904,
            26502002346, 26502065787, 26502129228, 26502192670])
        self.assertEqual(len(set(values)), 9)
        self.assertLessEqual(max(b - a for a, b in zip(values, values[1:])), 63442)
        self.assertTrue(all("accepted" not in row for row in d.plan()))

    def test_03_input_only_coordinate_changes_and_parent_not_mutated(self):
        actual = self.data["layers"][9]["parent"]
        snapshot = {key: value.copy() for key, value in actual.items()}
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

    def test_04_unknown_or_noninteger_controls_rejected(self):
        for control in ({}, {"label": "foreign", "input_Q24_units62": d.PASSING},
                        {**d.plan()[0], "input_Q24_units62": float(d.PASSING)},
                        {**d.plan()[0], "input_Q24_units62": d.PASSING - 1}):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "unknown"):
                d.input_parent(self.data["layers"][9]["parent"], control)

    def test_05_malformed_parent_rejected(self):
        parent = {key: value.copy() for key, value in self.data["layers"][9]["parent"].items()}
        parent["h"][0] ^= 1
        with self.assertRaises(ValueError):
            d.input_parent(parent, d.plan()[0])

    def test_06_contract_and_thresholds_unchanged(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        self.assertEqual(contract["thresholds"],
                         json.loads(d.prior.legacy.CONTRACT.read_text())["thresholds"])
        for key in contract:
            if key not in ("validation", "refinement", "claim_boundary"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                    d.check_contract({**contract, key: None})

    def test_07_source_origins_are_repository_local(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.prior.MODULE,
                     "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            self.assertTrue(Path(origins[name]["path"]).is_relative_to(d.ROOT))
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_08_original_reference_chain_authenticated(self):
        evidence = d.prior.validation_evidence(json.loads(d.CONTRACT.read_text()), self.data)
        refs = evidence["original_input_global_reference_bindings"]
        predecessor = refs["original_binary64_parent"]
        for row in refs["layers"]:
            self.assertEqual(row["input_binary64"], predecessor)
            self.assertEqual(self.data["bound"][row["binary64"]["path"]], row["binary64"])
            predecessor = row["binary64"]
        self.assertEqual([row["layer"] for row in refs["layers"]], [9, 10, 11, 12, 13])

    def test_09_missing_reference_binding_rejected(self):
        bound = dict(self.data["bound"])
        del bound[self.data["extension"]["original_binary64_parent"]["path"]]
        with self.assertRaisesRegex(ValueError, "unauthenticated"):
            d.prior.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                        {**self.data, "bound": bound})

    def test_10_reference_splice_rejected(self):
        extension = copy.deepcopy(self.data["extension"])
        extension["layers"]["10"]["input_binary64"] = extension["original_binary64_parent"]
        with self.assertRaisesRegex(ValueError, "predecessor mismatch"):
            d.prior.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                        {**self.data, "extension": extension})

    def test_11_observed_metadata_changes_rejected(self):
        for key, value in (("policy_id", "wrong"), ("native_L0_L8_invocations", 1),
                           ("rtl_invocations", 1), ("candidate_admitted", True),
                           ("normal_host_review", "OPTIONAL")):
            with self.subTest(key=key):
                self.reject_observed({**self.observed, key: value}, "metadata mismatch")

    def test_12_observed_source_changes_rejected(self):
        observed = copy.deepcopy(self.observed)
        observed["origins_after_execution"][d.prior.MODULE]["sha256"] = "0" * 64
        self.reject_observed(observed, "observed source mismatch")

    def test_13_observed_artifact_changes_rejected(self):
        observed = copy.deepcopy(self.observed)
        observed["artifacts"][0]["sha256"] = "0" * 64
        self.reject_observed(observed, "binding")

    def test_14_endpoint_decision_and_state_checks_rejected(self):
        observed = copy.deepcopy(self.observed)
        row = next(row for row in observed["controls"] if row["input_Q24_units62"] == d.PASSING)
        row["layers"][0]["reports"][0]["kv_lineage"] = "FAIL"
        self.reject_observed(observed, "state/KV/gate mismatch")
        observed = copy.deepcopy(self.observed)
        row = next(row for row in observed["controls"] if row["input_Q24_units62"] == d.PASSING)
        row["layers"][-1]["index62"]["accepted"] = False
        self.reject_observed(observed, "endpoint decision mismatch")

    def test_15_dispatch_or_selected_parent_linkage_changes_rejected(self):
        self.reject_observed({**self.observed, "native_execution_records": []},
                             "dispatch accounting mismatch")
        self.reject_observed({**self.observed, "reviewed_policy_bindings": []},
                             "selected-parent linkage mismatch")

    def test_16_wrong_frozen_result_pin_rejected(self):
        with patch.object(d.entry, "record", return_value={**self.binding, "sha256": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "wrong pinned producer result"):
                d.prior.producer.read_result(d.OBSERVED, {})

    def test_17_check_calls_validation_once_without_execution(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"synthetic": True}) as validate, \
                patch("builtins.print") as output, \
                patch.object(d.prior, "execute_sweep") as sweep, \
                patch.object(d, "execute_sweep") as refinement, \
                patch.object(d.entry, "write") as write:
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            output.assert_called_once_with(json.dumps({"synthetic": True}))
            sweep.assert_not_called()
            refinement.assert_not_called()
            write.assert_not_called()

    def test_18_cli_requires_one_explicit_mode(self):
        for args in ([], ["--out"], ["--check", "--out", "build/forbidden"], ["--rtl"]):
            with self.subTest(args=args), patch.object(d.sys, "argv", [d.MODULE, *args]), \
                    patch.object(d.sys, "stderr", io.StringIO()), \
                    patch.object(d, "validate") as validate:
                with self.assertRaises(SystemExit) as error:
                    d.main()
                self.assertEqual(error.exception.code, 2)
                validate.assert_not_called()

    def test_19_execution_plan_is_exactly_nine_L9_L13_P0_controls(self):
        records = d.execution_plan()
        self.assertEqual(records, [
            {"layer": layer, "position": 0} for _ in d.plan() for layer in (9, 10, 11, 12, 13)])
        self.assertEqual(len(records), 45)
        self.assertEqual(d.execution_plan([9, 10, 11, 12, 13]), records)

    def test_20_unauthorized_plans_fail_before_authentication_or_numerics(self):
        invalid = [{"layers": (layer, 9, 10, 11, 12, 13)} for layer in range(9)]
        invalid += [{"layers": route} for route in (
            (), (9,), (9, 10, 11, 12, 14), (10, 9, 11, 12, 13),
            (9.0, 10, 11, 12, 13), None)]
        invalid += [{"position": position} for position in (1, -1, False, 0.0)]
        invalid += [{"backend": backend} for backend in ("rtl", "gpu", "native", None)]
        with patch.object(d, "authenticate") as auth, \
                patch.object(d, "source_context") as origins, \
                patch.object(d, "execute_control") as control, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            for route in invalid:
                with self.subTest(route=route), self.assertRaisesRegex(ValueError, "execution"):
                    d.execute_sweep(**route)
            for guard in (auth, origins, control, native):
                guard.assert_not_called()

    def test_21_dispatch_and_stage_guards_reject_unplanned_work(self):
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
        for owner, name, args in ((d.subprocess, "Popen", (["forbidden"],)),
                                  (d.entry.os, "system", ("forbidden",))):
            with self.subTest(surface=name), patch.object(owner, name) as external:
                with self.assertRaisesRegex(AssertionError, "forbidden"):
                    with d.native_execution(d.execution_plan()):
                        getattr(owner, name)(*args)
                external.assert_not_called()
        with self.assertRaisesRegex(ValueError, "accounting"):
            with d.native_execution(d.execution_plan()):
                pass

    def test_22_output_scope_is_fresh_exclusive_and_versioned(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_directory(path), path)
            for bad in (d.ROOT / "build" / d.OUTPUT_PREFIX, d.ROOT / "build/foreign",
                        d.ROOT / (d.OUTPUT_PREFIX + "outside"),
                        path / (d.OUTPUT_PREFIX + "nested")):
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "scope"):
                    d.output_directory(bad)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_directory(path)
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "is_symlink", return_value=False):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_directory(path)

    def test_23_mocked_sweep_preserves_routes_references_endpoints_and_failures(self):
        frozen = {}
        for integer in (d.PASSING, d.FAILING):
            row = next(row for row in self.observed["controls"]
                       if row["input_Q24_units62"] == integer)
            for item in row["layers"]:
                saved = d.entry.native.load_state(
                    item["state_operand_KV_local_archive"],
                    lambda binding: d.entry.upstream.bind_input(binding, self.data["bound"]))
                frozen[integer, item["layer"]] = (saved, item["reports"])
        calls = []

        def fake_native(tensors, layer, parent, trajectory, reference):
            control_index = len(calls) // 5
            integer = d.PASSING if control_index % 2 == 0 and control_index != 8 else d.FAILING
            retained = self.data["layers"][layer]
            self.assertIs(tensors, retained["tensors"])
            self.assertIs(trajectory, retained["trajectory"])
            self.assertIs(reference, retained["reference"])
            if layer == 9:
                self.assertEqual(int(parent["i"][62]), d.plan()[control_index]["input_Q24_units62"])
            else:
                d.entry.prior.retained.verify_parent(parent, calls[-1])
            saved, reports = frozen[integer, layer]
            arrays = {key.removeprefix("arrays__"): value for key, value in saved.items()
                      if key.startswith("arrays__")}
            locals_ = {key.removeprefix("local_references__"): value for key, value in saved.items()
                       if key.startswith("local_references__")}
            d.entry.native.stages(tensors, layer, parent, arrays)
            calls.append(d.entry.prior.retained.state_from(arrays, "output", "stage18"))
            return arrays, locals_, reports

        with patch.object(d, "authenticate", return_value=self.data), \
                patch.object(d.entry.upstream, "execute_layer", side_effect=fake_native) as native, \
                patch.object(d.entry.native, "stages", return_value=None) as stages, \
                patch.object(d.entry.torch, "set_num_threads"), \
                patch.object(d.prior.legacy.runtime, "reauthenticate") as reauthenticate:
            result = d.execute_sweep()
        self.assertEqual(native.call_count, 45)
        self.assertEqual(stages.call_count, 45)
        reauthenticate.assert_called_once_with(self.data)
        self.assertEqual(result["native_execution_records"], d.execution_plan())
        self.assertEqual(result["native_stage_execution_records"], d.execution_plan())
        self.assertTrue(result["observed_endpoint_bitwise_reproduction"])
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertEqual(result["smallest_observed_transition_width_Q24_units"], 63441)
        self.assertEqual(len(result["L13_all_sample_transition_brackets"]), 7)
        self.assertTrue(any(not row["all_L9_L13_gates_pass"] for row in result["controls"]))
        self.assertEqual(result["artifacts"], [])
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(result[key], False)
        self.assertEqual(result["original_input_global_reference_bindings"],
                         d.prior.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                                     self.data)["original_input_global_reference_bindings"])

    def test_24_source_mismatch_rejected_before_authentication_and_dispatch(self):
        with patch.object(d, "authenticate") as auth, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            with self.assertRaisesRegex(ValueError, "changed before execution"):
                d.execute_sweep(validation={"origins": {}, "contract": {}})
            auth.assert_not_called()
            native.assert_not_called()

    def test_25_execution_cli_validates_once_and_publishes_separate_bindings(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        validation = {"origins": {}, "contract": {}}
        result = {
            "status": "DIAGNOSED", "native_L0_L8_invocations": 0,
            "native_L9_L13_invocations": 45, "rtl_invocations": 0, "native_layer_count": 45,
            "native_execution_records": d.execution_plan(),
            "native_stage_execution_records": d.execution_plan(),
        }
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(path)]), \
                patch.object(d, "output_directory", return_value=path), \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", return_value=validation) as validate, \
                patch.object(d, "execute_sweep", return_value=result) as sweep, \
                patch.object(d.entry, "write") as write, \
                patch.object(d.entry, "record", return_value={"synthetic": True}), \
                patch("builtins.print"):
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            mkdir.assert_called_once_with(exist_ok=False)
            self.assertEqual(sweep.call_count, 1)
            self.assertIs(sweep.call_args.kwargs["validation"], validation)
            self.assertEqual([call.args[0] for call in write.call_args_list], [
                path / "preflight_validation.json", path / "validation.json", path / "result.json"])
            self.assertEqual(result["validation"], {"synthetic": True})

    def test_26_failed_preflight_never_dispatches_or_publishes_result(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(path)]), \
                patch.object(d, "output_directory", return_value=path), \
                patch.object(Path, "mkdir"), \
                patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", side_effect=ValueError("preflight failed")) as validate, \
                patch.object(d, "execute_sweep") as sweep, \
                patch.object(d.entry, "write") as write:
            with self.assertRaisesRegex(ValueError, "preflight failed"):
                d.main()
            validate.assert_called_once_with()
            sweep.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
