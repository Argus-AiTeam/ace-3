"""Focused v3 checks; no native numerics, accepted-layer tests, or publication."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_threshold_refine_v3 as d


class ThresholdRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with d.zero_dispatch():
            cls.data, cls.observed = d.authenticate()
        cls.origins = d.source_context()

    def reject(self, observed, message):
        with self.assertRaisesRegex(ValueError, message):
            d.check_observation(observed, self.data, self.origins)

    def test_01_frozen_v2_and_historical_failure_preserved(self):
        evidence = self.data["v2_refinement_evidence"]
        self.assertEqual(evidence["result"]["sha256"], d.OBSERVED[1])
        self.assertTrue(evidence["read_only"])
        self.assertEqual(evidence["historical_status"], "DIAGNOSED")
        self.assertEqual(evidence["historical_v1_evidence"]["historical_status"], "BLOCKED")
        normalized = evidence["policy_binding"]["validation"]
        self.assertEqual(normalized, evidence["publication_bindings"]["validation"])
        self.assertEqual(normalized["bytes"], self.observed["validation"]["size"])
        self.assertNotIn("bytes", self.observed["validation"])
        self.assertEqual([(r["input_Q24_units62"], r["observed_accepted"])
                          for r in evidence["endpoints"]],
                         [(26501847707, True), (26501851672, False)])

    def test_02_exact_tighter_integer_plan(self):
        values = [r["input_Q24_units62"] for r in d.plan()]
        self.assertEqual(values, [
            26501847707, 26501847955, 26501848203, 26501848450, 26501848698,
            26501848946, 26501849194, 26501849442, 26501849690, 26501849937,
            26501850185, 26501850433, 26501850681, 26501850929, 26501851176,
            26501851424, 26501851672])
        self.assertEqual(len(set(values)), 17)
        self.assertTrue(all(type(v) is int for v in values))
        self.assertEqual({b - a for a, b in zip(values, values[1:])}, {247, 248})
        self.assertLessEqual(max(b - a for a, b in zip(values, values[1:])), 256)
        self.assertTrue(all("accepted" not in row for row in d.plan()))

    def test_03_planned_records_only_l9_l13_p0(self):
        records = d.execution_plan()
        self.assertEqual(records, [{"layer": layer, "position": 0}
                                   for _ in range(17) for layer in range(9, 14)])
        self.assertEqual(sum(row["layer"] <= 8 for row in records), 0)
        for layer in range(9):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "L0-L8"):
                d.execution_plan(layers=(layer, 10, 11, 12, 13))
        for layers in ((9, 10, 11, 12, 14), (9,), (9.0, 10, 11, 12, 13)):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                d.execution_plan(layers=layers)
        for position in (1, False, 0.0):
            with self.subTest(position=position), self.assertRaises(ValueError):
                d.execution_plan(position=position)
        with self.assertRaisesRegex(ValueError, "RTL forbidden"):
            d.execution_plan(backend="rtl")

    def test_04_only_coordinate62_changes_and_parent_is_read_only(self):
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

    def test_05_invalid_control_or_fp16_only_parent_rejected(self):
        actual = self.data["layers"][9]["parent"]
        for control in ({}, {**d.plan()[0], "label": "foreign"},
                        {**d.plan()[0], "input_Q24_units62": float(d.PASSING)}):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "unknown"):
                d.input_parent(actual, control)
        with self.assertRaisesRegex(ValueError, "FP16-only"):
            d.input_parent({"h": actual["h"]}, d.plan()[0])

    def test_06_contract_thresholds_and_claims_unchanged(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        self.assertEqual(contract["thresholds"], json.loads(d.base.legacy.CONTRACT.read_text())["thresholds"])
        for key in contract:
            if key not in ("validation", "refinement", "claim_boundary"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                    d.check_contract({**contract, key: None})

    def test_07_repository_origins_and_foreign_source_rejection(self):
        for name in (d.MODULE, d.TEST_MODULE, d.prior.MODULE,
                     "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            self.assertTrue(Path(self.origins[name]["path"]).is_relative_to(d.ROOT))
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()
        with patch.object(d, "CONTRACT", Path("/tmp/foreign.json")):
            with self.assertRaisesRegex(ValueError, "contract origin mismatch"):
                d.source_context()

    def test_08_original_input_global_reference_chain(self):
        refs = d.base.validation_evidence(json.loads(d.CONTRACT.read_text()), self.data)[
            "original_input_global_reference_bindings"]
        predecessor = refs["original_binary64_parent"]
        self.assertEqual([r["layer"] for r in refs["layers"]], list(d.LAYERS))
        for row in refs["layers"]:
            self.assertEqual(row["input_binary64"], predecessor)
            self.assertEqual(self.data["bound"][row["binary64"]["path"]], row["binary64"])
            predecessor = row["binary64"]
        extension = copy.deepcopy(self.data["extension"])
        extension["layers"]["10"]["input_binary64"] = extension["original_binary64_parent"]
        with self.assertRaisesRegex(ValueError, "predecessor mismatch"):
            d.base.validation_evidence(json.loads(d.CONTRACT.read_text()),
                                       {**self.data, "extension": extension})

    def test_09_observed_metadata_and_history_changes_rejected(self):
        for key, value in (("status", "PASS"), ("native_L0_L8_invocations", 1),
                           ("rtl_invocations", 1), ("candidate_admitted", True),
                           ("monotonicity_claim", True), ("control_count", 9)):
            with self.subTest(key=key):
                self.reject({**self.observed, key: value}, "metadata mismatch")
        self.reject({**self.observed, "v1_refinement_evidence": {}}, "history linkage")

    def test_10_observed_source_or_reference_substitution_rejected(self):
        origins = {**self.observed["origins_after_execution"]}
        origins[d.prior.MODULE] = {**origins[d.prior.MODULE], "sha256": "0" * 64}
        self.reject({**self.observed, "origins_after_execution": origins}, "source mismatch")
        self.reject({**self.observed, "original_input_global_reference_bindings": {}},
                    "reference binding mismatch")
        self.reject({**self.observed, "reviewed_policy_bindings": []}, "linkage mismatch")

    def test_11_observed_dispatch_or_bracket_changes_rejected(self):
        for key in ("native_execution_records", "native_stage_execution_records"):
            self.reject({**self.observed, key: []}, "dispatch accounting")
        self.reject({**self.observed, "L13_all_sample_transition_brackets": []}, "bracket mismatch")

    def test_12_observed_state_KV_gate_failures_are_not_erased(self):
        rows = list(self.observed["controls"])
        layers = list(rows[0]["layers"])
        reports = list(layers[0]["reports"])
        for key in ("kv_lineage", "residual_state_lineage"):
            changed = [{**reports[0], key: "FAIL"}, *reports[1:]]
            changed_layers = [{**layers[0], "reports": changed}, *layers[1:]]
            changed_rows = [{**rows[0], "layers": changed_layers}, *rows[1:]]
            self.reject({**self.observed, "controls": changed_rows}, "state/KV/gate mismatch")
        self.assertTrue(any(not row["all_L9_L13_gates_pass"] for row in rows))

    def test_13_changed_artifact_digest_rejected(self):
        item = {**self.observed["artifacts"][0], "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "binding"):
            d.entry.upstream.bind_input(item, dict(self.data["bound"]))

    def test_14_zero_dispatch_blocks_native_and_rtl_publication(self):
        for index in range(6):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                with d.zero_dispatch() as guards:
                    guards[index]()

    def test_15_check_only_cli_validates_once(self):
        with patch.object(d, "validate", return_value={"status": "mocked_check"}) as validate, \
                patch.object(d, "execute_sweep") as sweep, \
                patch.object(d.entry, "write") as write, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(d.main(["--check"]), 0)
            validate.assert_called_once_with()
            self.assertEqual(json.loads(output.getvalue())["status"], "mocked_check")
            sweep.assert_not_called()
            write.assert_not_called()

    def test_16_cli_rejects_execution_without_validation(self):
        with patch.object(d, "validate") as validate, \
                patch("sys.stderr", new_callable=io.StringIO):
            for argv in ([], ["--out"], ["--check", "--out", "build/unused"], ["--rtl"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as raised:
                    d.main(argv)
                self.assertEqual(raised.exception.code, 2)
            validate.assert_not_called()

    def test_17_unauthorized_execution_rejected_before_authentication(self):
        routes = [{"layers": (layer, 10, 11, 12, 13)} for layer in range(9)]
        routes += [{"layers": value} for value in ((), (9,), (9, 10, 11, 12, 14), None)]
        routes += [{"position": value} for value in (1, False, 0.0)]
        routes += [{"backend": value} for value in ("rtl", "gpu", None)]
        with patch.object(d, "authenticate") as auth, patch.object(d, "source_context") as origins, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            for route in routes:
                with self.subTest(route=route), self.assertRaisesRegex(ValueError, "execution"):
                    d.execute_sweep(**route)
            for guard in (auth, origins, native):
                guard.assert_not_called()

    def test_18_dispatch_and_stage_guards_reject_unplanned_work(self):
        self.assertEqual(d.execution_plan(), d.prior.execution_plan())
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
        for owner, name in ((d.prior.subprocess, "Popen"), (d.entry.os, "system")):
            with patch.object(owner, name) as external:
                with self.assertRaisesRegex(AssertionError, "forbidden"):
                    with d.native_execution(d.execution_plan()):
                        getattr(owner, name)("forbidden")
                external.assert_not_called()

    def test_19_output_scope_is_fresh_exclusive_and_versioned(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_directory(path), path)
            for bad in (d.ROOT / "build" / d.OUTPUT_PREFIX, d.ROOT / "build/foreign",
                        d.ROOT / (d.OUTPUT_PREFIX + "outside"), path / "nested",
                        d.ROOT / "build" / (d.prior.OUTPUT_PREFIX + "old")):
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, "scope"):
                    d.output_directory(bad)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_directory(path)
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_directory(path)

    def test_20_mocked_sweep_archives_all_controls_and_preserves_failures(self):
        frozen = {}
        for integer in (d.PASSING, d.FAILING):
            row = next(r for r in self.observed["controls"] if r["input_Q24_units62"] == integer)
            for item in row["layers"]:
                saved = d.entry.native.load_state(item["state_operand_KV_local_archive"],
                    lambda binding: d.entry.upstream.bind_input(binding, self.data["bound"]))
                frozen[integer, item["layer"]] = saved, item["reports"]
        calls, archive_paths = [], []
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        real_open, real_record = Path.open, d.entry.record

        def fake_open(file, mode="r", *args, **kwargs):
            if file.parent == path:
                self.assertEqual(mode, "xb")
                archive_paths.append(file)
                return io.BytesIO()
            return real_open(file, mode, *args, **kwargs)

        def fake_record(file):
            if file.parent == path:
                return {"path": str(file), "bytes": 0, "sha256": "synthetic"}
            return real_record(file)

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

        def fake_archive(stream, **arrays):
            self.assertIsInstance(stream, io.BytesIO)
            for kind in ("input_parent", "output_parent", "arrays", "local_references"):
                self.assertTrue(any(key.startswith(kind + "__") for key in arrays))
            for kind in ("input_parent", "output_parent"):
                self.assertTrue(all(f"{kind}__{key}" in arrays for key in ("i", "z", "h")))

        log = io.StringIO()
        with patch.object(d, "authenticate", return_value=(self.data, self.observed)), \
                patch.object(d.entry.upstream, "execute_layer", side_effect=fake_native) as native, \
                patch.object(d.entry.native, "stages", return_value=None) as stages, \
                patch.object(d.entry.torch, "set_num_threads"), \
                patch.object(d.base.legacy.runtime, "reauthenticate") as reauthenticate, \
                patch.object(Path, "open", new=fake_open), \
                patch.object(d.entry, "record", side_effect=fake_record), \
                patch.object(d.entry.np, "savez_compressed", side_effect=fake_archive) as archive:
            result = d.execute_sweep(path, log)
        self.assertEqual((native.call_count, stages.call_count, archive.call_count), (85, 85, 85))
        self.assertEqual(len(set(archive_paths)), 85)
        self.assertEqual(len(log.getvalue().splitlines()), 17)
        reauthenticate.assert_called_once_with(self.data)
        self.assertEqual(result["native_execution_records"], d.execution_plan())
        self.assertEqual(result["native_stage_execution_records"], d.execution_plan())
        self.assertTrue(result["observed_endpoint_bitwise_reproduction"])
        self.assertEqual(result["native_L0_L8_invocations"], 0)
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertLessEqual(result["smallest_observed_transition_width_Q24_units"], 256)
        self.assertEqual(len(result["L13_all_sample_transition_brackets"]), 15)
        self.assertTrue(any(not row["all_L9_L13_gates_pass"] for row in result["controls"]))
        self.assertEqual(result["status"], "DIAGNOSED")
        self.assertEqual(len(result["artifacts"]), 85)
        self.assertEqual([row["control"] for row in result["controls"]], d.plan())
        for row in result["controls"]:
            for item in row["layers"]:
                self.assertIn(item["state_operand_KV_local_archive"], result["artifacts"])
                for kind in ("input_parent", "output_parent", "arrays", "local_references"):
                    self.assertNotIn(kind, item)
        self.assertEqual(result["original_input_global_reference_bindings"],
                         self.observed["original_input_global_reference_bindings"])
        self.assertEqual(result["v2_refinement_evidence"], self.data["v2_refinement_evidence"])
        self.assertEqual(result["v1_refinement_evidence"]["historical_status"], "BLOCKED")
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "exact_threshold_claim", "monotonicity_claim", "accepted_L0_L8_execution"):
            self.assertIs(result[key], False)

    def test_21_source_mismatch_rejected_before_authentication_and_dispatch(self):
        with patch.object(d, "authenticate") as auth, \
                patch.object(d.entry.upstream, "execute_layer") as native:
            with self.assertRaisesRegex(ValueError, "changed before execution"):
                d.execute_sweep(validation={"origins": {}, "contract": {}})
            auth.assert_not_called()
            native.assert_not_called()

    def test_22_execution_cli_validates_once_with_separate_publication(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        validation = {"origins": {}, "contract": {}}
        result = {"status": "DIAGNOSED", "native_L0_L8_invocations": 0,
                  "native_L9_L13_invocations": 85, "rtl_invocations": 0, "native_layer_count": 85,
                  "native_execution_records": d.execution_plan(),
                  "native_stage_execution_records": d.execution_plan()}
        with patch.object(d, "output_directory", return_value=path), \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "open", return_value=io.StringIO()) as opened, \
                patch.object(d, "validate", return_value=validation) as validate, \
                patch.object(d, "execute_sweep", return_value=result) as sweep, \
                patch.object(d.entry, "write") as write, \
                patch.object(d.entry, "record", return_value={"synthetic": True}), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(d.main(["--out", str(path)]), 0)
            validate.assert_called_once_with()
            mkdir.assert_called_once_with(exist_ok=False)
            opened.assert_called_once_with("x")
            sweep.assert_called_once_with(path, opened.return_value, validation=validation)
            self.assertEqual([call.args[0] for call in write.call_args_list], [
                path / "preflight_validation.json", path / "validation.json", path / "result.json"])
            self.assertEqual(result["validation"], {"synthetic": True})
            self.assertEqual(json.loads(output.getvalue())["status"], "DIAGNOSED")

    def test_23_failed_preflight_or_sweep_never_publishes_a_result(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        for phase in ("preflight", "sweep"):
            with self.subTest(phase=phase), \
                    patch.object(d, "output_directory", return_value=path), \
                    patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=io.StringIO()), \
                    patch.object(d, "validate", return_value={},
                                 side_effect=ValueError("preflight failed") if phase == "preflight"
                                 else None) as validate, \
                    patch.object(d, "execute_sweep", side_effect=ValueError("sweep failed")) as sweep, \
                    patch.object(d.entry, "write") as write:
                with self.assertRaisesRegex(ValueError, phase + " failed"):
                    d.main(["--out", str(path)])
                validate.assert_called_once_with()
                if phase == "preflight":
                    sweep.assert_not_called()
                    write.assert_not_called()
                else:
                    sweep.assert_called_once()
                    self.assertEqual([call.args[0] for call in write.call_args_list],
                                     [path / "preflight_validation.json"])

    def test_24_endpoint_operands_and_gate_lineage_must_reproduce(self):
        endpoint = next(row for row in self.observed["controls"]
                        if row["input_Q24_units62"] == d.PASSING)
        item = endpoint["layers"][0]
        saved = d.entry.native.load_state(item["state_operand_KV_local_archive"],
            lambda binding: d.entry.upstream.bind_input(binding, self.data["bound"]))
        arrays = {k.removeprefix("arrays__"): v for k, v in saved.items() if k.startswith("arrays__")}
        locals_ = {k.removeprefix("local_references__"): v for k, v in saved.items()
                   if k.startswith("local_references__")}
        for key in ("kv_lineage", "residual_state_lineage"):
            reports = [{**item["reports"][0], key: "FAIL"}, *item["reports"][1:]]
            with self.subTest(key=key), patch.object(d.base.legacy, "execute_layer",
                    return_value=(arrays, locals_, reports)), \
                    self.assertRaisesRegex(ValueError, "incomplete native gate evidence"):
                d.execute_control(d.plan()[0], self.data, self.observed)
        changed = {key: value.copy() for key, value in locals_.items()}
        first = next(iter(changed))
        changed[first].flat[0] += 1
        with patch.object(d.base.legacy, "execute_layer",
                          return_value=(arrays, changed, item["reports"])) as native:
            with self.assertRaises(ValueError):
                d.execute_control(d.plan()[0], self.data, self.observed)
            native.assert_called_once()

    def test_25_authentication_failure_never_reaches_native_execution(self):
        with patch.object(d, "authenticate", side_effect=ValueError("changed frozen evidence")), \
                patch.object(d.entry.upstream, "execute_layer") as native, \
                patch.object(d, "native_execution") as dispatch, \
                patch.object(d.entry, "write") as write:
            with self.assertRaisesRegex(ValueError, "changed frozen evidence"):
                d.execute_sweep()
            native.assert_not_called()
            dispatch.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
