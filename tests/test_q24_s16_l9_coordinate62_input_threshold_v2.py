"""Read-only frozen evidence and synthetic CLI/dispatch tests, never native replay."""

import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_input_threshold_v2 as d


EXPECTED_TESTS = 30
# Exercise real publication inside the validation suite's outer no-output guards.
REAL_MKDIR = Path.mkdir
REAL_WRITE = d.entry.write


class InputThresholdV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(d.entry.upstream, "execute_layer",
                          side_effect=AssertionError("native authentication dispatch")) as dispatch, \
                patch.object(d.entry.native, "stages",
                             side_effect=AssertionError("native authentication stages")) as stages:
            cls.data = d.authenticate()
            dispatch.assert_not_called()
            stages.assert_not_called()
        cls.result_binding = cls.data["input_threshold_evidence"][1]["result"]
        cls.reviewed = json.loads(Path(cls.result_binding["path"]).read_text())

    def forged(self, reviewed):
        return patch.object(d.producer, "read_result", return_value=(self.result_binding, reviewed))

    def test_all_reviewed_bindings_authenticate_read_only(self):
        for evidence, pin in zip(self.data["input_threshold_evidence"], (d.REVIEWED_L9, d.REVIEWED_L10)):
            self.assertEqual(evidence["result"]["sha256"], pin[1])
            self.assertTrue(evidence["read_only"])
            reviewed = json.loads(Path(evidence["result"]["path"]).read_text())
            items = reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]
            items += [item for item in reviewed["origins_after_execution"].values() if "path" in item]
            for item in items:
                self.assertEqual(self.data["bound"][item["path"]],
                                 {key: item[key] for key in ("path", "bytes", "sha256")})
        self.assertEqual(self.reviewed["selected_L9_evidence"], self.data["selected_L9_evidence"])

    def test_v1_source_contract_and_tests_remain_frozen(self):
        for path, digest in (
            (Path(d.legacy.__file__), "eb18da9dc6664632c474f544ac68b56c866789d1d1539064dff9d5c56838745a"),
            (d.legacy.CONTRACT, "7d2a42e426cc811e7d5f4e1825227861517899cab2093f867286a1fae6134807"),
            (d.ROOT / "tests/test_q24_s16_l9_coordinate62_input_threshold_v1.py",
             "2074af4f89f63e63bc0a9a05053edd4f3de1110076a86bf17b96dc0df6bb0de5"),
        ):
            self.assertEqual(d.entry.record(path)["sha256"], digest)

    def test_plan_and_arithmetic_are_unchanged_functions(self):
        self.assertIs(d.plan, d.legacy.plan)
        self.assertIs(d.input_parent, d.legacy.input_parent)
        self.assertIs(d.threshold_intervals, d.legacy.threshold_intervals)
        self.assertEqual(len(d.plan()), 13)
        self.assertEqual(d.LAYERS, (9, 10, 11, 12, 13))

    def test_selected_parent_preserves_retained_reference_and_kv(self):
        d.entry.paired.same_arrays(self.data["layers"][9]["output"], self.data["layers"][10]["parent"])
        layer = self.data["layers"][10]
        self.assertIn("reference", layer)
        self.assertIn("trajectory", layer)
        self.assertIn("tensors", layer)
        self.assertTrue(all(report["kv_lineage"] == report["residual_state_lineage"] == "PASS"
                            for report in layer["reports"]))
        self.assertEqual(self.data["input_threshold_evidence"][1]["policy_binding"]["policy_id"],
                         d.entry.prior.gates.POLICY_ID)

    def test_historical_l9_cannot_replace_selected_pin(self):
        historical = d.entry.record(d.ROOT / "build" /
            "q24_s16_l9_coordinate62_entry_producer_cone_d0755047099e_attempt001/result.json")
        with patch.object(d.entry, "record", return_value=historical):
            with self.assertRaisesRegex(ValueError, "wrong pinned producer result"):
                d.producer.read_result(d.REVIEWED_L9, {})

    def test_historical_l10_cannot_replace_v2_pin(self):
        historical = d.entry.record(d.ROOT / "build" / d.producer.HISTORICAL[0] / "result.json")
        with patch.object(d.entry, "record", return_value=historical):
            with self.assertRaisesRegex(ValueError, "wrong pinned producer result"):
                d.authenticate_l10({"bound": {}})

    def test_historical_l10_is_rejected_even_if_re_pinned(self):
        with patch.object(d, "REVIEWED_L10", d.producer.HISTORICAL):
            with self.assertRaisesRegex(ValueError, "metadata mismatch: diagnostic_id"):
                d.authenticate_l10({"bound": {}})

    def test_source_mismatches_are_rejected(self):
        for name in (d.entry.MODULE, d.producer.MODULE, d.legacy.MODULE):
            reviewed = copy.deepcopy(self.reviewed)
            reviewed["origins_after_execution"][name]["sha256"] = "0" * 64
            with self.subTest(module=name), self.forged(reviewed):
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    d.authenticate_l10({**self.data, "bound": {}})

    def test_missing_required_origin_is_rejected(self):
        reviewed = copy.deepcopy(self.reviewed)
        del reviewed["origins_after_execution"][d.producer.MODULE]
        with self.forged(reviewed), self.assertRaisesRegex(ValueError, "source origin mismatch"):
            d.authenticate_l10({**self.data, "bound": {}})

    def test_changed_selected_l9_linkage_is_rejected(self):
        reviewed = {**self.reviewed, "selected_L9_evidence": {}}
        with self.forged(reviewed), self.assertRaisesRegex(ValueError, "selected L9 linkage"):
            d.authenticate_l10(self.data)

    def test_changed_l10_contract_is_rejected(self):
        reviewed = {**self.reviewed, "contract": {}}
        with self.forged(reviewed), self.assertRaisesRegex(ValueError, "v2 contract mismatch"):
            d.authenticate_l10(self.data)

    def test_changed_policy_and_scope_are_rejected(self):
        for key, value in (("policy_id", "wrong"), ("accepted_L0_L8_execution", True),
                           ("rtl_invocations", 1), ("normal_host_review", "OPTIONAL")):
            with self.subTest(key=key), self.forged({**self.reviewed, key: value}):
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    d.authenticate_l10(self.data)

    def test_changed_artifact_binding_is_rejected(self):
        reviewed = copy.deepcopy(self.reviewed)
        reviewed["artifacts"][0]["sha256"] = "0" * 64
        with self.forged(reviewed), self.assertRaisesRegex(ValueError, "conflicting binding: .*retained_authentication"):
            d.authenticate_l10(self.data)

    def test_contract_rejects_changed_normative_fields(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        for key in ("diagnostic_id", "policy_id", "cwd", "command", "selected_L9", "reviewed_L10_v2",
                    "output_prefix", "fresh_exclusive_directory", "controls", "native_layers",
                    "position", "coordinate", "max_native_layer_evaluations", "native_layer_invocations",
                    "rtl_invocations", "normal_host_review", "candidate_admitted", "policy_adopted",
                    "successor_published", "scientific_result_claim", "accepted_L0_L8_execution",
                    "thresholds"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contract mismatch"):
                d.check_contract({**contract, key: None})

    def test_all_module_origins_are_repository_local(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.producer.MODULE, d.legacy.MODULE,
                     "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            self.assertTrue(Path(origins[name]["path"]).is_relative_to(d.ROOT))
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_check_creates_no_output_and_never_sweeps(self):
        validation = {
            "status": "synthetic",
            **d.validation_evidence(json.loads(d.CONTRACT.read_text()), self.data),
        }
        thresholds = json.loads(d.legacy.CONTRACT.read_text())["thresholds"]
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value=validation) as validate, \
                patch.object(d, "execute_sweep") as sweep, patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "open") as opened, patch.object(d.entry, "write") as write, \
                patch("builtins.print") as output:
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            sweep.assert_not_called()
            mkdir.assert_not_called()
            opened.assert_not_called()
            write.assert_not_called()
            output.assert_called_once_with(json.dumps(validation))
            emitted = json.loads(output.call_args.args[0])
            self.assertEqual(emitted["thresholds"], thresholds)
            self.assertEqual(emitted["original_input_global_reference_bindings"],
                             validation["original_input_global_reference_bindings"])

    def test_cli_modes_are_required_and_exclusive(self):
        for args in ([], ["--check", "--out", "build/example"]):
            with patch.object(d.sys, "argv", [d.MODULE, *args]), \
                    patch.object(d.sys, "stderr", io.StringIO()), \
                    patch.object(d, "validate") as validate, patch.object(d, "execute_sweep") as sweep:
                with self.assertRaises(SystemExit) as error:
                    d.main()
                self.assertEqual(error.exception.code, 2)
                validate.assert_not_called()
                sweep.assert_not_called()

    def test_output_accepts_only_fresh_versioned_direct_children(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_directory(out), out)
            self.assertEqual(d.output_directory(out.relative_to(d.ROOT)), out)

    def test_output_rejects_old_prefixes_nested_and_outside_paths(self):
        for path in (Path("/tmp") / (d.OUTPUT_PREFIX + "synthetic"),
                     d.ROOT / "build/nested" / (d.OUTPUT_PREFIX + "synthetic"),
                     d.ROOT / "build" / d.OUTPUT_PREFIX, d.ROOT / "build/unrelated",
                     d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic") / "..",
                     *(d.ROOT / "build" / (prefix + "synthetic") for prefix in d.legacy.OUTPUT_PREFIXES)):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "versioned bounded"):
                d.output_directory(path)

    def test_existing_and_symlink_outputs_are_rejected_before_validation(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        for symlink, exists, message in ((True, False, "symlink"), (False, True, "already exists")):
            with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                    patch.object(Path, "is_symlink", return_value=symlink), \
                    patch.object(Path, "exists", return_value=exists), \
                    patch.object(d, "validate") as validate, patch.object(d, "execute_sweep") as sweep, \
                    patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, message):
                    d.main()
                validate.assert_not_called()
                sweep.assert_not_called()
                mkdir.assert_not_called()

    def test_exclusive_creation_race_never_validates_or_dispatches(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "output_directory", return_value=out), \
                patch.object(Path, "mkdir", side_effect=FileExistsError("race")) as mkdir, \
                patch.object(d, "validate") as validate, patch.object(d, "execute_sweep") as sweep:
            with self.assertRaises(FileExistsError):
                d.main()
            mkdir.assert_called_once_with(exist_ok=False)
            validate.assert_not_called()
            sweep.assert_not_called()

    def test_out_validates_once_and_publishes_only_bound_result(self):
        validation = {"origins": {}, "contract": {}}
        result = {"status": "DIAGNOSED", "origins_after_execution": {}, "contract": {},
                  "native_execution_records": [], "native_L0_L8_invocations": 0,
                  "rtl_invocations": 0, "native_layer_count": 0}
        events = []

        def publish(path, document):
            events.append(path.name)
            REAL_WRITE(path, document)

        with TemporaryDirectory(prefix=d.OUTPUT_PREFIX + "test_", dir=d.ROOT / "build") as temporary:
            out = Path(temporary) / "output"
            preflight_bytes = []

            def sweep(*args):
                events.append("sweep")
                preflight_bytes.append((out / "preflight_validation.json").read_bytes())
                self.assertEqual(json.loads(preflight_bytes[0]), validation)
                self.assertFalse((out / "validation.json").exists())
                self.assertFalse((out / "result.json").exists())
                return result

            with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                    patch.object(d, "output_directory", return_value=out), \
                    patch.object(Path, "mkdir", autospec=REAL_MKDIR, side_effect=REAL_MKDIR) as mkdir, \
                    patch.object(d, "validate",
                                 side_effect=lambda: events.append("validate") or validation) as validate, \
                    patch.object(d, "execute_sweep", side_effect=sweep), \
                    patch.object(d.entry, "write", side_effect=publish), patch("builtins.print"):
                self.assertEqual(d.main(), 0)
                mkdir.assert_called_once_with(out, exist_ok=False)
                validate.assert_called_once_with()
            self.assertEqual(events, ["validate", "preflight_validation.json", "sweep",
                                      "validation.json", "result.json"])
            self.assertEqual((out / "preflight_validation.json").read_bytes(), preflight_bytes[0])
            final = json.loads((out / "validation.json").read_text())
            self.assertNotIn("sweep_execution", json.loads(preflight_bytes[0]))
            self.assertEqual(final, validation)
            self.assertEqual(final["sweep_execution"]["native_layer_count"], 0)
            self.assertEqual(result["validation"], d.entry.record(out / "validation.json"))
            self.assertEqual(json.loads((out / "result.json").read_text()), result)
            for name in ("preflight_validation.json", "validation.json", "result.json"):
                path = out / name
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    REAL_WRITE(path, {})
                self.assertEqual(path.read_bytes(), before)

    def test_failed_validation_never_dispatches_or_publishes(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "output_directory", return_value=out), patch.object(Path, "mkdir"), \
                patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", side_effect=ValueError("synthetic validation failure")), \
                patch.object(d, "execute_sweep") as sweep, patch.object(d.entry, "write") as write:
            with self.assertRaisesRegex(ValueError, "synthetic validation failure"):
                d.main()
            sweep.assert_not_called()
            write.assert_not_called()

    def test_sweep_uses_private_v2_bindings_and_unchanged_v1_arithmetic(self):
        globals_before = d.legacy.execute_sweep.__globals__.copy()

        def factory(code, namespace, name):
            self.assertIs(code, d.legacy.execute_sweep.__code__)
            self.assertIs(namespace["authenticate"], d.authenticate)
            self.assertIs(namespace["source_context"], d.source_context)
            self.assertEqual(namespace["ID"], d.ID)
            self.assertEqual(namespace["CONTRACT"], d.CONTRACT)
            for key in ("execute_control", "input_parent", "plan", "threshold_intervals", "entry", "runtime"):
                self.assertIs(namespace[key], globals_before[key])
            return lambda out, log: (out, log)

        with patch.object(d, "FunctionType", side_effect=factory):
            self.assertEqual(d.execute_sweep("synthetic-out", "synthetic-log"),
                             ("synthetic-out", "synthetic-log"))
        self.assertEqual(d.legacy.execute_sweep.__globals__, globals_before)

    def test_native_dispatch_guard_rejects_all_out_of_scope_nodes(self):
        with patch.object(d.entry.upstream, "execute_layer", return_value="synthetic") as dispatch:
            for layer in (*range(9), *range(14, 24), -1, True, 9.0):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "restricted"):
                    d.legacy.execute_layer(layer, 0, {}, {})
            for position in (-1, 1, True, 0.0):
                with self.assertRaisesRegex(ValueError, "restricted"):
                    d.legacy.execute_layer(9, position, {}, {})
            dispatch.assert_not_called()
            for layer in d.LAYERS:
                data = {"layers": {layer: {"tensors": {}, "trajectory": [], "reference": []}}}
                self.assertEqual(d.legacy.execute_layer(layer, 0, {}, data), "synthetic")
            self.assertEqual(dispatch.call_count, 5)

    def test_native_stage_guard_and_accounting_fail_closed(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic")
        for layer in range(9):
            with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                    patch.object(d, "output_directory", return_value=out), patch.object(Path, "mkdir"), \
                    patch.object(Path, "open", return_value=io.StringIO()), \
                    patch.object(d, "validate", return_value={}), patch.object(d.entry, "write") as write, \
                    patch.object(d, "execute_sweep",
                                 side_effect=lambda *args: d.entry.native.stages({}, layer, {}, {})):
                with self.assertRaisesRegex(ValueError, "outside L9-L13"):
                    d.main()
                self.assertEqual(write.call_count, 1)
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "output_directory", return_value=out), patch.object(Path, "mkdir"), \
                patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", return_value={}), patch.object(d.entry, "write") as write, \
                patch.object(d, "execute_sweep", return_value={"native_execution_records": [9]}):
            with self.assertRaisesRegex(ValueError, "accounting mismatch"):
                d.main()
            self.assertEqual(write.call_count, 1)

    def test_thresholds_and_nonmonotonic_transitions_are_preserved(self):
        self.assertEqual(json.loads(d.CONTRACT.read_text())["thresholds"],
                         json.loads(d.legacy.CONTRACT.read_text())["thresholds"])
        rows = [{"control": control, "layers": [
            {"layer": layer, "index62": {"accepted": control["step"] in (1, 2)}}
            for layer in d.LAYERS]} for control in d.plan()]
        intervals = d.threshold_intervals(rows, 13)
        self.assertEqual([(row["left_step"], row["right_step"]) for row in intervals], [(0, 1), (2, 3)])
        self.assertEqual([row["left_accepted"] for row in intervals], [False, True])
        with self.assertRaisesRegex(ValueError, "incomplete or reordered"):
            d.threshold_intervals(rows[:-1], 13)

    def test_validation_reports_authenticated_original_input_reference_bindings(self):
        contract = json.loads(d.CONTRACT.read_text())
        evidence = d.validation_evidence(contract, self.data)
        self.assertEqual(evidence["thresholds"], contract["thresholds"])
        self.assertEqual(evidence["thresholds"], json.loads(d.legacy.CONTRACT.read_text())["thresholds"])
        self.assertIs(evidence["unchanged_baseline_gates"], True)
        self.assertIs(evidence["original_global_reference_unchanged"], True)
        references = evidence["original_input_global_reference_bindings"]
        extension = self.data["extension"]
        for key in ("original_specification", "original_binary64_parent"):
            self.assertEqual(references[key], extension[key])
            self.assertEqual(references[key], self.data["bound"][references[key]["path"]])
        self.assertEqual([row["layer"] for row in references["layers"]], list(d.LAYERS))
        predecessor = references["original_binary64_parent"]
        for row in references["layers"]:
            self.assertEqual(row["position"], 0)
            self.assertEqual(row["input_binary64"], predecessor)
            for key in ("input_binary64", "binary64"):
                self.assertEqual(row[key], extension["layers"][str(row["layer"])][key])
                self.assertEqual(row[key], self.data["bound"][row[key]["path"]])
            predecessor = row["binary64"]
        self.assertEqual(json.loads(json.dumps(evidence)), evidence)

    def test_validation_rejects_missing_or_changed_reference_authentication(self):
        contract = json.loads(d.CONTRACT.read_text())
        references = d.validation_evidence(contract, self.data)["original_input_global_reference_bindings"]
        bindings = [references["original_specification"], references["original_binary64_parent"]]
        bindings.extend(row[key] for row in references["layers"] for key in ("input_binary64", "binary64"))
        for binding in bindings:
            for missing in (True, False):
                bound = self.data["bound"].copy()
                if missing:
                    del bound[binding["path"]]
                else:
                    bound[binding["path"]] = {**binding, "sha256": "0" * 64}
                with self.subTest(path=binding["path"], missing=missing):
                    with self.assertRaisesRegex(ValueError, "unauthenticated original-input global-reference"):
                        d.validation_evidence(contract, {**self.data, "bound": bound})

    def test_validation_rejects_spliced_original_input_reference_chain(self):
        contract = json.loads(d.CONTRACT.read_text())
        for layer in d.LAYERS:
            extension = copy.deepcopy(self.data["extension"])
            extension["layers"][str(layer)]["input_binary64"] = extension["layers"][str(layer)]["binary64"]
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "predecessor mismatch"):
                d.validation_evidence(contract, {**self.data, "extension": extension})


if __name__ == "__main__":
    unittest.main()
