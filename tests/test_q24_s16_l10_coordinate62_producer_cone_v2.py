"""Read-only authentication and synthetic runtime wiring; zero native dispatch."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l10_coordinate62_producer_cone_v2 as d


EXPECTED_TESTS = 20


class SelectedParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = d.authenticate()

    def test_selected_current_source_and_all_recorded_bindings(self):
        evidence = self.data["selected_L9_evidence"]
        self.assertEqual(evidence["result"]["sha256"],
                         "c36ed0cb236b06b2fd72e2f73502f38f022e55c7fb67bf0f0b2fc705915539d4")
        self.assertEqual(evidence["entry_source"], d.entry.record(Path(d.entry.__file__)))
        reviewed = json.loads(Path(evidence["result"]["path"]).read_text())
        items = reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]
        items += [r for r in reviewed["origins_after_execution"].values() if "path" in r]
        for item in items:
            self.assertEqual(self.data["bound"][item["path"]],
                             {k: item[k] for k in ("path", "bytes", "sha256")})
        self.assertEqual(evidence["policy_binding"]["policy_id"], d.entry.prior.gates.POLICY_ID)
        self.assertTrue(evidence["read_only"])

    def test_selected_parent_is_consumed_without_reference_or_kv_substitution(self):
        parent = self.data["layers"][10]["parent"]
        binding = self.data["selected_L9_evidence"]["parent"]
        frozen = d.entry.native.load_state(
            binding, lambda item: d.entry.upstream.bind_input(item, self.data["bound"]))
        d.entry.paired.same_arrays(parent, frozen)
        d.entry.paired.same_arrays(parent, self.data["layers"][9]["output"])
        layer = self.data["layers"][10]
        references = (layer["reference"], layer["trajectory"], layer["tensors"])
        original = {key: np.zeros(896, dtype="<f8")
                    for key in ("input", "s11", "s17", "residual", "s18")}
        _, output = d.cut_parent(layer, original, "actual")
        d.entry.paired.same_arrays(
            output, d.entry.prior.retained.state_from(layer["arrays"], "output", "stage18"))
        self.assertTrue(all(a is b for a, b in zip(
            references, (layer["reference"], layer["trajectory"], layer["tensors"]))))

    def test_historical_result_and_old_entry_source_are_rejected_not_rebound(self):
        item, historical = d.read_result(d.HISTORICAL, {})
        self.assertEqual(item["sha256"],
                         "8830027d6e37ec77ad6d0a9b9815cedef0925d962b747fd38f84071e32f415cd")
        old = historical["origins_after_execution"][d.entry.MODULE]
        self.assertEqual(old["sha256"],
                         "75c8c7faa069e7b3fd8edd1b5c8cf1afe9db836f2401bf61d15f259c96b6e1d6")
        self.assertNotEqual(old, self.data["selected_L9_evidence"]["entry_source"])
        with self.assertRaisesRegex(ValueError, "binding mismatch: .*l9_coordinate62_entry"):
            d.entry.upstream.bind_input(old, {})
        self.assertEqual(d.entry.record(Path(item["path"])), item)

    def test_wrong_selected_result_pin_rejected(self):
        item = {**self.data["selected_L9_evidence"]["result"], "sha256": "wrong"}
        with patch.object(d.entry, "record", return_value=item):
            with self.assertRaisesRegex(ValueError, "wrong pinned producer result"):
                d.selected_parent({})

    def test_old_source_and_tampered_artifact_bindings_rejected(self):
        item, reviewed = d.read_result(d.SELECTED, {})
        _, historical = d.read_result(d.HISTORICAL, {})
        variants = []
        old = copy.deepcopy(reviewed)
        old["origins_after_execution"][d.entry.MODULE] = historical["origins_after_execution"][d.entry.MODULE]
        variants.append((old, "binding mismatch: .*l9_coordinate62_entry"))
        tampered = copy.deepcopy(reviewed)
        tampered["artifacts"][0]["sha256"] = "wrong"
        variants.append((tampered, "conflicting binding"))
        for value, message in variants:
            with self.subTest(message=message), patch.object(d, "read_result", return_value=(item, value)):
                bound = self.data["bound"].copy() if value is tampered else {}
                with self.assertRaisesRegex(ValueError, message):
                    d.selected_parent(bound)

    def test_selected_parent_mismatch_cannot_replace_authenticated_state(self):
        parent = {k: v.copy() for k, v in self.data["layers"][10]["parent"].items()}
        parent["i"][62] += 1
        current = {
            "parent": parent, "arrays": self.data["layers"][9]["arrays"],
            "L10_arrays": self.data["layers"][10]["arrays"],
            "L10_reports": self.data["layers"][10]["reports"],
            "evidence": self.data["selected_L9_evidence"],
        }
        with patch.object(d.entry, "authenticate", return_value=self.data), \
                patch.object(d, "selected_parent", return_value=current):
            with self.assertRaises(ValueError):
                d.authenticate()

    def test_missing_selected_parent_artifact_rejected(self):
        item, reviewed = d.read_result(d.SELECTED, {})
        reviewed["artifacts"] = [r for r in reviewed["artifacts"]
                                 if not r["path"].endswith("/actual_L9_parent.npz")]
        with patch.object(d, "read_result", return_value=(item, reviewed)):
            with self.assertRaisesRegex(ValueError, "missing or duplicate selected artifact"):
                d.selected_parent(self.data["bound"].copy())

    def test_mandatory_gate_report_change_rejected(self):
        current = {
            "parent": self.data["layers"][10]["parent"],
            "arrays": self.data["layers"][9]["arrays"],
            "L10_arrays": self.data["layers"][10]["arrays"],
            "L10_reports": copy.deepcopy(self.data["layers"][10]["reports"]),
            "evidence": self.data["selected_L9_evidence"],
        }
        report = current["L10_reports"][0]
        report["status"] = "FAIL" if report["status"] == "PASS" else "PASS"
        with patch.object(d.entry, "authenticate", return_value=self.data), \
                patch.object(d, "selected_parent", return_value=current):
            with self.assertRaises(ValueError):
                d.authenticate()

    def test_unchanged_arithmetic_plan_reference_and_thresholds(self):
        self.assertIs(d.plan, d.legacy.plan)
        self.assertIs(d.cut_parent, d.legacy.cut_parent)
        self.assertIs(d.original_branches, d.legacy.original_branches)
        self.assertEqual(len(d.plan()), 13)
        self.assertTrue(d.entry.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.entry.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.entry.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.entry.prior.measure(0x662f, reference)["accepted"])

    def test_foreign_module_origin_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_cli_only_checks_without_output_creation(self):
        with patch.object(d.sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"status": "synthetic"}) as validate, \
                patch.object(d, "diagnose") as diagnose, \
                patch.object(Path, "mkdir") as mkdir, patch("builtins.print") as output:
            self.assertEqual(d.main(), 0)
            validate.assert_called_once_with()
            mkdir.assert_not_called()
            diagnose.assert_not_called()
            output.assert_called_once_with(json.dumps({"status": "synthetic"}))

    def test_historical_candidate_and_contract_files_remain_frozen(self):
        _, historical = d.read_result(d.HISTORICAL, {})
        d.entry.upstream.bind_input(historical["origins_after_execution"][d.legacy.MODULE], {})
        gate = self.data["selected_L9_evidence"]["policy_binding"]["gate"]
        self.assertEqual(d.entry.record(Path(gate["path"])), gate)

    def test_output_path_requires_fresh_direct_bounded_directory(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic_test")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_path(out), out)
            self.assertEqual(d.output_path(out.relative_to(d.ROOT)), out)
            for invalid in (
                out / "nested", d.ROOT / out.name,
                d.ROOT / "build" / "wrong_prefix",
                d.ROOT / "build" / d.OUTPUT_PREFIX,
                d.ROOT / "build" / ".." / out.name,
            ):
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    d.output_path(invalid)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_path(out)
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(out)

    def test_cli_modes_are_required_and_exclusive(self):
        with patch.object(d, "validate") as validate, \
                patch.object(d, "diagnose") as diagnose, \
                patch.object(Path, "mkdir") as mkdir, patch.object(d.sys, "stderr", io.StringIO()):
            for args in ([], ["--check", "--out", "build/unused"]):
                with patch.object(d.sys, "argv", [d.MODULE, *args]):
                    with self.assertRaises(SystemExit) as error:
                        d.main()
                    self.assertEqual(error.exception.code, 2)
            validate.assert_not_called()
            diagnose.assert_not_called()
            mkdir.assert_not_called()

    def test_cli_out_validates_once_then_publishes_bound_result(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic_test")
        validation = {"status": "synthetic"}
        result = {"status": "DIAGNOSED", "diagnostic_id": d.ID}
        events = []

        def diagnose(path, log, checked):
            self.assertEqual(events, ["validate", "validation.json"])
            self.assertEqual(path, out)
            self.assertIs(checked, validation)
            events.append("diagnose")
            return result

        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "output_path", return_value=out), \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(Path, "open", return_value=io.StringIO()) as opened, \
                patch.object(d, "validate", side_effect=lambda: events.append("validate") or validation) as validate, \
                patch.object(d, "diagnose", side_effect=diagnose), \
                patch.object(d.entry, "write", side_effect=lambda p, v: events.append(p.name)) as write, \
                patch.object(d.entry, "record", return_value={"path": "synthetic_validation"}), \
                patch("builtins.print"):
            self.assertEqual(d.main(), 0)
            mkdir.assert_called_once_with(exist_ok=False)
            opened.assert_called_once_with("x")
            validate.assert_called_once_with()
            self.assertEqual(events, ["validate", "validation.json", "diagnose", "result.json"])
            self.assertEqual(write.call_args.args, (out / "result.json", result))
            self.assertEqual(result["validation"], {"path": "synthetic_validation"})

    def test_cli_existing_output_or_creation_race_cannot_validate_or_dispatch(self):
        out = d.ROOT / "build" / d.HISTORICAL[0]
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "validate") as validate, patch.object(d, "diagnose") as diagnose, \
                patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.main()
            mkdir.assert_not_called()
            with patch.object(d, "output_path", return_value=out):
                mkdir.side_effect = FileExistsError("synthetic creation race")
                with self.assertRaises(FileExistsError):
                    d.main()
            validate.assert_not_called()
            diagnose.assert_not_called()

    def test_failed_validation_never_dispatches_or_publishes_result(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "synthetic_test")
        with patch.object(d.sys, "argv", [d.MODULE, "--out", str(out)]), \
                patch.object(d, "output_path", return_value=out), \
                patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=io.StringIO()), \
                patch.object(d, "validate", side_effect=ValueError("synthetic validation failure")), \
                patch.object(d, "diagnose") as diagnose, patch.object(d.entry, "write") as write:
            with self.assertRaisesRegex(ValueError, "synthetic validation failure"):
                d.main()
            diagnose.assert_not_called()
            write.assert_not_called()

    def test_v1_body_is_reused_with_isolated_selected_parent_bindings(self):
        validation = {"origins": {"synthetic": True}, "contract": d.entry.record(d.CONTRACT)}
        original_globals = d.legacy.diagnose.__globals__.copy()
        expected = [10]
        for label, _ in d.plan():
            if label == "inherited_native":
                expected.append(10)
            expected.extend((11, 12, 13))
        self.assertEqual(len(expected), 41)

        def adapter(code, namespace, name):
            self.assertIs(code, d.legacy.diagnose.__code__)
            self.assertEqual(name, d.legacy.diagnose.__name__)
            self.assertEqual(namespace["ID"], d.ID)
            self.assertIs(namespace["origins"], d.source_context)
            self.assertEqual({key for key in namespace
                              if namespace[key] is not original_globals[key]},
                             {"ID", "origins", "authenticate"})
            self.assertIs(namespace["authenticate"](), self.data)

            def runner(out, log):
                for layer in expected:
                    d.entry.native.stages({}, layer, {}, {})
                return {"diagnostic_id": namespace["ID"], "native_layer_dispatches": expected}
            return runner

        with patch.object(d, "source_context", return_value=validation["origins"]), \
                patch.object(d, "authenticate", return_value=self.data) as authenticate, \
                patch.object(d, "FunctionType", side_effect=adapter) as factory, \
                patch.object(d.entry.native, "stages", return_value=None) as synthetic_stages:
            result = d.diagnose(d.ROOT / "build" / "unused", io.StringIO(), validation)
            authenticate.assert_called_once_with()
            factory.assert_called_once()
            self.assertEqual(synthetic_stages.call_count, 41)
            self.assertEqual(result["selected_L9_evidence"], self.data["selected_L9_evidence"])
            self.assertEqual(result["native_stage_dispatches"], expected)
            self.assertEqual(result["contract"], validation["contract"])
        self.assertTrue(all(d.legacy.diagnose.__globals__[key] is value
                            for key, value in original_globals.items()))

    def test_native_stage_guard_and_accounting_fail_closed(self):
        validation = {"origins": {}, "contract": d.entry.record(d.CONTRACT)}
        for layer in (*range(10), 14, True, 10.0):
            def runner(out, log):
                d.entry.native.stages({}, layer, {}, {})
            with self.subTest(layer=layer), \
                    patch.object(d, "source_context", return_value={}), \
                    patch.object(d, "FunctionType", return_value=runner), \
                    patch.object(d.entry.native, "stages") as stages:
                with self.assertRaisesRegex(ValueError, "native stage execution outside"):
                    d.diagnose(d.ROOT / "build" / "unused", io.StringIO(), validation)
                stages.assert_not_called()
        with patch.object(d, "source_context", return_value={}), \
                patch.object(d, "FunctionType", return_value=lambda out, log: {"native_layer_dispatches": []}):
            with self.assertRaisesRegex(ValueError, "accounting mismatch"):
                d.diagnose(d.ROOT / "build" / "unused", io.StringIO(), validation)

    def test_reused_native_dispatch_rejects_every_layer_outside_l10_l13(self):
        with patch.object(d.entry.upstream, "execute_layer") as dispatch:
            for layer in (*range(10), 14, True, 10.0, "10"):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "outside L10-L13"):
                    d.legacy.execute_layer(layer, {}, {}, [])
            dispatch.assert_not_called()
        parent = {"synthetic": True}
        calls = []
        with patch.object(d.legacy, "execute_layer", return_value=({}, {}, [])) as execute, \
                patch.object(d.entry.prior.retained, "state_from", return_value=parent):
            d.legacy.execute_suffix(parent, {}, lambda layer, *args: calls.append(layer), [])
        self.assertEqual(calls, [11, 12, 13])
        self.assertEqual([call.args[0] for call in execute.call_args_list], [11, 12, 13])


if __name__ == "__main__":
    unittest.main()
