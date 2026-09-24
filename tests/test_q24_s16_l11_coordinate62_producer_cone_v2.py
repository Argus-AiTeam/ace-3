"""Real retained binding and rejection tests; no diagnostic or native execution."""

import ast
import copy
from contextlib import ExitStack
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import preflight_q24_s16_l11_coordinate62_producer_cone_v2 as d


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = d.authenticate()

    def test_current_repository_origins(self):
        origins = d.source_context()
        for name in (d.MODULE, d.TEST_MODULE, d.legacy.ANCESTOR_TEST):
            self.assertEqual(origins[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))
        self.assertTrue(all(Path(item["path"]).is_relative_to(d.ROOT)
                            for item in origins.values() if "path" in item))

    def test_foreign_module_rejected(self):
        with patch.dict(sys.modules, {"ace3.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.source_context()

    def test_command_local_context_required(self):
        with patch.dict(d.os.environ, {"PYTHONPATH": "/tmp"}):
            with self.assertRaisesRegex(ValueError, "repository-bound command"):
                d.source_context()

    def test_contract_and_non_admission(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        self.assertEqual(contract["interface"], ["--check"])
        for key in ("execution_authorized", "candidate_admitted", "policy_adopted",
                    "successor_published", "scientific_result_claim"):
            self.assertIs(contract[key], False)

    def test_threshold_or_execution_contract_change_rejected(self):
        for key, value in (("thresholds", {}), ("execution_authorized", True),
                           ("native_layer_invocations", 1), ("rtl_invocations", 1)):
            contract = json.loads(d.CONTRACT.read_text())
            contract[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(contract)

    def test_v1_terminal_failure_preserved(self):
        items = d.history_records()
        self.assertEqual(len(items), 7)
        failed = json.loads((d.FAILED / "validation.json").read_text())
        self.assertEqual((failed["collected"], failed["executed"], failed["errors"]), (20, 0, 1))
        self.assertIn("binding mismatch", (d.FAILED / "unittest.log").read_text())

    def test_v1_mutation_rejected(self):
        original = d.record

        def changed(path):
            item = original(path)
            return {**item, "sha256": "wrong"} if path == d.legacy.CONTRACT else item

        with patch.object(d, "record", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "v1 history changed"):
                d.history_records()

    def test_retained_and_current_sources_are_distinct(self):
        updates = self.data["authentication"]["source_updates"]
        self.assertEqual(set(updates), set(d.retained.coordinate.v2.SOURCE_UPDATES))
        for item in updates.values():
            self.assertFalse(item["historical_source_relabelled"])
            self.assertNotEqual(item["retained_original"]["sha256"], item["current"]["sha256"])
            self.assertEqual(d.record(Path(item["current"]["path"])), item["current"])
            self.assertEqual(d.record(Path(item["retained_snapshot"]["path"])), item["retained_snapshot"])

    def test_stale_binding_still_rejected_by_strict_binder(self):
        for item in self.data["authentication"]["source_updates"].values():
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                d.upstream.bind_input(item["retained_original"], {})

    def test_unrecognized_source_drift_rejected(self):
        binder = d.retained.coordinate.v2
        update = next(iter(self.data["authentication"]["source_updates"].values()))
        item = {**update["retained_original"], "sha256": "unrecognized"}
        with self.assertRaisesRegex(ValueError, "unexpected historical source binding"):
            binder.bind_input(item, {}, {})

    def test_foreign_and_tampered_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside isolated repository"):
            d.upstream.bind_input({"path": "/tmp/input", "bytes": 0, "sha256": "wrong"}, {})
        item = self.data["L11_binding"]["L11_input"]
        with self.assertRaises(ValueError):
            d.upstream.bind_input({**item, "sha256": "wrong"}, self.data["bound"].copy())

    def test_actual_l10_l11_lineage_and_receipts(self):
        evidence = self.data["L11_binding"]
        self.assertEqual(evidence["L10_output"], evidence["L11_input"])
        receipt = json.loads(Path(evidence["L10_receipt"]["path"]).read_text())
        self.assertEqual(receipt["state"], evidence["L11_input"])
        self.assertEqual(receipt["next_layer"], 11)
        d.paired.same_arrays(self.data["layers"][10]["output"], self.data["layers"][11]["parent"])
        d.prior.retained.verify_parent(
            self.data["layers"][11]["output"],
            d.prior.retained.state_from(self.data["layers"][11]["arrays"], "output", "stage18"))

    def test_spliced_actual_l11_parent_rejected(self):
        data = {**self.data, "layers": self.data["layers"].copy()}
        layer = self.data["layers"][11]
        data["layers"][11] = {**layer, "parent": {k: v.copy() for k, v in layer["parent"].items()}}
        data["layers"][11]["parent"]["i"][62] += 1
        with self.assertRaises(ValueError):
            d.check_l11_lineage(data)

    def test_original_global_reference_chain_not_actual_state(self):
        refs = self.data["extension"]["layers"]
        for layer in (11, 12, 13):
            self.assertEqual(refs[str(layer)]["input_binary64"], refs[str(layer - 1)]["binary64"])
        binding = self.data["L11_binding"]
        self.assertNotEqual(binding["L11_input"]["path"], binding["L11_original_input"]["path"])
        self.assertEqual(binding["L11_original_output"], refs["11"]["binary64"])

    def test_spliced_reference_rejected(self):
        extension = copy.deepcopy(self.data["extension"])
        extension["layers"]["11"]["input_binary64"] = self.data["L11_binding"]["L11_input"]
        with self.assertRaisesRegex(ValueError, "spliced original-input reference"):
            d.retained.coordinate.check_reference_chain(extension)

    def test_retained_gates_kv_and_canonical_operands(self):
        layer = self.data["layers"][11]
        for stage, report in enumerate(layer["reports"]):
            self.assertEqual(report["node"], [11, 0, stage])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["policy_id"], d.prior.gates.POLICY_ID)
            self.assertEqual(report["residual_state_lineage"], "PASS")
            self.assertEqual(report["kv_lineage"], "PASS")
        self.assertEqual(len(layer["reports"]), 19)
        d.prior.local.authenticate_tensors(
            layer["tensors"], self.data["extension"]["layers"]["11"]["canonical"], 11)
        changed = {**layer["tensors"]}
        key = "model.layers.11.self_attn.q_proj.qweight"
        changed[key] = changed[key].copy()
        changed[key].flat[0] ^= 1
        with self.assertRaises(ValueError):
            d.prior.local.authenticate_tensors(
                changed, self.data["extension"]["layers"]["11"]["canonical"], 11)

    def test_exact_threshold_and_retained_failure(self):
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        failed = d.prior.measure(0x6630, reference)
        self.assertFalse(failed["accepted"])
        self.assertEqual(Fraction(failed["excess_budget"]), Fraction(1, 8))
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])
        self.assertEqual(self.data["actual_arrays"]["stage18"][62], 0x6630)

    def test_check_cli_validates_once(self):
        with patch.object(sys, "argv", [d.MODULE, "--check"]), \
                patch.object(d, "validate", return_value={"status": "synthetic"}) as validate, \
                patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(d.main(), 0)
        validate.assert_called_once_with()

    def test_cli_cannot_execute_or_resume(self):
        with patch.object(d, "validate") as validate, patch.object(sys, "stderr", io.StringIO()):
            for args in ([], ["--execute"], ["--check", "--out", str(d.FAILED)], ["--resume"]):
                with self.subTest(args=args), patch.object(sys, "argv", [d.MODULE, *args]):
                    with self.assertRaises(SystemExit) as error:
                        d.main()
                    self.assertEqual(error.exception.code, 2)
        validate.assert_not_called()

    def test_dispatch_guards_and_static_no_execution_surface(self):
        with ExitStack() as stack:
            guards = d.dispatch_guards(stack)
            with self.assertRaisesRegex(AssertionError, "dispatch forbidden"):
                d.upstream.execute_layer()
            self.assertEqual(sum(guard.call_count for guard in guards), 1)
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"diagnose", "execute_layer", "execute_layers", "stages",
                                 "Popen", "system", "original_branches", "publish_parent"})


if __name__ == "__main__":
    unittest.main()
