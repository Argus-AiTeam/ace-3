"""Repo-bound v3 interface tests; no native control or accepted-prefix replay."""

import ast
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v3 as d


EXPECTED_TESTS = 30


class ExecutionSurfaceTests(unittest.TestCase):
    def test_contract_matches_pinned_v2_science(self):
        contract = json.loads(d.CONTRACT.read_text())
        d.check_contract(contract)
        base = json.loads(d.v2.CONTRACT.read_text())
        for key in d.v2.PRESERVED_FIELDS:
            self.assertEqual(contract[key], base[key])

    def test_contract_rejects_every_changed_field(self):
        original = json.loads(d.CONTRACT.read_text())
        for key in original:
            with self.subTest(key=key):
                changed = deepcopy(original)
                changed[key] = None
                with self.assertRaises(ValueError):
                    d.check_contract(changed)

    def test_unknown_contract_overrides_rejected(self):
        contract = json.loads(d.CONTRACT.read_text())
        contract["threshold_override"] = 1
        with self.assertRaisesRegex(ValueError, "fields changed"):
            d.check_contract(contract)

    def test_history_is_exact_and_read_only(self):
        records = d.history_records()
        self.assertEqual(len(records), 6)
        for item in records:
            self.assertEqual(item["sha256"], d.HISTORY[str(Path(item["path"]).relative_to(d.ROOT))])

    def test_history_mutation_rejected(self):
        with patch.object(d, "record", return_value={"path": "changed", "sha256": "bad"}):
            with self.assertRaisesRegex(ValueError, "history changed"):
                d.history_records()

    def test_v3_origins_are_repo_bound(self):
        origins = d.origins()
        for name in (d.MODULE, d.TEST_MODULE, d.v2.MODULE, d.v2.TEST_MODULE,
                     "tests.test_q24_s16_toward_zero_l3_l8_v1"):
            self.assertEqual(origins[name]["path"],
                             str(d.ROOT.joinpath(*name.split(".")).with_suffix(".py")))

    def test_foreign_module_origin_rejected(self):
        with patch.dict(d.sys.modules, {"ace3.model.foreign": SimpleNamespace(__file__="/tmp/foreign.py")}):
            with self.assertRaisesRegex(ValueError, "module origin mismatch"):
                d.origins()

    def test_foreign_entry_origin_rejected(self):
        with patch.object(d, "__file__", "/tmp/foreign.py"):
            with self.assertRaisesRegex(ValueError, "entry source origin mismatch"):
                d.origins()

    def test_foreign_contract_origin_rejected(self):
        with patch.object(d, "CONTRACT", Path("/tmp/foreign.json")):
            with self.assertRaisesRegex(ValueError, "contract origin mismatch"):
                d.origins()

    def test_command_local_context_required(self):
        with patch.dict(d.os.environ, {"PYTHONPATH": "/tmp"}):
            with self.assertRaisesRegex(ValueError, "repository-bound command"):
                d.origins()

    def test_retained_and_current_bindings_remain_distinct(self):
        for relative, update in d.v2.SOURCE_UPDATES.items():
            retained, current, snapshot = d.v2.source_records(relative, update)
            records = {current["path"]: current, snapshot["path"]: snapshot}
            bound, updates = {}, {}
            with patch.object(d.upstream, "record", side_effect=lambda p: records[str(p)]):
                self.assertEqual(d.v2.bind_input(retained, bound, updates), Path(snapshot["path"]))
            self.assertEqual(bound, records)
            self.assertFalse(updates[relative]["historical_source_relabelled"])

    def test_changed_retained_or_current_source_rejected(self):
        for relative, update in d.v2.SOURCE_UPDATES.items():
            retained, current, snapshot = d.v2.source_records(relative, update)
            for target in (current, snapshot):
                records = {current["path"]: current, snapshot["path"]: snapshot}
                records[target["path"]] = {**target, "sha256": "bad"}
                with patch.object(d.upstream, "record", side_effect=lambda p: records[str(p)]):
                    with self.assertRaisesRegex(ValueError, "binding mismatch"):
                        d.v2.bind_input(retained, {}, {})

    def test_recheck_rejects_changed_bound_input(self):
        item = {"path": str(d.ROOT / "build/bound-input"), "bytes": 1, "sha256": "original"}
        with patch.object(d, "record", return_value={**item, "sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "authenticated input changed"):
                d.recheck([item])

    def test_all_non_l13_dispatch_rejected_before_native(self):
        with patch.object(d.upstream, "execute_layer") as dispatch:
            for layer in list(range(13)) + list(range(14, 24)) + [True, 13.0, "13"]:
                with self.subTest(layer=layer):
                    with self.assertRaisesRegex(ValueError, "only isolated native L13"):
                        d.execute_layer({}, layer, {}, {}, np.zeros(896))
            dispatch.assert_not_called()

    def test_l13_dispatch_keeps_actual_operands_and_original_reference(self):
        tensors, parent, trajectory, reference = object(), object(), object(), object()
        with patch.object(d.upstream, "execute_layer", return_value="native-result") as dispatch:
            self.assertEqual(d.execute_layer(tensors, 13, parent, trajectory, reference), "native-result")
            dispatch.assert_called_once_with(tensors, 13, parent, trajectory, reference)

    def test_no_rtl_or_prefix_execution_entrypoints(self):
        tree = ast.parse(Path(d.__file__).read_text())
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "execute_layers", "continuation_stages",
                                 "publish_parent", "run_rtl", "simulate", "_stages"})
        self.assertNotIn("subprocess", {node.name for node in ast.walk(tree) if isinstance(node, ast.alias)})

    def test_unchanged_science_and_exact_threshold(self):
        for name in ("interventions", "intervene", "classify", "compare_parents", "check_reports"):
            self.assertIs(getattr(d, name), getattr(d.legacy, name))
        self.assertEqual(len(list(d.interventions())), 954)
        self.assertTrue(d.prior.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.prior.measure(0x3c81, 1.0)["accepted"])
        reference = float.fromhex("0x1.8bd7b2092532cp+10")
        self.assertFalse(d.prior.measure(0x6630, reference)["accepted"])
        self.assertTrue(d.prior.measure(0x662f, reference)["accepted"])

    def test_fresh_versioned_output_path(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_test")
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_path(out), out)
        for path in (d.ROOT / "build/old_v1", d.ROOT / (d.OUTPUT_PREFIX + "bad"),
                     d.ROOT / "build" / d.OUTPUT_PREFIX):
            with self.assertRaisesRegex(ValueError, "v3 build scope"):
                d.output_path(path)

    def test_existing_output_and_symlink_rejected(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_test")
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(out)
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(out)

    def test_cli_rejects_ambiguous_or_missing_mode_and_output(self):
        for args in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--out", "unused"], ["--input", "unused"]):
            with patch.object(d.sys, "argv", [d.MODULE, *args]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    d.main()
                self.assertEqual(error.exception.code, 2)

    def test_check_cli_never_creates_output_or_dispatches(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(d.sys, "argv", [d.MODULE, "--check"]))
            stack.enter_context(patch.object(d, "validate", return_value={"status": "fixture"}))
            output = stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            mkdir = stack.enter_context(patch.object(Path, "mkdir"))
            dispatch = stack.enter_context(patch.object(d, "diagnose"))
            self.assertEqual(d.main(), 0)
            self.assertEqual(json.loads(output.getvalue()), {"status": "fixture"})
            mkdir.assert_not_called()
            dispatch.assert_not_called()

    def test_execute_cli_refuses_output_creation_race(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_test")
        with ExitStack() as stack:
            stack.enter_context(patch.object(d.sys, "argv", [d.MODULE, "--execute", "--out", str(out)]))
            stack.enter_context(patch.object(d, "output_path", return_value=out))
            stack.enter_context(patch.object(d, "validate", return_value={}))
            mkdir = stack.enter_context(patch.object(Path, "mkdir", side_effect=FileExistsError("exists")))
            dispatch = stack.enter_context(patch.object(d, "diagnose"))
            with self.assertRaises(FileExistsError):
                d.main()
            mkdir.assert_called_once_with(exist_ok=False)
            dispatch.assert_not_called()

    def test_execute_cli_wires_fresh_output_and_nonadmission(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_test")
        result = {"status": "DIAGNOSED", "candidate_admitted": False, "policy_adopted": False,
                  "successor_published": False}
        validation = {"authentication": "fixture"}
        with ExitStack() as stack:
            stack.enter_context(patch.object(d.sys, "argv", [d.MODULE, "--execute", "--out", str(out)]))
            stack.enter_context(patch.object(d, "output_path", return_value=out))
            stack.enter_context(patch.object(d, "validate", return_value=validation))
            stack.enter_context(patch.object(Path, "mkdir"))
            # Only command.log is mocked; the real contract is still read.
            original_open = Path.open
            stack.enter_context(patch.object(Path, "open", autospec=True, side_effect=lambda p, *a, **kw:
                io.StringIO() if p == out / "command.log" else original_open(p, *a, **kw)))
            dispatch = stack.enter_context(patch.object(d, "diagnose", return_value=result))
            writes = stack.enter_context(patch.object(d, "write"))
            stack.enter_context(patch.object(d, "record", return_value={"path": "fixture"}))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            self.assertEqual(d.main(), 0)
            self.assertEqual(dispatch.call_count, 1)
            self.assertEqual(dispatch.call_args.args[0], out)
            self.assertIs(dispatch.call_args.args[3], validation)
            self.assertEqual(writes.call_args.args[0], out / "result.json")
            for key in ("candidate_admitted", "policy_adopted", "successor_published"):
                self.assertIs(writes.call_args.args[1][key], False)

    def test_suite_rejects_incomplete_collection(self):
        module = SimpleNamespace(__name__="empty")
        with patch.object(d.unittest.defaultTestLoader, "loadTestsFromModule",
                          return_value=unittest.TestSuite()):
            with self.assertRaisesRegex(ValueError, "wrong focused collection"):
                d.run_suite(module, 1)

    def test_suite_rejects_skip_failure_and_error(self):
        def skipped():
            raise unittest.SkipTest("not evidence")

        def failed():
            raise AssertionError("not evidence")

        def errored():
            raise RuntimeError("not evidence")

        for function in (skipped, failed, errored):
            suite = unittest.TestSuite([unittest.FunctionTestCase(function)])
            with patch.object(d.unittest.defaultTestLoader, "loadTestsFromModule", return_value=suite):
                with self.assertRaisesRegex(ValueError, "focused validation failed"):
                    d.run_suite(SimpleNamespace(__name__="negative-fixture"), 1)

    def test_delivery_flags_and_review_boundary(self):
        contract = json.loads(d.CONTRACT.read_text())
        for key in ("candidate_admitted", "policy_adopted", "successor_published", "scientific_result_claim"):
            self.assertIs(contract[key], False)
        self.assertEqual(contract["native_layers_per_control"], [13])
        self.assertEqual(contract["maximum_native_layer_invocations"], 954)
        self.assertEqual(contract["native_L0_L8_invocations"], 0)
        self.assertEqual(contract["rtl_invocations"], 0)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")
        self.assertIn("separate bounded task", contract["execution_boundary"])

    def test_original_reference_chain_rejects_substitution_and_splices(self):
        extension = {
            "reference_only": True, "policy_id": d.prior.gates.POLICY_ID,
            "global_reference_policy": "legacy-binary64-AWQ-fully-independent-propagation",
            "original_binary64_parent": "binary64-8", "original_fp16_parent": "fp16-8",
            "layers": {str(layer): {
                "input_binary64": f"binary64-{layer - 1}", "input_fp16": f"fp16-{layer - 1}",
                "binary64": f"binary64-{layer}", "fp16": f"fp16-{layer}", "prior_kv": "own empty P0"}
                for layer in range(9, 14)},
        }
        d.check_reference_chain(extension)
        for key in ("reference_only", "policy_id", "global_reference_policy",
                    "original_binary64_parent", "original_fp16_parent"):
            changed = deepcopy(extension)
            changed[key] = None
            with self.assertRaises(ValueError):
                d.check_reference_chain(changed)
        for layer in range(9, 14):
            for key in ("input_binary64", "input_fp16", "prior_kv"):
                changed = deepcopy(extension)
                changed["layers"][str(layer)][key] = "mapped-or-other-layer"
                with self.assertRaisesRegex(ValueError, "spliced original-input reference"):
                    d.check_reference_chain(changed)

    def test_parent_receipt_rejects_state_kv_and_lineage_changes(self):
        freeze_record = {"path": "freeze"}
        freeze = {"contract": {"candidate_id": "candidate", "state_id": "state"}}
        previous = {"output_parent": "L12-state", "input_parent": "L11-state",
                    "reports": "L12-reports", "kv_state": "L12-KV"}
        entry = {"input_parent": "L12-state"}
        receipt = {
            "next_layer": 13, "position": 0, "history": [9707], "rtl_admissible": False,
            "evidence_kind": "cpu_software_q24", "candidate_id": "candidate", "state_id": "state",
            "policy_id": d.prior.gates.POLICY_ID, "arithmetic_lineage": freeze_record,
            "state": "L12-state", "state_lineage_parent": "L11-state",
            "numerical_report": "L12-reports", "kv": "L12-KV",
        }
        d.check_parent_receipt(receipt, freeze_record, freeze, previous, entry)
        for key in receipt:
            changed = deepcopy(receipt)
            changed[key] = None
            with self.assertRaisesRegex(ValueError, "receipt lineage mismatch"):
                d.check_parent_receipt(changed, freeze_record, freeze, previous, entry)

    def test_loader_reauthenticates_before_reading_or_dispatch(self):
        binding = {"path": str(d.ROOT / "build/bound-input"), "sha256": "original", "bytes": 1}
        with patch.object(d, "record", return_value={**binding, "sha256": "changed"}):
            with patch.object(d.native, "load_state") as load:
                with self.assertRaisesRegex(ValueError, "authenticated input changed"):
                    d.load_data({"input_bindings": [binding]})
                load.assert_not_called()

    def test_execution_reports_blocker_instead_of_success(self):
        out = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_test")
        with ExitStack() as stack:
            stack.enter_context(patch.object(d.sys, "argv", [d.MODULE, "--execute", "--out", str(out)]))
            stack.enter_context(patch.object(d, "output_path", return_value=out))
            stack.enter_context(patch.object(d, "validate", return_value={}))
            stack.enter_context(patch.object(Path, "mkdir"))
            original_open = Path.open
            stack.enter_context(patch.object(Path, "open", autospec=True, side_effect=lambda p, *a, **kw:
                io.StringIO() if p == out / "command.log" else original_open(p, *a, **kw)))
            stack.enter_context(patch.object(d, "diagnose", side_effect=ValueError("lineage mismatch")))
            writes = stack.enter_context(patch.object(d, "write"))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            self.assertEqual(d.main(), 2)
            result = writes.call_args.args[1]
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["missing_evidence"], ["ValueError: lineage mismatch"])
            self.assertIs(result["candidate_admitted"], False)


if __name__ == "__main__":
    unittest.main()
