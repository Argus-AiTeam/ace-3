"""Zero-native-dispatch L16 execution-surface and frozen L15 lineage tests."""

from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l16_from_l15_coordinate62_suffix_v1 as d


EXPECTED_TESTS = 46


class L16SuffixPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.result = json.loads((d.SELECTED / "result.json").read_text())
        cls.rows = json.loads((d.SELECTED / "interventions.json").read_text())
        cls.reports = json.loads((d.SELECTED / "actual_L15_gates.json").read_text())
        validation = json.loads(Path(cls.result["validation"]["path"]).read_text())
        freeze = json.loads(Path(validation["retained_freeze"]["path"]).read_text())
        cls.extension = json.loads(Path(freeze["reference_extension"]["path"]).read_text())

    def test_contract_matches(self):
        d.check_contract(self.contract)

    def test_thresholds_unchanged(self):
        for key in self.contract["thresholds"]:
            changed = deepcopy(self.contract)
            changed["thresholds"][key] = "relaxed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_non_admission_and_zero_dispatch_flags(self):
        for key in (*d.FLAGS, "native_control_dispatch_authorized",
                    "execution_surface_declared", "accepted_prefix_replay", "scientific_result_claim"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "changed"})

    def test_parent_pins_cannot_change(self):
        for key in ("selected_result", "selected_result_sha256", "inherits_sha256", "parent_review"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "substituted"})

    def test_consumer_is_only_l16_p0(self):
        for layer, position in ((True, 0), (15, 0), (17, 0), (16, 1)):
            with self.subTest(layer=layer, position=position), self.assertRaises(ValueError):
                d.check_contract({**self.contract, "consumer": {
                    "layer": layer, "position": position, "stages": list(range(19))}})

    def test_disclosed_command_and_execution_interface(self):
        for key, value in (("command", "python --check"), ("cwd", "/tmp"),
                           ("interface", ["--check"]), ("execution_command", "python --execute")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: value})

    def test_exact_thirteen_planned_controls(self):
        bounds = self.contract["planned_bounds"]
        self.assertEqual(bounds, d.planned_bounds())
        self.assertEqual(bounds["planned_native_layer_invocations"], 13)
        self.assertEqual(bounds["planned_native_L16_P0_invocations"], 13)
        self.assertEqual(bounds["layers"], [16])
        self.assertEqual(bounds["stages"], list(range(19)))
        self.assertEqual([row["control"] for row in bounds["schedule"]], list(d.CONTROLS))
        self.assertEqual(len(set(d.CONTROLS)), 13)
        self.assertEqual(bounds["native_L0_L15_invocations"], 0)
        self.assertEqual(bounds["rtl_invocations"], 0)

    def test_bounds_reject_replay_expansion_and_recomputation(self):
        for key, value in (("planned_native_layer_invocations", 14), ("layers", [15, 16]),
                           ("native_L0_L15_invocations", 1), ("rtl_invocations", 1),
                           ("reference_recomputation", True),
                           ("schedule", self.contract["planned_bounds"]["schedule"][:-1])):
            changed = deepcopy(self.contract)
            changed["planned_bounds"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_historical_l15_result_preserved(self):
        d.check_result(self.result)
        self.assertEqual(self.result["retained_status"], "FAIL")
        self.assertEqual(len(self.result["all_L15_gate_passing_controls"]), 9)

    def test_result_rejects_changed_claims_lineage_and_counts(self):
        for key, value in (("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True), ("unique_upstream_producer_attributed", True),
                           ("original_global_reference_unchanged", False),
                           ("source_operand_state_KV_lineage_checks", "FAIL"),
                           ("retained_status", "PASS"), ("native_layer_invocations", 12),
                           ("all_L15_gate_passing_controls", list(d.CONTROLS))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: value})

    def test_all_frozen_l15_controls_and_four_failures_retained(self):
        d.check_rows(self.rows)
        self.assertEqual(sum(row["all_L15_gates_pass"] for row in self.rows), 9)
        self.assertEqual([row["control"] for row in self.rows if not row["all_L15_gates_pass"]],
                         ["actual", "frozen_o", "frozen_down", "frozen_o_down"])

    def test_rows_reject_missing_duplicate_reordered_controls(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_reject_erased_failure_and_changed_gate(self):
        for key, value in (("mandatory_statuses", ["PASS"] * 19),
                           ("L15_status", "PASS"), ("L15_S18_failure_indices", [])):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_reject_kv_state_lineage_and_admission_changes(self):
        for key, value in (("prior_layer_kv_consumed", True), ("candidate_admitted", True),
                           ("L15_source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("prior_kv", "L15 KV"), ("conditional_acceptance_retained", True)):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_reports_match_retained_failure(self):
        d.check_reports(self.reports, self.rows[0])
        self.assertEqual(self.reports[18]["status"], "FAIL")

    def test_reports_reject_missing_and_wrong_stages(self):
        for reports in (self.reports[:-1], list(reversed(self.reports))):
            with self.subTest(count=len(reports)), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])

    def test_reports_reject_policy_state_kv_changes(self):
        for key in ("policy_id", "residual_state_lineage", "kv_lineage"):
            reports = deepcopy(self.reports)
            reports[0][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])

    def test_reports_reject_erased_global_failure(self):
        reports = deepcopy(self.reports)
        reports[18]["binary64_v1"]["failures"] = []
        with self.assertRaises(ValueError):
            d.check_reports(reports, self.rows[0])

    def test_immediate_parent_is_exact_paired_copy(self):
        with d.np.load(d.SELECTED / "actual_L15.npz", allow_pickle=False) as arrays:
            state = d.consumer_parent(arrays)
            for key, source in (("i", "output_i"), ("z", "output_z"), ("h", "stage18")):
                d.np.testing.assert_array_equal(state[key], arrays[source])
                self.assertFalse(d.np.shares_memory(state[key], arrays[source]))
        self.assertEqual(set(state), {"i", "z", "h"})

    def test_immediate_parent_rejects_fp16_only_and_spliced_state(self):
        with d.np.load(d.SELECTED / "actual_L15.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("output_i", "output_z", "stage18")}
        with self.assertRaises(KeyError):
            d.consumer_parent({"stage18": arrays["stage18"]})
        arrays["output_i"][62] = 0
        with self.assertRaises(ValueError):
            d.consumer_parent(arrays)

    def test_parent_does_not_carry_l15_kv(self):
        with d.np.load(d.SELECTED / "actual_L15.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["output_cache_k"].shape, (1, 128))
            self.assertEqual(set(d.consumer_parent(arrays)), {"i", "z", "h"})

    def test_changed_l15_source_fails_closed(self):
        original = d.record

        def changed(path):
            item = original(path)
            return {**item, "sha256": "0" * 64} if path == d.ROOT / next(iter(d.SOURCE_PINS)) else item

        with patch.object(d, "record", side_effect=changed), self.assertRaises(ValueError):
            d.history_records()

    def test_wrong_result_rejected_before_ancestry_authentication(self):
        with patch.object(d, "repository_record", return_value={"sha256": "0" * 64}), \
                patch.object(d.parent, "authenticate") as inherited, self.assertRaises(ValueError):
            d.authenticate({})
        inherited.assert_not_called()

    def test_independent_parent_review_required(self):
        review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
                  "mission_id": "9af9ceb10b68", "review": {"status": "done"}}
        d.check_parent_review(review)
        for key, value in (("producer_role", "engineer"), ("mission_id", "another-task"),
                           ("review", {"status": "blocked"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_parent_review({**review, key: value})

    def test_reference_extends_original_input_chain(self):
        self.assertIs(d.check_reference(self.extension), self.extension["layers"]["16"])

    def test_reference_rejects_reanchoring_and_prior_layer_kv(self):
        for layer, key in (("16", "input_binary64"), ("16", "input_fp16"),
                           ("16", "prior_kv"), ("15", "input_binary64"), ("12", "input_binary64")):
            extension = deepcopy(self.extension)
            extension["layers"][layer][key] = "actual-control"
            with self.subTest(layer=layer, key=key), self.assertRaises(ValueError):
                d.check_reference(extension)

    def test_reference_rejects_substituted_l16_arrays(self):
        for key in ("binary64", "fp16"):
            extension = deepcopy(self.extension)
            extension["layers"]["16"][key]["sha256"] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_reference(extension)

    def test_cli_rejects_invalid_modes_output_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--out", "build/forbidden"], ["--che"],
                     ["--exe", "--out", "build/forbidden"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), \
                    patch.object(d, "validate") as validation, self.assertRaises(SystemExit):
                d.main(argv)
            validation.assert_not_called()

    def test_cli_check_only_calls_validation(self):
        with patch.object(d, "validate", return_value={"L16_status": "NOT_EXECUTED"}) as check, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(d.main(["--check"]), 0)
        check.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), {"L16_status": "NOT_EXECUTED"})

    def test_candidate_origin_mismatch_rejected(self):
        with patch.object(d, "__file__", "/tmp/substituted.py"), self.assertRaises(ValueError):
            d.source_context()

    def test_execution_surface_does_not_change_check_dispatch_counts(self):
        for name in ("diagnose", "execute_layer", "NativeControls", "EXECUTION_COMMAND"):
            self.assertTrue(hasattr(d, name), name)
        self.assertEqual(d.FLAGS["native_layer_invocations"], 0)
        self.assertEqual(d.FLAGS["rtl_invocations"], 0)

    def test_non_repository_or_symlink_inputs_rejected_before_read(self):
        with patch.object(d, "record") as record:
            with self.assertRaises(ValueError):
                d.repository_record(Path("/tmp/substituted.json"))
            with patch.object(Path, "resolve", return_value=Path("/tmp/substituted.json")), \
                    self.assertRaises(ValueError):
                d.repository_record(d.CONTRACT)
        record.assert_not_called()

    def test_cli_declares_fresh_execution(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        args = d.parse_args(["--execute", "--out", str(path)])
        self.assertTrue(args.execute)
        self.assertFalse(args.check)
        self.assertEqual(args.out, path)
        self.assertEqual(d.EXECUTION_COMMAND, d.COMMAND.removesuffix("--check") +
                         f"--execute --out build/{d.OUTPUT_PREFIX}fresh")

    def test_execution_bounds_preserve_schedule_and_non_admission(self):
        bounds = self.contract["execution_bounds"]
        self.assertEqual(bounds, d.execution_bounds())
        self.assertEqual(bounds["schedule"], d.planned_bounds()["schedule"])
        self.assertEqual(bounds["native_layer_invocations"], 13)
        self.assertEqual(bounds["native_L16_P0_invocations"], 13)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(bounds[key], False)
        for key, value in (("layers", [15, 16]), ("native_layer_invocations", 14),
                           ("fresh_directory_required", False), ("ignored_directory_required", False),
                           ("symlinks_forbidden", False), ("separate_execution_task_required", False)):
            changed = deepcopy(self.contract)
            changed["execution_bounds"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_output_accepts_fresh_ignored_versioned_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_path_probe")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_path(path), path)

    def test_output_rejects_existing_paths(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "existing")
        with patch.object(Path, "exists", return_value=True), \
                patch.object(d.subprocess, "run") as git, self.assertRaises(ValueError):
            d.output_path(path)
        git.assert_not_called()

    def test_output_rejects_symlink_resolution(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "link")
        for resolved in (Path("/tmp/outside"), d.ROOT / "build" / (d.OUTPUT_PREFIX + "other")):
            with self.subTest(resolved=resolved), \
                    patch.object(Path, "resolve", return_value=resolved), \
                    self.assertRaises(ValueError):
                d.output_path(path)

    def test_output_rejects_dangling_symlink(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "dangling")
        with patch.object(Path, "resolve", return_value=path), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            d.output_path(path)

    def test_output_rejects_wrong_scope_prefix_and_traversal(self):
        for path in (Path("/tmp") / (d.OUTPUT_PREFIX + "fresh"), d.ROOT / "build" / "other",
                     d.ROOT / "build" / d.OUTPUT_PREFIX,
                     d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh") / "child",
                     d.ROOT / "build" / ".." / "build" / (d.OUTPUT_PREFIX + "fresh")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)

    def test_output_rejects_not_ignored_and_git_errors(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        for code in (1, 128):
            result = d.subprocess.CompletedProcess([], code, "", "git rejection")
            with self.subTest(code=code), patch.object(Path, "exists", return_value=False), \
                    patch.object(Path, "is_symlink", return_value=False), \
                    patch.object(d.subprocess, "run", return_value=result), \
                    self.assertRaisesRegex(ValueError, "output must be git-ignored"):
                d.output_path(path)

    def controls(self):
        with d.np.load(d.SELECTED / "actual_L15.npz", allow_pickle=False) as arrays:
            state = d.consumer_parent(arrays)
        parents = {label: {key: value.copy() for key, value in state.items()}
                   for label in d.CONTROLS}
        return parents, d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_dispatch_runs_thirteen_frozen_controls_once(self):
        parents, controls = self.controls()
        with patch.object(d, "execute_layer", return_value=({}, {}, [])) as execute:
            for label in d.CONTROLS:
                self.assertEqual(controls.run(label, parents[label]), ({}, {}, []))
            controls.finish()
        self.assertEqual(execute.call_count, 13)
        self.assertEqual(controls.count, 13)
        self.assertTrue(all(set(call.args[1]) == {"i", "z", "h"}
                            for call in execute.call_args_list))

    def test_dispatch_rejects_reordered_duplicate_and_exhausted_controls(self):
        parents, controls = self.controls()
        with patch.object(d, "execute_layer") as execute:
            with self.assertRaises(ValueError):
                controls.run(d.CONTROLS[1], parents[d.CONTROLS[1]])
            execute.assert_not_called()
            controls.run("actual", parents["actual"])
            with self.assertRaises(ValueError):
                controls.run("actual", parents["actual"])
            for label in d.CONTROLS[1:]:
                controls.run(label, parents[label])
            with self.assertRaises(ValueError):
                controls.run("actual", parents["actual"])
        self.assertEqual(execute.call_count, 13)

    def test_dispatch_rejects_spliced_state_before_native(self):
        parents, controls = self.controls()
        parents["actual"]["i"][62] += 1
        with patch.object(d, "execute_layer") as execute, self.assertRaises(ValueError):
            controls.run("actual", parents["actual"])
        execute.assert_not_called()
        self.assertEqual(controls.count, 0)

    def test_dispatch_rejects_missing_parents_and_partial_schedule(self):
        parents, controls = self.controls()
        with self.assertRaises(ValueError):
            controls.finish()
        del parents["actual"]
        with self.assertRaises(ValueError):
            d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_cli_rejects_bad_output_before_validation_or_dispatch(self):
        with patch.object(d, "validate") as validation, \
                patch.object(d, "diagnose") as dispatch, self.assertRaises(ValueError):
            d.main(["--execute", "--out", "/tmp/forbidden"])
        validation.assert_not_called()
        dispatch.assert_not_called()

    def test_cli_failed_validation_creates_no_output_or_dispatch(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        with patch.object(d, "output_path", return_value=path), \
                patch.object(d, "validate", side_effect=ValueError("invalid lineage")), \
                patch.object(Path, "mkdir") as mkdir, patch.object(d, "diagnose") as dispatch, \
                self.assertRaisesRegex(ValueError, "invalid lineage"):
            d.main(["--execute", "--out", str(path)])
        mkdir.assert_not_called()
        dispatch.assert_not_called()
