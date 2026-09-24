"""Frozen L19 lineage and L20 surface tests; native dispatch is always mocked."""

from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l20_from_l19_coordinate62_suffix_v1 as d


EXPECTED_TESTS = 34
STAGE_ROUTE = d.execute_layer


class L20SuffixPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.result = json.loads((d.SELECTED / "result.json").read_text())
        cls.rows = json.loads((d.SELECTED / "interventions.json").read_text())
        cls.reports = json.loads((d.SELECTED / "actual_L19_gates.json").read_text())
        validation = json.loads(Path(cls.result["validation"]["path"]).read_text())
        freeze = json.loads(Path(validation["retained_freeze"]["path"]).read_text())
        cls.extension = json.loads(Path(freeze["reference_extension"]["path"]).read_text())

    def test_contract_matches(self):
        d.check_contract(self.contract)

    def test_thresholds_and_policy_unchanged(self):
        for key in self.contract["thresholds"]:
            changed = deepcopy(self.contract)
            changed["thresholds"][key] = "relaxed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)
        with self.assertRaises(ValueError):
            d.check_contract({**self.contract, "policy_id": "changed"})

    def test_contract_rejects_flags_sources_and_classification_changes(self):
        for key in (*d.FLAGS, "selected_result", "selected_result_sha256", "inherits_sha256",
                    "parent_review", "parent_review_sha256", "native_control_dispatch_authorized",
                    "execution_surface_declared", "accepted_prefix_replay", "scientific_result_claim",
                    "retained_L19_gate_passing_controls", "retained_L18_gate_passing_controls",
                    "retained_L18_S18_coordinate62_failing_controls"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "changed"})

    def test_bounds_reject_expansion_replay_and_output_relaxation(self):
        for layer, position in ((True, 0), (19, 0), (21, 0), (20, 1)):
            with self.subTest(layer=layer, position=position), self.assertRaises(ValueError):
                d.check_contract({**self.contract, "consumer": {
                    "layer": layer, "position": position, "stages": list(range(19))}})
        for section in ("planned_bounds", "execution_bounds"):
            for key, value in (("layers", [19, 20]), ("native_L0_L19_invocations", 1),
                               ("rtl_invocations", 1), ("reference_recomputation", True),
                               ("schedule", self.contract[section]["schedule"][:-1]),
                               ("separate_execution_task_required", False)):
                changed = deepcopy(self.contract)
                changed[section][key] = value
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    d.check_contract(changed)
        for key in ("fresh_directory_required", "ignored_directory_required", "symlinks_forbidden"):
            changed = deepcopy(self.contract)
            changed["execution_bounds"][key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_exact_thirteen_planned_controls_and_zero_check_dispatch(self):
        bounds = self.contract["planned_bounds"]
        self.assertEqual(bounds, d.planned_bounds())
        self.assertEqual(bounds["planned_native_layer_invocations"], 13)
        self.assertEqual(bounds["planned_native_L20_P0_invocations"], 13)
        self.assertEqual(bounds["layers"], [20])
        self.assertEqual(bounds["stages"], list(range(19)))
        self.assertEqual(bounds["schedule"],
                         [{"control": c, "layer": 20, "position": 0} for c in d.CONTROLS])
        self.assertEqual(len(set(d.CONTROLS)), 13)
        for key in ("native_layer_invocations", "native_L0_L19_invocations",
                    "native_L19_P0_invocations", "native_L20_P0_invocations", "rtl_invocations"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_result_rejects_claim_count_and_lineage_changes(self):
        d.check_result(self.result)
        for key, value in (("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True), ("unique_upstream_producer_attributed", True),
                           ("original_global_reference_unchanged", False),
                           ("source_operand_state_KV_lineage_checks", "FAIL"),
                           ("retained_status", "PASS"), ("native_layer_invocations", 12),
                           ("native_L19_P0_invocations", 14),
                           ("all_L19_gate_passing_controls", list(d.HISTORICAL_PASSING)),
                           ("retained_L18_gate_passing_controls", list(d.CONTROLS)),
                           ("retained_L18_S18_coordinate62_failing_controls", []),
                           ("retained_L17_gate_passing_controls", list(d.HISTORICAL_PASSING)),
                           ("retained_L16_gate_passing_controls", list(d.HISTORICAL_PASSING)),
                           ("retained_L15_gate_passing_controls", list(d.CONTROLS))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: value})

    def test_thirteen_l19_passes_do_not_erase_historical_failures(self):
        d.check_rows(self.rows)
        self.assertEqual(self.rows[0]["control"], "actual")
        self.assertEqual([row["control"] for row in self.rows if row["all_L19_gates_pass"]],
                         list(d.CONTROLS))
        self.assertEqual(list(d.HISTORICAL_FAILING),
                         ["actual", "frozen_o", "frozen_down", "frozen_o_down"])
        for row in self.rows:
            self.assertEqual(row["mandatory_statuses"], ["PASS"] * 19)
            self.assertEqual(row["L19_S18_failure_indices"], [])
            self.assertTrue(row["retained_L16_all_gates_pass"])
            self.assertTrue(row["retained_L17_all_gates_pass"])
            for layer in (15, 18):
                self.assertEqual(row[f"retained_L{layer}_S18_failure_indices"],
                                 [62] if row["control"] in d.HISTORICAL_FAILING else [])
        self.assertEqual(sum(row["conditional_acceptance_retained"] for row in self.rows), 9)

    def test_rows_reject_erased_failures_changed_passes_and_state_kv(self):
        for index, key, value in (
                (0, "L19_status", "FAIL"), (0, "L19_S18_failure_indices", [62]),
                (0, "all_L19_gates_pass", False), (0, "conditional_acceptance_retained", True),
                (0, "retained_L15_all_gates_pass", True),
                (0, "retained_L15_S18_failure_indices", []),
                (0, "retained_L18_all_gates_pass", True),
                (0, "retained_L18_S18_failure_indices", []),
                (0, "retained_L16_all_gates_pass", False),
                (0, "retained_L17_S18_failure_indices", [62]),
                (0, "L19_source_operand_state_KV_RTZ_checks", "FAIL"),
                (0, "prior_kv", "L19 KV"), (0, "prior_layer_kv_consumed", True),
                (0, "candidate_admitted", True), (1, "L19_status", "FAIL"),
                (1, "all_L19_gates_pass", False), (1, "L19_S18_failure_indices", [62])):
            rows = deepcopy(self.rows)
            rows[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_reject_missing_duplicate_and_reordered_controls(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_reports_preserve_all_nineteen_passing_gates(self):
        for row in self.rows:
            reports = json.loads((d.SELECTED / (row["control"] + "_L19_gates.json")).read_text())
            d.check_reports(reports, row)
        for reports in (self.reports[:-1], list(reversed(self.reports))):
            with self.subTest(count=len(reports)), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])
        for key in ("policy_id", "residual_state_lineage", "kv_lineage", "status"):
            reports = deepcopy(self.reports)
            reports[0][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])
        reports = deepcopy(self.reports)
        reports[18]["binary64_v1"]["failures"] = [{"index": 62}]
        with self.assertRaises(ValueError):
            d.check_reports(reports, self.rows[0])

    def test_thirteen_complete_independent_izh_parents_exclude_l19_kv(self):
        for label in d.CONTROLS:
            with self.subTest(control=label), \
                    d.np.load(d.SELECTED / (label + "_L19.npz"), allow_pickle=False) as arrays:
                state = d.consumer_parent(arrays)
                self.assertEqual(set(state), {"i", "z", "h"})
                for key, source in (("i", "output_i"), ("z", "output_z"), ("h", "stage18")):
                    d.np.testing.assert_array_equal(state[key], arrays[source])
                    self.assertFalse(d.np.shares_memory(state[key], arrays[source]))
                self.assertEqual(arrays["output_cache_k"].shape, (1, 128))

    def test_parent_rejects_fp16_only_and_spliced_state(self):
        with d.np.load(d.SELECTED / "actual_L19.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("output_i", "output_z", "stage18")}
        with self.assertRaises(KeyError):
            d.consumer_parent({"stage18": arrays["stage18"]})
        arrays["output_i"][62] = 0
        with self.assertRaises(ValueError):
            d.consumer_parent(arrays)

    def test_frozen_source_pins_reject_substitutions(self):
        original = d.repository_record
        for relative in d.SOURCE_PINS:
            def changed(path):
                item = original(path)
                return {**item, "sha256": "0" * 64} if path == d.ROOT / relative else item
            with self.subTest(path=relative), \
                    patch.object(d, "repository_record", side_effect=changed), \
                    self.assertRaises(ValueError):
                d.history_records()

    def test_selected_result_rejected_before_ancestry_authentication(self):
        with patch.object(d, "repository_record", return_value={"sha256": "0" * 64}), \
                patch.object(d.parent, "authenticate") as inherited, self.assertRaises(ValueError):
            d.authenticate({})
        inherited.assert_not_called()

    def test_independent_cac153e063be_round_one_review_required(self):
        review = json.loads(d.PARENT_REVIEW.read_text())
        d.check_parent_review(review)
        for key, value in (("kind", "mission_context"), ("producer_role", "engineer"),
                           ("mission_id", "another-task"), ("round", 2), ("round", True),
                           ("review", {"status": "continue"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_parent_review({**review, key: value})

    def test_reference_follows_original_input_l20_chain(self):
        self.assertIs(d.check_reference(self.extension), self.extension["layers"]["20"])
        for key in ("binary64", "fp16"):
            self.assertEqual(self.extension["layers"]["20"][f"input_{key}"],
                             self.extension["layers"]["19"][key])

    def test_reference_rejects_reanchoring_kv_and_substitution(self):
        for layer, key in (("20", "input_binary64"), ("20", "input_fp16"), ("20", "prior_kv"),
                           ("19", "input_binary64"), ("18", "input_binary64"),
                           ("17", "input_binary64"), ("16", "input_binary64"),
                           ("15", "input_binary64")):
            extension = deepcopy(self.extension)
            extension["layers"][layer][key] = "actual-control"
            with self.subTest(layer=layer, key=key), self.assertRaises(ValueError):
                d.check_reference(extension)
        for key in ("binary64", "fp16"):
            extension = deepcopy(self.extension)
            extension["layers"]["20"][key]["sha256"] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_reference(extension)

    def test_candidate_test_and_input_origins_reject_substitution(self):
        with patch.object(d, "__file__", "/tmp/substituted.py"), self.assertRaises(ValueError):
            d.source_context()
        module = d.importlib.import_module(d.TEST_MODULE)
        with patch.object(module, "__file__", "/tmp/substituted_test.py"), \
                patch.object(d.parent, "source_context") as inherited, self.assertRaises(ValueError):
            d.source_context()
        inherited.assert_not_called()
        with self.assertRaises(ValueError):
            d.repository_record(Path("/tmp/substituted.json"))
        with patch.object(Path, "resolve", return_value=Path("/tmp/substituted.json")), \
                self.assertRaises(ValueError):
            d.repository_record(d.CONTRACT)

    def test_cli_rejects_invalid_modes_output_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--execute"],
                     ["--check", "--out", "build/forbidden"], ["--che"],
                     ["--exe", "--out", "build/forbidden"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), \
                    patch.object(d, "validate") as validation, self.assertRaises(SystemExit):
                d.main(argv)
            validation.assert_not_called()

    def test_cli_check_only_calls_validation(self):
        with patch.object(d, "validate", return_value={"L20_status": "NOT_EXECUTED"}) as check, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(d.main(["--check"]), 0)
        check.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), {"L20_status": "NOT_EXECUTED"})

    def test_execution_interface_preserves_bounds_and_non_admission(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        args = d.parse_args(["--execute", "--out", str(path)])
        self.assertTrue(args.execute)
        self.assertEqual(args.out, path)
        self.assertEqual(d.EXECUTION_COMMAND, d.COMMAND.removesuffix("--check") +
                         f"--execute --out build/{d.OUTPUT_PREFIX}fresh")
        bounds = self.contract["execution_bounds"]
        self.assertEqual(bounds, d.execution_bounds())
        self.assertEqual(bounds["schedule"], d.planned_bounds()["schedule"])
        self.assertEqual(bounds["native_layer_invocations"], 13)
        self.assertEqual(bounds["native_L20_P0_invocations"], 13)
        for key in ("candidate_admitted", "policy_adopted", "successor_published"):
            self.assertIs(bounds[key], False)
        self.assertEqual(bounds["normal_host_review"], "REQUIRED")

    def test_output_accepts_fresh_ignored_versioned_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "focused_path_probe")
        self.assertEqual(d.output_path(path), path)
        self.assertFalse(path.exists())

    def test_output_rejects_existing_paths_and_dangling_links(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "existing")
        for exists, symlink in ((True, False), (False, True)):
            with self.subTest(exists=exists, symlink=symlink), \
                    patch.object(Path, "resolve", return_value=path), \
                    patch.object(Path, "exists", return_value=exists), \
                    patch.object(Path, "is_symlink", return_value=symlink), \
                    patch.object(d.subprocess, "run") as git, self.assertRaises(ValueError):
                d.output_path(path)
            git.assert_not_called()

    def test_output_rejects_symlinks_traversal_wrong_scope_and_prefix(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "link")
        for resolved in (Path("/tmp/outside"), d.ROOT / "build" / (d.OUTPUT_PREFIX + "other")):
            with self.subTest(resolved=resolved), \
                    patch.object(Path, "resolve", return_value=resolved), self.assertRaises(ValueError):
                d.output_path(path)
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
        parents = {}
        for label in d.CONTROLS:
            with d.np.load(d.SELECTED / (label + "_L19.npz"), allow_pickle=False) as arrays:
                parents[label] = d.consumer_parent(arrays)
        return parents, d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_dispatch_runs_thirteen_distinct_frozen_controls_once(self):
        parents, controls = self.controls()
        with patch.object(d, "execute_layer", return_value=({}, {}, [])) as execute:
            for label in d.CONTROLS:
                self.assertEqual(controls.run(label, parents[label]), ({}, {}, []))
            controls.finish()
        self.assertEqual(execute.call_count, 13)
        self.assertEqual(controls.count, 13)
        for call, label in zip(execute.call_args_list, d.CONTROLS, strict=True):
            d.producer.paired.same_arrays(call.args[1], parents[label])

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

    def test_dispatch_rejects_spliced_state_missing_parents_and_partial_schedule(self):
        parents, controls = self.controls()
        parents["actual"]["i"][62] += 1
        with patch.object(d, "execute_layer") as execute, self.assertRaises(ValueError):
            controls.run("actual", parents["actual"])
        execute.assert_not_called()
        self.assertEqual(controls.count, 0)
        with self.assertRaises(ValueError):
            controls.finish()
        del parents["actual"]
        with self.assertRaises(ValueError):
            d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_cli_bad_output_or_validation_creates_no_output_or_dispatch(self):
        with patch.object(d, "validate") as validation, \
                patch.object(d, "diagnose") as dispatch, self.assertRaises(ValueError):
            d.main(["--execute", "--out", "/tmp/forbidden"])
        validation.assert_not_called()
        dispatch.assert_not_called()
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        with patch.object(d, "output_path", return_value=path), \
                patch.object(d, "validate", side_effect=ValueError("invalid lineage")), \
                patch.object(Path, "mkdir") as mkdir, patch.object(d, "diagnose") as dispatch, \
                self.assertRaisesRegex(ValueError, "invalid lineage"):
            d.main(["--execute", "--out", str(path)])
        mkdir.assert_not_called()
        dispatch.assert_not_called()

    def test_cli_second_path_check_precedes_exclusive_creation(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "fresh")
        with patch.object(d, "output_path", side_effect=[path, ValueError("output already exists")]), \
                patch.object(d, "validate", return_value={}), \
                patch.object(Path, "mkdir") as mkdir, patch.object(d, "diagnose") as dispatch, \
                self.assertRaisesRegex(ValueError, "output already exists"):
            d.main(["--execute", "--out", str(path)])
        mkdir.assert_not_called()
        dispatch.assert_not_called()

    def test_repo_bound_context_rejects_wrong_cwd_python_and_environment(self):
        with patch.object(Path, "cwd", return_value=Path("/tmp")), self.assertRaises(ValueError):
            d.source_context()
        with patch.object(d.sys, "executable", "/usr/bin/python3"), self.assertRaises(ValueError):
            d.source_context()
        with patch.dict(d.os.environ, {"PYTHONPATH": "/tmp"}), self.assertRaises(ValueError):
            d.source_context()
        with patch.object(d.sys, "dont_write_bytecode", False), self.assertRaises(ValueError):
            d.source_context()

    def test_mocked_stage_route_uses_only_l20_original_reference_and_all_gates(self):
        parents, _ = self.controls()
        state = parents["actual"]
        with d.np.load(d.SELECTED / "actual_L19.npz", allow_pickle=False) as saved:
            frozen = {key: saved[key] for key in saved.files}
        trajectory = {f"stage{stage:02d}": frozen[f"stage{stage:02d}"] for stage in range(19)}
        reference, tensors = d.np.zeros(896), {}
        sequence = list(range(19))

        def stages(actual_tensors, layer, actual_state, arrays):
            self.assertIs(actual_tensors, tensors)
            self.assertEqual(layer, 20)
            self.assertIs(actual_state, state)
            arrays.update(frozen)
            return iter(sequence)

        prior = d.producer.prior
        with patch.object(d.producer.native, "stages", side_effect=stages) as native, \
                patch.object(prior.retained, "check_stage_state", return_value=frozen["stage00"]), \
                patch.object(prior.local, "local_reference", return_value=frozen["stage00"]) as local, \
                patch.object(prior.gates, "evaluate_decoder_stage",
                             side_effect=lambda **kwargs: {"status": "PASS"}) as gates:
            _, locals_, reports = STAGE_ROUTE(tensors, state, trajectory, reference)
        native.assert_called_once()
        self.assertEqual(len(locals_), 18)
        self.assertEqual(local.call_count, 17)
        self.assertTrue(all(call.args[3] == 20 for call in local.call_args_list))
        self.assertEqual([report["node"] for report in reports], [[20, 0, s] for s in range(19)])
        self.assertEqual(gates.call_count, 19)
        for stage, call in enumerate(gates.call_args_list):
            self.assertEqual(call.kwargs["policy"], prior.gates.POLICY_ID)
            self.assertIs(call.kwargs["reference"], trajectory[f"stage{stage:02d}"])
            self.assertIs(call.kwargs["reference_binary64"], reference if stage == 18 else None)
        for sequence in ([], [1], [False], list(range(18)), list(range(19)) + [19]):
            with self.subTest(sequence=sequence), \
                    patch.object(d.producer.native, "stages", side_effect=stages), \
                    patch.object(prior.retained, "check_stage_state",
                                 return_value=frozen["stage00"]), \
                    patch.object(prior.local, "local_reference", return_value=frozen["stage00"]), \
                    patch.object(prior.gates, "evaluate_decoder_stage",
                                 side_effect=lambda **kwargs: {"status": "PASS"}), \
                    self.assertRaisesRegex(ValueError, "out-of-order|incomplete"):
                STAGE_ROUTE(tensors, state, trajectory, reference)

    def test_review_pin_rejected_before_ancestry_authentication(self):
        with patch.object(d, "record", return_value={"sha256": "0" * 64}), \
                patch.object(d.parent, "authenticate") as inherited, \
                self.assertRaisesRegex(ValueError, "frozen L19 review changed"):
            d.authenticate({})
        inherited.assert_not_called()

    def test_stage_route_retains_failure_and_rejects_missing_gate(self):
        parents, _ = self.controls()
        with d.np.load(d.SELECTED / "actual_L19.npz", allow_pickle=False) as saved:
            frozen = {key: saved[key] for key in saved.files}
        trajectory = {f"stage{s:02d}": frozen[f"stage{s:02d}"] for s in range(19)}
        prior = d.producer.prior

        def stages(tensors, layer, state, arrays):
            arrays.update(frozen)
            return iter(range(19))

        for status in ("FAIL", "BLOCKED"):
            with self.subTest(status=status), \
                    patch.object(d.producer.native, "stages", side_effect=stages), \
                    patch.object(prior.retained, "check_stage_state",
                                 return_value=frozen["stage00"]), \
                    patch.object(prior.local, "local_reference", return_value=frozen["stage00"]), \
                    patch.object(prior.gates, "evaluate_decoder_stage",
                                 side_effect=lambda **kwargs: {"status": status}):
                if status == "FAIL":
                    _, _, reports = STAGE_ROUTE({}, parents["actual"], trajectory, d.np.zeros(896))
                    self.assertEqual([r["status"] for r in reports], ["FAIL"] * 19)
                else:
                    with self.assertRaisesRegex(ValueError, "missing mandatory L20 gate"):
                        STAGE_ROUTE({}, parents["actual"], trajectory, d.np.zeros(896))
