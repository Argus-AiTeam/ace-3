"""Frozen-parent authentication and mocked L15-only execution surface tests."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l15_from_l14_coordinate62_suffix_v1 as d


EXPECTED_TESTS = 40


class L15SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.result = json.loads((d.SELECTED / "result.json").read_text())
        cls.rows = json.loads((d.SELECTED / "interventions.json").read_text())
        cls.reports = json.loads((d.SELECTED / "actual_L14_gates.json").read_text())

    def test_contract_matches(self):
        d.check_contract(self.contract)

    def test_thresholds_unchanged(self):
        changed = deepcopy(self.contract)
        changed["thresholds"]["S18"] = "relaxed"
        with self.assertRaises(ValueError):
            d.check_contract(changed)

    def test_non_admission_and_dispatch_flags(self):
        for key in (*d.FLAGS, "native_control_dispatch_authorized",
                    "accepted_prefix_replay", "scientific_result_claim"):
            value = self.contract[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: not value if type(value) is bool else "changed"})

    def test_selected_parent_and_review_pins(self):
        for key in ("selected_result", "selected_result_sha256", "inherits_sha256", "parent_review"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "substituted"})
        review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
                  "mission_id": "c55e934b5d70", "review": {"status": "done"}}
        d.check_parent_review(review)
        for key, value in (("producer_role", "engineer"), ("mission_id", "another-task"),
                           ("review", {"status": "blocked"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_parent_review({**review, key: value})

    def test_consumer_rejects_wrong_layer_position_and_boolean(self):
        for layer, position in ((True, 0), (14, 0), (16, 0), (15, 1)):
            with self.subTest(layer=layer, position=position), self.assertRaises(ValueError):
                d.check_contract({**self.contract, "consumer": {
                    "layer": layer, "position": position, "stages": list(range(19))}})

    def test_command_context_must_be_disclosed(self):
        for key in ("command", "execution_command", "cwd"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "bare-python"})

    def test_exact_invocation_bounds(self):
        bounds = self.contract["execution_bounds"]
        self.assertEqual(bounds, d.execution_bounds())
        self.assertEqual(bounds["native_layer_invocations"], 13)
        self.assertEqual(bounds["native_L15_P0_invocations"], 13)
        self.assertEqual(bounds["layers"], [15])
        self.assertEqual(bounds["stages"], list(range(19)))
        self.assertEqual([row["control"] for row in bounds["schedule"]], list(d.CONTROLS))
        for key in ("native_L12_invocations", "native_L13_invocations", "native_L14_invocations",
                    "native_L0_L8_invocations", "rtl_invocations"):
            self.assertEqual(bounds[key], 0)

    def test_bounds_reject_replay_and_expansion(self):
        for key, value in (("native_layer_invocations", 14), ("layers", [14, 15]),
                           ("native_L14_invocations", 1), ("native_L0_L8_invocations", 1),
                           ("rtl_invocations", 1), ("reference_recomputation", True),
                           ("schedule", self.contract["execution_bounds"]["schedule"][:-1])):
            changed = deepcopy(self.contract)
            changed["execution_bounds"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_historical_l14_result_preserved(self):
        d.check_result(self.result)
        self.assertEqual(self.result["retained_status"], "FAIL")
        self.assertEqual(self.result["native_layer_invocations"], 13)

    def test_result_rejects_admission_and_causality(self):
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "unique_upstream_producer_attributed"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: True})

    def test_result_rejects_reference_lineage_and_baseline_changes(self):
        for key, value in (("original_global_reference_unchanged", False),
                           ("source_operand_state_KV_lineage_checks", "FAIL"),
                           ("retained_status", "PASS"), ("accepted_prefix_replay", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: value})

    def test_result_rejects_missing_passing_control(self):
        with self.assertRaises(ValueError):
            d.check_result({**self.result, "all_L14_gate_passing_controls": list(d.CONTROLS[:-1])})

    def test_all_frozen_l14_controls_retained(self):
        d.check_rows(self.rows)
        self.assertEqual(len(self.rows), 13)
        self.assertTrue(all(row["all_L14_gates_pass"] for row in self.rows))
        self.assertFalse(self.rows[0]["retained_L13_all_gates_pass"])

    def test_rows_reject_missing_duplicate_reordered_controls(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_reject_changed_gate_and_failure_set(self):
        for key, value in (("mandatory_statuses", ["PASS"] * 18 + ["FAIL"]),
                           ("L14_status", "FAIL"), ("L14_S18_failure_indices", [62])):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_reject_kv_lineage_or_admission_changes(self):
        for key, value in (("prior_layer_kv_consumed", True), ("candidate_admitted", True),
                           ("L14_source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("prior_kv", "L14 KV")):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_reports_match_frozen_rows(self):
        d.check_reports(self.reports, self.rows[0])

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

    def test_reports_reject_changed_global_failure_set(self):
        reports = deepcopy(self.reports)
        reports[18]["binary64_v1"]["failures"] = [{"index": 62}]
        with self.assertRaises(ValueError):
            d.check_reports(reports, self.rows[0])

    def test_immediate_parent_is_exact_independent_paired_copy(self):
        with d.np.load(d.SELECTED / "actual_L14.npz", allow_pickle=False) as arrays:
            state = d.consumer_parent(arrays)
            for key, source in (("i", "output_i"), ("z", "output_z"), ("h", "stage18")):
                d.np.testing.assert_array_equal(state[key], arrays[source])
                self.assertFalse(d.np.shares_memory(state[key], arrays[source]))
        self.assertEqual(set(state), {"i", "z", "h"})

    def test_immediate_parent_rejects_fp16_only_and_spliced_state(self):
        with d.np.load(d.SELECTED / "actual_L14.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("output_i", "output_z", "stage18")}
        with self.assertRaises(KeyError):
            d.consumer_parent({"stage18": arrays["stage18"]})
        arrays["output_i"][62] = 0
        with self.assertRaises(ValueError):
            d.consumer_parent(arrays)

    def test_parent_does_not_carry_l14_kv(self):
        with d.np.load(d.SELECTED / "actual_L14.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["output_cache_k"].shape, (1, 128))
            state = d.consumer_parent(arrays)
        self.assertNotIn("k", state)
        self.assertNotIn("v", state)

    def test_stale_l14_source_fails_closed(self):
        original = d.record

        def changed(path):
            item = original(path)
            return {**item, "sha256": "0" * 64} if path == d.ROOT / next(iter(d.SOURCE_PINS)) else item

        with patch.object(d, "record", side_effect=changed), self.assertRaises(ValueError):
            d.history_records()

    def test_wrong_selected_result_rejected_before_ancestry_authentication(self):
        with patch.object(d, "record", return_value={"sha256": "0" * 64}), \
                patch.object(d.parent, "authenticate") as inherited, self.assertRaises(ValueError):
            d.authenticate({})
        inherited.assert_not_called()

    def reference_fixture(self):
        layers = {}
        previous64, previous16 = "original64", "original16"
        for layer in range(9, 16):
            layers[str(layer)] = {
                "input_binary64": previous64, "input_fp16": previous16,
                "binary64": f"original64_{layer}", "fp16": f"original16_{layer}",
                "prior_kv": "own empty P0",
            }
            previous64, previous16 = f"original64_{layer}", f"original16_{layer}"
        return {"reference_only": True, "policy_id": d.producer.prior.gates.POLICY_ID,
                "global_reference_policy": "legacy-binary64-AWQ-fully-independent-propagation",
                "original_binary64_parent": "original64", "original_fp16_parent": "original16",
                "layers": layers}

    def test_reference_extends_original_input_chain(self):
        extension = self.reference_fixture()
        self.assertIs(d.check_reference(extension), extension["layers"]["15"])

    def test_reference_rejects_reanchoring_and_inherited_kv(self):
        for layer, key in (("15", "input_binary64"), ("15", "input_fp16"),
                           ("15", "prior_kv"), ("14", "input_binary64"),
                           ("12", "input_binary64")):
            extension = self.reference_fixture()
            extension["layers"][layer][key] = "actual-control"
            with self.subTest(layer=layer, key=key), self.assertRaises(ValueError):
                d.check_reference(extension)

    def test_cli_rejects_incomplete_or_mixed_arguments(self):
        for argv in ([], ["--execute"], ["--check", "--out", "build/forbidden"],
                     ["--check", "--execute"], ["--out", "build/forbidden"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), \
                    patch.object(d, "validate") as validation, self.assertRaises(SystemExit):
                d.main(argv)
            validation.assert_not_called()

    def test_cli_discloses_execute_with_fresh_output(self):
        args = d.parse_args(["--execute", "--out", "build/" + d.OUTPUT_PREFIX + "fresh"])
        self.assertTrue(args.execute)
        self.assertFalse(args.check)
        self.assertTrue(self.contract["execution_command"].endswith(
            "--execute --out " + str(args.out)))

    def test_cli_check_never_dispatches(self):
        with patch.object(d, "validate", return_value={"L15_status": "NOT_EXECUTED"}) as check, \
                patch.object(d, "diagnose") as execute, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(d.main(["--check"]), 0)
        check.assert_called_once_with()
        execute.assert_not_called()

    def test_output_accepts_fresh_versioned_build_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_path(path), path)

    def test_output_rejects_existing_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=True), self.assertRaises(ValueError):
            d.output_path(path)

    def test_output_rejects_other_scopes(self):
        for path in (d.ROOT / d.OUTPUT_PREFIX, d.SELECTED, d.ROOT / "build" / d.OUTPUT_PREFIX,
                     d.ROOT / "build" / (d.OUTPUT_PREFIX + "child") / ".."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)

    def test_output_rejects_symlink(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            d.output_path(path)

    def dispatch_fixture(self):
        with d.np.load(d.SELECTED / "actual_L14.npz", allow_pickle=False) as arrays:
            state = d.consumer_parent(arrays)
        parents = {label: {key: value.copy() for key, value in state.items()} for label in d.CONTROLS}
        return parents, d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_dispatch_runs_each_frozen_control_once(self):
        parents, controls = self.dispatch_fixture()
        with patch.object(d, "execute_layer", return_value=({}, {}, [])) as execute:
            for label in d.CONTROLS:
                controls.run(label, parents[label])
            controls.finish()
        self.assertEqual(controls.count, 13)
        self.assertEqual(execute.call_count, 13)
        for call, state in zip(execute.call_args_list, parents.values(), strict=True):
            self.assertIs(call.args[1], state)
            self.assertIs(call.args[3], controls.reference)

    def test_dispatch_rejects_reordered_duplicate_exhausted_controls(self):
        parents, controls = self.dispatch_fixture()
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

    def test_dispatch_rejects_spliced_parent_before_native(self):
        parents, controls = self.dispatch_fixture()
        parents["actual"]["i"][62] += 1
        with patch.object(d, "execute_layer") as execute, self.assertRaises(ValueError):
            controls.run("actual", parents["actual"])
        execute.assert_not_called()
        self.assertEqual(controls.count, 0)

    def test_dispatch_rejects_partial_schedule(self):
        _, controls = self.dispatch_fixture()
        with self.assertRaises(ValueError):
            controls.finish()

    @contextmanager
    def layer_fixture(self, stages=range(19), rtz_ok=True):
        prior = d.producer.prior
        parents, _ = self.dispatch_fixture()
        reference = d.np.ones(896, dtype="<f8")
        trajectory = {f"stage{stage:02d}": d.np.zeros(1, dtype="<u2") for stage in range(19)}

        def produce(tensors, layer, state, arrays):
            self.assertEqual(layer, 15)
            arrays.update({key: d.np.zeros(1) for keys in prior.local.OPERANDS.values() for key in keys})
            arrays.update(trajectory)
            arrays["s16_unrounded_binary64"] = d.np.zeros(1)
            yield from stages

        with ExitStack() as stack:
            stack.enter_context(patch.object(d.producer.native, "stages", side_effect=produce))
            stack.enter_context(patch.object(prior.retained, "check_stage_state",
                                            return_value=d.np.zeros(1)))
            stack.enter_context(patch.object(prior.local, "local_reference", return_value=d.np.zeros(1)))
            stack.enter_context(patch.object(prior, "rtz_reference",
                                            return_value=d.np.array([0 if rtz_ok else 1])))
            evaluate = stack.enter_context(patch.object(
                prior.gates, "evaluate_decoder_stage",
                side_effect=lambda **kwargs: {"status": "FAIL" if kwargs["stage"] == 18 else "PASS"}))
            yield parents["actual"], trajectory, reference, evaluate

    def test_layer_evaluates_all_stages_with_original_reference(self):
        with self.layer_fixture() as (state, trajectory, reference, evaluate):
            _, locals_, reports = d.execute_layer({}, state, trajectory, reference)
        self.assertEqual(len(locals_), 18)
        self.assertEqual(reports[-1]["status"], "FAIL")
        self.assertEqual([row["node"] for row in reports], [[15, 0, stage] for stage in range(19)])
        for stage, call in enumerate(evaluate.call_args_list):
            self.assertIs(call.kwargs["reference"], trajectory[f"stage{stage:02d}"])
            self.assertIs(call.kwargs["reference_binary64"], reference if stage == 18 else None)
            self.assertEqual(call.kwargs["policy"], d.producer.prior.gates.POLICY_ID)

    def test_layer_rejects_incomplete_stages_and_rtz_mismatch(self):
        for stages, rtz_ok in ((range(18), True), ([1], True), ([True], True), (range(19), False)):
            with self.subTest(stages=stages, rtz_ok=rtz_ok), \
                    self.layer_fixture(stages, rtz_ok) as (state, trajectory, reference, _), \
                    self.assertRaises(ValueError):
                d.execute_layer({}, state, trajectory, reference)


if __name__ == "__main__":
    unittest.main()
