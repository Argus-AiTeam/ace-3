"""Focused interface and mocked-dispatch checks; no native control is executed."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l14_from_l12_v3_coordinate62_suffix_v1 as d


EXPECTED_TESTS = 40


class L14PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(d.CONTRACT.read_text())
        cls.result = json.loads((d.SELECTED / "result.json").read_text())
        cls.rows = json.loads((d.SELECTED / "interventions.json").read_text())
        cls.reports = json.loads((d.SELECTED / "actual_L13_gates.json").read_text())

    def test_contract_matches(self):
        d.check_contract(self.contract)

    def test_contract_forbids_dispatch_and_admission(self):
        for key in (*d.FLAGS, "native_control_dispatch_authorized", "accepted_prefix_replay",
                    "scientific_result_claim"):
            value = self.contract[key]
            changed = {**self.contract, key: not value if type(value) is bool else "changed"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_contract_preserves_thresholds(self):
        changed = deepcopy(self.contract)
        changed["thresholds"]["S18"] = "relaxed"
        with self.assertRaises(ValueError):
            d.check_contract(changed)

    def test_contract_pins_selected_evidence_and_source(self):
        for key in ("selected_result", "selected_result_sha256", "inherits_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract({**self.contract, key: "substituted"})

    def test_contract_rejects_boolean_layer_and_extended_suffix(self):
        for node in ({"layer": True, "position": 0, "stages": list(range(19))},
                     {"layer": 15, "position": 0, "stages": list(range(19))},
                     {"layer": 14, "position": 1, "stages": list(range(19))}):
            with self.subTest(node=node), self.assertRaises(ValueError):
                d.check_contract({**self.contract, "consumer": node})

    def test_contract_rejects_undisclosed_execution_command(self):
        with self.assertRaises(ValueError):
            d.check_contract({**self.contract, "execution_command": "--execute"})

    def test_result_preserves_historical_counts_and_failure(self):
        d.check_result(self.result)
        self.assertEqual(self.result["native_layer_invocations"], 15)
        self.assertEqual(self.result["retained_status"], "FAIL")

    def test_result_rejects_admission_or_causal_claim(self):
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "unique_upstream_producer_attributed"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: True})

    def test_result_rejects_reference_and_lineage_changes(self):
        for key, value in (("original_global_reference_unchanged", False),
                           ("source_operand_state_KV_lineage_checks", "FAIL"),
                           ("unchanged_baseline_gates", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_result({**self.result, key: value})

    def test_result_rejects_missing_passing_control(self):
        with self.assertRaises(ValueError):
            d.check_result({**self.result, "all_L13_gate_passing_controls": list(d.PASSING[:-1])})

    def test_all_thirteen_controls_retained(self):
        d.check_rows(self.rows)
        self.assertEqual(len(self.rows), 13)
        self.assertEqual(sum(row["all_L13_gates_pass"] for row in self.rows), 9)

    def test_rows_reject_missing_duplicate_and_reordered_controls(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_rows_preserve_failed_baseline(self):
        rows = deepcopy(self.rows)
        rows[0]["mandatory_statuses"][-1] = "PASS"
        with self.assertRaises(ValueError):
            d.check_rows(rows)

    def test_rows_reject_changed_failure_coordinate_or_admission(self):
        for key, value in (("L13_S18_failure_indices", [63]), ("candidate_admitted", True),
                           ("L13_source_operand_state_KV_RTZ_checks", "FAIL")):
            rows = deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_rows(rows)

    def test_reports_match_frozen_rows(self):
        d.check_reports(self.reports, self.rows[0])

    def test_reports_reject_incomplete_or_reordered_stages(self):
        for reports in (self.reports[:-1], list(reversed(self.reports))):
            with self.subTest(count=len(reports)), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])

    def test_reports_reject_policy_and_state_kv_changes(self):
        for key in ("policy_id", "residual_state_lineage", "kv_lineage"):
            reports = deepcopy(self.reports)
            reports[0][key] = "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_reports(reports, self.rows[0])

    def test_reports_reject_changed_global_failure_set(self):
        reports = deepcopy(self.reports)
        reports[18]["binary64_v1"]["failures"] = []
        with self.assertRaises(ValueError):
            d.check_reports(reports, self.rows[0])

    def test_consumer_parent_is_exact_independent_paired_copy(self):
        with d.np.load(d.SELECTED / "inherited_native_L13.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("output_i", "output_z", "stage18")}
        parent = d.consumer_parent(arrays)
        for key, source in (("i", "output_i"), ("z", "output_z"), ("h", "stage18")):
            d.np.testing.assert_array_equal(parent[key], arrays[source])
            self.assertFalse(d.np.shares_memory(parent[key], arrays[source]))
        self.assertEqual(set(parent), {"i", "z", "h"})

    def test_consumer_rejects_fp16_only_or_spliced_state(self):
        with d.np.load(d.SELECTED / "actual_L13.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ("output_i", "output_z", "stage18")}
        with self.assertRaises(KeyError):
            d.consumer_parent({"stage18": arrays["stage18"]})
        arrays["output_i"][62] = 0
        with self.assertRaises(ValueError):
            d.consumer_parent(arrays)

    def test_consumer_never_inherits_previous_layer_kv(self):
        with d.np.load(d.SELECTED / "actual_L13.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        parent = d.consumer_parent(arrays)
        self.assertEqual(arrays["output_cache_k"].shape, (1, 128))
        self.assertNotIn("k", parent)
        self.assertNotIn("v", parent)

    def test_stale_producer_source_fails_closed(self):
        original = d.record

        def changed(path):
            item = original(path)
            if path == d.ROOT / next(iter(d.SOURCE_PINS)):
                item = {**item, "sha256": "0" * 64}
            return item

        with patch.object(d, "record", side_effect=changed), self.assertRaises(ValueError):
            d.history_records()

    def test_wrong_selected_result_fails_before_parent_authentication(self):
        with patch.object(d, "record", return_value={"sha256": "0" * 64}), \
                patch.object(d.producer, "authenticate") as inherited, \
                self.assertRaises(ValueError):
            d.authenticate({})
        inherited.assert_not_called()

    def test_cli_rejects_incomplete_or_mixed_execution_arguments(self):
        for argv in ([], ["--execute"], ["--check", "--out", "build/forbidden"],
                     ["--check", "--execute"], ["--out", "build/forbidden"]):
            with self.subTest(argv=argv), patch("sys.stderr", new_callable=io.StringIO), \
                    patch.object(d, "validate") as validation, self.assertRaises(SystemExit):
                d.main(argv)
            validation.assert_not_called()

    def test_execution_bounds_are_exact(self):
        bounds = self.contract["execution_bounds"]
        self.assertEqual(bounds, d.execution_bounds())
        self.assertEqual(bounds["native_layer_invocations"], 13)
        self.assertEqual(bounds["native_L14_P0_invocations"], 13)
        self.assertEqual(bounds["layers"], [14])
        self.assertEqual([row["control"] for row in bounds["schedule"]], list(d.CONTROLS))
        self.assertEqual(bounds["native_L12_invocations"], 0)
        self.assertEqual(bounds["native_L13_invocations"], 0)

    def test_execution_bounds_reject_expansion(self):
        for key, value in (("native_layer_invocations", 14), ("layers", [13, 14]),
                           ("native_L12_invocations", 1), ("native_L0_L8_invocations", 1),
                           ("rtl_invocations", 1), ("reference_recomputation", True),
                           ("schedule", self.contract["execution_bounds"]["schedule"][:-1])):
            changed = deepcopy(self.contract)
            changed["execution_bounds"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def reference_fixture(self):
        layers = {}
        previous64, previous16 = "original64", "original16"
        for layer in range(9, 15):
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

    def test_reference_extends_only_original_input_chain(self):
        extension = self.reference_fixture()
        self.assertIs(d.check_reference(extension), extension["layers"]["14"])

    def test_reference_rejects_reanchoring_and_kv(self):
        for layer, key in (("14", "input_binary64"), ("14", "input_fp16"),
                           ("14", "prior_kv"), ("12", "input_binary64")):
            extension = self.reference_fixture()
            extension["layers"][layer][key] = "actual-control"
            with self.subTest(layer=layer, key=key), self.assertRaises(ValueError):
                d.check_reference(extension)

    def test_output_accepts_only_fresh_versioned_build_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=False):
            self.assertEqual(d.output_path(path), path)

    def test_output_rejects_existing_directory(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=True), self.assertRaises(ValueError):
            d.output_path(path)

    def test_output_rejects_wrong_scope(self):
        for path in (d.ROOT / d.OUTPUT_PREFIX, d.SELECTED,
                     d.ROOT / "build" / d.OUTPUT_PREFIX,
                     d.ROOT / "build" / (d.OUTPUT_PREFIX + "child") / ".."):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)

    def test_output_rejects_symlink(self):
        path = d.ROOT / "build" / (d.OUTPUT_PREFIX + "test-only")
        with patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            d.output_path(path)

    def test_cli_declares_execute_with_required_output(self):
        args = d.parse_args(["--execute", "--out", "build/" + d.OUTPUT_PREFIX + "fresh"])
        self.assertTrue(args.execute)
        self.assertFalse(args.check)
        self.assertEqual(str(args.out), "build/" + d.OUTPUT_PREFIX + "fresh")
        self.assertTrue(self.contract["execution_command"].endswith(
            "--execute --out " + str(args.out)))

    def test_cli_check_does_not_dispatch(self):
        with patch.object(d, "validate", return_value={"L14_status": "NOT_EXECUTED"}) as check, \
                patch.object(d, "diagnose") as execute, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(d.main(["--check"]), 0)
        check.assert_called_once_with()
        execute.assert_not_called()

    def dispatch_fixture(self):
        with d.np.load(d.SELECTED / "actual_L13.npz", allow_pickle=False) as saved:
            parent = d.consumer_parent(saved)
        parents = {label: {key: value.copy() for key, value in parent.items()}
                   for label in d.CONTROLS}
        return parents, d.NativeControls(parents, {}, {}, d.np.zeros(896))

    def test_dispatch_runs_each_frozen_control_once(self):
        parents, controls = self.dispatch_fixture()
        with patch.object(d, "execute_layer", return_value=({}, {}, [])) as execute:
            for label in d.CONTROLS:
                controls.run(label, parents[label])
            controls.finish()
        self.assertEqual(controls.count, 13)
        self.assertEqual(execute.call_count, 13)
        for call, parent in zip(execute.call_args_list, parents.values(), strict=True):
            self.assertIs(call.args[1], parent)
            self.assertIs(call.args[3], controls.reference)

    def test_dispatch_rejects_reordered_duplicate_or_exhausted_control(self):
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

    def test_dispatch_finish_rejects_partial_schedule(self):
        _, controls = self.dispatch_fixture()
        with self.assertRaises(ValueError):
            controls.finish()

    @contextmanager
    def layer_fixture(self, stages=range(19), rtz_ok=True):
        prior = d.producer.prior
        parents, _ = self.dispatch_fixture()
        reference = d.np.ones(896, dtype="<f8")
        trajectory = {f"stage{stage:02d}": d.np.zeros(1, dtype="<u2") for stage in range(19)}

        def produce(tensors, layer, parent, arrays):
            self.assertEqual(layer, 14)
            arrays.update({key: d.np.zeros(1) for keys in prior.local.OPERANDS.values()
                           for key in keys})
            arrays.update(trajectory)
            arrays["s16_unrounded_binary64"] = d.np.zeros(1)
            yield from stages

        with ExitStack() as stack:
            stack.enter_context(patch.object(d.producer.native, "stages", side_effect=produce))
            stack.enter_context(patch.object(prior.retained, "check_stage_state",
                                            return_value=d.np.zeros(1)))
            stack.enter_context(patch.object(prior.local, "local_reference",
                                            return_value=d.np.zeros(1)))
            stack.enter_context(patch.object(prior, "rtz_reference",
                                            return_value=d.np.array([0 if rtz_ok else 1])))
            evaluate = stack.enter_context(patch.object(
                prior.gates, "evaluate_decoder_stage",
                side_effect=lambda **kwargs: {"status": "FAIL" if kwargs["stage"] == 18 else "PASS"}))
            yield parents["actual"], trajectory, reference, evaluate

    def test_layer_evaluates_all_stages_with_original_reference(self):
        with self.layer_fixture() as (parent, trajectory, reference, evaluate):
            _, locals_, reports = d.execute_layer({}, parent, trajectory, reference)
        self.assertEqual(len(reports), 19)
        self.assertEqual(len(locals_), 18)
        self.assertEqual(reports[-1]["status"], "FAIL")
        self.assertEqual([row["node"] for row in reports], [[14, 0, i] for i in range(19)])
        for stage, call in enumerate(evaluate.call_args_list):
            self.assertIs(call.kwargs["reference"], trajectory[f"stage{stage:02d}"])
            self.assertIs(call.kwargs["reference_binary64"], reference if stage == 18 else None)
            self.assertEqual(call.kwargs["policy"], d.producer.prior.gates.POLICY_ID)

    def test_layer_rejects_incomplete_stages_and_rtz_mismatch(self):
        for stages, rtz_ok in ((range(18), True), ([1], True), (range(19), False)):
            with self.subTest(stages=stages, rtz_ok=rtz_ok), \
                    self.layer_fixture(stages, rtz_ok) as (parent, trajectory, reference, _), \
                    self.assertRaises(ValueError):
                d.execute_layer({}, parent, trajectory, reference)


if __name__ == "__main__":
    unittest.main()
