"""Focused read-only census checks; no ancestor suites or scientific execution."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l21_from_l20_s18_failure_census_v1 as d


EVIDENCE = None


class FailureCensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def pair(self):
        return (deepcopy(self.evidence["rows"][0]),
                deepcopy(self.evidence["gates"][d.CONTROLS[0]]))

    def test_live_count_and_boundaries(self):
        summary = self.evidence["summary"]
        self.assertEqual(summary["control_count"], 9)
        self.assertEqual(summary["S0_S17_PASS_reports"], 162)
        self.assertEqual(summary["S18_FAIL_reports"], 9)
        self.assertEqual(summary["artifact_count"], 37)
        self.assertEqual(summary["failure_coordinate_counts"], {"62": 9})
        for key, value in d.FLAGS.items():
            self.assertEqual(summary[key], value)
        self.assertEqual(summary["retained_L20_failing_controls"],
                         ["actual", "frozen_o", "frozen_down", "frozen_o_down"])

    def test_exact_failure_census_against_independent_constants(self):
        for control in self.evidence["summary"]["controls"]:
            with self.subTest(control=control["control"]):
                failure, = control["failures"]
                self.assertEqual(control["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
                self.assertEqual(control["failing_coordinates"], [62])
                self.assertEqual(failure["actual_error"], "3915202785233/4398046511104")
                self.assertEqual(failure["q"], "66912088017/4398046511104")
                self.assertEqual(failure["excess_error"], "7/8")
                self.assertEqual(failure["excess_budget"], "1/8")
                self.assertEqual(failure["excess_over_budget"], "3/4")
                self.assertEqual(failure["threshold_margin"], "-3/4")
                self.assertEqual(failure["actual_fp16_bits"], "55ed")
                self.assertEqual(failure["nearest_fp16_bits"], "55fb")

    def test_nearest_margin_independent_stored_error_oracle(self):
        for control in self.evidence["summary"]["controls"]:
            metrics = self.evidence["gates"][control["control"]][18]["binary64_v1"]["rows"]
            margins = [Fraction(1, 8) - (Fraction(m["actual_error"]) - Fraction(m["q"]))
                       for m in metrics]
            nearest = min(map(abs, margins))
            self.assertEqual(control["nearest_threshold"]["absolute_margin"], str(nearest))
            self.assertEqual(control["nearest_threshold"]["coordinates"], [
                {"index": i, "threshold_margin": str(m), "accepted": m >= 0}
                for i, m in enumerate(margins) if abs(m) == nearest])

    def test_contract_is_exact_and_command_is_disclosed(self):
        contract = json.loads(d.CONTRACT.read_bytes())
        d.check_contract(contract)
        self.assertEqual(contract["command_sidecar"], d.COMMAND)
        for key, value in (("excess_budget", "1/4"), ("expected_tests", 0),
                           ("interface", ["--execute"]), ("controls", [])):
            changed = deepcopy(contract)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_review_requires_terminal_independent_identity(self):
        for key, value in (("producer_role", "engineer"), ("mission_id", "other"),
                           ("round", 1), ("round", True)):
            review = deepcopy(self.evidence["review"])
            review[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_review(review)
        review = deepcopy(self.evidence["review"])
        review["review"]["status"] = "continue"
        with self.assertRaises(ValueError):
            d.check_review(review)

    def test_hash_size_and_missing_evidence_fail_closed(self):
        for key, value in (("sha256", "0" * 64), ("bytes", 0),
                           ("path", str(d.BASE / "nonexistent_census_input.json"))):
            record = dict(d.ANCHORS["result"], **{key: value})
            with self.subTest(key=key), self.assertRaises((ValueError, FileNotFoundError)):
                d.read_bound(record)

    def test_all_artifacts_required_exactly_once(self):
        records = self.evidence["result"]["artifacts"]
        for changed in (records[:-1], records + records[:1], records[:-1] + records[:1]):
            with self.assertRaises(ValueError):
                d.artifact_manifest(changed)

    def test_control_order_set_and_count(self):
        rows = self.evidence["rows"]
        for changed in (rows[::-1], rows[:-1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                d.summarize(changed, self.evidence["gates"])

    def test_stage_order_and_mandatory_local_gates(self):
        for mode in ("reorder", "missing", "local", "status"):
            row, gates = self.pair()
            if mode == "reorder":
                gates[0], gates[1] = gates[1], gates[0]
            elif mode == "missing":
                gates.pop()
            elif mode == "local":
                gates[0]["local_operator_fp16"]["passed"] = False
            else:
                gates[18]["status"] = "PASS"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                d.summarize_control(row, gates)

    def test_state_kv_lineage_and_history_gates(self):
        for key, value in (("prior_kv", "L20 KV"), ("prior_layer_kv_consumed", True),
                           ("L21_source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("retained_L15_all_gates_pass", False),
                           ("parent_fields", {"h": "stage18"})):
            row, gates = self.pair()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.summarize_control(row, gates)
        row, gates = self.pair()
        gates[17]["residual_state_lineage"] = "FAIL"
        with self.assertRaises(ValueError):
            d.summarize_control(row, gates)

    def test_thresholds_nonfinite_and_error_identity(self):
        for key, value in (("excess_budget", "1/4"), ("actual_error", "nan"),
                           ("q", "-1"), ("excess_error", "0")):
            row, gates = self.pair()
            gates[18]["binary64_v1"]["rows"][62][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.summarize_control(row, gates)

    def test_failure_coordinates_and_acceptance_not_inferred(self):
        for mode in ("accepted", "failures", "indices", "index", "count"):
            row, gates = self.pair()
            gate = gates[18]["binary64_v1"]
            if mode == "accepted":
                gate["rows"][62]["accepted"] = True
            elif mode == "failures":
                gate["failures"] = []
            elif mode == "indices":
                row["L21_S18_failure_indices"] = []
            elif mode == "index":
                gate["rows"][62]["index"] = 63
            else:
                gate["failure_count"] = 0
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                d.summarize_control(row, gates)

    def test_scalar_vector_consistency(self):
        row, gates = self.pair()
        row["index62"]["excess_over_budget"] = "0"
        with self.assertRaises(ValueError):
            d.summarize_control(row, gates)

    def test_result_flags_and_validation_cannot_promote(self):
        for key, value in (("candidate_admitted", True), ("policy_adopted", True),
                           ("successor_published", True), ("native_L0_L20_invocations", 1)):
            result = deepcopy(self.evidence["result"])
            result[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_lineage(result, self.evidence["validation"], self.evidence["parent"])
        validation = deepcopy(self.evidence["validation"])
        validation["skipped"] = 1
        with self.assertRaises(ValueError):
            d.check_lineage(self.evidence["result"], validation, self.evidence["parent"])
        result = deepcopy(self.evidence["result"])
        validation = deepcopy(self.evidence["validation"])
        for document in (result, validation):
            document["retained_L20_failing_controls"][0]["L20_S18_failure_indices"] = []
        with self.assertRaises(ValueError):
            d.check_lineage(result, validation, self.evidence["parent"])

    def test_reference_chain_and_source_bindings_cannot_splice(self):
        result, validation, parent = (deepcopy(self.evidence[key])
                                      for key in ("result", "validation", "parent"))
        parent["L20_original_reference"]["binary64"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.check_lineage(result, validation, parent)
        validation["input_bindings"] = []
        with self.assertRaises(ValueError):
            d.check_lineage(result, validation, self.evidence["parent"])

    def test_metrics_do_not_mutate_retained_inputs(self):
        row, gates = self.pair()
        before = deepcopy((row, gates))
        d.summarize_control(row, gates)
        self.assertEqual((row, gates), before)

    def test_no_scientific_execution_imports_or_write_calls(self):
        tree = ast.parse(d.SOURCE.read_bytes())
        allowed = {"argparse", "fractions", "hashlib", "importlib.util",
                   "json", "pathlib", "sys", "unittest"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name in allowed for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {
                    "write_bytes", "write_text", "mkdir", "unlink", "rename",
                    "execute_layer", "stages", "execute_layers", "save", "savez",
                })

    def test_cli_has_no_execute_output_or_replay_surface(self):
        for args in ([], ["--execute"], ["--check", "--out", "build/forbidden"],
                     ["--check", "--replay"]):
            with self.subTest(args=args), patch.object(d, "validate") as validate, \
                    patch("sys.stderr", new_callable=io.StringIO), \
                    self.assertRaises(SystemExit) as caught:
                d.main(args)
            self.assertEqual(caught.exception.code, 2)
            validate.assert_not_called()
