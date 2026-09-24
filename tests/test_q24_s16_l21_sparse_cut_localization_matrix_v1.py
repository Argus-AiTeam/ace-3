"""Focused live evidence and mutation checks; no ancestor suites or native imports."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1 as d


EVIDENCE = None


class LocalizationMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def closure_args(self):
        e = self.evidence
        return [deepcopy(e[key]) for key in (
            "closure", "validation", "reviewed", "closure_contract", "reports")] + [
                deepcopy(e["census"]["summary"]["original_reference"])]

    def test_live_exact_controls(self):
        output = self.evidence["summary"]
        self.assertEqual(output["control_count"], 9)
        self.assertEqual([r["control"] for r in output["matrix"]], [
            "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native"])
        self.assertGreaterEqual(output["closure_authenticated_file_count"], 278)

    def test_independent_exact_margin_oracle(self):
        for row in self.evidence["summary"]["matrix"]:
            raw = row["retained_raw_coordinate62"]
            self.assertEqual(raw["actual_error"], "3915202785233/4398046511104")
            self.assertEqual(raw["q"], "66912088017/4398046511104")
            self.assertEqual(raw["excess_error"], "7/8")
            self.assertEqual(raw["threshold_margin"], "-3/4")
            self.assertEqual(raw["status"], "FAIL")
            self.assertEqual(row["adjusted_sparse_cut_L21_status"], "PASS")
            self.assertEqual(row["adjusted_sparse_cut_threshold_margin"], "0")
            self.assertEqual(row["margin_contrast_not_causal_effect"], "3/4")
            self.assertIs(row["per_control_adjusted_execution_performed"], False)

    def test_distinct_closure_raw_failure_is_not_census_raw(self):
        comparator = self.evidence["summary"]["closure_comparator"]
        raw, adjusted = (comparator[k] for k in (
            "raw_L21_coordinate62", "adjusted_L21_coordinate62"))
        self.assertEqual(raw["actual_fp16_bits"], "55ef")
        self.assertEqual(raw["excess_error"], "3/4")
        self.assertEqual(raw["threshold_margin"], "-5/8")
        self.assertEqual(adjusted["actual_fp16_bits"], "55f9")
        self.assertEqual(adjusted["excess_error"], "1/8")
        self.assertEqual(adjusted["threshold_margin"], "0")
        self.assertEqual(comparator["retained_raw_failure_nodes"], [[15, 0, 18], [21, 0, 18]])
        self.assertEqual(comparator["L22_status"], "PASS")
        self.assertEqual(comparator["L23_status"], "PASS")

    def test_forbidden_invocations_and_non_admission_flags(self):
        output = self.evidence["summary"]
        for key, expected in d.FLAGS.items():
            self.assertIs(type(output[key]), type(expected))
            self.assertEqual(output[key], expected)
        for row in output["matrix"]:
            d.check_false_flags(row)
        self.assertTrue(output["historical_failures_preserved"])
        self.assertEqual(output["gate_evidence_scope"], "AUTHENTICATED_RETAINED_ONLY")

    def test_schema_and_contract_bindings(self):
        output = self.evidence["summary"]
        self.assertEqual(output["schema_version"], 1)
        self.assertEqual(output["diagnostic_id"], d.ID)
        self.assertEqual(output["status"], "BOUNDED_LOCALIZATION_MATRIX")
        self.assertEqual(output["evidence"], d.EXPECTED_CONTRACT["evidence"])
        self.assertEqual(output["node"], d.EXPECTED_CONTRACT["node"])
        self.assertEqual(output["claim_boundary"], d.EXPECTED_CONTRACT["claim_boundary"])
        self.assertEqual(output["comparison_scope"], d.EXPECTED_CONTRACT["comparison_scope"])
        self.assertEqual(d.record(d.CONTRACT)["path"], str(d.CONTRACT))

    def test_json_output_round_trip(self):
        summary = self.evidence["summary"]
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)
        with patch.object(d, "validate", return_value=summary), patch("sys.stdout", new_callable=io.StringIO) as out:
            d.main(["--check"])
        self.assertEqual(json.loads(out.getvalue()), summary)

    def test_contract_exactness(self):
        document = json.loads(d.CONTRACT.read_bytes())
        d.check_contract(document)
        for key, value in (("excess_budget", "1/4"), ("controls", []),
                           ("closure_revision", 1), ("interface", ["--execute"]),
                           ("schema_version", 2), ("output", "replace evidence")):
            changed = deepcopy(document)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(changed)
        document["flags"]["candidate_admitted"] = 0
        with self.assertRaises(ValueError):
            d.check_contract(document)

    def test_independent_review_identity(self):
        for key, value in (("producer_role", "engineer"), ("mission_id", "other"),
                           ("round", True), ("round", 2)):
            changed = deepcopy(self.evidence["reviews"]["closure_review"])
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_review(changed, "beb6029db7e6", 1)
        changed["round"] = 1
        changed["review"]["status"] = "continue"
        with self.assertRaises(ValueError):
            d.check_review(changed, "beb6029db7e6", 1)

    def test_artifact_hash_and_size_rejected(self):
        for key, value in (("sha256", "0" * 64), ("bytes", 0)):
            record = dict(d.PINS["census_review"], **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.census.read_bound(record)

    def test_missing_artifact_fails_closed(self):
        record = dict(d.PINS["closure_result"], path=str(d.BASE / "absent-matrix-input.json"))
        with self.assertRaises(FileNotFoundError):
            d.census.read_bound(record)

    def test_control_set_order_and_duplicates_rejected(self):
        controls = self.evidence["census"]["summary"]["controls"]
        closure = self.evidence["summary"]["closure_comparator"]
        for changed in (controls[:-1], controls + controls[:1], list(reversed(controls)),
                        controls[:-1] + controls[:1]):
            with self.assertRaises(ValueError):
                d.build_matrix(changed, closure)

    def test_missing_stage_rejected(self):
        args = self.closure_args()
        args[4][21].pop()
        with self.assertRaises(ValueError):
            d.check_closure(*args)

    def test_adjusted_gate_false_pass_rejected(self):
        report = deepcopy(self.evidence["closure"]["layers"][7]["adjusted_boundary_gate"])
        report["binary64_v1"]["passed"] = False
        with self.assertRaises(ValueError):
            d.check_gate(report, 21, [])

    def test_coordinate_set_order_rejected(self):
        report = deepcopy(self.evidence["reports"][21][18])
        report["binary64_v1"]["rows"][61]["index"] = 62
        with self.assertRaises(ValueError):
            d.check_gate(report, 21, [62])

    def test_exact_threshold_equality_and_no_relaxation(self):
        metric = deepcopy(self.evidence["closure"]["layers"][7][
            "adjusted_boundary_gate"]["binary64_v1"]["rows"][62])
        self.assertEqual(d.coordinate(metric)["status"], "PASS")
        for budget in ("1/4", "0.12500000000000001"):
            metric["excess_budget"] = budget
            with self.assertRaises(ValueError):
                d.coordinate(metric)

    def test_arithmetic_or_encoded_operand_tampering_rejected(self):
        original = self.evidence["reports"][21][18]["binary64_v1"]["rows"][62]
        for key, value in (("actual_error", "0"), ("q", "0"), ("excess_error", "0"),
                           ("actual_fp16_bits", "55f9"), ("actual_fp16_bits", "7c00"),
                           ("accepted", True)):
            metric = dict(original, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                d.coordinate(metric)

    def test_original_reference_reanchoring_rejected(self):
        for key, value in (("sha256", "0" * 64), ("path", "/tmp/reanchored.npy")):
            args = self.closure_args()
            args[0]["layers"][7]["original_reference"]["binary64"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_closure(*args)
        summary = self.evidence["summary"]
        left = summary["closure_comparator"]["original_reference"]["binary64"]
        right = summary["original_reference"]["binary64"]
        self.assertNotEqual(left["path"], right["path"])
        self.assertEqual(left["sha256"], right["sha256"])

    def test_suffix_state_and_kv_lineage_rejected(self):
        for key, value in (("prior_layer_kv_consumed", True), ("prior_kv", "shared"),
                           ("input_state_evidence", {})):
            args = self.closure_args()
            args[0]["layers"][7][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_closure(*args)

    def test_historical_schedule_flags_and_failures_preserved(self):
        for key, value in (("candidate_admitted", True), ("policy_adopted", 0),
                           ("successor_published", True), ("native_L0_L13_invocations", 1),
                           ("native_layer_invocations", 11)):
            args = self.closure_args()
            args[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_closure(*args)
        args = self.closure_args()
        args[0]["layers"][7]["raw_first_failure"] = None
        with self.assertRaises(ValueError):
            d.check_closure(*args)

    def test_read_only_import_and_call_surface(self):
        tree = ast.parse(d.SOURCE.read_bytes())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {
                    "execute_suffix", "drive_layer", "run_layer", "save", "write_bytes",
                    "write_text", "mkdir", "unlink", "system", "Popen"})
        self.assertEqual(imports, {
            "argparse", "fractions", "hashlib", "importlib.util", "json", "math",
            "pathlib", "struct", "sys", "unittest", "ace3.model.candidates"})
        self.assertNotIn("validate", [
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "census"])

    def test_execute_and_output_paths_rejected_before_validation(self):
        for argv in (["--execute"], ["--check", "--execute"],
                     ["--check", "--out", str(d.BASE)]):
            with patch.object(d, "validate") as validate, patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as failure:
                    d.main(argv)
                self.assertEqual(failure.exception.code, 2)
                validate.assert_not_called()

    def test_margin_contrast_is_not_paired_execution(self):
        summary = self.evidence["summary"]
        ids = {row["closure_comparator_id"] for row in summary["matrix"]}
        self.assertEqual(ids, {summary["closure_comparator"]["comparator_id"]})
        self.assertIn("not a paired intervention", summary["comparison_scope"])
        for row in summary["matrix"]:
            expected = Fraction(row["adjusted_sparse_cut_threshold_margin"]) - Fraction(
                row["retained_raw_coordinate62"]["threshold_margin"])
            self.assertEqual(Fraction(row["margin_contrast_not_causal_effect"]), expected)


if __name__ == "__main__":
    unittest.main()
