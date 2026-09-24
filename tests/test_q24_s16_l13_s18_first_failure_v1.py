"""Focused synthetic diagnostics; no accepted-prefix replay or RTL execution."""

import ast
import copy
import json
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_s18_first_failure_v1 as d


EXPECTED_TESTS = 14


def reports_fixture():
    reports = [
        {"stage": stage, "node": [13, 0, stage], "policy_id": d.gates.POLICY_ID,
         "residual_state_lineage": "PASS", "kv_lineage": "PASS", "status": "PASS",
         "local_operator_fp16": {"passed": True}} for stage in range(19)]
    failure = {"index": 62, "accepted": False}
    reports[18].update(status="FAIL", binary64_v1={
        "passed": False, "failure_count": 1, "failures": [failure],
        "rows": [{"index": 0, "accepted": True}]},
        fp16={"failures": [{"index": 9}]})
    result = {"first_failure": {"node": [13, 0, 18], "index": 62,
                                "failure_taxonomy": "global_numerical"}}
    return result, reports


def lineage_fixture():
    parent = {"i": np.zeros(896, dtype="<i8"), "z": np.zeros(896, dtype="u1"),
              "h": np.zeros(896, dtype="<u2")}
    arrays = {f"stage{stage:02d}": np.zeros(d.local.SIZES[stage], dtype="<u2")
              for stage in range(19)}
    arrays.update(input_i=parent["i"].copy(), input_z=parent["z"].copy(),
                  input_hidden=parent["h"].copy(), scratch_i=parent["i"].copy(),
                  scratch_z=parent["z"].copy(), output_i=parent["i"].copy(),
                  output_z=parent["z"].copy(), s16_unrounded_binary64=np.zeros(4864, dtype="<f8"))
    for kind in ("k", "v"):
        arrays["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
        arrays["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
    kv = {kind: arrays["output_cache_" + kind].copy() for kind in ("k", "v")}
    return arrays, parent, copy.deepcopy(arrays), kv


class DiagnosticTests(unittest.TestCase):
    def test_json_report_parsing_uses_mandatory_failures(self):
        result, reports = reports_fixture()
        self.assertEqual(d.select_failure(result, json.loads(json.dumps(reports)))["index"], 62)

    def test_missing_failure_list_is_not_rows_fallback(self):
        result, reports = reports_fixture()
        reports[18]["binary64_v1"]["failures"] = []
        with self.assertRaises(ValueError):
            d.select_failure(result, reports)

    def test_wrong_coordinate_or_order_rejected(self):
        for mutation in ("coordinate", "order", "earlier", "policy", "count"):
            with self.subTest(mutation=mutation):
                result, reports = reports_fixture()
                if mutation == "coordinate":
                    result["first_failure"]["index"] = 0
                elif mutation == "order":
                    reports[0]["node"] = [12, 0, 0]
                elif mutation == "earlier":
                    reports[17]["status"] = "FAIL"
                elif mutation == "policy":
                    reports[18]["policy_id"] = "other"
                else:
                    reports[18]["binary64_v1"]["failure_count"] = 2
                with self.assertRaises(ValueError):
                    d.select_failure(result, reports)

    def test_exact_retained_representation_floor(self):
        row = d.measure(0x6630, float.fromhex("0x1.8bd7b2092532cp+10"))
        self.assertEqual(Fraction(row["q"]), Fraction(407084766411, 1099511627776))
        self.assertEqual(Fraction(row["excess_error"]), Fraction(142671047477, 549755813888))
        self.assertFalse(row["accepted"])
        nearest = d.measure(0x662f, float.fromhex("0x1.8bd7b2092532cp+10"))
        self.assertTrue(nearest["accepted"])
        self.assertEqual(Fraction(nearest["excess_error"]), 0)

    def test_floor_ties_signed_zero_and_exact_budget(self):
        self.assertTrue(d.measure(0x3c80, 1.0)["accepted"])
        self.assertFalse(d.measure(0x3c81, 1.0)["accepted"])
        self.assertEqual(Fraction(d.measure(0x6400, 1024.5)["q"]), Fraction(1, 2))
        self.assertEqual(Fraction(d.measure(0x8000, -0.0)["q"]), 0)

    def test_nonfinite_or_out_of_range_reference_rejected(self):
        for value in (float("nan"), float("inf"), 65505.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                d.measure(0, value)

    def test_rtz_independent_normal_subnormal_and_zero(self):
        values = np.array([1.0008, -1.0008, 2**-25, -2**-25, 0.0, -0.0, 65504.0], dtype="<f8")
        self.assertEqual(d.rtz_reference(values).tolist(),
                         [0x3c00, 0xbc00, 0, 0x8000, 0, 0x8000, 0x7bff])

    def test_complete_lineage(self):
        self.assertEqual(d.check_lineage(*lineage_fixture())["S12_S18_exact_rational_IZH"], "PASS")

    def test_operand_state_and_parent_splices_rejected(self):
        for key in ("input_i", "input_z", "input_hidden", "scratch_i", "scratch_z",
                    "stage12", "output_i", "output_z", "stage18", "stage11", "stage17"):
            with self.subTest(key=key):
                arrays, parent, parent_arrays, kv = lineage_fixture()
                arrays[key][0] = 1
                with self.assertRaises(ValueError):
                    d.check_lineage(arrays, parent, parent_arrays, kv)

    def test_kv_and_rtz_mutations_rejected(self):
        for key in ("output_cache_k", "output_cache_v", "stage16"):
            with self.subTest(key=key):
                arrays, parent, parent_arrays, kv = lineage_fixture()
                arrays[key].flat[0] = 1
                with self.assertRaises(ValueError):
                    d.check_lineage(arrays, parent, parent_arrays, kv)
        arrays, parent, parent_arrays, kv = lineage_fixture()
        kv["k"][0, 0] = 1
        with self.assertRaises(ValueError):
            d.check_lineage(arrays, parent, parent_arrays, kv)

    def test_missing_state_and_invalid_dtype_are_explicit(self):
        arrays, parent, parent_arrays, kv = lineage_fixture()
        del arrays["scratch_z"]
        with self.assertRaises(KeyError):
            d.check_lineage(arrays, parent, parent_arrays, kv)
        arrays, parent, parent_arrays, kv = lineage_fixture()
        parent["i"] = parent["i"].astype("<f8")
        with self.assertRaises(ValueError):
            d.check_lineage(arrays, parent, parent_arrays, kv)

    def test_inconclusive_even_when_conditional_control_rescues(self):
        for rescued in (False, True):
            report = d.classify({"accepted": False}, {"accepted": rescued}, {"accepted": rescued})
            self.assertEqual(report["classification"], "inconclusive")
            self.assertEqual(report["S17_local_reference_substitution_rescues"], rescued)
            self.assertEqual(len(report["missing_evidence"]), 2)

    def test_no_rtl_dispatch_policy_adoption_or_successor_writer(self):
        source = Path(d.__file__).read_text()
        tree = ast.parse(source)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & {"system", "Popen", "execute_layers", "_stages",
                                 "continuation_stages", "run_factory", "save", "savez", "savez_compressed"})
        contract = json.loads(d.CONTRACT.read_text())
        self.assertEqual(contract["rtl_invocations"], 0)
        self.assertIs(contract["policy_adopted"], False)
        self.assertIs(contract["successor_published"], False)
        self.assertEqual(contract["normal_host_review"], "REQUIRED")

    def test_unselected_input_is_rejected_before_reading(self):
        with self.assertRaises(ValueError):
            d.diagnose(d.ROOT / "build/not_the_selected_attempt")


if __name__ == "__main__":
    unittest.main()
