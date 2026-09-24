"""Focused scalar, full-vector and fail-closed tests; no ancestor execution."""

import ast
from copy import deepcopy
from fractions import Fraction
import json
import struct
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l21_from_l20_s18_coordinate62_margin_sensitivity_v1 as d


REFERENCE = "0x1.7ecf94456fd10p+6"


def half(word):
    return Fraction.from_float(struct.unpack("<e", struct.pack("<H", word))[0])


def project(integer):
    return struct.unpack("<H", struct.pack("<e", integer / (1 << 24)))[0]


class MarginSensitivityTests(unittest.TestCase):
    def fixture(self):
        words = np.zeros(896, dtype="<u2")
        integers = np.zeros(896, dtype="<i8")
        zeros = np.zeros(896, dtype="u1")
        references = np.zeros(896, dtype="<f8")
        integers[62] = 1590863346
        words[62] = 0x55ED
        references[62] = float.fromhex(REFERENCE)
        rows, _ = d.vector_gate(words, references)
        reports = [{k: v for k, v in row.items() if k != "threshold_margin"} | {"index": i}
                   for i, row in enumerate(rows)]
        return {"output_i": integers, "output_z": zeros, "stage18": words}, references, reports

    def test_contract_exact(self):
        contract = json.loads(d.CONTRACT.read_bytes())
        d.check_contract(contract)
        self.assertEqual(contract["expected_tests"], 24)
        for key, value in (("excess_budget", "1/4"), ("controls", []), ("interface", ["--execute"])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_contract(contract | {key: value})

    def test_review_independence(self):
        review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
                  "mission_id": "ecf478e49bef", "round": 1, "review": {"status": "done"}}
        d.check_review(review)
        for key, value in (("round", True), ("producer_role", "engineer"),
                           ("mission_id", "other"), ("review", {"status": "continue"})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_review(review | {key: value})

    def test_hash_mismatch_fails_closed(self):
        pin = d.PINS["census_source"]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.census.read_bound(pin | {"sha256": "0" * 64})

    def test_cell_odd_tie_ownership(self):
        cell = d.cell(0x55F9)
        self.assertEqual((cell["lower"], cell["upper"]), ("3057/32", "3059/32"))
        self.assertFalse(cell["lower_inclusive"])
        self.assertEqual((cell["lower_tie_owner"], cell["upper_tie_owner"]), ("55f8", "55fa"))
        self.assertEqual(cell["minimum_Q24_integer"], 3057 * (1 << 19) + 1)
        self.assertEqual(project(cell["minimum_Q24_integer"] - 1), 0x55F8)
        self.assertEqual(project(cell["minimum_Q24_integer"]), 0x55F9)

    def test_cell_even_tie_ownership(self):
        cell = d.cell(0x55FA)
        self.assertTrue(cell["lower_inclusive"])
        self.assertTrue(cell["upper_inclusive"])
        self.assertEqual(project(cell["minimum_Q24_integer"]), 0x55FA)
        self.assertEqual(project(cell["maximum_Q24_integer"]), 0x55FA)
        self.assertNotEqual(project(cell["minimum_Q24_integer"] - 1), 0x55FA)
        self.assertNotEqual(project(cell["maximum_Q24_integer"] + 1), 0x55FA)

    def test_cell_subnormal(self):
        for word in (1, 2, 0x3FF):
            cell = d.cell(word)
            self.assertEqual(cell["minimum_Q24_integer"], word)
            self.assertEqual(cell["maximum_Q24_integer"], word)
            self.assertEqual(Fraction(cell["lower"]), (half(word - 1) + half(word)) / 2)

    def test_cell_binade_asymmetry(self):
        cell = d.cell(0x4000)
        self.assertEqual(Fraction(cell["lower"]), (half(0x3FFF) + half(0x4000)) / 2)
        self.assertNotEqual(half(0x4000) - Fraction(cell["lower"]),
                            Fraction(cell["upper"]) - half(0x4000))

    def test_invalid_cell_domain(self):
        for word in (0, -1, True, 0x7BFF, 0x7C00, 0x8000):
            with self.subTest(word=word), self.assertRaises(ValueError):
                d.cell(word)

    def test_minimum_from_below_independent_exhaustive_oracle(self):
        result = d.sensitivity(1590863346, 0, REFERENCE)
        exact = Fraction.from_float(float.fromhex(REFERENCE))
        finite = [(word, half(word)) for word in range(0x7C00)]
        nearest, value = min(finite, key=lambda item: (abs(item[1] - exact), item[0] & 1))
        passing = [word for word, value in finite
                   if abs(value - exact) - abs(half(nearest) - exact) <= Fraction(1, 8)]
        self.assertEqual((min(passing), max(passing)), (0x55F9, 0x55FD))
        cut = result["minimal_cut"]
        self.assertEqual(cut["adjusted_Q24_integer"], 1602748417)
        self.assertEqual(cut["delta_Q24_units"], 11885071)
        self.assertEqual(project(cut["adjusted_Q24_integer"]), min(passing))
        self.assertNotIn(project(cut["one_unit_less_Q24_integer"]), passing)
        self.assertEqual(result["baseline"]["excess_error"], "7/8")
        self.assertEqual(result["baseline"]["threshold_margin"], "-3/4")
        self.assertEqual(cut["gate"]["threshold_margin"], "0")

    def test_minimum_from_above(self):
        integer = int(half(0x5605) * (1 << 24))
        result = d.sensitivity(integer, 0, REFERENCE)
        cut = result["minimal_cut"]
        self.assertLess(cut["delta_Q24_units"], 0)
        self.assertEqual(cut["gate"]["actual_fp16_bits"], "55fd")
        self.assertEqual(cut["one_unit_less_Q24_integer"], cut["adjusted_Q24_integer"] + 1)
        self.assertEqual(project(cut["adjusted_Q24_integer"]), 0x55FD)
        self.assertEqual(project(cut["one_unit_less_Q24_integer"]), 0x55FE)

    def test_scalar_reference_and_floor_exact(self):
        row = d.scalar(0x55ED, REFERENCE)
        self.assertEqual(row["reference_binary64_hex"], REFERENCE)
        self.assertEqual(row["nearest_fp16_bits"], "55fb")
        self.assertEqual(Fraction(row["q"]), abs(half(0x55FB) - Fraction.from_float(float.fromhex(REFERENCE))))
        self.assertEqual(Fraction(row["actual_error"]) - Fraction(row["q"]), Fraction(7, 8))
        self.assertEqual(row["excess_budget"], "1/8")

    def test_invalid_scalar_or_nonfailing_baseline(self):
        for integer, zero, ref in ((1590863346, 1, REFERENCE),
                                    (0, 0, REFERENCE), (1590863346, 0, "nan"),
                                    (int(half(0x55FB) * (1 << 24)), 0, REFERENCE)):
            with self.subTest(integer=integer, ref=ref), self.assertRaises(ValueError):
                d.sensitivity(integer, zero, ref)

    def test_signed_zero_state(self):
        arrays, _, _ = self.fixture()
        arrays["stage18"][0] = 0x8000
        arrays["output_z"][0] = 1
        d.check_state(arrays)
        arrays["output_z"][62] = 1
        with self.assertRaises(ValueError):
            d.check_state(arrays)

    def test_state_shape_dtype_and_projection(self):
        arrays, _, _ = self.fixture()
        for key, value in (("output_i", arrays["output_i"].astype("<i4")),
                           ("output_z", arrays["output_z"][:-1]),
                           ("stage18", np.zeros(896, dtype="<u2"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_state(arrays | {key: value})

    def test_full_vector_only_coordinate62_and_no_mutation(self):
        arrays, references, reports = self.fixture()
        before = {k: v.copy() for k, v in arrays.items()}
        reference_before, reports_before = references.copy(), deepcopy(reports)
        result = d.analyze_control(d.CONTROLS[0], arrays, references, reports)
        vectors = result["counterfactual_vectors"]
        self.assertEqual(vectors["minimum"]["changed_coordinates"], [62])
        self.assertEqual(vectors["minimum"]["failure_indices"], [])
        self.assertEqual(vectors["one_unit_less"]["failure_indices"], [62])
        for key, value in before.items():
            np.testing.assert_array_equal(arrays[key], value)
        np.testing.assert_array_equal(references, reference_before)
        self.assertEqual(reports, reports_before)

    def test_non62_failure_rejected(self):
        arrays, references, reports = self.fixture()
        arrays["stage18"][63] = 0x3C00
        arrays["output_i"][63] = 1 << 24
        with self.assertRaisesRegex(ValueError, "non-62"):
            d.analyze_control(d.CONTROLS[0], arrays, references, reports)

    def test_reference_or_stored_metric_splice_rejected(self):
        arrays, references, reports = self.fixture()
        for key, value in (("reference_binary64_hex", "0x0.0p+0"),
                           ("excess_budget", "1/4"), ("actual_fp16_bits", "55ef")):
            changed = deepcopy(reports)
            changed[62][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "metric/reference"):
                d.analyze_control(d.CONTROLS[0], arrays, references, changed)

    def test_nonfinite_vector_rejected(self):
        arrays, references, _ = self.fixture()
        references[0] = np.nan
        with self.assertRaises(ValueError):
            d.vector_gate(arrays["stage18"], references)

    def test_nine_control_agreement_and_material_disagreement(self):
        analysis = d.sensitivity(1590863346, 0, REFERENCE)
        accounts = [{"control": label, "status": "ANALYZED", "analysis": deepcopy(analysis)}
                    for label in d.CONTROLS]
        accounts[1]["analysis"]["baseline_Q24_integer"] -= 1
        d.check_agreement(accounts)
        with self.assertRaises(ValueError):
            d.check_agreement(accounts[:-1])
        accounts[1]["analysis"]["minimal_cut"]["adjusted_Q24_integer"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            d.check_agreement(accounts)

    def test_missing_authentication_is_unknown_without_arithmetic(self):
        with patch.object(d, "authenticate", side_effect=FileNotFoundError("missing binding")), \
                patch.object(d, "analyze_control") as analyze:
            result = d.diagnose()
        self.assertEqual(result["classification"], "UNKNOWN")
        self.assertEqual(result["analyzed_control_count"], 0)
        self.assertEqual(len(result["controls"]), 9)
        self.assertIn("missing binding", result["error"]["message"])
        analyze.assert_not_called()
        for key, value in d.FLAGS.items():
            self.assertEqual(result[key], value)

    def test_arithmetic_failure_rejects_and_stops(self):
        summary = {"original_reference": {}, "thresholds": {}, "retained_L20_failing_controls": []}
        retained = {"summary": summary, "gates": {d.CONTROLS[0]: [None] * 18 + [
            {"binary64_v1": {"rows": []}}]}}
        with patch.object(d, "authenticate", return_value=(retained, {d.CONTROLS[0]: {}}, None)), \
                patch.object(d.census, "artifact_manifest", return_value={}), \
                patch.object(d, "analyze_control", side_effect=ValueError("bad cell")) as analyze:
            retained["result"] = {"artifacts": []}
            result = d.diagnose()
        self.assertEqual(result["classification"], "REJECTED")
        self.assertEqual(result["error"]["control"], d.CONTROLS[0])
        self.assertEqual(analyze.call_count, 1)

    def test_output_scope_and_existing_evidence_refusal(self):
        for path in ("build/legacy", f"build/{d.NAME}_attempt1/nested",
                     f"ace3/{d.NAME}_attempt1"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)
        with patch.object(d.Path, "exists", return_value=True), self.assertRaisesRegex(ValueError, "overwrite"):
            d.output_path(f"build/{d.NAME}_attempt_reviewer")

    def test_json_write_is_exclusive(self):
        path = unittest.mock.Mock()
        path.open.side_effect = FileExistsError("preserved")
        with self.assertRaises(FileExistsError):
            d.write_json(path, {"status": "DIAGNOSED"})
        path.open.assert_called_once_with("x", encoding="utf-8")

    def test_no_dispatch_or_legacy_imports(self):
        tree = ast.parse(d.SOURCE.read_bytes())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        allowed = {
            "argparse", "fractions", "hashlib", "importlib.util", "io", "json", "math",
            "os", "pathlib", "shlex", "sys", "time", "unittest", "numpy", "ace3.model.candidates",
        }
        for node in imports:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            self.assertTrue(set(names) <= allowed)
            if isinstance(node, ast.ImportFrom) and node.module == "ace3.model.candidates":
                self.assertTrue({a.name for a in node.names} <= {
                    "binary64_fp16_excess_v1", "residual_exact_grid_q24_reference_v1",
                    "diagnose_q24_s16_l21_from_l20_s18_failure_census_v1",
                })
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"Popen", "system", "spawn", "save", "savez", "validate"})


if __name__ == "__main__":
    unittest.main()
