"""Integer-grid and rounding-cell oracles; never regenerate scientific anchors."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import math
import struct
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_mixed_factor_cutoff_classifier_v1 as d


EVIDENCE = None


def half_grid(word):
    exponent, mantissa = (int(word) >> 10) & 31, int(word) & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16 oracle input")
    integer = mantissa if exponent == 0 else (1024 + mantissa) << (exponent - 1)
    return -integer if int(word) & 32768 else integer


def rounding_cell(value):
    previous = Fraction(math.nextafter(value, -math.inf))
    following = Fraction(math.nextafter(value, math.inf))
    exact = Fraction(value)
    even = struct.unpack("<Q", struct.pack("<d", value))[0] % 2 == 0
    return (previous + exact) / 2, (exact + following) / 2, even


class TenAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            result = d.validate_capture(d.OUTPUT)
            cls.e = {"result": result, "words": {
                row["label"]: d.load_words(row["archive"]) for row in result["anchors"]}}
        else:
            cls.e = EVIDENCE
        cls.result = cls.e["result"]

    def assert_in_cell(self, exact, rounded, square=False):
        low, high, even = rounding_cell(rounded)
        if square:
            low, high = low * low, high * high
        self.assertTrue(low <= exact <= high if even else low < exact < high)

    def test_exact_ten_case_census(self):
        self.assertEqual(tuple(r["label"] for r in self.result["anchors"]), d.LABELS)
        self.assertEqual(len(self.e["words"]), 10)
        self.assertEqual(self.result["dispatch"]["exact_square_terms"], 8960)

    def test_independent_integer_mean_oracle(self):
        for row in self.result["anchors"]:
            grid = [half_grid(v) for v in self.e["words"][row["label"]]]
            mean = Fraction(sum(v * v for v in grid), 896 * 2**48)
            self.assertEqual(Fraction(row["mean_square"]), mean)
            self.assertEqual(Fraction(row["radicand"]), mean + Fraction(1, 1000000))

    def test_binary64_conversion_rounding_cells(self):
        for row in self.result["anchors"]:
            self.assert_in_cell(Fraction(row["radicand"]),
                                float.fromhex(row["radicand_binary64_hex"]))

    def test_sqrt_cells_without_another_sqrt(self):
        for row in self.result["anchors"]:
            self.assert_in_cell(Fraction(float.fromhex(row["radicand_binary64_hex"])),
                                float.fromhex(row["root_binary64_hex"]), square=True)

    def test_reciprocal_cells_without_binary64_reciprocal(self):
        for row in self.result["anchors"]:
            root = Fraction(float.fromhex(row["root_binary64_hex"]))
            self.assert_in_cell(1 / root, float.fromhex(row["inverse_norm_anchor_hex"]))

    def test_input_byte_and_shape_bindings(self):
        for row in self.result["anchors"]:
            words = self.e["words"][row["label"]]
            self.assertEqual(words.shape, (896,))
            self.assertEqual(words.dtype.str, "<u2")
            self.assertEqual(row["input"]["bytes"], 1792)
            self.assertEqual(row["input"]["sha256"], hashlib.sha256(words.tobytes()).hexdigest())

    def test_closure_validation_never_regenerates(self):
        with patch.object(d.math, "sqrt", side_effect=AssertionError("scalar replay")):
            for row in self.result["anchors"]:
                d.validate_anchor(row)

    def test_anchor_tamper_rejected(self):
        for field in ("inverse_norm_anchor", "mean_square", "radicand_closure",
                      "inverse_square_identity_defect"):
            row = deepcopy(self.result["anchors"][0])
            row[field] = str(Fraction(row[field]) + 1)
            with self.assertRaises(ValueError):
                d.validate_anchor(row)

    def test_invalid_operands_fail_before_sqrt(self):
        bad = [np.zeros(895, dtype="<u2"), np.zeros(897, dtype="<u2"),
               np.zeros(896, dtype="<f2"), np.full(896, 0x7c00, dtype="<u2"),
               np.full(896, 0x7e00, dtype="<u2")]
        with patch.object(d.math, "sqrt", side_effect=AssertionError("invalid-input replay")):
            for words in bad:
                with self.assertRaises(ValueError):
                    d.scalar_anchor(words)

    def test_native_review_rejects_self_review_or_wrong_artifact(self):
        review = {"kind": "round_reviewed_handoff", "schema_version": 3,
                  "producer_role": "reviewer", "mission_id": d.TASK,
                  "review": {"status": "done", "artifact_bindings": [
                      {"ref": d.RESUME_PIN["path"], "sha256": d.RESUME_PIN["sha256"]}]}}
        d.verify_review(review, d.RESUME_PIN, d.TASK)
        review["producer_role"] = "engineer"
        with self.assertRaises(ValueError):
            d.verify_review(review, d.RESUME_PIN, d.TASK)
        review["producer_role"] = "reviewer"
        review["review"]["artifact_bindings"] = []
        with self.assertRaises(ValueError):
            d.verify_review(review, d.RESUME_PIN, d.TASK)

    def test_framed_capture_keeps_distinct_arbitrary_stream_bytes(self):
        command, stdout, stderr = b'{"argv":[]}\n', b"\x00out\n\xff", b"\xfeerr\x00\n"
        capture = d.framed_capture(command, stdout, stderr, 7)
        self.assertTrue(capture.endswith(command + stdout + stderr))
        self.assertNotEqual(capture, d.framed_capture(command, stderr, stdout, 7))
        self.assertNotEqual(capture, d.framed_capture(command, stdout, stderr, 0))

    def test_operator_budget_and_review_boundary(self):
        dispatch = self.result["dispatch"]
        for key in ("scientific_cases", "binary64_radicand_conversions",
                    "binary64_sqrt_calls", "binary64_reciprocals"):
            self.assertEqual(dispatch[key], 10)
        for key in ("norm_weight_multiplications", "rmsnorm_vector_outputs",
                    "selected_head_calls", "model_operator_calls"):
            self.assertEqual(dispatch[key], 0)
        self.assertEqual(self.result["normal_host_review"],
                         "REQUIRED_BEFORE_CLASSIFIER_CONSUMPTION")
        self.assertNotIn("decision", self.result)
        self.assertEqual(self.result["preregistration"]["later_classifier"]["cases"], 27)
        history = self.result["original_execution_sources"]
        producer = "ace3.model.candidates.q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1"
        self.assertEqual(history[producer]["sha256"],
                         "5a3a3e3700afef82b149c8cff60e69b0bf0c9ba0784ad35501596fa1f44adb2a")
        current = {p["path"]: p["sha256"] for p in self.result["authenticated_pins"]}
        self.assertEqual(current[history[producer]["path"]],
                         "b406b5f3bc89eebadc89156c3a09ad5b7d3a25d3bf1da55f3c81186835e915e1")


def vector_fixture(before=Fraction(1, 4), after=Fraction(-1, 8)):
    return {
        "control": d.CONTROLS[0], "left_id": 34319, "right_id": 319,
        "branch": "fp16", "role": "synthetic",
        "before_exact_margin": str(before), "after_exact_margin": str(after),
        "vector_swap_margin_movement": str(after-before),
        "observed_movement_sign": d.sign(after-before),
        "observed_crossing_pattern": [d.sign(before), d.sign(after)],
        "retained_actual_margin": str(before), "immutable_reference_margin": str(after),
        "parent_account": {"vector_coordinate_sum": str(before-after)},
    }


class MixedFactorClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live = EVIDENCE if isinstance(EVIDENCE, dict) and "classifier_result" in EVIDENCE else None

    def test_factor_vector_formula_closure(self):
        actual = np.zeros(d.WIDTH, dtype="<f2")
        reference = np.zeros(d.WIDTH, dtype="<f2")
        weight = np.ones(d.WIDTH, dtype="<f2")
        actual[0], reference[0], weight[0] = np.float16(3), np.float16(1), np.float16(2)
        vectors = d.factor_vectors(
            actual.view("<u2"), reference.view("<u2"), weight,
            {"inverse_norm_anchor": "5/7"}, {"inverse_norm_anchor": "3/7"})
        self.assertEqual(vectors["direct_hidden"][0], Fraction(12, 7))
        self.assertEqual(vectors["scalar_scale"][0], Fraction(4, 7))
        self.assertEqual(vectors["interaction"][0], Fraction(8, 7))
        self.assertEqual(vectors["full"][0], Fraction(24, 7))
        self.assertEqual(sum(vectors[f][0] for f in d.FACTORS), vectors["full"][0])

    def test_exact_factor_dot_counts_and_value(self):
        audit = {"selected_row_factor_dot_computations": 0, "selected_row_scalar_products": 0}
        vector = [Fraction(1, 3)] * d.WIDTH
        row = np.ones(d.WIDTH, dtype="<f2")
        self.assertEqual(d.exact_factor_dot(vector, row, audit), Fraction(d.WIDTH, 3))
        self.assertEqual(audit["selected_row_factor_dot_computations"], 1)
        self.assertEqual(audit["selected_row_scalar_products"], d.WIDTH)

    def test_classify_case_matches_one_factor(self):
        row = vector_fixture()
        dots = {
            "direct_hidden": {34319: Fraction(3, 8), 319: Fraction()},
            "scalar_scale": {34319: Fraction(), 319: Fraction()},
            "interaction": {34319: Fraction(), 319: Fraction()},
            "full": {34319: Fraction(3, 8), 319: Fraction()},
        }
        result = d.classify_case(row, dots)
        self.assertTrue(result["factor_results"]["direct_hidden"]["direction_agrees"])
        self.assertTrue(result["factor_results"]["direct_hidden"]["crossing_pattern_agrees"])
        self.assertEqual(set(result["closure_residuals"].values()), {"0"})

    def test_classify_case_rejects_wrong_factor_pattern(self):
        row = vector_fixture()
        dots = {
            "direct_hidden": {34319: Fraction(-1, 8), 319: Fraction()},
            "scalar_scale": {34319: Fraction(), 319: Fraction()},
            "interaction": {34319: Fraction(), 319: Fraction()},
            "full": {34319: Fraction(-1, 8), 319: Fraction()},
        }
        result = d.classify_case(row, dots)
        self.assertFalse(result["factor_results"]["direct_hidden"]["direction_agrees"])
        self.assertFalse(result["factor_results"]["direct_hidden"]["crossing_pattern_agrees"])

    def test_classifier_decision_support_and_rejection(self):
        rows = []
        for control in d.CONTROLS:
            for left, right in d.PAIRS:
                rows.append({"control": control, "left_id": left, "right_id": right,
                             "factor_results": {
                                 "direct_hidden": {"direction_agrees": True, "crossing_pattern_agrees": True},
                                 "scalar_scale": {"direction_agrees": False, "crossing_pattern_agrees": True},
                                 "interaction": {"direction_agrees": True, "crossing_pattern_agrees": False},
                             }, "closure_residuals": {"x": "0"}})
        self.assertEqual(d.classifier_decision(rows), ("SUPPORTED", ["direct_hidden"]))
        for row in rows:
            row["factor_results"]["direct_hidden"]["direction_agrees"] = False
        self.assertEqual(d.classifier_decision(rows), ("REJECTED", []))
        rows[0]["closure_residuals"]["x"] = "1"
        with self.assertRaises(ValueError):
            d.classifier_decision(rows)

    def test_forbidden_classifier_replay_guards(self):
        audit = {"forbidden_calls": 0}
        with d.classifier_only(audit):
            for operation in (d.anchor_worker, d.capture_anchors, d.vector_cutoff.run_vectors,
                              d.vector_cutoff.parent.rmsnorm,
                              d.vector_cutoff.parent.logits):
                with self.assertRaises(RuntimeError):
                    operation(None)
        self.assertEqual(audit["forbidden_calls"], 5)

    def test_live_classifier_result_census_and_dispatch(self):
        if self.live is None:
            self.assertIsNone(self.live)
            return
        result = self.live["classifier_result"]
        self.assertEqual(len(result["accounts"]), 27)
        self.assertEqual(result["dispatch"]["selected_head_rows"], [13, 319, 34319])
        self.assertEqual(result["dispatch"]["selected_row_factor_dot_computations"], 108)
        self.assertEqual(result["dispatch"]["selected_row_scalar_products"], 96768)
        self.assertEqual(result["dispatch"]["rmsnorm_invocations"], 0)
        self.assertEqual(result["dispatch"]["scalar_anchor_regeneration"], 0)
        self.assertEqual(result["dispatch"]["model_operator_calls"], 0)
        self.assertEqual({tuple(row["closure_residuals"].values()) for row in result["accounts"]}, {("0", "0", "0")})

    def test_live_classifier_stored_validation(self):
        if self.live is None:
            self.assertIsNone(self.live)
            return
        result = deepcopy(self.live["classifier_result"])
        result["tests"] = {"executed": d.EXPECTED_TESTS, "compiled": [{}, {}],
                           "failures": 0, "errors": 0, "skipped": 0}
        d.validate_classifier_result(result)
        changed = deepcopy(result)
        changed["accounts"][0]["factor_results"]["direct_hidden"]["margin_movement"] = "tampered"
        with self.assertRaises(ValueError):
            d.validate_classifier_result(changed)
        changed = deepcopy(result)
        changed["accounts"][0]["closure_residuals"]["full_plus_boundary_minus_vector"] = "1"
        with self.assertRaises(ValueError):
            d.validate_classifier_result(changed)
