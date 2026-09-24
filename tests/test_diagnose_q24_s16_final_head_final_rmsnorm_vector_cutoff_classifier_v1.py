"""Independent signed-integer FP16 dot oracle and vector-classifier fault tests."""

from copy import deepcopy
from fractions import Fraction
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_vector_cutoff_classifier_v1 as d


EVIDENCE = None


def integer_half(word):
    word = int(word)
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16 oracle operand")
    magnitude = fraction if exponent == 0 else (1024+fraction) << (exponent-1)
    return -magnitude if word & 32768 else magnitude


def fixture(a=Fraction(1, 8), r=Fraction(-1, 16),
            ea=Fraction(9, 64), er=Fraction(-3, 64)):
    values = [a, r, a-r, ea, er, ea-er, a-ea, er-r, (a-ea)+(er-r)]
    return dict(zip(d.previous.COLUMNS,
                    [d.parent.CONTROLS[0], 34319, 319, "fp16", *map(str, values)], strict=True))


class VectorClassifierTests(unittest.TestCase):
    def test_exact_vector_swap_and_parent_closure(self):
        row = fixture()
        result = d.classify_pair(row, (Fraction(9, 64), Fraction()),
                                (Fraction(-3, 64), Fraction()))
        self.assertEqual(result["vector_swap_margin_movement"], "-3/16")
        self.assertEqual(result["observed_crossing_pattern"], [1, -1])
        self.assertTrue(result["direction_agrees"])
        self.assertTrue(result["crossing_pattern_agrees"])
        self.assertEqual(set(result["exact_closure_residuals"].values()), {"0"})

    def test_direction_prediction_can_be_rejected(self):
        row = fixture(ea=Fraction(1), er=Fraction(2))
        result = d.classify_pair(row, (Fraction(1), Fraction()), (Fraction(2), Fraction()))
        self.assertFalse(result["direction_agrees"])
        self.assertFalse(result["crossing_pattern_agrees"])

    def test_zero_is_not_directional_or_crossing_agreement(self):
        row = fixture(ea=Fraction(), er=Fraction())
        result = d.classify_pair(row, (Fraction(), Fraction()), (Fraction(), Fraction()))
        self.assertEqual(result["observed_movement_sign"], 0)
        self.assertFalse(result["direction_agrees"])
        self.assertFalse(result["crossing_pattern_agrees"])
        row = fixture(a=Fraction(), r=Fraction(), ea=Fraction(), er=Fraction())
        result = d.classify_pair(row, (Fraction(), Fraction()), (Fraction(), Fraction()))
        self.assertTrue(result["direction_agrees"])
        self.assertTrue(result["crossing_pattern_agrees"])

    def test_parent_splices_and_dot_splices_rejected(self):
        for column in d.previous.COLUMNS[4:]:
            row = fixture()
            row[column] = str(Fraction(row[column])+1)
            with self.subTest(column=column), self.assertRaises(ValueError):
                d.prediction(row)
        with self.assertRaises(ValueError):
            d.classify_pair(fixture(), (Fraction(1), Fraction()), (Fraction(), Fraction()))
        with self.assertRaises(ValueError):
            d.classify_pair(fixture(), (0.0, 0.0), (0.0, 0.0))

    def test_complete_decision_and_distinct_successors(self):
        rows = [{"control": c, "left_id": a, "right_id": b, "branch": "fp16",
                 "direction_agrees": True, "crossing_pattern_agrees": True}
                for c in d.parent.CONTROLS for a, b in d.PAIRS]
        self.assertEqual(d.decision(rows), "SUPPORTED")
        for field in ("direction_agrees", "crossing_pattern_agrees"):
            changed = deepcopy(rows)
            changed[4][field] = False
            self.assertEqual(d.decision(changed), "REJECTED")
        for changed in (rows[:-1], rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):
                d.decision(changed)
        self.assertEqual(len(set(d.SUCCESSORS.values())), 3)

    def test_integer_oracle_all_live_selected_dots(self):
        self.assertIsNotNone(EVIDENCE, "task-native tests require fresh vector execution")
        self.assertEqual(len(EVIDENCE["dots"]), 30)
        for key, (vector, weights) in EVIDENCE["operands"].items():
            x = [integer_half(w) for w in vector.view("<u2")]
            w = [integer_half(v) for v in weights.view("<u2")]
            oracle = Fraction(sum(a*b for a, b in zip(x, w, strict=True)), 1 << 48)
            self.assertEqual(EVIDENCE["dots"][key], oracle)

    def test_live_preregistration_and_reference_anchors(self):
        self.assertIsNotNone(EVIDENCE)
        records = deepcopy(EVIDENCE["records"])
        prereg = d.preregister(records)
        self.assertEqual(len(prereg["rows"]), 27)
        self.assertEqual(prereg["frozen_selected_row_ids"], [13, 319, 34319])
        self.assertEqual(records, EVIDENCE["records"])
        for row in EVIDENCE["accounts"]:
            self.assertEqual(row["immutable_reference_margin"],
                             row["parent_account"]["retained_reference_margin"])
            self.assertEqual(set(row["exact_closure_residuals"].values()), {"0"})
        for alteration in ("duplicate", "row", "branch"):
            changed = deepcopy(records)
            if alteration == "duplicate":
                changed[-1] = changed[0]
            else:
                changed[0]["left_id" if alteration == "row" else "branch"] = "changed"
            with self.assertRaises(ValueError):
                d.preregister(changed)

    def test_runtime_pins_and_capture_byte_tamper(self):
        runtime = d.previous.runtime_pins()
        result = {"runtime_pins": runtime, "preregistration": {"runtime_pins": runtime}}
        command = {"runtime_pins": runtime}
        self.assertEqual(d.previous.validate_runtime_pins(result, command, command), runtime)
        with self.assertRaises(ValueError):
            d.previous.validate_runtime_pins(result, {}, command)
        with self.assertRaises(ValueError):
            d.previous.pin_bytes(b"tampered", d.PINS[3])

    def test_forbidden_dispatch_and_evidence_writes(self):
        audit = {"forbidden_calls": 0}
        with d.selected_only(audit):
            for operation in (d.previous.execute_check, d.previous.run_selected,
                              d.previous.round_half, d.bridge.mixed.exact_head,
                              d.parent.rmsnorm, d.parent.logits):
                with self.assertRaises(RuntimeError):
                    operation(None)
            with self.assertRaises(RuntimeError):
                d.SOURCE.open("wb")
        self.assertEqual(audit["forbidden_calls"], 7)

    def test_stored_scalar_validation_has_no_vector_dispatch(self):
        self.assertIsNotNone(EVIDENCE)
        accounts = EVIDENCE["accounts"]
        selected = d.decision(accounts)
        result = {
            "diagnostic_id": d.NAME, "task": d.TASK, "command": d.COMMAND,
            "execution_valid": True, "normal_host_review": "REQUIRED",
            "runtime_pins": {}, "parent_records": EVIDENCE["records"],
            "preregistration": {**d.preregister(EVIDENCE["records"]), "runtime_pins": {}},
            "accounts": accounts, "decision": selected, "successor": d.SUCCESSORS[selected],
            "direction_agreements": sum(r["direction_agrees"] for r in accounts),
            "crossing_pattern_agreements": sum(r["crossing_pattern_agrees"] for r in accounts),
            "tests": {"executed": d.EXPECTED_TESTS, "compiled": [{}, {}],
                      "failures": 0, "errors": 0, "skipped": 0},
            "dispatch_and_write_audit": {
                "forbidden_calls": 0, "artifact_overwrites": 0, "scalar_rne_computations": 0,
                "reference_trajectory_changes": 0, "full_vocabulary_ranking_calls": 0,
                "local_exact_row_dot_computations": 30, "selected_row_scalar_products": 26880,
            },
        }
        with patch.object(d, "run_vectors", side_effect=AssertionError("forbidden recomputation")):
            d.validate_result(result)
            for field in ("vector_swap_margin_movement", "direction_agrees",
                          "observed_crossing_pattern", "immutable_reference_margin"):
                changed = deepcopy(result)
                changed["accounts"][0][field] = "tampered"
                with self.subTest(field=field), self.assertRaises(ValueError):
                    d.validate_result(changed)


if __name__ == "__main__":
    unittest.main()
