"""Independent bit-decoded exact-dot/RNE oracle and replay-free mutation guards."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_scalar_rne_threshold_classifier_v1 as d


EVIDENCE = REPORT = SUMMARIES = PARENT_SUMMARY = COMPARISONS = PINS = None


def decode(bits):
    bits = int(bits)
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    if exponent == 31:
        raise ValueError("nonfinite oracle word")
    sign = -1 if bits & 0x8000 else 1
    shift = exponent - 25 if exponent else -24
    numerator = sign * (mantissa + (1024 if exponent else 0))
    return Fraction(numerator << shift) if shift >= 0 else Fraction(numerator, 1 << -shift)


def nearest(value):
    sign, magnitude = (0x8000 if value < 0 else 0), abs(value)
    lo, hi = 0, 0x7bff
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode(mid) <= magnitude:
            lo = mid
        else:
            hi = mid - 1
    best = min((lo, min(lo + 1, 0x7bff)),
               key=lambda bits: (abs(decode(bits) - magnitude), bits & 1))
    return sign | best


class ScalarTests(unittest.TestCase):
    def test_independent_hidden_operands(self):
        delta = tuple(decode(a) - Fraction(*float(b).as_integer_ratio()) for a, b in zip(
            EVIDENCE["references"]["rmsnorm_fp16"], EVIDENCE["references"]["rmsnorm_binary64"], strict=True))
        self.assertEqual(REPORT["frozen_final_hidden_residual"], list(map(str, delta)))
        for operand in REPORT["operands"]:
            words = EVIDENCE["arrays"][operand["control"]]["rmsnorm"]
            sign = -1 if operand["polarity"] == "forward" else 1
            targets = tuple(decode(a) + Fraction(sign * 3, 64) * h for a, h in zip(words, delta, strict=True))
            self.assertEqual(operand["baseline_hidden_words"], words.tolist())
            self.assertEqual(operand["exact_targets"], list(map(str, targets)))
            self.assertEqual(operand["working_hidden_words"], [nearest(v) for v in targets])
            self.assertEqual(operand["coordinate_order"], list(range(896)))

    def test_independent_individual_accumulator_scalars(self):
        operands = {(r["control"], r["polarity"]): r for r in REPORT["operands"]}
        for row in REPORT["scalars"]:
            operand = operands[row["control"], row["polarity"]]
            weights = [decode(v) for v in EVIDENCE["rows"][row["row_id"]].view("<u2")]
            for key, values in (
                ("baseline", map(decode, operand["baseline_hidden_words"])),
                ("exact_target", map(Fraction, operand["exact_targets"])),
                ("working", map(decode, operand["working_hidden_words"])),
            ):
                expected = sum((a * b for a, b in zip(values, weights, strict=True)), Fraction())
                result = row[key]
                self.assertEqual(result["exact"], str(expected))
                self.assertEqual(result["fp16_word"], nearest(expected))
                self.assertEqual(result["rounded"], str(decode(result["fp16_word"])))
                self.assertEqual(result["rne_remainder"], str(decode(result["fp16_word"]) - expected))
            self.assertEqual(Fraction(row["hidden_rne_dot_remainder"]),
                             Fraction(row["working"]["exact"]) - Fraction(row["exact_target"]["exact"]))

    def test_rne_ties_subnormals_sign_and_saturation(self):
        for value in (Fraction(), Fraction(1, 1 << 25), Fraction(3, 1 << 25),
                      Fraction(1) + Fraction(1, 2048), Fraction(1) + Fraction(3, 2048),
                      Fraction(2047, 1 << 25), Fraction(65504)):
            for signed in (value, -value):
                self.assertEqual(d.rne(signed)["fp16_word"], nearest(signed))
        for value in (Fraction(65520), Fraction(-65520), Fraction(1, 3), 1.0, float("nan")):
            with self.assertRaises(ValueError):
                d.rne(value)
        a, b = Fraction(1) + Fraction(1, 2048), Fraction(1)
        self.assertNotEqual(decode(nearest(a)) - decode(nearest(b)), decode(nearest(a - b)))

    def test_selection_and_operand_mutations(self):
        for key, value in (("IDS", (319, 13)), ("PAIRS", d.PAIRS[::-1]),
                           ("BRANCHES", ("fp16",)), ("POLARITIES", ("forward",)),
                           ("COORDINATES", (363,)), ("DOSE", Fraction(1, 32)),
                           ("DOSE", 0.046875)):
            with patch.object(d, key, value), self.assertRaises(ValueError):
                d.selection_gate()
        values = (Fraction(),) * 896
        rows = np.zeros((2, 896), dtype="<f2")
        for bad_values, bad_rows in ((values[:-1], rows), (list(values), rows),
                                     (values, rows[:1]), (values, rows.astype("<f8")),
                                     (values, np.full((2, 896), np.nan, dtype="<f2"))):
            with self.assertRaises(ValueError):
                d.accumulate(bad_values, bad_rows, {"exact_selected_row_accumulations": 0})

    def test_protected_evidence_identity_mutations(self):
        identity = d.closed.digest(EVIDENCE)
        d.closed.protect(EVIDENCE, identity)
        for key in EVIDENCE:
            with self.assertRaises(ValueError):
                d.closed.protect({**EVIDENCE, key: "mutation"}, identity)
        for row_id in d.IDS:
            changed = {**EVIDENCE, "rows": {**EVIDENCE["rows"]}}
            changed["rows"][row_id] = changed["rows"][row_id].copy()
            changed["rows"][row_id].view("<u2")[0] ^= 1
            with self.assertRaises(ValueError):
                d.closed.protect(changed, identity)

    def test_authenticated_pin_and_review_mutations(self):
        for pin in (*PINS, *d.retained.RECEIPTS, d.PARENT_RECEIPT,
                    d.parent.record(d.SOURCE), d.parent.record(d.TEST)):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        with patch.object(d, "PARENT_CALL", "missing"), self.assertRaises(ValueError):
            d.authenticate_parent()

    def test_closed_receipt_mutations(self):
        for index in range(8):
            summaries = deepcopy(SUMMARIES)
            summaries[index]["status"] = "UNKNOWN"
            with self.assertRaises(ValueError):
                d.retained.bind_doses(EVIDENCE, summaries)
        for key, value in (("status", "UNKNOWN"), ("pattern_agreements", 71),
                           ("protected_input_identity", "foreign"), ("flags", {}),
                           ("dispatch_and_write_audit", {}), ("tests", {}),
                           ("changed_coordinate_counts", {}), ("contrast_fields", []),
                           ("contrast_values", PARENT_SUMMARY["contrast_values"][::-1])):
            summary = {**PARENT_SUMMARY, key: value}
            with self.assertRaises((ValueError, KeyError)):
                d.bind_parent(EVIDENCE, summary, COMPARISONS)

    def test_forbidden_dispatch_and_writes(self):
        audit = {"forbidden_calls": 0}
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.head.decode_array_q24(None), lambda: d.bridge.measure(),
            lambda: d.bridge.margin.selected_coordinates(None),
            lambda: d.retained.hotspot.audit_coordinates(None, None, None),
            lambda: subprocess.Popen(["false"]), lambda: io.open(d.SOURCE, "w"),
            lambda: os.open(d.TEST, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
        ]
        for module in (d.retained, d.closed, d.retained.bracket, d.retained.bracket.vector,
                       d.retained.bracket.complement, d.retained.bracket.reverse,
                       d.retained.bracket.half, d.retained.bracket.quarter, d.retained.bracket.eighth,
                       d.retained.bracket.sixteenth, d.retained.bracket.thirtysecond):
            for name in ("check", "run_tests", "prepare"):
                operations.append(lambda module=module, name=name: getattr(module, name)())
        with d.read_only(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], len(operations))

    def test_falsifiable_successors_and_unknown(self):
        for target, working, expected in (
            (72, 72, ("supported", d.EXACT_SUCCESSOR)),
            (72, 0, ("supported", d.EXACT_SUCCESSOR)),
            (71, 72, ("supported", d.HIDDEN_SUCCESSOR)),
            (0, 72, ("supported", d.HIDDEN_SUCCESSOR)),
            (71, 71, ("UNKNOWN", None)), (0, 0, ("UNKNOWN", None)),
        ):
            self.assertEqual(d.decide(target, working, 72), expected)
        for values in ((73, 72, 72), (True, 72, 72), (72, 72, 71)):
            with self.assertRaises(ValueError):
                d.decide(*values)

    def test_contrast_integrity_mutations(self):
        rows = REPORT["contrasts"]
        for changed in (rows[:-1], rows[::-1], [rows[0]] * 72):
            with self.assertRaises(ValueError):
                d.classify(changed)
        for key in ("closed_bracket_delta", "fixed_reference_margin", "retained_margin_change",
                    "target_delta", "working_delta", "target_margin_change", "working_margin_change"):
            changed = deepcopy(rows)
            changed[0][key] = str(Fraction(changed[0][key]) + 1)
            with self.assertRaises(ValueError):
                d.classify(changed)

    def test_independent_ordered_margin_agreement_counts(self):
        scalars = {(r["control"], r["polarity"], r["row_id"]): r for r in REPORT["scalars"]}
        counts = {"target": 0, "working": 0}
        for row in REPORT["contrasts"]:
            left = scalars[row["control"], row["polarity"], row["left_id"]]
            right = scalars[row["control"], row["polarity"], row["right_id"]]
            old = decode(left["baseline"]["fp16_word"]) - decode(right["baseline"]["fp16_word"])
            for name, key in (("target", "exact_target"), ("working", "working")):
                new = decode(left[key]["fp16_word"]) - decode(right[key]["fp16_word"])
                self.assertEqual(row[name + "_delta"], str(new - old))
                counts[name] += str(new - old) == row["closed_bracket_delta"]
        self.assertEqual(counts["target"], REPORT["target_pattern_agreements"])
        self.assertEqual(counts["working"], REPORT["working_pattern_agreements"])
        self.assertEqual(d.classify(REPORT["contrasts"])["successor"], REPORT["successor"])

    def test_single_json_unknown_and_claim_boundary(self):
        for error in (ValueError("authentication defect"), RuntimeError("scope defect")):
            output = io.StringIO()
            with patch.object(d, "check", side_effect=error), redirect_stdout(output):
                code = d.main(["--check"])
            value = json.loads(output.getvalue())
            self.assertEqual((code, value["status"], value["successor"]), (1, "UNKNOWN", None))
            self.assertIn(str(error), value["integrity_error"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--dose", "1/32"])
        for key, value in d.FLAGS.items():
            if key not in ("historical_failures_preserved", "original_global_reference_unchanged"):
                self.assertIn(value, (False, 0), key)
