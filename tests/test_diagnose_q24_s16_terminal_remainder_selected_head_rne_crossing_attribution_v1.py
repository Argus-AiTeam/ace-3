"""Independent integer-bit nearest-even oracle and replay-free attribution guards."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import struct
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_rne_crossing_attribution_v1 as d


EVIDENCE = REPORT = SUMMARY = ROWS = COMPARISONS = HEAD_ROWS = PINS = None


def decode(word):
    return Fraction(struct.unpack("<e", struct.pack("<H", word))[0])


def nearest(value):
    magnitude = abs(value)
    lo, hi = 0, 0x7bff
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode(mid) <= magnitude:
            lo = mid
        else:
            hi = mid - 1
    word = min((lo, min(lo + 1, 0x7bff)),
               key=lambda w: (abs(decode(w) - magnitude), w & 1))
    return word | (0x8000 if value < 0 else 0)


class CrossingTests(unittest.TestCase):
    def test_independent_retained_scalar_rounding_and_cells(self):
        self.assertEqual(len(REPORT["rows"]), 36)
        for row in REPORT["rows"]:
            for name in ("baseline", "target", "working"):
                exact, word = Fraction(row[name + "_exact"]), row[name + "_word"]
                self.assertEqual(nearest(exact), word)
                cell = row["cells"][name]
                self.assertEqual(Fraction(cell["rounded"]), decode(word))
                lower = (decode(word - 1) + decode(word)) / 2
                upper = (decode(word) + decode(word + 1)) / 2
                self.assertEqual(Fraction(cell["lower_midpoint"]), lower)
                self.assertEqual(Fraction(cell["upper_midpoint"]), upper)
                self.assertEqual(Fraction(cell["lower_distance"]), exact - lower)
                self.assertEqual(Fraction(cell["upper_distance"]), upper - exact)
                self.assertEqual(Fraction(cell["rne_remainder"]), decode(word) - exact)
                self.assertEqual(cell["tie"], exact in (lower, upper))

    def test_independent_slack_and_signed_dose_accounts(self):
        slopes = {}
        for row in REPORT["rows"]:
            start = Fraction(row["baseline_exact"])
            for name in ("target", "working"):
                account = row[name]
                end = Fraction(row[name + "_exact"])
                direction = 1 if end > start else -1
                threshold = (decode(row["baseline_word"]) +
                             decode(row["baseline_word"] + direction)) / 2
                self.assertEqual(Fraction(account["signed_movement"]), end - start)
                self.assertEqual(Fraction(account["initial_slack"]), direction * (threshold - start))
                self.assertEqual(Fraction(account["remaining_slack"]), direction * (threshold - end))
                self.assertEqual(account["word_changed"], row[name + "_word"] != row["baseline_word"])
                self.assertEqual(account["slack_closure_residual"], "0")
            slope = (Fraction(row["target_exact"]) - start) / Fraction(row["signed_dose"])
            self.assertEqual(slope, slopes.setdefault(row["row_id"], slope))
            self.assertEqual(Fraction(row["hidden_rne_dot_remainder"]),
                             Fraction(row["working_exact"]) - Fraction(row["target_exact"]))

    def test_independent_ordered_reference_closure(self):
        rows = {(r["control"], r["polarity"], r["row_id"]): r for r in REPORT["rows"]}
        single = coupled = 0
        for contrast in REPORT["contrasts"]:
            c, p, left, right, branch = (contrast[k] for k in (
                "control", "polarity", "left_id", "right_id", "branch"))
            lrow, rrow = rows[c, p, left], rows[c, p, right]
            old = decode(lrow["baseline_word"]) - decode(rrow["baseline_word"])
            refs = EVIDENCE["references"]["logits_" + branch]
            ref = (decode(int(refs[left])) - decode(int(refs[right])) if branch == "fp16"
                   else Fraction(float(refs[left])) - Fraction(float(refs[right])))
            self.assertEqual(Fraction(contrast["fixed_reference_margin"]), ref)
            for name in ("target", "working"):
                new = decode(lrow[name + "_word"]) - decode(rrow[name + "_word"])
                self.assertEqual(Fraction(contrast[name]["delta"]), new - old)
                self.assertEqual(str(new - old), COMPARISONS[c, p, left, right])
                self.assertEqual(Fraction(contrast[name]["margin_change"]), new - ref)
            count = sum(rows[c, p, i]["target_word"] != rows[c, p, i]["baseline_word"]
                        for i in (left, right))
            single += count == 1
            coupled += count == 2
        self.assertEqual((single, coupled), (REPORT["single_row_contrasts"], REPORT["two_row_contrasts"]))
        self.assertEqual(single + coupled, 72)

    def test_tie_parity_signed_zero_subnormal_and_overflow(self):
        for word in (0, 1, 2, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0x4800):
            midpoint = (decode(word) + decode(word + 1)) / 2
            for exact in (midpoint, -midpoint):
                result = d.cell(exact, nearest(exact))
                self.assertTrue(result["tie"])
                self.assertTrue(result["even_significand"])
                with self.assertRaises(ValueError):
                    d.cell(exact, nearest(exact) ^ 1)
        self.assertEqual(d.cell(Fraction(), 0)["rounded"], "0")
        for value, word in ((Fraction(65520), 0x7bff), (Fraction(-65520), 0xfbff),
                            (Fraction(), 0x8000)):
            with self.assertRaises(ValueError):
                d.cell(value, word)
        for word in (-1, 65536, True, 0x7c00, 0x7e00):
            with self.assertRaises(ValueError):
                d.decode(word)
        for value in ("1/3", "2/4", 1.0, "nan"):
            with self.assertRaises(ValueError):
                d.rational(value)

    def test_threshold_ties_and_no_movement(self):
        for start_word in (0x3c00, 0x3c01, 0xbc00, 0xbc01):
            start = decode(start_word)
            for neighbor in (start_word - 1, start_word + 1):
                midpoint = (start + decode(neighbor)) / 2
                baseline = d.cell(start, start_word)
                end = d.cell(midpoint, nearest(midpoint))
                account = d.movement(baseline, end)
                self.assertEqual(account["word_changed"], bool(start_word & 1))
                self.assertEqual(account["remaining_slack"], "0")
                self.assertTrue(account["endpoint_at_baseline_midpoint"])
            self.assertFalse(d.movement(baseline, baseline)["word_changed"])

    def test_scalar_midpoint_baseline_and_dose_mutations(self):
        for key, value in (("baseline_exact", "0"), ("target_exact", "0"),
                           ("working_exact", "0"), ("target_word", 18461),
                           ("baseline_word", 18460)):
            rows = deepcopy(ROWS)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                d.attribute(rows, EVIDENCE, COMPARISONS, HEAD_ROWS)
        rows = deepcopy(ROWS)
        rows[0]["target_exact"] = str(Fraction(rows[0]["target_exact"]) + Fraction(1, 1 << 60))
        with self.assertRaises(ValueError):
            d.attribute(rows, EVIDENCE, COMPARISONS, HEAD_ROWS)

    def test_receipt_shape_outcome_and_identity_mutations(self):
        for key, value in (("status", "UNKNOWN"), ("successor", d.scalar.HIDDEN_SUCCESSOR),
                           ("native_exit", 1), ("stdout_json_documents", 2),
                           ("target_pattern_agreements", 71), ("working_pattern_agreements", 71),
                           ("protected_input_identity", "foreign"), ("tests", {}),
                           ("dispatch_and_write_audit", {}), ("scalar_fields", []),
                           ("scalar_values", SUMMARY["scalar_values"][::-1])):
            with self.assertRaises((ValueError, KeyError)):
                d.bind_receipt({**SUMMARY, key: value}, EVIDENCE, PINS)
        for pin in (*d.PINS, d.RECEIPT, d.parent.record(d.SOURCE), d.parent.record(d.TEST)):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        with patch.object(d, "CALL", "missing"), self.assertRaises(ValueError):
            d.authenticate_receipt()
        identity = d.closed.digest(EVIDENCE)
        for key in EVIDENCE:
            with self.assertRaises(ValueError):
                d.closed.protect({**EVIDENCE, key: "mutation"}, identity)

    def test_selection_and_contrast_mutations(self):
        for key, value in (("IDS", (319, 13)), ("PAIRS", d.PAIRS[::-1]),
                           ("BRANCHES", ("fp16",)), ("POLARITIES", ("forward",)),
                           ("DOSE", Fraction(1, 32)), ("DOSE", 0.046875)):
            with patch.object(d, key, value), self.assertRaises(ValueError):
                d.selection_gate()
        for changed in (REPORT["contrasts"][:-1], REPORT["contrasts"][::-1]):
            with self.assertRaises(ValueError):
                d.classify(changed)
        for key in ("delta", "rounded_margin", "margin_change", "left_contribution"):
            rows = deepcopy(REPORT["contrasts"])
            rows[0]["target"][key] = str(Fraction(rows[0]["target"][key]) + 1)
            with self.assertRaises(ValueError):
                d.classify(rows)
        rows = deepcopy(REPORT["contrasts"])
        rows[0]["target"]["crossing_rows"] = [34319]
        with self.assertRaises(ValueError):
            d.classify(rows)

    def test_falsifiable_single_coupled_and_compensation_successors(self):
        rows = deepcopy(REPORT["contrasts"])
        for row in rows:
            for name in ("target", "working"):
                branch = row[name]
                branch.update(left_contribution=branch["delta"], right_contribution="0",
                              crossing_rows=[row["left_id"]], opposite_row_compensation=False)
        self.assertEqual(d.classify(rows)["successor"], d.SINGLE)
        for compensating in (False, True):
            changed = deepcopy(rows)
            for name in ("target", "working"):
                branch = changed[0][name]
                delta = Fraction(branch["delta"])
                left = 2 * delta if compensating else delta / 2
                branch.update(left_contribution=str(left), right_contribution=str(delta - left),
                              crossing_rows=[changed[0]["left_id"], changed[0]["right_id"]],
                              opposite_row_compensation=compensating)
            self.assertEqual(d.classify(changed)["successor"], d.COUPLED)

    def test_forbidden_replay_and_artifact_writes(self):
        audit = {"forbidden_calls": 0}
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.measure(),
            lambda: d.scalar.retained.dots(None, None), lambda: d.scalar.retained.prepare(None, None, None),
            lambda: subprocess.Popen(["false"]), lambda: io.open(d.SOURCE, "w"),
            lambda: os.open(d.TEST, os.O_WRONLY | os.O_TRUNC), lambda: d.SOURCE.unlink(),
        ]
        for module, names in (
            (d.scalar, ("check", "run_tests", "measure", "accumulate", "rne", "classify")),
            (d.scalar.retained, ("check", "run_tests", "head_only")),
            (d.scalar.retained.bracket, ("check", "run_tests", "prepare")),
        ):
            for name in names:
                operations.append(lambda module=module, name=name: getattr(module, name)())
        with d.read_only(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], len(operations))

    def test_single_json_unknown_and_fixed_cli(self):
        for error in (ValueError("receipt integrity"), RuntimeError("forbidden replay")):
            stream = io.StringIO()
            with patch.object(d, "check", side_effect=error), redirect_stdout(stream):
                code = d.main(["--check"])
            result = json.loads(stream.getvalue())
            self.assertEqual((code, result["classification"], result["successor"]),
                             (1, "UNKNOWN/integrity", None))
            self.assertIn(str(error), result["integrity_error"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--dose", "1/32"])

    def test_claim_boundary_and_closed_exact_precedence(self):
        self.assertEqual(REPORT["selected_branch"], "exact_target")
        self.assertEqual(SUMMARY["successor"], d.scalar.EXACT_SUCCESSOR)
        self.assertEqual(REPORT["target_contrast_closures"], 72)
        self.assertEqual(REPORT["working_contrast_closures"], 72)
        for key, value in d.FLAGS.items():
            if key not in ("historical_failures_preserved", "original_global_reference_unchanged"):
                self.assertIn(value, (False, 0), key)
