"""Independent FP16-bit oracle for retained affine dose thresholds, without replay."""

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

from ace3.model.candidates import diagnose_q24_s16_terminal_remainder_selected_head_coupled_rne_dose_threshold_bracket_v1 as d


EVIDENCE = REPORT = RETAINED = RECEIPT = SUMMARY = ROWS = SUMMARIES = HEAD_SUMMARY = None


def decode(word):
    return Fraction(struct.unpack("<e", struct.pack("<H", word))[0])


def nearest(value):
    lo, hi = 0, 0x7bff
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode(mid) <= abs(value):
            lo = mid
        else:
            hi = mid - 1
    word = min((lo, min(lo + 1, 0x7bff)), key=lambda w: (abs(decode(w) - abs(value)), w & 1))
    return word | (0x8000 if value < 0 else 0)


class DoseThresholdTests(unittest.TestCase):
    def test_independent_exact_thresholds_and_slacks(self):
        self.assertEqual(len(REPORT["rows"]), 36)
        for row in REPORT["rows"]:
            b, slope = Fraction(row["baseline_exact"]), Fraction(row["signed_row_dose_slope"])
            t, word = row["threshold"], row["baseline_word"]
            direction = 1 if slope > 0 else -1
            midpoint = (decode(word) + decode(word + direction)) / 2
            dose = (midpoint - b) / slope
            self.assertEqual(Fraction(t["midpoint"]), midpoint)
            self.assertEqual(Fraction(t["dose_to_midpoint"]), dose)
            self.assertEqual(t["tie_winner_word"], nearest(midpoint))
            for dose_point in (d.LOW, d.HIGH):
                account = t["bracket"][str(dose_point)]
                self.assertEqual(Fraction(account["dose_slack"]), dose - dose_point)
                self.assertEqual(Fraction(account["accumulator_slack"]),
                                 direction * (midpoint - b - slope * dose_point))
                self.assertEqual(account["crossed"], nearest(b + slope * dose_point) != word)
            self.assertEqual(nearest(b + slope * d.LOW), row["lower_word"])
            self.assertEqual(nearest(b + slope * d.HIGH), row["upper_word"])

    def test_independent_cells_affine_identity_and_working_remainder(self):
        slopes = {}
        for row in REPORT["rows"]:
            for name in ("baseline", "target", "working"):
                self.assertEqual(nearest(Fraction(row[name + "_exact"])), row[name + "_word"])
            slope = Fraction(row["frozen_row_dose_slope"])
            self.assertEqual(slope, slopes.setdefault(row["row_id"], slope))
            signed = -slope if row["polarity"] == "forward" else slope
            self.assertEqual(Fraction(row["target_exact"]),
                             Fraction(row["baseline_exact"]) + d.HIGH * signed)
            self.assertEqual(Fraction(row["hidden_rne_dot_remainder"]),
                             Fraction(row["working_exact"]) - Fraction(row["target_exact"]))
            self.assertIsNone(row["working_dose_threshold"])

    def test_independent_reference_and_endpoint_margin_closure(self):
        lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in REPORT["rows"]}
        for item in REPORT["contrasts"]:
            c, p, l, r, branch = (item[k] for k in ("control", "polarity", "left_id", "right_id", "branch"))
            left, right = lookup[c, p, l], lookup[c, p, r]
            refs = EVIDENCE["references"]["logits_" + branch]
            reference = (decode(int(refs[l])) - decode(int(refs[r])) if branch == "fp16"
                         else Fraction(float(refs[l])) - Fraction(float(refs[r])))
            old = decode(left["baseline_word"]) - decode(right["baseline_word"])
            self.assertEqual(Fraction(item["fixed_reference_margin"]), reference)
            for name, dose in (("lower", d.LOW), ("upper", d.HIGH)):
                words = [nearest(Fraction(x["baseline_exact"]) +
                                 dose * Fraction(x["signed_row_dose_slope"])) for x in (left, right)]
                margin = decode(words[0]) - decode(words[1])
                self.assertEqual(Fraction(item[name + "_exact_delta"]), margin - old)
                self.assertEqual(Fraction(item[name + "_margin_change"]), margin - reference)

    def test_independent_all_closed_dose_nonzero_consistency(self):
        lookup = {(r["control"], r["polarity"], r["row_id"]): r for r in REPORT["rows"]}
        agreements = 0
        for item in REPORT["closed_dose_consistency"]:
            dose = Fraction(item["dose"])
            selected = [lookup[item["control"], item["polarity"], i]
                        for i in (item["left_id"], item["right_id"])]
            active = [r["row_id"] for r in selected if
                      nearest(Fraction(r["baseline_exact"]) + dose * Fraction(r["signed_row_dose_slope"]))
                      != r["baseline_word"]]
            self.assertEqual(item["first_threshold_crossed_rows"], active)
            expected = bool(active) == bool(Fraction(item["closed_delta"]))
            self.assertEqual(item["nonzero_consistent"], expected)
            agreements += item["consistent"]
        self.assertEqual(len(REPORT["closed_dose_consistency"]), 504)
        self.assertEqual(REPORT["closed_dose_consistency_agreements"], agreements)
        self.assertEqual(sum(x["agreements"] for x in REPORT["closed_dose_counts"].values()), agreements)

    def test_signed_ties_zero_subnormals_and_binade_boundaries(self):
        for word in (0, 1, 2, 1023, 1024, 0x3bff, 0x3c00, 0x3c01, 0xbc00, 0xbc01, 0x8001):
            for slope in (Fraction(1), Fraction(-1)):
                b = decode(word)
                account = d.threshold(b, word, slope)
                dose = Fraction(account["dose_to_midpoint"])
                midpoint = b + slope * dose
                self.assertEqual(account["tie_winner_word"], nearest(midpoint))
                self.assertEqual(d.reached(account, dose), bool(word & 1))
                epsilon = Fraction(1, 1 << 80)
                if dose:
                    self.assertFalse(d.reached(account, dose - epsilon))
                else:
                    self.assertEqual(account["boundary_kind"], "signed_zero_transition")
                    self.assertFalse(d.reached(account, Fraction()))
                    with self.assertRaises(ValueError):
                        d.reached(account, -epsilon)
                self.assertTrue(d.reached(account, dose + epsilon))
                self.assertNotEqual(nearest(b + slope * (dose + epsilon)), word)
                self.assertEqual(account["baseline_dose_interval"]["upper_inclusive"], not bool(word & 1))
                self.assertEqual(account["crossed_dose_interval"]["lower_inclusive"], bool(word & 1))
        negative_zero = d.threshold(-Fraction(1, 1 << 26), 0x8000, Fraction(1))
        self.assertEqual(negative_zero["boundary_kind"], "signed_zero_transition")
        self.assertEqual(negative_zero["dose_to_midpoint"], str(Fraction(1, 1 << 26)))
        self.assertEqual(negative_zero["tie_winner_word"], 0)
        self.assertTrue(d.reached(negative_zero, Fraction(1, 1 << 26)))
        with self.assertRaises(ValueError):
            d.threshold(Fraction(65504), 0x7bff, Fraction(1))
        with self.assertRaises(ValueError):
            d.threshold(Fraction(), 0x8000, Fraction(1))

    def test_nondyadic_dose_and_exact_endpoint_ownership(self):
        account = d.threshold(Fraction(1), 0x3c00, Fraction(3))
        self.assertEqual(account["dose_to_midpoint"], "1/6144")
        self.assertFalse(d.reached(account, Fraction(1, 6144)))
        for dose in (d.LOW, d.HIGH):
            odd = 0x3c01
            midpoint = (decode(odd) + decode(odd + 1)) / 2
            slope = (midpoint - decode(odd)) / dose
            account = d.threshold(decode(odd), odd, slope)
            self.assertTrue(d.reached(account, dose))
            even = d.threshold(decode(odd + 1), odd + 1, -slope)
            self.assertFalse(d.reached(even, dose))

    def test_invalid_inputs_and_selection_mutations(self):
        for text in ("2/4", 0.5, "nan"):
            with self.assertRaises((ValueError, TypeError)):
                d.exact(text)
        for slope in (Fraction(), 1.0):
            with self.assertRaises(ValueError):
                d.threshold(Fraction(1), 0x3c00, slope)
        for key, value in (("IDS", (319, 13)), ("PAIRS", d.PAIRS[::-1]),
                           ("BRANCHES", ("fp16",)), ("LOW", 0.03125),
                           ("HIGH", Fraction(1, 16)), ("DOSES", d.DOSES[::-1])):
            with patch.object(d, key, value), self.assertRaises(ValueError):
                d.selection_gate()

    def test_scalar_slope_midpoint_and_row_census_mutations(self):
        for key, value in (("baseline_exact", "0"), ("baseline_word", 18460),
                           ("target_exact", "0"), ("target_word", 18461),
                           ("working_exact", "0"), ("frozen_row_dose_slope", "1"),
                           ("signed_dose", "3/64"), ("hidden_rne_dot_remainder", "0")):
            rows = deepcopy(RETAINED["rows"])
            rows[0][key] = value
            with self.assertRaises(ValueError):
                d.solve_rows(rows, EVIDENCE)
        rows = deepcopy(RETAINED["rows"])
        rows[0]["cells"]["baseline"]["lower_midpoint"] = "0"
        with self.assertRaises(ValueError):
            d.solve_rows(rows, EVIDENCE)
        for rows in (RETAINED["rows"][:-1], RETAINED["rows"][::-1]):
            with self.assertRaises(ValueError):
                d.solve_rows(rows, EVIDENCE)

    def test_authenticated_receipt_and_historical_gate_mutations(self):
        for key, value in (("status", "UNKNOWN"), ("protected_input_identity", "foreign"),
                           ("flags", {}), ("tests", {}), ("dispatch_and_write_audit", {}),
                           ("retained_scalar_receipt", {}), ("retained_dose_summaries", []),
                           ("retained_thresholds", {}), ("retained_controls_and_failure_gates", []),
                           ("final_reference_authority", {}), ("retained_common_component", "PASS")):
            receipt = deepcopy(RECEIPT)
            receipt["native_result"][key] = value
            with self.assertRaises((ValueError, KeyError)):
                d.bind_crossing(receipt, EVIDENCE, SUMMARY, ROWS, SUMMARIES, HEAD_SUMMARY)
        for pin in (*d.PINS, d.RECEIPT):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        identity = d.closed.digest(EVIDENCE)
        for key in EVIDENCE:
            with self.assertRaises(ValueError):
                d.closed.protect({**EVIDENCE, key: "mutation"}, identity)

    def test_contrast_and_fixed_reference_mutations(self):
        for key in ("fixed_reference_margin", "baseline_rounded_margin", "closed_bracket_delta"):
            retained = deepcopy(RETAINED)
            retained["contrasts"][0][key] = "0"
            with self.assertRaises(ValueError):
                d.solve(retained, EVIDENCE, SUMMARIES)
        for key, value in (("crossing_rows", []), ("delta", "0"), ("left_contribution", "0"),
                           ("margin_change", "0"), ("opposite_row_compensation", True)):
            retained = deepcopy(RETAINED)
            retained["contrasts"][0]["target"][key] = value
            with self.assertRaises(ValueError):
                d.solve(retained, EVIDENCE, SUMMARIES)
        retained = {**RETAINED, "contrasts": RETAINED["contrasts"][::-1]}
        with self.assertRaises(ValueError):
            d.solve(retained, EVIDENCE, SUMMARIES)

    def test_falsifiable_joint_and_asymmetric_successors(self):
        coupled = [{"both_new_in_bracket": [True, True]} for _ in range(4)]
        self.assertEqual(d.decide(coupled, 68, 72, 504)["successor"], d.JOINT)
        changed = deepcopy(coupled)
        changed[0]["both_new_in_bracket"] = [True, False]
        for args in ((changed, 68, 72, 504), (coupled, 67, 72, 504),
                     (coupled, 68, 71, 504), (coupled, 68, 72, 503)):
            result = d.decide(*args)
            self.assertEqual((result["classification"], result["successor"]), ("rejected", d.ASYMMETRIC))
        for rows in ([], coupled[:-1]):
            with self.assertRaises(ValueError):
                d.decide(rows, 68, 72, 504)

    def test_forbidden_dispatch_and_artifact_overwrites(self):
        audit = {"forbidden_calls": 0}
        operations = [
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.bridge.measure(),
            lambda: d.scalar.retained.dots(None, None), lambda: subprocess.Popen(["false"]),
            lambda: io.open(d.SOURCE, "w"), lambda: os.open(d.TEST, os.O_WRONLY | os.O_TRUNC),
            lambda: d.SOURCE.unlink(),
        ]
        for module, names in (
            (d.crossing, ("check", "run_tests", "attribute", "classify")),
            (d.scalar, ("check", "run_tests", "measure", "accumulate", "rne", "classify")),
            (d.scalar.retained, ("check", "run_tests", "head_only", "prepare")),
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
        for error in (ValueError("arithmetic integrity"), RuntimeError("forbidden replay")):
            stream = io.StringIO()
            with patch.object(d, "check", side_effect=error), redirect_stdout(stream):
                code = d.main(["--check"])
            result = json.loads(stream.getvalue())
            self.assertEqual((code, result["classification"], result["successor"]),
                             (1, "UNKNOWN/integrity", None))
            self.assertIn(str(error), result["integrity_error"])
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"status": "rejected"}), redirect_stdout(stream):
            self.assertEqual(d.main(["--check"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), {"status": "rejected"})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--dose", "1/32"])

    def test_boundary_counts_and_contract_mutation_guards(self):
        self.assertEqual(REPORT["selected_branch"], "exact_target")
        self.assertEqual(REPORT["row_threshold_closures"], 36)
        self.assertEqual(REPORT["target_contrast_closures"], 72)
        self.assertEqual(REPORT["working_contrast_closures"], 72)
        self.assertEqual(REPORT["upper_exact_margin_agreements"], 72)
        self.assertEqual(len(REPORT["coupled_contrasts"]), 4)
        self.assertEqual(REPORT["other_single_row_contrasts"], 68)
        for key, value in d.FLAGS.items():
            if key not in ("historical_failures_preserved", "original_global_reference_unchanged"):
                self.assertIn(value, (False, 0), key)
