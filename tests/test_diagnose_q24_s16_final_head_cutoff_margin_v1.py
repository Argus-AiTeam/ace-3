"""Independent sorting/Fraction oracles and fail-closed cutoff diagnostics."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_cutoff_margin_v1 as d


EVIDENCE = None


def vector(values):
    return np.asarray(values, dtype="<f8")


class CutoffMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is not None:
            cls.result, cls.arrays, cls.references, cls.report = EVIDENCE
        else:
            with d.base.read_only({"forbidden_calls": 0}):
                d.base.read_bound(d.AUTHENTICATOR_PIN)
                cls.result, cls.arrays, cls.references, _ = d.base.authenticate()
                cls.report = d.report(cls.result, cls.arrays, cls.references)

    def test_exact_nine_control_census(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.base.parent.CONTROLS))
        self.assertEqual(self.report["retained_L23_failures"], 9)
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})

    def test_live_rank_and_gap_independent_oracle(self):
        refs = {"binary64": self.references["logits_binary64"],
                "fp16": self.references["logits_fp16"].view("<f2")}
        for row in self.report["controls"]:
            values_by_kind = {
                **refs, "actual_fp16": self.arrays[row["control"]]["logits"].view("<f2")}
            for key, values in values_by_kind.items():
                order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
                ranks = {index: rank for rank, index in enumerate(order, 1)}
                summary = row["summaries"][key]
                self.assertEqual(summary["top_k_ids_diagnostic_only"], order[:10])
                self.assertEqual(summary["highest_excluded"]["token_id"], order[10])
                gap = Fraction(float(values[order[9]])) - Fraction(float(values[order[10]]))
                self.assertEqual(Fraction(summary["cutoff_gap"]), gap)
                for position in row["rank_positions"]:
                    self.assertEqual(position[key]["rank"], ranks[position["token_id"]])

    def test_live_overlap_preserved(self):
        for row in self.report["controls"]:
            self.assertEqual(row["comparisons"]["binary64"]["overlap_count"], 10)
            self.assertTrue(row["comparisons"]["binary64"]["same_order"])
            self.assertEqual(row["comparisons"]["fp16"]["overlap_count"], 9)
            self.assertFalse(row["comparisons"]["fp16"]["same_order"])

    def test_live_reversal_fraction_oracle(self):
        reference = self.references["logits_fp16"].view("<f2")
        for row in self.report["controls"]:
            actual = self.arrays[row["control"]]["logits"].view("<f2")
            for pair in row["comparisons"]["fp16"]["cutoff_reversals"]:
                a, b = pair["actual_only_id"], pair["reference_only_id"]
                da = Fraction(float(actual[a])) - Fraction(float(reference[a]))
                db = Fraction(float(actual[b])) - Fraction(float(reference[b]))
                margin = Fraction(float(reference[b])) - Fraction(float(reference[a]))
                self.assertEqual(Fraction(pair["relative_value_change"]), da - db)
                self.assertEqual(Fraction(pair["reference_preference_margin"]), margin)
                self.assertEqual(Fraction(pair["actual_preference_margin"]), da - db - margin)

    def test_live_differing_ids_and_margins(self):
        for row in self.report["controls"]:
            tops = [set(summary["top_k_ids_diagnostic_only"])
                    for summary in row["summaries"].values()]
            expected = set.union(*tops) - set.intersection(*tops)
            self.assertEqual({entry["token_id"] for entry in row["differing_token_diagnostics"]}, expected)
            for entry in row["differing_token_diagnostics"]:
                self.assertIs(type(entry["token_id"]), int)
                for key in row["summaries"]:
                    p, s = entry[key], row["summaries"][key]
                    value = Fraction(float.fromhex(p["value_hex"]))
                    self.assertEqual(Fraction(p["margin_to_lowest_included"]),
                                     value - Fraction(float.fromhex(s["lowest_included"]["value_hex"])))
                    self.assertEqual(Fraction(p["margin_to_highest_excluded"]),
                                     value - Fraction(float.fromhex(s["highest_excluded"]["value_hex"])))

    def test_adjacent_finite_reversal(self):
        actual, ref = vector([9, 5, 4.875, 0]), vector([9, 4.75, 5, 0])
        row = d.cutoff_control(actual, actual, ref, 2)
        self.assertTrue(row["explanation"]["finite_near_boundary_explained"])
        self.assertEqual(row["explanation"]["classification"], "FINITE_ADJACENT_CUTOFF_REVERSAL")
        pair = row["comparisons"]["fp16"]["cutoff_reversals"][0]
        self.assertEqual(pair["relative_value_change"], "3/8")
        self.assertEqual(pair["reference_preference_margin"], "1/4")
        self.assertEqual(pair["actual_preference_margin"], "1/8")

    def test_distant_reversal_not_near_boundary(self):
        actual, ref = vector([9, 6, 5, 4]), vector([9, 0, 5, 4])
        row = d.cutoff_control(actual, actual, ref, 2)
        self.assertFalse(row["explanation"]["finite_near_boundary_explained"])
        self.assertEqual(row["explanation"]["classification"], "FINITE_REVERSAL_NOT_ADJACENT_TO_CUTOFF")

    def test_tie_break_disclosed(self):
        actual, ref = vector([9, 5, 5, 0]), vector([9, 4.75, 5, 0])
        row = d.cutoff_control(actual, actual, ref, 2)
        self.assertEqual(row["summaries"]["actual_fp16"]["top_k_ids_diagnostic_only"], [0, 1])
        self.assertTrue(row["summaries"]["actual_fp16"]["cutoff_tie"])
        self.assertFalse(row["explanation"]["finite_near_boundary_explained"])
        self.assertEqual(row["explanation"]["classification"], "FINITE_REVERSAL_WITH_TIE_BREAK")

    def test_signed_zero_tie_and_negative_values(self):
        p = d.profile(vector([-1, -0.0, 0.0, -2]), 2)
        self.assertEqual(p["summary"]["top_k_ids_diagnostic_only"], [1, 2])
        self.assertEqual(p["summary"]["cutoff_gap"], "1")
        p = d.profile(vector([-0.0, 0.0, -1]), 1)
        self.assertTrue(p["summary"]["cutoff_tie"])

    def test_no_membership_change_not_explanation(self):
        values = vector([4, 3, 2, 1])
        row = d.cutoff_control(values, values, values, 2)
        self.assertEqual(row["differing_token_diagnostics"], [])
        self.assertFalse(row["explanation"]["finite_near_boundary_explained"])

    def test_order_change_distinct_from_overlap(self):
        row = d.cutoff_control(vector([4, 3, 1]), vector([3, 4, 1]), vector([3, 4, 1]), 2)
        self.assertEqual(row["comparisons"]["fp16"]["overlap_count"], 2)
        self.assertFalse(row["comparisons"]["fp16"]["same_order"])
        self.assertEqual(row["comparisons"]["fp16"]["cutoff_reversals"], [])

    def test_multiple_exchanges(self):
        row = d.cutoff_control(vector([5, 4, 3, 2, 1]), vector([5, 4, 3, 2, 1]),
                               vector([1, 2, 4, 5, 0]), 2)
        self.assertEqual(len(row["comparisons"]["fp16"]["cutoff_reversals"]), 4)
        self.assertFalse(row["explanation"]["finite_near_boundary_explained"])

    def test_nearest_opposite_membership(self):
        p = d.profile(vector([7, 6, 4, 1]), 2)
        for index, neighbor, margin in ((0, 2, "3"), (1, 2, "2"), (2, 1, "-2"), (3, 1, "-5")):
            actual = d.token_position(index, p, 2)["nearest_opposite_membership"]
            self.assertEqual(actual["token_id"], neighbor)
            self.assertEqual(actual["signed_margin"], margin)

    def test_exact_gap_no_subtraction_overflow(self):
        p = d.profile(vector([np.finfo(np.float64).max, -np.finfo(np.float64).max]), 1)
        self.assertEqual(Fraction(p["summary"]["cutoff_gap"]),
                         2 * Fraction(float(np.finfo(np.float64).max)))
        p = d.profile(vector([np.nextafter(0.0, 1.0), 0.0]), 1)
        self.assertEqual(Fraction(p["summary"]["cutoff_gap"]), Fraction(1, 2 ** 1074))

    def test_nonfinite_schema_and_k_refused(self):
        for values in (vector([1, float("nan")]), vector([1, float("inf")]),
                       vector([1, -float("inf")]), np.array([2, 1], dtype="<f2"),
                       vector([[2, 1]])):
            with self.assertRaises(ValueError):
                d.profile(values, 1)
        for k in (0, -1, 2, True, 1.5):
            with self.assertRaises(ValueError):
                d.profile(vector([2, 1]), k)
        with self.assertRaises(ValueError):
            d.cutoff_control(vector([3, 2, 1]), vector([2, 1]), vector([2, 1]), 1)

    def test_retained_history_and_flags(self):
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            l23 = row["retained_L23_and_ancestral_lineage"]
            for history in (l23, l23["retained_L21"], l23["retained_L22"]):
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(history[flag], False)
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(self.report["original_global_reference"],
                         self.result["preflight"]["final_reference"])

    def test_history_and_gate_mutations_refused(self):
        for key, value in (("candidate_admitted", True),
                           ("source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("S18_failure_indices", [])):
            result = deepcopy(self.result)
            result["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)

    def test_threshold_mutation_refused(self):
        result = deepcopy(self.result)
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)

    def test_result_array_and_authenticator_hash_refused(self):
        for pin in (d.base.PINS["result"], self.result["controls"][0]["arrays"]["logits"],
                    d.AUTHENTICATOR_PIN):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_array_nonfinite_and_trailing_bytes_refused(self):
        for array, suffix in ((np.array([0x7c00], dtype="<u2"), b""),
                              (np.array([0], dtype="<u2"), b"x")):
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            with self.assertRaises(ValueError):
                d.base.parent.checked_array(stream.getvalue() + suffix, (1,), "<u2")

    def test_artifact_census_refused(self):
        names = [d.Path(pin["path"]).name for pin in self.result["artifacts"]] + ["result.json"]
        for changed in (names[:-1], names + ["unexpected.npy"]):
            with self.assertRaises(ValueError):
                d.base.artifact_manifest(self.result, changed)

    def test_terminal_reviewer_mutation_refused(self):
        review = json.loads(d.base.read_bound(d.base.PINS["review"]))
        mission = json.loads(d.base.read_bound(d.base.PINS["mission"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.base.PINS["review"]["path"]},
                  "mission": {"path": d.base.PINS["mission"]["path"]}}
        backlog = [{"id": d.base.MISSION, "status": "done", "outcome": {"review_status": "done"},
                    "finished_ts": review["created_at"]}]
        for changed in ({**review, "producer_role": "engineer"},
                        {**review, "review": {"status": "pending"}}):
            with self.assertRaises(ValueError):
                d.base.terminal_review(changed, latest, backlog, mission)
        with self.assertRaises(ValueError):
            d.base.terminal_review(review, latest, [{**backlog[0], "status": "running"}], mission)

    def test_dispatch_and_write_guards(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.base.parent.rmsnorm(None, None),
            lambda: d.base.parent.logits(None, None),
            lambda: d.base.parent.execute(None),
            lambda: d.base.parent.norm.rmsnorm(None, None),
            lambda: open(d.SOURCE, "w"),
            lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE),
        ]
        with d.base.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_json_only_stdout(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"status": "fixture"}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(json.loads(stream.getvalue()), {"status": "fixture"})
        self.assertEqual(stream.getvalue().count("\n"), 1)

    def test_cli_forbidden_modes_refused(self):
        for args in ([], ["--execute"], ["--check", "--out", "build/new"],
                     ["--check", "--execute"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_report_json_and_boundary(self):
        json.dumps(self.report, allow_nan=False)
        for key in ("evidence_writes", "final_rmsnorm_invocations", "lm_head_invocations",
                    "native_layer_invocations", "decoder_invocations", "tokenizer_decode_invocations"):
            self.assertEqual(d.FLAGS[key], 0)
        self.assertFalse(d.FLAGS["token_published"])
        self.assertFalse(d.FLAGS["token_selected_for_feedback"])
        self.assertFalse(d.FLAGS["upstream_cause_identified"])
        self.assertIn("Normal independent Reviewer", d.BOUNDARY)
        self.assertIn("REQUIRED", d.BOUNDARY)


if __name__ == "__main__":
    unittest.main()
