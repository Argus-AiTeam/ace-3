"""Independent scalar Fraction oracles and read-only contribution boundary tests."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_coordinate_contributions_v1 as d


EVIDENCE = None


class CoordinateContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is not None:
            evidence = EVIDENCE
        else:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        cls.result, cls.arrays, cls.refs, cls.geometry, cls.weights, cls.report = evidence

    def test_exact_nine_histories(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"][stage] for stage in ("L21", "L22", "L23")))

    def test_live_all_coordinate_sums_and_rankings_independent_oracle(self):
        for row in self.report["controls"]:
            hidden = {"actual_fp16": self.arrays[row["control"]]["rmsnorm"].view("<f2"),
                      "fp16": self.refs["rmsnorm_fp16"].view("<f2"),
                      "binary64": self.refs["rmsnorm_binary64"]}
            for pair in row["coordinate_pairs"]:
                l, r = pair["left_id"], pair["right_id"]
                for name, x in hidden.items():
                    left = [Fraction(float(a)) * Fraction(float(b))
                            for a, b in zip(x, self.weights[l], strict=True)]
                    right = [Fraction(float(a)) * Fraction(float(b))
                             for a, b in zip(x, self.weights[r], strict=True)]
                    values = [a - b for a, b in zip(left, right, strict=True)]
                    branch = pair["branches"][name]
                    self.assertEqual(Fraction(branch["exact_sum"]), sum(left) - sum(right))
                    self.assertEqual(Fraction(branch["independent_left_dot"]), sum(left))
                    self.assertEqual(Fraction(branch["independent_right_dot"]), sum(right))
                    order = sorted(range(896), key=lambda i: (-abs(values[i]), i))[:d.RANK_LIMIT]
                    self.assertEqual(branch["largest_absolute_signed"],
                                     [{"coordinate": i, "signed_contribution": str(values[i])}
                                      for i in order])
                    self.assertEqual(Fraction(branch["ranked_signed_sum"])
                                     + Fraction(branch["unranked_exact_remainder"]), sum(values))

    def test_live_coordinate_changes_and_boundary_accounting(self):
        for row in self.report["controls"]:
            x = self.arrays[row["control"]]["rmsnorm"].view("<f2")
            for pair in row["coordinate_pairs"]:
                l, r = pair["left_id"], pair["right_id"]
                for name, ref in (("fp16", self.refs["rmsnorm_fp16"].view("<f2")),
                                  ("binary64", self.refs["rmsnorm_binary64"])):
                    values = [(Fraction(float(a)) - Fraction(float(b)))
                              * (Fraction(float(wl)) - Fraction(float(wr)))
                              for a, b, wl, wr in zip(x, ref, self.weights[l], self.weights[r], strict=True)]
                    effect = pair["actual_minus_independent_reference"][name]
                    self.assertEqual(Fraction(effect["exact_sum"]), sum(values))
                    for entry in effect["largest_absolute_signed"]:
                        self.assertEqual(Fraction(entry["signed_contribution"]), values[entry["coordinate"]])
                    self.assertEqual(Fraction(effect["retained_actual_margin"]),
                                     Fraction(effect["reference_retained_margin"]) + sum(values)
                                     + Fraction(effect["boundary_remainder_change"]))
                    self.assertEqual(Fraction(effect["margin_after_ranked_coordinate_changes"]),
                                     Fraction(effect["reference_retained_margin"])
                                     + Fraction(effect["ranked_signed_sum"]))
                    self.assertEqual(Fraction(effect["margin_after_all_coordinate_changes"]),
                                     Fraction(effect["reference_retained_margin"]) + sum(values))

    def test_live_retained_logit_margins(self):
        for row in self.report["controls"]:
            logits = {"actual_fp16": self.arrays[row["control"]]["logits"].view("<f2"),
                      "fp16": self.refs["logits_fp16"].view("<f2"),
                      "binary64": self.refs["logits_binary64"]}
            for pair in row["coordinate_pairs"]:
                for name, values in logits.items():
                    margin = Fraction(float(values[pair["left_id"]])) - Fraction(float(values[pair["right_id"]]))
                    branch = pair["branches"][name]
                    self.assertEqual(Fraction(branch["retained_margin"]), margin)
                    self.assertEqual(Fraction(branch["exact_sum"])
                                     + Fraction(branch["head_accumulation_rounding_remainder"]), margin)

    def test_exchanged_and_cutoff_coverage(self):
        for row, geometry in zip(self.report["controls"], self.geometry["controls"], strict=True):
            pairs = {(p["left_id"], p["right_id"]): p["roles"] for p in row["coordinate_pairs"]}
            for name, summary in geometry["summaries"].items():
                self.assertIn(name + "_cutoff_neighbors",
                              pairs[(summary["lowest_included"]["token_id"],
                                     summary["highest_excluded"]["token_id"])])
            for name, comparison in geometry["comparisons"].items():
                for pair in comparison["cutoff_reversals"]:
                    self.assertIn("exchanged_vs_" + name,
                                  pairs[(pair["actual_only_id"], pair["reference_only_id"])])
            for entry in geometry["differing_token_diagnostics"]:
                for name in geometry["summaries"]:
                    self.assertIn((entry["token_id"],
                                   entry[name]["nearest_opposite_membership"]["token_id"]), pairs)

    def test_signed_ranking_ties_cancellation(self):
        result = d.ranked(list(map(Fraction, [2, -2, 0, 1, -1])), 2)
        self.assertEqual(result["largest_absolute_signed"],
                         [{"coordinate": 0, "signed_contribution": "2"},
                          {"coordinate": 1, "signed_contribution": "-2"}])
        self.assertEqual(result["exact_sum"], "0")
        self.assertEqual(result["unranked_exact_remainder"], "0")
        self.assertEqual(result["exact_sum_of_absolute_contributions"], "6")
        self.assertEqual(result["largest_positive"][0]["coordinate"], 0)
        self.assertEqual(result["most_negative"][0]["coordinate"], 1)

    def test_zero_contributions(self):
        result = d.ranked([Fraction(0)] * 8)
        self.assertEqual(result["zero_coordinate_count"], 8)
        self.assertEqual(result["largest_positive"], [])
        self.assertEqual(result["most_negative"], [])
        self.assertEqual(result["exact_sum"], "0")

    def test_dyadic_dot_subnormal_and_cancellation(self):
        a = np.array([np.nextafter(0.0, 1.0), 1, -1], dtype="<f8")
        w = np.array([2, 3, 3], dtype="<f2")
        self.assertEqual(d.dyadic_dot(a, w), Fraction(1, 2 ** 1073))
        a = np.array([2 ** -24, -0.0], dtype="<f2")
        self.assertEqual(d.dyadic_dot(a, a), Fraction(1, 2 ** 48))

    def test_invalid_vectors_refused(self):
        for value in (np.array([np.nan]), np.array([np.inf]), np.array([1], dtype="<f4"),
                      np.zeros((1, 2)), np.array([], dtype="<f8")):
            with self.assertRaises(ValueError):
                d.exact_vector(value)

    def test_invalid_rankings_refused(self):
        for limit in (0, -1, True, 9, 1.5):
            with self.assertRaises(ValueError):
                d.ranked([Fraction(1)] * 8, limit)
        with self.assertRaises(ValueError):
            d.ranked([1.0] * 8)

    def test_independent_references_not_casts(self):
        self.assertFalse(np.array_equal(self.refs["rmsnorm_fp16"].view("<f2"),
                                        self.refs["rmsnorm_binary64"].astype("<f2")))
        self.assertEqual(self.report["original_global_reference"], self.result["preflight"]["final_reference"])

    def test_actual_logit_tamper_refused(self):
        row = self.report["controls"][0]
        arrays = {key: dict(value) for key, value in self.arrays.items()}
        arrays[row["control"]]["logits"] = arrays[row["control"]]["logits"].copy()
        arrays[row["control"]]["logits"][row["coordinate_pairs"][0]["left_id"]] ^= 1
        with self.assertRaisesRegex(ValueError, "reproduce retained actual logit"):
            d.report(self.result, arrays, self.refs, self.geometry, self.weights)

    def test_row_dot_mismatch_refused(self):
        with patch.object(d, "dyadic_dot", return_value=Fraction(0)), self.assertRaises(ValueError):
            d.report(self.result, self.arrays, self.refs, self.geometry, self.weights)

    def test_control_census_refused(self):
        arrays = dict(self.arrays)
        arrays.pop(next(iter(arrays)))
        with self.assertRaises(ValueError):
            d.report(self.result, arrays, self.refs, self.geometry, self.weights)

    def test_history_and_threshold_mutations_refused(self):
        for key, value in (("candidate_admitted", True),
                           ("source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("S18_failure_indices", [])):
            result = deepcopy(self.result)
            result["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(self.result)
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)

    def test_source_result_and_test_pins_refused(self):
        for pin in (*d.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_terminal_reviewer_required(self):
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

    def test_dispatch_and_write_guards(self):
        audit = {"forbidden_calls": 0}
        calls = [lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
                 lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
                 lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
                 lambda: open(d.SOURCE, "w"), lambda: os.open(d.SOURCE, os.O_WRONLY),
                 lambda: os.unlink(d.SOURCE),
                 lambda: d.preflight.parent.producer.legacy.torch.cuda._lazy_init()]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_single_json_stdout(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"control_count": 9}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(json.loads(stream.getvalue()), {"control_count": 9})
        self.assertEqual(stream.getvalue().count("\n"), 1)

    def test_cli_forbidden_modes_refused(self):
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_non_admission_and_lineage_separation(self):
        for row in self.report["controls"]:
            l23 = row["retained_L23_and_ancestral_lineage"]
            for history in (l23, l23["retained_L21"], l23["retained_L22"]):
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(history[flag], False)
        self.assertFalse(self.report["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])
        self.assertFalse(self.report["lineage_separation"]["old_states_substituted_or_propagated"])
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])

    def test_numeric_ids_only_and_local_compute_count(self):
        count = 0
        for row in self.report["controls"]:
            for pair in row["coordinate_pairs"]:
                self.assertIs(type(pair["left_id"]), int)
                self.assertIs(type(pair["right_id"]), int)
                self.assertEqual(set(pair["branches"]), {"actual_fp16", "fp16", "binary64"})
                count += 6
        self.assertEqual(self.report["local_exact_row_dot_computations"], count)
        self.assertGreater(count, 0)

    def test_row_id_bounds_refused(self):
        for ids in (set(), {-1}, {151936}, {True}):
            with self.assertRaises(ValueError):
                d.load_rows({}, ids)

    def test_report_serialization_and_claim_boundary(self):
        json.dumps(self.report, allow_nan=False)
        for name in ("native_dispatch", "decoder_dispatch", "prefix_dispatch",
                     "admission_dispatch", "RTL_dispatch", "GPU_dispatch", "hardware_dispatch",
                     "reference_producer_dispatch", "evidence_writes", "tokenizer_decode_invocations"):
            self.assertEqual(d.FLAGS[name], 0)
        for name in ("token_published", "token_selected_for_feedback", "upstream_cause_identified",
                     "reference_reanchored", "full_model_claim"):
            self.assertIs(d.FLAGS[name], False)
        self.assertIn("independent Reviewer validation REQUIRED", d.BOUNDARY)


if __name__ == "__main__":
    unittest.main()
