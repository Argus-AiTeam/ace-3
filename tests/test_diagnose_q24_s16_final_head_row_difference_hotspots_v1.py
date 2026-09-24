"""Independent bit-level row and scalar contribution oracles; no operator replay."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_row_difference_hotspots_v1 as d


EVIDENCE = None


def fp16(word):
    sign = -1 if word & 0x8000 else 1
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite oracle input")
    if exponent == 0:
        return sign * Fraction(fraction, 1 << 24)
    return sign * Fraction(1024 + fraction) * Fraction(2) ** (exponent - 25)


def exact(value):
    return Fraction(*float(value).as_integer_ratio())


def order(values, field, limit=8):
    if field == "largest_positive":
        indices = sorted((i for i, v in enumerate(values) if v > 0), key=lambda i: (-values[i], i))
    elif field == "most_negative":
        indices = sorted((i for i, v in enumerate(values) if v < 0), key=lambda i: (values[i], i))
    else:
        indices = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    return indices[:limit]


class RowDifferenceHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        else:
            evidence = EVIDENCE
        cls.result, cls.arrays, cls.refs, cls.geometry, cls.weights, cls.alignment, cls.report = evidence
        cls.row_oracles = {
            key: [fp16(int(word)) for word in row.view("<u2")] for key, row in cls.weights.items()}

    def pairs(self):
        for row in self.report["controls"]:
            for pair in row["row_difference_pairs"]:
                yield row, pair

    def differences(self, pair):
        return [a - b for a, b in zip(self.row_oracles[pair["left_id"]],
                                      self.row_oracles[pair["right_id"]], strict=True)]

    def vectors(self, label):
        return {
            "actual_fp16": [fp16(int(w)) for w in self.arrays[label]["rmsnorm"]],
            "fp16": [fp16(int(w)) for w in self.refs["rmsnorm_fp16"]],
            "binary64": [exact(v) for v in self.refs["rmsnorm_binary64"]],
        }

    def test_all_896_row_differences_bit_oracle(self):
        for _, pair in self.pairs():
            values = self.differences(pair)
            summary = pair["row_difference_hotspots"]
            entries = summary["full_absolute_signed_ranking"]
            self.assertEqual([e["coordinate"] for e in entries], order(values, "", 896))
            for entry in entries:
                i = entry["coordinate"]
                self.assertEqual(Fraction(entry["signed_row_delta"]), values[i])
                self.assertEqual(Fraction(entry["absolute_magnitude"]), abs(values[i]))
                self.assertEqual(Fraction(entry["left_exact"]), self.row_oracles[pair["left_id"]][i])
                self.assertEqual(Fraction(entry["right_exact"]), self.row_oracles[pair["right_id"]][i])

    def test_signed_top_rankings(self):
        for _, pair in self.pairs():
            values = self.differences(pair)
            for field in d.RANK_FIELDS:
                self.assertEqual([e["coordinate"] for e in pair["row_difference_hotspots"][field]],
                                 order(values, field))

    def test_exact_sums_and_remainders(self):
        for _, pair in self.pairs():
            values = self.differences(pair)
            summary = pair["row_difference_hotspots"]
            self.assertEqual(Fraction(summary["exact_sum"]), sum(values))
            self.assertEqual(Fraction(summary["ranked_signed_sum"]), sum(values[i] for i in order(values, "")))
            self.assertEqual(Fraction(summary["exact_sum"]), Fraction(summary["ranked_signed_sum"])
                             + Fraction(summary["unranked_exact_remainder"]))
            self.assertEqual(Fraction(summary["exact_sum_of_absolute_row_deltas"]), sum(map(abs, values)))
            self.assertEqual(summary["zero_coordinate_count"], values.count(Fraction()))

    def test_coordinate62_and_forced_ranks(self):
        for _, pair in self.pairs():
            values = self.differences(pair)
            ranking = order(values, "", 896)
            forced = {e["coordinate"]: e for e in pair["row_difference_hotspots"]["forced_coordinates"]}
            self.assertIn("required_coordinate_62", forced[62]["reasons"])
            for i, entry in forced.items():
                self.assertEqual(entry["absolute_rank"], ranking.index(i) + 1)
                self.assertEqual(entry["in_largest_absolute"], i in ranking[:8])
                self.assertEqual(Fraction(entry["signed_row_delta"]), values[i])

    def test_all_interpretive_hotspots_forced(self):
        for row, pair in self.pairs():
            forced = {e["coordinate"] for e in pair["row_difference_hotspots"]["forced_coordinates"]}
            summaries = [*row["rmsnorm_delta_hotspots"].values(), *pair["branches"].values(),
                         *pair["actual_minus_independent_reference"].values()]
            for summary in summaries:
                for field in d.RANK_FIELDS:
                    self.assertTrue({e["coordinate"] for e in summary[field]} <= forced)

    def test_branch_products_and_rankings_independent_oracle(self):
        for row, pair in self.pairs():
            values = self.differences(pair)
            for name, hidden in self.vectors(row["control"]).items():
                products = [h * w for h, w in zip(hidden, values, strict=True)]
                summary = pair["branches"][name]
                self.assertEqual(Fraction(summary["exact_sum"]), sum(products))
                for field in d.RANK_FIELDS:
                    self.assertEqual(summary[field], [
                        {"coordinate": i, "signed_contribution": str(products[i])} for i in order(products, field)])

    def test_delta_products_expanded_independent_oracle(self):
        for row, pair in self.pairs():
            vectors = self.vectors(row["control"])
            for name in d.BRANCHES:
                products = [(a * l - a * r) - (ref * l - ref * r)
                            for a, ref, l, r in zip(vectors["actual_fp16"], vectors[name],
                                                   self.row_oracles[pair["left_id"]],
                                                   self.row_oracles[pair["right_id"]], strict=True)]
                summary = pair["actual_minus_independent_reference"][name]
                self.assertEqual(Fraction(summary["exact_sum"]), sum(products))
                for field in d.RANK_FIELDS:
                    self.assertEqual(summary[field], [
                        {"coordinate": i, "signed_contribution": str(products[i])} for i in order(products, field)])
                self.assertEqual(Fraction(summary["retained_margin_change"]),
                                 sum(products) + Fraction(summary["boundary_remainder_change"]))

    def test_forced_coordinate_relationships(self):
        for row, pair in self.pairs():
            vectors = self.vectors(row["control"])
            values = self.differences(pair)
            multipliers = {"branches": vectors, "actual_minus_independent_reference": {
                name: [a - r for a, r in zip(vectors["actual_fp16"], vectors[name], strict=True)]
                for name in d.BRANCHES}}
            for entry in pair["row_difference_hotspots"]["forced_coordinates"]:
                i = entry["coordinate"]
                for kind, branches in multipliers.items():
                    for name, vector in branches.items():
                        relationship = entry["coordinate_relationships"][kind][name]
                        self.assertEqual(Fraction(relationship["multiplier"]), vector[i])
                        self.assertEqual(Fraction(relationship["signed_contribution"]), vector[i] * values[i])

    def test_overlap_and_signed_interpretation(self):
        for row, pair in self.pairs():
            values = self.differences(pair)
            vectors = self.vectors(row["control"])
            multipliers = {"branches": vectors, "actual_minus_independent_reference": {
                name: [a - r for a, r in zip(vectors["actual_fp16"], vectors[name], strict=True)]
                for name in d.BRANCHES}}
            for kind, branches in multipliers.items():
                for name, vector in branches.items():
                    products = [h * w for h, w in zip(vector, values, strict=True)]
                    relation = pair["contribution_relationships"][kind][name]
                    matches = []
                    for field in d.RANK_FIELDS:
                        left, right = order(values, field), order(products, field)
                        overlap = relation["row_vs_contribution_hotspots"][field]
                        self.assertEqual(overlap["shared_coordinates"], sorted(set(left) & set(right)))
                        self.assertEqual(overlap["same_order"], left == right)
                        matches.append(left == right)
                    self.assertEqual(relation["top_rankings_match_row_geometry"], all(matches))
                    self.assertEqual(relation["row_sign_reversed_coordinates"],
                                     [i for i, (v, w) in enumerate(zip(vector, values)) if w and v < 0])
                    self.assertEqual(relation["nonzero_row_erased_coordinates"],
                                     [i for i, (v, w) in enumerate(zip(vector, values)) if w and v == 0])
                    self.assertEqual(relation["row_sign_preserved_coordinates"],
                                     [i for i, (v, w) in enumerate(zip(vector, values)) if w and v > 0])

    def test_row_vs_hidden_delta_overlap(self):
        for row, pair in self.pairs():
            for name in d.BRANCHES:
                for field in d.RANK_FIELDS:
                    a = {e["coordinate"] for e in pair["row_difference_hotspots"][field]}
                    b = {e["coordinate"] for e in row["rmsnorm_delta_hotspots"][name][field]}
                    overlap = pair["row_vs_rmsnorm_delta_hotspots"][name][field]
                    self.assertEqual(overlap["shared_coordinates"], sorted(a & b))
                    self.assertEqual(overlap["jaccard"], str(Fraction(len(a & b), len(a | b))) if a | b else "1")

    def test_exact_existing_pair_selection(self):
        for row, geometry in zip(self.report["controls"], self.geometry["controls"], strict=True):
            self.assertEqual({(p["left_id"], p["right_id"]): p["roles"] for p in row["row_difference_pairs"]},
                             d.contributions.pairs_for(geometry))
            self.assertEqual(len(row["row_difference_pairs"]), 3)
        self.assertEqual(self.report["selected_pair_count"], 27)
        self.assertEqual(self.report["coordinate_product_vectors"], 135)

    def test_nine_histories_thresholds_and_separate_lineages(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
            history = row["retained_L23_and_ancestral_lineage"]
            for item in (history, history["retained_L21"], history["retained_L22"]):
                self.assertEqual(item["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(item[flag], False)
        for key, value in self.report["lineage_separation"].items():
            if key.startswith("old_"):
                self.assertIs(value, False)

    def test_independent_original_reference_identity(self):
        self.assertEqual(self.report["original_global_reference"], self.result["preflight"]["final_reference"])
        final = self.report["original_global_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_" + branch],
                             self.result["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertFalse(np.array_equal(self.refs["rmsnorm_fp16"].view("<f2"),
                                       self.refs["rmsnorm_binary64"].astype("<f2")))

    def test_retained_history_and_threshold_mutations_refused(self):
        for key, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            result = deepcopy(self.result)
            result["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(self.result)
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)

    def test_source_test_result_pins_refuse_tampering(self):
        for pin in (*d.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_terminal_review_gate(self):
        review = json.loads(d.base.read_bound(d.base.PINS["review"]))
        mission = json.loads(d.base.read_bound(d.base.PINS["mission"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.base.PINS["review"]["path"]},
                  "mission": {"path": d.base.PINS["mission"]["path"]}}
        backlog = [{"id": d.base.MISSION, "status": "done", "outcome": {"review_status": "done"},
                    "finished_ts": review["created_at"]}]
        for changed in ({**review, "producer_role": "engineer"}, {**review, "review": {"status": "pending"}}):
            with self.assertRaises(ValueError):
                d.base.terminal_review(changed, latest, backlog, mission)

    def test_tokenizer_tied_head_and_reference_splices_refused(self):
        preflight = d.contributions.preflight
        assets = self.result["preflight"]["assets"]
        with patch.object(d.base, "authenticate", return_value=(self.result, self.arrays, self.refs, [])):
            with patch.object(preflight, "bind_assets", return_value={}), self.assertRaises(ValueError):
                d.authenticate()
            with patch.object(preflight, "bind_assets", return_value=assets):
                with patch.object(preflight, "bind_final_reference", return_value={}), self.assertRaises(ValueError):
                    d.authenticate()

    def test_invalid_rows_and_forced_inputs(self):
        zero = np.zeros(896, dtype="<f2")
        for row in (np.zeros(895, dtype="<f2"), np.zeros(896, dtype="<f8"),
                    np.full(896, np.inf, dtype="<f2"), np.full(896, np.nan, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.row_summary(row, zero, {})
        for forced in ({-1: ["bad"]}, {896: ["bad"]}, {True: ["bad"]}, {1: []}, []):
            with self.assertRaises(ValueError):
                d.row_summary(zero, zero, forced)
        for limit in (0, -1, True, 897, 1.5):
            with self.assertRaises(ValueError):
                d.row_summary(zero, zero, {}, limit)

    def test_signed_ties_subnormal_zero_and_forced62(self):
        left, right = np.zeros(896, dtype="<f2"), np.zeros(896, dtype="<f2")
        left[:8] = [2, -2, 2, -2, 2, -2, 2, -2]
        left[62] = -1
        left[100] = 2 ** -24
        summary, values, _, _ = d.row_summary(left, right, {62: ["required_coordinate_62"]})
        self.assertEqual([e["coordinate"] for e in summary["largest_absolute_signed"]], list(range(8)))
        self.assertEqual(summary["ranked_signed_sum"], "0")
        self.assertEqual(Fraction(summary["exact_sum"]), -1 + Fraction(1, 2 ** 24))
        self.assertEqual(values[100], fp16(1))
        entry = summary["forced_coordinates"][0]
        self.assertEqual(entry["absolute_rank"], 9)
        self.assertFalse(entry["in_largest_absolute"])
        zero, _, _, _ = d.row_summary(right, right, {})
        self.assertEqual(zero["zero_coordinate_count"], 896)
        self.assertEqual(zero["largest_positive"], [])
        self.assertEqual(zero["most_negative"], [])

    def test_geometry_agreement_and_disagreement_are_distinct(self):
        values = [Fraction(896 - i) * (-1 if i % 2 else 1) for i in range(896)]
        right = [Fraction()] * 896
        for multiplier, matches in (([Fraction(2)] * 896, True), ([Fraction(-2)] * 896, False),
                                    ([Fraction()] * 896, False)):
            summary = d.contributions.ranked([h * v for h, v in zip(multiplier, values)])
            relation = d.relationship(values, multiplier, values, right, summary)
            self.assertEqual(relation["top_rankings_match_row_geometry"], matches)
            self.assertIn("coincide" if matches else "does not reproduce", relation["interpretation"])
        summary["exact_sum"] = "1"
        with self.assertRaises(ValueError):
            d.relationship(values, multiplier, values, right, summary)
        with self.assertRaises(ValueError):
            d.relationship(values, multiplier, right, values, d.contributions.ranked(values))

    def test_control_and_pair_splices_refused(self):
        arrays = dict(self.arrays)
        arrays.pop(next(iter(arrays)))
        with self.assertRaises(ValueError):
            d.report(self.result, arrays, self.refs, self.geometry, self.weights, self.alignment)
        aligned = deepcopy(self.alignment)
        aligned["controls"][0]["selected_pair_coordinate_alignment"][0]["left_id"] = 0
        with self.assertRaises(ValueError):
            d.report(self.result, self.arrays, self.refs, self.geometry, self.weights, aligned)

    def test_forbidden_dispatch_and_writes_intercepted(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.contributions.analyze_pair(None), lambda: d.contributions.dyadic_dot(None, None),
            lambda: d.hotspots.measure(), lambda: d.hotspots.check(), lambda: d.contributions.check(),
            lambda: d.cutoff.check(), lambda: d.base.check(),
            lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
            lambda: open(d.SOURCE, "w"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE),
            lambda: d.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_single_json_and_forbidden_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"control_count": 9}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(json.loads(stream.getvalue()), {"control_count": 9})
        self.assertEqual(stream.getvalue().count("\n"), 1)
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_json_flags_and_non_admission_boundary(self):
        json.dumps(self.report, allow_nan=False)
        for name in ("native_dispatch", "decoder_dispatch", "prefix_dispatch", "admission_dispatch",
                     "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "simulation_dispatch",
                     "reference_producer_dispatch", "evidence_writes", "tokenizer_decode_invocations",
                     "final_rmsnorm_invocations", "lm_head_invocations", "selected_row_head_replay",
                     "local_exact_row_dot_computations"):
            self.assertEqual(d.FLAGS[name], 0)
        for name in ("token_published", "token_selected_for_feedback", "upstream_cause_identified",
                     "reference_reanchored", "full_model_claim", "full_vocabulary_head_recomputation"):
            self.assertIs(d.FLAGS[name], False)
        self.assertIn("independent Reviewer validation REQUIRED", d.BOUNDARY)


if __name__ == "__main__":
    unittest.main()
