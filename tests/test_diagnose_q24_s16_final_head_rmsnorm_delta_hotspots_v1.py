"""Independent scalar oracles and retained-input/read-only boundary regressions."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_rmsnorm_delta_hotspots_v1 as d


EVIDENCE = None


def rational(value):
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def oracle_entries(values, field, limit=8, key="signed_delta"):
    if field == "largest_positive":
        order = sorted((i for i, v in enumerate(values) if v > 0), key=lambda i: (-values[i], i))
    elif field == "most_negative":
        order = sorted((i for i, v in enumerate(values) if v < 0), key=lambda i: (values[i], i))
    else:
        order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    return [{"coordinate": i, key: str(values[i])} for i in order[:limit]]


class RMSNormDeltaHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is not None:
            evidence = EVIDENCE
        else:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        cls.result, cls.arrays, cls.refs, cls.geometry, cls.weights, cls.report = evidence

    def reference(self, name):
        return self.refs["rmsnorm_fp16"].view("<f2") if name == "fp16" else self.refs["rmsnorm_binary64"]

    def deltas(self, label, branch):
        return [rational(a) - rational(r) for a, r in zip(
            self.arrays[label]["rmsnorm"].view("<f2"), self.reference(branch), strict=True)]

    def test_full_live_signed_rankings(self):
        for row in self.report["controls"]:
            for branch, summary in row["rmsnorm_delta_hotspots"].items():
                values = self.deltas(row["control"], branch)
                self.assertEqual(summary["full_absolute_signed_ranking"],
                                 oracle_entries(values, "largest_absolute_signed", 896))
                for field in d.RANK_FIELDS:
                    self.assertEqual(summary[field], oracle_entries(values, field))

    def test_exact_sums_remainders_and_zero_counts(self):
        for row in self.report["controls"]:
            for branch, summary in row["rmsnorm_delta_hotspots"].items():
                values = self.deltas(row["control"], branch)
                chosen = [entry["coordinate"] for entry in summary["largest_absolute_signed"]]
                self.assertEqual(Fraction(summary["exact_sum"]), sum(values))
                self.assertEqual(Fraction(summary["ranked_signed_sum"]), sum(values[i] for i in chosen))
                self.assertEqual(Fraction(summary["unranked_exact_remainder"]),
                                 sum(v for i, v in enumerate(values) if i not in chosen))
                self.assertEqual(Fraction(summary["exact_sum_of_absolute_contributions"]), sum(map(abs, values)))
                self.assertEqual(summary["zero_coordinate_count"], values.count(Fraction()))
                self.assertEqual(summary["coordinate_count"], 896)

    def test_coordinate62_and_forced_exact_values(self):
        for row in self.report["controls"]:
            for branch, summary in row["rmsnorm_delta_hotspots"].items():
                forced = {entry["coordinate"]: entry for entry in summary["forced_coordinates"]}
                self.assertIn("required_coordinate_62", forced[62]["reasons"])
                values = self.deltas(row["control"], branch)
                order = sorted(range(896), key=lambda i: (-abs(values[i]), i))
                for i, entry in forced.items():
                    self.assertEqual(Fraction(entry["signed_delta"]), values[i])
                    self.assertEqual(entry["absolute_rank"], order.index(i) + 1)
                    self.assertEqual(Fraction(entry["actual_exact"]),
                                     rational(self.arrays[row["control"]]["rmsnorm"].view("<f2")[i]))
                    self.assertEqual(Fraction(entry["reference_exact"]), rational(self.reference(branch)[i]))
                    self.assertEqual(Fraction(entry["actual_exact"]) - Fraction(entry["reference_exact"]), values[i])
                    self.assertEqual(entry["in_largest_absolute"], i in order[:8])

    def test_live_pair_coordinate_products_independent_oracle(self):
        for row in self.report["controls"]:
            actual = self.arrays[row["control"]]["rmsnorm"].view("<f2")
            hidden = {"actual_fp16": actual, **{name: self.reference(name) for name in d.BRANCHES}}
            for pair in row["selected_pair_coordinate_alignment"]:
                l, r = pair["left_id"], pair["right_id"]
                weights = [rational(a) - rational(b) for a, b in zip(self.weights[l], self.weights[r], strict=True)]
                for branch, vector in hidden.items():
                    values = [rational(h) * w for h, w in zip(vector, weights, strict=True)]
                    summary = pair["branches"][branch]
                    self.assertEqual(Fraction(summary["exact_sum"]), sum(values))
                    for field in d.RANK_FIELDS:
                        self.assertEqual(summary[field], oracle_entries(values, field, key="signed_contribution"))
                for branch in d.BRANCHES:
                    values = [v * w for v, w in zip(self.deltas(row["control"], branch), weights, strict=True)]
                    summary = pair["actual_minus_independent_reference"][branch]
                    self.assertEqual(Fraction(summary["exact_sum"]), sum(values))
                    for field in d.RANK_FIELDS:
                        self.assertEqual(summary[field], oracle_entries(values, field, key="signed_contribution"))
                    self.assertEqual(Fraction(summary["retained_margin_change"]),
                                     sum(values) + Fraction(summary["boundary_remainder_change"]))

    def test_retained_pair_margin_accounting(self):
        for row in self.report["controls"]:
            a = self.arrays[row["control"]]["logits"].view("<f2")
            for pair in row["selected_pair_coordinate_alignment"]:
                l, r = pair["left_id"], pair["right_id"]
                for name in d.BRANCHES:
                    ref = self.refs["logits_" + name]
                    if name == "fp16":
                        ref = ref.view("<f2")
                    summary = pair["actual_minus_independent_reference"][name]
                    self.assertEqual(Fraction(summary["retained_actual_margin"]), rational(a[l]) - rational(a[r]))
                    self.assertEqual(Fraction(summary["retained_reference_margin"]), rational(ref[l]) - rational(ref[r]))

    def test_all_cutoff_exchange_and_signed_hotspots_forced(self):
        for row, geometry in zip(self.report["controls"], self.geometry["controls"], strict=True):
            pairs = row["selected_pair_coordinate_alignment"]
            self.assertEqual({(p["left_id"], p["right_id"]): p["roles"] for p in pairs},
                             d.contributions.pairs_for(geometry))
            for branch in d.BRANCHES:
                forced = {entry["coordinate"] for entry in row["rmsnorm_delta_hotspots"][branch]["forced_coordinates"]}
                for pair in pairs:
                    for kind in ("branches", "actual_minus_independent_reference"):
                        for summary in pair[kind].values():
                            for field in d.RANK_FIELDS:
                                self.assertTrue({entry["coordinate"] for entry in summary[field]} <= forced)

    def test_pair_hotspot_alignment(self):
        for row in self.report["controls"]:
            for name in d.BRANCHES:
                hidden = set(d.indices(row["rmsnorm_delta_hotspots"][name]))
                for pair in row["selected_pair_coordinate_alignment"]:
                    effect = pair["actual_minus_independent_reference"][name]
                    self.assertEqual(effect["hidden_delta_hotspot_overlap"]["shared_coordinates"],
                                     sorted(hidden & set(d.indices(effect))))
                    for branch, summary in pair["branches"].items():
                        self.assertEqual(effect["branch_contribution_hotspot_overlap"][branch]["shared_coordinates"],
                                         sorted(hidden & set(d.indices(summary))))

    def test_all_cross_control_and_branch_overlaps(self):
        rows = {row["control"]: row for row in self.report["controls"]}
        cross = self.report["cross_control_hotspots"]
        self.assertEqual(cross["pair_count"], 36)
        self.assertEqual(len({(p["left"], p["right"]) for p in cross["pairs"]}), 36)
        for pair in cross["pairs"]:
            for name in d.BRANCHES:
                left = set(d.indices(rows[pair["left"]]["rmsnorm_delta_hotspots"][name]))
                right = set(d.indices(rows[pair["right"]]["rmsnorm_delta_hotspots"][name]))
                self.assertEqual(pair["per_reference"][name]["shared_coordinates"], sorted(left & right))
                self.assertEqual(pair["per_reference"][name]["jaccard"], str(Fraction(len(left & right), len(left | right))))
        for row in rows.values():
            a, b = (set(d.indices(row["rmsnorm_delta_hotspots"][name])) for name in d.BRANCHES)
            self.assertEqual(row["reference_branch_hotspot_overlap"]["shared_coordinates"], sorted(a & b))

    def test_byte_identical_groups_preserve_all_histories(self):
        for kind, groups in self.report["artifact_equivalence_classes"].items():
            self.assertEqual(sorted(g["control_count"] for g in groups), [1, 8])
            self.assertEqual(sorted(label for g in groups for label in g["controls"]), sorted(d.parent.CONTROLS))
            for group in groups:
                first = self.arrays[group["controls"][0]]
                for label in group["controls"]:
                    for array_kind in (("rmsnorm",) if kind == "rmsnorm" else ("rmsnorm", "logits")):
                        self.assertEqual(first[array_kind].tobytes(), self.arrays[label][array_kind].tobytes())
                self.assertTrue(group["byte_identity_and_delta_rankings_verified"])

    def test_exact_nine_failure_histories_and_lineages(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual([row["control"] for row in self.report["controls"]], list(d.parent.CONTROLS))
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertEqual(set(row["retained_failures"]), {"L21", "L22", "L23"})
            self.assertTrue(all(row["retained_failures"].values()))
            history = row["retained_L23_and_ancestral_lineage"]
            for item in (history, history["retained_L21"], history["retained_L22"]):
                self.assertEqual(item["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(item[flag], False)
        lineage = self.report["lineage_separation"]
        for flag in ("old_adjusted_closure_certifies_new_parents", "old_states_substituted_or_propagated",
                     "old_sparse_cut_replayed_or_revalidated"):
            self.assertIs(lineage[flag], False)
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])

    def test_failure_gate_threshold_and_control_mutations_refused(self):
        for key, value in (("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("S18_failure_indices", [])):
            result = deepcopy(self.result)
            result["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(self.result)
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)
        arrays = dict(self.arrays)
        arrays.pop(next(iter(arrays)))
        with self.assertRaises(ValueError):
            d.report(self.result, arrays, self.refs, self.geometry, self.weights)

    def test_independent_reference_authority(self):
        self.assertFalse(np.array_equal(self.reference("fp16"), self.reference("binary64").astype("<f2")))
        self.assertEqual(self.report["original_global_reference"], self.result["preflight"]["final_reference"])
        final = self.report["original_global_reference"]["reference"]
        for name in d.BRANCHES:
            self.assertEqual(final["input_" + name], self.result["preflight"]["L23_original_reference"]["reference"][name])

    def test_pinned_source_test_result_tampering_refused(self):
        for pin in (*d.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_terminal_independent_reviewer_gate(self):
        review = json.loads(d.base.read_bound(d.base.PINS["review"]))
        mission = json.loads(d.base.read_bound(d.base.PINS["mission"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.base.PINS["review"]["path"]},
                  "mission": {"path": d.base.PINS["mission"]["path"]}}
        backlog = [{"id": d.base.MISSION, "status": "done", "outcome": {"review_status": "done"},
                    "finished_ts": review["created_at"]}]
        for changed in ({**review, "producer_role": "engineer"}, {**review, "review": {"status": "pending"}}):
            with self.assertRaises(ValueError):
                d.base.terminal_review(changed, latest, backlog, mission)

    def test_live_asset_and_final_authority_splices_refused(self):
        retained = (self.result, self.arrays, self.refs, [])
        assets = self.result["preflight"]["assets"]
        preflight = d.contributions.preflight
        with patch.object(d.base, "authenticate", return_value=retained):
            with patch.object(preflight, "bind_assets", return_value={}), self.assertRaises(ValueError):
                d.authenticate()
            with patch.object(preflight, "bind_assets", return_value=assets):
                with patch.object(preflight, "bind_final_reference", return_value={}), self.assertRaises(ValueError):
                    d.authenticate()

    def test_invalid_vectors_and_shape_refused(self):
        for value in (np.array([np.nan]), np.array([np.inf]), np.array([1], dtype="<f4"),
                      np.zeros((1, 2)), np.array([], dtype="<f8")):
            with self.assertRaises(ValueError):
                d.delta_summary(value, value, {}, 1)
        with self.assertRaises(ValueError):
            d.delta_summary(np.zeros(8), np.zeros(9), {}, 8)

    def test_signed_ties_cancellation_and_subnormal_oracle(self):
        actual = np.array([2, -2, 0, 1, -1, 2 ** -24, -0.0, 0], dtype="<f2")
        reference = np.zeros(8, dtype="<f8")
        reference[7] = np.nextafter(0.0, 1.0)
        summary = d.delta_summary(actual, reference, {}, 2)
        values = [rational(a) - rational(b) for a, b in zip(actual, reference, strict=True)]
        self.assertEqual(summary["largest_absolute_signed"], oracle_entries(values, "largest_absolute_signed", 2))
        self.assertEqual(Fraction(summary["exact_sum"]), Fraction(1, 2 ** 24) - Fraction(1, 2 ** 1074))
        self.assertEqual(summary["ranked_signed_sum"], "0")
        self.assertEqual(summary["largest_positive"][0]["coordinate"], 0)
        self.assertEqual(summary["most_negative"][0]["coordinate"], 1)

    def test_force_coordinate62_outside_top_ranked(self):
        actual, ref = np.zeros(896, dtype="<f2"), np.zeros(896, dtype="<f8")
        actual[:8] = 2
        actual[62] = -1
        summary = d.delta_summary(actual, ref, {62: ["required_coordinate_62"]})
        entry = summary["forced_coordinates"][0]
        self.assertEqual(entry["coordinate"], 62)
        self.assertEqual(entry["signed_delta"], "-1")
        self.assertEqual(entry["absolute_rank"], 9)
        self.assertFalse(entry["in_largest_absolute"])

    def test_zero_vectors_and_invalid_ranking_inputs(self):
        summary = d.delta_summary(np.zeros(8, dtype="<f2"), np.zeros(8), {}, 8)
        self.assertEqual(summary["zero_coordinate_count"], 8)
        self.assertEqual(summary["largest_positive"], [])
        self.assertEqual(summary["most_negative"], [])
        self.assertEqual(summary["exact_sum"], "0")
        for forced in ({-1: ["bad"]}, {8: ["bad"]}, {True: ["bad"]}, {1: []}, []):
            with self.assertRaises(ValueError):
                d.delta_summary(np.zeros(8), np.zeros(8), forced, 8)
        for limit in (0, -1, True, 9, 1.5):
            with self.assertRaises(ValueError):
                d.delta_summary(np.zeros(8), np.zeros(8), {}, limit)

    def test_overlap_disjoint_partial_and_order(self):
        self.assertEqual(d.overlap([1, 2], [3, 4])["classification"], "DISJOINT_HOTSPOTS")
        self.assertEqual(d.overlap([1, 2], [2, 3])["jaccard"], "1/3")
        self.assertEqual(d.overlap([1, 2], [2, 3])["classification"], "PARTIAL_HOTSPOT_OVERLAP")
        self.assertEqual(d.overlap([1, 2], [2, 1])["classification"], "IDENTICAL_HOTSPOT_SET")
        self.assertFalse(d.overlap([1, 2], [2, 1])["same_order"])

    def test_dispatch_write_prior_check_and_head_guards(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.contributions.analyze_pair(None), lambda: d.contributions.dyadic_dot(None, None),
            lambda: d.contributions.measure(), lambda: d.contributions.check(), lambda: d.cutoff.check(),
            lambda: d.base.check(), lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
            lambda: open(d.SOURCE, "w"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE),
            lambda: d.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_single_json_document(self):
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

    def test_json_claims_flags_and_honest_compute_count(self):
        json.dumps(self.report, allow_nan=False)
        count = sum(5 * len(row["selected_pair_coordinate_alignment"]) for row in self.report["controls"])
        self.assertEqual(self.report["coordinate_product_vectors"], count)
        self.assertGreater(count, 0)
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
