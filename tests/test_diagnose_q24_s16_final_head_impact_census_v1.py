"""Independent small-array oracles and fail-closed retained-census checks."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_impact_census_v1 as d


EVIDENCE = None


class FinalHeadImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is not None:
            cls.result, cls.arrays, cls.references, cls.report = EVIDENCE
        else:
            with d.read_only({"forbidden_calls": 0}):
                cls.result, cls.arrays, cls.references, _ = d.authenticate()
                cls.report = d.census(cls.result, cls.arrays, cls.references)

    def test_exact_census(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual(self.report["arrays_verified"], 18)
        self.assertEqual(self.report["cross_control"]["pair_count"], 36)

    def test_equivalence_independent_bytes(self):
        for kind, _ in d.KINDS:
            groups = {}
            for label in d.parent.CONTROLS:
                groups.setdefault(self.arrays[label][kind].tobytes(), []).append(label)
            self.assertEqual(list(groups.values()), [
                group["controls"] for group in self.report["artifact_equivalence_classes"][kind]])
            self.assertEqual(len(groups), 2)

    def test_metrics_independent_counts(self):
        for row in self.report["controls"]:
            for kind, _ in d.KINDS:
                actual = self.arrays[row["control"]][kind]
                metric = row["metrics"][kind]
                self.assertEqual(metric["reviewed_fp16_mismatch_count"], int(
                    np.count_nonzero(actual != self.references[kind + "_fp16"])))
                self.assertEqual(metric["binary64_mismatch_count"], int(
                    np.count_nonzero(actual.view("<f2") != self.references[kind + "_binary64"])))

    def test_metrics_fraction_oracle(self):
        a = np.array([1, -2, 0, -0.0], dtype="<f2").view("<u2")
        b = np.array([1.125, -2.5, 0, 0], dtype="<f8")
        f = np.array([1, -1, -0.0, 0], dtype="<f2").view("<u2")
        result = d.metric(a, b, f)
        exact = [abs(Fraction(float(x)) - Fraction(float(y)))
                 for x, y in zip(a.view("<f2"), b, strict=True)]
        self.assertEqual(Fraction(result["binary64_max_absolute_error"]), max(exact))
        self.assertEqual(result["binary64_worst_index"], exact.index(max(exact)))
        self.assertEqual(result["reviewed_fp16_mismatch_count"], 3)
        self.assertEqual(result["binary64_mismatch_count"], 2)
        self.assertEqual(result["reviewed_fp16_max_absolute_error"], "1")

    def test_zero_error_and_tie(self):
        words = np.array([0, -0.0, 1], dtype="<f2").view("<u2")
        result = d.metric(words, words.view("<f2").astype("<f8"), words)
        self.assertEqual(result["binary64_max_absolute_error"], "0")
        self.assertEqual(result["binary64_worst_index"], 0)
        self.assertEqual(result["reviewed_fp16_mismatch_count"], 0)

    def test_topk_independent_sort(self):
        for row in self.report["controls"]:
            actual = self.arrays[row["control"]]["logits"].view("<f2")
            ids = sorted(range(len(actual)), key=lambda i: (-float(actual[i]), i))[:10]
            for key in ("binary64", "fp16"):
                ref = self.references["logits_" + key]
                values = ref.view("<f2") if key == "fp16" else ref
                expected = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))[:10]
                self.assertEqual(row["top_k"][key], {
                    "same_order": ids == expected, "overlap_count": len(set(ids) & set(expected))})

    def test_overlap_not_order(self):
        self.assertEqual(d.overlap([1, 2], [2, 1]), {"overlap_count": 2, "same_order": False})
        self.assertEqual(d.overlap([1, 2], [2, 3]), {"overlap_count": 1, "same_order": False})

    def test_pairwise_independent(self):
        for pair in self.report["cross_control"]["pairs"]:
            for kind, _ in d.KINDS:
                a, b = (self.arrays[pair[key]][kind] for key in ("left", "right"))
                self.assertEqual(pair[kind]["mismatch_count"], int(np.count_nonzero(a != b)))
                error = np.abs(a.view("<f2").astype("<f8") - b.view("<f2").astype("<f8"))
                self.assertEqual(Fraction(pair[kind]["max_absolute_error"]),
                                 Fraction(float(error.max())))
                self.assertEqual(pair[kind]["worst_index"], int(error.argmax()))

    def test_extrema_exact(self):
        for kind, _ in d.KINDS:
            for prefix in ("binary64", "reviewed_fp16"):
                values = [Fraction(row["metrics"][kind][prefix + "_max_absolute_error"])
                          for row in self.report["controls"]]
                result = self.report["extrema"][kind]["errors"][prefix]
                self.assertEqual(Fraction(result["max_absolute_error"]), max(values))
                self.assertEqual(Fraction(result["min_max_absolute_error"]), min(values))

    def test_retained_failures(self):
        d.check_history(self.result)
        self.assertEqual(self.report["retained_L23_failures"], 9)
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertEqual([f["index"] for f in row["retained_failures"]["L23"]],
                             [62] if row["control"] == "mapped_all" else [62, 241])

    def test_history_mutation_refused(self):
        for stage in ("retained_L21", "retained_L22"):
            result = deepcopy(self.result)
            result["controls"][0]["parent"]["retained_L23"][stage]["candidate_admitted"] = True
            with self.assertRaises(ValueError):
                d.check_history(result)

    def test_threshold_mutation_refused(self):
        result = deepcopy(self.result)
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.check_history(result)

    def test_control_reordering_refused(self):
        result = deepcopy(self.result)
        result["controls"].reverse()
        with self.assertRaises(ValueError):
            d.check_history(result)

    def test_terminal_review_mutations_refused(self):
        review = json.loads(d.read_bound(d.PINS["review"]))
        mission = json.loads(d.read_bound(d.PINS["mission"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.PINS["review"]["path"]},
                  "mission": {"path": d.PINS["mission"]["path"]}}
        backlog = [{"id": d.MISSION, "status": "done", "outcome": {"review_status": "done"},
                    "finished_ts": review["created_at"]}]
        d.terminal_review(review, latest, backlog, mission)
        for key, value in (("producer_role", "engineer"), ("review", {"status": "pending"})):
            with self.assertRaises(ValueError):
                d.terminal_review({**review, key: value}, latest, backlog, mission)
        with self.assertRaises(ValueError):
            d.terminal_review(review, latest, [{**backlog[0], "status": "running"}], mission)
        with self.assertRaises(ValueError):
            d.terminal_review(review, {**latest, "handoff": {"path": "other"}}, backlog, mission)

    def test_artifact_census_refusals(self):
        names = [Path(pin["path"]).name for pin in self.result["artifacts"]] + ["result.json"]
        d.artifact_manifest(self.result, reversed(names))
        for changed in (names[:-1], names + ["failure.json"]):
            with self.assertRaises(ValueError):
                d.artifact_manifest(self.result, changed)
        result = deepcopy(self.result)
        result["artifacts"][-1] = result["artifacts"][0]
        with self.assertRaises(ValueError):
            d.artifact_manifest(result, names)

    def test_hash_mutation_refused(self):
        pin = {**d.PINS["result"], "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            d.read_bound(pin)

    def test_size_mutation_refused(self):
        pin = {**d.PINS["result"], "bytes": 1}
        with self.assertRaises(ValueError):
            d.read_bound(pin)

    def test_array_schema_refusals(self):
        for array in (np.ones(2, dtype="<f8"),
                      np.array([0x7c00, 0], dtype="<u2")):
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            with self.assertRaises(ValueError):
                d.parent.checked_array(stream.getvalue(), (2,), "<u2")
        stream = io.BytesIO()
        np.save(stream, np.ones(2, dtype="<u2"), allow_pickle=False)
        with self.assertRaises(ValueError):
            d.parent.checked_array(stream.getvalue() + b"x", (2,), "<u2")

    def test_comparison_mutation_refused(self):
        result = deepcopy(self.result)
        result["controls"][0]["comparisons"]["logits"]["binary64_worst_index"] = -1
        with self.assertRaises(ValueError):
            d.census(result, self.arrays, self.references)

    def test_operator_dispatch_refused(self):
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for name in ("rmsnorm", "logits", "execute", "validate", "load_operands", "authenticate"):
                with self.assertRaises(RuntimeError):
                    getattr(d.parent, name)()
        self.assertEqual(audit["forbidden_calls"], 6)

    def test_external_dispatch_refused(self):
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            with self.assertRaises(RuntimeError):
                d.parent.subprocess.run(["false"])
            with self.assertRaises(RuntimeError):
                os.system("false")
        self.assertEqual(audit["forbidden_calls"], 2)

    def test_evidence_writes_refused(self):
        audit = {"forbidden_calls": 0}
        target = d.parent.OUTPUT / "result.json"
        with d.read_only(audit):
            for operation in (lambda: target.open("wb"), lambda: open(target, "a"),
                              lambda: os.open(target, os.O_WRONLY),
                              lambda: target.unlink(), lambda: target.mkdir(),
                              lambda: target.rename(target)):
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], 6)

    def test_json_and_nonadmission(self):
        json.dumps(self.report, allow_nan=False)
        self.assertTrue(all(value is False if isinstance(value, bool) else value == 0
                            for value in d.FLAGS.values()))
        self.assertEqual(self.report["retained_flags"], d.parent.FLAGS)
        self.assertIn("Q24 state is wider than FP16", d.BOUNDARY)
        self.assertIn("REQUIRED", d.BOUNDARY)

    def test_no_execute_or_output_interface(self):
        for args in ([], ["--execute"], ["--check", "--out", "build/other"],
                     ["--check", "--validate"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(args)


if __name__ == "__main__":
    unittest.main()
