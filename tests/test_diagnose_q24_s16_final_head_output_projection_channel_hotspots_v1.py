"""Independent bit-level and scalar oracles for retained tied-row channel accounting."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_output_projection_channel_hotspots_v1 as d


EVIDENCE = None


def fp16(word):
    sign = -1 if word & 0x8000 else 1
    exponent, fraction = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite bit oracle input")
    if exponent == 0:
        return sign * Fraction(fraction, 1 << 24)
    return sign * Fraction(1024 + fraction) * Fraction(2) ** (exponent - 25)


def vector(array):
    if array.dtype.str == "<f8":
        return [Fraction(*float(value).as_integer_ratio()) for value in array]
    return [fp16(int(word)) for word in array.view("<u2")]


def ordered(values, field="", limit=896):
    if field == "largest_positive":
        return sorted((i for i, v in enumerate(values) if v > 0), key=lambda i: (-values[i], i))[:limit]
    if field == "most_negative":
        return sorted((i for i, v in enumerate(values) if v < 0), key=lambda i: (values[i], i))[:limit]
    return sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))[:limit]


class ChannelHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        else:
            evidence = EVIDENCE
        cls.result, cls.arrays, cls.refs, cls.selected, cls.weights, cls.report = evidence
        cls.oracles = {}
        for row in cls.report["controls"]:
            actual = vector(cls.arrays[row["control"]]["rmsnorm"])
            for name, branch in row["branches"].items():
                reference, weights = vector(cls.refs["rmsnorm_" + name]), vector(cls.weights[branch["numeric_id"]])
                cls.oracles[row["control"], name] = [
                    a * w - r * w for a, r, w in zip(actual, reference, weights, strict=True)]

    def branches(self):
        for row in self.report["controls"]:
            for name, branch in row["branches"].items():
                yield row["control"], name, branch

    def test_worst_rows_full_vocabulary_independent_oracle(self):
        references = {name: vector(self.refs["logits_" + name]) for name in d.BRANCHES}
        seen = {}
        for label, name, branch in self.branches():
            words = self.arrays[label]["logits"]
            key = (words.tobytes(), name)
            if key not in seen:
                errors = [abs(a - r) for a, r in zip(vector(words), references[name], strict=True)]
                seen[key] = (errors.index(max(errors)), max(errors))
            index, error = seen[key]
            self.assertEqual(branch["numeric_id"], index)
            self.assertEqual(Fraction(branch["absolute_residual"]), error)

    def test_all_column_products_bit_oracle(self):
        for label, name, branch in self.branches():
            actual, reference = vector(self.arrays[label]["rmsnorm"]), vector(self.refs["rmsnorm_" + name])
            weights = vector(self.weights[branch["numeric_id"]])
            for entry in branch["channels"]["full_absolute_signed_ranking"]:
                i = entry["coordinate"]
                self.assertEqual(Fraction(entry["signed_contribution"]), self.oracles[label, name][i])
                self.assertEqual(Fraction(entry["input_delta"]), actual[i] - reference[i])
                self.assertEqual(Fraction(entry["column_weight"]), weights[i])

    def test_deterministic_complete_ranking(self):
        for label, name, branch in self.branches():
            self.assertEqual([e["coordinate"] for e in branch["channels"]["full_absolute_signed_ranking"]],
                             ordered(self.oracles[label, name]))

    def test_signed_top_eight_rankings(self):
        for label, name, branch in self.branches():
            for field in d.hotspots.RANK_FIELDS:
                self.assertEqual([e["coordinate"] for e in branch["channels"][field]],
                                 ordered(self.oracles[label, name], field, 8))

    def test_exact_sums_and_unranked_remainders(self):
        for label, name, branch in self.branches():
            values, summary = self.oracles[label, name], branch["channels"]
            chosen = sum(values[i] for i in ordered(values, limit=8))
            self.assertEqual(Fraction(summary["exact_sum"]), sum(values))
            self.assertEqual(Fraction(summary["ranked_signed_sum"]), chosen)
            self.assertEqual(Fraction(summary["unranked_exact_remainder"]), sum(values) - chosen)
            self.assertEqual(Fraction(summary["exact_sum_of_absolute_contributions"]), sum(map(abs, values)))
            self.assertEqual(summary["zero_coordinate_count"], values.count(Fraction()))

    def test_retained_residual_and_boundary_remainder(self):
        for label, name, branch in self.branches():
            i = branch["numeric_id"]
            actual = fp16(int(self.arrays[label]["logits"][i]))
            reference = (fp16(int(self.refs["logits_fp16"][i])) if name == "fp16"
                         else Fraction(*float(self.refs["logits_binary64"][i]).as_integer_ratio()))
            residual = actual - reference
            self.assertEqual(Fraction(branch["retained_signed_residual"]), residual)
            self.assertEqual(residual, sum(self.oracles[label, name]) + Fraction(branch["boundary_remainder"]))

    def test_coordinate62_always_reported(self):
        for label, name, branch in self.branches():
            values = self.oracles[label, name]
            entry = branch["channels"]["coordinate62"]
            self.assertEqual(entry["coordinate"], 62)
            self.assertEqual(entry["absolute_rank"], ordered(values).index(62) + 1)
            self.assertEqual(entry["in_largest_absolute"], 62 in ordered(values, limit=8))
            self.assertEqual(Fraction(entry["signed_contribution"]), values[62])

    def test_bounded_counts_and_control_order(self):
        self.assertEqual([row["control"] for row in self.report["controls"]], list(d.parent.CONTROLS))
        self.assertEqual(self.report["selected_row_count"], 18)
        self.assertEqual(self.report["coordinate_product_count"], 16128)
        ids = {branch["numeric_id"] for _, _, branch in self.branches()}
        self.assertEqual(self.report["unique_numeric_rows"], sorted(ids))
        self.assertLessEqual(len(ids), 18)
        for row in self.report["controls"]:
            self.assertEqual(list(row["branches"]), list(d.BRANCHES))

    def test_json_schema_and_serialization(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        self.assertEqual(json.loads(json.dumps(self.report, allow_nan=False)), self.report)

    def test_schema_rejects_count_type_and_field_mutations(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for key, value in (("control_count", 10), ("selected_row_count", 19),
                           ("coordinate_product_count", 0), ("controls", []), ("extra", 1)):
            bad = {**self.report, key: value}
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(bad)
        for value in ([], [{"coordinate": True}], [{"coordinate": 896}], None):
            bad = deepcopy(self.report)
            bad["controls"][0]["branches"]["fp16"]["channels"]["full_absolute_signed_ranking"] = value
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(bad)

    def test_preserved_histories_thresholds_and_lineages(self):
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertEqual(self.report["retained_flags"], self.result["flags"])
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
            history = row["retained_L23_and_ancestral_lineage"]
            for stage in (history, history["retained_L21"], history["retained_L22"]):
                self.assertEqual(stage["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(stage[flag], False)
        for key, value in self.report["lineage_separation"].items():
            if key.startswith("old_"):
                self.assertIs(value, False)

    def test_original_reference_not_reanchored(self):
        final = self.report["original_global_reference"]
        self.assertEqual(final, self.result["preflight"]["final_reference"])
        for name in d.BRANCHES:
            self.assertEqual(final["reference"]["input_" + name],
                             self.result["preflight"]["L23_original_reference"]["reference"][name])
        self.assertFalse(np.array_equal(self.refs["rmsnorm_fp16"].view("<f2"),
                                       self.refs["rmsnorm_binary64"].astype("<f2")))

    def test_history_and_gate_mutations_refused(self):
        for key, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.selection(bad, self.arrays, self.refs)
        bad = deepcopy(self.result)
        bad["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.selection(bad, self.arrays, self.refs)

    def test_source_and_input_pin_tampering_refused(self):
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

    def test_asset_and_reference_splices_refused(self):
        preflight = d.contributions.preflight
        with patch.object(d.base, "authenticate", return_value=(self.result, self.arrays, self.refs, [])):
            with patch.object(preflight, "bind_assets", return_value={}), self.assertRaises(ValueError):
                d.hotspots.authenticate()
            with patch.object(preflight, "bind_assets", return_value=self.result["preflight"]["assets"]):
                with patch.object(preflight, "bind_final_reference", return_value={}), self.assertRaises(ValueError):
                    d.hotspots.authenticate()

    def test_invalid_column_and_logit_operands(self):
        zero = np.zeros(896, dtype="<f2")
        for bad in (zero[:-1], zero.astype("<f8"), np.full(896, np.inf, dtype="<f2"),
                    np.full(896, np.nan, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.channel_summary(bad, zero, zero)
        logits = np.zeros(d.VOCABULARY, dtype="<f2")
        for bad in (logits[:-1], logits.astype("<f8"), np.full(d.VOCABULARY, np.nan, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.worst_row(bad, logits)

    def test_channel_ties_signs_zeros_and_subnormal(self):
        actual, reference, weights = (np.zeros(896, dtype="<f2"), np.zeros(896, dtype="<f2"),
                                      np.ones(896, dtype="<f2"))
        actual[:8] = [2, -2] * 4
        actual[62], actual[100] = -1, 2 ** -24
        summary = d.channel_summary(actual, reference, weights)
        self.assertEqual([e["coordinate"] for e in summary["largest_absolute_signed"]], list(range(8)))
        self.assertEqual(summary["coordinate62"]["absolute_rank"], 9)
        self.assertEqual(Fraction(summary["exact_sum"]), -1 + Fraction(1, 2 ** 24))
        weights[:8] = -1
        reversed_summary = d.channel_summary(actual, reference, weights)
        self.assertEqual(reversed_summary["largest_positive"][0]["coordinate"], 1)
        zero = d.channel_summary(reference, reference, weights)
        self.assertEqual(zero["zero_coordinate_count"], 896)
        self.assertEqual(zero["largest_positive"], [])
        self.assertEqual(zero["most_negative"], [])

    def test_row_ties_use_exact_residual_then_numeric_id(self):
        actual = np.zeros(d.VOCABULARY, dtype="<f2")
        reference = np.zeros(d.VOCABULARY, dtype="<f8")
        actual[1:3] = 1
        reference[1], reference[2] = 2 ** -55, -(2 ** -55)
        self.assertEqual(d.worst_row(actual, reference), (2, 1 + Fraction(1, 2 ** 55)))
        reference[1] = reference[2]
        self.assertEqual(d.worst_row(actual, reference)[0], 1)

    def test_control_branch_and_weight_splices_refused(self):
        arrays = dict(self.arrays)
        arrays.pop(next(iter(arrays)))
        with self.assertRaises(ValueError):
            d.selection(self.result, arrays, self.refs)
        weights = dict(self.weights)
        weights.pop(next(iter(weights)))
        with self.assertRaises(ValueError):
            d.report(self.result, self.arrays, self.refs, self.selected, weights)
        selected = deepcopy(self.selected)
        selected[next(iter(selected))] = {}
        with self.assertRaises(ValueError):
            d.report(self.result, self.arrays, self.refs, selected, self.weights)
        selected = deepcopy(self.selected)
        branch = selected[next(iter(selected))]
        index, residual = branch["fp16"]
        branch["fp16"] = (index, residual + 1)
        with self.assertRaises(ValueError):
            d.report(self.result, self.arrays, self.refs, selected, self.weights)

    def test_forbidden_dispatch_and_evidence_writes_intercepted(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.contributions.dyadic_dot(None, None), lambda: d.contributions.check(),
            lambda: d.hotspots.check(), lambda: d.hotspots.measure(), lambda: d.hotspots.report(None),
            lambda: d.hotspots.cutoff.check(), lambda: d.base.check(),
            lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
            lambda: open(d.SOURCE, "w"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE), lambda: d.parent.write(None, None),
            lambda: d.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        for flag in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                     "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "selected_row_head_replay"):
            self.assertEqual(d.FLAGS[flag], 0)
        self.assertIn("or ACE2 changes", d.BOUNDARY)
        self.assertIn("independent Reviewer validation REQUIRED", d.BOUNDARY)

    def test_cli_single_json_and_forbidden_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value=self.report), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(json.loads(stream.getvalue()), self.report)
        self.assertEqual(stream.getvalue().count("\n"), 1)
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
