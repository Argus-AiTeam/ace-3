"""Independent FP16-bit and binary64-rational oracles for tied-head complements."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_output_projection_unselected_channel_complement_v1 as d


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


def synthetic(actual, reference=None, weights=None, boundary=Fraction(3, 7)):
    reference = np.zeros(d.WIDTH, dtype="<f2") if reference is None else reference
    weights = np.ones(d.WIDTH, dtype="<f2") if weights is None else weights
    summary = d.channels.channel_summary(actual, reference, weights)
    residual = Fraction(summary["exact_sum"]) + boundary
    return {
        "numeric_id": 0, "absolute_residual": str(abs(residual)),
        "retained_signed_residual": str(residual),
        "retained_actual_hex": float(residual).hex(), "retained_reference_hex": "0x0.0p+0",
        "boundary_remainder": str(boundary), "exact_residual_identity": True,
        "channels": summary,
    }


class UnselectedChannelComplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        else:
            evidence = EVIDENCE
        cls.evidence, cls.report = evidence
        cls.result, cls.arrays, cls.refs, cls.selected, cls.weights, cls.inherited = cls.evidence
        cls.oracles = {}
        for row in cls.inherited["controls"]:
            actual = vector(cls.arrays[row["control"]]["rmsnorm"])
            for name, account in row["branches"].items():
                reference = vector(cls.refs["rmsnorm_" + name])
                weights = vector(cls.weights[account["numeric_id"]])
                cls.oracles[row["control"], name] = [
                    a * w - r * w for a, r, w in zip(actual, reference, weights, strict=True)]

    def accounts(self):
        for row in self.report["controls"]:
            for name, account in row["branches"].items():
                yield row["control"], name, account, account["unselected_input_channels"]

    def test_inherited_report_preserved_without_mutation(self):
        restored = deepcopy(self.report)
        restored.pop("unselected_input_coordinate_count")
        restored.pop("complement_group_count")
        for row in restored["controls"]:
            for account in row["branches"].values():
                account.pop("unselected_input_channels")
        self.assertEqual(restored, self.inherited)
        self.assertNotIn("unselected_input_coordinate_count", self.inherited)

    def test_original_row_and_reference_selection_preserved(self):
        for label, name, account, _ in self.accounts():
            index, residual = self.selected[label][name]
            self.assertEqual((account["numeric_id"], Fraction(account["retained_signed_residual"])),
                             (index, residual))
            self.assertEqual(index, 48298 if name == "fp16" else 20806)

    def test_selected_coordinates_independent_bit_oracle(self):
        for label, name, _, split in self.accounts():
            values = self.oracles[label, name]
            ordered = sorted(range(d.WIDTH), key=lambda i: (-abs(values[i]), i))[:d.LIMIT]
            self.assertEqual(split["excluded_ranked_coordinates"], ordered)

    def test_group_signed_sums_independent_oracle(self):
        for label, name, _, split in self.accounts():
            values, excluded = self.oracles[label, name], set(split["excluded_ranked_coordinates"])
            for group in split["groups_in_input_order"]:
                indices = range(group["input_group"] * 128, (group["input_group"] + 1) * 128)
                self.assertEqual(Fraction(group["signed_contribution"]),
                                 sum((values[i] for i in indices if i not in excluded), Fraction()))

    def test_group_absolute_sums_independent_oracle(self):
        for label, name, _, split in self.accounts():
            values, excluded = self.oracles[label, name], set(split["excluded_ranked_coordinates"])
            for group in split["groups_in_input_order"]:
                indices = range(group["input_group"] * 128, (group["input_group"] + 1) * 128)
                self.assertEqual(Fraction(group["sum_absolute_contributions"]),
                                 sum((abs(values[i]) for i in indices if i not in excluded), Fraction()))

    def test_total_signed_and_absolute_complements(self):
        for label, name, _, split in self.accounts():
            values = [value for i, value in enumerate(self.oracles[label, name])
                      if i not in split["excluded_ranked_coordinates"]]
            self.assertEqual(len(values), 888)
            self.assertEqual(Fraction(split["unselected_signed_sum"]), sum(values, Fraction()))
            self.assertEqual(Fraction(split["unselected_absolute_sum"]), sum(map(abs, values), Fraction()))

    def test_selected_and_full_absolute_accounts(self):
        for label, name, account, split in self.accounts():
            values = self.oracles[label, name]
            signed = sum((values[i] for i in split["excluded_ranked_coordinates"]), Fraction())
            absolute = sum((abs(values[i]) for i in split["excluded_ranked_coordinates"]), Fraction())
            self.assertEqual(Fraction(split["selected_signed_sum"]), signed)
            self.assertEqual(Fraction(split["selected_absolute_sum"]), absolute)
            self.assertEqual(absolute + Fraction(split["unselected_absolute_sum"]),
                             Fraction(account["channels"]["exact_sum_of_absolute_contributions"]))

    def test_retained_residual_and_boundary_independent_oracle(self):
        for label, name, account, split in self.accounts():
            index = account["numeric_id"]
            actual = fp16(int(self.arrays[label]["logits"][index]))
            reference = (fp16(int(self.refs["logits_fp16"][index])) if name == "fp16"
                         else Fraction(*float(self.refs["logits_binary64"][index]).as_integer_ratio()))
            residual = actual - reference
            self.assertEqual(Fraction(account["retained_signed_residual"]), residual)
            self.assertEqual(Fraction(account["boundary_remainder"]),
                             residual - sum(self.oracles[label, name], Fraction()))
            self.assertEqual(Fraction(split["selected_signed_sum"]) +
                             Fraction(split["unselected_signed_sum"]) +
                             Fraction(account["boundary_remainder"]), residual)

    def test_exact_group_partition_counts(self):
        for _, _, _, split in self.accounts():
            counts = []
            for index, group in enumerate(split["groups_in_input_order"]):
                excluded = sum(i // 128 == index for i in split["excluded_ranked_coordinates"])
                self.assertEqual(group["input_group"], index)
                self.assertEqual(group["start_coordinate"], index * 128)
                self.assertEqual(group["end_coordinate_exclusive"], (index + 1) * 128)
                self.assertEqual(group["coordinate_count"], 128 - excluded)
                self.assertEqual(group["excluded_ranked_coordinate_count"], excluded)
                counts.append(group["coordinate_count"])
            self.assertEqual(len(counts), 7)
            self.assertEqual(sum(counts), 888)

    def test_control_branch_and_logical_census(self):
        self.assertEqual([row["control"] for row in self.report["controls"]], list(d.parent.CONTROLS))
        for row in self.report["controls"]:
            self.assertEqual(list(row["branches"]), list(d.BRANCHES))
        self.assertEqual(len(list(self.accounts())), 18)
        self.assertEqual(self.report["unselected_input_coordinate_count"], 15984)
        self.assertEqual(self.report["complement_group_count"], 126)
        self.assertTrue(all(split[key] for _, _, _, split in self.accounts()
                            for key in ("exact_complement_identity", "exact_absolute_identity",
                                        "exact_residual_identity")))

    def test_zero_terms_ties_and_selected_exclusion(self):
        account = synthetic(np.zeros(d.WIDTH, dtype="<f2"))
        split = d.split_unselected(account)
        self.assertEqual(split["excluded_ranked_coordinates"], list(range(8)))
        self.assertEqual([g["coordinate_count"] for g in split["groups_in_input_order"]],
                         [120, 128, 128, 128, 128, 128, 128])
        self.assertEqual(split["unselected_signed_sum"], "0")
        self.assertEqual(split["unselected_absolute_sum"], "0")

    def test_cancellation_is_not_absolute_net_group_sum(self):
        actual = np.zeros(d.WIDTH, dtype="<f2")
        actual[:8], actual[128:130] = 4, [1, -1]
        split = d.split_unselected(synthetic(actual))
        self.assertEqual(split["groups_in_input_order"][1]["signed_contribution"], "0")
        self.assertEqual(split["groups_in_input_order"][1]["sum_absolute_contributions"], "2")
        self.assertEqual(split["unselected_absolute_sum"], "2")

    def test_subnormal_and_binary64_terms_remain_exact(self):
        actual = np.zeros(d.WIDTH, dtype="<f2")
        actual[:8], actual[128] = 4, 2 ** -24
        reference = np.zeros(d.WIDTH, dtype="<f8")
        reference[129] = 2 ** -100
        weights = np.ones(d.WIDTH, dtype="<f2")
        weights[128] = 2 ** -24
        split = d.split_unselected(synthetic(actual, reference, weights))
        self.assertEqual(Fraction(split["unselected_signed_sum"]),
                         Fraction(1, 2 ** 48) - Fraction(1, 2 ** 100))
        self.assertEqual(Fraction(split["unselected_absolute_sum"]),
                         Fraction(1, 2 ** 48) + Fraction(1, 2 ** 100))

    def test_group_edges_and_negative_weights(self):
        actual = np.zeros(d.WIDTH, dtype="<f2")
        chosen = [0, 127, 128, 255, 256, 767, 768, 895]
        actual[chosen] = 4
        actual[126], actual[769] = 1, -1
        weights = -np.ones(d.WIDTH, dtype="<f2")
        split = d.split_unselected(synthetic(actual, weights=weights))
        self.assertEqual(split["excluded_ranked_coordinates"], chosen)
        self.assertEqual([g["coordinate_count"] for g in split["groups_in_input_order"]],
                         [126, 126, 127, 128, 128, 127, 126])
        self.assertEqual(split["groups_in_input_order"][0]["signed_contribution"], "-1")
        self.assertEqual(split["groups_in_input_order"][6]["signed_contribution"], "1")

    def test_schema_and_json_roundtrip(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        self.assertEqual(json.loads(json.dumps(self.report, allow_nan=False)), self.report)

    def test_schema_rejects_census_order_type_and_extra_fields(self):
        template = next(self.accounts())[3]
        validator = jsonschema.Draft202012Validator(d.SPLIT_SCHEMA)
        for key, value in (("coordinate_count", 889), ("unselected_signed_sum", 0),
                           ("exact_absolute_identity", False), ("extra", 0),
                           ("excluded_ranked_coordinates", [0] * 8)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**template, key: value})
        for key, value in (("input_group", 1), ("coordinate_count", True),
                           ("start_coordinate", 1), ("sum_absolute_contributions", "nan")):
            bad = deepcopy(template)
            bad["groups_in_input_order"][0][key] = value
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(bad)
        bad = deepcopy(template)
        bad["groups_in_input_order"].reverse()
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(bad)

    def test_split_rejects_corrupt_selected_ranking_and_identities(self):
        template = synthetic(np.zeros(d.WIDTH, dtype="<f2"))
        for key in ("ranked_signed_sum", "unranked_exact_remainder", "exact_sum",
                    "exact_sum_of_absolute_contributions"):
            bad = deepcopy(template)
            bad["channels"][key] = "1"
            with self.assertRaises(ValueError):
                d.split_unselected(bad)
        for key in ("retained_signed_residual", "boundary_remainder"):
            with self.assertRaises(ValueError):
                d.split_unselected({**template, key: "1"})
        for field in ("full_absolute_signed_ranking", "largest_absolute_signed"):
            bad = deepcopy(template)
            bad["channels"][field][0]["coordinate"] = 9
            with self.assertRaises(ValueError):
                d.split_unselected(bad)

    def test_report_rejects_inherited_and_selection_splices(self):
        for key, value in (("numeric_id", 1), ("boundary_remainder", "1")):
            bad = deepcopy(self.inherited)
            bad["controls"][0]["branches"]["fp16"][key] = value
            with self.assertRaises(ValueError):
                d.report((*self.evidence[:5], bad))
        selected = deepcopy(self.selected)
        label = next(iter(selected))
        index, residual = selected[label]["fp16"]
        selected[label]["fp16"] = (index, residual + 1)
        with self.assertRaises(ValueError):
            d.report((*self.evidence[:3], selected, *self.evidence[4:]))
        bad = deepcopy(self.inherited)
        bad["controls"][0]["branches"]["fp16"]["channels"]["coordinate62"]["input_delta"] = "0"
        with self.assertRaises(ValueError):
            d.report((*self.evidence[:5], bad))

    def test_history_threshold_reference_and_lineage_gates_preserved(self):
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(self.report["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(self.report["retained_flags"], self.result["flags"])
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
        for key, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.report((bad, *self.evidence[1:]))
        bad = deepcopy(self.result)
        bad["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.report((bad, *self.evidence[1:]))

    def test_source_input_and_asset_authentication_gates(self):
        for pin in (*d.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        preflight = d.channels.contributions.preflight
        with patch.object(d.base, "authenticate", return_value=(self.result, self.arrays, self.refs, [])):
            with patch.object(preflight, "bind_assets", return_value={}), self.assertRaises(ValueError):
                d.channels.hotspots.authenticate()
            with patch.object(preflight, "bind_assets", return_value=self.result["preflight"]["assets"]):
                with patch.object(preflight, "bind_final_reference", return_value={}), self.assertRaises(ValueError):
                    d.channels.hotspots.authenticate()

    def test_terminal_reviewer_gate_preserved(self):
        review = json.loads(d.base.read_bound(d.base.PINS["review"]))
        mission = json.loads(d.base.read_bound(d.base.PINS["mission"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.base.PINS["review"]["path"]},
                  "mission": {"path": d.base.PINS["mission"]["path"]}}
        backlog = [{"id": d.base.MISSION, "status": "done", "outcome": {"review_status": "done"},
                    "finished_ts": review["created_at"]}]
        for changed in ({**review, "producer_role": "engineer"}, {**review, "review": {"status": "pending"}}):
            with self.assertRaises(ValueError):
                d.base.terminal_review(changed, latest, backlog, mission)

    def test_forbidden_dispatch_and_write_guards(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
            lambda: d.channels.hotspots.measure(), lambda: d.channels.hotspots.check(),
            lambda: d.channels.contributions.dyadic_dot(None, None),
            lambda: d.channels.contributions.check(), lambda: d.base.check(),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: subprocess.Popen(["false"]), lambda: os.system("false"),
            lambda: open(d.SOURCE, "w"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE), lambda: d.parent.write(None, None),
            lambda: d.channels.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        self.assertEqual(d.FLAGS, d.channels.FLAGS)
        for key in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                    "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "selected_row_head_replay"):
            self.assertEqual(d.FLAGS[key], 0)

    def test_cli_one_json_and_no_execution_or_output_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value=self.report), patch("sys.stdout", stream):
            d.main(["--check"])
        text = stream.getvalue()
        document, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(document, self.report)
        self.assertEqual(text[end:], "\n")
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_source_workdir_account_and_interpreter_gate(self):
        contexts = (
            patch.object(d, "SOURCE", d.TEST), patch.object(d.Path, "cwd", return_value=d.ROOT.parent),
            patch.object(d.os, "getuid", return_value=0),
            patch.object(d.sys, "executable", "/invalid/python"),
            patch.dict(d.os.environ, {"PYTHONPATH": ""}),
            patch.object(d.sys, "dont_write_bytecode", False),
        )
        for context in contexts:
            with context, self.assertRaisesRegex(ValueError, "isolated source/workdir"):
                d.check()


if __name__ == "__main__":
    unittest.main()
