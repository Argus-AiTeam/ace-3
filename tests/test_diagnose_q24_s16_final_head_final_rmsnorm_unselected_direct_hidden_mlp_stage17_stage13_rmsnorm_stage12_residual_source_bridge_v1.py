"""Independent half-bit/native-nibble oracles for retained stage12 residual accounts."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_stage12_residual_source_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None
NAMES = (
    "input_hidden_delta", "attention_stage11_delta", "actual_input_q24_carry",
    "actual_attention_state_update", "actual_stage12_fp16_conversion",
    "negative_original_input_fp16_attention_residual_boundary",
)
FIELDS = (
    "stage12_delta_contribution", "stage13_direct_input_contribution",
    "projection_contribution", "coordinate_linear_operand_contribution",
)


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weight(tensors, output, coordinate):
    shift = 4*(0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    q = (int(tensors["qweight"][coordinate, output//8]) % (1 << 32))//(1 << shift) % 16
    z = (int(tensors["qzeros"][coordinate//128, output//8]) % (1 << 32))//(1 << shift) % 16
    return (q-z)*half(int(tensors["scales"].view("<u2")[coordinate//128, output]))


def mass(values):
    values = list(values)
    signed = sum(values, Fraction())
    absolute = sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


def replace_path(value, path, replacement):
    if not path:
        return replacement
    result = value.copy()
    key = path[0]
    result[key] = replace_path(value[key], path[1:], replacement)
    return result


class Stage12ResidualSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            audit = {"forbidden_calls": 0}
            cls.inputs = d.collect(audit)
            with d.read_only(audit):
                cls.result = d.report(*cls.inputs)
            if audit["forbidden_calls"]:
                raise AssertionError("forbidden fixture work")
        else:
            cls.inputs, cls.result = EVIDENCE
        cls.previous_inputs, cls.retained = cls.inputs
        cls.projection_inputs, cls.norm, cls.anchors, _ = cls.previous_inputs
        cls.e = cls.projection_inputs[0][0]
        cls.actual, cls.reference, cls.norm_values = d.previous.bind_inputs(cls.e, cls.norm)
        cls.sr = Fraction(cls.anchors["original_input_fp16"]["inverse_norm_anchor"])

    def projections(self):
        for row in self.result["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        yield row, hotspot, selected, kind, selected[kind]

    def first_account(self):
        row, _, _, _, projection = next(self.projections())
        source = projection["ranked_stage13_rmsnorm_sources"][0]
        args = (self.actual[row["control"]], self.reference, source,
                Fraction(source["norm_weight"]), self.sr,
                Fraction(source["gate_up_weight"]), Fraction(source["coordinate_linear_operand_factor"]))
        return source, args

    def test_every_retained_direct_input_closes_independent_half_and_nibble_oracle(self):
        count = 0
        for row, hotspot, selected, kind, projection in self.projections():
            actual, reference = self.e["archives"][row["control"]], self.e["reference_archive"]
            output = selected["selected_down_input"]["coordinate"]
            down = weight(self.projection_inputs[0][4], hotspot["output_coordinate"], output)
            opposite_stage = "stage15" if kind == "gate_proj" else "stage14"
            opposite = half(int(reference[opposite_stage][output]))
            factor = Fraction(hotspot["retained_weighted_row_factor"])*down*opposite
            for rank, source in enumerate(projection["ranked_stage13_rmsnorm_sources"], 1):
                i = source["retained_ranked_input"]["coordinate"]
                a, h, att = (half(int(actual[k][i])) for k in ("stage12", "input_hidden", "stage11"))
                r, rh, ratt = (half(int(reference[k][i])) for k in ("stage12", "input_hidden", "stage11"))
                iq, sq = (Fraction(int(actual[k][i]), 1 << 24) for k in ("input_i", "scratch_i"))
                values = [h-rh, att-ratt, iq-h, sq-iq-att, a-sq, rh+ratt-r]
                norm = half(int(self.norm.view("<u2")[i]))
                column = weight(self.projection_inputs[1][kind], output, i)
                direct = [v*norm*self.sr for v in values]
                projected = [v*column for v in direct]
                weighted = [v*factor for v in projected]
                account = source["direct_stage12_residual_source"]
                self.assertEqual(source["retained_input_rank"], rank)
                self.assertEqual(account["terms"], [
                    {"term": name, **{field: str(v) for field, v in zip(FIELDS, vs, strict=True)}}
                    for name, vs in zip(NAMES, zip(values, direct, projected, weighted, strict=True), strict=True)])
                self.assertEqual(account["totals"], {
                    field: mass(vs) for field, vs in zip(FIELDS, (values, direct, projected, weighted), strict=True)})
                self.assertEqual(sum(values, Fraction()), a-r)
                pin = source["terms"][0]
                self.assertEqual(account["retained_parent_direct_stage12_input"], pin)
                for vs, field in ((direct, "stage13_delta_contribution"),
                                  (projected, "projection_contribution"),
                                  (weighted, "coordinate_linear_operand_contribution")):
                    self.assertEqual(sum(vs, Fraction()), Fraction(pin[field]))
                self.assertEqual(account["norm_weight"], str(norm))
                self.assertEqual(account["original_input_fp16_scalar_anchor"], str(self.sr))
                self.assertEqual(account["gate_up_weight"], str(column))
                self.assertEqual(account["coordinate_linear_operand_factor"], str(factor))
                for key in ("stage12_closure_residual", "actual_boundary_closure_residual",
                            "parent_direct_stage13_closure_residual", "parent_direct_projection_closure_residual",
                            "parent_direct_coordinate_closure_residual"):
                    self.assertEqual(account[key], "0")
                count += 1
        self.assertEqual(count, 2304)

    def test_actual_and_original_boundary_disclosures_remain_separate(self):
        for row, _, _, _, projection in self.projections():
            actual, reference = self.e["archives"][row["control"]], self.e["reference_archive"]
            for source in projection["ranked_stage13_rmsnorm_sources"]:
                i = source["retained_ranked_input"]["coordinate"]
                result = source["direct_stage12_residual_source"]
                h, att, x = (half(int(actual[k][i])) for k in ("input_hidden", "stage11", "stage12"))
                rh, ratt, rx = (half(int(reference[k][i])) for k in ("input_hidden", "stage11", "stage12"))
                iq, sq = (Fraction(int(actual[k][i]), 1 << 24) for k in ("input_i", "scratch_i"))
                self.assertEqual(result["actual_operands"], {
                    "input_hidden": str(h), "attention_stage11": str(att), "input_q24": str(iq),
                    "scratch_q24": str(sq), "stage12_fp16": str(x)})
                self.assertEqual(result["original_input_fp16_operands"], {
                    "input_hidden": str(rh), "attention_stage11": str(ratt), "stage12_fp16": str(rx)})
                for field, value in (
                    ("actual_attention_residual_boundary", x-h-att),
                    ("actual_input_q24_carry", iq-h), ("actual_attention_state_update", sq-iq-att),
                    ("actual_stage12_fp16_conversion", x-sq),
                    ("original_input_fp16_attention_residual_boundary", rx-rh-ratt),
                    ("actual_minus_original_attention_residual_boundary", x-h-att-rx+rh+ratt),
                ):
                    self.assertEqual(result[field], str(value))

    def assert_summary(self, summary, accounts):
        self.assertEqual(summary["account_count"], len(accounts))
        self.assertEqual(summary["component_count"], len(accounts)*6)
        self.assertEqual(summary["totals"], {
            field: mass(Fraction(t[field]) for a in accounts for t in a["terms"]) for field in FIELDS})
        self.assertEqual(summary["component_totals"], {
            name: {field: mass(Fraction(a["terms"][i][field]) for a in accounts) for field in FIELDS}
            for i, name in enumerate(NAMES)})

    def test_projection_row_and_global_signed_absolute_cancellation_totals(self):
        all_accounts = []
        for row in self.result["rows"]:
            row_accounts = []
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        projection = selected[kind]
                        accounts = [s["direct_stage12_residual_source"]
                                    for s in projection["ranked_stage13_rmsnorm_sources"]]
                        self.assert_summary(projection["stage12_residual_source_summary"], accounts)
                        row_accounts.extend(accounts)
            self.assert_summary(row["stage12_residual_source_summary"], row_accounts)
            all_accounts.extend(row_accounts)
        self.assert_summary(self.result["stage12_residual_source_summary"], all_accounts)
        self.assertEqual(len(all_accounts), 2304)
        self.assertEqual(self.result["stage12_residual_source_summary"]["component_count"], 13824)

    def test_parent_report_survives_lossless_removal_of_only_new_accounts(self):
        stripped = deepcopy(self.result)
        for key in ("stage12_residual_source_summary", "stage12_residual_source_identity",
                    "stage12_residual_source_weighting", "stage12_residual_reference"):
            stripped.pop(key)
        for row in stripped["rows"]:
            row.pop("stage12_residual_source_summary")
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        projection = selected[kind]
                        projection.pop("stage12_residual_source_summary")
                        for source in projection["ranked_stage13_rmsnorm_sources"]:
                            source.pop("direct_stage12_residual_source")
        self.assertEqual(stripped, self.retained)
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])

    def test_coordinate741_every_control_and_both_reference_branches(self):
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.previous.previous.hotspot.census.CONTROLS
                          for b in ("fp16", "binary64")])
        for row in self.result["rows"]:
            self.assertEqual(len(row["hotspots"]), 1)
            self.assertEqual(row["stage12_residual_source_summary"]["account_count"], 128)
            if row["control"] == "frozen_inherited":
                self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
                self.assertEqual(row["hotspots"][0]["output_coordinate"], 741)
        self.assertEqual(self.result["internal_reference"], "original_input_L23_fp16")
        self.assertEqual(self.result["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")

    def test_tampered_retained_actual_original_operands_geometry_and_bits(self):
        control = d.previous.previous.hotspot.census.CONTROLS[0]
        for is_reference in (False, True):
            archive = self.e["reference_archive"] if is_reference else self.e["archives"][control]
            for stage in ("input_hidden", "stage11", "stage12", "stage13"):
                changed = archive[stage].copy()
                changed[0] ^= 1
                for value in (changed, archive[stage][:-1], archive[stage].astype("<i4")):
                    path = ["reference_archive", stage] if is_reference else ["archives", control, stage]
                    with self.assertRaises(ValueError):
                        d.previous.bind_inputs(replace_path(self.e, path, value), self.norm)

    def test_tampered_q24_signed_zero_and_kv_bindings(self):
        control = d.previous.previous.hotspot.census.CONTROLS[0]
        for stage in ("input_i", "input_z", "scratch_i", "scratch_z", "output_i", "output_z",
                      "output_cache_k", "output_cache_v", "stage03", "stage06", "stage07"):
            changed = self.e["archives"][control][stage].copy()
            changed.flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.previous.bind_inputs(replace_path(self.e, ["archives", control, stage], changed), self.norm)
        with self.assertRaises(ValueError):
            d.previous.bind_inputs({**self.e, "archives": dict(reversed(list(self.e["archives"].items())))}, self.norm)

    def test_tampered_parent_ranked_selection_direct_term_and_row_factor(self):
        path = ["rows", 0, "hotspots", 0, "selected_stage16_coordinates", 0,
                "gate_proj", "ranked_stage13_rmsnorm_sources"]
        sources = self.retained["rows"][0]["hotspots"][0]["selected_stage16_coordinates"][0]["gate_proj"][
            "ranked_stage13_rmsnorm_sources"]
        for changed in (
            replace_path(self.retained, path, list(reversed(sources))),
            replace_path(self.retained, [*path, 0, "terms", 0, "stage13_delta_contribution"], "1"),
            replace_path(self.retained, ["rows", 0, "hotspots", 0, "retained_weighted_row_factor"], "1"),
        ):
            with self.assertRaises(ValueError):
                d.report(self.previous_inputs, changed)

    def test_tampered_source_operands_multipliers_and_direct_targets(self):
        source, args = self.first_account()
        for field in ("actual_stage12", "original_input_fp16_stage12", "stage12_delta", "norm_weight",
                      "actual_raw_scratch_q24", "actual_q24_to_stage12_conversion",
                      "gate_up_weight", "coordinate_linear_operand_factor"):
            with self.assertRaises(ValueError):
                d.account(*args[:2], {**source, field: "tampered"}, *args[3:])
        for field in ("stage13_delta_contribution", "projection_contribution",
                      "coordinate_linear_operand_contribution"):
            changed = replace_path(source, ["terms", 0, field], "tampered")
            with self.assertRaises(ValueError):
                d.account(*args[:2], changed, *args[3:])
        for i in (True, -1, 896):
            changed = replace_path(source, ["retained_ranked_input", "coordinate"], i)
            with self.assertRaises(ValueError):
                d.account(*args[:2], changed, *args[3:])
        with self.assertRaises(ValueError):
            d.account(*args[:-1], 0.0)

    def test_native_tensor_norm_anchor_and_source_bindings_reject_tampering(self):
        for kind in ("gate_proj", "up_proj"):
            for suffix in ("qweight", "qzeros", "scales"):
                value = self.projection_inputs[1][kind][suffix].copy()
                value.view("u1").flat[0] ^= 1
                tensors = replace_path(self.projection_inputs[1], [kind, suffix], value)
                with self.assertRaises(ValueError):
                    d.previous.previous.bind_inputs(self.e, tensors)
        pin = self.retained["post_attention_rmsnorm_tensor_binding"]
        changed = self.norm.copy()
        changed.view("<u2")[0] ^= 1
        for value in (changed, self.norm[:-1], self.norm.astype("<f8")):
            with self.assertRaises(ValueError):
                d.previous.bind_weight(value, pin)
        for field, value in (("name", "model.norm.weight"), ("sha256", "0"*64),
                             ("shape", [895]), ("dtype", "float64")):
            with self.assertRaises(ValueError):
                d.previous.bind_weight(self.norm, {**pin, field: value})
        with self.assertRaises(ValueError):
            d.previous.bind_anchors(self.actual, self.reference, replace_path(
                self.anchors, ["original_input_fp16", "inverse_norm_anchor"], "1"))
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.previous.SOURCE), "sha256": "0"*64})

    def test_zero_weight_factor_and_exact_cancellation(self):
        source, args = self.first_account()
        for column, factor in ((Fraction(0), args[-1]), (args[-2], Fraction(0))):
            changed = deepcopy(source)
            changed["gate_up_weight"] = str(column)
            changed["coordinate_linear_operand_factor"] = str(factor)
            direct = Fraction(changed["terms"][0]["stage13_delta_contribution"])
            changed["terms"][0]["projection_contribution"] = str(direct*column)
            changed["terms"][0]["coordinate_linear_operand_contribution"] = str(direct*column*factor)
            result = d.account(*args[:2], changed, *args[3:5], column, factor)
            self.assertEqual(result["totals"]["coordinate_linear_operand_contribution"], mass([]))
        self.assertEqual(d.bridge.mass([Fraction(3), Fraction(-3), Fraction(0)]), mass([Fraction(3), Fraction(-3)]))

    def test_original_references_thresholds_history_and_claim_boundaries(self):
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        for field, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)
        self.assertTrue(all(not v for v in d.FLAGS.values()))
        for text in ("common UNKNOWN", "not solely", "Q24", "original-input FP16", "strict-FP16-state"):
            self.assertIn(text, d.BOUNDARY)

    def test_forbidden_parent_operator_replay_and_write_paths(self):
        target = d.parent.OUTPUT / "forbidden-stage12-residual-source"
        torch = d.parent.preflight.parent.producer.legacy.torch
        calls = (
            lambda: d.previous.check(), lambda: d.previous.collect(None),
            lambda: d.previous.run_tests(None, None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.parent.head.decode_array_q24(None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: d.previous.gate_up.measure(), lambda: d.previous.hidden.measure(),
            lambda: math.exp(0), lambda: np.exp(0), lambda: torch.sigmoid(None),
            lambda: torch.nn.functional.silu(None), lambda: torch.cuda._lazy_init(),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
            lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
            lambda: os.replace(target, target), lambda: target.unlink(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_requires_check_and_emits_only_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"tests": {"executed": 14}}), patch("sys.stdout", output):
            d.main(["--check"])
        self.assertEqual(json.loads(output.getvalue()), {"tests": {"executed": 14}})
        self.assertEqual(output.getvalue().count("\n"), 1)
        for argv in ([], ["--execute"], ["--che"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--output-root", "/tmp/forbidden"],
                     ["--check", "--reference", "/tmp/forbidden"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
