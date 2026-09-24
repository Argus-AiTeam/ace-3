"""Independent half-bit, native-nibble and scalar-anchor retained-account oracles."""

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_stage13_rmsnorm_source_bridge_v1 as d
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    value = (Fraction(mantissa, 1 << 24) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def mass(values):
    total = sum(values, Fraction())
    absolute = sum(map(abs, values), Fraction())
    return {"signed": str(total), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(total))}


def anchor(words):
    mean = sum((half(int(w))**2 for w in words), Fraction())/896
    radicand = mean+Fraction(1, 1000000)
    with localcontext() as context:
        context.prec = 90
        root = float(Decimal.from_float(float(radicand)).sqrt())
    inverse = Fraction(1.0/root)
    return {
        "mean_square": str(mean), "epsilon": "1/1000000", "radicand": str(radicand),
        "radicand_binary64_conversion_delta": str(Fraction(float(radicand))-radicand),
        "root_binary64_hex": root.hex(), "inverse_norm_anchor": str(inverse),
        "inverse_norm_anchor_hex": float(inverse).hex(),
        "inverse_square_identity_defect": str(inverse**2*radicand-1),
    }


def weight(tensors, output, coordinate):
    shift = 4*(0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    group = coordinate//128
    q = (int(tensors["qweight"][coordinate, output//8]) % (1 << 32))//(1 << shift) % 16
    z = (int(tensors["qzeros"][group, output//8]) % (1 << 32))//(1 << shift) % 16
    return (q-z)*half(int(tensors["scales"].view("<u2")[group, output]))


def replace_path(value, path, replacement):
    if not path:
        return replacement
    result = value.copy()
    key = path[0]
    result[key] = replace_path(value[key], path[1:], replacement)
    return result


class RMSNormSourceTests(unittest.TestCase):
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
        cls.previous_inputs, cls.norm, cls.anchors, cls.retained = cls.inputs
        cls.e = cls.previous_inputs[0][0]
        cls.actual, cls.reference, cls.norm_values = d.bind_inputs(cls.e, cls.norm)
        cls.oracle_anchors = {
            "original_input_fp16": anchor(cls.e["reference_archive"]["stage12"]),
            "actual": {c: anchor(a["stage12"]) for c, a in cls.e["archives"].items()}}

    def accounts(self):
        for row in self.result["rows"]:
            for hotspot in row["hotspots"]:
                for selected in hotspot["selected_stage16_coordinates"]:
                    for kind in ("gate_proj", "up_proj"):
                        yield row, hotspot, selected, kind, selected[kind]

    def test_all_ranked_four_term_identities_independent_half_and_nibble_oracle(self):
        count = 0
        for row, hotspot, selected, kind, projection in self.accounts():
            control = row["control"]
            actual, reference = self.e["archives"][control], self.e["reference_archive"]
            sa = Fraction(self.oracle_anchors["actual"][control]["inverse_norm_anchor"])
            sr = Fraction(self.oracle_anchors["original_input_fp16"]["inverse_norm_anchor"])
            opposite = "up" if kind == "gate_proj" else "gate"
            factor = (Fraction(hotspot["retained_weighted_row_factor"])
                      * Fraction(selected["selected_down_input"]["weight"])
                      * Fraction(selected["bridge"]["reference_operands"][opposite]))
            for rank, source in enumerate(projection["ranked_stage13_rmsnorm_sources"], 1):
                pin = projection["largest_absolute_coordinates"][rank-1]
                i = pin["coordinate"]
                a, r, ya, yr = (half(int(archive[stage][i]))
                               for archive, stage in ((actual, "stage12"), (reference, "stage12"),
                                                      (actual, "stage13"), (reference, "stage13")))
                w = half(int(self.norm.view("<u2")[i]))
                ba, br = ya-w*a*sa, yr-w*r*sr
                values = [w*(a-r)*sr, w*r*(sa-sr), w*(a-r)*(sa-sr), ba-br]
                column = weight(self.previous_inputs[1][kind], projection["output_coordinate"], i)
                projected = [v*column for v in values]
                weighted = [v*factor for v in projected]
                self.assertEqual(source["retained_input_rank"], rank)
                self.assertEqual(source["retained_ranked_input"], pin)
                self.assertEqual(source["terms"], [
                    {"term": name, "stage13_delta_contribution": str(v),
                     "projection_contribution": str(p), "coordinate_linear_operand_contribution": str(c)}
                    for name, v, p, c in zip(d.TERMS, values, projected, weighted, strict=True)])
                for field, value in (
                    ("actual_stage12", a), ("original_input_fp16_stage12", r),
                    ("stage12_delta", a-r), ("norm_weight", w),
                    ("actual_stage13", ya), ("original_input_fp16_stage13", yr),
                    ("retained_stage13_delta", ya-yr), ("scalar_anchor_delta", sa-sr),
                    ("actual_output_boundary_remainder", ba), ("original_output_boundary_remainder", br),
                    ("gate_up_weight", column), ("coordinate_linear_operand_factor", factor),
                    ("actual_raw_scratch_q24", Fraction(int(actual["scratch_i"][i]), 1 << 24)),
                    ("actual_q24_to_stage12_conversion", a-Fraction(int(actual["scratch_i"][i]), 1 << 24)),
                ):
                    self.assertEqual(source[field], str(value))
                self.assertEqual(source["stage13_totals"], mass(values))
                self.assertEqual(source["projection_totals"], mass(projected))
                self.assertEqual(source["coordinate_linear_operand_totals"], mass(weighted))
                self.assertEqual(sum(values, Fraction()), ya-yr)
                self.assertEqual(sum(projected, Fraction()), Fraction(pin["signed_contribution"]))
                self.assertEqual(sum(weighted, Fraction()), Fraction(pin["signed_contribution"])*factor)
                for field in ("stage13_closure_residual", "parent_ranked_term_closure_residual",
                              "coordinate_linear_operand_closure_residual"):
                    self.assertEqual(source[field], "0")
                count += 1
        self.assertEqual(count, 2304)

    def test_scalar_anchors_exact_energy_conversion_and_identity_defects(self):
        self.assertEqual(self.anchors, self.oracle_anchors)
        self.assertEqual(self.result["stage12_scalar_anchors"], self.oracle_anchors)
        self.assertEqual(self.result["native_epsilon_q48"], 281474977)
        self.assertEqual(self.result["native_epsilon_minus_contract"],
                         str(Fraction(281474977, 1 << 48)-Fraction(1, 1000000)))

    def test_all_projection_component_totals_and_retained_linear_operand_closure(self):
        for _, _, selected, kind, projection in self.accounts():
            sources = projection["ranked_stage13_rmsnorm_sources"]
            summary = projection["rmsnorm_source_summary"]
            projected = [Fraction(t["projection_contribution"]) for s in sources for t in s["terms"]]
            weighted = [Fraction(t["coordinate_linear_operand_contribution"])
                        for s in sources for t in s["terms"]]
            factor = Fraction(summary["coordinate_linear_operand_factor"])
            unranked = Fraction(projection["unranked_input_totals"]["signed"])
            boundary = Fraction(projection["projection_boundary_remainder"])
            self.assertEqual(summary["ranked_projection_totals"], mass(projected))
            self.assertEqual(summary["ranked_coordinate_linear_operand_totals"], mass(weighted))
            self.assertEqual(summary["retained_ranked_input_totals"], projection["ranked_input_totals"])
            self.assertEqual(summary["retained_unranked_input_totals"], projection["unranked_input_totals"])
            self.assertEqual(summary["retained_projection_boundary_remainder"], str(boundary))
            for i, term in enumerate(d.TERMS):
                for prefix, field in (("projection", "projection_contribution"),
                                      ("coordinate_linear_operand", "coordinate_linear_operand_contribution")):
                    self.assertEqual(summary["component_"+prefix+"_totals"][term],
                                     mass([Fraction(s["terms"][i][field]) for s in sources]))
            self.assertEqual(summary["expanded_projection_and_boundary_totals"],
                             mass([*projected, unranked, boundary]))
            self.assertEqual(summary["expanded_coordinate_linear_operand_totals"],
                             mass([*weighted, unranked*factor, boundary*factor]))
            linear = selected["coordinate_weighted_terms_in_operand_order"][0 if kind == "gate_proj" else 1]
            self.assertEqual(summary["retained_coordinate_linear_operand_term"], linear["signed_contribution"])
            self.assertEqual(sum(weighted, Fraction())+(unranked+boundary)*factor,
                             Fraction(linear["signed_contribution"]))
            self.assertEqual(sum(projected, Fraction())+unranked+boundary,
                             Fraction(projection["retained_projection_delta"]))
            self.assertEqual(summary["projection_closure_residual"], "0")
            self.assertEqual(summary["retained_linear_operand_closure_residual"], "0")

    def test_preserved_parent_rows_selections_remainders_and_stop_gates(self):
        for row, old in zip(self.result["rows"], self.retained["rows"], strict=True):
            self.assertEqual({k: v for k, v in row.items() if k != "hotspots"},
                             {k: v for k, v in old.items() if k != "hotspots"})
        for row, hotspot, selected, kind, projection in self.accounts():
            old_row = next(r for r in self.retained["rows"]
                           if (r["control"], r["branch"]) == (row["control"], row["branch"]))
            old_hotspot = next(h for h in old_row["hotspots"]
                               if h["output_coordinate"] == hotspot["output_coordinate"])
            old_selected = next(s for s in old_hotspot["selected_stage16_coordinates"]
                                if s["selected_down_input"] == selected["selected_down_input"])
            self.assertEqual({k: v for k, v in projection.items()
                              if k not in ("ranked_stage13_rmsnorm_sources", "rmsnorm_source_summary",
                                           "rmsnorm_source_factor_bindings")}, old_selected[kind])
            self.assertEqual(selected["bridge"], old_selected["bridge"])
            self.assertEqual(hotspot["summary"], old_hotspot["summary"])
        for key, value in self.retained.items():
            if key != "rows":
                self.assertEqual(self.result[key], value)
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        self.assertEqual(self.result["ranked_stage13_source_account_count"], 2304)
        self.assertEqual(self.result["rmsnorm_source_component_count"], 9216)

    def test_coordinate741_and_every_control_reference_branch_representative(self):
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.previous.hotspot.census.CONTROLS for b in ("fp16", "binary64")])
        for row in self.result["rows"]:
            self.assertEqual(len(row["hotspots"]), 1)
            if row["control"] == "frozen_inherited":
                self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
                self.assertEqual(row["hotspots"][0]["output_coordinate"], 741)
        self.assertEqual(self.result["internal_reference"], "original_input_L23_fp16")
        self.assertEqual(self.result["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")

    def test_tampered_stage12_stage13_actual_reference_geometry_and_bits(self):
        control = d.previous.hotspot.census.CONTROLS[0]
        for is_reference in (False, True):
            archive = self.e["reference_archive"] if is_reference else self.e["archives"][control]
            for stage in ("stage12", "stage13"):
                changed = archive[stage].copy()
                changed[0] ^= 1
                for value in (changed, archive[stage][:-1], archive[stage].astype("<i4")):
                    path = ["reference_archive", stage] if is_reference else ["archives", control, stage]
                    with self.assertRaises(ValueError):
                        d.bind_inputs(replace_path(self.e, path, value), self.norm)

    def test_tampered_raw_q24_kv_and_control_order(self):
        control = d.previous.hotspot.census.CONTROLS[0]
        for stage in ("scratch_i", "scratch_z", "output_cache_k", "output_cache_v"):
            changed = self.e["archives"][control][stage].copy()
            changed.flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.bind_inputs(replace_path(self.e, ["archives", control, stage], changed), self.norm)
        with self.assertRaises(ValueError):
            d.bind_inputs({**self.e, "archives": dict(reversed(list(self.e["archives"].items())))}, self.norm)

    def test_post_attention_norm_tensor_name_bits_dtype_shape_and_hash_binding(self):
        pin = self.result["post_attention_rmsnorm_tensor_binding"]
        self.assertEqual(pin["name"], d.NORM)
        self.assertEqual(pin["sha256"], hashlib.sha256(self.norm.tobytes()).hexdigest())
        changed = self.norm.copy()
        changed.view("<u2")[0] ^= 1
        for value in (changed, self.norm[:-1], self.norm.astype("<f8"),
                      np.full(896, np.inf, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.bind_weight(value, pin)
        for key, value in (("name", "model.norm.weight"), ("shape", [895]),
                           ("dtype", "float64"), ("sha256", "0"*64)):
            with self.assertRaises(ValueError):
                d.bind_weight(self.norm, {**pin, key: value})

    def test_existing_gate_up_tensor_bindings_reject_tampering(self):
        for kind in ("gate_proj", "up_proj"):
            for suffix in ("qweight", "qzeros", "scales"):
                value = self.previous_inputs[1][kind][suffix].copy()
                value.view("u1").flat[0] ^= 1
                tensors = replace_path(self.previous_inputs[1], [kind, suffix], value)
                with self.assertRaises(ValueError):
                    d.previous.bind_inputs(self.e, tensors)

    def test_tampered_scalar_anchors_and_disclosures(self):
        for root in (["original_input_fp16"], ["actual", d.previous.hotspot.census.CONTROLS[0]]):
            for field in self.anchors["original_input_fp16"]:
                with self.assertRaises(ValueError):
                    d.bind_anchors(self.actual, self.reference,
                                   replace_path(self.anchors, [*root, field], "tampered"))

    def test_tampered_parent_selection_and_row_factor(self):
        path = ["rows", 0, "hotspots", 0, "selected_stage16_coordinates", 0, "gate_proj"]
        for changed in (
            replace_path(self.retained, [*path, "largest_absolute_coordinates"],
                         list(reversed(self.retained["rows"][0]["hotspots"][0]
                                       ["selected_stage16_coordinates"][0]["gate_proj"]
                                       ["largest_absolute_coordinates"]))),
            replace_path(self.retained, ["rows", 0, "hotspots", 0, "retained_weighted_row_factor"], "1"),
        ):
            with self.assertRaises(ValueError):
                d.report(self.previous_inputs, self.norm, self.anchors, changed)

    def test_changed_ranked_delta_weight_signed_absolute_and_coordinate_rejected(self):
        row, _, _, _, projection = next(self.accounts())
        pin = projection["largest_absolute_coordinates"][0]
        for field, value in (("input_delta", "1"), ("weight", "1"),
                             ("signed_contribution", "1"), ("absolute_contribution", "-1"),
                             ("coordinate", True), ("coordinate", -1), ("coordinate", 896)):
            with self.assertRaises(ValueError):
                d.account(self.actual[row["control"]], self.reference, self.norm_values, self.anchors,
                          row["control"], {**pin, field: value}, Fraction(pin["weight"]), Fraction(0))

    def test_zero_factors_equal_inputs_and_cancellation(self):
        row, _, _, _, projection = next(self.accounts())
        pin = projection["largest_absolute_coordinates"][0]
        control = row["control"]
        zero = d.account(self.actual[control], self.reference, self.norm_values, self.anchors,
                         control, pin, Fraction(pin["weight"]), Fraction(0))
        self.assertEqual(zero["coordinate_linear_operand_totals"], mass([]))
        i = pin["coordinate"]
        actual = {**self.reference, "scratch_q24": self.reference["stage12"]}
        anchors = {"actual": {control: self.anchors["original_input_fp16"]},
                   "original_input_fp16": self.anchors["original_input_fp16"]}
        equal_pin = {**pin, "input_delta": "0", "signed_contribution": "0", "absolute_contribution": "0"}
        equal = d.account(actual, self.reference, self.norm_values, anchors, control,
                          equal_pin, Fraction(pin["weight"]), Fraction(-1))
        self.assertEqual(equal["stage13_totals"], mass([]))
        self.assertEqual(equal["actual_stage12"], str(self.reference["stage12"][i]))
        self.assertEqual(d.bridge.mass([Fraction(3), Fraction(-3), Fraction(0)]),
                         {"signed": "0", "absolute": "6", "cancellation_absolute_mass": "6"})

    def test_forbidden_replay_dispatch_nonlinear_and_write_paths(self):
        target = d.parent.OUTPUT / "forbidden-stage13-rmsnorm-source"
        torch = d.parent.preflight.parent.producer.legacy.torch
        calls = (
            lambda: d.previous.check(), lambda: d.previous.collect(None),
            lambda: d.previous.run_tests(None, None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.parent.head.decode_array_q24(None),
            lambda: native._stages(None, None, None, None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: d.gate_up.measure(), lambda: d.hidden.measure(),
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

    def test_sources_original_references_history_thresholds_and_non_admission(self):
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.previous.SOURCE), "sha256": "0"*64})
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
        self.assertTrue(all(not value for value in d.FLAGS.values()))
        for text in ("common UNKNOWN", "not solely rounding", "Q24", "original-input FP16"):
            self.assertIn(text, d.BOUNDARY)

    def test_cli_exactly_one_json_document_and_no_extra_paths(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"tests": {"executed": 16}}), patch("sys.stdout", output):
            d.main(["--check"])
        self.assertEqual(json.loads(output.getvalue()), {"tests": {"executed": 16}})
        self.assertEqual(output.getvalue().count("\n"), 1)
        for argv in ([], ["--execute"], ["--che"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--output-root", "/tmp/forbidden"],
                     ["--check", "--reference", "/tmp/forbidden"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
