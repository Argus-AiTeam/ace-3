"""Independent retained-bit oracles for selected RMSNorm boundary accounting."""

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
import io
import json
import os
import struct
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_boundary_residual_margin_bridge_v1 as d


EVIDENCE = None


def bits(word, mantissa_bits, exponent_bits, bias):
    exponent = (word >> mantissa_bits) & ((1 << exponent_bits)-1)
    mantissa = word & ((1 << mantissa_bits)-1)
    if exponent == (1 << exponent_bits)-1:
        raise ValueError("nonfinite oracle operand")
    value = Fraction(mantissa if exponent == 0 else (1 << mantissa_bits)+mantissa)
    value *= Fraction(2) ** ((1 if exponent == 0 else exponent)-bias-mantissa_bits)
    return -value if word >> (mantissa_bits+exponent_bits) else value


def half(word):
    return bits(int(word), 10, 5, 15)


def double(value):
    return bits(struct.unpack("<Q", struct.pack("<d", value))[0], 52, 11, 1023)


class BoundaryResidualMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, mc, ic in zip(self.e["report"]["controls"], self.e["margin_report"]["controls"],
                                   self.e["interaction_report"]["controls"], strict=True):
            for pair, mp, ip in zip(control["pairs"], mc["pairs"], ic["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield control, pair, branch, account, mp["branches"][branch], ip["branches"][branch]

    def assert_mass(self, observed, values):
        values = list(values)
        signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
        self.assertEqual(observed, {"signed": str(signed), "absolute": str(absolute),
                                   "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_boundary_components_independent_retained_bit_oracle(self):
        weights = [half(v) for v in self.e["weight_array"].view("<u2")]
        for control, pair, branch, account, original, _ in self.branches():
            label = control["control"]
            a = self.e["archives"][label]["stage18"]
            r = self.e["reference_archive"]["stage18"] if branch == "fp16" else self.e["binary64"]
            ya = self.e["arrays"][label]["rmsnorm"]
            yr = self.e["references"]["rmsnorm_"+branch]
            sa, sr = (Fraction(n["inverse_norm_anchor"])
                      for n in (control["actual_norm"], account["reference_norm"]))
            left, right = (self.e["rows"][pair[k]].view("<u2") for k in ("left_id", "right_id"))
            for row, old in zip(account["selected_coordinates"], original["selected_coordinates"], strict=True):
                i = row["coordinate"]
                ah, rh = half(a[i]), half(r[i]) if branch == "fp16" else double(r[i])
                ay, ry = half(ya[i]), half(yr[i]) if branch == "fp16" else double(yr[i])
                dw, w = half(left[i])-half(right[i]), weights[i]
                ba, br = ay-w*ah*sa, ry-w*rh*sr
                self.assertEqual(row["weighted_components"], {
                    d.COMPONENTS[0]: str(dw*ba), d.COMPONENTS[1]: str(-dw*br)})
                for side, h, y, s, b in (("actual", ah, ay, sa, ba), ("reference", rh, ry, sr, br)):
                    self.assertEqual(row[side], {
                        "retained_hidden": str(h), "retained_rmsnorm": str(y),
                        "inverse_norm_anchor": str(s), "anchored_product": str(w*h*s),
                        "boundary_conversion_remainder": str(b),
                        "weighted_boundary_conversion_remainder": str(dw*b)})
                self.assertEqual(row["boundary_remainder_delta_weighted_term"],
                                 old["weighted_terms"]["boundary_remainder_delta"])
                q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
                self.assertEqual(Fraction(row["actual_raw_q24_hidden"]), q)
                self.assertEqual(Fraction(row["actual_q24_to_stage18_conversion"]), ah-q)
                self.assert_mass(row["weighted_boundary_mass"], (dw*ba, -dw*br))

    def test_scalar_anchors_and_conversion_disclosures(self):
        for control in self.e["hidden_report"]["controls"]:
            actual = [half(v) for v in self.e["archives"][control["control"]]["stage18"]]
            for branch in d.BRANCHES:
                reference = ([half(v) for v in self.e["reference_archive"]["stage18"]]
                             if branch == "fp16" else [double(v) for v in self.e["binary64"]])
                for values, norm in ((actual, control["actual_norm"]),
                                     (reference, control["branches"][branch]["reference_norm"])):
                    radicand = sum(v*v for v in values)/896 + Fraction(1, 1000000)
                    self.assertEqual(Fraction(norm["radicand"]), radicand)
                    anchor = Fraction(norm["inverse_norm_anchor"])
                    with localcontext() as context:
                        context.prec = 70
                        expected = 1/(Decimal(radicand.numerator)/Decimal(radicand.denominator)).sqrt()
                        observed = Decimal(anchor.numerator)/Decimal(anchor.denominator)
                        self.assertLess(abs(observed/expected-1), Decimal("4e-16"))
                    self.assertEqual(Fraction(norm["inverse_square_identity_defect"]), anchor*anchor*radicand-1)
                    self.assertEqual(Fraction(norm["radicand_binary64_conversion_delta"]),
                                     double(float(radicand))-radicand)
        h = self.e["hidden_report"]
        self.assertEqual(h["native_epsilon_q48"], 281474977)
        self.assertEqual(Fraction(h["native_epsilon_minus_contract"]),
                         Fraction(281474977, 1 << 48)-Fraction(1, 1000000))

    def test_selected_totals_and_two_levels_of_cancellation(self):
        for _, _, _, account, original, _ in self.branches():
            rows, totals = account["selected_coordinates"], account["boundary_accounting"]
            absolute = Fraction()
            for k in d.COMPONENTS:
                values = [Fraction(row["weighted_components"][k]) for row in rows]
                self.assert_mass(totals["component_totals"][k], values)
                absolute += sum(map(abs, values))
            targets = [Fraction(row["boundary_remainder_delta_weighted_term"]) for row in rows]
            self.assert_mass(totals["selected_boundary_remainder_delta"], targets)
            self.assertEqual(Fraction(totals["component_absolute_sum"]), absolute)
            self.assertEqual(Fraction(totals["within_coordinate_cancellation_mass"]), absolute-sum(map(abs, targets)))
            self.assertEqual(Fraction(totals["across_coordinate_cancellation_mass"]),
                             sum(map(abs, targets))-abs(sum(targets)))
            self.assertEqual(Fraction(totals["total_component_cancellation_mass"]), absolute-abs(sum(targets)))
            self.assertEqual(str(sum(targets)),
                             original["accounting"]["selected_term_totals"]["boundary_remainder_delta"]["signed"])

    def test_selected_union_pairs_and_other_remainders_unchanged(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_interaction_residual_margin_bridge"],
                      self.e["interaction_report"])
        for control, geometry in zip(self.e["report"]["controls"], self.e["geometry"]["controls"], strict=True):
            self.assertEqual([(p["left_id"], p["right_id"], p["roles"]) for p in control["pairs"]],
                             [(a, b, roles) for (a, b), roles in d.margin.contributions.pairs_for(geometry).items()])
        for _, _, _, account, original, old in self.branches():
            for retained in (original, old):
                self.assertEqual([(r["coordinate"], r["selection_reasons"]) for r in account["selected_coordinates"]],
                                 [(r["coordinate"], r["selection_reasons"]) for r in retained["selected_coordinates"]])
            self.assertIs(account["unchanged_margin_accounting"], original["accounting"])
            totals = original["accounting"]
            self.assertEqual(sum(Fraction(v["signed"]) for v in totals["selected_term_totals"].values())
                             +Fraction(totals["unselected_coordinate_signed_remainder"])
                             +Fraction(totals["head_boundary_remainder_change"]),
                             Fraction(totals["retained_margin_change"]))
            self.assertGreaterEqual(Fraction(totals["unselected_coordinate_absolute_remainder"]),
                                    abs(Fraction(totals["unselected_coordinate_signed_remainder"])))

    def test_independent_reference_and_lineage_boundaries(self):
        self.assertNotEqual([half(v) for v in self.e["reference_archive"]["stage18"]],
                            [double(v) for v in self.e["binary64"]])
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_"+branch], original[branch])
        for _, _, branch, account, _, _ in self.branches():
            self.assertEqual(account["rmsnorm_reference"], "original_input_final_attempt003_"+branch)
            self.assertEqual(account["hidden_reference"], "original_input_L23_"+branch)
            self.assertEqual(account["residual_internal_reference"], "original_input_L23_fp16")
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])

    def fixture(self, ba=Fraction(2, 7), br=Fraction(2, 7), dw=Fraction(-3, 5), w=Fraction(4, 3)):
        a, r, sa, sr = Fraction(2), Fraction(3), Fraction(5), Fraction(7)
        terms = dict(zip(d.hidden.TERMS, (w*(a-r)*sr, w*r*(sa-sr), w*(a-r)*(sa-sr), ba-br), strict=True))
        ya, yr = w*a*sa+ba, w*r*sr+br
        h = {"coordinate": 62, "actual_hidden": str(a), "reference_hidden": str(r),
             "actual_rmsnorm": str(ya), "reference_rmsnorm": str(yr), "weight": str(w),
             "hidden_delta": str(a-r), "inverse_norm_anchor_delta": str(sa-sr),
             **{k: str(v) for k, v in terms.items()},
             "actual_boundary_conversion_remainder": str(ba), "reference_boundary_conversion_remainder": str(br),
             "retained_rmsnorm_delta": str(ya-yr), "signed_term_sum": str(ya-yr),
             "actual_raw_q24_hidden": "1", "actual_q24_to_stage18_conversion": "1"}
        row = {"coordinate": 62, "selection_reasons": ["required_coordinate_62"],
               "left_weight": str(dw), "right_weight": "0", "row_difference": str(dw), "hidden_bridge": h,
               "weighted_terms": {k: str(dw*v) for k, v in terms.items()},
               "weighted_actual_boundary_conversion_remainder": str(dw*ba),
               "weighted_reference_boundary_conversion_remainder": str(dw*br),
               "coordinate_margin_change": str(dw*(ya-yr))}
        return row, {"inverse_norm_anchor": str(sa)}, {"inverse_norm_anchor": str(sr)}

    def test_zero_net_boundary_preserves_cancelling_sides(self):
        row = d.split_coordinate(*self.fixture())
        self.assertEqual(row["boundary_remainder_delta_weighted_term"], "0")
        self.assertGreater(Fraction(row["weighted_boundary_mass"]["absolute"]), 0)
        self.assertEqual(row["weighted_boundary_mass"]["absolute"],
                         row["weighted_boundary_mass"]["cancellation_absolute_mass"])

    def test_zero_row_difference_and_zero_norm_weight(self):
        row = d.split_coordinate(*self.fixture(dw=Fraction()))
        self.assertEqual(row["weighted_boundary_mass"]["absolute"], "0")
        row = d.split_coordinate(*self.fixture(w=Fraction(), ba=Fraction(1, 3), br=Fraction(-1, 7)))
        self.assertNotEqual(row["boundary_remainder_delta_weighted_term"], "0")
        self.assertEqual(row["actual"]["anchored_product"], "0")

    def test_non_dyadic_signed_boundary_identity(self):
        row = d.split_coordinate(*self.fixture(ba=Fraction(-5, 11), br=Fraction(3, 13)))
        self.assertEqual(Fraction(row["boundary_remainder_delta_weighted_term"]),
                         -Fraction(3, 5)*(-Fraction(5, 11)-Fraction(3, 13)))

    def test_coordinate_and_operand_mutations_refused(self):
        for value in (-1, 896, True):
            args = self.fixture()
            args[0]["coordinate"] = value
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)
        for key in ("actual_hidden", "reference_hidden", "actual_rmsnorm", "reference_rmsnorm", "weight",
                    "hidden_delta", "inverse_norm_anchor_delta", "actual_boundary_conversion_remainder",
                    "reference_boundary_conversion_remainder", "actual_q24_to_stage18_conversion"):
            args = self.fixture()
            args[0]["hidden_bridge"][key] = str(Fraction(args[0]["hidden_bridge"][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)

    def test_weighted_terms_and_scalar_mutations_refused(self):
        for key in ("left_weight", "right_weight", "row_difference",
                    "weighted_actual_boundary_conversion_remainder",
                    "weighted_reference_boundary_conversion_remainder", "coordinate_margin_change"):
            args = self.fixture()
            args[0][key] = str(Fraction(args[0][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)
        for term in d.hidden.TERMS:
            args = self.fixture()
            args[0]["weighted_terms"][term] = "999"
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)
        for index in (1, 2):
            args = self.fixture()
            args[index]["inverse_norm_anchor"] = "0"
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)

    def test_selected_component_and_total_mutations_refused(self):
        _, _, _, account, original, _ = next(self.branches())
        rows = deepcopy(account["selected_coordinates"])
        rows[0]["weighted_components"][d.COMPONENTS[0]] = "999"
        with self.assertRaises(ValueError):
            d.selected_account(rows, original["accounting"])
        totals = deepcopy(original["accounting"])
        totals["selected_term_totals"]["boundary_remainder_delta"]["absolute"] = "-1"
        with self.assertRaises(ValueError):
            d.selected_account(account["selected_coordinates"], totals)

    def test_report_linkage_census_and_selection_mutations_refused(self):
        old = self.e["interaction_report"]
        for field, value in (
            ("retained_final_rmsnorm_scalar_scale_residual_margin_bridge", {}),
            ("controls", list(reversed(old["controls"]))), ("coordinate62_accounts", 0),
        ):
            with self.assertRaises(ValueError):
                d.report({**self.e, "interaction_report": {**old, field: value}})
        c = old["controls"][0]
        p = c["pairs"][0]
        b = p["branches"]["fp16"]
        rows = b["selected_coordinates"]
        changed_pair = {**p, "branches": {**p["branches"], "fp16": {
            **b, "selected_coordinates": [{**rows[0], "selection_reasons": []}, *rows[1:]]}}}
        changed = {**old, "controls": [{**c, "pairs": [changed_pair, *c["pairs"][1:]]}, *old["controls"][1:]]}
        with self.assertRaises(ValueError):
            d.report({**self.e, "interaction_report": changed})

    def test_history_threshold_source_and_authority_gates(self):
        for key, value in (("mandatory_statuses", ["PASS"]*19), ("S18_failure_indices", []),
                           ("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)
        for pin in (d.base.PINS["review"], *self.e["hidden_pins"],
                    d.margin.rows.PINS["rmsnorm_hotspot_source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        changed = deepcopy(self.e["result"])
        changed["preflight"]["final_reference"]["reference"]["input_fp16"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            d.residual.load_residuals(changed)
        self.assertEqual(self.e["margin_report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])

    def test_state_kv_tokenizer_and_tensor_gates(self):
        archive = self.e["archives"][d.parent.CONTROLS[0]]
        for key in ("output_z", "output_cache_k"):
            changed = {**archive, key: archive[key].copy()}
            changed[key].flat[0] = 2 if key == "output_z" else int(changed[key].flat[0]) ^ 1
            with self.assertRaises(ValueError):
                d.residual.operands(changed, actual=True)
        assets = self.e["assets"]
        self.assertEqual(assets["tokenizer_status"], "BOUND")
        self.assertFalse(assets["missing_tokenizer"])
        self.assertEqual(assets["tensors"]["lm_head.weight"], assets["tensors"]["model.embed_tokens.weight"])
        with self.assertRaises(ValueError):
            d.hidden.validate_weight(self.e["weight_array"],
                                     {**assets["tensors"]["model.norm.weight"], "sha256": "0"*64})

    def test_forbidden_dispatch_and_prior_checks(self):
        calls = [
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.residual.local.local_reference(None), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.margin.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden-native"]), lambda: os.system("forbidden-native"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        for module in (d.interaction, d.scale, d.direct, d.margin, d.hidden, d.residual):
            calls.extend((lambda m=module: m.check(), lambda m=module: m.measure(),
                          lambda m=module: m.focused_tests(None)))
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-boundary-bridge"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_rejects_execution_output_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value=self.e["report"]), patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        self.assertEqual(json.loads(text), self.e["report"])
        self.assertEqual(text.count("\n"), 1)

    def test_report_shape_and_census(self):
        report = self.e["report"]
        self.assertEqual(set(report), {
            "control_count", "diagnostic_pair_count", "pair_branch_count", "coordinate62_accounts",
            "selected_coordinate_accounts", "weighted_boundary_component_count", "lineage_separation",
            "reference_scope", "identity", "remainder_scope", "scalar_anchor",
            "retained_final_rmsnorm_interaction_residual_margin_bridge", "controls"})
        self.assertEqual(report["control_count"], 9)
        pairs = sum(len(c["pairs"]) for c in report["controls"])
        self.assertGreater(pairs, 0)
        self.assertEqual(report["diagnostic_pair_count"], pairs)
        self.assertEqual(report["pair_branch_count"], 2*pairs)
        self.assertEqual(report["coordinate62_accounts"], 2*pairs)
        count = 0
        for _, _, _, account, _, _ in self.branches():
            indices = [r["coordinate"] for r in account["selected_coordinates"]]
            self.assertEqual(indices, sorted(set(indices)))
            self.assertIn(62, indices)
            self.assertTrue(8 <= len(indices) <= 25)
            count += len(indices)
        self.assertEqual(report["selected_coordinate_accounts"], count)
        self.assertEqual(report["weighted_boundary_component_count"], 2*count)

    def test_nonadmission_and_accounting_scope(self):
        for key in ("native_dispatch", "decoder_dispatch", "prefix_dispatch", "admission_dispatch",
                    "rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "upstream_causality_claimed", "rounding_only_attribution", "candidate_admitted",
                    "policy_adopted", "successor_published", "binary64_internal_stages_substituted",
                    "scalar_split_causal_allocation", "interaction_split_causal_allocation",
                    "boundary_split_causal_allocation", "RTL_dispatch", "GPU_dispatch",
                    "hardware_dispatch", "simulation_dispatch"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "accounting closure only", "not causal attribution",
                       "not rounding-only", "Normal independent Host Reviewer REQUIRED",
                       "not a binary64 input/attention/MLP decomposition"):
            self.assertIn(phrase, d.BOUNDARY)
        self.assertIn("not separately identifiable", self.e["report"]["remainder_scope"])
