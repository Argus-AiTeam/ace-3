"""Independent retained-bit, scalar and algebra oracles for the scale split."""

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import io
import json
import os
import struct
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_scalar_scale_residual_margin_bridge_v1 as d


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


class ScalarScaleResidualMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, old in zip(self.e["report"]["controls"], self.e["margin_report"]["controls"], strict=True):
            for pair, original in zip(control["pairs"], old["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield control, pair, branch, account, original["branches"][branch]

    @lru_cache(maxsize=None)
    def vector(self, label, stage):
        archive = self.e["reference_archive"] if label == "reference" else self.e["archives"][label]
        return [half(word) for word in archive[stage]]

    @lru_cache(maxsize=None)
    def terminal(self, branch):
        return (self.vector("reference", "stage18") if branch == "fp16"
                else [double(v) for v in self.e["binary64"]])

    @lru_cache(maxsize=None)
    def oracle(self, label, branch):
        a = {k: self.vector(label, k) for k in d.residual.STAGES}
        r = {k: self.vector("reference", k) for k in d.residual.STAGES}
        result = []
        for i, terminal in enumerate(self.terminal(branch)):
            q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
            components = (
                a["input_hidden"][i]-r["input_hidden"][i], a["stage11"][i]-r["stage11"][i],
                a["stage17"][i]-r["stage17"][i],
                q-a["input_hidden"][i]-a["stage11"][i]-a["stage17"][i],
                r["input_hidden"][i]+r["stage11"][i]+r["stage17"][i]-r["stage18"][i],
                a["stage18"][i]-q, r["stage18"][i]-terminal,
            )
            result.append((components, a["stage18"][i], terminal))
        return result

    def test_global_energy_independent_retained_bit_oracle(self):
        for control in self.e["report"]["controls"]:
            for branch, scalar in control["global_scalar_accounts"].items():
                oracle = self.oracle(control["control"], branch)
                for index, key in enumerate(d.ENERGY_COMPONENTS):
                    for field, multiplier in (
                        ("energy_components", lambda a, r: a+r),
                        ("linear_energy_components", lambda a, r: 2*r),
                        ("quadratic_energy_components", lambda a, r: a-r),
                    ):
                        values = [parts[index]*multiplier(a, r)/d.WIDTH for parts, a, r in oracle]
                        total = scalar["residual_energy_totals"][field][key]
                        self.assertEqual(Fraction(total["signed"]), sum(values))
                        self.assertEqual(Fraction(total["absolute"]), sum(map(abs, values)))
                        self.assertEqual(Fraction(total["cancellation_absolute_mass"]),
                                         sum(map(abs, values))-abs(sum(values)))
                expected = sum((a*a-r*r)/d.WIDTH for _, a, r in oracle)
                self.assertEqual(Fraction(scalar["global_mean_square_delta"]), expected)

    def test_scalar_anchors_independent_decimal_oracle(self):
        for control in self.e["report"]["controls"]:
            for branch, scalar in control["global_scalar_accounts"].items():
                for values, norm in ((self.vector(control["control"], "stage18"), scalar["actual_norm"]),
                                     (self.terminal(branch), scalar["reference_norm"])):
                    radicand = sum(v*v for v in values)/d.WIDTH + Fraction(1, 1000000)
                    self.assertEqual(Fraction(norm["radicand"]), radicand)
                    with localcontext() as context:
                        context.prec = 70
                        expected = 1/(Decimal(radicand.numerator)/Decimal(radicand.denominator)).sqrt()
                        anchor = Fraction(norm["inverse_norm_anchor"])
                        value = Decimal(anchor.numerator)/Decimal(anchor.denominator)
                        self.assertLess(abs(value/expected-1), Decimal("4e-16"))
                    self.assertEqual(Fraction(norm["inverse_square_identity_defect"]), anchor*anchor*radicand-1)
                    self.assertEqual(Fraction(norm["radicand_binary64_conversion_delta"]),
                                     double(float(radicand))-radicand)

    def test_scalar_identity_independent_cross_multiplication(self):
        for control in self.e["report"]["controls"]:
            for scalar in control["global_scalar_accounts"].values():
                an, rn = scalar["actual_norm"], scalar["reference_norm"]
                sa, sr = (Fraction(n["inverse_norm_anchor"]) for n in (an, rn))
                qa, qr = (Fraction(n["radicand"]) for n in (an, rn))
                denominator = qa*qr*(sa+sr)
                components = {k: Fraction(v) for k, v in scalar["scale_components"].items()}
                for k in d.ENERGY_COMPONENTS:
                    self.assertEqual(components[k]*denominator,
                                     -Fraction(scalar["residual_energy_totals"]["energy_components"][k]["signed"]))
                self.assertEqual(components[d.DEFECT_COMPONENTS[0]]*denominator, qr*(sa*sa*qa-1))
                self.assertEqual(components[d.DEFECT_COMPONENTS[1]]*denominator, -qa*(sr*sr*qr-1))
                self.assertEqual(sum(components.values()), sa-sr)
                self.assertEqual(Fraction(scalar["component_mass"]["absolute"]), sum(map(abs, components.values())))
                self.assertEqual(Fraction(scalar["component_mass"]["cancellation_absolute_mass"]),
                                 sum(map(abs, components.values()))-abs(sa-sr))

    def test_weighted_components_independent_row_bit_oracle(self):
        weights = [half(v) for v in self.e["weight_array"].view("<u2")]
        for control, pair, branch, account, original in self.branches():
            scalar = control["global_scalar_accounts"][branch]
            left, right = ([half(v) for v in self.e["rows"][pair[k]].view("<u2")]
                           for k in ("left_id", "right_id"))
            for row, old in zip(account["selected_coordinates"], original["selected_coordinates"], strict=True):
                i = row["coordinate"]
                factor = (left[i]-right[i])*weights[i]*self.terminal(branch)[i]
                self.assertEqual(Fraction(row["row_difference_times_weight_times_reference_hidden"]), factor)
                for k in d.COMPONENTS:
                    self.assertEqual(Fraction(row["weighted_components"][k]),
                                     factor*Fraction(scalar["scale_components"][k]))
                self.assertEqual(row["global_scale_weighted_term"], old["weighted_terms"]["global_scale"])
                self.assertEqual(Fraction(row["weighted_component_mass"]["signed"]),
                                 Fraction(row["global_scale_weighted_term"]))

    def test_selected_energy_and_unselected_remainders(self):
        for control, _, branch, account, _ in self.branches():
            oracle = self.oracle(control["control"], branch)
            selected = {row["coordinate"]: row for row in account["selected_coordinates"]}
            for i, row in selected.items():
                parts, a, r = oracle[i]
                energy = row["selected_residual_energy"]
                for k, value in zip(d.ENERGY_COMPONENTS, parts, strict=True):
                    self.assertEqual(Fraction(energy["hidden_components"][k]), value)
                    self.assertEqual(Fraction(energy["energy_components"][k]), value*(a+r)/d.WIDTH)
                    self.assertEqual(Fraction(energy["linear_energy_components"][k]), value*2*r/d.WIDTH)
                    self.assertEqual(Fraction(energy["quadratic_energy_components"][k]), value*(a-r)/d.WIDTH)
            for index, k in enumerate(d.ENERGY_COMPONENTS):
                values = [parts[index]*(a+r)/d.WIDTH for i, (parts, a, r) in enumerate(oracle)
                          if i not in selected]
                total = account["residual_energy_selected_and_unselected"][k]
                self.assertEqual(Fraction(total["unselected_signed_remainder"]), sum(values))
                self.assertEqual(Fraction(total["unselected_absolute_remainder"]), sum(map(abs, values)))

    def test_selected_mass_and_cancellation(self):
        for _, _, _, account, original in self.branches():
            totals = account["global_scale_accounting"]
            rows = account["selected_coordinates"]
            target = [Fraction(row["global_scale_weighted_term"]) for row in rows]
            absolute = Fraction()
            for k in d.COMPONENTS:
                values = [Fraction(row["weighted_components"][k]) for row in rows]
                actual = totals["component_totals"][k]
                self.assertEqual(Fraction(actual["signed"]), sum(values))
                self.assertEqual(Fraction(actual["absolute"]), sum(map(abs, values)))
                self.assertEqual(Fraction(actual["cancellation_absolute_mass"]),
                                 sum(map(abs, values))-abs(sum(values)))
                absolute += sum(map(abs, values))
            self.assertEqual(Fraction(totals["within_coordinate_cancellation_mass"]),
                             absolute-sum(map(abs, target)))
            self.assertEqual(Fraction(totals["across_coordinate_cancellation_mass"]),
                             sum(map(abs, target))-abs(sum(target)))
            self.assertEqual(Fraction(totals["total_component_cancellation_mass"]), absolute-abs(sum(target)))
            self.assertEqual(Fraction(totals["selected_global_scale"]["signed"]),
                             Fraction(original["accounting"]["selected_term_totals"]["global_scale"]["signed"]))

    def test_original_pairs_selections_and_margin_remainders_unchanged(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_direct_hidden_residual_margin_bridge"],
                      self.e["direct_report"])
        for control, geometry in zip(self.e["report"]["controls"], self.e["geometry"]["controls"], strict=True):
            pairs = d.margin.contributions.pairs_for(geometry)
            self.assertEqual([(p["left_id"], p["right_id"]) for p in control["pairs"]], list(pairs))
        for _, _, _, account, original in self.branches():
            self.assertEqual([r["coordinate"] for r in account["selected_coordinates"]],
                             [r["coordinate"] for r in original["selected_coordinates"]])
            self.assertIs(account["unchanged_margin_accounting"], original["accounting"])
            totals = original["accounting"]
            signed = sum(Fraction(v["signed"]) for v in totals["selected_term_totals"].values())
            self.assertEqual(signed+Fraction(totals["unselected_coordinate_signed_remainder"])
                             +Fraction(totals["head_boundary_remainder_change"]),
                             Fraction(totals["retained_margin_change"]))

    def test_independent_reference_boundaries(self):
        self.assertNotEqual(self.terminal("fp16"), self.terminal("binary64"))
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_"+branch], original[branch])
        nonzero = False
        for _, _, branch, account, _ in self.branches():
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            self.assertEqual(account["residual_internal_reference"], "original_input_L23_fp16")
            for row in account["selected_coordinates"]:
                value = Fraction(row["selected_residual_energy"]["hidden_components"]["fp16_to_branch_terminal_remainder"])
                if branch == "fp16":
                    self.assertEqual(value, 0)
                else:
                    nonzero |= value != 0
        self.assertTrue(nonzero)

    def test_zero_energy_nonzero_anchor_defects_and_cancellation(self):
        energy = {"global_mean_square_delta": "0", "totals": {"energy_components": {
            k: {"signed": "0"} for k in d.ENERGY_COMPONENTS}}}
        an = {"radicand": "4", "inverse_norm_anchor": "1/2", "inverse_square_identity_defect": "0"}
        rn = {"radicand": "4", "inverse_norm_anchor": "1/4", "inverse_square_identity_defect": "-3/4"}
        result = d.scalar_account(energy, an, rn)
        self.assertEqual(Fraction(result["inverse_norm_anchor_delta"]), Fraction(1, 4))
        self.assertEqual(Fraction(result["scale_components"][d.DEFECT_COMPONENTS[1]]), Fraction(1, 4))
        energy["totals"]["energy_components"]["input_hidden"]["signed"] = "7"
        energy["totals"]["energy_components"]["attention_stage11"]["signed"] = "-7"
        result = d.scalar_account(energy, an, an)
        self.assertEqual(result["component_mass"]["signed"], "0")
        self.assertGreater(Fraction(result["component_mass"]["absolute"]), 0)

    def test_zero_tied_row_difference(self):
        control, _, branch, account, original = next(self.branches())
        row = deepcopy(original["selected_coordinates"][0])
        row["right_weight"], row["row_difference"] = row["left_weight"], "0"
        row["weighted_terms"]["global_scale"] = "0"
        old = self.e["direct_report"]["controls"][0]["pairs"][0]["branches"][branch]["selected_coordinates"][0]
        result = d.split_coordinate(row, old, control["global_scalar_accounts"][branch],
                                    account["selected_coordinates"][0]["selected_residual_energy"],
                                    self.terminal(branch)[row["coordinate"]], Fraction(row["hidden_bridge"]["weight"]))
        self.assertTrue(all(Fraction(v) == 0 for v in result["weighted_components"].values()))

    def test_coordinate_operand_and_scale_splices_refused(self):
        control, _, branch, account, original = next(self.branches())
        old = self.e["direct_report"]["controls"][0]["pairs"][0]["branches"][branch]["selected_coordinates"][0]
        energy = account["selected_coordinates"][0]["selected_residual_energy"]
        source = original["selected_coordinates"][0]
        for key in ("reference_hidden", "weight", "inverse_norm_anchor_delta", "global_scale"):
            row = deepcopy(source)
            row["hidden_bridge"][key] = str(Fraction(row["hidden_bridge"][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(row, old, control["global_scalar_accounts"][branch], energy,
                                   self.terminal(branch)[source["coordinate"]], Fraction(source["hidden_bridge"]["weight"]))
        for coordinate in (-1, 896, True):
            with self.assertRaises(ValueError):
                d.split_coordinate({**source, "coordinate": coordinate}, old,
                                   control["global_scalar_accounts"][branch], energy, Fraction(), Fraction())
        row = deepcopy(source)
        row["weighted_terms"]["global_scale"] = "1"
        with self.assertRaises(ValueError):
            d.split_coordinate(row, old, control["global_scalar_accounts"][branch], energy,
                               self.terminal(branch)[source["coordinate"]], Fraction(source["hidden_bridge"]["weight"]))
        changed = deepcopy(account["selected_coordinates"])
        changed[0]["weighted_components"]["attention_stage11"] = "999"
        with self.assertRaises(ValueError):
            d.selected_account(changed, original["accounting"])

    def test_scalar_and_direct_report_mutations_refused(self):
        control = self.e["report"]["controls"][0]
        scalar = control["global_scalar_accounts"]["fp16"]
        energy = {"global_mean_square_delta": scalar["global_mean_square_delta"],
                  "totals": scalar["residual_energy_totals"]}
        for key in ("radicand", "inverse_norm_anchor", "inverse_square_identity_defect"):
            changed = {**scalar["actual_norm"], key: "999"}
            with self.assertRaises(ValueError):
                d.scalar_account(energy, changed, scalar["reference_norm"])
        changed = {**self.e, "direct_report": {**self.e["direct_report"],
                    "retained_final_rmsnorm_logit_margin_bridge": {}}}
        with self.assertRaises(ValueError):
            d.report(changed)
        changed = {**self.e, "margin_report": {**self.e["margin_report"],
                    "controls": list(reversed(self.e["margin_report"]["controls"]))}}
        with self.assertRaises(ValueError):
            d.report(changed)

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
        calls = (
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.residual.local.local_reference(None), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.direct.check(), lambda: d.direct.measure(), lambda: d.margin.check(),
            lambda: d.hidden.check(), lambda: d.residual.check(),
            lambda: d.margin.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden-native"]), lambda: os.system("forbidden-native"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-scalar-scale-bridge"
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
        self.assertEqual(text, json.dumps(self.e["report"], sort_keys=True, allow_nan=False)+"\n")
        self.assertEqual(text.count("\n"), 1)

    def test_report_shape_and_census(self):
        report = self.e["report"]
        self.assertEqual(set(report), {
            "control_count", "diagnostic_pair_count", "pair_branch_count", "coordinate62_accounts",
            "lineage_separation", "selected_coordinate_accounts", "weighted_component_count",
            "global_energy_coordinate_count", "reference_scope", "energy_identity", "scalar_identity",
            "allocation_scope", "retained_final_rmsnorm_direct_hidden_residual_margin_bridge", "controls"})
        self.assertEqual(report["control_count"], 9)
        self.assertEqual(report["global_energy_coordinate_count"], 9*2*896)
        pairs = sum(len(c["pairs"]) for c in report["controls"])
        self.assertGreater(pairs, 0)
        self.assertEqual(report["diagnostic_pair_count"], pairs)
        self.assertEqual(report["pair_branch_count"], 2*pairs)
        self.assertEqual(report["coordinate62_accounts"], 2*pairs)
        count = 0
        for _, _, _, account, _ in self.branches():
            self.assertIn(62, [r["coordinate"] for r in account["selected_coordinates"]])
            count += len(account["selected_coordinates"])
        self.assertEqual(report["selected_coordinate_accounts"], count)
        self.assertEqual(report["weighted_component_count"], count*9)

    def test_nonadmission_and_reference_scope(self):
        for key in ("rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "upstream_causality_claimed", "rounding_only_attribution", "candidate_admitted",
                    "policy_adopted", "successor_published", "binary64_internal_stages_substituted",
                    "scalar_split_causal_allocation"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "not causal percentages", "not rounding-only",
                       "Normal independent Host Reviewer REQUIRED"):
            self.assertIn(phrase, d.BOUNDARY)
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])
        self.assertIn("not a binary64 input/attention/MLP decomposition", d.BOUNDARY)


if __name__ == "__main__":
    unittest.main()
