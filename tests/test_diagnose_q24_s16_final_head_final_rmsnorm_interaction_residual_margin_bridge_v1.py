"""Independent retained-bit and exact-algebra oracles for interaction accounting."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_interaction_residual_margin_bridge_v1 as d


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


class InteractionResidualMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, old, dc, sc in zip(
                self.e["report"]["controls"], self.e["margin_report"]["controls"],
                self.e["direct_report"]["controls"], self.e["scalar_report"]["controls"], strict=True):
            for pair, original, dp, sp in zip(control["pairs"], old["pairs"], dc["pairs"],
                                              sc["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield (control, pair, branch, account, original["branches"][branch],
                           dp["branches"][branch], sp["branches"][branch],
                           sc["global_scalar_accounts"][branch])

    @lru_cache(maxsize=None)
    def vector(self, label, stage):
        archive = self.e["reference_archive"] if label == "reference" else self.e["archives"][label]
        return [half(v) for v in archive[stage]]

    @lru_cache(maxsize=None)
    def terminal(self, branch):
        return (self.vector("reference", "stage18") if branch == "fp16"
                else [double(v) for v in self.e["binary64"]])

    @lru_cache(maxsize=None)
    def oracle(self, label, branch):
        a = {k: self.vector(label, k) for k in d.residual.STAGES}
        r = {k: self.vector("reference", k) for k in d.residual.STAGES}
        components, energies = [], [Fraction() for _ in d.HIDDEN_COMPONENTS]
        for i, terminal in enumerate(self.terminal(branch)):
            q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
            parts = (
                a["input_hidden"][i]-r["input_hidden"][i],
                a["stage11"][i]-r["stage11"][i], a["stage17"][i]-r["stage17"][i],
                q-a["input_hidden"][i]-a["stage11"][i]-a["stage17"][i],
                r["input_hidden"][i]+r["stage11"][i]+r["stage17"][i]-r["stage18"][i],
                a["stage18"][i]-q, r["stage18"][i]-terminal,
            )
            self.assertEqual(sum(parts), a["stage18"][i]-terminal)
            components.append(parts)
            for j, value in enumerate(parts):
                energies[j] += value*(a["stage18"][i]+terminal)/896
        return components, energies

    def assert_mass(self, observed, values):
        values = list(values)
        signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
        self.assertEqual(observed, {"signed": str(signed), "absolute": str(absolute),
                                   "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_cross_products_independent_retained_bit_oracle(self):
        weights = [half(v) for v in self.e["weight_array"].view("<u2")]
        for control, pair, branch, account, original, _, _, scalar in self.branches():
            components, energies = self.oracle(control["control"], branch)
            qa, qr = (Fraction(n["radicand"]) for n in (scalar["actual_norm"], scalar["reference_norm"]))
            sa, sr = (Fraction(n["inverse_norm_anchor"])
                      for n in (scalar["actual_norm"], scalar["reference_norm"]))
            denominator = qa*qr*(sa+sr)
            scalars = [-v/denominator for v in energies]
            scalars.extend((qr*(sa*sa*qa-1)/denominator, -qa*(sr*sr*qr-1)/denominator))
            left, right = ([half(v) for v in self.e["rows"][pair[k]].view("<u2")]
                           for k in ("left_id", "right_id"))
            for row, old in zip(account["selected_coordinates"], original["selected_coordinates"], strict=True):
                i = row["coordinate"]
                factor = (left[i]-right[i])*weights[i]
                self.assertEqual(Fraction(row["row_difference_times_norm_weight"]), factor)
                for key, c in zip(d.HIDDEN_COMPONENTS, components[i], strict=True):
                    for column, s in zip(d.SCALE_COMPONENTS, scalars, strict=True):
                        self.assertEqual(Fraction(row["weighted_cross_products"][key][column]), factor*c*s)
                self.assertEqual(row["interaction_weighted_term"], old["weighted_terms"]["interaction"])

    def test_scalar_anchor_and_defect_disclosures_independent_oracle(self):
        for control in self.e["scalar_report"]["controls"]:
            for branch, scalar in control["global_scalar_accounts"].items():
                for values, norm in (
                    (self.vector(control["control"], "stage18"), scalar["actual_norm"]),
                    (self.terminal(branch), scalar["reference_norm"]),
                ):
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
                self.assertEqual(list(scalar["scale_components"])[-2:], list(d.scale.DEFECT_COMPONENTS))

    def test_coordinate_row_column_and_grid_masses(self):
        for _, _, _, account, _, _, _, _ in self.branches():
            for row in account["selected_coordinates"]:
                grid = row["weighted_cross_products"]
                for k, cells in grid.items():
                    self.assert_mass(row["hidden_component_totals"][k], map(Fraction, cells.values()))
                for j in d.SCALE_COMPONENTS:
                    self.assert_mass(row["scalar_component_totals"][j],
                                     (Fraction(cells[j]) for cells in grid.values()))
                self.assert_mass(row["weighted_cross_product_mass"],
                                 (Fraction(v) for cells in grid.values() for v in cells.values()))
                self.assertEqual(row["weighted_cross_product_mass"]["signed"], row["interaction_weighted_term"])

    def test_selected_cell_totals_and_cancellation(self):
        for _, _, _, account, original, _, _, _ in self.branches():
            rows, total = account["selected_coordinates"], account["interaction_accounting"]
            targets = [Fraction(row["interaction_weighted_term"]) for row in rows]
            absolute = Fraction()
            for k in d.HIDDEN_COMPONENTS:
                for j in d.SCALE_COMPONENTS:
                    values = [Fraction(row["weighted_cross_products"][k][j]) for row in rows]
                    self.assert_mass(total["cross_product_totals"][k][j], values)
                    absolute += sum(map(abs, values))
            self.assert_mass(total["selected_interaction"], targets)
            self.assertEqual(Fraction(total["cross_product_absolute_sum"]), absolute)
            self.assertEqual(Fraction(total["within_coordinate_cancellation_mass"]),
                             absolute-sum(map(abs, targets)))
            self.assertEqual(Fraction(total["across_coordinate_cancellation_mass"]),
                             sum(map(abs, targets))-abs(sum(targets)))
            self.assertEqual(Fraction(total["total_cross_product_cancellation_mass"]), absolute-abs(sum(targets)))
            self.assertEqual(Fraction(original["accounting"]["selected_term_totals"]["interaction"]["signed"]),
                             sum(targets))

    def test_retained_pairs_coordinates_and_remainders_unchanged(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_scalar_scale_residual_margin_bridge"],
                      self.e["scalar_report"])
        for control, geometry in zip(self.e["report"]["controls"], self.e["geometry"]["controls"], strict=True):
            self.assertEqual([(p["left_id"], p["right_id"], p["roles"]) for p in control["pairs"]],
                             [(a, b, roles) for (a, b), roles in d.margin.contributions.pairs_for(geometry).items()])
        for _, _, _, account, original, _, _, _ in self.branches():
            self.assertEqual([(r["coordinate"], r["selection_reasons"]) for r in account["selected_coordinates"]],
                             [(r["coordinate"], r["selection_reasons"]) for r in original["selected_coordinates"]])
            self.assertIs(account["unchanged_margin_accounting"], original["accounting"])
            totals = original["accounting"]
            signed = sum(Fraction(v["signed"]) for v in totals["selected_term_totals"].values())
            self.assertEqual(signed+Fraction(totals["unselected_coordinate_signed_remainder"])
                             +Fraction(totals["head_boundary_remainder_change"]),
                             Fraction(totals["retained_margin_change"]))
            self.assertGreaterEqual(Fraction(totals["unselected_coordinate_absolute_remainder"]),
                                    abs(Fraction(totals["unselected_coordinate_signed_remainder"])))

    def test_independent_reference_and_coordinate62_boundaries(self):
        self.assertNotEqual(self.terminal("fp16"), self.terminal("binary64"))
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_"+branch], original[branch])
        nonzero = False
        for _, _, branch, account, _, old, _, _ in self.branches():
            self.assertIn(62, [r["coordinate"] for r in account["selected_coordinates"]])
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            self.assertEqual(account["residual_internal_reference"], "original_input_L23_fp16")
            for row in old["selected_coordinates"]:
                v = Fraction(row["hidden_components"]["fp16_to_branch_terminal_remainder"])
                if branch == "fp16":
                    self.assertEqual(v, 0)
                else:
                    nonzero |= v != 0
        self.assertTrue(nonzero)

    def fixture(self, components, scalars, dw=Fraction(2, 3), weight=Fraction(-3, 5)):
        components = dict(zip(d.HIDDEN_COMPONENTS, map(Fraction, components), strict=True))
        scalars = dict(zip(d.SCALE_COMPONENTS, map(Fraction, scalars), strict=True))
        dh, ds = sum(components.values()), sum(scalars.values())
        r, sr = Fraction(2), Fraction(4)
        factor = dw*weight
        reasons = ["required_coordinate_62"]
        row = {
            "coordinate": 62, "selection_reasons": reasons, "left_weight": str(dw), "right_weight": "0",
            "row_difference": str(dw),
            "hidden_bridge": {"coordinate": 62, "actual_hidden": str(r+dh), "reference_hidden": str(r),
                              "weight": str(weight), "hidden_delta": str(dh), "inverse_norm_anchor_delta": str(ds),
                              "direct_hidden": str(weight*sr*dh), "global_scale": str(weight*r*ds),
                              "interaction": str(weight*dh*ds)},
            "weighted_terms": {"direct_hidden": str(factor*sr*dh), "global_scale": str(factor*r*ds),
                               "interaction": str(factor*dh*ds)},
        }
        dr = {"coordinate": 62, "reference_inverse_norm_anchor": str(sr),
              "weight_times_reference_anchor_times_row_difference": str(factor*sr),
              "hidden_components": {k: str(v) for k, v in components.items()},
              "weighted_components": {k: str(factor*sr*v) for k, v in components.items()},
              "direct_hidden_weighted_term": str(factor*sr*dh)}
        sc = {"coordinate": 62, "selection_reasons": reasons,
              "selected_residual_energy": {"coordinate": 62, "hidden_components": dr["hidden_components"]},
              "row_difference_times_weight_times_reference_hidden": str(factor*r),
              "weighted_components": {k: str(factor*r*v) for k, v in scalars.items()},
              "global_scale_weighted_term": str(factor*r*ds)}
        scalar = {"actual_norm": {"inverse_norm_anchor": str(sr+ds)},
                  "reference_norm": {"inverse_norm_anchor": str(sr)},
                  "inverse_norm_anchor_delta": str(ds),
                  "scale_components": {k: str(v) for k, v in scalars.items()}}
        return row, dr, sc, scalar

    def test_zero_hidden_delta_retains_cancelling_cells(self):
        row = d.split_coordinate(*self.fixture([1, -1, 0, 0, 0, 0, 0], [1]+[0]*8))
        self.assertEqual(row["interaction_weighted_term"], "0")
        self.assertGreater(Fraction(row["weighted_cross_product_mass"]["absolute"]), 0)
        self.assertEqual(row["weighted_cross_product_mass"]["absolute"],
                         row["weighted_cross_product_mass"]["cancellation_absolute_mass"])

    def test_zero_scale_delta_retains_anchor_defect_columns(self):
        row = d.split_coordinate(*self.fixture([1]+[0]*6, [0]*7+[1, -1]))
        self.assertEqual(row["interaction_weighted_term"], "0")
        for column in d.scale.DEFECT_COMPONENTS:
            self.assertGreater(Fraction(row["scalar_component_totals"][column]["absolute"]), 0)

    def test_zero_row_difference_and_norm_weight(self):
        for kw in ({"dw": Fraction()}, {"weight": Fraction()}):
            row = d.split_coordinate(*self.fixture([1]*7, [1]*9, **kw))
            self.assertEqual(row["weighted_cross_product_mass"]["absolute"], "0")
            self.assertTrue(all(Fraction(v) == 0 for cells in row["weighted_cross_products"].values()
                                for v in cells.values()))

    def test_signed_non_dyadic_cross_product_closure(self):
        c = [Fraction((-1)**i, i+1) for i in range(7)]
        s = [Fraction((-1)**i, i+2) for i in range(9)]
        row = d.split_coordinate(*self.fixture(c, s))
        self.assertEqual(Fraction(row["interaction_weighted_term"]), -Fraction(2, 5)*sum(c)*sum(s))
        self.assertGreater(Fraction(row["weighted_cross_product_mass"]["cancellation_absolute_mass"]), 0)

    def live_coordinate(self):
        _, _, _, _, original, dr, sr, scalar = next(self.branches())
        return original["selected_coordinates"][0], dr["selected_coordinates"][0], sr["selected_coordinates"][0], scalar

    def test_coordinate_and_operand_splices_refused(self):
        source = self.live_coordinate()
        for i in (-1, 896, True):
            row = {**source[0], "coordinate": i}
            with self.assertRaises(ValueError):
                d.split_coordinate(row, *source[1:])
        for key in ("actual_hidden", "reference_hidden", "weight", "hidden_delta",
                    "inverse_norm_anchor_delta", "interaction", "direct_hidden", "global_scale"):
            args = deepcopy(source)
            args[0]["hidden_bridge"][key] = str(Fraction(args[0]["hidden_bridge"][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)
        for key in ("left_weight", "row_difference"):
            args = deepcopy(source)
            args[0][key] = str(Fraction(args[0][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)

    def test_retained_component_and_target_splices_refused(self):
        source = self.live_coordinate()
        for index, field, key in (
            (1, "hidden_components", "input_hidden"),
            (1, "weighted_components", "attention_stage11"),
            (2, "weighted_components", "actual_scalar_anchor_defect"),
            (3, "scale_components", "negative_reference_scalar_anchor_defect"),
            (0, "weighted_terms", "interaction"),
        ):
            args = deepcopy(source)
            args[index][field][key] = str(Fraction(args[index][field][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(*args)
        args = deepcopy(source)
        args[2]["selection_reasons"] = []
        with self.assertRaises(ValueError):
            d.split_coordinate(*args)

    def test_selected_total_mutation_refused(self):
        _, _, _, account, original, _, _, _ = next(self.branches())
        rows = deepcopy(account["selected_coordinates"])
        rows[0]["weighted_cross_products"]["input_hidden"]["attention_stage11"] = "999"
        with self.assertRaises(ValueError):
            d.selected_account(rows, original["accounting"])

    def test_report_linkage_census_and_scalar_allocation_mutations_refused(self):
        changed = {**self.e, "scalar_report": {**self.e["scalar_report"],
                   "retained_final_rmsnorm_direct_hidden_residual_margin_bridge": {}}}
        with self.assertRaises(ValueError):
            d.report(changed)
        for field, value in (("controls", list(reversed(self.e["scalar_report"]["controls"]))),
                             ("coordinate62_accounts", 0)):
            changed = {**self.e, "scalar_report": {**self.e["scalar_report"], field: value}}
            with self.assertRaises(ValueError):
                d.report(changed)
        first = self.e["scalar_report"]["controls"][0]
        scalar = first["global_scalar_accounts"]["fp16"]
        values = dict(scalar["scale_components"])
        values["input_hidden"] = str(Fraction(values["input_hidden"])+1)
        values["attention_stage11"] = str(Fraction(values["attention_stage11"])-1)
        changed_control = {**first, "global_scalar_accounts": {
            **first["global_scalar_accounts"], "fp16": {**scalar, "scale_components": values}}}
        changed = {**self.e, "scalar_report": {**self.e["scalar_report"], "controls": [
            changed_control, *self.e["scalar_report"]["controls"][1:]]}}
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
            lambda: d.scale.check(), lambda: d.scale.measure(), lambda: d.scale.focused_tests(None),
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
        target = d.parent.OUTPUT / "forbidden-interaction-bridge"
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
            "lineage_separation", "selected_coordinate_accounts", "weighted_cross_product_count",
            "cross_product_shape", "reference_scope", "identity", "allocation_scope",
            "retained_final_rmsnorm_scalar_scale_residual_margin_bridge", "controls"})
        self.assertEqual(report["control_count"], 9)
        self.assertEqual(report["cross_product_shape"], {"hidden_components": 7, "scalar_components": 9})
        pairs = sum(len(c["pairs"]) for c in report["controls"])
        self.assertGreater(pairs, 0)
        self.assertEqual(report["diagnostic_pair_count"], pairs)
        self.assertEqual(report["pair_branch_count"], 2*pairs)
        self.assertEqual(report["coordinate62_accounts"], 2*pairs)
        count = 0
        for _, _, _, account, _, _, _, _ in self.branches():
            indices = [r["coordinate"] for r in account["selected_coordinates"]]
            self.assertEqual(indices, sorted(set(indices)))
            self.assertIn(62, indices)
            self.assertTrue(8 <= len(indices) <= 25)
            for row in account["selected_coordinates"]:
                self.assertEqual(list(row["weighted_cross_products"]), list(d.HIDDEN_COMPONENTS))
                self.assertTrue(all(list(cells) == list(d.SCALE_COMPONENTS)
                                    for cells in row["weighted_cross_products"].values()))
            count += len(indices)
        self.assertEqual(report["selected_coordinate_accounts"], count)
        self.assertEqual(report["weighted_cross_product_count"], count*63)

    def test_nonadmission_and_accounting_scope(self):
        for key in ("native_dispatch", "decoder_dispatch", "prefix_dispatch", "admission_dispatch",
                    "rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "upstream_causality_claimed", "rounding_only_attribution", "candidate_admitted",
                    "policy_adopted", "successor_published", "binary64_internal_stages_substituted",
                    "scalar_split_causal_allocation", "interaction_split_causal_allocation",
                    "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "simulation_dispatch"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "accounting closure only", "not causal attribution",
                       "not rounding-only", "Normal independent Host Reviewer REQUIRED",
                       "not a binary64 input/attention/MLP decomposition"):
            self.assertIn(phrase, d.BOUNDARY)
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])


if __name__ == "__main__":
    unittest.main()
