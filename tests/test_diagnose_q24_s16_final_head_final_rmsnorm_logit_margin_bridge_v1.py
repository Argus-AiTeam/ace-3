"""Independent retained-bit and algebra oracles for the local margin bridge."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_logit_margin_bridge_v1 as d


EVIDENCE = None


def bits(word, fraction_bits, exponent_bits, bias):
    exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    mantissa = word & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        raise ValueError("nonfinite oracle input")
    value = Fraction(mantissa if exponent == 0 else (1 << fraction_bits) + mantissa)
    value *= Fraction(2) ** ((1 if exponent == 0 else exponent) - bias - fraction_bits)
    return -value if word >> (fraction_bits + exponent_bits) else value


def half(word):
    return bits(int(word), 10, 5, 15)


def double(value):
    return bits(struct.unpack("<Q", struct.pack("<d", value))[0], 52, 11, 1023)


class MarginBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, bridge in zip(self.e["report"]["controls"],
                                   self.e["hidden_report"]["controls"], strict=True):
            for pair in control["pairs"]:
                for branch, account in pair["branches"].items():
                    yield control["control"], bridge, pair, branch, account

    @lru_cache(maxsize=None)
    def operands(self, label, branch):
        a = [half(v) for v in self.e["archives"][label]["stage18"]]
        r = ([half(v) for v in self.e["reference_archive"]["stage18"]]
             if branch == "fp16" else [double(v) for v in self.e["binary64"]])
        ya = [half(v) for v in self.e["arrays"][label]["rmsnorm"]]
        raw = self.e["references"]["rmsnorm_" + branch]
        yr = [half(v) if branch == "fp16" else double(v) for v in raw]
        return a, r, ya, yr

    @lru_cache(maxsize=None)
    def row(self, index):
        return [half(v) for v in self.e["rows"][index].view("<u2")]

    def test_live_weighted_terms_independent_bit_oracle(self):
        w = [half(v) for v in self.e["weight_array"].view("<u2")]
        for label, bridge, pair, branch, account in self.branches():
            a, r, ya, yr = self.operands(label, branch)
            sa = Fraction(bridge["actual_norm"]["inverse_norm_anchor"])
            sr = Fraction(bridge["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
            left, right = self.row(pair["left_id"]), self.row(pair["right_id"])
            for row in account["selected_coordinates"]:
                i = row["coordinate"]
                dw = left[i] - right[i]
                ba, br = ya[i] - w[i]*a[i]*sa, yr[i] - w[i]*r[i]*sr
                expected = ((a[i]-r[i])*w[i]*sr, r[i]*w[i]*(sa-sr),
                            (a[i]-r[i])*w[i]*(sa-sr), ba-br)
                self.assertEqual(Fraction(row["left_weight"]), left[i])
                self.assertEqual(Fraction(row["right_weight"]), right[i])
                self.assertEqual(Fraction(row["row_difference"]), dw)
                for key, value in zip(d.TERMS, expected, strict=True):
                    self.assertEqual(Fraction(row["hidden_bridge"][key]), value)
                    self.assertEqual(Fraction(row["weighted_terms"][key]), value*dw)
                self.assertEqual(Fraction(row["coordinate_margin_change"]), (ya[i]-yr[i])*dw)
                self.assertEqual(Fraction(row["weighted_actual_boundary_conversion_remainder"]), ba*dw)
                self.assertEqual(Fraction(row["weighted_reference_boundary_conversion_remainder"]), br*dw)

    def test_scalar_anchors_independent_decimal_oracle(self):
        for bridge in self.e["hidden_report"]["controls"]:
            label = bridge["control"]
            a, _, _, _ = self.operands(label, "fp16")
            norms = [(a, bridge["actual_norm"])]
            norms.extend((self.operands(label, b)[1], bridge["branches"][b]["reference_norm"])
                         for b in d.BRANCHES)
            for values, norm in norms:
                mean = sum(v*v for v in values) / d.WIDTH
                radicand = mean + Fraction(1, 1000000)
                anchor = Fraction(norm["inverse_norm_anchor"])
                self.assertEqual(Fraction(norm["mean_square"]), mean)
                self.assertEqual(Fraction(norm["radicand"]), radicand)
                self.assertEqual(Fraction(norm["radicand_binary64_conversion_delta"]),
                                 double(float(radicand))-radicand)
                self.assertEqual(Fraction(norm["inverse_square_identity_defect"]),
                                 anchor*anchor*radicand-1)
                with localcontext() as context:
                    context.prec = 70
                    exact = 1 / (Decimal(radicand.numerator)/Decimal(radicand.denominator)).sqrt()
                    actual = Decimal(anchor.numerator)/Decimal(anchor.denominator)
                    self.assertLess(abs(actual/exact-1), Decimal("4e-16"))

    def test_pair_identity_and_roles_preserved(self):
        for control, geometry in zip(self.e["report"]["controls"],
                                     self.e["geometry"]["controls"], strict=True):
            expected = d.contributions.pairs_for(geometry)
            self.assertEqual([(p["left_id"], p["right_id"]) for p in control["pairs"]], list(expected))
            for pair in control["pairs"]:
                self.assertEqual(pair["roles"], expected[pair["left_id"], pair["right_id"]])
                self.assertEqual(list(pair["branches"]), list(d.BRANCHES))

    def test_selection_and_row_ranking_independent_oracle(self):
        for label, bridge, pair, branch, account in self.branches():
            _, _, ya, yr = self.operands(label, branch)
            left, right = self.row(pair["left_id"]), self.row(pair["right_id"])
            dw = [x-y for x, y in zip(left, right)]
            delta = [x-y for x, y in zip(ya, yr)]
            products = [x*y for x, y in zip(dw, delta)]
            def top(values):
                return sorted(range(d.WIDTH), key=lambda i: (-abs(values[i]), i))[:8]
            row_order = sorted(range(d.WIDTH), key=lambda i: (-abs(dw[i]), i))
            expected = {62: ["required_coordinate_62"]}
            for reason, indices in (
                ("retained_rmsnorm_delta_hotspot", top(delta)),
                ("pair_margin_delta_hotspot", top(products)),
                ("tied_head_row_difference_hotspot", top(dw)),
            ):
                for i in indices:
                    expected.setdefault(i, []).append(reason)
            self.assertEqual(pair["row_difference_hotspots"], top(dw))
            self.assertEqual([r["coordinate"] for r in account["selected_coordinates"]], sorted(expected))
            for row in account["selected_coordinates"]:
                self.assertEqual(row["selection_reasons"], sorted(expected[row["coordinate"]]))
                self.assertEqual(row["row_difference_absolute_rank"], row_order.index(row["coordinate"])+1)

    def test_margin_and_unselected_remainders_independent_oracle(self):
        for label, _, pair, branch, account in self.branches():
            _, _, ya, yr = self.operands(label, branch)
            left, right = self.row(pair["left_id"]), self.row(pair["right_id"])
            values = [(a-r)*(l-w) for a, r, l, w in zip(ya, yr, left, right)]
            chosen = {r["coordinate"] for r in account["selected_coordinates"]}
            unselected = [v for i, v in enumerate(values) if i not in chosen]
            actual_logits = self.e["arrays"][label]["logits"]
            reference = self.e["references"]["logits_"+branch]
            l, r = pair["left_id"], pair["right_id"]
            decode = half if branch == "fp16" else double
            margin = half(actual_logits[l])-half(actual_logits[r])-decode(reference[l])+decode(reference[r])
            totals = account["accounting"]
            self.assertEqual(Fraction(totals["unselected_coordinate_signed_remainder"]), sum(unselected))
            self.assertEqual(Fraction(totals["unselected_coordinate_absolute_remainder"]), sum(map(abs, unselected)))
            self.assertEqual(Fraction(totals["retained_margin_change"]), margin)
            self.assertEqual(Fraction(totals["head_boundary_remainder_change"]), margin-sum(values))
            self.assertEqual(Fraction(account["retained_pair_effect"]["exact_sum"]), sum(values))

    def test_signed_absolute_totals_and_cancellation(self):
        for _, _, _, _, account in self.branches():
            rows, totals = account["selected_coordinates"], account["accounting"]
            products = [Fraction(r["coordinate_margin_change"]) for r in rows]
            term_abs = Fraction()
            for key in d.TERMS:
                values = [Fraction(r["weighted_terms"][key]) for r in rows]
                self.assertEqual(Fraction(totals["selected_term_totals"][key]["signed"]), sum(values))
                self.assertEqual(Fraction(totals["selected_term_totals"][key]["absolute"]), sum(map(abs, values)))
                term_abs += sum(map(abs, values))
            self.assertEqual(Fraction(totals["selected_coordinate_signed_sum"]), sum(products))
            self.assertEqual(Fraction(totals["selected_coordinate_absolute_sum"]), sum(map(abs, products)))
            self.assertEqual(Fraction(totals["selected_term_absolute_sum"]), term_abs)
            self.assertEqual(Fraction(totals["selected_term_cancellation_mass"]), term_abs-abs(sum(products)))
            self.assertEqual(Fraction(totals["selected_coordinate_cancellation_mass"]),
                             sum(map(abs, products))-abs(sum(products)))
            effect = account["retained_pair_effect"]
            self.assertEqual(Fraction(totals["full_coordinate_cancellation_mass"]),
                             Fraction(effect["exact_sum_of_absolute_contributions"])-abs(Fraction(effect["exact_sum"])))
            for row in rows:
                values = [Fraction(row["weighted_terms"][key]) for key in d.TERMS]
                self.assertEqual(sum(values), Fraction(row["coordinate_margin_change"]))
                self.assertEqual(Fraction(row["absolute_term_sum"]), sum(map(abs, values)))
                self.assertEqual(Fraction(row["cancellation_absolute_mass"]),
                                 sum(map(abs, values))-abs(sum(values)))
                self.assertIs(row["exact_weighted_identity"], True)
            self.assertIs(totals["exact_margin_identity"], True)

    def test_coordinate62_and_q24_conversion_disclosure(self):
        for label, _, _, branch, account in self.branches():
            a, _, _, _ = self.operands(label, branch)
            selected = {r["coordinate"]: r for r in account["selected_coordinates"]}
            self.assertIn("required_coordinate_62", selected[62]["selection_reasons"])
            for i, row in selected.items():
                raw = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
                self.assertEqual(Fraction(row["hidden_bridge"]["actual_raw_q24_hidden"]), raw)
                self.assertEqual(Fraction(row["hidden_bridge"]["actual_q24_to_stage18_conversion"]), a[i]-raw)

    def test_reference_branches_never_cast_or_reanchored(self):
        self.assertNotEqual(self.e["hidden"]["fp16"], self.e["hidden"]["binary64"])
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_"+branch], original[branch])
            self.assertEqual(self.e["hidden"][branch], self.operands(d.parent.CONTROLS[0], branch)[1])
        for _, _, _, branch, account in self.branches():
            self.assertEqual(account["hidden_reference"], "original_input_L23_"+branch)
            self.assertEqual(account["rmsnorm_reference"], "original_input_final_attempt003_"+branch)

    def test_synthetic_negative_zero_and_cancellation(self):
        zero = [Fraction()] * d.WIDTH
        hidden = d.hidden.account(Fraction(2), Fraction(1), Fraction(3), Fraction(3),
                                  Fraction(1), Fraction(1), Fraction(2), 0, zero, zero, Fraction(2))
        row = d.weighted_account(hidden, Fraction(-1), Fraction(1))
        self.assertEqual([Fraction(row["weighted_terms"][k]) for k in d.TERMS], [-4, 2, 2, 0])
        self.assertEqual(Fraction(row["cancellation_absolute_mass"]), 8)
        row = d.weighted_account(hidden, Fraction(1), Fraction(1))
        self.assertTrue(all(Fraction(v) == 0 for v in row["weighted_terms"].values()))

    def test_selection_ties_and_union_deduplication(self):
        entries = [{"coordinate": i} for i in range(8)]
        selected = d.selected_coordinates({"hotspots": entries},
                                         {"largest_absolute_signed": entries}, list(range(d.WIDTH)))
        self.assertEqual(list(selected), [*range(8), 62])
        self.assertEqual(len(selected[0]), 3)

    def test_census_and_output_shape(self):
        report = self.e["report"]
        self.assertEqual(set(report), {
            "control_count", "diagnostic_pair_count", "pair_branch_count",
            "selected_coordinate_accounts", "weighted_term_count", "coordinate62_accounts",
            "selection", "identity", "remainder_scope", "hidden_delta_bridge",
            "cutoff_margin_report", "lineage_separation", "controls"})
        self.assertEqual(report["control_count"], 9)
        self.assertEqual([r["control"] for r in report["controls"]], list(d.parent.CONTROLS))
        pairs = sum(len(c["pairs"]) for c in report["controls"])
        accounts = sum(len(a["selected_coordinates"]) for _, _, _, _, a in self.branches())
        self.assertGreater(pairs, 0)
        self.assertEqual(report["diagnostic_pair_count"], pairs)
        self.assertEqual(report["pair_branch_count"], 2*pairs)
        self.assertEqual(report["coordinate62_accounts"], 2*pairs)
        self.assertEqual(report["selected_coordinate_accounts"], accounts)
        self.assertEqual(report["weighted_term_count"], 4*accounts)
        for _, _, _, _, account in self.branches():
            self.assertGreaterEqual(len(account["selected_coordinates"]), 8)
            self.assertLessEqual(len(account["selected_coordinates"]), 25)

    def test_history_and_threshold_mutations_refused(self):
        for key, value in (("mandatory_statuses", ["PASS"]*19),
                           ("S18_failure_indices", []), ("candidate_admitted", True),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)

    def test_input_and_source_pin_mutations_refused(self):
        for pin in (self.e["hidden_pins"][-1], d.rows.PINS["rmsnorm_hotspot_source"],
                    d.base.PINS["review"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})

    def test_tokenizer_tied_head_norm_and_epsilon_bindings(self):
        assets = self.e["assets"]
        self.assertEqual(assets["tokenizer_status"], "BOUND")
        self.assertFalse(assets["missing_tokenizer"])
        self.assertEqual(assets["tensors"]["lm_head.weight"], assets["tensors"]["model.embed_tokens.weight"])
        self.assertEqual(d.parent.norm.EPSILON_Q48, 281474977)
        self.assertEqual(d.hidden.EPSILON, Fraction(1, 1000000))
        with self.assertRaises(ValueError):
            d.hidden.validate_weight(self.e["weight_array"],
                                     {**assets["tensors"]["model.norm.weight"], "sha256": "0"*64})

    def test_local_identity_and_control_mutations_refused(self):
        _, _, _, _, account = next(self.branches())
        row = deepcopy(account["selected_coordinates"][0]["hidden_bridge"])
        row["direct_hidden"] = str(Fraction(row["direct_hidden"])+1)
        with self.assertRaises(ValueError):
            d.weighted_account(row, Fraction(1), Fraction(0))
        changed = {**self.e, "arrays": dict(reversed(list(self.e["arrays"].items())))}
        with self.assertRaises(ValueError):
            d.report(changed)
        changed_effect = {**account["retained_pair_effect"],
                          "retained_margin_change": str(Fraction(account["accounting"]["retained_margin_change"])+1)}
        with self.assertRaises(ValueError):
            d.margin_account(account["selected_coordinates"], changed_effect)

    def test_dispatch_guards(self):
        calls = (
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.hidden.residual.local.local_reference(None),
            lambda: d.parent.head.decode_array_q24(None),
            lambda: d.hidden.check(), lambda: d.hidden.measure(),
            lambda: d.rows.check(), lambda: d.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden-native"]), lambda: os.system("forbidden-native"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_write_guards(self):
        target = d.parent.OUTPUT / "forbidden-margin-bridge"
        audit = {"forbidden_calls": 0}
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_refuses_replay_output_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_one_deterministic_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value=self.e["report"]), patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        self.assertEqual(json.loads(text), self.e["report"])
        self.assertEqual(text, json.dumps(self.e["report"], sort_keys=True, allow_nan=False)+"\n")
        self.assertEqual(text.count("\n"), 1)

    def test_nonadmission_and_lineage_boundaries(self):
        for key in ("rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "upstream_causality_claimed", "rounding_only_attribution", "candidate_admitted",
                    "policy_adopted", "successor_published"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "not causal percentages", "not rounding-only",
                       "No RMSNorm, head, row-dot", "Normal independent Host Reviewer REQUIRED"):
            self.assertIn(phrase, d.BOUNDARY)
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])
        self.assertEqual(self.e["report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])


if __name__ == "__main__":
    unittest.main()
