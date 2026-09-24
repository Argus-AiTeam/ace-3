"""Independent retained-bit oracles and negative gates for the direct-hidden split."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_direct_hidden_residual_margin_bridge_v1 as d


EVIDENCE = None


def bits(word, mantissa_bits, exponent_bits, bias):
    exponent = (word >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = word & ((1 << mantissa_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        raise ValueError("nonfinite retained oracle operand")
    value = Fraction(mantissa if exponent == 0 else (1 << mantissa_bits) + mantissa)
    value *= Fraction(2) ** ((1 if exponent == 0 else exponent) - bias - mantissa_bits)
    return -value if word >> (mantissa_bits + exponent_bits) else value


def half(word):
    return bits(int(word), 10, 5, 15)


def double(value):
    return bits(struct.unpack("<Q", struct.pack("<d", value))[0], 52, 11, 1023)


class DirectHiddenResidualMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, upstream, hidden in zip(
                self.e["report"]["controls"], self.e["margin_report"]["controls"],
                self.e["hidden_report"]["controls"], strict=True):
            for pair, original in zip(control["pairs"], upstream["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield control["control"], pair, branch, account, original["branches"][branch], hidden

    @lru_cache(maxsize=None)
    def vector(self, label, stage):
        archive = self.e["reference_archive"] if label == "reference" else self.e["archives"][label]
        return [half(word) for word in archive[stage]]

    @lru_cache(maxsize=None)
    def row(self, index):
        return [half(word) for word in self.e["rows"][index].view("<u2")]

    def test_live_components_independent_retained_bit_oracle(self):
        weights = [half(word) for word in self.e["weight_array"].view("<u2")]
        binary64 = [double(value) for value in self.e["binary64"]]
        for label, pair, branch, account, original, hidden in self.branches():
            sr = Fraction(hidden["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
            left, right = self.row(pair["left_id"]), self.row(pair["right_id"])
            for row, old in zip(account["selected_coordinates"], original["selected_coordinates"], strict=True):
                i = row["coordinate"]
                a = {key: self.vector(label, key)[i] for key in d.residual.STAGES}
                r = {key: self.vector("reference", key)[i] for key in d.residual.STAGES}
                terminal = r["stage18"] if branch == "fp16" else binary64[i]
                q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
                expected = (
                    a["input_hidden"]-r["input_hidden"], a["stage11"]-r["stage11"],
                    a["stage17"]-r["stage17"],
                    q-a["input_hidden"]-a["stage11"]-a["stage17"],
                    r["input_hidden"]+r["stage11"]+r["stage17"]-r["stage18"],
                    a["stage18"]-q, r["stage18"]-terminal,
                )
                factor = weights[i]*sr*(left[i]-right[i])
                self.assertEqual(Fraction(row["weight_times_reference_anchor_times_row_difference"]), factor)
                self.assertEqual(list(row["weighted_components"]), list(d.COMPONENTS))
                for key, value in zip(d.COMPONENTS, expected, strict=True):
                    self.assertEqual(Fraction(row["hidden_components"][key]), value)
                    self.assertEqual(Fraction(row["weighted_components"][key]), value*factor)
                self.assertEqual(sum(expected), a["stage18"]-terminal)
                self.assertEqual(Fraction(row["direct_hidden_weighted_term"]), factor*sum(expected))
                self.assertEqual(row["direct_hidden_weighted_term"], old["weighted_terms"]["direct_hidden"])

    def test_boundary_subparts_independent_oracle(self):
        for label, _, _, account, _, _ in self.branches():
            archive = self.e["archives"][label]
            for row in account["selected_coordinates"]:
                i = row["coordinate"]
                f = Fraction(row["weight_times_reference_anchor_times_row_difference"])
                qi, qs, qo = (Fraction(int(archive[k+"_i"][i]), 1 << 24)
                              for k in ("input", "scratch", "output"))
                expected = (qi-self.vector(label, "input_hidden")[i],
                            qs-qi-self.vector(label, "stage11")[i],
                            qo-qs-self.vector(label, "stage17")[i])
                actual = row["weighted_actual_residual_boundary_parts"]
                self.assertEqual([Fraction(v) for v in actual.values()], [v*f for v in expected])
                r = {k: self.vector("reference", k)[i]
                     for k in (*d.residual.STAGES, "stage12")}
                refs = (r["input_hidden"]+r["stage11"]-r["stage12"],
                        r["stage12"]+r["stage17"]-r["stage18"])
                self.assertEqual(
                    [Fraction(v) for v in row["weighted_negative_reference_residual_boundary_parts"].values()],
                    [v*f for v in refs])

    def test_scalar_anchor_independent_decimal_oracle(self):
        for branch in d.BRANCHES:
            values = (self.vector("reference", "stage18") if branch == "fp16"
                      else [double(v) for v in self.e["binary64"]])
            radicand = sum(v*v for v in values)/d.WIDTH + Fraction(1, 1000000)
            with localcontext() as context:
                context.prec = 70
                expected = 1/(Decimal(radicand.numerator)/Decimal(radicand.denominator)).sqrt()
                for control in self.e["hidden_report"]["controls"]:
                    norm = control["branches"][branch]["reference_norm"]
                    self.assertEqual(Fraction(norm["radicand"]), radicand)
                    anchor = Fraction(norm["inverse_norm_anchor"])
                    actual = Decimal(anchor.numerator)/Decimal(anchor.denominator)
                    self.assertLess(abs(actual/expected-1), Decimal("4e-16"))

    def test_binary64_terminal_remainder_not_stage_substitution(self):
        self.assertNotEqual(self.e["hidden"]["fp16"], self.e["hidden"]["binary64"])
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        nonzero = False
        for _, _, branch, account, _, _ in self.branches():
            self.assertEqual(final["input_"+branch], original[branch])
            self.assertEqual(account["residual_internal_reference"], "original_input_L23_fp16")
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            for row in account["selected_coordinates"]:
                remainder = Fraction(row["hidden_components"]["fp16_to_branch_terminal_remainder"])
                if branch == "fp16":
                    self.assertEqual(remainder, 0)
                else:
                    nonzero |= remainder != 0
        self.assertTrue(nonzero)

    def test_coordinate62_and_q24_conversion_disclosed(self):
        count = 0
        for label, _, _, account, _, _ in self.branches():
            selected = {r["coordinate"]: r for r in account["selected_coordinates"]}
            self.assertIn(62, selected)
            count += 1
            for i, row in selected.items():
                q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
                self.assertEqual(Fraction(row["hidden_components"]["q24_to_fp16_conversion"]),
                                 self.vector(label, "stage18")[i]-q)
        self.assertEqual(count, self.e["report"]["coordinate62_accounts"])

    def test_pairs_selection_and_unsplit_outputs_preserved(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_logit_margin_bridge"],
                      self.e["margin_report"])
        for control, geometry in zip(self.e["report"]["controls"], self.e["geometry"]["controls"], strict=True):
            pairs = d.margin.contributions.pairs_for(geometry)
            self.assertEqual([(p["left_id"], p["right_id"]) for p in control["pairs"]], list(pairs))
            for pair in control["pairs"]:
                self.assertEqual(pair["roles"], pairs[pair["left_id"], pair["right_id"]])
        for _, _, _, account, original, _ in self.branches():
            self.assertEqual([r["coordinate"] for r in account["selected_coordinates"]],
                             [r["coordinate"] for r in original["selected_coordinates"]])
            self.assertIs(account["unchanged_margin_accounting"], original["accounting"])
            totals = original["accounting"]
            signed = sum(Fraction(v["signed"]) for v in totals["selected_term_totals"].values())
            self.assertEqual(signed + Fraction(totals["unselected_coordinate_signed_remainder"])
                             + Fraction(totals["head_boundary_remainder_change"]),
                             Fraction(totals["retained_margin_change"]))

    def test_selection_independent_bit_ranking(self):
        for label, pair, branch, account, _, _ in self.branches():
            ya = [half(v) for v in self.e["arrays"][label]["rmsnorm"]]
            decode = half if branch == "fp16" else double
            yr = [decode(v) for v in self.e["references"]["rmsnorm_"+branch]]
            delta = [a-r for a, r in zip(ya, yr, strict=True)]
            dw = [l-r for l, r in zip(self.row(pair["left_id"]), self.row(pair["right_id"]), strict=True)]
            expected = {62}
            for values in (delta, dw, [v*w for v, w in zip(delta, dw, strict=True)]):
                expected.update(sorted(range(d.WIDTH), key=lambda i: (-abs(values[i]), i))[:8])
            self.assertEqual([r["coordinate"] for r in account["selected_coordinates"]], sorted(expected))

    def test_signed_absolute_and_cancellation_accounts(self):
        for _, _, _, account, _, _ in self.branches():
            rows, totals = account["selected_coordinates"], account["direct_hidden_accounting"]
            all_abs = Fraction()
            direct = [Fraction(r["direct_hidden_weighted_term"]) for r in rows]
            for key in d.COMPONENTS:
                values = [Fraction(r["weighted_components"][key]) for r in rows]
                actual = totals["component_totals"][key]
                self.assertEqual(Fraction(actual["signed"]), sum(values))
                self.assertEqual(Fraction(actual["absolute"]), sum(map(abs, values)))
                self.assertEqual(Fraction(actual["cancellation_absolute_mass"]),
                                 sum(map(abs, values))-abs(sum(values)))
                all_abs += sum(map(abs, values))
            self.assertEqual(Fraction(totals["within_coordinate_cancellation_mass"]), all_abs-sum(map(abs, direct)))
            self.assertEqual(Fraction(totals["across_coordinate_cancellation_mass"]), sum(map(abs, direct))-abs(sum(direct)))
            self.assertEqual(Fraction(totals["total_component_cancellation_mass"]), all_abs-abs(sum(direct)))
            for row in rows:
                for source, target in (("hidden_components", "hidden_component_mass"),
                                       ("weighted_components", "weighted_component_mass")):
                    values = [Fraction(v) for v in row[source].values()]
                    self.assertEqual(Fraction(row[target]["signed"]), sum(values))
                    self.assertEqual(Fraction(row[target]["absolute"]), sum(map(abs, values)))
                    self.assertEqual(Fraction(row[target]["cancellation_absolute_mass"]),
                                     sum(map(abs, values))-abs(sum(values)))

    def test_synthetic_signed_zero_cancellation(self):
        self.assertEqual(half(0x8000), 0)
        self.assertEqual(d.mass([Fraction(-4), Fraction(2), Fraction(2)]),
                         {"signed": "0", "absolute": "8", "cancellation_absolute_mass": "8"})
        label, _, branch, _, original, hidden = next(self.branches())
        row = deepcopy(original["selected_coordinates"][0])
        row["right_weight"] = row["left_weight"]
        row["row_difference"] = "0"
        row["weighted_terms"]["direct_hidden"] = "0"
        sr = Fraction(hidden["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
        result = d.split_coordinate(self.e["actual"][label],
                                    d.residual.operands(self.e["reference_archive"], actual=False),
                                    self.e["hidden"][branch][row["coordinate"]], sr, row)
        self.assertTrue(all(Fraction(v) == 0 for v in result["weighted_components"].values()))

    def test_report_shape_and_census(self):
        report = self.e["report"]
        self.assertEqual(set(report), {
            "control_count", "diagnostic_pair_count", "pair_branch_count", "coordinate62_accounts",
            "selected_coordinate_accounts", "weighted_component_count", "reference_scope",
            "identity", "lineage_separation", "retained_final_rmsnorm_logit_margin_bridge", "controls"})
        self.assertEqual(report["control_count"], 9)
        pairs = sum(len(c["pairs"]) for c in report["controls"])
        accounts = sum(len(a["selected_coordinates"]) for _, _, _, a, _, _ in self.branches())
        self.assertGreater(pairs, 0)
        self.assertEqual(report["diagnostic_pair_count"], pairs)
        self.assertEqual(report["pair_branch_count"], 2*pairs)
        self.assertEqual(report["coordinate62_accounts"], 2*pairs)
        self.assertEqual(report["selected_coordinate_accounts"], accounts)
        self.assertEqual(accounts, self.e["margin_report"]["selected_coordinate_accounts"])
        self.assertEqual(report["weighted_component_count"], 7*accounts)

    def test_coordinate_and_operand_splices_refused(self):
        label, _, branch, _, original, hidden = next(self.branches())
        reference = d.residual.operands(self.e["reference_archive"], actual=False)
        sr = Fraction(hidden["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
        for key in ("actual_hidden", "reference_hidden", "actual_raw_q24_hidden",
                    "hidden_delta", "direct_hidden", "actual_q24_to_stage18_conversion"):
            row = deepcopy(original["selected_coordinates"][0])
            row["hidden_bridge"][key] = str(Fraction(row["hidden_bridge"][key])+1)
            with self.assertRaises(ValueError):
                d.split_coordinate(self.e["actual"][label], reference,
                                   self.e["hidden"][branch][row["coordinate"]], sr, row)
        for coordinate in (-1, 896, True):
            row = {**original["selected_coordinates"][0], "coordinate": coordinate}
            with self.assertRaises(ValueError):
                d.split_coordinate(self.e["actual"][label], reference, Fraction(), sr, row)

    def test_weighted_and_selected_total_mutations_refused(self):
        label, _, branch, account, original, hidden = next(self.branches())
        row = deepcopy(original["selected_coordinates"][0])
        row["weighted_terms"]["direct_hidden"] = str(Fraction(row["weighted_terms"]["direct_hidden"])+1)
        sr = Fraction(hidden["branches"][branch]["reference_norm"]["inverse_norm_anchor"])
        with self.assertRaises(ValueError):
            d.split_coordinate(self.e["actual"][label],
                               d.residual.operands(self.e["reference_archive"], actual=False),
                               self.e["hidden"][branch][row["coordinate"]], sr, row)
        rows = deepcopy(account["selected_coordinates"])
        rows[0]["weighted_components"]["attention_stage11"] = str(
            Fraction(rows[0]["weighted_components"]["attention_stage11"])+1)
        with self.assertRaises(ValueError):
            d.selected_account(rows, original["accounting"])

    def test_history_threshold_and_lineage_mutations_refused(self):
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
        self.assertEqual(self.e["margin_report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])

    def test_pins_and_authority_mutations_refused(self):
        for pin in (self.e["hidden_pins"][0], self.e["hidden_pins"][1],
                    self.e["hidden_pins"][-1], d.base.PINS["review"],
                    d.margin.rows.PINS["rmsnorm_hotspot_source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        changed = deepcopy(self.e["result"])
        changed["preflight"]["final_reference"]["reference"]["input_fp16"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            d.residual.load_residuals(changed)

    def test_state_and_kv_mutations_refused(self):
        archive = self.e["archives"][d.parent.CONTROLS[0]]
        changed = {**archive, "output_cache_k": archive["output_cache_k"].copy()}
        changed["output_cache_k"][0, 0] ^= 1
        with self.assertRaises(ValueError):
            d.residual.operands(changed, actual=True)
        changed = {**archive, "output_z": archive["output_z"].copy()}
        changed["output_z"][0] = 2
        with self.assertRaises(ValueError):
            d.residual.operands(changed, actual=True)

    def test_tokenizer_tied_head_and_norm_bindings(self):
        assets = self.e["assets"]
        self.assertEqual(assets["tokenizer_status"], "BOUND")
        self.assertFalse(assets["missing_tokenizer"])
        self.assertEqual(assets["tensors"]["lm_head.weight"], assets["tensors"]["model.embed_tokens.weight"])
        self.assertEqual(d.parent.norm.EPSILON_Q48, 281474977)
        with self.assertRaises(ValueError):
            d.hidden.validate_weight(self.e["weight_array"],
                                     {**assets["tensors"]["model.norm.weight"], "sha256": "0"*64})

    def test_control_order_mutation_refused(self):
        changed = {**self.e, "margin_report": {
            **self.e["margin_report"], "controls": list(reversed(self.e["margin_report"]["controls"]))}}
        with self.assertRaises(ValueError):
            d.report(changed)

    def test_forbidden_dispatch_and_prior_checks(self):
        calls = (
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.residual.local.local_reference(None),
            lambda: d.parent.head.decode_array_q24(None),
            lambda: d.margin.check(), lambda: d.margin.measure(), lambda: d.hidden.check(),
            lambda: d.residual.check(), lambda: d.margin.rows.check(),
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
        target = d.parent.OUTPUT / "forbidden-direct-hidden-bridge"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_rejects_replay_output_and_abbreviations(self):
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

    def test_nonadmission_and_reference_scope(self):
        for key in ("rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "upstream_causality_claimed", "rounding_only_attribution", "candidate_admitted",
                    "policy_adopted", "successor_published", "binary64_internal_stages_substituted"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "not causal percentages", "not rounding-only",
                       "No RMSNorm, head, row-dot", "Normal independent Host Reviewer REQUIRED"):
            self.assertIn(phrase, d.BOUNDARY)
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])
        self.assertIn("not a binary64 input/attention/MLP decomposition", d.REFERENCE_SCOPE)


if __name__ == "__main__":
    unittest.main()
