"""Independent retained-bit oracles for the unselected-coordinate margin split."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_coordinate_residual_margin_bridge_v1 as d


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


class UnselectedResidualMarginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def branches(self):
        for control, mc in zip(self.e["report"]["controls"], self.e["margin_report"]["controls"], strict=True):
            for pair, mp in zip(control["pairs"], mc["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield control, pair, branch, account, mp["branches"][branch]

    def assert_mass(self, observed, values):
        values = list(values)
        signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
        self.assertEqual(observed, {"signed": str(signed), "absolute": str(absolute),
                                   "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_independent_retained_bit_oracle(self):
        weights = [half(v) for v in self.e["weight_array"].view("<u2")]
        rows = {i: [half(v) for v in a.view("<u2")] for i, a in self.e["rows"].items()}
        references = {
            "fp16": [half(v) for v in self.e["reference_archive"]["stage18"]],
            "binary64": [double(v) for v in self.e["binary64"]],
        }
        outputs = {
            "fp16": [half(v) for v in self.e["references"]["rmsnorm_fp16"]],
            "binary64": [double(v) for v in self.e["references"]["rmsnorm_binary64"]],
        }
        for control, pair, branch, account, original in self.branches():
            label = control["control"]
            actual = [half(v) for v in self.e["archives"][label]["stage18"]]
            output = [half(v) for v in self.e["arrays"][label]["rmsnorm"]]
            sa, sr = (Fraction(n["inverse_norm_anchor"])
                      for n in (control["actual_norm"], account["reference_norm"]))
            selected = {r["coordinate"] for r in original["selected_coordinates"]}
            components = {k: [] for k in d.TERMS}
            deltas = []
            for i in range(896):
                if i in selected:
                    continue
                a, r, ya, yr, w = actual[i], references[branch][i], output[i], outputs[branch][i], weights[i]
                dw = rows[pair["left_id"]][i]-rows[pair["right_id"]][i]
                # Expanded products provide an independent algebraic evaluation order.
                direct = dw*w*a*sr-dw*w*r*sr
                scale = dw*w*r*sa-dw*w*r*sr
                interaction = dw*w*a*sa-dw*w*a*sr-dw*w*r*sa+dw*w*r*sr
                boundary = dw*ya-dw*w*a*sa-dw*yr+dw*w*r*sr
                for k, v in zip(d.TERMS, (direct, scale, interaction, boundary), strict=True):
                    components[k].append(v)
                deltas.append(dw*ya-dw*yr)
            result = account["unselected_accounting"]
            for k, values in components.items():
                self.assert_mass(result["component_totals"][k], values)
            self.assert_mass(result["unselected_coordinate_remainder"], deltas)
            absolute = sum(abs(v) for values in components.values() for v in values)
            net_absolute = sum(abs(sum(values)) for values in components.values())
            self.assertEqual(Fraction(result["component_absolute_sum"]), absolute)
            self.assertEqual(Fraction(result["within_coordinate_cancellation_mass"]),
                             absolute-sum(map(abs, deltas)))
            self.assertEqual(Fraction(result["across_coordinate_cancellation_mass"]),
                             sum(map(abs, deltas))-abs(sum(deltas)))
            self.assertEqual(Fraction(result["total_component_cancellation_mass"]), absolute-abs(sum(deltas)))
            self.assertEqual(Fraction(result["within_category_across_coordinate_cancellation_mass"]),
                             absolute-net_absolute)
            self.assertEqual(Fraction(result["between_category_net_cancellation_mass"]), net_absolute-abs(sum(deltas)))

    def test_exact_remainder_and_head_closure(self):
        for _, _, _, account, original in self.branches():
            result, totals = account["unselected_accounting"], original["accounting"]
            target = result["unselected_coordinate_remainder"]
            self.assertEqual(target["signed"], totals["unselected_coordinate_signed_remainder"])
            self.assertEqual(target["absolute"], totals["unselected_coordinate_absolute_remainder"])
            self.assertEqual(sum(Fraction(v["signed"]) for v in result["component_totals"].values())
                             +Fraction(totals["selected_coordinate_signed_sum"])
                             +Fraction(totals["head_boundary_remainder_change"]),
                             Fraction(totals["retained_margin_change"]))
            for term in d.TERMS:
                self.assertEqual(Fraction(result["neutralized_retained_margin_change"][term]),
                                 Fraction(totals["retained_margin_change"])
                                 -Fraction(result["component_totals"][term]["signed"]))

    def test_selected_union_and_retained_chain_unchanged(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_boundary_residual_margin_bridge"],
                      self.e["boundary_report"])
        for control, geometry in zip(self.e["report"]["controls"], self.e["geometry"]["controls"], strict=True):
            self.assertEqual([(p["left_id"], p["right_id"], p["roles"]) for p in control["pairs"]],
                             [(a, b, roles) for (a, b), roles in d.margin.contributions.pairs_for(geometry).items()])
        for _, _, _, account, original in self.branches():
            self.assertIs(account["selected_coordinates"], original["selected_coordinates"])
            self.assertIs(account["unchanged_margin_accounting"], original["accounting"])
            indices = [r["coordinate"] for r in original["selected_coordinates"]]
            self.assertIn(62, indices)
            self.assertEqual(account["unselected_accounting"]["excluded_selected_coordinates"], indices)
            self.assertEqual(account["unselected_accounting"]["coordinate_count"], 896-len(indices))

    def test_independent_references_and_lineage(self):
        self.assertNotEqual([half(v) for v in self.e["reference_archive"]["stage18"]],
                            [double(v) for v in self.e["binary64"]])
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for b in d.BRANCHES:
            self.assertEqual(final["input_"+b], original[b])
        for _, _, b, account, _ in self.branches():
            self.assertEqual(account["hidden_reference"], "original_input_L23_"+b)
            self.assertEqual(account["rmsnorm_reference"], "original_input_final_attempt003_"+b)
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])

    def test_independent_scalar_anchor_oracle(self):
        for control in self.e["hidden_report"]["controls"]:
            actual = [half(v) for v in self.e["archives"][control["control"]]["stage18"]]
            for b in d.BRANCHES:
                reference = ([half(v) for v in self.e["reference_archive"]["stage18"]]
                             if b == "fp16" else [double(v) for v in self.e["binary64"]])
                for values, norm in ((actual, control["actual_norm"]),
                                     (reference, control["branches"][b]["reference_norm"])):
                    radicand = sum(v*v for v in values)/896+Fraction(1, 1000000)
                    self.assertEqual(Fraction(norm["radicand"]), radicand)
                    anchor = Fraction(norm["inverse_norm_anchor"])
                    with localcontext() as context:
                        context.prec = 70
                        expected = 1/(Decimal(radicand.numerator)/Decimal(radicand.denominator)).sqrt()
                        observed = Decimal(anchor.numerator)/Decimal(anchor.denominator)
                        self.assertLess(abs(observed/expected-1), Decimal("4e-16"))
                    self.assertEqual(Fraction(norm["inverse_square_identity_defect"]), anchor*anchor*radicand-1)
        self.assertEqual(self.e["hidden_report"]["native_epsilon_q48"], 281474977)

    def fixture(self, zero=False):
        a, r, ya, yr, w, left, right = [[Fraction() for _ in range(896)] for _ in range(7)]
        a[63], r[63], w[63], left[63] = Fraction(2), Fraction(3), Fraction(4, 3), Fraction(-3, 5)
        ya[63], yr[63] = (Fraction(), Fraction()) if zero else (Fraction(5, 11), Fraction(-3, 13))
        target = left[63]*(ya[63]-yr[63])
        head = Fraction(1, 7)
        totals = {"unselected_coordinate_signed_remainder": str(target),
                  "unselected_coordinate_absolute_remainder": str(abs(target)),
                  "selected_coordinate_signed_sum": "0", "head_boundary_remainder_change": str(head),
                  "retained_margin_change": str(target+head)}
        return [a, r, ya, yr, w, left, right, Fraction(5), Fraction(7), [0, 1, 2, 3, 4, 5, 6, 62], totals]

    def test_zero_net_nonzero_cancellation(self):
        result = d.split_unselected(*self.fixture(zero=True))
        self.assertEqual(result["unselected_coordinate_remainder"]["signed"], "0")
        self.assertEqual(result["unselected_coordinate_remainder"]["absolute"], "0")
        self.assertGreater(Fraction(result["within_coordinate_cancellation_mass"]), 0)
        self.assertEqual(result["component_absolute_sum"], result["total_component_cancellation_mass"])

    def test_non_dyadic_signed_identity(self):
        result = d.split_unselected(*self.fixture())
        target = -Fraction(3, 5)*(Fraction(5, 11)+Fraction(3, 13))
        self.assertEqual(Fraction(result["unselected_coordinate_remainder"]["signed"]), target)
        self.assertEqual(Fraction(result["component_totals"]["direct_hidden"]["signed"]), Fraction(28, 5))

    def test_invalid_selected_coordinates(self):
        for indices in ([0, 1, 2, 3, 4, 5, 6, True, 62], [0, 1, 2, 3, 4, 5, 6, -1, 62],
                        [0, 1, 2, 3, 4, 5, 6, 62, 896], [0, 1, 2, 3, 4, 5, 6, 62, 62],
                        [0, 1, 2, 3, 4, 5, 6, 7], [62, 0, 1, 2, 3, 4, 5, 6]):
            args = self.fixture()
            args[9] = indices
            with self.assertRaises(ValueError):
                d.split_unselected(*args)

    def test_invalid_exact_vectors_and_anchors(self):
        for index in range(7):
            args = self.fixture()
            args[index] = args[index][:-1]
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
            args = self.fixture()
            args[index][63] = 1.0
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
        for index in (7, 8):
            for value in (Fraction(), Fraction(-1), 1.0):
                args = self.fixture()
                args[index] = value
                with self.assertRaises(ValueError):
                    d.split_unselected(*args)

    def test_remainder_and_operand_splices(self):
        for key in self.fixture()[-1]:
            args = self.fixture()
            args[-1][key] = str(Fraction(args[-1][key])+1)
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
        for index in (2, 3, 5, 6):
            args = self.fixture()
            args[index][63] += 1
            with self.assertRaises(ValueError):
                d.split_unselected(*args)

    def test_report_linkage_and_census_splices(self):
        old = self.e["boundary_report"]
        for key, value in (
            ("retained_final_rmsnorm_interaction_residual_margin_bridge", {}),
            ("controls", list(reversed(old["controls"]))), ("coordinate62_accounts", 0),
        ):
            with self.assertRaises(ValueError):
                d.report({**self.e, "boundary_report": {**old, key: value}})

    def test_history_failure_threshold_gates(self):
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

    def test_source_pins_and_original_authority(self):
        for pin in (d.base.PINS["review"], *self.e["hidden_pins"],
                    d.margin.rows.PINS["rmsnorm_hotspot_source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        changed = deepcopy(self.e["result"])
        changed["preflight"]["final_reference"]["reference"]["input_fp16"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            d.residual.load_residuals(changed)

    def test_state_kv_tokenizer_and_weights(self):
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
        for module in (d.boundary, d.interaction, d.scale, d.direct, d.margin, d.hidden, d.residual):
            calls.extend((lambda m=module: m.check(), lambda m=module: m.measure(),
                          lambda m=module: m.focused_tests(None)))
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-unselected-bridge"
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

    def selection_rows(self, values, remainder, retained):
        return [
            {"control": c, "branch": b,
             "unselected_accounting": {
                 "component_totals": {k: {"signed": str(v)} for k, v in zip(d.TERMS, values, strict=True)},
                 "neutralized_retained_margin_change": {
                     k: str(retained-v) for k, v in zip(d.TERMS, values, strict=True)}},
             "unchanged_margin_accounting": {"unselected_coordinate_signed_remainder": str(remainder),
                                             "retained_margin_change": str(retained)}}
            for c in d.parent.CONTROLS for b in d.BRANCHES]

    def test_predeclared_selection_and_stop_rule(self):
        rows = self.selection_rows([6, 2, 1, 1], 10, 20)
        self.assertEqual(d.select_mechanism(rows)["mechanism"], "direct_hidden")
        self.assertEqual(d.select_mechanism(rows[:-1])["mechanism"], "UNKNOWN")
        self.assertEqual(d.select_mechanism(list(reversed(rows)))["mechanism"], "UNKNOWN")
        for values, remainder, retained, expected in (
            ([5, 2, 2, 1], 10, 20, "UNKNOWN"),
            ([5, 2, 2, 1], 10, 1, "direct_hidden"),
            ([5, 2, 2, 1], 10, 5, "UNKNOWN"),
            ([5, 5, 0, 0], 10, 1, "UNKNOWN"),
            ([0, 0, 0, 0], 0, 0, "UNKNOWN"),
            ([-6, -2, -1, -1], -10, -20, "direct_hidden"),
            ([6, -2, -2, -2], 0, 0, "UNKNOWN"),
        ):
            self.assertEqual(d.select_mechanism(self.selection_rows(values, remainder, retained))["mechanism"],
                             expected)
        changed = deepcopy(rows)
        changed[1] = self.selection_rows([1, 7, 1, 1], 10, 20)[1]
        self.assertEqual(d.select_mechanism(changed)["mechanism"], "UNKNOWN")
        changed = deepcopy(rows)
        changed[-1] = self.selection_rows([-6, -2, -1, -1], -10, -20)[-1]
        self.assertEqual(d.select_mechanism(changed)["mechanism"], "UNKNOWN")
        changed = deepcopy(rows)
        changed[0]["unselected_accounting"]["neutralized_retained_margin_change"]["direct_hidden"] = "99"
        with self.assertRaises(ValueError):
            d.select_mechanism(changed)

    def test_report_census_selection_and_nonadmission(self):
        report = self.e["report"]
        self.assertEqual(report["control_count"], 9)
        self.assertEqual(report["pair_branch_count"], 2*report["diagnostic_pair_count"])
        self.assertEqual(report["coordinate62_accounts"], report["pair_branch_count"])
        self.assertEqual(report["selected_coordinate_accounts"]+report["unselected_coordinate_accounts"],
                         896*report["pair_branch_count"])
        self.assertEqual(report["weighted_unselected_term_count"], 4*report["unselected_coordinate_accounts"])
        for decision in report["selection"]["pairs"]:
            rows = []
            for control in report["controls"]:
                for pair in control["pairs"]:
                    if (pair["left_id"], pair["right_id"]) == (decision["left_id"], decision["right_id"]):
                        rows.extend({"control": control["control"], "branch": b, **a}
                                    for b, a in pair["branches"].items())
            qualifying = []
            for term in d.TERMS:
                values = [Fraction(r["unselected_accounting"]["component_totals"][term]["signed"]) for r in rows]
                stable = bool(values) and all(v*values[0] > 0 for v in values)
                eligible = len(rows) == 18 and stable
                for row, value in zip(rows, values, strict=True):
                    terms = row["unselected_accounting"]["component_totals"]
                    remainder = Fraction(row["unchanged_margin_accounting"]["unselected_coordinate_signed_remainder"])
                    retained = Fraction(row["unchanged_margin_accounting"]["retained_margin_change"])
                    eligible &= all(abs(value) > abs(Fraction(terms[k]["signed"])) for k in d.TERMS if k != term)
                    eligible &= ((remainder*value > 0 and 2*abs(value) > abs(remainder))
                                 or retained*(retained-value) < 0)
                if eligible:
                    qualifying.append(term)
            self.assertEqual(decision["mechanism"], qualifying[0] if qualifying else "UNKNOWN")
        for key in ("native_dispatch", "decoder_dispatch", "prefix_dispatch", "admission_dispatch",
                    "rmsnorm_operator_replay", "head_operator_replay", "row_dot_operator_replay",
                    "reference_producer_replay", "evidence_writes", "precision_or_scale_expansion",
                    "candidate_admitted", "policy_adopted", "successor_published",
                    "binary64_internal_stages_substituted", "unselected_split_causal_allocation",
                    "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "simulation_dispatch"):
            self.assertFalse(d.FLAGS[key])
        for phrase in ("wider than FP16", "not a binary64", "P0-only", "Normal independent Host Reviewer REQUIRED"):
            self.assertIn(phrase, d.BOUNDARY)
