"""Independent bit-decoding and algebra oracles for retained hidden accounting."""

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

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_hidden_delta_bridge_v1 as d


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


class HiddenBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def selections(self):
        for control in self.e["report"]["controls"]:
            for branch, report in control["branches"].items():
                for row in [*report["hotspots"], report["forced_coordinate62"]]:
                    yield control, branch, report, row

    def hidden(self, label, branch):
        a = [half(v) for v in self.e["archives"][label]["stage18"]]
        r = ([half(v) for v in self.e["reference_archive"]["stage18"]]
             if branch == "fp16" else [double(v) for v in self.e["binary64"]])
        return a, r

    def outputs(self, label, branch):
        ya = [half(v) for v in self.e["arrays"][label]["rmsnorm"]]
        raw = self.e["references"]["rmsnorm_" + branch]
        return ya, [half(v) if branch == "fp16" else double(v) for v in raw]

    def test_independent_live_coordinate_oracle(self):
        for control, branch, report, row in self.selections():
            label, i = control["control"], row["coordinate"]
            a, r = self.hidden(label, branch)
            ya, yr = self.outputs(label, branch)
            w = half(self.e["weight_array"].view("<u2")[i])
            sa = Fraction(control["actual_norm"]["inverse_norm_anchor"])
            sr = Fraction(report["reference_norm"]["inverse_norm_anchor"])
            expected = {
                "actual_hidden": a[i], "reference_hidden": r[i],
                "actual_rmsnorm": ya[i], "reference_rmsnorm": yr[i],
                "hidden_delta": a[i] - r[i], "weight": w,
                "direct_hidden": (a[i] - r[i]) * w * sr,
                "global_scale": r[i] * w * (sa - sr),
                "interaction": (a[i] - r[i]) * w * (sa - sr),
                "actual_boundary_conversion_remainder": ya[i] - a[i] * w * sa,
                "reference_boundary_conversion_remainder": yr[i] - r[i] * w * sr,
            }
            for key, value in expected.items():
                self.assertEqual(Fraction(row[key]), value, (label, branch, i, key))

    def test_all_closures_and_cancellation(self):
        for _, _, _, row in self.selections():
            terms = [Fraction(row[key]) for key in d.TERMS]
            delta = Fraction(row["retained_rmsnorm_delta"])
            self.assertEqual(sum(terms), delta)
            self.assertEqual(Fraction(row["signed_term_sum"]), delta)
            self.assertEqual(Fraction(row["absolute_term_sum"]), sum(map(abs, terms)))
            self.assertEqual(Fraction(row["cancellation_absolute_mass"]),
                             sum(map(abs, terms)) - abs(delta))
            self.assertEqual(Fraction(row["boundary_remainder_delta"]),
                             Fraction(row["actual_boundary_conversion_remainder"])
                             - Fraction(row["reference_boundary_conversion_remainder"]))
            self.assertIs(row["exact_accounting_identity"], True)

    def test_global_energy_independent_oracle(self):
        for control in self.e["report"]["controls"]:
            for branch, report in control["branches"].items():
                a, r = self.hidden(control["control"], branch)
                delta = sum(x*x-y*y for x, y in zip(a, r)) / d.WIDTH
                linear = sum(2*y*(x-y) for x, y in zip(a, r)) / d.WIDTH
                quadratic = sum((x-y)**2 for x, y in zip(a, r)) / d.WIDTH
                self.assertEqual(Fraction(report["global_mean_square_delta"]), delta)
                self.assertEqual(Fraction(report["global_mean_square_linear_change"]), linear)
                self.assertEqual(Fraction(report["global_mean_square_quadratic_change"]), quadratic)
                self.assertEqual(delta, linear + quadratic)

    def test_selected_and_other_energy(self):
        for control, branch, report, row in self.selections():
            a, r = self.hidden(control["control"], branch)
            i = row["coordinate"]
            selected = (a[i]**2 - r[i]**2) / d.WIDTH
            self.assertEqual(Fraction(row["selected_mean_square_linear_change"])
                             + Fraction(row["selected_mean_square_quadratic_change"]), selected)
            self.assertEqual(Fraction(row["other_coordinates_mean_square_change"]) + selected,
                             Fraction(report["global_mean_square_delta"]))

    def test_scalar_anchors_against_decimal_oracle(self):
        for control in self.e["report"]["controls"]:
            for norm in [control["actual_norm"],
                         *(v["reference_norm"] for v in control["branches"].values())]:
                q = Fraction(norm["radicand"])
                s = Fraction(norm["inverse_norm_anchor"])
                with localcontext() as context:
                    context.prec = 70
                    exact = 1 / (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
                    anchor = Decimal(s.numerator) / Decimal(s.denominator)
                    self.assertLess(abs(anchor / exact - 1), Decimal("4e-16"))
                self.assertEqual(Fraction(norm["inverse_square_identity_defect"]), s*s*q-1)
                self.assertEqual(Fraction(norm["radicand_binary64_conversion_delta"]),
                                 double(float(q))-q)

    def test_hotspot_order_and_counts(self):
        self.assertEqual(self.e["report"]["logical_hotspot_accounts"], 144)
        self.assertEqual(self.e["report"]["forced_coordinate_accounts"], 18)
        self.assertEqual([v["control"] for v in self.e["report"]["controls"]],
                         list(d.parent.CONTROLS))
        for control in self.e["report"]["controls"]:
            self.assertEqual(list(control["branches"]), list(d.BRANCHES))
            for branch, report in control["branches"].items():
                ya, yr = self.outputs(control["control"], branch)
                order = sorted(range(d.WIDTH), key=lambda i: (-abs(ya[i]-yr[i]), i))
                self.assertEqual([r["coordinate"] for r in report["hotspots"]], order[:8])
                self.assertEqual([r["absolute_rank"] for r in report["hotspots"]], list(range(1, 9)))
                self.assertEqual(report["forced_coordinate62"]["coordinate"], 62)
                self.assertEqual(report["forced_coordinate62"]["absolute_rank"], order.index(62)+1)

    def test_separate_original_reference_branches(self):
        self.assertNotEqual(self.e["hidden"]["fp16"], self.e["hidden"]["binary64"])
        for branch in d.BRANCHES:
            _, reference = self.hidden(d.parent.CONTROLS[0], branch)
            self.assertEqual(self.e["hidden"][branch], reference)
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for branch in d.BRANCHES:
            self.assertEqual(final["input_"+branch], original[branch])

    def test_q24_conversion_not_raw_state_propagation(self):
        for control, _, _, row in self.selections():
            archive = self.e["archives"][control["control"]]
            i = row["coordinate"]
            raw = Fraction(int(archive["output_i"][i]), 1 << 24)
            self.assertEqual(Fraction(row["actual_raw_q24_hidden"]), raw)
            self.assertEqual(Fraction(row["actual_q24_to_stage18_conversion"]),
                             half(archive["stage18"][i]) - raw)

    def test_synthetic_interaction_and_cancellation(self):
        linear = [Fraction(0)] * d.WIDTH
        row = d.account(Fraction(2), Fraction(1), Fraction(3), Fraction(3),
                        Fraction(1), Fraction(1), Fraction(2), 0,
                        linear, linear, Fraction(2))
        self.assertEqual([Fraction(row[k]) for k in d.TERMS],
                         [Fraction(2), Fraction(-1), Fraction(-1), Fraction(0)])
        self.assertEqual(Fraction(row["cancellation_absolute_mass"]), 4)

    def test_zero_and_signed_hidden_vectors(self):
        zero = [Fraction(0)] * d.WIDTH
        norm = d.norm_summary(zero)
        self.assertEqual(Fraction(norm["mean_square"]), 0)
        a, r = zero.copy(), zero.copy()
        a[0], r[0] = Fraction(-2), Fraction(2)
        linear, quadratic = d.energy_delta(a, r)
        self.assertEqual(linear[0], -quadratic[0])
        self.assertEqual(d.norm_summary(a), d.norm_summary(r))

    def test_invalid_array_and_weight_bindings(self):
        pin = self.e["assets"]["tensors"]["model.norm.weight"]
        weight = self.e["weight_array"]
        for bad in (weight[:1], weight.astype("<f8"),
                    np.full(d.WIDTH, np.inf, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.validate_weight(bad, pin)
        bad_pin = {**pin, "sha256": "0"*64}
        with self.assertRaises(ValueError):
            d.validate_weight(weight, bad_pin)
        for bad in ([], [0]*d.WIDTH):
            with self.assertRaises(ValueError):
                d.norm_summary(bad)

    def test_history_gate_mutations_refused(self):
        for key, value in (("mandatory_statuses", ["PASS"]*19),
                           ("S18_failure_indices", []),
                           ("candidate_admitted", True),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)

    def test_input_pin_mutation_refused(self):
        pin = self.e["hidden_pins"][-1]
        with self.assertRaises(ValueError):
            d.base.read_bound({**pin, "sha256": "0"*64})

    def test_dispatch_guards(self):
        calls = (lambda: d.parent.rmsnorm(None, None),
                 lambda: d.parent.logits(None, None),
                 lambda: d.parent.norm.rmsnorm(None, None),
                 lambda: d.parent.execute("forbidden"),
                 lambda: d.residual.local.local_reference(None),
                 lambda: d.parent.head.decode_array_q24(None),
                 lambda: d.hotspots.check(),
                 lambda: subprocess.run(["forbidden-native"]),
                 lambda: os.system("forbidden-native"),
                 lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_write_guards(self):
        target = d.parent.OUTPUT / "forbidden-hidden-bridge"
        audit = {"forbidden_calls": 0}
        calls = (lambda: target.write_text("forbidden"),
                 lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
                 lambda: target.mkdir(),
                 lambda: os.replace(target, target))
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_refuses_replay_and_output_paths(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_is_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value=self.e["report"]), \
                patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        self.assertEqual(json.loads(text), self.e["report"])
        self.assertEqual(text.count("\n"), 1)

    def test_claims_and_output_shape(self):
        self.assertEqual(set(self.e["report"]), {
            "selection", "scalar_anchor", "identity", "energy_identity", "remainder_scope",
            "native_epsilon_q48", "native_epsilon_minus_contract", "logical_hotspot_accounts",
            "forced_coordinate_accounts", "controls"})
        for key in ("rmsnorm_operator_replay", "head_operator_replay",
                    "reference_producer_replay", "evidence_writes",
                    "precision_or_scale_expansion", "upstream_causality_claimed",
                    "rounding_only_attribution"):
            self.assertFalse(d.FLAGS[key])
        self.assertIn("not rounding-only", d.BOUNDARY)
        self.assertIn("wider than FP16", d.BOUNDARY)
        self.assertEqual(self.e["assets"]["tensors"]["lm_head.weight"],
                         self.e["assets"]["tensors"]["model.embed_tokens.weight"])
        self.assertEqual(self.e["assets"]["tokenizer_status"], "BOUND")
        self.assertEqual(d.parent.norm.EPSILON_Q48, 281474977)


if __name__ == "__main__":
    unittest.main()
