"""Independent retained-bit oracles for selected-row head boundary accounting."""

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import io
import json
import os
import struct
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_accumulation_boundary_residual_margin_bridge_v1 as d


EVIDENCE = None


def bits(word, fraction_bits, exponent_bits, bias):
    exponent = (word >> fraction_bits) & ((1 << exponent_bits)-1)
    mantissa = word & ((1 << fraction_bits)-1)
    if exponent == (1 << exponent_bits)-1:
        raise ValueError("nonfinite oracle operand")
    value = Fraction(mantissa if exponent == 0 else (1 << fraction_bits)+mantissa)
    value *= Fraction(2) ** ((1 if exponent == 0 else exponent)-bias-fraction_bits)
    return -value if word >> (fraction_bits+exponent_bits) else value


def half(word):
    return bits(int(word), 10, 5, 15)


def double(value):
    return bits(struct.unpack("<Q", struct.pack("<d", value))[0], 52, 11, 1023)


def audit():
    return {"forbidden_calls": 0, "local_exact_row_dot_computations": 0,
            "selected_row_scalar_products": 0}


class HeadBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only(audit()):
                cls.e = d.measure(audit())
        else:
            cls.e = EVIDENCE

    def accounts(self):
        for control in self.e["report"]["controls"]:
            for pair in control["pairs"]:
                for branch, row in pair["branches"].items():
                    yield control["control"], pair, branch, row["accounting"]

    @lru_cache(maxsize=None)
    def vector(self, trajectory):
        if trajectory in d.BRANCHES:
            raw = self.e["references"]["rmsnorm_"+trajectory]
            return tuple((half if trajectory == "fp16" else double)(v) for v in raw)
        return tuple(half(v) for v in self.e["arrays"][trajectory]["rmsnorm"])

    @lru_cache(maxsize=None)
    def row(self, index):
        return tuple(half(v) for v in self.e["rows"][index].view("<u2"))

    @lru_cache(maxsize=None)
    def dot(self, trajectory, index):
        return sum((a*w for a, w in zip(self.vector(trajectory), self.row(index), strict=True)),
                   Fraction())

    def test_live_selected_row_dots_independent_bit_oracle(self):
        for label, pair, branch, a in self.accounts():
            for side, index in (("left", pair["left_id"]), ("right", pair["right_id"])):
                self.assertEqual(Fraction(a["actual_exact_"+side+"_row_dot"]), self.dot(label, index))
                self.assertEqual(Fraction(a["reference_exact_"+side+"_row_dot"]), self.dot(branch, index))

    def test_live_retained_margins_and_boundary_signs(self):
        for label, pair, branch, a in self.accounts():
            l, r = pair["left_id"], pair["right_id"]
            al = self.e["arrays"][label]["logits"]
            rl = self.e["references"]["logits_"+branch]
            decode = half if branch == "fp16" else double
            am, rm = half(al[l])-half(al[r]), decode(rl[l])-decode(rl[r])
            ea, er = self.dot(label, l)-self.dot(label, r), self.dot(branch, l)-self.dot(branch, r)
            expected = {
                "retained_actual_margin": am, "retained_reference_margin": rm,
                "retained_margin_change": am-rm,
                "actual_exact_row_dot_margin": ea, "reference_exact_row_dot_margin": er,
                "actual_head_boundary": am-ea, "reference_head_boundary": rm-er,
                "negative_reference_head_boundary": er-rm,
                "head_boundary_remainder_change": am-ea-rm+er,
            }
            for key, value in expected.items():
                self.assertEqual(Fraction(a[key]), value)

    def test_vector_coordinate_identity_independent_oracle(self):
        for label, pair, branch, a in self.accounts():
            values = [(x-y)*(l-r) for x, y, l, r in zip(
                self.vector(label), self.vector(branch),
                self.row(pair["left_id"]), self.row(pair["right_id"]), strict=True)]
            self.assertEqual(Fraction(a["vector_coordinate_sum"]), sum(values))
            self.assertEqual(Fraction(a["vector_coordinate_absolute_sum"]), sum(map(abs, values)))
            self.assertEqual(Fraction(a["vector_coordinate_cancellation_mass"]),
                             sum(map(abs, values))-abs(sum(values)))

    def test_closure_and_mass_oracle(self):
        for _, _, _, a in self.accounts():
            components = [Fraction(a[k]) for k in d.COMPONENTS]
            for key, values in (("boundary_accounting", components),
                                ("margin_accounting", [Fraction(a["vector_coordinate_sum"]), *components])):
                self.assertEqual(Fraction(a[key]["signed"]), sum(values))
                self.assertEqual(Fraction(a[key]["absolute"]), sum(map(abs, values)))
                self.assertEqual(Fraction(a[key]["cancellation_absolute_mass"]),
                                 sum(map(abs, values))-abs(sum(values)))
            self.assertEqual(set(a["closure_residuals"].values()), {"0"})
            old = a["unchanged_margin_accounting"]
            self.assertEqual(Fraction(old["selected_coordinate_signed_sum"])
                             + Fraction(old["unselected_coordinate_signed_remainder"])
                             + sum(components), Fraction(a["retained_margin_change"]))
            self.assertTrue(a["exact_boundary_identity"] and a["exact_margin_identity"])

    def test_branch_totals_keep_repeated_accounts_separate(self):
        for branch in d.BRANCHES:
            accounts = [a for _, _, b, a in self.accounts() if b == branch]
            total = self.e["report"]["branch_totals"][branch]
            self.assertEqual(total["pair_accounts"], len(accounts))
            for key, value in total["component_totals"].items():
                terms = [Fraction(a[key]) for a in accounts]
                self.assertEqual(Fraction(value["signed"]), sum(terms))
                self.assertEqual(Fraction(value["absolute"]), sum(map(abs, terms)))
            self.assertEqual(Fraction(total["boundary_absolute_account"]), sum(
                Fraction(a["boundary_accounting"]["absolute"]) for a in accounts))
            self.assertEqual(Fraction(total["within_pair_boundary_cancellation"]), sum(
                Fraction(a["boundary_accounting"]["cancellation_absolute_mass"]) for a in accounts))

    def test_census_pairs_roles_and_unchanged_upstream(self):
        report = self.e["report"]
        self.assertEqual(report["control_count"], 9)
        self.assertEqual(report["diagnostic_pair_count"], 27)
        self.assertEqual(report["pair_branch_count"], 54)
        self.assertEqual(report["selected_row_ids"], [13, 319, 34319])
        self.assertEqual(report["retained_final_rmsnorm_logit_margin_bridge"], self.e["margin_report"])
        for control, old in zip(report["controls"], self.e["margin_report"]["controls"], strict=True):
            self.assertEqual(control["control"], old["control"])
            for pair, p in zip(control["pairs"], old["pairs"], strict=True):
                for key in ("left_id", "right_id", "roles"):
                    self.assertEqual(pair[key], p[key])
                self.assertEqual(list(pair["branches"]), list(d.BRANCHES))
                for b, account in pair["branches"].items():
                    self.assertEqual(account["accounting"]["unchanged_margin_accounting"],
                                     p["branches"][b]["accounting"])

    def test_bounded_arithmetic_count(self):
        report = self.e["report"]
        expected = set()
        for label, pair, branch, _ in self.accounts():
            for index in (pair["left_id"], pair["right_id"]):
                expected.update((("actual", label, index), ("reference", branch, index)))
        self.assertEqual(len(expected), 33)
        self.assertEqual(report["local_exact_row_dot_computations"], len(expected))
        self.assertEqual(report["selected_row_scalar_products"], len(expected)*896)

    def test_synthetic_exact_cancellation_and_negative_reference(self):
        effect = {"exact_sum": "0", "exact_sum_of_absolute_contributions": "0",
                  "retained_actual_margin": "3", "retained_reference_margin": "3",
                  "retained_margin_change": "0", "boundary_remainder_change": "0"}
        old = {"head_boundary_remainder_change": "0", "retained_margin_change": "0",
               "selected_coordinate_signed_sum": "0", "unselected_coordinate_signed_remainder": "0",
               "unselected_coordinate_absolute_remainder": "0"}
        a = d.split_boundary(Fraction(3), Fraction(3), (Fraction(2), Fraction(0)),
                             (Fraction(2), Fraction(0)), effect, old)
        self.assertEqual(a["actual_head_boundary"], "1")
        self.assertEqual(a["negative_reference_head_boundary"], "-1")
        self.assertEqual(a["boundary_accounting"], {
            "signed": "0", "absolute": "2", "cancellation_absolute_mass": "2"})

    def test_split_mutations_refused(self):
        _, _, _, a = next(self.accounts())
        old = a["unchanged_margin_accounting"]
        effect = {"exact_sum": a["vector_coordinate_sum"],
                  "exact_sum_of_absolute_contributions": a["vector_coordinate_absolute_sum"],
                  **{k: a[k] for k in ("retained_actual_margin", "retained_reference_margin",
                                       "retained_margin_change")},
                  "boundary_remainder_change": a["head_boundary_remainder_change"]}
        ad = tuple(Fraction(a["actual_exact_"+s+"_row_dot"]) for s in ("left", "right"))
        rd = tuple(Fraction(a["reference_exact_"+s+"_row_dot"]) for s in ("left", "right"))
        for key in effect:
            changed = {**effect, key: str(Fraction(effect[key])+1)}
            if key == "exact_sum_of_absolute_contributions":
                changed[key] = "-1"
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.split_boundary(Fraction(a["retained_actual_margin"]),
                                 Fraction(a["retained_reference_margin"]), ad, rd, changed, old)
        with self.assertRaises(ValueError):
            d.split_boundary(Fraction(a["retained_actual_margin"]),
                             Fraction(a["retained_reference_margin"]), ad, rd,
                             effect, {**old, "selected_coordinate_signed_sum": "999"})

    def test_report_census_and_lineage_mutations_refused(self):
        for mutation in ("controls", "pair", "branch", "lineage", "row"):
            e = {**self.e, "margin_report": deepcopy(self.e["margin_report"])}
            pair = e["margin_report"]["controls"][0]["pairs"][0]
            if mutation == "controls":
                e["margin_report"]["controls"].reverse()
            elif mutation == "pair":
                pair["left_id"] = 0
            elif mutation == "branch":
                del pair["branches"]["binary64"]
            elif mutation == "lineage":
                pair["branches"]["fp16"]["rmsnorm_reference"] = "substituted"
            else:
                e["rows"] = {**e["rows"], 0: next(iter(e["rows"].values()))}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d.report(e, audit())

    def test_operand_shape_dtype_and_nonfinite_refusals(self):
        np = d.mixed.np
        a = audit()
        good = np.zeros(d.WIDTH, dtype="<f2")
        for vector, row in ((good[:-1], good), (good, good.astype("<f8")),
                            (np.full(d.WIDTH, np.inf, dtype="<f2"), good),
                            (good.astype("<i2"), good)):
            with self.assertRaises(ValueError):
                d.selected_row_dot(vector, row, a)
        self.assertEqual(a, audit())

    def test_authentication_pin_and_history_mutations_refused(self):
        for pin in (d.base.PINS["review"], d.margin.rows.PINS["contribution_source"],
                    self.e["hidden_pins"][-1], self.e["mixed_input_binding"]["source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        for key, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                           ("source_operand_state_KV_RTZ_checks", "FAIL")):
            changed = deepcopy(self.e["result"])
            changed["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.e["result"])
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)

    def test_original_reference_and_mixed_input_binding(self):
        final = self.e["result"]["preflight"]["final_reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for b in d.BRANCHES:
            self.assertEqual(final["reference"]["input_"+b], original[b])
        self.assertNotEqual(self.vector("fp16"), self.vector("binary64"))
        self.assertEqual(self.e["mixed_input_binding"]["final_reference_manifest"], final["manifest"])
        self.assertEqual(self.e["mixed_input_binding"]["runtime"],
                         {"torch": d.mixed.torch.__version__, "numpy": d.mixed.np.__version__})
        self.assertEqual(self.e["assets"]["tokenizer_status"], "BOUND")
        self.assertEqual(self.e["assets"]["tensors"]["lm_head.weight"],
                         self.e["assets"]["tensors"]["model.embed_tokens.weight"])

    def test_dispatch_guards(self):
        calls = (
            lambda: d.parent.execute("forbidden"), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.norm.rmsnorm(None, None),
            lambda: d.parent.head.decode_array_q24(None),
            lambda: d.mixed.exact_head(None, None), lambda: d.mixed.float_head(None, None),
            lambda: d.mixed.compute(None, None, None), lambda: d.mixed.check(),
            lambda: d.margin.check(), lambda: d.hidden.measure(),
            lambda: d.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden-native"]), lambda: os.system("forbidden-native"),
            lambda: d.mixed.torch.cuda._lazy_init(),
        )
        a = audit()
        with d.read_only(a):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(a["forbidden_calls"], len(calls))

    def test_write_guards(self):
        target = d.parent.OUTPUT / "forbidden-head-boundary-write"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
                 lambda: target.mkdir(), lambda: os.replace(target, target), lambda: target.unlink())
        a = audit()
        with d.read_only(a):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(a["forbidden_calls"], len(calls))

    def test_cli_refuses_replay_outputs_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "forbidden"],
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

    def test_nonadmission_and_lineage_scope(self):
        for key, value in d.FLAGS.items():
            if key == "bounded_selected_row_arithmetic_permitted":
                self.assertIs(value, True)
            else:
                self.assertFalse(value, key)
        self.assertEqual(self.e["report"]["lineage_separation"], d.contributions.LINEAGE)
        self.assertFalse(d.contributions.LINEAGE["old_adjusted_closure_certifies_new_parents"])
        for phrase in ("not rounding-only", "wider than FP16", "no full-vocabulary head",
                       "No token decode", "not nested unselected-coordinate",
                       "Normal independent Reviewer validation REQUIRED"):
            self.assertIn(phrase, d.BOUNDARY)


if __name__ == "__main__":
    unittest.main()
