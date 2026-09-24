"""Independent retained-bit oracle and negative gates for complement residual accounting."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_residual_margin_bridge_v1 as d


EVIDENCE = None


def bits(word, mantissa_bits, exponent_bits, bias):
    exponent = (word >> mantissa_bits) & ((1 << exponent_bits)-1)
    mantissa = word & ((1 << mantissa_bits)-1)
    if exponent == (1 << exponent_bits)-1:
        raise ValueError("nonfinite retained oracle operand")
    value = Fraction(mantissa if exponent == 0 else (1 << mantissa_bits)+mantissa)
    value *= Fraction(2) ** ((1 if exponent == 0 else exponent)-bias-mantissa_bits)
    return -value if word >> (mantissa_bits+exponent_bits) else value


def half(word):
    return bits(int(word), 10, 5, 15)


def double(value):
    return bits(struct.unpack("<Q", struct.pack("<d", value))[0], 52, 11, 1023)


class UnselectedDirectHiddenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.e = d.measure()
        else:
            cls.e = EVIDENCE

    def rows(self):
        for control in self.e["report"]["controls"]:
            for pair in control["pairs"]:
                for branch, account in pair["branches"].items():
                    yield control["control"], pair, branch, account

    @lru_cache(maxsize=None)
    def vector(self, label, stage):
        archive = self.e["reference_archive"] if label == "reference" else self.e["archives"][label]
        return [half(v) for v in archive[stage]]

    def assert_mass(self, observed, values):
        signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
        self.assertEqual(observed, {"signed": str(signed), "absolute": str(absolute),
                                   "cancellation_absolute_mass": str(absolute-abs(signed))})

    def test_independent_retained_bit_components_and_cancellation(self):
        weights = [half(v) for v in self.e["weight_array"].view("<u2")]
        rows = {i: [half(v) for v in a.view("<u2")] for i, a in self.e["rows"].items()}
        binary64 = [double(v) for v in self.e["binary64"]]
        for label, pair, branch, account in self.rows():
            gate = account["gate_values"]
            selected = set(gate["excluded_selected_coordinates"])
            sr = Fraction(account["unchanged_unselected_bridge"]["reference_norm"]["inverse_norm_anchor"])
            values = {k: [] for k in d.COMPONENTS}
            corrections = []
            for i in range(d.WIDTH):
                if i in selected:
                    continue
                a = {k: self.vector(label, k)[i] for k in d.residual.STAGES}
                r = {k: self.vector("reference", k)[i] for k in d.residual.STAGES}
                terminal = r["stage18"] if branch == "fp16" else binary64[i]
                q = Fraction(int(self.e["archives"][label]["output_i"][i]), 1 << 24)
                parts = (
                    a["input_hidden"]-r["input_hidden"], a["stage11"]-r["stage11"],
                    a["stage17"]-r["stage17"],
                    q-a["input_hidden"]-a["stage11"]-a["stage17"],
                    r["input_hidden"]+r["stage11"]+r["stage17"]-r["stage18"],
                    a["stage18"]-q, r["stage18"]-terminal,
                )
                factor = (rows[pair["left_id"]][i]-rows[pair["right_id"]][i])*weights[i]*sr
                for k, v in zip(d.COMPONENTS, parts, strict=True):
                    values[k].append(v*factor)
                corrections.append((a["stage18"]-terminal)*factor)
            for k, v in values.items():
                self.assert_mass(gate["component_totals"][k], v)
            self.assert_mass(gate["unselected_direct_hidden"], corrections)
            absolute = sum(abs(v) for vs in values.values() for v in vs)
            net_absolute = sum(abs(sum(vs)) for vs in values.values())
            target_absolute, target = sum(map(abs, corrections)), sum(corrections)
            for key, value in (
                ("component_absolute_sum", absolute),
                ("within_coordinate_cancellation_mass", absolute-target_absolute),
                ("across_coordinate_cancellation_mass", target_absolute-abs(target)),
                ("total_component_cancellation_mass", absolute-abs(target)),
                ("within_component_across_coordinate_cancellation_mass", absolute-net_absolute),
                ("between_component_net_cancellation_mass", net_absolute-abs(target)),
            ):
                self.assertEqual(Fraction(gate[key]), value)

    def test_exact_margin_change_and_fixed_accounts(self):
        for _, _, _, account in self.rows():
            row, old = account["gate_values"], account["unchanged_unselected_bridge"]
            margin = old["unchanged_margin_accounting"]
            retained = Fraction(row["retained_actual_margin"])-Fraction(row["retained_reference_margin"])
            self.assertEqual(Fraction(row["retained_margin_change"]), retained)
            self.assertEqual(row["retained_margin_change"], margin["retained_margin_change"])
            self.assertEqual(row["selected_coordinate_signed_sum"], margin["selected_coordinate_signed_sum"])
            self.assertEqual(row["head_boundary_remainder_change"], margin["head_boundary_remainder_change"])
            self.assertEqual(row["unselected_coordinate_remainder"],
                             old["unselected_accounting"]["unselected_coordinate_remainder"])
            other = sum(Fraction(v["signed"]) for v in row["unchanged_unselected_categories"].values())
            self.assertEqual(sum(Fraction(v["signed"]) for v in row["component_totals"].values())
                             +other+Fraction(row["selected_coordinate_signed_sum"])
                             +Fraction(row["head_boundary_remainder_change"]), retained)
            for key in d.COMPONENTS:
                self.assertEqual(Fraction(row["neutralized_retained_margin_change"][key]),
                                 retained-Fraction(row["component_totals"][key]["signed"]))

    def test_retained_selected_chain_and_coordinate62(self):
        self.assertIs(self.e["report"]["retained_final_rmsnorm_unselected_coordinate_residual_margin_bridge"],
                      self.e["unselected_report"])
        for label, pair, branch, account in self.rows():
            original = next(p for c in self.e["direct_report"]["controls"] if c["control"] == label
                            for p in c["pairs"] if (p["left_id"], p["right_id"])
                            == (pair["left_id"], pair["right_id"]))["branches"][branch]
            self.assertIs(account["unchanged_selected_direct_hidden"], original)
            selected = [r["coordinate"] for r in original["selected_coordinates"]]
            self.assertEqual(selected, account["gate_values"]["excluded_selected_coordinates"])
            self.assertIn(62, selected)

    def test_independent_terminal_and_lineage(self):
        for _, _, branch, account in self.rows():
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            terminal = account["gate_values"]["component_totals"]["fp16_to_branch_terminal_remainder"]
            if branch == "fp16":
                self.assertEqual(terminal["signed"], "0")
                self.assertEqual(terminal["absolute"], "0")
            else:
                self.assertGreater(Fraction(terminal["absolute"]), 0)
        final = self.e["result"]["preflight"]["final_reference"]["reference"]
        original = self.e["result"]["preflight"]["L23_original_reference"]["reference"]
        for b in d.BRANCHES:
            self.assertEqual(final["input_"+b], original[b])
        self.assertFalse(self.e["report"]["lineage_separation"]["old_adjusted_closure_certifies_new_parents"])

    def fixture(self):
        components = {k: [Fraction() for _ in range(d.WIDTH)] for k in d.COMPONENTS}
        components[d.COMPONENTS[0]][63] = Fraction(3, 7)
        components[d.COMPONENTS[1]][63] = -Fraction(3, 7)
        components[d.COMPONENTS[0]][64] = -Fraction(2, 7)
        components[d.COMPONENTS[1]][64] = Fraction(2, 7)
        one, zero = [Fraction(1)]*d.WIDTH, [Fraction()]*d.WIDTH
        upstream = {"component_totals": {"direct_hidden": {
            "signed": "0", "absolute": "0", "cancellation_absolute_mass": "0"}}}
        return [components, one, one, zero, Fraction(1),
                [0, 1, 2, 3, 4, 5, 6, 62], upstream, Fraction(3, 11)]

    def test_zero_net_nonzero_cancellation(self):
        row = d.split_unselected(*self.fixture())
        self.assertEqual(row["unselected_direct_hidden"]["absolute"], "0")
        self.assertEqual(Fraction(row["component_absolute_sum"]), Fraction(10, 7))
        self.assertEqual(Fraction(row["within_coordinate_cancellation_mass"]), Fraction(10, 7))
        self.assertEqual(Fraction(row["between_component_net_cancellation_mass"]), Fraction(2, 7))
        self.assertEqual(Fraction(row["within_component_across_coordinate_cancellation_mass"]), Fraction(8, 7))

    def test_invalid_union_vectors_and_anchor(self):
        for selected in ([62], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 62, 62],
                         [0, 1, 2, 3, 4, 5, 6, True, 62], [0, 1, 2, 3, 4, 5, 6, 62, 896]):
            args = self.fixture()
            args[5] = selected
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
        for index in (1, 2, 3):
            args = self.fixture()
            args[index] = args[index][:-1]
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
        for anchor in (Fraction(), Fraction(-1), 1.0):
            args = self.fixture()
            args[4] = anchor
            with self.assertRaises(ValueError):
                d.split_unselected(*args)
        args = self.fixture()
        args[0][d.COMPONENTS[0]][63] = 1.0
        with self.assertRaises(ValueError):
            d.split_unselected(*args)

    def test_direct_hidden_account_splices(self):
        for k in ("signed", "absolute", "cancellation_absolute_mass"):
            args = self.fixture()
            args[6]["component_totals"]["direct_hidden"][k] = "1"
            with self.assertRaises(ValueError):
                d.split_unselected(*args)

    def selection_rows(self, values, remainder=10, retained=20):
        totals = {k: {"signed": str(v), "absolute": str(abs(v))}
                  for k, v in zip(d.COMPONENTS, values, strict=True)}
        return [{"control": c, "branch": b, "component_totals": deepcopy(totals),
                 "unselected_direct_hidden": {"signed": str(remainder)},
                 "retained_actual_margin": str(retained+100), "retained_reference_margin": "100",
                 "retained_margin_change": str(retained),
                 "neutralized_retained_margin_change": {
                     k: str(retained-v) for k, v in zip(d.COMPONENTS, values, strict=True)}}
                for c in d.parent.CONTROLS for b in d.BRANCHES]

    def test_selection_half_tie_and_zero_boundaries(self):
        for values, retained, expected in (
            ([6, 2, 1, 1, 0, 0, 0], 20, d.COMPONENTS[0]),
            ([5, 2, 2, 1, 0, 0, 0], 20, "UNKNOWN"),
            ([5, 2, 2, 1, 0, 0, 0], 1, d.COMPONENTS[0]),
            ([5, 2, 2, 1, 0, 0, 0], 5, "UNKNOWN"),
            ([5, 2, 2, 1, 0, 0, 0], 0, "UNKNOWN"),
            ([5, 5, 0, 0, 0, 0, 0], 1, "UNKNOWN"),
            ([0, 0, 0, 0, 0, 0, 0], 0, "UNKNOWN"),
        ):
            self.assertEqual(d.select_component(self.selection_rows(values, retained=retained))["component"],
                             expected)
        self.assertEqual(d.select_component(self.selection_rows(
            [-6, -2, -1, -1, 0, 0, 0], -10, -20))["component"], d.COMPONENTS[0])

    def test_selection_change_not_preference_margin_or_absolute_mass(self):
        rows = self.selection_rows([5, 2, 2, 1, 0, 0, 0], retained=1)
        self.assertEqual(d.select_component(rows)["component"], d.COMPONENTS[0])
        rows = self.selection_rows([5, 2, 2, 1, 0, 0, 0], retained=-1)
        self.assertEqual(d.select_component(rows)["component"], "UNKNOWN")
        rows = self.selection_rows([6, 2, 1, 1, 0, 0, 0])
        for row in rows:
            row["component_totals"][d.COMPONENTS[1]]["absolute"] = "1000000"
        self.assertEqual(d.select_component(rows)["component"], d.COMPONENTS[0])
        rows[0]["retained_reference_margin"] = "0"
        with self.assertRaises(ValueError):
            d.select_component(rows)

    def test_selection_census_and_direction_instability(self):
        rows = self.selection_rows([6, 2, 1, 1, 0, 0, 0])
        for changed in (rows[:-1], list(reversed(rows)), rows+rows[:1]):
            self.assertEqual(d.select_component(changed)["component"], "UNKNOWN")
        rows[-1] = self.selection_rows([-6, -2, -1, -1, 0, 0, 0], -10, -20)[-1]
        self.assertEqual(d.select_component(rows)["component"], "UNKNOWN")
        rows = self.selection_rows([6, 2, 1, 1, 0, 0, 0])
        rows[1] = self.selection_rows([1, 7, 1, 1, 0, 0, 0])[1]
        self.assertEqual(d.select_component(rows)["component"], "UNKNOWN")
        rows[0]["neutralized_retained_margin_change"][d.COMPONENTS[0]] = "0"
        with self.assertRaises(ValueError):
            d.select_component(rows)

    def test_common_selection_requires_all_shared_pairs(self):
        def decision(value, complete=True):
            return {"component": value, "complete_nine_control_two_branch_census": complete}
        self.assertEqual(d.common_selection([]), "UNKNOWN")
        self.assertEqual(d.common_selection([decision("input_hidden"), decision("UNKNOWN")]), "UNKNOWN")
        self.assertEqual(d.common_selection([decision("input_hidden"), decision("mlp_stage17")]), "UNKNOWN")
        self.assertEqual(d.common_selection([decision("input_hidden"), decision("UNKNOWN", False)]),
                         "input_hidden")

    def test_live_gate_independent_oracle(self):
        report = self.e["report"]
        shared_choices = []
        for decision in report["selection"]["pairs"]:
            rows = [a["gate_values"] for _, p, _, a in self.rows()
                    if (p["left_id"], p["right_id"]) == (decision["left_id"], decision["right_id"])]
            choices = []
            for key, evaluation in zip(d.COMPONENTS, decision["component_evaluations"], strict=True):
                values = [Fraction(r["component_totals"][key]["signed"]) for r in rows]
                stable = bool(values) and all(v*values[0] > 0 for v in values)
                complete = [(r["control"], r["branch"]) for r in rows] == [
                    (c, b) for c in d.parent.CONTROLS for b in d.BRANCHES]
                eligible = stable and complete
                for row, v, observed in zip(rows, values, evaluation["rows"], strict=True):
                    largest = all(abs(v) > abs(Fraction(a["signed"]))
                                  for k, a in row["component_totals"].items() if k != key)
                    remainder = Fraction(row["unselected_direct_hidden"]["signed"])
                    half = v*remainder > 0 and 2*abs(v) > abs(remainder)
                    change = Fraction(row["retained_actual_margin"])-Fraction(row["retained_reference_margin"])
                    reverse = change*(change-v) < 0
                    self.assertEqual(observed["unique_largest_net_signed_magnitude"], largest)
                    self.assertEqual(observed["explains_more_than_half"], half)
                    self.assertEqual(observed["retained_margin_change_sign_reversed"], reverse)
                    self.assertEqual(observed["directionally_stable"], stable)
                    self.assertEqual(observed["row_qualifies"], largest and (half or reverse) and stable and complete)
                    eligible &= largest and (half or reverse)
                if eligible:
                    choices.append(key)
            expected = choices[0] if choices else "UNKNOWN"
            self.assertEqual(decision["component"], expected)
            if decision["complete_nine_control_two_branch_census"]:
                shared_choices.append(expected)
        common = shared_choices[0] if shared_choices and len(set(shared_choices)) == 1 else "UNKNOWN"
        self.assertEqual(report["selection"]["component"], common)
        self.assertEqual(report["selection"]["stop_nested_bridge_expansion"], common == "UNKNOWN")

    def test_report_parent_gate_and_linkage_splices(self):
        old = self.e["unselected_report"]
        for key, value in (("retained_final_rmsnorm_boundary_residual_margin_bridge", {}),
                           ("controls", list(reversed(old["controls"]))),
                           ("selection", {**old["selection"], "mechanism": "UNKNOWN"})):
            with self.assertRaises(ValueError):
                d.report({**self.e, "unselected_report": {**old, key: value}})

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

    def test_source_original_authority_and_tokenizer_gates(self):
        for pin in (d.base.PINS["review"], *self.e["hidden_pins"],
                    d.margin.rows.PINS["rmsnorm_hotspot_source"]):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0"*64})
        changed = deepcopy(self.e["result"])
        changed["preflight"]["final_reference"]["reference"]["input_fp16"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            d.residual.load_residuals(changed)
        assets = self.e["assets"]
        self.assertEqual(assets["tokenizer_status"], "BOUND")
        self.assertFalse(assets["missing_tokenizer"])
        self.assertEqual(assets["tensors"]["lm_head.weight"], assets["tensors"]["model.embed_tokens.weight"])
        with self.assertRaises(ValueError):
            d.hidden.validate_weight(self.e["weight_array"],
                                     {**assets["tensors"]["model.norm.weight"], "sha256": "0"*64})

    def test_state_kv_gates_and_nonadmission_census(self):
        archive = self.e["archives"][d.parent.CONTROLS[0]]
        for key in ("output_z", "output_cache_k"):
            changed = {**archive, key: archive[key].copy()}
            changed[key].flat[0] = 2 if key == "output_z" else int(changed[key].flat[0]) ^ 1
            with self.assertRaises(ValueError):
                d.residual.operands(changed, actual=True)
        report = self.e["report"]
        self.assertEqual(report["control_count"], 9)
        self.assertEqual(report["pair_branch_count"], 2*report["diagnostic_pair_count"])
        self.assertEqual(report["coordinate62_accounts"], report["pair_branch_count"])
        self.assertEqual(report["selected_coordinate_accounts"]+report["unselected_coordinate_accounts"],
                         896*report["pair_branch_count"])
        self.assertEqual(report["weighted_residual_component_count"], 7*report["unselected_coordinate_accounts"])
        self.assertTrue(all(not v for v in d.FLAGS.values()))

    def test_forbidden_dispatch_and_earlier_checks(self):
        calls = [
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None), lambda: d.parent.execute("forbidden"),
            lambda: d.residual.local.local_reference(None), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.margin.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden-native"]), lambda: os.system("forbidden-native"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        for module in (d.unselected, d.boundary, d.interaction, d.scale, d.direct, d.margin, d.hidden, d.residual):
            calls.extend((lambda m=module: m.check(), lambda m=module: m.measure(),
                          lambda m=module: m.focused_tests(None)))
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-unselected-direct-hidden-bridge"
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

    def test_stdout_exactly_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value=self.e["report"]), patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        self.assertEqual(json.loads(text), self.e["report"])
        self.assertEqual(text.count("\n"), 1)
