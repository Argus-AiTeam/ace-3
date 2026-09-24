"""Independent exact gate examples and authenticated bridge mutation refusals."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_gate_obstruction_census_v1 as d


EVIDENCE = OBSERVED = None
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
    "negative_reference_residual_boundary", "q24_to_fp16_conversion",
    "fp16_to_branch_terminal_remainder",
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down", "frozen_inherited_o_down",
    "scratch", "scratch_down", "mapped62", "mapped_all", "inherited_native",
)


def rows(values=(6, 2, 1, 1, 0, 0, 0), retained=20, absolute=None, aggregate_absolute=None):
    values = list(map(Fraction, values))
    absolute = list(map(Fraction, absolute)) if absolute is not None else list(map(abs, values))
    aggregate = sum(values)
    aa = sum(map(abs, values)) if aggregate_absolute is None else Fraction(aggregate_absolute)
    total, net = sum(absolute), sum(map(abs, values))
    return [{
        "control": c, "branch": b,
        "component_totals": {
            k: {"signed": str(v), "absolute": str(a), "cancellation_absolute_mass": str(a-abs(v))}
            for k, v, a in zip(COMPONENTS, values, absolute, strict=True)},
        "unselected_direct_hidden": {
            "signed": str(aggregate), "absolute": str(aa), "cancellation_absolute_mass": str(aa-abs(aggregate))},
        "component_absolute_sum": str(total),
        "within_coordinate_cancellation_mass": str(total-aa),
        "across_coordinate_cancellation_mass": str(aa-abs(aggregate)),
        "total_component_cancellation_mass": str(total-abs(aggregate)),
        "within_component_across_coordinate_cancellation_mass": str(total-net),
        "between_component_net_cancellation_mass": str(net-abs(aggregate)),
        "retained_actual_margin": str(Fraction(retained)+100), "retained_reference_margin": "100",
        "retained_margin_change": str(Fraction(retained)),
        "neutralized_retained_margin_change": {k: str(Fraction(retained)-v)
                                              for k, v in zip(COMPONENTS, values, strict=True)},
    } for c in CONTROLS for b in ("fp16", "binary64")]


class ObstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.bridge.read_only({"forbidden_calls": 0}):
                cls.e = d.bridge.measure()
            with d.read_only({"forbidden_calls": 0}):
                cls.observed = d.census(cls.e["report"])
        else:
            cls.e, cls.observed = EVIDENCE, OBSERVED

    def first(self, fixture):
        return d.pair_census(fixture)["component_evaluations"][0]

    def changed_gate(self, change):
        report = self.e["report"]
        control = report["controls"][0]
        pair = control["pairs"][0]
        branch = pair["branches"]["fp16"]
        gate = deepcopy(branch["gate_values"])
        change(gate)
        return {**report, "controls": [
            {**control, "pairs": [{**pair, "branches": {
                **pair["branches"], "fp16": {**branch, "gate_values": gate}}},
                *control["pairs"][1:]]}, *report["controls"][1:]]}

    def test_unique_largest_uses_net_not_absolute_account(self):
        self.assertEqual(d.pair_census(rows(absolute=(6, 1000000, 1, 1, 0, 0, 0)))["component"],
                         "input_hidden")
        e = self.first(rows((2, 6, 1, 1, 0, 0, 0)))
        self.assertEqual(e["gate_failure_counts"]["unique_largest_net_signed_magnitude"], 18)
        self.assertEqual(e["rows"][0]["larger_components"], ["attention_stage11"])
        self.assertIn("unique_largest_net_signed_magnitude", e["rows"][0]["blocking_conditions"])

    def test_direction_instability(self):
        fixture = rows()
        fixture[-1] = rows((-6, -2, -1, -1, 0, 0, 0), -20)[-1]
        e = self.first(fixture)
        self.assertEqual(e["direction_counts"], {"-1": 1, "0": 0, "1": 17})
        self.assertEqual(e["gate_failure_counts"]["directionally_stable"], 18)
        self.assertFalse(e["qualifies"])

    def test_strict_half_boundary(self):
        e = self.first(rows((5, 2, 2, 1, 0, 0, 0)))
        self.assertEqual(e["gate_failure_counts"]["explains_more_than_half"], 18)
        self.assertEqual(e["strict_half_equality_count"], 18)
        self.assertEqual(e["rows"][0]["twice_magnitude_minus_aggregate_magnitude"], "0")
        self.assertEqual(e["neither_half_nor_reversal_count"], 18)
        self.assertFalse(e["qualifies"])

    def test_reversal_failure_and_alternative_success(self):
        fixture = (5, 2, 2, 1, 0, 0, 0)
        failure = self.first(rows(fixture, -1))
        self.assertEqual(failure["gate_failure_counts"]["retained_margin_change_sign_reversed"], 18)
        self.assertEqual(failure["rows"][0]["reversal_product"], "6")
        self.assertFalse(failure["qualifies"])
        success = self.first(rows(fixture, 1))
        self.assertEqual(success["rows"][0]["reversal_product"], "-4")
        self.assertEqual(success["rows"][0]["failed_gates"], ["explains_more_than_half"])
        self.assertEqual(success["rows"][0]["blocking_conditions"], [])
        self.assertTrue(success["qualifies"])
        self.assertTrue(self.first(rows())["qualifies"])

    def test_complete_census_missing_duplicate_and_order(self):
        fixture = rows()
        missing = d.pair_census(fixture[:-1])
        self.assertEqual(missing["missing_rows"], [["inherited_native", "binary64"]])
        self.assertEqual(missing["component"], "UNKNOWN")
        self.assertEqual(missing["component_evaluations"][0]["gate_failure_counts"][
            "complete_nine_control_two_branch_census"], 17)
        duplicate = d.pair_census(fixture+fixture[:1])
        self.assertEqual(duplicate["duplicate_rows"], [
            {"control": "frozen_inherited", "branch": "fp16", "count": 2}])
        self.assertEqual(duplicate["component"], "UNKNOWN")
        self.assertFalse(d.pair_census(list(reversed(fixture)))["canonical_row_order"])
        self.assertEqual(d.pair_census([])["component"], "UNKNOWN")

    def test_ties(self):
        e = self.first(rows((5, -5, 0, 0, 0, 0, 0), 1))
        self.assertEqual(e["tied_for_largest_count"], 18)
        self.assertEqual(e["rows"][0]["equal_net_magnitude_components"], ["attention_stage11"])
        self.assertTrue(e["rows"][0]["retained_margin_change_sign_reversed"])
        self.assertFalse(e["qualifies"])

    def test_zero_boundaries_are_not_reversals(self):
        e = self.first(rows((0, 0, 0, 0, 0, 0, 0), 0))
        self.assertEqual(e["zero_boundary_counts"], {
            "correction": 18, "direct_hidden_aggregate": 18,
            "retained_margin_change": 18, "neutralized_margin_change": 18})
        self.assertFalse(e["qualifies"])
        for retained, field in ((0, "retained_margin_change"), (5, "neutralized_margin_change")):
            e = self.first(rows((5, 2, 2, 1, 0, 0, 0), retained))
            self.assertEqual(e["zero_boundary_counts"][field], 18)
            self.assertEqual(e["gate_failure_counts"]["retained_margin_change_sign_reversed"], 18)
            self.assertFalse(e["qualifies"])

    def test_exact_cancellation_totals(self):
        fixture = rows((Fraction(1, 7), Fraction(-1, 7), 0, 0, 0, 0, 0),
                       Fraction(3, 11), (Fraction(5, 7), Fraction(5, 7), 0, 0, 0, 0, 0), 0)
        p = d.pair_census(fixture)
        account = p["exact_accounting_rows"][0]
        self.assertEqual(account["component_absolute_sum"], "10/7")
        self.assertEqual(account["within_coordinate_cancellation_mass"], "10/7")
        self.assertEqual(account["within_component_across_coordinate_cancellation_mass"], "8/7")
        self.assertEqual(account["between_component_net_cancellation_mass"], "2/7")
        self.assertEqual(p["component_evaluations"][0]["rows"][0]["component_net_magnitude_sum"], "2/7")
        fixture[0]["between_component_net_cancellation_mass"] = "0"
        with self.assertRaises(ValueError):
            d.pair_census(fixture)

    def test_independent_live_gate_oracle(self):
        shared = []
        for pair in self.observed["pairs"]:
            source = pair["exact_accounting_rows"]
            complete = [(r["control"], r["branch"]) for r in source] == [
                (c, b) for c in CONTROLS for b in ("fp16", "binary64")]
            choices = []
            for key, evaluation in zip(COMPONENTS, pair["component_evaluations"], strict=True):
                corrections = [Fraction(r["component_totals"][key]["signed"]) for r in source]
                stable = bool(corrections) and all(v*corrections[0] > 0 for v in corrections)
                qualifies = complete and stable
                observed_failures = dict.fromkeys((
                    "unique_largest_net_signed_magnitude", "directionally_stable",
                    "explains_more_than_half", "retained_margin_change_sign_reversed",
                    "complete_nine_control_two_branch_census"), 0)
                for original, observed, value in zip(source, evaluation["rows"], corrections, strict=True):
                    competitors = [abs(Fraction(a["signed"])) for k, a in original["component_totals"].items()
                                   if k != key]
                    aggregate = Fraction(original["unselected_direct_hidden"]["signed"])
                    change = Fraction(original["retained_actual_margin"])-Fraction(original["retained_reference_margin"])
                    largest = abs(value) > max(competitors)
                    half = value*aggregate > 0 and abs(value) > abs(aggregate)/2
                    reversal = (change < 0 < change-value) or (change-value < 0 < change)
                    for gate, expected in zip(observed_failures,
                                              (largest, stable, half, reversal, complete), strict=True):
                        self.assertEqual(observed[gate], expected)
                        observed_failures[gate] += not expected
                    self.assertEqual(observed["row_qualifies"], complete and stable and largest and (half or reversal))
                    qualifies &= largest and (half or reversal)
                self.assertEqual(evaluation["gate_failure_counts"], observed_failures)
                self.assertEqual(evaluation["qualifies"], qualifies)
                if qualifies:
                    choices.append(key)
            expected = choices[0] if choices else "UNKNOWN"
            self.assertEqual(pair["component"], expected)
            if complete:
                shared.append(expected)
        self.assertTrue(shared)
        common = shared[0] if len(set(shared)) == 1 else "UNKNOWN"
        self.assertEqual(self.observed["common_component"], common)
        self.assertEqual(self.observed["stop_nested_bridge_expansion"], common == "UNKNOWN")

    def test_tampered_bridge_and_gate_inputs(self):
        for field, value in (
            ("retained_actual_margin", "0"), ("component_absolute_sum", "0"),
            ("selected_coordinate_signed_sum", "0"), ("control", "scratch"),
            ("excluded_selected_coordinates", []),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.census(self.changed_gate(lambda g: g.__setitem__(field, value)))
        with self.assertRaises(ValueError):
            d.census(self.changed_gate(lambda g: g["component_totals"]["input_hidden"].__setitem__("signed", "0")))
        with self.assertRaises(ValueError):
            d.census(self.changed_gate(lambda g: g["neutralized_retained_margin_change"].__setitem__("input_hidden", "0")))
        report = self.e["report"]
        decision = deepcopy(report["selection"]["pairs"][0])
        gate = decision["component_evaluations"][0]["rows"][0]
        gate["unique_largest_net_signed_magnitude"] = not gate["unique_largest_net_signed_magnitude"]
        with self.assertRaises(ValueError):
            d.census({**report, "selection": {
                **report["selection"], "pairs": [decision, *report["selection"]["pairs"][1:]]}})

    def test_parent_gate_census_and_reference_refusals(self):
        report = self.e["report"]
        for changed in (
            {**report, "controls": report["controls"][:-1]},
            {**report, "reference_scope": "substituted"},
            {**report, "selection": {**report["selection"], "parent_category": "UNKNOWN"}},
            {**report, "selection": {**report["selection"], "component": "input_hidden"}},
        ):
            with self.assertRaises(ValueError):
                d.census(changed)
        original = report["retained_final_rmsnorm_unselected_coordinate_residual_margin_bridge"]
        changed = {**original, "selection": {**original["selection"], "mechanism": "UNKNOWN"}}
        with self.assertRaises(ValueError):
            d.census({**report, "retained_final_rmsnorm_unselected_coordinate_residual_margin_bridge": changed})

    def test_history_source_and_nonadmission_gates(self):
        for field, value in (("mandatory_statuses", ["PASS"]*19), ("S18_failure_indices", []),
                             ("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            result = deepcopy(self.e["result"])
            result["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(result)
        result = deepcopy(self.e["result"])
        result["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1"
        with self.assertRaises(ValueError):
            d.base.check_history(result)
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.bridge.margin.rows.PINS["rmsnorm_hotspot_source"], "sha256": "0"*64})
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.e["margin_report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])
        self.assertTrue(all(not v for v in d.FLAGS.values()))

    def test_forbidden_dispatch(self):
        calls = (
            lambda: d.bridge.check(), lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.bridge.focused_tests(None), lambda: d.bridge.component_vectors(None, None, None),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: d.bridge.hidden.check(), lambda: subprocess.run(["forbidden"]),
            lambda: os.system("forbidden"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-obstruction-census"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_rejects_dispatch_output_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_is_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.observed}), patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        self.assertEqual(json.loads(text), {"report": self.observed})
        self.assertEqual(text.count("\n"), 1)
