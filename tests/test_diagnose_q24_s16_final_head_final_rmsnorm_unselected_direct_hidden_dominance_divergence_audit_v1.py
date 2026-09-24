"""Independent rational input oracle, census binding and no-replay refusals."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_dominance_divergence_audit_v1 as d


EVIDENCE = CENSUS = OBSERVED = None
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
    "negative_reference_residual_boundary", "q24_to_fp16_conversion",
    "fp16_to_branch_terminal_remainder",
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down", "frozen_inherited_o_down",
    "scratch", "scratch_down", "mapped62", "mapped_all", "inherited_native",
)
BRANCHES = ("fp16", "binary64")
PAIRS = ((319, 34319), (34319, 13), (34319, 319))
MLP, TERMINAL = COMPONENTS[2], COMPONENTS[6]


def input_oracle(source):
    values = [Fraction(source["component_totals"][k]["signed"]) for k in COMPONENTS]
    magnitudes = list(map(abs, values))
    winners = [k for k, value in zip(COMPONENTS, magnitudes, strict=True)
               if all(value >= other for other in magnitudes)]
    return values, magnitudes, winners


class DominanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.bridge.read_only({"forbidden_calls": 0}):
                cls.e = d.bridge.measure()
            with d.read_only({"forbidden_calls": 0}):
                cls.census = d.census.census(cls.e["report"])
                cls.observed = d.audit_census(cls.census, cls.e["report"])
        else:
            cls.e, cls.census, cls.observed = EVIDENCE, CENSUS, OBSERVED
        cls.inputs = {
            (p["left_id"], p["right_id"]): p for p in cls.census["pairs"]
        }
        cls.pairs = {(p["left_id"], p["right_id"]): p for p in cls.observed["pairs"]}

    def test_exact_all_shared_rows_against_independent_input_oracle(self):
        count = 0
        for identity in PAIRS:
            sources = self.inputs[identity]["exact_accounting_rows"]
            actual = self.pairs[identity]["rows"]
            self.assertEqual(len(sources), 18)
            for source, row in zip(sources, actual, strict=True):
                values, magnitudes, winners = input_oracle(source)
                self.assertEqual((row["control"], row["branch"]), (source["control"], source["branch"]))
                self.assertEqual(row["largest_components"], winners)
                self.assertEqual(row["unique_largest_component"], winners[0])
                self.assertEqual(row["largest_net_signed_magnitude"], str(max(magnitudes)))
                self.assertEqual(row["mlp_minus_terminal_net_magnitude"],
                                 str(magnitudes[2]-magnitudes[6]))
                for index, component in enumerate(row["components"]):
                    self.assertEqual(component["component"], COMPONENTS[index])
                    self.assertEqual(component["signed_correction"], str(values[index]))
                    self.assertEqual(component["net_signed_magnitude"], str(magnitudes[index]))
                    self.assertEqual(component["magnitude_gap_against_mlp_stage17"],
                                     str(magnitudes[index]-magnitudes[2]))
                    self.assertEqual(component["magnitude_gap_against_terminal_remainder"],
                                     str(magnitudes[index]-magnitudes[6]))
                    self.assertEqual(component["larger_components"], [
                        k for k, v in zip(COMPONENTS, magnitudes, strict=True) if v > magnitudes[index]])
                    self.assertEqual(component["equal_net_magnitude_components"], [
                        k for k, v in zip(COMPONENTS, magnitudes, strict=True)
                        if k != COMPONENTS[index] and v == magnitudes[index]])
                    self.assertEqual(component["unique_largest_net_signed_magnitude"],
                                     all(magnitudes[index] > v for i, v in enumerate(magnitudes) if i != index))
                    self.assertEqual(component["retained_gate_row"],
                                     self.inputs[identity]["component_evaluations"][index]["rows"][
                                         CONTROLS.index(source["control"])*2+BRANCHES.index(source["branch"])])
                    count += 1
        self.assertEqual(count, 378)

    def test_nine_mlp_failures_and_exact_representative_rows(self):
        for identity in ((319, 34319), (34319, 319)):
            expected = []
            for source in self.inputs[identity]["exact_accounting_rows"]:
                values, magnitudes, winners = input_oracle(source)
                if any(v >= magnitudes[2] for i, v in enumerate(magnitudes) if i != 2):
                    expected.append((source["control"], source["branch"], values, magnitudes, winners))
            failures = self.pairs[identity]["mlp_stage17_unique_largest_failure_rows"]
            self.assertEqual(len(expected), 9)
            self.assertEqual(self.pairs[identity]["mlp_stage17_unique_largest_failure_count"], 9)
            for (control, branch, values, magnitudes, winners), row in zip(expected, failures, strict=True):
                self.assertEqual((row["control"], row["branch"]), (control, branch))
                self.assertEqual(row["signed_correction"], str(values[2]))
                self.assertEqual(row["mlp_minus_terminal_net_magnitude"], str(magnitudes[2]-magnitudes[6]))
                self.assertEqual(row["magnitude_gap_against_terminal_remainder"],
                                 str(abs(values[2])-abs(values[6])))
                self.assertEqual(row["largest_components"], winners)
                self.assertEqual(row["larger_components"], [
                    k for k, v in zip(COMPONENTS, magnitudes, strict=True) if v > magnitudes[2]])
                self.assertEqual(row["equal_net_magnitude_components"], [])
                self.assertTrue(row["retained_gate_row"]["explains_more_than_half"])
                self.assertIn("unique_largest_net_signed_magnitude", row["retained_gate_row"]["blocking_conditions"])
            self.assertEqual(self.pairs[identity]["component"], "UNKNOWN")
        selected = self.pairs[(34319, 13)]
        self.assertEqual(selected["component"], MLP)
        self.assertEqual(selected["mlp_stage17_unique_largest_failure_rows"], [])
        self.assertTrue(all(r["unique_largest_component"] == MLP for r in selected["rows"]))

    def test_branch_and_control_dominance_partitions(self):
        for identity in PAIRS:
            sources = self.inputs[identity]["exact_accounting_rows"]
            oracle = {(r["control"], r["branch"]): input_oracle(r)[2] for r in sources}
            expected_branches = [{
                "branch": b, "controls_by_unique_largest_component": {
                    k: [c for c in CONTROLS if oracle[c, b] == [k]] for k in COMPONENTS},
                "tied_controls": [c for c in CONTROLS if len(oracle[c, b]) != 1],
            } for b in BRANCHES]
            expected_controls = [{
                "control": c, "largest_components_by_branch": {b: oracle[c, b] for b in BRANCHES},
                "branch_winners_differ": oracle[c, "fp16"] != oracle[c, "binary64"],
            } for c in CONTROLS]
            self.assertEqual(self.pairs[identity]["dominance_partitions"],
                             {"by_branch": expected_branches, "by_control": expected_controls})

    def test_failure_partitions_keep_all_controls_including_empty_buckets(self):
        for identity in PAIRS:
            oracle = {(r["control"], r["branch"]): input_oracle(r)[2]
                      for r in self.inputs[identity]["exact_accounting_rows"]}
            expected = {
                "by_branch": [{"branch": b, "controls": [
                    c for c in CONTROLS if oracle[c, b] != [MLP]]} for b in BRANCHES],
                "by_control": [{"control": c, "branches": [
                    b for b in BRANCHES if oracle[c, b] != [MLP]]} for c in CONTROLS],
            }
            self.assertEqual(self.pairs[identity]["mlp_stage17_failure_partitions"], expected)

    def test_terminal_fp16_zero_boundaries_are_exact_not_reversals(self):
        for identity in PAIRS:
            rows = self.pairs[identity]["terminal_fp16_zero_boundary_rows"]
            self.assertEqual([r["control"] for r in rows], list(CONTROLS))
            for row, source in zip(rows, self.inputs[identity]["exact_accounting_rows"][::2], strict=True):
                self.assertEqual(row["branch"], "fp16")
                self.assertEqual(row["signed_correction"], "0")
                self.assertEqual(row["magnitude_gap_against_terminal_remainder"], "0")
                self.assertEqual(row["magnitude_gap_against_mlp_stage17"],
                                 str(-abs(Fraction(source["component_totals"][MLP]["signed"]))))
                gate = row["retained_gate_row"]
                retained = Fraction(source["retained_actual_margin"])-Fraction(source["retained_reference_margin"])
                self.assertEqual(gate["neutralized_retained_margin_change"], str(retained))
                self.assertEqual(gate["reversal_product"], str(retained*retained))
                self.assertTrue(gate["zero_boundaries"]["correction"])
                self.assertFalse(gate["directionally_stable"])
                self.assertFalse(gate["explains_more_than_half"])
                self.assertFalse(gate["retained_margin_change_sign_reversed"])

    def test_full_census_no_ties_and_stop_preserved(self):
        self.assertEqual(tuple(self.pairs), PAIRS)
        self.assertEqual(self.observed["shared_pair_count"], 3)
        self.assertEqual(self.observed["shared_row_count"], 54)
        self.assertEqual(self.observed["shared_component_row_count"], 378)
        self.assertEqual(self.observed["common_component"], "UNKNOWN")
        self.assertTrue(self.observed["stop_nested_bridge_expansion"])
        self.assertTrue(self.observed["all_shared_censuses_complete"])
        self.assertEqual(self.observed["shared_largest_tie_row_count"], 0)
        equal_rows = 0
        for identity in PAIRS:
            for source in self.inputs[identity]["exact_accounting_rows"]:
                _, magnitudes, _ = input_oracle(source)
                equal_rows += sum(any(v == other for j, other in enumerate(magnitudes) if j != i)
                                  for i, v in enumerate(magnitudes))
        self.assertEqual(self.observed["shared_equal_magnitude_component_row_count"], equal_rows)
        self.assertEqual(self.observed["retained_obstruction_census"], self.census)
        nonshared = [{"left_id": p["left_id"], "right_id": p["right_id"]}
                     for p in self.census["pairs"] if (p["left_id"], p["right_id"]) not in PAIRS]
        self.assertEqual(self.observed["nonshared_pair_identities"], nonshared)
        self.assertEqual(self.observed["nonshared_pair_count"], len(nonshared))
        for pair in self.pairs.values():
            self.assertEqual(pair["row_count"], 18)
            self.assertEqual(pair["component_row_count"], 126)
            self.assertEqual(pair["largest_tie_rows"], [])
            self.assertEqual(pair["census_proof"], {
                "complete_nine_control_two_branch_census": True, "expected_row_count": 18,
                "observed_row_count": 18, "missing_rows": [], "unexpected_rows": [],
                "duplicate_rows": [], "canonical_row_order": True,
            })
            self.assertEqual([(r["control"], r["branch"]) for r in pair["rows"]],
                             [(c, b) for c in CONTROLS for b in BRANCHES])

    def test_signed_magnitude_gaps_ties_and_absolute_mass_are_distinct(self):
        source = {"control": "fixture", "branch": "fp16", "component_totals": {
            k: {"signed": v, "absolute": "1000000"}
            for k, v in zip(COMPONENTS, ("1/7", "-3/7", "2/7", "0", "0", "0", "-3/7"), strict=True)}}
        row = d.row_dominance(source)
        self.assertEqual(row["largest_components"], ["attention_stage11", TERMINAL])
        self.assertEqual(row["unique_largest_component"], "UNKNOWN")
        self.assertEqual(row["mlp_minus_terminal_net_magnitude"], "-1/7")
        mlp = row["components"][2]
        self.assertEqual(mlp["magnitude_gap_against_mlp_stage17"], "0")
        self.assertEqual(mlp["magnitude_gap_against_terminal_remainder"], "-1/7")
        self.assertEqual(mlp["larger_components"], ["attention_stage11", TERMINAL])
        terminal = row["components"][6]
        self.assertEqual(terminal["signed_correction"], "-3/7")
        self.assertEqual(terminal["magnitude_gap_against_mlp_stage17"], "1/7")
        self.assertEqual(terminal["equal_net_magnitude_components"], ["attention_stage11"])
        self.assertFalse(terminal["unique_largest_net_signed_magnitude"])

    def test_tampered_census_rows_gates_and_counts_are_rejected(self):
        original = self.census["pairs"][0]
        changes = (
            lambda p: p["exact_accounting_rows"].pop(),
            lambda p: p["component_evaluations"][0]["rows"][0].__setitem__(
                "unique_largest_net_signed_magnitude",
                not p["component_evaluations"][0]["rows"][0]["unique_largest_net_signed_magnitude"]),
            lambda p: p["component_evaluations"][0]["rows"][0].__setitem__("larger_components", []),
            lambda p: p["exact_accounting_rows"][0]["component_totals"][MLP].__setitem__("signed", "0"),
            lambda p: p.__setitem__("component", "input_hidden"),
            lambda p: p.__setitem__("observed_row_count", 0),
        )
        for change in changes:
            pair = deepcopy(original)
            change(pair)
            with self.assertRaises(ValueError):
                d.audit_census({**self.census, "pairs": [pair, *self.census["pairs"][1:]]}, self.e["report"])
        for field, value in (("pairs", self.census["pairs"][1:]), ("shared_pair_count", 0),
                             ("reference_scope", "substituted"), ("stop_nested_bridge_expansion", False)):
            with self.assertRaises(ValueError):
                d.audit_census({**self.census, field: value}, self.e["report"])

    def test_bridge_gate_and_reference_splices_are_rejected(self):
        report = self.e["report"]
        for field, value in (("reference_scope", "substituted"), ("controls", report["controls"][:-1])):
            with self.assertRaises(ValueError):
                d.audit_census(self.census, {**report, field: value})
        pair = deepcopy(report["selection"]["pairs"][0])
        gate = pair["component_evaluations"][0]["rows"][0]
        gate["unique_largest_net_signed_magnitude"] = not gate["unique_largest_net_signed_magnitude"]
        with self.assertRaises(ValueError):
            d.audit_census(self.census, {**report, "selection": {
                **report["selection"], "pairs": [pair, *report["selection"]["pairs"][1:]]}})

    def test_historical_failures_exact_thresholds_and_reference_gates(self):
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
        for branch in BRANCHES:
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.e["margin_report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])
        self.assertTrue(all(not value for value in d.FLAGS.values()))

    def test_source_pin_tamper_is_rejected(self):
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.bridge.margin.rows.PINS["rmsnorm_hotspot_source"], "sha256": "0"*64})

    def test_forbidden_predecessor_checks_and_operator_dispatch(self):
        calls = (
            lambda: d.census.check(), lambda: d.census.run_tests(None, None),
            lambda: d.bridge.check(), lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.bridge.component_vectors(None, None, None),
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
        target = d.parent.OUTPUT / "forbidden-dominance-audit"
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
        decoded, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(decoded, {"report": self.observed})
        self.assertEqual(text[end:], "\n")
