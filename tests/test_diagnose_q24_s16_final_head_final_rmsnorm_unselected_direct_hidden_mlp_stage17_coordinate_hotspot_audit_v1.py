"""Independent retained-bit MLP coordinate oracle and no-replay refusals."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_coordinate_hotspot_audit_v1 as d


EVIDENCE = CENSUS = DOMINANCE = OBSERVED = None
CONTROLS = ("frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native")


def oracle(evidence, control, branch, selected, anchor):
    ref, actual = evidence["reference_archive"], evidence["archives"][control]
    fp16 = ref["stage18"].view("<f2")
    terminal = fp16 if branch == "fp16" else evidence["binary64"]
    weights = evidence["weight_array"]
    left, right = evidence["rows"][34319], evidence["rows"][13]
    mlp, ref_mlp = actual["stage17"].view("<f2"), ref["stage17"].view("<f2")
    q = lambda v: Fraction(float(v))
    result = []
    for i in range(896):
        if i not in selected:
            factor = q(weights[i])*anchor*(q(left[i])-q(right[i]))
            result.append((i, factor, q(mlp[i])-q(ref_mlp[i]), q(fp16[i])-q(terminal[i])))
    return result


class MlpCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.bridge.read_only({"forbidden_calls": 0}):
                cls.e = d.bridge.measure()
            with d.read_only({"forbidden_calls": 0}):
                cls.c = d.census.census(cls.e["report"])
                cls.dom = d.dominance.audit_census(cls.c, cls.e["report"])
                cls.report = d.audit_coordinates(cls.e, cls.c, cls.dom)
        else:
            cls.e, cls.c, cls.dom, cls.report = EVIDENCE, CENSUS, DOMINANCE, OBSERVED
        cls.inputs = {}
        cls.expected = {}
        for control in cls.e["report"]["controls"]:
            pair = next(p for p in control["pairs"] if (p["left_id"], p["right_id"]) == (34319, 13))
            for branch, source in pair["branches"].items():
                old = source["unchanged_unselected_bridge"]
                selected = [r["coordinate"] for r in old["selected_coordinates"]]
                anchor = Fraction(old["reference_norm"]["inverse_norm_anchor"])
                key = control["control"], branch
                cls.inputs[key] = source
                cls.expected[key] = oracle(cls.e, *key, selected, anchor)
        cls.pair = next(p for p in cls.e["report"]["controls"][0]["pairs"]
                        if (p["left_id"], p["right_id"]) == (34319, 13))

    def test_exact_coordinate_oracle_and_signed_absolute_closure(self):
        rows = self.report["pairs"][0]["rows"]
        self.assertEqual([(r["control"], r["branch"]) for r in rows],
                         [(c, b) for c in CONTROLS for b in ("fp16", "binary64")])
        for row in rows:
            key = row["control"], row["branch"]
            source = self.inputs[key]
            selected = [r["coordinate"] for r in
                        source["unchanged_unselected_bridge"]["selected_coordinates"]]
            expected = self.expected[key]
            self.assertEqual(row["excluded_selected_coordinates"], selected)
            self.assertEqual(row["coordinate_count"], 896-len(selected))
            self.assertEqual([r["coordinate"] for r in row["coordinates"]], [r[0] for r in expected])
            self.assertNotIn(62, [r[0] for r in expected])
            for actual, (i, factor, mlp, terminal) in zip(row["coordinates"], expected, strict=True):
                self.assertEqual(actual["coordinate"], i)
                for field, value in (
                    ("retained_weighted_row_factor", factor), ("mlp_stage17_hidden_delta", mlp),
                    ("terminal_hidden_delta", terminal), ("mlp_stage17_signed_contribution", factor*mlp),
                    ("terminal_signed_contribution", factor*terminal),
                    ("mlp_stage17_absolute_contribution", abs(factor*mlp)),
                    ("terminal_absolute_contribution", abs(factor*terminal)),
                    ("mlp_minus_terminal_coordinate_magnitude", abs(factor*mlp)-abs(factor*terminal)),
                ):
                    self.assertEqual(actual[field], str(value))
            for index, component in ((2, d.MLP), (3, d.TERMINAL)):
                values = [r[1]*r[index] for r in expected]
                signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
                total = {"signed": str(signed), "absolute": str(absolute),
                         "cancellation_absolute_mass": str(absolute-abs(signed))}
                self.assertEqual(row["component_totals"][component], total)
                self.assertEqual(source["gate_values"]["component_totals"][component], total)
                self.assertEqual(row["coordinate_closure_residuals"][component],
                                 dict.fromkeys(total, "0"))
            self.assertEqual(row["retained_gate_values"], source["gate_values"])
            self.assertTrue(row["exact_signed_and_absolute_closure"])

    def test_representative_exact_mlp_hotspot_rows(self):
        for row in self.report["pairs"][0]["rows"]:
            ranked = sorted(self.expected[row["control"], row["branch"]],
                            key=lambda r: (-abs(r[1]*r[2]), r[0]))
            top = [r for r in ranked if r[1]*r[2]][:8]
            self.assertEqual(len(top), 8)
            self.assertEqual(row["mlp_stage17_absolute_coordinate_order"], [r[0] for r in ranked])
            self.assertEqual([r["coordinate"] for r in row["top_mlp_stage17_coordinates"]],
                             [r[0] for r in top])
            for observed, (i, factor, mlp, terminal) in zip(
                    row["top_mlp_stage17_coordinates"], top, strict=True):
                self.assertEqual(observed["coordinate"], i)
                self.assertEqual(observed["mlp_stage17_signed_contribution"], str(factor*mlp))
                self.assertEqual(observed["terminal_signed_contribution"], str(factor*terminal))
                self.assertEqual(observed["mlp_minus_terminal_coordinate_magnitude"],
                                 str(abs(factor*mlp)-abs(factor*terminal)))
            maximum = abs(ranked[0][1]*ranked[0][2])
            self.assertEqual(row["top_mlp_stage17_magnitude_ties"],
                             [r[0] for r in ranked if abs(r[1]*r[2]) == maximum])
            groups = {}
            for i, factor, mlp, _ in ranked:
                groups.setdefault(str(abs(factor*mlp)), []).append(i)
            self.assertEqual(row["mlp_stage17_magnitude_tie_groups"], [
                {"absolute_contribution": value, "coordinates": ids}
                for value, ids in groups.items() if len(ids) > 1])
            self.assertEqual(row["nonzero_mlp_stage17_coordinate_count"],
                             sum(bool(r[1]*r[2]) for r in ranked))
            self.assertTrue(row["mlp_stage17_hotspot_order_meaningful"])

    def test_coordinate_ties_include_zeros_and_use_coordinate_identity(self):
        rows = [{"coordinate": i, "mlp_stage17_absolute_contribution": str(abs(v))}
                for i, v in ((8, -2), (3, 2), (7, 0), (1, 0), (6, 1))]
        result = d.coordinate_order(rows)
        self.assertEqual(result["mlp_stage17_absolute_coordinate_order"], [3, 8, 6, 1, 7])
        self.assertEqual(result["top_mlp_stage17_magnitude_ties"], [3, 8])
        self.assertEqual(result["mlp_stage17_magnitude_tie_groups"], [
            {"absolute_contribution": "2", "coordinates": [3, 8]},
            {"absolute_contribution": "0", "coordinates": [1, 7]}])
        zeros = d.coordinate_order([r for r in rows if r["mlp_stage17_absolute_contribution"] == "0"])
        self.assertEqual(zeros["top_mlp_stage17_coordinates"], [])
        self.assertEqual(zeros["top_mlp_stage17_magnitude_ties"], [])
        self.assertFalse(zeros["mlp_stage17_hotspot_order_meaningful"])

    def test_fp16_terminal_zero_and_binary64_comparisons(self):
        zero_rows = comparison_rows = 0
        for row in self.report["pairs"][0]["rows"]:
            self.assertEqual(row["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            self.assertEqual(row["hidden_reference"], "original_input_L23_"+row["branch"])
            self.assertEqual(row["rmsnorm_reference"], "original_input_final_attempt003_"+row["branch"])
            if row["branch"] == "fp16":
                zero_rows += 1
                self.assertTrue(row["fp16_terminal_zero_boundary"])
                self.assertFalse(row["terminal_remainder_comparison_meaningful"])
                self.assertFalse(row["terminal_hotspot_order_meaningful"])
                self.assertEqual(row["terminal_remainder_comparison_rows"], [])
                self.assertEqual(row["top_terminal_coordinates"], [])
                self.assertEqual(row["top_terminal_magnitude_ties"], [])
                self.assertEqual(row["nonzero_terminal_coordinate_count"], 0)
                for coordinate in row["coordinates"]:
                    for field in ("terminal_hidden_delta", "terminal_signed_contribution",
                                  "terminal_absolute_contribution"):
                        self.assertEqual(coordinate[field], "0")
            else:
                comparison_rows += 1
                self.assertFalse(row["fp16_terminal_zero_boundary"])
                self.assertTrue(row["terminal_remainder_comparison_meaningful"])
                self.assertEqual(row["terminal_remainder_comparison_rows"],
                                 row["top_mlp_stage17_coordinates"])
        self.assertEqual((zero_rows, comparison_rows), (9, 9))

    def test_all_prior_pair_and_common_gates_unchanged(self):
        self.assertEqual([(p["left_id"], p["right_id"]) for p in self.report["pairs"]], [(34319, 13)])
        self.assertEqual((self.report["pair_count"], self.report["row_count"]), (1, 18))
        self.assertEqual(self.report["retained_dominance_audit"], self.dom)
        self.assertEqual(self.dom["retained_obstruction_census"], self.c)
        self.assertEqual(self.report["retained_bridge_selection"], self.e["report"]["selection"])
        self.assertEqual([(p["left_id"], p["right_id"], p["component"]) for p in self.dom["pairs"]],
                         [(319, 34319, "UNKNOWN"), (34319, 13, d.MLP), (34319, 319, "UNKNOWN")])
        self.assertEqual(self.report["pairs"][0]["component"], d.MLP)
        self.assertFalse(self.report["pairs"][0]["stop_nested_bridge_expansion"])
        middle = self.dom["pairs"][1]
        self.assertEqual(middle["mlp_stage17_unique_largest_failure_count"], 0)
        self.assertEqual(len(middle["rows"]), 18)
        for row in middle["rows"]:
            self.assertEqual(row["unique_largest_component"], d.MLP)
            self.assertGreater(Fraction(row["mlp_minus_terminal_net_magnitude"]), 0)
        self.assertEqual(self.report["common_component"], "UNKNOWN")
        self.assertTrue(self.report["stop_nested_bridge_expansion"])
        self.assertTrue(self.report["prior_pair_and_common_gates_unchanged"])

    def test_tampered_vectors_rejected(self):
        for field, key in (("hidden", "fp16"), ("hidden", "binary64"), ("actual", CONTROLS[0])):
            altered = {**self.e, field: deepcopy(self.e[field])}
            vector = altered[field][key]
            if field == "actual":
                vector = vector["stage17"]
            vector[0] += Fraction(1, 16)
            with self.assertRaises(ValueError):
                d.terminal.bound_vectors(altered)
        with self.assertRaises(ValueError):
            d.terminal.bound_vectors({**self.e, "weights": [self.e["weights"][0]+1, *self.e["weights"][1:]]})
        reference, rows = d.terminal.bound_vectors(self.e)
        for label in (34319, 13):
            with self.assertRaises(ValueError):
                d.coordinate_row(self.e, reference, {**rows, label: [Fraction()]*896},
                                 self.pair, "binary64")

    def test_tampered_coordinate_inputs_rejected(self):
        reference, rows = d.terminal.bound_vectors(self.e)
        original = self.pair["branches"]["binary64"]["gate_values"]["excluded_selected_coordinates"]
        for selected in (original[:-1], list(reversed(original)), [*original, 896],
                         [False, *original[1:]], [*original, original[-1]]):
            pair = deepcopy(self.pair)
            pair["branches"]["binary64"]["gate_values"]["excluded_selected_coordinates"] = selected
            with self.assertRaises(ValueError):
                d.coordinate_row(self.e, reference, rows, pair, "binary64")

    def test_tampered_pair_control_and_branch_inputs_rejected(self):
        reference, rows = d.terminal.bound_vectors(self.e)
        for field, value in (("left_id", 319), ("right_id", 319)):
            with self.assertRaises(ValueError):
                d.coordinate_row(self.e, reference, rows, {**self.pair, field: value}, "binary64")
        with self.assertRaises(ValueError):
            d.coordinate_row(self.e, reference, rows, self.pair, "fp32")
        for mutate in (
            lambda r: r["controls"].reverse(),
            lambda r: r["controls"][0]["pairs"].pop(1),
            lambda r: r["controls"][0]["pairs"][1]["branches"].pop("fp16"),
            lambda r: r["controls"][0]["pairs"][1]["branches"]["binary64"]["gate_values"].update(control="scratch"),
            lambda r: r["controls"][0]["pairs"][1]["branches"]["binary64"]["gate_values"].update(branch="fp16"),
        ):
            report = deepcopy(self.e["report"])
            mutate(report)
            with self.assertRaises(ValueError):
                d.audit_coordinates({**self.e, "report": report}, self.c, self.dom)

    def test_tampered_census_dominance_and_gate_inputs_rejected(self):
        for field, value in (("common_component", d.MLP), ("stop_nested_bridge_expansion", False)):
            with self.assertRaises(ValueError):
                d.audit_coordinates(self.e, self.c, {**self.dom, field: value})
            with self.assertRaises(ValueError):
                d.audit_coordinates(self.e, {**self.c, field: value}, self.dom)
        report = deepcopy(self.e["report"])
        report["selection"]["pairs"][1]["component"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            d.audit_coordinates({**self.e, "report": report}, self.c, self.dom)
        pair = deepcopy(self.pair)
        pair["branches"]["binary64"]["gate_values"]["component_totals"][d.MLP]["signed"] = "0"
        reference, rows = d.terminal.bound_vectors(self.e)
        with self.assertRaises(ValueError):
            d.coordinate_row(self.e, reference, rows, pair, "binary64")

    def test_history_references_thresholds_and_source_authentication(self):
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
        for branch in ("fp16", "binary64"):
            self.assertEqual(self.e["result"]["preflight"]["final_reference"]["reference"]["input_"+branch],
                             self.e["result"]["preflight"]["L23_original_reference"]["reference"][branch])
        self.assertEqual(self.e["margin_report"]["cutoff_margin_report"]["thresholds"],
                         self.e["result"]["preflight"]["thresholds"])
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.bridge.margin.rows.PINS["rmsnorm_hotspot_source"], "sha256": "0"*64})
        self.assertEqual(d.FLAGS, {**d.dominance.FLAGS, "mlp_stage17_coordinate_causal_allocation": False})
        self.assertTrue(all(not value for value in d.FLAGS.values()))

    def test_forbidden_dispatch_and_predecessor_replay(self):
        calls = (
            lambda: d.terminal.check(), lambda: d.terminal.run_tests(None, None, None, None),
            lambda: d.terminal.audit_coordinates(None, None, None),
            lambda: d.dominance.check(), lambda: d.dominance.run_tests(None, None, None),
            lambda: d.census.check(), lambda: d.census.run_tests(None, None),
            lambda: d.bridge.check(), lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.bridge.component_vectors(None, None, None),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_forbidden_writes(self):
        target = d.parent.OUTPUT / "forbidden-mlp-coordinate-audit"
        calls = (lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
                 lambda: os.open(target, os.O_WRONLY | os.O_CREAT), lambda: target.mkdir(),
                 lambda: os.replace(target, target), lambda: target.unlink())
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_rejects_forbidden_paths_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_is_exactly_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.report}), patch("sys.stdout", output):
            d.main(["--check"])
        text = output.getvalue()
        decoded, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(decoded, {"report": self.report})
        self.assertEqual(text[end:], "\n")
