"""Independent retained-input rational coordinate oracle and no-replay refusals."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_terminal_remainder_coordinate_hotspot_audit_v1 as d


EVIDENCE = CENSUS = DOMINANCE = OBSERVED = None
PAIRS = ((319, 34319), (34319, 319))
CONTROLS = ("frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
            "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
            "mapped_all", "inherited_native")


def oracle(evidence, identity, control, branch, selected, anchor):
    ref = evidence["reference_archive"]
    actual = evidence["archives"][control]
    fp16 = ref["stage18"].view("<f2")
    terminal = fp16 if branch == "fp16" else evidence["binary64"]
    weights = evidence["weight_array"]
    left, right = (evidence["rows"][i] for i in identity)
    mlp, ref_mlp = actual["stage17"].view("<f2"), ref["stage17"].view("<f2")
    q = lambda v: Fraction(float(v))
    result = []
    for i in range(896):
        if i not in selected:
            factor = q(weights[i])*anchor*(q(left[i])-q(right[i]))
            result.append((i, factor*(q(fp16[i])-q(terminal[i])),
                           factor*(q(mlp[i])-q(ref_mlp[i]))))
    return result


class TerminalCoordinateTests(unittest.TestCase):
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
        for control in cls.e["report"]["controls"]:
            for pair in control["pairs"]:
                for branch, source in pair["branches"].items():
                    cls.inputs[(pair["left_id"], pair["right_id"]), control["control"], branch] = source

    def test_exact_coordinate_oracle_and_signed_absolute_closure(self):
        count = 0
        for pair in self.report["pairs"]:
            identity = pair["left_id"], pair["right_id"]
            self.assertEqual([(r["control"], r["branch"]) for r in pair["rows"]],
                             [(c, b) for c in CONTROLS for b in ("fp16", "binary64")])
            for row in pair["rows"]:
                source = self.inputs[identity, row["control"], row["branch"]]
                selected = [r["coordinate"] for r in
                            source["unchanged_unselected_bridge"]["selected_coordinates"]]
                anchor = Fraction(source["unchanged_unselected_bridge"]["reference_norm"]["inverse_norm_anchor"])
                expected = oracle(self.e, identity, row["control"], row["branch"], selected, anchor)
                self.assertEqual(row["excluded_selected_coordinates"], selected)
                self.assertNotIn(62, [v["coordinate"] for v in row["coordinates"]])
                self.assertEqual(row["coordinate_count"], 896-len(selected))
                self.assertEqual([(v["coordinate"], Fraction(v["terminal_signed_contribution"]),
                                   Fraction(v["mlp_stage17_signed_contribution"])) for v in row["coordinates"]],
                                 expected)
                for index, field in ((1, d.TERMINAL), (2, d.MLP)):
                    values = [v[index] for v in expected]
                    signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
                    total = {"signed": str(signed), "absolute": str(absolute),
                             "cancellation_absolute_mass": str(absolute-abs(signed))}
                    self.assertEqual(row["component_totals"][field], total)
                    self.assertEqual(source["gate_values"]["component_totals"][field], total)
                self.assertEqual(row["retained_gate_values"], source["gate_values"])
                self.assertTrue(row["exact_signed_and_absolute_closure"])
                count += 1
        self.assertEqual(count, 36)

    def test_representative_exact_hotspots_both_divergent_pairs(self):
        for pair in self.report["pairs"]:
            identity = pair["left_id"], pair["right_id"]
            for row in pair["rows"]:
                source = self.inputs[identity, row["control"], row["branch"]]
                anchor = Fraction(source["unchanged_unselected_bridge"]["reference_norm"]["inverse_norm_anchor"])
                expected = oracle(self.e, identity, row["control"], row["branch"],
                                  row["excluded_selected_coordinates"], anchor)
                ranked = sorted(expected, key=lambda v: (-abs(v[1]), v[0]))
                self.assertEqual(row["terminal_absolute_coordinate_order"], [v[0] for v in ranked])
                top = [v for v in ranked if v[1]][:8]
                self.assertEqual([r["coordinate"] for r in row["top_terminal_coordinates"]],
                                 [v[0] for v in top])
                for actual, (i, terminal, mlp) in zip(row["top_terminal_coordinates"], top, strict=True):
                    self.assertEqual(actual["coordinate"], i)
                    self.assertEqual(actual["terminal_signed_contribution"], str(terminal))
                    self.assertEqual(actual["terminal_absolute_contribution"], str(abs(terminal)))
                    self.assertEqual(actual["mlp_stage17_signed_contribution"], str(mlp))
                    self.assertEqual(actual["mlp_stage17_absolute_contribution"], str(abs(mlp)))
                    self.assertEqual(actual["terminal_minus_mlp_coordinate_magnitude"],
                                     str(abs(terminal)-abs(mlp)))
                if row["branch"] == "binary64":
                    self.assertEqual(len(top), 8)
                    self.assertTrue(row["terminal_hotspot_order_meaningful"])
                self.assertEqual(row["top_terminal_magnitude_ties"],
                                 [i for i, t, _ in ranked if t and abs(t) == abs(ranked[0][1])])

    def test_fp16_coordinate_zero_boundaries(self):
        count = 0
        for pair in self.report["pairs"]:
            for row in pair["rows"]:
                if row["branch"] != "fp16":
                    continue
                self.assertTrue(row["fp16_terminal_zero_boundary"])
                self.assertFalse(row["terminal_hotspot_order_meaningful"])
                self.assertEqual(row["top_terminal_coordinates"], [])
                self.assertEqual(row["top_terminal_magnitude_ties"], [])
                self.assertEqual(row["nonzero_terminal_coordinate_count"], 0)
                for coordinate in row["coordinates"]:
                    self.assertEqual(coordinate["terminal_hidden_delta"], "0")
                    self.assertEqual(coordinate["terminal_signed_contribution"], "0")
                    self.assertEqual(coordinate["terminal_absolute_contribution"], "0")
                    self.assertEqual(Fraction(coordinate["terminal_minus_mlp_coordinate_magnitude"]),
                                     -abs(Fraction(coordinate["mlp_stage17_signed_contribution"])))
                count += 1
        self.assertEqual(count, 18)

    def test_prior_pair_and_common_gates_unchanged(self):
        self.assertEqual([(p["left_id"], p["right_id"]) for p in self.report["pairs"]], list(PAIRS))
        self.assertEqual(self.report["row_count"], 36)
        self.assertEqual(self.report["retained_dominance_audit"], self.dom)
        self.assertEqual(self.report["retained_bridge_selection"], self.e["report"]["selection"])
        self.assertEqual([p["component"] for p in self.dom["pairs"]], ["UNKNOWN", d.MLP, "UNKNOWN"])
        self.assertEqual(self.report["common_component"], "UNKNOWN")
        self.assertTrue(self.report["stop_nested_bridge_expansion"])
        for pair in self.report["pairs"]:
            self.assertEqual(pair["component"], "UNKNOWN")
            self.assertTrue(pair["stop_nested_bridge_expansion"])
        self.assertTrue(self.report["prior_pair_and_common_gates_unchanged"])

    def test_tampered_vectors_rejected(self):
        for field, key in (("hidden", "binary64"), ("hidden", "fp16"),
                           ("actual", "frozen_inherited")):
            altered = dict(self.e)
            altered[field] = deepcopy(self.e[field])
            vector = altered[field][key]
            if field == "actual":
                vector = vector["stage17"]
            vector[0] += Fraction(1, 16)
            with self.assertRaises(ValueError):
                d.bound_vectors(altered)
        altered = {**self.e, "weights": [self.e["weights"][0]+1, *self.e["weights"][1:]]}
        with self.assertRaises(ValueError):
            d.bound_vectors(altered)
        for label in (319, 34319):
            altered = {**self.e, "rows": {**self.e["rows"], label: self.e["rows"][label]*0}}
            with self.assertRaises(ValueError):
                d.audit_coordinates(altered, self.c, self.dom)

    def test_tampered_coordinate_inputs_rejected(self):
        source = self.e["report"]["controls"][0]["pairs"][0]
        reference, rows = d.bound_vectors(self.e)
        original = source["branches"]["binary64"]["gate_values"]["excluded_selected_coordinates"]
        for selected in (original[:-1], list(reversed(original)), [*original, 896],
                         [False, *original[1:]], [*original, original[-1]]):
            pair = deepcopy(source)
            pair["branches"]["binary64"]["gate_values"]["excluded_selected_coordinates"] = selected
            with self.assertRaises(ValueError):
                d.coordinate_row(self.e, reference, rows, pair, "binary64")

    def test_tampered_census_dominance_and_gate_inputs_rejected(self):
        for field, value in (("common_component", d.TERMINAL), ("stop_nested_bridge_expansion", False)):
            with self.assertRaises(ValueError):
                d.audit_coordinates(self.e, self.c, {**self.dom, field: value})
            with self.assertRaises(ValueError):
                d.audit_coordinates(self.e, {**self.c, field: value}, self.dom)
        report = deepcopy(self.e["report"])
        report["selection"]["pairs"][0]["component"] = d.TERMINAL
        with self.assertRaises(ValueError):
            d.audit_coordinates({**self.e, "report": report}, self.c, self.dom)
        report = deepcopy(self.e["report"])
        report["controls"][0]["pairs"][0]["branches"]["binary64"]["gate_values"][
            "component_totals"][d.TERMINAL]["signed"] = "0"
        with self.assertRaises(ValueError):
            d.audit_coordinates({**self.e, "report": report}, self.c, self.dom)

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
        self.assertEqual(d.FLAGS, {**d.dominance.FLAGS, "terminal_coordinate_causal_allocation": False})
        self.assertTrue(all(not value for value in d.FLAGS.values()))

    def test_forbidden_dispatch_and_predecessor_replay(self):
        calls = (
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
        target = d.parent.OUTPUT / "forbidden-terminal-coordinate-audit"
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
