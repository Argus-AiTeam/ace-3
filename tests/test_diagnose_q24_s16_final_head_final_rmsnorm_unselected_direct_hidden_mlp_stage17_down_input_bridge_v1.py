"""Independent FP16-bit oracle for the retained middle-pair down-input bridge."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_unselected_direct_hidden_mlp_stage17_down_input_bridge_v1 as d


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite FP16")
    value = (Fraction(mantissa, 16777216) if exponent == 0
             else Fraction(1024+mantissa)*Fraction(2)**(exponent-25))
    return -value if word & 32768 else value


def weights(tensors, output):
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[output % 8]
    result = []
    for i in range(4864):
        q = (int(tensors["qweight"][i, output//8]) % (1 << 32)) // 16**lane % 16
        z = (int(tensors["qzeros"][i//128, output//8]) % (1 << 32)) // 16**lane % 16
        result.append((q-z)*half(int(tensors["scales"].view("<u2")[i//128, output])))
    return result


def mass(values):
    signed, absolute = sum(values, Fraction()), sum(map(abs, values), Fraction())
    return {"signed": str(signed), "absolute": str(absolute),
            "cancellation_absolute_mass": str(absolute-abs(signed))}


class DownInputBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.bridge.read_only({"forbidden_calls": 0}):
                evidence = d.bridge.measure()
            with d.read_only({"forbidden_calls": 0}):
                census = d.hotspot.census.census(evidence["report"])
                dominance = d.hotspot.dominance.audit_census(census, evidence["report"])
                observed = d.hotspot.audit_coordinates(evidence, census, dominance)
                tensors = d.load_tensors(evidence)
                report = d.report(evidence, census, dominance, observed, tensors)
            cls.inputs = evidence, census, dominance, observed, tensors
            cls.result = report
        else:
            *inputs, cls.result = EVIDENCE
            cls.inputs = tuple(inputs)
        cls.e, cls.c, cls.dom, cls.old, cls.tensors = cls.inputs
        cls.expected = {}
        columns = {}
        for row in cls.result["rows"]:
            control, branch = row["control"], row["branch"]
            pair = next(p for c in cls.e["report"]["controls"] if c["control"] == control
                        for p in c["pairs"] if (p["left_id"], p["right_id"]) == (34319, 13))
            anchor = Fraction(pair["branches"][branch]["unchanged_unselected_bridge"]
                              ["reference_norm"]["inverse_norm_anchor"])
            for account in row["hotspots"]:
                output = account["output_coordinate"]
                if output not in columns:
                    columns[output] = weights(cls.tensors, output)
                delta = [half(int(a))-half(int(r)) for a, r in zip(
                    cls.e["archives"][control]["stage16"],
                    cls.e["reference_archive"]["stage16"], strict=True)]
                factor = (half(int(cls.e["weight_array"].view("<u2")[output]))*anchor
                          * (half(int(cls.e["rows"][34319].view("<u2")[output]))
                             - half(int(cls.e["rows"][13].view("<u2")[output]))))
                terms = [a*w for a, w in zip(delta, columns[output], strict=True)]
                retained = (half(int(cls.e["archives"][control]["stage17"][output]))
                            - half(int(cls.e["reference_archive"]["stage17"][output])))
                cls.expected[control, branch, output] = delta, columns[output], terms, factor, retained

    def accounts(self):
        for row in self.result["rows"]:
            for account in row["hotspots"]:
                yield row, account, self.expected[row["control"], row["branch"],
                                                 account["output_coordinate"]]

    def test_exact_input_rows_independent_bit_oracle(self):
        count = 0
        for row, account, (delta, column, terms, factor, retained) in self.accounts():
            self.assertEqual(len(account["input_coordinates"]), 4864)
            for i, observed in enumerate(account["input_coordinates"]):
                self.assertEqual(observed, {
                    "coordinate": i, "input_group": i//128, "input_delta": str(delta[i]),
                    "weight": str(column[i]), "signed_contribution": str(terms[i]),
                    "absolute_contribution": str(abs(terms[i])),
                    "weighted_signed_contribution": str(factor*terms[i]),
                    "weighted_absolute_contribution": str(abs(factor*terms[i]))})
                count += 1
            self.assertEqual(account["retained_hotspot"]["retained_weighted_row_factor"], str(factor))
            self.assertEqual(account["retained_coordinate_signed_contribution"], str(factor*retained))
        self.assertEqual(count, self.result["projection_account_count"]*4864)

    def test_coordinate741_representative_exact_rows(self):
        expected = {
            "fp16": "972712872287995455840195/19342813113834066795298816",
            "binary64": "1955800403415784668441195/38685626227668133590597632",
        }
        for branch in ("fp16", "binary64"):
            row = next(r for r in self.result["rows"]
                       if r["control"] == "frozen_inherited" and r["branch"] == branch)
            self.assertEqual(row["retained_maximum_coordinate_ties"], [741])
            account = row["hotspots"][0]
            self.assertEqual(account["retained_coordinate_signed_contribution"], expected[branch])
            delta, column, terms, factor, retained = self.expected["frozen_inherited", branch, 741]
            for i in (0, 127, 128, 741, 4863):
                self.assertEqual(account["input_coordinates"][i]["signed_contribution"], str(terms[i]))
                self.assertEqual(account["input_coordinates"][i]["weighted_signed_contribution"],
                                 str(factor*delta[i]*column[i]))
            self.assertEqual(account["weighted_projection_boundary_remainder"],
                             str(factor*(retained-sum(terms, Fraction()))))

    def test_signed_absolute_selected_unselected_and_group_closure(self):
        for row, account, (_, _, terms, factor, retained) in self.accounts():
            weighted = [factor*t for t in terms]
            top = sorted(range(4864), key=lambda i: (-abs(terms[i]), i))[:8]
            self.assertEqual(account["weighted_input_total"], mass(weighted))
            self.assertEqual(account["weighted_selected_total"], mass([weighted[i] for i in top]))
            self.assertEqual(account["weighted_unselected_total"],
                             mass([v for i, v in enumerate(weighted) if i not in top]))
            for g, group in enumerate(account["weighted_groups_in_input_order"]):
                self.assertEqual(group, {"input_group": g, **mass(weighted[g*128:(g+1)*128])})
            for key in ("signed", "absolute"):
                total = Fraction(account["weighted_input_total"][key])
                self.assertEqual(total, sum(Fraction(account[name][key]) for name in
                                           ("weighted_selected_total", "weighted_unselected_total")))
                self.assertEqual(total, sum(Fraction(g[key]) for g in account["weighted_groups_in_input_order"]))
            raw = account["down_projection"]
            for g, group in enumerate(raw["groups_in_input_order"]):
                chunk = terms[g*128:(g+1)*128]
                self.assertEqual(group["signed_contribution"], str(sum(chunk, Fraction())))
                self.assertEqual(group["sum_absolute_contributions"], str(sum(map(abs, chunk), Fraction())))
            boundary = factor*(retained-sum(terms, Fraction()))
            self.assertEqual(account["weighted_projection_closure_residual"], "0")
            self.assertEqual(sum(weighted, Fraction())+boundary, factor*retained)
            self.assertEqual(account["input_and_boundary_cancellation_mass"],
                             str(sum(map(abs, weighted), Fraction())+abs(boundary)-abs(factor*retained)))
            self.assertTrue(account["exact_signed_and_absolute_closure"])

    def test_order_ties_and_zero_tie_boundary(self):
        for row, account, (_, _, terms, factor, _) in self.accounts():
            ordered = sorted(range(4864), key=lambda i: (-abs(terms[i]), i))
            self.assertEqual(account["input_order"]["absolute_coordinate_order"], ordered)
            groups = {}
            for i in ordered:
                groups.setdefault(str(abs(terms[i])), []).append(i)
            self.assertEqual(account["input_order"]["magnitude_tie_groups"], [
                {"absolute_contribution": value, "coordinates": ids}
                for value, ids in groups.items() if len(ids) > 1])
            self.assertEqual(sorted(range(4864), key=lambda i: (-abs(factor*terms[i]), i)), ordered)
        synthetic = d.input_order([Fraction(-2), Fraction(0), Fraction(2), Fraction(0)])
        self.assertEqual(synthetic["absolute_coordinate_order"], [0, 2, 1, 3])
        self.assertEqual(synthetic["maximum_magnitude_ties"], [0, 2])
        zero = d.input_order([Fraction(), Fraction()])
        self.assertEqual(zero["maximum_magnitude_ties"], [])
        self.assertFalse(zero["hotspot_order_meaningful"])

    def test_all_retained_pair_common_and_reference_gates_unchanged(self):
        self.assertEqual((self.result["left_id"], self.result["right_id"]), (34319, 13))
        self.assertEqual(self.result["row_count"], 18)
        self.assertEqual([(r["control"], r["branch"]) for r in self.result["rows"]],
                         [(c, b) for c in d.hotspot.census.CONTROLS for b in ("fp16", "binary64")])
        self.assertEqual(self.result["retained_hotspot_audit"], self.old)
        self.assertEqual(self.old["retained_dominance_audit"], self.dom)
        self.assertEqual(self.dom["retained_obstruction_census"], self.c)
        self.assertEqual(self.old["retained_bridge_selection"], self.e["report"]["selection"])
        self.assertEqual([p["component"] for p in self.dom["pairs"]], ["UNKNOWN", d.hotspot.MLP, "UNKNOWN"])
        self.assertEqual(self.result["common_component"], "UNKNOWN")
        self.assertTrue(self.result["stop_nested_bridge_expansion"])
        for row, old in zip(self.result["rows"], self.old["pairs"][0]["rows"], strict=True):
            self.assertEqual(row["retained_gate_values"], old["retained_gate_values"])
            self.assertEqual(row["retained_maximum_coordinate_ties"], old["top_mlp_stage17_magnitude_ties"])
            self.assertEqual(row["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
            self.assertEqual(row["hidden_reference"], "original_input_L23_"+row["branch"])
            self.assertEqual(row["rmsnorm_reference"], "original_input_final_attempt003_"+row["branch"])
        self.assertTrue(self.result["prior_pair_and_common_gates_unchanged"])

    def test_tampered_pair_control_branch_hotspot_and_gate_inputs(self):
        for mutate in (
            lambda r: r["pairs"][0].update(left_id=319),
            lambda r: r["pairs"][0]["rows"][0].update(control="scratch"),
            lambda r: r["pairs"][0]["rows"][0].update(branch="binary64"),
            lambda r: r["pairs"][0]["rows"][0].update(top_mlp_stage17_magnitude_ties=[62]),
            lambda r: r["pairs"][0]["rows"][0]["coordinates"][0].update(mlp_stage17_signed_contribution="1"),
            lambda r: r.update(common_component=d.hotspot.MLP),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            altered = deepcopy(self.old)
            mutate(altered)
            with self.assertRaises(ValueError):
                d.report(self.e, self.c, self.dom, altered, self.tensors)

    def test_tampered_vectors_and_native_tensor_operands(self):
        control = d.hotspot.census.CONTROLS[0]
        for stage in ("stage16", "stage17"):
            archive = {**self.e["archives"][control], stage: self.e["archives"][control][stage].copy()}
            archive[stage][0] ^= 1
            changed = {**self.e, "archives": {**self.e["archives"], control: archive}}
            with self.assertRaises(ValueError):
                d.bind_inputs(changed, self.tensors)
        archive = {**self.e["reference_archive"], "stage16": self.e["reference_archive"]["stage16"].copy()}
        archive["stage16"][0] ^= 1
        with self.assertRaises(ValueError):
            d.bind_inputs({**self.e, "reference_archive": archive}, self.tensors)
        for suffix in self.tensors:
            changed = {**self.tensors, suffix: self.tensors[suffix].copy()}
            changed[suffix].flat[0] += 1
            with self.assertRaises(ValueError):
                d.bind_inputs(self.e, changed)
        altered = deepcopy(self.e["actual"])
        altered[control]["stage17"][741] += 1
        with self.assertRaises(ValueError):
            d.bind_inputs({**self.e, "actual": altered}, self.tensors)

    def test_tampered_group_vector_selection_and_closure_reports(self):
        for mutate in (
            lambda r: r["rows"][0]["hotspots"][0]["weighted_groups_in_input_order"][0].update(signed="1"),
            lambda r: r["rows"][0]["hotspots"][0]["down_projection"]["groups_in_input_order"].reverse(),
            lambda r: r["rows"][0]["hotspots"][0]["input_coordinates"][741].update(absolute_contribution="1"),
            lambda r: r["rows"][0]["hotspots"][0]["input_order"]["selected_coordinates"].reverse(),
            lambda r: r["rows"][0]["hotspots"][0].update(weighted_projection_boundary_remainder="1"),
            lambda r: r.update(stop_nested_bridge_expansion=False),
        ):
            altered = deepcopy(self.result)
            mutate(altered)
            with self.assertRaises(ValueError):
                d.validate_report(*self.inputs, altered)

    def test_history_thresholds_and_source_bindings(self):
        for field, value in (("candidate_admitted", True), ("S18_failure_indices", []),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
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
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.parent.record(d.hotspot.SOURCE), "sha256": "0"*64})
        self.assertTrue(all(not value for value in d.FLAGS.values()))

    def test_forbidden_replay_dispatch_and_writes(self):
        target = d.parent.OUTPUT / "forbidden-stage17-down-input"
        calls = (
            lambda: d.hotspot.check(), lambda: d.hotspot.run_tests(None, None, None, None),
            lambda: d.down.check(), lambda: d.down.measure(),
            lambda: d.down.report(None, None, None, None, None),
            lambda: d.bridge.measure(), lambda: d.bridge.report(None),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.parent.execute("forbidden"), lambda: d.parent.head.decode_array_q24(None),
            lambda: d.bridge.residual.local.local_reference(None),
            lambda: d.bridge.margin.contributions.dyadic_dot(None, None),
            lambda: subprocess.run(["forbidden"]), lambda: os.system("forbidden"),
            lambda: d.parent.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
            lambda: target.write_text("forbidden"), lambda: open(target, "wb"),
            lambda: os.open(target, os.O_WRONLY | os.O_CREAT),
            lambda: target.mkdir(), lambda: os.replace(target, target), lambda: target.unlink(),
        )
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_cli_refuses_extra_paths_and_abbreviations(self):
        for argv in ([], ["--execute"], ["--check", "--out", "/tmp/forbidden"],
                     ["--check", "--reference"], ["--check", "--che"]):
            with patch.object(d, "check", side_effect=AssertionError("dispatch")), \
                    patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_stdout_exactly_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.result}), patch("sys.stdout", output):
            d.main(["--check"])
        value, end = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(value, {"report": self.result})
        self.assertEqual(output.getvalue()[end:], "\n")
