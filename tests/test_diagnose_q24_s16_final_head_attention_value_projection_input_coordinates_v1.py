"""Independent FP16 bit and AWQ integer oracles for retained V input accounts."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_value_projection_input_coordinates_v1 as d
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    sign = -1 if word & 32768 else 1
    return sign * (Fraction(mantissa, 2 ** 24) if exponent == 0
                   else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))


def column(tensors, coordinate):
    lane = (0, 2, 4, 6, 1, 3, 5, 7).index(coordinate % 8)
    packed = coordinate // 8
    return [
        (((int(tensors["qweight"][i, packed]) % 2 ** 32) // 16 ** lane % 16)
         - ((int(tensors["qzeros"][i // 128, packed]) % 2 ** 32) // 16 ** lane % 16))
        * half(int(tensors["scales"].view("<u2")[i // 128, coordinate]))
        for i in range(896)]


def integer_products(actual, reference, weights):
    terms = []
    for a, r, w in zip(actual, reference, weights, strict=True):
        an, rn, wn = a * 2 ** 24, r * 2 ** 24, w * 2 ** 24
        if any(v.denominator != 1 for v in (an, rn, wn)):
            raise ValueError("not FP16-grid dyadics")
        terms.append(Fraction((an.numerator - rn.numerator) * wn.numerator, 2 ** 48))
    return terms


class ValueProjectionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.actual, cls.reference, cls.tensors, cls.report = cls.evidence
        cls.result = cls.previous[0][0]
        cls.columns, cls.values = {}, {}
        for label, _, _, entry in cls.accounts():
            coordinate = entry["value_projection"]["output_coordinate"]
            if coordinate not in cls.columns:
                cls.columns[coordinate] = column(cls.tensors, coordinate)
            if (label, coordinate) not in cls.values:
                cls.values[(label, coordinate)] = integer_products(
                    cls.actual[label], cls.reference, cls.columns[coordinate])

    @classmethod
    def accounts(cls):
        for row in cls.report["controls"]:
            for name, branch in row["branches"].items():
                for hotspot in branch["hotspots"]:
                    for entry in hotspot["value_components"]:
                        yield row["control"], name, hotspot, entry

    def test_retained_inputs_and_v_independent_half_bit_oracle(self):
        pins = [(row["parent"]["terminal_archive"], self.actual[row["control"]],
                 self.previous[1][row["control"]]) for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference, self.previous[2]))
        for pin, values, attention in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                self.assertEqual(values, [half(int(w)) for w in archive["stage00"]])
                for stage in ("stage03", "stage07"):
                    self.assertEqual(attention[1], [half(int(w)) for w in archive[stage]])

    def test_awq_columns_independent_lane_oracle(self):
        for coordinate, expected in self.columns.items():
            self.assertEqual(d.weight_column(self.tensors, coordinate), expected)

    def test_native_lanes_all_groups_signed_words_no_plus_one(self):
        tensors = {"qweight": np.full((896, 16), 0x76543210, dtype="<i4"),
                   "qzeros": np.full((7, 16), 0x01234567, dtype="<i4"),
                   "scales": np.full((7, 128), 2 ** -24, dtype="<f2")}
        tensors["scales"][1::2] = -0.5
        for coordinate in range(8):
            self.assertEqual(d.weight_column(tensors, coordinate), column(tensors, coordinate))
        self.assertEqual(d.weight_column(tensors, 0)[0], -7 * Fraction(1, 2 ** 24))
        tensors["qweight"][:] = -1
        self.assertEqual(d.weight_column(tensors, 127), column(tensors, 127))

    def test_all_coordinate_sums_rankings_groups_integer_oracle(self):
        for label, _, _, entry in self.accounts():
            account = entry["value_projection"]
            coordinate = account["output_coordinate"]
            values = self.values[(label, coordinate)]
            order = sorted(range(896), key=lambda i: (-abs(values[i]), i))
            self.assertEqual([e["coordinate"] for e in account["largest_absolute_coordinates"]], order[:8])
            for term in account["largest_absolute_coordinates"]:
                i = term["coordinate"]
                self.assertEqual(Fraction(term["signed_contribution"]), values[i])
                self.assertEqual(Fraction(term["weight"]), self.columns[coordinate][i])
                self.assertEqual(Fraction(term["input_delta"]), self.actual[label][i] - self.reference[i])
                self.assertEqual(term["input_group"], i // 128)
            total = sum(values)
            self.assertEqual(Fraction(account["exact_input_delta"]), total)
            self.assertEqual(Fraction(account["sum_absolute_contributions"]), sum(map(abs, values)))
            selected = sum(values[i] for i in order[:8])
            self.assertEqual(Fraction(account["ranked_signed_sum"]), selected)
            self.assertEqual(Fraction(account["unranked_signed_remainder"]), total - selected)
            totals = [sum(values[i:i + 128]) for i in range(0, 896, 128)]
            groups = account["groups_by_absolute_signed_sum"]
            self.assertEqual([g["input_group"] for g in groups],
                             sorted(range(7), key=lambda i: (-abs(totals[i]), i)))
            for group in groups:
                i = group["input_group"]
                self.assertEqual((group["start_coordinate"], group["end_coordinate_exclusive"],
                                  group["coordinate_count"]), (i * 128, (i + 1) * 128, 128))
                self.assertEqual(Fraction(group["signed_contribution"]), totals[i])
                self.assertEqual(Fraction(group["sum_absolute_contributions"]),
                                 sum(map(abs, values[i * 128:(i + 1) * 128])))

    def test_projection_remainders_and_weighted_value_identity(self):
        columns = {}
        for label, _, hotspot, entry in self.accounts():
            account, component = entry["value_projection"], entry["retained_value_component"]
            coordinate = account["output_coordinate"]
            output = hotspot["final_head_channel"]["coordinate"]
            if output not in columns:
                columns[output] = column(self.previous[3], output)
            kv, dim = divmod(coordinate, 64)
            factor = sum(columns[output][head * 64 + dim] for head in range(kv * 7, (kv + 1) * 7))
            actual, reference = self.previous[1][label][1][coordinate], self.previous[2][1][coordinate]
            self.assertEqual(Fraction(account["actual_v"]), actual)
            self.assertEqual(Fraction(account["reference_v"]), reference)
            retained = actual - reference
            exact = sum(self.values[(label, coordinate)])
            self.assertEqual(Fraction(account["retained_projection_delta"]), retained)
            self.assertEqual(Fraction(account["projection_boundary_remainder"]), retained - exact)
            self.assertEqual(Fraction(account["value_component_factor"]), factor)
            self.assertEqual(Fraction(account["weighted_input_contribution"]), exact * factor)
            self.assertEqual(Fraction(account["weighted_boundary_contribution"]), (retained - exact) * factor)
            self.assertEqual(Fraction(account["retained_value_contribution"]), retained * factor)
            self.assertEqual(Fraction(component["value"]), retained * factor)
            self.assertEqual(component["value"], component["signed_contribution"])
            self.assertEqual((component["probability"], component["interaction"]), ("0", "0"))

    def test_selection_counts_order_and_inherited_report(self):
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        counts = [0, 0, 0, 0, 0, 0]
        for row, old in zip(self.report["controls"], self.previous[4]["controls"], strict=True):
            self.assertEqual(row["control"], old["control"])
            self.assertEqual(list(row["branches"]), list(d.channels.BRANCHES))
            for name, branch in row["branches"].items():
                counts[0] += 1
                prior = old["branches"][name]
                self.assertEqual(branch["numeric_id"], prior["numeric_id"])
                self.assertEqual(branch["retained_signed_residual"], prior["retained_signed_residual"])
                self.assertEqual([h["hotspot_rank"] for h in branch["hotspots"]], list(range(1, 9)))
                for h, oh in zip(branch["hotspots"], prior["hotspots"], strict=True):
                    counts[1] += 1
                    self.assertEqual(h["final_head_channel"], oh["final_head_channel"])
                    self.assertEqual([e["retained_value_component"] for e in h["value_components"]],
                                     oh["attention"]["largest_absolute_value_components"])
                    self.assertEqual([e["value_hotspot_rank"] for e in h["value_components"]],
                                     list(range(1, 9)))
                    for entry in h["value_components"]:
                        a, c = entry["value_projection"], entry["retained_value_component"]
                        self.assertEqual(a["output_coordinate"], c["value_coordinate"])
                        kv, dim = divmod(c["value_coordinate"], 64)
                        self.assertEqual((c["kv_head"], c["head_dimension"]), (kv, dim))
                        self.assertEqual(c["query_heads"], list(range(kv * 7, (kv + 1) * 7)))
                        for i, n in enumerate((1, a["coordinate_count"],
                                               len(a["largest_absolute_coordinates"]),
                                               len(a["groups_by_absolute_signed_sum"])), 2):
                            counts[i] += n
        self.assertEqual(counts, [18, 144, 1152, 1032192, 9216, 8064])
        self.assertEqual(self.report["source_value_attribution_report"], self.previous[4])

    def test_json_schema_and_deterministic_rebuild(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        rebuilt = d.report(self.previous, self.actual, self.reference, self.tensors)
        self.assertEqual(text, json.dumps(rebuilt, sort_keys=True, allow_nan=False))

    def test_schema_rejects_counts_extra_fields_and_scope(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("projection_account_count", 1151), ("selected_value_component_count", 0),
                             ("input_coordinate_term_count", 0), ("input_group_count", 0),
                             ("selected_input_coordinate_count", 0), ("attention_reference", "binary64"),
                             ("hotspot_count", 143), ("selected_row_count", 17), ("extra", 1),
                             ("control_count", 8), ("controls", [])):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        account = next(self.accounts())[3]["value_projection"]
        for field, value in (("largest_absolute_coordinates", []), ("groups_by_absolute_signed_sum", []),
                             ("input_stage", "input_i"), ("output_coordinate", 128), ("position", 1),
                             ("exact_local_identity", False), ("projection", "q_proj"),
                             ("exact_value_component_identity", False)):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate({**account, field: value})

    def test_synthetic_zero_ties_cancellation_and_boundary(self):
        component = {"source_position": 0, "source_token_id": 9707, "value_coordinate": 0,
                     "kv_head": 0, "head_dimension": 0, "query_heads": list(range(7)),
                     "probability": "0", "interaction": "0", "value": "-4", "signed_contribution": "-4"}
        x, r, w = [Fraction()] * 896, [Fraction()] * 896, [Fraction(1)] * 896
        x[0], x[1], x[128], x[129] = Fraction(4), Fraction(-4), Fraction(4), Fraction(-4)
        account = d.account(x, r, w, 0, Fraction(3), Fraction(1), Fraction(-2), component)
        self.assertEqual([e["coordinate"] for e in account["largest_absolute_coordinates"]],
                         [0, 1, 128, 129, 2, 3, 4, 5])
        self.assertEqual(account["exact_input_delta"], "0")
        self.assertEqual(account["projection_boundary_remainder"], "2")
        self.assertEqual(account["weighted_boundary_contribution"], "-4")
        self.assertEqual(account["sum_absolute_contributions"], "16")
        self.assertEqual([g["input_group"] for g in account["groups_by_absolute_signed_sum"]], list(range(7)))
        for factor, retained in ((Fraction(), Fraction(2)), (Fraction(1), Fraction())):
            zero = d.account(r, r, w, 0, retained, Fraction(), factor,
                             {**component, "value": "0", "signed_contribution": "0"})
            self.assertEqual([e["coordinate"] for e in zero["largest_absolute_coordinates"]], list(range(8)))
            self.assertEqual(zero["retained_value_contribution"], "0")

    def test_invalid_exact_inputs_and_coordinates_rejected(self):
        component = next(self.accounts())[3]["retained_value_component"]
        for actual in ([], [0] * 896):
            with self.assertRaises(ValueError):
                d.account(actual, self.reference, [Fraction()] * 896, component["value_coordinate"],
                          Fraction(), Fraction(), Fraction(), component)
        for coordinate in (-1, 128, True):
            with self.assertRaises(ValueError):
                d.weight_column(self.tensors, coordinate)

    def test_four_tensor_bindings_reject_tamper_and_nonfinite(self):
        binding = self.result["preflight"]["L23_original_reference"]["reference"]["canonical"]
        for suffix, tensor in self.tensors.items():
            pin = binding[d.PROJECTION + suffix]
            self.assertIs(d.bind_tensor(tensor, pin, suffix), tensor)
            for value, bad_pin in ((tensor, {**pin, "sha256": "0" * 64}),
                                   (tensor, {**pin, "name": "wrong"}),
                                   (tensor.reshape(-1)[:1], pin), (tensor.astype("<f8"), pin)):
                with self.assertRaises(ValueError):
                    d.bind_tensor(value, bad_pin, suffix)
        for suffix in ("scales", "bias"):
            bad = self.tensors[suffix].copy()
            bad.flat[0] = np.inf
            pin = binding[d.PROJECTION + suffix]
            with self.assertRaises(ValueError):
                d.bind_tensor(bad, {**pin, "sha256": hashlib.sha256(bad.tobytes()).hexdigest()}, suffix)

    def test_authenticated_archive_splice_rejected(self):
        row = self.result["controls"][0]
        pin, expected = row["parent"]["terminal_archive"], self.previous[1][row["control"]]
        for p, attention in (({**pin, "sha256": "0" * 64}, expected),
                             (pin, ([Fraction()] * 14, *expected[1:]))):
            with self.assertRaises(ValueError):
                d.retained_input(p, actual=True, expected_attention=attention)
        for pin in (*d.channels.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_stage00_v_cache_lineage_and_probability_splices_rejected(self):
        row = self.result["controls"][0]
        pins = [(row["parent"]["terminal_archive"], True, self.previous[1][row["control"]]),
                (self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                 False, self.previous[2])]
        for pin, actual, expected in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            changes = [("stage00", np.zeros(895, dtype="<u2")),
                       ("stage00", np.zeros(896, dtype="<f2")),
                       ("stage00", np.full(896, 0x7e00, dtype="<u2"))]
            for name in ("stage03", "stage07", "stage09", "stage10"):
                value = original[name].copy()
                value.flat[0] ^= 1
                changes.append((name, value))
            if actual:
                for kind in ("k", "v"):
                    changes.append(("input_cache_" + kind, np.zeros((1, 128), dtype="<u2")))
                    value = original["output_cache_" + kind].copy()
                    value.flat[0] ^= 1
                    changes.append(("output_cache_" + kind, value))
            for name, value in changes:
                payload = io.BytesIO()
                np.savez(payload, **{**original, name: value})
                with patch.object(d.base, "read_bound", return_value=payload.getvalue()):
                    with self.assertRaises(ValueError):
                        d.retained_input(pin, actual=actual, expected_attention=expected)

    def test_control_selection_mapping_and_component_tamper_rejected(self):
        for actual in ({}, dict(reversed(list(self.actual.items())))):
            with self.assertRaises(ValueError):
                d.report(self.previous, actual, self.reference, self.tensors)
        for mutation in ("controls", "branches", "hotspot", "ranking", "selection", "mapping", "value"):
            inherited = deepcopy(self.previous[4])
            row = inherited["controls"][0]
            branch = next(iter(row["branches"].values()))
            hotspot = branch["hotspots"][0]
            ordered = hotspot["attention"]["full_value_component_ranking"]
            if mutation == "controls":
                inherited["controls"].reverse()
            elif mutation == "branches":
                row["branches"] = dict(reversed(list(row["branches"].items())))
            elif mutation == "hotspot":
                branch["hotspots"].reverse()
            elif mutation == "ranking":
                ordered.reverse()
            elif mutation == "selection":
                hotspot["attention"]["largest_absolute_value_components"].reverse()
            else:
                field = "kv_head" if mutation == "mapping" else "value"
                value = 1 - ordered[0]["kv_head"] if mutation == "mapping" else "999"
                ordered[0][field] = value
                hotspot["attention"]["largest_absolute_value_components"][0][field] = value
            with self.assertRaises(ValueError):
                d.report((*self.previous[:4], inherited), self.actual, self.reference, self.tensors)

    def test_original_references_histories_thresholds_nonadmission(self):
        retained = self.report["source_value_attribution_report"]["final_head_channel_report"]
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        ref = self.report["source_value_attribution_report"]["attention_reference"]
        self.assertEqual(ref["binary64_stage_status"], "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertIs(ref["final_head_branches_reanchored"], False)
        self.assertEqual(ref["fp16"], retained["original_global_reference"]["reference"]["input_fp16"])
        for row in retained["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
        for field, value in (("candidate_admitted", True), ("policy_adopted", True),
                             ("successor_published", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)
        for text in ("wider than FP16", "NOT propagation", "no qzero",
                     "No root-cause", "post-RMSNorm", "not solely rounding"):
            self.assertIn(text, d.BOUNDARY)

    def test_forbidden_dispatch_prior_checks_and_evidence_writes(self):
        calls = [
            lambda: native.projection(None, None, None), lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.attribution.check(), lambda: d.attribution.focused_tests(None),
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
            lambda: d.channels.hotspots.check(), lambda: d.base.check(),
            lambda: d.channels.contributions.dyadic_dot(None, None),
            lambda: d.parent.write(None, None), lambda: subprocess.Popen(["false"]),
            lambda: os.system("false"), lambda: open(d.SOURCE, "w"),
            lambda: d.SOURCE.write_text("forbidden"), lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.unlink(d.SOURCE), lambda: os.rename(d.SOURCE, d.TEST),
            lambda: d.channels.contributions.preflight.parent.producer.legacy.torch.cuda._lazy_init(),
        ]
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        for flag in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                     "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "attention_replay",
                     "o_projection_replay", "v_projection_replay"):
            self.assertEqual(d.FLAGS[flag], 0)

    def test_cli_stdout_only_rejects_execution_and_output_modes(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"result": 1}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"result": 1}\n')
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_workdir_account_interpreter_and_bytecode_gates(self):
        for change in (patch.object(d.Path, "cwd", return_value=d.ROOT.parent),
                       patch.object(d.os, "getuid", return_value=0),
                       patch.object(d.sys, "executable", "/wrong/python"),
                       patch.object(d.sys, "dont_write_bytecode", False),
                       patch.dict(d.os.environ, {"PYTHONPATH": "/wrong"})):
            with change, patch.object(d, "measure", side_effect=AssertionError("must not measure")):
                with self.assertRaises(ValueError):
                    d.check()


if __name__ == "__main__":
    unittest.main()
