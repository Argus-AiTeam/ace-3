"""Independent bit/integer oracle and read-only projection attribution checks."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_qk_projection_input_coordinates_v1 as d
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
    lane = (0, 4, 1, 5, 2, 6, 3, 7)[coordinate % 8]
    packed = coordinate // 8
    values = []
    for i in range(896):
        q = (int(tensors["qweight"][i, packed]) % 2 ** 32) // 16 ** lane % 16
        z = (int(tensors["qzeros"][i // 128, packed]) % 2 ** 32) // 16 ** lane % 16
        s = half(int(tensors["scales"].view("<u2")[i // 128, coordinate]))
        values.append((q - z) * s)
    return values


def integer_products(actual, reference, weights):
    scale = 2 ** 48
    terms = []
    for a, r, w in zip(actual, reference, weights, strict=True):
        an, rn, wn = a * 2 ** 24, r * 2 ** 24, w * 2 ** 24
        if any(v.denominator != 1 for v in (an, rn, wn)):
            raise ValueError("not FP16-grid dyadics")
        terms.append(Fraction((an.numerator - rn.numerator) * wn.numerator, scale))
    return terms


class ProjectionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.actual, cls.reference, cls.tensors, cls.report = cls.evidence
        cls.result = cls.previous[0][0][0][0]
        cls.columns = {}
        cls.values = {}
        for label, _, account in cls.accounts():
            kind, coordinate = account["projection"], account["output_coordinate"]
            key = (kind, coordinate)
            if key not in cls.columns:
                cls.columns[key] = column(cls.tensors[kind], coordinate)
            if (label, *key) not in cls.values:
                cls.values[(label, *key)] = integer_products(
                    cls.actual[label], cls.reference, cls.columns[key])

    @classmethod
    def accounts(cls):
        for row in cls.report["controls"]:
            for hotspot in row["score_hotspots"]:
                for entry in hotspot["dimensions"]:
                    for kind in ("query", "key"):
                        yield row["control"], entry, entry[kind]

    def test_retained_inputs_independent_half_bit_oracle(self):
        pins = [(row["parent"]["terminal_archive"], self.actual[row["control"]])
                for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference))
        for pin, values in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                self.assertEqual(values, [half(int(w)) for w in archive["stage00"]])

    def test_awq_columns_independent_native_lane_oracle(self):
        for (kind, coordinate), expected in self.columns.items():
            self.assertEqual(d.weight_column(self.tensors, kind, coordinate), expected)

    def test_all_coordinate_sums_rankings_and_groups_integer_oracle(self):
        for label, _, account in self.accounts():
            kind, coordinate = account["projection"], account["output_coordinate"]
            values = self.values[(label, kind, coordinate)]
            order = sorted(range(896), key=lambda i: (-abs(values[i]), i))
            entries = account["largest_absolute_coordinates"]
            self.assertEqual([e["coordinate"] for e in entries], order[:8])
            for entry in entries:
                i = entry["coordinate"]
                self.assertEqual(Fraction(entry["signed_contribution"]), values[i])
                self.assertEqual(Fraction(entry["weight"]), self.columns[(kind, coordinate)][i])
                self.assertEqual(Fraction(entry["input_delta"]), self.actual[label][i] - self.reference[i])
                self.assertEqual(entry["input_group"], i // 128)
            self.assertEqual(Fraction(account["exact_input_delta"]), sum(values))
            self.assertEqual(Fraction(account["sum_absolute_contributions"]), sum(map(abs, values)))
            selected = sum(values[i] for i in order[:8])
            self.assertEqual(Fraction(account["ranked_signed_sum"]), selected)
            self.assertEqual(Fraction(account["unranked_signed_remainder"]), sum(values) - selected)
            groups = account["groups_by_absolute_signed_sum"]
            totals = [sum(values[start:start + 128]) for start in range(0, 896, 128)]
            self.assertEqual([g["input_group"] for g in groups],
                             sorted(range(7), key=lambda i: (-abs(totals[i]), i)))
            for group in groups:
                i = group["input_group"]
                self.assertEqual((group["start_coordinate"], group["end_coordinate_exclusive"]),
                                 (i * 128, (i + 1) * 128))
                self.assertEqual(Fraction(group["signed_contribution"]), totals[i])
                self.assertEqual(Fraction(group["sum_absolute_contributions"]),
                                 sum(map(abs, values[i * 128:(i + 1) * 128])))

    def test_boundary_terms_and_both_qk_swap_orders(self):
        for _, entry, account in self.accounts():
            component = entry["retained_qk_component"]
            kind = account["projection"]
            partner = "key" if kind == "query" else "query"
            retained = Fraction(component["actual_" + kind]) - Fraction(component["reference_" + kind])
            self.assertEqual(Fraction(account["retained_projection_delta"]), retained)
            exact = Fraction(account["exact_input_delta"])
            boundary = Fraction(account["projection_boundary_delta"])
            self.assertEqual(exact + boundary, retained)
            for side in ("reference", "actual"):
                factor = Fraction(component[side + "_" + partner]) / 8
                self.assertEqual(Fraction(account[side + "_partner_over_8"]), factor)
                for name, value in (("input", exact), ("boundary", boundary), ("retained", retained)):
                    self.assertEqual(Fraction(account[f"qk_{name}_at_{side}_partner"]), value * factor)
                for term in account["largest_absolute_coordinates"] + account["groups_by_absolute_signed_sum"]:
                    self.assertEqual(Fraction(term[f"qk_at_{side}_partner"]),
                                     Fraction(term["signed_contribution"]) * factor)
            q, k = entry["query"], entry["key"]
            delta = Fraction(component["signed_contribution"])
            self.assertEqual(delta, Fraction(q["qk_retained_at_reference_partner"])
                             + Fraction(k["qk_retained_at_actual_partner"]))
            self.assertEqual(delta, Fraction(k["qk_retained_at_reference_partner"])
                             + Fraction(q["qk_retained_at_actual_partner"]))

    def test_hotspot_selection_counts_and_inherited_report(self):
        counts = [0, 0, 0, 0, 0, 0]
        for row, old in zip(self.report["controls"], self.previous[3]["controls"], strict=True):
            self.assertEqual(row["control"], old["control"])
            for h, oh in zip(row["score_hotspots"], old["score_hotspots"], strict=True):
                self.assertEqual(h["retained_score"], oh["retained_score"])
                self.assertEqual((h["source_position"], h["source_token_id"]), (0, 9707))
                self.assertEqual(h["retained_score"]["probability_delta"], "0")
                self.assertEqual([e["retained_qk_component"] for e in h["dimensions"]],
                                 oh["source_tokens"][0]["qk_account"]["largest_absolute_components"])
                self.assertEqual([e["dimension_hotspot_rank"] for e in h["dimensions"]], list(range(1, 9)))
                counts[0] += 1
                for entry in h["dimensions"]:
                    counts[1] += 1
                    for kind in ("query", "key"):
                        a = entry[kind]
                        for i, n in enumerate((1, a["coordinate_count"],
                                               len(a["largest_absolute_coordinates"]),
                                               len(a["groups_by_absolute_signed_sum"])), 2):
                            counts[i] += n
        self.assertEqual(counts, [72, 576, 1152, 1032192, 9216, 8064])
        self.assertEqual(self.report["qk_logit_attribution_report"], self.previous[3])

    def test_schema_json_and_deterministic_rebuild(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        rebuilt = d.report(self.previous[3], self.actual, self.reference, self.tensors)
        self.assertEqual(text, json.dumps(rebuilt, sort_keys=True, allow_nan=False))

    def test_schema_rejects_counts_extra_fields_and_scope(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("projection_account_count", 1151), ("input_coordinate_term_count", 0),
                             ("input_group_count", 8063), ("selected_input_coordinate_count", 0),
                             ("attention_reference", "binary64"), ("extra", 1), ("controls", [])):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        account = next(self.accounts())[2]
        for field, value in (("largest_absolute_coordinates", []), ("groups_by_absolute_signed_sum", []),
                             ("input_stage", "input_i"), ("exact_local_identity", False)):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate({**account, field: value})
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate(
                {**account, "projection": "key", "output_coordinate": 128})

    def test_synthetic_cancellation_zero_ties_and_boundary_remainder(self):
        component = deepcopy(next(self.accounts())[1]["retained_qk_component"])
        component.update(actual_query="3", reference_query="1", actual_key="0", reference_key="2")
        x, r, w = [Fraction()] * 896, [Fraction()] * 896, [Fraction(1)] * 896
        x[0], x[1], x[128], x[129] = Fraction(4), Fraction(-4), Fraction(4), Fraction(-4)
        account = d.account(x, r, w, component, "query")
        self.assertEqual([e["coordinate"] for e in account["largest_absolute_coordinates"]],
                         [0, 1, 128, 129, 2, 3, 4, 5])
        self.assertEqual(account["exact_input_delta"], "0")
        self.assertEqual(account["projection_boundary_delta"], "2")
        self.assertEqual(account["qk_boundary_at_reference_partner"], "1/2")
        self.assertEqual(account["qk_retained_at_actual_partner"], "0")
        self.assertEqual([g["input_group"] for g in account["groups_by_absolute_signed_sum"]], list(range(7)))
        zero = d.account(r, r, w, component, "query")
        self.assertEqual([e["coordinate"] for e in zero["largest_absolute_coordinates"]], list(range(8)))

    def test_invalid_projection_operands_and_coordinates(self):
        component = next(self.accounts())[1]["retained_qk_component"]
        weights = [Fraction()] * 896
        for actual, kind in (([], "query"), ([0] * 896, "query"), (self.reference, "value")):
            with self.assertRaises(ValueError):
                d.account(actual, self.reference, weights, component, kind)
        for kind, coordinate in (("query", -1), ("query", 896), ("key", 128), ("key", True), ("v", 0)):
            with self.assertRaises(ValueError):
                d.weight_column(self.tensors, kind, coordinate)

    def test_all_tensor_bindings_reject_corruption_shapes_and_nonfinite(self):
        binding = self.result["preflight"]["L23_original_reference"]["reference"]["canonical"]
        for kind in ("query", "key"):
            for suffix, tensor in self.tensors[kind].items():
                pin = binding[d.PREFIXES[kind] + suffix]
                self.assertIs(d.bind_tensor(tensor, pin, kind, suffix), tensor)
                for value, bad_pin in ((tensor, {**pin, "sha256": "0" * 64}),
                                       (tensor, {**pin, "name": "wrong"}),
                                       (tensor.reshape(-1)[:1], pin),
                                       (tensor.astype("<f8"), pin)):
                    with self.assertRaises(ValueError):
                        d.bind_tensor(value, bad_pin, kind, suffix)
        bad = self.tensors["key"]["scales"].copy()
        bad.flat[0] = np.inf
        pin = binding[d.PREFIXES["key"] + "scales"]
        with self.assertRaises(ValueError):
            d.bind_tensor(bad, {**pin, "sha256": hashlib.sha256(bad.tobytes()).hexdigest()}, "key", "scales")

    def test_archive_authentication_splice_and_stage00_type(self):
        label = self.result["controls"][0]["control"]
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        expected = self.previous[1][label]
        for p, qk in (({**pin, "sha256": "0" * 64}, expected),
                      (pin, ([Fraction()] * 896, [Fraction()] * 128))):
            with self.assertRaises(ValueError):
                d.retained_input(p, actual=True, expected_qk=qk)
        with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for bad in (np.zeros(895, dtype="<u2"), np.zeros(896, dtype="<f2"),
                    np.full(896, 0x7e00, dtype="<u2")):
            payload = io.BytesIO()
            np.savez(payload, **{**arrays, "stage00": bad})
            with patch.object(d.base, "read_bound", return_value=payload.getvalue()):
                with self.assertRaises(ValueError):
                    d.retained_input(pin, actual=True, expected_qk=expected)

    def test_control_selection_mapping_and_component_tamper(self):
        with self.assertRaises(ValueError):
            d.report(self.previous[3], {}, self.reference, self.tensors)
        for mutation in ("controls", "ranking", "selection", "mapping", "term", "gqa"):
            bad = deepcopy(self.previous[3])
            a = bad["controls"][0]["score_hotspots"][0]["source_tokens"][0]["qk_account"]
            if mutation == "controls":
                bad["controls"].reverse()
            elif mutation == "ranking":
                a["full_absolute_component_ranking"].reverse()
            elif mutation == "selection":
                a["largest_absolute_components"].reverse()
            elif mutation == "gqa":
                h = bad["controls"][0]["score_hotspots"][0]
                h["retained_score"]["kv_head"] = 1 - h["retained_score"]["kv_head"]
            else:
                field = "key_coordinate" if mutation == "mapping" else "interaction"
                value = (a["largest_absolute_components"][0]["key_coordinate"] + 1) % 128 if mutation == "mapping" else "999"
                a["largest_absolute_components"][0][field] = value
                a["full_absolute_component_ranking"][0][field] = value
            with self.assertRaises(ValueError):
                d.report(bad, self.actual, self.reference, self.tensors)

    def test_original_references_failures_thresholds_and_state_gates(self):
        retained = self.previous[3]["score_vs_value_report"]["source_value_attribution_report"]["final_head_channel_report"]
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for field, value in (("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)
        for text in ("Q24 state", "wider than FP16", "NOT propagation", "no qzero",
                     "No root-cause", "post-RMSNorm", "No multi-token ranking"):
            self.assertIn(text, d.BOUNDARY)

    def test_forbidden_dispatch_prior_checks_and_evidence_writes(self):
        calls = [
            lambda: native.projection(None, None, None), lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.parent.rmsnorm(None, None), lambda: d.parent.logits(None, None),
            lambda: d.qk.check(), lambda: d.qk.focused_tests(None),
            lambda: d.qk.scores.check(), lambda: d.attribution.check(), lambda: d.channels.check(),
            lambda: d.base.check(), lambda: d.channels.contributions.dyadic_dot(None, None),
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
                     "qk_projection_replay", "rope_replay", "score_replay", "softmax_replay"):
            self.assertEqual(d.FLAGS[flag], 0)

    def test_cli_stdout_only_rejects_execution_and_output_options(self):
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
