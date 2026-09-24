"""Independent scalar oracle and mutation tests for attention complement accounting."""

from copy import deepcopy
from fractions import Fraction
import io
import json
import os
import subprocess
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_source_value_unselected_component_complement_v1 as d
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as native


EVIDENCE = None


def half(word):
    sign = -1 if word & 32768 else 1
    exponent, mantissa = (word >> 10) & 31, word & 1023
    if exponent == 31:
        raise ValueError("nonfinite half")
    return sign * (Fraction(mantissa, 2 ** 24) if exponent == 0
                   else Fraction(1024 + mantissa) * Fraction(2) ** (exponent - 25))


def independent_column(tensors, coordinate):
    physical = (0, 2, 4, 6, 1, 3, 5, 7).index(coordinate % 8)
    column = coordinate // 8
    result = []
    for j in range(896):
        q = ((int(tensors["qweight"][j, column]) & 0xffffffff) // 16 ** physical) % 16
        z = ((int(tensors["qzeros"][j // 128, column]) & 0xffffffff) // 16 ** physical) % 16
        scale = half(int(tensors["scales"].view("<u2")[j // 128, coordinate]))
        result.append((q - z) * scale)
    return result


class AttentionComplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.previous, cls.report = cls.evidence
        cls.channel_evidence, cls.actual, cls.reference, cls.tensors, cls.inherited = cls.previous
        cls.result = cls.channel_evidence[0]

    def accounts(self, report=None):
        for row in (self.report if report is None else report)["controls"]:
            for branch, item in row["branches"].items():
                for hotspot in item["hotspots"]:
                    yield row["control"], branch, hotspot

    def original(self):
        return deepcopy(next(self.accounts(self.inherited))[2]["attention"])

    def test_independent_all_component_and_complement_oracle(self):
        columns = {}
        for label, _, hotspot in self.accounts():
            coordinate = hotspot["final_head_channel"]["coordinate"]
            if coordinate not in columns:
                columns[coordinate] = independent_column(self.tensors, coordinate)
            weights = columns[coordinate]
            va, vr = self.actual[label][1], self.reference[1]
            terms = [
                (va[kv * 64 + dim] - vr[kv * 64 + dim])
                * sum(weights[head * 64 + dim] for head in range(kv * 7, kv * 7 + 7))
                for kv in range(2) for dim in range(64)]
            order = sorted(range(128), key=lambda i: (-abs(terms[i]), i))
            account = hotspot["attention"]
            split = account["unselected_value_component_complement"]
            self.assertEqual(split["excluded_ranked_value_coordinates"], order[:8])
            self.assertEqual(split["unselected_value_coordinates_in_input_order"], sorted(order[8:]))
            for entry in account["full_value_component_ranking"]:
                self.assertEqual(Fraction(entry["signed_contribution"]), terms[entry["value_coordinate"]])
            for name, indices in (("selected", order[:8]), ("unselected", order[8:]),
                                  ("full", order)):
                total = split[name]
                signed = sum(terms[i] for i in indices)
                absolute = sum(abs(terms[i]) for i in indices)
                self.assertEqual(Fraction(total["signed_contribution"]), signed)
                self.assertEqual(Fraction(total["sum_absolute_contributions"]), absolute)
                self.assertEqual(Fraction(total["value"]), signed)
                self.assertEqual(Fraction(total["absolute_parts"]["value"]), absolute)
                for key in ("probability", "interaction"):
                    self.assertEqual(total[key], "0")
                    self.assertEqual(total["absolute_parts"][key], "0")
            delta = self.actual[label][3][coordinate] - self.reference[3][coordinate]
            self.assertEqual(Fraction(account["retained_o_projection_delta"]), delta)
            self.assertEqual(Fraction(account["av_boundary_remainder"]), 0)
            self.assertEqual(Fraction(account["o_projection_boundary_remainder"]), delta - sum(terms))

    def test_inherited_report_is_preserved_without_mutation(self):
        for (_, _, new), (_, _, old) in zip(self.accounts(), self.accounts(self.inherited), strict=True):
            self.assertEqual({k: v for k, v in new.items() if k != "attention"},
                             {k: v for k, v in old.items() if k != "attention"})
            self.assertEqual({k: v for k, v in new["attention"].items()
                              if k != "unselected_value_component_complement"}, old["attention"])
            self.assertNotIn("unselected_value_component_complement", old["attention"])
        for key in ("final_head_channel_report", "attention_reference"):
            self.assertEqual(self.report[key], self.inherited[key])

    def test_complete_deterministic_census(self):
        rows = list(self.accounts())
        self.assertEqual(len(rows), 144)
        self.assertEqual(self.report["selected_value_component_count"], 1152)
        self.assertEqual(self.report["unselected_value_component_count"], 17280)
        self.assertEqual(self.report["unselected_query_value_product_count"], 120960)
        for _, _, hotspot in rows:
            split = hotspot["attention"]["unselected_value_component_complement"]
            excluded = split["excluded_ranked_value_coordinates"]
            unselected = split["unselected_value_coordinates_in_input_order"]
            self.assertEqual((len(excluded), len(unselected)), (8, 120))
            self.assertEqual(sorted(excluded + unselected), list(range(128)))
            self.assertEqual(unselected, sorted(unselected))
            self.assertEqual(split["unselected_query_value_product_count"], 840)

    def test_source_token_signed_absolute_totals(self):
        for _, _, hotspot in self.accounts():
            account = hotspot["attention"]
            split = account["unselected_value_component_complement"]
            source = split["source_tokens"][0]
            self.assertEqual((source["source_position"], source["source_token_id"]), (0, 9707))
            self.assertEqual((source["selected_component_count"], source["unselected_component_count"]),
                             (8, 120))
            for name in ("selected", "unselected", "full"):
                self.assertEqual(source[name], split[name])
            for key in (*d.PARTS, "signed_contribution"):
                self.assertEqual(source["full"][key], account["source_tokens"][0][key])

    def test_selected_complement_and_boundary_closure(self):
        for _, _, hotspot in self.accounts():
            a = hotspot["attention"]
            split = a["unselected_value_component_complement"]
            for key in (*d.PARTS, "signed_contribution", "sum_absolute_contributions"):
                self.assertEqual(Fraction(split["selected"][key]) + Fraction(split["unselected"][key]),
                                 Fraction(split["full"][key]))
            self.assertEqual(split["selected"]["signed_contribution"], a["ranked_signed_sum"])
            self.assertEqual(split["unselected"]["signed_contribution"], a["unranked_signed_remainder"])
            self.assertEqual(split["full"]["signed_contribution"], a["exact_attention_sum"])
            self.assertEqual(sum(Fraction(a[key]) for key in (
                "ranked_signed_sum", "unranked_signed_remainder", "av_boundary_remainder",
                "o_projection_boundary_remainder")), Fraction(a["retained_o_projection_delta"]))

    def test_original_references_thresholds_and_failures(self):
        retained = self.report["final_head_channel_report"]
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in retained["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertTrue(all(row["retained_failures"].values()))
        ref = self.report["attention_reference"]
        self.assertEqual(ref["fp16"], retained["original_global_reference"]["reference"]["input_fp16"])
        self.assertEqual(ref["binary64_stage_status"], "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertIs(ref["final_head_branches_reanchored"], False)

    def test_authenticated_fp16_operands_independent_bit_decoder(self):
        pins = [(row["parent"]["terminal_archive"], self.actual[row["control"]])
                for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference))
        for pin, operands in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                for stage, values in zip(("stage09", "stage07", "stage10", "stage11"),
                                         operands, strict=True):
                    self.assertEqual(values, [half(int(word)) for word in archive[stage]])

    def test_ties_zero_and_cancellation_absolute_not_absolute_sum(self):
        a = self.original()
        for entry in a["full_value_component_ranking"]:
            i = entry["value_coordinate"]
            entry.update(probability="0", interaction="0",
                         value="1" if i % 2 == 0 else "-1",
                         signed_contribution="1" if i % 2 == 0 else "-1")
        a["full_value_component_ranking"].sort(key=lambda e: e["value_coordinate"])
        a["largest_absolute_value_components"] = deepcopy(a["full_value_component_ranking"][:8])
        a["source_tokens"][0].update(probability="0", value="0", interaction="0", signed_contribution="0")
        for key in ("exact_attention_sum", "ranked_signed_sum", "unranked_signed_remainder",
                    "av_boundary_remainder", "o_projection_boundary_remainder",
                    "retained_o_projection_delta"):
            a[key] = "0"
        split = d.split_unselected(a)
        self.assertEqual(split["unselected"]["signed_contribution"], "0")
        self.assertEqual(split["unselected"]["sum_absolute_contributions"], "120")
        self.assertEqual(split["excluded_ranked_value_coordinates"], list(range(8)))
        for entry in a["full_value_component_ranking"]:
            entry.update(value="0", signed_contribution="0")
        a["largest_absolute_value_components"] = deepcopy(a["full_value_component_ranking"][:8])
        zero = d.split_unselected(a)
        self.assertEqual(zero["unselected"]["sum_absolute_contributions"], "0")
        self.assertEqual(zero["unselected_value_coordinates_in_input_order"], list(range(8, 128)))

    def test_duplicate_missing_and_reordered_components_rejected(self):
        a = self.original()
        variants = (a["full_value_component_ranking"][:-1],
                    a["full_value_component_ranking"][:-1] + [a["full_value_component_ranking"][0]],
                    list(reversed(a["full_value_component_ranking"])))
        for entries in variants:
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.split_unselected({**a, "full_value_component_ranking": entries})

    def test_selected_entry_and_order_mutations_rejected(self):
        for key, value in (("value", "999"), ("source_token_id", 1)):
            a = self.original()
            a["largest_absolute_value_components"][0][key] = value
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.split_unselected(a)
        a = self.original()
        a["largest_absolute_value_components"].reverse()
        with self.assertRaises(ValueError):
            d.split_unselected(a)

    def test_component_parts_source_and_gqa_mutations_rejected(self):
        for key, value in (("value", "999"), ("source_position", 1), ("kv_head", 2),
                           ("head_dimension", 64), ("query_heads", [0] * 7)):
            a = self.original()
            a["full_value_component_ranking"][-1][key] = value
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.split_unselected(a)

    def test_source_totals_and_remainders_mutations_rejected(self):
        for key in ("exact_attention_sum", "ranked_signed_sum", "unranked_signed_remainder",
                    "av_boundary_remainder", "o_projection_boundary_remainder",
                    "retained_o_projection_delta"):
            a = self.original()
            a[key] = str(Fraction(a[key]) + 1)
            with self.assertRaises(ValueError):
                d.split_unselected(a)
        for key in (*d.PARTS, "signed_contribution"):
            a = self.original()
            a["source_tokens"][0][key] = str(Fraction(a["source_tokens"][0][key]) + 1)
            with self.assertRaises(ValueError):
                d.split_unselected(a)

    def test_scope_and_nonfinite_rejected(self):
        for key, value in (("position", 1), ("layer", 22), ("attention_reference", "binary64"),
                           ("exact_attention_sum", "nan"), ("source_token_count", 2)):
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.split_unselected({**self.original(), key: value})

    def test_schema_serialization_and_counts(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        validator.validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        for key, value in (("hotspot_count", 143), ("unselected_value_component_count", 17279),
                           ("selected_value_component_count", 0), ("extra", True)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, key: value})
        split = next(self.accounts())[2]["attention"]["unselected_value_component_complement"]
        for key, value in (("unselected_component_count", 119),
                           ("unselected_query_value_product_count", 839),
                           ("exact_local_identity", False), ("attention_reference", "binary64")):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.SPLIT_SCHEMA).validate({**split, key: value})

    def test_control_branch_and_hotspot_splices_rejected(self):
        bad = {**self.inherited, "controls": list(reversed(self.inherited["controls"]))}
        with self.assertRaises(ValueError):
            d.report(bad)
        bad = deepcopy(self.inherited)
        row = bad["controls"][0]
        row["branches"] = dict(reversed(list(row["branches"].items())))
        with self.assertRaises(ValueError):
            d.report(bad)
        bad = deepcopy(self.inherited)
        branch = next(iter(bad["controls"][0]["branches"].values()))
        branch["hotspots"].reverse()
        with self.assertRaises(ValueError):
            d.report(bad)

    def test_source_and_archive_pins_reject_tamper(self):
        for pin in (self.result["controls"][0]["parent"]["terminal_archive"],
                    *d.channels.PINS.values(), *d.base.PINS.values()):
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_state_and_admission_history_gates(self):
        for field, value in (("candidate_admitted", True),
                             ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)

    def test_probability_cache_and_lineage_gates(self):
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as arrays:
            original = {key: arrays[key] for key in arrays.files}
        for key in ("stage03", "stage07", "stage09", "stage10", "output_cache_v"):
            bad = deepcopy(original)
            bad[key].flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.attribution.attention_operands(bad, actual=True)
        with self.assertRaises(ValueError):
            d.attribution.attention_operands({
                **original, "input_cache_k": np.zeros((1, 128), dtype="<u2")}, actual=True)

    def test_attention_reference_and_awq_binding_gates(self):
        assets = self.result["preflight"]["assets"]
        for field, value in (("prior_kv", "other"), ("fp16", {})):
            bad = deepcopy(self.result)
            bad["preflight"]["L23_original_reference"]["reference"][field] = value
            with self.assertRaises(ValueError):
                d.attribution.load_attention(bad, assets)
        bad = deepcopy(self.result)
        pin = bad["preflight"]["L23_original_reference"]["reference"]["canonical"][
            d.attribution.PROJECTION + "qzeros"]
        pin["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.attribution.load_attention(bad, assets)

    def test_forbidden_dispatch_and_writes(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: native.projection(None, None, None),
            lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
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
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        self.assertEqual(d.FLAGS, d.attribution.FLAGS)
        for key in ("native_dispatch", "prefix_dispatch", "admission_dispatch", "evidence_writes",
                    "RTL_dispatch", "GPU_dispatch", "hardware_dispatch", "attention_replay",
                    "o_projection_replay"):
            self.assertEqual(d.FLAGS[key], 0)
        self.assertIs(d.FLAGS["binary64_attention_reference_reconstructed"], False)

    def test_cli_exactly_one_document(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"result": 1}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"result": 1}\n')
        self.assertEqual(json.loads(stream.getvalue()), {"result": 1})

    def test_cli_rejects_execute_output_and_abbreviation(self):
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_workdir_account_interpreter_and_bytecode_gates(self):
        for target, name, value in ((d.Path, "cwd", d.ROOT.parent),
                                    (d.os, "getuid", 0)):
            with patch.object(target, name, return_value=value), \
                    patch.object(d, "measure", side_effect=AssertionError("must not measure")):
                with self.assertRaises(ValueError):
                    d.check()
        for name, value in (("executable", "/wrong/python"), ("dont_write_bytecode", False)):
            with patch.object(d.sys, name, value), \
                    patch.object(d, "measure", side_effect=AssertionError("must not measure")):
                with self.assertRaises(ValueError):
                    d.check()

    def test_measure_reuses_accounting_not_prior_check_or_producer(self):
        with patch.object(d.attribution, "measure", return_value=(self.previous, [], {})) as inherited, \
                patch.object(d, "report", return_value=self.report) as extend:
            result, files, assets = d.measure()
        inherited.assert_called_once_with()
        extend.assert_called_once_with(self.inherited)
        self.assertIs(result[0], self.previous)
        self.assertIs(result[1], self.report)
        self.assertEqual((files, assets), ([], {}))


if __name__ == "__main__":
    unittest.main()
