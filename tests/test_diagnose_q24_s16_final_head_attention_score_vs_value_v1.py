"""Independent bit/scalar checks of retained scores and fixed-operand contrasts."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_score_vs_value_v1 as d
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


def weights(tensors, coordinate):
    physical = (0, 2, 4, 6, 1, 3, 5, 7).index(coordinate % 8)
    def nibble(word):
        return ((int(word) & 0xffffffff) // 16 ** physical) % 16
    return [(nibble(tensors["qweight"][j, coordinate // 8])
             - nibble(tensors["qzeros"][j // 128, coordinate // 8]))
            * half(int(tensors["scales"].view("<u2")[j // 128, coordinate]))
            for j in range(896)]


class ScoreValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.prior, cls.scores, cls.reference_scores, cls.report = cls.evidence
        cls.channel_evidence, cls.actual, cls.reference, cls.tensors, cls.previous = cls.prior
        cls.result = cls.channel_evidence[0]

    def hotspots(self):
        for row in self.report["controls"]:
            for branch in row["branches"].values():
                for hotspot in branch["hotspots"]:
                    yield row["control"], hotspot

    def test_retained_scores_independent_bit_oracle(self):
        pins = [(row["parent"]["terminal_archive"], self.scores[row["control"]])
                for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference_scores))
        for pin, scores in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                self.assertEqual(scores, [half(int(word)) for word in archive["stage08"]])
                self.assertEqual([half(int(word)) for word in archive["stage09"]], [1] * 14)

    def test_score_deltas_order_mapping_and_counts(self):
        for row in self.report["controls"]:
            delta = [a - r for a, r in zip(
                self.scores[row["control"]], self.reference_scores, strict=True)]
            expected = sorted(range(14), key=lambda h: (-abs(delta[h]), h))
            self.assertEqual([s["query_head"] for s in row["scores_by_absolute_delta"]], expected)
            self.assertEqual(row["changed_score_head_count"], sum(x != 0 for x in delta))
            self.assertEqual(row["changed_probability_head_count"], 0)
            for entry in row["scores_by_absolute_delta"]:
                h = entry["query_head"]
                self.assertEqual(Fraction(entry["score_delta"]), delta[h])
                self.assertEqual(entry["kv_head"], h // 7)
                self.assertEqual(entry["probability_delta"], "0")

    def test_all_source_and_selected_component_products_scalar_oracle(self):
        cache = {}
        for label, hotspot in self.hotspots():
            coordinate = hotspot["final_head_channel"]["coordinate"]
            if coordinate not in cache:
                cache[coordinate] = weights(self.tensors, coordinate)
            w = cache[coordinate]
            pa, va = self.actual[label][:2]
            pr, vr = self.reference[:2]
            sums = [Fraction()] * 4
            component_sums = [[Fraction()] * 4 for _ in range(128)]
            for j in range(896):
                head, dim = divmod(j, 64)
                i = head // 7 * 64 + dim
                terms = (pr[head] * vr[i] * w[j], pa[head] * vr[i] * w[j],
                         pr[head] * va[i] * w[j], pa[head] * va[i] * w[j])
                for k, term in enumerate(terms):
                    sums[k] += term
                    component_sums[i][k] += term
            source = hotspot["source_tokens"][0]
            for key, expected in zip(d.PRODUCTS, sums, strict=True):
                self.assertEqual(Fraction(source["contrasts"][key]), expected)
            for entry in source["largest_absolute_value_components"]:
                for key, expected in zip(d.PRODUCTS, component_sums[entry["value_coordinate"]], strict=True):
                    self.assertEqual(Fraction(entry["contrasts"][key]), expected)

    def test_both_swap_orders_interaction_and_boundary_identities(self):
        for _, hotspot in self.hotspots():
            source = hotspot["source_tokens"][0]
            for item in [source, *source["largest_absolute_value_components"]]:
                c = {k: Fraction(v) for k, v in item["contrasts"].items()}
                self.assertEqual(c["signed_contribution"], c["actual_product"] - c["reference_product"])
                self.assertEqual(c["signed_contribution"],
                                 c["probability_at_reference_value"] + c["value_at_actual_probability"])
                self.assertEqual(c["signed_contribution"],
                                 c["value_at_reference_probability"] + c["probability_at_actual_value"])
                self.assertEqual(c["interaction"],
                                 c["probability_at_actual_value"] - c["probability_at_reference_value"])
                self.assertEqual(c["probability_at_reference_value"], 0)
                self.assertEqual(c["probability_at_actual_value"], 0)
                self.assertEqual(c["interaction"], 0)
            self.assertEqual(Fraction(source["contrasts"]["signed_contribution"])
                             + Fraction(hotspot["av_boundary_remainder"])
                             + Fraction(hotspot["o_projection_boundary_remainder"]),
                             Fraction(hotspot["retained_o_projection_delta"]))

    def test_synthetic_nonzero_weighting_value_interaction(self):
        a = ([Fraction(2)] * 14, [Fraction(3)] * 128)
        r = ([Fraction(1)] * 14, [Fraction(1)] * 128)
        w = [Fraction()] * 896
        w[0], w[64] = Fraction(2), Fraction(-1)
        products = d.product_contrasts(a, r, w)
        self.assertEqual(products[0], (1, 2, 3, 6))
        self.assertEqual([products[i] for i in range(1, 128)], [(0, 0, 0, 0)] * 127)
        c = d.contrast(products[0])
        self.assertEqual([c[k] for k in d.EFFECTS], ["1", "4", "2", "3", "2", "5"])
        self.assertEqual(d.contrast((Fraction(1),) * 4)["signed_contribution"], "0")

    def test_singleton_score_shift_ties_and_zero_scores(self):
        reference = [Fraction()] * 14
        actual = [Fraction(3), Fraction(-3)] + [Fraction()] * 12
        rows = d.score_rows(actual, reference, [Fraction(1)] * 14, [Fraction(1)] * 14)
        self.assertEqual([r["query_head"] for r in rows], list(range(14)))
        self.assertTrue(all(r["probability_delta"] == "0" for r in rows))
        self.assertEqual([r["score_delta"] for r in rows[:2]], ["3", "-3"])

    def test_census_hotspot_selection_and_stable_order(self):
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        count = components = 0
        for row, prior in zip(self.report["controls"], self.previous["controls"], strict=True):
            self.assertEqual(list(row["branches"]), list(d.channels.BRANCHES))
            for name, branch in row["branches"].items():
                old = prior["branches"][name]
                self.assertEqual(branch["numeric_id"], old["numeric_id"])
                for hotspot, previous in zip(branch["hotspots"], old["hotspots"], strict=True):
                    count += 1
                    self.assertEqual(hotspot["hotspot_rank"], previous["hotspot_rank"])
                    self.assertEqual(hotspot["final_head_channel"], previous["final_head_channel"])
                    source = hotspot["source_tokens"][0]
                    self.assertEqual((source["source_rank"], source["source_position"],
                                      source["source_token_id"]), (1, 0, 9707))
                    selected = source["largest_absolute_value_components"]
                    components += len(selected)
                    self.assertEqual([e["value_coordinate"] for e in selected], [
                        e["value_coordinate"] for e in previous["attention"]["largest_absolute_value_components"]])
        self.assertEqual((count, components), (144, 1152))
        self.assertEqual(self.report["score_pair_count"], len(self.report["controls"]) * 14)

    def test_schema_json_and_deterministic_rebuild(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        rebuilt = d.report(self.prior, self.scores, self.reference_scores)
        self.assertEqual(text, json.dumps(rebuilt, sort_keys=True, allow_nan=False))

    def test_schema_rejects_counts_scope_and_extra_fields(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("score_pair_count", 125), ("source_account_count", 0),
                             ("selected_value_component_count", 1151), ("controls", []),
                             ("attention_reference", "binary64"), ("extra", 0)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        row = self.report["controls"][0]["scores_by_absolute_delta"][0]
        for field, value in (("actual_probability", "2"), ("query_head", 14)):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**row, field: value}, d.SCORE_SCHEMA)

    def test_original_references_failures_and_attribution_preserved(self):
        self.assertEqual(self.report["source_value_attribution_report"], self.previous)
        retained = self.previous["final_head_channel_report"]
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertEqual(self.previous["attention_reference"]["binary64_stage_status"],
                         "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertIs(self.previous["attention_reference"]["final_head_branches_reanchored"], False)
        for row in retained["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
        for text in ("Q24 state", "wider than FP16", "NOT propagation", "no qzero",
                     "No root-cause", "not a multi-token source ranking"):
            self.assertIn(text, d.BOUNDARY)

    def test_invalid_scores_probabilities_and_product_dimensions(self):
        for a, r, pa, pr in (([Fraction()] * 13, [Fraction()] * 14, [1] * 14, [1] * 14),
                             ([Fraction()] * 14, [Fraction()] * 14, [2] * 14, [1] * 14),
                             ([Fraction()] * 14, [Fraction()] * 14, [1] * 14, [0] * 14)):
            with self.assertRaises(ValueError):
                d.score_rows(a, r, pa, pr)
        with self.assertRaises(ValueError):
            d.product_contrasts(([1], [1]), ([1], [1]), [])
        for array in (np.array([0x7c00] * 14, dtype="<u2"),
                      np.array([0x7e00] * 14, dtype="<u2"),
                      np.zeros(13, dtype="<u2"), np.zeros(14, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.attribution.words(array, (14,))

    def test_control_order_rank_and_attribution_tamper_rejected(self):
        for scores in ({}, dict(reversed(list(self.scores.items())))):
            with self.assertRaises(ValueError):
                d.report(self.prior, scores, self.reference_scores)
        row = self.previous["controls"][0]
        hotspot = row["branches"][d.channels.BRANCHES[0]]["hotspots"][0]
        original = hotspot["attention"]
        products = d.product_contrasts(
            self.actual[row["control"]], self.reference,
            weights(self.tensors, hotspot["final_head_channel"]["coordinate"]))
        for field, value in (("probability", "1"), ("signed_contribution", "1000000"),
                             ("value_coordinate", 128)):
            bad = deepcopy(original)
            bad["full_value_component_ranking"][0][field] = value
            with self.assertRaises(ValueError):
                d.source_account(bad, products)
        bad = deepcopy(original)
        bad["full_value_component_ranking"].reverse()
        with self.assertRaises(ValueError):
            d.source_account(bad, products)

    def test_archive_source_and_lineage_tamper_rejected(self):
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        with self.assertRaises(ValueError):
            d.retained_scores({**pin, "sha256": "0" * 64})
        for pin in d.channels.PINS.values():
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        for field, value in (("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)
        bad = deepcopy(self.result)
        bad["preflight"]["L23_original_reference"]["reference"]["fp16"] = {}
        with self.assertRaises(ValueError):
            d.attribution.load_attention(bad, self.result["preflight"]["assets"])

    def test_forbidden_dispatch_prior_checks_and_evidence_writes(self):
        calls = [
            lambda: native.projection(None, None, None),
            lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.attribution.check(), lambda: d.attribution.focused_tests(None),
            lambda: d.channels.check(), lambda: d.channels.focused_tests(None),
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
                     "o_projection_replay", "score_replay", "softmax_replay"):
            self.assertEqual(d.FLAGS[flag], 0)

    def test_cli_stdout_only_and_no_execution_or_output_mode(self):
        stream = io.StringIO()
        with patch.object(d, "check", return_value={"result": 1}), patch("sys.stdout", stream):
            d.main(["--check"])
        self.assertEqual(stream.getvalue(), '{"result": 1}\n')
        for args in ([], ["--execute"], ["--check", "--out", "build/new"], ["--che"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
                d.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_workdir_account_interpreter_and_bytecode_gate(self):
        changes = [
            patch.object(d.Path, "cwd", return_value=d.ROOT.parent),
            patch.object(d.os, "getuid", return_value=0),
            patch.object(d.sys, "executable", "/wrong/python"),
            patch.object(d.sys, "dont_write_bytecode", False),
        ]
        for change in changes:
            with change, patch.object(d, "measure", side_effect=AssertionError("must not measure")):
                with self.assertRaises(ValueError):
                    d.check()


if __name__ == "__main__":
    unittest.main()
