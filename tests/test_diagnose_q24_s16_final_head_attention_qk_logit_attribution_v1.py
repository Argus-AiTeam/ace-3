"""Independent integer-ratio oracle and fail-closed retained Q/K checks."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_qk_logit_attribution_v1 as d
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


def scaled_dot(q, k):
    terms = [(a.numerator * b.numerator, a.denominator * b.denominator * 8)
             for a, b in zip(q, k, strict=True)]
    denominator = max(b for _, b in terms)
    return Fraction(sum(a * (denominator // b) for a, b in terms), denominator)


class QKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                cls.evidence, _, _ = d.measure()
        else:
            cls.evidence = EVIDENCE
        cls.prior, cls.actual, cls.reference, cls.report = cls.evidence
        cls.result = cls.prior[0][0][0]

    def accounts(self):
        for row in self.report["controls"]:
            for hotspot in row["score_hotspots"]:
                yield row["control"], hotspot, hotspot["source_tokens"][0]["qk_account"]

    def test_retained_qk_independent_bit_oracle(self):
        pins = [(row["parent"]["terminal_archive"], self.actual[row["control"]])
                for row in self.result["controls"]]
        pins.append((self.result["preflight"]["L23_original_reference"]["reference"]["fp16"],
                     self.reference))
        for pin, (q, k) in pins:
            with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
                self.assertEqual(q, [half(int(w)) for w in archive["stage04"]])
                self.assertEqual(k, [half(int(w)) for w in archive["stage06"]])

    def test_all_selected_components_independent_four_product_oracle(self):
        for label, hotspot, account in self.accounts():
            score = hotspot["retained_score"]
            h, kv = score["query_head"], score["kv_head"]
            qa, ka = self.actual[label]
            qr, kr = self.reference
            expected = {}
            for dim in range(64):
                qi, ki = h * 64 + dim, (h // 7) * 64 + dim
                products = [scaled_dot([q], [k]) for q, k in
                            ((qr[qi], kr[ki]), (qa[qi], kr[ki]),
                             (qr[qi], ka[ki]), (qa[qi], ka[ki]))]
                c00, c10, c01, c11 = products
                expected[dim] = (c10 - c00, c01 - c00, c11 - c10 - c01 + c00, c11 - c00)
            for entry in account["full_absolute_component_ranking"]:
                dim = entry["head_dimension"]
                self.assertEqual((entry["query_coordinate"], entry["key_coordinate"]),
                                 (h * 64 + dim, kv * 64 + dim))
                self.assertEqual(tuple(Fraction(entry[k]) for k in d.PARTS), expected[dim])
                self.assertEqual([Fraction(entry[k]) for k in
                                  ("actual_query", "reference_query", "actual_key", "reference_key")],
                                 [qa[h * 64 + dim], qr[h * 64 + dim],
                                  ka[kv * 64 + dim], kr[kv * 64 + dim]])
            for field, q, k in (("actual_exact_scaled_dot", qa, ka),
                                 ("reference_exact_scaled_dot", qr, kr)):
                self.assertEqual(Fraction(account[field]), scaled_dot(
                    q[h * 64:(h + 1) * 64], k[kv * 64:(kv + 1) * 64]))

    def test_swap_orders_and_retained_boundary_identities(self):
        for _, hotspot, account in self.accounts():
            a = {k: Fraction(account[k]) for k in (*d.PARTS, "query_at_actual_key",
                 "key_at_actual_query", "score_boundary_delta", "retained_score_delta",
                 "actual_exact_scaled_dot", "reference_exact_scaled_dot",
                 "actual_score_boundary_remainder", "reference_score_boundary_remainder")}
            delta = a["signed_contribution"]
            self.assertEqual(delta, a["query_at_reference_key"] + a["key_at_actual_query"])
            self.assertEqual(delta, a["key_at_reference_query"] + a["query_at_actual_key"])
            self.assertEqual(a["query_at_actual_key"] - a["query_at_reference_key"], a["interaction"])
            self.assertEqual(delta + a["score_boundary_delta"], a["retained_score_delta"])
            self.assertEqual(a["score_boundary_delta"],
                             a["actual_score_boundary_remainder"] - a["reference_score_boundary_remainder"])
            for side in ("actual", "reference"):
                self.assertEqual(a[side + "_exact_scaled_dot"] + a[side + "_score_boundary_remainder"],
                                 Fraction(hotspot["retained_score"][side + "_score"]))

    def test_groups_and_full_and_selected_deterministic_rankings(self):
        for _, _, account in self.accounts():
            rows = account["full_absolute_component_ranking"]
            self.assertEqual(sorted(e["head_dimension"] for e in rows), list(range(64)))
            expected = sorted(rows, key=lambda e: (-abs(Fraction(e["signed_contribution"])),
                                                 e["head_dimension"]))
            self.assertEqual(rows, expected)
            self.assertEqual(account["largest_absolute_components"], expected[:8])
            self.assertEqual(Fraction(account["ranked_signed_sum"]),
                             sum(Fraction(e["signed_contribution"]) for e in expected[:8]))
            self.assertEqual(Fraction(account["ranked_signed_sum"])
                             + Fraction(account["unranked_signed_remainder"]),
                             Fraction(account["signed_contribution"]))
            for group, start in zip(account["component_groups"], (0, 32), strict=True):
                self.assertEqual((group["group"], group["start_dimension"], group["end_dimension_exclusive"]),
                                 (f"dimensions_{start}_{start + 31}", start, start + 32))
                for part in d.PARTS:
                    self.assertEqual(Fraction(group[part]), sum(
                        Fraction(e[part]) for e in rows if start <= e["head_dimension"] < start + 32))
            for part in d.PARTS:
                self.assertEqual(sum(Fraction(g[part]) for g in account["component_groups"]),
                                 Fraction(account[part]))

    def test_hotspot_source_census_and_probability_invariance(self):
        counts = [0, 0, 0, 0]
        for row, old in zip(self.report["controls"], self.prior[3]["controls"], strict=True):
            self.assertEqual(row["control"], old["control"])
            self.assertEqual([h["retained_score"] for h in row["score_hotspots"]],
                             old["scores_by_absolute_delta"][:8])
            self.assertEqual([h["score_hotspot_rank"] for h in row["score_hotspots"]], list(range(1, 9)))
            for hotspot in row["score_hotspots"]:
                source = hotspot["source_tokens"][0]
                self.assertEqual((source["source_rank"], source["source_position"], source["source_token_id"]),
                                 (1, 0, 9707))
                self.assertEqual(hotspot["retained_score"]["probability_delta"], "0")
                a = source["qk_account"]
                for i, n in enumerate((1, len(a["full_absolute_component_ranking"]),
                                       len(a["largest_absolute_components"]), len(a["component_groups"]))):
                    counts[i] += n
        self.assertEqual(counts, [72, 4608, 576, 144])

    def test_schema_json_and_deterministic_rebuild(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.report)
        text = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(text), self.report)
        rebuilt = d.report(self.prior[3], self.actual, self.reference)
        self.assertEqual(text, json.dumps(rebuilt, sort_keys=True, allow_nan=False))

    def test_schema_rejects_counts_scope_and_extra_fields(self):
        validator = jsonschema.Draft202012Validator(d.REPORT_SCHEMA)
        for field, value in (("dimension_account_count", 4607), ("source_account_count", 0),
                             ("selected_dimension_count", 575), ("controls", []),
                             ("attention_reference", "binary64"), ("extra", 0)):
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate({**self.report, field: value})
        account = next(self.accounts())[2]
        for field, value in (("score_scale", "1/64"), ("component_groups", []),
                             ("exact_local_identity", False)):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate({**account, field: value})

    def test_synthetic_interaction_cancellation_and_ties(self):
        r = ([Fraction(1)] * 896, [Fraction(1)] * 128)
        a = ([Fraction(1)] * 896, [Fraction(1)] * 128)
        a[0][0], a[1][0] = Fraction(2), Fraction(3)
        a[0][1] = Fraction(-4)
        score = d.scores.score_rows([Fraction(9)] * 14, [Fraction(8)] * 14, [1] * 14, [1] * 14)[0]
        account = d.account(a, r, score)
        self.assertEqual([e["head_dimension"] for e in account["largest_absolute_components"]], list(range(8)))
        first = account["full_absolute_component_ranking"][0]
        self.assertEqual([Fraction(first[k]) for k in d.PARTS],
                         [Fraction(1, 8), Fraction(1, 4), Fraction(1, 4), Fraction(5, 8)])
        self.assertEqual(account["signed_contribution"], "0")
        self.assertEqual(account["score_boundary_delta"], "1")
        zero = d.account(r, r, score)
        self.assertEqual(zero["signed_contribution"], "0")
        self.assertEqual([e["head_dimension"] for e in zero["full_absolute_component_ranking"]], list(range(64)))

    def test_invalid_operand_shapes_types_head_mapping_and_scores(self):
        score = self.prior[3]["controls"][0]["scores_by_absolute_delta"][0]
        for actual in (([Fraction()] * 895, [Fraction()] * 128), ([0] * 896, [0] * 128)):
            with self.assertRaises(ValueError):
                d.account(actual, self.reference, score)
        for field, value in (("kv_head", 1 - score["kv_head"]), ("score_delta", "100000"),
                             ("actual_probability", "0")):
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.account(self.reference, self.reference, {**score, field: value})

    def test_p0_projection_rope_cache_lineage_and_invalid_words(self):
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in
                      ("stage01", "stage02", "stage04", "stage05", "stage06",
                       "input_cache_k", "output_cache_k")}
        for key in ("stage01", "stage02", "stage05", "output_cache_k"):
            bad = deepcopy(arrays)
            bad[key].flat[0] ^= 1
            with self.assertRaises(ValueError):
                d.qk_operands(bad, actual=True)
        for array in (np.full(896, 0x7c00, dtype="<u2"), np.full(896, 0x7e00, dtype="<u2"),
                      np.zeros(895, dtype="<u2"), np.zeros(896, dtype="<f2")):
            with self.assertRaises(ValueError):
                d.qk_operands({**arrays, "stage04": array}, actual=True)
        with self.assertRaises(ValueError):
            d.qk_operands({**arrays, "input_cache_k": np.zeros((1, 128), dtype="<u2")}, actual=True)

    def test_control_order_duplicate_heads_and_ranking_tamper(self):
        for actual in ({}, dict(reversed(list(self.actual.items())))):
            with self.assertRaises(ValueError):
                d.report(self.prior[3], actual, self.reference)
        for mutation in ("reverse", "duplicate", "count"):
            bad = deepcopy(self.prior[3])
            row = bad["controls"][0]
            if mutation == "reverse":
                row["scores_by_absolute_delta"].reverse()
            elif mutation == "duplicate":
                row["scores_by_absolute_delta"][1] = row["scores_by_absolute_delta"][0]
            else:
                row["changed_score_head_count"] = 0
            with self.assertRaises(ValueError):
                d.report(bad, self.actual, self.reference)

    def test_archive_source_score_splice_and_history_tamper(self):
        pin = self.result["controls"][0]["parent"]["terminal_archive"]
        expected = self.prior[1][self.result["controls"][0]["control"]]
        with self.assertRaises(ValueError):
            d.retained_qk({**pin, "sha256": "0" * 64}, actual=True, expected_scores=expected)
        with self.assertRaises(ValueError):
            d.retained_qk(pin, actual=True, expected_scores=[Fraction()] * 14)
        for pin in d.channels.PINS.values():
            with self.assertRaises(ValueError):
                d.base.read_bound({**pin, "sha256": "0" * 64})
        for field, value in (("candidate_admitted", True), ("source_operand_state_KV_RTZ_checks", "FAIL")):
            bad = deepcopy(self.result)
            bad["controls"][0]["parent"]["retained_L23"][field] = value
            with self.assertRaises(ValueError):
                d.base.check_history(bad)

    def test_original_references_failures_thresholds_and_scope_preserved(self):
        self.assertEqual(self.report["score_vs_value_report"], self.prior[3])
        previous = self.report["score_vs_value_report"]["source_value_attribution_report"]
        retained = previous["final_head_channel_report"]
        self.assertEqual(retained["original_global_reference"], self.result["preflight"]["final_reference"])
        self.assertEqual(retained["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertTrue(all(r["L21_L22_L23_status"] == ["FAIL"] * 3 for r in retained["controls"]))
        self.assertEqual(previous["attention_reference"]["binary64_stage_status"],
                         "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertIs(previous["attention_reference"]["final_head_branches_reanchored"], False)
        for text in ("Q24 state", "wider than FP16", "NOT propagation", "no qzero",
                     "No root-cause", "No multi-token ranking"):
            self.assertIn(text, d.BOUNDARY)

    def test_forbidden_native_dispatch_prior_checks_and_evidence_writes(self):
        calls = [
            lambda: native.projection(None, None, None), lambda: native._stages(None, None, None, None),
            lambda: local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.rmsnorm(None, None),
            lambda: d.parent.logits(None, None), lambda: d.parent.preflight.parent.execute(None),
            lambda: d.scores.check(), lambda: d.scores.focused_tests(None),
            lambda: d.scores.attribution.check(), lambda: d.channels.check(),
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
