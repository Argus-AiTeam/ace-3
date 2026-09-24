"""Independent integer-oracle complement checks and retained Q/K regressions."""

from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
import unittest
from unittest.mock import patch

import jsonschema
import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_qk_projection_unselected_stage00_complement_v1 as d


EVIDENCE = None
spec = importlib.util.spec_from_file_location("retained_qk_input_oracles", d.inputs.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("retained Q/K input oracle loader unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ProjectionComplementTests(oracle.ProjectionInputTests):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            with d.read_only({"forbidden_calls": 0}):
                evidence, _, _ = d.measure()
        else:
            evidence = EVIDENCE
        oracle.EVIDENCE, cls.extended = evidence
        super().setUpClass()

    @classmethod
    def complements(cls, report=None):
        for row in (cls.extended if report is None else report)["controls"]:
            for hotspot in row["score_hotspots"]:
                for entry in hotspot["dimensions"]:
                    for kind in ("query", "key"):
                        yield row["control"], entry, entry[kind]

    def synthetic(self, kind="query"):
        actual = [Fraction(10)] * 8 + [Fraction()] * 888
        actual[128], actual[129] = Fraction(3), Fraction(-3)
        actual[256], actual[384] = Fraction(2), Fraction(-1)
        reference, weights = [Fraction()] * 896, [Fraction(1)] * 896
        component = deepcopy(next(self.accounts())[1]["retained_qk_component"])
        component.update(actual_query="3", reference_query="1", actual_key="0", reference_key="-2")
        account = d.inputs.account(actual, reference, weights, component, kind)
        return actual, reference, weights, component, kind, account

    def test_complement_census_and_preserved_report(self):
        stripped = deepcopy(self.extended)
        self.assertEqual(stripped.pop("unselected_input_coordinate_count"), 1022976)
        accounts = coordinates = groups = 0
        for _, _, account in self.complements(stripped):
            split = account.pop("unselected_stage00_inputs")
            accounts += 1
            coordinates += split["coordinate_count"]
            groups += len(split["groups_in_input_order"])
        self.assertEqual((accounts, coordinates, groups), (1152, 1022976, 8064))
        self.assertEqual(stripped, self.report)

    def test_all_complement_groups_independent_integer_oracle(self):
        for label, _, account in self.complements():
            values = self.values[label, account["projection"], account["output_coordinate"]]
            selected = sorted(range(896), key=lambda i: (-abs(values[i]), i))[:8]
            split = account["unselected_stage00_inputs"]
            self.assertEqual(split["excluded_ranked_coordinates"], selected)
            self.assertEqual(split["coordinate_count"], 888)
            for group, entry in enumerate(split["groups_in_input_order"]):
                indices = [i for i in range(group * 128, (group + 1) * 128) if i not in selected]
                signed = sum((values[i] for i in indices), Fraction())
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group * 128,
                    "end_coordinate_exclusive": (group + 1) * 128,
                    "coordinate_count": len(indices),
                    "excluded_ranked_coordinate_count": 128 - len(indices),
                    "signed_contribution": str(signed),
                    "sum_absolute_contributions": str(sum((abs(values[i]) for i in indices), Fraction())),
                    **{f"qk_at_{side}_partner": str(signed * Fraction(account[side + "_partner_over_8"]))
                       for side in ("reference", "actual")},
                })
            remaining = [values[i] for i in range(896) if i not in selected]
            self.assertEqual(Fraction(split["unselected_signed_sum"]), sum(remaining))
            self.assertEqual(Fraction(split["unselected_absolute_sum"]), sum(map(abs, remaining)))

    def test_complement_projection_and_both_swap_closures(self):
        for _, entry, account in self.complements():
            split = account["unselected_stage00_inputs"]
            signed, absolute = Fraction(split["unselected_signed_sum"]), Fraction(split["unselected_absolute_sum"])
            selected = [Fraction(e["signed_contribution"]) for e in account["largest_absolute_coordinates"]]
            self.assertEqual(signed, sum(Fraction(g["signed_contribution"]) for g in split["groups_in_input_order"]))
            self.assertEqual(absolute, sum(Fraction(g["sum_absolute_contributions"]) for g in split["groups_in_input_order"]))
            self.assertEqual(signed, Fraction(account["unranked_signed_remainder"]))
            self.assertEqual(absolute + sum(map(abs, selected)), Fraction(account["sum_absolute_contributions"]))
            self.assertGreaterEqual(absolute, abs(signed))
            total = sum(selected) + signed + Fraction(account["projection_boundary_delta"])
            self.assertEqual(total, Fraction(account["retained_projection_delta"]))
            for side in ("reference", "actual"):
                factor = Fraction(account[side + "_partner_over_8"])
                self.assertEqual(Fraction(split[f"qk_unselected_at_{side}_partner"]), signed * factor)
                self.assertEqual(total * factor, Fraction(account[f"qk_retained_at_{side}_partner"]))
            q, k = entry["query"], entry["key"]
            for first, second in ((q, k), (k, q)):
                pieces = []
                for operand, side in ((first, "reference"), (second, "actual")):
                    pieces.append(
                        (Fraction(operand["ranked_signed_sum"])
                         + Fraction(operand["unselected_stage00_inputs"]["unselected_signed_sum"])
                         + Fraction(operand["projection_boundary_delta"]))
                        * Fraction(operand[side + "_partner_over_8"]))
                self.assertEqual(sum(pieces), Fraction(entry["retained_qk_component"]["signed_contribution"]))
            self.assertTrue(all(split[key] for key in (
                "exact_complement_identity", "exact_projection_identity", "exact_qk_partner_identities")))

    def test_complement_schema_and_json_shape(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.extended)
        self.assertEqual(json.loads(json.dumps(self.extended, allow_nan=False)), self.extended)
        original = next(self.complements())[2]["unselected_stage00_inputs"]
        for mutation in ("count", "order", "duplicate", "extra", "reference", "identity"):
            bad = deepcopy(original)
            if mutation == "count":
                bad["coordinate_count"] = 896
            elif mutation == "order":
                bad["groups_in_input_order"].reverse()
            elif mutation == "duplicate":
                bad["excluded_ranked_coordinates"][1] = bad["excluded_ranked_coordinates"][0]
            elif mutation == "extra":
                bad["admitted"] = True
            elif mutation == "reference":
                bad["attention_reference"] = "binary64"
            else:
                bad["exact_complement_identity"] = False
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(d.SPLIT_SCHEMA).validate(bad)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(1032192, d.REPORT_SCHEMA["properties"]["unselected_input_coordinate_count"])
        account = next(self.complements())[2]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(d.ACCOUNT_SCHEMA).validate(
                {**account, "projection": "key", "output_coordinate": 128})

    def test_complement_deterministic_rebuild_without_input_mutation(self):
        original = json.dumps(self.report, sort_keys=True, allow_nan=False)
        self.assertEqual(d.report(self.evidence), self.extended)
        self.assertEqual(json.dumps(self.report, sort_keys=True, allow_nan=False), original)

    def test_complement_zero_ties_cancellation_and_partner_signs(self):
        for kind in ("query", "key"):
            args = self.synthetic(kind)
            split = d.split_unselected(*args)
            self.assertEqual(split["excluded_ranked_coordinates"], list(range(8)))
            self.assertEqual((split["unselected_signed_sum"], split["unselected_absolute_sum"]), ("1", "9"))
            self.assertEqual((split["groups_in_input_order"][1]["signed_contribution"],
                              split["groups_in_input_order"][1]["sum_absolute_contributions"]), ("0", "6"))
            self.assertEqual(args[-1]["projection_boundary_delta"], "-79")
            if kind == "query":
                self.assertEqual(split["qk_unselected_at_reference_partner"], "-1/4")
                self.assertEqual(split["qk_unselected_at_actual_partner"], "0")
            else:
                self.assertEqual(split["qk_unselected_at_reference_partner"], "1/8")
                self.assertEqual(split["qk_unselected_at_actual_partner"], "3/8")
            _, reference, weights, component, _, _ = args
            account = d.inputs.account(reference, reference, weights, component, kind)
            zero = d.split_unselected(reference, reference, weights, component, kind, account)
            self.assertEqual(zero["excluded_ranked_coordinates"], list(range(8)))
            self.assertEqual((zero["unselected_signed_sum"], zero["unselected_absolute_sum"]), ("0", "0"))
            self.assertEqual([g["coordinate_count"] for g in zero["groups_in_input_order"]],
                             [120, 128, 128, 128, 128, 128, 128])

    def test_complement_selected_tamper_rejected(self):
        *operands, account = self.synthetic()
        for field in ("coordinate", "input_group", "input_delta", "weight", "signed_contribution",
                      "qk_at_reference_partner", "qk_at_actual_partner"):
            bad = deepcopy(account)
            bad["largest_absolute_coordinates"][0][field] = 128 if field in ("coordinate", "input_group") else "999"
            with self.assertRaises(ValueError):
                d.split_unselected(*operands, bad)
        for mutation in ("duplicate", "order", "short"):
            bad = deepcopy(account)
            selected = bad["largest_absolute_coordinates"]
            if mutation == "duplicate":
                selected[1] = selected[0]
            elif mutation == "order":
                selected.reverse()
            else:
                selected.pop()
            with self.assertRaises(ValueError):
                d.split_unselected(*operands, bad)

    def test_complement_totals_groups_and_partner_tamper_rejected(self):
        *operands, account = self.synthetic()
        fields = ("unranked_signed_remainder", "ranked_signed_sum", "exact_input_delta",
                  "sum_absolute_contributions", "retained_projection_delta", "projection_boundary_delta",
                  "reference_partner_over_8", "actual_partner_over_8")
        fields += tuple(f"qk_{term}_at_{side}_partner"
                        for term in ("input", "boundary", "retained") for side in ("reference", "actual"))
        for field in fields:
            with self.assertRaises(ValueError):
                d.split_unselected(*operands, {**account, field: "999"})
        for field in ("signed_contribution", "sum_absolute_contributions",
                      "qk_at_reference_partner", "qk_at_actual_partner"):
            bad = deepcopy(account)
            bad["groups_by_absolute_signed_sum"][0][field] = "999"
            with self.assertRaises(ValueError):
                d.split_unselected(*operands, bad)

    def test_complement_invalid_operands_and_scope_rejected(self):
        args = self.synthetic()
        for index in range(3):
            for replacement in ([], [0] * 896, [Fraction()] * 895):
                bad = list(args)
                bad[index] = replacement
                with self.assertRaises(ValueError):
                    d.split_unselected(*bad)
        for field, value in (("projection", "key"), ("output_coordinate", 896),
                             ("input_stage", "input_i"), ("coordinate_count", 888)):
            with self.assertRaises(ValueError):
                d.split_unselected(*args[:-1], {**args[-1], field: value})
        for field in ("actual_key", "reference_query"):
            bad = list(args)
            bad[3] = {**bad[3], field: "999"}
            with self.assertRaises(ValueError):
                d.split_unselected(*bad)

    def test_complement_report_control_and_component_splices_rejected(self):
        for mutation in ("control", "component", "score", "dimension"):
            bad = deepcopy(self.report)
            hotspot = bad["controls"][0]["score_hotspots"][0]
            if mutation == "control":
                bad["controls"].reverse()
            elif mutation == "component":
                hotspot["dimensions"][0]["retained_qk_component"]["actual_key"] = "999"
            elif mutation == "score":
                hotspot["score_hotspot_rank"] = 2
            else:
                hotspot["dimensions"].reverse()
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                d.report((*self.evidence[:4], bad))

    def test_complement_source_and_kv_lineage_rejections(self):
        row = self.result["controls"][0]
        pin = row["parent"]["terminal_archive"]
        with np.load(io.BytesIO(d.base.read_bound(pin)), allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        for key in ("stage01", "stage02", "stage05", "stage06", "output_cache_k", "input_cache_k"):
            bad = {**original, key: original[key].copy()}
            if key == "input_cache_k":
                bad[key] = np.zeros((1, 128), dtype="<u2")
            else:
                bad[key].flat[0] ^= np.uint16(1)
            with self.assertRaises(ValueError):
                d.inputs.qk.qk_operands(bad, actual=True)
        with self.assertRaises(ValueError):
            d.base.read_bound({**pin, "sha256": "0" * 64})

    def test_complement_read_only_cli_and_prior_check_guards(self):
        calls = [
            lambda: d.inputs.check(), lambda: d.inputs.focused_tests(None),
            lambda: d.inputs.qk.check(), lambda: d.parent.execute(None),
            lambda: oracle.native.projection(None, None, None),
            lambda: oracle.local.local_reference(None, None, None, None),
            lambda: d.parent.write(None, None), lambda: open(d.SOURCE, "w"),
            lambda: d.SOURCE.write_text("forbidden"),
            lambda: oracle.os.unlink(d.SOURCE),
            lambda: oracle.subprocess.Popen(["false"]),
        ]
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        self.assertEqual(d.FLAGS, d.inputs.FLAGS)
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
                       patch.object(d, "SOURCE", d.TEST),
                       patch.object(d.os, "getuid", return_value=0),
                       patch.object(d.sys, "executable", "/wrong/python"),
                       patch.object(d.sys, "dont_write_bytecode", False),
                       patch.dict(d.os.environ, {"PYTHONPATH": "/wrong"})):
            with change, patch.object(d, "measure", side_effect=AssertionError("must not measure")):
                with self.assertRaises(ValueError):
                    d.check()


if __name__ == "__main__":
    unittest.main()
