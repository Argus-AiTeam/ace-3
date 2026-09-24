"""Independent integer-oracle complement checks and retained V input regressions."""

from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
import unittest
from unittest.mock import patch

import jsonschema

from ace3.model.candidates import diagnose_q24_s16_final_head_attention_value_projection_unselected_stage00_complement_v1 as d


EVIDENCE = None
spec = importlib.util.spec_from_file_location("retained_v_input_oracles", d.inputs.TEST)
if spec is None or spec.loader is None:
    raise RuntimeError("retained V input oracle loader unavailable")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class ValueProjectionComplementTests(oracle.ValueProjectionInputTests):
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
    def complements(cls):
        for row in cls.extended["controls"]:
            for branch in row["branches"].values():
                for hotspot in branch["hotspots"]:
                    for entry in hotspot["value_components"]:
                        yield row["control"], entry["value_projection"]

    @staticmethod
    def synthetic():
        actual = [Fraction(10)] * 8 + [Fraction()] * 888
        actual[128], actual[129] = Fraction(3), Fraction(-3)
        actual[256], actual[384] = Fraction(2), Fraction(-1)
        reference, weights = [Fraction()] * 896, [Fraction(1)] * 896
        component = {
            "source_position": 0, "source_token_id": 9707, "value_coordinate": 0,
            "kv_head": 0, "head_dimension": 0, "query_heads": list(range(7)),
            "probability": "0", "interaction": "0", "value": "-4", "signed_contribution": "-4",
        }
        account = d.inputs.account(actual, reference, weights, 0, Fraction(3),
                                   Fraction(1), Fraction(-2), component)
        return actual, reference, weights, account

    def test_complement_census_and_preserved_report(self):
        stripped = deepcopy(self.extended)
        self.assertEqual(stripped.pop("unselected_input_coordinate_count"), 1022976)
        accounts = coordinates = groups = 0
        for row in stripped["controls"]:
            for branch in row["branches"].values():
                for hotspot in branch["hotspots"]:
                    for entry in hotspot["value_components"]:
                        split = entry["value_projection"].pop("unselected_stage00_inputs")
                        accounts += 1
                        coordinates += split["coordinate_count"]
                        groups += len(split["groups_in_input_order"])
        self.assertEqual((accounts, coordinates, groups), (1152, 1022976, 8064))
        self.assertEqual(stripped, self.report)

    def test_all_complement_groups_independent_integer_oracle(self):
        for label, account in self.complements():
            values = self.values[label, account["output_coordinate"]]
            selected = sorted(range(896), key=lambda i: (-abs(values[i]), i))[:8]
            split = account["unselected_stage00_inputs"]
            self.assertEqual(split["excluded_ranked_coordinates"], selected)
            self.assertEqual(split["coordinate_count"], 888)
            for group, entry in enumerate(split["groups_in_input_order"]):
                indices = [i for i in range(group * 128, (group + 1) * 128) if i not in selected]
                self.assertEqual(entry, {
                    "input_group": group, "start_coordinate": group * 128,
                    "end_coordinate_exclusive": (group + 1) * 128,
                    "coordinate_count": len(indices),
                    "excluded_ranked_coordinate_count": 128 - len(indices),
                    "signed_contribution": str(sum((values[i] for i in indices), Fraction())),
                    "sum_absolute_contributions": str(sum((abs(values[i]) for i in indices), Fraction())),
                })
            remaining = [values[i] for i in range(896) if i not in selected]
            self.assertEqual(Fraction(split["unselected_signed_sum"]), sum(remaining))
            self.assertEqual(Fraction(split["unselected_absolute_sum"]), sum(map(abs, remaining)))

    def test_complement_projection_and_value_closure(self):
        for _, account in self.complements():
            split = account["unselected_stage00_inputs"]
            signed, absolute = Fraction(split["unselected_signed_sum"]), Fraction(split["unselected_absolute_sum"])
            selected = [Fraction(e["signed_contribution"]) for e in account["largest_absolute_coordinates"]]
            self.assertEqual(signed, sum(Fraction(g["signed_contribution"])
                                         for g in split["groups_in_input_order"]))
            self.assertEqual(absolute, sum(Fraction(g["sum_absolute_contributions"])
                                           for g in split["groups_in_input_order"]))
            self.assertEqual(signed, Fraction(account["unranked_signed_remainder"]))
            self.assertEqual(absolute + sum(map(abs, selected)), Fraction(account["sum_absolute_contributions"]))
            self.assertGreaterEqual(absolute, abs(signed))
            boundary, factor = Fraction(account["projection_boundary_remainder"]), Fraction(account["value_component_factor"])
            self.assertEqual(sum(selected) + signed + boundary, Fraction(account["retained_projection_delta"]))
            self.assertEqual((sum(selected) + signed + boundary) * factor, Fraction(account["retained_value_contribution"]))
            self.assertTrue(all(split[key] for key in (
                "exact_complement_identity", "exact_projection_identity", "exact_value_component_identity")))

    def test_complement_schema_and_single_json_shape(self):
        jsonschema.Draft202012Validator.check_schema(d.OUTPUT_SCHEMA)
        jsonschema.Draft202012Validator(d.REPORT_SCHEMA).validate(self.extended)
        self.assertEqual(json.loads(json.dumps(self.extended, allow_nan=False)), self.extended)
        _, account = next(self.complements())
        original = account["unselected_stage00_inputs"]
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
        schema = d.REPORT_SCHEMA["properties"]["unselected_input_coordinate_count"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(1032192, schema)

    def test_complement_zero_ties_and_signed_cancellation(self):
        actual, reference, weights, account = self.synthetic()
        split = d.split_unselected(actual, reference, weights, account)
        self.assertEqual(split["excluded_ranked_coordinates"], list(range(8)))
        self.assertEqual((split["unselected_signed_sum"], split["unselected_absolute_sum"]), ("1", "9"))
        self.assertEqual((split["groups_in_input_order"][1]["signed_contribution"],
                          split["groups_in_input_order"][1]["sum_absolute_contributions"]), ("0", "6"))
        self.assertEqual(account["projection_boundary_remainder"], "-79")
        self.assertEqual(account["weighted_boundary_contribution"], "158")
        zero = d.inputs.account(reference, reference, weights, 0, Fraction(3), Fraction(1),
                                Fraction(-2), {
                                    "source_position": 0, "source_token_id": 9707,
                                    "value_coordinate": 0, "kv_head": 0, "head_dimension": 0,
                                    "query_heads": list(range(7)), "probability": "0",
                                    "interaction": "0", "value": "-4", "signed_contribution": "-4"})
        zero_split = d.split_unselected(reference, reference, weights, zero)
        self.assertEqual(zero_split["excluded_ranked_coordinates"], list(range(8)))
        self.assertEqual((zero_split["unselected_signed_sum"], zero_split["unselected_absolute_sum"]), ("0", "0"))
        self.assertEqual([g["coordinate_count"] for g in zero_split["groups_in_input_order"]],
                         [120, 128, 128, 128, 128, 128, 128])

    def test_complement_selected_tamper_rejected(self):
        actual, reference, weights, account = self.synthetic()
        for field in ("coordinate", "input_group", "input_delta", "weight", "signed_contribution"):
            bad = deepcopy(account)
            bad["largest_absolute_coordinates"][0][field] = 128 if field in ("coordinate", "input_group") else "999"
            with self.assertRaises(ValueError):
                d.split_unselected(actual, reference, weights, bad)
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
                d.split_unselected(actual, reference, weights, bad)

    def test_complement_retained_totals_and_groups_tamper_rejected(self):
        actual, reference, weights, account = self.synthetic()
        fields = ("unranked_signed_remainder", "ranked_signed_sum", "exact_input_delta",
                  "sum_absolute_contributions", "retained_projection_delta",
                  "projection_boundary_remainder", "weighted_input_contribution",
                  "weighted_boundary_contribution", "retained_value_contribution")
        for field in fields:
            with self.assertRaises(ValueError):
                d.split_unselected(actual, reference, weights, {**account, field: "999"})
        for field in ("signed_contribution", "sum_absolute_contributions"):
            bad = deepcopy(account)
            bad["groups_by_absolute_signed_sum"][0][field] = "999"
            with self.assertRaises(ValueError):
                d.split_unselected(actual, reference, weights, bad)

    def test_complement_invalid_operands_and_scope_rejected(self):
        actual, reference, weights, account = self.synthetic()
        for index in range(3):
            for replacement in ([], [0] * 896, [Fraction()] * 895):
                values = [actual, reference, weights]
                values[index] = replacement
                with self.assertRaises(ValueError):
                    d.split_unselected(*values, account)
        for field, value in (("projection", "q_proj"), ("input_stage", "stage13_fp16"),
                             ("output_stage", "stage07_fp16"), ("position", 1), ("layer", 22)):
            with self.assertRaises(ValueError):
                d.split_unselected(actual, reference, weights, {**account, field: value})

    def test_complement_cli_stdout_only(self):
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

    def test_complement_forbidden_prior_checks_and_writes(self):
        calls = [
            lambda: d.inputs.check(), lambda: d.inputs.focused_tests(None),
            lambda: d.inputs.attribution.check(), lambda: d.channels.check(),
            lambda: oracle.native.projection(None, None, None),
            lambda: oracle.local.local_reference(None, None, None, None),
            lambda: d.parent.execute(None), lambda: d.parent.write(None, None),
            lambda: d.SOURCE.write_text("forbidden"), lambda: d.TEST.unlink(),
            lambda: oracle.subprocess.Popen(["false"]),
        ]
        audit = {"forbidden_calls": 0}
        with d.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))
        self.assertEqual(d.FLAGS, d.inputs.FLAGS)


if __name__ == "__main__":
    unittest.main()
