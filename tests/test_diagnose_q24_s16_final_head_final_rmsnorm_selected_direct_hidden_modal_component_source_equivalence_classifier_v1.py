"""Independent integer-pair oracle and hostile retained-evidence controls."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from functools import cmp_to_key
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_component_source_equivalence_classifier_v1 as d


def pair(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, q):
    common = gcd(n, q)
    n, q = n // common, q // common
    return str(n) if q == 1 else f"{n}/{q}"


def add(left, right):
    a, b = pair(left)
    c, e = pair(right)
    return exact(a * e + c * b, b * e)


def negate(value):
    n, q = pair(value)
    return exact(-n, q)


def subtract(left, right):
    return add(left, negate(right))


def multiply(left, right):
    a, b = pair(left)
    c, e = pair(right)
    return exact(a * c, b * e)


def ordered(left, right):
    a, b = pair(left)
    c, e = pair(right)
    return (a * e > c * b) - (a * e < c * b)


def summed(values):
    result = "0"
    for value in values:
        result = add(result, value)
    return result


class ModalSourceEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for binding in d.HELPER_PINS:
            d.retained_io.bound_bytes(binding)
        cls.audit = {"forbidden_calls": 0}
        with d.retained_io.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.retained_io.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)
        node = cls.retained["retained_667"]["retained_3fd"]
        cls.comparators = node
        cls.hotspots = node["retained_0146"]["retained_6184"]["retained_e795"]["retained_449b"]
        cls.source = cls.hotspots["retained_50f"]

    def assert_spread_oracle(self, account):
        values = account["values"]
        self.assertEqual(tuple(values), d.CONTROLS)
        ordered_values = sorted(values.values(), key=cmp_to_key(ordered))
        low, high = ordered_values[0], ordered_values[-1]
        self.assertEqual(account["minimum"], low)
        self.assertEqual(account["maximum"], high)
        self.assertEqual(account["spread"], subtract(high, low))
        self.assertEqual(account["minimum_controls"], [c for c, v in values.items() if v == low])
        self.assertEqual(account["maximum_controls"], [c for c, v in values.items() if v == high])
        deltas = {c: subtract(v, values[d.CANONICAL]) for c, v in values.items()}
        self.assertEqual(account["deltas_from_canonical"], deltas)
        mismatches = [c for c, value in deltas.items() if value != "0"]
        self.assertEqual([r["control"] for r in account["mismatch_rows"]], mismatches)
        self.assertEqual(account["integrity_status"], "REJECTED" if mismatches else "SUPPORTED")

    def test_all_280_components_40_sums_and_retained_factor_spreads_with_integer_oracle(self):
        components = coordinates = 0
        for partition in self.result["partitions"]:
            for account in partition["coordinates"]:
                for spread in account["field_spreads"].values():
                    self.assert_spread_oracle(spread)
                for control in d.CONTROLS:
                    row = d.selected_row(self.source, partition, control, account["coordinate"])
                    values = {k: v["values"][control] for k, v in account["field_spreads"].items()}
                    self.assertEqual(values[d.FACTOR], row[d.FACTOR])
                    self.assertEqual(values[d.ANCHOR], row[d.ANCHOR])
                    for component in d.COMPONENTS:
                        hidden, weighted = row["hidden_components"][component], row["weighted_components"][component]
                        self.assertEqual(values["hidden." + component], hidden)
                        self.assertEqual(values["weighted." + component], weighted)
                        self.assertEqual(multiply(row[d.FACTOR], hidden), weighted)
                        components += 1
                    self.assertEqual(summed(list(row["hidden_components"].values())), values["hidden.sum"])
                    self.assertEqual(summed(list(row["weighted_components"].values())), values["weighted.sum"])
                    self.assertEqual(values["weighted.sum"], row["direct_hidden_weighted_term"])
                    self.assertEqual(multiply(row[d.FACTOR], values["hidden.sum"]), values["weighted.sum"])
                    coordinates += 1
        self.assertEqual((components, coordinates), (280, 40))

    def test_candidates_context_and_parent_compatibility_with_integer_oracle(self):
        for partition, parent in zip(self.result["partitions"], self.comparators["report"]["partitions"], strict=True):
            for family in ("candidate_spreads", "parent_comparator_spreads"):
                for account in partition[family].values():
                    self.assert_spread_oracle(account)
            for profile in parent["control_profiles"]:
                control = profile["control"]
                values = {k: v["values"][control] for k, v in partition["candidate_spreads"].items()}
                for label, identity in partition["candidate_identities"].items():
                    row = d.selected_row(self.source, partition, control, identity["coordinate"])
                    expected = (row["weighted_components"][identity["component"]] if "component" in identity
                                else summed(list(row["weighted_components"].values())))
                    self.assertEqual(values[label + ".signed"], expected)
                    self.assertEqual(expected, profile["fields"]["comparator." + label + ".signed"])
                expected = subtract(values["exception.signed"], values["modal.signed"])
                self.assertEqual(values["signed_comparator"], expected)
                self.assertEqual(profile["fields"]["comparator.signed_comparator"], expected)
                if "selected_sum" in values:
                    self.assertEqual(values["selected_sum"], add(values["exception.signed"], values["modal.signed"]))
                    row = d.selected_row(self.source, partition, control, 241)
                    self.assertEqual(add(values["selected_sum"], values["context_sum"]), row["direct_hidden_weighted_term"])
                    self.assertNotEqual(values["selected_sum"], values["signed_comparator"])

    def test_reversed_pair_oracle_and_input_immutability(self):
        checks = self.result["reversed_pair"]["checks"]
        self.assertTrue(checks)
        for row in checks:
            self.assertEqual(row["actual"], row["expected"])
            self.assertEqual(row["integrity_status"], "SUPPORTED")
        forward, reverse = self.result["partitions"][:2]
        for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
            for control in d.CONTROLS:
                for component in d.COMPONENTS:
                    self.assertEqual(left["field_spreads"]["hidden." + component]["values"][control],
                                     right["field_spreads"]["hidden." + component]["values"][control])
                    self.assertEqual(negate(left["field_spreads"]["weighted." + component]["values"][control]),
                                     right["field_spreads"]["weighted." + component]["values"][control])
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual(self.result["separate_weight_and_row_difference_equality"], "UNKNOWN")

    def test_exact_spreads_nonzero_ties_cancellation_and_large_denominators(self):
        for entries in (
            ["-7/3", "2/3", "2/3", "-7/3", "0", "1/3", "0", "1/3"],
            ["1/9007199254740993", "1/9007199254740992", "0", "-1", "2", "-1", "2", "0"],
            ["0"] * 8,
        ):
            self.assert_spread_oracle(d.spread(dict(zip(d.CONTROLS, entries, strict=True))))
        self.assertEqual(d.total(["1/3", "-1/3"]), "0")

    def test_missing_scalars_are_unknown_not_equal_and_mismatch_precedes_unknown(self):
        values = dict.fromkeys(d.CONTROLS, "0")
        values[d.CONTROLS[1]] = None
        result = d.spread(values)
        self.assertEqual(result["integrity_status"], "UNKNOWN")
        self.assertFalse(result["extrema_cover_all_controls"])
        values[d.CONTROLS[2]] = "1/7"
        self.assertEqual(d.spread(values)["integrity_status"], "REJECTED")
        values[d.CANONICAL] = None
        self.assertEqual(d.spread(values)["integrity_status"], "UNKNOWN")
        self.assertEqual(d.spread(dict.fromkeys(d.CONTROLS))["spread"], None)

    def classify_changed_row(self, change):
        source, hotspots = deepcopy(self.source), deepcopy(self.hotspots)
        parent = self.retained["report"]["partitions"][0]
        for document in (source, hotspots):
            change(d.selected_row(document, parent, d.CONTROLS[1], 62))
        return d.classify_partition(parent, self.comparators["report"]["partitions"][0], hotspots, source)

    def test_source_mismatch_is_not_hidden_by_equal_parent_candidate(self):
        def mutate(row):
            row["hidden_components"]["input_hidden"] = add(row["hidden_components"]["input_hidden"], "1")
        result = self.classify_changed_row(mutate)
        self.assertEqual(result["integrity_status"], "REJECTED")
        self.assertTrue(any(r["field"] == "product.input_hidden" for r in result["mismatch_rows"]))

    def test_factor_weighted_sum_and_anchor_tampering_rejected(self):
        for key in (d.FACTOR, d.ANCHOR, "direct_hidden_weighted_term"):
            with self.subTest(key=key):
                result = self.classify_changed_row(lambda row: row.__setitem__(key, add(row[key], "1")))
                self.assertEqual(result["integrity_status"], "REJECTED")

    def test_missing_component_produces_explicit_unknown(self):
        result = self.classify_changed_row(lambda row: row["hidden_components"].pop("input_hidden"))
        missing = result["coordinates"][0]["field_spreads"]["hidden.input_hidden"]
        self.assertEqual(missing["unknown_controls"], [d.CONTROLS[1]])
        self.assertIsNone(missing["deltas_from_canonical"][d.CONTROLS[1]])
        self.assertFalse(missing["extrema_cover_all_controls"])
        self.assertTrue(any(r["control"] == d.CONTROLS[1]
                            and r["field"] == "coordinate.62.hidden.input_hidden"
                            for r in result["unknown_rows"]))
        self.assertEqual(result["integrity_status"], "REJECTED" if result["mismatch_rows"] else "UNKNOWN")

    def test_parent_splice_and_reverse_sign_tampering_rejected(self):
        parent = deepcopy(self.comparators["report"]["partitions"][0])
        parent["control_profiles"][1]["fields"]["comparator.signed_comparator"] = "0"
        result = d.classify_partition(self.retained["report"]["partitions"][0], parent, self.hotspots, self.source)
        self.assertEqual(result["integrity_status"], "REJECTED")
        reverse = deepcopy(self.result["partitions"][1])
        reverse["coordinates"][0]["field_spreads"][d.FACTOR]["values"][d.CANONICAL] = "1"
        self.assertEqual(d.orientation(self.result["partitions"][0], reverse)["integrity_status"], "REJECTED")

    def test_duplicate_controls_and_out_of_scope_fail_closed(self):
        for control in ("mapped_all", d.CONTROLS[0]):
            comparator = deepcopy(self.comparators["report"]["partitions"][0])
            comparator["control_profiles"][1]["control"] = control
            with self.assertRaises(ValueError):
                d.classify_partition(self.retained["report"]["partitions"][0], comparator, self.hotspots, self.source)
        parent = deepcopy(self.retained["report"]["partitions"][0])
        parent["branch"] = "fp16"
        with self.assertRaises(ValueError):
            d.classify_partition(parent, self.comparators["report"]["partitions"][0], self.hotspots, self.source)

    def test_noncanonical_float_nonfinite_and_duplicate_json_rejected(self):
        for value in (0.5, True, "2/4", "0.5", "NaN", "1/0"):
            with self.subTest(value=value), self.assertRaises((ValueError, ZeroDivisionError)):
                d.rational(value)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{} {}'):
            with self.assertRaises(ValueError):
                d.retained_io.decode(text)
        tree = ast.parse(d.SOURCE.read_bytes())
        self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == "float" for n in ast.walk(tree)))

    def test_authentication_rejects_byte_review_capture_source_and_counter_mutation(self):
        pin = d.retained_io.PINS["review"]
        with patch.object(Path, "read_bytes", return_value=b"tampered"):
            with self.assertRaises(ValueError):
                d.retained_io.bound_bytes(pin)
        original = d.retained_io.bound_bytes
        for kind in ("review", "capture"):
            changed = deepcopy(self.review if kind == "review" else self.capture)
            if kind == "review":
                changed["producer_role"] = "engineer"
            else:
                changed["results"][-1]["exit_status"] = 1
            with patch.object(d.retained_io, "bound_bytes", side_effect=lambda binding:
                              d.retained_io.json.dumps(changed).encode()
                              if binding == d.retained_io.PINS[kind] else original(binding)):
                with self.assertRaises(ValueError):
                    d.retained_io.authenticate()
        changed = deepcopy(self.retained)
        changed["compiled_sources"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.retained_io.validate_retained(changed)
        changed = deepcopy(self.retained)
        changed["flags"]["head_operator_replay"] = 1
        with self.assertRaises(ValueError):
            d.retained_io.validate_retained(changed)

    def test_read_only_guard_denies_write_tensor_read_and_dispatch(self):
        import os
        import subprocess
        operations = (
            lambda: open(d.retained_io.PINS["stdout"]["path"], "wb"),
            lambda: Path("/unretained-tensor").read_bytes(),
            lambda: subprocess.run(["true"], check=True),
            lambda: os.mkdir("/forbidden-modal-classifier-write"),
        )
        for operation in operations:
            audit = {"forbidden_calls": 0}
            with d.retained_io.read_only(audit), self.assertRaises(RuntimeError):
                operation()
            self.assertEqual(audit["forbidden_calls"], 1)

    def test_helper_authentication_and_boundary(self):
        for binding in d.HELPER_PINS:
            self.assertEqual(hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest(), binding["sha256"])
        self.assertTrue(self.result["descriptive_accounting_only"])
        self.assertEqual(self.result["modal_control_count"], 8)
        self.assertEqual(self.result["unstable_partition_count"], 3)
        self.assertEqual(self.result["coordinate_account_count"], 5)
        self.assertIn(self.result["integrity_status"], ("SUPPORTED", "REJECTED", "UNKNOWN"))
        self.assertEqual(d.status([], []), "SUPPORTED")
        self.assertEqual(d.status([], ["missing"]), "UNKNOWN")
        self.assertEqual(d.status(["mismatch"], ["missing"]), "REJECTED")

    def test_stdout_parser_and_authenticated_helper_snapshot(self):
        output, error = io.StringIO(), io.StringIO()
        with patch.object(d, "check", return_value={"integrity_status": "UNKNOWN"}) as check:
            with redirect_stdout(output), redirect_stderr(error):
                d.main(["--check"])
        check.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue()), {"integrity_status": "UNKNOWN"})
        self.assertEqual(error.getvalue(), "")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            d.main([])
        for replacement in (b"invalid", b"x" * d.HELPER_PINS[0]["bytes"]):
            with patch.object(Path, "read_bytes", return_value=replacement):
                with self.assertRaises(ValueError):
                    d.load_retained_io()


if __name__ == "__main__":
    unittest.main()
