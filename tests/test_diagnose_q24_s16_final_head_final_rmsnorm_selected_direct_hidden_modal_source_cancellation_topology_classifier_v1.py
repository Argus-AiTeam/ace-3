"""Independent integer-pair oracle and fail-closed retained-only controls."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_source_cancellation_topology_classifier_v1 as d


def pair(text):
    pieces = text.split("/")
    return int(pieces[0]), int(pieces[1]) if len(pieces) == 2 else 1


def exact(n, q):
    if q < 0:
        n, q = -n, -q
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


def magnitude(value):
    n, q = pair(value)
    return exact(abs(n), q)


def summed(values):
    result = "0"
    for value in values:
        result = add(result, value)
    return result


class CancellationTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def test_native_domain_and_all_80_rows_accounted_without_admission(self):
        r = self.result
        self.assertEqual((r["input_equivalence_status"], r["input_mismatch_count"],
                          r["input_unknown_count"], r["cancellation_topology_status"]),
                         ("REJECTED", 80, 0, "SUPPORTED"))
        self.assertEqual((r["failure_count"], r["failures"], r["unaccounted_mismatch_rows"]), (0, [], []))
        original = self.retained["report"]["mismatch_rows"]
        accounted = [{k: v for k, v in row.items() if k != "classification"}
                     for row in r["accounted_mismatch_rows"]]
        sort = lambda rows: sorted(json.dumps(row, sort_keys=True) for row in rows)
        self.assertEqual(sort(accounted), sort(original))
        self.assertEqual(len(accounted), 80)
        categories = [row["classification"] for row in r["accounted_mismatch_rows"]]
        self.assertEqual(categories.count("equal_opposite_source_redistribution"), 40)
        self.assertEqual(categories.count("absolute_cancellation_mass_redistribution"), 40)
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual(r["lineage_separation"], self.retained["report"]["lineage_separation"])
        self.assertEqual(r["reference_scope"], self.retained["report"]["reference_scope"])
        self.assertEqual(r["separate_weight_and_row_difference_equality"], "UNKNOWN")

    def test_all_40_coordinate_profiles_and_280_products_with_integer_pair_oracle(self):
        count = products = 0
        output = iter(self.result["coordinate_accounts"])
        for partition in self.retained["report"]["partitions"]:
            for coordinate in partition["coordinates"]:
                actual = next(output)
                fields = coordinate["field_spreads"]
                self.assertEqual(actual["coordinate"], coordinate["coordinate"])
                canonical = {k: field["values"]["frozen_inherited"] for k, field in fields.items()}
                self.assertEqual(actual["canonical_values"], canonical)
                for profile in actual["controls"]:
                    control = profile["control"]
                    values = {k: field["values"][control] for k, field in fields.items()}
                    deltas = {k: subtract(v, canonical[k]) for k, v in values.items()}
                    self.assertEqual(profile["field_deltas"], deltas)
                    for component in (
                        "input_hidden", "attention_stage11", "mlp_stage17", "actual_residual_boundary",
                        "negative_reference_residual_boundary", "q24_to_fp16_conversion",
                        "fp16_to_branch_terminal_remainder",
                    ):
                        self.assertEqual(multiply(values[d.FACTOR], values["hidden." + component]),
                                         values["weighted." + component])
                        products += 1
                    for family in ("hidden", "weighted"):
                        source_values = [v for k, v in values.items() if k.startswith(family + ".")
                                         and ".mass." not in k and not k.endswith(".sum")]
                        signed = summed(source_values)
                        absolute = summed([magnitude(v) for v in source_values])
                        self.assertEqual(values[family + ".sum"], signed)
                        self.assertEqual(values[family + ".mass.signed"], signed)
                        self.assertEqual(values[family + ".mass.absolute"], absolute)
                        self.assertEqual(values[family + ".mass.cancellation_absolute_mass"],
                                         subtract(absolute, magnitude(signed)))
                        pair_sum = add(deltas[family + ".actual_residual_boundary"],
                                       deltas[family + ".q24_to_fp16_conversion"])
                        self.assertEqual(profile[family + "_source_delta_sum"], pair_sum)
                        self.assertEqual(pair_sum, "0")
                        self.assertEqual(deltas[family + ".sum"], "0")
                    self.assertEqual(values["weighted.sum"], values["direct_hidden_weighted_term"])
                    self.assertEqual(multiply(values[d.FACTOR], values["hidden.sum"]), values["weighted.sum"])
                    count += 1
        self.assertEqual((count, products), (40, 280))
        self.assertIsNone(next(output, None))

    def test_three_exact_signature_groups_and_mass_movements(self):
        expected = [
            (["frozen_inherited_o", "frozen_inherited_o_down"], "-3/32768"),
            (["scratch", "scratch_down"], "-321/4194304"),
            (["mapped62"], "-933/16777216"),
        ]
        for account in self.result["coordinate_accounts"]:
            if account["coordinate"] != 62:
                self.assertEqual(account["signature_groups"], [])
                continue
            self.assertEqual(len(account["signature_groups"]), 3)
            factor = account["canonical_values"][d.FACTOR]
            for group, (members, hidden_delta) in zip(account["signature_groups"], expected, strict=True):
                self.assertEqual(group["controls"], members)
                signature = group["signature"]
                self.assertEqual(signature["hidden.actual_residual_boundary"], hidden_delta)
                self.assertEqual(signature["hidden.q24_to_fp16_conversion"], negate(hidden_delta))
                weighted = multiply(factor, hidden_delta)
                self.assertEqual(signature["weighted.actual_residual_boundary"], weighted)
                self.assertEqual(signature["weighted.q24_to_fp16_conversion"], negate(weighted))
                hidden_mass = multiply("2", hidden_delta)
                weighted_mass = multiply(magnitude(factor), hidden_mass)
                for kind in ("absolute", "cancellation_absolute_mass"):
                    self.assertEqual(signature["hidden.mass." + kind], hidden_mass)
                    self.assertEqual(signature["weighted.mass." + kind], weighted_mass)
                self.assertNotEqual(hidden_mass, "0")
                self.assertNotEqual(weighted_mass, "0")
                for member in members:
                    profile = next(p for p in account["controls"] if p["control"] == member)
                    self.assertEqual({k: v for k, v in profile["field_deltas"].items() if v != "0"},
                                     signature)

    def test_canonical_equal_controls_coordinate241_and_component_partition(self):
        for account in self.result["coordinate_accounts"]:
            for profile in account["controls"]:
                if account["coordinate"] == 241 or profile["control"] in (
                    "frozen_inherited", "frozen_inherited_down", "inherited_native",
                ):
                    self.assertEqual(set(profile["field_deltas"].values()), {"0"})
        parent = self.retained["report"]["partitions"]
        self.assertEqual([len(p["mismatch_rows"]) for p in parent], [40, 40, 0])
        self.assertEqual(parent[2]["integrity_status"], "SUPPORTED")

    def test_reversed_pairs_identical_hidden_and_negated_weighted_orientation(self):
        accounts = self.result["coordinate_accounts"]
        for forward, reverse in zip(accounts[:2], accounts[2:4], strict=True):
            self.assertEqual((forward["left_id"], forward["right_id"]),
                             (reverse["right_id"], reverse["left_id"]))
            for left, right in zip(forward["controls"], reverse["controls"], strict=True):
                for field, delta in left["field_deltas"].items():
                    signed_weighted = field.startswith("weighted.") and field not in (
                        "weighted.mass.absolute", "weighted.mass.cancellation_absolute_mass",
                    )
                    expected = negate(delta) if signed_weighted or field in (
                        d.FACTOR, "direct_hidden_weighted_term",
                    ) else delta
                    self.assertEqual(right["field_deltas"][field], expected)

    def test_exact_large_denominator_and_invalid_rational_controls(self):
        values = dict.fromkeys(d.CONTROLS, "1/9007199254740993")
        values["scratch"] = "1/9007199254740992"
        result = d.spread(values)
        self.assertEqual(result["deltas_from_canonical"]["scratch"],
                         subtract(values["scratch"], values["frozen_inherited"]))
        self.assertNotEqual(result["spread"], "0")
        for invalid in (None, 0, True, 0.5, "0.5", "2/4", "01", "1/0", "NaN"):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, ZeroDivisionError)):
                d.rational(invalid)
        for raw in (b'{"x":1,"x":2}', b'{"x":0.5}', b'{"x":NaN}', b'{} {}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                d.decode(raw)

    def test_missing_unknown_duplicate_and_nonzero_dispatch_inputs_fail_closed(self):
        for change in (
            lambda x: x["report"].update(integrity_status="SUPPORTED"),
            lambda x: x["report"]["unknown_rows"].append({"field": "missing"}),
            lambda x: x["report"]["partitions"].append(x["report"]["partitions"][0]),
            lambda x: x["report"]["partitions"][0]["modal_controls"].append("mapped_all"),
            lambda x: x["report"]["partitions"][0]["coordinates"][0]["field_spreads"].pop("hidden.sum"),
            lambda x: x["report"]["partitions"][0]["coordinates"][0]["field_spreads"]["hidden.sum"]["values"].update(scratch=None),
            lambda x: x["report"]["partitions"][0]["coordinates"][0]["field_spreads"]["hidden.sum"]["values"].update(scratch=0.5),
            lambda x: x["flags"].update(head_operator_replay=1),
            lambda x: x["dispatch_and_write_audit"].update(evidence_writes=1),
            lambda x: x["dispatch_and_write_audit"].update(forbidden_calls=True),
        ):
            changed = deepcopy(self.retained)
            change(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                d.report(changed)

    def test_changed_scalars_metadata_closures_and_rows_cannot_support_topology(self):
        def values(x):
            return x["report"]["partitions"][0]["coordinates"][0]["field_spreads"]

        changes = [
            lambda x: values(x)["hidden.actual_residual_boundary"]["values"].update(scratch="0"),
            lambda x: values(x)["weighted.actual_residual_boundary"]["values"].update(scratch="0"),
            lambda x: values(x)["hidden.mass.absolute"]["values"].update(scratch="0"),
            lambda x: values(x)["hidden.mass.cancellation_absolute_mass"]["values"].update(scratch="0"),
            lambda x: values(x)["hidden.actual_residual_boundary"]["values"].update(frozen_inherited_down="0"),
            lambda x: values(x)["hidden.sum"]["values"].update(scratch="0"),
            lambda x: values(x)[d.FACTOR]["values"].update(scratch="0"),
            lambda x: values(x)[d.ANCHOR]["values"].update(scratch="0"),
            lambda x: values(x)["hidden.actual_residual_boundary"].update(spread="0"),
            lambda x: x["report"]["mismatch_rows"].__setitem__(0, x["report"]["mismatch_rows"][1]),
            lambda x: x["report"]["partitions"][0]["closure_checks"][0].update(integrity_status="REJECTED"),
            lambda x: x["report"]["reversed_pair"]["checks"][0].update(actual="0"),
            lambda x: x["report"]["partitions"][1]["coordinates"][0]["field_spreads"]
            ["weighted.actual_residual_boundary"]["values"].update(scratch="0"),
            lambda x: x["report"]["partitions"][2]["coordinates"][0]["field_spreads"]
            ["hidden.input_hidden"]["values"].update(scratch="0"),
        ]
        for change in changes:
            changed = deepcopy(self.retained)
            change(changed)
            result = d.report(changed)
            with self.subTest(change=change):
                self.assertEqual(result["cancellation_topology_status"], "REJECTED")
                self.assertGreater(result["failure_count"], 0)

    def test_each_primary_and_capture_member_pin_rejects_byte_tampering(self):
        bindings = [*d.PINS.values(), *d.SOURCE_PINS, *d.SNAPSHOT_PINS]
        bindings.extend(p for r in self.capture["results"] for p in r["files"])
        unique = {p["path"]: p for p in bindings}
        original_read = Path.read_bytes
        for binding in unique.values():
            def changed_read(path):
                data = original_read(path)
                return data + b" " if str(path) == binding["path"] else data

            with self.subTest(path=binding["path"]), patch.object(Path, "read_bytes", changed_read):
                with self.assertRaisesRegex(ValueError, "retained byte count changed"):
                    d.authenticate()
        bad = {**d.PINS["stdout"], "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "retained hash changed"):
            d.bound_bytes(bad)

    def test_review_identity_and_capture_splices_fail_even_with_mocked_byte_gate(self):
        mutations = [
            ("review", lambda x: x.update(producer_role="engineer")),
            ("review", lambda x: x.update(mission_id="other")),
            ("review", lambda x: x["review"].update(status="continue")),
            ("capture", lambda x: x.update(success=False)),
            ("capture", lambda x: x["checks"].update(zero_stderr=False)),
            ("capture", lambda x: x["preflight"].update(uid=0)),
            ("capture", lambda x: x["preflight"]["environment"].update(PYTHONDONTWRITEBYTECODE="0")),
            ("capture", lambda x: x["results"][-1].update(exit_status=1)),
            ("capture", lambda x: x["results"][-1].update(timed_out=True)),
            ("capture", lambda x: x["results"][-1].update(argv=[d.PYTHON, "-c", "pass"])),
            ("capture", lambda x: x["results"][-1]["files"][0].update(path="/foreign")),
        ]
        original = d.bound_bytes
        for kind, mutate in mutations:
            changed = deepcopy(self.review if kind == "review" else self.capture)
            mutate(changed)

            def mocked(binding):
                return json.dumps(changed).encode() if binding == d.PINS[kind] else original(binding)

            with self.subTest(kind=kind, mutate=mutate), patch.object(d, "bound_bytes", mocked):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_read_only_guard_blocks_unbound_reads_writes_imports_execution_and_dispatch(self):
        operations = [
            lambda: Path("/unbound-retained-input").read_bytes(),
            lambda: d.SOURCE.open("wb"),
            lambda: sys.audit("import", "forbidden_operator", None, None, None, None),
            lambda: sys.audit("exec", None),
            lambda: sys.audit("subprocess.Popen", "forbidden", [], None, None),
            lambda: sys.audit("socket.connect", None, None),
            lambda: sys.audit("os.remove", "/forbidden", -1),
            lambda: sys.audit("os.mkdir", "/forbidden", 0, -1),
        ]
        for operation in operations:
            audit = {"forbidden_calls": 0}
            with self.subTest(operation=operation):
                with d.read_only(audit), self.assertRaises(RuntimeError):
                    operation()
                self.assertEqual(audit, {"forbidden_calls": 1})

    def test_no_producer_imports_or_float_arithmetic_and_stdout_only_main(self):
        source = d.SOURCE.read_bytes()
        tree = ast.parse(source)
        allowed = {"argparse", "contextlib", "fractions", "hashlib", "json", "os", "pathlib", "sys"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name in allowed for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Constant):
                self.assertFalse(isinstance(node.value, float))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("float", "eval", "exec", "__import__"))
        stdout, stderr = io.StringIO(), io.StringIO()
        payload = {"report": self.result}
        with patch.object(d, "check", return_value=payload) as check:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                d.main(["--check"])
        check.assert_called_once_with()
        self.assertEqual(d.decode(stdout.getvalue()), payload)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().count("\n"), 1)
        for binding in (*d.PINS.values(), *d.SOURCE_PINS, *d.SNAPSHOT_PINS):
            data = Path(binding["path"]).read_bytes()
            self.assertEqual((len(data), hashlib.sha256(data).hexdigest()),
                             (binding["bytes"], binding["sha256"]))


if __name__ == "__main__":
    unittest.main()
