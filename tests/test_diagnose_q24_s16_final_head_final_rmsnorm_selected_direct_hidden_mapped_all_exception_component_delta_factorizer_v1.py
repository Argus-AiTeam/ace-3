"""Independent integer-rational oracle for retained component/source movements."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from math import gcd, lcm
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_component_delta_factorizer_v1 as d


def ratio(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, q):
    if q < 0:
        n, q = -n, -q
    factor = gcd(n, q)
    n, q = n // factor, q // factor
    return str(n) if q == 1 else f"{n}/{q}"


def add(*values):
    pairs = [ratio(v) for v in values]
    denominator = lcm(*(q for _, q in pairs))
    return exact(sum(n * (denominator // q) for n, q in pairs), denominator)


def negate(value):
    n, q = ratio(value)
    return exact(-n, q)


def subtract(left, right):
    return add(left, negate(right))


def multiply(left, right):
    a, b = ratio(left)
    c, e = ratio(right)
    return exact(a * c, b * e)


def move(modal, mapped):
    return {"modal_class": modal, "mapped_all": mapped, "delta": subtract(mapped, modal)}


def oracle_classes(deltas):
    denominator = lcm(*(ratio(v)[1] for v in deltas.values()))
    grid = {k: ratio(v)[0] * (denominator // ratio(v)[1]) for k, v in deltas.items()}
    net = sum(grid.values())
    maximum = max(abs(v) for v in grid.values())
    return {
        "signed_sum": exact(net, denominator),
        "absolute_mass": exact(sum(abs(v) for v in grid.values()), denominator),
        "dominant": [k for k, v in grid.items() if v != 0 and abs(v) == maximum],
        "aligned": [k for k, v in grid.items() if v * net > 0],
        "opposing": [k for k, v in grid.items() if v * net < 0],
        "zero": [k for k, v in grid.items() if v == 0],
        "cancelling_nonzero": [k for k, v in grid.items() if v != 0 and net == 0],
        "net_zero": net == 0,
    }


class ComponentDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)
        cls.hotspots = cls.retained
        for child, _ in d.NESTED_PINS[:-1]:
            cls.hotspots = cls.hotspots[child]
        cls.source = cls.hotspots["retained_50f"]

    def row(self, partition, control, coordinate):
        controls = self.source["report"]["controls"]
        account = [c for c in controls if c["control"] == control]
        self.assertEqual(len(account), 1)
        pairs = [p for p in account[0]["pairs"] if (p["left_id"], p["right_id"]) ==
                 (partition["left_id"], partition["right_id"])]
        self.assertEqual(len(pairs), 1)
        rows = [r for r in pairs[0]["branches"]["binary64"]["selected_coordinates"]
                if r["coordinate"] == coordinate]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_all_source_weight_and_seven_component_accounts_against_integer_oracle(self):
        count = 0
        self.assertEqual([tuple(p[k] for k in d.SCOPE_FIELDS) for p in self.result["partitions"]],
                         list(d.SCOPES))
        for partition in self.result["partitions"]:
            for account in partition["coordinates"]:
                modal, mapped = [self.row(partition, c, account["coordinate"])
                                 for c in ("frozen_inherited", "mapped_all")]
                weights = [r["weight_times_reference_anchor_times_row_difference"] for r in (modal, mapped)]
                self.assertEqual(account["weight"], move(*weights))
                self.assertEqual([c["component"] for c in account["components"]], list(d.COMPONENTS))
                for component in account["components"]:
                    name = component["component"]
                    source = move(modal["hidden_components"][name], mapped["hidden_components"][name])
                    weighted = move(*(multiply(r["hidden_components"][name], w)
                                      for r, w in zip((modal, mapped), weights, strict=True)))
                    self.assertEqual(weighted, move(modal["weighted_components"][name], mapped["weighted_components"][name]))
                    self.assertEqual(component["source"], source)
                    self.assertEqual(component["weighted"], weighted)
                    dw = subtract(weights[1], weights[0])
                    terms = {
                        "modal_weight_times_source_delta": multiply(weights[0], source["delta"]),
                        "modal_source_times_weight_delta": multiply(source["modal_class"], dw),
                        "source_delta_times_weight_delta": multiply(source["delta"], dw),
                    }
                    self.assertEqual(component["product_delta_terms"], terms)
                    self.assertEqual(add(*terms.values()), weighted["delta"])
                    self.assertTrue(component["exact_product_delta_closure"])
                    count += 1
                self.assertEqual(account["signed"], move(modal["direct_hidden_weighted_term"],
                                                          mapped["direct_hidden_weighted_term"]))
                for side in d.SIDES:
                    self.assertEqual(add(*(c["weighted"][side] for c in account["components"])),
                                     account["signed"][side])
                for label, field in (("component_delta_classes", "weighted"), ("source_delta_classes", "source")):
                    self.assertEqual(account[label], oracle_classes({c["component"]: c[field]["delta"]
                                                                   for c in account["components"]}))
                self.assertTrue(account["exact_seven_component_closure"])
        self.assertEqual(count, 35)
        self.assertEqual(self.result["coordinate_account_count"], 5)
        self.assertEqual(self.result["weighted_component_account_count"], count)

    def test_parent_candidate_comparator_and_selected_context_closures(self):
        for parent, partition in zip(self.retained["report"]["partitions"], self.result["partitions"], strict=True):
            expected_components = {name: {} for name in d.COMPONENTS}
            for side, control in (("modal_class", "frozen_inherited"), ("mapped_all", "mapped_all")):
                values = {}
                for label, candidate in partition["candidates"].items():
                    identity = candidate["identity"]
                    row = self.row(partition, control, identity["coordinate"])
                    values[label] = (row["direct_hidden_weighted_term"] if partition["table"] == "coordinate"
                                     else row["weighted_components"][identity["component"]])
                    self.assertEqual(candidate["signed"][side], values[label])
                    self.assertTrue(candidate["exact_parent_closure"])
                self.assertEqual(partition["signed_comparator"][side], subtract(values["exception"], values["modal"]))
                for name in d.COMPONENTS:
                    terms = {}
                    for label, candidate in partition["candidates"].items():
                        identity = candidate["identity"]
                        terms[label] = (self.row(partition, control, identity["coordinate"])["weighted_components"][name]
                                        if partition["table"] == "coordinate" or name == identity["component"] else "0")
                    expected_components[name][side] = subtract(terms["exception"], terms["modal"])
            for name, values in expected_components.items():
                values["delta"] = subtract(values["mapped_all"], values["modal_class"])
            self.assertEqual(partition["comparator_components"], expected_components)
            for label, candidate in partition["candidates"].items():
                self.assertEqual(candidate["signed"], {k: parent["field_comparisons"][label + ".signed"][k] for k in d.SIDES})
            for side in d.SIDES:
                self.assertEqual(add(*(v[side] for v in expected_components.values())), partition["signed_comparator"][side])
                self.assertEqual(partition["signed_comparator"][side], parent["field_comparisons"]["signed_comparator"][side])
            self.assertEqual(partition["comparator_delta_classes"],
                             oracle_classes({k: v["delta"] for k, v in expected_components.items()}))
            context = partition["selected_component_context"]
            if partition["table"] == "coordinate":
                self.assertIsNone(context)
                continue
            self.assertEqual(context["selected_components"], ["fp16_to_branch_terminal_remainder", "mlp_stage17"])
            self.assertEqual(context["remaining_components"], [c for c in d.COMPONENTS if c not in context["selected_components"]])
            self.assertEqual(len(context["remaining_components"]), 5)
            for field, names in (("selected_sum", context["selected_components"]), ("context_sum", context["remaining_components"])):
                sums = [add(*(self.row(partition, control, 241)["weighted_components"][c] for c in names))
                        for control in ("frozen_inherited", "mapped_all")]
                self.assertEqual(context[field], move(*sums))
            for side in d.SIDES:
                self.assertEqual(add(context["selected_sum"][side], context["context_sum"][side]),
                                 context["coordinate_signed"][side])
            self.assertTrue(context["exact_selected_context_closure"])

    def test_reversed_orientation_and_nonlinear_boundary(self):
        forward, reverse = self.result["partitions"][:2]
        self.assertEqual(self.result["orientation_symmetry"]["component_signed_orientation_checks"], 42)
        for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
            for a, b in zip(left["components"], right["components"], strict=True):
                self.assertEqual(a["source"], b["source"])
                for side in d.SIDES:
                    self.assertEqual(a["weighted"][side], negate(b["weighted"][side]))
        self.assertEqual(forward["nonlinear_parent_fields"], reverse["nonlinear_parent_fields"])
        self.assertIn("not abs(mapped-modal)", self.result["nonlinear_boundary"])
        self.assertIn("not a causal source", self.result["nonlinear_boundary"])
        changed = deepcopy(reverse)
        changed["coordinates"][0]["components"][0]["weighted"]["delta"] = "1"
        with self.assertRaises(ValueError):
            d.orientation(forward, changed)

    def test_class_ties_opposition_zeros_and_cancellation(self):
        for values in ({"a": "3/2", "b": "-1/2", "c": "0"},
                       {"a": "2", "b": "-2", "c": "0"},
                       {"a": "0", "b": "0"},
                       {"a": "-3", "b": "-3", "c": "1"}):
            self.assertEqual(d.classes(values), oracle_classes(values))
        self.assertEqual(d.classes({"a": "0"})["dominant"], [])

    def test_product_difference_with_both_factors_changed(self):
        modal = deepcopy(self.row(self.result["partitions"][0], "frozen_inherited", 62))
        mapped = deepcopy(modal)
        for row, weight, source in ((modal, "2/3", "-4/5"), (mapped, "-7/11", "13/17")):
            row["weight_times_reference_anchor_times_row_difference"] = weight
            row["hidden_components"] = {c: source for c in d.COMPONENTS}
            row["weighted_components"] = {c: multiply(weight, source) for c in d.COMPONENTS}
            row["direct_hidden_weighted_term"] = add(*row["weighted_components"].values())
        result = d.coordinate_account(modal, mapped)
        terms = result["components"][0]["product_delta_terms"]
        self.assertTrue(all(v != "0" for v in terms.values()))
        self.assertEqual(add(*terms.values()), subtract(multiply("-7/11", "13/17"), multiply("2/3", "-4/5")))

    def test_rejects_component_source_weight_and_parent_tampering(self):
        partition = self.retained["report"]["partitions"][0]
        modal = self.row(partition, "frozen_inherited", 62)
        mapped = self.row(partition, "mapped_all", 62)
        for field in ("hidden_components", "weighted_components"):
            changed = deepcopy(mapped)
            changed[field]["input_hidden"] = "999"
            with self.assertRaises(ValueError):
                d.coordinate_account(modal, changed)
            changed = deepcopy(mapped)
            del changed[field]["input_hidden"]
            with self.assertRaises(ValueError):
                d.coordinate_account(modal, changed)
        for field in ("weight_times_reference_anchor_times_row_difference", "direct_hidden_weighted_term"):
            changed = deepcopy(mapped)
            changed[field] = "999"
            with self.assertRaises(ValueError):
                d.coordinate_account(modal, changed)
        for field in ("exception.signed", "modal.signed", "signed_comparator"):
            changed = deepcopy(partition)
            changed["field_comparisons"][field]["delta"] = "999"
            with self.assertRaises(ValueError):
                d.factor_partition(changed, self.hotspots, self.source)
        changed = deepcopy(partition)
        changed["modal_class"]["canonical_control"] = "scratch"
        with self.assertRaises(ValueError):
            d.factor_partition(changed, self.hotspots, self.source)
        changed = deepcopy(self.hotspots)
        control = next(c for c in changed["report"]["controls"] if c["control"] == "frozen_inherited")
        pair = next(p for p in control["pairs"] if (p["left_id"], p["right_id"]) == (34319, 319))
        pair["branches"]["binary64"]["selected_coordinates"] *= 2
        with self.assertRaises(ValueError):
            d.factor_partition(partition, changed, self.source)

    def test_nested_pins_scope_and_source_census_rejected_when_changed(self):
        for depth in range(6):
            changed = deepcopy(self.retained)
            node = changed
            for child, _ in d.NESTED_PINS[:depth]:
                node = node[child]
            node["input_pins"]["stdout"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                d.validate_retained(changed)
        for counters in ({}, {"operator": 1}, {"operator": True}, {"operator": "0"}):
            with self.assertRaises(ValueError):
                d.zero_counters(counters)
        for field in ("compiled_sources", "report"):
            changed = deepcopy(self.retained)
            if field == "compiled_sources":
                changed[field][0]["bytes"] += 1
            else:
                changed[field]["partitions"].append(deepcopy(changed[field]["partitions"][0]))
            with self.assertRaises(ValueError):
                d.validate_retained(changed)

    def test_authentication_rejects_byte_capture_and_review_splices(self):
        for binding in [*d.PINS.values(), *d.SOURCE_PINS]:
            with patch.object(Path, "read_bytes", return_value=b"x"):
                with self.assertRaises(ValueError):
                    d.bound_bytes(binding)
        real = d.bound_bytes
        for label, field in (("review", "status"), ("review", "mission"), ("capture", "exit"), ("capture", "overwrite")):
            modified = deepcopy(self.review if label == "review" else self.capture)
            if field == "status":
                modified["review"]["status"] = "pending"
            elif field == "mission":
                modified["mission_id"] = "wrong"
            elif field == "exit":
                modified["results"][-1]["exit_status"] = 1
            else:
                modified["accepted_artifacts_after"] = []

            def altered(binding):
                return json.dumps(modified).encode() if binding == d.PINS[label] else real(binding)

            with patch.object(d, "bound_bytes", side_effect=altered):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_read_only_guard_blocks_forbidden_operations(self):
        events = (
            ("open", (str(d.TEST), "w", 1)),
            ("open", (d.NESTED_PINS[0][1]["stdout"]["path"], "r", 0)),
            ("import", ("ace3.model.projection_oracle", None, [], [], [])),
            ("subprocess.Popen", ("anything", [], None, None)),
            ("socket.connect", (None, ("localhost", 1))),
            ("os.mkdir", ("anything", 511, -1)),
        )
        for event, args in events:
            audit = {"forbidden_calls": 0}
            with d.read_only(audit):
                with self.assertRaises(RuntimeError):
                    sys.audit(event, *args)
            self.assertEqual(audit, {"forbidden_calls": 1})

    def test_no_float_preservation_and_stdout_only_cli(self):
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        for field in ("reference_scope", "lineage_separation"):
            self.assertEqual(self.result[field], self.retained["report"][field])

        def no_float(value):
            self.assertNotIsInstance(value, float)
            if isinstance(value, dict):
                for child in value.values():
                    no_float(child)
            elif isinstance(value, list):
                for child in value:
                    no_float(child)

        no_float(self.result)
        for path in (d.SOURCE, d.TEST):
            tree = ast.parse(path.read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    self.assertNotIsInstance(node.value, float)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("float", "eval", "exec", "__import__"))
                if path == d.SOURCE and isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("ace3", "numpy", "torch")))
        with patch.object(d, "check", return_value={"report": self.result}):
            output = io.StringIO()
            with redirect_stdout(output):
                d.main(["--check"])
            self.assertEqual(json.loads(output.getvalue()), {"report": self.result})
            self.assertEqual(output.getvalue().count("\n"), 1)

    def test_rejects_noncanonical_rationals_and_json(self):
        for value in (1, True, "1.0", "2/2", "NaN", "1/0"):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                d.rational(value)
        for value in ('{"x":1,"x":2}', '{"x":NaN}', "{} {}"):
            with self.assertRaises(ValueError):
                d.decode(value)


if __name__ == "__main__":
    unittest.main()
