"""Independent integer-pair oracle for the retained source-only product collapse."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_source_only_product_term_collapse_classifier_v1 as d


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


class SourceOnlyCollapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def test_all_35_products_and_five_weights_against_independent_integer_oracle(self):
        counts = {d.SOURCE_ONLY: 0, d.ZERO_MOVEMENT: 0}
        coordinates = zero_terms = 0
        for parent, partition in zip(self.retained["report"]["partitions"], self.result["partitions"], strict=True):
            self.assertEqual(tuple(partition[k] for k in d.SCOPE_FIELDS),
                             tuple(parent[k] for k in d.SCOPE_FIELDS))
            for original, account in zip(parent["coordinates"], partition["coordinates"], strict=True):
                coordinates += 1
                weight = account["weight"]
                self.assertEqual(subtract(weight["mapped_all"], weight["modal_class"]), "0")
                self.assertEqual(weight["delta"], "0")
                self.assertTrue(account["exact_zero_weight_delta"])
                for old, component in zip(original["components"], account["components"], strict=True):
                    self.assertEqual({k: component[k] for k in old}, old)
                    source, weighted = component["source"], component["weighted"]
                    for value in (source, weighted):
                        self.assertEqual(subtract(value["mapped_all"], value["modal_class"]), value["delta"])
                    for side in d.SIDES[:2]:
                        self.assertEqual(multiply(weight[side], source[side]), weighted[side])
                    terms = {
                        "modal_weight_times_source_delta": multiply(weight["modal_class"], source["delta"]),
                        "modal_source_times_weight_delta": multiply(source["modal_class"], weight["delta"]),
                        "source_delta_times_weight_delta": multiply(source["delta"], weight["delta"]),
                    }
                    self.assertEqual(component["product_delta_terms"], terms)
                    self.assertEqual(add(*terms.values()), weighted["delta"])
                    self.assertEqual(terms[d.TERMS[0]], weighted["delta"])
                    self.assertEqual([terms[k] for k in d.TERMS[1:]], ["0", "0"])
                    zero_terms += 2
                    label = d.ZERO_MOVEMENT if ratio(weighted["delta"])[0] == 0 else d.SOURCE_ONLY
                    if label == d.ZERO_MOVEMENT:
                        self.assertEqual(source["delta"], "0")
                    else:
                        self.assertNotEqual(source["delta"], "0")
                    self.assertEqual(component["classification"], label)
                    self.assertTrue(component["exact_source_only_collapse"])
                    counts[label] += 1
                for side in d.SIDES:
                    self.assertEqual(add(*(c["weighted"][side] for c in account["components"])),
                                     account["signed"][side])
        self.assertEqual((coordinates, sum(counts.values()), zero_terms), (5, 35, 70))
        self.assertEqual(counts, {d.SOURCE_ONLY: 20, d.ZERO_MOVEMENT: 15})
        self.assertEqual(self.result["classification_counts"], counts)
        self.assertEqual(self.result["zero_coordinate_weight_delta_count"], 5)
        self.assertEqual(self.result["zero_weight_delta_product_term_count"], 70)
        self.assertEqual([p["classification_counts"] for p in self.result["partitions"]],
                         [{d.SOURCE_ONLY: 8, d.ZERO_MOVEMENT: 6}] * 2
                         + [{d.SOURCE_ONLY: 4, d.ZERO_MOVEMENT: 3}])

    def test_parent_comparator_and_selected_context_integer_closures(self):
        for p, parent in zip(self.result["partitions"], self.retained["retained_667"]["report"]["partitions"], strict=True):
            selected = {}
            for label, candidate in p["candidates"].items():
                identity = candidate["identity"]
                account = next(a for a in p["coordinates"] if a["coordinate"] == identity["coordinate"])
                selected[label] = {c["component"]: c["weighted"] for c in account["components"]
                                   if p["table"] == "coordinate" or c["component"] == identity["component"]}
                for side in d.SIDES:
                    self.assertEqual(add(*(v[side] for v in selected[label].values())), candidate["signed"][side])
                    self.assertEqual(candidate["signed"][side], parent["field_comparisons"][label + ".signed"][side])
            for name in d.COMPONENTS:
                for side in d.SIDES:
                    self.assertEqual(p["comparator_components"][name][side],
                                     subtract(selected["exception"].get(name, {}).get(side, "0"),
                                              selected["modal"].get(name, {}).get(side, "0")))
            for side in d.SIDES:
                self.assertEqual(add(*(v[side] for v in p["comparator_components"].values())),
                                 p["signed_comparator"][side])
                self.assertEqual(p["signed_comparator"][side], parent["field_comparisons"]["signed_comparator"][side])
            context = p["selected_component_context"]
            if context is None:
                self.assertEqual(p["table"], "coordinate")
            else:
                for field, names in (("selected_sum", context["selected_components"]),
                                     ("context_sum", context["remaining_components"])):
                    for side in d.SIDES:
                        self.assertEqual(context[field][side],
                                         add(*(c["weighted"][side] for c in p["coordinates"][0]["components"]
                                               if c["component"] in names)))
                for side in d.SIDES:
                    self.assertEqual(add(context["selected_sum"][side], context["context_sum"][side]),
                                     context["coordinate_signed"][side])

    def test_reversed_signed_orientation_and_preservation(self):
        forward, reverse = self.result["partitions"][:2]
        for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
            for field in ("weight", "signed"):
                for side in d.SIDES:
                    self.assertEqual(left[field][side], negate(right[field][side]))
            for a, b in zip(left["components"], right["components"], strict=True):
                self.assertEqual(a["source"], b["source"])
                self.assertEqual(a["classification"], b["classification"])
                for side in d.SIDES:
                    self.assertEqual(a["weighted"][side], negate(b["weighted"][side]))
                for term in d.TERMS:
                    self.assertEqual(a["product_delta_terms"][term], negate(b["product_delta_terms"][term]))
        for name in d.COMPONENTS:
            for side in d.SIDES:
                self.assertEqual(forward["comparator_components"][name][side],
                                 negate(reverse["comparator_components"][name][side]))
        for side in d.SIDES:
            self.assertEqual(forward["signed_comparator"][side], negate(reverse["signed_comparator"][side]))
        symmetry = self.result["orientation_symmetry"]
        self.assertEqual(symmetry["component_signed_orientation_checks"], 42)
        self.assertEqual(symmetry["product_term_signed_orientation_checks"], 42)
        self.assertEqual(forward["nonlinear_parent_fields"], reverse["nonlinear_parent_fields"])
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        for field in ("reference_scope", "lineage_separation", "delta_rule", "source_rule", "nonlinear_boundary"):
            self.assertEqual(self.result[field], self.retained["report"][field])

    def test_each_product_term_source_weight_and_signed_tamper_is_rejected(self):
        original = self.retained["report"]["partitions"][0]["coordinates"][0]
        paths = [
            ("weight", side) for side in d.SIDES
        ] + [("signed", side) for side in d.SIDES]
        paths += [("components", index, field, side) for index in range(7)
                  for field in ("source", "weighted") for side in d.SIDES]
        paths += [("components", index, "product_delta_terms", term)
                  for index in range(7) for term in d.TERMS]
        for path in paths:
            with self.subTest(path=path):
                changed = deepcopy(original)
                node = changed
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = add(node[path[-1]], "1")
                with self.assertRaises(ValueError):
                    d.classify_coordinate(changed)
        changed = deepcopy(original)
        changed["components"].append(deepcopy(changed["components"][0]))
        with self.assertRaises(ValueError):
            d.classify_coordinate(changed)

    def test_zero_weight_masking_nonzero_source_is_not_zero_source_movement(self):
        changed = deepcopy(self.retained["report"]["partitions"][0]["coordinates"][0])
        changed["weight"] = dict.fromkeys(d.SIDES, "0")
        changed["signed"] = dict.fromkeys(d.SIDES, "0")
        for c in changed["components"]:
            c["weighted"] = dict.fromkeys(d.SIDES, "0")
            c["product_delta_terms"] = dict.fromkeys(d.TERMS, "0")
        with self.assertRaisesRegex(ValueError, "must not mask"):
            d.classify_coordinate(changed)

    def test_parent_and_orientation_tamper_is_rejected(self):
        parent = self.retained["retained_667"]["report"]["partitions"][0]
        original = self.result["partitions"][0]
        paths = [
            ("signed_comparator", "delta"),
            ("comparator_components", "input_hidden", "delta"),
            ("candidates", "exception", "signed", "delta"),
            ("candidates", "modal", "identity", "coordinate"),
            ("modal_class", "canonical_control"),
        ]
        for path in paths:
            changed = deepcopy(original)
            node = changed
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = "999"
            with self.assertRaises(ValueError):
                d.verify_partition(changed, parent)
        changed = deepcopy(self.result["partitions"][2])
        changed["selected_component_context"]["context_sum"]["delta"] = "999"
        with self.assertRaises(ValueError):
            d.verify_partition(changed, self.retained["retained_667"]["report"]["partitions"][2])
        for field, key in (("weighted", "delta"), ("product_delta_terms", d.TERMS[0]),
                           ("source", "delta")):
            changed = deepcopy(self.result["partitions"][1])
            changed["coordinates"][0]["components"][0][field][key] = "999"
            with self.assertRaises(ValueError):
                d.orientation(original, changed)

    def test_authentication_rejects_all_artifact_and_sidecar_bytes(self):
        for binding in [*d.PINS.values(), *d.SOURCE_PINS,
                        *(p for r in self.capture["results"] for p in r["files"])]:
            with patch.object(Path, "read_bytes", return_value=b"x"):
                with self.assertRaises(ValueError):
                    d.bound_bytes(binding)
        real = d.bound_bytes
        for label, field in (("review", "status"), ("review", "mission"), ("capture", "exit"),
                             ("capture", "overwrite"), ("capture", "argv"), ("capture", "source")):
            changed = deepcopy(self.review if label == "review" else self.capture)
            if field == "status":
                changed["review"]["status"] = "pending"
            elif field == "mission":
                changed["mission_id"] = "wrong"
            elif field == "exit":
                changed["results"][-1]["exit_status"] = 1
            elif field == "overwrite":
                changed["accepted_artifacts_after"] = []
            elif field == "argv":
                changed["results"][-1]["argv"][-1] = "--wrong"
            else:
                changed["sources_after"] = []

            def altered(binding):
                return json.dumps(changed).encode() if binding == d.PINS[label] else real(binding)

            with patch.object(d, "bound_bytes", side_effect=altered):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_retained_gates_and_counter_types(self):
        for values in ({}, {"operator": 1}, {"operator": True}, {"operator": "0"}):
            with self.assertRaises(ValueError):
                d.zero_counters(values)
        children = ("retained_667", "retained_3fd", "retained_0146", "retained_6184",
                    "retained_e795", "retained_449b", "retained_50f")
        for depth in range(8):
            changed = deepcopy(self.retained)
            node = changed
            for child in children[:depth]:
                node = node[child]
            node["dispatch_and_write_audit"]["forbidden_calls"] = 1
            with self.assertRaises(ValueError):
                d.validate_retained(changed)
        for field in ("compiled_sources", "version", "report"):
            changed = deepcopy(self.retained)
            if field == "compiled_sources":
                changed[field][0]["bytes"] += 1
            elif field == "version":
                changed[field] = True
            else:
                changed[field]["partitions"].reverse()
            with self.assertRaises(ValueError):
                d.validate_retained(changed)

    def test_read_only_blocks_producers_writes_imports_and_dispatch(self):
        events = (
            ("open", (str(d.SOURCE), "w", 1)),
            ("open", (str(d.ROOT / "build/unauthorized"), "r", 0)),
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

    def test_no_float_and_one_stdout_document(self):
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
            arithmetic = (
                ("ratio", "exact", "add", "negate", "subtract", "multiply")
                if path == d.TEST else
                ("rational", "movement", "classify_coordinate", "verify_partition", "orientation", "report")
            )
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name in arithmetic:
                    for node in ast.walk(function):
                        if isinstance(node, ast.BinOp):
                            self.assertNotIsInstance(node.op, ast.Div)
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
        for value in (1, True, "1.0", "2/2", "NaN", "1/0"):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                d.rational(value)
        for value in ('{"x":1,"x":2}', '{"x":NaN}', "{} {}"):
            with self.assertRaises(ValueError):
                d.decode(value)


if __name__ == "__main__":
    unittest.main()
