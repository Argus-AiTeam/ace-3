"""Independent integer-grid oracle for the three retained exception archetypes."""

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

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_archetype_classifier_v1 as d


def ratio(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, q):
    if q < 0:
        n, q = -n, -q
    factor = gcd(n, q)
    n, q = n // factor, q // factor
    return str(n) if q == 1 else f"{n}/{q}"


def subtract(left, right):
    a, b = ratio(left)
    c, e = ratio(right)
    return exact(a * e - c * b, b * e)


def polarity(text):
    n, _ = ratio(text)
    return (n > 0) - (n < 0)


def oracle_comparison(mapped, modal):
    a, b = ratio(mapped)
    c, e = ratio(modal)
    delta = subtract(mapped, modal)
    return {
        "modal_class": modal, "mapped_all": mapped, "delta": delta,
        "signs": {"modal_class": polarity(modal), "mapped_all": polarity(mapped), "delta": polarity(delta)},
        "ratio": {"status": "EXACT", "value": exact(a * e, b * c)} if c else
                 {"status": "UNDEFINED_ZERO_MODAL_DENOMINATOR", "value": None},
    }


class ArchetypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def grid(self, partition, control):
        source = self.retained["retained_0146"]["retained_6184"]["retained_e795"]["retained_449b"]
        account = next(c for c in source["report"]["controls"] if c["control"] == control)
        pair = next(p for p in account["pairs"] if (p["left_id"], p["right_id"]) ==
                    (partition["left_id"], partition["right_id"]))
        terms = {}
        for row in pair["branches"]["binary64"]["selected_coordinates"]:
            if partition["table"] == "coordinate":
                terms[(row["coordinate"], None)] = ratio(row["direct_hidden_weighted_term"])
            else:
                for component, value in row["weighted_components"].items():
                    terms[(row["coordinate"], component)] = ratio(value)
        denominator = lcm(*(q for _, q in terms.values()))
        grid = {key: n * (denominator // q) for key, (n, q) in terms.items()}
        magnitudes = sorted((abs(v) for v in grid.values()), reverse=True)
        candidates = (("exception", (62, None)), ("modal", (241, None)))
        if partition["table"] == "coordinate_component":
            candidates = (("exception", (241, "fp16_to_branch_terminal_remainder")),
                          ("modal", (241, "mlp_stage17")))

        def identity(key):
            return {"coordinate": key[0], **({"component": key[1]} if key[1] is not None else {})}

        fields, ids = {}, {}
        for label, key in candidates:
            value = grid[key]
            ids[label] = identity(key)
            fields[label + ".signed"] = exact(value, denominator)
            fields[label + ".absolute"] = exact(abs(value), denominator)
            fields[label + ".gap_from_maximum"] = exact(magnitudes[0] - abs(value), denominator)
            fields[label + ".rank"] = str(1 + sum(abs(v) > abs(value) for v in grid.values()))
        for field in ("signed", "absolute"):
            fields[field + "_comparator"] = subtract(fields["exception." + field], fields["modal." + field])
        fields["rank_difference"] = subtract(fields["exception.rank"], fields["modal.rank"])
        fields["dominance_gap"] = exact(magnitudes[0] - magnitudes[1], denominator)
        winners = [key for key, value in grid.items() if abs(value) == magnitudes[0]]
        self.assertEqual(len(winners), 1)
        ids["observed_winner"] = identity(winners[0])
        ids["runner_up_ids"] = [identity(key) for key, value in grid.items() if abs(value) == magnitudes[1]]
        ties = {
            "candidate_absolute_tie": fields["exception.absolute"] == fields["modal.absolute"],
            "candidate_signed_tie": fields["exception.signed"] == fields["modal.signed"],
            "candidate_rank_tie": fields["exception.rank"] == fields["modal.rank"],
            "zero_candidate_ids": [identity(k) for _, k in candidates if grid[k] == 0],
            "maximum_candidate_ids": [identity(k) for _, k in candidates if abs(grid[k]) == magnitudes[0]],
            "runner_up_tied": sum(abs(v) == magnitudes[1] for v in grid.values()) > 1,
        }
        return fields, ids, ties

    def test_all_fields_contrasts_memberships_against_deeper_integer_grid(self):
        self.assertEqual(len(self.result["partitions"]), 3)
        for partition in self.result["partitions"]:
            modal, modal_ids, modal_ties = self.grid(partition, "frozen_inherited")
            mapped, mapped_ids, mapped_ties = self.grid(partition, "mapped_all")
            for control in ("frozen_inherited_o", "frozen_inherited_down", "frozen_inherited_o_down",
                            "scratch", "scratch_down", "mapped62", "inherited_native"):
                self.assertEqual(self.grid(partition, control), (modal, modal_ids, modal_ties))
            self.assertEqual(partition["modal_class"]["count"], 8)
            self.assertEqual(partition["exception_class"], {"controls": ["mapped_all"], "count": 1})
            expected = {k: oracle_comparison(mapped[k], modal[k]) for k in modal}
            self.assertEqual(partition["field_comparisons"], expected)
            contrast = {}
            for candidate in ("exception", "modal"):
                for field in ("signed", "absolute", "rank"):
                    contrast[candidate + "_" + field + "_delta"] = expected[candidate + "." + field]["delta"]
            for field in ("absolute_comparator", "signed_comparator", "dominance_gap"):
                contrast[field + "_delta"] = expected[field]["delta"]
            self.assertEqual(partition["contrast_vector"], contrast)
            self.assertEqual(partition["identity_membership"], {"modal_class": modal_ids, "mapped_all": mapped_ids})
            self.assertEqual(partition["tie_membership"], {"modal_class": modal_ties, "mapped_all": mapped_ties})
            signature = partition["categorical_signature"]
            delta_signs = [expected[c + ".absolute"]["signs"]["delta"] for c in ("exception", "modal")]
            self.assertEqual(signature["candidate_absolute_delta_signs"], delta_signs)
            self.assertEqual(signature["magnitude_archetype"],
                             "BOTH_CANDIDATES_CONTRACT" if delta_signs == [-1, -1]
                             else "EXCEPTION_FIXED_MODAL_CONTRACTS")
            self.assertEqual(signature["candidate_signed_signs"], {
                side: [polarity(f[c + ".signed"]) for c in ("exception", "modal")]
                for side, f in (("modal_class", modal), ("mapped_all", mapped))})
            self.assertEqual(signature["comparator_signs"],
                             {f: expected[f]["signs"] for f in ("signed_comparator", "absolute_comparator")})
            self.assertEqual(signature["rank_transition"], {
                k: [modal[k], mapped[k]] for k in ("exception.rank", "modal.rank", "rank_difference")})
            self.assertEqual(signature["dominance_gap_delta_sign"], polarity(expected["dominance_gap"]["delta"]))
            self.assertEqual(signature["winner_changed"], modal_ids["observed_winner"] != mapped_ids["observed_winner"])
            self.assertEqual(signature["winner_runner_up_exchange"],
                             modal_ids["runner_up_ids"] == [mapped_ids["observed_winner"]]
                             and mapped_ids["runner_up_ids"] == [modal_ids["observed_winner"]])
            for side, ids, ties in (("modal_class", modal_ids, modal_ties), ("mapped_all", mapped_ids, mapped_ties)):
                self.assertEqual(signature["ties"][side], {k: v for k, v in ties.items() if not k.endswith("_ids")})
                self.assertEqual(signature["candidate_membership"][side], {
                    "maximum": [c for c in ("exception", "modal") if ids[c] in ties["maximum_candidate_ids"]],
                    "zero": [c for c in ("exception", "modal") if ids[c] in ties["zero_candidate_ids"]],
                    "runner_up": [c for c in ("exception", "modal") if ids[c] in ids["runner_up_ids"]],
                })

    def test_cross_partition_signatures_and_orientation(self):
        cross = self.result["cross_partition"]
        partitions = self.result["partitions"]
        self.assertEqual([len(g["partitions"]) for g in cross["magnitude_archetype_groups"]], [2, 1])
        self.assertEqual(len(cross["categorical_classes"]), 3)
        self.assertEqual(len(cross["pair_comparisons"]), 3)
        for pair, (left, right) in zip(cross["pair_comparisons"], ((0, 1), (0, 2), (1, 2)), strict=True):
            a, b = partitions[left], partitions[right]
            x, y = a["categorical_signature"], b["categorical_signature"]
            self.assertEqual(pair["equal_categories"], [k for k in x if x[k] == y[k]])
            self.assertEqual(pair["different_categories"], [k for k in x if x[k] != y[k]])
            self.assertEqual(pair["absolute_rank_gap_fields_equal"], (left, right) == (0, 1))
            self.assertEqual(pair["signed_fields_are_exact_negatives"], (left, right) == (0, 1))
            self.assertEqual(pair["identity_membership_equal"], a["identity_membership"] == b["identity_membership"])
        self.assertEqual(cross["shared_categories"], {
            k: v for k, v in partitions[0]["categorical_signature"].items()
            if all(p["categorical_signature"][k] == v for p in partitions[1:])})
        for group, partition in zip(cross["categorical_classes"], partitions, strict=True):
            self.assertEqual(group["signature"], partition["categorical_signature"])
            self.assertEqual(group["partitions"], [{k: partition[k] for k in ("table", "left_id", "right_id", "branch")}])

    def test_exact_ratio_zero_and_large_integer_cases(self):
        for mapped, modal in (("0", "0"), ("1", "0"), ("0", "-3/7"), ("-3/7", "2/9"),
                              ("9007199254740993/7", "9007199254740992/7")):
            self.assertEqual(d.field_comparison(mapped, modal), oracle_comparison(mapped, modal))
        for bad in (1, True, "2/2", "0.5", "01", "1/-2"):
            with self.assertRaises(ValueError):
                d.rational(bad)
        for bad in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                d.decode(bad)

    def test_ties_zero_and_membership_are_not_silently_dropped(self):
        fields = {"exception.absolute": "0", "modal.absolute": "0",
                  "exception.signed": "0", "modal.signed": "0",
                  "exception.rank": "1", "modal.rank": "1",
                  "exception.gap_from_maximum": "0", "modal.gap_from_maximum": "0"}
        ids = {"exception": {"coordinate": 62}, "modal": {"coordinate": 241},
               "runner_up_ids": [{"coordinate": 1}, {"coordinate": 2}]}
        ties = d.memberships(fields, ids)
        self.assertTrue(all(ties[k] for k in ("candidate_absolute_tie", "candidate_signed_tie",
                                             "candidate_rank_tie", "runner_up_tied")))
        self.assertEqual(ties["zero_candidate_ids"], [ids["exception"], ids["modal"]])
        self.assertEqual(ties["maximum_candidate_ids"], ties["zero_candidate_ids"])

    def test_scope_class_and_contrast_mutations_fail_closed(self):
        eq = self.retained["report"]["partitions"][0]
        raw = self.retained["retained_0146"]["report"]["partitions"][0]
        for key, value in (("equivalence_classes", [["frozen_inherited"]]),
                           ("modal_control_count", 7), ("classification", "REJECTED"),
                           ("left_id", 13)):
            changed = deepcopy(eq)
            changed[key] = value
            with self.assertRaises(ValueError):
                d.classify_partition(changed, raw)
        changed = deepcopy(eq)
        changed["control_profiles"][1]["fields"]["comparator.dominance_gap"] = "0"
        with self.assertRaises(ValueError):
            d.classify_partition(changed, raw)
        for mutation in ("control", "contrast", "identity"):
            changed = deepcopy(raw)
            if mutation == "control":
                changed["control_membership"]["exception"].append("scratch")
            elif mutation == "contrast":
                changed["mapped_all_vs_modal_contrast_rows"][0]["dominance_gap_delta"] = "0"
            else:
                changed["exception_winner_id"] = {"coordinate": 63}
            with self.assertRaises(ValueError):
                d.classify_partition(eq, changed)

    def test_nested_pins_counters_and_source_binding(self):
        for depth in range(5):
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
        changed = deepcopy(self.retained)
        changed["compiled_sources"][0]["bytes"] += 1
        with self.assertRaises(ValueError):
            d.validate_retained(changed)

    def test_authentication_rejects_bytes_and_capture_or_review_tampering(self):
        with patch.object(Path, "read_bytes", return_value=b"x"):
            with self.assertRaises(ValueError):
                d.bound_bytes(d.PINS["stdout"])
        real = d.bound_bytes
        for label in ("capture", "review"):
            modified = deepcopy(self.capture if label == "capture" else self.review)
            if label == "capture":
                modified["results"][-1]["exit_status"] = 1
            else:
                modified["review"]["status"] = "pending"

            def altered(binding):
                return json.dumps(modified).encode() if binding == d.PINS[label] else real(binding)

            with patch.object(d, "bound_bytes", side_effect=altered):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_read_only_guard_blocks_dispatch_and_writes(self):
        for event, args in (("open", (str(d.TEST), "w", 1)),
                            ("open", (str(d.ROOT / "unknown"), "r", 0)),
                            ("import", ("ace3.model.projection_oracle", None, [], [], [])),
                            ("subprocess.Popen", ("anything", [], None, None)),
                            ("os.mkdir", ("anything", 511, -1))):
            audit = {"forbidden_calls": 0}
            with d.read_only(audit):
                with self.assertRaises(RuntimeError):
                    sys.audit(event, *args)
            self.assertEqual(audit, {"forbidden_calls": 1})

    def test_preservation_no_float_or_producer_imports(self):
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertTrue(self.result["descriptive_accounting_only"])
        self.assertEqual(self.result["reference_scope"], self.retained["report"]["reference_scope"])
        self.assertEqual(self.result["lineage_separation"], self.retained["report"]["lineage_separation"])

        def no_float(value):
            self.assertNotIsInstance(value, float)
            if isinstance(value, dict):
                for child in value.values():
                    no_float(child)
            elif isinstance(value, list):
                for child in value:
                    no_float(child)

        no_float(self.result)
        tree = ast.parse(d.SOURCE.read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                self.assertNotIsInstance(node.value, float)
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name.split(".")[0] in sys.stdlib_module_names for n in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("float", "eval", "exec", "__import__"))

    def test_cli_emits_one_document_without_running_check_again(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.result}), redirect_stdout(output):
            d.main(["--check"])
        text = output.getvalue()
        value, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(text[end:], "\n")
        self.assertEqual(value, {"report": self.result})


if __name__ == "__main__":
    unittest.main()
