"""Independent integer-grid closure and hostile retained-only profiler controls."""

import ast
from copy import deepcopy
import io
import json
from math import gcd, lcm
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_modal_exception_contrast_profiler_v1 as d


def ratio(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(numerator, denominator):
    factor = gcd(numerator, denominator)
    n, q = numerator // factor, denominator // factor
    return str(n) if q == 1 else f"{n}/{q}"


def difference(left, right):
    a, b = ratio(left)
    c, e = ratio(right)
    return exact(a * e - c * b, b * e)


def polarity(text):
    n, _ = ratio(text)
    return (n > 0) - (n < 0)


class ProfilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def oracle(self, profile, control):
        controls = self.retained["retained_e795"]["retained_449b"]["report"]["controls"]
        source = next(c for c in controls if c["control"] == control)
        pair = next(p for p in source["pairs"] if
                    (p["left_id"], p["right_id"]) == (profile["left_id"], profile["right_id"]))
        cells = {}
        for row in pair["branches"][profile["branch"]]["selected_coordinates"]:
            if profile["table"] == "coordinate":
                cells[(row["coordinate"], None)] = ratio(row["direct_hidden_weighted_term"])
            else:
                for component, text in row["weighted_components"].items():
                    cells[(row["coordinate"], component)] = ratio(text)
        denominator = lcm(*(q for _, q in cells.values()))
        grid = {key: n * (denominator // q) for key, (n, q) in cells.items()}
        magnitudes = sorted((abs(n) for n in grid.values()), reverse=True)
        rows = {}
        for key, value in grid.items():
            ident = {"coordinate": key[0]}
            if key[1] is not None:
                ident["component"] = key[1]
            rows[key] = {
                **ident, "signed": exact(value, denominator),
                "absolute": exact(abs(value), denominator),
                "rank": 1 + sum(abs(v) > abs(value) for v in grid.values()),
                "gap_from_maximum": exact(magnitudes[0] - abs(value), denominator),
            }
        winner = next(row for row in rows.values() if row["rank"] == 1)
        return rows, exact(magnitudes[0] - magnitudes[1], denominator), {
            k: winner[k] for k in ("coordinate", "component") if k in winner
        }, pair["roles"]

    def test_exact_domain_membership_and_nested_preservation(self):
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual((self.result["unstable_partition_count"],
                          self.result["same_partition_comparator_row_count"],
                          self.result["mapped_all_vs_modal_contrast_row_count"]), (3, 27, 24))
        self.assertEqual([(p["table"], p["left_id"], p["right_id"], p["branch"])
                          for p in self.result["partitions"]], [
            ("coordinate", 34319, 319, "binary64"),
            ("coordinate", 319, 34319, "binary64"),
            ("coordinate_component", 34319, 13, "binary64"),
        ])
        for p in self.result["partitions"]:
            self.assertEqual(p["control_membership"], {
                "exception": ["mapped_all"],
                "modal": ["frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
                          "frozen_inherited_o_down", "scratch", "scratch_down",
                          "mapped62", "inherited_native"],
            })
        for key in ("reference_scope", "lineage_separation"):
            self.assertEqual(self.result[key], self.retained["report"][key])
        self.assertEqual(self.retained["input_pins"], d.RETAINED_E795_PINS)
        self.assertEqual(self.retained["retained_e795"]["input_pins"], d.RETAINED_449B_PINS)
        self.assertEqual(self.retained["retained_e795"]["retained_449b"]["input_pins"],
                         d.RETAINED_50F_PINS)
        d.require_expected(self.result)

    def test_all_27_comparators_against_independent_integer_grid(self):
        count = 0
        for p in self.result["partitions"]:
            for row in p["same_partition_comparator_rows"]:
                cells, gap, winner, roles = self.oracle(p, row["control"])
                chosen = []
                for label in ("exception", "modal"):
                    ident = p[label + "_winner_id"]
                    expected = cells[(ident["coordinate"], ident.get("component"))]
                    self.assertEqual(row[label + "_winner_row"], expected)
                    chosen.append(expected)
                e, m = chosen
                self.assertEqual(row["dominance_gap"], gap)
                self.assertEqual(row["observed_winner_id"], winner)
                self.assertEqual(row["roles"], roles)
                self.assertEqual((row["left_id"], row["right_id"], row["branch"]),
                                 (p["left_id"], p["right_id"], p["branch"]))
                self.assertEqual(row["membership"],
                                 "exception" if row["control"] == "mapped_all" else "modal")
                for field in ("absolute", "signed"):
                    value = difference(e[field], m[field])
                    self.assertEqual(row[field + "_comparator"], value)
                    self.assertEqual(row[field + "_comparator_sign"], polarity(value))
                self.assertEqual(row["rank_difference"], e["rank"] - m["rank"])
                self.assertEqual(row["absolute_comparator_sign"],
                                 1 if row["control"] == "mapped_all" else -1)
                ranks = sorted({c["rank"] for c in cells.values()})
                self.assertEqual(
                    {(i["coordinate"], i.get("component")) for i in row["runner_up_ids"]},
                    {k for k, c in cells.items() if c["rank"] == ranks[1]})
                count += 1
        self.assertEqual(count, 27)

    def test_all_24_contrasts_against_independent_integer_grid(self):
        count = 0
        for p in self.result["partitions"]:
            mapped, mapped_gap, _, _ = self.oracle(p, "mapped_all")
            for row in p["mapped_all_vs_modal_contrast_rows"]:
                modal, modal_gap, _, _ = self.oracle(p, row["modal_control"])
                values = {}
                selected = {}
                for label in ("exception", "modal"):
                    ident = p[label + "_winner_id"]
                    key = (ident["coordinate"], ident.get("component"))
                    selected[label] = (mapped[key], modal[key])
                    for field in ("signed", "absolute"):
                        name = label + "_" + field + "_delta"
                        values[name] = difference(mapped[key][field], modal[key][field])
                    self.assertEqual(row[label + "_rank_delta"],
                                     mapped[key]["rank"] - modal[key]["rank"])
                for field in ("signed", "absolute"):
                    left = difference(selected["exception"][0][field], selected["modal"][0][field])
                    right = difference(selected["exception"][1][field], selected["modal"][1][field])
                    values[field + "_comparator_delta"] = difference(left, right)
                    self.assertEqual(values[field + "_comparator_delta"],
                                     difference(values["exception_" + field + "_delta"],
                                                values["modal_" + field + "_delta"]))
                values["dominance_gap_delta"] = difference(mapped_gap, modal_gap)
                for key, value in values.items():
                    self.assertEqual(row[key], value)
                    self.assertEqual(row["delta_signs"][key], polarity(value))
                self.assertEqual(row["absolute_comparator_signs"],
                                 {"mapped_all": 1, "modal_control": -1})
                self.assertEqual(row["rank_order_signs"],
                                 {"mapped_all": -1, "modal_control": 1})
                self.assertIs(row["rank_flip"], True)
                self.assertIs(row["winner_changed"], True)
                self.assertEqual(row["exception_control"], "mapped_all")
                count += 1
        self.assertEqual(count, 24)

    def test_reversed_outer_pairs_have_exact_sign_reversal_not_new_samples(self):
        left, right = self.result["partitions"][:2]
        for a, b in zip(left["same_partition_comparator_rows"],
                        right["same_partition_comparator_rows"], strict=True):
            self.assertEqual(a["control"], b["control"])
            for label in ("exception_winner_row", "modal_winner_row"):
                self.assertEqual(a[label]["absolute"], b[label]["absolute"])
                self.assertEqual(a[label]["rank"], b[label]["rank"])
                self.assertEqual(a[label]["signed"], difference("0", b[label]["signed"]))
            self.assertEqual(a["dominance_gap"], b["dominance_gap"])
            self.assertEqual(a["absolute_comparator"], b["absolute_comparator"])
            self.assertEqual(a["signed_comparator"], difference("0", b["signed_comparator"]))

    def test_review_role_mission_and_terminal_status_are_required(self):
        for path, value in (
            (("kind",), "round_engineer_handoff"), (("mission_id",), "e795493d08bc"),
            (("producer_role",), "engineer"), (("review", "status"), "pending"),
        ):
            changed = deepcopy(self.review)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(ValueError, "independent terminal"):
                d.validate_review(changed)

    def test_nested_pin_and_forbidden_flag_splices_fail(self):
        for depth in range(4):
            changed = deepcopy(self.retained)
            node = changed
            for key in ("retained_e795", "retained_449b", "retained_50f")[:depth]:
                node = node[key]
            node["flags"]["candidate_admitted"] = True
            with self.assertRaisesRegex(ValueError, "forbidden"):
                d.report(changed)
        for depth in range(3):
            changed = deepcopy(self.retained)
            node = changed
            for key in ("retained_e795", "retained_449b")[:depth]:
                node = node[key]
            node["input_pins"]["stdout"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "nested pins"):
                d.report(changed)

    def test_capture_failures_and_source_identity_splices_fail(self):
        for fault in ("success", "exit", "source", "accepted", "account", "command", "files"):
            capture = deepcopy(self.capture)
            if fault == "success":
                capture["success"] = False
            elif fault == "exit":
                capture["results"][-1]["exit_status"] = 1
            elif fault == "source":
                capture["sources_after"][0]["sha256"] = "0" * 64
            elif fault == "accepted":
                capture["accepted_artifacts_after"] = []
            elif fault == "account":
                capture["preflight"]["uid"] = 0
            elif fault == "command":
                capture["results"][-1]["command"] += " --replay"
            else:
                capture["results"][-1]["files"][3]["path"] = str(d.SOURCE)
            original = d.bound_bytes

            def altered(pin):
                return json.dumps(capture).encode() if pin == d.PINS["capture"] else original(pin)

            with patch.object(d, "bound_bytes", side_effect=altered):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_exact_hash_and_byte_count_required(self):
        pin = d.PINS["review"]
        data = d.bound_bytes(pin)
        for changed in (data + b" ", b"x" + data[1:]):
            with patch.object(Path, "read_bytes", return_value=changed):
                with self.assertRaises(ValueError):
                    d.bound_bytes(pin)

    def test_partition_membership_ties_missing_and_extra_scope_fail(self):
        for fault in ("mode", "duplicate", "cross", "stable", "extra", "exception"):
            changed = deepcopy(self.retained)
            parts = changed["report"]["tables"]["coordinate"]["partitions"]["by_pair_branch"]
            p = parts[0]
            if fault == "mode":
                p["modal_winner_ids"].append({"coordinate": 62})
            elif fault == "duplicate":
                p["winner_classes"][0]["members"].append(p["winner_classes"][0]["members"][0])
            elif fault == "cross":
                p["winner_classes"][0]["members"][0]["branch"] = "fp16"
            elif fault == "stable":
                p["stable"] = True
            elif fault == "extra":
                parts[1]["stable"] = False
            else:
                p["exceptions"] = []
            with self.assertRaises(ValueError):
                d.report(changed)

    def test_corrupt_ranked_values_identities_and_gap_closure_fail(self):
        source = self.retained["retained_e795"]["report"]["accounts"][0]["tables"]["coordinate"]
        for fault in ("signed", "rank", "gap", "identity", "order", "tie", "zero", "missing"):
            table = deepcopy(source)
            if fault == "signed":
                table["rows"][0]["signed"] = "0"
            elif fault == "rank":
                table["rows"][0]["rank"] = True
            elif fault == "gap":
                table["dominance_gap"] = "0"
            elif fault == "identity":
                table["rows"][1]["coordinate"] = table["rows"][0]["coordinate"]
            elif fault == "order":
                table["rows"].reverse()
            elif fault == "tie":
                table["rows"][1]["absolute"] = table["rows"][0]["absolute"]
            elif fault == "zero":
                for row in table["rows"]:
                    row["absolute"] = "0"
                    row["signed"] = "0"
            else:
                table["rows"] = table["rows"][:1]
            with self.assertRaises(ValueError):
                d.ranked_rows(table, "coordinate")

    def test_selected_row_splice_is_rejected_in_full_report(self):
        changed = deepcopy(self.retained)
        changed["retained_e795"]["report"]["accounts"][0]["tables"]["coordinate"]["rows"][0]["signed"] = "0"
        with self.assertRaisesRegex(ValueError, "signed/absolute"):
            d.report(changed)

    def test_exact_rationals_json_and_no_float_arithmetic(self):
        for value in (0.1, 1, True, "2/4", "0/1", "-0", "1.0", "1e-5"):
            with self.assertRaises(ValueError):
                d.rational(value)
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{} {}'):
            with self.assertRaises(ValueError):
                d.decode(value)
        tree = ast.parse(d.SOURCE.read_bytes())
        self.assertFalse(any(isinstance(n, ast.Constant) and type(n.value) is float
                             for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == "float" for n in ast.walk(tree)))
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
        names += [n.module for n in imports if isinstance(n, ast.ImportFrom)]
        self.assertEqual(set(names), {
            "argparse", "contextlib", "fractions", "hashlib", "json", "os", "pathlib", "sys",
        })

        def no_float(value):
            self.assertIsNot(type(value), float)
            if isinstance(value, dict):
                for item in value.values():
                    no_float(item)
            elif isinstance(value, list):
                for item in value:
                    no_float(item)

        no_float(self.result)

    def test_read_only_guard_blocks_writes_ancestor_reads_and_dispatch(self):
        audit = {"forbidden_calls": 0}
        operations = (
            lambda: d.SOURCE.write_bytes(b"forbidden"),
            lambda: Path(d.RETAINED_E795_PINS["stdout"]["path"]).read_bytes(),
            lambda: os.system(":"),
            lambda: sys.audit("subprocess.Popen", "forbidden", [], None, None),
            lambda: sys.audit("import", "numpy", None, None, None, None),
        )
        with d.read_only(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], len(operations))

    def test_cli_emits_one_json_and_zero_stderr_without_reexecuting_check(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(d, "check", return_value={"report": self.result}) as check:
            with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
                d.main(["--check"])
        check.assert_called_once_with()
        self.assertEqual(d.decode(out.getvalue()), {"report": self.result})
        self.assertEqual(err.getvalue(), "")
        self.assertTrue(out.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
