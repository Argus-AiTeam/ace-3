"""Independent integer-grid oracle and hostile controls for retained partitions."""

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import io
import json
from math import lcm
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_partition_localizer_v1 as d


class LocalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def test_exact_consumption_and_unchanged_nested_evidence(self):
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.result["pair_branch_count"], 54)
        self.assertEqual(self.result["localized_table_count"], 2)
        self.assertEqual(self.result["selected_coordinate_accounts"], 1035)
        self.assertEqual(self.result["weighted_component_count"], 7245)
        for table in self.result["tables"].values():
            self.assertEqual([tuple(a[k] for k in d.ACCOUNT_FIELDS) for a in table["accounts"]],
                             [(c, l, r, b) for c in d.CONTROLS
                              for l, r in d.PAIRS for b in d.BRANCHES])
        for key in ("lineage_separation", "reference_scope", "ranking_rule"):
            self.assertEqual(self.result[key], self.retained["report"][key])
        self.assertEqual(self.retained["input_pins"], d.RETAINED_449B_PINS)
        self.assertEqual(self.retained["retained_449b"]["input_pins"], d.RETAINED_50F_PINS)

    def test_every_winner_and_gap_against_nested_integer_grid_oracle(self):
        controls = self.retained["retained_449b"]["report"]["controls"]
        for kind, output in self.result["tables"].items():
            for member in output["accounts"]:
                control = next(c for c in controls if c["control"] == member["control"])
                pair = next(p for p in control["pairs"]
                            if (p["left_id"], p["right_id"]) ==
                            (member["left_id"], member["right_id"]))
                rows = pair["branches"][member["branch"]]["selected_coordinates"]
                cells = {}
                for row in rows:
                    if kind == "coordinate":
                        cells[(row["coordinate"], None)] = Fraction(row["direct_hidden_weighted_term"])
                    else:
                        for component, value in row["weighted_components"].items():
                            cells[(row["coordinate"], component)] = Fraction(value)
                denominator = lcm(*(v.denominator for v in cells.values()))
                integers = {k: abs(v.numerator) * (denominator // v.denominator)
                            for k, v in cells.items()}
                top = max(integers.values())
                winner = next(k for k, v in integers.items() if v == top)
                self.assertEqual(sum(v == top for v in integers.values()), 1)
                second = max(v for k, v in integers.items() if k != winner)
                self.assertEqual((member["winner_id"]["coordinate"],
                                  member["winner_id"].get("component")), winner)
                self.assertEqual(Fraction(member["maximum_absolute"]), Fraction(top, denominator))
                self.assertEqual(Fraction(member["runner_up_absolute"]), Fraction(second, denominator))
                self.assertEqual(Fraction(member["dominance_gap"]), Fraction(top - second, denominator))
                self.assertEqual(
                    {(r["coordinate"], r.get("component")) for r in member["runner_up_ids"]},
                    {k for k, v in integers.items() if v == second})
                self.assertEqual(member["roles"], pair["roles"])

    def test_exact_global_winner_facts(self):
        self.assertEqual(self.result["tables"]["coordinate"]["global"]["winner_counts"],
                         [{"coordinate": 62, "count": 20}, {"coordinate": 241, "count": 34}])
        self.assertEqual(self.result["tables"]["coordinate_component"]["global"]["winner_counts"], [
            {"coordinate": 62, "component": "input_hidden", "count": 36},
            {"coordinate": 241, "component": "mlp_stage17", "count": 17},
            {"coordinate": 241, "component": "fp16_to_branch_terminal_remainder", "count": 1},
        ])
        d.require_expected(self.result)

    def test_exact_pair_branch_exceptions_and_control_gaps(self):
        expected = {
            "coordinate": {
                (34319, 319, "binary64"): (62, None),
                (319, 34319, "binary64"): (62, None),
            },
            "coordinate_component": {
                (34319, 13, "binary64"): (241, "fp16_to_branch_terminal_remainder"),
            },
        }
        gaps = {
            "coordinate": Fraction(1827610068439962304726431454702255,
                                   1329227995784915872903807060280344576),
            "coordinate_component": Fraction(3832757189268268603981093761768175,
                                             1329227995784915872903807060280344576),
        }
        for kind, table in self.result["tables"].items():
            unstable = [p for p in table["partitions"]["by_pair_branch"] if not p["stable"]]
            self.assertEqual({(p["left_id"], p["right_id"], p["branch"]) for p in unstable},
                             set(expected[kind]))
            self.assertEqual(len(table["pair_branch_exceptions"]), len(expected[kind]))
            for p in unstable:
                self.assertEqual(sorted(c["count"] for c in p["winner_classes"]), [1, 8])
                self.assertEqual(len(p["exceptions"]), 1)
                exception = p["exceptions"][0]
                self.assertEqual(exception["control"], "mapped_all")
                self.assertEqual((exception["winner_id"]["coordinate"],
                                  exception["winner_id"].get("component")),
                                 expected[kind][(p["left_id"], p["right_id"], p["branch"])])
                self.assertEqual(Fraction(exception["dominance_gap"]), gaps[kind])
                majority = next(c for c in p["winner_classes"] if c["count"] == 8)
                self.assertEqual({m["control"] for m in majority["members"]},
                                 set(d.CONTROLS) - {"mapped_all"})
            for p in table["partitions"]["by_pair_branch"]:
                if p["stable"]:
                    self.assertEqual(p["winner_counts"][0]["count"], 9)
                    self.assertEqual(p["exceptions"], [])

    def test_every_partition_membership_map_modes_exceptions_and_gap_extrema(self):
        fields = {
            "by_branch": ("branch",), "by_control": ("control",),
            "by_ordered_pair": ("left_id", "right_id"),
            "by_control_branch": ("control", "branch"),
            "by_pair_branch": ("left_id", "right_id", "branch"),
            "by_control_pair": ("control", "left_id", "right_id"),
        }
        sizes = {"by_branch": (2, 27), "by_control": (9, 6),
                 "by_ordered_pair": (3, 18), "by_control_branch": (18, 3),
                 "by_pair_branch": (6, 9), "by_control_pair": (27, 2)}
        for table in self.result["tables"].values():
            for name, keys in fields.items():
                groups = table["partitions"][name]
                size, count = sizes[name]
                self.assertEqual(len(groups), size)
                for group in groups:
                    members = [a for a in table["accounts"]
                               if all(a[k] == group[k] for k in keys)]
                    counts = Counter(tuple(sorted(m["winner_id"].items())) for m in members)
                    self.assertEqual(len(members), count)
                    self.assertEqual(group["account_count"], count)
                    self.assertEqual({tuple(sorted((k, v) for k, v in row.items() if k != "count")):
                                      row["count"] for row in group["winner_counts"]}, counts)
                    stable = len(counts) == 1
                    self.assertEqual(group["stable"], stable)
                    state = "stable" if stable else "unstable"
                    self.assertIn({k: group[k] for k in keys}, table["partition_map"][name][state])
                    modes = [dict(k) for k, n in counts.items() if n == max(counts.values())]
                    self.assertCountEqual(group["modal_winner_ids"], modes)
                    mode = modes[0] if len(modes) == 1 else None
                    self.assertEqual(group["unique_modal_winner_id"], mode)
                    self.assertEqual(group["exceptions"], [] if mode is None else
                                     [m for m in members if m["winner_id"] != mode])
                    self.assertCountEqual([m for c in group["winner_classes"] for m in c["members"]],
                                          members)
                    self.assertEqual(sum(c["count"] for c in group["winner_classes"]), count)
                    for c in [group, *group["winner_classes"]]:
                        subset = members if c is group else c["members"]
                        gaps = [Fraction(m["dominance_gap"]) for m in subset]
                        self.assertEqual(Fraction(c["minimum_dominance_gap"]), min(gaps))
                        self.assertEqual(Fraction(c["maximum_dominance_gap"]), max(gaps))
                mapping = table["partition_map"][name]
                self.assertEqual(len(mapping["stable"]) + len(mapping["unstable"]), size)

    def test_frequency_ties_do_not_manufacture_exception_baselines(self):
        rows = [{"winner_id": {"coordinate": c}, "dominance_gap": "1/3"} for c in (62, 241)]
        group = d.partition(rows)
        self.assertFalse(group["stable"])
        self.assertIsNone(group["unique_modal_winner_id"])
        self.assertEqual(group["exceptions"], [])
        self.assertEqual(group["modal_winner_ids"], [{"coordinate": 62}, {"coordinate": 241}])
        with self.assertRaisesRegex(ValueError, "empty partition"):
            d.partition([])

    def test_hostile_ranking_and_gap_mutations_fail_closed(self):
        original = self.retained["report"]["accounts"][0]["tables"]["coordinate"]
        mutations = [
            lambda t: t.update(dominance_gap="0"),
            lambda t: t.update(winner_ids=[{"coordinate": 0}]),
            lambda t: t.update(unique_nonzero_hotspot=False),
            lambda t: t.update(row_count=True),
            lambda t: t.update(all_zero=True),
            lambda t: t["rows"].reverse(),
            lambda t: t["rows"][0].update(absolute="-1"),
            lambda t: t["rows"][0].update(absolute="2/2"),
            lambda t: t["rows"][0].update(signed="0"),
            lambda t: t["rows"][0].update(rank=True),
            lambda t: t["rows"][0].update(gap_from_maximum="1"),
            lambda t: t["rows"][0].update(coordinate=True),
            lambda t: t["rows"][0].update(component="input_hidden"),
            lambda t: t["rows"][1].update(coordinate=t["rows"][0]["coordinate"]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                table = deepcopy(original)
                mutate(table)
                with self.assertRaises(ValueError):
                    d.localize_table(table, "coordinate")

    def test_exact_near_tie_and_retained_unique_domain(self):
        epsilon = Fraction(1, 2**120)
        values = [Fraction(1), 1 - epsilon]
        table = {
            "rows": [{"coordinate": i, "signed": str(v), "absolute": str(v),
                      "rank": i + 1, "gap_from_maximum": str(1 - v)}
                     for i, v in enumerate(values)],
            "row_count": 2, "zero_row_count": 0, "maximum_tie_count_including_zeros": 1,
            "maximum_absolute": "1", "runner_up_absolute": str(values[1]),
            "dominance_gap": str(epsilon), "next_distinct_absolute": str(values[1]),
            "gap_to_next_distinct": str(epsilon), "winner_ids": [{"coordinate": 0}],
            "maximizer_ids_including_zeros": [{"coordinate": 0}],
            "unique_nonzero_hotspot": True, "all_zero": False, "exact_gap_and_tie_closure": True,
        }
        self.assertEqual(d.localize_table(table, "coordinate")["dominance_gap"], str(epsilon))
        for values in (("1", "1"), ("0", "0")):
            changed = deepcopy(table)
            for row, value in zip(changed["rows"], values, strict=True):
                row.update(signed=value, absolute=value)
            with self.assertRaisesRegex(ValueError, "unique nonzero winners"):
                d.localize_table(changed, "coordinate")

    def test_identity_census_rejects_missing_duplicate_and_reordered_accounts(self):
        for transform in (lambda a: a[:-1], lambda a: [a[1], *a[1:]], lambda a: list(reversed(a))):
            changed = dict(self.retained)
            changed["report"] = {**self.retained["report"],
                                 "accounts": transform(self.retained["report"]["accounts"])}
            with self.assertRaisesRegex(ValueError, "54-account"):
                d.report(changed)

    def test_nested_pin_and_non_admission_mutations_fail_closed(self):
        for layer in ("449b", "50f", "flags"):
            changed = dict(self.retained)
            if layer == "449b":
                changed["input_pins"] = {}
            elif layer == "50f":
                changed["retained_449b"] = {**self.retained["retained_449b"], "input_pins": {}}
            else:
                changed["flags"] = {**self.retained["flags"], "new_token_claim": True}
            with self.assertRaises(ValueError):
                d.validate_retained(changed)

    def test_authentication_rejects_changed_bytes_counts_and_nonterminal_review(self):
        pin = d.PINS["review"]
        data = Path(pin["path"]).read_bytes()
        for changed in (data + b" ", b"X" + data[1:]):
            with patch.object(Path, "read_bytes", return_value=changed):
                with self.assertRaisesRegex(ValueError, "retained (byte count|hash) changed"):
                    d.bound_bytes(pin)
        review = d.decode(data)
        for field, value in (("mission_id", "449b1c24ea06"), ("producer_role", "engineer"),
                             ("kind", "mission_context"), ("review", {"status": "continue"})):
            with self.assertRaisesRegex(ValueError, "independent terminal"):
                d.validate_review({**review, field: value})
        for invalid in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{}{}'):
            with self.assertRaises(ValueError):
                d.decode(invalid)
        for invalid in ("2/2", "0.5", 0.5, True):
            with self.assertRaises(ValueError):
                d.rational(invalid)
        for invalid in ({"calls": 1}, {"claim": True}, {"calls": 0.0}, {}):
            with self.assertRaises(ValueError):
                d.zero_counters(invalid)

    def test_capture_member_source_environment_and_stdout_splices_fail_closed(self):
        raw = {name: Path(pin["path"]).read_bytes() for name, pin in d.PINS.items()
               if name in ("review", "capture")}
        original = d.bound_bytes
        changes = [
            lambda c: c.update(success=False),
            lambda c: c["sources_after"][0].update(sha256="0" * 64),
            lambda c: c["preflight"].update(uid=0),
            lambda c: c["results"][-1].update(exit_status=1),
            lambda c: c["results"][-1]["files"][2].update(path="/not-retained"),
        ]
        for change in changes:
            capture = d.decode(raw["capture"])
            change(capture)

            def substituted(pin):
                if pin == d.PINS["capture"]:
                    return json.dumps(capture).encode()
                return original(pin)

            with patch.object(d, "bound_bytes", side_effect=substituted):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_runtime_guard_forbids_write_decode_import_and_dispatch(self):
        audit = {"forbidden_calls": 0}
        events = [
            ("open", (d.PINS["stdout"]["path"], "w", os.O_WRONLY | os.O_TRUNC)),
            ("open", ("/not-retained/tensor", "r", os.O_RDONLY)),
            ("import", ("ace3.model.closed_producer",)),
            ("import", ("numpy",)),
            ("import", ("torch",)),
            ("import", ("safetensors",)),
            ("subprocess.Popen", ()),
            ("socket.connect", ()),
            ("os.mkdir", ()),
            ("os.rename", ()),
            ("os.system", ()),
        ]
        with d.read_only(audit):
            for event, args in events:
                with self.assertRaisesRegex(RuntimeError, "retained-only localizer"):
                    sys.audit(event, *args)
        self.assertEqual(audit["forbidden_calls"], len(events))

    def test_production_imports_are_stdlib_only_and_cli_is_single_json(self):
        tree = ast.parse(d.SOURCE.read_bytes())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertTrue(set(imports) <= {
            "argparse", "contextlib", "fractions", "hashlib", "json", "os", "pathlib", "sys"})
        output = io.StringIO()
        with patch.object(d, "check", return_value={"report": self.result}), patch("sys.stdout", output):
            d.main(["--check"])
        self.assertEqual(d.decode(output.getvalue()), {"report": self.result})
        self.assertEqual(output.getvalue().count("\n"), 1)
        with patch.object(d, "check", side_effect=ValueError("blocked")), patch("sys.stdout", io.StringIO()) as out:
            with self.assertRaisesRegex(ValueError, "blocked"):
                d.main(["--check"])
            self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
