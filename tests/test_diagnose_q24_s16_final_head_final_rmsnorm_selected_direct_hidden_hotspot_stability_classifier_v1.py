"""Independent integer-grid checks of retained hotspot stability and exact gaps."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
from math import lcm
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_hotspot_stability_classifier_v1 as d


def grid(values):
    fractions = [Fraction(v) for v in values]
    denominator = lcm(*(v.denominator for v in fractions))
    return [v.numerator * (denominator // v.denominator) for v in fractions], denominator


def fixture(values):
    rows = [{"coordinate": i, "signed": str(v), "absolute": str(abs(v))}
            for i, v in enumerate(values)]
    rows.sort(key=lambda r: (-Fraction(r["absolute"]), r["coordinate"]))
    for row in rows:
        row["rank"] = 1 + sum(Fraction(r["absolute"]) > Fraction(row["absolute"])
                              for r in rows)
    nonzero = any(values)
    return {"ranking": rows, "nonzero_hotspot": nonzero,
            "maximum_ties": [r for r in rows if r["rank"] == 1] if nonzero else []}


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def test_exact_census_and_retained_immutability(self):
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual(self.retained, self.before)
        self.assertEqual(self.retained["input_pins"], d.RETAINED_50F_PINS)
        self.assertEqual([self.result[k] for k in (
            "control_count", "diagnostic_pair_count", "pair_branch_count",
            "selected_coordinate_accounts", "weighted_component_count")],
            [9, 27, 54, 1035, 7245])
        identities = [(a["control"], a["left_id"], a["right_id"], a["branch"])
                      for a in self.result["accounts"]]
        self.assertEqual(identities, [(c, l, r, b) for c in d.CONTROLS
                                     for l, r in d.PAIRS for b in d.BRANCHES])
        for key in ("lineage_separation", "reference_scope", "ranking_rule"):
            self.assertEqual(self.result[key], self.retained["report"][key])

    def test_every_ranking_against_independent_integer_grid(self):
        for a in self.result["accounts"]:
            control = next(c for c in self.retained["report"]["controls"]
                           if c["control"] == a["control"])
            pair = next(p for p in control["pairs"]
                        if (p["left_id"], p["right_id"]) == (a["left_id"], a["right_id"]))
            source = pair["branches"][a["branch"]]["selected_coordinates"]
            ints, denominator = grid(row["weighted_components"][k]
                                     for row in source for k in d.COMPONENTS)
            cells = {(row["coordinate"], k): ints[7 * i + j]
                     for i, row in enumerate(source) for j, k in enumerate(d.COMPONENTS)}
            expected = {
                "component": {(None, k): (sum(cells[(r["coordinate"], k)] for r in source),
                                          sum(abs(cells[(r["coordinate"], k)]) for r in source))
                              for k in d.COMPONENTS},
                "coordinate": {(r["coordinate"], None):
                               (sum(ints[7*i:7*i+7]), abs(sum(ints[7*i:7*i+7])))
                               for i, r in enumerate(source)},
                "coordinate_component": {key: (v, abs(v)) for key, v in cells.items()},
            }
            expected.update({k: {key: (v, abs(v)) for key, v in cells.items() if key[1] == k}
                             for k in d.COMPONENTS})
            for key, table in a["tables"].items():
                oracle = expected[key]
                self.assertEqual(len(table["rows"]), len(oracle))
                magnitudes = sorted((v[1] for v in oracle.values()), reverse=True)
                top, second = magnitudes[0], magnitudes[1]
                self.assertEqual(Fraction(table["dominance_gap"]),
                                 Fraction(top-second, denominator))
                self.assertEqual(Fraction(table["maximum_absolute"]), Fraction(top, denominator))
                self.assertEqual(Fraction(table["runner_up_absolute"]), Fraction(second, denominator))
                lower = next((v for v in magnitudes if v < top), None)
                self.assertEqual(table["next_distinct_absolute"],
                                 None if lower is None else str(Fraction(lower, denominator)))
                self.assertEqual(table["gap_to_next_distinct"],
                                 None if lower is None else str(Fraction(top-lower, denominator)))
                for row in table["rows"]:
                    signed, absolute = oracle[(row.get("coordinate"), row.get("component"))]
                    self.assertEqual(Fraction(row["signed"]), Fraction(signed, denominator))
                    self.assertEqual(Fraction(row["absolute"]), Fraction(absolute, denominator))
                    self.assertEqual(Fraction(row["gap_from_maximum"]),
                                     Fraction(top-absolute, denominator))
                    self.assertEqual(row["rank"], 1 + sum(v > absolute for v in magnitudes))
                winners = {(r.get("coordinate"), r.get("component")) for r in table["winner_ids"]}
                self.assertEqual(winners, {k for k, v in oracle.items() if v[1] == top and top})
                self.assertEqual(table["zero_row_count"], magnitudes.count(0))
                self.assertEqual(table["maximum_tie_count_including_zeros"], magnitudes.count(top))
                self.assertEqual(table["unique_nonzero_hotspot"], bool(top) and magnitudes.count(top) == 1)

    def test_required_measured_winners(self):
        tables = self.result["global"]["tables"]
        self.assertEqual(tables["component"]["unique_winner_counts"],
                         [{"component": "mlp_stage17", "count": 54}])
        self.assertEqual(tables["component"]["classification"], "STABLE_UNIQUE_HOTSPOT")
        self.assertEqual(tables["coordinate"]["unique_winner_counts"],
                         [{"coordinate": 62, "count": 20}, {"coordinate": 241, "count": 34}])
        self.assertEqual(tables["coordinate_component"]["unique_winner_counts"], [
            {"coordinate": 62, "component": "input_hidden", "count": 36},
            {"coordinate": 241, "component": "mlp_stage17", "count": 17},
            {"coordinate": 241, "component": d.COMPONENTS[-1], "count": 1}])
        for key in ("coordinate", "coordinate_component"):
            self.assertEqual(tables[key]["classification"], "VARIABLE_HOTSPOTS")
            self.assertTrue(tables[key]["all_accounts_unique_nonzero"])

    def test_all_partition_counts_winners_and_exact_gap_extrema(self):
        fields = {
            "by_branch": ("branch",), "by_control": ("control",),
            "by_ordered_pair": ("left_id", "right_id"),
            "by_control_branch": ("control", "branch"),
            "by_pair_branch": ("left_id", "right_id", "branch"),
            "by_control_pair": ("control", "left_id", "right_id"),
        }
        expected_sizes = [(2, 27), (9, 6), (3, 18), (18, 3), (6, 9), (27, 2)]
        for (name, keys), (size, count) in zip(fields.items(), expected_sizes, strict=True):
            groups = self.result["partitions"][name]
            self.assertEqual(len(groups), size)
            for group in groups:
                members = [a for a in self.result["accounts"]
                           if all(a[k] == group[k] for k in keys)]
                self.assertEqual(group["account_count"], count)
                self.assertEqual(len(members), count)
                for key, summary in group["tables"].items():
                    tables = [a["tables"][key] for a in members]
                    winners, unique = {}, {}
                    for table in tables:
                        for row in table["winner_ids"]:
                            ident = tuple(sorted(row.items()))
                            winners[ident] = winners.get(ident, 0) + 1
                            if len(table["winner_ids"]) == 1:
                                unique[ident] = unique.get(ident, 0) + 1
                    for field, expected in (("winner_membership_counts", winners),
                                            ("unique_winner_counts", unique)):
                        self.assertEqual({tuple(sorted((k, v) for k, v in row.items()
                                                       if k != "count")): row["count"]
                                          for row in summary[field]}, expected)
                    gaps = [Fraction(t["dominance_gap"]) for t in tables]
                    self.assertEqual(Fraction(summary["minimum_dominance_gap"]), min(gaps))
                    self.assertEqual(Fraction(summary["maximum_dominance_gap"]), max(gaps))
                    sets = [{tuple(sorted(r.items())) for r in t["winner_ids"]} for t in tables]
                    stable = bool(sets[0]) and all(s == sets[0] for s in sets)
                    self.assertEqual(summary["stable_nonzero_winner_set"], stable)
                    self.assertEqual(summary["stable_unique_nonzero_hotspot"],
                                     stable and len(sets[0]) == 1)
                    self.assertEqual(summary["zero_only_accounts"], sum(not s for s in sets))
                    self.assertEqual(summary["tied_nonzero_accounts"], sum(len(s) > 1 for s in sets))
                    self.assertEqual(summary["zero_rows"], sum(t["zero_row_count"] for t in tables))

    def test_fp16_terminal_zero_is_not_a_stable_hotspot(self):
        fp16 = next(p for p in self.result["partitions"]["by_branch"] if p["branch"] == "fp16")
        summary = fp16["tables"][d.COMPONENTS[-1]]
        self.assertEqual(summary["classification"], "ZERO_ONLY_NO_NONZERO_HOTSPOT")
        self.assertEqual(summary["zero_only_accounts"], 27)
        self.assertEqual(summary["winner_membership_counts"], [])
        self.assertFalse(summary["stable_unique_nonzero_hotspot"])
        for a in self.result["accounts"]:
            if a["branch"] == "fp16":
                table = a["tables"][d.COMPONENTS[-1]]
                self.assertEqual(table["zero_row_count"], a["selected_coordinate_count"])
                self.assertEqual(table["maximum_tie_count_including_zeros"], table["row_count"])
                self.assertEqual(table["gap_to_next_distinct"], None)
                self.assertEqual(table["dominance_gap"], "0")

    def test_exact_ties_near_ties_zero_rows_and_singletons(self):
        tied = d.classify_table(fixture([Fraction(-4), Fraction(4), Fraction(2), Fraction()]))
        self.assertEqual(tied["dominance_gap"], "0")
        self.assertEqual(tied["gap_to_next_distinct"], "2")
        self.assertEqual(tied["winner_ids"], [{"coordinate": 0}, {"coordinate": 1}])
        self.assertEqual(tied["zero_row_count"], 1)
        self.assertEqual(d.summarize([tied, tied])["classification"], "STABLE_TIED_HOTSPOTS")
        near = d.classify_table(fixture([Fraction(1), Fraction(2**100-1, 2**100)]))
        self.assertEqual(near["dominance_gap"], str(Fraction(1, 2**100)))
        self.assertTrue(near["unique_nonzero_hotspot"])
        singleton = d.classify_table(fixture([Fraction(-3)]))
        self.assertEqual(singleton["dominance_gap"], "3")
        self.assertIsNone(singleton["gap_to_next_distinct"])
        zero = d.classify_table(fixture([Fraction(), Fraction()]))
        self.assertEqual(zero["winner_ids"], [])
        self.assertEqual(zero["maximum_tie_count_including_zeros"], 2)
        self.assertEqual(d.summarize([zero, near])["classification"],
                         "MIXED_ZERO_AND_NONZERO_HOTSPOTS")
        with self.assertRaises(ValueError):
            d.classify_table(fixture([]))
        with self.assertRaises(ValueError):
            d.summarize([])

    def test_table_mutations_refused(self):
        for mutate in (
            lambda t: t["ranking"].reverse(),
            lambda t: t["ranking"].append(t["ranking"][0]),
            lambda t: t["ranking"][0].update(rank=2),
            lambda t: t["ranking"][0].update(rank=True),
            lambda t: t["ranking"][0].update(coordinate=True),
            lambda t: t["ranking"][0].update(absolute="-1"),
            lambda t: t.update(maximum_ties=[]),
            lambda t: t.update(nonzero_hotspot=False),
        ):
            table = fixture([Fraction(3), Fraction(2)])
            mutate(table)
            with self.assertRaises(ValueError):
                d.classify_table(table)
        for value in (1.0, True, "1.0", "2/2", "NaN"):
            with self.assertRaises(ValueError):
                d.rational(value)

    def test_account_operand_reference_and_ranked_mass_splices_refused(self):
        source = self.retained["report"]["controls"][0]["pairs"][0]["branches"]["fp16"]
        for mutate in (
            lambda a: a.update(hidden_reference="reanchored"),
            lambda a: a.update(binary64_internal_stages="reconstructed"),
            lambda a: a.update(exact_final_margin_identity=False),
            lambda a: a["selected_coordinates"][0]["weighted_components"].update(input_hidden="0"),
            lambda a: a["coordinate_hotspots"]["ranking"][0].update(signed="0"),
            lambda a: a["component_hotspots_by_coordinate_absolute_mass"]["ranking"][0].update(absolute="1"),
            lambda a: a["per_component_coordinate_hotspots"].pop("mlp_stage17"),
        ):
            changed = deepcopy(source)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.classify_account(changed, "fp16")

    def test_control_pair_branch_and_parent_splices_refused(self):
        for mutate in (
            lambda p: p["controls"].reverse(),
            lambda p: p["controls"][0]["pairs"][0].update(left_id=13),
            lambda p: p["controls"][0]["pairs"][0].update(roles=["changed"]),
            lambda p: p["controls"][0]["pairs"][0]["branches"].pop("fp16"),
            lambda p: p.update(weighted_component_count=7244),
            lambda p: p["controls"][0]["pairs"][0]["branches"]["fp16"]["unchanged_margin_accounting"].update(retained_margin_change="0"),
        ):
            changed = deepcopy(self.retained)
            mutate(changed["report"])
            with self.assertRaises(ValueError):
                d.report(changed)

    def test_pin_json_review_capture_and_50f_failures_refused(self):
        with patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "byte count"):
                d.bound_bytes(d.PINS["stdout"])
            with self.assertRaisesRegex(ValueError, "hash"):
                d.bound_bytes({**d.PINS["stdout"], "bytes": 2})
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{} {}'):
            with self.assertRaises(ValueError):
                d.decode(value)
        for target, mutate in (
            ("review", lambda r: r["review"].update(status="pending")),
            ("capture", lambda r: r.update(success=False)),
            ("capture", lambda r: r["results"][1].update(exit_status=1)),
            ("stdout", lambda r: r["input_pins"]["stdout"].update(sha256="0"*64)),
            ("stdout", lambda r: r["dispatch_and_write_audit"].update(forbidden_calls=1)),
        ):
            def altered(pin):
                data = Path(pin["path"]).read_bytes()
                if pin["path"] == d.PINS[target]["path"]:
                    value = json.loads(data)
                    mutate(value)
                    return json.dumps(value).encode()
                return data
            with patch.object(d, "bound_bytes", side_effect=altered):
                with self.assertRaises(ValueError):
                    d.authenticate()

    def test_dispatch_write_and_nonretained_read_guards(self):
        probes = (
            lambda: d.SOURCE.open("wb"),
            lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.remove(d.SOURCE),
            lambda: os.system("false"),
            lambda: Path(d.RETAINED_50F_PINS["stdout"]["path"]).read_bytes(),
            lambda: __import__("ace3.model.decoder_layer0_oracle", fromlist=["unused"]),
        )
        for probe in probes:
            audit = {"forbidden_calls": 0}
            with d.read_only(audit):
                with self.assertRaises(RuntimeError):
                    probe()
            self.assertEqual(audit["forbidden_calls"], 1)

    def test_cli_single_json_and_no_output_path(self):
        with patch.object(d, "check", return_value={"status": "fixture", "report": self.result}):
            output = io.StringIO()
            with patch("sys.stdout", output):
                d.main(["--check"])
        text = output.getvalue()
        value, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(text[end:], "\n")
        self.assertEqual(value["report"], self.result)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--output", "forbidden"])

    def test_runtime_gate_and_operator_free_imports(self):
        with patch.object(os, "getuid", return_value=-1):
            with self.assertRaisesRegex(ValueError, "gate failed"):
                d.check()
        tree = ast.parse(d.SOURCE.read_bytes())
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(modules, {"argparse", "contextlib", "fractions", "hashlib",
                                   "json", "os", "pathlib", "sys"})
