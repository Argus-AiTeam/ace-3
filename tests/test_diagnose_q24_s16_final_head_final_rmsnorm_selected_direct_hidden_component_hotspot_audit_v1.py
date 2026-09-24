"""Independent integer-grid oracles for retained-only selected hotspot accounts."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
from math import lcm
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_component_hotspot_audit_v1 as d


def grid(values):
    fractions = [Fraction(v) for v in values]
    denominator = lcm(*(v.denominator for v in fractions))
    integers = [v.numerator * (denominator // v.denominator) for v in fractions]
    return integers, denominator


class SelectedHotspotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with d.read_only({"forbidden_calls": 0}):
            cls.retained = d.authenticate()
            cls.result = d.report(cls.retained)

    def accounts(self):
        for control, original in zip(self.result["controls"],
                                     self.retained["report"]["controls"], strict=True):
            self.assertEqual(control["control"], original["control"])
            for pair, prior in zip(control["pairs"], original["pairs"], strict=True):
                for branch, account in pair["branches"].items():
                    yield control, pair, branch, account, prior["branches"][branch]

    def test_census_and_ordered_identities(self):
        self.assertEqual(self.result["selected_coordinate_accounts"], 1035)
        self.assertEqual(self.result["weighted_component_count"], 7245)
        self.assertEqual(sum(1 for _ in self.accounts()), 54)
        for control, original in zip(self.result["controls"],
                                     self.retained["report"]["controls"], strict=True):
            for pair, prior in zip(control["pairs"], original["pairs"], strict=True):
                self.assertEqual({k: pair[k] for k in ("left_id", "right_id", "roles")},
                                 {k: prior[k] for k in ("left_id", "right_id", "roles")})

    def test_independent_integer_grid_signed_absolute_and_cancellation(self):
        for _, _, _, account, old in self.accounts():
            rows = old["selected_coordinates"]
            for key in d.COMPONENTS:
                integers, denominator = grid(row["weighted_components"][key] for row in rows)
                total = account["per_component_coordinate_hotspots"][key]["mass"]
                expected = {
                    "signed": Fraction(sum(integers), denominator),
                    "absolute": Fraction(sum(abs(v) for v in integers), denominator),
                    "cancellation_absolute_mass":
                        Fraction(sum(abs(v) for v in integers) - abs(sum(integers)), denominator),
                    "positive_mass": Fraction(sum(v for v in integers if v > 0), denominator),
                    "negative_absolute_mass": Fraction(-sum(v for v in integers if v < 0), denominator),
                }
                self.assertEqual({k: Fraction(total[k]) for k in expected}, expected)
                self.assertEqual(total["zero_count"], integers.count(0))

    def test_independent_nested_cancellation_closure(self):
        for _, _, _, account, old in self.accounts():
            rows = old["selected_coordinates"]
            values = [r["weighted_components"][k] for r in rows for k in d.COMPONENTS]
            integers, denominator = grid(values)
            direct = [sum(integers[i:i+7]) for i in range(0, len(integers), 7)]
            gross, net_abs = sum(abs(v) for v in integers), sum(abs(v) for v in direct)
            totals = account["direct_hidden_accounting"]
            self.assertEqual(Fraction(totals["within_coordinate_cancellation_mass"]),
                             Fraction(gross-net_abs, denominator))
            self.assertEqual(Fraction(totals["across_coordinate_cancellation_mass"]),
                             Fraction(net_abs-abs(sum(direct)), denominator))
            self.assertEqual(Fraction(totals["total_component_cancellation_mass"]),
                             Fraction(gross-abs(sum(direct)), denominator))
            self.assertEqual(totals, old["direct_hidden_accounting"])

    def test_all_ranked_cells_and_independent_pairwise_order(self):
        for _, _, _, account, old in self.accounts():
            entries = account["coordinate_component_hotspots"]["ranking"]
            expected = {(r["coordinate"], k): r["weighted_components"][k]
                        for r in old["selected_coordinates"] for k in d.COMPONENTS}
            self.assertEqual(len(entries), len(expected))
            self.assertEqual({(r["coordinate"], r["component"]): r["signed"] for r in entries},
                             expected)
            magnitudes = [abs(Fraction(r["signed"])) for r in entries]
            for i, row in enumerate(entries):
                self.assertEqual(Fraction(row["absolute"]), magnitudes[i])
                self.assertEqual(row["rank"], 1 + sum(v > magnitudes[i] for v in magnitudes))
                if i:
                    self.assertGreaterEqual(magnitudes[i-1], magnitudes[i])
                    if magnitudes[i-1] == magnitudes[i]:
                        previous = entries[i-1]
                        self.assertLess((previous["coordinate"], d.COMPONENTS.index(previous["component"])),
                                        (row["coordinate"], d.COMPONENTS.index(row["component"])))

    def test_all_coordinate_and_component_rankings(self):
        for _, _, _, account, old in self.accounts():
            coordinates = account["coordinate_hotspots"]["ranking"]
            self.assertEqual({r["coordinate"]: r["signed"] for r in coordinates},
                             {r["coordinate"]: r["direct_hidden_weighted_term"]
                              for r in old["selected_coordinates"]})
            components = account["component_hotspots_by_coordinate_absolute_mass"]["ranking"]
            self.assertEqual({r["component"] for r in components}, set(d.COMPONENTS))
            for table in (account["coordinate_hotspots"],
                          account["component_hotspots_by_coordinate_absolute_mass"],
                          *account["per_component_coordinate_hotspots"].values()):
                entries = table["ranking"]
                magnitudes = [Fraction(r["absolute"]) for r in entries]
                self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))
                self.assertEqual(table["maximum_ties"],
                                 [r for r in entries if Fraction(r["absolute"]) == max(magnitudes)]
                                 if any(magnitudes) else [])
                for row in entries:
                    self.assertEqual(row["rank"],
                                     1 + sum(v > Fraction(row["absolute"]) for v in magnitudes))
            for row in components:
                self.assertEqual(row["absolute"],
                                 old["direct_hidden_accounting"]["component_totals"][row["component"]]["absolute"])

    def test_coordinate62_is_once_and_has_exact_complement(self):
        for _, _, _, account, _ in self.accounts():
            isolated = account["coordinate62"]
            self.assertTrue(isolated["included_once"])
            self.assertEqual(isolated["selected"]["coordinate"], 62)
            self.assertEqual(sum(r["coordinate"] == 62 for r in account["selected_coordinates"]), 1)
            for key in ("signed", "absolute"):
                self.assertEqual(Fraction(isolated["selected"][key])
                                 + Fraction(isolated["other_coordinates_mass"][key]),
                                 Fraction(account["direct_hidden_accounting"]["selected_direct_hidden"][key]))
            for component, table in account["per_component_coordinate_hotspots"].items():
                for key in ("signed", "absolute"):
                    self.assertEqual(Fraction(table["coordinate62"][key])
                                     + Fraction(table["other_coordinates_mass"][key]),
                                     Fraction(table["mass"][key]))

    def test_fp16_zero_terminal_and_binary64_retained_remainder(self):
        seen_nonzero = False
        for _, _, branch, account, _ in self.accounts():
            table = account["per_component_coordinate_hotspots"][d.COMPONENTS[-1]]
            if branch == "fp16":
                self.assertFalse(table["nonzero_hotspot"])
                self.assertEqual(table["maximum_ties"], [])
                self.assertTrue(all(r["rank"] == 1 and r["signed"] == "0" for r in table["ranking"]))
            else:
                seen_nonzero |= table["nonzero_hotspot"]
            self.assertEqual(account["binary64_internal_stages"], "NOT_RETAINED_NO_RECONSTRUCTION")
        self.assertTrue(seen_nonzero)

    def test_margin_and_reference_data_not_replaced(self):
        for _, _, _, account, old in self.accounts():
            self.assertIs(account["selected_coordinates"], old["selected_coordinates"])
            self.assertIs(account["unchanged_margin_accounting"], old["unchanged_margin_accounting"])
            margin = account["unchanged_margin_accounting"]
            values = [v["signed"] for v in margin["selected_term_totals"].values()]
            values += [margin["unselected_coordinate_signed_remainder"],
                       margin["head_boundary_remainder_change"]]
            integers, denominator = grid(values)
            self.assertEqual(Fraction(sum(integers), denominator),
                             Fraction(margin["retained_margin_change"]))
        self.assertEqual(self.result["lineage_separation"],
                         self.retained["report"]["lineage_separation"])

    def test_ties_zeros_and_signed_cancellation_synthetic(self):
        entries = [{"coordinate": i, "signed": str(v), "absolute": str(abs(v))}
                   for i, v in [(62, -4), (7, 4), (8, 0), (3, 2)]]
        result = d.hotspot(entries)
        self.assertEqual([r["coordinate"] for r in result["ranking"]], [7, 62, 3, 8])
        self.assertEqual([r["rank"] for r in result["ranking"]], [1, 1, 3, 4])
        self.assertEqual(len(result["maximum_ties"]), 2)
        self.assertEqual(d.signed_masses([Fraction(4), Fraction(-4)]),
                         {"signed": "0", "absolute": "8", "cancellation_absolute_mass": "8",
                          "positive_mass": "4", "negative_absolute_mass": "4", "zero_count": 0})
        zeros = d.hotspot([{"coordinate": 62, "signed": "0", "absolute": "0"}])
        self.assertFalse(zeros["nonzero_hotspot"])
        self.assertEqual(zeros["maximum_ties"], [])

    def test_coordinate_and_component_mutations_refused(self):
        old = self.retained["report"]["controls"][0]["pairs"][0]["branches"]["fp16"]
        upstream = self.retained["report"]["retained_final_rmsnorm_logit_margin_bridge"]["controls"][0]["pairs"][0]["branches"]["fp16"]
        for mutate in (
            lambda b: b["selected_coordinates"].append(b["selected_coordinates"][0]),
            lambda b: b["selected_coordinates"][0]["weighted_components"].update(input_hidden="123"),
            lambda b: b["selected_coordinates"][0].update(coordinate=True),
            lambda b: b.update(selected_coordinates=[r for r in b["selected_coordinates"] if r["coordinate"] != 62]),
            lambda b: b["direct_hidden_accounting"].update(component_absolute_sum="0"),
            lambda b: b["unchanged_margin_accounting"].update(retained_margin_change="0"),
            lambda b: b.update(hidden_reference="local_reanchored"),
            lambda b: b.update(binary64_internal_stages="reconstructed"),
        ):
            changed = deepcopy(old)
            mutate(changed)
            with self.assertRaises(ValueError):
                d.branch_account(changed, upstream, "fp16")

    def test_control_pair_and_branch_splices_refused(self):
        for mutate in (
            lambda r: r["controls"].reverse(),
            lambda r: r["controls"][0]["pairs"][0].update(left_id=13),
            lambda r: r["controls"][0]["pairs"][0]["branches"].pop("fp16"),
            lambda r: r.update(selected_coordinate_accounts=1034),
            lambda r: r["controls"][0]["pairs"][0].update(roles=["different"]),
        ):
            changed = deepcopy(self.retained)
            mutate(changed["report"])
            with self.assertRaises(ValueError):
                d.report(changed)

    def test_pin_and_json_corruption_refused(self):
        pin = {**d.PINS["stdout"], "sha256": "0"*64}
        with patch.object(Path, "read_bytes", return_value=b"{}"):
            with self.assertRaisesRegex(ValueError, "byte count"):
                d.bound_bytes(pin)
            pin["bytes"] = 2
            with self.assertRaisesRegex(ValueError, "hash"):
                d.bound_bytes(pin)
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{} {}'):
            with self.assertRaises(ValueError):
                d.decode(data)

    def test_review_and_parent_failure_refused(self):
        def altered(pin):
            data = Path(pin["path"]).read_bytes()
            if pin == d.PINS["review"]:
                review = json.loads(data)
                review["review"]["status"] = "pending"
                return json.dumps(review).encode()
            return data
        with patch.object(d, "bound_bytes", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "independent reviewed"):
                d.authenticate()
        retained = deepcopy(self.retained)
        retained["dispatch_and_write_audit"]["forbidden_calls"] = 1
        original = d.decode
        with patch.object(d, "decode", side_effect=lambda data:
                          retained if hashlib.sha256(data).hexdigest() == d.PINS["stdout"]["sha256"]
                          else original(data)):
            with self.assertRaisesRegex(ValueError, "retained forbidden"):
                d.authenticate()

    def test_write_dispatch_and_operand_access_guards(self):
        probes = (
            lambda: d.SOURCE.open("wb"),
            lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.remove(d.SOURCE),
            lambda: os.system("false"),
            lambda: Path("/not-a-retained-input").read_bytes(),
            lambda: __import__("ace3.model.decoder_layer0_oracle", fromlist=["unused"]),
        )
        for probe in probes:
            audit = {"forbidden_calls": 0}
            with d.read_only(audit):
                with self.assertRaises(RuntimeError):
                    probe()
            self.assertEqual(audit["forbidden_calls"], 1)

    def test_cli_exact_single_json_serialization(self):
        with patch.object(d, "check", return_value={"status": "fixture", "report": self.result}):
            output = io.StringIO()
            with patch("sys.stdout", output):
                d.main(["--check"])
        text = output.getvalue()
        decoded, end = json.JSONDecoder().raw_decode(text)
        self.assertEqual(text[end:], "\n")
        self.assertEqual(decoded["report"], self.result)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            d.main(["--check", "--output", "forbidden"])

    def test_source_has_no_operator_or_third_party_imports(self):
        import ast
        tree = ast.parse(d.SOURCE.read_bytes())
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(modules, {"argparse", "contextlib", "fractions", "hashlib",
                                   "json", "os", "pathlib", "sys"})
