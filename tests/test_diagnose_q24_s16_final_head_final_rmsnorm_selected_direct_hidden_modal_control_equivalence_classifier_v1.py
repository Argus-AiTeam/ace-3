"""Independent integer-grid oracle for all 84 retained modal-control pairs."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from functools import cmp_to_key
import io
import json
from math import gcd, lcm
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_control_equivalence_classifier_v1 as d


def ratio(text):
    parts = text.split("/")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def exact(n, q):
    factor = gcd(n, q)
    n, q = n // factor, q // factor
    return str(n) if q == 1 else f"{n}/{q}"


def difference(left, right):
    a, b = ratio(left)
    c, e = ratio(right)
    return exact(a * e - c * b, b * e)


def polarity(text):
    n, _ = ratio(text)
    return (n > 0) - (n < 0)


def order(left, right):
    return polarity(difference(left, right))


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = {"forbidden_calls": 0}
        with d.read_only(cls.audit):
            cls.retained, cls.capture, cls.review = d.authenticate()
            cls.before = deepcopy(cls.retained)
            cls.result = d.report(cls.retained)

    def grid(self, partition, control):
        retained = self.retained["retained_6184"]["retained_e795"]["retained_449b"]
        source = next(c for c in retained["report"]["controls"] if c["control"] == control)
        pair = next(p for p in source["pairs"] if
                    (p["left_id"], p["right_id"]) ==
                    (partition["left_id"], partition["right_id"]))
        terms = {}
        for row in pair["branches"][partition["branch"]]["selected_coordinates"]:
            if partition["table"] == "coordinate":
                terms[(row["coordinate"], None)] = ratio(row["direct_hidden_weighted_term"])
            else:
                for component, text in row["weighted_components"].items():
                    terms[(row["coordinate"], component)] = ratio(text)
        denominator = lcm(*(q for _, q in terms.values()))
        grid = {key: n * (denominator // q) for key, (n, q) in terms.items()}
        magnitudes = sorted((abs(n) for n in grid.values()), reverse=True)
        candidates = (("exception", (62, None)), ("modal", (241, None)))
        if partition["table"] == "coordinate_component":
            candidates = (
                ("exception", (241, "fp16_to_branch_terminal_remainder")),
                ("modal", (241, "mlp_stage17")),
            )
        fields = {}
        identities = {}
        for label, key in candidates:
            value = grid[key]
            identities[label] = {"coordinate": key[0]}
            if key[1] is not None:
                identities[label]["component"] = key[1]
            fields[label + ".signed"] = exact(value, denominator)
            fields[label + ".absolute"] = exact(abs(value), denominator)
            fields[label + ".gap_from_maximum"] = exact(magnitudes[0] - abs(value), denominator)
            fields[label + ".rank"] = str(1 + sum(abs(v) > abs(value) for v in grid.values()))
        for field in ("signed", "absolute"):
            fields[field + "_comparator"] = difference(
                fields["exception." + field], fields["modal." + field])
        fields["rank_difference"] = difference(fields["exception.rank"], fields["modal.rank"])
        fields["dominance_gap"] = exact(magnitudes[0] - magnitudes[1], denominator)
        keys = sorted(grid, key=lambda k: (
            k[0], d.COMPONENTS.index(k[1]) if k[1] is not None else -1))
        runner_up = []
        for key in keys:
            if abs(grid[key]) == magnitudes[1]:
                item = {"coordinate": key[0]}
                if key[1] is not None:
                    item["component"] = key[1]
                runner_up.append(item)
        identities["observed_winner"] = identities["exception" if control == "mapped_all" else "modal"]
        identities["runner_up_ids"] = runner_up
        return fields, identities, pair["roles"]

    def oracle_profile(self, partition, control):
        base, identities, roles = self.grid(partition, control)
        mapped, _, _ = self.grid(partition, "mapped_all")
        fields = {"comparator." + k: v for k, v in base.items()}
        for candidate in ("exception", "modal"):
            for field in ("signed", "absolute", "rank"):
                fields["contrast." + candidate + "_" + field + "_delta"] = difference(
                    mapped[candidate + "." + field], base[candidate + "." + field])
        for field in ("absolute_comparator", "signed_comparator", "dominance_gap"):
            fields["contrast." + field + "_delta"] = difference(mapped[field], base[field])
        ties = {
            "candidate_absolute_tie": base["exception.absolute"] == base["modal.absolute"],
            "candidate_signed_tie": base["exception.signed"] == base["modal.signed"],
            "candidate_rank_tie": base["exception.rank"] == base["modal.rank"],
            "zero_candidate_ids": [identities[c] for c in ("exception", "modal")
                                   if base[c + ".absolute"] == "0"],
            "maximum_candidate_ids": [identities[c] for c in ("exception", "modal")
                                      if base[c + ".gap_from_maximum"] == "0"],
            "runner_up_tied": len(identities["runner_up_ids"]) > 1,
        }
        ranks = {k: int(base[k]) for k in ("exception.rank", "modal.rank", "rank_difference")}
        ranks.update({c + "_rank_delta": int(fields["contrast." + c + "_rank_delta"])
                      for c in ("exception", "modal")})
        return {
            "control": control, "roles": roles, "fields": fields,
            "signs": {k: polarity(v) for k, v in fields.items()},
            "ranks": ranks, "identities": identities, "ties": ties,
            "contrast_membership": {
                "absolute_comparator_signs": {
                    "mapped_all": polarity(mapped["absolute_comparator"]),
                    "modal_control": polarity(base["absolute_comparator"]),
                },
                "rank_order_signs": {
                    "mapped_all": polarity(mapped["rank_difference"]),
                    "modal_control": polarity(base["rank_difference"]),
                },
                "rank_flip": polarity(mapped["rank_difference"]) * polarity(base["rank_difference"]) == -1,
                "winner_changed": identities["observed_winner"] != identities["exception"],
            },
        }

    def test_domain_preservation_and_zero_dispatch(self):
        self.assertEqual(self.before, self.retained)
        self.assertEqual(self.audit, {"forbidden_calls": 0})
        self.assertEqual(self.result["unstable_partition_count"], 3)
        self.assertEqual(self.result["modal_control_pair_comparison_count"], 84)
        self.assertEqual([(p["table"], p["left_id"], p["right_id"], p["branch"])
                          for p in self.result["partitions"]], list(d.SCOPES))
        for p in self.result["partitions"]:
            self.assertEqual(p["modal_controls"], list(d.MODAL_CONTROLS))
            self.assertEqual((p["modal_control_count"], p["pair_comparison_count"]), (8, 28))
            pairs = [(r["left_control"], r["right_control"]) for r in p["pair_comparisons"]]
            self.assertEqual(pairs, [(a, b) for i, a in enumerate(d.MODAL_CONTROLS)
                                     for b in d.MODAL_CONTROLS[i + 1:]])
        for field in ("reference_scope", "lineage_separation"):
            self.assertEqual(self.result[field], self.retained["report"][field])

    def test_all_profiles_and_84_pairs_against_independent_integer_grid(self):
        count = 0
        for p in self.result["partitions"]:
            oracles = {c: self.oracle_profile(p, c) for c in d.MODAL_CONTROLS}
            self.assertEqual(p["control_profiles"], list(oracles.values()))
            for pair in p["pair_comparisons"]:
                left, right = [oracles[pair[k]] for k in ("left_control", "right_control")]
                deltas = {k: difference(v, right["fields"][k]) for k, v in left["fields"].items()}
                self.assertEqual(pair["field_deltas"], deltas)
                self.assertEqual(pair["delta_signs"], {k: polarity(v) for k, v in deltas.items()})
                self.assertEqual(set(pair["zero_fields"]), {k for k, v in deltas.items() if v == "0"})
                self.assertEqual(set(pair["nonzero_fields"]), {k for k, v in deltas.items() if v != "0"})
                expected = {
                    "comparator_exact": all(v == "0" for k, v in deltas.items() if k.startswith("comparator.")),
                    "contrast_vector_exact": all(v == "0" for k, v in deltas.items() if k.startswith("contrast.")),
                    "sign": left["signs"] == right["signs"],
                    "rank": left["ranks"] == right["ranks"],
                    "dominance_gap": deltas["comparator.dominance_gap"] == "0"
                                     and deltas["contrast.dominance_gap_delta"] == "0",
                    "identity_membership": left["identities"] == right["identities"],
                    "tie_membership": left["ties"] == right["ties"],
                    "contrast_membership": left["contrast_membership"] == right["contrast_membership"],
                }
                self.assertEqual(pair["agreement"], expected)
                self.assertEqual(pair["equivalent"], all(expected.values()))
                count += 1
        self.assertEqual(count, 84)

    def test_spreads_classes_and_partition_decisions_against_independent_oracle(self):
        supported = 0
        for p in self.result["partitions"]:
            profiles = [self.oracle_profile(p, c) for c in d.MODAL_CONTROLS]
            classes = {}
            for profile in profiles:
                signature = json.dumps({k: v for k, v in profile.items()
                                        if k not in ("control", "roles")}, sort_keys=True)
                classes.setdefault(signature, []).append(profile["control"])
            self.assertEqual(p["equivalence_classes"], list(classes.values()))
            expected_pairs = sum(len(c) * (len(c) - 1) // 2 for c in classes.values())
            self.assertEqual(p["equivalent_pair_count"], expected_pairs)
            expected_status = "SUPPORTED" if len(classes) == 1 else "REJECTED"
            self.assertEqual(p["classification"], expected_status)
            supported += expected_status == "SUPPORTED"
            for field, spread in p["field_spreads"].items():
                values = [v["fields"][field] for v in profiles]
                ordered = sorted(values, key=cmp_to_key(order))
                low, high = ordered[0], ordered[-1]
                self.assertEqual(spread, {
                    "minimum": low, "maximum": high, "spread": difference(high, low),
                    "zero": low == high,
                    "minimum_controls": [c for c, v in zip(d.MODAL_CONTROLS, values, strict=True) if v == low],
                    "maximum_controls": [c for c, v in zip(d.MODAL_CONTROLS, values, strict=True) if v == high],
                })
            self.assertEqual(p["agreement_counts"], {
                k: sum(pair["agreement"][k] for pair in p["pair_comparisons"])
                for k in p["pair_comparisons"][0]["agreement"]})
        self.assertEqual(self.result["supported_partition_count"], supported)
        self.assertEqual(self.result["rejected_partition_count"], 3 - supported)

    def test_supported_branch_and_sign_only_equivalence_rejection(self):
        partition = deepcopy(self.retained["report"]["partitions"][0])
        first_row = partition["same_partition_comparator_rows"][0]
        first_contrast = partition["mapped_all_vs_modal_contrast_rows"][0]
        partition["same_partition_comparator_rows"] = [
            row if row["control"] == "mapped_all" else
            {**deepcopy(first_row), "control": row["control"]}
            for row in partition["same_partition_comparator_rows"]]
        partition["mapped_all_vs_modal_contrast_rows"] = [
            {**deepcopy(first_contrast), "modal_control": c} for c in d.MODAL_CONTROLS]
        uniform = d.classify_partition(partition)
        self.assertEqual(uniform["classification"], "SUPPORTED")
        self.assertEqual(uniform["equivalent_pair_count"], 28)
        self.assertEqual(uniform["equivalence_classes"], [list(d.MODAL_CONTROLS)])
        self.assertTrue(all(v["zero"] for v in uniform["field_spreads"].values()))
        left = deepcopy(self.result["partitions"][0]["control_profiles"][0])
        right = deepcopy(left)
        right["control"] = d.MODAL_CONTROLS[1]
        pair = d.compare_controls(left, right)
        self.assertTrue(pair["equivalent"])
        self.assertFalse(pair["nonzero_fields"])
        right["fields"]["comparator.exception.signed"] = difference(
            left["fields"]["comparator.exception.signed"], "1/10000000000000000000000000000000000000000")
        pair = d.compare_controls(left, right)
        self.assertTrue(pair["agreement"]["sign"])
        self.assertTrue(pair["agreement"]["identity_membership"])
        self.assertFalse(pair["equivalent"])
        self.assertEqual(pair["nonzero_fields"], ["comparator.exception.signed"])

    def test_identity_and_tie_membership_prevent_false_equivalence(self):
        left = self.result["partitions"][0]["control_profiles"][0]
        for field, value in (("identities", {}), ("ties", {}), ("ranks", {}),
                             ("contrast_membership", {})):
            right = deepcopy(left)
            right[field] = value
            pair = d.compare_controls(left, right)
            self.assertFalse(pair["equivalent"])
            self.assertFalse(pair["nonzero_fields"])

    def test_review_mission_role_and_terminal_status_required(self):
        for path, value in (
            (("kind",), "round_engineer_handoff"), (("mission_id",), "6184a5063c9d"),
            (("producer_role",), "engineer"), (("review", "status"), "pending"),
        ):
            changed = deepcopy(self.review)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaisesRegex(ValueError, "independent terminal"):
                d.validate_review(changed)

    def test_nested_pin_and_forbidden_claim_splices_fail(self):
        for depth in range(5):
            changed = deepcopy(self.retained)
            node = changed
            for child, _ in d.NESTED_PINS[:depth]:
                node = node[child]
            node["flags"]["candidate_admitted"] = True
            with self.assertRaisesRegex(ValueError, "forbidden"):
                d.report(changed)
        for depth in range(4):
            changed = deepcopy(self.retained)
            node = changed
            for child, _ in d.NESTED_PINS[:depth]:
                node = node[child]
            node["input_pins"]["stdout"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "nested pins"):
                d.report(changed)

    def test_capture_failures_and_identity_splices_fail(self):
        for fault in ("success", "exit", "timeout", "source", "accepted", "account", "command", "files"):
            capture = deepcopy(self.capture)
            if fault == "success":
                capture["success"] = False
            elif fault == "exit":
                capture["results"][-1]["exit_status"] = 1
            elif fault == "timeout":
                capture["results"][-1]["timed_out"] = True
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

    def test_exact_hash_byte_count_and_whole_command_required(self):
        data = d.bound_bytes(d.PINS["review"])
        for changed in (data + b" ", b"x" + data[1:]):
            with patch.object(Path, "read_bytes", return_value=changed):
                with self.assertRaises(ValueError):
                    d.bound_bytes(d.PINS["review"])
        original = d.bound_bytes

        def altered(pin):
            data = original(pin)
            return data + b"x" if pin["path"].endswith(".whole-command.log") else data

        with patch.object(d, "bound_bytes", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "whole-command"):
                d.authenticate()

    def test_missing_extra_duplicate_cross_partition_and_nonmodal_rows_fail(self):
        for fault in ("missing", "extra", "duplicate", "cross", "modal", "contrast"):
            changed = dict(self.retained)
            changed["report"] = deepcopy(self.retained["report"])
            p = changed["report"]["partitions"][0]
            if fault == "missing":
                changed["report"]["partitions"].pop()
            elif fault == "extra":
                changed["report"]["partitions"].append(deepcopy(p))
            elif fault == "duplicate":
                p["same_partition_comparator_rows"][1] = p["same_partition_comparator_rows"][0]
            elif fault == "cross":
                p["same_partition_comparator_rows"][0]["branch"] = "fp16"
            elif fault == "modal":
                p["control_membership"]["modal"][0] = "mapped_all"
            else:
                p["mapped_all_vs_modal_contrast_rows"][0]["modal_control"] = "mapped_all"
            with self.assertRaises(ValueError):
                d.report(changed)

    def test_corrupt_comparator_and_contrast_fields_fail(self):
        for fault in ("signed", "rank", "gap", "identity", "sign", "delta", "delta_sign", "delta_rank"):
            p = deepcopy(self.retained["report"]["partitions"][0])
            row = p["same_partition_comparator_rows"][0]
            contrast = p["mapped_all_vs_modal_contrast_rows"][0]
            if fault == "signed":
                row["exception_winner_row"]["signed"] = "0"
            elif fault == "rank":
                row["exception_winner_row"]["rank"] = True
            elif fault == "gap":
                row["modal_winner_row"]["gap_from_maximum"] = "1"
            elif fault == "identity":
                row["observed_winner_id"] = {"coordinate": 62}
            elif fault == "sign":
                row["signed_comparator_sign"] = 0
            elif fault == "delta":
                contrast["dominance_gap_delta"] = "0"
            elif fault == "delta_sign":
                contrast["delta_signs"]["dominance_gap_delta"] = 0
            else:
                contrast["modal_rank_delta"] = True
            with self.assertRaises(ValueError):
                d.classify_partition(p)

    def test_canonical_rationals_strict_json_and_no_float(self):
        for value in (0.1, 1, True, "2/4", "0/1", "-0", "1.0", "1e-5"):
            with self.assertRaises(ValueError):
                d.rational(value)
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{} {}'):
            with self.assertRaises(ValueError):
                d.decode(value)
        tree = ast.parse(d.SOURCE.read_bytes())
        self.assertFalse(any(isinstance(n, ast.Constant) and type(n.value) is float for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id == "float" for n in ast.walk(tree)))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(names, {
            "argparse", "contextlib", "fractions", "hashlib", "itertools", "json",
            "os", "pathlib", "sys",
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

    def test_guard_blocks_writes_ancestor_reads_imports_and_dispatch(self):
        audit = {"forbidden_calls": 0}
        operations = (
            lambda: d.SOURCE.write_bytes(b"forbidden"),
            lambda: Path(d.NESTED_PINS[0][1]["stdout"]["path"]).read_bytes(),
            lambda: os.system(":"),
            lambda: sys.audit("subprocess.Popen", "forbidden", [], None, None),
            lambda: sys.audit("import", "ace3.model.projection_oracle", None, None, None, None),
            lambda: sys.audit("socket.connect", None, None),
        )
        with d.read_only(audit):
            for operation in operations:
                with self.assertRaises(RuntimeError):
                    operation()
        self.assertEqual(audit["forbidden_calls"], len(operations))

    def test_cli_emits_one_json_document(self):
        output = io.StringIO()
        with patch.object(d, "check", return_value=self.result), redirect_stdout(output):
            d.main(["--check"])
        self.assertEqual(d.decode(output.getvalue()), self.result)
        self.assertTrue(output.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
