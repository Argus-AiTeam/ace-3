"""Focused preparation checks; no native evaluator or ancestor test suites."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1 as d


EVIDENCE = None
ORDER = [
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
]


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def summary(self):
        return deepcopy(self.evidence["summary"])

    def extension(self):
        previous = deepcopy(self.previous())
        return {
            "schema": "ace3-v3-original-reference-suffix-p0-v1",
            "policy_id": d.census.POLICY, "binary64_profile": d.census.PROFILE,
            "global_reference_policy": d.census.REFERENCE_POLICY,
            "reference_only": True, "accepted_ancestor_oracle_replays": 0,
            "scope": {"history": [9707], "layers": list(range(9, 24)), "position": 0},
            "layers": {
                "21": previous,
                "22": {
                    "input_binary64": previous["binary64"], "input_fp16": previous["fp16"],
                    "prior_kv": "own empty P0",
                    "binary64": {
                        "path": str(d.REFERENCE_MANIFEST.parent / "layer22_binary64.npy"),
                        "bytes": 7296,
                        "sha256": "99a12c20ac500e15555c89efd6aeb354e09ab4025d48ef78c7823b7b7b979934",
                    },
                    "fp16": {
                        "path": str(d.REFERENCE_MANIFEST.parent / "layer22_fp16.npz"),
                        "bytes": 53392,
                        "sha256": "38212043e4575c539133fce45b384deb354ddedee05426c82dadf556687f6a7b",
                    },
                },
            },
        }

    def previous(self):
        return self.evidence["retained"]["summary"]["original_reference"]

    def test_live_exact_nine_order(self):
        self.assertEqual([r["control"] for r in self.summary()["controls"]], ORDER)
        self.assertEqual(len(self.evidence["states"]), 9)

    def test_reordered_and_duplicate_controls_rejected(self):
        for labels in (list(reversed(ORDER)), ORDER[:-1] + [ORDER[0]]):
            s = self.summary()
            for row, label in zip(s["controls"], labels):
                row["control"] = label
            with self.assertRaises(ValueError):
                d.check_summary(s)

    def test_original_failures_exact_fraction_oracle(self):
        for row in self.summary()["controls"]:
            retained = row["retained_L21"]
            self.assertEqual(retained["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
            self.assertEqual(retained["failing_coordinates"], [62])
            metric = retained["failures"][0]
            reference = Fraction(float.fromhex(metric["reference_binary64_hex"]))
            actual = d.rational.fp16_value(int(metric["actual_fp16_bits"], 16))
            nearest = d.rational.fp16_value(int(metric["nearest_fp16_bits"], 16))
            error, floor = abs(actual - reference), abs(nearest - reference)
            self.assertEqual(error, Fraction(3915202785233, 4398046511104))
            self.assertEqual(floor, Fraction(66912088017, 4398046511104))
            self.assertEqual(error - floor, Fraction(7, 8))
            self.assertEqual(Fraction(1, 8) - (error - floor), Fraction(-3, 4))

    def test_failure_cannot_be_relabeled_pass(self):
        s = self.summary()
        s["controls"][0]["retained_L21"]["S18"] = "PASS"
        with self.assertRaises(ValueError):
            d.check_summary(s)

    def test_all_original_parent_archives_bound(self):
        manifest = d.census.artifact_manifest(self.evidence["retained"]["result"]["artifacts"])
        for row in self.summary()["controls"]:
            expected = str(d.census.BASE / f"{row['control']}_L21.npz")
            self.assertEqual(row["parent_archive"], manifest[expected])
            self.assertEqual(row["parent_state"]["fields"],
                             {"i": "output_i", "z": "output_z", "h": "stage18"})

    def test_artifact_hash_mutation_rejected(self):
        record = deepcopy(self.summary()["controls"][0]["parent_archive"])
        record["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.census.read_bound(record)

    def test_review_identity_and_terminal_status(self):
        review = json.loads(d.census.read_bound(d.PINS["localization_review"]))
        d.matrix.check_review(review, "6c38406b5e07", 1)
        for field, value in (("producer_role", "engineer"), ("mission_id", "ecf478e49bef")):
            changed = deepcopy(review)
            changed[field] = value
            with self.assertRaises(ValueError):
                d.matrix.check_review(changed, "6c38406b5e07", 1)
        review["review"]["status"] = "pending"
        with self.assertRaises(ValueError):
            d.matrix.check_review(review, "6c38406b5e07", 1)

    def test_state_projection_mutation_rejected(self):
        state = deepcopy(self.evidence["states"][ORDER[0]])
        state["stage18"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "projection mismatch"):
            d.check_state(state)

    def test_state_shape_dtype_and_zero_tag_rejected(self):
        for field, value in (
            ("output_i", self.evidence["states"][ORDER[0]]["output_i"][:-1]),
            ("output_i", self.evidence["states"][ORDER[0]]["output_i"].astype("<i4")),
            ("output_z", d.np.full(896, 2, dtype="u1")),
        ):
            state = deepcopy(self.evidence["states"][ORDER[0]])
            state[field] = value
            with self.assertRaises(ValueError):
                d.check_state(state)

    def test_empty_per_consumer_fp16_kv(self):
        caches = [d.empty_kv() for _ in ORDER]
        self.assertEqual(len({id(c[k]) for c in caches for k in ("k", "v")}), 18)
        for cache in caches:
            self.assertEqual(set(cache), {"k", "v"})
            for array in cache.values():
                self.assertEqual(array.shape, (0, 128))
                self.assertEqual(array.dtype.str, "<u2")
        for row in self.summary()["controls"]:
            self.assertIs(row["consumer_kv"]["prior_layer_kv_consumed"], False)
            self.assertIs(row["consumer_kv"]["shared_between_consumers"], False)

    def test_live_original_l22_reference_or_explicit_blocked(self):
        s = self.summary()
        d.check_summary(s)
        binding = s["L22_original_reference"]
        if binding["status"] == "BLOCKED_MISSING_L22_REFERENCE":
            self.assertTrue(binding["missing"])
            self.assertEqual(s["status"], "BLOCKED_MISSING_L22_REFERENCE")
        else:
            current = binding["reference"]
            self.assertEqual(current["input_binary64"], self.previous()["binary64"])
            self.assertEqual(current["input_fp16"], self.previous()["fp16"])
            self.assertEqual(current["binary64"]["sha256"],
                             "99a12c20ac500e15555c89efd6aeb354e09ab4025d48ef78c7823b7b7b979934")
            self.assertEqual(current["fp16"]["sha256"],
                             "38212043e4575c539133fce45b384deb354ddedee05426c82dadf556687f6a7b")

    def test_missing_manifest_is_blocked(self):
        result = d.bind_reference([], self.previous())
        self.assertEqual(result["status"], "BLOCKED_MISSING_L22_REFERENCE")
        self.assertIsNone(result["reference"])
        with patch.object(d.census, "read_bound", side_effect=FileNotFoundError):
            result = d.bind_reference([{"path": str(d.REFERENCE_MANIFEST)}], self.previous())
        self.assertEqual(result["missing"], [str(d.REFERENCE_MANIFEST)])

    def test_missing_l22_entry_is_blocked(self):
        extension = self.extension()
        del extension["layers"]["22"]
        with patch.object(d.census, "read_bound", return_value=json.dumps(extension).encode()):
            result = d.bind_reference([{"path": str(d.REFERENCE_MANIFEST)}], self.previous())
        self.assertEqual(result["status"], "BLOCKED_MISSING_L22_REFERENCE")
        self.assertEqual(result["missing"], [str(d.REFERENCE_MANIFEST) + "#layers/22"])

    def test_missing_l22_payload_is_blocked(self):
        extension = self.extension()
        payload = json.dumps(extension).encode()
        with patch.object(d.census, "read_bound",
                          side_effect=[payload, FileNotFoundError(), b"", b"", b""]):
            result = d.bind_reference([{"path": str(d.REFERENCE_MANIFEST)}], self.previous())
        self.assertEqual(result["status"], "BLOCKED_MISSING_L22_REFERENCE")
        self.assertEqual(result["missing"], [extension["layers"]["22"]["binary64"]["path"]])

    def test_corrupt_reference_is_not_missing_fallback(self):
        with patch.object(d.census, "read_bound", side_effect=ValueError("hash mismatch")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                d.bind_reference([{"path": str(d.REFERENCE_MANIFEST)}], self.previous())

    def test_reference_reanchoring_and_kv_rejected(self):
        for field, value in (("input_binary64", {"path": "candidate"}),
                             ("input_fp16", {"path": "candidate"}), ("prior_kv", "L21 own KV")):
            extension = self.extension()
            extension["layers"]["22"][field] = value
            with self.assertRaises(ValueError):
                d.check_reference_manifest(extension, self.previous())

    def test_reference_policy_history_and_parent_rejected(self):
        for field, value in (("policy_id", "relaxed"), ("global_reference_policy", "local"),
                             ("scope", {"position": 1}), ("reference_only", False)):
            extension = self.extension()
            extension[field] = value
            with self.assertRaises(ValueError):
                d.check_reference_manifest(extension, self.previous())
        extension = self.extension()
        extension["layers"]["21"]["prior_kv"] = "shared"
        with self.assertRaises(ValueError):
            d.check_reference_manifest(extension, self.previous())

    def test_output_fresh_ignored_direct_child(self):
        s = self.summary()
        self.assertEqual(d.output_path(s["future_output"]["selected"]).parent, d.ROOT / "build")
        self.assertFalse(Path(s["future_output"]["selected"]).exists())
        self.assertFalse(s["future_output"]["create_during_check"])

    def test_output_unsafe_paths_rejected_before_git(self):
        with patch.object(d.subprocess, "run", side_effect=AssertionError("unexpected git")):
            for path in (
                d.ROOT, d.ROOT / "build", d.ROOT.parent / d.OUTPUT.name,
                d.OUTPUT / "nested", d.ROOT / "build" / ".." / d.OUTPUT.name,
                d.ROOT / "build" / "unversioned", d.ROOT / "build" / (d.NAME + "_attempt000"),
            ):
                with self.subTest(path=str(path)), self.assertRaises(ValueError):
                    d.output_path(path)

    def test_occupied_and_symlink_output_rejected(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "resolve", return_value=d.ROOT / "elsewhere"):
            with self.assertRaisesRegex(ValueError, "canonical"):
                d.output_path(d.OUTPUT)

    def test_nonignored_and_git_failure_rejected(self):
        for code in (1, 128):
            with patch.object(d.subprocess, "run",
                              return_value=SimpleNamespace(returncode=code, stderr=b"refused")):
                with self.assertRaisesRegex(ValueError, "not confirmed ignored"):
                    d.output_path(d.OUTPUT)

    def test_contract_exact_schema_and_thresholds(self):
        contract = json.loads(d.CONTRACT.read_bytes())
        d.check_contract(contract)
        self.assertEqual(contract["excess_budget"], "1/8")
        for field, value in (("version", True), ("schema_version", 2),
                             ("controls", ORDER[::-1]), ("excess_budget", "1"),
                             ("local_threshold", "relaxed")):
            changed = deepcopy(contract)
            changed[field] = value
            with self.assertRaises(ValueError):
                d.check_contract(changed)

    def test_json_result_contract_binding(self):
        s = json.loads(json.dumps(self.summary(), allow_nan=False))
        d.check_summary(s)
        for field, value in (("diagnostic_id", "other"), ("schema_version", True),
                             ("contract", {"path": "other"})):
            changed = deepcopy(s)
            changed[field] = value
            with self.assertRaises(ValueError):
                d.check_summary(changed)

    def test_zero_invocations_and_false_admission_flags(self):
        s = self.summary()
        for key, value in d.FLAGS.items():
            self.assertIs(type(s[key]), type(value))
            self.assertEqual(s[key], value)
            changed = deepcopy(s)
            changed[key] = 1 if type(value) is int else True
            with self.assertRaises(ValueError):
                d.check_summary(changed)

    def test_only_read_only_imports_and_git_query(self):
        allowed = {
            "argparse", "importlib.util", "io", "json", "pathlib", "re", "subprocess",
            "sys", "unittest", "numpy", "fractions", "hashlib", "math", "struct",
        }
        for path in (d.SOURCE, d.matrix.SOURCE, d.census.SOURCE, Path(d.PINS["state_oracle"]["path"])):
            tree = ast.parse(path.read_bytes())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in allowed for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    if node.module == "ace3.model.candidates":
                        self.assertTrue(all(alias.name in {
                            "diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1",
                            "diagnose_q24_s16_l21_from_l20_s18_failure_census_v1",
                            "residual_exact_grid_q24_reference_v1",
                        } for alias in node.names))
                    else:
                        self.assertIn(node.module, allowed)
        with patch.object(d.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0, stderr=b"")) as run:
            d.output_path(d.OUTPUT)
        self.assertEqual(run.call_args.args[0],
                         ["git", "-C", str(d.ROOT), "check-ignore", "--quiet", "--", str(d.OUTPUT)])

    def test_check_cli_and_execute_rejection(self):
        with patch.object(d, "validate", return_value={"candidate_admitted": False}) as validate:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                d.main(["--check", "--out", str(d.OUTPUT)])
            validate.assert_called_once_with(str(d.OUTPUT))
            self.assertEqual(json.loads(stdout.getvalue()), {"candidate_admitted": False})
        with patch.object(d, "validate", side_effect=AssertionError("must not validate")):
            with patch("sys.stderr", new_callable=io.StringIO):
                for args in ([], ["--execute"], ["--check", "--execute"]):
                    with self.assertRaises(SystemExit):
                        d.main(args)

    def test_old_closure_never_substitutes_for_new_parents(self):
        s = self.summary()
        self.assertFalse(s["old_sparse_cut_used_as_parent"])
        self.assertEqual(s["localization_comparison"]["diagnostic_id"], d.matrix.ID)
        s["controls"][0]["parent_archive"]["path"] = str(d.matrix.BASE / "adjusted_L21.npz")
        with self.assertRaises(ValueError):
            d.check_summary(s)

    def test_historical_failures_and_scope_preserved(self):
        s = self.summary()
        self.assertTrue(s["historical_failures_preserved"])
        self.assertEqual([r["control"] for r in s["retained_L20_failing_controls"]],
                         ["actual", "frozen_o", "frozen_down", "frozen_o_down"])
        for row in s["retained_L20_failing_controls"]:
            self.assertEqual(row["L20_status"], "FAIL")
            self.assertEqual(row["retained_L15_S18_failure_indices"], [62])
            self.assertEqual(row["retained_L18_S18_failure_indices"], [62])
        self.assertIn("wider than FP16", s["claim_boundary"])
        self.assertEqual(s["normal_host_review"], "REQUIRED")


if __name__ == "__main__":
    unittest.main()
