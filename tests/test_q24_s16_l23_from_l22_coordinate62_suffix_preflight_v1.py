"""Fresh preparation tests; no native execution or ancestor regression suites."""

import ast
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l23_from_l22_coordinate62_suffix_preflight_v1 as d


EVIDENCE = None
ORDER = ["frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
         "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
         "mapped_all", "inherited_native"]


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = EVIDENCE if EVIDENCE is not None else d.authenticate()

    def summary(self):
        return deepcopy(self.evidence["summary"])

    def previous(self):
        return self.evidence["result"]["preflight"]["L22_original_reference"]["reference"]

    def extension(self):
        previous = deepcopy(self.previous())
        return {
            "schema": "ace3-v3-original-reference-suffix-p0-v1",
            "policy_id": d.census.POLICY, "binary64_profile": d.census.PROFILE,
            "global_reference_policy": d.census.REFERENCE_POLICY, "reference_only": True,
            "accepted_ancestor_oracle_replays": 0,
            "scope": {"history": [9707], "layers": list(range(9, 24)), "position": 0},
            "layers": {
                "21": {"binary64": previous["input_binary64"], "fp16": previous["input_fp16"]},
                "22": previous,
                "23": {
                    "input_binary64": previous["binary64"], "input_fp16": previous["fp16"],
                    "prior_kv": "own empty P0",
                    "canonical": {"model.layers.23.input_layernorm.weight": {"dtype": "float16"}},
                    "binary64": {"path": str(d.REFERENCE_MANIFEST.parent / "layer23_binary64.npy")},
                    "fp16": {"path": str(d.REFERENCE_MANIFEST.parent / "layer23_fp16.npz")},
                },
            },
        }

    def test_live_nine_order(self):
        self.assertEqual([r["control"] for r in self.summary()["controls"]], ORDER)
        self.assertEqual(list(self.evidence["states"]), ORDER)

    def test_reordering_and_duplication_rejected(self):
        for labels in (list(reversed(ORDER)), ORDER[:-1] + [ORDER[0]]):
            summary = self.summary()
            for row, label in zip(summary["controls"], labels):
                row["control"] = label
            with self.assertRaises(ValueError):
                d.check_summary(summary)

    def test_exact_nine_l22_failures(self):
        for row in self.summary()["controls"]:
            numerator = 724834487455 if row["control"] == "mapped_all" else 759194225823
            self.assertEqual(Fraction(row["L22_failure"]["excess_error"]),
                             Fraction(numerator, 1099511627776))
            self.assertLess(Fraction(row["L22_failure"]["threshold_margin"]), 0)
        self.assertEqual(self.summary()["retained_L22_stage_reports"], {"PASS": 162, "FAIL": 9})

    def test_failure_relabel_rejected(self):
        summary = self.summary()
        summary["controls"][0]["retained_L22"]["mandatory_statuses"][18] = "PASS"
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_raw_parent_archives_bound(self):
        for row in self.summary()["controls"]:
            self.assertEqual(row["parent_archive"],
                             d.record(d.BASE / f"{row['control']}_L22.npz"))
            d.parent.check_state(self.evidence["states"][row["control"]])

    def test_parent_substitution_rejected(self):
        summary = self.summary()
        summary["controls"][0]["parent_archive"]["path"] = str(d.census.BASE / "mapped62_L21.npz")
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_hash_mutation_rejected(self):
        pin = deepcopy(d.PINS["result"])
        pin["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.read_bound(pin)

    def test_terminal_review_and_backlog(self):
        review = json.loads(d.read_bound(d.PINS["review"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.PINS["review"]["path"]}}
        rows = [{"id": d.MISSION, "status": "done", "outcome": {"review_status": "done"},
                 "finished_ts": review["created_at"]}]
        d.check_review(review, latest, rows)
        for key, value in (("producer_role", "engineer"), ("mission_id", "other"), ("round", 2)):
            bad = deepcopy(review)
            bad[key] = value
            with self.assertRaises(ValueError):
                d.check_review(bad, latest, rows)
        for field in ("status", "review_status"):
            bad = deepcopy(rows)
            (bad[0] if field == "status" else bad[0]["outcome"])[field] = "pending"
            with self.assertRaises(ValueError):
                d.check_review(review, latest, bad)
        review["review"]["status"] = "pending"
        with self.assertRaises(ValueError):
            d.check_review(review, latest, rows)

    def test_state_projection_mutation(self):
        state = deepcopy(self.evidence["states"][ORDER[0]])
        state["stage18"][62] ^= 1
        with self.assertRaises(ValueError):
            d.parent.check_state(state)

    def test_state_schema_and_zero_tags(self):
        for field, value in (("output_i", np.zeros(896, dtype="<i4")),
                             ("output_z", np.full(896, 255, dtype="u1")),
                             ("stage18", np.zeros(895, dtype="<u2"))):
            state = deepcopy(self.evidence["states"][ORDER[0]])
            state[field] = value
            with self.assertRaises(ValueError):
                d.parent.check_state(state)

    def test_own_empty_fp16_kv(self):
        arrays = [a for kv in self.evidence["kvs"].values() for a in kv.values()]
        self.assertEqual(len({id(a) for a in arrays}), 18)
        for a in arrays:
            self.assertEqual((a.shape, a.dtype.str), ((0, 128), "<u2"))
        self.assertFalse(d.KV["prior_layer_kv_consumed"])

    def test_original_reference_or_explicit_blocked(self):
        reference = self.summary()["L23_original_reference"]
        if reference["status"] == "BOUND_ORIGINAL_INPUT_L23":
            for key in ("binary64", "fp16"):
                self.assertEqual(reference["reference"]["input_" + key], self.previous()[key])
            self.assertEqual(reference["missing"], [])
        else:
            self.assertEqual(reference["status"], "BLOCKED_MISSING_L23_REFERENCE")
            self.assertTrue(reference["missing"])

    def test_missing_manifest_blocked(self):
        with patch.object(d, "read_bound", side_effect=FileNotFoundError):
            result = d.bind_reference({"path": str(d.REFERENCE_MANIFEST)}, self.previous())
        self.assertEqual(result["status"], "BLOCKED_MISSING_L23_REFERENCE")

    def test_missing_l23_entry_blocked(self):
        extension = self.extension()
        del extension["layers"]["23"]
        with patch.object(d, "read_bound", return_value=json.dumps(extension).encode()):
            result = d.bind_reference({"path": str(d.REFERENCE_MANIFEST)}, self.previous())
        self.assertEqual(result["missing"], [str(d.REFERENCE_MANIFEST) + "#layers/23"])

    def test_missing_l23_payload_blocked(self):
        with patch.object(d, "read_bound", side_effect=[
                json.dumps(self.extension()).encode(), FileNotFoundError(),
                b"unused", b"unused", b"unused"]):
            result = d.bind_reference({"path": str(d.REFERENCE_MANIFEST)}, self.previous())
        self.assertEqual(result["status"], "BLOCKED_MISSING_L23_REFERENCE")

    def test_corrupt_reference_not_missing(self):
        with patch.object(d, "read_bound", side_effect=ValueError("hash mismatch")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                d.bind_reference({"path": str(d.REFERENCE_MANIFEST)}, self.previous())

    def test_reanchoring_and_kv_rejected(self):
        for key, value in (("input_binary64", {"path": "candidate.npy"}),
                           ("input_fp16", {"path": "candidate.npz"}), ("prior_kv", "L22 KV")):
            extension = self.extension()
            extension["layers"]["23"][key] = value
            with self.assertRaises(ValueError):
                d.check_reference_manifest(extension, self.previous())

    def test_reference_policy_history_parent_rejected(self):
        for key, value in (("policy_id", "other"), ("global_reference_policy", "candidate"),
                           ("reference_only", False), ("accepted_ancestor_oracle_replays", 1),
                           ("scope", {"history": [0]})):
            extension = self.extension()
            extension[key] = value
            with self.assertRaises(ValueError):
                d.check_reference_manifest(extension, self.previous())
        extension = self.extension()
        extension["layers"]["22"]["input_binary64"] = {"path": "candidate"}
        with self.assertRaises(ValueError):
            d.check_reference_manifest(extension, self.previous())

    def test_fresh_output_not_created(self):
        self.assertEqual(d.output_path(d.OUTPUT), d.OUTPUT)
        self.assertFalse(d.OUTPUT.exists())
        self.assertFalse(d.FUTURE_OUTPUT["create_during_check"])

    def test_nested_external_and_noncanonical_outputs(self):
        for path in (d.OUTPUT / "nested", Path("/tmp") / d.OUTPUT.name,
                     d.ROOT / "build/../build" / d.OUTPUT.name,
                     d.ROOT / "build" / (d.NAME + "_attempt000"),
                     d.ROOT / "build" / "unversioned"):
            with patch.object(d.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    d.output_path(path)
                run.assert_not_called()

    def test_occupied_and_symlink_outputs(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaises(ValueError):
                    d.output_path(d.OUTPUT)
        with patch.object(Path, "resolve", return_value=Path("/tmp/elsewhere")):
            with self.assertRaises(ValueError):
                d.output_path(d.OUTPUT)

    def test_nonignored_and_git_error(self):
        for code in (1, 128):
            with patch.object(d.subprocess, "run", return_value=SimpleNamespace(
                    returncode=code, stderr=b"not ignored")):
                with self.assertRaises(ValueError):
                    d.output_path(d.OUTPUT)

    def test_exact_contract_and_threshold(self):
        contract = json.loads(d.CONTRACT.read_bytes())
        d.check_contract(contract)
        for key, value in (("excess_budget", "1/4"), ("schema_version", True),
                           ("interface", ["--execute"]), ("controls", list(reversed(ORDER)))):
            bad = deepcopy(contract)
            bad[key] = value
            with self.assertRaises(ValueError):
                d.check_contract(bad)

    def test_json_schema_binding(self):
        summary = json.loads(json.dumps(self.summary()))
        d.check_summary(summary)
        summary["contract"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.check_summary(summary)

    def test_zero_invocations_and_nonadmission(self):
        for key, value in d.FLAGS.items():
            summary = self.summary()
            self.assertEqual(summary[key], value)
            summary[key] = True if type(value) is bool else 1
            with self.assertRaises(ValueError):
                d.check_summary(summary)

    def test_read_only_imports_no_native_dependencies(self):
        allowed = {
            "argparse", "fractions", "importlib.util", "io", "json", "os", "pathlib", "re",
            "subprocess", "sys", "unittest", "numpy", "hashlib", "math", "struct",
        }
        modules = {
            "diagnose_q24_s16_l22_from_l21_coordinate62_suffix_preflight_v1",
            "diagnose_q24_s16_l21_sparse_cut_localization_matrix_v1",
            "diagnose_q24_s16_l21_from_l20_s18_failure_census_v1",
            "residual_exact_grid_q24_reference_v1",
        }
        for path in (d.SOURCE, d.parent.SOURCE, d.parent.matrix.SOURCE, d.census.SOURCE,
                     Path(d.parent.PINS["state_oracle"]["path"])):
            for node in ast.walk(ast.parse(path.read_bytes())):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in allowed for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    if node.module == "ace3.model.candidates":
                        self.assertTrue(all(alias.name in modules for alias in node.names))
                    else:
                        self.assertIn(node.module, allowed)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {
                        "write_bytes", "write_text", "mkdir", "unlink", "symlink_to",
                        "save", "savez", "savez_compressed", "Popen", "system",
                    })

    def test_cli_json_and_execute_rejection(self):
        with patch.object(d, "validate", return_value={"candidate_admitted": False}) as validate:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                d.main(["--check", "--out", str(d.OUTPUT)])
            validate.assert_called_once_with(str(d.OUTPUT))
            self.assertEqual(json.loads(stdout.getvalue()), {"candidate_admitted": False})
        with patch("sys.stderr", new_callable=io.StringIO):
            for args in ([], ["--execute"], ["--check", "--execute"]):
                with self.assertRaises(SystemExit):
                    d.main(args)

    def test_history_and_claim_boundary(self):
        summary = self.summary()
        self.assertEqual(summary["retained_L21_history"]["retained_L20_failing_controls"],
                         ["actual", "frozen_o", "frozen_down", "frozen_o_down"])
        self.assertIn("wider than FP16", summary["claim_boundary"])
        self.assertIn("source/identity/role-model/account/budget/access/lock/concurrency",
                      summary["claim_boundary"])
        self.assertEqual(summary["normal_host_review"], "REQUIRED")
        for row in summary["controls"]:
            self.assertEqual(row["retained_L21"]["failures"][0]["threshold_margin"], "-3/4")

    def test_metric_and_operand_lineage_mutation(self):
        row = self.evidence["result"]["controls"][0]
        reports = json.loads(d.read_bound(row["gates"]))
        reference = np.load(io.BytesIO(d.read_bound(self.previous()["binary64"])),
                            allow_pickle=False)
        for change in ("kv", "metric", "operand"):
            bad = deepcopy(reports)
            if change == "kv":
                bad[0]["kv_lineage"] = "FAIL"
            elif change == "metric":
                bad[18]["binary64_v1"]["rows"][62]["excess_budget"] = "1"
            else:
                bad[0]["local_operator_fp16"]["passed"] = False
            with self.assertRaises(ValueError):
                d.check_reports(row, bad, self.evidence["states"][ORDER[0]], reference)

    def test_future_output_contract_mutation(self):
        summary = self.summary()
        summary["future_output"]["overwrite"] = True
        with self.assertRaises(ValueError):
            d.check_summary(summary)


if __name__ == "__main__":
    unittest.main()
