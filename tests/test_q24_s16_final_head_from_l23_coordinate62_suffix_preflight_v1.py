"""Focused read-only terminal L23/final-head preparation checks."""

import ast
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_preflight_v1 as d


class FinalHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.real_authenticate = d.authenticate
        original_run = d.subprocess.run

        def git_only(command, **kwargs):
            if command[:4] != ["git", "-C", str(d.ROOT), "check-ignore"]:
                raise AssertionError("forbidden subprocess: " + str(command))
            return original_run(command, **kwargs)

        def capture(out):
            cls.evidence = cls.real_authenticate(out)
            return cls.evidence

        def no_execution(*args, **kwargs):
            raise AssertionError("execution/write forbidden during --check")

        with ExitStack() as stack:
            stack.enter_context(patch.object(d, "authenticate", side_effect=capture))
            stack.enter_context(patch.object(d.subprocess, "run", side_effect=git_only))
            for name in ("execute", "execute_layer", "load_inputs", "validate_evidence",
                         "expected_stage", "stage_report", "write"):
                stack.enter_context(patch.object(d.parent, name, side_effect=no_execution))
            for name in ("write_text", "write_bytes", "mkdir", "touch"):
                stack.enter_context(patch.object(Path, name, side_effect=no_execution))
            with redirect_stdout(cls.stdout):
                cls.exit_code = d.main(["--check"])
        if cls.exit_code != 0:
            raise AssertionError(cls.stdout.getvalue())
        cls.reviewed_manifest = json.loads(d.read_bound(d.FINAL_REFERENCE_PINS["manifest"]))

    def test_live_cli_json_only(self):
        self.assertEqual(json.loads(self.stdout.getvalue()), self.evidence["summary"])
        self.assertEqual(len(self.stdout.getvalue().splitlines()), 1)
        self.assertEqual(self.exit_code, 0)

    def test_nine_terminal_states_and_failures(self):
        summary = self.evidence["summary"]
        self.assertEqual(list(self.evidence["states"]), list(d.CONTROLS))
        self.assertEqual(summary["control_count"], 9)
        self.assertEqual(summary["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in summary["controls"]:
            self.assertEqual(row["retained_L23"]["L23_status"], "FAIL")
            self.assertEqual([m["index"] for m in row["L23_failures"]],
                             [62] if row["control"] == "mapped_all" else [62, 241])
            self.assertEqual(row["retained_L23"]["retained_L22"]["L22_status"], "FAIL")
            self.assertEqual(row["retained_L23"]["retained_L21"]["S18"], "FAIL")

    def test_exact_reviewed_final_reference_is_ready(self):
        summary = self.evidence["summary"]
        self.assertEqual(summary["status"], "PREFLIGHT_READY")
        self.assertEqual(summary["final_reference"]["missing"], [])
        self.assertEqual(summary["asset_binding_revision"], 3)
        self.assertEqual(summary["final_reference"]["manifest"], {
            "path": str(d.ROOT / "build/q24_s16_final_original_reference_v1_attempt003/freeze.json"),
            "sha256": "87b4746306ecaba3c106f0d6f3ec6c112c2ad3856386458a23ef463c04a98feb",
            "bytes": 316192,
        })
        self.assertEqual(summary["final_reference"]["terminal_review"], {
            "mission_id": "eff468ba33e5", "round": 1, "producer_role": "reviewer",
            "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
        })
        self.assertEqual(summary["thresholds"], self.reviewed_manifest["retained_parent"]["thresholds"])
        self.assertEqual(summary["final_head_status"], "NOT_EXECUTED")

    def test_boundary_and_no_output_creation(self):
        summary = self.evidence["summary"]
        self.assertFalse(d.OUTPUT.exists())
        self.assertEqual({key: summary[key] for key in d.FLAGS}, d.FLAGS)
        self.assertFalse(summary["future_output"]["created"])
        self.assertEqual(summary["normal_host_review"], "REQUIRED")
        self.assertEqual(summary["assets"]["tensors"]["lm_head.weight"],
                         summary["assets"]["tensors"]["model.embed_tokens.weight"])
        self.assertEqual(summary["final_contract"]["rmsnorm"]["epsilon"], "1/1000000")
        self.assertEqual(summary["final_contract"]["head"]["top_k"], 10)
        self.assertEqual(summary["assets"]["tokenizer_status"], "BOUND")
        self.assertEqual(summary["assets"]["missing_tokenizer"], [])
        self.assertEqual([pin["path"] for pin in summary["assets"]["tokenizer"]],
                         [str(d.TOKENIZER_DIR / name)
                          for name in ("tokenizer.json", "tokenizer_config.json")])
        self.assertEqual(summary["assets"]["tensors"]["model.norm.weight"]["shape"], [896])
        self.assertEqual(summary["assets"]["tensors"]["lm_head.weight"]["shape"], [151936, 896])

    def test_review_round_and_backlog_fail_closed(self):
        review = json.loads(d.read_bound(d.PINS["review"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.PINS["review"]["path"]}}
        backlog = [{"id": d.MISSION, "status": "done", "finished_ts": review["created_at"],
                    "outcome": {"review_status": "done"}}]
        d.check_review(review, latest, backlog)
        for key, value in (("producer_role", "engineer"), ("round", 1),
                           ("review", {"status": "pending"}), ("mission_id", "other")):
            bad = deepcopy(review)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_review(bad, latest, backlog)
        for records in ([], backlog * 2, [{**backlog[0], "status": "running"}],
                        [{**backlog[0], "outcome": {"review_status": "pending"}}],
                        [{**backlog[0], "finished_ts": review["created_at"] - 1}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                d.check_review(review, latest, records)
        latest["handoff"]["path"] = "other"
        with self.assertRaises(ValueError):
            d.check_review(review, latest, backlog)

    def test_final_review_round_and_backlog_fail_closed(self):
        pin = d.FINAL_REFERENCE_PINS["review"]
        review = json.loads(d.read_bound(pin))
        latest = {"kind": "handoff_ref", "handoff": {"path": pin["path"]}}
        backlog = [{"id": d.FINAL_REFERENCE_MISSION, "status": "done",
                    "finished_ts": review["created_at"], "outcome": {"review_status": "done"}}]
        kwargs = {"mission": d.FINAL_REFERENCE_MISSION, "round_number": 1, "pin": pin}
        d.check_review(review, latest, backlog, **kwargs)
        for key, value in (("producer_role", "engineer"), ("round", 2),
                           ("review", {"status": "pending"}), ("mission_id", "other"),
                           ("kind", "mission_context")):
            bad = {**review, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.check_review(bad, latest, backlog, **kwargs)
        for rows in ([], backlog * 2, [{**backlog[0], "status": "running"}],
                     [{**backlog[0], "outcome": {"review_status": "pending"}}],
                     [{**backlog[0], "finished_ts": review["created_at"] - 1}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                d.check_review(review, latest, rows, **kwargs)
        latest["handoff"]["path"] = d.PINS["review"]["path"]
        with self.assertRaises(ValueError):
            d.check_review(review, latest, backlog, **kwargs)

    def test_output_rejections(self):
        for path in ("/tmp/" + d.OUTPUT.name, d.ROOT / d.OUTPUT.name,
                     d.ROOT / "build/nested" / d.OUTPUT.name,
                     d.ROOT / "build" / (d.NAME + "_attempt000"),
                     d.ROOT / "build/../build" / d.OUTPUT.name,
                     d.ROOT / "build/unversioned"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.output_path(path)
        for method in ("exists", "is_symlink"):
            with patch.object(Path, method, return_value=True), self.assertRaises(ValueError):
                d.output_path(d.OUTPUT)
        with patch.object(Path, "resolve", return_value=d.ROOT / "elsewhere"), self.assertRaises(ValueError):
            d.output_path(d.OUTPUT)
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stderr=b"not ignored")), self.assertRaises(ValueError):
            d.output_path(d.OUTPUT)

    def test_control_and_output_identity_mutations(self):
        result = self.evidence["result"]
        evidence = {"summary": result["preflight"]}
        for mutation in ("reorder", "duplicate", "output", "admission", "history"):
            bad = deepcopy(result)
            if mutation == "reorder":
                bad["controls"].reverse()
            elif mutation == "duplicate":
                bad["controls"][-1] = bad["controls"][0]
            elif mutation == "output":
                bad["output"] += "_other"
            elif mutation == "admission":
                bad["candidate_admitted"] = True
            else:
                bad["controls"][0]["retained_L21"]["S18"] = "PASS"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d.parent.check_result(bad, evidence, d.BASE)

    def test_terminal_gate_mutations(self):
        row = self.evidence["result"]["controls"][0]
        reports = json.loads(d.read_bound(row["gates"]))
        arrays = self.evidence["states"][row["control"]]
        import numpy as np
        binding = self.evidence["summary"]["L23_original_reference"]
        reference = np.load(io.BytesIO(d.read_bound(binding["reference"]["binary64"])),
                            allow_pickle=False)
        for mutation in ("status", "kv", "budget", "reference", "actual", "local"):
            bad = deepcopy(reports)
            if mutation == "status":
                bad[18]["status"] = "PASS"
            elif mutation == "kv":
                bad[0]["kv_lineage"] = "FAIL"
            elif mutation == "budget":
                bad[18]["binary64_v1"]["rows"][62]["excess_budget"] = "1"
            elif mutation == "reference":
                bad[18]["binary64_v1"]["rows"][0]["reference_binary64_hex"] = "0x0.0p+0"
            elif mutation == "actual":
                bad[18]["binary64_v1"]["rows"][0]["actual_fp16_bits"] = "0000"
            else:
                bad[0]["local_operator_fp16"]["passed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                d.check_reports(row, bad, arrays, reference)
        bad_arrays = {key: value.copy() for key, value in arrays.items()}
        bad_arrays["stage18"][0] ^= 1
        with self.assertRaises(ValueError):
            d.check_reports(row, reports, bad_arrays, reference)

    def test_reference_reanchoring_rejected(self):
        binding = self.evidence["summary"]["L23_original_reference"]
        assets = self.evidence["summary"]["assets"]
        manifest = json.loads(d.read_bound(binding["manifest"]))
        manifest["final"] = {"input_binary64": {"path": "candidate-output.npy"}}
        with patch.object(d, "read_bound", return_value=json.dumps(manifest).encode()):
            with self.assertRaises(ValueError):
                d.bind_final_reference(binding, assets)

    def test_check_failure_is_json_and_nonzero(self):
        stdout = io.StringIO()
        with patch.object(d, "authenticate", side_effect=ValueError("occupied output")):
            with redirect_stdout(stdout):
                status = d.main(["--check"])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "CHECK_FAILED")

    def synthetic_final(self):
        """In-memory format fixtures only, never propagated reference outputs."""
        binding = deepcopy(self.evidence["summary"]["L23_original_reference"])
        assets = deepcopy(self.evidence["summary"]["assets"])
        manifest = deepcopy(self.reviewed_manifest)
        final = manifest["final"]
        files = {}
        for key, shape, dtype in d.FINAL_ARRAYS:
            stream = io.BytesIO()
            np.save(stream, np.zeros(shape, dtype=dtype), allow_pickle=False)
            payload = stream.getvalue()
            path = str(d.FINAL_REFERENCE_OUTPUT / (key + ".npy"))
            files[path] = payload
            final[key] = {"path": path, "sha256": hashlib.sha256(payload).hexdigest(),
                          "bytes": len(payload), "shape": list(shape), "dtype": dtype}
        return binding, assets, manifest, files

    def bind_synthetic(self, binding, assets, manifest, files):
        payload = json.dumps(manifest).encode()
        pins = deepcopy(d.FINAL_REFERENCE_PINS)
        pins["manifest"].update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        files = {**files, pins["manifest"]["path"]: payload}
        original_read = Path.read_bytes

        def read_bytes(path):
            if str(path) in files:
                return files[str(path)]
            if path.parent == d.FINAL_REFERENCE_OUTPUT:
                raise FileNotFoundError(str(path))
            return original_read(path)

        # Only format tests replace the trust-root digest; live authority tests do not.
        with patch.object(d, "FINAL_REFERENCE_PINS", pins), \
                patch.object(Path, "read_bytes", new=read_bytes):
            return d.bind_final_reference(binding, assets)

    def test_complete_authenticated_final_format(self):
        result = self.bind_synthetic(*self.synthetic_final())
        self.assertEqual(result["status"], "BOUND_ORIGINAL_INPUT_FINAL")
        self.assertEqual(result["missing"], [])

    def test_final_identity_and_lineage_mutations(self):
        for key, value in (
            ("history", [9707, 1879]), ("position", 1), ("reference_only", False),
            ("reference_policy", "candidate-local"), ("input_binary64", {}),
            ("input_fp16", {}), ("checkpoint", {}), ("tensors", {}),
            ("tokenizer", []),
        ):
            binding, assets, manifest, files = self.synthetic_final()
            manifest["final"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bind_synthetic(binding, assets, manifest, files)

    def test_manifest_identity_and_scope_mutations(self):
        for mutation in ("path", "identity", "revision", "output", "original_manifest",
                         "L23", "parents", "failures", "source", "historical_source", "null_final"):
            binding, assets, manifest, files = self.synthetic_final()
            if mutation == "path":
                binding["manifest"]["path"] += ".substitute"
            elif mutation == "identity":
                manifest["diagnostic_id"] = "other"
            elif mutation == "revision":
                manifest["implementation_revision"] = 1
            elif mutation == "output":
                manifest["output"]["path"] = str(d.FINAL_REFERENCE_OUTPUT).replace("003", "001")
            elif mutation == "original_manifest":
                manifest["original_reference_manifest"] = {}
            elif mutation == "L23":
                manifest["retained_parent"]["L23_original_reference"] = {}
            elif mutation == "parents":
                manifest["retained_parent"]["evidence"] = {}
            elif mutation == "failures":
                manifest["retained_parent"]["retained_L23_stage_reports"]["FAIL"] = 0
            elif mutation == "source":
                manifest["source_bindings"][0]["path"] += ".substitute"
            elif mutation == "historical_source":
                manifest["source_bindings"][2]["sha256"] = "0" * 64
            else:
                manifest["final"] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.bind_synthetic(binding, assets, manifest, files)

    def test_final_array_shape_dtype_and_finiteness(self):
        for key, shape, dtype in d.FINAL_ARRAYS:
            for mutation in ("shape", "dtype", "nonfinite", "trailing", "not_npy"):
                binding, assets, manifest, files = self.synthetic_final()
                array = np.zeros(shape, dtype=dtype)
                if mutation == "shape":
                    array = array[:-1]
                elif mutation == "dtype":
                    array = array.astype("<f4")
                elif mutation == "nonfinite":
                    array[0] = 0x7c00 if dtype == "<u2" else np.inf
                stream = io.BytesIO()
                np.save(stream, array, allow_pickle=False)
                payload = stream.getvalue()
                if mutation == "trailing":
                    payload += b"unbound"
                elif mutation == "not_npy":
                    payload = b"PK-not-an-npy-array"
                pin = manifest["final"][key]
                files[pin["path"]] = payload
                pin.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
                with self.subTest(key=key, mutation=mutation), self.assertRaises(ValueError):
                    self.bind_synthetic(binding, assets, manifest, files)

    def test_final_missing_corrupt_and_substitute_arrays(self):
        for key, _, _ in d.FINAL_ARRAYS:
            for mutation in ("missing", "hash", "size", "path"):
                binding, assets, manifest, files = self.synthetic_final()
                pin = manifest["final"][key]
                if mutation == "missing":
                    del files[pin["path"]]
                    result = self.bind_synthetic(binding, assets, manifest, files)
                    self.assertEqual(result["status"], "BLOCKED_MISSING_FINAL_REFERENCE")
                    self.assertEqual(result["missing"], [pin["path"]])
                    continue
                if mutation == "hash":
                    files[pin["path"]] += b"corrupt"
                elif mutation == "size":
                    pin["bytes"] += 1
                else:
                    pin["path"] = str(d.ROOT / "build/substitute.npy")
                with self.subTest(key=key, mutation=mutation), self.assertRaises(ValueError):
                    self.bind_synthetic(binding, assets, manifest, files)

    def test_final_noncanonical_and_unbound_arrays(self):
        original_resolve = Path.resolve
        target = d.FINAL_REFERENCE_OUTPUT / "logits_fp16.npy"
        args = self.synthetic_final()

        def redirected(path, *args, **kwargs):
            return d.ROOT / "elsewhere.npy" if path == target else original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", new=redirected), self.assertRaises(ValueError):
            self.bind_synthetic(*args)

    def test_missing_reviewed_manifest_never_uses_unbound_or_default_attempt(self):
        binding = self.evidence["summary"]["L23_original_reference"]
        assets = self.evidence["summary"]["assets"]
        original_read = Path.read_bytes
        target = Path(d.FINAL_REFERENCE_PINS["manifest"]["path"])
        reads = []

        def missing(path):
            reads.append(path)
            if path == target:
                raise FileNotFoundError(str(path))
            return original_read(path)

        for present in (False, True):
            with self.subTest(present=present), patch.object(Path, "read_bytes", new=missing), \
                    patch.object(Path, "exists", return_value=present):
                result = d.bind_final_reference(binding, assets)
            self.assertEqual(result["status"], "BLOCKED_MISSING_FINAL_REFERENCE")
            self.assertEqual(result["missing"][0], str(target) + "#final")
            self.assertEqual(len(result["missing"]), 1 if present else 5)
            self.assertTrue(all(row["status"] == ("PRESENT_UNBOUND" if present else "MISSING")
                                for row in result["required_arrays"]))
        self.assertFalse(any("q24_s16_final_original_reference_v1_attempt001" in str(p)
                             for p in reads))

    def test_live_final_authority_bytes_fail_closed(self):
        binding = self.evidence["summary"]["L23_original_reference"]
        assets = self.evidence["summary"]["assets"]
        original_read = Path.read_bytes
        for key in ("review", "mission", "manifest"):
            target = Path(d.FINAL_REFERENCE_PINS[key]["path"])

            def corrupt(path):
                return b"{}" if path == target else original_read(path)

            with self.subTest(key=key), patch.object(Path, "read_bytes", new=corrupt), \
                    self.assertRaisesRegex(ValueError, "hash mismatch"):
                d.bind_final_reference(binding, assets)

    def test_stale_or_noncanonical_final_manifest_pin_rejected(self):
        binding = self.evidence["summary"]["L23_original_reference"]
        assets = self.evidence["summary"]["assets"]
        for path in (d.FINAL_REFERENCE_PINS["manifest"]["path"].replace("003", "001"),
                     str(d.FINAL_REFERENCE_OUTPUT / "../freeze.json")):
            pins = deepcopy(d.FINAL_REFERENCE_PINS)
            pins["manifest"]["path"] = path
            with self.subTest(path=path), patch.object(d, "FINAL_REFERENCE_PINS", pins), \
                    self.assertRaisesRegex(ValueError, "manifest substituted"):
                d.bind_final_reference(binding, assets)

    def test_tokenizer_missing_corrupt_and_noncanonical(self):
        with patch.object(d, "read_bound", side_effect=FileNotFoundError("missing")):
            result = d.bind_tokenizer()
        self.assertEqual(result["tokenizer_status"], "BLOCKED_MISSING_TOKENIZER_BINDING")
        self.assertEqual(result["missing_tokenizer"], [
            str(d.TOKENIZER_DIR / name) for name in ("tokenizer.json", "tokenizer_config.json")])
        original_read = Path.read_bytes
        original_resolve = Path.resolve
        for name in ("tokenizer.json", "tokenizer_config.json"):
            target = d.TOKENIZER_DIR / name

            def corrupt(path):
                return b"corrupt" if path == target else original_read(path)

            def redirected(path, *args, **kwargs):
                return d.ROOT / "elsewhere" if path == target else original_resolve(path, *args, **kwargs)

            with self.subTest(name=name, mutation="hash"):
                with patch.object(Path, "read_bytes", new=corrupt), self.assertRaises(ValueError):
                    d.bind_tokenizer()
            with self.subTest(name=name, mutation="path"):
                with patch.object(Path, "resolve", new=redirected), self.assertRaises(ValueError):
                    d.bind_tokenizer()

    def test_checkpoint_substitute_and_corruption(self):
        checkpoint = deepcopy(self.evidence["result"]["checkpoint"])
        with self.assertRaises(ValueError):
            d.bind_assets({**checkpoint, "path": str(d.ROOT / "build/substitute.safetensors")})
        with patch.object(Path, "open", return_value=io.BytesIO(b"corrupt")):
            with self.assertRaises(ValueError):
                d.bind_assets(checkpoint)

    def test_no_execute_interface_or_execution_calls(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                d.main(["--check", "--execute"])
        self.assertEqual(error.exception.code, 2)
        tree = ast.parse(d.SOURCE.read_text())
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"execute", "execute_layer", "load_inputs", "validate_evidence",
                                  "expected_stage", "stage_report", "run_token", "rmsnorm",
                                  "mkdir", "write_bytes", "write_text", "touch"})


if __name__ == "__main__":
    unittest.main()
