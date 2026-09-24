"""One live reference-only creation plus independent oracle and refusal tests."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from ace3.model.candidates import q24_s16_final_original_reference_v1 as d


class FinalOriginalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output_patch = patch.object(d, "OUTPUT", Path(os.environ.get(
            "ACE3_FINAL_REFERENCE_TEST_OUT", str(d.OUTPUT))))
        output_patch.start()
        cls.addClassCleanup(output_patch.stop)
        cls.created = not d.OUTPUT.exists()
        cls.stdout = io.StringIO()
        authenticate = d.authenticate
        subprocess_run = d.parent.subprocess.run

        def capture():
            cls.summary = authenticate()
            return cls.summary

        def forbidden(*args, **kwargs):
            raise AssertionError("candidate/prefix/admission/native/service execution forbidden")

        def git_only(command, **kwargs):
            if command[:4] != ["git", "-C", str(d.ROOT), "check-ignore"]:
                forbidden()
            return subprocess_run(command, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(d, "authenticate", side_effect=capture))
            stack.enter_context(patch.object(d.parent.subprocess, "run", side_effect=git_only))
            stack.enter_context(patch.object(d.original, "bind_references", side_effect=forbidden))
            for name in ("execute", "execute_layer", "load_inputs", "validate_evidence",
                         "expected_stage", "stage_report", "write"):
                stack.enter_context(patch.object(d.parent.parent, name, side_effect=forbidden))
            stack.enter_context(patch.object(d.parent.parent.producer.native.candidate,
                                            "_stages", side_effect=forbidden))
            with redirect_stdout(cls.stdout):
                cls.exit_code = d.main(["--create" if cls.created else "--verify"])
        if cls.exit_code != 0:
            raise AssertionError(cls.stdout.getvalue())
        cls.result = json.loads(cls.stdout.getvalue())
        cls.manifest = json.loads((d.OUTPUT / "freeze.json").read_bytes())
        cls.arrays = {key: np.load(d.OUTPUT / (key + ".npy"), allow_pickle=False)
                      for key, _, _ in d.ARRAYS}

    def test_live_json_only(self):
        self.assertEqual(self.exit_code, 0)
        self.assertEqual(len(self.stdout.getvalue().splitlines()), 1)
        self.assertEqual(self.result["status"], "CREATED_AND_VERIFIED" if self.created else "VERIFIED")
        self.assertEqual(self.result["final_binding"], str(d.OUTPUT / "freeze.json") + "#final")
        self.assertEqual({p.name for p in d.OUTPUT.iterdir()}, d.FILES)

    def test_independent_full_array_oracle(self):
        binding = self.summary["L23_original_reference"]["reference"]
        h64 = np.load(io.BytesIO(d.read_bound(binding["binary64"])), allow_pickle=False)
        with np.load(io.BytesIO(d.read_bound(binding["fp16"])), allow_pickle=False) as archive:
            h16 = archive["stage18"].view("<f2").astype("<f8")
        with d.safe_open(self.summary["assets"]["checkpoint"]["path"], framework="numpy") as model:
            norm = model.get_tensor("model.norm.weight").astype("<f8")

            def independent_norm(hidden):
                variance = math.fsum(float(x) ** 2 for x in hidden) / len(hidden)
                return hidden / math.sqrt(variance + 1e-6) * norm

            norm64 = independent_norm(h64)
            norm16 = independent_norm(h16).astype("<f4").astype("<f2")
            np.testing.assert_allclose(self.arrays["rmsnorm_binary64"], norm64,
                                       rtol=0, atol=1e-12)
            np.testing.assert_array_equal(self.arrays["rmsnorm_fp16"], norm16.view("<u2"))
            head = model.get_slice("lm_head.weight")
            for start in range(0, 151936, 1024):
                end = min(start + 1024, 151936)
                weights = np.asarray(head[start:end], dtype="<f8")
                expected64 = np.sum(weights * norm64, axis=1, dtype=np.float64)
                expected16 = np.sum(weights * norm16.astype("<f8"), axis=1,
                                    dtype=np.float64).astype("<f4").astype("<f2").view("<u2")
                np.testing.assert_allclose(self.arrays["logits_binary64"][start:end],
                                           expected64, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(self.arrays["logits_fp16"][start:end], expected16)

    def test_fp16_trajectory_is_not_binary64_rounding(self):
        self.assertFalse(np.array_equal(
            self.arrays["rmsnorm_fp16"],
            self.arrays["rmsnorm_binary64"].astype("<f2").view("<u2")))
        self.assertFalse(np.array_equal(
            self.arrays["logits_fp16"],
            self.arrays["logits_binary64"].astype("<f2").view("<u2")))

    def test_frozen_fp16_stage_conversion_midpoint(self):
        value = float.fromhex("0x1.772000024b4f7p+1")
        source = np.array([value], dtype="<f8")
        actual = d.torch.from_numpy(source).to(d.torch.float16).numpy().view("<u2")
        direct = source.astype("<f2").view("<u2")
        independent = source.astype("<f4").astype("<f2").view("<u2")
        np.testing.assert_array_equal(actual, independent)
        self.assertEqual(int(actual[0]), 0x41DC)
        self.assertEqual(int(direct[0]), 0x41DD)

    def test_all_nine_failures_and_history_unchanged(self):
        retained = self.manifest["retained_parent"]
        self.assertEqual(retained, self.summary)
        self.assertEqual(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        self.assertEqual([row["control"] for row in retained["controls"]], list(d.parent.CONTROLS))
        for row in retained["controls"]:
            self.assertEqual(row["retained_L23"]["L23_status"], "FAIL")
            self.assertEqual(row["retained_L23"]["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
            self.assertEqual([metric["index"] for metric in row["L23_failures"]],
                             [62] if row["control"] == "mapped_all" else [62, 241])
        self.assertTrue(retained["historical_failures_preserved"])
        self.assertEqual(retained["thresholds"], self.summary["thresholds"])

    def test_nonadmission_review_boundary(self):
        flags = self.manifest["flags"]
        for key in ("candidate_admitted", "policy_adopted", "successor_published",
                    "strict_FP16_state_claim", "new_token_claim", "full_model_claim",
                    "reference_reanchoring", "original_prefix_replay", "admission_replay"):
            self.assertIs(flags[key], False)
        for key in ("native_L0_L23_invocations", "prefix_invocations", "admission_invocations",
                    "final_rmsnorm_invocations", "lm_head_invocations", "top_k_invocations",
                    "tokenizer_decode_invocations", "rtl_invocations", "hardware_invocations",
                    "gpu_invocations", "simulation_invocations", "retained_evidence_writes"):
            self.assertEqual(flags[key], 0)
        self.assertEqual(self.manifest["normal_host_review"], "REQUIRED")
        self.assertEqual(self.result["normal_host_review"], "REQUIRED")
        self.assertEqual(flags["reference_rmsnorm_invocations"], 2)
        self.assertEqual(flags["reference_lm_head_invocations"], 2)

    def test_retained_manifest_not_extended_or_replaced(self):
        binding = self.summary["L23_original_reference"]
        old = json.loads(d.read_bound(binding["manifest"]))
        self.assertNotIn("final", old)
        self.assertEqual(self.manifest["original_reference_manifest"], binding["manifest"])
        self.assertEqual(self.manifest["final"]["input_binary64"], old["layers"]["23"]["binary64"])
        self.assertEqual(self.manifest["final"]["input_fp16"], old["layers"]["23"]["fp16"])
        self.assertEqual(d.parent.bind_final_reference(binding, self.summary["assets"])["status"],
                         "BLOCKED_MISSING_FINAL_REFERENCE")

    def test_bound_checkpoint_head_and_tokenizer(self):
        final = self.manifest["final"]
        self.assertEqual(final["checkpoint"]["sha256"],
                         "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b")
        self.assertEqual(final["tensors"]["model.norm.weight"]["sha256"],
                         "1dd25d7720c68bc10838374200238c26626a624119cac0b45bff44bc43c354fe")
        self.assertEqual(final["tensors"]["lm_head.weight"],
                         final["tensors"]["model.embed_tokens.weight"])
        self.assertEqual(final["tokenizer"], self.summary["assets"]["tokenizer"])
        self.assertEqual(final["history"], [9707])
        self.assertEqual(final["position"], 0)

    def test_output_array_contracts(self):
        for key, shape, dtype in d.ARRAYS:
            pin = self.manifest["final"][key]
            self.assertEqual(pin["shape"], list(shape))
            self.assertEqual(pin["dtype"], dtype)
            array = d.checked_array(d.read_bound(pin), shape, dtype)
            np.testing.assert_array_equal(array, self.arrays[key])

    def test_verify_recomputes_without_writing(self):
        output = io.StringIO()
        with patch.object(d, "authenticate", return_value=self.summary), \
                patch.object(Path, "mkdir", side_effect=AssertionError("verify mkdir")), \
                redirect_stdout(output):
            self.assertEqual(d.main(["--verify"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["evidence_writes"], 0)
        self.assertEqual(result["manifest"], self.result["manifest"])

    def test_output_refuses_occupied_and_invalid_paths(self):
        for path in (d.OUTPUT, d.ROOT / "outside", d.ROOT / "build" / (d.NAME + "_attempt000"),
                     d.ROOT / "build" / "nested" / (d.NAME + "_attempt002"),
                     d.ROOT / "build" / (d.NAME + "_attempt2"),
                     d.ROOT / "build" / ".." / "build" / (d.NAME + "_attempt002")):
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                d.output_path(path)

    def test_output_refuses_symlink(self):
        with TemporaryDirectory(prefix="final_reference_test_", dir=d.ROOT / "build") as temp:
            link = Path(temp) / (d.NAME + "_attempt002")
            link.symlink_to(d.OUTPUT, target_is_directory=True)
            with self.assertRaises(ValueError):
                d.output_path(link, existing=True)

    def test_output_requires_ignore_confirmation(self):
        with patch.object(d.parent.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = b"not ignored"
            with self.assertRaisesRegex(ValueError, "not confirmed ignored"):
                d.output_path(d.OUTPUT, existing=True)

    def test_manifest_rejects_reanchoring_substitution_and_flags(self):
        for key, value in (
            ("input_binary64", self.manifest["final"]["input_fp16"]),
            ("input_fp16", self.manifest["final"]["input_binary64"]),
            ("checkpoint", {}), ("tensors", {}), ("tokenizer", []),
            ("history", [358]), ("position", 1), ("reference_only", False),
            ("reference_policy", "control-derived"),
        ):
            changed = deepcopy(self.manifest)
            changed["final"][key] = value
            with self.subTest(key=key), patch.object(
                    Path, "read_bytes", return_value=json.dumps(changed).encode()), \
                    self.assertRaisesRegex(ValueError, "binding changed"):
                d.check_output(d.OUTPUT, self.manifest)
        changed = deepcopy(self.manifest)
        changed["flags"]["candidate_admitted"] = True
        with patch.object(Path, "read_bytes", return_value=json.dumps(changed).encode()), \
                self.assertRaisesRegex(ValueError, "binding changed"):
            d.check_output(d.OUTPUT, self.manifest)

    def test_rehashed_wrong_array_cannot_substitute(self):
        changed = deepcopy(self.manifest)
        changed["final"]["logits_fp16"]["sha256"] = "0" * 64
        changed["final"]["logits_fp16"]["path"] = str(d.OUTPUT / "rmsnorm_fp16.npy")
        with patch.object(Path, "read_bytes", return_value=json.dumps(changed).encode()), \
                self.assertRaisesRegex(ValueError, "binding changed"):
            d.check_output(d.OUTPUT, self.manifest)

    def test_corrupt_array_hash_fails(self):
        pin = self.manifest["final"]["logits_fp16"]
        payload = bytearray(Path(pin["path"]).read_bytes())
        payload[-1] ^= 1
        with patch.object(Path, "read_bytes", return_value=bytes(payload)), \
                self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.read_bound(pin)

    def test_array_structure_and_finiteness_fail_closed(self):
        for dtype in ("<f8", "<u2"):
            good = np.ones(896, dtype=dtype)
            nonfinite = good.copy()
            nonfinite[0] = np.inf if dtype == "<f8" else 0x7C00
            nan = good.copy()
            nan[0] = np.nan if dtype == "<f8" else 0x7E00
            cases = [d.npy_bytes(good[:-1]), d.npy_bytes(good.astype("<f4")),
                     d.npy_bytes(nonfinite), d.npy_bytes(nan),
                     d.npy_bytes(good) + b"extra", d.npy_bytes(good)[:-1], b"not-npy"]
            for payload in cases:
                with self.subTest(dtype=dtype, size=len(payload)), \
                        self.assertRaises((ValueError, EOFError)):
                    d.checked_array(payload, (896,), dtype)

    def test_noncanonical_npy_version_rejected(self):
        stream = io.BytesIO()
        np.lib.format.write_array(stream, np.ones(896, dtype="<f8"), version=(2, 0))
        with self.assertRaisesRegex(ValueError, "noncanonical NPY"):
            d.checked_array(stream.getvalue(), (896,), "<f8")

    def test_fp16_archive_trailing_duplicate_and_missing_members(self):
        pin = self.summary["L23_original_reference"]["reference"]["fp16"]
        with self.assertRaisesRegex(ValueError, "trailing"):
            d.fp16_input(d.read_bound(pin) + b"extra")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("stage18.npy", d.npy_bytes(np.ones(896, dtype="<u2")))
        with self.assertRaisesRegex(ValueError, "members"):
            d.fp16_input(stream.getvalue())

    def test_original_input_reanchoring_rejected_before_arithmetic(self):
        summary = deepcopy(self.summary)
        summary["L23_original_reference"]["reference"]["binary64"] = \
            self.manifest["final"]["rmsnorm_binary64"]
        with patch.object(d.torch, "from_numpy", side_effect=AssertionError("arithmetic")), \
                self.assertRaisesRegex(ValueError, "reanchored"):
            d.reference_arrays(summary)

    def test_checkpoint_substitute_and_corruption_rejected(self):
        checkpoint = deepcopy(self.summary["assets"]["checkpoint"])
        checkpoint["path"] = str(d.OUTPUT / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            d.parent.bind_assets(checkpoint)
        with patch.object(Path, "open", return_value=io.BytesIO(b"corrupt")), \
                self.assertRaisesRegex(ValueError, "checkpoint identity"):
            d.parent.bind_assets(self.summary["assets"]["checkpoint"])

    def test_tokenizer_corruption_and_noncanonical_path_rejected(self):
        pin = self.summary["assets"]["tokenizer"][0]
        with patch.object(Path, "read_bytes", return_value=b"corrupt"), \
                self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.read_bound(pin)
        bad = {**pin, "path": str(d.ROOT / "build" / ".." / Path(pin["path"]).name)}
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            d.read_bound(bad)

    def test_nonterminal_review_rejected(self):
        review = json.loads(d.read_bound(d.parent.PINS["review"]))
        latest = {"kind": "handoff_ref", "handoff": {"path": d.parent.PINS["review"]["path"]}}
        backlog = [{"id": d.parent.MISSION, "status": "done",
                    "outcome": {"review_status": "done"}, "finished_ts": review["created_at"]}]
        review["review"]["status"] = "pending"
        with self.assertRaises(ValueError):
            d.parent.check_review(review, latest, backlog)

    def test_wrong_account_fails_json_before_output(self):
        output = io.StringIO()
        with patch.object(d.os, "getuid", return_value=0), \
                patch.object(d, "run", side_effect=AssertionError("output")), redirect_stdout(output):
            self.assertEqual(d.main(["--create"]), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("account", result["error"])

    def test_occupied_cli_fails_json_without_authentication(self):
        output = io.StringIO()
        with patch.object(d, "authenticate", side_effect=AssertionError("authentication")), \
                redirect_stdout(output):
            self.assertEqual(d.main(["--create"]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
