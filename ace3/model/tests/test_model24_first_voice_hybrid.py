#!/usr/bin/env python3
"""Focused fail-closed tests for the bounded First Voice Hybrid RTL driver."""

from __future__ import annotations

import json
import gzip
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import FIXED_CHAT_TOKEN_IDS  # noqa: E402
from model24_first_voice_hybrid import (  # noqa: E402
    COMPACT_SELF_TEST_MARKER,
    HIDDEN_SIZE,
    STATE_KIND,
    STATE_SCHEMA_VERSION,
    HybridRtlError,
    _canonical_bytes,
    authenticate_build,
    authenticate_compact_layer,
    authenticate_fixed_inputs,
    binary_path,
    blocker_document,
    build_compact_layer,
    canonical_json,
    compact_layer_manifest_path,
    _compact_trace,
    contract_binding,
    hash_file,
    plan_execution,
    sha256_bytes,
    state_record_paths,
    validate_state_envelope,
    write_json,
)
from model24_oracle import (  # noqa: E402
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
)


class Model24FirstVoiceHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = (
            REPOSITORY_ROOT
            / "build"
            / "model24_first_voice_hybrid_unit"
            / self._testMethodName
        )
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)

    @staticmethod
    def _write_hidden(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                f"00{index:04x}{int(value):04x}\n"
                for index, value in enumerate(values)
            ),
            encoding="ascii",
        )

    def _write_state_record(
        self,
        *,
        layer_index: int,
        next_position: int,
        seed: int,
    ) -> tuple[Path, Path]:
        states_dir = self.work / "states"
        runtime_dir = self.work / "transactions"
        input_values = np.asarray(
            [(index + seed) & 0xFFFF for index in range(HIDDEN_SIZE)],
            dtype="<u2",
        )
        output_values = np.asarray(
            [(index + seed + 1) & 0xFFFF for index in range(HIDDEN_SIZE)],
            dtype="<u2",
        )
        transaction_dir = (
            runtime_dir
            / f"position{next_position - 1:03d}"
            / f"layer{layer_index:02d}"
        )
        self._write_hidden(transaction_dir / "inputs.hex", input_values)
        self._write_hidden(transaction_dir / "raw" / "final.hex", output_values)
        state_path, envelope_path = state_record_paths(
            states_dir,
            layer_index,
            next_position,
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(
            f"opaque-layer-{layer_index}-position-{next_position}-seed-{seed}".encode()
        )
        parent_state_sha256 = None
        parent_envelope_sha256 = None
        if next_position > 1:
            parent_state, parent_envelope = state_record_paths(
                states_dir,
                layer_index,
                next_position - 1,
            )
            parent_state_sha256 = hash_file(parent_state)["sha256"]
            parent_envelope_sha256 = hash_file(parent_envelope)["sha256"]
        envelope = {
            "schema_version": STATE_SCHEMA_VERSION,
            "kind": STATE_KIND,
            "model_binding": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            "build_manifest_sha256": "1" * 64,
            "binary_sha256": "2" * 64,
            "layer_index": layer_index,
            "cache_slot": 0,
            "next_position": next_position,
            "parent_state_sha256": parent_state_sha256,
            "parent_envelope_sha256": parent_envelope_sha256,
            "input_activation_sha256": sha256_bytes(_canonical_bytes(input_values)),
            "output_hidden_sha256": sha256_bytes(_canonical_bytes(output_values)),
            "state": hash_file(state_path),
        }
        write_json(envelope_path, envelope)
        return state_path, envelope_path

    def _write_two_record_chain(self) -> tuple[Path, Path]:
        self._write_state_record(layer_index=7, next_position=1, seed=10)
        return self._write_state_record(layer_index=7, next_position=2, seed=20)

    def _validate_chain(self, next_position: int = 2) -> dict[str, object]:
        state_path, envelope_path = state_record_paths(
            self.work / "states", 7, next_position
        )
        return validate_state_envelope(
            envelope_path,
            state_path,
            states_dir=self.work / "states",
            runtime_dir=self.work / "transactions",
            layer_index=7,
            next_position=next_position,
            build_manifest_sha256="1" * 64,
            binary_sha256="2" * 64,
        )

    def _mutate_current_envelope(self, field: str, value: object) -> None:
        _, envelope_path = state_record_paths(self.work / "states", 7, 2)
        envelope = json.loads(envelope_path.read_text(encoding="ascii"))
        envelope[field] = value
        write_json(envelope_path, envelope)

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def _fake_compact_build(self, layer_index: int = 3) -> tuple[Path, Path]:
        compiled_dir = self.work / "compiled"
        temporary_mdir = self.work / "compact_mdir"

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            arguments = [str(value) for value in command]
            if arguments == ["verilator", "--version"]:
                return subprocess.CompletedProcess(
                    arguments, 0, stdout="Verilator test-version\n", stderr=""
                )
            if "--Mdir" in arguments:
                mdir = Path(arguments[arguments.index("--Mdir") + 1])
                built_binary = mdir / "Vace3_decoder_layer0_token_engine"
                built_binary.write_bytes(b"fake-self-contained-executable")
                built_binary.chmod(0o755)
                return subprocess.CompletedProcess(arguments, 0)
            if arguments[0] == "strip":
                return subprocess.CompletedProcess(arguments, 0)
            if "--compact-build-self-test" in arguments:
                self.assertFalse(temporary_mdir.exists())
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=f"{COMPACT_SELF_TEST_MARKER} layer_index={layer_index}\n",
                    stderr="",
                )
            raise AssertionError(f"unexpected command: {arguments}")

        with mock.patch(
            "model24_first_voice_hybrid.subprocess.run", side_effect=fake_run
        ):
            build_compact_layer(
                REPOSITORY_ROOT,
                compiled_dir,
                layer_index,
                temporary_mdir,
                verilator="verilator",
                strip="strip",
            )
        return compiled_dir, temporary_mdir

    def test_compact_builder_removes_mdir_and_keeps_only_binary_manifest(self) -> None:
        compiled_dir, temporary_mdir = self._fake_compact_build()
        layer_dir = compiled_dir / "layer3"
        self.assertFalse(temporary_mdir.exists())
        self.assertEqual(
            sorted(
                str(path.relative_to(layer_dir))
                for path in layer_dir.rglob("*")
                if path.is_file()
            ),
            ["bin/Vace3_decoder_layer0_token_engine", "layer_manifest.json"],
        )
        accepted, _ = authenticate_compact_layer(REPOSITORY_ROOT, compiled_dir, 3)
        self.assertEqual(accepted["layer_index"], 3)

    def test_compact_builder_removes_mdir_after_build_failure(self) -> None:
        compiled_dir = self.work / "compiled"
        temporary_mdir = self.work / "compact_mdir"

        def fail_compile(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            arguments = [str(value) for value in command]
            if arguments == ["verilator", "--version"]:
                return subprocess.CompletedProcess(
                    arguments, 0, stdout="Verilator test-version\n", stderr=""
                )
            raise subprocess.CalledProcessError(2, arguments)

        with mock.patch(
            "model24_first_voice_hybrid.subprocess.run", side_effect=fail_compile
        ):
            with self.assertRaises(HybridRtlError) as raised:
                build_compact_layer(
                    REPOSITORY_ROOT,
                    compiled_dir,
                    3,
                    temporary_mdir,
                    verilator="verilator",
                    strip="strip",
                )
        self.assertEqual(raised.exception.code, "compact_build_failed")
        self.assertFalse(temporary_mdir.exists())
        self.assertFalse((compiled_dir / ".layer3.partial").exists())

    def test_compact_manifest_and_binary_tampering_are_rejected(self) -> None:
        compiled_dir, _ = self._fake_compact_build()
        manifest_path = compact_layer_manifest_path(compiled_dir, 3)
        compact_binary = binary_path(compiled_dir, 3)
        original_manifest = manifest_path.read_bytes()
        original_binary = compact_binary.read_bytes()
        for case in (
            "layer", "source", "configuration", "configuration_hash",
            "binary_hash", "manifest_bytes", "binary_bytes",
        ):
            with self.subTest(case=case):
                compact_binary.write_bytes(original_binary)
                compact_binary.chmod(0o755)
                manifest_path.write_bytes(original_manifest)
                document = json.loads(original_manifest)
                if case == "layer":
                    document["layer_index"] = 4
                    write_json(manifest_path, document)
                elif case == "source":
                    source = next(iter(document["sources"]))
                    document["sources"][source] = "0" * 64
                    write_json(manifest_path, document)
                elif case == "configuration":
                    document["configuration"]["top_module"] = "tampered"
                    write_json(manifest_path, document)
                elif case == "configuration_hash":
                    document["configuration_sha256"] = "0" * 64
                    write_json(manifest_path, document)
                elif case == "binary_hash":
                    document["binary"]["sha256"] = "0" * 64
                    write_json(manifest_path, document)
                elif case == "manifest_bytes":
                    manifest_path.write_bytes(original_manifest + b" ")
                elif case == "binary_bytes":
                    compact_binary.write_bytes(original_binary + b"tampered")
                with self.assertRaises(HybridRtlError) as raised:
                    authenticate_compact_layer(REPOSITORY_ROOT, compiled_dir, 3)
                self.assertEqual(raised.exception.code, "stale_compiled_rtl")

    def test_fixed_prompt_and_multiple_generation_fit_real_rtl_depth(self) -> None:
        plan = plan_execution(FIXED_CHAT_TOKEN_IDS, 4)
        self.assertEqual(plan["prompt_positions"], 25)
        self.assertEqual(plan["maximum_generated_feedback_positions"], 3)
        self.assertEqual(plan["maximum_represented_positions"], 28)
        self.assertEqual(plan["cache_positions"], 128)
        contract, _ = contract_binding(REPOSITORY_ROOT)
        self.assertEqual(
            contract["token0_disclosure"]["selection_is_rtl_input"],
            False,
        )
        self.assertEqual(
            contract["numeric_profile"]["qzero_adjustment"],
            "none",
        )

    def test_capacity_overflow_returns_precise_non_rtl_blocker(self) -> None:
        with self.assertRaises(HybridRtlError) as raised:
            plan_execution(FIXED_CHAT_TOKEN_IDS, 105)
        self.assertEqual(raised.exception.code, "rtl_context_capacity_exceeded")
        blocker = blocker_document(raised.exception)
        self.assertEqual(blocker["status"], "blocked")
        self.assertFalse(blocker["rtl_claim"])
        self.assertFalse(blocker["software_fallback_used"])
        self.assertEqual(
            blocker["details"]["required_represented_positions"],
            129,
        )

    def test_wrong_checkpoint_is_rejected_before_tokenizer_use(self) -> None:
        checkpoint = self.work / "stale.safetensors"
        checkpoint.write_bytes(b"not-the-fixed-checkpoint")
        with self.assertRaises(HybridRtlError) as raised:
            authenticate_fixed_inputs(checkpoint, self.work / "missing-tokenizer")
        self.assertEqual(
            raised.exception.code,
            "checkpoint_authentication_failed",
        )
        self.assertIn("checkpoint size mismatch", str(raised.exception))

    def test_missing_compiled_manifest_has_precise_blocker(self) -> None:
        _, contract_record = contract_binding(REPOSITORY_ROOT)
        with self.assertRaises(HybridRtlError) as raised:
            authenticate_build(REPOSITORY_ROOT, self.work, contract_record)
        self.assertEqual(raised.exception.code, "compiled_layer_missing")
        self.assertEqual(
            raised.exception.details["path"],
            str(self.work / "build_manifest.json"),
        )

    def test_valid_retained_state_chain_is_accepted(self) -> None:
        state, _ = self._write_two_record_chain()
        accepted = self._validate_chain()
        self.assertEqual(accepted["state"], hash_file(state))

    def test_opaque_state_mutation_and_stale_position_are_rejected(self) -> None:
        state, _ = self._write_two_record_chain()
        with self.assertRaises(HybridRtlError) as raised:
            self._validate_chain(next_position=3)
        self.assertEqual(raised.exception.code, "stale_rtl_state")

        with self.assertRaises(HybridRtlError) as raised:
            state_path, envelope_path = state_record_paths(
                self.work / "states", 7, 2
            )
            validate_state_envelope(
                envelope_path,
                state_path,
                states_dir=self.work / "states",
                runtime_dir=self.work / "transactions",
                layer_index=7,
                next_position=2,
                build_manifest_sha256="1" * 64,
                binary_sha256="9" * 64,
            )
        self.assertEqual(raised.exception.code, "stale_rtl_state")

        state.write_bytes(state.read_bytes() + b"-tampered")
        with self.assertRaises(HybridRtlError) as raised:
            self._validate_chain()
        self.assertEqual(raised.exception.code, "stale_rtl_state")
        self.assertIn("state hash mismatch", str(raised.exception))

    def test_modified_parent_state_hash_is_rejected(self) -> None:
        self._write_two_record_chain()
        self._mutate_current_envelope("parent_state_sha256", "9" * 64)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_modified_parent_envelope_hash_is_rejected(self) -> None:
        self._write_two_record_chain()
        self._mutate_current_envelope("parent_envelope_sha256", "9" * 64)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_modified_input_activation_hash_is_rejected(self) -> None:
        self._write_two_record_chain()
        self._mutate_current_envelope("input_activation_sha256", "9" * 64)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_modified_output_hidden_hash_is_rejected(self) -> None:
        self._write_two_record_chain()
        self._mutate_current_envelope("output_hidden_sha256", "9" * 64)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_wrong_predecessor_is_rejected(self) -> None:
        self._write_two_record_chain()
        wrong_state, wrong_envelope = self._write_state_record(
            layer_index=8, next_position=1, seed=30
        )
        predecessor_state, predecessor_envelope = state_record_paths(
            self.work / "states", 7, 1
        )
        shutil.copyfile(wrong_state, predecessor_state)
        shutil.copyfile(wrong_envelope, predecessor_envelope)
        self._mutate_current_envelope(
            "parent_state_sha256", hash_file(predecessor_state)["sha256"]
        )
        self._mutate_current_envelope(
            "parent_envelope_sha256", hash_file(predecessor_envelope)["sha256"]
        )
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_missing_predecessor_is_rejected(self) -> None:
        self._write_two_record_chain()
        predecessor_state, _ = state_record_paths(self.work / "states", 7, 1)
        shutil.rmtree(predecessor_state.parent)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_noncanonical_envelope_is_rejected(self) -> None:
        self._write_two_record_chain()
        _, envelope_path = state_record_paths(self.work / "states", 7, 2)
        envelope = json.loads(envelope_path.read_text(encoding="ascii"))
        reordered = dict(reversed(list(envelope.items())))
        envelope_path.write_bytes(
            (json.dumps(reordered, separators=(",", ":")) + "\n").encode("ascii")
        )
        self.assertNotEqual(envelope_path.read_bytes(), canonical_json(envelope))
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_extra_state_envelope_field_is_rejected(self) -> None:
        self._write_two_record_chain()
        self._mutate_current_envelope("unexpected", True)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_missing_state_envelope_field_is_rejected(self) -> None:
        self._write_two_record_chain()
        _, envelope_path = state_record_paths(self.work / "states", 7, 2)
        envelope = json.loads(envelope_path.read_text(encoding="ascii"))
        del envelope["output_hidden_sha256"]
        write_json(envelope_path, envelope)
        with self.assertRaises(HybridRtlError):
            self._validate_chain()

    def test_contract_is_explicit_json_and_has_no_software_rtl_alias(self) -> None:
        contract_path = (
            REPOSITORY_ROOT
            / "ace3"
            / "contracts"
            / "model24_first_voice_hybrid.json"
        )
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        self.assertEqual(contract["rtl"]["cache_positions"], 128)
        self.assertEqual(
            contract["host"]["fixed_prompt_tokens"],
            len(FIXED_CHAT_TOKEN_IDS),
        )
        self.assertIn(
            "Software/oracle Model24 execution must never be labeled RTL",
            contract["claim_boundary"]["software_only_prohibition"],
        )

    def test_trace_compaction_preserves_authenticated_raw_bytes(self) -> None:
        trace = self.work / "trace.hex"
        payload = (b"0011223344556677\n" * 4096) + b"terminal\n"
        trace.write_bytes(payload)
        record = _compact_trace(trace, REPOSITORY_ROOT)
        archive = REPOSITORY_ROOT / record["storage"]["path"]
        self.assertFalse(trace.exists())
        self.assertTrue(archive.is_file())
        self.assertEqual(gzip.decompress(archive.read_bytes()), payload)
        self.assertEqual(record["bytes"], len(payload))
        self.assertLess(record["storage"]["bytes"], record["bytes"])


if __name__ == "__main__":
    unittest.main()
