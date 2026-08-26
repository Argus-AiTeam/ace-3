#!/usr/bin/env python3
"""Focused fail-closed tests for the bounded First Voice Hybrid RTL driver."""

from __future__ import annotations

import json
import gzip
import shutil
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import FIXED_CHAT_TOKEN_IDS  # noqa: E402
from model24_first_voice_hybrid import (  # noqa: E402
    STATE_KIND,
    HybridRtlError,
    authenticate_build,
    authenticate_fixed_inputs,
    blocker_document,
    _compact_trace,
    contract_binding,
    hash_file,
    plan_execution,
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

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

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

    def test_opaque_state_mutation_and_stale_position_are_rejected(self) -> None:
        state = self.work / "layer07.state"
        envelope_path = self.work / "layer07.json"
        state.write_bytes(b"real-opaque-verilator-state-image")
        envelope = {
            "schema_version": 1,
            "kind": STATE_KIND,
            "model_binding": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            "build_manifest_sha256": "1" * 64,
            "binary_sha256": "2" * 64,
            "layer_index": 7,
            "cache_slot": 0,
            "next_position": 3,
            "parent_state_sha256": "3" * 64,
            "parent_envelope_sha256": "4" * 64,
            "input_activation_sha256": "5" * 64,
            "output_hidden_sha256": "6" * 64,
            "state": hash_file(state),
        }
        write_json(envelope_path, envelope)
        accepted = validate_state_envelope(
            envelope_path,
            state,
            layer_index=7,
            next_position=3,
            build_manifest_sha256="1" * 64,
            binary_sha256="2" * 64,
        )
        self.assertEqual(accepted["state"], hash_file(state))

        with self.assertRaises(HybridRtlError) as raised:
            validate_state_envelope(
                envelope_path,
                state,
                layer_index=7,
                next_position=3,
                build_manifest_sha256="1" * 64,
                binary_sha256="9" * 64,
            )
        self.assertEqual(raised.exception.code, "stale_rtl_state")

        with self.assertRaises(HybridRtlError) as raised:
            validate_state_envelope(
                envelope_path,
                state,
                layer_index=7,
                next_position=4,
                build_manifest_sha256="1" * 64,
                binary_sha256="2" * 64,
            )
        self.assertEqual(raised.exception.code, "stale_rtl_state")

        state.write_bytes(state.read_bytes() + b"-tampered")
        with self.assertRaises(HybridRtlError) as raised:
            validate_state_envelope(
                envelope_path,
                state,
                layer_index=7,
                next_position=3,
                build_manifest_sha256="1" * 64,
                binary_sha256="2" * 64,
            )
        self.assertEqual(raised.exception.code, "stale_rtl_state")
        self.assertIn("state hash mismatch", str(raised.exception))

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
