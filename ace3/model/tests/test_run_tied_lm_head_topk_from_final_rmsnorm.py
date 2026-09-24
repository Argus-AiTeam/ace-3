from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

import run_tied_lm_head_topk_from_final_rmsnorm as run  # noqa: E402


class TiedLmHeadTerminalContractTests(unittest.TestCase):
    def test_attempt002_manifest_records_exact_module_surface(self) -> None:
        manifest = json.loads(run.DESIGN_MANIFEST.read_text(encoding="utf-8"))
        contract = next(
            row
            for row in manifest["authoritative_contracts"]
            if row["path"] == "ace3/contracts/streaming_tied_lm_head_topk.json"
        )
        self.assertEqual(
            contract["sha256"],
            "c98f1541787a879a3ce9c28ad7cc2a5fe83a8b41421f44058469530a3493f2d4",
        )
        source = next(
            row
            for row in manifest["source_units"]
            if row["path"] == "ace3/rtl/ace3_streaming_tied_lm_head_topk.sv"
        )
        module = source["modules"][0]
        self.assertEqual(run.ATTEMPT_ID, "model24_tied_lm_head_topk_attempt002")
        self.assertEqual(
            [row["name"] for row in module["parameters"]],
            [
                "HIDDEN_SIZE",
                "VOCAB_SIZE",
                "TOP_K",
                "TOKEN_INDEX_WIDTH",
                "FEATURE_INDEX_WIDTH",
                "TOP_RANK_WIDTH",
            ],
        )
        self.assertEqual(len(module["ports"]), 37)
        self.assertEqual(
            [row["module"] for row in module["dependencies"]],
            ["ace3_fp16_to_q24", "ace3_q47_48_to_f16_rne"],
        )
        self.assertEqual(
            module["requirement_ids"],
            [
                "LMHEAD-MODEL-BINDING-001",
                "LMHEAD-NUMERIC-001",
                "LMHEAD-PROTOCOL-001",
                "LMHEAD-TOPK-001",
                "LMHEAD-EVIDENCE-001",
                "LMHEAD-PROVENANCE-001",
            ],
        )

    def test_natural_terminal_requires_exact_complete_counts(self) -> None:
        payload = (
            b"schema=ace3_tied_lm_head_raw_v1 natural_terminal=1 exit_code=0 "
            b"hidden_count=896 weight_count=136134656 logit_count=151936 "
            b"top_count=10 cycles=136287550\n"
        )
        self.assertIsNotNone(run.NATURAL_TERMINAL_RE.fullmatch(payload))

    def test_duplicate_terminal_field_is_rejected(self) -> None:
        payload = (
            b"schema=ace3_tied_lm_head_raw_v1 natural_terminal=1 exit_code=0 "
            b"hidden_count=896 weight_count=136134656 logit_count=151936 "
            b"top_count=10 cycles=136287550 natural_terminal=1\n"
        )
        self.assertIsNone(run.NATURAL_TERMINAL_RE.fullmatch(payload))

    def test_failure_terminal_cannot_pass_natural_gate(self) -> None:
        payload = (
            b"schema=ace3_tied_lm_head_raw_v1 natural_terminal=0 exit_code=2 "
            b"hidden_count=896 weight_count=896 logit_count=1 top_count=0 "
            b"cycles=1802\n"
        )
        self.assertIsNotNone(run.FAILURE_TERMINAL_RE.fullmatch(payload))
        self.assertIsNone(run.NATURAL_TERMINAL_RE.fullmatch(payload))


if __name__ == "__main__":
    unittest.main()
