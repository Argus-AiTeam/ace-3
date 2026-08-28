#!/usr/bin/env python3
"""Focused tests for the layer-3 Token-0 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from layer3_token0_diagnostic import (  # noqa: E402
    DiagnosticError,
    authenticate_predecessor_handoff,
    decode_stage_records,
    diagnose,
    distribution,
)


def valid_handoff_payload() -> bytes:
    return "".join(
        f"{token:02x}{index:04x}{0:04x}\n"
        for token in range(2)
        for index in range(896)
    ).encode("ascii")


class TraceDecodeTests(unittest.TestCase):
    def test_pair_interleaved_vector_is_restored_by_index(self) -> None:
        records = [(index, index + 1) for index in range(896)]
        records[0], records[32] = records[32], records[0]
        decoded = decode_stage_records(records, stage=4, token=0)
        self.assertEqual(decoded[0], 1)
        self.assertEqual(decoded[32], 33)

    def test_attention_duplicate_indices_preserve_head_order(self) -> None:
        records = [
            (key_position, head * 2 + key_position)
            for head in range(14)
            for key_position in range(2)
        ]
        decoded = decode_stage_records(records, stage=9, token=1)
        self.assertEqual(decoded.tolist(), list(range(28)))

    def test_attention_order_mismatch_fails_closed(self) -> None:
        records = [(index & 1, index) for index in range(28)]
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(DiagnosticError, "ordering"):
            decode_stage_records(records, stage=8, token=1)


class HandoffAuthenticationTests(unittest.TestCase):
    def test_expected_predecessor_and_digest_positive_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            handoff = Path(temporary) / "handoff.hex"
            payload = valid_handoff_payload()
            handoff.write_bytes(payload)
            rows, binding = authenticate_predecessor_handoff(
                handoff,
                layer_index=3,
                expected_predecessor_layer=2,
                expected_handoff_sha256=hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(binding["source_layer_index"], 2)
            self.assertEqual(binding["consumer_layer_index"], 3)

    def test_valid_wrong_handoff_and_wrong_layer_fail_before_computation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff = root / "handoff.hex"
            payload = valid_handoff_payload()
            handoff.write_bytes(payload)
            base = {
                "layer_index": 3,
                "focus_dimension": [],
                "handoff": handoff,
                "expected_predecessor_layer": 2,
                "expected_handoff_sha256": hashlib.sha256(payload).hexdigest(),
                "rtl_trace": root / "not-read-trace.hex",
                "oracle_trace": root / "not-read-oracle-trace.hex",
                "rtl_final": root / "not-read-final.hex",
                "oracle_final": root / "not-read-oracle-final.hex",
                "checkpoint": root / "not-read-checkpoint",
                "tensor_map": root / "not-read-map",
                "output": root / "not-written.json",
            }
            cases = {
                "wrong_predecessor_layer": {
                    "expected_predecessor_layer": 1,
                },
                "valid_wrong_handoff": {
                    "expected_handoff_sha256": hashlib.sha256(
                        payload[:-11] + b"0000000000\n"
                    ).hexdigest(),
                },
            }
            for label, override in cases.items():
                with self.subTest(label=label), patch(
                    "layer3_token0_diagnostic.load_trace"
                ) as load_trace:
                    args = argparse.Namespace(**{**base, **override})
                    with self.assertRaises(DiagnosticError):
                        diagnose(args)
                    load_trace.assert_not_called()


class DistributionTests(unittest.TestCase):
    def test_reports_sparse_affected_dimension_and_ulp(self) -> None:
        bits = np.asarray([0x3C00, 0x6400], dtype="<u2")
        result = distribution(
            bits,
            np.asarray([1.0, 1023.6], dtype=np.float64),
            stage=18,
            token=0,
        )
        self.assertEqual(result["count_abs_gt_0_1"], 1)
        self.assertEqual(
            result["affected_coordinates_abs_gt_0_1"],
            [{"dimension": 1}],
        )
        self.assertLessEqual(result["fp16_ulps_at_worst"], 1.0)


if __name__ == "__main__":
    unittest.main()
