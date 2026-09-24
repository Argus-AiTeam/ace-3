#!/usr/bin/env python3
"""Focused stale and tamper rejection tests for position-2 lm_head evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ace3.model.position2_live_lm_head import (
    EvidenceError,
    authenticate_parent_traversal,
    file_record,
    load_terminal_bits,
    parse_rtl_receipt,
    require_expected_hash,
    require_file_record,
    require_source_bindings,
    source_records,
    topk_from_bits,
)


class Position2LiveLmHeadTests(unittest.TestCase):
    def parent_document(self, terminal: Path) -> dict[str, Any]:
        terminal_record = file_record(terminal)
        terminal_bits = load_terminal_bits(terminal)
        terminal_record["semantic_sha256"] = hashlib.sha256(
            terminal_bits.tobytes()
        ).hexdigest()
        layers = [{"layer_index": index} for index in range(23)]
        layers.append({"layer_index": 23, "output": terminal_record})
        return {
            "schema_version": 2,
            "kind": "ace3_selected_token_position2_continuation_evidence",
            "status": "COMPLETE",
            "model": {
                "repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
                "revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
                "checkpoint_sha256": (
                    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7d"
                    "f53940831ed0c1b"
                ),
            },
            "selected_token": {"selected_token_id": 271},
            "current_continuation_attempt": {
                "status": "COMPLETE",
                "execution": "current-worktree compiled Verilator RTL",
                "selected_token_id": 271,
                "position": 2,
                "layer_order": list(range(24)),
                "natural_terminal_layers": 24,
                "layers": layers,
                "post_layer23": {
                    "hidden_sha256": terminal_record["semantic_sha256"],
                    "natural_terminal": True,
                    "independent_oracle_within_tolerance": True,
                },
            },
        }

    def write_parent_document(
        self, path: Path, document: dict[str, Any]
    ) -> None:
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_missing_parent_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with self.assertRaisesRegex(EvidenceError, "evidence is missing"):
                authenticate_parent_traversal(path, lambda _: None)

    def test_incomplete_parent_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            self.write_parent_document(
                path,
                {
                    "schema_version": 2,
                    "kind": (
                        "ace3_selected_token_position2_continuation_evidence"
                    ),
                    "status": "INCOMPLETE",
                },
            )
            with self.assertRaisesRegex(EvidenceError, "is not COMPLETE"):
                authenticate_parent_traversal(path, lambda _: None)

    def test_stale_parent_source_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            terminal.write_text(
                "".join(f"00{index:04x}0000\n" for index in range(896)),
                encoding="ascii",
            )
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            self.write_parent_document(evidence, document)

            def reject_stale_source(_: Path) -> None:
                raise EvidenceError(
                    "parent traversal evidence is stale: "
                    "validator source binding mismatch"
                )

            with self.assertRaisesRegex(
                EvidenceError, "stale: validator source binding mismatch"
            ):
                authenticate_parent_traversal(evidence, reject_stale_source)

    def test_incomplete_parent_layer_closure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            terminal.write_text(
                "".join(f"00{index:04x}0000\n" for index in range(896)),
                encoding="ascii",
            )
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            document["current_continuation_attempt"]["layers"].pop(8)
            self.write_parent_document(evidence, document)
            with self.assertRaisesRegex(EvidenceError, "layer ordering mismatch"):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_missing_parent_terminal_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            terminal.write_text(
                "".join(f"00{index:04x}0000\n" for index in range(896)),
                encoding="ascii",
            )
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            self.write_parent_document(evidence, document)
            terminal.unlink()
            with self.assertRaisesRegex(
                EvidenceError, "parent layer-23 terminal is missing"
            ):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_tampered_parent_terminal_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            terminal.write_text(
                "".join(f"00{index:04x}0000\n" for index in range(896)),
                encoding="ascii",
            )
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            self.write_parent_document(evidence, document)
            terminal.write_text(
                "".join(f"00{index:04x}3c00\n" for index in range(896)),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                EvidenceError, "parent layer-23 terminal content binding mismatch"
            ):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_terminal_parser_binds_ordered_little_endian_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.hex"
            path.write_text("0000003c00\n000001bc00\n", encoding="ascii")
            values = load_terminal_bits(path, hidden_size=2)
            self.assertEqual(values.tolist(), [0x3C00, 0xBC00])
            self.assertEqual(values.tobytes(), b"\x00<\x00\xbc")

    def test_tampered_parent_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text('{"position":2}\n', encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_text('{"position":3}\n', encoding="utf-8")
            with self.assertRaisesRegex(EvidenceError, "parent SHA-256 mismatch"):
                require_expected_hash(path, expected, "parent")

    def test_tampered_bound_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtl.log"
            path.write_text("terminal\n", encoding="ascii")
            record = file_record(path)
            path.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(EvidenceError, "content binding mismatch"):
                require_file_record(path, record, "RTL terminal")

    def test_stale_consumed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"orchestrator": "model/position2.py", "rtl": "rtl/head.sv"}
            for relative_path in paths.values():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative_path}\n", encoding="ascii")
            stored = source_records(root, paths)
            (root / paths["rtl"]).write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(EvidenceError, "rtl source binding mismatch"):
                require_source_bindings(stored, root, paths)

    def test_incomplete_consumed_source_closure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"one": "one.py", "two": "two.py"}
            for relative_path in paths.values():
                (root / relative_path).write_text(relative_path, encoding="ascii")
            stored = source_records(root, paths)
            stored.pop("two")
            with self.assertRaisesRegex(EvidenceError, "closure is incomplete"):
                require_source_bindings(stored, root, paths)

    def test_rtl_terminal_rejects_truncated_vocabulary(self) -> None:
        oracle = {
            "selected_token_id": 42,
            "selected_logit_f16_bits": 0x4500,
            "selected_check_token_ids": list(range(12)),
            "top_k": [],
        }
        receipt = (
            "STREAMING_LM_HEAD_OFFICIAL_PASS hidden=896 vocab=896 "
            "weights=802816 top_token=42 checks=12 cycles=9 "
            "integrated_dialogue=not_run synthesis=not_run ppa=not_measured "
            "fpga=not_run latency=not_claimed\n"
        )
        with self.assertRaisesRegex(EvidenceError, "full-vocabulary count mismatch"):
            parse_rtl_receipt(receipt, oracle)

    def test_topk_ties_use_ascending_token_id(self) -> None:
        bits = np.asarray([0x3C00, 0x4000, 0x4000, 0xBC00], dtype="<u2")
        winners = topk_from_bits(bits)
        self.assertEqual([item[0] for item in winners[:3]], [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
