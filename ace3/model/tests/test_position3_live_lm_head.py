#!/usr/bin/env python3
"""Focused stale and tamper rejection tests for position-3 lm_head evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ace3.model.position3_live_lm_head import (
    EvidenceError,
    NUMERIC_PROFILE,
    PROMPT_TOKEN_HISTORY,
    authenticate_parent_traversal,
    authenticate_tokenizer,
    file_record,
    load_terminal_bits,
    prepare,
    require_source_bindings,
    source_records,
)


class Position3LiveLmHeadTests(unittest.TestCase):
    def parent_document(
        self, terminal: Path, selected_token_id: int = 42
    ) -> dict[str, Any]:
        terminal_record = file_record(terminal)
        terminal_bits = load_terminal_bits(terminal)
        terminal_record["semantic_sha256"] = hashlib.sha256(
            terminal_bits.tobytes()
        ).hexdigest()
        layers = [{"layer_index": index} for index in range(23)]
        layers.append({"layer_index": 23, "output": terminal_record})
        traversal_history = [*PROMPT_TOKEN_HISTORY, selected_token_id]
        return {
            "schema_version": 1,
            "kind": "ace3_selected_token_position3_continuation_evidence",
            "status": "COMPLETE",
            "model": {
                "repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
                "revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
                "checkpoint_sha256": (
                    "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7d"
                    "f53940831ed0c1b"
                ),
                "numeric_profile": NUMERIC_PROFILE,
            },
            "authenticated_launch_preflight": {
                "path": "/bound/launch_preflight.json",
                "bytes": 1,
                "sha256": "0" * 64,
            },
            "position3_input": {
                "position": 3,
                "selected_token_id": selected_token_id,
                "prompt_token_history": PROMPT_TOKEN_HISTORY,
                "traversal_token_history": traversal_history,
            },
            "current_continuation_attempt": {
                "status": "COMPLETE",
                "execution": "current-worktree compiled Verilator RTL",
                "operation": "selected-token-position3-full-traversal",
                "selected_token_id": selected_token_id,
                "position": 3,
                "prompt_token_history": PROMPT_TOKEN_HISTORY,
                "traversal_token_history": traversal_history,
                "layer_order": list(range(24)),
                "natural_terminal_layers": 24,
                "layers": layers,
                "post_layer23": {
                    "hidden_sha256": terminal_record["semantic_sha256"],
                    "natural_terminal": True,
                    "independent_integer_oracle_match": True,
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

    def write_terminal(self, path: Path) -> None:
        path.write_text(
            "".join(f"00{index:04x}0000\n" for index in range(896)),
            encoding="ascii",
        )

    def test_missing_parent_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with self.assertRaisesRegex(
                EvidenceError, "position-3 traversal evidence is missing"
            ):
                prepare(
                    root / "checkpoint",
                    root / "tokenizer",
                    output,
                    root / "missing.json",
                )
            self.assertFalse(output.exists())

    def test_incomplete_parent_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            self.write_parent_document(
                path,
                {
                    "schema_version": 1,
                    "kind": (
                        "ace3_selected_token_position3_continuation_evidence"
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
            self.write_terminal(terminal)
            evidence = root / "evidence.json"
            self.write_parent_document(evidence, self.parent_document(terminal))

            def reject_stale_source(_: Path) -> None:
                raise EvidenceError(
                    "parent traversal evidence is stale: source binding mismatch"
                )

            with self.assertRaisesRegex(EvidenceError, "stale: source binding"):
                authenticate_parent_traversal(evidence, reject_stale_source)

    def test_incomplete_parent_layer_closure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            self.write_terminal(terminal)
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            document["current_continuation_attempt"]["layers"].pop(8)
            self.write_parent_document(evidence, document)
            with self.assertRaisesRegex(EvidenceError, "layer ordering mismatch"):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_wrong_parent_operation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            self.write_terminal(terminal)
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            document["current_continuation_attempt"]["operation"] = "position2"
            self.write_parent_document(evidence, document)
            with self.assertRaisesRegex(EvidenceError, "identity mismatch"):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_tampered_parent_terminal_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            self.write_terminal(terminal)
            evidence = root / "evidence.json"
            self.write_parent_document(evidence, self.parent_document(terminal))
            terminal.write_text(
                "".join(f"00{index:04x}3c00\n" for index in range(896)),
                encoding="ascii",
            )
            with self.assertRaisesRegex(EvidenceError, "content binding mismatch"):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_selected_token_history_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = root / "final.hex"
            self.write_terminal(terminal)
            evidence = root / "evidence.json"
            document = self.parent_document(terminal)
            document["position3_input"]["traversal_token_history"][-1] += 1
            self.write_parent_document(evidence, document)
            with self.assertRaisesRegex(EvidenceError, "history binding mismatch"):
                authenticate_parent_traversal(evidence, lambda _: None)

    def test_tokenizer_artifacts_are_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_dir = Path(directory)
            tokenizer = tokenizer_dir / "tokenizer.json"
            config = tokenizer_dir / "tokenizer_config.json"
            tokenizer.write_bytes(b"tokenizer\n")
            config.write_bytes(b"config\n")
            tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            stored = authenticate_tokenizer(
                tokenizer_dir, tokenizer_hash, config_hash
            )
            self.assertEqual(stored["tokenizer"]["sha256"], tokenizer_hash)
            tokenizer.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(
                EvidenceError, "official tokenizer.json SHA-256 mismatch"
            ):
                authenticate_tokenizer(
                    tokenizer_dir, tokenizer_hash, config_hash
                )

    def test_stale_consumed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"orchestrator": "model/position3.py", "rtl": "rtl/head.sv"}
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


if __name__ == "__main__":
    unittest.main()
