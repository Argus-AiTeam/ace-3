#!/usr/bin/env python3
"""Tests for fail-closed position-4 continuation preparation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ace3.model.prepare_position4_continuation import (
    HIDDEN_SIZE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NUMERIC_PROFILE,
    PreparationError,
    RECHECK_CONDITION,
    canonical_json,
    file_record,
    prepare,
    validate,
)


class Position4ContinuationPreparationTests(unittest.TestCase):
    def write_json(self, path: Path, document: dict[str, Any]) -> None:
        path.write_bytes(canonical_json(document))

    def make_fixture(self, root: Path) -> dict[str, Any]:
        repository = root / "repository"
        repository.mkdir()
        source_paths = {
            "position4_continuation_preparer": "source/preparer.py",
            "focused_tests": "source/test_preparer.py",
            "contract": "contracts/position4.json",
        }
        for index, relative in enumerate(source_paths.values()):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source-{index}\n", encoding="ascii")

        parent_source = root / "position3-source.py"
        parent_source.write_text("position3-source\n", encoding="ascii")
        selected_position3_token = 2
        traversal_history = [151644, 2114, 271, selected_position3_token]
        layers = []
        for layer_index in range(24):
            layer_root = root / f"layer{layer_index:02d}"
            layer_root.mkdir()
            predecessor = layer_root / "position003.state"
            output_state = layer_root / "position004.state"
            output_hidden = layer_root / "final.hex"
            predecessor.write_bytes(f"predecessor-{layer_index}\n".encode("ascii"))
            output_state.write_bytes(f"output-state-{layer_index}\n".encode("ascii"))
            output_hidden.write_bytes(f"hidden-{layer_index}\n".encode("ascii"))
            predecessor_record = file_record(predecessor)
            output_record = file_record(output_hidden)
            output_record["semantic_sha256"] = hashlib.sha256(
                output_hidden.read_bytes()
            ).hexdigest()
            layers.append(
                {
                    "layer_index": layer_index,
                    "position": 3,
                    "authenticated_kv_parentage": {
                        "position2_output_state": predecessor_record
                    },
                    "predecessor_state": predecessor_record,
                    "output_state": file_record(output_state),
                    "output": output_record,
                }
            )

        parent_path = root / "position3-evidence.json"
        parent = {
            "schema_version": 1,
            "kind": "ace3_selected_token_position3_continuation_evidence",
            "status": "COMPLETE",
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": "fixture-checkpoint",
                "numeric_profile": NUMERIC_PROFILE,
            },
            "position3_input": {
                "position": 3,
                "selected_token_id": selected_position3_token,
                "prompt_token_history": [151644, 2114, 271],
                "traversal_token_history": traversal_history,
            },
            "current_continuation_attempt": {
                "status": "COMPLETE",
                "execution": "current-worktree compiled Verilator RTL",
                "operation": "selected-token-position3-full-traversal",
                "selected_token_id": selected_position3_token,
                "position": 3,
                "prompt_token_history": [151644, 2114, 271],
                "traversal_token_history": traversal_history,
                "layer_order": list(range(24)),
                "natural_terminal_layers": 24,
                "layers": layers,
                "post_layer23": {
                    "hidden_sha256": layers[-1]["output"]["semantic_sha256"],
                    "natural_terminal": True,
                    "independent_integer_oracle_match": True,
                },
            },
            "consumed_sources": {"parent": file_record(parent_source)},
        }

        checkpoint = root / "model.safetensors"
        checkpoint.write_bytes(b"synthetic checkpoint\n")
        parent["model"]["checkpoint_sha256"] = file_record(checkpoint)["sha256"]
        self.write_json(parent_path, parent)

        tokenizer_dir = root / "tokenizer"
        tokenizer_dir.mkdir()
        tokenizer = tokenizer_dir / "tokenizer.json"
        tokenizer_config = tokenizer_dir / "tokenizer_config.json"
        tokenizer.write_text('{"fixture":"tokenizer"}\n', encoding="ascii")
        tokenizer_config.write_text('{"fixture":"config"}\n', encoding="ascii")
        artifact = root / "lm-head-oracle.json"
        artifact.write_text('{"fixture":true}\n', encoding="ascii")
        binary = root / "lm-head.bin"
        binary.write_bytes(b"fixture-binary\n")
        terminal = root / "rtl.log"
        terminal.write_text("fixture-pass\n", encoding="ascii")
        lm_source = root / "lm-head-source.py"
        lm_source.write_text("lm-head-source\n", encoding="ascii")
        lm_head_path = root / "lm-head-evidence.json"
        lm_head = {
            "schema_version": 1,
            "kind": (
                "ace3_position3_live_final_rmsnorm_streaming_lm_head_evidence"
            ),
            "status": "COMPLETE",
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint": file_record(checkpoint),
                "tokenizer": {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "tokenizer": file_record(tokenizer),
                    "tokenizer_config": file_record(tokenizer_config),
                },
                "geometry": {
                    "hidden_size": 896,
                    "vocab_size": 151936,
                    "top_k": 10,
                },
            },
            "input": {
                "token_id": selected_position3_token,
                "position": 3,
                "source": (
                    "authenticated position-3 traversal layer-23 terminal "
                    "hidden state"
                ),
            },
            "parent_traversal": {
                "evidence": file_record(parent_path),
                "status": "COMPLETE",
                "selected_token_id": selected_position3_token,
                "position": 3,
                "prompt_token_history": [151644, 2114, 271],
                "traversal_token_history": traversal_history,
                "natural_terminal_layers": 24,
            },
            "full_vocabulary_oracles": {
                "agreement": {
                    "all_rounded_fp16_logits_equal": True,
                    "selected_token_equal": True,
                    "top_k_equal": True,
                },
                "coverage": {
                    "vocab_size": 151936,
                    "rounded_fp16_mismatches": 0,
                },
                "selected_token_id": 42,
            },
            "official_shape_streaming_tied_lm_head": {
                "natural_terminal": True,
                "accepted_logits": 151936,
                "selected_token_id": 42,
                "binary": file_record(binary),
                "terminal_log": file_record(terminal),
            },
            "selected_next_token": {
                "token_id": 42,
                "logit_f16_bits": 19465,
                "exact_integer_pytorch_agreement": True,
                "rtl_top_k_agreement": True,
            },
            "consumed_sources": {"lm_head": file_record(lm_source)},
            "artifacts": {"oracle.json": file_record(artifact)},
        }
        self.write_json(lm_head_path, lm_head)
        return {
            "repository": repository,
            "source_paths": source_paths,
            "parent_path": parent_path,
            "lm_head_path": lm_head_path,
            "checkpoint_sha256": file_record(checkpoint)["sha256"],
            "tokenizer_sha256": file_record(tokenizer)["sha256"],
            "tokenizer_config_sha256": file_record(tokenizer_config)["sha256"],
            "output": root / "package",
            "layers": layers,
            "tokenizer": tokenizer,
        }

    @staticmethod
    def embedding(_: Path, token_id: int) -> list[int]:
        return [((token_id << 8) + index) & 0xFFFF for index in range(HIDDEN_SIZE)]

    def arguments(self, fixture: dict[str, Any]) -> dict[str, Any]:
        return {
            "embedding_provider": self.embedding,
            "repository_root": fixture["repository"],
            "own_source_paths": fixture["source_paths"],
            "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
            "expected_tokenizer_sha256": fixture["tokenizer_sha256"],
            "expected_tokenizer_config_sha256": (
                fixture["tokenizer_config_sha256"]
            ),
        }

    def test_complete_evidence_produces_bound_position4_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            document = prepare(
                fixture["parent_path"],
                fixture["lm_head_path"],
                fixture["output"],
                **self.arguments(fixture),
            )
            validated = validate(
                fixture["parent_path"],
                fixture["lm_head_path"],
                fixture["output"],
                **self.arguments(fixture),
            )
            self.assertEqual(document, validated)
            self.assertEqual(document["status"], "READY")
            self.assertEqual(document["position4_input"]["selected_token_id"], 42)
            self.assertEqual(
                document["position4_input"]["prompt_token_history"],
                [151644, 2114, 271, 2],
            )
            self.assertEqual(
                document["position4_input"]["traversal_token_history"],
                [151644, 2114, 271, 2, 42],
            )
            self.assertEqual(
                [item["layer_index"] for item in document["layer_kv_parentage"]["layers"]],
                list(range(24)),
            )
            self.assertEqual(
                document["consumed_artifacts"]["official_tokenizer"]["tokenizer"],
                file_record(fixture["tokenizer"]),
            )
            self.assertFalse(document["launch"]["execution_performed"])
            self.assertFalse(document["launch"]["execution_authority"])

    def test_missing_lm_head_emits_not_ready_with_exact_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            missing = Path(directory) / "missing-lm-head.json"
            with self.assertRaisesRegex(
                PreparationError, "position-3 lm_head evidence is missing"
            ):
                prepare(
                    fixture["parent_path"],
                    missing,
                    fixture["output"],
                    **self.arguments(fixture),
                )
            record = json.loads(
                (fixture["output"] / "not_ready.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "NOT_READY")
            self.assertEqual(record["recheck_condition"], RECHECK_CONDITION)
            self.assertFalse((fixture["output"] / "launch_manifest.json").exists())
            self.assertFalse((fixture["output"] / "position4_input.hex").exists())

    def test_incomplete_lm_head_emits_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            lm_head = json.loads(
                fixture["lm_head_path"].read_text(encoding="utf-8")
            )
            lm_head["status"] = "INCOMPLETE"
            self.write_json(fixture["lm_head_path"], lm_head)
            with self.assertRaisesRegex(
                PreparationError, "lm_head evidence is not COMPLETE"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            record = json.loads(
                (fixture["output"] / "not_ready.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "NOT_READY")
            self.assertIn("not COMPLETE", record["reason"])

    def test_lm_head_bound_to_different_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            lm_head = json.loads(
                fixture["lm_head_path"].read_text(encoding="utf-8")
            )
            lm_head["parent_traversal"]["evidence"]["sha256"] = "0" * 64
            self.write_json(fixture["lm_head_path"], lm_head)
            with self.assertRaisesRegex(
                PreparationError, "exact COMPLETE traversal evidence"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            self.assertTrue((fixture["output"] / "not_ready.json").is_file())

    def test_tampered_position3_kv_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            state = Path(fixture["layers"][7]["output_state"]["path"])
            state.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(
                PreparationError, "layer 7 output_state content binding mismatch"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            self.assertTrue((fixture["output"] / "not_ready.json").is_file())

    def test_tampered_tokenizer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["tokenizer"].write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(
                PreparationError, "tokenizer.json content binding mismatch"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            self.assertTrue((fixture["output"] / "not_ready.json").is_file())

    def test_tampered_generated_embedding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            prepare(
                fixture["parent_path"],
                fixture["lm_head_path"],
                fixture["output"],
                **self.arguments(fixture),
            )
            (fixture["output"] / "position4_input.hex").write_text(
                "tampered\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                PreparationError, "embedding content binding mismatch"
            ):
                validate(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )


if __name__ == "__main__":
    unittest.main()
