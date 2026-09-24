#!/usr/bin/env python3
"""Fixture tests for fail-closed position-3 continuation preparation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from prepare_position3_continuation import (  # noqa: E402
    CHECKPOINT_SHA256,
    HIDDEN_SIZE,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    PreparationError,
    canonical_json,
    file_record,
    prepare,
    validate,
)


class Position3ContinuationPreparationTests(unittest.TestCase):
    def write_json(self, path: Path, document: dict[str, Any]) -> None:
        path.write_bytes(canonical_json(document))

    def make_fixture(self, root: Path) -> dict[str, Any]:
        repository = root / "repository"
        repository.mkdir()
        source_paths = {
            "position3_continuation_preparer": "source/preparer.py",
            "focused_tests": "source/test_preparer.py",
            "contract": "contracts/position3.json",
        }
        for index, relative in enumerate(source_paths.values()):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"source-{index}\n", encoding="ascii")

        parent_source = root / "parent-source.py"
        parent_source.write_text("parent-source\n", encoding="ascii")
        layers = []
        for layer_index in range(24):
            layer_root = root / f"layer{layer_index:02d}"
            layer_root.mkdir()
            predecessor = layer_root / "position002.state"
            output_state = layer_root / "position003.state"
            output_hidden = layer_root / "final.hex"
            predecessor.write_bytes(f"predecessor-{layer_index}\n".encode("ascii"))
            output_state.write_bytes(f"output-state-{layer_index}\n".encode("ascii"))
            output_hidden.write_bytes(f"hidden-{layer_index}\n".encode("ascii"))
            predecessor_record = file_record(predecessor)
            layers.append(
                {
                    "layer_index": layer_index,
                    "position": 2,
                    "predecessor_state": predecessor_record,
                    "predecessor_replay": {
                        "position1": {"output_state": predecessor_record}
                    },
                    "output_state": file_record(output_state),
                    "output": file_record(output_hidden),
                }
            )

        parent_path = root / "position2-evidence.json"
        parent = {
            "schema_version": 2,
            "kind": "ace3_selected_token_position2_continuation_evidence",
            "status": "COMPLETE",
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "numeric_profile": "fixture W4A16 FP16 K/V",
            },
            "selected_token": {"selected_token_id": 271},
            "current_continuation_attempt": {
                "status": "COMPLETE",
                "position": 2,
                "selected_token_id": 271,
                "layer_order": list(range(24)),
                "natural_terminal_layers": 24,
                "predecessor_seed_bindings": {
                    "position0": {
                        "token_id": 151644,
                        "embedding_sha256": "1" * 64,
                    },
                    "position1": {
                        "token_id": 2114,
                        "embedding_sha256": "2" * 64,
                    },
                },
                "layers": layers,
            },
            "consumed_sources": {"parent": file_record(parent_source)},
        }
        self.write_json(parent_path, parent)

        checkpoint = root / "model.safetensors"
        checkpoint.write_bytes(b"synthetic checkpoint\n")
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
                "ace3_position2_live_final_rmsnorm_streaming_lm_head_evidence"
            ),
            "status": "COMPLETE",
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint": file_record(checkpoint),
                "geometry": {
                    "hidden_size": 896,
                    "vocab_size": 151936,
                    "top_k": 10,
                },
            },
            "input": {
                "token_id": 271,
                "position": 2,
                "source": (
                    "authenticated parent position-2 layer-23 terminal hidden state"
                ),
            },
            "parent_traversal": {
                "evidence": file_record(parent_path),
                "status": "COMPLETE",
                "selected_token_id": 271,
                "position": 2,
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
                "selected_token_id": 2,
            },
            "official_shape_streaming_tied_lm_head": {
                "natural_terminal": True,
                "accepted_logits": 151936,
                "selected_token_id": 2,
                "binary": file_record(binary),
                "terminal_log": file_record(terminal),
            },
            "selected_next_token": {
                "token_id": 2,
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
            "output": root / "package",
            "layers": layers,
        }

    @staticmethod
    def embedding(_: Path, token_id: int) -> list[int]:
        return [((token_id << 8) + index) & 0xFFFF for index in range(HIDDEN_SIZE)]

    @staticmethod
    def accept(_: Path) -> None:
        return None

    def arguments(self, fixture: dict[str, Any]) -> dict[str, Any]:
        return {
            "position2_validator": self.accept,
            "lm_head_validator": self.accept,
            "embedding_provider": self.embedding,
            "repository_root": fixture["repository"],
            "own_source_paths": fixture["source_paths"],
            "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
        }

    def test_complete_fixtures_produce_fully_bound_position3_package(self) -> None:
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
            self.assertEqual(document["position3_input"]["selected_token_id"], 2)
            self.assertEqual(
                document["position3_input"]["prompt_token_history"],
                [151644, 2114, 271],
            )
            self.assertEqual(
                document["position3_input"]["traversal_token_history"],
                [151644, 2114, 271, 2],
            )
            self.assertEqual(
                [item["layer_index"] for item in document["layer_kv_parentage"]["layers"]],
                list(range(24)),
            )
            self.assertEqual(
                set(document["source_bindings"]["preparation"]),
                set(fixture["source_paths"]),
            )
            self.assertEqual(
                document["parents"]["position2_traversal"],
                file_record(fixture["parent_path"]),
            )
            self.assertEqual(
                document["parents"]["position2_lm_head"],
                file_record(fixture["lm_head_path"]),
            )
            self.assertEqual(
                len(document["consumed_artifacts"]["position2_layer_states"]),
                24,
            )
            self.assertFalse(document["launch"]["execution_performed"])
            self.assertFalse(document["launch"]["execution_authority"])

    def test_incomplete_parent_fails_without_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            parent = json.loads(
                fixture["parent_path"].read_text(encoding="utf-8")
            )
            parent["status"] = "INCOMPLETE"
            self.write_json(fixture["parent_path"], parent)
            with self.assertRaisesRegex(
                PreparationError, "traversal evidence is not COMPLETE"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            self.assertFalse(fixture["output"].exists())

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
            self.assertFalse(fixture["output"].exists())

    def test_tampered_position2_kv_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            state = Path(fixture["layers"][7]["output_state"]["path"])
            state.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(
                PreparationError, "layer 7 output state content binding mismatch"
            ):
                prepare(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )
            self.assertFalse(fixture["output"].exists())

    def test_tampered_consumed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            source = fixture["repository"] / fixture["source_paths"]["contract"]
            source.write_text("tampered\n", encoding="ascii")
            document = prepare(
                fixture["parent_path"],
                fixture["lm_head_path"],
                fixture["output"],
                **self.arguments(fixture),
            )
            self.assertNotEqual(
                document["source_bindings"]["preparation"]["contract"]["sha256"],
                "0" * 64,
            )
            source.write_text("tampered-again\n", encoding="ascii")
            with self.assertRaisesRegex(
                PreparationError, "stored position-3 launch manifest is stale"
            ):
                validate(
                    fixture["parent_path"],
                    fixture["lm_head_path"],
                    fixture["output"],
                    **self.arguments(fixture),
                )

    def test_tampered_generated_embedding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            prepare(
                fixture["parent_path"],
                fixture["lm_head_path"],
                fixture["output"],
                **self.arguments(fixture),
            )
            (fixture["output"] / "position3_input.hex").write_text(
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
