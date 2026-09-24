#!/usr/bin/env python3
"""Fixture tests for fail-closed position-3 launch acceptance."""

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

from accept_position3_traversal_launch import (  # noqa: E402
    EMBEDDING_NAME,
    HIDDEN_SIZE,
    MANIFEST_NAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NUMERIC_PROFILE,
    PREFLIGHT_NAME,
    RECHECK_CONDITION,
    TRAVERSAL_OPERATION,
    LaunchAcceptanceError,
    accept,
    canonical_json,
    file_record,
    load_json,
    validate,
)


class Position3TraversalLaunchAcceptanceTests(unittest.TestCase):
    def write_json(self, path: Path, document: dict[str, Any]) -> None:
        path.write_bytes(canonical_json(document))

    def make_fixture(self, root: Path) -> dict[str, Any]:
        repository = root / "repository"
        repository.mkdir()
        source_paths = {
            "position3_launch_acceptor": "model/accept.py",
            "focused_tests": "tests/test_accept.py",
            "contract": "contracts/accept.json",
            "position3_package_preparer": "model/prepare.py",
            "position3_package_contract": "contracts/prepare.json",
            "makefile": "Makefile",
        }
        for index, relative in enumerate(source_paths.values()):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"acceptance-source-{index}\n", encoding="ascii")

        package = root / "package"
        package.mkdir()
        embedding = package / EMBEDDING_NAME
        embedding.write_text(
            "".join(
                f"00{index:04x}{((2 << 8) + index) & 0xffff:04x}\n"
                for index in range(HIDDEN_SIZE)
            ),
            encoding="ascii",
        )

        parent_traversal = root / "position2-traversal.json"
        parent_traversal.write_text('{"status":"COMPLETE"}\n', encoding="ascii")
        parent_lm_head = root / "position2-lm-head.json"
        parent_lm_head.write_text('{"status":"COMPLETE"}\n', encoding="ascii")
        checkpoint = root / "model.safetensors"
        checkpoint.write_bytes(b"fixture official checkpoint\n")
        lm_head_binary = root / "lm-head.bin"
        lm_head_binary.write_bytes(b"fixture lm-head binary\n")
        terminal_log = root / "lm-head-terminal.log"
        terminal_log.write_text("fixture natural terminal\n", encoding="ascii")
        oracle = root / "lm-head-oracle.json"
        oracle.write_text('{"mismatches":0}\n', encoding="ascii")
        package_source = root / "package-source.py"
        package_source.write_text("package source\n", encoding="ascii")
        traversal_source = root / "traversal-source.py"
        traversal_source.write_text("traversal source\n", encoding="ascii")
        lm_head_source = root / "lm-head-source.py"
        lm_head_source.write_text("lm-head source\n", encoding="ascii")

        layers = []
        layer_artifacts = []
        for layer_index in range(24):
            predecessor = root / f"layer{layer_index:02d}-position1.state"
            output_state = root / f"layer{layer_index:02d}-position2.state"
            output_hidden = root / f"layer{layer_index:02d}-position2.hex"
            predecessor.write_bytes(f"predecessor-{layer_index}\n".encode())
            output_state.write_bytes(f"output-state-{layer_index}\n".encode())
            output_hidden.write_bytes(f"output-hidden-{layer_index}\n".encode())
            records = {
                "position1_predecessor_state": file_record(predecessor),
                "position2_output_state": file_record(output_state),
                "position2_output_hidden": file_record(output_hidden),
            }
            layers.append(
                {
                    "layer_index": layer_index,
                    "parent_position": 2,
                    "next_position": 3,
                    **records,
                }
            )
            layer_artifacts.append({"layer_index": layer_index, **records})

        checkpoint_sha256 = file_record(checkpoint)["sha256"]
        embedding_record = file_record(embedding)
        manifest = {
            "schema_version": 1,
            "kind": "ace3_selected_token_position3_traversal_launch_package",
            "status": "READY",
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": checkpoint_sha256,
                "numeric_profile": NUMERIC_PROFILE,
            },
            "position3_input": {
                "position": 3,
                "selected_token_id": 2,
                "selected_logit_f16_bits": 19465,
                "prompt_token_history": [151644, 2114, 271],
                "traversal_token_history": [151644, 2114, 271, 2],
                "embedding": {
                    **embedding_record,
                    "dtype": "FP16",
                    "elements": HIDDEN_SIZE,
                    "tensor": "model.embed_tokens.weight",
                    "token_id": 2,
                },
            },
            "layer_kv_parentage": {
                "source_position": 2,
                "target_position": 3,
                "layer_order": list(range(24)),
                "layers": layers,
            },
            "parents": {
                "position2_traversal": file_record(parent_traversal),
                "position2_lm_head": file_record(parent_lm_head),
            },
            "source_bindings": {
                "preparation": {"preparer": file_record(package_source)},
                "position2_traversal": {
                    "validator": file_record(traversal_source)
                },
                "position2_lm_head": {"lm_head": file_record(lm_head_source)},
            },
            "consumed_artifacts": {
                "official_checkpoint": file_record(checkpoint),
                "position2_lm_head_artifacts": {
                    "oracle": file_record(oracle)
                },
                "position2_lm_head_binary": file_record(lm_head_binary),
                "position2_lm_head_terminal_log": file_record(terminal_log),
                "position2_layer_states": layer_artifacts,
            },
            "launch": {
                "operation": TRAVERSAL_OPERATION,
                "input_position": 3,
                "layer_order": list(range(24)),
                "required_embedding": embedding_record["path"],
                "required_fp16_kv_state_count": 24,
                "execution_performed": False,
                "execution_authority": False,
            },
            "claim_boundary": {
                "position3_traversal": "not executed",
                "dialogue": "not claimed",
            },
        }
        self.write_json(package / MANIFEST_NAME, manifest)
        return {
            "repository": repository,
            "source_paths": source_paths,
            "package": package,
            "output": root / "preflight",
            "manifest": manifest,
            "checkpoint_sha256": checkpoint_sha256,
            "layers": layers,
        }

    @staticmethod
    def authenticate_fixture(package: Path) -> dict[str, Any]:
        return load_json(package / MANIFEST_NAME)

    def arguments(self, fixture: dict[str, Any]) -> dict[str, Any]:
        return {
            "package_authenticator": self.authenticate_fixture,
            "repository_root": fixture["repository"],
            "own_source_paths": fixture["source_paths"],
            "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
        }

    def assert_rejected(
        self,
        fixture: dict[str, Any],
        message: str,
    ) -> None:
        with self.assertRaisesRegex(LaunchAcceptanceError, message):
            accept(
                fixture["package"],
                fixture["output"],
                **self.arguments(fixture),
            )
        self.assertFalse(fixture["output"].exists())

    def test_authenticated_ready_package_produces_only_inert_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            document = accept(
                fixture["package"],
                fixture["output"],
                **self.arguments(fixture),
            )
            validated = validate(
                fixture["package"],
                fixture["output"],
                **self.arguments(fixture),
            )
            self.assertEqual(document, validated)
            self.assertEqual(
                {path.name for path in fixture["output"].iterdir()},
                {PREFLIGHT_NAME},
            )
            self.assertEqual(document["status"], "ACCEPTED")
            self.assertEqual(
                document["launch_authority"]["state"],
                "INERT_PREFLIGHT_ONLY",
            )
            self.assertTrue(
                document["launch_authority"]["all_required_bindings_valid"]
            )
            self.assertFalse(
                document["launch_authority"]["execution_authority"]
            )
            self.assertFalse(
                document["launch_authority"]["execution_performed"]
            )
            self.assertFalse(
                document["launch_authority"]["traversal_output_created"]
            )

    def test_missing_package_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            (fixture["package"] / EMBEDDING_NAME).unlink()
            self.assert_rejected(fixture, "artifact closure mismatch")

    def test_stale_package_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            state = Path(fixture["layers"][7]["position2_output_state"]["path"])
            state.write_text("stale substituted bytes\n", encoding="ascii")
            self.assert_rejected(
                fixture,
                "layer 7 position2_output_state content binding mismatch",
            )

    def test_incomplete_package_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["manifest"]["status"] = "NOT_READY"
            self.write_json(
                fixture["package"] / MANIFEST_NAME,
                fixture["manifest"],
            )
            self.assert_rejected(fixture, "package is not READY")

    def test_substituted_embedding_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            replacement = Path(directory) / "substituted-position3-input.hex"
            replacement.write_bytes(
                (fixture["package"] / EMBEDDING_NAME).read_bytes()
            )
            replacement_record = file_record(replacement)
            fixture["manifest"]["position3_input"]["embedding"].update(
                replacement_record
            )
            fixture["manifest"]["launch"]["required_embedding"] = (
                replacement_record["path"]
            )
            self.write_json(
                fixture["package"] / MANIFEST_NAME,
                fixture["manifest"],
            )
            self.assert_rejected(fixture, "substituted path binding")

    def test_token_history_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["manifest"]["position3_input"]["prompt_token_history"][1] = 0
            self.write_json(
                fixture["package"] / MANIFEST_NAME,
                fixture["manifest"],
            )
            self.assert_rejected(fixture, "token history binding mismatch")

    def test_kv_parentage_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["manifest"]["layer_kv_parentage"]["layers"][4][
                "parent_position"
            ] = 1
            self.write_json(
                fixture["package"] / MANIFEST_NAME,
                fixture["manifest"],
            )
            self.assert_rejected(fixture, "layer 4 K/V parentage mismatch")

    def test_source_and_artifact_mutations_fail_closed(self) -> None:
        for binding in ("source", "artifact"):
            with self.subTest(binding=binding):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_fixture(Path(directory))
                    if binding == "source":
                        path = Path(
                            fixture["manifest"]["source_bindings"][
                                "position2_traversal"
                            ]["validator"]["path"]
                        )
                        expected = "sources position2_traversal validator"
                    else:
                        path = Path(
                            fixture["manifest"]["consumed_artifacts"][
                                "position2_lm_head_artifacts"
                            ]["oracle"]["path"]
                        )
                        expected = "lm_head artifacts oracle"
                    path.write_text("mutated\n", encoding="ascii")
                    self.assert_rejected(fixture, expected)

    def test_intended_operation_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))
            fixture["manifest"]["launch"]["operation"] = (
                "selected-token-position2-full-traversal"
            )
            self.write_json(
                fixture["package"] / MANIFEST_NAME,
                fixture["manifest"],
            )
            self.assert_rejected(
                fixture,
                "intended traversal operation mismatch",
            )

    def test_upstream_authentication_failure_creates_no_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_fixture(Path(directory))

            def reject(_: Path) -> dict[str, Any]:
                raise LaunchAcceptanceError("fresh package validation failed")

            arguments = self.arguments(fixture)
            arguments["package_authenticator"] = reject
            with self.assertRaisesRegex(
                LaunchAcceptanceError,
                "fresh package validation failed",
            ):
                accept(fixture["package"], fixture["output"], **arguments)
            self.assertFalse(fixture["output"].exists())

    def test_recheck_condition_names_exact_ready_package_and_bindings(self) -> None:
        self.assertEqual(
            RECHECK_CONDITION,
            "recheck when "
            "build/model24_selected_token_position3_preparation/package/"
            "launch_manifest.json exists with status READY and passes fresh "
            "validation of its exact embedding, token history, 24-layer FP16 "
            "K/V parentage, position-2 parents, source/artifact closure, and "
            "selected-token-position3-full-traversal operation",
        )


if __name__ == "__main__":
    unittest.main()
