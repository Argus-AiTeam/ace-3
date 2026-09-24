#!/usr/bin/env python3
"""Fail-closed tests for authenticated position-4 RTL traversal."""

from __future__ import annotations

from contextlib import contextmanager
import json
import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
from unittest import mock

import numpy as np
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from ace3.model.tests import test_prepare_position4_continuation as prep_tests  # noqa: E402
from official_model24_next_token import _layer_tensor_names  # noqa: E402
from prepare_position4_continuation import (  # noqa: E402
    PreparationError,
    canonical_json,
    file_record,
    prepare,
)
import validate_selected_token_position4_traversal as validator  # noqa: E402
from validate_selected_token_position4_traversal import (  # noqa: E402
    HIDDEN_SIZE,
    LAYER_COUNT,
    POSITION4,
    RECHECK_CONDITION,
    TRAVERSAL_OPERATION,
    TraversalError,
    generate,
    hidden_payload,
    semantic_hidden_sha256,
    sha256_bytes,
    validate,
)


class SelectedTokenPosition4TraversalTests(unittest.TestCase):
    def make_ready_fixture(self, root: Path) -> dict[str, Any]:
        preparation = prep_tests.Position4ContinuationPreparationTests()
        fixture = preparation.make_fixture(root)
        checkpoint = Path(
            json.loads(
                fixture["lm_head_path"].read_text(encoding="utf-8")
            )["model"]["checkpoint"]["path"]
        )
        tensors = {}
        for layer_index in range(LAYER_COUNT):
            for tensor_index, name in enumerate(_layer_tensor_names(layer_index)):
                if name.endswith((".qweight", ".qzeros")):
                    value = np.asarray(
                        [layer_index * 100 + tensor_index],
                        dtype="<i4",
                    )
                else:
                    value = np.asarray(
                        [layer_index + tensor_index / 32],
                        dtype="<f2",
                    )
                tensors[name] = value
        checkpoint.unlink()
        save_file(tensors, checkpoint)
        checkpoint_record = file_record(checkpoint)

        parent = json.loads(
            fixture["parent_path"].read_text(encoding="utf-8")
        )
        parent["model"]["checkpoint_sha256"] = checkpoint_record["sha256"]
        fixture["parent_path"].write_bytes(canonical_json(parent))

        lm_head = json.loads(
            fixture["lm_head_path"].read_text(encoding="utf-8")
        )
        lm_head["model"]["checkpoint"] = checkpoint_record
        lm_head["parent_traversal"]["evidence"] = file_record(
            fixture["parent_path"]
        )
        fixture["lm_head_path"].write_bytes(canonical_json(lm_head))

        fixture["checkpoint_sha256"] = checkpoint_record["sha256"]
        preparation_arguments = preparation.arguments(fixture)
        prepare(
            fixture["parent_path"],
            fixture["lm_head_path"],
            fixture["output"],
            **preparation_arguments,
        )

        executor_sources = {
            "executor": "source/preparer.py",
            "tests": "source/test_preparer.py",
            "contract": "contracts/position4.json",
        }
        fixture.update(
            {
                "manifest": fixture["output"] / "launch_manifest.json",
                "evidence": root / "execution/evidence.json",
                "not_ready": root / "execution/current_state.json",
                "preparation_arguments": preparation_arguments,
                "executor_sources": executor_sources,
            }
        )
        return fixture

    @staticmethod
    def fake_build_canonical_layer(
        layer_index: int,
        build_root: Path,
    ) -> dict[str, dict[str, Any]]:
        binary = (
            build_root
            / f"compiled/layer{layer_index}/obj_dir"
            / "Vace3_decoder_layer0_token_engine"
        )
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(f"canonical-rtl-layer-{layer_index}\n".encode())
        compile_log = build_root / f"layer{layer_index:02d}-compile.log"
        compile_log.write_text(
            f"canonical compile layer={layer_index}\n",
            encoding="ascii",
        )
        return {
            "binary": file_record(binary),
            "compile_log": file_record(compile_log),
        }

    @staticmethod
    def fake_execute_transaction(
        binary: Path,
        layer_index: int,
        position: int,
        hidden_bits: np.ndarray,
        vectors: Mapping[str, Any],
        tensor_vector_dir: Path,
        transaction_dir: Path,
        state_out: Path,
        state_in: Path | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        del binary, tensor_vector_dir, state_in
        raw_dir = transaction_dir / "raw"
        raw_dir.mkdir(parents=True)
        simulation_log = transaction_dir / "simulation.log"
        simulation_log.write_text(
            f"canonical simulation layer={layer_index}\n",
            encoding="ascii",
        )
        bits = np.asarray(hidden_bits, dtype="<u2")
        output = raw_dir / "final.hex"
        output.write_bytes(hidden_payload(bits))
        state_out.parent.mkdir(parents=True, exist_ok=True)
        state_out.write_bytes(f"position4-state-{layer_index}\n".encode())
        terminal = raw_dir / "terminal.txt"
        terminal.write_text(
            "schema=ace3_decoder_token_transaction_v1 "
            f"layer_index={layer_index} position={position} natural_terminal=1 "
            "exit_code=0 trace_count=1 final_count=896 done_count=1\n",
            encoding="ascii",
        )
        trace = raw_dir / "trace.hex"
        trace.write_text(
            f"canonical-trace-layer-{layer_index}\n",
            encoding="ascii",
        )
        semantic_sha256 = sha256_bytes(bits.tobytes())
        transaction = {
            "layer_index": layer_index,
            "position": position,
            "input": {
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "sha256": semantic_sha256,
            },
            "output": {
                **file_record(output),
                "dtype": "FP16",
                "elements": HIDDEN_SIZE,
                "semantic_sha256": semantic_sha256,
            },
            "output_state": file_record(state_out),
            "vectors": dict(vectors),
            "simulation_log": file_record(simulation_log),
            "raw": {
                "terminal": file_record(terminal),
                "trace": file_record(trace),
                "trace_count": 1,
                "final_count": HIDDEN_SIZE,
                "done_count": 1,
            },
        }
        return transaction, bits.copy()

    @contextmanager
    def replay_environment(self) -> Iterator[None]:
        def identity_step(
            _: object,
            hidden: np.ndarray,
            __: int,
        ) -> np.ndarray:
            return np.asarray(hidden, dtype="<u2")

        with (
            mock.patch.object(
                validator,
                "seed_integer_oracle",
                return_value=[object() for _ in range(LAYER_COUNT)],
            ),
            mock.patch.object(
                validator,
                "_primary_layer_step",
                side_effect=identity_step,
            ),
            mock.patch.object(
                validator,
                "build_canonical_layer",
                side_effect=self.fake_build_canonical_layer,
            ),
            mock.patch.object(
                validator,
                "execute_transaction",
                side_effect=self.fake_execute_transaction,
            ),
        ):
            yield

    def arguments(self, fixture: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "package_validation_kwargs": fixture["preparation_arguments"],
            "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
            "repository_root": fixture["repository"],
            "consumed_source_paths": fixture["executor_sources"],
        }

    def test_complete_ready_package_executes_and_replays_24_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_ready_fixture(Path(directory))
            with self.replay_environment():
                document = generate(
                    fixture["manifest"],
                    fixture["evidence"],
                    not_ready_output=fixture["not_ready"],
                    **self.arguments(fixture),
                )
                validated = validate(
                    fixture["manifest"],
                    fixture["evidence"],
                    **self.arguments(fixture),
                )
            self.assertEqual(document, validated)
            self.assertEqual(document["status"], "COMPLETE")
            traversal = document["current_continuation_attempt"]
            self.assertEqual(traversal["operation"], TRAVERSAL_OPERATION)
            self.assertEqual(traversal["position"], POSITION4)
            self.assertEqual(traversal["layer_order"], list(range(LAYER_COUNT)))
            self.assertEqual(len(traversal["layers"]), LAYER_COUNT)
            self.assertTrue(
                all(
                    layer["predecessor_state"]
                    == layer["authenticated_kv_parentage"][
                        "position3_output_state"
                    ]
                    for layer in traversal["layers"]
                )
            )
            self.assertFalse(fixture["not_ready"].exists())

    def test_missing_package_emits_exact_not_ready_without_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "missing/launch_manifest.json"
            evidence = root / "execution/evidence.json"
            not_ready = root / "execution/current_state.json"
            with self.assertRaisesRegex(
                TraversalError, "READY launch manifest is missing"
            ):
                generate(
                    manifest,
                    evidence,
                    not_ready_output=not_ready,
                    repository_root=root,
                    consumed_source_paths={},
                )
            current_state = json.loads(not_ready.read_text(encoding="utf-8"))
            self.assertEqual(current_state["status"], "NOT_READY")
            self.assertEqual(
                current_state["recheck_condition"], RECHECK_CONDITION
            )
            self.assertFalse(current_state["execution_performed"])
            self.assertFalse(
                current_state["complete_traversal_evidence_created"]
            )
            self.assertFalse(evidence.exists())
            self.assertFalse((evidence.parent / "traversal").exists())

    def test_incomplete_parent_emits_not_ready_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_ready_fixture(Path(directory))
            parent = json.loads(
                fixture["parent_path"].read_text(encoding="utf-8")
            )
            parent["status"] = "INCOMPLETE"
            fixture["parent_path"].write_bytes(canonical_json(parent))
            manifest = json.loads(
                fixture["manifest"].read_text(encoding="utf-8")
            )
            manifest["parents"]["position3_traversal"] = file_record(
                fixture["parent_path"]
            )
            fixture["manifest"].write_bytes(canonical_json(manifest))
            with self.assertRaises(PreparationError):
                generate(
                    fixture["manifest"],
                    fixture["evidence"],
                    not_ready_output=fixture["not_ready"],
                    **self.arguments(fixture),
                )
            current_state = json.loads(
                fixture["not_ready"].read_text(encoding="utf-8")
            )
            self.assertEqual(current_state["status"], "NOT_READY")
            self.assertIn("not COMPLETE", current_state["reason"])
            self.assertFalse(fixture["evidence"].exists())
            self.assertFalse(
                (fixture["evidence"].parent / "traversal").exists()
            )

    def test_tampered_embedding_and_kv_parent_fail_before_execution(self) -> None:
        for artifact in ("embedding", "kv"):
            with self.subTest(artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_ready_fixture(Path(directory))
                    manifest = json.loads(
                        fixture["manifest"].read_text(encoding="utf-8")
                    )
                    if artifact == "embedding":
                        path = Path(
                            manifest["position4_input"]["embedding"]["path"]
                        )
                    else:
                        path = Path(
                            manifest["layer_kv_parentage"]["layers"][7][
                                "position3_output_state"
                            ]["path"]
                        )
                    path.write_text("tampered\n", encoding="ascii")
                    with self.assertRaises(
                        (TraversalError, PreparationError)
                    ):
                        generate(
                            fixture["manifest"],
                            fixture["evidence"],
                            not_ready_output=fixture["not_ready"],
                            **self.arguments(fixture),
                        )
                    self.assertEqual(
                        json.loads(
                            fixture["not_ready"].read_text(encoding="utf-8")
                        )["status"],
                        "NOT_READY",
                    )
                    self.assertFalse(fixture["evidence"].exists())
                    self.assertFalse(
                        (fixture["evidence"].parent / "traversal").exists()
                    )

    def test_non_ordered_traversal_cannot_emit_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_ready_fixture(Path(directory))

            def non_ordered_executor(
                live_root: Path,
                authenticated: Mapping[str, Any],
            ) -> dict[str, Any]:
                traversal = validator.execute_live_traversal(
                    live_root, authenticated
                )
                traversal["layer_order"] = list(reversed(range(LAYER_COUNT)))
                return traversal

            with self.replay_environment(), self.assertRaisesRegex(
                TraversalError, "traversal identity mismatch"
            ):
                generate(
                    fixture["manifest"],
                    fixture["evidence"],
                    not_ready_output=fixture["not_ready"],
                    traversal_executor=non_ordered_executor,
                    **self.arguments(fixture),
                )
            self.assertFalse(fixture["evidence"].exists())


if __name__ == "__main__":
    unittest.main()
