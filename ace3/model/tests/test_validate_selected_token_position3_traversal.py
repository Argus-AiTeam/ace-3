#!/usr/bin/env python3
"""Fail-closed tests for position-3 tensor closure and execution replay."""

from __future__ import annotations

from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping
from unittest import mock

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from accept_position3_traversal_launch import (  # noqa: E402
    PREFLIGHT_NAME,
    accept,
    canonical_json,
)
from ace3.model.tests import (  # noqa: E402
    test_accept_position3_traversal_launch as acceptance_tests,
)
from official_model24_next_token import _layer_tensor_names  # noqa: E402
import validate_selected_token_position3_traversal as validator  # noqa: E402
import validate_selected_token_position2_traversal as position2_validator  # noqa: E402
from validate_selected_token_position3_traversal import (  # noqa: E402
    HIDDEN_SIZE,
    LAYER_COUNT,
    POSITION3,
    TRAVERSAL_OPERATION,
    TraversalError,
    comparison_record,
    file_record,
    generate,
    hidden_payload,
    load_json,
    materialize_position3_vectors,
    semantic_hidden_sha256,
    sha256_bytes,
    validate,
)


class SelectedTokenPosition3TraversalTests(unittest.TestCase):
    def test_position3_uses_q24_exp_silu_layer_oracle(self) -> None:
        self.assertIs(
            validator._primary_layer_step,
            position2_validator.selected_primary_layer_step,
        )
        previous_gate = position2_validator.dialogue_oracle._fp16_silu_gate
        observed: dict[str, tuple[int, bool, bool]] = {}

        def probe(
            _state: object,
            hidden: np.ndarray,
            _start_position: int,
        ) -> np.ndarray:
            observed["result"] = (
                position2_validator.dialogue_oracle._fp16_silu_gate(
                    0xBB29,
                    0x2D20,
                )
            )
            return hidden

        with mock.patch.object(
            position2_validator.dialogue_oracle,
            "_primary_layer_step",
            side_effect=probe,
        ):
            hidden = np.asarray([[0x3C00]], dtype="<u2")
            self.assertTrue(
                np.array_equal(
                    validator._primary_layer_step(object(), hidden, POSITION3),
                    hidden,
                )
            )

        self.assertEqual(observed["result"], (0xA552, False, False))
        self.assertIs(
            position2_validator.dialogue_oracle._fp16_silu_gate,
            previous_gate,
        )

    def make_accepted_fixture(self, root: Path) -> dict[str, Any]:
        acceptance = (
            acceptance_tests.Position3TraversalLaunchAcceptanceTests()
        )
        fixture = acceptance.make_fixture(root)
        checkpoint = Path(
            fixture["manifest"]["consumed_artifacts"]["official_checkpoint"][
                "path"
            ]
        )
        tensors = {}
        for layer_index in range(LAYER_COUNT):
            for tensor_index, name in enumerate(
                _layer_tensor_names(layer_index)
            ):
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
        save_file(tensors, checkpoint)
        checkpoint_record = file_record(checkpoint)
        fixture["manifest"]["model"]["checkpoint_sha256"] = checkpoint_record[
            "sha256"
        ]
        fixture["manifest"]["consumed_artifacts"][
            "official_checkpoint"
        ] = checkpoint_record
        acceptance.write_json(
            fixture["package"] / "launch_manifest.json",
            fixture["manifest"],
        )
        fixture["checkpoint_sha256"] = checkpoint_record["sha256"]
        arguments = acceptance.arguments(fixture)
        preflight_dir = root / "accepted-preflight"
        accept(
            fixture["package"],
            preflight_dir,
            **arguments,
        )
        fixture.update(
            {
                "preflight": preflight_dir / PREFLIGHT_NAME,
                "evidence": root / "execution/evidence.json",
                "acceptance_arguments": arguments,
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
        output = raw_dir / "final.hex"
        bits = np.asarray(hidden_bits, dtype="<u2")
        output.write_bytes(hidden_payload(bits))
        state_out.parent.mkdir(parents=True, exist_ok=True)
        state_out.write_bytes(f"position3-state-{layer_index}\n".encode())
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

    def fake_traversal(
        self,
        live_root: Path,
        authenticated: Mapping[str, Any],
    ) -> dict[str, Any]:
        live_root.mkdir(parents=True)
        bits = np.asarray(authenticated["embedding_bits"], dtype="<u2")
        layers = []
        checkpoint_path = Path(authenticated["checkpoint"]["path"])
        with safe_open(checkpoint_path, framework="np") as checkpoint:
            for layer_index, parentage in enumerate(
                authenticated["kv_parentage"]
            ):
                layer_dir = live_root / f"layer{layer_index:02d}"
                vectors = materialize_position3_vectors(
                    checkpoint,
                    layer_index,
                    bits,
                    layer_dir / "vectors",
                )
                binary = (
                    live_root
                    / f"compiled/layer{layer_index}/obj_dir"
                    / "Vace3_decoder_layer0_token_engine"
                )
                binary.parent.mkdir(parents=True)
                binary.write_bytes(
                    f"canonical-rtl-layer-{layer_index}\n".encode()
                )
                compile_log = layer_dir / "compile.log"
                compile_log.write_text(
                    f"canonical compile layer={layer_index}\n",
                    encoding="ascii",
                )
                transaction, actual_bits = self.fake_execute_transaction(
                    binary,
                    layer_index,
                    POSITION3,
                    bits,
                    vectors,
                    Path(vectors["manifest"]["path"]).parent,
                    layer_dir / "position003",
                    layer_dir / "position004.state",
                    Path(parentage["position2_output_state"]["path"]),
                )
                comparison = comparison_record(actual_bits, actual_bits)
                comparison_path = layer_dir / "comparison.json"
                comparison_path.write_bytes(canonical_json(comparison))
                layers.append(
                    {
                        **transaction,
                        "authenticated_kv_parentage": parentage,
                        "predecessor_state": parentage[
                            "position2_output_state"
                        ],
                        "live_binary": file_record(binary),
                        "compile_log": file_record(compile_log),
                        "independent_oracle_comparison": comparison,
                        "comparison_report": file_record(comparison_path),
                    }
                )
                bits = actual_bits
        return {
            "status": "COMPLETE",
            "execution": "current-worktree compiled Verilator RTL",
            "operation": TRAVERSAL_OPERATION,
            "selected_token_id": authenticated["selected_token_id"],
            "position": POSITION3,
            "prompt_token_history": authenticated["prompt_token_history"],
            "traversal_token_history": authenticated[
                "traversal_token_history"
            ],
            "layer_order": list(range(LAYER_COUNT)),
            "natural_terminal_layers": LAYER_COUNT,
            "parent_preflight": authenticated["preflight_record"],
            "parent_launch_manifest": authenticated["manifest_record"],
            "layers": layers,
            "post_layer23": {
                "hidden_sha256": sha256_bytes(bits.tobytes()),
                "natural_terminal": True,
                "independent_integer_oracle_match": True,
            },
        }

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

    def execution_arguments(self, fixture: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "launch_validation_kwargs": fixture[
                "acceptance_arguments"
            ],
            "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
        }

    def generate_fixture(self, fixture: Mapping[str, Any]) -> dict[str, Any]:
        with self.replay_environment():
            return generate(
                fixture["preflight"],
                fixture["evidence"],
                traversal_executor=self.fake_traversal,
                **self.execution_arguments(fixture),
            )

    def assert_mutated_generation_rejected(
        self,
        fixture: Mapping[str, Any],
        mutation: Any,
        message: str,
    ) -> None:
        def executor(
            live_root: Path,
            authenticated: Mapping[str, Any],
        ) -> dict[str, Any]:
            traversal = self.fake_traversal(live_root, authenticated)
            mutation(traversal)
            return traversal

        with self.replay_environment(), self.assertRaisesRegex(
            TraversalError,
            message,
        ):
            generate(
                fixture["preflight"],
                fixture["evidence"],
                traversal_executor=executor,
                **self.execution_arguments(fixture),
            )

    @staticmethod
    def rewrite_manifest(layer: dict[str, Any]) -> None:
        vectors = layer["vectors"]
        path = Path(vectors["manifest"]["path"])
        manifest = load_json(path)
        manifest["tensors"] = vectors["tensors"]
        path.write_bytes(canonical_json(manifest))
        vectors["manifest"] = file_record(path)

    def test_complete_real_shaped_closure_is_replayed_on_generate_and_validate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))
            document = self.generate_fixture(fixture)
            with self.replay_environment():
                validated = validate(
                    fixture["preflight"],
                    fixture["evidence"],
                    **self.execution_arguments(fixture),
                )
            self.assertEqual(document, validated)
            layers = document["current_continuation_attempt"]["layers"]
            self.assertEqual(len(layers), LAYER_COUNT)
            self.assertTrue(
                all(len(layer["vectors"]["tensors"]) == 26 for layer in layers)
            )

    def test_empty_and_truncated_tensor_closures_are_rejected(self) -> None:
        for count in (0, 25):
            with self.subTest(count=count):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_accepted_fixture(Path(directory))

                    def mutate(traversal: dict[str, Any]) -> None:
                        layer = traversal["layers"][0]
                        layer["vectors"]["tensors"] = layer["vectors"][
                            "tensors"
                        ][:count]
                        self.rewrite_manifest(layer)

                    self.assert_mutated_generation_rejected(
                        fixture,
                        mutate,
                        "layer 0 tensor closure mismatch",
                    )

    def test_dummy_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))

            def mutate(traversal: dict[str, Any]) -> None:
                vectors = traversal["layers"][0]["vectors"]
                path = Path(vectors["manifest"]["path"])
                path.write_text('{"fixture":true}\n', encoding="ascii")
                vectors["manifest"] = file_record(path)

            self.assert_mutated_generation_rejected(
                fixture,
                mutate,
                "layer 0 vector manifest identity mismatch",
            )

    def test_extra_and_substituted_tensor_closures_are_rejected(self) -> None:
        for mutation, message in (
            ("extra", "tensor closure mismatch"),
            ("substituted", "tensor inventory mismatch"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_accepted_fixture(Path(directory))

                    def mutate(traversal: dict[str, Any]) -> None:
                        layer = traversal["layers"][0]
                        tensors = layer["vectors"]["tensors"]
                        if mutation == "extra":
                            tensors.append(dict(tensors[-1]))
                        else:
                            tensors[0]["checkpoint_tensor"] = dict(
                                tensors[0]["checkpoint_tensor"]
                            )
                            tensors[0]["checkpoint_tensor"]["name"] = (
                                _layer_tensor_names(1)[0]
                            )
                        self.rewrite_manifest(layer)

                    self.assert_mutated_generation_rejected(
                        fixture,
                        mutate,
                        message,
                    )

    def test_vector_records_cannot_redirect_harness_consumed_paths(self) -> None:
        cases = {
            "manifest": "layer 0 vector manifest substituted path binding",
            "input": "layer 0 vector input substituted path binding",
            "rope": (
                "layer 0 vector rope_coefficients substituted path binding"
            ),
            "tensor": "layer 0 tensor model.layers.0.input_layernorm.weight substituted path binding",
        }
        for artifact, message in cases.items():
            with self.subTest(artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_accepted_fixture(Path(directory))

                    def mutate(traversal: dict[str, Any]) -> None:
                        vectors = traversal["layers"][0]["vectors"]
                        manifest_path = Path(vectors["manifest"]["path"])
                        manifest = load_json(manifest_path)
                        if artifact == "manifest":
                            alternate = manifest_path.with_name(
                                "alternate-manifest.json"
                            )
                            alternate.write_bytes(manifest_path.read_bytes())
                            manifest_path.write_bytes(b"{}\n")
                            vectors["manifest"] = file_record(alternate)
                            return
                        if artifact == "input":
                            record = vectors["input"]
                            manifest_key = "input"
                        elif artifact == "rope":
                            record = vectors["rope_coefficients"]
                            manifest_key = "rope_coefficients"
                        else:
                            record = vectors["tensors"][0]["serialized"]
                            manifest_key = ""
                        canonical = Path(record["path"])
                        alternate = canonical.with_name(
                            f"alternate-{canonical.name}"
                        )
                        alternate.write_bytes(canonical.read_bytes())
                        canonical.write_bytes(b"ffff\n")
                        alternate_record = file_record(alternate)
                        if artifact == "tensor":
                            vectors["tensors"][0][
                                "serialized"
                            ] = alternate_record
                            manifest["tensors"] = vectors["tensors"]
                        else:
                            vectors[manifest_key] = alternate_record
                            manifest[manifest_key] = alternate_record
                        manifest_path.write_bytes(canonical_json(manifest))
                        vectors["manifest"] = file_record(manifest_path)

                    self.assert_mutated_generation_rejected(
                        fixture,
                        mutate,
                        message,
                    )

    def mutate_evidence(
        self,
        fixture: Mapping[str, Any],
        mutation: Any,
    ) -> None:
        evidence = load_json(fixture["evidence"])
        mutation(evidence["current_continuation_attempt"])
        fixture["evidence"].write_bytes(canonical_json(evidence))

    def assert_stored_mutation_rejected(
        self,
        fixture: Mapping[str, Any],
        mutation: Any,
        message: str,
    ) -> None:
        self.generate_fixture(fixture)
        self.mutate_evidence(fixture, mutation)
        with self.replay_environment(), self.assertRaisesRegex(
            TraversalError,
            message,
        ):
            validate(
                fixture["preflight"],
                fixture["evidence"],
                **self.execution_arguments(fixture),
            )

    def test_substituted_binary_is_rejected_by_canonical_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))

            def mutate(traversal: dict[str, Any]) -> None:
                layer = traversal["layers"][4]
                path = Path(layer["live_binary"]["path"])
                path.write_bytes(b"substituted executable\n")
                layer["live_binary"] = file_record(path)

            self.assert_stored_mutation_rejected(
                fixture,
                mutate,
                "layer 4 canonical binary replay mismatch",
            )

    def test_fabricated_terminal_and_logs_are_rejected_by_replay(self) -> None:
        for artifact, message in (
            ("terminal", "terminal replay mismatch"),
            ("simulation_log", "simulation log replay mismatch"),
            ("compile_log", "compile log replay mismatch"),
        ):
            with self.subTest(artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self.make_accepted_fixture(Path(directory))

                    def mutate(traversal: dict[str, Any]) -> None:
                        layer = traversal["layers"][6]
                        if artifact == "terminal":
                            path = Path(layer["raw"]["terminal"]["path"])
                            path.write_text(
                                "schema=ace3_decoder_token_transaction_v1 "
                                "layer_index=6 position=3 natural_terminal=1 "
                                "exit_code=0 trace_count=2 final_count=896 "
                                "done_count=1\n",
                                encoding="ascii",
                            )
                            layer["raw"]["terminal"] = file_record(path)
                            layer["raw"]["trace_count"] = 2
                        else:
                            path = Path(layer[artifact]["path"])
                            path.write_text(
                                f"fabricated {artifact}\n",
                                encoding="ascii",
                            )
                            layer[artifact] = file_record(path)

                    self.assert_stored_mutation_rejected(
                        fixture,
                        mutate,
                        message,
                    )

    def test_oracle_only_output_is_rejected_by_fresh_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))

            def mutate(traversal: dict[str, Any]) -> None:
                layer = traversal["layers"][-1]
                output = Path(layer["output"]["path"])
                altered = np.zeros(HIDDEN_SIZE, dtype="<u2")
                altered[0] = 1
                output.write_bytes(hidden_payload(altered))
                digest = semantic_hidden_sha256(output)
                layer["output"] = {
                    **file_record(output),
                    "dtype": "FP16",
                    "elements": HIDDEN_SIZE,
                    "semantic_sha256": digest,
                }
                comparison = comparison_record(altered, altered)
                layer["independent_oracle_comparison"] = comparison
                report = Path(layer["comparison_report"]["path"])
                report.write_bytes(canonical_json(comparison))
                layer["comparison_report"] = file_record(report)
                traversal["post_layer23"]["hidden_sha256"] = digest

            self.assert_stored_mutation_rejected(
                fixture,
                mutate,
                "layer 23 output replay mismatch",
            )

    def test_missing_preflight_fails_before_execution_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))
            fixture["preflight"].unlink()
            with self.assertRaisesRegex(
                TraversalError,
                "position-3 launch preflight is missing",
            ):
                generate(
                    fixture["preflight"],
                    fixture["evidence"],
                    traversal_executor=self.fake_traversal,
                    **self.execution_arguments(fixture),
                )
            self.assertFalse(fixture["evidence"].parent.exists())

    def test_stale_parent_artifact_fails_before_execution_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_accepted_fixture(Path(directory))
            state = Path(
                fixture["layers"][9]["position2_output_state"]["path"]
            )
            state.write_text("stale-parent\n", encoding="ascii")
            with self.assertRaisesRegex(
                (TraversalError, RuntimeError),
                "content binding mismatch",
            ):
                generate(
                    fixture["preflight"],
                    fixture["evidence"],
                    traversal_executor=self.fake_traversal,
                    **self.execution_arguments(fixture),
                )


if __name__ == "__main__":
    unittest.main()
