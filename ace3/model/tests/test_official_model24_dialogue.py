#!/usr/bin/env python3
"""Focused lineage, stop, and mutation tests for Model24 dialogue evidence."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import EOS_TOKEN_ID, FIXED_CHAT_TOKEN_IDS  # noqa: E402
from official_model24_dialogue import (  # noqa: E402
    ARTIFACT_NAME,
    DialogueExecutionError,
    POSITION2_TARGET_TOKEN_ID,
    REPAIR8_KV_AXES,
    REPAIR8_KV_SHAPE,
    generation_stop_reason,
    load_repair8_position2_kv,
    validate_document,
    validate_repair8_position2_kv,
)
from model24_one_shot_python_launcher import (  # noqa: E402
    file_record,
    probe_project_python,
)
from model24_oracle import CHECKPOINT_SHA256  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )


def _canonical_trace_rows(values: list[int], stage: int, position: int) -> bytes:
    return b"".join(
        f"00{position:04x}{stage:02x}{index:04x}{value:04x}\n".encode("ascii")
        for index, value in enumerate(values)
    )


def _make_repair8_fixture(root: Path) -> tuple[Path, str]:
    layer_id = 0
    semantic_dir = root / "semantic_kv"
    transaction_dir = root / "execution/transactions/position001/layer00/raw"
    state_dir = root / "execution/states/layer00/position002"
    semantic_dir.mkdir(parents=True)
    transaction_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    position0_k = list(range(128))
    position0_v = [0x0800 + index for index in range(128)]
    position1_k = [0x1000 + index for index in range(128)]
    position1_v = [0x1800 + index for index in range(128)]
    semantic_lines = []
    for index in range(128):
        semantic_lines.append(
            f"{layer_id:02x}00{0:04x}06{index:04x}{position0_k[index]:04x}\n"
        )
        semantic_lines.append(
            f"{layer_id:02x}00{0:04x}07{index:04x}{position0_v[index]:04x}\n"
        )
    semantic_payload = "".join(semantic_lines).encode("ascii")
    payload_path = semantic_dir / "layer00.hex"
    readback_path = semantic_dir / "layer00.readback.hex"
    payload_path.write_bytes(semantic_payload)
    readback_path.write_bytes(semantic_payload)

    trace_payload = b"".join(
        line
        for index in range(128)
        for line in (
            f"00{1:04x}06{index:04x}{position1_k[index]:04x}\n".encode("ascii"),
            f"00{1:04x}07{index:04x}{position1_v[index]:04x}\n".encode("ascii"),
        )
    )
    trace_path = transaction_dir / "trace.hex.gz"
    with trace_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(trace_payload)

    state_path = state_dir / "state"
    state_path.write_bytes(b"synthetic-verilator-state")
    state_record = _record(state_path)
    envelope_path = state_dir / "envelope.json"
    _write_json(
        envelope_path,
        {
            "schema_version": 2,
            "kind": "ace3_model24_first_voice_layer_state",
            "layer_index": layer_id,
            "next_position": 2,
            "model_binding": {"checkpoint_sha256": CHECKPOINT_SHA256},
            "state": state_record,
        },
    )

    parent_kv = {
        "elements_each": 128,
        "format": "FP16",
        "k_sha256": _sha256(_canonical_trace_rows(position0_k, 6, 0)),
        "v_sha256": _sha256(_canonical_trace_rows(position0_v, 7, 0)),
    }
    manifest_path = semantic_dir / "layer00.json"
    _write_json(
        manifest_path,
        {
            "schema": "ace3-semantic-kv-preload-v1",
            "layer_index": layer_id,
            "cache_slot": 0,
            "source_position": 0,
            "execution_position": 1,
            "execution_token": 2114,
            "model_binding": {"checkpoint_sha256": CHECKPOINT_SHA256},
            "tensor_binding": {
                "key": "trace-stage-6-rotated-key-fp16",
                "ordering": "kv-head-major-dimension-minor",
                "value": "trace-stage-7-value-fp16",
            },
            "payload": {"path": payload_path.name, **_record(payload_path)},
            "parent_kv": parent_kv,
        },
    )

    layers = [{"layer_index": index} for index in range(24)]
    layers[0] = {
        "layer_index": 0,
        "semantic_kv_manifest": _record(manifest_path),
        "semantic_kv_payload": _record(payload_path),
        "semantic_kv_readback": _record(readback_path),
        "semantic_parent_kv": parent_kv,
        "independent_reference": {"inherited_parent_kv": parent_kv},
        "transaction": {
            "state_bytes": state_record["bytes"],
            "state_sha256": state_record["sha256"],
            "trace": {
                "sha256": _sha256(trace_payload),
                "storage": _record(trace_path),
            },
        },
    }
    result_path = root / "result.json"
    _write_json(
        result_path,
        {
            "schema": "ace3-position1-model24-causal-traversal-v2",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "selected_token": 2114,
            "position": 1,
            "natural_terminal_layers": 24,
            "layers": layers,
        },
    )
    return root, _sha256(result_path.read_bytes())


def _rebind_manifest(root: Path, mutate) -> str:
    manifest_path = root / "semantic_kv/layer00.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    mutate(manifest)
    _write_json(manifest_path, manifest)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    result["layers"][0]["semantic_kv_manifest"] = _record(manifest_path)
    _write_json(result_path, result)
    return _sha256(result_path.read_bytes())


class OfficialModel24DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact = (
            REPOSITORY_ROOT
            / "build"
            / "official_model24_dialogue"
            / ARTIFACT_NAME
        )
        cls.document = json.loads(artifact.read_text(encoding="ascii"))

    def test_complete_readable_dialogue_evidence(self) -> None:
        summary = validate_document(self.document)
        generation = self.document["generation"]
        self.assertEqual(
            summary["generated_token_ids"],
            generation["generated_token_ids"],
        )
        self.assertEqual(summary["decoded_text"], generation["decoded_text"])
        self.assertEqual(summary["stop_reason"], generation["stop_reason"])
        self.assertEqual(summary["steps"], len(generation["steps"]))
        self.assertTrue(summary["decoded_text"])

    def test_cache_lineage_growth_is_non_vacuous(self) -> None:
        steps = self.document["generation"]["steps"]
        self.assertEqual(
            [step["cache_lineage"]["position_count"] for step in steps],
            [25, 26, 27, 28],
        )
        self.assertEqual(
            steps[1]["cache_lineage"]["parent_cache_sha256"],
            steps[0]["cache_lineage"]["cache_sha256"],
        )
        self.assertNotEqual(
            steps[0]["cache_lineage"]["layers"][0]["k_sha256"],
            steps[1]["cache_lineage"]["layers"][0]["k_sha256"],
        )

        broken = copy.deepcopy(self.document)
        broken["generation"]["steps"][1]["cache_lineage"][
            "parent_cache_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "aggregate cache parentage",
        ):
            validate_document(broken)

    def test_argmax_and_logit_mutations_are_rejected(self) -> None:
        excessive_hidden_error = copy.deepcopy(self.document)
        excessive_hidden_error["generation"]["steps"][0]["terminal_hidden"][
            "independent_reference"
        ]["max_abs_error"] = 1.0
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "terminal hidden comparison",
        ):
            validate_document(excessive_hidden_error)

        false_argmax = copy.deepcopy(self.document)
        false_argmax["generation"]["steps"][1]["token"][
            "argmax_matches_independent_reference"
        ] = False
        with self.assertRaisesRegex(DialogueExecutionError, "argmax comparison"):
            validate_document(false_argmax)

        excessive_error = copy.deepcopy(self.document)
        excessive_error["generation"]["steps"][0]["logits"][
            "independent_reference"
        ]["max_abs_error"] = 1.0
        with self.assertRaisesRegex(DialogueExecutionError, "logits comparison"):
            validate_document(excessive_error)

        broken_top_k = copy.deepcopy(self.document)
        broken_top_k["generation"]["steps"][0]["token"]["top_k"][0][
            "token_id"
        ] += 1
        with self.assertRaisesRegex(DialogueExecutionError, "top-k evidence"):
            validate_document(broken_top_k)

    def test_official_eos_and_maximum_stop_rules(self) -> None:
        self.assertIsNone(generation_stop_reason(9707, 1, 3))
        self.assertEqual(
            generation_stop_reason(EOS_TOKEN_ID, 2, 3),
            "eos_token",
        )
        self.assertEqual(
            generation_stop_reason(1879, 3, 3),
            "max_new_tokens",
        )

        continued_after_eos = copy.deepcopy(self.document)
        continued_after_eos["generation"]["steps"][1]["token"][
            "argmax_token_id"
        ] = EOS_TOKEN_ID
        continued_after_eos["generation"]["steps"][1]["token"][
            "independent_reference_argmax_token_id"
        ] = EOS_TOKEN_ID
        continued_after_eos["generation"]["generated_token_ids"][1] = EOS_TOKEN_ID
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "continued after EOS",
        ):
            validate_document(continued_after_eos)


class Repair8Position2KvTests(unittest.TestCase):
    def test_import_uses_official_gqa_axes_and_head_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result_sha256 = _make_repair8_fixture(Path(directory))
            imported = load_repair8_position2_kv(
                root,
                0,
                POSITION2_TARGET_TOKEN_ID,
                expected_result_sha256=result_sha256,
            )
            self.assertEqual(tuple(imported.contract["axes"]), REPAIR8_KV_AXES)
            self.assertEqual(imported.k_bits.shape, REPAIR8_KV_SHAPE)
            self.assertEqual(tuple(imported.reference_k.shape), REPAIR8_KV_SHAPE)
            self.assertEqual(imported.contract["sequence_length"], 2)
            self.assertEqual(imported.contract["kv_heads"], 2)
            self.assertEqual(imported.contract["head_dim"], 64)
            self.assertEqual(int(imported.k_bits[0, 0, 0, 63]), 63)
            self.assertEqual(int(imported.k_bits[0, 0, 1, 0]), 64)
            self.assertEqual(int(imported.k_bits[0, 1, 0, 0]), 0x1000)
            validate_repair8_position2_kv(imported)

            wrong_geometry = replace(
                imported,
                contract={**imported.contract, "kv_heads": 4, "head_dim": 32},
            )
            with self.assertRaisesRegex(
                DialogueExecutionError, "geometry contract"
            ):
                validate_repair8_position2_kv(wrong_geometry)

    def test_transposed_doubled_and_missing_heads_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result_sha256 = _make_repair8_fixture(Path(directory))
            imported = load_repair8_position2_kv(
                root,
                0,
                POSITION2_TARGET_TOKEN_ID,
                expected_result_sha256=result_sha256,
            )
            transposed = replace(
                imported,
                k_bits=imported.k_bits.transpose(0, 1, 3, 2),
                v_bits=imported.v_bits.transpose(0, 1, 3, 2),
                reference_k=imported.reference_k.transpose(2, 3),
                reference_v=imported.reference_v.transpose(2, 3),
                contract={
                    **imported.contract,
                    "shape": [1, 2, 64, 2],
                    "flattening": (
                        "batch-major, sequence-major, head-dim-major, "
                        "kv-head-minor"
                    ),
                },
            )
            with self.assertRaisesRegex(DialogueExecutionError, "GQA shape"):
                validate_repair8_position2_kv(transposed)

            for head_count in (1, 4):
                with self.subTest(head_count=head_count):
                    repeats = 1 if head_count == 1 else 2
                    k_bits = (
                        imported.k_bits[:, :, :1, :]
                        if head_count == 1
                        else np.repeat(imported.k_bits, repeats, axis=2)
                    )
                    v_bits = (
                        imported.v_bits[:, :, :1, :]
                        if head_count == 1
                        else np.repeat(imported.v_bits, repeats, axis=2)
                    )
                    broken = replace(
                        imported,
                        k_bits=k_bits,
                        v_bits=v_bits,
                        reference_k=torch.from_numpy(
                            k_bits.view("<f2").astype(np.float64)
                        ),
                        reference_v=torch.from_numpy(
                            v_bits.view("<f2").astype(np.float64)
                        ),
                        contract={
                            **imported.contract,
                            "shape": [1, 2, head_count, 64],
                            "kv_heads": head_count,
                        },
                    )
                    with self.assertRaisesRegex(
                        DialogueExecutionError, "GQA shape"
                    ):
                        validate_repair8_position2_kv(broken)

    def test_wrong_layer_position_and_tokens_are_rejected(self) -> None:
        mutations = (
            (
                "layer",
                lambda manifest: manifest.__setitem__("layer_index", 1),
                "manifest identity",
            ),
            (
                "position",
                lambda manifest: manifest.__setitem__("source_position", 1),
                "manifest identity",
            ),
            (
                "source_token",
                lambda manifest: manifest.__setitem__("execution_token", 271),
                "manifest identity",
            ),
        )
        for label, mutation, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root, _ = _make_repair8_fixture(Path(directory))
                result_sha256 = _rebind_manifest(root, mutation)
                with self.assertRaisesRegex(DialogueExecutionError, message):
                    load_repair8_position2_kv(
                        root,
                        0,
                        POSITION2_TARGET_TOKEN_ID,
                        expected_result_sha256=result_sha256,
                    )

        with tempfile.TemporaryDirectory() as directory:
            root, result_sha256 = _make_repair8_fixture(Path(directory))
            with self.assertRaisesRegex(DialogueExecutionError, "target token"):
                load_repair8_position2_kv(
                    root,
                    0,
                    272,
                    expected_result_sha256=result_sha256,
                )

    def test_tampered_cache_and_transposition_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, result_sha256 = _make_repair8_fixture(Path(directory))
            readback = root / "semantic_kv/layer00.readback.hex"
            payload = bytearray(readback.read_bytes())
            payload[-2] = ord("1") if payload[-2] != ord("1") else ord("2")
            readback.write_bytes(payload)
            with self.assertRaisesRegex(DialogueExecutionError, "hash binding"):
                load_repair8_position2_kv(
                    root,
                    0,
                    POSITION2_TARGET_TOKEN_ID,
                    expected_result_sha256=result_sha256,
                )

        with tempfile.TemporaryDirectory() as directory:
            root, _ = _make_repair8_fixture(Path(directory))
            result_sha256 = _rebind_manifest(
                root,
                lambda manifest: manifest["tensor_binding"].__setitem__(
                    "ordering", "head-dim-major-kv-head-minor"
                ),
            )
            with self.assertRaisesRegex(
                DialogueExecutionError, "transposition contract"
            ):
                load_repair8_position2_kv(
                    root,
                    0,
                    POSITION2_TARGET_TOKEN_ID,
                    expected_result_sha256=result_sha256,
                )


class Model24OneShotRuntimeTests(unittest.TestCase):
    def test_stale_runtime_contract_is_rejected_before_receipt(self) -> None:
        launcher = MODEL_DIR / "model24_one_shot_python_launcher.py"
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            entrypoint = scratch / "entrypoint.py"
            entrypoint.write_text("raise SystemExit(0)\n", encoding="ascii")
            python_path = Path(sys.executable).resolve()
            probe = probe_project_python(python_path)
            runtime_identity = {
                "project_python": {
                    **file_record(python_path),
                    "implementation": probe["implementation"],
                    "python_version": probe["python_version"],
                    "required_imports": {
                        "numpy": probe["numpy_version"],
                        "torch": probe["torch_version"],
                    },
                },
                "entrypoint": file_record(entrypoint),
            }
            contract_path = scratch / "runtime.json"
            stale_identity = copy.deepcopy(runtime_identity)
            stale_identity["project_python"]["python_version"] = "0.0-stale"
            _write_json(
                contract_path,
                {
                    "schema_version": 1,
                    "kind": "ace3_model24_python_runtime_contract",
                    "runtime_identity": stale_identity,
                },
            )
            receipt = scratch / "receipt.json"
            rejected = subprocess.run(
                [
                    str(python_path),
                    str(launcher),
                    "--project-python",
                    str(python_path),
                    "--entrypoint",
                    str(entrypoint),
                    "--runtime-contract",
                    str(contract_path),
                    "--receipt",
                    str(receipt),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("stale Python/runtime", rejected.stderr)
            self.assertFalse(receipt.exists())

            _write_json(
                contract_path,
                {
                    "schema_version": 1,
                    "kind": "ace3_model24_python_runtime_contract",
                    "runtime_identity": runtime_identity,
                },
            )
            accepted = subprocess.run(
                [
                    str(python_path),
                    str(launcher),
                    "--project-python",
                    str(python_path),
                    "--entrypoint",
                    str(entrypoint),
                    "--runtime-contract",
                    str(contract_path),
                    "--receipt",
                    str(receipt),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            document = json.loads(receipt.read_text(encoding="ascii"))
            self.assertEqual(document["runtime_contract"], file_record(contract_path))


if __name__ == "__main__":
    unittest.main()
