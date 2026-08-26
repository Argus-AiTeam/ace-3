#!/usr/bin/env python3
"""Trusted chain-tip and exact JSON type regression tests."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_first_voice_hybrid import (  # noqa: E402
    HIDDEN_SIZE,
    STATE_KIND,
    STATE_SCHEMA_VERSION,
    HybridRtlError,
    _canonical_bytes,
    _retained_hidden_sha256,
    _run_transaction,
    hash_file,
    load_trusted_tips_checkpoint,
    sha256_bytes,
    state_record_paths,
    state_tip_commitment,
    validate_state_envelope,
    write_json,
    write_trusted_tips_checkpoint,
)
from model24_oracle import (  # noqa: E402
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
)


class TrustedTipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = REPOSITORY_ROOT / "build" / "trusted_tip_unit" / self._testMethodName
        self._reset()

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def _reset(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self.tips: dict[int, dict[str, object]] = {}

    @staticmethod
    def _write_hidden(path: Path, offset: int) -> np.ndarray:
        values = np.asarray(
            [(index + offset) & 0xFFFF for index in range(HIDDEN_SIZE)],
            dtype="<u2",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                f"00{index:04x}{int(value):04x}\n"
                for index, value in enumerate(values)
            ),
            encoding="ascii",
        )
        return values

    def _write_record(self) -> tuple[Path, Path]:
        states_dir = self.work / "states"
        runtime_dir = self.work / "transactions"
        activation = self._write_hidden(
            runtime_dir / "position000/layer00/inputs.hex", 10
        )
        output = self._write_hidden(
            runtime_dir / "position000/layer00/raw/final.hex", 11
        )
        state_path, envelope_path = state_record_paths(states_dir, 0, 1)
        state_path.parent.mkdir(parents=True)
        state_path.write_bytes(b"x")
        envelope = {
            "schema_version": STATE_SCHEMA_VERSION,
            "kind": STATE_KIND,
            "model_binding": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            },
            "build_manifest_sha256": "1" * 64,
            "binary_sha256": "2" * 64,
            "layer_index": 0,
            "cache_slot": 0,
            "next_position": 1,
            "parent_state_sha256": None,
            "parent_envelope_sha256": None,
            "input_activation_sha256": sha256_bytes(_canonical_bytes(activation)),
            "output_hidden_sha256": sha256_bytes(_canonical_bytes(output)),
            "state": hash_file(state_path),
        }
        write_json(envelope_path, envelope)
        self.tips[0] = state_tip_commitment(envelope, envelope_path.read_bytes())
        return state_path, envelope_path

    def _validate(
        self,
        state_path: Path,
        envelope_path: Path,
        expected_tip: object,
    ) -> dict[str, object]:
        return validate_state_envelope(
            envelope_path,
            state_path,
            states_dir=self.work / "states",
            runtime_dir=self.work / "transactions",
            layer_index=0,
            next_position=1,
            build_manifest_sha256="1" * 64,
            binary_sha256="2" * 64,
            expected_tip=expected_tip,
        )

    def test_self_rehashed_tip_artifacts_are_rejected(self) -> None:
        for artifact in ("state", "activation", "output"):
            with self.subTest(artifact=artifact):
                self._reset()
                state_path, envelope_path = self._write_record()
                envelope = json.loads(envelope_path.read_text(encoding="ascii"))
                if artifact == "state":
                    state_path.write_bytes(b"y")
                    envelope["state"] = hash_file(state_path)
                else:
                    relative = (
                        "position000/layer00/inputs.hex"
                        if artifact == "activation"
                        else "position000/layer00/raw/final.hex"
                    )
                    path = self.work / "transactions" / relative
                    self._write_hidden(path, 100 if artifact == "activation" else 101)
                    field = (
                        "input_activation_sha256"
                        if artifact == "activation"
                        else "output_hidden_sha256"
                    )
                    envelope[field] = _retained_hidden_sha256(path, artifact)
                write_json(envelope_path, envelope)
                with self.assertRaises(HybridRtlError):
                    self._validate(state_path, envelope_path, self.tips[0])

    def test_missing_wrong_and_pre_restore_tip_fail_closed(self) -> None:
        state_path, envelope_path = self._write_record()
        with self.assertRaises(HybridRtlError):
            self._validate(state_path, envelope_path, None)
        wrong = copy.deepcopy(self.tips[0])
        wrong["envelope"]["sha256"] = "9" * 64
        with self.assertRaises(HybridRtlError):
            self._validate(state_path, envelope_path, wrong)
        with mock.patch("model24_first_voice_hybrid.subprocess.run") as run:
            with self.assertRaises(HybridRtlError):
                _run_transaction(
                    repository_root=REPOSITORY_ROOT,
                    compiled_dir=self.work / "compiled",
                    vector_dir=self.work / "vectors",
                    runtime_dir=self.work / "transactions",
                    states_dir=self.work / "states",
                    layer_index=0,
                    position=1,
                    hidden_bits=np.zeros(HIDDEN_SIZE, dtype="<u2"),
                    build_manifest_sha256="1" * 64,
                    binary_sha256="2" * 64,
                    trusted_tips={},
                )
        run.assert_not_called()

    def test_in_memory_and_authenticated_checkpoint_resume(self) -> None:
        state_path, envelope_path = self._write_record()
        self.assertEqual(
            self._validate(state_path, envelope_path, self.tips[0])["state"],
            hash_file(state_path),
        )
        checkpoint = self.work / "tips.json"
        expected = write_trusted_tips_checkpoint(
            checkpoint,
            self.tips,
            build_manifest_sha256="1" * 64,
        )
        restored = load_trusted_tips_checkpoint(
            checkpoint,
            expected,
            build_manifest_sha256="1" * 64,
        )
        self._validate(state_path, envelope_path, restored[0])
        with self.assertRaises(HybridRtlError):
            load_trusted_tips_checkpoint(
                checkpoint, None, build_manifest_sha256="1" * 64
            )
        wrong = dict(expected)
        wrong["sha256"] = "9" * 64
        with self.assertRaises(HybridRtlError):
            load_trusted_tips_checkpoint(
                checkpoint, wrong, build_manifest_sha256="1" * 64
            )
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
        with self.assertRaises(HybridRtlError):
            load_trusted_tips_checkpoint(
                checkpoint, expected, build_manifest_sha256="1" * 64
            )

    def test_every_integer_field_rejects_bool_and_float(self) -> None:
        envelope_paths = (
            ("schema_version",),
            ("layer_index",),
            ("cache_slot",),
            ("next_position",),
            ("state", "bytes"),
        )
        tip_paths = (
            ("schema_version",),
            ("layer_index",),
            ("cache_slot",),
            ("next_position",),
            ("envelope", "bytes"),
            ("state", "bytes"),
        )
        for source, paths in (("envelope", envelope_paths), ("tip", tip_paths)):
            for path in paths:
                for kind in ("bool", "float"):
                    with self.subTest(source=source, path=path, kind=kind):
                        self._reset()
                        state_path, envelope_path = self._write_record()
                        document = (
                            json.loads(envelope_path.read_text(encoding="ascii"))
                            if source == "envelope"
                            else copy.deepcopy(self.tips[0])
                        )
                        target = document
                        for key in path[:-1]:
                            target = target[key]
                        value = target[path[-1]]
                        target[path[-1]] = bool(value) if kind == "bool" else float(value)
                        if source == "envelope":
                            write_json(envelope_path, document)
                            expected_tip = copy.deepcopy(self.tips[0])
                            expected_tip["envelope"] = hash_file(envelope_path)
                        else:
                            expected_tip = document
                        with self.assertRaises(HybridRtlError):
                            self._validate(state_path, envelope_path, expected_tip)

        self._reset()
        self._write_record()
        checkpoint = self.work / "tips.json"
        write_trusted_tips_checkpoint(
            checkpoint,
            self.tips,
            build_manifest_sha256="1" * 64,
        )
        for replacement in (True, 1.0):
            document = json.loads(checkpoint.read_text(encoding="ascii"))
            document["schema_version"] = replacement
            write_json(checkpoint, document)
            with self.assertRaises(HybridRtlError):
                load_trusted_tips_checkpoint(
                    checkpoint,
                    hash_file(checkpoint),
                    build_manifest_sha256="1" * 64,
                )
            document["schema_version"] = 1
            write_json(checkpoint, document)


if __name__ == "__main__":
    unittest.main()
