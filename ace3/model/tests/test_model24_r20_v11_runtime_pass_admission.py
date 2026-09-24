#!/usr/bin/env python3
"""Tests for the r20/V11 zero-workload runtime-pass admission candidate."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ace3.model import validate_model24_r20_v11_runtime_pass_admission as contract


class RuntimePassAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from ace3.model import prepare_model24_r20_v11_runtime_pass_admission

        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.pristine = cls.root / "pristine"
        prepare_model24_r20_v11_runtime_pass_admission.build_candidate(
            cls.pristine
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._make_writable(cls.root)
        cls._temporary.cleanup()

    @staticmethod
    def _make_writable(root: Path) -> None:
        if not root.exists():
            return
        for path in root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        root.chmod(0o700)

    def _copy(self, name: str) -> Path:
        candidate = self.root / name
        shutil.copytree(self.pristine, candidate)
        self._make_writable(candidate)
        return candidate

    @staticmethod
    def _load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="ascii"))

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.write_bytes(contract.canonical_json(value))

    def _refresh_snapshot_record(
        self,
        candidate: Path,
        label: str,
    ) -> None:
        admission_path = candidate / "admission.json"
        admission = self._load(admission_path)
        relative_path = contract.SNAPSHOT_PATHS[label]
        payload = (candidate / relative_path).read_bytes()
        admission["snapshots"][label] = contract.snapshot_record(
            candidate,
            relative_path,
            contract.SNAPSHOT_SOURCES[label],
            payload,
        )
        self._write(admission_path, admission)

    def test_valid_candidate_binds_zero_workload_admission(self) -> None:
        result = contract.validate_candidate(self.pristine)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["admission_status"], "AWAITING_INDEPENDENT_REVIEW"
        )
        self.assertEqual(result["activity_counters"], "all_zero")
        self.assertFalse(result["authority_consumed"])
        self.assertFalse(result["execution_authorized"])

    def test_stale_preflight_is_rejected(self) -> None:
        candidate = self._copy("stale-preflight")
        path = candidate / contract.SNAPSHOT_PATHS[
            "v11_no_execution_preflight"
        ]
        value = self._load(path)
        value["current_state"]["observed_at_unix_ns"] += 1
        self._write(path, value)
        self._refresh_snapshot_record(candidate, "v11_no_execution_preflight")
        with self.assertRaisesRegex(SystemExit, "preflight SHA256 differs"):
            contract.validate_candidate(candidate)

    def test_receipt_chain_drift_is_rejected(self) -> None:
        candidate = self._copy("receipt-chain-drift")
        path = candidate / contract.SNAPSHOT_PATHS["receipt_chain_review"]
        value = self._load(path)
        value["review"]["status"] = "blocked"
        self._write(path, value)
        self._refresh_snapshot_record(candidate, "receipt_chain_review")
        with self.assertRaisesRegex(
            SystemExit, "receipt_chain_review SHA256 differs"
        ):
            contract.validate_candidate(candidate)

    def test_directive_substitution_and_absence_are_rejected(self) -> None:
        substituted = self._copy("directive-substitution")
        path = substituted / contract.SNAPSHOT_PATHS[
            "manager_v11_directive"
        ]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            SystemExit, "manager_v11_directive record differs"
        ):
            contract.validate_candidate(substituted)

        absent = self._copy("directive-absence")
        (absent / contract.SNAPSHOT_PATHS["manager_v11_directive"]).unlink()
        with self.assertRaises(FileNotFoundError):
            contract.validate_candidate(absent)

    def test_consumed_authority_is_rejected(self) -> None:
        candidate = self._copy("consumed-authority")
        path = candidate / contract.SNAPSHOT_PATHS[
            "adopted_project_authority_preimage"
        ]
        value = self._load(path)
        value["authority_consumed"] = True
        self._write(path, value)
        self._refresh_snapshot_record(
            candidate, "adopted_project_authority_preimage"
        )
        with self.assertRaisesRegex(
            SystemExit, "project_authority_preimage SHA256 differs"
        ):
            contract.validate_candidate(candidate)

    def test_replay_marker_is_rejected(self) -> None:
        candidate = self._copy("replay-marker")
        path = candidate / "admission.json"
        value = self._load(path)
        value["one_shot_controls"]["replay"] = True
        self._write(path, value)
        with self.assertRaisesRegex(SystemExit, "one-shot controls differ"):
            contract.validate_candidate(candidate)

    def test_launch_tuple_drift_is_rejected(self) -> None:
        candidate = self._copy("launch-tuple-drift")
        path = candidate / "admission.json"
        value = self._load(path)
        value["launch"]["cwd"] += "-drift"
        self._write(path, value)
        with self.assertRaisesRegex(SystemExit, "launch tuple differs"):
            contract.validate_candidate(candidate)


if __name__ == "__main__":
    unittest.main()
