#!/usr/bin/env python3
"""Focused tests for the pre-authority durable-runner registry gate."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import preflight_argus_subagent_registry as preflight  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[3]
V5_TASK_ID = "ace3-position2-fresh-v5-20260831T032100Z"
V5_AUTHORITY_CONSUMED = (
    REPOSITORY
    / "build/model24_selected_token_position2_authority_consumed"
    / f"{V5_TASK_ID}.json"
)


class ArgusSubagentRegistryPreflightTests(unittest.TestCase):
    def test_real_project_root_passes_from_0555_evidence_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "immutable-evidence"
            evidence_root.mkdir()
            evidence_root.chmod(0o555)
            runtime_output = Path(directory) / "runtime-output"
            validator_output = Path(directory) / "validator-output"
            original_cwd = Path.cwd()
            try:
                os.chdir(evidence_root)
                result = preflight.preflight_registry(
                    REPOSITORY,
                    V5_AUTHORITY_CONSUMED,
                    canonical_task_id=V5_TASK_ID,
                    evidence_root=evidence_root,
                    forbidden_paths=(runtime_output, validator_output),
                )
            finally:
                os.chdir(original_cwd)
                evidence_root.chmod(0o755)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["submitter_cwd"], str(REPOSITORY))
            self.assertEqual(
                result["registry"]["path"],
                str(REPOSITORY / ".argus_subagents"),
            )
            self.assertEqual(
                result["real_bootstrap"]["cwd"],
                str(REPOSITORY),
            )
            self.assertEqual(result["canonical_task_id"], V5_TASK_ID)
            self.assertEqual(result["canonical_task_allocations"], 0)
            self.assertEqual(result["durable_runner_submissions"], 0)
            self.assertEqual(result["payload_invocations"], 0)
            self.assertEqual(result["runtime_invocations"], 0)
            self.assertEqual(result["validator_invocations"], 0)
            self.assertEqual(result["authority_consumptions"], 0)
            self.assertFalse(
                (evidence_root / ".argus_subagents").exists(),
                "v4 regression: registry must not be created below evidence cwd",
            )
            self.assertFalse(V5_AUTHORITY_CONSUMED.exists())
            self.assertFalse(runtime_output.exists())
            self.assertFalse(validator_output.exists())

    def test_real_bootstrap_creates_missing_project_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"

            result = preflight.preflight_registry(
                root,
                marker,
                canonical_task_id="ace3-position2-fresh-unit-test-00001",
            )

            self.assertEqual(result["status"], "PASS")
            self.assertRegex(
                result["registry"]["probe_name"],
                r"^\.argus-registry-preflight-[0-9a-f]{32}$",
            )
            self.assertTrue(registry.is_dir())
            self.assertEqual(list(registry.iterdir()), [])
            self.assertFalse(marker.exists())

    def test_successive_preflights_use_distinct_probe_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"
            registry.mkdir()

            first = preflight.preflight_registry(
                root,
                marker,
                canonical_task_id="ace3-position2-fresh-unit-test-00002",
            )
            second = preflight.preflight_registry(
                root,
                marker,
                canonical_task_id="ace3-position2-fresh-unit-test-00002",
            )

            self.assertNotEqual(
                first["registry"]["probe_name"],
                second["registry"]["probe_name"],
            )
            self.assertEqual(list(registry.iterdir()), [])
            self.assertFalse(marker.exists())

    def test_dangling_symlink_fails_before_authority_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"
            registry.symlink_to(root / "missing-registry", target_is_directory=True)

            with self.assertRaisesRegex(
                preflight.RegistryPreflightError,
                "real subagent registry bootstrap probe failed",
            ):
                preflight.preflight_registry(root, marker)

            self.assertFalse(marker.exists())
            self.assertTrue(registry.is_symlink())

    def test_resolving_symlink_is_also_rejected_before_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "registry-target"
            target.mkdir()
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"
            registry.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(
                preflight.RegistryPreflightError,
                "real subagent registry bootstrap probe failed",
            ):
                preflight.preflight_registry(root, marker)

            self.assertFalse(marker.exists())
            self.assertEqual(list(target.iterdir()), [])

    def test_non_directory_registry_fails_without_consuming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"

            registry.write_text("not a directory\n", encoding="ascii")
            with self.assertRaisesRegex(
                preflight.RegistryPreflightError,
                "real subagent registry bootstrap probe failed",
            ):
                preflight.preflight_registry(root, marker)
            self.assertFalse(marker.exists())

    def test_existing_consumption_marker_blocks_before_registry_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / ".argus_subagents"
            marker = root / "future-authority-consumed.json"
            registry.mkdir()
            marker.write_text("{}\n", encoding="ascii")

            with self.assertRaisesRegex(
                preflight.RegistryPreflightError,
                "already consumed",
            ):
                preflight.preflight_registry(root, marker)

            self.assertEqual(list(registry.iterdir()), [])
            self.assertEqual(marker.read_text(encoding="ascii"), "{}\n")

    def test_unwritable_control_root_fails_before_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "control"
            root.mkdir()
            marker = root / "future-authority-consumed.json"
            root.chmod(0o555)
            try:
                with self.assertRaisesRegex(
                    preflight.RegistryPreflightError,
                    "real subagent registry bootstrap probe failed",
                ):
                    preflight.preflight_registry(root, marker)
                self.assertFalse(marker.exists())
                self.assertFalse((root / ".argus_subagents").exists())
                self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
            finally:
                root.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
