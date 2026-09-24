#!/usr/bin/env python3
"""Focused tests for the pre-authority durable-runner registry gate."""

from __future__ import annotations

import os
import io
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import preflight_argus_subagent_registry as preflight  # noqa: E402


REPOSITORY = Path(__file__).resolve().parents[3]
PROBE_TASK_ID = "ace3-position2-fresh-source-archive-preflight-unit-00001"


def write_source_archive(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as stream:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            stream.addfile(info, io.BytesIO(payload))


class ArgusSubagentRegistryPreflightTests(unittest.TestCase):
    def test_real_project_root_passes_from_0555_evidence_cwd(self) -> None:
        scratch_parent = REPOSITORY / "build/preflight-source-archive-tests"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_parent) as directory:
            evidence_root = Path(directory) / "immutable-evidence"
            evidence_root.mkdir()
            evidence_root.chmod(0o555)
            runtime_output = Path(directory) / "runtime-output"
            validator_output = Path(directory) / "validator-output"
            authority_consumed = Path(directory) / "future-authority-consumed.json"
            source_archive = Path(directory) / "source-archive.tar"
            write_source_archive(
                source_archive,
                {"Makefile": (REPOSITORY / "Makefile").read_bytes()},
            )
            build_root = Path(directory) / "intended-build-boundary"
            original_cwd = Path.cwd()
            try:
                os.chdir(evidence_root)
                result = preflight.preflight_registry(
                    REPOSITORY,
                    authority_consumed,
                    canonical_task_id=PROBE_TASK_ID,
                    evidence_root=evidence_root,
                    forbidden_paths=(runtime_output, validator_output),
                    source_archive=source_archive.relative_to(REPOSITORY),
                    build_root=build_root,
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
            archive_gate = result["source_archive_preflight"]
            self.assertEqual(archive_gate["status"], "PASS")
            self.assertEqual(archive_gate["cwd"], str(REPOSITORY))
            self.assertEqual(
                archive_gate["source_archive"]["path"],
                str(source_archive),
            )
            self.assertEqual(
                archive_gate["make_target"],
                "model24-rtl-layer-compile",
            )
            self.assertEqual(archive_gate["build_root"], str(build_root))
            self.assertEqual(
                set(archive_gate["make_target_resolution"]),
                {"0", "23"},
            )
            for layer in ("0", "23"):
                self.assertEqual(
                    archive_gate["make_target_resolution"][layer]["exit_code"],
                    0,
                )
            self.assertEqual(archive_gate["runtime_invocations"], 0)
            self.assertEqual(archive_gate["validator_invocations"], 0)
            self.assertFalse(build_root.exists())
            self.assertEqual(result["canonical_task_id"], PROBE_TASK_ID)
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
            self.assertFalse(authority_consumed.exists())
            self.assertFalse(runtime_output.exists())
            self.assertFalse(validator_output.exists())

    def test_missing_archive_target_fails_before_registry_or_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Makefile").write_text(
                ".PHONY: model24-rtl-layer-compile\n"
                "model24-rtl-layer-compile:\n"
                "\t@echo live-root-target-would-mask-archive-drift\n",
                encoding="ascii",
            )
            source_archive = root / "build/source-archive.tar"
            write_source_archive(
                source_archive,
                {
                    "Makefile": (
                        b".PHONY: other-target\n"
                        b"other-target:\n"
                        b"\t@echo wrong archive\n"
                    )
                },
            )
            marker = root / "future-authority-consumed.json"
            build_root = root / "build/preflight-boundary"

            with self.assertRaisesRegex(
                preflight.RegistryPreflightError,
                "sealed source Makefile target is missing",
            ):
                preflight.preflight_registry(
                    root,
                    marker,
                    canonical_task_id=(
                        "ace3-position2-fresh-source-archive-preflight-unit-00002"
                    ),
                    source_archive=source_archive.relative_to(root),
                    build_root=build_root,
                )

            self.assertFalse(marker.exists())
            self.assertFalse((root / ".argus_subagents").exists())
            self.assertFalse(build_root.exists())

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
