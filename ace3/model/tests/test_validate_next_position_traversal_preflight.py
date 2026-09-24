from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
TESTS = Path(__file__).resolve().parent
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import validate_next_position_traversal_preflight as preflight  # noqa: E402
import launch_next_position_traversal as launcher  # noqa: E402
from next_position_readiness_fixture import (  # noqa: E402
    build_cursor8_readiness,
)


class NextPositionTraversalPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readiness_directory = tempfile.TemporaryDirectory()
        try:
            cls.readiness_path = build_cursor8_readiness(
                ROOT,
                Path(cls.readiness_directory.name),
            )
        except FileNotFoundError as error:
            cls.readiness_directory.cleanup()
            raise unittest.SkipTest(str(error)) from error
        cls.document = json.loads(
            cls.readiness_path.read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.readiness_directory.cleanup()

    def test_cursor8_checkpoint007_inputs_pass_without_launch_authority(
        self,
    ) -> None:
        report = preflight.validate_preflight(self.readiness_path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["inputs_status"], "READY_INPUTS")
        self.assertEqual(report["launch_status"], "NOT_READY")
        self.assertFalse(report["launch_authorized"])
        self.assertFalse(report["execution_performed"])
        self.assertEqual(report["generation"], 8)
        self.assertEqual(report["cursor"], 8)
        self.assertEqual(report["parent_checkpoint_index"], 7)
        self.assertEqual(report["transaction_index"], 8)
        self.assertEqual(report["layer_index"], 7)
        self.assertEqual(report["transaction_position"], 3)
        self.assertEqual(report["output_state_position"], 4)
        self.assertEqual(report["authenticated_fixture_tensor_count"], 26)
        self.assertEqual(report["authenticated_rope_coefficient_rows"], 96)

    def test_missing_selected_token_feedback_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.document)
        candidate["embedding_feedback"]["vector"]["path"] = str(
            ROOT / "build/absent-selected-token-feedback.hex"
        )
        with self.assertRaisesRegex(
            preflight.PreflightError,
            "feedback vector|artifact is missing",
        ):
            preflight.validate_preflight_document(candidate)

    def test_mismatched_parentage_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.document)
        candidate["cursor"]["checkpoint"] = 6
        with self.assertRaisesRegex(
            preflight.PreflightError,
            "cursor, generation, or parent checkpoint bounds differ",
        ):
            preflight.validate_preflight_document(candidate)

    def test_mismatched_fixture_binding_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.document)
        candidate["next_layer_inputs"]["layer7_fixture_manifest"]["sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(
            preflight.PreflightError,
            "fixture|binding",
        ):
            preflight.validate_preflight_document(candidate)

    def test_mismatched_rope_and_cache_position_fails_closed(self) -> None:
        candidate = copy.deepcopy(self.document)
        candidate["next_layer_inputs"]["transaction_position"] = 4
        with self.assertRaisesRegex(
            preflight.PreflightError,
            "transaction, layer, or input-binding bounds differ",
        ):
            preflight.validate_preflight_document(candidate)

    def test_launcher_preserves_preflight_and_refuses_missing_authority(
        self,
    ) -> None:
        protected = [
            self.readiness_path,
            Path(self.document["cursor"]["authoritative_pointer"]["path"]),
            Path(self.document["cursor"]["generation_manifest"]["path"]),
            Path(self.document["cursor"]["ledger"]["path"]),
            Path(self.document["cursor"]["checkpoint007"]["path"]),
        ]
        before = {path: path.read_bytes() for path in protected}
        with mock.patch.object(launcher.subprocess, "run") as run:
            report, returncode = launcher.launch(self.readiness_path, None)
        self.assertEqual(returncode, 2)
        self.assertEqual(report["status"], "REFUSED")
        self.assertEqual(report["reason_code"], "MISSING_AUTHORITY_PACKAGE")
        self.assertEqual(report["reason"], "authority package was not provided")
        self.assertEqual(report["preflight"]["status"], "PASS")
        self.assertEqual(report["preflight"]["inputs_status"], "READY_INPUTS")
        self.assertEqual(report["preflight"]["transaction_index"], 8)
        self.assertFalse(report["authority_package_created"])
        self.assertFalse(report["authority_consumed"])
        self.assertFalse(report["consumption_artifact_created"])
        self.assertFalse(report["traversal_performed"])
        self.assertFalse(report["traversal_artifact_created"])
        self.assertFalse(report["runtime_state_mutated"])
        self.assertFalse(report["transaction_consumed"])
        self.assertFalse(report["execution_artifact_created"])
        self.assertFalse(report["publication_performed"])
        self.assertFalse(report["publication_artifact_created"])
        run.assert_not_called()
        self.assertEqual(
            {path: path.read_bytes() for path in protected},
            before,
        )

    def test_host_entrypoint_refuses_cursor8_without_creating_artifacts(
        self,
    ) -> None:
        protected = [
            self.readiness_path,
            Path(self.document["cursor"]["authoritative_pointer"]["path"]),
            Path(self.document["cursor"]["generation_manifest"]["path"]),
            Path(self.document["cursor"]["ledger"]["path"]),
            Path(self.document["cursor"]["checkpoint007"]["path"]),
        ]
        before = {path: path.read_bytes() for path in protected}
        with tempfile.TemporaryDirectory() as temporary:
            launch_namespace = Path(temporary) / "transaction008-launch"
            authority = launch_namespace / "authority-package.json"
            output = launch_namespace / "execution-result.json"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODEL / "launch_next_position_traversal.py"),
                    "--readiness",
                    str(self.readiness_path),
                    "--authority-package",
                    str(authority),
                ],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stderr, "")
            self.assertEqual(report["status"], "REFUSED")
            self.assertEqual(
                report["reason_code"],
                "MISSING_AUTHORITY_PACKAGE",
            )
            self.assertEqual(report["preflight"]["generation"], 8)
            self.assertEqual(report["preflight"]["cursor"], 8)
            self.assertEqual(
                report["preflight"]["parent_checkpoint_index"],
                7,
            )
            for field in (
                "authority_package_created",
                "authority_consumed",
                "consumption_artifact_created",
                "traversal_performed",
                "traversal_artifact_created",
                "runtime_state_mutated",
                "transaction_consumed",
                "execution_artifact_created",
                "publication_performed",
                "publication_artifact_created",
            ):
                self.assertFalse(report[field], field)
            self.assertFalse(authority.exists())
            self.assertFalse(output.exists())
            self.assertFalse(launch_namespace.exists())
        self.assertEqual(
            {path: path.read_bytes() for path in protected},
            before,
        )

    def test_host_dry_run_contract_is_stable_complete_and_effect_free(
        self,
    ) -> None:
        parent_checkpoint = self.document["cursor"]["checkpoint007"]
        protected = [
            self.readiness_path,
            Path(self.document["cursor"]["authoritative_pointer"]["path"]),
            Path(self.document["cursor"]["generation_manifest"]["path"]),
            Path(self.document["cursor"]["ledger"]["path"]),
            Path(parent_checkpoint["path"]),
        ]
        before = {path: path.read_bytes() for path in protected}
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            sys.executable,
            str(MODEL / "launch_next_position_traversal.py"),
            "--readiness",
            str(self.readiness_path),
            "--dry-run-readiness",
        ]

        completed = [
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            for _ in range(2)
        ]
        reports = [json.loads(run.stdout) for run in completed]

        self.assertEqual([run.returncode for run in completed], [2, 2])
        self.assertEqual([run.stderr for run in completed], ["", ""])
        self.assertEqual(completed[0].stdout, completed[1].stdout)
        self.assertEqual(reports[0], reports[1])

        report = reports[0]
        self.assertEqual(
            report["kind"],
            "ace3_next_position_runtime_readiness_contract",
        )
        self.assertEqual(report["status"], "NOT_READY")
        self.assertEqual(
            report["runtime_identity"],
            self.document["runtime_identity"],
        )
        self.assertEqual(
            report["cursor_identity"],
            {
                "cursor_index": 8,
                "generation": 8,
                "parent_generation": 7,
                "parent_checkpoint": "checkpoint007",
                "parent_checkpoint_index": 7,
            },
        )
        expected_parent_artifacts = {
            key: self.document["cursor"][key]
            for key in (
                "authoritative_pointer",
                "generation_manifest",
                "ledger",
                "checkpoint007",
            )
        }
        self.assertEqual(
            report["required_parent_artifacts"],
            expected_parent_artifacts,
        )
        for record in report["required_parent_artifacts"].values():
            self.assertEqual(
                preflight.file_record(Path(record["path"])),
                record,
            )
        self.assertEqual(
            report["next_position_bounds"],
            {
                "transaction_index": 8,
                "layer_index": 7,
                "transaction_position": 3,
                "output_state_position": 4,
                "transaction_count_bound": 26,
                "model_layer_count_bound": 24,
            },
        )
        self.assertEqual(
            report["launch_verdict"],
            {
                "status": "REFUSED",
                "launch_authorized": False,
                "reason_code": "MISSING_AUTHORITY_PACKAGE",
                "reason": "authority package was not provided",
            },
        )
        self.assertEqual(
            report["zero_effects"],
            {
                field: False
                for field in launcher.ZERO_EFFECT_FIELDS
            },
        )
        self.assertEqual(
            {path: path.read_bytes() for path in protected},
            before,
        )

    def test_launcher_reports_invalid_authority_without_runtime_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            authority = Path(temporary) / "invalid-authority.json"
            authority.write_text(
                '{"schema_version":1,"kind":"not-an-authority"}\n',
                encoding="utf-8",
            )
            with mock.patch.object(launcher.subprocess, "run") as run:
                report, returncode = launcher.launch(
                    self.readiness_path,
                    authority,
                )
        self.assertEqual(returncode, 2)
        self.assertEqual(report["status"], "REFUSED")
        self.assertEqual(report["reason_code"], "INVALID_AUTHORITY_PACKAGE")
        self.assertEqual(
            report["reason"],
            "authority schema or kind differs from the pending transaction",
        )
        self.assertEqual(report["preflight"]["status"], "PASS")
        self.assertFalse(report["authority_package_created"])
        self.assertFalse(report["authority_consumed"])
        self.assertFalse(report["consumption_artifact_created"])
        self.assertFalse(report["traversal_performed"])
        self.assertFalse(report["traversal_artifact_created"])
        self.assertFalse(report["runtime_state_mutated"])
        self.assertFalse(report["transaction_consumed"])
        self.assertFalse(report["execution_artifact_created"])
        self.assertFalse(report["publication_performed"])
        self.assertFalse(report["publication_artifact_created"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
