from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
TESTS = Path(__file__).resolve().parent
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import launch_next_position_traversal as launcher  # noqa: E402
import prepare_selected_token_feedback_input_package as feedback  # noqa: E402
import prepare_transaction8_layer7_authority_readiness as readiness  # noqa: E402
from next_position_readiness_fixture import (  # noqa: E402
    build_cursor8_readiness,
)


TRANSACTION8_PACKAGE_MANIFEST = ROOT / (
    "build/model24_selected_token_position3_runs/"
    "ace3-position3-fresh-r11-20260831t215500z/"
    "transaction8-layer7-continuation-package-r5/package-manifest.json"
)


class Transaction8Layer7AuthorityReadinessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TRANSACTION8_PACKAGE_MANIFEST.is_file():
            raise unittest.SkipTest(
                "reviewer-gated transaction008 package manifest is absent"
            )
        gate_manifest = json.loads(
            TRANSACTION8_PACKAGE_MANIFEST.read_text(encoding="utf-8")
        )
        cls.post_publication_review = Path(
            gate_manifest["post_publication_independent_review"]["path"]
        )
        cls.reviewer_role_receipt = Path(
            gate_manifest["host_reviewer_role_receipt"]["path"]
        )
        if not (
            cls.post_publication_review.is_file()
            and cls.reviewer_role_receipt.is_file()
        ):
            raise unittest.SkipTest(
                "Reviewer gate or durable role receipt is absent"
            )

    def test_host_validates_twice_deterministically_without_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            cursor8_readiness = build_cursor8_readiness(
                ROOT,
                temporary_path / "cursor8",
            )
            feedback_path = temporary_path / "selected-token-feedback.json"
            feedback.emit_input_package(cursor8_readiness, feedback_path)
            package_path = temporary_path / "authority-readiness.json"
            package = readiness.emit_authority_readiness_package(
                feedback_path,
                self.post_publication_review,
                self.reviewer_role_receipt,
                package_path,
            )
            protected_paths = {
                cursor8_readiness,
                feedback_path,
                package_path,
                self.post_publication_review,
                self.reviewer_role_receipt,
                *(
                    Path(record["path"])
                    for record in package["bindings"][
                        "required_parent_artifacts"
                    ].values()
                ),
            }
            before = {
                path: launcher.preflight.file_record(path)
                for path in protected_paths
            }
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                str(MODEL / "launch_next_position_traversal.py"),
                "--authority-readiness-package",
                str(package_path),
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

            self.assertEqual([run.returncode for run in completed], [0, 0])
            self.assertEqual([run.stderr for run in completed], ["", ""])
            self.assertEqual(completed[0].stdout, completed[1].stdout)
            reports = [json.loads(run.stdout) for run in completed]
            self.assertEqual(reports[0], reports[1])

            report = reports[0]
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                report["kind"],
                "ace3_transaction8_layer7_authority_readiness_validation",
            )
            self.assertEqual(
                report["cursor_identity"],
                {
                    "generation": 8,
                    "cursor": 8,
                    "parent_generation": 7,
                    "parent_checkpoint": "checkpoint007",
                    "parent_checkpoint_index": 7,
                },
            )
            self.assertEqual(
                report["transaction_identity"]["transaction_index"],
                8,
            )
            self.assertEqual(report["transaction_identity"]["layer_index"], 7)
            self.assertEqual(
                report["transaction_identity"]["permitted_transaction_indices"],
                [8],
            )
            self.assertEqual(
                report["launch_verdict"],
                {
                    "status": "AUTHORITY_VALIDATED",
                    "reason_code": "EXECUTION_NOT_REQUESTED",
                    "launch_authorized": False,
                    "authority_missing": False,
                },
            )
            self.assertEqual(
                report["zero_effects"],
                {
                    field: False
                    for field in launcher.ZERO_EFFECT_FIELDS
                },
            )
            future = report["transactions009_025"]
            self.assertEqual(future["transaction_indices"], list(range(9, 26)))
            self.assertEqual(
                {
                    key: future[key]
                    for key in launcher.FUTURE_TRANSACTION_EFFECT_FIELDS
                },
                {
                    field: False
                    for field in launcher.FUTURE_TRANSACTION_EFFECT_FIELDS
                },
            )
            self.assertEqual(
                {
                    path: launcher.preflight.file_record(path)
                    for path in protected_paths
                },
                before,
            )


if __name__ == "__main__":
    unittest.main()
