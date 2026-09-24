from __future__ import annotations

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

import launch_next_position_traversal as launcher  # noqa: E402
import prepare_selected_token_feedback_input_package as input_package  # noqa: E402
from next_position_readiness_fixture import (  # noqa: E402
    build_cursor8_readiness,
)


class SelectedTokenFeedbackInputPackageTest(unittest.TestCase):
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
        cls.readiness = json.loads(
            cls.readiness_path.read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.readiness_directory.cleanup()

    def test_package_is_repeatable_bound_and_effect_free(self) -> None:
        parent_records = [
            self.readiness["cursor"][key]
            for key in (
                "authoritative_pointer",
                "generation_manifest",
                "ledger",
                "checkpoint007",
            )
        ]
        protected_paths = [
            self.readiness_path,
            *(Path(record["path"]) for record in parent_records),
            Path(self.readiness["selected_token"]["lm_head_receipt"]["path"]),
            Path(self.readiness["embedding_feedback"]["vector"]["path"]),
            Path(self.readiness["selected_token"]["official_tokenizer"]["path"]),
            Path(
                self.readiness["selected_token"][
                    "official_tokenizer_config"
                ]["path"]
            ),
        ]
        before = {path: path.read_bytes() for path in protected_paths}
        with tempfile.TemporaryDirectory() as temporary:
            output_a = Path(temporary) / "run-a/input-package.json"
            output_b = Path(temporary) / "run-b/input-package.json"
            with (
                mock.patch.object(launcher, "consume_authority") as consume,
                mock.patch.object(launcher.subprocess, "run") as traversal,
            ):
                package_a = input_package.emit_input_package(
                    self.readiness_path,
                    output_a,
                )
                package_b = input_package.emit_input_package(
                    self.readiness_path,
                    output_b,
                )
            consume.assert_not_called()
            traversal.assert_not_called()

            self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
            self.assertEqual(package_a, package_b)
            self.assertEqual(
                json.loads(output_a.read_text(encoding="utf-8")),
                package_a,
            )
            self.assertEqual(
                sorted(
                    path.relative_to(temporary).as_posix()
                    for path in Path(temporary).rglob("*")
                    if path.is_file()
                ),
                [
                    "run-a/input-package.json",
                    "run-b/input-package.json",
                ],
            )

        self.assertEqual(package_a["status"], "VERIFIED_INPUT_ONLY")
        self.assertEqual(package_a["selected_token"]["token_id"], 2)
        self.assertEqual(
            package_a["selected_token"]["decoded_text_fragment"],
            "#",
        )
        self.assertEqual(
            package_a["tied_embedding"]["tensor_shape"],
            [151936, 896],
        )
        self.assertEqual(
            package_a["tied_embedding"]["selected_row_shape"],
            [896],
        )
        self.assertEqual(
            package_a["tied_embedding"]["selected_row_semantic_sha256"],
            "8367b1f56e896acd2d99b64c3f0bd73f3090b8310ec7b294614074836a8af06a",
        )
        self.assertEqual(
            package_a["parentage"]["parent_checkpoint"],
            "checkpoint007",
        )
        self.assertEqual(package_a["parentage"]["parent_checkpoint_index"], 7)
        self.assertEqual(
            package_a["next_position_bounds"],
            {
                "transaction_index": 8,
                "layer_index": 7,
                "transaction_position": 3,
                "output_state_position": 4,
                "transaction_count_bound": 26,
                "model_layer_count_bound": 24,
            },
        )
        dry_run = package_a["dry_run_readiness_contract"]
        self.assertEqual(dry_run["status"], "NOT_READY")
        self.assertEqual(
            dry_run["launch_verdict"]["reason_code"],
            "MISSING_AUTHORITY_PACKAGE",
        )
        self.assertFalse(package_a["authority_package_present"])
        self.assertFalse(package_a["launch_authorized"])
        self.assertEqual(
            dry_run["zero_effects"],
            {field: False for field in launcher.ZERO_EFFECT_FIELDS},
        )
        self.assertEqual(
            {path: path.read_bytes() for path in protected_paths},
            before,
        )

    def test_host_dry_run_from_package_is_repeatable_and_effect_free(
        self,
    ) -> None:
        parent_records = [
            self.readiness["cursor"][key]
            for key in (
                "authoritative_pointer",
                "generation_manifest",
                "ledger",
                "checkpoint007",
            )
        ]
        protected_paths = [
            self.readiness_path,
            *(Path(record["path"]) for record in parent_records),
            Path(self.readiness["selected_token"]["lm_head_receipt"]["path"]),
            Path(self.readiness["embedding_feedback"]["vector"]["path"]),
            Path(self.readiness["selected_token"]["official_tokenizer"]["path"]),
            Path(
                self.readiness["selected_token"][
                    "official_tokenizer_config"
                ]["path"]
            ),
        ]
        before = {
            path: launcher.preflight.file_record(path)
            for path in protected_paths
        }
        with tempfile.TemporaryDirectory() as temporary:
            package_path = Path(temporary) / "selected-token-package.json"
            input_package.emit_input_package(
                self.readiness_path,
                package_path,
            )
            package_before = package_path.read_bytes()
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable,
                str(MODEL / "launch_next_position_traversal.py"),
                "--selected-token-feedback-package",
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
            reports = [json.loads(run.stdout) for run in completed]

            self.assertEqual([run.returncode for run in completed], [2, 2])
            self.assertEqual([run.stderr for run in completed], ["", ""])
            self.assertEqual(completed[0].stdout, completed[1].stdout)
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(package_path.read_bytes(), package_before)
            self.assertEqual(
                [
                    path.relative_to(temporary).as_posix()
                    for path in Path(temporary).rglob("*")
                    if path.is_file()
                ],
                ["selected-token-package.json"],
            )

            report = reports[0]
            validation = report[
                "selected_token_feedback_package_validation"
            ]
            self.assertEqual(validation["status"], "PASS")
            self.assertTrue(validation["package_sha256_validated"])
            self.assertEqual(
                validation["package"],
                launcher.preflight.file_record(package_path),
            )
            self.assertEqual(validation["token_id"], 2)
            self.assertEqual(validation["decoded_text_fragment"], "#")
            self.assertEqual(
                validation["embedding_semantic_sha256"],
                "8367b1f56e896acd2d99b64c3f0bd73f3090b8310ec7b294614074836a8af06a",
            )
            self.assertEqual(
                validation["parent_checkpoint_identity"],
                {
                    "name": "checkpoint007",
                    "index": 7,
                    "artifact": self.readiness["cursor"]["checkpoint007"],
                },
            )
            self.assertEqual(
                validation["next_position_bounds"],
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
                report["launch_verdict"]["reason_code"],
                "MISSING_AUTHORITY_PACKAGE",
            )
            self.assertEqual(
                report["zero_effects"],
                {
                    field: False
                    for field in launcher.ZERO_EFFECT_FIELDS
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
