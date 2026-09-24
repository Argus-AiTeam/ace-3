#!/usr/bin/env python3
"""Focused tests for non-executing fresh position-2 package preparation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import prepare_selected_token_position2_fresh_package as package  # noqa: E402


class Position2FreshPackageTests(unittest.TestCase):
    def make_repository(self, root: Path) -> tuple[Path, Path]:
        checkpoint_payload = b"authenticated-checkpoint-fixture\n"
        checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        validator_source = f'''
CHECKPOINT_SHA256 = "{checkpoint_sha256}"
EMBEDDING_SHA256 = "{"a" * 64}"
POSITION0_EMBEDDING_SHA256 = "{"b" * 64}"
POSITION1_EMBEDDING_SHA256 = "{"c" * 64}"
POSITION0_TOKEN_ID = 151644
POSITION1_TOKEN_ID = 2114
SELECTED_TOKEN_ID = 271
POSITION = 2
CONSUMED_SOURCE_PATHS = {{
    "validator": "ace3/model/validate_selected_token_position2_traversal.py",
    "dependency": "ace3/model/dependency.py",
    "makefile_target": "Makefile",
}}
'''
        makefile_payload = (
            b"model24-selected-token-position2-tests:\n"
            b"\t@true\n"
        )
        files = {
            package.VALIDATOR_RELATIVE: validator_source.encode("ascii"),
            Path("ace3/model/dependency.py"): b"DEPENDENCY = True\n",
            Path("Makefile"): makefile_payload,
            package.PACKAGE_TOOLING_RELATIVE[0]: Path(
                package.__file__
            ).read_bytes(),
            package.PACKAGE_TOOLING_RELATIVE[1]: Path(__file__).read_bytes(),
            package.CHECKPOINT_RELATIVE: checkpoint_payload,
            package.TENSOR_MAP_RELATIVE: b'{"fixture":true}\n',
        }
        for relative, payload in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        source_files = [
            {
                "bytes": (root / relative).stat().st_size,
                "path": relative.as_posix(),
                "sha256": package.sha256_file(root / relative),
            }
            for relative in sorted(
                {
                    package.VALIDATOR_RELATIVE,
                    Path("ace3/model/dependency.py"),
                    Path("Makefile"),
                },
                key=lambda item: item.as_posix(),
            )
        ]
        narrow_target = {
            "bytes": len(makefile_payload),
            "end_line": 2,
            "path": "Makefile",
            "sha256": hashlib.sha256(makefile_payload).hexdigest(),
            "start_line": 1,
            "target": "model24-selected-token-position2-tests",
        }
        canonical_source_set = {
            "files": source_files,
            "narrow_makefile_target": narrow_target,
            "schema": package.SOURCE_SET_SCHEMA,
        }
        canonical_payload = package.compact_canonical_json(
            canonical_source_set
        )
        source_review = (
            root
            / package.SOURCE_REVIEW_PARENT_RELATIVE
            / "ace3-position2-source-review-unit-test.json"
        )
        source_review.parent.mkdir(parents=True)
        source_review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": package.SOURCE_REVIEW_KIND,
                    "review_id": "ace3-position2-source-review-unit-test",
                    "verdict": "PASS",
                    "scope": {
                        "position": 2,
                        "selected_token_id": 271,
                        "review_type": (
                            "fresh read-only post-repair source review"
                        ),
                    },
                    "accepted_source_set": {
                        **canonical_source_set,
                        "canonical_digest": {
                            "algorithm": (
                                package.SOURCE_SET_DIGEST_ALGORITHM
                            ),
                            "canonical_bytes": len(canonical_payload),
                            "sha256": hashlib.sha256(
                                canonical_payload
                            ).hexdigest(),
                        },
                        "declared_source_labels": {
                            "validator": (
                                package.VALIDATOR_RELATIVE.as_posix()
                            ),
                            "dependency": "ace3/model/dependency.py",
                            "makefile_target": "Makefile",
                        },
                        "local_python_import_closure": {
                            "missing": [],
                            "status": "PASS",
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        source_review.chmod(0o444)
        return root, source_review

    @staticmethod
    def make_writable(root: Path) -> None:
        if not root.exists():
            return
        root.chmod(stat.S_IMODE(root.stat().st_mode) | 0o700)
        for path in root.rglob("*"):
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o600)

    def test_live_validator_contract_is_fresh_only(self) -> None:
        contract = package.validator_contract(package.ROOT)
        self.assertEqual(contract["POSITION"], 2)
        self.assertEqual(
            [
                contract["POSITION0_TOKEN_ID"],
                contract["POSITION1_TOKEN_ID"],
                contract["SELECTED_TOKEN_ID"],
            ],
            [151644, 2114, 271],
        )
        self.assertEqual(
            contract["source_paths"]["validator"],
            package.VALIDATOR_RELATIVE.as_posix(),
        )

    def test_prepare_and_validate_sealed_nonexecuting_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            _, source_review = self.make_repository(repository)
            run_id = "ace3-position2-fresh-unit-test-00001"
            package_root = (
                repository / package.PACKAGE_PARENT_RELATIVE / run_id
            )
            output_root = (
                repository / package.OUTPUT_PARENT_RELATIVE / run_id
            )
            try:
                manifest = package.prepare_package(
                    repository,
                    package_root,
                    output_root,
                    run_id,
                    source_review,
                )
                result = package.validate_package(repository, package_root)
                self.assertEqual(
                    manifest["status"],
                    "SEALED_AWAITING_INDEPENDENT_PACKAGE_REVIEW",
                )
                self.assertEqual(
                    manifest["accepted_source_review"]["verdict"],
                    "PASS",
                )
                self.assertEqual(
                    manifest["accepted_source_review"][
                        "accepted_source_set"
                    ]["sha256"],
                    package.load_json(source_review)[
                        "accepted_source_set"
                    ]["canonical_digest"]["sha256"],
                )
                self.assertEqual(result["execution_invocations"], 0)
                self.assertEqual(result["model_invocations"], 0)
                self.assertFalse(output_root.exists())
                self.assertEqual(
                    stat.S_IMODE(package_root.stat().st_mode) & 0o222,
                    0,
                )
            finally:
                self.make_writable(package_root)

    def test_validation_rejects_created_output_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            _, source_review = self.make_repository(repository)
            run_id = "ace3-position2-fresh-unit-test-00002"
            package_root = (
                repository / package.PACKAGE_PARENT_RELATIVE / run_id
            )
            output_root = (
                repository / package.OUTPUT_PARENT_RELATIVE / run_id
            )
            try:
                package.prepare_package(
                    repository,
                    package_root,
                    output_root,
                    run_id,
                    source_review,
                )
                output_root.mkdir(parents=True)
                with self.assertRaisesRegex(
                    package.PackageError,
                    "fresh output namespace is not absent",
                ):
                    package.validate_package(repository, package_root)
            finally:
                self.make_writable(package_root)

    def test_preparation_rejects_changed_reviewed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            _, source_review = self.make_repository(repository)
            dependency = repository / "ace3/model/dependency.py"
            dependency.write_bytes(b"DEPENDENCY = False\n")
            run_id = "ace3-position2-fresh-unit-test-00003"
            with self.assertRaisesRegex(
                package.PackageError,
                "accepted reviewed source changed",
            ):
                package.prepare_package(
                    repository,
                    repository / package.PACKAGE_PARENT_RELATIVE / run_id,
                    repository / package.OUTPUT_PARENT_RELATIVE / run_id,
                    run_id,
                    source_review,
                )

    def test_package_tool_has_no_execution_route(self) -> None:
        source = inspect.getsource(package)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("validate_selected_token_position2_traversal import", source)
        self.assertEqual(
            set(package.PERFORMED_ACTIONS),
            {
                "model_execution",
                "lm_head_work",
                "position3_work",
                "position4_work",
                "dialogue_work",
                "synthesis",
                "ppa_measurement",
                "fpga_work",
                "git_operation",
                "ace2_access",
                "direct_make_execution",
            },
        )


if __name__ == "__main__":
    unittest.main()
