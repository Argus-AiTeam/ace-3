#!/usr/bin/env python3
"""Coverage, freshness, and mutation tests for the systematic Model24 batch."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import (  # noqa: E402
    ContractError,
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    authenticate_tokenizer,
)
from official_model24_dialogue import DialogueExecutionError  # noqa: E402
from official_model24_showcase import _canonical_json, _sha256_bytes  # noqa: E402
from official_model24_systematic_continuations import (  # noqa: E402
    ARTIFACT_NAME,
    EXPECTED_CASE_COUNT,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_LANGUAGE_COUNTS,
    EXCLUDED_UNREVIEWED_EVIDENCE,
    MANIFEST_NAME,
    MARKDOWN_NAME,
    REVIEWED_BASELINE,
    RUN_LOG_NAME,
    SUMMARY_NAME,
    _load_prompt_suite,
    _parse_jsonl,
    _validate_rows,
    validate_directory,
)


class OfficialModel24SystematicContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vector_dir = (
            REPOSITORY_ROOT
            / "build"
            / "official_model24_systematic_continuations"
        )
        cls.summary = json.loads(
            (cls.vector_dir / SUMMARY_NAME).read_text(encoding="ascii")
        )
        cls.rows = _parse_jsonl((cls.vector_dir / ARTIFACT_NAME).read_bytes())
        cls.tokenizer = authenticate_tokenizer(DEFAULT_OFFICIAL_TOKENIZER_DIR)
        cls.metadata = {
            key: cls.summary[key]
            for key in (
                "provenance",
                "suite_binding",
                "asset_bindings",
                "model_binding",
                "tokenizer_binding",
                "numeric_profile",
                "runtime",
                "claim_boundary",
            )
        }

    def test_full_directory_regenerates_complete_authenticated_batch(self) -> None:
        summary = validate_directory(
            self.vector_dir,
            DEFAULT_OFFICIAL_CHECKPOINT,
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
        )
        self.assertEqual(summary["results"]["cases"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["results"]["completed_cases"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["results"]["execution_failures"], 0)
        self.assertGreaterEqual(summary["results"]["steps"], EXPECTED_CASE_COUNT * 2)
        self.assertEqual(
            {path.name for path in self.vector_dir.iterdir() if path.is_file()},
            {
                ARTIFACT_NAME,
                SUMMARY_NAME,
                MANIFEST_NAME,
                MARKDOWN_NAME,
                RUN_LOG_NAME,
            },
        )

    def test_missing_tokenizer_configuration_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-tokenizer"
            with self.assertRaisesRegex(
                ContractError,
                "pass --official-tokenizer-dir or set "
                "ACE3_OFFICIAL_MODEL24_TOKENIZER_DIR",
            ):
                authenticate_tokenizer(missing)

    def test_checked_in_suite_is_balanced_unique_and_ordered(self) -> None:
        suite = _load_prompt_suite()
        cases = suite["cases"]
        self.assertEqual(suite["reviewed_baseline"], REVIEWED_BASELINE)
        self.assertEqual(len(cases), EXPECTED_CASE_COUNT)
        self.assertEqual(len({case["id"] for case in cases}), EXPECTED_CASE_COUNT)
        self.assertEqual(
            {
                category: sum(case["category"] == category for case in cases)
                for category in EXPECTED_CATEGORY_COUNTS
            },
            EXPECTED_CATEGORY_COUNTS,
        )
        self.assertEqual(
            {
                language: sum(case["language"] == language for case in cases)
                for language in EXPECTED_LANGUAGE_COUNTS
            },
            EXPECTED_LANGUAGE_COUNTS,
        )
        self.assertEqual(
            [row["id"] for row in self.rows],
            [case["id"] for case in cases],
        )

    def test_every_step_records_lockstep_result_and_kv_lineage(self) -> None:
        for row in self.rows:
            self.assertIsNotNone(row["generation"])
            self.assertGreaterEqual(len(row["generation"]["steps"]), 2)
            previous_cache = None
            for step in row["generation"]["steps"]:
                self.assertEqual(
                    set(step["ace_vs_pytorch"]),
                    {"argmax", "terminal_hidden_tolerance", "logit_tolerance"},
                )
                self.assertTrue(
                    all(
                        result in ("agreement", "mismatch")
                        for result in step["ace_vs_pytorch"].values()
                    )
                )
                cache = step["cache_lineage"]
                self.assertEqual(len(cache["layers"]), 24)
                self.assertEqual(
                    cache["parent_cache_sha256"],
                    None if previous_cache is None else previous_cache["cache_sha256"],
                )
                previous_cache = cache

    def test_prompt_omission_duplicate_and_partial_batch_are_rejected(self) -> None:
        missing = copy.deepcopy(self.rows[:-1])
        with self.assertRaisesRegex(DialogueExecutionError, "row count"):
            _validate_rows(
                missing,
                self.metadata,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                require_complete=True,
            )

        duplicate = copy.deepcopy(self.rows)
        duplicate[-1] = copy.deepcopy(duplicate[-2])
        with self.assertRaisesRegex(DialogueExecutionError, "order or coverage"):
            _validate_rows(
                duplicate,
                self.metadata,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                require_complete=True,
            )

        partial = copy.deepcopy(self.rows)
        partial[0]["generation"] = None
        partial[0]["status"] = "execution_failure"
        partial[0]["raw_decoded_output"] = None
        partial[0]["max_hidden_abs_error"] = None
        partial[0]["max_logit_abs_error"] = None
        partial[0]["failures"] = [
            {
                "prompt_id": partial[0]["id"],
                "type": "execution_failure",
                "exception_type": "DialogueExecutionError",
                "message": "injected partial row",
            }
        ]
        with self.assertRaisesRegex(DialogueExecutionError, "partial rows"):
            _validate_rows(
                partial,
                self.metadata,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                require_complete=True,
            )

    def test_truncation_vacuity_and_token_mutation_are_rejected(self) -> None:
        payload = (self.vector_dir / ARTIFACT_NAME).read_bytes()
        with self.assertRaisesRegex(DialogueExecutionError, "truncated"):
            _parse_jsonl(payload[:-1])

        vacuous = copy.deepcopy(self.rows)
        vacuous[0]["generation"]["steps"] = []
        with self.assertRaisesRegex(DialogueExecutionError, "step count"):
            _validate_rows(
                vacuous,
                self.metadata,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                require_complete=True,
            )

        mutated = copy.deepcopy(self.rows)
        mutated[0]["generation"]["generated_token_ids"][0] += 1
        with self.assertRaisesRegex(DialogueExecutionError, "token stream"):
            _validate_rows(
                mutated,
                self.metadata,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                require_complete=True,
            )

    def test_stale_asset_binding_is_rejected_even_with_updated_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            vector_dir = Path(temporary)
            for name in (
                ARTIFACT_NAME,
                SUMMARY_NAME,
                MANIFEST_NAME,
                MARKDOWN_NAME,
                RUN_LOG_NAME,
            ):
                shutil.copyfile(self.vector_dir / name, vector_dir / name)
            summary = json.loads((vector_dir / SUMMARY_NAME).read_bytes())
            summary["asset_bindings"][0]["sha256"] = "0" * 64
            summary_payload = _canonical_json(summary)
            (vector_dir / SUMMARY_NAME).write_bytes(summary_payload)
            manifest = json.loads((vector_dir / MANIFEST_NAME).read_bytes())
            manifest["artifacts"][SUMMARY_NAME] = {
                "bytes": len(summary_payload),
                "sha256": _sha256_bytes(summary_payload),
            }
            (vector_dir / MANIFEST_NAME).write_bytes(_canonical_json(manifest))
            with self.assertRaisesRegex(DialogueExecutionError, "stale or mutated"):
                validate_directory(
                    vector_dir,
                    DEFAULT_OFFICIAL_CHECKPOINT,
                    DEFAULT_OFFICIAL_TOKENIZER_DIR,
                )

    def test_report_names_accepted_and_excluded_ancestry(self) -> None:
        report = (self.vector_dir / MARKDOWN_NAME).read_text(encoding="utf-8")
        self.assertIn(f"`{REVIEWED_BASELINE}`", report)
        self.assertIn(f"`{EXCLUDED_UNREVIEWED_EVIDENCE}`", report)
        self.assertIn("explicitly excluded", report)
        self.assertIn("No full-model RTL", report)


if __name__ == "__main__":
    unittest.main()
