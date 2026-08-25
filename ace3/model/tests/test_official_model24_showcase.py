#!/usr/bin/env python3
"""Focused validation and mutation tests for the official showcase."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import (  # noqa: E402
    DEFAULT_OFFICIAL_CHECKPOINT,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    authenticate_tokenizer,
)
from official_model24_dialogue import DialogueExecutionError  # noqa: E402
from official_model24_showcase import (  # noqa: E402
    ARTIFACT_NAME,
    PROMPT_SPECS,
    validate_directory,
    validate_showcase_document,
)


class OfficialModel24ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vector_dir = REPOSITORY_ROOT / "build" / "official_model24_showcase"
        cls.document = json.loads(
            (cls.vector_dir / ARTIFACT_NAME).read_text(encoding="ascii")
        )
        cls.tokenizer = authenticate_tokenizer(DEFAULT_OFFICIAL_TOKENIZER_DIR)

    def test_complete_showcase_directory(self) -> None:
        summary = validate_directory(
            self.vector_dir,
            DEFAULT_OFFICIAL_CHECKPOINT,
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
        )
        self.assertEqual(summary["prompts"], 6)
        self.assertEqual(summary["failures"], len(self.document["failures"]))
        self.assertGreaterEqual(summary["steps"], 12)

    def test_required_prompt_coverage_and_raw_outputs(self) -> None:
        rows = self.document["rows"]
        self.assertEqual(
            [row["id"] for row in rows],
            [spec["id"] for spec in PROMPT_SPECS],
        )
        self.assertIn("I am", [row["input_text"] for row in rows])
        self.assertIn(
            "The capital of France is",
            [row["input_text"] for row in rows],
        )
        self.assertTrue(
            all(
                row["raw_decoded_output"] == row["generation"]["decoded_text"]
                for row in rows
            )
        )

    def test_per_token_argmax_error_and_fp16_kv_lineage(self) -> None:
        for row in self.document["rows"]:
            steps = row["generation"]["steps"]
            self.assertGreaterEqual(len(steps), 2)
            for ordinal, step in enumerate(steps):
                self.assertTrue(
                    step["token"]["argmax_matches_independent_reference"]
                )
                self.assertTrue(
                    step["terminal_hidden"]["independent_reference"][
                        "within_tolerance"
                    ]
                    == (
                        step["terminal_hidden"]["independent_reference"][
                            "max_abs_error"
                        ]
                        <= step["terminal_hidden"]["independent_reference"][
                            "absolute_tolerance"
                        ]
                    )
                )
                self.assertTrue(
                    step["logits"]["independent_reference"]["within_tolerance"]
                    == (
                        step["logits"]["independent_reference"]["max_abs_error"]
                        <= step["logits"]["independent_reference"][
                            "absolute_tolerance"
                        ]
                    )
                )
                self.assertEqual(step["logits"]["vocab_size"], 151936)
                if ordinal:
                    self.assertEqual(
                        step["cache_lineage"]["parent_cache_sha256"],
                        steps[ordinal - 1]["cache_lineage"]["cache_sha256"],
                    )
            expected_failures = [
                failure
                for failure in self.document["failures"]
                if failure["prompt_id"] == row["id"]
            ]
            self.assertEqual(row["failures"], expected_failures)

    def test_prompt_and_token_mutations_are_rejected(self) -> None:
        missing_prompt = copy.deepcopy(self.document)
        missing_prompt["rows"].pop()
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "prompt coverage",
        ):
            validate_showcase_document(
                missing_prompt,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
            )

        false_token = copy.deepcopy(self.document)
        false_token["rows"][0]["generation"]["steps"][0]["token"][
            "argmax_matches_independent_reference"
        ] = False
        with self.assertRaisesRegex(DialogueExecutionError, "argmax comparison"):
            validate_showcase_document(
                false_token,
                self.tokenizer,
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
            )


if __name__ == "__main__":
    unittest.main()
