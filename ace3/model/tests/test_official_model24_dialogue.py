#!/usr/bin/env python3
"""Focused lineage, stop, and mutation tests for Model24 dialogue evidence."""

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

from model24_execution_oracle import EOS_TOKEN_ID, FIXED_CHAT_TOKEN_IDS  # noqa: E402
from official_model24_dialogue import (  # noqa: E402
    ARTIFACT_NAME,
    DialogueExecutionError,
    generation_stop_reason,
    validate_document,
)


class OfficialModel24DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact = (
            REPOSITORY_ROOT
            / "build"
            / "official_model24_dialogue"
            / ARTIFACT_NAME
        )
        cls.document = json.loads(artifact.read_text(encoding="ascii"))

    def test_complete_readable_dialogue_evidence(self) -> None:
        summary = validate_document(self.document)
        generation = self.document["generation"]
        self.assertEqual(
            summary["generated_token_ids"],
            generation["generated_token_ids"],
        )
        self.assertEqual(summary["decoded_text"], generation["decoded_text"])
        self.assertEqual(summary["stop_reason"], generation["stop_reason"])
        self.assertEqual(summary["steps"], len(generation["steps"]))
        self.assertTrue(summary["decoded_text"])

    def test_cache_lineage_growth_is_non_vacuous(self) -> None:
        steps = self.document["generation"]["steps"]
        self.assertEqual(
            [step["cache_lineage"]["position_count"] for step in steps],
            [25, 26, 27, 28],
        )
        self.assertEqual(
            steps[1]["cache_lineage"]["parent_cache_sha256"],
            steps[0]["cache_lineage"]["cache_sha256"],
        )
        self.assertNotEqual(
            steps[0]["cache_lineage"]["layers"][0]["k_sha256"],
            steps[1]["cache_lineage"]["layers"][0]["k_sha256"],
        )

        broken = copy.deepcopy(self.document)
        broken["generation"]["steps"][1]["cache_lineage"][
            "parent_cache_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "aggregate cache parentage",
        ):
            validate_document(broken)

    def test_argmax_and_logit_mutations_are_rejected(self) -> None:
        excessive_hidden_error = copy.deepcopy(self.document)
        excessive_hidden_error["generation"]["steps"][0]["terminal_hidden"][
            "independent_reference"
        ]["max_abs_error"] = 1.0
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "terminal hidden comparison",
        ):
            validate_document(excessive_hidden_error)

        false_argmax = copy.deepcopy(self.document)
        false_argmax["generation"]["steps"][1]["token"][
            "argmax_matches_independent_reference"
        ] = False
        with self.assertRaisesRegex(DialogueExecutionError, "argmax comparison"):
            validate_document(false_argmax)

        excessive_error = copy.deepcopy(self.document)
        excessive_error["generation"]["steps"][0]["logits"][
            "independent_reference"
        ]["max_abs_error"] = 1.0
        with self.assertRaisesRegex(DialogueExecutionError, "logits comparison"):
            validate_document(excessive_error)

    def test_official_eos_and_maximum_stop_rules(self) -> None:
        self.assertIsNone(generation_stop_reason(9707, 1, 3))
        self.assertEqual(
            generation_stop_reason(EOS_TOKEN_ID, 2, 3),
            "eos_token",
        )
        self.assertEqual(
            generation_stop_reason(1879, 3, 3),
            "max_new_tokens",
        )

        continued_after_eos = copy.deepcopy(self.document)
        continued_after_eos["generation"]["steps"][1]["token"][
            "argmax_token_id"
        ] = EOS_TOKEN_ID
        continued_after_eos["generation"]["steps"][1]["token"][
            "independent_reference_argmax_token_id"
        ] = EOS_TOKEN_ID
        continued_after_eos["generation"]["generated_token_ids"][1] = EOS_TOKEN_ID
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "continued after EOS",
        ):
            validate_document(continued_after_eos)


if __name__ == "__main__":
    unittest.main()
