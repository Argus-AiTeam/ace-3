#!/usr/bin/env python3
"""Focused default, explicit-prompt, and stale-binding host/runtime tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPOSITORY_ROOT / "ace3" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model24_execution_oracle import (  # noqa: E402
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    FIXED_CHAT_TOKEN_IDS,
    authenticate_tokenizer,
)
from model24_host_runtime import (  # noqa: E402
    ARTIFACT_NAME,
    FOCUSED_EXPLICIT_PROMPT,
    HostRuntimeError,
    SELECTED_TOKEN_AUTHORITY_KIND,
    SELECTED_TOKEN_CHAIN_KIND,
    SELECTED_TOKEN_POLICY,
    SELECTED_TOKEN_RECEIPT_KIND,
    _model_binding,
    form_dialogue_from_receipt_chain,
    form_next_dialogue_step_from_receipt,
    validate_directory,
    validate_document,
    validate_selected_token_receipt,
)
from fp16_adaptation_oracle import decode_f16_q24  # noqa: E402
from model24_oracle import (  # noqa: E402
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
)
from official_model24_dialogue import (  # noqa: E402
    DialogueExecutionError,
    create_binding_lineage,
    official_tokenizer_binding,
    validate_binding_lineage,
)


RECEIPT_TERMINAL_EVIDENCE = {
    "kind": "ace3_streaming_tied_lm_head_terminal",
    "sha256": "a" * 64,
}
RECEIPT_AUTHORITY_LINEAGE = {
    "authority_id": "accepted-receipt-bridge-fixture",
    "scope": "selected-token-dialogue-receipt-read-only",
}


class _FixtureTokenizer:
    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        if skip_special_tokens:
            raise AssertionError("receipt bridge changed tokenizer semantics")
        return "".join(
            {271: " moon", 2: "#", 3: "$"}.get(token_id, f"<{token_id}>")
            for token_id in token_ids
        )


def _receipt_prompt() -> dict:
    return {
        "source": "default",
        "serialization": FIXED_CHAT_SERIALIZATION,
        "serialization_utf8_sha256": hashlib.sha256(
            FIXED_CHAT_SERIALIZATION.encode("utf-8")
        ).hexdigest(),
        "token_ids": list(FIXED_CHAT_TOKEN_IDS),
    }


def _top_k_fixture(
    token_ids: list[int] | None = None,
    logit_bits: list[int] | None = None,
) -> list[dict]:
    tokens = token_ids or [271, 9707, 11, 311, 498, 2, 3, 4, 5, 6]
    bits = logit_bits or [
        0x4C0F,
        0x4B00,
        0x4A00,
        0x4900,
        0x4800,
        0x4700,
        0x4600,
        0x4500,
        0x4400,
        0x4300,
    ]
    return [
        {
            "rank": rank,
            "token_id": token_id,
            "logit_f16_bits": raw,
            "logit_q24": decode_f16_q24(raw)[0],
        }
        for rank, (token_id, raw) in enumerate(zip(tokens, bits, strict=True))
    ]


def _selected_token_receipt(
    *,
    top_k: list[dict] | None = None,
) -> dict:
    winners = top_k or _top_k_fixture()
    return {
        "schema_version": 1,
        "kind": SELECTED_TOKEN_RECEIPT_KIND,
        "model_binding": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "revision": "db09cd27ead7fee40cdee309693cf83601b9c899",
            "checkpoint": {
                "filename": "model.safetensors",
                "sha256": CHECKPOINT_SHA256,
                "bytes": CHECKPOINT_SIZE,
            },
        },
        "tokenizer_binding": official_tokenizer_binding(),
        "prompt_lineage": {
            key: _receipt_prompt()[key]
            for key in (
                "source",
                "serialization_utf8_sha256",
                "token_ids",
            )
        },
        "terminal_evidence": copy.deepcopy(RECEIPT_TERMINAL_EVIDENCE),
        "authority": {
            "kind": SELECTED_TOKEN_AUTHORITY_KIND,
            "lineage": copy.deepcopy(RECEIPT_AUTHORITY_LINEAGE),
            "receipt_use_authorized": True,
            "authority_consumed": False,
        },
        "selection": {
            "generation_ordinal": 0,
            "vocab_size": 151936,
            "selection_policy": SELECTED_TOKEN_POLICY,
            "selected_token_id": winners[0]["token_id"],
            "selected_logit_f16_bits": winners[0]["logit_f16_bits"],
            "top_k": winners,
        },
    }


def _receipt_chain(
    length: int = 3,
) -> tuple[list[dict], list[dict], list[dict]]:
    selected_ids = [271, 2, 3, 4]
    ranked_ids = [271, 9707, 11, 311, 498, 2, 3, 4, 5, 6, 7, 8]
    terminal_evidence_chain = [
        {
            "kind": "ace3_streaming_tied_lm_head_terminal",
            "sha256": f"{position + 1:064x}",
        }
        for position in range(length + 1)
    ]
    authority_lineages = [
        {
            "authority_id": f"accepted-receipt-bridge-fixture-{position}",
            "scope": "selected-token-dialogue-receipt-read-only",
        }
        for position in range(length)
    ]
    receipts = []
    token_history = list(FIXED_CHAT_TOKEN_IDS)
    for position, selected_id in enumerate(selected_ids[:length]):
        token_ids = [
            selected_id,
            *(
                token_id
                for token_id in ranked_ids
                if token_id != selected_id
            ),
        ][:10]
        receipt = _selected_token_receipt(
            top_k=_top_k_fixture(token_ids=token_ids)
        )
        receipt["selection"]["generation_ordinal"] = position
        receipt["input_token_history"] = list(token_history)
        receipt["parent_terminal_evidence"] = copy.deepcopy(
            terminal_evidence_chain[position]
        )
        receipt["terminal_evidence"] = copy.deepcopy(
            terminal_evidence_chain[position + 1]
        )
        receipt["authority"]["lineage"] = copy.deepcopy(
            authority_lineages[position]
        )
        receipts.append(receipt)
        token_history.append(selected_id)
    return receipts, terminal_evidence_chain, authority_lineages


def _binding_only_document() -> dict:
    prompt = {
        "source": "default",
        "input_text": FIXED_CHAT_MESSAGES[-1][1],
        "messages": [
            {"role": role, "content": content}
            for role, content in FIXED_CHAT_MESSAGES
        ],
        "serialization": FIXED_CHAT_SERIALIZATION,
        "serialization_utf8_sha256": hashlib.sha256(
            FIXED_CHAT_SERIALIZATION.encode("utf-8")
        ).hexdigest(),
        "token_ids": list(FIXED_CHAT_TOKEN_IDS),
        "decoded_roundtrip": FIXED_CHAT_SERIALIZATION,
    }
    document = {
        "model_binding": _model_binding(Path("model.safetensors")),
        "tokenizer_binding": official_tokenizer_binding(),
        "prompt": prompt,
        "generation": {
            "generated_token_ids": [9707, 11, 311, 498],
            "decoded_token_ids": [9707, 11, 311, 498],
            "decoded_text": "Hello, to you",
            "stop_reason": "max_new_tokens",
        },
    }
    document["binding_lineage"] = create_binding_lineage(document)
    return document


class SelectedTokenDialogueReceiptBridgeTests(unittest.TestCase):
    def test_accepted_shape_receipt_forms_next_step_without_execution(
        self,
    ) -> None:
        receipt = _selected_token_receipt()
        original = copy.deepcopy(receipt)
        with (
            patch(
                "model24_host_runtime.execute_host_runtime",
                side_effect=AssertionError("runtime workload invoked"),
            ) as runtime,
            patch(
                "model24_host_runtime._load_model",
                side_effect=AssertionError("model workload invoked"),
            ) as model,
            patch(
                "model24_host_runtime.execute_loaded_prompt",
                side_effect=AssertionError("model workload invoked"),
            ) as executor,
            patch(
                "controller_model24_rtl_cascade.execute",
                side_effect=AssertionError("controller or RTL invoked"),
            ) as controller,
            patch(
                "subprocess.run",
                side_effect=AssertionError("durable submission invoked"),
            ) as submission,
        ):
            step = form_next_dialogue_step_from_receipt(
                receipt,
                _FixtureTokenizer(),
                _receipt_prompt(),
                RECEIPT_TERMINAL_EVIDENCE,
                RECEIPT_AUTHORITY_LINEAGE,
            )

        for call in (runtime, model, executor, controller, submission):
            call.assert_not_called()
        self.assertEqual(receipt, original)
        self.assertEqual(step["selected_token"]["token_id"], 271)
        self.assertEqual(step["selected_token"]["logit_f16_bits"], 0x4C0F)
        self.assertEqual(step["selected_token"]["decoded_token"], " moon")
        self.assertEqual(
            step["next_model_input_token_ids"],
            [*FIXED_CHAT_TOKEN_IDS, 271],
        )
        self.assertEqual(
            step["effect_boundary"],
            {
                "runtime_workload_invoked": False,
                "controller_invoked": False,
                "model_or_rtl_invoked": False,
                "durable_submission_created": False,
                "authority_created": False,
                "authority_consumed": False,
            },
        )

    def test_synthetic_equal_logit_tie_uses_ascending_token_id(self) -> None:
        bits = [0x3C00, 0x3C00, 0x3800, 0x3400, 0x3000]
        bits.extend([0x2C00, 0x2800, 0x2400, 0x2000, 0x1C00])
        receipt = _selected_token_receipt(
            top_k=_top_k_fixture(
                token_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                logit_bits=bits,
            )
        )

        summary = validate_selected_token_receipt(
            receipt,
            _receipt_prompt(),
            RECEIPT_TERMINAL_EVIDENCE,
            RECEIPT_AUTHORITY_LINEAGE,
        )

        self.assertEqual(summary["selected_token_id"], 2)

    def test_stale_lineage_is_rejected(self) -> None:
        mutations = (
            (
                "checkpoint",
                lambda receipt: receipt["model_binding"]["checkpoint"].__setitem__(
                    "sha256", "0" * 64
                ),
                "checkpoint lineage",
            ),
            (
                "tokenizer",
                lambda receipt: receipt["tokenizer_binding"].__setitem__(
                    "tokenizer_sha256", "0" * 64
                ),
                "tokenizer lineage",
            ),
            (
                "prompt",
                lambda receipt: receipt["prompt_lineage"]["token_ids"].append(0),
                "prompt lineage",
            ),
            (
                "terminal",
                lambda receipt: receipt["terminal_evidence"].__setitem__(
                    "sha256", "0" * 64
                ),
                "terminal evidence lineage",
            ),
            (
                "authority",
                lambda receipt: receipt["authority"]["lineage"].__setitem__(
                    "authority_id", "stale-authority"
                ),
                "authority lineage",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipt = _selected_token_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    validate_selected_token_receipt(
                        receipt,
                        _receipt_prompt(),
                        RECEIPT_TERMINAL_EVIDENCE,
                        RECEIPT_AUTHORITY_LINEAGE,
                    )

    def test_token_logit_and_top_k_mismatches_are_rejected(self) -> None:
        mutations = (
            (
                "token",
                lambda receipt: receipt["selection"].__setitem__(
                    "selected_token_id", 0
                ),
                "token ID does not match",
            ),
            (
                "logit",
                lambda receipt: receipt["selection"].__setitem__(
                    "selected_logit_f16_bits", 0x3C00
                ),
                "selected logit does not match",
            ),
            (
                "top_k",
                lambda receipt: receipt["selection"]["top_k"][1].__setitem__(
                    "logit_q24", 0
                ),
                "top-k payload",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipt = _selected_token_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    validate_selected_token_receipt(
                        receipt,
                        _receipt_prompt(),
                        RECEIPT_TERMINAL_EVIDENCE,
                        RECEIPT_AUTHORITY_LINEAGE,
                    )

    def test_unauthorized_or_consumed_authority_is_rejected(self) -> None:
        mutations = (
            (
                "unauthorized",
                lambda receipt: receipt["authority"].__setitem__(
                    "receipt_use_authorized", False
                ),
                "not authorized",
            ),
            (
                "consumed",
                lambda receipt: receipt["authority"].__setitem__(
                    "authority_consumed", True
                ),
                "already consumed",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipt = _selected_token_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    form_next_dialogue_step_from_receipt(
                        receipt,
                        _FixtureTokenizer(),
                        _receipt_prompt(),
                        RECEIPT_TERMINAL_EVIDENCE,
                        RECEIPT_AUTHORITY_LINEAGE,
                    )


class SelectedTokenDialogueReceiptChainTests(unittest.TestCase):
    def _form_chain(
        self,
        receipts: list[dict],
        terminal_evidence_chain: list[dict],
        authority_lineages: list[dict],
    ) -> dict:
        return form_dialogue_from_receipt_chain(
            receipts,
            _FixtureTokenizer(),
            _receipt_prompt(),
            terminal_evidence_chain,
            authority_lineages,
        )

    def test_valid_chain_assembles_history_and_transcript_without_execution(
        self,
    ) -> None:
        receipts, terminal_chain, authorities = _receipt_chain()
        originals = copy.deepcopy(
            (receipts, terminal_chain, authorities)
        )
        with (
            patch(
                "model24_host_runtime.execute_host_runtime",
                side_effect=AssertionError("runtime workload invoked"),
            ) as runtime,
            patch(
                "model24_host_runtime._load_model",
                side_effect=AssertionError("model workload invoked"),
            ) as model,
            patch(
                "model24_host_runtime.execute_loaded_prompt",
                side_effect=AssertionError("model or oracle invoked"),
            ) as executor,
            patch(
                "controller_model24_rtl_cascade.execute",
                side_effect=AssertionError("controller or RTL invoked"),
            ) as controller,
            patch(
                "model24_r15_lifecycle.submit",
                side_effect=AssertionError("lifecycle invoked"),
            ) as lifecycle,
            patch(
                "subprocess.run",
                side_effect=AssertionError("durable submission invoked"),
            ) as submission,
        ):
            dialogue = self._form_chain(
                receipts,
                terminal_chain,
                authorities,
            )

        for call in (
            runtime,
            model,
            executor,
            controller,
            lifecycle,
            submission,
        ):
            call.assert_not_called()
        self.assertEqual(
            (receipts, terminal_chain, authorities),
            originals,
        )
        self.assertEqual(dialogue["kind"], SELECTED_TOKEN_CHAIN_KIND)
        self.assertEqual(dialogue["receipt_count"], 3)
        self.assertEqual(dialogue["generated_token_ids"], [271, 2, 3])
        self.assertEqual(
            dialogue["resulting_token_history"],
            [*FIXED_CHAT_TOKEN_IDS, 271, 2, 3],
        )
        self.assertEqual(
            dialogue["decoded_dialogue_transcript"],
            _FixtureTokenizer().decode(
                [*FIXED_CHAT_TOKEN_IDS, 271, 2, 3],
                skip_special_tokens=False,
            ),
        )
        self.assertEqual(
            dialogue["effect_boundary"],
            {
                "runtime_workload_invoked": False,
                "lifecycle_invoked": False,
                "controller_invoked": False,
                "model_oracle_or_rtl_invoked": False,
                "durable_submission_created": False,
                "authority_created": False,
                "authority_consumed": False,
            },
        )

    def test_chain_requires_two_or_more_receipts(self) -> None:
        receipts, terminal_chain, authorities = _receipt_chain(length=1)
        with self.assertRaisesRegex(
            HostRuntimeError,
            "at least two receipts",
        ):
            self._form_chain(receipts, terminal_chain, authorities)

    def test_stale_checkpoint_tokenizer_and_prompt_lineage_are_rejected(
        self,
    ) -> None:
        mutations = (
            (
                "checkpoint",
                lambda receipts: receipts[1]["model_binding"][
                    "checkpoint"
                ].__setitem__("sha256", "0" * 64),
                "checkpoint lineage",
            ),
            (
                "tokenizer",
                lambda receipts: receipts[1][
                    "tokenizer_binding"
                ].__setitem__("tokenizer_sha256", "0" * 64),
                "tokenizer lineage",
            ),
            (
                "prompt",
                lambda receipts: receipts[1][
                    "prompt_lineage"
                ]["token_ids"].append(0),
                "prompt lineage",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipts, terminal_chain, authorities = _receipt_chain()
                mutate(receipts)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    self._form_chain(
                        receipts,
                        terminal_chain,
                        authorities,
                    )

    def test_out_of_order_duplicate_and_gapped_positions_are_rejected(
        self,
    ) -> None:
        mutations = (
            ("out_of_order", lambda receipts: receipts.reverse()),
            (
                "duplicate",
                lambda receipts: receipts[1]["selection"].__setitem__(
                    "generation_ordinal", 0
                ),
            ),
            (
                "gapped",
                lambda receipts: receipts[1]["selection"].__setitem__(
                    "generation_ordinal", 2
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                receipts, terminal_chain, authorities = _receipt_chain()
                mutate(receipts)
                with self.assertRaisesRegex(
                    HostRuntimeError,
                    "positions are not monotone and ungapped",
                ):
                    self._form_chain(
                        receipts,
                        terminal_chain,
                        authorities,
                    )

    def test_token_history_discontinuity_is_rejected(self) -> None:
        receipts, terminal_chain, authorities = _receipt_chain()
        receipts[1]["input_token_history"][-1] += 1
        with self.assertRaisesRegex(
            HostRuntimeError,
            "token history discontinuity",
        ):
            self._form_chain(receipts, terminal_chain, authorities)

    def test_terminal_evidence_parent_mismatch_is_rejected(self) -> None:
        receipts, terminal_chain, authorities = _receipt_chain()
        receipts[1]["parent_terminal_evidence"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            HostRuntimeError,
            "terminal evidence parent mismatch",
        ):
            self._form_chain(receipts, terminal_chain, authorities)

    def test_token_logit_and_top_k_mutations_are_rejected(self) -> None:
        def make_nonfinite(receipts: list[dict]) -> None:
            receipts[1]["selection"]["top_k"][1][
                "logit_f16_bits"
            ] = 0x7C00

        def break_order(receipts: list[dict]) -> None:
            top_k = receipts[1]["selection"]["top_k"]
            top_k[1], top_k[2] = top_k[2], top_k[1]
            top_k[1]["rank"] = 1
            top_k[2]["rank"] = 2

        mutations = (
            (
                "token",
                lambda receipts: receipts[1]["selection"].__setitem__(
                    "selected_token_id", 0
                ),
                "token ID does not match",
            ),
            (
                "logit",
                lambda receipts: receipts[1]["selection"].__setitem__(
                    "selected_logit_f16_bits", 0x3C00
                ),
                "selected logit does not match",
            ),
            (
                "q24",
                lambda receipts: receipts[1]["selection"]["top_k"][
                    1
                ].__setitem__("logit_q24", 0),
                "top-k payload",
            ),
            ("nonfinite", make_nonfinite, "top-k payload"),
            ("ordering", break_order, "top-k ordering"),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipts, terminal_chain, authorities = _receipt_chain()
                mutate(receipts)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    self._form_chain(
                        receipts,
                        terminal_chain,
                        authorities,
                    )

    def test_each_authority_must_be_authorized_and_unconsumed(self) -> None:
        mutations = (
            (
                "unauthorized",
                lambda receipts: receipts[1]["authority"].__setitem__(
                    "receipt_use_authorized", False
                ),
                "not authorized",
            ),
            (
                "consumed",
                lambda receipts: receipts[1]["authority"].__setitem__(
                    "authority_consumed", True
                ),
                "already consumed",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                receipts, terminal_chain, authorities = _receipt_chain()
                mutate(receipts)
                with self.assertRaisesRegex(HostRuntimeError, message):
                    self._form_chain(
                        receipts,
                        terminal_chain,
                        authorities,
                    )


class Model24HostBindingValidationTests(unittest.TestCase):
    def test_valid_official_binding_is_accepted_without_runtime(self) -> None:
        document = _binding_only_document()
        with patch(
            "official_model24_dialogue._load_model",
            side_effect=AssertionError("runtime workload invoked"),
        ), patch(
            "official_model24_dialogue.execute_loaded_prompt",
            side_effect=AssertionError("runtime workload invoked"),
        ):
            lineage = validate_binding_lineage(document)
        self.assertEqual(lineage, document["binding_lineage"])

    def test_stale_asset_and_record_bindings_are_rejected(self) -> None:
        mutations = (
            (
                "model_revision",
                lambda document: document["model_binding"].__setitem__(
                    "revision", "stale-revision"
                ),
                "checkpoint lineage",
            ),
            (
                "checkpoint",
                lambda document: document["model_binding"]["checkpoint"].__setitem__(
                    "sha256", "0" * 64
                ),
                "checkpoint lineage",
            ),
            (
                "tokenizer_revision",
                lambda document: document["tokenizer_binding"].__setitem__(
                    "revision", "stale-revision"
                ),
                "tokenizer lineage",
            ),
            (
                "tokenizer",
                lambda document: document["tokenizer_binding"].__setitem__(
                    "tokenizer_sha256", "0" * 64
                ),
                "tokenizer lineage",
            ),
            (
                "prompt",
                lambda document: document["prompt"].__setitem__(
                    "serialization", "stale prompt"
                ),
                "record lineage hash",
            ),
            (
                "generated_tokens",
                lambda document: document["generation"][
                    "generated_token_ids"
                ].__setitem__(0, 0),
                "record lineage hash",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                document = _binding_only_document()
                mutate(document)
                with self.assertRaisesRegex(DialogueExecutionError, message):
                    validate_binding_lineage(document)


class Model24HostRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer_dir = (
            Path(
                os.environ.get(
                    "ACE3_OFFICIAL_TOKENIZER_DIR",
                    REPOSITORY_ROOT
                    / "build"
                    / "model24_host_runtime"
                    / "tokenizer",
                )
            )
        )
        cls.checkpoint = Path(
            os.environ.get(
                "ACE3_OFFICIAL_CHECKPOINT",
                REPOSITORY_ROOT
                / "model24_execution_vectors"
                / "model.safetensors",
            )
        )
        cls.tokenizer = authenticate_tokenizer(cls.tokenizer_dir)
        cls.evidence_root = Path(
            os.environ.get(
                "ACE3_MODEL24_HOST_RUNTIME_DIR",
                REPOSITORY_ROOT / "build" / "model24_host_runtime",
            )
        )
        cls.default_dir = cls.evidence_root / "default"
        cls.explicit_dir = cls.evidence_root / "explicit"
        cls.default_document = json.loads(
            (cls.default_dir / ARTIFACT_NAME).read_text(encoding="ascii")
        )
        cls.explicit_document = json.loads(
            (cls.explicit_dir / ARTIFACT_NAME).read_text(encoding="ascii")
        )

    def test_default_prompt_produces_authenticated_multitoken_output(self) -> None:
        summary = validate_document(
            self.default_document,
            self.tokenizer,
            self.tokenizer_dir,
            None,
        )
        self.assertEqual(summary["prompt_source"], "default")
        self.assertGreaterEqual(summary["generated_tokens"], 2)
        self.assertTrue(summary["decoded_text"].strip())

    def test_explicit_prompt_produces_authenticated_multitoken_output(self) -> None:
        summary = validate_document(
            self.explicit_document,
            self.tokenizer,
            self.tokenizer_dir,
            FOCUSED_EXPLICIT_PROMPT,
        )
        self.assertEqual(summary["prompt_source"], "caller_provided")
        self.assertGreaterEqual(summary["generated_tokens"], 2)
        self.assertTrue(summary["decoded_text"].strip())

    def test_stale_checkpoint_and_tokenizer_bindings_are_rejected(self) -> None:
        stale_checkpoint = copy.deepcopy(self.default_document)
        stale_checkpoint["model_binding"]["checkpoint"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(HostRuntimeError, "checkpoint binding"):
            validate_document(
                stale_checkpoint,
                self.tokenizer,
                self.tokenizer_dir,
                None,
            )

        stale_tokenizer = copy.deepcopy(self.default_document)
        stale_tokenizer["tokenizer_binding"]["tokenizer_sha256"] = "0" * 64
        with self.assertRaisesRegex(HostRuntimeError, "tokenizer binding"):
            validate_document(
                stale_tokenizer,
                self.tokenizer,
                self.tokenizer_dir,
                None,
            )

    def test_prompt_token_mismatch_is_rejected(self) -> None:
        mismatched = copy.deepcopy(self.explicit_document)
        mismatched["prompt"]["token_ids"][0] += 1
        with self.assertRaisesRegex(HostRuntimeError, "prompt token_ids mismatch"):
            validate_document(
                mismatched,
                self.tokenizer,
                self.tokenizer_dir,
                FOCUSED_EXPLICIT_PROMPT,
            )

    def test_out_of_tolerance_oracle_comparison_is_rejected(self) -> None:
        mismatched = copy.deepcopy(self.default_document)
        comparison = mismatched["generation"]["steps"][0]["logits"][
            "independent_reference"
        ]
        comparison["max_abs_error"] = comparison["absolute_tolerance"] + 1.0
        comparison["within_tolerance"] = False
        with self.assertRaisesRegex(DialogueExecutionError, "logits comparison"):
            validate_document(
                mismatched,
                self.tokenizer,
                self.tokenizer_dir,
                None,
            )

    def test_cache_and_top_k_substitutions_are_rejected(self) -> None:
        broken_cache = copy.deepcopy(self.default_document)
        broken_cache["generation"]["steps"][1]["cache_lineage"][
            "parent_cache_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            DialogueExecutionError,
            "aggregate cache parentage",
        ):
            validate_document(
                broken_cache,
                self.tokenizer,
                self.tokenizer_dir,
                None,
            )

        broken_top_k = copy.deepcopy(self.default_document)
        broken_top_k["generation"]["steps"][0]["token"]["top_k"][0][
            "token_id"
        ] += 1
        with self.assertRaisesRegex(DialogueExecutionError, "top-k evidence"):
            validate_document(
                broken_top_k,
                self.tokenizer,
                self.tokenizer_dir,
                None,
            )

    def test_validation_reauthenticates_checkpoint_and_tokenizer(self) -> None:
        with patch(
            "model24_host_runtime.authenticate_checkpoint",
            side_effect=RuntimeError("stale checkpoint"),
        ):
            with self.assertRaisesRegex(
                HostRuntimeError,
                "official checkpoint authentication failed",
            ):
                validate_directory(
                    self.default_dir,
                    self.checkpoint,
                    self.tokenizer_dir,
                )

        with patch(
            "model24_host_runtime.authenticate_tokenizer",
            side_effect=RuntimeError("stale tokenizer"),
        ):
            with self.assertRaisesRegex(
                HostRuntimeError,
                "official tokenizer authentication failed",
            ):
                validate_directory(
                    self.default_dir,
                    self.checkpoint,
                    self.tokenizer_dir,
                )


if __name__ == "__main__":
    unittest.main()
