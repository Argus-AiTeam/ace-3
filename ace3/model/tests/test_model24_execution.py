#!/usr/bin/env python3
"""Exhaustive reduced-geometry and mutation tests for model24 execution."""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODEL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(MODEL_DIR))

from generate_model24_execution_vectors import generate  # noqa: E402
from model24_execution_oracle import (  # noqa: E402
    ContractError,
    CHECKPOINT_SHA256,
    DEFAULT_OFFICIAL_TOKENIZER_DIR,
    DECODER_INTERFACE_SHA256,
    DECODER_SOURCE_SHA256,
    EOS_TOKEN_ID,
    ExecutionMachine,
    FIXED_CHAT_MESSAGES,
    FIXED_CHAT_SERIALIZATION,
    FIXED_CHAT_TOKEN_IDS,
    Geometry,
    FIXED_TERMINAL_HIDDEN_SHA256,
    FINAL_NORM_SHA256,
    LAYER_DESCRIPTOR_SHA256,
    OFFICIAL_GEOMETRY,
    PER_LAYER_OPERATIONS,
    REDUCED_RESPONSE_TOKEN_IDS,
    TIED_WEIGHT_SHA256,
    StructuralModel24Host,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_SHA256,
    authenticate_tokenizer,
    argmax_first,
    build_vector_artifacts,
    expected_schedule,
    grouped_lm_head_interface,
    host_generation_document,
    indexed_layer_input_handoff_binding,
    kv_address,
    kv_owner,
    load_json_bytes,
    residual_handoffs,
    reduced_token_label,
    require_provenance_commit,
    sha256_bytes,
    serialize_chat_prompt,
    stable_top_k_f16,
    validate_execution_contract,
    validate_decoder_snapshot,
    validate_trajectory,
    validate_vector_bindings,
)
from validate_model24_execution_vectors import validate_vector_directory  # noqa: E402


class Model24ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        contracts = REPOSITORY_ROOT / "ace3" / "contracts"
        cls.contract_payload = (contracts / "model24_execution.json").read_bytes()
        cls.bindings_payload = (
            contracts / "model24_execution_vector_bindings.json"
        ).read_bytes()
        cls.tensor_payload = (contracts / "model24_tensor_map.json").read_bytes()
        cls.control_payload = (contracts / "model24_control.json").read_bytes()
        cls.contract = load_json_bytes(cls.contract_payload, "execution contract")
        cls.bindings = load_json_bytes(cls.bindings_payload, "vector bindings")

    def test_official_contract_binds_all_reviewed_surfaces(self) -> None:
        validate_execution_contract(
            self.contract,
            self.tensor_payload,
            self.control_payload,
        )
        validate_vector_bindings(
            self.bindings,
            sha256_bytes(self.contract_payload),
        )
        self.assertEqual(self.contract["parent_commit"], "3cf65b762d928e02e2b64fbba4389e294e1aa2c5")
        self.assertEqual(len(self.contract["layers"]["bindings"]), 24)
        checkpoint = self.contract["model_binding"]["checkpoint"]
        self.assertEqual(checkpoint["sha256"], CHECKPOINT_SHA256)
        self.assertEqual(checkpoint["bytes"], 730652248)
        self.assertEqual(
            [
                binding["namespace"]
                for binding in self.contract["layers"]["bindings"]
            ],
            [f"model.layers.{layer_id}." for layer_id in range(24)],
        )
        self.assertEqual(
            tuple(
                binding["descriptor_sha256"]
                for binding in self.contract["layers"]["bindings"]
            ),
            LAYER_DESCRIPTOR_SHA256,
        )
        snapshot = self.contract["decoder_snapshot_compatibility"]
        self.assertIn("not 24-layer execution", snapshot["status"])
        self.assertEqual(snapshot["source"]["sha256"], DECODER_SOURCE_SHA256)
        self.assertEqual(snapshot["interface"]["sha256"], DECODER_INTERFACE_SHA256)
        layer0 = self.contract["official_single_decoder_layer"]
        self.assertEqual(layer0["prompt"]["token_ids"], [9707, 1879])
        self.assertEqual(layer0["consumed_layer_tensor_count"], 26)
        self.assertEqual(
            layer0["independent_reference"]["sampled_projection_bit_oracle_outputs"],
            42,
        )
        validate_decoder_snapshot(REPOSITORY_ROOT)

    def test_decoder_snapshot_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "ace3" / "rtl" / "ace3_decoder_layer0_token_engine.sv"
            interface = root / "ace3" / "tb" / "ace3_decoder_width_boundary_tb.sv"
            source.parent.mkdir(parents=True)
            interface.parent.mkdir(parents=True)
            source.write_bytes(
                (
                    REPOSITORY_ROOT
                    / "ace3"
                    / "rtl"
                    / "ace3_decoder_layer0_token_engine.sv"
                ).read_bytes()
            )
            interface.write_bytes(
                (
                    REPOSITORY_ROOT
                    / "ace3"
                    / "tb"
                    / "ace3_decoder_width_boundary_tb.sv"
                ).read_bytes()
            )
            validate_decoder_snapshot(root)
            interface.write_bytes(interface.read_bytes() + b"\n")
            with self.assertRaisesRegex(ContractError, "SHA256 mismatch"):
                validate_decoder_snapshot(root)

    def test_provenance_accepts_additive_commits_and_fails_closed(
        self,
    ) -> None:
        def git(repository: Path, *arguments: str) -> str:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Model24 Test",
                    "-c",
                    "user.email=model24-test@example.invalid",
                    *arguments,
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "published"
            repository.mkdir()
            git(repository, "init", "--quiet")
            (repository / "contract").write_text("pinned\n", encoding="ascii")
            git(repository, "add", "contract")
            git(repository, "commit", "--quiet", "-m", "pinned provenance")
            provenance_commit = git(repository, "rev-parse", "HEAD")

            for commit_number in (1, 2):
                additive_path = repository / f"additive-{commit_number}"
                additive_path.write_text(f"{commit_number}\n", encoding="ascii")
                git(repository, "add", additive_path.name)
                git(
                    repository,
                    "commit",
                    "--quiet",
                    "-m",
                    f"additive commit {commit_number}",
                )
                require_provenance_commit(repository, provenance_commit)
            git(repository, "checkout", "--quiet", "--orphan", "unrelated")
            (repository / "other").write_text("unrelated\n", encoding="ascii")
            git(repository, "add", "other")
            git(repository, "commit", "--quiet", "-m", "unrelated history")
            with self.assertRaisesRegex(
                ContractError,
                "required provenance commit .* is not an ancestor of HEAD",
            ):
                require_provenance_commit(repository, provenance_commit)

            non_repository = root / "not-a-repository"
            non_repository.mkdir()
            with self.assertRaisesRegex(
                ContractError,
                "unable to verify required provenance commit",
            ):
                require_provenance_commit(non_repository, provenance_commit)

            with mock.patch(
                "model24_execution_oracle.subprocess.run",
                side_effect=FileNotFoundError("git"),
            ) as run_git:
                with self.assertRaisesRegex(ContractError, "unable to execute git"):
                    require_provenance_commit(repository, provenance_commit)
                run_git.assert_called_once_with(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        provenance_commit,
                        "HEAD",
                    ],
                    cwd=repository,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                )

    def test_official_schedule_and_lineage_are_complete(self) -> None:
        events = expected_schedule()
        self.assertEqual(len(events), 483)
        self.assertEqual(events[0]["operation"], "embedding_lookup")
        self.assertEqual(
            [event["operation"] for event in events[-2:]],
            ["final_rmsnorm", "lm_head"],
        )
        for layer_id in range(24):
            block = [
                event
                for event in events
                if event["layer_id"] == layer_id
            ]
            self.assertEqual(
                [event["operation"] for event in block],
                list(PER_LAYER_OPERATIONS),
            )
            self.assertEqual(
                {event["namespace"] for event in block},
                {f"model.layers.{layer_id}."},
            )
        handoffs = residual_handoffs(OFFICIAL_GEOMETRY)
        self.assertEqual(handoffs[0]["input"], "embedding.output")
        for layer_id in range(1, 24):
            self.assertEqual(
                handoffs[layer_id]["input"],
                handoffs[layer_id - 1]["output"],
            )
        validate_trajectory(events)

    def test_layer2_input_handoff_names_official_layer1_dependency(self) -> None:
        self.assertEqual(
            indexed_layer_input_handoff_binding(
                2,
                {
                    "sha256": "a" * 64,
                    "rows": 1792,
                    "shape": [2, 896],
                    "dtype": "F16",
                },
            ),
            {
                "sha256": "a" * 64,
                "rows": 1792,
                "shape": [2, 896],
                "dtype": "F16",
                "source": "authenticated decoder layer 1 raw final rows",
                "source_layer_index": 1,
                "consumer_layer_index": 2,
                "byte_preserved_as": "inputs.hex",
            },
        )

    def test_tokenizer_binding_prompt_serialization_and_decode(self) -> None:
        tokenizer = authenticate_tokenizer(DEFAULT_OFFICIAL_TOKENIZER_DIR)
        messages = [
            {"role": role, "content": content}
            for role, content in FIXED_CHAT_MESSAGES
        ]
        serialized = serialize_chat_prompt(messages)
        self.assertEqual(serialized, FIXED_CHAT_SERIALIZATION)
        encoded = tokenizer.encode(serialized, add_special_tokens=False).ids
        self.assertEqual(encoded, list(FIXED_CHAT_TOKEN_IDS))
        self.assertEqual(
            tokenizer.decode(encoded, skip_special_tokens=False),
            serialized,
        )
        self.assertEqual(
            tokenizer.decode(
                list(REDUCED_RESPONSE_TOKEN_IDS),
                skip_special_tokens=False,
            ),
            "Hello world",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            tokenizer_dir = Path(temporary_directory)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (tokenizer_dir / name).write_bytes(
                    (DEFAULT_OFFICIAL_TOKENIZER_DIR / name).read_bytes()
                )
            (tokenizer_dir / "tokenizer.json").write_bytes(
                (tokenizer_dir / "tokenizer.json").read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(ContractError, "tokenizer.json SHA256"):
                authenticate_tokenizer(tokenizer_dir)
        self.assertEqual(
            sha256_bytes(
                (DEFAULT_OFFICIAL_TOKENIZER_DIR / "tokenizer.json").read_bytes()
            ),
            TOKENIZER_SHA256,
        )
        self.assertEqual(
            sha256_bytes(
                (
                    DEFAULT_OFFICIAL_TOKENIZER_DIR / "tokenizer_config.json"
                ).read_bytes()
            ),
            TOKENIZER_CONFIG_SHA256,
        )

    def test_host_steps_prompt_and_generation_with_stop_and_slot_reuse(self) -> None:
        document = host_generation_document(DEFAULT_OFFICIAL_TOKENIZER_DIR)
        self.assertEqual(document["prompt"]["serialization"], FIXED_CHAT_SERIALIZATION)
        self.assertEqual(document["prompt"]["decoded_roundtrip"], FIXED_CHAT_SERIALIZATION)
        self.assertEqual(
            document["output"]["generated_token_ids"],
            [*REDUCED_RESPONSE_TOKEN_IDS, EOS_TOKEN_ID],
        )
        self.assertEqual(document["output"]["decoded_text"], "Hello world")
        self.assertEqual(document["stop"]["reason"], "eos_token")
        self.assertTrue(document["stop"]["eos_emitted"])
        token_flow = document["token_flow"]
        self.assertEqual(len(token_flow), len(FIXED_CHAT_TOKEN_IDS) + 3)
        self.assertEqual(
            [step["position"] for step in token_flow],
            list(range(len(token_flow))),
        )
        self.assertFalse(token_flow[0]["cache_slot_reused"])
        self.assertTrue(all(step["cache_slot_reused"] for step in token_flow[1:]))
        self.assertTrue(all(step["cache_slot"] == 0 for step in token_flow))
        self.assertTrue(
            all(step["schedule_event_count"] == 483 for step in token_flow)
        )
        self.assertEqual(
            document["cache_slot_flow"]["reuse_count"],
            len(token_flow) - 1,
        )
        self.assertIn("official-geometry logits", document["claim_boundary"])
        self.assertIn(
            "readable official-checkpoint dialogue",
            document["claim_boundary"],
        )

    def test_host_max_token_stop_and_outside_vocabulary_rejection(self) -> None:
        document = host_generation_document(
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
            max_new_tokens=2,
            cache_slot=1,
        )
        self.assertEqual(document["stop"]["reason"], "max_new_tokens")
        self.assertFalse(document["stop"]["eos_emitted"])
        self.assertEqual(document["output"]["decoded_text"], "Hello world")
        self.assertTrue(all(step["cache_slot"] == 1 for step in document["token_flow"]))
        with self.assertRaisesRegex(
            ContractError,
            "outside_reduced_execution_vocabulary",
        ):
            reduced_token_label(42)
        host = StructuralModel24Host()
        with self.assertRaisesRegex(
            ContractError,
            "outside_reduced_execution_vocabulary",
        ):
            host.step(42, "prompt")
        self.assertEqual(host.steps, [])
        with self.assertRaisesRegex(ContractError, "positive integer"):
            host_generation_document(
                DEFAULT_OFFICIAL_TOKENIZER_DIR,
                max_new_tokens=0,
            )

    def test_official_kv_ownership_and_final_interfaces(self) -> None:
        addresses = {
            kv_address(1, layer_id, 37, 1, 63, kind)
            for layer_id in range(24)
            for kind in ("K", "V")
        }
        self.assertEqual(len(addresses), 48)
        self.assertNotEqual(kv_owner(0, "K"), kv_owner(0, "V"))
        lm_head = grouped_lm_head_interface()
        self.assertEqual(lm_head["groups_per_logit"], 7)
        self.assertEqual(lm_head["output_logits"], 151936)
        self.assertEqual(argmax_first([5, 5, 4]), 0)
        self.assertEqual(argmax_first([-4, 9, 9]), 1)
        with self.assertRaises(ContractError):
            argmax_first([])
        with self.assertRaises(ContractError):
            argmax_first([1, True])
        logits = [0xBC00] * OFFICIAL_GEOMETRY.vocab_size
        logits[7] = 0x3C00
        logits[9] = 0x3C00
        self.assertEqual(stable_top_k_f16(logits, 3), [7, 9, 0])
        with self.assertRaisesRegex(ContractError, "exactly 151936"):
            stable_top_k_f16(logits[:-1], 1)

    def test_small_geometries_exhaust_every_legal_address(self) -> None:
        for layers in (1, 2, 3):
            geometry = Geometry(layers, 2, 2, 2, 2, 4, 2, 5)
            events = expected_schedule(geometry)
            validate_trajectory(events, geometry)
            addresses = [
                kv_address(slot, layer, position, head, dimension, kind, geometry)
                for kind in ("K", "V")
                for slot in range(geometry.cache_slots)
                for layer in range(geometry.layers)
                for position in range(geometry.positions)
                for head in range(geometry.kv_heads)
                for dimension in range(geometry.head_dim)
            ]
            self.assertEqual(
                len(addresses),
                2
                * geometry.cache_slots
                * geometry.layers
                * geometry.positions
                * geometry.kv_heads
                * geometry.head_dim,
            )
            self.assertEqual(len(addresses), len(set(addresses)))
            self.assertEqual(sorted(addresses), list(range(len(addresses))))

    def test_every_kv_bound_and_type_fails_closed(self) -> None:
        geometry = Geometry(2, 2, 2, 2, 2, 4, 2, 5)
        illegal = (
            (-1, 0, 0, 0, 0, "K"),
            (2, 0, 0, 0, 0, "K"),
            (0, -1, 0, 0, 0, "K"),
            (0, 2, 0, 0, 0, "K"),
            (0, 0, -1, 0, 0, "K"),
            (0, 0, 2, 0, 0, "K"),
            (0, 0, 0, -1, 0, "K"),
            (0, 0, 0, 2, 0, "K"),
            (0, 0, 0, 0, -1, "K"),
            (0, 0, 0, 0, 2, "K"),
            (0, 0, 0, 0, 0, "X"),
            (False, 0, 0, 0, 0, "K"),
        )
        for arguments in illegal:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ContractError):
                    kv_address(*arguments, geometry=geometry)

    def test_schedule_fault_latches_until_reset(self) -> None:
        geometry = Geometry(1, 1, 1, 1, 1, 2, 1, 2)
        events = expected_schedule(geometry)
        machine = ExecutionMachine(geometry)
        bad = dict(events[0])
        bad["operation"] = "q_proj"
        with self.assertRaises(ContractError):
            machine.accept(bad)
        self.assertTrue(machine.faulted)
        with self.assertRaisesRegex(ContractError, "faulted"):
            machine.accept(events[0])
        machine.reset()
        for event in events:
            machine.accept(event)
        self.assertTrue(machine.done)
        with self.assertRaisesRegex(ContractError, "completed"):
            machine.accept(events[-1])

    def test_contract_and_reviewed_map_mutations_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["schedule"]["event_count"] = 482
        with self.assertRaises(ContractError):
            validate_execution_contract(
                mutated,
                self.tensor_payload,
                self.control_payload,
            )
        mutated_tensor = self.tensor_payload.replace(
            b'"layers": 24',
            b'"layers": 23',
            1,
        )
        with self.assertRaisesRegex(ContractError, "tensor map SHA256"):
            validate_execution_contract(
                self.contract,
                mutated_tensor,
                self.control_payload,
            )

    def test_vectors_are_byte_reproducible_and_validate(self) -> None:
        first = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
        )
        second = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
        )
        self.assertEqual(first, second)
        official = load_json_bytes(
            first["official_token_decision.json"],
            "official token decision",
        )
        layer0 = load_json_bytes(
            first["official_layer0_slice.json"],
            "official layer-0 slice",
        )
        self.assertEqual(len(layer0["model_binding"]["consumed_tensors"]), 26)
        self.assertEqual(len(layer0["intermediates"]), 20)
        self.assertTrue(
            all(
                item["within_tolerance"]
                for item in layer0["independent_reference"]["comparisons"].values()
            )
        )
        self.assertEqual(
            layer0["independent_reference"]["sampled_projection_bit_oracle_checks"],
            42,
        )
        self.assertEqual(
            len(layer0["final_token_decision_handoff"]["f16_bits"]),
            OFFICIAL_GEOMETRY.hidden_size,
        )
        self.assertIn(
            "layers 1 through 23 must execute",
            layer0["final_token_decision_handoff"]["status"],
        )
        consumed = official["model_binding"]["consumed_tensors"]
        self.assertEqual(consumed["lm_head.weight"]["sha256"], TIED_WEIGHT_SHA256)
        self.assertEqual(
            consumed["model.embed_tokens.weight"]["sha256"],
            TIED_WEIGHT_SHA256,
        )
        self.assertEqual(consumed["model.norm.weight"]["sha256"], FINAL_NORM_SHA256)
        self.assertNotEqual(
            consumed["lm_head.weight"]["absolute_file_offsets"],
            consumed["model.embed_tokens.weight"]["absolute_file_offsets"],
        )
        self.assertEqual(
            official["terminal_hidden_state"]["sha256"],
            FIXED_TERMINAL_HIDDEN_SHA256,
        )
        logits = official["lm_head"]["logits_f16_bits"]
        self.assertEqual(official["lm_head"]["vocab_size"], 151936)
        self.assertEqual(len(logits), 151936)
        top_k = official["token_decision"]["top_k"]
        self.assertEqual(len(top_k), 10)
        self.assertEqual(
            official["token_decision"]["argmax_token_id"],
            stable_top_k_f16(logits, 1)[0],
        )
        self.assertEqual([record["rank"] for record in top_k], list(range(10)))
        self.assertIn(
            "no official 24-layer numerical execution",
            official["claim_boundary"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "vectors"
            generated = generate(REPOSITORY_ROOT, output_dir)
            self.assertEqual(generated, first)
            summary = validate_vector_directory(REPOSITORY_ROOT, output_dir)
            self.assertEqual(summary["official_events"], 483)

    def test_vector_mutations_missing_extra_and_duplicate_keys_are_rejected(self) -> None:
        artifacts = build_vector_artifacts(
            sha256_bytes(self.contract_payload),
            sha256_bytes(self.bindings_payload),
            DEFAULT_OFFICIAL_TOKENIZER_DIR,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for name, payload in artifacts.items():
                (output_dir / name).write_bytes(payload)

            official_path = output_dir / "official_schedule.json"
            original = official_path.read_bytes()
            official_path.write_bytes(original.replace(b'"event_count":483', b'"event_count":482', 1))
            with self.assertRaisesRegex(ContractError, "oracle regeneration"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            official_path.write_bytes(original)

            (output_dir / "extra.json").write_text("{}\n", encoding="ascii")
            with self.assertRaisesRegex(ContractError, "artifact set"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            (output_dir / "extra.json").unlink()

            manifest_path = output_dir / "manifest.json"
            original_manifest = manifest_path.read_bytes()
            manifest_path.write_bytes(
                original_manifest.replace(
                    b'{"algorithm":',
                    b'{"schema_version":1,"algorithm":',
                    1,
                )
            )
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)
            manifest_path.write_bytes(original_manifest)

            official_path.unlink()
            with self.assertRaisesRegex(ContractError, "artifact set"):
                validate_vector_directory(REPOSITORY_ROOT, output_dir)

if __name__ == "__main__":
    unittest.main()
