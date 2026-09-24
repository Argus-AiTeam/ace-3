#!/usr/bin/env python3
"""Admit the reviewed position-2 tail without invoking either RTL runtime."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ace3/model"))

import numpy as np

import model24_host_runtime as host
import prepare_position2_tail_binding as binding
import prepare_position3_continuation as embedding

CONTRACT = ROOT / "ace3/contracts/token358_position3_preflight.json"
TEST = ROOT / "ace3/model/tests/test_token358_position3_preflight.py"
require = binding.require
load = binding.load
record = binding.record
authenticate = binding.authenticate


def write_new(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)


def source_records() -> list[dict]:
    paths = {CONTRACT, *binding.source_paths()}
    pending = [Path(__file__).resolve(), TEST, ROOT / "ace3/model/model24_host_runtime.py",
               ROOT / "ace3/model/prepare_position3_continuation.py"]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        paths.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([item.name for item in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [])
            for name in names:
                local = ROOT / "ace3/model" / (name.split(".")[0] + ".py")
                if local.is_file():
                    pending.append(local)
    return [record(path) for path in sorted(paths)]


def pinned(path: Path, digest: str) -> dict:
    item = record(path)
    require(item["sha256"] == digest, f"foreign parent: {path}")
    authenticate(item)
    return item


def authenticate_seal(path: Path, digest: str) -> dict:
    item = pinned(path, digest)
    rows = load(path)["files"]
    require(len({row["path"] for row in rows}) == len(rows), "duplicate sealed path")
    for row in rows:
        authenticate(row)
    return item


def embedding_bits(checkpoint: Path, tensor: dict, token: int) -> list[int]:
    bits = embedding.load_official_embedding(checkpoint, token)
    # The second reader uses the reviewed raw tensor offset, not safetensors slicing.
    raw = np.memmap(checkpoint, dtype="<u2", mode="r",
                    offset=tensor["offset"] + token * 896 * 2, shape=(896,))
    require(bits == raw.tolist(), "independent embedding reader mismatch")
    require(np.isfinite(raw.view("<f2")).all(), "nonfinite embedding")
    return bits


def semantic_bits(bits: list[int]) -> str:
    return hashlib.sha256(np.asarray(bits, dtype="<u2").tobytes()).hexdigest()


def authenticated_context() -> dict:
    contract = load(CONTRACT)
    tail = ROOT / contract["tail_directory"]
    tail_seal = authenticate_seal(tail / "seal.json", contract["tail_seal_sha256"])
    review_record = pinned(Path(contract["tail_review"]), contract["tail_review_sha256"])
    review = load(Path(review_record["path"]))
    require(review["producer_role"] == "reviewer"
            and review["mission_id"] == "1693a83a95c6"
            and review["round"] == 3 and review["review"]["status"] == "done",
            "tail lacks independent Reviewer admission")
    parent, hidden = binding.admitted_parent(load(binding.CONTRACT))
    frozen_tail = load(tail / "frozen.json")
    package = load(tail / "parent_runtime_pass_package.json")
    require(frozen_tail["parent"] == package["parent"] == parent,
            "foreign layer23/tail parent")
    for source in frozen_tail["sources"]:
        relative = Path(source["path"]).relative_to(tail / "sources")
        require(record(ROOT / relative)["sha256"] == source["sha256"],
                f"reviewed tail source drift: {relative}")
    result = load(tail / "result.json")
    require(result["status"] == "TAIL_PASS_REVIEW_REQUIRED"
            and result["natural_exit"] is True and result["runtime_tail_invocations"] == 1
            and result["earliest_tail_mismatch"] is None
            and all(result[key] == 0 for key in (
                "final_rmsnorm_material_mismatches", "logit_mismatches",
                "selected_accumulator_mismatches", "topk_mismatches")),
            "tail terminal did not pass")
    inputs = load(tail / "authenticated_inputs.json")
    binding.authenticate_tree(inputs)
    require(inputs["checkpoint"] == package["checkpoint"]
            and inputs["layer23_terminal"] == parent["terminal"]
            and inputs["tensors"] == package["tensors"], "tail input lineage mismatch")
    checkpoint = Path(inputs["checkpoint"]["path"])
    require(inputs["checkpoint"]["sha256"] == embedding.CHECKPOINT_SHA256,
            "foreign official checkpoint")
    tensors = inputs["tensors"]
    tied = tensors["model.embed_tokens.weight"]
    require(tied["dtype"] == "F16" and tied["shape"] == [151936, 896]
            and tied["sha256"] == tensors["lm_head.weight"]["sha256"],
            "tied weight lineage mismatch")
    head_input = load(tail / "head_input.json")
    rms = load(tail / "final_rmsnorm.json")
    require(head_input["weights"] == tied
            and head_input["actual_rmsnorm"] == rms["raw_output"],
            "tied head did not consume the reviewed RMSNorm")
    weights = np.memmap(checkpoint, dtype="<u2", mode="r",
                        offset=tensors["model.norm.weight"]["offset"], shape=(896,))
    expected_rms = binding.independent_rmsnorm(hidden, weights.tolist())
    actual_rms = [int(line, 16) for line in Path(rms["raw_output"]["path"]).read_text().split()]
    require(expected_rms == actual_rms, "independent reviewed-parent RMSNorm mismatch")

    prefix = ROOT / contract["prefix_directory"]
    prefix_seal = authenticate_seal(prefix / "seal.json", contract["prefix_seal_sha256"])
    prefix_result = pinned(prefix / "result.json", contract["prefix_result_sha256"])
    prefix_frozen = pinned(prefix / "frozen.json", contract["prefix_frozen_sha256"])
    chain = load(prefix / "result.json")["layers"]
    frozen = load(prefix / "frozen.json")
    authenticate(frozen["parent_admission"])
    policy = load(Path(frozen["parent_admission"]["path"]))["policy"]
    require(policy["input_token_history"] == contract["input_token_history"]
            and all(policy[key] == value for key, value in contract["profile"].items()),
            "reviewed token history/profile mismatch")
    for row in frozen["sources"]:
        authenticate(row)
    layer22_path = Path(parent["result"]["path"]).parent.parent / "layer22/result.json"
    layer22_execution = load(layer22_path.parent / "execution.json")
    layer23_execution = load(Path(parent["result"]["path"]).parent / "execution.json")
    require(layer22_execution["predecessor"] == chain[-1]
            and layer23_execution["predecessor"] == record(layer22_path),
            "foreign layer22/23 predecessor chain")
    chain = [*chain, record(layer22_path), parent["result"]]
    require(len(chain) == 24, "missing K/V layer results")
    kv = []
    previous = None
    for index, item in enumerate(chain):
        authenticate(item)
        layer = load(Path(item["path"]))
        require(layer["layer_index"] == index and layer["actual_output_fed_rtl_chain"] is True
                and [row["position"] for row in layer["positions"]] == [0, 1, 2]
                and [row["stage"] for row in layer["independent_comparisons"]] == list(range(19))
                and all(row["failure_count"] == 0 for row in layer["independent_comparisons"]),
                f"incomplete reviewed layer {index}")
        authenticate(layer["live_binary"])
        for position, transaction in enumerate(layer["positions"]):
            state = transaction["output_state"]
            require(state["path"] == str(Path(item["path"]).parent /
                                        f"position{position + 1:03d}.state"),
                    f"foreign K/V state at layer {index}")
            for row in (state, transaction["output"], transaction["raw"]["terminal"],
                        transaction["raw"]["trace"]):
                authenticate(row)
            require(transaction["raw"]["done_count"] == 1
                    and transaction["exact_comparison"]["trace"]["mismatch_count"] == 0
                    and transaction["exact_comparison"]["final_hidden"]["mismatch_count"] == 0,
                    f"nonterminal or mismatching layer {index}")
            expected_input = (
                previous["positions"][position]["output"]["semantic_sha256"] if previous
                else semantic_bits(embedding_bits(checkpoint, tied,
                                                   contract["input_token_history"][position])))
            require(transaction["input"]["sha256"] == expected_input,
                    f"token history or actual-output chain mismatch at layer {index}")
        kv.append({
            "layer_index": index, "position": 2, "valid_positions": [0, 1, 2],
            "dtype": "FP16", "kv_heads": 2, "head_dimension": 64,
            "encoding": "Verilator serialized decoder state; restore requires bound binary",
            "layer_result": item, "live_binary": layer["live_binary"],
            "state": layer["positions"][2]["output_state"],
            "position_states": [row["output_state"] for row in layer["positions"]],
            "actual_kv_traces": [row["raw"]["trace"] for row in layer["positions"]],
        })
        previous = layer

    tokenizer_dir = ROOT / contract["tokenizer_directory"]
    tokenizer = host.authenticate_tokenizer(tokenizer_dir)
    history = contract["input_token_history"]
    serialization = tokenizer.decode(history, skip_special_tokens=False)
    require(tokenizer.encode(serialization, add_special_tokens=False).ids == history,
            "tokenizer/history round trip mismatch")
    prompt = {"source": "reviewed corrected-Q position-2 runtime input history",
              "serialization": serialization, "token_ids": history,
              "serialization_utf8_sha256": hashlib.sha256(serialization.encode()).hexdigest()}
    selected = load(tail / "selected_token.json")
    topk = load(tail / "topk.json")
    logits = []
    for index, line in enumerate((tail / "raw/logits.txt").read_text().splitlines()):
        token, bits, accumulator = line.split()
        raw = int(bits, 16)
        require(int(token) == index and 0 <= raw <= 65535, "raw logit index/payload mismatch")
        value = np.asarray(raw, dtype="<u2").view("<f2").item()
        require(np.isfinite(value), "nonfinite final logit")
        logits.append((value, index, raw))
    require(len(logits) == 151936, "incomplete vocabulary receipt")
    ordered = sorted(logits, key=lambda row: (-row[0], row[1]))[:10]
    expected_topk = [[rank, token, bits] for rank, (_, token, bits) in enumerate(ordered)]
    raw_topk = [[int(rank), int(token), int(bits, 16)]
                for rank, token, bits in
                (line.split() for line in (tail / "raw/topk.txt").read_text().splitlines())]
    require(topk["actual"] == topk["expected"] == raw_topk == expected_topk
            and topk["mismatch_count"] == 0
            and topk["selection_policy"] == package["selection_policy"],
            "final Top-K receipt mismatch")
    require(selected["actual_token_id"] == selected["expected_token_id"]
            == result["selected_token_id"] == expected_topk[0][1] == contract["selected_token_id"]
            and selected["actual_logit_bits"] == selected["expected_logit_bits"] == expected_topk[0][2]
            and selected["matches"] is True and selected["parent_layer23"] == parent["terminal"]
            and selected["tail_input"] == rms["raw_output"], "selected-token lineage mismatch")
    terminal = {"tail_seal": tail_seal, "tail_review": review_record,
                "tail_result": record(tail / "result.json"), "layer23": parent,
                "topk": record(tail / "topk.json"),
                "selected_token": record(tail / "selected_token.json")}
    authority = {"purpose": "receipt-to-host-preflight only; no decoder launch authority",
                 "terminal": terminal}
    receipt = {
        "schema_version": 1, "kind": host.SELECTED_TOKEN_RECEIPT_KIND,
        "model_binding": host._receipt_model_binding(),
        "tokenizer_binding": host.official_tokenizer_binding(),
        "prompt_lineage": host._receipt_prompt_lineage(prompt),
        "terminal_evidence": terminal,
        "authority": {"kind": host.SELECTED_TOKEN_AUTHORITY_KIND, "lineage": authority,
                      "receipt_use_authorized": True, "authority_consumed": False},
        "selection": {
            "generation_ordinal": 1, "vocab_size": 151936,
            "selection_policy": host.SELECTED_TOKEN_POLICY,
            "selected_token_id": selected["actual_token_id"],
            "selected_logit_f16_bits": selected["actual_logit_bits"],
            "top_k": [{"rank": rank, "token_id": token, "logit_f16_bits": bits,
                       "logit_q24": host.decode_f16_q24(bits)[0]}
                      for rank, token, bits in expected_topk],
        },
    }
    step = host.form_next_dialogue_step_from_receipt(receipt, tokenizer, prompt, terminal, authority)
    bits = embedding_bits(checkpoint, tied, selected["actual_token_id"])
    return {
        "contract": contract, "tokenizer": tokenizer, "embedding_bits": bits,
        "package": {
            "schema": contract["schema"], "position": 3, "selected_token_id": 358,
            "profile": contract["profile"], "sources": source_records(),
            "parents": terminal,
            "prefix": {"result": prefix_result, "seal": prefix_seal, "frozen": prefix_frozen,
                       "admission": frozen["parent_admission"]},
            "checkpoint": inputs["checkpoint"], "tied_weights": tensors,
            "tokenizer_files": [record(tokenizer_dir / name)
                                for name in ("tokenizer.json", "tokenizer_config.json")],
            "prompt": prompt, "selected_token_receipt": receipt, "host_step": step,
            "token_history": step["next_model_input_token_ids"], "kv_parents": kv,
            "embedding": {"token_id": 358, "position": 3, "dtype": "FP16", "elements": 896,
                          "semantic_sha256": semantic_bits(bits)},
            "public_interfaces": {"decoder": frozen["public_contract"],
                                  "tail": frozen_tail["public_interfaces"]},
            "runtime_tail_invocations": 0, "position3_execution_performed": False,
            "authority_consumed": False, "execution_authorized": False,
            "claim_boundary": contract["claim_boundary"],
        },
    }


def validate_document(document: dict, expected: dict, tokenizer: object) -> None:
    require(document["runtime_tail_invocations"] == 0
            and document["position3_execution_performed"] is False
            and document["authority_consumed"] is False
            and document["execution_authorized"] is False, "consumed authority or execution")
    receipt = document["selected_token_receipt"]
    terminal = expected["parents"]
    authority = expected["selected_token_receipt"]["authority"]["lineage"]
    step = host.form_next_dialogue_step_from_receipt(
        receipt, tokenizer, expected["prompt"], terminal, authority)
    require(step == expected["host_step"] == document["host_step"], "host receipt mismatch")
    require(document == expected, "preflight package semantic/binding mismatch")


def negative_controls(expected: dict, tokenizer: object) -> list[dict]:
    mutations = {
        "stale_token_271": lambda d: d["selected_token_receipt"]["selection"].update(selected_token_id=271),
        "foreign_parent": lambda d: d["selected_token_receipt"]["terminal_evidence"].update(layer23={}),
        "consumed_authority": lambda d: d["selected_token_receipt"]["authority"].update(authority_consumed=True),
        "missing_kv": lambda d: d["kv_parents"].pop(),
        "mismatched_token_history": lambda d: d["token_history"].__setitem__(2, 271),
        "embedding_mismatch": lambda d: d["embedding"].update(semantic_sha256="0" * 64),
        "topk_mismatch": lambda d: d["selected_token_receipt"]["selection"]["top_k"][0].update(logit_q24=0),
        "source_drift": lambda d: d["sources"][0].update(sha256="0" * 64),
    }
    rows = []
    for name, mutate in mutations.items():
        candidate = copy.deepcopy(expected)
        mutate(candidate)
        try:
            validate_document(candidate, expected, tokenizer)
        except (RuntimeError, ValueError) as error:
            rows.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise RuntimeError(f"negative control accepted: {name}")
    return rows


def bind_embedding(context: dict, path: Path) -> dict:
    payload = embedding.embedding_payload(context["embedding_bits"])
    require(path.read_bytes() == payload, "materialized embedding mismatch")
    document = copy.deepcopy(context["package"])
    document["embedding"]["file"] = record(path)
    return document


def run(operation: str, output: Path) -> dict:
    require(output.parent == ROOT / "build", "package must be directly under build/")
    if operation == "prepare":
        output.mkdir(exist_ok=False)
        write_new(output / "frozen.json", {
            "contract": load(CONTRACT), "sources": source_records(),
            "argv": sys.argv, "python": sys.version,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "torch", "safetensors", "tokenizers")},
            "runtime_tail_invocations": 0, "position3_execution_performed": False,
        })
    try:
        frozen = load(output / "frozen.json")
        require(frozen["contract"] == load(CONTRACT)
                and frozen["sources"] == source_records(), "current-source admission drift")
        if operation == "validate":
            binding.authenticate_tree(load(output / "seal.json"))
        context = authenticated_context()
        embedding_path = output / "position003_token358_embedding.hex"
        if operation == "prepare":
            with embedding_path.open("xb") as stream:
                stream.write(embedding.embedding_payload(context["embedding_bits"]))
        expected = bind_embedding(context, embedding_path)
        package_path = output / "runtime_pass_package.json"
        if operation == "prepare":
            write_new(package_path, expected)
        validate_document(load(package_path), expected, context["tokenizer"])
        negatives = negative_controls(expected, context["tokenizer"])
        require([row["case"] for row in negatives] == context["contract"]["negative_controls"],
                "negative-control coverage mismatch")
        result = {
            "status": "PREFLIGHT_PASS_REVIEW_REQUIRED",
            "selected_token_id": 358, "target_position": 3,
            "input_token_history": expected["token_history"],
            "kv_parent_count": len(expected["kv_parents"]),
            "runtime_tail_invocations": 0, "position3_execution_performed": False,
            "authority_consumed": False, "execution_authorized": False,
            "independent_embedding_mismatches": 0,
            "independent_review": "pending Host Reviewer",
            "negative_controls": negatives,
            "package": record(package_path), "claim_boundary": expected["claim_boundary"],
        }
        if operation == "prepare":
            write_new(output / "result.json", result)
            write_new(output / "seal.json", {
                "files": [record(path) for path in sorted(output.iterdir()) if path.is_file()]
            })
        return result
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        if operation == "prepare":
            write_new(output / "failure.json", {
                "status": "PREFLIGHT_FAIL_NOT_EXECUTED",
                "taxonomy": "preexecution_binding_failure",
                "root_cause_hypothesis": str(error),
                "regression": "Fresh current-source admission and the eight fail-closed controls",
                "runtime_tail_invocations": 0, "position3_execution_performed": False,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.operation, args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
