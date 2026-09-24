#!/usr/bin/env python3
"""Run one reviewed token-358 layer0 continuation without replaying its prefix."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import os
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

import decoder_layer0_oracle as decoder
import replay_layer00_q_projection_repair as comparison
import run_corrected_q_selected_token_frontier as receipts
import run_repaired_q_selected_token_frontier as frontier
import validate_selected_token_position2_traversal as traversal

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "build/token358_position3_preflight_attempt001"
REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "3e8badd5ba3c/round-0001.json"
)
REFERENCE = ROOT / "build/independent_fp16_trajectory_20260906_1133"
TOP = "ace3_decoder_layer0_token_engine"
require = frontier.require
record = frontier.file_record
load = receipts.load
write = receipts.write
authenticate = receipts.authenticate


def independent_helpers() -> dict:
    namespace = {
        "np": np, "torch": torch, "torch_functional": functional,
        "_require": require, "require": require, "DiagnosticError": RuntimeError,
        "GROUP_SIZE": 128, "HEAD_DIM": 64, "HIDDEN_SIZE": 896,
        "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
        "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7),
    }
    # Extract only pure definitions: importing the archived driver writes its tree.
    for filename, names in (
        ("official_single_decoder_layer.py",
         {"_torch_unpack", "_torch_linear", "_torch_rmsnorm"}),
        ("propagated_reference.py", {"propagated_reference"}),
    ):
        path = REFERENCE / "source" / filename
        nodes = [
            node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        require({node.name for node in nodes} == names, "independent helper missing")
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace


def array_input(item: dict) -> np.ndarray:
    path = Path(item["path"])
    require(record(path)["sha256"] == item["file_sha256"], f"array drift: {path}")
    values = np.load(path, allow_pickle=False)
    require(values.dtype == np.dtype("<u2") and list(values.shape) == item["shape"]
            and traversal.sha256_bytes(values.tobytes()) == item["semantic_sha256"],
            f"array semantics drift: {path}")
    return values


def run(root: Path) -> dict:
    started = time.monotonic()
    torch.set_num_threads(2)
    review = load(REVIEW)
    require(review["producer_role"] == "reviewer"
            and review["mission_id"] == "3e8badd5ba3c"
            and review["review"]["status"] == "done", "preflight review not accepted")
    for item in load(PACKAGE / "seal.json")["files"]:
        authenticate(item)
    admission = load(PACKAGE / "result.json")
    require(admission["status"] == "PREFLIGHT_PASS_REVIEW_REQUIRED",
            "preflight did not pass")
    authenticate(admission["package"])
    package = load(Path(admission["package"]["path"]))
    require(package["selected_token_id"] == 358 and package["position"] == 3
            and package["token_history"] == [9707, 1879, 0, 358],
            "selected-token input lineage mismatch")
    authenticate(package["prefix"]["frozen"])
    parent_frozen = load(Path(package["prefix"]["frozen"]["path"]))
    parent = package["kv_parents"][0]
    require(parent["layer_index"] == 0 and parent["position"] == 2
            and parent["valid_positions"] == [0, 1, 2] and parent["dtype"] == "FP16",
            "layer0 K/V parent incompatible")
    for item in (parent["state"], parent["live_binary"], parent["layer_result"],
                 package["embedding"]["file"], package["checkpoint"],
                 *parent["actual_kv_traces"]):
        authenticate(item)
    sources = {item["path"]: item
               for item in parent_frozen["sources"] + package["sources"]}
    for item in sources.values():
        authenticate(item)
    sources[str(Path(__file__).resolve())] = record(Path(__file__).resolve())
    for filename in ("official_single_decoder_layer.py", "propagated_reference.py"):
        path = REFERENCE / "source" / filename
        sources[str(path)] = record(path)
    header = (ROOT / f"ace3/rtl/{TOP}.sv").read_text().split(");", 1)[0] + ");"
    require(header == parent_frozen["public_contract"]
            == package["public_interfaces"]["decoder"], "public decoder contract drift")
    for name in ("verilator", "g++", "make"):
        authenticate(parent_frozen["tools"][name]["executable"])
    for name, value in parent_frozen["build_environment"].items():
        if name not in ("PATH", "PYTHONPATH"):
            require(os.environ.get(name) == value, f"build environment drift: {name}")

    reference_path = REFERENCE / "recovery001/layer00_generation1.json"
    matches = [item for item in parent_frozen["independent_references"]
               if item["path"] == str(reference_path)]
    require(len(matches) == 1, "independent parent reference is not bound")
    authenticate(matches[0])
    reference = load(reference_path)
    require(reference["history"] == package["token_history"][:3]
            and reference["layer"] == 0, "independent K/V history mismatch")
    own_k = array_input(reference["own_cache"]["k"])
    own_v = array_input(reference["own_cache"]["v"])
    require(own_k.shape == own_v.shape == (3, 2, 64), "independent cache geometry")
    policy = load(REFERENCE / "recovery001/POLICY.json")
    require(policy["comparison"] == {
        "absolute_tolerance": 0.125,
        "accept": "finite AND (abs_error<=0.125 OR (relative_error<0.001 AND orderedFP16_ULP<=1))",
        "max_ulp_distance": 1,
        "relative_denominator": "max(abs(reference),2^-14)",
        "relative_tolerance_strict": 0.001,
    }, "accepted comparison policy drift")
    require((comparison.ABSOLUTE_TOLERANCE, comparison.RELATIVE_TOLERANCE,
             comparison.MAX_ULP_DISTANCE) == (0.125, 0.001, 1),
            "material comparator thresholds drift")
    write(root / "frozen.json", {
        "argv": [sys.executable, "-B", *sys.argv], "python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "safetensors")},
        "review": record(REVIEW), "package": admission["package"],
        "package_seal": record(PACKAGE / "seal.json"),
        "sources": list(sources.values()), "public_contract": header,
        "parameters": {"LAYER_INDEX": 0, "ACCURATE_SILU": 1},
        "parent_frozen": package["prefix"]["frozen"], "kv_parent": parent,
        "embedding": package["embedding"], "checkpoint": package["checkpoint"],
        "independent_parent": matches[0], "independent_cache": reference["own_cache"],
        "independent_policy": record(REFERENCE / "recovery001/POLICY.json"),
        "comparison_policy": policy["comparison"], "tools": parent_frozen["tools"],
        "build_environment": {name: os.environ.get(name)
                              for name in parent_frozen["build_environment"]},
        "scope": "One position3 layer0 transaction; no prefix replay or later layer",
    })
    for name in ("verilator", "g++", "make"):
        receipts.logged([parent_frozen["tools"][name]["executable"]["path"], "--version"],
                        root / f"{name}.version.log")
    receipts.logged([
        "make", "--no-print-directory", "model24-rtl-layer-compile",
        "MODEL24_RTL_LAYER_INDEX=0", "MODEL24_RTL_ACCURATE_SILU=1",
        f"MODEL24_RTL_CASCADE_DIR={root}",
    ], root / "compile.log")
    binary = root / f"compiled/layer0/obj_dir/V{TOP}"
    old_objects = Path(parent["live_binary"]["path"]).parent
    new_objects = binary.parent
    old_layout = {p.name: record(p) for p in old_objects.iterdir()
                  if p.suffix in (".cpp", ".h")}
    new_layout = {p.name: record(p) for p in new_objects.iterdir()
                  if p.suffix in (".cpp", ".h")}
    abi = {
        "parent_binary": parent["live_binary"], "fresh_binary": record(binary),
        "parent_generated_sources": old_layout, "fresh_generated_sources": new_layout,
        "same_generated_layout": bool(old_layout) and old_layout.keys() == new_layout.keys()
        and all(old_layout[name]["sha256"] == new_layout[name]["sha256"]
                for name in old_layout),
    }
    write(root / "state_abi.json", abi)
    require(abi["same_generated_layout"], "fresh compiled serialized-state ABI differs")
    print("CURRENT_WORKTREE_COMPILED_STATE_ABI_MATCH", flush=True)

    hidden = traversal.load_hidden_bits(Path(package["embedding"]["file"]["path"]))
    require(traversal.sha256_bytes(hidden.tobytes()) == package["embedding"]["semantic_sha256"],
            "embedding semantic mismatch")
    layer_root = root / "layer00"
    with safe_open(package["checkpoint"]["path"], framework="np") as checkpoint:
        official = np.ascontiguousarray(
            checkpoint.get_slice("model.embed_tokens.weight")[358]
        ).view("<u2")
        require(np.array_equal(official, hidden), "token358 checkpoint embedding mismatch")
        vectors = traversal.materialize_transaction_vectors(
            checkpoint, 0, 3, hidden, layer_root / "vectors"
        )
        values = traversal.layer_oracle_values(checkpoint, 0, vectors)
        tensors = {item["name"]: np.ascontiguousarray(checkpoint.get_tensor(item["name"]))
                   for item in reference["checkpoint_tensor_hashes"]}
    require(len(tensors) == len(vectors["tensors"]) == 26, "layer0 tensor closure mismatch")
    for item in reference["checkpoint_tensor_hashes"]:
        require(traversal.sha256_bytes(tensors[item["name"]].tobytes()) == item["sha256"],
                f"independent tensor binding drift: {item['name']}")
    expected, next_k, next_v = independent_helpers()["propagated_reference"](
        torch.from_numpy(hidden.view("<f2").astype(np.float64)[None, :]), tensors, 0,
        cached_k=torch.from_numpy(own_k.view("<f2").astype(np.float64)),
        cached_v=torch.from_numpy(own_v.view("<f2").astype(np.float64)), position_offset=3,
    )
    independent_dir = root / "independent"
    independent_dir.mkdir()
    expected_bits = {
        stage: np.asarray(value, dtype="<f2").view("<u2")
        for stage, value in expected[0].items()
    }
    independent_records = {
        str(stage): comparison.write_hex(independent_dir / f"stage{stage:02d}.hex", bits)
        for stage, bits in expected_bits.items()
    }
    for name, cache in (("k", next_k), ("v", next_v)):
        np.save(independent_dir / f"position003_own_{name}.npy",
                cache.numpy().astype("<f2").view("<u2"), allow_pickle=False)

    cache_k, cache_v = [], []
    for position, item in enumerate(parent["actual_kv_traces"]):
        payload = Path(item["path"]).read_bytes()
        cache_k.append(frontier.trace_stage(payload, position, 6, 128).tolist())
        cache_v.append(frontier.trace_stage(payload, position, 7, 128).tolist())
    oracle_started = time.monotonic()
    final, trace = decoder.run_token(values, hidden.tolist(), 3, cache_k, cache_v,
                                     accurate_silu=True)
    exact, expected_trace, expected_final = traversal._write_exact_oracle(layer_root, final, trace)
    vectors.update(traversal.materialize_runtime_vector_contract(
        0, 3, final, trace, layer_root / "vectors"
    ))
    write(root / "prepared.json", {
        "vectors": vectors, "exact_oracle": exact, "independent_stages": independent_records,
        "oracle_seconds": time.monotonic() - oracle_started,
        "binary": record(binary), "input_state": parent["state"],
    })
    for item in [*sources.values(), parent["state"]]:
        authenticate(item)
    traversal.run_logged = receipts.logged
    write(root / "simulation_started.json", {"started_at": time.time(), "position": 3, "layer": 0})
    transaction, actual_hidden = traversal.execute_transaction(
        binary, 0, 3, hidden, vectors, layer_root / "vectors", layer_root / "position003",
        layer_root / "position004.state", Path(parent["state"]["path"]),
    )
    write(layer_root / "transaction.json", transaction)
    payload = Path(transaction["raw"]["trace"]["path"]).read_bytes()
    exact_comparison = {
        "trace": traversal.exact_hex_comparison(payload, expected_trace, "trace"),
        "final_hidden": traversal.exact_hex_comparison(
            Path(transaction["output"]["path"]).read_bytes(), expected_final, "final_hidden"),
    }
    comparisons = []
    for stage in range(19):
        bits = frontier.trace_stage(payload, 3, stage, len(expected_bits[stage]),
                                    4 if stage in (8, 9) else None)
        comparisons.append({
            "layer": 0, "position": 3, "stage": stage,
            "actual_trace": transaction["raw"]["trace"],
            "independent_reference": independent_records[str(stage)],
            **comparison.comparison(bits, expected_bits[stage]),
        })
    first = next((item for item in comparisons if item["failure_count"]), None)
    require(np.array_equal(actual_hidden, frontier.trace_stage(payload, 3, 18, 896)),
            "actual final hidden/trace mismatch")
    require(all(item["exact_match"] for item in exact_comparison.values()),
            "exact runtime oracle mismatch")
    for item in sources.values():
        authenticate(item)
    authenticate(parent["state"])
    result = {
        "status": "EARLIEST_MATERIAL_MISMATCH_LOCALIZED" if first
                  else "ZERO_MATERIAL_MISMATCHES_LAYER00_POSITION003",
        "selected_token_id": 358, "position": 3, "layer": 0,
        "token_history": package["token_history"],
        "natural_terminal_transactions": 1, "prefix_rtl_replays": 0,
        "transaction": record(layer_root / "transaction.json"),
        "output_hidden": transaction["output"], "output_state": transaction["output_state"],
        "kv_valid_positions": [0, 1, 2, 3], "kv_parent": parent,
        "state_abi": record(root / "state_abi.json"),
        "exact_comparison": exact_comparison, "independent_comparisons": comparisons,
        "material_mismatches": sum(item["failure_count"] for item in comparisons),
        "earliest_material_mismatch": first,
        "elapsed_seconds": time.monotonic() - started,
        "independent_review": "pending Host Reviewer",
        "boundary": "Computer-local W4A16 layer0 only; no third-token or hardware claim",
    }
    if first:
        result.update(
            taxonomy="numerical_policy_material_mismatch",
            root_cause_hypothesis="RTL propagation diverges from the independent FP16 policy",
            regression="Reproduce the earliest recorded stage with unchanged policy in a fresh attempt",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.attempt_dir.absolute()
    require(root.parent == ROOT / "build" and root.name.startswith("token358_position3_layer0_attempt"),
            "attempt must be in its build namespace")
    root.mkdir(exist_ok=False)
    write(root / "execution.json", {
        "argv": [sys.executable, "-B", *sys.argv], "cwd": str(ROOT),
        "started_at": time.time(), "runner": record(Path(__file__).resolve()),
        "command": shlex.join([sys.executable, "-B", *sys.argv]),
    })
    try:
        result = run(root)
    except (RuntimeError, OSError, ValueError, AssertionError, KeyError) as error:
        simulated = (root / "simulation_started.json").exists()
        result = {
            "status": "RTL_ATTEMPT_FAILED" if simulated else "EVALUATOR_NO_EXECUTION",
            "taxonomy": "runtime_or_output_contract_failure" if simulated
                        else "preexecution_binding_or_build_failure",
            "root_cause_hypothesis": str(error),
            "regression": "Resolve the named failure; use a fresh attempt without prefix replay",
            "position3_simulation_invoked": simulated, "rtl_correctness_conclusion": None,
            "independent_review": "pending Host Reviewer",
        }
        receipts.seal(root, result)
        raise
    receipts.seal(root, result)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
