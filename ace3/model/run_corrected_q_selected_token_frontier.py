#!/usr/bin/env python3
"""Execute a fresh corrected-Q RTL chain with an ordered independent frontier."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from safetensors import safe_open

import model24_persistent_kv_runtime as persistent
import run_repaired_q_selected_token_frontier as frontier
import validate_selected_token_position2_traversal as traversal


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "build/model24_persistent_kv_selected_token_corrected_q_admission001"
TOP = "ace3_decoder_layer0_token_engine"
LAST_LAYER = 21
CAUSAL = ROOT / "build/layer08_causal_replay_attempt002"
SAME_OPERAND = ROOT / "build/layer08_upstream_projection_policy_attempt001/result.json"


def write(path: Path, value: object) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def authenticate(record: dict) -> None:
    actual = frontier.file_record(Path(record["path"]))
    frontier.require(
        actual["sha256"] == record["sha256"]
        and actual["bytes"] == record["bytes"],
        f"binding drift: {record['path']}",
    )


def logged(command: list[str], log_path: Path) -> None:
    receipt = log_path.with_suffix(log_path.suffix + ".command.json")
    write(receipt, {"argv": command, "cwd": str(ROOT), "started_at": time.time()})
    started = time.monotonic()
    with log_path.open("xb") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    write(log_path.with_suffix(log_path.suffix + ".exit.json"), {
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "command": frontier.file_record(receipt),
        "log": frontier.file_record(log_path),
    })
    frontier.require(result.returncode == 0, f"command failed: {receipt}")


def prepare(root: Path) -> None:
    frontier.require(not root.exists() and not root.is_symlink(), "attempt exists")
    root.mkdir()
    plan = load(PARENT / "admission.json")
    accepted = load(PARENT / "result.json")
    authenticate(accepted["admission"])
    frontier.require(accepted["status"] == "ADMISSION_PREFLIGHT_PASS",
                     "reviewed admission is not passing")
    causal_seal = load(CAUSAL / "seal.json")
    for item in causal_seal["files"]:
        authenticate(item)
    causal = load(CAUSAL / "result.json")
    frontier.require(
        causal["status"] == "CONTROLLED_NUMERICAL_REPLAY_COMPLETE"
        and causal["exact_baseline_layers"] == list(range(8))
        and causal["exact_baseline_positions"] == [0, 1, 2]
        and causal["fresh_rtl_executed"] is False,
        "controlled causal replay is incomplete",
    )
    experiments = {row["name"]: row for row in causal["experiments"]}
    failures = {
        name: experiments[name]["layer08_fixed_k_scores"]["failure_count"]
        for name in ("baseline", "single_round_k_all", "single_round_v_all",
                     "single_round_kv_all")
    }
    frontier.require(
        failures == {"baseline": 3, "single_round_k_all": 0,
                     "single_round_v_all": 3, "single_round_kv_all": 3},
        "K-only controlled comparison no longer supports this candidate",
    )
    same_operand = load(SAME_OPERAND)
    for item in same_operand["authenticated_inputs"]:
        authenticate(item)
    frontier.require(
        same_operand["status"] == "NUMERICAL_DIAGNOSIS_COMPLETE"
        and same_operand["kv_double_round_exact"]
        and same_operand["q_single_round_exact"],
        "same-operand policy attribution is incomplete",
    )
    first_k = min(
        (row for row in same_operand["comparisons"]
         if row["projection"] == "k"
         and row["single_round_vs_rtl"]["different_count"] > 0),
        key=lambda row: (row["layer"], row["position"], row["stage"]),
    )
    repair_paths = {
        str(ROOT / path) for path in (
            "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
            "ace3/model/decoder_layer0_oracle.py",
            "ace3/contracts/decoder_layer0_token_engine.json",
            "ace3/model/run_corrected_q_selected_token_frontier.py",
        )
    }
    source_deltas = []
    for item in plan["sources"] + plan["parents"]:
        if item["path"] in repair_paths:
            source_deltas.append({
                "parent": item, "candidate": frontier.file_record(Path(item["path"])),
            })
        else:
            authenticate(item)
    for item in plan["official_inputs"]:
        authenticate(item)
    for tool in plan["tools"].values():
        authenticate(tool["executable"])
    spec = importlib.util.spec_from_file_location("corrected_q_admission", PARENT / "admit.py")
    frontier.require(spec is not None and spec.loader is not None, "admission import missing")
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    header = (ROOT / f"ace3/rtl/{TOP}.sv").read_text().split(");", 1)[0] + ");"
    frontier.require(header == plan["public_contract"], "public contract drift")
    sources = admission.source_paths({"sources": [
        *plan["sources"],
        {"path": str(Path(__file__).resolve())},
        {"path": str(Path(frontier.__file__).resolve())},
    ]})
    references = []
    for layer in range(LAST_LAYER + 1):
        record_path = frontier.REFERENCE / f"layer{layer:02d}_generation1.json"
        references.append(frontier.file_record(record_path))
        for item in load(record_path)["stages"]:
            record = frontier.file_record(Path(item["path"]))
            frontier.require(record["sha256"] == item["file_sha256"],
                             f"reference file binding drift: {record['path']}")
            references.append(record)
    frozen = {
        "parent_admission": frontier.file_record(PARENT / "admission.json"),
        "parent_result": frontier.file_record(PARENT / "result.json"),
        "sources": [frontier.file_record(path) for path in sources],
        "official_inputs": plan["official_inputs"],
        "independent_references": references,
        "public_contract": header,
        "policy": {
            **plan["policy"], "layers": list(range(LAST_LAYER + 1)),
            "k": "SINGLE_ROUND_BIAS=1; causal-replay candidate, review pending",
            "other_projections": "V/O/gate/up/down unchanged reviewed policy",
        },
        "repair": {
            "causal_seal": frontier.file_record(CAUSAL / "seal.json"),
            "causal_result": frontier.file_record(CAUSAL / "result.json"),
            "same_operand_result": frontier.file_record(SAME_OPERAND),
            "source_deltas": source_deltas,
            "controlled_score_failure_counts": failures,
            "earliest_observed_k_policy_difference": first_k,
            "taxonomy": "upstream_projection_bias_rounding_policy",
            "hypothesis": "Accumulated upstream K double rounding contributes to the layer08 score mismatch",
            "regression": "Fresh Q/K-single-round RTL from official embeddings; all stages through the next mismatch or layer21",
            "boundary": "The diagnostic holds layer08 K fixed; neither a unique causative layer nor full layer08 clearance is established",
        },
        "comparison_policy": {
            "implementation": "replay_layer00_q_projection_repair.comparison",
            "source_bound": True,
            "scope": "all 19 stages at selected-token position 2 after each fresh layer",
            "stop": "first material mismatch or layer21",
        },
        "runtime_argv": [
            sys.executable, "-B", str(Path(__file__).resolve()),
            "execute", "--attempt-dir", str(root),
        ],
        "tools": plan["tools"],
        "python": {"version": sys.version, "executable": frontier.file_record(Path(sys.executable))},
        "numpy_version": np.__version__,
        "build_environment": {key: os.environ.get(key) for key in admission.BUILD_ENV},
        "parents_are_runtime_inputs": False,
    }
    write(root / "frozen.json", frozen)
    rtl = [str(path) for path in sources
           if path.suffix == ".sv" and path.parent == ROOT / "ace3/rtl"]
    logged([
        plan["tools"]["iverilog"]["executable"]["path"], "-g2012", "-s", TOP,
        f"-P{TOP}.ACCURATE_SILU=1", "-o", str(root / "public-contract.vvp"), *rtl,
    ], root / "public-contract.log")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "attempt": str(root)}))


def execute(root: Path) -> dict:
    frozen = load(root / "frozen.json")
    frontier.require(not (root / "execution.json").exists(), "attempt already consumed")
    frontier.require(
        frozen["runtime_argv"] == [sys.executable, "-B", *sys.argv],
        "runtime command differs from frozen command",
    )
    frontier.require(
        frozen["build_environment"] == {
            key: os.environ.get(key) for key in frozen["build_environment"]
        }, "build environment drift",
    )
    for item in (frozen["sources"] + frozen["official_inputs"]
                 + frozen["independent_references"]):
        authenticate(item)
    for name in ("causal_seal", "causal_result", "same_operand_result"):
        authenticate(frozen["repair"][name])
    write(root / "execution.json", {
        "argv": frozen["runtime_argv"], "started_at": time.time(),
        "frozen": frontier.file_record(root / "frozen.json"),
    })
    traversal.run_logged = logged
    hidden = [persistent.token_embedding(token) for token in persistent.INPUT_TOKEN_HISTORY]
    layers = []
    first_failure = None
    with persistent.configured_transaction_engine(), safe_open(
        persistent.CHECKPOINT, framework="np"
    ) as checkpoint:
        for layer in range(LAST_LAYER + 1):
            started = time.monotonic()
            layer_root = root / f"layer{layer:02d}"
            tensors = layer_root / "vectors"
            vectors = traversal.materialize_transaction_vectors(
                checkpoint, layer, 2, hidden[2], tensors
            )
            inputs = [
                traversal.materialize_input_vectors(
                    layer, position, hidden[position],
                    layer_root / f"position{position:03d}/vectors",
                ) for position in (0, 1)
            ] + [vectors]
            binary = root / f"compiled/layer{layer}/obj_dir/V{TOP}"
            logged([
                "make", "--no-print-directory", "model24-rtl-layer-compile",
                f"MODEL24_RTL_LAYER_INDEX={layer}", "MODEL24_RTL_ACCURATE_SILU=1",
                f"MODEL24_RTL_CASCADE_DIR={root}",
            ], layer_root / "compile.log")
            binary_record = frontier.file_record(binary)
            values = traversal.layer_oracle_values(checkpoint, layer, vectors)
            cache_k, cache_v = [], []
            positions = []
            state_in = None
            for position in range(3):
                row, actual_hidden = traversal.execute_exact_transaction(
                    binary, layer, position, hidden[position], inputs[position],
                    tensors, layer_root / f"position{position:03d}",
                    layer_root / f"position{position + 1:03d}.state",
                    values, cache_k, cache_v, state_in,
                )
                hidden[position] = actual_hidden
                state_in = Path(row["output_state"]["path"])
                positions.append(row)
            authenticate(binary_record)
            raw_trace = positions[2]["raw"]["trace"]
            authenticate(raw_trace)
            payload = Path(raw_trace["path"]).read_bytes()
            stages = {
                stage: frontier.trace_stage(
                    payload, 2, stage, count,
                    frontier.PERIODIC_STAGE_INDICES.get(stage),
                ) for stage, count in frontier.STAGE_COUNTS.items()
            }
            comparisons, first_failure = frontier.compare_stages(layer, stages)
            record = {
                "layer_index": layer, "positions": positions,
                "live_binary": binary_record, "independent_comparisons": comparisons,
                "actual_output_fed_rtl_chain": True,
                "elapsed_seconds": time.monotonic() - started,
            }
            write(layer_root / "result.json", record)
            layers.append(frontier.file_record(layer_root / "result.json"))
            print(json.dumps({
                "completed_layer": layer,
                "material_mismatches": sum(row["failure_count"] for row in comparisons),
                "elapsed_seconds": record["elapsed_seconds"],
            }), flush=True)
            if first_failure is not None:
                break
    for item in frozen["sources"]:
        authenticate(item)
    result = {
        "status": ("ZERO_MATERIAL_MISMATCHES_THROUGH_LAYER21" if first_failure is None
                   else "EARLIEST_MATERIAL_MISMATCH_LOCALIZED"),
        "layers": layers, "reached_layer": len(layers) - 1,
        "actual_output_fed_rtl_chain": True,
        "current_source_bindings_unchanged": True,
        "earliest_material_mismatch": first_failure,
        "layer08_stage08_repaired": (
            len(layers) > 8 and (
                first_failure is None or first_failure["layer_index"] > 8
                or (first_failure["layer_index"] == 8 and first_failure["stage"] > 8)
            )
        ),
        "natural_terminal_transactions": 3 * len(layers),
        "independent_review": "pending Host Reviewer",
        "boundary": "Computer-local W4A16 RTL decoder frontier only; no tail or hardware claim",
    }
    if first_failure is not None:
        result.update(
            taxonomy="numerical_policy_material_mismatch",
            hypothesis="Fresh RTL stage propagation diverges from the independently propagated policy",
            regression="Reproduce the recorded earliest layer/stage under unchanged policy in a fresh attempt",
        )
    return result


def seal(root: Path, result: dict) -> None:
    write(root / "result.json", result)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    write(root / "seal.json", {"files": [frontier.file_record(path) for path in files]})
    for path in [*files, root / "seal.json"]:
        path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "execute"))
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.attempt_dir.absolute()
    frontier.require(root.parent == ROOT / "build" and re.fullmatch(
        r"model24_persistent_kv_selected_token_corrected_q_attempt[0-9]{3}", root.name
    ) is not None, "attempt outside corrected-Q namespace")
    if args.operation == "prepare":
        frontier.require(not root.exists() and not root.is_symlink(), "attempt exists")
    try:
        if args.operation == "prepare":
            prepare(root)
            return 0
        result = execute(root)
    except (RuntimeError, ValueError, OSError, AssertionError) as error:
        result = {
            "status": ("PREPARATION_FAILED" if args.operation == "prepare"
                       else "EXECUTION_FAILED"),
            "error": str(error),
            "exception_type": type(error).__name__,
            "taxonomy": "execution_or_binding_failure",
            "hypothesis": "The recorded command, exact runtime comparison, or bound input failed",
            "regression": "Inspect the preserved failing boundary before any fresh repair attempt",
            "rtl_correctness_conclusion": "none; inspect completed per-layer evidence",
            "independent_review": "pending Host Reviewer",
        }
        if root.exists() and not (root / "result.json").exists():
            seal(root, result)
        print(json.dumps(result), file=sys.stderr)
        return 1
    seal(root, result)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
