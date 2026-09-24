#!/usr/bin/env python3
"""Execute the reviewed corrected-Q layer23 continuation exactly once."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path("/home/argustest/ace3-argus")
PREPARATION = ROOT / "build/layer22_23_continuation_attempt003"
HERE = ROOT / "build/layer23_runtime_launch_attempt002"
SOURCE = Path(__file__).resolve()
TASK_ID = "ace3-layer23-0ffba022ad92-attempt002"
REVIEW = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "0ffba022ad92/round-0003.json"
)
CHECKPOINT = Path(
    "/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/"
    "0ffba022ad92/CHECKPOINT.md"
)
sys.path.insert(0, str(PREPARATION))
import prepare as binding
import validate_selected_token_position2_traversal as traversal


def execute():
    manifest = binding.validate()
    frozen = binding.load(HERE / "frozen.json")
    for key in ("source", "review", "predecessor", "predecessor_seal"):
        binding.authenticate(frozen[key])
    review = binding.load(REVIEW)
    # "continue" refers to the two-layer mission, not rejection of layer22.
    binding.require(
        review["producer_role"] == "reviewer"
        and review["mission_id"] == "0ffba022ad92"
        and review["round"] == 3
        and review["review"]["status"] == "continue"
        and review["review"]["reason"].startswith("Independent review admits layer22:")
        and review["review"]["next_action"].startswith(
            "Launch a fresh immutable layer23 attempt from the admitted layer22 actual RTL outputs"
        )
        and review["checkpoint"]["path"] == str(CHECKPOINT),
        "missing explicit Host admission of layer22",
    )
    predecessor_path = binding.OUTPUT / "layer22/result.json"
    predecessor = binding.load(predecessor_path)
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    binding.require(
        binding.record(PREPARATION / "seal.json")["sha256"] in checkpoint
        and binding.record(predecessor_path)["sha256"] in checkpoint,
        "Host checkpoint does not bind preparation and layer22",
    )
    binding.authenticate_tree(binding.load(
        ROOT / "build/layer22_runtime_launch_attempt003/seal.json"
    ))
    binding.authenticate_tree(predecessor)
    execution = binding.load(binding.OUTPUT / "layer22/execution.json")
    binding.require(
        execution["preparation_seal"] == binding.record(PREPARATION / "seal.json")
        and execution["predecessor"] == binding.record(binding.PARENT / "layer21/result.json")
        and predecessor["preparation_seal"] == execution["preparation_seal"],
        "layer22 preparation or parent binding drift",
    )
    parent = binding.load(binding.PARENT / "layer21/result.json")
    binding.require(
        predecessor["layer_index"] == 22
        and predecessor["actual_output_fed_rtl_chain"] is True
        and [tx["position"] for tx in predecessor["positions"]] == [0, 1, 2]
        and len(predecessor["independent_comparisons"]) == 19
        and all(row["stage"] == stage and row["failure_count"] == 0
                for stage, row in enumerate(predecessor["independent_comparisons"]))
        and predecessor["earliest_material_mismatch"] is None,
        "layer22 has not cleared the ordered independent oracle",
    )
    for position, tx in enumerate(predecessor["positions"]):
        binding.require(
            tx["output"]["path"] ==
            str(binding.OUTPUT / f"layer22/position{position:03d}/raw/final.hex")
            and tx["input"]["sha256"] ==
            parent["positions"][position]["output"]["semantic_sha256"],
            "layer22 did not consume actual layer21 outputs",
        )
    layer_root = binding.OUTPUT / "layer23"
    binding.require(
        not layer_root.exists() and not (binding.OUTPUT / "compiled/layer23").exists(),
        "layer23 output or binary already exists",
    )
    import numpy as np
    from safetensors import safe_open
    import model24_persistent_kv_runtime as persistent
    import run_corrected_q_selected_token_frontier as current
    import run_repaired_q_selected_token_frontier as frontier

    hidden = [np.asarray(binding.hidden_bits(tx), dtype="<u2")
              for tx in predecessor["positions"]]
    traversal.run_logged = current.logged
    with persistent.configured_transaction_engine(), safe_open(
        persistent.CHECKPOINT, framework="np"
    ) as model:
        binding.require(
            binding.tensor_catalog(model) == manifest["canonical_tensors"],
            "canonical layer tensor drift",
        )
        layer_root.mkdir()
        binding.write(layer_root / "execution.json", {
            "argv": [sys.executable, "-B", *sys.argv],
            "source": binding.record(SOURCE),
            "review": binding.record(REVIEW),
            "review_scope": "Explicit layer22 admission within a continuing two-layer mission",
            "predecessor": binding.record(predecessor_path),
            "preparation_seal": binding.record(PREPARATION / "seal.json"),
            "layer": 23,
        })
        tensors = layer_root / "vectors"
        vectors = traversal.materialize_transaction_vectors(model, 23, 2, hidden[2], tensors)
        inputs = [
            traversal.materialize_input_vectors(
                23, position, hidden[position], layer_root / f"position{position:03d}/vectors"
            )
            for position in (0, 1)
        ] + [vectors]
        current.logged(binding.build_command(23), layer_root / "compile.log")
        binary = binding.OUTPUT / f"compiled/layer23/obj_dir/V{binding.TOP}"
        binary_record = binding.record(binary)
        values = traversal.layer_oracle_values(model, 23, vectors)
        cache_k, cache_v, positions = [], [], []
        state_in = None
        for position in (0, 1, 2):
            row, actual = traversal.execute_exact_transaction(
                binary, 23, position, hidden[position], inputs[position], tensors,
                layer_root / f"position{position:03d}",
                layer_root / f"position{position + 1:03d}.state",
                values, cache_k, cache_v, state_in,
            )
            hidden[position] = actual
            state_in = Path(row["output_state"]["path"])
            positions.append(row)
        binding.authenticate(binary_record)
        raw = positions[2]["raw"]["trace"]
        binding.authenticate(raw)
        payload = Path(raw["path"]).read_bytes()
        stages = {
            stage: frontier.trace_stage(
                payload, 2, stage, count, frontier.PERIODIC_STAGE_INDICES.get(stage)
            )
            for stage, count in frontier.STAGE_COUNTS.items()
        }
        comparisons, failure = frontier.compare_stages(23, stages)
        binding.write(layer_root / "result.json", {
            "layer_index": 23,
            "positions": positions,
            "live_binary": binary_record,
            "preparation_seal": binding.record(PREPARATION / "seal.json"),
            "independent_comparisons": comparisons,
            "actual_output_fed_rtl_chain": True,
            "earliest_material_mismatch": failure,
            "review_required_before_next_layer": True,
        })
        binding.authenticate_tree(manifest["sources"])
        binding.require(failure is None, "material mismatch; preserve attempt and stop")
    print("LAYER23_COMPARISON_COMPLETE_HOST_REVIEW_REQUIRED", flush=True)


def main():
    binding.require(
        not (binding.OUTPUT / "layer23").exists()
        and not (binding.OUTPUT / "compiled/layer23").exists(),
        "layer23 attempt already exists",
    )
    HERE.mkdir()
    manifest = binding.load(PREPARATION / "continuation.json")
    environment = os.environ.copy()
    changes = {}
    for key, expected in manifest["current_candidate"]["build_environment"].items():
        if environment.get(key) != expected:
            changes[key] = {"inherited": environment.get(key), "accepted": expected}
        if expected is None:
            environment.pop(key, None)
        else:
            environment[key] = expected
    changes["PYTHONSAFEPATH"] = {
        "inherited": environment.pop("PYTHONSAFEPATH", None),
        "accepted": None,
        "reason": "The admitted preparation imports trusted sibling modules.",
    }
    command = [sys.executable, "-B", str(SOURCE), "--execute"]
    binding.write(HERE / "frozen.json", {
        "attempt_id": HERE.name,
        "task_id": TASK_ID,
        "source": binding.record(SOURCE),
        "preparation_seal": binding.record(PREPARATION / "seal.json"),
        "preparation_manifest": binding.record(PREPARATION / "continuation.json"),
        "admitted_recipe": binding.record(
            ROOT / "build/layer22_23_continuation_attempt002/layer_recipe.py"
        ),
        "review_gate_change": (
            "Bind the exact round3 explicit layer22 admission, whose mission status "
            "is continue; do not require completion of unexecuted layer23."
        ),
        "review": binding.record(REVIEW),
        "predecessor": binding.record(binding.OUTPUT / "layer22/result.json"),
        "predecessor_seal": binding.record(
            ROOT / "build/layer22_runtime_launch_attempt003/seal.json"
        ),
        "prior_no_execution": binding.record(
            ROOT / "build/layer23_runtime_launch_attempt001/seal.json"
        ),
        "argv": command,
        "cwd": str(ROOT),
        "environment_changes": changes,
        "accepted_build_environment": manifest["current_candidate"]["build_environment"],
        "started_at": time.time(),
    })
    started = time.monotonic()
    with (HERE / "stdout.log").open("xb") as stdout, (HERE / "stderr.log").open("xb") as stderr:
        child = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False
        )
    binding.write(HERE / "exit.json", {
        "exit_code": child.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": binding.record(HERE / "stdout.log"),
        "stderr": binding.record(HERE / "stderr.log"),
    })
    layer_root = binding.OUTPUT / "layer23"
    layer_path = layer_root / "result.json"
    summary = {
        "attempt_id": HERE.name,
        "independent_review": "pending normal Host Reviewer",
        "command_receipt": binding.record(HERE / "frozen.json"),
        "exit_receipt": binding.record(HERE / "exit.json"),
        "boundary": "Computer-local W4A16 RTL only; no tail, dialogue or hardware claim",
    }
    if layer_path.exists():
        result = binding.load(layer_path)
        binding.authenticate_tree(result)
        parent = binding.load(binding.OUTPUT / "layer22/result.json")
        binding.require(
            result["layer_index"] == 23
            and result["actual_output_fed_rtl_chain"] is True
            and [row["position"] for row in result["positions"]] == [0, 1, 2],
            "layer23 identity mismatch",
        )
        for position, row in enumerate(result["positions"]):
            receipt = binding.load(
                layer_root / f"position{position:03d}/simulation.log.exit.json"
            )
            binding.require(receipt["exit_code"] == 0, "simulator did not exit zero")
            traversal.parse_natural_terminal(Path(row["raw"]["terminal"]["path"]), 23, position)
            binding.require(
                row["input"]["sha256"] == parent["positions"][position]["output"]["semantic_sha256"],
                "layer23 actual-output parent mismatch",
            )
        comparisons = result["independent_comparisons"]
        binding.require(
            [row["stage"] for row in comparisons] == list(range(19)),
            "independent stage coverage mismatch",
        )
        first = next((row for row in comparisons if row["failure_count"]), None)
        binding.require(
            first == result["earliest_material_mismatch"]
            and (child.returncode == 0 if first is None else child.returncode != 0),
            "executor status disagrees with independent oracle",
        )
        summary.update(
            status=("LAYER23_ZERO_MATERIAL_MISMATCHES_REVIEW_REQUIRED" if first is None
                    else "EARLIEST_MATERIAL_MISMATCH_LOCALIZED"),
            layer_result=binding.record(layer_path),
            actual_output_fed_rtl_chain=True,
            natural_terminal_transactions=3,
            material_mismatches=sum(row["failure_count"] for row in comparisons),
            earliest_material_mismatch=first,
        )
        if first is not None:
            summary.update(
                taxonomy="numerical_policy_material_mismatch",
                root_cause_hypothesis=(
                    "Actual RTL differs from independent accepted-policy propagation; "
                    "the causal origin is not yet localized."
                ),
                regression="Reproduce the sealed earliest layer23 stage in a fresh attempt.",
            )
    else:
        binding.require(child.returncode != 0, "successful executor omitted its result")
        summary.update(
            status="EXECUTION_FAILED_WITHOUT_LAYER_RESULT",
            taxonomy="executor_failure_before_comparison_completion",
            root_cause_hypothesis="Inspect preserved executor stderr and partial artifacts.",
            regression="Resolve the recorded failing boundary before a fresh attempt.",
            rtl_correctness_conclusion="none",
        )
    binding.write(HERE / "result.json", summary)
    directories = (HERE, layer_root, binding.OUTPUT / "compiled/layer23")
    paths = sorted(
        path for directory in directories if directory.exists()
        for path in directory.rglob("*") if path.is_file()
    )
    binding.write(HERE / "seal.json", {"files": [binding.record(path) for path in paths]})
    for path in [*paths, HERE / "seal.json"]:
        path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    for directory in directories:
        if directory.exists():
            for path in sorted(directory.rglob("*"), reverse=True):
                if path.is_dir():
                    path.chmod(0o555)
            directory.chmod(0o555)
    print(summary["status"], flush=True)
    return child.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        raise SystemExit(main())
