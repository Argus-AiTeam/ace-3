#!/usr/bin/env python3
"""Continue from sealed layer07 without replaying accepted decoder prefixes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "build/token358_position3_full_continuation_attempt001"
PREVIOUS = ROOT / "build/token358_position3_full_continuation_attempt003"
COMPLETION = ROOT / "build/token358_position3_full_continuation_attempt003_completion/completion.json"
DIAGNOSIS = PREVIOUS / "score_diagnostic004"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text("ascii"))


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def authenticate(item):
    actual = record(item["path"])
    require(all(actual[key] == item[key] for key in actual),
            f"binding drift: {item['path']}")


def write(path, value):
    with Path(path).open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def trusted_runner(out):
    completion = load(COMPLETION)
    authenticate(completion["attempt_seal"])
    authenticate(completion["attempt_result"])
    for item in load(PREVIOUS / "seal.json")["files"]:
        authenticate(item)
    inherited = load(PREVIOUS / "frozen.json")
    authenticate(inherited["shared_runner"])
    require(inherited["shared_runner"]["path"] == str(ORIGINAL / "execute.py"),
            "unexpected shared runner")
    spec = importlib.util.spec_from_file_location(
        "token358_bound_continuation", inherited["shared_runner"]["path"])
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = out
    return runner


def execute(out, runner):
    frozen = load(ORIGINAL / "frozen.json")
    for item in [*load(ORIGINAL / "seal.json")["files"], *runner.all_records(frozen)]:
        authenticate(item)
    require(Path(sys.executable).resolve() ==
            Path(frozen["preflight"]["python_executable"]["path"]).resolve(),
            "interpreter drift")
    package = load(runner.PACKAGE / "launch_package.json")
    require(package["layer_order"] == list(range(2, 24))
            and package["position"] == 3
            and package["token_history"] == [9707, 1879, 0, 358],
            "public execution contract drift")
    review = load(runner.REVIEW)
    require(review["producer_role"] == "reviewer"
            and review["review"]["status"] == "done"
            and review["mission_id"] == "2d24ed43c02e", "prerequisite review mismatch")
    for layer in range(3, 8):
        accepted = load(PREVIOUS / f"layer{layer:02d}/result.json")
        require(accepted["layer"] == layer and accepted["position"] == 3
                and accepted["status"] == "ZERO_MATERIAL_MISMATCHES"
                and accepted["material_mismatches"] == 0
                and len(accepted["comparisons"]) == 19
                and all(row["failure_count"] == 0 for row in accepted["comparisons"])
                and all(row["exact_match"] for row in accepted["exact"].values())
                and accepted["kv_valid_positions"] == [0, 1, 2, 3],
                f"layer{layer:02d} is not a passing numerical boundary")
    diagnosis = load(DIAGNOSIS / "result.json")
    require(diagnosis["fresh_rtl"]["all_match_binary64_dots"]
            and diagnosis["fresh_rtl"]["baseline_matches_preserved_trace"]
            and diagnosis["controls"]["baseline"]["failure_indices"] == [13, 15]
            and diagnosis["controls"]["independent_norm_same_hidden_q_only"]["failure_indices"]
            == [13, 15]
            and diagnosis["controls"]["independent_q_only"]["failure_indices"] == [],
            "diagnosis does not justify unchanged layer08 arithmetic")
    write(out / "frozen.json", {
        "source": record(__file__), "command": record(out / "launch.command.sh"),
        "previous_completion": record(COMPLETION),
        "previous_seal": record(PREVIOUS / "seal.json"),
        "previous_frozen": record(PREVIOUS / "frozen.json"),
        "original_frozen": record(ORIGINAL / "frozen.json"),
        "shared_runner": record(ORIGINAL / "execute.py"),
        "diagnosis": record(DIAGNOSIS / "result.json"),
        "diagnostic_contract": record(DIAGNOSIS / "frozen.json"),
        "diagnostic_explanation": record(DIAGNOSIS / "root_cause.md"),
        "accepted_hidden_boundary": record(PREVIOUS / "layer07/result.json"),
        "recovered_layer2": load(PREVIOUS / "frozen.json")["recovered_layer2"],
        "public_interfaces": frozen["public_interfaces"],
        "parameters": {"LAYER_INDEX": list(range(8, 24)), "ACCURATE_SILU": 1},
        "profile": frozen["profile"], "prompt": frozen["prompt"],
        "score_policy": frozen["score_policy"],
        "comparison_policy": frozen["comparison_policy"],
        "binary64_continuous_driver": frozen["binary64_continuous_driver"],
        "preflight": frozen["preflight"], "layer_order": list(range(8, 24)),
        "rtl_prefix_replays": 0, "numerical_policy_changes": 0,
        "repair_decision": (
            "No layer08 arithmetic or contract edit is justified. Reproduce the "
            "diagnosed inherited-trajectory mismatch in a fresh official attempt; "
            "advance only if the unchanged independent gate passes."
        ),
    })
    sys.path.insert(0, str(ROOT / "ace3/model"))
    import run_token358_position3_layer0 as prior

    np, torch = prior.np, prior.torch
    traversal, frontier = prior.traversal, prior.frontier
    torch.set_num_threads(1)
    require((prior.comparison.ABSOLUTE_TOLERANCE, prior.comparison.RELATIVE_TOLERANCE,
             prior.comparison.MAX_ULP_DISTANCE) == (0.125, 0.001, 1), "policy drift")
    helpers = prior.independent_helpers()
    actual_hidden = traversal.load_hidden_bits(Path(accepted["output_hidden"]["path"]))
    independent_hidden = np.asarray([
        int(row, 16) for row in Path(accepted["independent_hidden"]["path"])
        .read_text("ascii").splitlines()], dtype="<u2")
    require(actual_hidden.shape == independent_hidden.shape == (896,), "hidden geometry")
    previous_hidden = accepted["output_hidden"]
    runner.COMPLETED.append(accepted)
    for tool in ("verilator", "g++", "make"):
        runner.logged([tool, "--version"], out / f"{tool}.version.log")
    with prior.safe_open(package["checkpoint"]["path"], framework="np") as checkpoint:
        for unit in package["layers"]:
            layer = unit["layer_index"]
            if layer < 8:
                continue
            require(layer == runner.COMPLETED[-1]["layer"] + 1, "nonsequential traversal")
            runner.ACTIVE_LAYER = layer
            runner.PHASE = "layer_preparation"
            started = time.monotonic()
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            vector_dir = directory / "vectors"
            parent = unit["kv_parent"]
            authenticate(previous_hidden)
            for item in runner.all_records(unit):
                authenticate(item)
            vectors = traversal.materialize_input_vectors(layer, 3, actual_hidden, vector_dir)
            require(vectors["rope_coefficients"]["sha256"] ==
                    unit["rope_coefficients"]["sha256"], "position3 RoPE drift")
            tensor_dir = vector_dir / "tensors"
            tensor_dir.mkdir()
            tensors = []
            for item in unit["tensors"]:
                destination = tensor_dir / Path(item["serialized"]["path"]).name
                shutil.copyfile(item["serialized"]["path"], destination)
                copied = record(destination)
                require(copied["sha256"] == item["serialized"]["sha256"], "tensor copy drift")
                tensors.append({"checkpoint_tensor": item["checkpoint_tensor"], "serialized": copied})
            manifest = {
                "schema_version": 1, "kind": "ace3_position2_live_transaction_vectors",
                "layer_index": layer, "position": 3,
                "input_activation_sha256": traversal.sha256_bytes(actual_hidden.tobytes()),
                "input": vectors["input"], "rope_coefficients": vectors["rope_coefficients"],
                "tensors": tensors,
            }
            write(vector_dir / "manifest.json", manifest)
            vectors.update(manifest=record(vector_dir / "manifest.json"), tensors=tensors)
            ref_path = runner.REFERENCE / f"recovery001/layer{layer:02d}_generation1.json"
            ref = load(ref_path)
            require(ref["layer"] == layer and ref["history"] == package["token_history"][:3],
                    "independent cache identity drift")
            official = {item["name"]: np.ascontiguousarray(checkpoint.get_tensor(item["name"]))
                        for item in ref["checkpoint_tensor_hashes"]}
            require(len(official) == 26, "checkpoint tensor count")
            for item in ref["checkpoint_tensor_hashes"]:
                require(traversal.sha256_bytes(official[item["name"]].tobytes()) == item["sha256"],
                        "independent checkpoint tensor drift")
            own_k, own_v = [prior.array_input(ref["own_cache"][name]) for name in ("k", "v")]
            require(own_k.shape == own_v.shape == (3, 2, 64), "independent K/V geometry")
            runner.PHASE = "layer_independent_oracle"
            expected, next_k, next_v = helpers["propagated_reference"](
                torch.from_numpy(independent_hidden.view("<f2").astype(np.float64)[None, :]),
                official, layer, cached_k=torch.from_numpy(own_k.view("<f2").astype(np.float64)),
                cached_v=torch.from_numpy(own_v.view("<f2").astype(np.float64)), position_offset=3)
            expected_bits = {stage: np.asarray(value, dtype="<f2").view("<u2")
                             for stage, value in expected[0].items()}
            require(set(expected_bits) == set(range(19)), "independent stage coverage")
            oracle_dir = directory / "independent"
            oracle_dir.mkdir()
            expected_records = {str(stage): prior.comparison.write_hex(
                oracle_dir / f"stage{stage:02d}.hex", bits) for stage, bits in expected_bits.items()}
            for name, cache in (("k", next_k), ("v", next_v)):
                with (oracle_dir / f"position003_own_{name}.npy").open("xb") as stream:
                    np.save(stream, cache.numpy().astype("<f2").view("<u2"), allow_pickle=False)
            runner.PHASE = "layer_local_oracle"
            caches = {
                name: [frontier.trace_stage(Path(item["path"]).read_bytes(), p, stage, 128).tolist()
                       for p, item in enumerate(parent["actual_kv_traces"])]
                for name, stage in (("k", 6), ("v", 7))
            }
            require(all(len(rows) == 3 for rows in caches.values()), "actual K/V parent geometry")
            values = traversal.layer_oracle_values(checkpoint, layer, vectors)
            final, trace = prior.decoder.run_token(
                values, actual_hidden.tolist(), 3,
                [row.copy() for row in caches["k"]], [row.copy() for row in caches["v"]],
                accurate_silu=True)
            require(all(len(rows) == 3 for rows in caches.values()), "K/V mutation regression")
            vectors.update(traversal.materialize_runtime_vector_contract(
                layer, 3, final, trace, vector_dir))
            runner.PHASE = "public_top_compile"
            runner.logged([
                "make", "--no-print-directory", "model24-rtl-layer-compile",
                f"MODEL24_RTL_LAYER_INDEX={layer}", "MODEL24_RTL_ACCURATE_SILU=1",
                f"MODEL24_RTL_CASCADE_DIR={out}",
            ], directory / "compile.log")
            binary = out / f"compiled/layer{layer}/obj_dir/Vace3_decoder_layer0_token_engine"
            layouts = [
                {p.name: record(p) for p in folder.iterdir() if p.suffix in (".cpp", ".h")}
                for folder in (Path(parent["live_binary"]["path"]).parent, binary.parent)
            ]
            compatible = bool(layouts[0]) and layouts[0].keys() == layouts[1].keys() and all(
                layouts[0][name]["sha256"] == layouts[1][name]["sha256"] for name in layouts[0])
            write(directory / "state_abi.json", {
                "parent_binary": parent["live_binary"], "fresh_binary": record(binary),
                "parent_generated_sources": layouts[0], "fresh_generated_sources": layouts[1],
                "same_generated_layout": compatible,
            })
            require(compatible, "fresh compiled serialized-state ABI differs")
            replacements = {
                str(runner.PACKAGE / f"layer{layer:02d}/vectors"): str(vector_dir),
                str(runner.PACKAGE.parent / f"future_execution/layer{layer:02d}/position003/raw"):
                    str(directory / "position003/raw"),
                str(runner.PACKAGE.parent / f"future_execution/layer{layer:02d}/position004.state"):
                    str(directory / "position004.state"),
                parent["live_binary"]["path"]: str(binary),
            }
            expected_argv = [replacements.get(arg, arg) for arg in unit["argv"]]
            binary_record = record(binary)
            write(directory / "prepared.json", {
                "input_hidden": previous_hidden, "kv_parent": parent, "vectors": vectors,
                "independent_parent": record(ref_path), "independent_stages": expected_records,
                "binary": binary_record, "argv": expected_argv,
                "preparation_seconds": time.monotonic() - started,
            })

            def run_once(argv, log_path):
                require(argv == expected_argv, "runtime command differs from reviewed relocation")
                authenticate(parent["state"])
                authenticate(binary_record)
                for item in runner.all_records(vectors):
                    authenticate(item)
                runner.logged(argv, log_path)

            traversal.run_logged = run_once
            runner.PHASE = "layer_rtl_runtime"
            transaction, output_hidden = traversal.execute_transaction(
                binary, layer, 3, actual_hidden, vectors, vector_dir,
                directory / "position003", directory / "position004.state",
                Path(parent["state"]["path"]))
            write(directory / "transaction.json", transaction)
            runner.PHASE = "layer_comparison"
            payload = Path(transaction["raw"]["trace"]["path"]).read_bytes()
            comparisons = []
            for stage in range(19):
                actual = frontier.trace_stage(payload, 3, stage, expected_bits[stage].size,
                                              4 if stage in (8, 9) else None)
                comparisons.append({
                    "stage": stage, "independent_reference": expected_records[str(stage)],
                    **prior.comparison.comparison(actual, expected_bits[stage]),
                })
            exact = {
                "trace": traversal.exact_hex_comparison(
                    payload, Path(vectors["trace"]["path"]).read_bytes(), "trace"),
                "final_hidden": traversal.exact_hex_comparison(
                    Path(transaction["output"]["path"]).read_bytes(),
                    Path(vectors["final_hidden"]["path"]).read_bytes(), "final_hidden"),
            }
            actual_caches = {}
            for name, stage in (("k", 6), ("v", 7)):
                bits = np.asarray([*caches[name], frontier.trace_stage(
                    payload, 3, stage, 128).tolist()], dtype="<u2")
                require(bits.shape == (4, 128), "actual K/V receipt geometry")
                bits = bits.reshape(4, 2, 64)
                path = directory / f"position003_actual_{name}.npy"
                with path.open("xb") as stream:
                    np.save(stream, bits, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                actual_caches[name] = {**record(path), "shape": [4, 2, 64],
                                      "semantic_sha256": traversal.sha256_bytes(bits.tobytes())}
            first = next((row for row in comparisons if row["failure_count"]), None)
            receipt = {
                "layer": layer, "position": 3, "transaction": record(directory / "transaction.json"),
                "output_hidden": transaction["output"], "output_state": transaction["output_state"],
                "independent_hidden": expected_records["18"], "kv_parent": parent,
                "actual_kv": actual_caches, "kv_valid_positions": [0, 1, 2, 3],
                "comparisons": comparisons, "exact": exact,
                "material_mismatches": sum(row["failure_count"] for row in comparisons),
                "earliest_material_mismatch": first,
                "elapsed_seconds": time.monotonic() - started,
                "status": "MATERIAL_MISMATCH" if first else
                          "ZERO_MATERIAL_MISMATCHES" if all(x["exact_match"] for x in exact.values())
                          else "LOCAL_EXACT_ORACLE_MISMATCH",
            }
            write(directory / "result.json", receipt)
            write(directory / "seal.json", {
                "files": [record(p) for p in sorted(directory.rglob("*")) if p.is_file()]})
            runner.COMPLETED.append(receipt)
            print(json.dumps({"layer": layer, "status": receipt["status"],
                              "material_mismatches": receipt["material_mismatches"]}), flush=True)
            if first:
                result = runner.mismatch(first["stage"], comparisons,
                                         layer_result=record(directory / "result.json"))
                if layer == 8 and first["stage"] == 8:
                    result.update(
                        diagnosis=record(DIAGNOSIS / "result.json"),
                        root_cause_hypothesis=(
                            "Incoming hidden-trajectory drift is amplified by Q and head3 QK. "
                            "The upstream operation creating the drift remains unlocalized; "
                            "no layer08 arithmetic edit is justified by the controlled diagnosis."),
                        regression=(
                            "Preserve this fresh layer08 failure and its exact-local agreement. "
                            "Localize the upstream producer using sealed intermediates; then "
                            "rebuild only an explicitly affected compatible prefix and rerun "
                            "elements13/15 with unchanged policy and independent K/V."),
                    )
                return result
            require(all(row["exact_match"] for row in exact.values()), "local exact oracle mismatch")
            actual_hidden, independent_hidden = output_hidden, expected_bits[18].reshape(896)
            previous_hidden = transaction["output"]
    return runner.tail_run(package, actual_hidden, independent_hidden, prior)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--close", type=int, metavar="PROCESS_EXIT_CODE")
    args = parser.parse_args()
    out = args.attempt_dir.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("token358_position3_full_continuation_attempt004"),
            "attempt outside authorized namespace")
    if args.close is not None:
        result = load(out / "result.json")
        expected_code = 0 if result["status"] == "ZERO_MATERIAL_MISMATCHES_LAYERS02_23_AND_TAIL" else 2
        require(args.close == expected_code, "process exit differs from result")
        write(out / "completion.json", {
            "process_exit_code": args.close, "result": record(out / "result.json"),
            "seal": record(out / "seal.json"), "closed_run_log": record(out / "run.log"),
            "source": record(__file__), "command": record(out / "launch.command.sh"),
            "semantic_comparison_replays": 0, "independent_review": "pending normal Host Reviewer",
        })
        print(json.dumps({"status": result["status"], "completion": str(out / "completion.json")}))
        return 0
    require(out.is_dir() and {p.name for p in out.iterdir()} <= {
        "launch.command.sh", "submit.command.sh", "wait.command.sh",
        "submit.json", "run.log", "durable_terminal.json",
    }, "attempt is not fresh")
    write(out / "execution_started.json", {
        "time": time.time(), "pid": os.getpid(), "argv": sys.argv,
        "python": sys.version, "cwd": str(ROOT), "prefix_rtl_replays": 0,
    })
    runner = None
    started = time.monotonic()
    try:
        runner = trusted_runner(out)
        result = execute(out, runner)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, ImportError) as error:
        result = {
            "status": "EXECUTION_OR_EVIDENCE_BLOCKER" if runner and runner.ACTIVE_LAYER is not None
                      and runner.PHASE in ("layer_rtl_runtime", "layer_comparison")
                      else "EVALUATOR_NO_EXECUTION",
            "phase": runner.PHASE if runner else "authentication",
            "layer": runner.ACTIVE_LAYER if runner else None,
            "error": str(error), "traceback": traceback.format_exc(),
            "failure_taxonomy": "execution_or_evidence_contract",
            "root_cause_hypothesis": "The preserved exception identifies a prerequisite boundary, not RTL correctness.",
            "regression": "Resolve only the named blocker in a fresh attempt; preserve accepted prefixes.",
        }
    result.update(
        elapsed_seconds=time.monotonic() - started,
        invocations=runner.INVOCATIONS if runner else [],
        completed_layer_receipts=[
            record(out / f"layer{row['layer']:02d}/result.json")
            for row in runner.COMPLETED if row["layer"] >= 8] if runner else [],
        prefix_rtl_replays=0, token_history=[9707, 1879, 0, 358],
        independent_review="pending normal Host Reviewer",
        boundary="Computer-local W4A16 position3 only; no third-token, dialogue or hardware claim",
    )
    write(out / "result.json", result)
    write(out / "seal.json", {
        "files": [record(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name not in {
                      "submit.json", "run.log", "durable_terminal.json"}]})
    print(json.dumps({"status": result["status"], "result": str(out / "result.json")}), flush=True)
    return 0 if result["status"] == "ZERO_MATERIAL_MISMATCHES_LAYERS02_23_AND_TAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
