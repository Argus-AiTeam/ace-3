#!/usr/bin/env python3
"""Read-only layer08 operand diagnosis; write only a fresh diagnostic attempt."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from accepted_awq_projection import project_fp16_stage
from diagnose_layer01_position2_stage08 import (
    authenticated_array,
    comparison,
    file_record,
    load_hex,
    load_json,
    reference_rope,
    require,
    rtl_rope,
    score_matrix,
    sha256_bytes,
)
from layer3_token0_diagnostic import decode_stage_records
from official_model24_next_token import _torch_rmsnorm
from qwen2_rope_oracle import qwen2_coefficient, rotate_pair
from validate_selected_token_position2_traversal import load_hidden_bits


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = ROOT / "build/model24_persistent_kv_selected_token_corrected_q_attempt002"
REFERENCE = ROOT / "build/independent_fp16_trajectory_20260906_1133/recovery001"
DEFAULT_OUTPUT = ROOT / "build/layer08_stage08_diagnostic_attempt001"


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def values(bits: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype="<u2").view("<f2").astype(np.float64)


def bits(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype="<f2").reshape(-1).view("<u2")


def detail(actual: np.ndarray, expected: np.ndarray) -> dict:
    result = comparison(actual.reshape(-1), expected.reshape(-1))
    differences = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    result["different_count"] = int(differences.size)
    result["first_difference"] = None
    if differences.size:
        index = int(differences[0])
        result["first_difference"] = {
            "index": index,
            "actual_bits": f"{int(actual.reshape(-1)[index]):04x}",
            "expected_bits": f"{int(expected.reshape(-1)[index]):04x}",
            "actual": float(values(actual).reshape(-1)[index]),
            "expected": float(values(expected).reshape(-1)[index]),
        }
    return result


def run(output: Path) -> dict:
    require(output.resolve().is_relative_to(ROOT / "build"),
            "diagnostic output must be inside this worktree's build directory")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    authenticated: dict[str, dict] = {}

    def authenticate(record: dict) -> Path:
        path = Path(record["path"])
        actual = file_record(path)
        require(actual["sha256"] == record["sha256"]
                and actual["bytes"] == record["bytes"],
                f"artifact binding mismatch: {path}")
        authenticated[str(path)] = actual
        return path

    execution = load_json(ATTEMPT / "execution.json")
    frozen = load_json(authenticate(execution["frozen"]))
    frozen_refs = {record["path"]: record
                   for record in frozen["independent_references"]}
    layer_result = load_json(ATTEMPT / "layer08/result.json")
    require(layer_result["layer_index"] == 8, "wrong layer result")
    require([p["position"] for p in layer_result["positions"]] == [0, 1, 2],
            "wrong position history")
    original_results = file_record(ATTEMPT / "layer08/result.json")
    authenticated[original_results["path"]] = original_results
    record_path = REFERENCE / "layer08_generation1.json"
    reference = load_json(authenticate(frozen_refs[str(record_path)]))
    reference_stages = {}
    for stage in range(9):
        suffix = f"/layer08/position002/stage{stage:02d}.npy"
        matches = [r for r in reference["stages"] if r["path"].endswith(suffix)]
        require(len(matches) == 1, f"ambiguous independent stage {stage}")
        reference_stages[stage] = authenticated_array(matches[0]).reshape(-1)
        authenticated[matches[0]["path"]] = file_record(Path(matches[0]["path"]))
    expected_k = authenticated_array(reference["own_cache"]["k"])
    authenticated[reference["own_cache"]["k"]["path"]] = file_record(
        Path(reference["own_cache"]["k"]["path"]))
    require(expected_k.shape == (3, 2, 64), "independent K geometry")
    stages: dict[int, dict[int, np.ndarray]] = {}
    for position, transaction in enumerate(layer_result["positions"]):
        terminal_path = authenticate(transaction["raw"]["terminal"])
        fields = terminal_path.read_text(encoding="ascii").split()
        terminal = dict(field.split("=", 1) for field in fields)
        require(len(terminal) == len(fields), "duplicate terminal field")
        require(terminal == {
            "schema": "ace3_decoder_token_transaction_v1",
            "layer_index": "8", "position": str(position),
            "natural_terminal": "1", "exit_code": "0",
            "trace_count": str(transaction["raw"]["trace_count"]),
            "final_count": "896", "done_count": "1",
        }, "not a naturally completed layer08 transaction")
        trace_path = authenticate(transaction["raw"]["trace"])
        records: dict[int, list[tuple[int, int]]] = {}
        rows = trace_path.read_text(encoding="ascii").splitlines()
        require(len(rows) == transaction["raw"]["trace_count"],
                "trace row count mismatch")
        for row in rows:
            require(len(row) == 16 and int(row[:2], 16) == 0
                    and int(row[2:6], 16) == position,
                    "trace owner/width mismatch")
            stage = int(row[6:8], 16)
            if stage <= 8:
                records.setdefault(stage, []).append(
                    (int(row[8:12], 16), int(row[12:], 16)))
        stages[position] = {
            stage: decode_stage_records(records[stage], stage, position)
            for stage in range(9)
        }
        require(np.array_equal(stages[position][5], stages[position][6]),
                "RoPE K / cache-write stage interface mismatch")
    actual = stages[2]
    actual_k = np.stack([stages[p][6].reshape(2, 64) for p in range(3)])

    # Authenticate serialized projection operands against checkpoint tensor bytes.
    transaction = layer_result["positions"][2]
    tensor_values = {}
    tensor_records = []
    for record in transaction["vectors"]["tensors"]:
        checkpoint = record["checkpoint_tensor"]
        name = checkpoint["name"]
        if not any(part in name for part in
                   ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj")):
            continue
        path = authenticate(record["serialized"])
        data = load_hex(path, 4 if checkpoint["dtype"] == "F16" else 8)
        require(data.nbytes == checkpoint["bytes"]
                and sha256_bytes(data.tobytes()) == checkpoint["sha256"],
                f"checkpoint/serialization mismatch: {name}")
        tensor_values[name] = data
        tensor_records.append(record)
    input_path = authenticate(transaction["vectors"]["input"])
    input_rows = input_path.read_text(encoding="ascii").splitlines()
    require(len(input_rows) == 896
            and all(len(row) == 10 and int(row[:2], 16) == 0
                    and int(row[2:6], 16) == index
                    for index, row in enumerate(input_rows)),
            "hidden-input serialization mismatch")
    hidden = np.asarray([int(row[6:], 16) for row in input_rows], dtype="<u2")
    require(sha256_bytes(hidden.tobytes()) == transaction["input"]["sha256"],
            "hidden input semantic binding mismatch")
    upstream_result = load_json(authenticate(file_record(
        ATTEMPT / "layer07/result.json")))
    require(upstream_result["layer_index"] == 7
            and [p["position"] for p in upstream_result["positions"]] == [0, 1, 2],
            "wrong layer07 handoff history")
    upstream = upstream_result["positions"][2]
    upstream_path = authenticate(upstream["output"])
    require(upstream_path == ATTEMPT / "layer07/position002/raw/final.hex",
            "layer07 handoff is not the position2 raw final")
    upstream_hidden = load_hidden_bits(upstream_path)
    require(sha256_bytes(upstream_hidden.tobytes())
            == upstream["output"]["semantic_sha256"]
            == transaction["input"]["sha256"],
            "layer07 final / layer08 input semantic binding mismatch")
    require(np.array_equal(upstream_hidden, hidden),
            "layer07 raw final differs from layer08 serialized input")
    require(upstream_path.read_bytes() == input_path.read_bytes(),
            "layer07 raw final / layer08 input serialization differs")
    handoff = {
        "producer": authenticated[str(upstream_path)],
        "consumer": authenticated[str(input_path)],
        "position": 2, "elements": 896, "dtype": "little-endian FP16 bits",
        "exact_value_matches": int(np.sum(upstream_hidden == hidden)),
        "serialized_bytes_equal": True,
        "semantic_binding_equal": True,
        "comparison": detail(upstream_hidden, hidden),
    }
    predecessor_path = REFERENCE / "layer07_generation1.json"
    predecessor = load_json(authenticate(frozen_refs[str(predecessor_path)]))
    predecessor_rows = [r for r in predecessor["stages"]
                        if r["path"].endswith("/layer07/position002/stage18.npy")]
    require(len(predecessor_rows) == 1, "independent incoming hidden ambiguous")
    expected_hidden = authenticated_array(predecessor_rows[0]).reshape(-1)
    authenticated[predecessor_rows[0]["path"]] = file_record(
        Path(predecessor_rows[0]["path"]))
    gamma = tensor_values["model.layers.8.input_layernorm.weight"]

    def norm(activation: np.ndarray) -> np.ndarray:
        result = _torch_rmsnorm(
            torch.from_numpy(values(activation)[None, :]), values(gamma))
        return bits(result.detach().cpu().numpy())

    def project(activation: np.ndarray, kind: str) -> np.ndarray:
        prefix = f"model.layers.8.self_attn.{kind}_proj"
        return project_fp16_stage(
            activation, tensor_values[prefix + ".qweight"],
            tensor_values[prefix + ".qzeros"], tensor_values[prefix + ".scales"],
            896 if kind == "q" else 128, tensor_values[prefix + ".bias"])

    accepted_norm_actual_hidden = norm(hidden)
    require(np.array_equal(norm(expected_hidden), reference_stages[0]),
            "independent RMSNorm recomputation does not reproduce accepted stage00")
    accepted_q_actual_norm = project(actual[0], "q")
    require(np.array_equal(accepted_norm_actual_hidden, actual[0])
            and np.array_equal(accepted_q_actual_norm, actual[1]),
            "local RMSNorm or Q projection introduces additional drift")
    accepted_q_reference_norm = project(reference_stages[0], "q")
    require(np.array_equal(accepted_q_reference_norm, reference_stages[1]),
            "independent AWQ Q projection does not reproduce accepted stage01")
    require(np.array_equal(reference_rope(reference_stages[1]), reference_stages[4]),
            "independent RoPE does not reproduce accepted stage04")
    require(np.array_equal(rtl_rope(actual[1]), actual[4]),
            "current RoPE operand policy does not reproduce preserved stage04")
    require(np.array_equal(score_matrix(actual[4], actual_k), actual[8]),
            "current score arithmetic does not reproduce preserved stage08")

    def independent_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:
        q_values = values(q).reshape(14, 64)
        k_values = values(k)
        return bits(np.stack([k_values[:, head // 7] @ q_values[head] / 8.0
                              for head in range(14)]))

    def rotate_k(projection: np.ndarray, position: int) -> np.ndarray:
        rotated = np.empty_like(projection)
        for head in range(2):
            for pair in range(32):
                low = head * 64 + pair
                high = low + 32
                cosine, sine = qwen2_coefficient(position, pair)
                a, b, invalid, saturation = rotate_pair(
                    int(projection[low]), int(projection[high]), cosine, sine)
                require(not invalid and not saturation, "K RoPE numeric flags")
                rotated[low], rotated[high] = a, b
        return rotated.reshape(2, 64)

    require(np.array_equal(independent_scores(reference_stages[4], expected_k),
                           reference_stages[8]),
            "binary64 Q/K dot does not reproduce accepted stage08")

    cases = {
        "preserved_q_preserved_k": (actual[4], actual_k),
        "reference_q_preserved_k": (reference_stages[4], actual_k),
        "preserved_q_reference_k": (actual[4], expected_k),
        "reference_q_reference_k": (reference_stages[4], expected_k),
        "accepted_q_projection_on_actual_norm": (
            rtl_rope(accepted_q_actual_norm), actual_k),
        "accepted_rope_on_actual_q_projection": (reference_rope(actual[1]), actual_k),
        "reference_norm_and_projection_rtl_rope": (
            rtl_rope(accepted_q_reference_norm), actual_k),
        "accepted_norm_on_actual_hidden_q_projection_rtl_rope": (
            rtl_rope(project(accepted_norm_actual_hidden, "q")), actual_k),
        "accepted_local_q_on_actual_hidden": (
            reference_rope(project(accepted_norm_actual_hidden, "q")), actual_k),
    }
    for position in range(3):
        replaced = actual_k.copy()
        replaced[position] = expected_k[position]
        cases[f"reference_k_position{position}_only"] = (actual[4], replaced)
        require(np.array_equal(rotate_k(stages[position][2], position),
                               actual_k[position]),
                f"current K RoPE does not reproduce cache position {position}")
        accepted_projection = project(stages[position][0], "k")
        replaced_projection = actual_k.copy()
        replaced_projection[position] = rotate_k(accepted_projection, position)
        cases[f"accepted_k_projection_policy_position{position}_only"] = (
            actual[4], replaced_projection)
    controlled = {}
    for name, (q, k) in cases.items():
        result = score_matrix(q, k)
        controlled[name] = {
            **detail(result, reference_stages[8]),
            "score_bits": [f"{int(value):04x}" for value in result],
        }
    stage_comparisons = {str(stage): detail(actual[stage], reference_stages[stage])
                         for stage in range(9)}
    require(all(stage_comparisons[str(s)]["failure_count"] == 0 for s in range(8)),
            "preserved first material divergence is earlier than stage08")
    require(stage_comparisons["8"]["failure_count"] == 3,
            "preserved layer08 failure count changed")
    require(controlled["reference_q_preserved_k"]["failure_count"] == 0
            and controlled["reference_norm_and_projection_rtl_rope"]["failure_count"] == 0
            and controlled["preserved_q_reference_k"]["failure_count"] == 3,
            "inherited-hidden-to-Q controlled comparison changed")

    write_json(output / "numerical.json", {
        "stage_comparisons": stage_comparisons,
        "controlled_comparisons": controlled,
        "boundary": "Numerical diagnosis only; simulator has not yet executed.",
    })

    def module_source(name: str) -> Path:
        matches = [path for path in (ROOT / "ace3/rtl").glob("*.sv")
                   if re.search(r"\bmodule\s+" + re.escape(name) + r"\b",
                                path.read_text(encoding="utf-8"))]
        require(len(matches) == 1, f"public module definition ambiguous: {name}")
        return matches[0]

    sources = [
        ROOT / "ace3/rtl/ace3_attention_score_core.sv",
        module_source("ace3_fp16_to_q24"),
        module_source("ace3_q24_to_fp16_rne"),
        ROOT / "ace3/rtl/ace3_decoder_layer0_token_engine.sv",
        ROOT / "ace3/rtl/ace3_awq_w4a16_projection_engine.sv",
        ROOT / "ace3/model/accepted_awq_projection.py",
        ROOT / "ace3/model/official_model24_next_token.py",
        Path(inspect.getfile(_torch_rmsnorm)).resolve(),
        ROOT / "ace3/model/qwen2_rope_oracle.py",
        ROOT / "ace3/model/attention_oracle.py",
        ROOT / "ace3/model/diagnose_layer01_position2_stage08.py",
        ROOT / "ace3/model/run_corrected_q_selected_token_frontier.py",
        ROOT / "ace3/model/validate_selected_token_position2_traversal.py",
        Path(__file__).resolve(),
        ROOT / "ace3/tb/ace3_layer08_stage08_diagnostic_tb.sv",
    ]
    source_records = [file_record(path) for path in sources]
    public_contract = sources[0].read_text(encoding="ascii").split(");", 1)[0] + ");\n"
    (output / "public_contract.sv.txt").write_text(public_contract, encoding="ascii")
    for name, data in (("q", actual[4]), ("k", actual_k.reshape(-1))):
        with (output / f"{name}.hex").open("x", encoding="ascii") as stream:
            stream.writelines(f"{int(value):04x}\n" for value in data)
    policy = load_json(REFERENCE / "POLICY.json")
    compile_command = ["iverilog", "-g2012", "-s", "ace3_attention_score_core",
                       "-Pace3_attention_score_core.HEAD_DIM=64",
                       "-o", str(output / "public_contract.vvp"),
                       *map(str, dict.fromkeys(sources[:3]))]
    simulation_compile = [
        "iverilog", "-g2012", "-s", "ace3_layer08_stage08_diagnostic_tb",
        "-o", str(output / "score.vvp"),
        *map(str, dict.fromkeys(sources[:3])), str(sources[-1])]
    simulator_command = ["vvp", str(output / "score.vvp")]
    write_json(output / "frozen.json", {
        "input_attempt": str(ATTEMPT), "sources": source_records,
        "authenticated_inputs": list(authenticated.values()),
        "public_module": "ace3_attention_score_core", "parameters": {"HEAD_DIM": 64},
        "compile_contract": compile_command, "compile_simulation": simulation_compile,
        "simulate": simulator_command, "policy": policy,
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "iverilog": subprocess.check_output(
            ["iverilog", "-V"], stderr=subprocess.STDOUT, text=True),
        "q_input": file_record(output / "q.hex"), "k_input": file_record(output / "k.hex"),
        "scope": "42 layer08/position2 scores only; no decoder traversal",
    })
    for name, command in (("contract_compile", compile_command),
                          ("score_compile", simulation_compile),
                          ("simulation", simulator_command)):
        with (output / f"{name}.log").open("x", encoding="ascii") as log:
            completed = subprocess.run(command, cwd=output, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=60)
        write_json(output / f"{name}.exit.json", {"command": command,
                                                "exit_code": completed.returncode})
        require(completed.returncode == 0, f"{name} failed; see preserved log")
    require((output / "terminal.txt").read_text(encoding="ascii") ==
            "natural_terminal=1 exit_code=0 scores=42\n",
            "score replay did not terminate naturally")
    replay = load_hex(output / "scores.hex", 4)
    require(np.array_equal(replay, actual[8]),
            "fresh current-source score RTL differs from immutable trace")

    failed = stage_comparisons["8"]["first_failure"]
    head, key_position = failed["head"], failed["key_position"]
    q_actual = values(actual[4]).reshape(14, 64)[head]
    q_expected = values(reference_stages[4]).reshape(14, 64)[head]
    k_actual = values(actual_k)[key_position, head // 7]
    k_expected = values(expected_k)[key_position, head // 7]
    dq, dk = q_actual - q_expected, k_actual - k_expected
    contributions = np.stack((dq * k_expected, q_expected * dk, dq * dk), axis=1) / 8
    with (output / "first_failure_terms.csv").open("x", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dimension", "q_actual", "q_reference", "k_actual",
                         "k_reference", "q_delta", "k_delta", "interaction"])
        for index in range(64):
            writer.writerow([index, q_actual[index], q_expected[index],
                             k_actual[index], k_expected[index], *contributions[index]])
    component_comparisons = {
        "incoming_hidden": detail(hidden, expected_hidden),
        "accepted_norm_on_actual_hidden_vs_actual_stage00": detail(
            accepted_norm_actual_hidden, actual[0]),
        "accepted_q_on_actual_norm_vs_actual_stage01": detail(
            accepted_q_actual_norm, actual[1]),
        "accepted_k_on_actual_norm_vs_actual_stage02": detail(
            project(actual[0], "k"), actual[2]),
        "accepted_rope_on_actual_projection_vs_actual_stage04": detail(
            reference_rope(actual[1]), actual[4]),
    }
    for position in range(3):
        component_comparisons[f"k_cache_position{position}"] = detail(
            actual_k[position], expected_k[position])
        component_comparisons[
            f"accepted_k_on_actual_norm_vs_actual_stage02_position{position}"
        ] = detail(project(stages[position][0], "k"), stages[position][2])
    for record in [*authenticated.values(), *source_records]:
        require(file_record(Path(record["path"])) == record,
                f"input/source changed during diagnosis: {record['path']}")
    report = {
        "status": "LOCALIZED_TO_PRE_SCORE_Q_K_OPERANDS",
        "taxonomy": "numerical_policy_material_mismatch",
        "input_attempt": str(ATTEMPT), "attempt002_preserved": True,
        "layer07_to_layer08_handoff": handoff,
        "earliest_material_stage": 8, "first_material_value": failed,
        "stage_comparisons": stage_comparisons,
        "component_comparisons": component_comparisons,
        "earliest_observed_value_class": (
            "incoming_layer08_hidden" if
            component_comparisons["incoming_hidden"]["different_count"] else
            "layer08_input_rmsnorm"),
        "source_attribution": {
            "hidden_handoff": (
                "ace3/model/run_corrected_q_selected_token_frontier.py:155-192 "
                "(execute_exact_transaction returns raw final; hidden[position] "
                "carries it into the next layer)"),
            "hidden_serialization": (
                "ace3/model/validate_selected_token_position2_traversal.py:466-505,"
                "592-661 (serialize hidden_payload, bind simulator input, and "
                "return load_hidden_bits(raw/final.hex))"),
            "score_arithmetic": "ace3/rtl/ace3_attention_score_core.sv:69-125 "
                                "(Q24 decode, exact Q48 dot, /8 and FP16 RNE)",
            "score_stage_interface": "ace3/rtl/ace3_decoder_layer0_token_engine.sv:"
                                     "781-785 (cache read, pair handshake, stage08)",
            "q_vs_k_bias_policy": "ace3/rtl/ace3_decoder_layer0_token_engine.sv:"
                                  "353,384-387 (Q single-round; K legacy double-round)",
            "projection_rounding": "ace3/rtl/ace3_awq_w4a16_projection_engine.sv:"
                                   "153-157,196-203",
            "accepted_norm_implementation": inspect.getfile(_torch_rmsnorm),
            "accepted_projection": "ace3/model/accepted_awq_projection.py:48-70",
            "accepted_rope": "ace3/model/layer3_token0_diagnostic.py:265-286",
            "trace_serialization": "ace3/tb/ace3_decoder_layer0_token_engine_main.cpp:"
                                   "285-291; stage08 key indices repeat per query head",
        },
        "controlled_comparisons": controlled,
        "conclusion": (
            "The earliest observed layer08 divergence is inherited incoming-hidden "
            "drift: layer07 position2 raw final equals layer08 serialized input "
            "in every FP16 value and every serialized byte. The named harness "
            "handoff propagates, rather than introduces, that drift. Accepted "
            "RMSNorm and Q projection reproduce actual stages00/01 exactly on "
            "actual inputs; replacing reference norm/Q removes the three material "
            "stage08 failures, while replacing K alone does not. The fresh score "
            "RTL reproduces all 42 preserved scores. The material mismatch is "
            "inherited-hidden-to-Q drift amplified by the dot product, not a "
            "local score arithmetic, stage-interface, or serialization defect "
            "on these operands. The cause upstream of layer07 raw final remains "
            "outside this bounded diagnosis."),
        "first_failure_dot_decomposition": {
            "unrounded_actual": float(q_actual @ k_actual / 8),
            "unrounded_reference": float(q_expected @ k_expected / 8),
            "q_delta": float(contributions[:, 0].sum()),
            "k_delta": float(contributions[:, 1].sum()),
            "interaction": float(contributions[:, 2].sum()),
            "largest_absolute_contribution_dimensions": np.argsort(
                -np.abs(contributions.sum(axis=1)))[:8].tolist(),
        },
        "fresh_rtl": {"top": "ace3_attention_score_core", "scores": 42,
                      "exact_preserved_matches": int(np.sum(replay == actual[8])),
                      "accepted_policy": detail(replay, reference_stages[8]),
                      "terminal": file_record(output / "terminal.txt"),
                      "trace": file_record(output / "scores.hex")},
        "regression": "All 42 replay scores must equal immutable stage08; "
                      "all independent Q/K dot scores must equal accepted stage08; "
                      "the original three material failures must remain observable; "
                      "layer07 raw final must equal layer08 serialized input in "
                      "all 896 FP16 values and all bytes, with semantic bindings "
                      "equal; actual-input norm/Q must reproduce stages00/01; "
                      "reference Q or norm replacement must remove all three failures.",
        "boundary": "Local score replay plus numerical controlled comparisons, not a "
                    "passing layer08 repair, full decoder replay, or layer09 advance. "
                    "No synthesis, PPA, board, dialogue, or end-to-end claim.",
        "independent_review": "required; Host Reviewer has not yet adjudicated",
        "frozen": file_record(output / "frozen.json"),
        "tensor_bindings": tensor_records,
        "evidence_seal": "After report creation, remove all write permission bits "
                         "from every attempt artifact and the attempt directory. "
                         "This is a filesystem read-only seal, not tamper-proof storage.",
    }
    write_json(output / "report.json", report)
    for artifact in output.iterdir():
        artifact.chmod(artifact.stat().st_mode & ~0o222)
        require(artifact.stat().st_mode & 0o222 == 0,
                f"diagnostic artifact remains writable: {artifact}")
    output.chmod(output.stat().st_mode & ~0o222)
    require(output.stat().st_mode & 0o222 == 0, "diagnostic directory remains writable")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({
        "status": report["status"], "report": str(args.output / "report.json"),
        "first_failure": report["first_material_value"],
        "layer07_to_layer08_handoff": report["layer07_to_layer08_handoff"],
        "conclusion": report["conclusion"],
        "component_comparisons": report["component_comparisons"],
        "controlled_failure_counts": {
            name: result["failure_count"]
            for name, result in report["controlled_comparisons"].items()},
        "dot_decomposition": report["first_failure_dot_decomposition"],
    }, indent=2))


if __name__ == "__main__":
    main()
