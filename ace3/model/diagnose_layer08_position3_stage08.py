#!/usr/bin/env python3
"""Attribute sealed position-3 score drift without replaying decoder prefixes."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from qwen2_rope_oracle import qwen2_coefficient, rotate_pair
from layer3_token0_diagnostic import decode_stage_records


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = ROOT / "build/token358_position3_full_continuation_attempt003"
OUTPUT = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else ATTEMPT / "score_diagnostic001"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def values(bits):
    return np.ascontiguousarray(bits, dtype="<u2").view("<f2").astype(np.float64)


def half_bits(value):
    return np.asarray(value, dtype="<f2").view("<u2").reshape(-1)


def detail(actual, expected):
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    a, e = values(actual), values(expected)
    error = np.abs(a - e)
    relative = error / np.maximum(np.abs(e), 2.0**-14)

    def ordered(bits):
        b = bits.astype(np.int64)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7fff), 0x8000 + b)

    ulp = np.abs(ordered(actual) - ordered(expected))
    finite = np.isfinite(a) & np.isfinite(e)
    failed = ~(finite & ((error <= 0.125) | ((relative < 0.001) & (ulp <= 1))))
    return {
        "different_count": int(np.count_nonzero(actual != expected)),
        "failure_indices": np.flatnonzero(failed).tolist(),
        "max_absolute_error": float(error.max()),
        "elements": [
            {"index": i, "actual_bits": f"{int(actual[i]):04x}",
             "expected_bits": f"{int(expected[i]):04x}",
             "actual": float(a[i]), "expected": float(e[i]),
             "absolute_error": float(error[i]), "relative_error": float(relative[i]),
             "ulp": int(ulp[i]), "accepted": bool(not failed[i])}
            for i in (13, 15) if i < actual.size
        ],
    }


def run():
    require(OUTPUT.parent == ATTEMPT, "diagnostic must remain inside attempt003")
    require(OUTPUT.is_dir() and {p.name for p in OUTPUT.iterdir()}
            <= {"score_tb.sv", "diagnostic.log"}, "diagnostic output is not fresh")
    torch.set_num_threads(1)
    bindings, consumed = {}, {}

    def register(value):
        if isinstance(value, dict):
            if {"path", "sha256", "bytes"} <= value.keys():
                previous = bindings.get(value["path"])
                require(previous is None or previous["sha256"] == value["sha256"],
                        f"conflicting binding: {value['path']}")
                bindings[value["path"]] = value
            for child in value.values():
                register(child)
        elif isinstance(value, list):
            for child in value:
                register(child)

    def read(path):
        path = Path(path).resolve()
        record = bindings[str(path)]
        data = path.read_bytes()
        require(len(data) == record["bytes"]
                and hashlib.sha256(data).hexdigest() == record["sha256"],
                f"binding mismatch: {path}")
        consumed[str(path)] = record
        return data

    def load(path):
        value = json.loads(read(path))
        register(value)
        return value

    # The preserved attempt seal is the trust root, not a newly accepted run.
    seal_path = ATTEMPT / "seal.json"
    seal_bytes = seal_path.read_bytes()
    register(json.loads(seal_bytes))
    load(ATTEMPT / "layer08/seal.json")
    frozen = load(ATTEMPT / "frozen.json")
    original = load(frozen["original_frozen"]["path"])
    prepared = load(ATTEMPT / "layer08/prepared.json")
    result = load(ATTEMPT / "layer08/result.json")
    reference = load(prepared["independent_parent"]["path"])
    require(result["layer"] == 8 and result["position"] == 3
            and prepared["kv_parent"]["layer_index"] == 8
            and prepared["kv_parent"]["valid_positions"] == [0, 1, 2]
            and reference["layer"] == 8 and reference["history"] == [9707, 1879, 0],
            "layer/history/cache parent mismatch")
    require(original["comparison_policy"] == frozen["comparison_policy"],
            "comparison policy drift")
    read(prepared["binary"]["path"])
    read(prepared["kv_parent"]["state"]["path"])

    def trace(path, position):
        stages = {}
        for row in read(path).decode("ascii").splitlines():
            require(len(row) == 16 and int(row[:2], 16) == 0
                    and int(row[2:6], 16) == position, "trace identity mismatch")
            stage, index, value = int(row[6:8], 16), int(row[8:12], 16), int(row[12:], 16)
            if stage <= 8:
                stage_rows = stages.setdefault(stage, [])
                stage_rows.append((index, value))
        return {s: decode_stage_records(v, s, position) for s, v in stages.items()}

    actual = trace(ATTEMPT / "layer08/position003/raw/trace.hex", 3)
    caches = [trace(r["path"], p) for p, r in
              enumerate(prepared["kv_parent"]["actual_kv_traces"])]
    caches.append(actual)

    def hex_bits(path):
        return np.asarray([int(row, 16) for row in read(path).splitlines()], dtype="<u2")

    independent = {s: hex_bits(prepared["independent_stages"][str(s)]["path"])
                   for s in range(9)}
    for s in range(9):
        require(actual[s].shape == independent[s].shape, "stage geometry mismatch")
    require(actual[8].size == 56, "score geometry mismatch")
    for cache in caches:
        require(np.array_equal(cache[5], cache[6]), "K cache write/order mismatch")
    actual_k = np.stack([c[6].reshape(2, 64) for c in caches])

    def array(path):
        import io
        return np.load(io.BytesIO(read(path)), allow_pickle=False)

    independent_k = array(ATTEMPT / "layer08/independent/position003_own_k.npy")
    independent_v = array(ATTEMPT / "layer08/independent/position003_own_v.npy")
    require(independent_k.shape == independent_v.shape == (4, 2, 64),
            "independent cache geometry mismatch")
    require(np.array_equal(independent_k[:3], array(reference["own_cache"]["k"]["path"]))
            and np.array_equal(independent_v[:3], array(reference["own_cache"]["v"]["path"])),
            "independent cache parentage mismatch")
    require(np.array_equal(independent_k[3].reshape(-1), independent[6]),
            "independent current K order mismatch")
    require(np.array_equal(actual_k, array(ATTEMPT / "layer08/position003_actual_k.npy"))
            and np.array_equal(np.stack([c[7].reshape(2, 64) for c in caches]),
                               array(ATTEMPT / "layer08/position003_actual_v.npy")),
            "actual K/V receipt parentage mismatch")
    hidden_rows = read(prepared["vectors"]["input"]["path"]).decode("ascii").splitlines()
    require(len(hidden_rows) == 896 and all(
        len(row) == 10 and int(row[:6], 16) == i for i, row in enumerate(hidden_rows)),
        "hidden vector geometry/order mismatch")
    hidden = np.asarray([int(row[6:], 16) for row in hidden_rows], dtype="<u2")
    require(read(prepared["vectors"]["input"]["path"])
            == read(prepared["input_hidden"]["path"]), "actual hidden parent mismatch")
    independent_hidden = hex_bits(ATTEMPT / "layer07/independent/stage18.hex")

    tensors = {}
    tensor_hashes = {r["name"]: r["sha256"] for r in reference["checkpoint_tensor_hashes"]}
    for item in prepared["vectors"]["tensors"]:
        tensor = item["checkpoint_tensor"]
        name = tensor["name"]
        if not any(part in name for part in ("input_layernorm", "self_attn.q_proj",
                                             "self_attn.k_proj")):
            continue
        dtype = "<u2" if tensor["dtype"] == "F16" else "<u4"
        data = np.asarray([int(row, 16) for row in
                           read(item["serialized"]["path"]).splitlines()], dtype=dtype)
        require(hashlib.sha256(data.tobytes()).hexdigest()
                == tensor["sha256"] == tensor_hashes[name], "AWQ tensor binding mismatch")
        tensors[name] = data.view("<f2" if dtype == "<u2" else "<i4").reshape(tensor["shape"])

    source = ROOT / "build/independent_fp16_trajectory_20260906_1133/source/official_single_decoder_layer.py"
    names = {"_torch_unpack", "_torch_linear", "_torch_rmsnorm"}
    nodes = [n for n in ast.parse(read(source)).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "missing independent numerical definitions")
    namespace = {"np": np, "torch": torch, "GROUP_SIZE": 128, "_require": require,
                 "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7)}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)

    def norm(h):
        x = torch.from_numpy(values(h)[None, :])
        return half_bits(namespace["_torch_rmsnorm"](x, tensors["model.layers.8.input_layernorm.weight"]).numpy())

    def project(x, kind):
        prefix = f"model.layers.8.self_attn.{kind}_proj"
        output = namespace["_torch_linear"](torch.from_numpy(values(x)[None, :]),
                                             tensors, prefix, prefix + ".bias")
        return half_bits(output.numpy())

    def rotate(x, position, policy):
        shaped = values(x).reshape(-1, 64)
        if policy == "independent":
            frequencies = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
            angle = position * frequencies
            cosine, sine = torch.cos(angle).numpy(), torch.sin(angle).numpy()
            low, high = shaped[:, :32], shaped[:, 32:]
            return half_bits(np.concatenate((low * cosine - high * sine,
                                              high * cosine + low * sine), axis=1))
        out = np.empty_like(x)
        for head in range(shaped.shape[0]):
            for pair in range(32):
                lo, hi = head * 64 + pair, head * 64 + pair + 32
                c, s = qwen2_coefficient(position, pair)
                a, b, invalid, saturation = rotate_pair(int(x[lo]), int(x[hi]), c, s)
                require(not invalid and not saturation, "RoPE flags")
                out[lo], out[hi] = a, b
        return out

    def scores(q, k):
        qv, kv = values(q).reshape(14, 64), values(k)
        return np.stack([kv[:, h // 7] @ qv[h] / 8 for h in range(14)]).reshape(-1)

    same_norm = norm(hidden)
    same_q, same_k = project(actual[0], "q"), project(actual[0], "k")
    require(np.array_equal(norm(independent_hidden), independent[0]),
            "independent norm reconstruction mismatch")
    require(np.array_equal(project(independent[0], "q"), independent[1])
            and np.array_equal(project(independent[0], "k"), independent[2]),
            "independent AWQ reconstruction mismatch")
    require(np.array_equal(rotate(independent[1], 3, "independent"), independent[4])
            and np.array_equal(rotate(independent[2], 3, "independent"), independent[5]),
            "independent RoPE reconstruction mismatch")
    require(np.array_equal(rotate(actual[1], 3, "rtl"), actual[4])
            and all(np.array_equal(rotate(c[2], p, "rtl"), c[5])
                    for p, c in enumerate(caches)), "RTL RoPE reconstruction mismatch")
    require(np.array_equal(half_bits(scores(independent[4], independent_k)), independent[8]),
            "independent score reconstruction mismatch")
    require(np.array_equal(half_bits(scores(actual[4], actual_k)), actual[8]),
            "exact operand dot does not reproduce preserved scores")
    require(detail(actual[8], independent[8])["failure_indices"] == [13, 15],
            "original failure identity changed")

    rope_records = read(prepared["vectors"]["rope_coefficients"]["path"]).splitlines()
    expected_coefficients = [qwen2_coefficient(3, p) for p in range(32)]
    require(rope_records == [
        f"{position:04x}{pair:02x}{c:04x}{s:04x}".encode("ascii")
        for position in range(4) for pair in range(32)
        for c, s in (qwen2_coefficient(position, pair),)
    ], "RoPE coefficient position/pair/precision binding mismatch")
    write_json(OUTPUT / "rope_coefficients.json", {
        "serialized_rows": [r.decode("ascii") for r in rope_records],
        "expected_fp16_cos_sin": [[f"{c:04x}", f"{s:04x}"] for c, s in expected_coefficients],
    })
    controls = {
        "baseline": (actual[4], actual_k),
        "independent_q_only": (independent[4], actual_k),
        "independent_k_only": (actual[4], independent_k),
        "independent_q_and_k": (independent[4], independent_k),
        "independent_rope_on_actual_q": (rotate(actual[1], 3, "independent"), actual_k),
        "rtl_rope_on_independent_q_projection": (rotate(independent[1], 3, "rtl"), actual_k),
        "independent_norm_same_hidden_q_only": (rotate(project(same_norm, "q"), 3, "rtl"), actual_k),
        "independent_projection_same_norm_q_only": (rotate(same_q, 3, "rtl"), actual_k),
    }
    for p in range(4):
        for kind in ("independent_cache", "independent_rope"):
            k = actual_k.copy()
            k[p] = independent_k[p] if kind == "independent_cache" else rotate(
                caches[p][2], p, "independent").reshape(2, 64)
            controls[f"{kind}_position{p}_only"] = (actual[4], k)
    corrected_rope_k = np.stack([rotate(c[2], p, "independent").reshape(2, 64)
                                for p, c in enumerate(caches)])
    controls["independent_rope_all_qk"] = (rotate(actual[1], 3, "independent"), corrected_rope_k)
    stage_details = {str(s): detail(actual[s], independent[s]) for s in range(9)}
    control_details = {}
    for name, (q, k) in controls.items():
        score_values = scores(q, k)
        control_details[name] = {
            **detail(half_bits(score_values), independent[8]),
            "pre_round_offending": [float(score_values[i]) for i in (13, 15)],
            "offending_bits": [f"{int(half_bits(score_values)[i]):04x}" for i in (13, 15)],
        }
    with (OUTPUT / "offending_products.csv").open("x", encoding="ascii", newline="") as stream:
        fields = ["element", "head", "key_position", "dimension", "q_actual", "q_reference",
                  "k_actual", "k_reference", "actual_product_scaled",
                  "reference_product_scaled", "q_contribution", "k_contribution", "cross_contribution"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in (13, 15):
            h, p = divmod(i, 4)
            qa, qr = values(actual[4]).reshape(14, 64)[h], values(independent[4]).reshape(14, 64)[h]
            ka, kr = values(actual_k)[p, h // 7], values(independent_k)[p, h // 7]
            for d in range(64):
                writer.writerow(dict(zip(fields, (i, h, p, d, qa[d], qr[d], ka[d], kr[d],
                    qa[d] * ka[d] / 8, qr[d] * kr[d] / 8, (qa[d] - qr[d]) * kr[d] / 8,
                    qr[d] * (ka[d] - kr[d]) / 8, (qa[d] - qr[d]) * (ka[d] - kr[d]) / 8))))

    # One score-only RTL run: all controlled cases, with the unchanged public top.
    for name, (q, k) in controls.items():
        with (OUTPUT / f"{name}.hex").open("x", encoding="ascii") as stream:
            for h in range(14):
                for p in range(4):
                    for d in range(64):
                        stream.write(f"{int(q[h * 64 + d]):04x}{int(k[p, h // 7, d]):04x}\n")
    source_paths = [ROOT / "ace3/rtl" / n for n in (
        "ace3_attention_score_core.sv", "ace3_fp16_fixed.sv")]
    for path in source_paths:
        read(path)
    bench = OUTPUT / "score_tb.sv"
    bench_bytes = bench.read_bytes()
    compile_top = ["iverilog", "-g2012", "-s", "ace3_attention_score_core",
                   "-Pace3_attention_score_core.HEAD_DIM=64", "-o", str(OUTPUT / "public_top.vvp"),
                   *map(str, source_paths)]
    compile_bench = ["iverilog", "-g2012", "-s", "score_tb",
                     "-Pscore_tb.CASES=" + str(len(controls)), "-o", str(OUTPUT / "score.vvp"),
                     *map(str, source_paths), str(bench)]
    all_pairs = "".join((OUTPUT / f"{name}.hex").read_text("ascii") for name in controls)
    with (OUTPUT / "pairs.hex").open("x", encoding="ascii") as stream:
        stream.write(all_pairs)
    frozen_diagnostic = {
        "kind": "controlled_score_diagnostic_not_official_decoder_attempt",
        "original_seal": {"path": str(seal_path), "sha256": hashlib.sha256(seal_bytes).hexdigest()},
        "inputs": list(consumed.values()), "comparison_policy": frozen["comparison_policy"],
        "public_top": read(source_paths[0]).decode("ascii").split(");", 1)[0] + ");",
        "parameters": {"HEAD_DIM": 64}, "compile_top": compile_top, "compile_bench": compile_bench,
        "bench_sha256": hashlib.sha256(bench_bytes).hexdigest(),
        "pair_input_sha256": hashlib.sha256(all_pairs.encode("ascii")).hexdigest(),
        "case_order": list(controls), "python": sys.version, "numpy": np.__version__,
        "torch": torch.__version__, "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "iverilog_version": subprocess.run(["iverilog", "-V"], capture_output=True, text=True, check=True).stdout,
        "vvp_version": subprocess.run(["vvp", "-V"], capture_output=True, text=True, check=True).stderr,
        "prefix_replays": 0, "official_decoder_attempts": 0,
    }
    write_json(OUTPUT / "frozen.json", frozen_diagnostic)
    for phase, command in (("public_compile", compile_top), ("bench_compile", compile_bench),
                           ("simulation", ["vvp", str(OUTPUT / "score.vvp")])):
        with (OUTPUT / f"{phase}.log").open("x") as stream:
            proc = subprocess.run(command, cwd=OUTPUT, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=60, check=False)
        require(proc.returncode == 0, f"{phase} failed: see preserved log")
    rtl = np.asarray([int(r, 16) for r in (OUTPUT / "scores.hex").read_text("ascii").splitlines()],
                     dtype="<u2").reshape(len(controls), 56)
    for index, (name, (q, k)) in enumerate(controls.items()):
        require(np.array_equal(rtl[index], half_bits(scores(q, k))),
                f"fresh RTL/exact-dot disagreement: {name}")
    require(np.array_equal(rtl[0], actual[8]), "fresh RTL/preserved trace disagreement")
    summary = {
        "status": "DIAGNOSTIC_COMPLETE_NOT_LAYER_ACCEPTANCE",
        "original_status": result["status"], "stage08": stage_details["8"],
        "stages": stage_details, "controls": control_details,
        "same_operand_checks": {
            "input_hidden_trajectory": detail(hidden, independent_hidden),
            "norm_actual_hidden": detail(actual[0], same_norm),
            "q_projection_actual_norm": detail(actual[1], same_q),
            "k_projection_actual_norm": detail(actual[2], same_k),
            "q_rope_same_projection": detail(actual[4], rotate(actual[1], 3, "independent")),
            "k_rope_same_projection": detail(actual[5], rotate(actual[2], 3, "independent")),
        },
        "fresh_rtl": {"cases": len(controls), "scores": int(rtl.size),
                      "all_match_binary64_dots": True, "baseline_matches_preserved_trace": True},
        "boundaries": "No decoder replay, accepted-prefix execution, tolerance change, RTL repair, or end-to-end claim.",
    }
    write_json(OUTPUT / "result.json", summary)
    print(json.dumps({"status": summary["status"],
                      "same_operand_differences": {k: v["different_count"] for k, v in summary["same_operand_checks"].items()},
                      "controls": {k: {"failures": v["failure_indices"], "bits": v["offending_bits"],
                                       "pre_round": v["pre_round_offending"]} for k, v in control_details.items()}}))


if __name__ == "__main__":
    run()
