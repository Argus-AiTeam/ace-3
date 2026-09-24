#!/usr/bin/env python3
"""Compare retained layer2-8 operators independently, without replaying a prefix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as functional

from diagnose_layer08_position3_stage08 import detail, half_bits, require, values, write_json
from layer3_token0_diagnostic import decode_stage_records


ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "build/token358_position3_full_continuation_attempt003"
B = ROOT / "build/token358_position3_full_continuation_attempt004"
REFERENCE = ROOT / "build/independent_fp16_trajectory_20260906_1133/source"


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def compared(actual, expected):
    result = detail(actual, expected)
    result.pop("elements")
    a, e = values(actual).reshape(-1), values(expected).reshape(-1)
    indices = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    order = indices[np.argsort(-np.abs(a[indices] - e[indices]), kind="stable")]
    result["witnesses"] = [
        {"index": int(i), "actual_bits": f"{int(actual.reshape(-1)[i]):04x}",
         "independent_bits": f"{int(expected.reshape(-1)[i]):04x}",
         "actual": float(a[i]), "independent": float(e[i])}
        for i in order[:5]
    ]
    return result


class Evidence:
    def __init__(self):
        self.bindings = {}
        self.consumed = {}

    def register(self, value):
        if isinstance(value, dict):
            if "path" in value and ("file_sha256" in value or "sha256" in value):
                digest = value.get("file_sha256", value.get("sha256"))
                key = str(Path(value["path"]).resolve())
                prior = self.bindings.get(key)
                require(prior is None or prior["sha256"] == digest, f"binding conflict: {key}")
                bound = {"path": key, "sha256": digest}
                if "bytes" in value:
                    bound["bytes"] = value["bytes"]
                self.bindings[key] = bound
            for child in value.values():
                self.register(child)
        elif isinstance(value, list):
            for child in value:
                self.register(child)

    def read(self, path):
        key = str(Path(path).resolve())
        bound = self.bindings[key]
        data = Path(key).read_bytes()
        require(hashlib.sha256(data).hexdigest() == bound["sha256"]
                and ("bytes" not in bound or len(data) == bound["bytes"]),
                f"evidence binding mismatch: {key}")
        self.consumed[key] = {**bound, "bytes": len(data)}
        return data

    def load(self, path):
        result = json.loads(self.read(path))
        self.register(result)
        return result

    def bits(self, path, indexed=False):
        rows = self.read(path).decode("ascii").splitlines()
        width = 10 if indexed else 4
        require(all(len(row) == width for row in rows), f"hex width: {path}")
        if indexed:
            require(all(int(row[:6], 16) == i for i, row in enumerate(rows)),
                    f"hidden address order: {path}")
        return np.asarray([int(row[-4:], 16) for row in rows], dtype="<u2")

    def array(self, path):
        result = np.load(io.BytesIO(self.read(path)), allow_pickle=False)
        require(result.dtype == np.dtype("<u2"), f"cache precision: {path}")
        return result

    def trace(self, path, position, last_stage=18):
        grouped = {s: [] for s in range(last_stage + 1)}
        for row in self.read(path).decode("ascii").splitlines():
            require(len(row) == 16 and int(row[:2], 16) == 0
                    and int(row[2:6], 16) == position, f"trace identity: {path}")
            stage, index, value = int(row[6:8], 16), int(row[8:12], 16), int(row[12:], 16)
            require(0 <= stage <= 18, f"unknown trace stage: {stage}")
            if stage <= last_stage:
                grouped[stage].append((index, value))
        return {s: decode_stage_records(rows, s, position) for s, rows in grouped.items()}


def diagnose(out, command):
    started = time.monotonic()
    torch.set_num_threads(1)
    evidence = Evidence()
    roots = [record(folder / "seal.json") for folder in (A, B)]
    for root in roots:
        evidence.register(root)
        evidence.load(root["path"])
    frozen = evidence.load(A / "frozen.json")
    original = evidence.load(frozen["original_frozen"]["path"])
    package = evidence.load(original["launch_package"]["path"])
    require(original["comparison_policy"] == frozen["comparison_policy"]
            and package["position"] == 3 and package["token_history"] == [9707, 1879, 0, 358],
            "frozen profile/history drift")
    recovered = evidence.load(frozen["recovered_layer2"]["path"])
    require(recovered["status"] == "ZERO_MATERIAL_MISMATCHES_LAYER02",
            "layer2 acceptance receipt changed")
    evidence.load(original["review"]["path"])
    evidence.load(package["layer1_review"]["path"])
    evidence.load(package["layer1_result"]["path"])
    for folder in (A, B):
        for layer in (range(3, 9) if folder == A else [8]):
            evidence.load(folder / f"layer{layer:02d}/seal.json")
    plans = []
    for layer in range(2, 9):
        directory = (ROOT / "build/token358_position3_full_continuation_attempt001/layer02"
                     if layer == 2 else A / f"layer{layer:02d}")
        result = recovered if layer == 2 else evidence.load(directory / "result.json")
        prepared = evidence.load(directory / "prepared.json")
        transaction = evidence.load(result["transaction"]["path"])
        ref = evidence.load(prepared["independent_parent"]["path"])
        require(result["layer"] == layer and result["position"] == 3
                and ref["layer"] == layer and ref["history"] == [9707, 1879, 0]
                and prepared["kv_parent"]["layer_index"] == layer
                and prepared["kv_parent"]["valid_positions"] == [0, 1, 2],
                f"layer/cache identity drift: {layer}")
        if layer < 8:
            require(result["material_mismatches"] == 0, f"prefix gate changed: {layer}")
        require(all(row["exact_match"] for row in result["exact"].values()),
                f"preserved local-exact receipt changed: {layer}")
        plans.append((layer, directory, result, prepared, transaction, ref))

    namespace = {"np": np, "torch": torch, "GROUP_SIZE": 128, "_require": require,
                 "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7)}
    source = REFERENCE / "official_single_decoder_layer.py"
    names = {"_torch_unpack", "_torch_linear", "_torch_rmsnorm"}
    nodes = [node for node in ast.parse(evidence.read(source)).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == names, "missing independent numerical operator")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    evidence.read(REFERENCE / "propagated_reference.py")
    write_json(out / "frozen.json", {
        "kind": "retained_operand_numerical_diagnostic_not_an_official_RTL_attempt",
        "roots": roots, "original_frozen": frozen["original_frozen"],
        "recovered_layer2": frozen["recovered_layer2"], "comparison_policy": frozen["comparison_policy"],
        "source": record(__file__), "command": record(command),
        "helpers": [record(Path(__file__).parent / name) for name in (
            "diagnose_layer08_position3_stage08.py", "layer3_token0_diagnostic.py")],
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "layers": list(range(2, 9)), "position": 3,
        "method": "Each operator consumes retained operands. No output is propagated into another operator.",
        "prefix_replays": 0, "RTL_invocations": 0, "acceptance_policy_changes": 0,
    })

    prior_actual = evidence.bits(package["layer1_hidden"]["path"], indexed=True)
    prior_independent = evidence.bits(package["independent_layer1_hidden"]["path"])
    boundary = {"actual": package["layer1_hidden"],
                "independent": package["independent_layer1_hidden"],
                "comparison": compared(prior_actual, prior_independent)}
    require(boundary["comparison"]["different_count"] > 0,
            "layer2-input drift hypothesis not reproduced")
    rows = []
    for layer, directory, receipt, prepared, transaction, ref in plans:
        layer_start = time.monotonic()
        actual = evidence.trace(transaction["raw"]["trace"]["path"], 3)
        independent = {s: evidence.bits(prepared["independent_stages"][str(s)]["path"])
                       for s in range(19)}
        hidden = evidence.bits(prepared["vectors"]["input"]["path"], indexed=True)
        require(np.array_equal(hidden, prior_actual)
                and evidence.read(prepared["vectors"]["input"]["path"])
                == evidence.read(prepared["input_hidden"]["path"]),
                f"actual hidden parent mismatch: {layer}")
        require(hashlib.sha256(hidden.tobytes()).hexdigest() == transaction["input"]["sha256"],
                f"actual input semantic hash mismatch: {layer}")
        terminal = evidence.read(transaction["raw"]["terminal"]["path"]).decode("ascii").split()
        fields = dict(item.split("=", 1) for item in terminal)
        require(len(fields) == len(terminal) and fields["natural_terminal"] == "1"
                and fields["exit_code"] == "0" and fields["done_count"] == "1"
                and fields["final_count"] == "896" and fields["position"] == "3"
                and fields["layer_index"] == str(layer), f"terminal mismatch: {layer}")
        require(np.array_equal(actual[18], evidence.bits(receipt["output_hidden"]["path"], True)),
                f"final hidden/trace mismatch: {layer}")
        require(np.array_equal(independent[18], evidence.bits(receipt["independent_hidden"]["path"])),
                f"independent final binding mismatch: {layer}")
        require(evidence.read(transaction["raw"]["trace"]["path"])
                == evidence.read(prepared["vectors"]["trace"]["path"])
                and evidence.read(receipt["output_hidden"]["path"])
                == evidence.read(prepared["vectors"]["final_hidden"]["path"]),
                f"retained exact-local oracle disagreement: {layer}")

        parent = prepared["kv_parent"]
        evidence.read(parent["state"]["path"])
        evidence.read(prepared["binary"]["path"])
        old = [evidence.trace(item["path"], p, 7)
               for p, item in enumerate(parent["actual_kv_traces"])]
        actual_caches, independent_caches = {}, {}
        for kind, stage in (("k", 6), ("v", 7)):
            ak = np.stack([item[stage].reshape(2, 64) for item in old] +
                          [actual[stage].reshape(2, 64)])
            ik = evidence.array(directory / f"independent/position003_own_{kind}.npy")
            require(ak.shape == ik.shape == (4, 2, 64), f"cache shape: {layer}/{kind}")
            require(np.array_equal(ik[:3], evidence.array(ref["own_cache"][kind]["path"]))
                    and np.array_equal(ik[3].reshape(-1), independent[stage]),
                    f"independent cache parentage: {layer}/{kind}")
            require(np.array_equal(ak, evidence.array(receipt["actual_kv"][kind]["path"])),
                    f"actual cache parentage: {layer}/{kind}")
            actual_caches[kind], independent_caches[kind] = ak, ik
        require(all(np.array_equal(item[5], item[6]) and np.array_equal(item[3], item[7])
                    for item in [*old, actual]), f"cache write ordering: {layer}")

        tensors = {}
        tensor_hashes = {item["name"]: item["sha256"] for item in ref["checkpoint_tensor_hashes"]}
        for item in prepared["vectors"]["tensors"]:
            meta = item["checkpoint_tensor"]
            dtype = "<u2" if meta["dtype"] == "F16" else "<u4"
            data = np.asarray([int(row, 16) for row in
                               evidence.read(item["serialized"]["path"]).splitlines()], dtype=dtype)
            require(data.nbytes == meta["bytes"] and hashlib.sha256(data.tobytes()).hexdigest()
                    == meta["sha256"] == tensor_hashes[meta["name"]],
                    f"AWQ/norm tensor binding: {meta['name']}")
            tensors[meta["name"]] = data.view("<f2" if dtype == "<u2" else "<i4").reshape(meta["shape"])
        require(len(tensors) == 26, f"tensor coverage: {layer}")
        prefix = f"model.layers.{layer}"

        def tensor(bits):
            return torch.from_numpy(values(bits))

        def norm(bits, suffix):
            return namespace["_torch_rmsnorm"](
                tensor(bits)[None, :], tensors[prefix + "." + suffix])

        def linear(bits, suffix, bias=False):
            name = prefix + "." + suffix
            return namespace["_torch_linear"](
                tensor(bits)[None, :], tensors, name, name + ".bias" if bias else None)

        def rope(bits):
            x = tensor(bits).reshape(-1, 64)
            frequency = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
            angle = torch.outer(torch.arange(3, 4, dtype=torch.float64), frequency)
            low, high = x[:, :32], x[:, 32:]
            return torch.cat((low * angle.cos() - high * angle.sin(),
                              high * angle.cos() + low * angle.sin()), dim=1)

        def operator(stage, stages, incoming, caches):
            if stage in (0, 13):
                return norm(incoming if stage == 0 else stages[12],
                            "input_layernorm.weight" if stage == 0 else "post_attention_layernorm.weight")
            if stage in (1, 2, 3, 11, 14, 15, 17):
                operand, suffix, bias = {
                    1: (0, "self_attn.q_proj", True), 2: (0, "self_attn.k_proj", True),
                    3: (0, "self_attn.v_proj", True), 11: (10, "self_attn.o_proj", False),
                    14: (13, "mlp.gate_proj", False), 15: (13, "mlp.up_proj", False),
                    17: (16, "mlp.down_proj", False),
                }[stage]
                return linear(stages[operand], suffix, bias)
            if stage in (4, 5):
                return rope(stages[1 if stage == 4 else 2])
            if stage in (6, 7):
                return tensor(stages[5 if stage == 6 else 3])
            if stage == 8:
                q, k = tensor(stages[4]).reshape(14, 64), tensor(caches["k"])
                return torch.stack([q[h] @ k[:, h // 7].T / 8 for h in range(14)])
            if stage == 9:
                return torch.softmax(tensor(stages[8]).reshape(14, 4), dim=-1)
            if stage == 10:
                p, v = tensor(stages[9]).reshape(14, 4), tensor(caches["v"])
                return torch.stack([p[h] @ v[:, h // 7] for h in range(14)])
            if stage == 12:
                return tensor(stages[11]) + tensor(incoming)
            if stage == 16:
                return functional.silu(tensor(stages[14])) * tensor(stages[15])
            require(stage == 18, f"unknown operator: {stage}")
            return tensor(stages[12]) + tensor(stages[17])

        def reference_bits(raw):
            return raw.to(torch.float16).numpy().view("<u2").reshape(-1)

        stages = []
        for stage in range(19):
            independent_raw = operator(stage, independent, prior_independent, independent_caches)
            reconstructed = reference_bits(independent_raw)
            require(np.array_equal(reconstructed, independent[stage]),
                    f"independent retained-input reconstruction mismatch: {layer}/{stage}")
            actual_raw = operator(stage, actual, hidden, actual_caches)
            conditional = reference_bits(actual_raw)
            direct = half_bits(independent_raw.numpy())
            stages.append({
                "stage": stage, "actual_trace": transaction["raw"]["trace"]["path"],
                "independent": prepared["independent_stages"][str(stage)]["path"],
                "trajectory": compared(actual[stage], independent[stage]),
                "independent_operator_on_actual_operands": compared(actual[stage], conditional),
                "binary64_numpy_cast_on_actual_operands": compared(
                    actual[stage], half_bits(actual_raw.numpy())),
                "numpy_cast_vs_preserved_torch_cast": compared(direct, reconstructed),
                "independent_retained_operands_exact": True,
            })
            if layer == 2 and stage == 4:
                witness = 709
                raw = float(independent_raw.reshape(-1)[witness])
                require(int(direct[witness]) == 0xbd49 and int(reconstructed[witness]) == 0xbd4a,
                        "layer2 RoPE cast-discrimination regression changed")
                stages[-1]["cast_discrimination"] = {
                    "index": witness, "binary64": raw, "binary32": float(np.float32(raw)),
                    "numpy_direct_fp16": f"{int(direct[witness]):04x}",
                    "torch_fp16": f"{int(reconstructed[witness]):04x}",
                    "explicit_fp32_then_fp16": f"{int(half_bits(np.float32(raw))[0]):04x}",
                    "fp16_midpoint": -1.32177734375,
                }
        require(sum(len(s["trajectory"]["failure_indices"]) for s in stages)
                == receipt["material_mismatches"], f"unchanged gate regression: {layer}")
        layer_row = {
            "layer": layer, "input_actual": prepared["input_hidden"]["path"],
            "input_independent": (package["independent_layer1_hidden"]["path"] if layer == 2
                                  else plans[layer - 3][2]["independent_hidden"]["path"]),
            "incoming_hidden": compared(hidden, prior_independent), "stages": stages,
            "cache_parentage_exact": True,
            "cache_trajectory": {kind: compared(actual_caches[kind], independent_caches[kind])
                                 for kind in ("k", "v")},
            "first_local_policy_difference": next(
                (s["stage"] for s in stages if
                 s["independent_operator_on_actual_operands"]["different_count"]), None),
            "seconds": time.monotonic() - layer_start,
        }
        if layer == 8:
            other = evidence.load(B / "layer08/prepared.json")
            other_result = evidence.load(B / "layer08/result.json")
            other_transaction = evidence.load(other_result["transaction"]["path"])
            require(evidence.read(other_transaction["raw"]["trace"]["path"])
                    == evidence.read(transaction["raw"]["trace"]["path"]),
                    "attempt003/004 trace identity changed")
            require(evidence.read(other["vectors"]["input"]["path"])
                    == evidence.read(prepared["vectors"]["input"]["path"]),
                    "attempt003/004 hidden identity changed")
            require(all(np.array_equal(evidence.bits(other["independent_stages"][str(s)]["path"]),
                                       independent[s]) for s in range(19)),
                    "attempt003/004 independent trajectory changed")
            require(stages[8]["trajectory"]["failure_indices"] == [13, 15],
                    "layer08 original score failure changed")
            layer_row["layer08_score_witnesses"] = detail(actual[8], independent[8])["elements"]
            layer_row["layer08_q_witnesses"] = [
                {"index": i, "actual": float(values(actual[1])[i]),
                 "independent": float(values(independent[1])[i])} for i in (222, 223, 253)]
            layer_row["attempt003_004_same_retained_trajectory"] = True
        write_json(out / f"layer{layer:02d}.json", layer_row)
        rows.append(layer_row)
        prior_actual, prior_independent = actual[18], independent[18]
        print(json.dumps({
            "layer": layer, "incoming_differences": layer_row["incoming_hidden"]["different_count"],
            "final_differences": stages[18]["trajectory"]["different_count"],
            "local_policy_stages": [s["stage"] for s in stages
                                    if s["independent_operator_on_actual_operands"]["different_count"]],
            "material_failures": sum(len(s["trajectory"]["failure_indices"]) for s in stages),
        }), flush=True)

    summary = {
        "status": "UPSTREAM_LOCALIZATION_COMPLETE_NOT_RTL_ACCEPTANCE",
        "earliest_observed_boundary": boundary,
        "earliest_causal_operator": None,
        "reason": (
            "Drift is present at the authenticated layer1-output/layer2-input cut. "
            "Multiple retained-input operator policy differences coexist downstream. "
            "A first local bit difference is not proof of a necessary cause of the layer08 score failures. "
            "This stagewise diagnostic does not propagate counterfactuals across layers and cannot "
            "uniquely assign those failures to a prior operator."
        ),
        "smallest_supported_intervention": (
            "No production arithmetic, reference, cache, or input replacement is justified. "
            "The next causal diagnostic must begin before the differing layer2 input, using retained "
            "layer0/1 intermediates and independently owned caches; any proposed repair still needs "
            "an explicitly isolated downstream sensitivity calculation, not an unchanged RTL replay."
        ),
        "layers": [{"layer": row["layer"], "path": str(out / f"layer{row['layer']:02d}.json")}
                   for row in rows],
        "authenticated_inputs": list(evidence.consumed.values()),
        "frozen_gate_reproduced": True, "independent_stage_reconstructions": 19 * len(rows),
        "prefix_replays": 0, "RTL_invocations": 0, "numerical_policy_changes": 0,
        "independent_reviewer": "pending normal Host Reviewer",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "result.json", summary)
    write_json(out / "seal.json", {"files": [record(p) for p in sorted(out.iterdir())]})
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("authenticated_inputs", "layers", "earliest_observed_boundary")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent.parent == ROOT / "build"
            and output.parent.name.startswith("token358_position3_upstream_hidden_trajectory_localization"),
            "output outside diagnostic namespace")
    output.mkdir(exist_ok=False)
    try:
        diagnose(output, args.command.resolve())
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        write_json(output / "failure.json", {
            "taxonomy": "retained_evidence_diagnostic_failure",
            "hypothesis": str(error),
            "regression": "Authenticate and reconstruct the named retained boundary in a fresh diagnostic.",
            "RTL_invocations": 0,
        })
        raise
