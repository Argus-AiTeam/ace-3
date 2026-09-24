#!/usr/bin/env python3
"""Software-only fixed-history B continuation; never a candidate admission."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ace3/model"
B = ROOT / "build/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010"
SPLIT = ROOT / "build/layer08_position2_rope_split_86938809f0a2_attempt007"
REFERENCE = ROOT / "build/independent_fp16_trajectory_20260906_1133/source"
TOKENS = [9707, 1879, 0, 358]
COUNTS = [896, 896, 128, 128, 896, 128, 128, 128, 56, 56,
          896, 896, 896, 896, 4864, 4864, 4864, 896, 896]
POLICY = {
    "accept": "finite AND (abs_error<=0.125 OR (relative_error<0.001 AND orderedFP16_ULP<=1))",
    "absolute_tolerance": 0.125, "relative_tolerance_strict": 0.001,
    "max_ulp_distance": 1, "relative_denominator": "max(abs(reference),2^-14)",
}


class BindingError(RuntimeError):
    pass


class MaterialFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BindingError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == set(names), f"missing functions: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    bindings, comparisons, history, outputs = {}, [], [], []
    active = None
    write(out / "preregistered.json", {
        "fixed_experiment_inputs": TOKENS, "candidate_native_generation": False,
        "profile": "B: general Q24 exponential, Q-only fused-Q48 RoPE; K/V/SiLU unchanged",
        "layers": list(range(9)), "position": 3, "stages": list(range(19)),
        "stage_record_counts": COUNTS, "comparison_policy": POLICY,
        "stop": "first material stage failure; no subsequent candidate arithmetic",
        "software_contract": "(values, activation, position, cache_k, cache_v, accurate_silu=True, accepted_q_projection=True)",
        "rtl_execution": False, "binary64_v1": "NOT_EVALUATED; six known B failures remain FAIL",
        "source": record(Path(__file__)), "launch": record(out / "run.command.sh"),
        "accepted_freeze_sha256": "03ab838054fe8318d369e8f260801c1e0193ad1c43720133afb96fbb5b90f785",
        "independent_review": 1, "self_maintenance": 0,
    })

    def bind(item):
        actual = record(item["path"])
        require(all(actual[key] == item[key] for key in ("bytes", "sha256")),
                f"binding mismatch: {item['path']}; expected={item}; actual={actual}")
        bindings[actual["path"]] = actual
        return Path(actual["path"])

    def load(item):
        return json.loads(bind(item).read_text())

    try:
        frozen_record = record(B / "frozen.json")
        require(frozen_record["sha256"] ==
                "03ab838054fe8318d369e8f260801c1e0193ad1c43720133afb96fbb5b90f785",
                "accepted B freeze identity changed")
        frozen = load(frozen_record)
        result = load(record(B / "result.json"))
        require(result["freeze"] == frozen_record
                and result["status"] == "PASS_BOUNDED_RTL_CANDIDATE"
                and result["position3_executed"] is False
                and len(result["transactions"]) == 27, "B result scope mismatch")
        review = load(record(Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
                             / "86938809f0a2/round-0009.json"))
        require(review["producer_role"] == "reviewer" and review["round"] == 9
                and review["mission_id"] == "86938809f0a2"
                and review["review"]["status"] == "done", "B independent review mismatch")
        require(frozen["token_history"] == TOKENS[:3]
                and frozen["comparison_policy"] == POLICY, "fixed history/policy mismatch")
        software_freeze = load(frozen["software_freeze"])
        software_result = load(frozen["software_result"])
        require(software_result["freeze"] == frozen["software_freeze"]
                and software_result["all_affected_stage_gates_clear"] == ["q_only"],
                "split selection mismatch")
        expected = {item["path"]: item for item in
                    software_freeze["inputs_and_sources"] + frozen["retained_source_closure"]}

        def source(path):
            path = path.resolve()
            require(str(path) in expected, f"no accepted source binding: {path}")
            return bind(expected[str(path)])

        for name in ("decoder_layer0_oracle", "attention_oracle", "awq_bit_oracle",
                     "fp16_adaptation_oracle", "qwen2_rope_oracle",
                     "accepted_awq_projection", "projection_oracle"):
            source(MODEL / f"{name}.py")
        for item in frozen["source_closure"]:
            bind(item)
        bind(frozen["python"])
        require(Path(sys.executable).resolve() == Path(frozen["python"]["path"]),
                "accepted Python executable differs")
        for name, version in frozen["python_packages"].items():
            require(importlib.metadata.version(name) == version, f"{name} version drift")
        checkpoint = bind(frozen["checkpoint"])
        sys.path.insert(0, str(MODEL))
        import numpy as np
        import torch
        import torch.nn.functional as torch_functional
        from safetensors import safe_open
        from awq_bit_oracle import AWQ_REVERSE_ORDER, q47_48_to_f16
        from fp16_adaptation_oracle import decode_f16_q24, residual_add
        from qwen2_rope_oracle import qwen2_coefficient

        torch.set_num_threads(1)
        candidate_path = source(ROOT / "build/layer08_position3_softmax_exp_replay_attempt003"
                                "/dependency_closure/candidate_oracle.py")
        candidate = module("fixed_history_b_candidate", candidate_path)
        operator_source = source(ROOT / "build/layer08_position2_operator_factorial_86938809f0a2_attempt001"
                                 "/experiment.py")
        operator_namespace = {
            "np": np, "AWQ_REVERSE_ORDER": AWQ_REVERSE_ORDER,
            "decode_f16_q24": decode_f16_q24, "residual_add": residual_add,
            "q47_48_to_f16": q47_48_to_f16, "qwen2_coefficient": qwen2_coefficient,
            "d": sys.modules[__name__],
        }
        definitions(operator_source, ("unpack", "projection", "fused_rope"), operator_namespace)
        reference_namespace = {
            "np": np, "torch": torch, "torch_functional": torch_functional,
            "AWQ_REVERSE_ORDER": AWQ_REVERSE_ORDER, "GROUP_SIZE": 128,
            "HEAD_DIM": 64, "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
            "HIDDEN_SIZE": 896, "_require": require, "require": require,
            "DiagnosticError": BindingError,
        }
        definitions(source(REFERENCE / "official_single_decoder_layer.py"),
                    ("_torch_unpack", "_torch_linear", "_torch_rmsnorm"), reference_namespace)
        definitions(source(REFERENCE / "propagated_reference.py"),
                    ("propagated_reference",), reference_namespace)
        comparator_path = MODEL / "diagnose_layer01_position2_stage08.py"
        comparator = module("fixed_history_gate", comparator_path)
        bindings[str(comparator_path)] = record(comparator_path)

        def bits(path, width=4):
            rows = path.read_text("ascii").splitlines()
            require(all(len(row) == width for row in rows), f"hex schema mismatch: {path}")
            return np.array([int(row, 16) for row in rows], dtype="<u2")

        def decoded(rows, stage, position):
            if stage in (8, 9):
                require([index for index, _ in rows] ==
                        list(range(position + 1)) * 14, "score/probability head order mismatch")
            else:
                rows = sorted(rows)
                require([index for index, _ in rows] == list(range(len(rows))),
                        "stage indices incomplete or duplicated")
            return np.array([value for _, value in rows], dtype="<u2")

        stage_arrays, reference_arrays = {}, {}
        software_outputs = {item["path"]: item for item in software_result["outputs"]}
        with safe_open(checkpoint, framework="numpy") as model:
            embedding = model.get_tensor("model.embed_tokens.weight")
            require(embedding.dtype == np.float16 and embedding.shape[1] == 896,
                    "official embedding representation mismatch")
            for item in frozen["embeddings"]:
                rows = bind(item["input"]).read_text("ascii").splitlines()
                actual = np.array([int(row[-4:], 16) for row in rows], dtype="<u2")
                require(np.array_equal(actual, embedding[item["token"]].view("<u2")),
                        "historical fixed embedding mismatch")
            current = embedding[358].copy().view("<u2")
            reference_hidden = torch.from_numpy(current.view("<f2").astype(np.float64)).reshape(1, 896)
            tensor_values = {}
            for item in frozen["checkpoint_tensors"]:
                value = model.get_tensor(item["name"])
                require(list(value.shape) == item["shape"] and str(value.dtype) == item["dtype"]
                        and hashlib.sha256(value.tobytes()).hexdigest() == item["sha256"],
                        f"official tensor mismatch: {item['name']}")
                tensor_values[item["name"]] = value
        for transaction in result["transactions"]:
            layer, position = transaction["layer"], transaction["position"]
            key = layer, position
            row = load(transaction["result"])
            prep = load(row["prepared"])
            raw_record = row["transaction"]["raw"]["trace"]
            raw = bind(raw_record).read_text("ascii").splitlines()
            groups = {}
            for line in raw:
                require(len(line) == 16 and int(line[:2], 16) == 0
                        and int(line[2:6], 16) == position, "B trace owner mismatch")
                groups.setdefault(int(line[6:8], 16), []).append(
                    (int(line[8:12], 16), int(line[12:], 16)))
            require(set(groups) == set(range(19)), "incomplete B stage export")
            stage_arrays[key] = {stage: decoded(rows, stage, position)
                                 for stage, rows in groups.items()}
            software_path = SPLIT / f"q_only_layer{layer:02d}_position{position:03d}.npz"
            with np.load(bind(software_outputs[str(software_path)]), allow_pickle=False) as arrays:
                for stage, actual in stage_arrays[key].items():
                    require(np.array_equal(actual, arrays[f"stage{stage:02d}"]),
                            f"B RTL/software representation mismatch: {key}/stage{stage}")
            reference_arrays[key] = {
                int(stage): bits(bind(item)) for stage, item in row["independent_stages"].items()}
            require(set(reference_arrays[key]) == set(range(19)), "reference export incomplete")
            for field, arrays in (("input_hidden", stage_arrays),
                                  ("independent_input", reference_arrays)):
                rows = bind(prep[field]).read_text("ascii").splitlines()
                hidden = np.array([int(line[-4:], 16) for line in rows], dtype="<u2")
                parent = (embedding[TOKENS[position]].view("<u2") if layer == 0
                          else arrays[layer - 1, position][18])
                require(np.array_equal(hidden, parent), f"{field} lineage mismatch: {key}")
            history.append({"layer": layer, "position": position,
                            "accepted_transaction": transaction["result"],
                            "software_stage_archive": software_outputs[str(software_path)],
                            "all_19_stages_bit_exact": True})

        write(out / "frozen.json", {
            "preregistration": record(out / "preregistered.json"),
            "bindings": list(bindings.values()), "historical_representation": history,
            "history_policy": "B RTL stage06/stage07 decoded FP16, bit-exact with accepted q_only software; independent caches use only independent stage06/stage07",
            "reference_policy": "source-bound independently propagated FP16 recurrence; position_offset=3",
            "fixed_embedding_token358_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
            "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
            "instrumentation": "replace only run_token trace initialization with a stage-gating list; arithmetic AST and globals retained",
        })
        tree = ast.parse((MODEL / "decoder_layer0_oracle.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_token")
        trace_nodes = [node for node in ast.walk(function)
                       if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                       and node.target.id == "trace"]
        require(len(trace_nodes) == 1 and isinstance(trace_nodes[0].value, ast.List),
                "trace instrumentation contract changed")
        trace_nodes[0].value = ast.Call(func=ast.Name(id="Trace", ctx=ast.Load()), args=[], keywords=[])
        instrumented = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        baseline_rope = candidate.run_token.__globals__["_rope"]
        latest_trace = None
        failure = None
        for layer in range(9):
            active = {"layer": layer, "position": 3, "stage": 0}
            prefix = f"model.layers.{layer}."
            tensors = {name: value for name, value in tensor_values.items() if name.startswith(prefix)}
            values = {name.replace(prefix, "model.layers.0.", 1) + ":":
                      value.reshape(-1).view("<u2" if value.dtype == np.float16 else "<u4")
                      for name, value in tensors.items()}
            keys = [stage_arrays[layer, p][6].astype(int).tolist() for p in range(3)]
            vals = [stage_arrays[layer, p][7].astype(int).tolist() for p in range(3)]
            ref_keys = torch.from_numpy(np.stack([reference_arrays[layer, p][6].view("<f2")
                                                 for p in range(3)]).astype(np.float64)).reshape(3, 2, 64)
            ref_vals = torch.from_numpy(np.stack([reference_arrays[layer, p][7].view("<f2")
                                                 for p in range(3)]).astype(np.float64)).reshape(3, 2, 64)
            refs, _, _ = reference_namespace["propagated_reference"](
                reference_hidden, tensors, layer, cached_k=ref_keys, cached_v=ref_vals,
                position_offset=3, reference_policy="w4a16_fp16_interstage")
            reference_bits = {stage: np.asarray(value, dtype="<f2").view("<u2")
                              for stage, value in refs[0].items()}

            class Trace(list):
                def __init__(self):
                    super().__init__()
                    self.groups = {}
                    self.next_stage = 0

                def append(self, item):
                    stage, index, value, position = item
                    require(position == 3, "candidate trace position mismatch")
                    super().append(item)
                    self.groups.setdefault(stage, []).append((index, value))
                    while (self.next_stage < 19 and
                           len(self.groups.get(self.next_stage, [])) == COUNTS[self.next_stage]):
                        stage = self.next_stage
                        active["stage"] = stage
                        actual = decoded(self.groups[stage], stage, 3)
                        expected_bits = reference_bits[stage]
                        gate = comparator.comparison(actual, expected_bits)
                        comparisons.append({"layer": layer, "position": 3, "stage": stage,
                                            "comparison": gate})
                        if not gate["within_tolerance"]:
                            raise MaterialFailure(f"L{layer}/P3/S{stage}: {gate}")
                        self.next_stage += 1

                def extend(self, items):
                    for item in items:
                        self.append(item)

            latest_trace = Trace()
            namespace = dict(candidate.run_token.__globals__)
            namespace.update({
                "Trace": lambda: latest_trace,
                "_module": operator_namespace["projection"](values, False),
                "_rope": lambda vector, heads, position:
                    (operator_namespace["fused_rope"] if heads == 14 else baseline_rope)(
                        vector, heads, position),
            })
            exec(compile(instrumented, str(MODEL / "decoder_layer0_oracle.py"), "exec"), namespace)
            operands = {"actual_input": current.copy(),
                        "reference_input": reference_hidden.numpy().copy(),
                        "actual_history_k": np.asarray(keys, dtype="<u2"),
                        "actual_history_v": np.asarray(vals, dtype="<u2"),
                        "reference_history_k": ref_keys.numpy().copy(),
                        "reference_history_v": ref_vals.numpy().copy()}
            try:
                final, _ = namespace["run_token"](
                    {name: value.astype(int).tolist() for name, value in values.items()
                     if not name.endswith((".qweight:", ".qzeros:", ".scales:"))},
                    current.astype(int).tolist(), 3, keys, vals,
                    accurate_silu=True, accepted_q_projection=True)
                require(latest_trace.next_stage == 19, "incomplete measured layer")
            except MaterialFailure as exc:
                failure = {"active": dict(active), "detail": str(exc),
                           "failure_taxonomy": "upstream_trajectory_divergence",
                           "root_cause_hypothesis": "B own-input and independent FP16 trajectories diverge; unique producer not established",
                           "regression": "ordered fixed-history B position3 stages under unchanged finite/absolute/relative/ULP gate"}
            arrays = {f"actual_stage{stage:02d}": decoded(rows, stage, 3)
                      for stage, rows in latest_trace.groups.items()
                      if len(rows) == COUNTS[stage]}
            arrays.update({f"reference_stage{stage:02d}": value for stage, value in reference_bits.items()})
            path = out / f"layer{layer:02d}_position003.npz"
            with path.open("xb") as stream:
                np.savez(stream, **operands, **arrays,
                         partial_actual_trace=np.asarray(latest_trace, dtype=np.int64))
            outputs.append(record(path))
            write(out / f"layer{layer:02d}_comparisons.json",
                  [row for row in comparisons if row["layer"] == layer])
            print(f"layer={layer} measured_stages={latest_trace.next_stage} failure={failure is not None}", flush=True)
            if failure is not None:
                break
            current = np.array(final, dtype="<u2")
            reference_hidden = torch.from_numpy(refs[0][18]).reshape(1, 896)
        witnesses = "NOT_REACHED"
        if active["layer"] == 8 and latest_trace.next_stage > 8:
            actual = decoded(latest_trace.groups[8], 8, 3)
            witnesses = {str(index): comparator.comparison(actual[index:index + 1],
                                                          reference_bits[8][index:index + 1])
                         for index in (13, 15)}
        write(out / "result.json", {
            "status": "SOFTWARE_FP16_MATERIAL_FAILURE" if failure else "SOFTWARE_FIXED_HISTORY_FP16_CLEAR",
            "freeze": record(out / "frozen.json"), "failure": failure,
            "comparisons": comparisons, "outputs": outputs,
            "layer08_position3_stage08_witnesses13_15": witnesses,
            "fixed_experiment_inputs": TOKENS, "position3_executed": True,
            "new_rtl_transactions": 0, "historical_transactions_reexecuted": 0,
            "historical_software_reconstruction": False, "candidate_native_generation": False,
            "candidate_admitted": False, "binary64_v1": "NOT_EVALUATED; original six B scalar failures in five outputs remain FAIL",
            "independent_review": "PENDING_NORMAL_HOST_REVIEWER",
            "elapsed_seconds": time.monotonic() - started,
        })
    except (BindingError, FileNotFoundError, KeyError, ImportError) as exc:
        write(out / "result.json", {
            "status": "AUTHENTICATED_REPRESENTATION_REFERENCE_BLOCKER",
            "failure_taxonomy": "state_reference_representation",
            "blocker": f"{type(exc).__name__}: {exc}", "active": active,
            "authenticated_bindings": list(bindings.values()), "comparisons": comparisons,
            "root_cause_hypothesis": "required source/state/reference contract does not match the retained B interface",
            "regression": "authenticate B source, independent recurrence and bit-exact software/RTL history before propagation",
            "new_rtl_transactions": 0, "candidate_admitted": False,
            "binary64_v1": "NOT_EVALUATED; known FAILs unchanged",
            "independent_review": "PENDING_NORMAL_HOST_REVIEWER",
        })
        raise


if __name__ == "__main__":
    main()
