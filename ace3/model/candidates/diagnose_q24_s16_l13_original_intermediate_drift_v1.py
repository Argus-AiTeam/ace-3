"""Source-bound L13 CPU intermediate control; never publishes a software parent.

From /home/argustest/ace3-argus:
  PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 \
    /home/argustest/miniconda3/bin/python -B -m \
    ace3.model.candidates.diagnose_q24_s16_l13_original_intermediate_drift_v1 \
    --input build/q24_software_l9_l23_4d1cfdfc15c0_attempt002 \
    --out build/q24_s16_l13_original_intermediate_drift_attempt001

Compiles the new files, executes only focused tests, authenticates retained
evidence, and records full-vector controls against the unchanged global gate.
"""

import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import py_compile
import sys
import time
from types import SimpleNamespace
import unittest
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_s18_first_failure_v1 as prior
from ace3.model.candidates import remaining_layers_v3 as references


ROOT = prior.ROOT
ID = "ace3-q24-s16-l13-original-intermediate-drift-v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_original_intermediate_drift_v1.json"
TEST_MODULE = "tests.test_q24_s16_l13_original_intermediate_drift_v1"
require = prior.require
record = prior.retained.record
write = prior.retained.write


def exact(value):
    return Fraction.from_float(float(value))


def same_binary64(actual, expected):
    require(actual.dtype == expected.dtype == np.dtype("<f8")
            and actual.shape == expected.shape == (896,)
            and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
            and np.array_equal(actual.view("<u8"), expected.view("<u8")),
            "original S18 does not reproduce the retained binary64 reference bit-for-bit")


def decompose(parent_i, s11_word, s17_word, s18_word, original, incoming, index):
    parent = Fraction(int(parent_i), 1 << 24)
    s11, s17, s18 = (prior.rational.fp16_value(int(word))
                     for word in (s11_word, s17_word, s18_word))
    ref = {key: exact(values[index]) for key, values in original.items()}
    parts = {
        "incoming_L12_Q24_drift": parent - ref["input"],
        "S11_drift": s11 - ref["s11"],
        "S17_drift": s17 - ref["s17"],
        "Q24_S18_RNE_projection": s18 - (parent + s11 + s17),
        "negative_original_binary64_addition_roundoff":
            ref["input"] + ref["s11"] + ref["s17"] - ref["s18"],
    }
    total = s18 - ref["s18"]
    require(sum(parts.values()) == total, "signed intermediate decomposition does not close")
    input_effect = exact(incoming["s18"][index]) - ref["s18"]
    local_effect = s18 - exact(incoming["s18"][index])
    require(input_effect + local_effect == total, "controlled decomposition does not close")
    return {
        "index": index, "signed_parts": {key: str(value) for key, value in parts.items()},
        "total_signed_error": str(total),
        "same_binary64_operator_input_effect": str(input_effect),
        "actual_minus_same_binary64_operator_on_Q24_parent": str(local_effect),
        "scratch_drift": str(parent + s11 - ref["residual"]),
        "original_residual_addition_roundoff":
            str(ref["residual"] - ref["input"] - ref["s11"]),
        "original_final_addition_roundoff":
            str(ref["s18"] - ref["residual"] - ref["s17"]),
    }


def capture(namespace, state, hidden):
    """Observe the unmodified recurrence's projections, not a rewritten oracle."""
    projection = namespace["_reference_projection"]
    observed = {}

    def observe(activation, operand):
        value = projection(activation, operand)
        for key, name in (("o", "s11"), ("down", "s17")):
            if operand is state.projections[key]:
                require(name not in observed, "duplicate reference projection")
                observed[name] = value.detach().numpy().reshape(-1).copy()
        return value

    require(state.reference_k.shape == state.reference_v.shape == (0, 2, 64),
            "control requires its own empty P0 cache")
    namespace["_reference_projection"] = observe
    try:
        with torch.no_grad():
            output = namespace["_reference_layer_step"](
                state, torch.from_numpy(hidden.copy().reshape(1, 896)), 0)
    finally:
        namespace["_reference_projection"] = projection
    require(set(observed) == {"s11", "s17"}, "missing bound S11/S17 observations")
    observed.update(input=hidden.copy(), residual=hidden + observed["s11"],
                    s18=output.numpy().reshape(-1).copy())
    same_binary64(observed["residual"] + observed["s17"], observed["s18"])
    require(state.reference_k.shape == state.reference_v.shape == (1, 2, 64),
            "reference control P0 KV shape mismatch")
    for value in observed.values():
        require(value.dtype == np.dtype("<f8") and value.shape == (896,)
                and np.all(np.isfinite(value)), "invalid binary64 intermediate")
    return observed


def attribution(baseline, input_control):
    require(baseline["accepted"] is False, "expected retained index 62 failure")
    return {
        "classification": "inconclusive",
        "incoming_drift_sufficient_in_binary64_then_RNE_control":
            not input_control["accepted"],
        "rationale": (
            "Original S11/S17/residual/S18 and the same binary64 recurrence on actual "
            "Q24 L12 values are now available. Exact signed decompositions close. "
            "This isolates an incoming-parent effect within that binary64 control, "
            "not a forced outcome of the native FP16/RTZ operators. The complementary "
            "local term includes nonlinear interactions, FP16 operator boundaries, "
            "native arithmetic and final projection; it is not a rounding-only defect."
        ),
        "missing_evidence": [
            "A full-vector same-native-S16-RTZ L13 operator control with a separately "
            "specified non-admitted original-input parent mapping: the original "
            "binary64 L12 vector is not itself a valid exact Q24 I/Z/H parent. "
            "This diagnostic does not invent or admit that mapping.",
            "If unique local rounding attribution is required, paired native "
            "operator interventions separating individual L13 arithmetic/boundary "
            "effects and their interaction with the changed incoming parent.",
        ],
    }


def diagnose(input_dir, out):
    started = time.monotonic()
    baseline = prior.diagnose(input_dir)
    authenticated = {item["path"]: item for item in baseline["input_bindings"]}

    def bind(item):
        path = Path(item["path"])
        require(path.resolve().is_relative_to(ROOT), f"non-repository binding: {path}")
        if str(path) not in authenticated:
            require(record(path) == item, f"source/operand binding mismatch: {path}")
            authenticated[str(path)] = item
        else:
            require(authenticated[str(path)] == item, f"conflicting binding: {path}")
        return path

    def document(item):
        return json.loads(bind(item).read_text())

    freeze = document(authenticated[str(prior.INPUT / "freeze.json")])
    extension = document(freeze["reference_extension"])
    expected64, expected16 = extension["original_binary64_parent"], extension["original_fp16_parent"]
    for layer in range(9, 14):
        item = extension["layers"][str(layer)]
        require(item["input_binary64"] == expected64 and item["input_fp16"] == expected16
                and item["prior_kv"] == "own empty P0", "spliced original reference trajectory")
        bind(item["binary64"])
        bind(item["fp16"])
        expected64, expected16 = item["binary64"], item["fp16"]
    specification = document(extension["original_specification"])
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
    require(packages == extension["python_packages"] == specification["python_packages"],
            "independent oracle package mismatch")
    namespace = {
        "np": np, "torch": torch, "torch_functional": functional, "math": math,
        "AWQ_REVERSE_ORDER": (0, 4, 1, 5, 2, 6, 3, 7), "GROUP_SIZE": 128,
        "HEAD_DIM": 64, "HIDDEN_SIZE": 896, "QUERY_HEADS": 14, "KEY_VALUE_HEADS": 2,
    }
    for suffix, names in (
        ("/official_single_decoder_layer.py", ("_torch_unpack", "_torch_rmsnorm")),
        ("/official_model24_dialogue.py", ("_reference_projection", "_reference_layer_step")),
    ):
        matches = [r for r in extension["generators"] if r["path"].endswith(suffix)]
        require(len(matches) == 1 and matches[0] in specification["sources"],
                f"unbound independent generator: {suffix}")
        bind(matches[0])
        references.definitions(matches[0], names, namespace)
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {name: np.ascontiguousarray(model.get_tensor(name))
                   for name in prior.local.tensor_shapes(13)}
    prior.local.authenticate_tensors(tensors, item["canonical"], 13)
    torch.set_num_threads(1)
    projections = {}
    for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                        ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
        prefix = f"model.layers.13.{suffix}"
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    preparation_seconds = time.monotonic() - started
    with np.load(bind(baseline["operand_bindings"]["L12_parent"]), allow_pickle=False) as data:
        parent = {key: data[key].copy() for key in data.files}
    with np.load(bind(baseline["operand_bindings"]["L13_actual"]), allow_pickle=False) as data:
        actual = {key: data[key].copy() for key in data.files}
    original_input = np.load(bind(item["input_binary64"]), allow_pickle=False)
    global_reference = np.load(bind(item["binary64"]), allow_pickle=False)
    q24_input = parent["i"].astype("<f8") / (1 << 24)
    require(all(exact(value) == Fraction(int(integer), 1 << 24)
                for value, integer in zip(q24_input, parent["i"])),
            "Q24 parent not exactly representable by this binary64 control")
    q24_input[(parent["i"] == 0) & (parent["z"] != 0)] = -0.0
    controls, timings = {}, {}
    for name, hidden in (("original", original_input), ("incoming_Q24", q24_input)):
        state = SimpleNamespace(
            input_norm=tensors["model.layers.13.input_layernorm.weight"],
            post_attention_norm=tensors["model.layers.13.post_attention_layernorm.weight"],
            projections=projections,
            reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
            reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
        begin = time.monotonic()
        controls[name] = capture(namespace, state, hidden)
        timings[name] = time.monotonic() - begin
    same_binary64(controls["original"]["s18"], global_reference)
    incoming = controls["incoming_Q24"]
    require(np.all(np.abs(incoming["s18"]) <= 65504), "control S18 outside FP16 finite range")
    incoming_words = incoming["s18"].astype("<f2").view("<u2")
    rows = []
    for index in range(896):
        row = decompose(parent["i"][index], actual["stage11"][index], actual["stage17"][index],
                        actual["stage18"][index], controls["original"], incoming, index)
        row["actual_gate"] = prior.measure(int(actual["stage18"][index]), float(global_reference[index]))
        row["incoming_binary64_then_RNE_gate"] = prior.measure(
            int(incoming_words[index]), float(global_reference[index]))
        rows.append(row)
    reports = document(authenticated[str(input_dir / "layer13/reports.json")])
    failure_indices = [row["index"] for row in rows if not row["actual_gate"]["accepted"]]
    require(failure_indices == [row["index"] for row in reports[18]["binary64_v1"]["failures"]],
            "full-vector mandatory failures changed")
    write(out / "coordinate_decomposition.json", rows)
    np.savez(out / "reference_intermediates.npz",
             **{f"{name}_{stage}": values for name, values_by_stage in controls.items()
                for stage, values in values_by_stage.items()})
    write(out / "retained_control.json", baseline)
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [13, 0, 18], "index": 62,
        "candidate_admitted": False, "rtl_invocations": 0, "policy_adopted": False,
        "successor_published": False, "normal_host_review": "REQUIRED",
        "retained_status": "FAIL", "lineage": baseline["lineage"],
        "original_S18_bitwise_reproduction": True,
        "material_set": {"selection": "all 896 output coordinates, without prefiltering",
                         "count": len(rows), "mandatory_failure_indices": failure_indices},
        "selected_coordinate": rows[62],
        **attribution(rows[62]["actual_gate"], rows[62]["incoming_binary64_then_RNE_gate"]),
        "input_bindings": list(authenticated.values()), "python_packages": packages,
        "artifacts": [record(out / name) for name in (
            "coordinate_decomposition.json", "reference_intermediates.npz", "retained_control.json")],
        "timing_seconds": {"authentication_local_gates_and_weights": preparation_seconds, **timings,
                           "total_diagnosis": time.monotonic() - started},
        "claim_boundary": "Binary64 parent A/B is diagnostic only, not a Q24 successor or "
                          "reference replacement. Native INT4/FP16 operators/KV and wide Q24 "
                          "state remain unchanged. No RTL, hardware, strict-FP16-state W4A16, "
                          "new-token or full-model PASS. No dominant runtime-stage claim.",
    }


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT) and sys.dont_write_bytecode
            and not sys.flags.optimize, "use published repo-bound command context")
    tests = importlib.import_module(TEST_MODULE)
    legacy = "tests.test_q24_s16_toward_zero_l3_l8_v1"
    importlib.import_module(legacy)
    origins = {}
    for name, module in sorted(list(sys.modules.items())):
        if name.startswith("ace3.model.candidates.") or name in (TEST_MODULE, legacy):
            expected = ROOT.joinpath(*name.split(".")).with_suffix(".py")
            require(Path(module.__file__).resolve() == expected, f"module origin mismatch: {name}")
            origins[name] = record(expected)
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["rtl_invocations"] == 0
            and contract["policy_adopted"] is False and contract["successor_published"] is False
            and contract["normal_host_review"] == "REQUIRED", "diagnostic contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    validation = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": origins, "compiled": compiled, "contract": record(CONTRACT),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False,
    }
    write(out / "validation.json", validation)
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l13_original_intermediate_drift_"),
            "output outside bounded build scope")
    require(args.input.resolve() == prior.INPUT, "unselected retained input")
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({"cwd": str(Path.cwd()), "executable": sys.executable,
                              "argv": sys.argv, "PYTHONPATH": os.environ.get("PYTHONPATH"),
                              "loadavg": os.getloadavg(), "torch_threads_before": torch.get_num_threads()}) + "\n")
        log.flush()
        try:
            validation = validate(out)
            result = diagnose(args.input.resolve(), out)
            result["validation"] = record(out / "validation.json")
            result["tests_executed"] = validation["executed"]
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"],
                "candidate_admitted": False, "rtl_invocations": 0, "policy_adopted": False,
                "successor_published": False, "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {key: result[key] for key in ("status", "classification", "candidate_admitted")}
        summary["result"] = str(out / "result.json")
        if result["status"] == "BLOCKED":
            summary["missing_evidence"] = result["missing_evidence"]
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
