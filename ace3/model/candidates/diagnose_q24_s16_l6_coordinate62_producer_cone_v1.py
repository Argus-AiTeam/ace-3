"""Frozen L6/P0 coordinate-62 producer cuts; non-admitting native L7-L13 CPU suffix.

From /home/argustest/ace3-argus, run once with a fresh bounded output:
PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 \
/home/argustest/miniconda3/bin/python -B -m \
ace3.model.candidates.diagnose_q24_s16_l6_coordinate62_producer_cone_v1 \
--out build/q24_s16_l6_coordinate62_producer_cone_<fresh-name>

The entry point compiles both repository files and executes the focused tests
once before diagnosis. It does not rerun the accepted 19-test prefix suite.
Use the durable direct runner for this over-two-minute command. Host independent
review must reproduce it with another exclusive output; this is not admission.
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
import traceback
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l7_coordinate62_producer_cone_v1 as previous


producer, coordinate, upstream = previous.producer, previous.coordinate, previous.upstream
paired, prior, native = previous.paired, previous.prior, previous.native
ROOT = previous.ROOT
ID = "ace3-q24-s16-l6-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l6_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l6_coordinate62_producer_cone_v1"
REVIEWED = ROOT / "build/q24_s16_l7_coordinate62_producer_cone_6b7120b2b23e_attempt001"
REVIEWED_SHA = "d049db823487bf591007ab0e2819adf8bb35a2420c86e0a196fd9fce9a4dcd3a"
REPRODUCED = ROOT / "build/q24_s16_l7_coordinate62_producer_cone_reviewer_round001"
REPRODUCED_SHA = "c4f940de5289730998932269cf1ca48195cf50ca95ee627bd5e047b6670030fe"
require, record, write = prior.require, paired.record, paired.write
BRANCH_DEFINITIONS = {
    "inherited": "L5 output/L6 input complete I/Z/H coordinate 62, mapped nearest-Q24 "
                 "ties-even from original-input L5 binary64; freeze actual L6 O and down.",
    "o": "L6 S11 coordinate 62 mapped direct RNE-FP16 from original-input L6 O.",
    "S12": "Exact incoming Q24 plus actual FP16 S11 O, retaining I/Z/H and RNE16 H.",
    "down": "L6 S17 coordinate 62 mapped direct RNE-FP16 from original-input L6 down.",
    "S18": "Exact unrounded S12 Q24 plus FP16 down; unchanged final RNE16 projection.",
    "scratch": "Complete L6 S12 coordinate 62 mapped nearest-Q24 from original-input "
               "L6 residual; actual down is frozen, not recomputed.",
    "scratch_down": "Combine the mapped S12 and original-input RNE-FP16 down cuts.",
    "mapped62": "Complete L6 S18 coordinate 62 mapped nearest-Q24 from original-input L6.",
}
CLAIM_BOUNDARY = (
    "Only frozen L6/P0 coordinate-62 producer mappings and unchanged native L7-L13 CPU "
    "suffixes. INT4 native AWQ G128 asymmetric GEMM order, no qzero plus-one, FP16 "
    "scales/operator boundaries/KV, Q24 residuals, thresholds and original-input "
    "global reference are unchanged. No L0-L6 native replay, accepted test replay, "
    "L14+, strict-FP16-state W4A16, new-token, full-model PASS, RTL, hardware, GPU, "
    "FPGA, simulation, synthesis, PPA, performance attribution, model/profile/budget/"
    "configuration change, policy adoption, successor publication or ACE2 changes."
)


def plan():
    return previous.plan()


def origins():
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name not in ("ace3", "tests") and not name.startswith(("ace3.", "tests.")):
            continue
        expected = ROOT.joinpath(*name.split("."))
        filename = getattr(module, "__file__", None)
        if filename is None:
            paths = list(getattr(module, "__path__", ()))
            require(paths and all(Path(p).resolve() == expected for p in paths),
                    f"namespace origin mismatch: {name}")
            result[name] = {"namespace_paths": paths}
        else:
            path = Path(filename).resolve()
            require(path.is_relative_to(ROOT)
                    and path in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {name}")
            result[name] = record(path)
    require(MODULE in result and TEST_MODULE in result, "missing candidate/test origins")
    return result


def validate(out):
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "entry source origin mismatch")
    tests = importlib.import_module(TEST_MODULE)
    source_origins = origins()
    compiled = []
    for index, path in enumerate((Path(__file__).resolve(), Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(record(path))
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(out / "validation.json", {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "accepted_L0_L8_tests_executed": False,
    })
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")


def authenticate_reviewed(reviewed):
    require(reviewed["diagnostic_id"] == previous.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["classification"] == "conditional_inherited_L6_branch_sufficiency"
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["original_L7_S18_bitwise_reproduction"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["rtl_invocations"] == 0
            and all(reviewed[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "reviewed L7 scientific parent mismatch")


def authenticate():
    data = previous.authenticate()
    bind = lambda item: upstream.bind_input(item, data["bound"])
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    for directory, digest in ((REVIEWED, REVIEWED_SHA), (REPRODUCED, REPRODUCED_SHA)):
        item = record(directory / "result.json")
        require(item["sha256"] == digest, "wrong reviewed L7 result")
        reviewed = read(item)
        authenticate_reviewed(reviewed)
        for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
            bind(item)
        for item in reviewed["origins_after_execution"].values():
            if "path" in item:
                bind(item)
    extension_freeze = read(record(prior.INPUT / "freeze.json"))
    accepted = extension_freeze["state_lineage"]["prior_residual"]
    result = read(accepted["result"])
    freeze = read(accepted["parent"]["arithmetic_lineage"])
    entries = {item["layer"]: item for item in result["layers"]}
    require(entries[5]["output_parent"] == entries[6]["input_parent"]
            and entries[6]["output_parent"] == entries[7]["input_parent"],
            "accepted L5/L6/L7 parent splice")
    for layer in (5, 6):
        item = entries[layer]
        incoming, arrays = archive(item["input_parent"]), archive(item["actual_stages"])
        for stage in range(19):
            prior.retained.check_stage_state(stage, arrays, incoming)
        output = archive(item["output_parent"])
        paired.same_arrays(output, prior.retained.state_from(arrays, "output", "stage18"))
        paired.same_arrays(archive(item["kv_state"]),
                           {k: arrays["output_cache_" + k] for k in ("k", "v")})
        reports = read(item["reports"])
        require(len(reports) == 19 and all(
            r["node"] == [layer, 0, s] and r["status"] == "PASS"
            and r["policy_id"] == prior.gates.POLICY_ID
            and r["residual_state_lineage"] == r["kv_lineage"] == "PASS"
            for s, r in enumerate(reports)), "accepted L5/L6 stage/state/KV gates changed")
        data[f"L{layer}"] = {
            "parent": incoming, "arrays": arrays, "output": output, "reports": reports}
    paired.same_arrays(data["L5"]["output"], data["L6"]["parent"])
    paired.same_arrays(data["L6"]["output"], data["L7"]["parent"])
    control = read(freeze["global_reference_lineage"]["control"])
    model_freeze = read(freeze["global_reference_lineage"]["model"])
    for layer in (5, 6, 7):
        cases = [c for c in control["cases"] if c["layer"] == layer and c["position"] == 0]
        require(len(cases) == 1, "ambiguous original-input reference")
        case = cases[0]
        reference = np.load(bind(case["binary64_reference_array"]), allow_pickle=False)
        paired.drift.same_binary64(reference, reference)
        if layer > 5:
            paired.drift.same_binary64(reference, data[f"original{layer}"])
        data[f"original{layer}"] = reference
        if layer == 7:
            trajectory = {}
            for stage in range(19):
                path = str(Path(case["fp16_reference"]["path"]).with_name(f"stage{stage:02d}.hex"))
                matches = [r for r in freeze["input_bindings"] if r["path"] == path]
                require(len(matches) == 1, "missing original FP16 trajectory binding")
                trajectory[f"stage{stage:02d}"] = np.array(
                    [int(word, 16) for word in bind(matches[0]).read_text().split()], dtype="<u2")
            bind(case["fp16_reference"])
            data["L7"].update(trajectory=trajectory, reference=reference,
                              locals=archive(entries[7]["local_references"]))
    data["layers"][7] = data["L7"]
    with safe_open(str(bind(data["extension"]["checkpoint"])), framework="numpy") as model:
        tensors = {k: np.ascontiguousarray(model.get_tensor(k)) for k in prior.local.tensor_shapes(6)}
    prior.local.authenticate_tensors(
        tensors, {r["name"]: r for r in model_freeze["checkpoint_tensors"]}, 6)
    data["L6"]["tensors"] = tensors
    data["review_identity_basis"].update(
        L7_repository_result=record(REVIEWED / "result.json"),
        L7_reviewer_reproduction=record(REPRODUCED / "result.json"))
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["L6"]["tensors"]
    bind = lambda item: upstream.bind_input(item, data["bound"])
    specification = json.loads(bind(extension["original_specification"]).read_text())
    require({name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
            == extension["python_packages"] == specification["python_packages"],
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
                "unbound independent reference generator")
        bind(matches[0])
        paired.drift.references.definitions(matches[0], names, namespace)
    projections = {}
    for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                        ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
        prefix = "model.layers.6." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.6.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.6.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    observed = paired.drift.capture(namespace, state, data["original5"])
    paired.drift.same_binary64(observed["s18"], data["original6"])
    return observed


def cut_parent(layer, original, label):
    require(label in dict(plan()), "unknown L6 control")
    return previous.cut_parent(layer, original, label)


def execute_l7(tensors, parent, trajectory, binary64):
    # Preserve the existing L8 and L9-L13 dispatch guards.
    prior.retained.verify_parent(parent, parent)
    arrays, locals_, reports, stages = {}, {}, [], []
    for stage in native.candidate._stages(tensors, 7, parent, arrays):
        require(stage == len(stages), "out-of-order native L7 stage")
        stages.append(stage)
        expected = prior.retained.check_stage_state(stage, arrays, parent)
        if stage < 18 and stage != 12:
            expected = prior.local.local_reference(
                stage, {key: arrays[key].copy() for key in prior.local.OPERANDS[stage]}, tensors, 7)
        if stage < 18:
            locals_[f"stage{stage:02d}"] = expected
        report = prior.gates.evaluate_decoder_stage(
            stage=stage, actual=arrays[f"stage{stage:02d}"],
            reference=trajectory[f"stage{stage:02d}"], policy=prior.gates.POLICY_ID,
            local_reference=expected, reference_binary64=binary64 if stage == 18 else None)
        require(report["status"] in ("PASS", "FAIL"), "missing mandatory L7 gate evidence")
        report.update(node=[7, 0, stage], residual_state_lineage="PASS", kv_lineage="PASS")
        reports.append(report)
    require(stages == list(range(19)), "incomplete native L7 suffix layer")
    require(np.array_equal(prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                           arrays["stage16"]), "S16 RTZ operand check failed")
    return arrays, locals_, reports


def execute_suffix(parent, data, observe):
    item = data["layers"][7]
    arrays, locals_, reports = execute_l7(item["tensors"], parent, item["trajectory"], item["reference"])
    first = observe(7, arrays, locals_, reports)
    incoming = prior.retained.state_from(arrays, "output", "stage18")
    return [first] + previous.execute_suffix(incoming, data, observe)


def classify(rows):
    result = previous.classify(rows)
    singles = result["sufficient_frozen_single_branches"]
    result["classification"] = (
        "conditional_inherited_L5_branch_sufficiency" if singles == ["inherited"]
        else "conditional_local_L6_branch_sufficiency"
        if len(singles) == 1 and singles[0] in ("o", "down")
        else "unresolved_under_tested_branch_mappings")
    result["missing_evidence"] = [
        "Frozen singleton sufficiency is conditional on actual uncut coordinates and branches, "
        "not necessity or unique producer attribution. Multiple/no singletons remain unresolved.",
        "Input/S12/S18 use nearest-Q24 ties-even (error <= 2^-25, signed zero retained); O/down "
        "use direct RNE-FP16. These exogenous cuts do not replace the original global reference.",
        "L6 native operators are not replayed: inherited changes do not recompute L6 attention "
        "or MLP. Reachable inherited-native, other-coordinate and pre-L5 causes remain untested.",
        "Exact residual closure is not a residual arithmetic defect. Scalar L13 rescue does "
        "not imply all L7-L13 gates pass; even a passing frozen cut is not an admitted repair.",
        "No strict-FP16-state W4A16, new-token, full-model, hardware or performance-cause claim.",
    ]
    return result


def diagnose(out, log):
    started = time.monotonic()
    data = authenticate()
    timing = {"authentication": time.monotonic() - started}
    artifacts, rows = [], []

    def document(name, value):
        write(out / name, value)
        item = record(out / name)
        artifacts.append(item)
        return item

    def save(name, arrays):
        item = prior.retained.save(out / name, arrays)
        artifacts.append(item)
        return item

    document("diagnostic_contract.json", {
        "diagnostic_id": ID, "policy_id": prior.gates.POLICY_ID,
        "branch_definitions": BRANCH_DEFINITIONS, "controls": [label for label, _ in plan()],
        "claim_boundary": CLAIM_BOUNDARY, "normal_host_review": "REQUIRED",
        "candidate_admitted": False, "policy_adopted": False,
        "successor_published": False, "rtl_invocations": 0,
        "thresholds": {
            "S0_S17": "finite AND (abs_error <= 0.125 OR (relative_error < 0.001 AND "
                      "ordered_FP16_ULP <= 1)); denominator max(abs(reference), 2^-14)",
            "S18": "Original-input binary64-fp16-excess-v1 exact excess <= 1/8",
        },
    })
    document("retained_authentication.json", data["authentication"])
    began = time.monotonic()
    original = original_branches(data)
    save("L6_original_branches.npz", original)
    timing["original_L6_branch_capture"] = time.monotonic() - began
    layer = data["L6"]
    document("L5_L6_coordinate62.json", {
        name: coordinate.compare_parents(actual, paired.mapped_parent(reference), reference)[62]
        for name, actual, reference in (
            ("L5_output_L6_input", layer["parent"], original["input"]),
            ("L6_S12", prior.retained.state_from(layer["arrays"], "scratch", "stage12"), original["residual"]),
            ("L6_output_L7_input", layer["output"], original["s18"]),
        )
    })
    document("L6_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], layer["arrays"], original, 62),
        "branch_definitions": BRANCH_DEFINITIONS,
        "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
    })
    for label, parts in plan():
        began = time.monotonic()
        operands, parent = cut_parent(layer, original, label)
        unchanged = np.arange(896) != 62
        require(all(np.array_equal(parent[k][unchanged], layer["output"][k][unchanged]) for k in parent),
                "L6 intervention changed unselected coordinates")
        if operands is not None:
            save(label + "_L6_cut_operands.npz", operands)
        parent_record = save(label + "_L6_parent.npz", parent)

        def observe(suffix_layer, arrays, locals_, reports):
            item = data["layers"][suffix_layer]
            if label == "actual":
                paired.same_arrays(arrays, item["arrays"])
                paired.same_arrays(locals_, item["locals"])
                coordinate.check_reports(reports, item["reports"])
            stem = f"{label}_L{suffix_layer}"
            vectors = save(stem + ".npz", arrays)
            local_record = save(stem + "_local_references.npz", locals_)
            gates = document(stem + "_gates.json", reports)
            scalar = prior.measure(int(arrays["stage18"][62]), float(item["reference"][62]))
            failures = [r["index"] for r in reports[18]["binary64_v1"]["failures"]]
            require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
            return {
                "layer": suffix_layer, "vectors": vectors, "local_references": local_record,
                "gates": gates, "index62": scalar, "S18_failure_indices": failures,
                "mandatory_statuses": [r["status"] for r in reports],
                "all_gates_pass": all(r["status"] == "PASS" for r in reports),
                "source_operand_state_KV_RTZ_checks": "PASS",
                "coordinate62_output": coordinate.compare_parents(
                    prior.retained.state_from(arrays, "output", "stage18"),
                    paired.mapped_parent(item["reference"]), item["reference"])[62],
            }

        suffix = execute_suffix(parent, data, observe)
        if label == "actual":
            require(suffix[-1]["S18_failure_indices"] == [62], "retained L13 failure set changed")
        rows.append({
            "label": label, "parts": list(parts), "parent": parent_record,
            "kind": "retained_actual" if label == "actual" else "exogenous_L6_branch_mapping",
            "L6_operator_gates": "retained_PASS_not_reexecuted" if label == "actual" else "not_claimed",
            "L6_output_Q24_units62": int(parent["i"][62]),
            "L6_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(layer["output"]["i"][62]),
            "L6_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "suffix": suffix, "index62": suffix[-1]["index62"],
            "all_L7_L13_gates_pass": all(r["all_gates_pass"] for r in suffix),
            "candidate_admitted": False, "seconds": time.monotonic() - began,
        })
        log.write(json.dumps(rows[-1]) + "\n")
        log.flush()
    document("interventions.json", rows)
    by_name = {r["label"]: r for r in rows}
    interactions = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            delta = by_name[label]["L6_output_minus_actual_Q24_units62"]
            delta -= sum(by_name["frozen_" + b]["L6_output_minus_actual_Q24_units62"] for b in parts)
            require(delta == 0, "frozen additive decomposition did not close")
            interactions[label] = delta
    document("frozen_Q24_additive_interactions.json", interactions)
    after = origins()
    require(after == json.loads((out / "validation.json").read_text())["origins"],
            "source/module changed during execution")
    for item in data["bound"].values():
        require(Path(item["path"]).resolve().is_relative_to(ROOT)
                and record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    timing["native_controls_and_gates"] = sum(r["seconds"] for r in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [6, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "accepted_L0_L6_native_execution": False, "native_retained_bitwise_reproduction": True,
        "original_L6_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "source_operand_state_KV_lineage_checks": "PASS",
        "control_count": len(rows), "native_suffix_layer_evaluations": 7 * len(rows),
        "timing_seconds": timing, "origins_after_execution": after,
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "review_identity_basis": data["review_identity_basis"],
        "all_L7_L13_gate_passing_controls": [r["label"] for r in rows if r["all_L7_L13_gates_pass"]],
        "claim_boundary": CLAIM_BOUNDARY, **classify(rows),
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l6_coordinate62_producer_cone_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = output_path(parser.parse_args().out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": os.environ.get("PYTHONPATH"), "loadavg": os.getloadavg(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads_before": torch.get_num_threads(), "rtl_invocations": 0,
        }) + "\n")
        log.flush()
        try:
            torch.set_num_threads(1)
            validate(out)
            result = diagnose(out, log)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
            traceback.print_exc(file=log)
            result = {
                "diagnostic_id": ID, "status": "BLOCKED", "classification": "inconclusive",
                "missing_evidence": [f"{type(exc).__name__}: {exc}"], "candidate_admitted": False,
                "policy_adopted": False, "successor_published": False, "rtl_invocations": 0,
                "normal_host_review": "REQUIRED",
            }
        write(out / "result.json", result)
        summary = {k: result[k] for k in ("status", "classification", "missing_evidence")}
        summary["result"] = str(out / "result.json")
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0 if result["status"] == "DIAGNOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
