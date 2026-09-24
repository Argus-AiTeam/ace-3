"""Non-admitting CPU L12 producer cuts; see the adjacent versioned contract."""

import argparse
from fractions import Fraction
import importlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import py_compile
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1 as coordinate


upstream = coordinate.upstream
paired = coordinate.paired
prior = coordinate.prior
native = coordinate.native
ROOT = coordinate.ROOT
ID = "ace3-q24-s16-l12-coordinate62-producer-cone-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l12_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l12_coordinate62_producer_cone_v1"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l12_coordinate62_producer_cone_v1.json"
REVIEWED = ROOT / "build/q24_s16_l13_l12_parent_coordinate_cut_4937ea22f941_attempt001"
REVIEWED_SHA = "08230294f36233d6b535490c35222d030dfb671c13dd70df79ec0cb4361cbe99"
BRANCHES = ("inherited", "o", "down")
require = prior.require
record = paired.record
write = paired.write


def plan():
    rows = [("actual", ())]
    for size in range(1, 4):
        rows.extend(("frozen_" + "_".join(parts), parts)
                    for parts in itertools.combinations(BRANCHES, size))
    return rows + [
        ("scratch", ("scratch",)), ("scratch_down", ("scratch", "down")),
        ("mapped62", ("mapped62",)), ("mapped_all", ("mapped_all",)),
        ("inherited_native", ("inherited_native",)),
    ]


def compose(parent, o, down):
    prior.retained.verify_parent(parent, parent)
    scratch = native.candidate.state.add(parent, o)
    prior.retained.verify_parent(scratch, prior.retained.transition_reference(parent, o))
    output = native.candidate.state.add(scratch, down)
    prior.retained.verify_parent(output, prior.retained.transition_reference(scratch, down))
    return scratch, output


def frozen_parent(parent, actual, original, parts):
    require(tuple(parts) in [p for _, p in plan()[:8]], "unknown frozen branch combination")
    incoming = {k: v.copy() for k, v in parent.items()}
    o, down = actual["stage11"].copy(), actual["stage17"].copy()
    if "inherited" in parts:
        incoming = coordinate.intervene(incoming, paired.mapped_parent(original["input"]), [62])
    for name, words, source in (("o", o, "s11"), ("down", down, "s17")):
        if name in parts:
            mapped = native.candidate.rne(torch.from_numpy(original[source].copy()))
            words[62] = mapped[62]
    scratch, output = compose(incoming, o, down)
    return incoming, o, down, scratch, output


def classify(rows):
    require([r["label"] for r in rows] == [label for label, _ in plan()],
            "incomplete or reordered producer controls")
    by_name = {r["label"]: r for r in rows}
    require(not by_name["actual"]["index62"]["accepted"]
            and by_name["actual"]["index62"]["actual_fp16_bits"] == "6630",
            "retained baseline changed")
    require(by_name["mapped62"]["index62"]["accepted"]
            and by_name["mapped62"]["index62"]["actual_fp16_bits"] == "662f"
            and by_name["mapped_all"]["index62"]["accepted"],
            "reviewed coordinate endpoints changed")
    singles = [b for b in BRANCHES if by_name["frozen_" + b]["index62"]["accepted"]]
    joint = [label for label, parts in plan()[:8]
             if len(parts) > 1 and by_name[label]["index62"]["accepted"]]
    return {
        "classification": ("conditional_single_branch_sufficiency" if len(singles) == 1
                           else "unresolved_producer_interaction"),
        "sufficient_frozen_single_branches": singles,
        "sufficient_frozen_joint_branches": joint,
        "inherited_native_rescue": by_name["inherited_native"]["index62"]["accepted"],
        "scratch_rescue": by_name["scratch"]["index62"]["accepted"],
        "unique_upstream_producer_attributed": False,
        "missing_evidence": [
            "Frozen branch cuts isolate scalar additive contributions on the actual background, "
            "not independently realizable L12 operator repairs. Multiple sufficient cuts do "
            "not identify a unique producer; no rescue does not establish necessity.",
            "Inherited-native changes only the complete L11 I/Z/H coordinate 62 and recomputes "
            "native L12. Other inherited coordinates and producers before L9 are not isolated.",
            "O/down binary64 reference contributions are mapped to FP16 RNE; input and scratch "
            "are mapped nearest-Q24 ties-even. Mapping and binary64 addition roundoff are "
            "disclosed, and the original global reference is never replaced.",
            "No exhaustive nonlinear attribution, policy adoption, admitted repair, strict-FP16 "
            "state, new-token, full-model, hardware or performance-bottleneck claim.",
        ],
    }


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
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID and contract["policy_id"] == prior.gates.POLICY_ID
            and contract["reviewed_result_sha256"] == REVIEWED_SHA
            and contract["reviewed_upstream_sha256"] == coordinate.REVIEWED_SHA
            and contract["controls"] == [label for label, _ in plan()]
            and contract["normal_host_review"] == "REQUIRED" and contract["rtl_invocations"] == 0
            and all(contract[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "producer diagnostic contract mismatch")
    with (out / "unittest.log").open("x") as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    write(out / "validation.json", {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "sys_path": sys.path, "origins": source_origins, "compiled": compiled,
        "contract": record(CONTRACT), "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "accepted_L0_L8_tests_executed": False,
    })
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    return contract


def authenticate():
    data = coordinate.authenticate({"reviewed_result_sha256": coordinate.REVIEWED_SHA})
    bound = data["bound"]
    bind = lambda item: upstream.bind_input(item, bound)
    read = lambda item: json.loads(bind(item).read_text())
    archive = lambda item: native.load_state(item, bind)
    item = record(REVIEWED / "result.json")
    require(item["sha256"] == REVIEWED_SHA, "wrong reviewed coordinate evidence")
    reviewed = read(item)
    require(reviewed["diagnostic_id"] == coordinate.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["sufficient_single_coordinates"] == [62]
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["rtl_invocations"] == 0
            and all(reviewed[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "reviewed coordinate scope mismatch")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)
    for label in ("actual", "single_062", "mapped"):
        path = str(REVIEWED / f"{label}_native.npz")
        matches = [r for r in reviewed["artifacts"] if r["path"] == path]
        require(len(matches) == 1, f"missing reviewed endpoint: {label}")
        data[label + "_reviewed"] = archive(matches[0])
    freeze_record = record(prior.INPUT / "freeze.json")
    freeze = read(freeze_record)
    extension = read(freeze["reference_extension"])
    retained = read(record(prior.INPUT / "result.json"))
    entries = {e["layer"]: e for e in retained["layers"]}
    expected = freeze["state_lineage"]["prior_residual"]["parent"]["state"]
    require(freeze["state_lineage"]["prior_kv"] == "own empty P0"
            and freeze["state_lineage"]["prior_residual"]["prior_layer_kv_consumed"] is False,
            "wrong retained KV boundary")
    layers = {}
    for layer in range(9, 13):
        entry, reference = entries[layer], extension["layers"][str(layer)]
        require(entry["input_parent"] == expected, f"spliced actual parent at L{layer}")
        parent, arrays = archive(entry["input_parent"]), archive(entry["actual_stages"])
        for stage in range(19):
            prior.retained.check_stage_state(stage, arrays, parent)
        reports = read(entry["reports"])
        require(len(reports) == 19 and all(
            r["node"] == [layer, 0, s] and r["status"] == "PASS"
            and r["policy_id"] == prior.gates.POLICY_ID
            and r["residual_state_lineage"] == r["kv_lineage"] == "PASS"
            for s, r in enumerate(reports)), "retained prefix gates changed")
        output = archive(entry["output_parent"])
        prior.retained.verify_parent(output, prior.retained.state_from(arrays, "output", "stage18"))
        paired.same_arrays(archive(entry["kv_state"]),
                           {k: arrays["output_cache_" + k] for k in ("k", "v")})
        receipt = read(record(prior.INPUT / f"layer{layer:02d}/software_parent.json"))
        require(receipt["state"] == entry["output_parent"]
                and receipt["state_lineage_parent"] == entry["input_parent"]
                and receipt["numerical_report"] == entry["reports"]
                and receipt["kv"] == entry["kv_state"] and receipt["arithmetic_lineage"] == freeze_record
                and receipt["candidate_id"] == freeze["contract"]["candidate_id"]
                and receipt["state_id"] == freeze["contract"]["state_id"]
                and receipt["policy_id"] == prior.gates.POLICY_ID
                and receipt["evidence_kind"] == "cpu_software_q24"
                and receipt["position"] == 0 and receipt["history"] == [9707]
                and receipt["next_layer"] == layer + 1 and receipt["rtl_admissible"] is False
                and receipt["normal_host_review"] == "REQUIRED", "retained receipt mismatch")
        layers[layer] = {
            "parent": parent, "arrays": arrays, "output": output, "reports": reports,
            "locals": archive(entry["local_references"]),
            "reference": np.load(bind(reference["binary64"]), allow_pickle=False),
            "trajectory": archive(reference["fp16"]),
        }
        expected = entry["output_parent"]
    require(entries[13]["input_parent"] == expected, "spliced L13 parent")
    with safe_open(str(bind(extension["checkpoint"])), framework="numpy") as model:
        tensors = {k: np.ascontiguousarray(model.get_tensor(k))
                   for k in prior.local.tensor_shapes(12)}
    prior.local.authenticate_tensors(tensors, extension["layers"]["12"]["canonical"], 12)
    data.update(layers=layers, tensors12=tensors, extension=extension)
    return data


def original_branches(data):
    extension, tensors = data["extension"], data["tensors12"]
    bind = lambda item: upstream.bind_input(item, data["bound"])
    specification = json.loads(bind(extension["original_specification"]).read_text())
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
                "unbound independent reference generator")
        bind(matches[0])
        paired.drift.references.definitions(matches[0], names, namespace)
    projections = {}
    for key, suffix in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                        ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")):
        prefix = "model.layers.12." + suffix
        q = namespace["_torch_unpack"](tensors[prefix + ".qweight"]).to(torch.float64)
        z = namespace["_torch_unpack"](tensors[prefix + ".qzeros"]).to(torch.float64)
        scales = torch.from_numpy(tensors[prefix + ".scales"].astype("<f8"))
        weight = (q - z.repeat_interleave(128, dim=0)) * scales.repeat_interleave(128, dim=0)
        bias = torch.from_numpy(tensors[prefix + ".bias"].astype("<f8")) if key in ("q", "k", "v") else None
        projections[key] = SimpleNamespace(reference_weight=weight, reference_bias=bias)
    state = SimpleNamespace(
        input_norm=tensors["model.layers.12.input_layernorm.weight"],
        post_attention_norm=tensors["model.layers.12.post_attention_layernorm.weight"],
        projections=projections, reference_k=torch.empty((0, 2, 64), dtype=torch.float64),
        reference_v=torch.empty((0, 2, 64), dtype=torch.float64))
    hidden = np.load(bind(extension["layers"]["12"]["input_binary64"]), allow_pickle=False)
    paired.drift.same_binary64(hidden, data["layers"][11]["reference"])
    observed = paired.drift.capture(namespace, state, hidden)
    paired.drift.same_binary64(observed["s18"], data["layers"][12]["reference"])
    return observed


def diagnose(out, log, contract):
    started = time.monotonic()
    torch.set_num_threads(1)
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

    document("retained_authentication.json", data["authentication"])
    trajectory = []
    for layer, item in data["layers"].items():
        mapped = paired.mapped_parent(item["reference"])
        delta = coordinate.compare_parents(item["output"], mapped, item["reference"])[62]
        trajectory.append({"layer": layer, "position": 0, **delta})
    document("L9_L12_coordinate62.json", trajectory)
    began = time.monotonic()
    original = original_branches(data)
    save("L12_original_branches.npz", original)
    timing["original_L12_branch_capture"] = time.monotonic() - began
    layer = data["layers"][12]
    began = time.monotonic()
    actual, locals_, reports = upstream.execute_layer(
        data["tensors12"], 12, layer["parent"], layer["trajectory"], layer["reference"])
    paired.same_arrays(actual, layer["arrays"])
    paired.same_arrays(locals_, layer["locals"])
    coordinate.check_reports(reports, layer["reports"])
    timing["native_L12_baseline_and_gates"] = time.monotonic() - began
    save("actual_L12.npz", actual)
    document("actual_L12_gates.json", reports)
    document("L12_coordinate62_decomposition.json", {
        **paired.parts(layer["parent"], actual, original, 62),
        "S12_actual_vs_original": coordinate.compare_parents(
            prior.retained.state_from(actual, "scratch", "stage12"),
            paired.mapped_parent(original["residual"]), original["residual"])[62],
        "branch_definitions": contract["branch_definitions"],
        "original_values_hex": {k: float(v[62]).hex() for k, v in original.items()},
    })
    baseline_scratch = prior.retained.state_from(actual, "scratch", "stage12")
    for label, parts in plan():
        began = time.monotonic()
        l12_reports = None
        if label in ("mapped62", "mapped_all"):
            parent = coordinate.intervene(data["actual"], data["mapped"],
                                          [62] if label == "mapped62" else list(range(896)))
            cut = {"kind": "reviewed_output_coordinate_control", "parts": list(parts)}
        elif label == "inherited_native":
            incoming = coordinate.intervene(layer["parent"], paired.mapped_parent(original["input"]), [62])
            generated, _, l12_reports = upstream.execute_layer(
                data["tensors12"], 12, incoming, layer["trajectory"], layer["reference"])
            save(label + "_L12.npz", generated)
            document(label + "_L12_gates.json", l12_reports)
            parent = prior.retained.state_from(generated, "output", "stage18")
            cut = {"kind": "native_L12_recomputed_from_inherited_coordinate62", "parts": list(parts)}
        else:
            if "scratch" in parts:
                scratch = coordinate.intervene(
                    baseline_scratch, paired.mapped_parent(original["residual"]), [62])
                down = actual["stage17"].copy()
                if "down" in parts:
                    down[62] = native.candidate.rne(torch.from_numpy(original["s17"].copy()))[62]
                parent = native.candidate.state.add(scratch, down)
                prior.retained.verify_parent(parent, prior.retained.transition_reference(scratch, down))
                branch_vectors = {"scratch_" + k: v for k, v in scratch.items()}
            else:
                incoming, o, down, scratch, parent = frozen_parent(layer["parent"], actual, original, parts)
                branch_vectors = {"input_" + k: v for k, v in incoming.items()}
                branch_vectors.update({"scratch_" + k: v for k, v in scratch.items()})
                branch_vectors["o"] = o
                if label == "actual":
                    paired.same_arrays(scratch, baseline_scratch)
                    prior.retained.verify_parent(parent, data["actual"])
            branch_vectors["down"] = down
            save(label + "_L12_cut_operands.npz", branch_vectors)
            for key in parent:
                unchanged = np.arange(896) != 62
                require(np.array_equal(parent[key][unchanged], data["actual"][key][unchanged]),
                        "frozen cut changed unselected coordinates")
            cut = {"kind": "exogenous_frozen_scalar_branch_cut", "parts": list(parts),
                   "exact_residual_transitions": "PASS",
                   "L12_operator_repair_or_gate_pass_claimed": False}
        parent_record = save(label + "_L12_parent.npz", parent)
        arrays, _, suffix_reports = coordinate.execute(parent, data)
        if label == "actual":
            paired.same_arrays(arrays, data["actual_arrays"])
            paired.same_arrays(arrays, data["actual_reviewed"])
            coordinate.check_reports(suffix_reports, data["actual_reports"])
        elif label in ("mapped62", "mapped_all"):
            paired.same_arrays(arrays, data["single_062_reviewed" if label == "mapped62" else "mapped_reviewed"])
            if label == "mapped_all":
                coordinate.check_reports(suffix_reports, data["mapped_reports"])
        save(label + "_L13.npz", arrays)
        gate_record = document(label + "_L13_gates.json", suffix_reports)
        scalar = prior.measure(int(arrays["stage18"][62]), float(data["reference"][62]))
        failures = [r["index"] for r in suffix_reports[18]["binary64_v1"]["failures"]]
        require(scalar["accepted"] == (62 not in failures), "scalar/full-vector gate disagreement")
        if label == "actual":
            require(failures == [62], "retained failure set changed")
        row = {
            "label": label, "cut": cut, "parent": parent_record, "gates": gate_record,
            "index62": scalar, "L12_output_Q24_units62": int(parent["i"][62]),
            "L12_output_minus_actual_Q24_units62": int(parent["i"][62]) - int(data["actual"]["i"][62]),
            "L12_output_FP16_bits62": f"{int(parent['h'][62]):04x}",
            "L13_S18_failure_indices": failures,
            "L13_S18_changed_indices": np.flatnonzero(
                arrays["stage18"] != data["actual_arrays"]["stage18"]).tolist(),
            "mandatory_statuses": [r["status"] for r in suffix_reports],
            "all_L13_gates_pass": all(r["status"] == "PASS" for r in suffix_reports),
            "native_L12_mandatory_statuses": None if l12_reports is None else [r["status"] for r in l12_reports],
            "L13_source_operand_state_KV_RTZ_checks": "PASS", "candidate_admitted": False,
            "seconds": time.monotonic() - began,
        }
        rows.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
    document("interventions.json", rows)
    # Exact Q24 additive interaction, separate from thresholded FP16 suffix rescue.
    by_name = {r["label"]: r for r in rows}
    base = by_name["actual"]["L12_output_Q24_units62"]
    interaction = {}
    for label, parts in plan()[:8]:
        if len(parts) > 1:
            value = by_name[label]["L12_output_Q24_units62"] - base
            value -= sum(by_name["frozen_" + b]["L12_output_Q24_units62"] - base for b in parts)
            require(value == 0, "frozen exact additive decomposition did not close")
            interaction[label] = value
    document("frozen_Q24_additive_interactions.json", interaction)
    after = origins()
    validation = json.loads((out / "validation.json").read_text())
    require(after == validation["origins"], "module/source origins changed during execution")
    for item in data["bound"].values():
        require(record(Path(item["path"])) == item, f"authenticated input changed: {item['path']}")
    timing["native_controls_and_gates"] = sum(row["seconds"] for row in rows)
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [12, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED", "retained_status": "FAIL",
        "native_retained_bitwise_reproduction": True, "reviewed_coordinate_rescue_reproduced": True,
        "original_L12_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True, "source_operand_state_KV_lineage_checks": "PASS",
        "control_count": len(rows), "timing_seconds": timing, "origins_after_execution": after,
        "input_bindings": list(data["bound"].values()), "artifacts": artifacts,
        "all_L13_gate_passing_controls": [r["label"] for r in rows if r["all_L13_gates_pass"]],
        "claim_boundary": contract["claim_boundary"], **classify(rows),
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l12_coordinate62_producer_cone_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
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
            contract = validate(out)
            result = diagnose(out, log, contract)
            result["validation"] = record(out / "validation.json")
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, py_compile.PyCompileError) as exc:
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
