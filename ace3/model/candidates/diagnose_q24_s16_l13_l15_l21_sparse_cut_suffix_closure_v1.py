"""One non-admitting L14-L23/P0 suffix with the frozen L13/L15/L21 cuts.

The adjacent contract publishes the repo-bound execution and read-only review
commands. Raw native arrays and failures remain distinct from adjusted states.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import py_compile
import sys
import time
import unittest

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l13_minimal_cut_l14_suffix_v1 as base
from ace3.model.candidates import diagnose_q24_s16_l15_cut_l16_suffix_v1 as l15
from ace3.model.candidates import diagnose_q24_s16_l21_coordinate62_adjusted_l22_l23_suffix_v1 as l21
from ace3.model.candidates import diagnose_q24_s16_l20_interval_l21_l23_execute_v2 as interior


ROOT, PYTHON = base.ROOT, base.PYTHON
retained, require, native = base.retained, base.require, base.native
local, gates = base.local, base.gates
NAME = "q24_s16_l13_l15_l21_sparse_cut_suffix_closure_v1"
ID = "ace3-q24-s16-l13-l15-l21-sparse-cut-suffix-closure-v1-r2"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
OUTPUT = ROOT / "build" / (NAME + "_attempt002")
LAYERS = tuple(range(14, 24))
EXPECTED_TESTS = 29
FLAGS = dict(base.FLAGS, native_L0_L13_invocations=0,
             L20_endpoint_interval_replays=0)
REFERENCE_POLICY = l21.REFERENCE_POLICY
UPSTREAM = {
    "historical_attempt002": (
        "build/q24_software_l9_l23_4d1cfdfc15c0_attempt002/result.json",
        "addadcb0beba17e7ea0bed56201353a2001febade99fbf515ad82e091b70a61a"),
    "selected_L13_attempt": (
        "build/q24_software_l9_l23_0c02042f0e21_attempt001/result.json",
        "0aa53205f23fbf79bb4d00b011fac3850e5809819d8e543804579978d1d29996"),
    "L13_margin": (
        "build/q24_s16_l13_s18_margin_sensitivity_v1_attempt001/result.json",
        "5acc6ca0c7bc0765f1aa373b953715c1da743b8b3bca3a9d9a19d4ce33b591ef"),
    "L13_suffix": (
        "build/q24_s16_l13_minimal_cut_l14_suffix_v1_attempt001/scientific_binding/result.json",
        "331d2536e78faf9e51258da1e614754a57e22c2e39e640473680ac02d041f491"),
    "L15_margin": (
        "build/q24_s16_l15_s18_margin_sensitivity_v1_attempt001/closure_signature/result.json",
        "237ccabf99aec92b4d724b1c3a7aba2d873dbd137b628f7d0d2af235a9ed42a6"),
    "L15_preflight": (
        "build/q24_s16_l15_cut_l16_suffix_preflight_v1_attempt001/array_equality/result.json",
        "9307ef34309418bd95da01b3b3843246f1190b915e4b489ab263a693547e294e"),
    "L15_suffix": (
        "build/q24_s16_l15_cut_l16_suffix_v1_attempt001/result.json",
        "6d827ad412e70d59ba7bfcb28b9da056fdfc66c75299852c2645eeeab684290b"),
    "L21_margin": (
        "build/q24_s16_l21_s18_margin_sensitivity_v1_attempt001/array_equality/result.json",
        "334b47246ce4f8d66257f1cdbe6265fe462da6eb58923e020b5c8ac4d2bcd5bd"),
    "L21_suffix": (
        "build/q24_s16_l21_coordinate62_adjusted_l22_l23_suffix_v1_attempt001/result.json",
        "dedab0b177ae9fb7dc97445d5fedad1e9b2520f221ddf0febbae4acc9a4c16b9"),
    "L20_endpoints": (
        "build/q24_s16_l20_endpoint_l21_l23_execute_v2_attempt001/native_execution_v1/diagnostic.stdout.json",
        "64558fa6ee596b58ceda0546eadf51391d48cabae5a1bc7426ea1ac5fd9b6c91"),
    "L20_interior": (
        "build/q24_s16_l20_interval_l21_l23_execute_v2_attempt001/execute.stdout.json",
        "fb21b2ef0e46d9e333c0d620af1129d7e1b4b4f02c8ca034c715592bbe8d8cac"),
}
CUTS = {
    "13": {"index": 62, "baseline_Q24_integer": 26568588224,
           "delta_Q24_units": -1866689, "adjusted_Q24_integer": 26566721535,
           "output_word": "662f", "application": "frozen input only; no native L13"},
    "15": {"index": 62, "baseline_Q24_integer": 26617418751,
           "delta_Q24_units": -365567, "adjusted_Q24_integer": 26617053184,
           "output_word": "6632", "application": "after raw S18 checks, before L16"},
    "21": {"index": 62, "baseline_Q24_integer": 1592645376,
           "delta_Q24_units": 10103041, "adjusted_Q24_integer": 1602748417,
           "output_word": "55f9", "application": "after raw S18 checks, before L22"},
}
COMMAND = (
    "cd /home/argustest/ace3-argus && "
    "PYTHONPATH=/home/argustest/ace3-argus /home/argustest/miniconda3/bin/python -B "
    "-m " + MODULE
)
no_execution = interior.v1.no_execution
HISTORICAL_L20_SOURCES = frozenset((
    interior.v1.SOURCE,
    interior.v1.CONTRACT,
    ROOT / "tests/test_q24_s16_l20_endpoint_l21_l23_executor_v1.py",
))


def check_flags(document):
    for key, value in FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed boundary: {key}")


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 2, "focused_tests": EXPECTED_TESTS,
        "layers": list(LAYERS), "maximum_native_layer_invocations": 10,
        "parent_node": [13, 0, 18], "cuts": CUTS,
        "upstream": {key: list(value) for key, value in UPSTREAM.items()},
        "policy_id": gates.POLICY_ID, "excess_budget": "1/8",
        "local_gate": interior.upstream.producer.LOCAL_GATE,
        "reference_policy": REFERENCE_POLICY,
        "execution": COMMAND + " --execute --out build/" + NAME + "_attempt002",
        "review": COMMAND + " --check --out build/" + NAME + "_review002",
        "historical_source_paths": sorted(str(path) for path in HISTORICAL_L20_SOURCES),
        **FLAGS,
    }
    for key, value in expected.items():
        interior.equal(document[key], value, f"contract mismatch: {key}")


def output_path(path):
    out = path.absolute()
    require(out.resolve() == out and out.parent == ROOT / "build"
            and out.name.startswith(NAME + "_"), "output outside writable scope")
    require(not out.exists(), "output already exists; preserve historical evidence")
    return out


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = l15.preflight.margin.origins()
    for name in (MODULE, TEST_MODULE, base.prior.LEGACY_TEST):
        require(records[name]["path"] ==
                str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"repository origin mismatch: {name}")
    return records


def pinned_documents(inputs):
    return {name: base.margin.pinned(inputs, ROOT / path, digest)
            for name, (path, digest) in UPSTREAM.items()}


def check_historical_origin(name, record):
    require(name == "ace3" or name.startswith("ace3.")
            or name == "tests" or name.startswith("tests."),
            f"unexpected repository module: {name}")
    path = ROOT.joinpath(*name.split("."))
    require(record["path"] in (str(path.with_suffix(".py")), str(path / "__init__.py")),
            f"historical module origin mismatch: {name}")


def bind_l20_record(inputs, record, historical):
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(ROOT),
            f"historical input outside isolated repository: {path}")
    if path not in HISTORICAL_L20_SOURCES:
        return inputs.bind(record)
    # Pinned result bytes authenticate the old signature, not today's source bytes.
    signature = {key: record[key] for key in ("path", "bytes", "sha256")}
    if str(path) in historical:
        require(historical[str(path)]["historical"] == signature,
                f"conflicting historical source binding: {path}")
    current = retained.record(path)
    inputs.bind(current)
    historical[str(path)] = {"historical": signature, "current": current}
    return path


def authenticate_l20(inputs, documents, source_origins, extension):
    historical = {}
    for name, module in (("L20_endpoints", interior.interior.endpoint),
                         ("L20_interior", interior)):
        document = documents[name]
        validation = document["validation"]
        interior.upstream.check_validation(validation, module.EXPECTED_TESTS)
        require(document["diagnostic_id"] == module.ID
                and document["executor"] == module.EXECUTOR
                and validation["command"] == module.COMMAND
                and document["reference_policy"] == REFERENCE_POLICY
                and document["original_L20_reference"] == extension["layers"]["20"]
                and document["parent_binding"] ==
                documents["L15_suffix"]["layers"][-1]["input_state_evidence"],
                "historical L20 context/reference/lineage mismatch")
        for key, value in base.FLAGS.items():
            require(type(document[key]) is type(value) and document[key] == value,
                    f"historical L20 boundary changed: {key}")
        require(sorted(validation["compiled"], key=lambda row: row["path"]) ==
                sorted(validation["origins"].values(), key=lambda row: row["path"]),
                "historical L20 compiled closure mismatch")
        for module_name, record in validation["origins"].items():
            check_historical_origin(module_name, record)
            require(source_origins[module_name]["path"] == record["path"],
                    f"repository origin mismatch: {module_name}")
        for record in (*document["authenticated_inputs"], *validation["compiled"],
                       validation["contract"], document["parent_binding"]):
            bind_l20_record(inputs, record, historical)
    frozen = documents["L20_interior"]
    endpoint = frozen["historical_endpoint_execution"]
    require(frozen["sampled_set_complete"] is True
            and endpoint["status"] == documents["L20_endpoints"]["status"]
            and endpoint["native_layers"] == [21, 21]
            and endpoint["native_layer_invocations"] == 2
            and endpoint["evidence"]["diagnostic.stdout.json"] ==
            retained.record(ROOT / UPSTREAM["L20_endpoints"][0]),
            "wrong accepted interior dependency")
    for record in endpoint["evidence"].values():
        inputs.bind(record)
    return {"pinned_results": {name: retained.record(ROOT / UPSTREAM[name][0])
                               for name in ("L20_endpoints", "L20_interior")},
            "source_bindings": list(historical.values()),
            "historical_source_bytes_claimed_current": False}


def check_history(documents):
    for name in ("historical_attempt002", "selected_L13_attempt"):
        document = documents[name]
        require(document["status"] == "FAIL"
                and document["first_failure"]["node"] == [13, 0, 18]
                and document["first_failure"]["index"] == 62
                and document["candidate_admitted"] is False
                and document["policy_id"] == gates.POLICY_ID,
                "historical L13 failure relabeled")
    for name, key, count in (("L20_endpoints", "candidate_set", 2),
                             ("L20_interior", "probes", 15)):
        document = documents[name]
        require(document["status"] == ("ENDPOINT_GATE_FAIL" if count == 2
                                       else "INTERIOR_GATE_FAIL")
                and document["native_layers"] == [21] * count
                and document["native_layer_invocations"] == count
                and len(document[key]) == count, "historical no-pass accounting changed")
        for row in document[key]:
            require(row["status"] == "FAIL" and row["native_layers"] == [21]
                    and row["first_failure"] ==
                    {"node": [21, 0, 18], "index": 62, "gate": "binary64_v1"},
                    "historical no-pass result changed")


def global_gate(layer, parent, trajectory, reference):
    retained.verify_parent(parent, parent)
    report = gates.evaluate_decoder_stage(
        stage=18, actual=parent["h"], reference=trajectory["stage18"],
        policy=gates.POLICY_ID, local_reference=None, reference_binary64=reference)
    report["node"] = [layer, 0, 18]
    return report


def authenticate(inputs, documents, source_origins):
    check_history(documents)
    with no_execution():
        verified, _, extension, _ = l21.authenticate()
    for record in verified.records.values():
        inputs.bind(record)
    provenance = authenticate_l20(inputs, documents, source_origins, extension)
    base.check_reference_suffix(extension)
    source = documents["L13_suffix"]["layers"][0]["input_state_evidence"]
    require(source["path"] == str(l15.preflight.margin.INPUT / "counterfactual_L13_state.npz"),
            "wrong frozen adjusted L13 parent")
    parent = inputs.archive(source)
    raw = inputs.archive(documents["selected_L13_attempt"]["layers"][-1]["actual_stages"])
    expected = base.minimal_cut(retained.state_from(raw, "output", "stage18"),
                                documents["L13_margin"]["minimal_residual_cut"])
    retained.verify_parent(parent, expected)
    cuts = {}
    for layer, name, suffix, function in (
            (15, "L15_margin", "L13_suffix", l15.preflight.minimal_cut),
            (21, "L21_margin", "L15_suffix", l21.minimal_cut)):
        entry = documents[suffix]["layers"][-1]
        require(entry["layer"] == layer, "wrong cut baseline layer")
        arrays = inputs.archive(entry["actual_stages"])
        baseline = retained.state_from(arrays, "output", "stage18")
        cut = documents[name]["minimal_residual_cut"]
        adjusted = function(baseline, cut)
        require(int(adjusted["i"][62]) == CUTS[str(layer)]["adjusted_Q24_integer"],
                "cut definition mismatch")
        cuts[layer] = (baseline, cut)
    item = extension["layers"]["13"]
    gate = global_gate(13, parent, inputs.archive(item["fp16"]),
                       base.global_reference(inputs, item))
    require(gate["status"] == "PASS", "frozen adjusted L13 global gate failed")
    return extension, parent, cuts, source, gate, provenance


def apply_boundary(layer, parent, failure, cuts, trajectory, reference):
    if layer not in cuts:
        return failure, parent, None
    expected_failure = {"node": [layer, 0, 18], "index": 62, "gate": "binary64_v1"}
    if failure is not None and failure != expected_failure:
        return failure, parent, None
    baseline, cut = cuts[layer]
    retained.verify_parent(parent, baseline)
    function = l15.preflight.minimal_cut if layer == 15 else l21.minimal_cut
    changed = function(parent, cut)
    report = global_gate(layer, changed, trajectory, reference)
    require(report["status"] in ("PASS", "FAIL"), "adjusted boundary gate blocked")
    failure = (None if report["status"] == "PASS" else {
        "node": [layer, 0, 18], "index": native.first_failure_index(report),
        "gate": "binary64_v1"})
    return failure, changed, report


def walk(parent, run_layer):
    for layer in LAYERS:
        retained.verify_parent(parent, parent)
        failure, changed = run_layer(layer, parent)
        if failure is not None:
            return failure
        retained.verify_parent(changed, changed)
        parent = changed
    return None


def execute_suffix(out, inputs, extension, parent, cuts, result):
    parent_binding = retained.save(out / "counterfactual_L13_state.npz", parent)
    with base.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        def run_layer(layer, incoming):
            nonlocal parent_binding
            item = extension["layers"][str(layer)]
            tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
            local.authenticate_tensors(tensors, item["canonical"], layer)
            trajectory = inputs.archive(item["fp16"])
            reference = base.global_reference(inputs, item)
            arrays, local_refs, reports = {}, {}, []
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            entry = {"layer": layer, "position": 0, "status": "BLOCKED",
                     "input_state_evidence": parent_binding, "original_reference": item,
                     "prior_kv": "own empty P0", "prior_layer_kv_consumed": False}
            result["layers"].append(entry)
            try:
                failure, timing = base.drive_layer(
                    tensors, layer, incoming, trajectory, reference, arrays, local_refs, reports)
                entry.update(timing, raw_first_failure=failure)
                if failure is not None and failure["node"][2] < 18:
                    entry.update(status="FAIL", first_failure=failure)
                    return failure, None
                raw_parent = retained.state_from(arrays, "output", "stage18")
                failure, changed, boundary = apply_boundary(
                    layer, raw_parent, failure, cuts, trajectory, reference)
                entry.update(status="FAIL" if failure else "PASS",
                             first_failure=failure, adjusted_boundary_gate=boundary)
                if boundary is not None:
                    entry["cut"] = CUTS[str(layer)]
                    entry["adjusted_state_evidence"] = retained.save(
                        directory / "adjusted_state.npz", changed)
                if failure is not None:
                    return failure, None
                parent_binding = retained.save(directory / "counterfactual_state.npz", changed)
                entry["output_state_evidence"] = parent_binding
                entry["own_kv_evidence"] = retained.save(directory / "own_kv.npz", {
                    kind: arrays["output_cache_" + kind] for kind in ("k", "v")})
                return None, {key: value.copy() for key, value in changed.items()}
            finally:
                entry["actual_stages"] = retained.save(directory / "actual_stages.npz", arrays)
                entry["local_references"] = retained.save(
                    directory / "local_references.npz", local_refs)
                retained.write(directory / "reports.json", reports)
                entry["reports"] = retained.record(directory / "reports.json")
                print(json.dumps({"layer": layer, "status": entry["status"]}), flush=True)

        with base.suffix_only(result["audit"]):
            return walk(parent, run_layer)


def validate(out, inputs, documents):
    require(Path.cwd() == ROOT and Path(sys.executable) == PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "published repository-bound command required")
    frozen_origins = documents["L20_interior"]["validation"]["origins"]
    historical = {}
    for name, record in frozen_origins.items():
        check_historical_origin(name, record)
        bind_l20_record(inputs, record, historical)
        if name == "tests" or name.startswith("tests."):
            importlib.import_module(name)
    tests = importlib.import_module(TEST_MODULE)
    current = origins()
    for name, record in frozen_origins.items():
        require(current[name]["path"] == record["path"],
                f"frozen source origin mismatch: {name}")
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for index, path in enumerate((SOURCE, Path(tests.__file__).resolve())):
        py_compile.compile(str(path), cfile=str(out / f"compiled_{index}.pyc"), doraise=True)
        compiled.append(retained.record(path))
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with (out / "unittest.log").open("x") as stream:
        with no_execution():
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    validation = {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "dont_write_bytecode": sys.dont_write_bytecode, "origins": current,
        "compiled": compiled, "contract": retained.record(CONTRACT), "contract_parsed": True,
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "legacy_19_tests_executed": False, "upstream_tests_executed": False,
        "native_layer_invocations": 0, **FLAGS,
    }
    retained.write(out / "validation.json", validation)
    require(count == result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed, skipped or incomplete")
    for record in (*current.values(), *compiled, validation["contract"]):
        inputs.bind(record)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    result = {
        "diagnostic_id": ID, "status": "BLOCKED", "history": [9707], "position": 0,
        "audit": {"native_layers": []}, "layers": [], "cuts": CUTS, **FLAGS,
        "reference_policy": REFERENCE_POLICY, "upstream": UPSTREAM,
        "historical_attempt002_used_as_parent": False,
        "claim_boundary": "Isolated non-admitting Q24 CPU counterfactual only. Wide residual "
        "state is not strict-FP16-state W4A16. INT4 weights and FP16 boundaries/KV unchanged. "
        "No token, full-model, RTL, GPU, FPGA, hardware, simulation, synthesis or PPA claim. "
        "Historical failures and independent bounded acceptances remain unchanged.",
        "timing_seconds": {},
    }
    retained.write(out / "command.json", {
        "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
        "PYTHONPATH": os.environ.get("PYTHONPATH"), "dont_write_bytecode": sys.dont_write_bytecode})
    phase, started = "pin_compile_and_focused_tests", time.monotonic()
    inputs = base.prior.BoundInputs()
    try:
        documents = pinned_documents(inputs)
        validation = validate(out, inputs, documents)
        result["validation"] = retained.record(out / "validation.json")
        result["tests_executed"] = validation["executed"]
        result["timing_seconds"][phase] = time.monotonic() - started
        phase, started = "read_only_authentication", time.monotonic()
        extension, parent, cuts, source, gate, provenance = authenticate(
            inputs, documents, validation["origins"])
        result.update(frozen_L13_parent=source, frozen_L13_global_gate=gate,
                      historical_L20_provenance=provenance,
                      lineage="selected frozen adjusted L13; fresh sequential I/Z/H; own P0 KV",
                      review_provenance=documents["L13_suffix"]["review_provenance"])
        result["timing_seconds"][phase] = time.monotonic() - started
        phase, started = "native_suffix" if args.execute else "read_only_check", time.monotonic()
        if args.execute:
            failure = execute_suffix(out, inputs, extension, parent, cuts, result)
            if failure is not None:
                failure = dict(failure, evidence=result["layers"][-1]["reports"])
            result.update(first_failure=failure,
                          status="COMBINED_SUFFIX_GATE_FAIL" if failure else "BOUNDED_COMBINED_SUFFIX_PASS")
        else:
            result["status"] = "READ_ONLY_CHECK_PASS"
        result["timing_seconds"][phase] = time.monotonic() - started
        phase, started = "final_binding_integrity", time.monotonic()
        require(origins() == validation["origins"], "module source changed during diagnostic")
        for record in inputs.records.values():
            require(retained.record(record["path"]) == record,
                    f"source/input changed during diagnostic: {record['path']}")
        result["authenticated_inputs"] = list(inputs.records.values())
        result["timing_seconds"][phase] = time.monotonic() - started
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            StopIteration, RuntimeError, py_compile.PyCompileError) as exc:
        result.update(status="BLOCKED", phase=phase, error=f"{type(exc).__name__}: {exc}")
        result["timing_seconds"][phase] = time.monotonic() - started
        print(result["error"], file=sys.stderr, flush=True)
    sequence = result["audit"]["native_layers"]
    result.update(native_layer_invocations=len(sequence),
                  native_per_layer={str(layer): sequence.count(layer) for layer in range(24)})
    result["command"] = retained.record(out / "command.json")
    retained.write(out / "result.json", result)
    print(json.dumps({key: result[key] for key in
                      ("status", "native_layer_invocations", "tests_executed")
                      if key in result}), flush=True)
    return 1 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
