"""L20-output/L21-input coordinate-62 Q24 CPU sensitivity diagnostic.

--check compiles and tests the repository surface, authenticates frozen evidence,
and describes prospective probes without native execution. --execute additionally
runs the bounded search and conditional L22-L23 suffix, reporting only on stdout.
"""

import argparse
from contextlib import ExitStack, contextmanager
import importlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_l21_coordinate62_producer_cone_v1 as producer


suffix, margin = producer.dependency, producer.margin
retained, require, gates = producer.retained, producer.require, producer.gates
ROOT, PYTHON = producer.ROOT, producer.PYTHON
NAME = "q24_s16_l20_to_l21_coordinate62_input_sensitivity_v1"
ID = "ace3-q24-s16-l20-to-l21-coordinate62-input-sensitivity-v1"
MODULE = "ace3.model.candidates.diagnose_" + NAME
TEST_MODULE = "tests.test_" + NAME
SOURCE = ROOT.joinpath(*MODULE.split(".")).with_suffix(".py")
CONTRACT = ROOT / f"ace3/contracts/candidates/{NAME}.json"
LEGACY_TEST = margin.margin.prior.LEGACY_TEST
EXPECTED_TESTS = 20
FLAGS = dict(producer.FLAGS)
PRODUCER_INPUT = ROOT / ".argus_subagents/72ea3c5dc077-l21-producer-cone-r2_logs/stdout.log"
PRODUCER_SHA256 = "aa504fefcf27f9be43d55ea1bc9da61eb66697f08e17ff5e6c1875989a21598f"
SUFFIX_PINS = {
    "result.json": "dedab0b177ae9fb7dc97445d5fedad1e9b2520f221ddf0febbae4acc9a4c16b9",
    "validation.json": "5966493f2fcd555734d5b65df3f39f9f034e9481c50764c943fbe5140b66f095",
    "command.json": "b124b5c19fa2b4da9c1cd0b03e01425d3c7f0077b1709a59e5440972c2b6cbdd",
}
SEARCH = {
    "boundary": "frozen L20 output / L21 input", "coordinate": 62,
    "position": 0, "history": [9707],
    "units": "signed integer Q24 units", "radius_Q24_units": 16777216,
    "order": "0, then +2**k, -2**k for k=0..24", "probe_count": 51,
    "state": "copy I/Z/H; change only I[62]; preserve Z; recompute H[62] by RNE16",
    "references": "unchanged independently propagated original-input global reference",
    "precondition": "authenticate parent, then require full-vector L20 S18 global gate",
    "prospective_native_layers": [21],
    "mandatory_gates": "own empty P0 FP16 KV; source/operand/I/Z/H lineage; S0-S17 local; S18 global",
    "stop": "first full-vector L21 pass or new mandatory failure; exhaustion is INCONCLUSIVE",
    "minimality": "no monotonicity assumption, binary search, or minimal-input-delta claim",
    "execution": "--execute only; bounded CPU search with mandatory independent Host review",
}
EXECUTION = {
    "version": 1, "native_layers": [21, 22, 23],
    "maximum_native_layer_invocations": 54,
    "output": "stdout only; no state publication or admission",
    "suffix": "L22 then L23 only after a full-vector L21 pass",
    "neighbor": "test one Q24 unit toward zero after a passing nonzero probe",
    "minimality": "sampled sufficiency is not global minimality; unresolved minimality is INCONCLUSIVE",
}
no_execution = producer.no_execution


def origins():
    require(Path(__file__).resolve() == SOURCE and CONTRACT.resolve() == CONTRACT,
            "candidate/contract origin mismatch")
    records = margin.margin.origins()
    for name in (MODULE, TEST_MODULE, LEGACY_TEST):
        require(name in records and records[name]["path"]
                == str(ROOT.joinpath(*name.split(".")).with_suffix(".py")),
                f"required repository origin missing: {name}")
    return records


def check_contract(document):
    expected = {
        "diagnostic_id": ID, "version": 1, "focused_tests": EXPECTED_TESTS,
        "interface": ["--check", "--execute"], "native_layers": [], "search": SEARCH,
        "executor": EXECUTION,
        "producer_input": str(PRODUCER_INPUT.relative_to(ROOT)),
        "producer_sha256": PRODUCER_SHA256,
        "suffix_input": str(suffix.OUTPUT.relative_to(ROOT)),
        "suffix_sha256": SUFFIX_PINS, "margin_sha256": suffix.PINS,
        "L15_cut_suffix_sha256": margin.PINS,
        "policy_id": gates.POLICY_ID, "local_gate": producer.LOCAL_GATE,
        "excess_budget": "1/8", "reference_policy": suffix.REFERENCE_POLICY, **FLAGS,
    }
    for key, value in expected.items():
        require(type(document[key]) is type(value)
                and json.dumps(document[key], sort_keys=True) == json.dumps(value, sort_keys=True),
                f"versioned contract mismatch: {key}")


def check_validation(validation, count):
    require(validation["cwd"] == str(ROOT) and validation["executable"] == str(PYTHON)
            and validation["PYTHONPATH"] == str(ROOT)
            and validation["dont_write_bytecode"] is True,
            "frozen repo-bound context mismatch")
    for key, value in (("collected", count), ("executed", count),
                       ("failures", 0), ("errors", 0), ("skipped", 0)):
        require(type(validation[key]) is int and validation[key] == value,
                f"frozen validation count mismatch: {key}")
    require(validation["legacy_19_tests_executed"] is False
            and validation["upstream_tests_executed"] is False,
            "accepted upstream tests must not be replayed")


def bind_validation(inputs, validation, source, contract):
    margin.upstream.preflight.bind_origins(inputs, validation["origins"])
    require(validation["contract"] == retained.record(contract),
            "frozen contract/source mismatch")
    require(validation["compiled"] == [
        retained.record(source),
        retained.record(ROOT / "tests" / ("test_" + source.stem.removeprefix("diagnose_") + ".py")),
    ], "frozen compiled source mismatch")
    for record in (*validation["compiled"], validation["contract"]):
        inputs.bind(record)


def check_producer(document, frozen, accepted_margin):
    require(document["diagnostic_id"] == producer.ID
            and document["status"] == "INCONCLUSIVE"
            and document["native_layers"] == [21] * 11
            and type(document["native_invocations"]) is int
            and document["native_invocations"] == document["native_layer_invocations"] == 11
            and document["native_retained_bitwise_reproduction"] is True
            and document["executor"] == producer.EXECUTION,
            "wrong inconclusive producer-cone result")
    for key, value in FLAGS.items():
        if key not in ("native_invocations", "native_layer_invocations"):
            require(type(document[key]) is type(value) and document[key] == value,
                    f"changed producer boundary: {key}")
    require(document["upstream_sha256"] == suffix.PINS
            and document["suffix_sha256"] == margin.PINS
            and document["reference_policy"] == suffix.REFERENCE_POLICY
            and document["lineage"] == producer.LINEAGE
            and document["first_failure"] == frozen["first_failure"]
            and document["parent_binding"] == frozen["layers"][-1]["input_state_evidence"]
            and document["original_reference"] == frozen["layers"][-1]["original_reference"],
            "producer parent/reference/lineage substitution")
    controls = document["controls"]
    require(len(controls) == 11, "incomplete frozen producer controls")
    for row, (label, cut) in zip(controls, producer.controls()):
        require(row["label"] == label and row["canonical_stages"] == list(cut)
                and row["last_stage"] == 18 and row["status"] == "FAIL"
                and row["index62"] == accepted_margin["baseline"],
                "changed frozen producer control")
        reports = row["reports"]
        require(len(reports) == 19 and producer.decision(reports) is None,
                "producer control reached a different boundary")
        for stage, report in enumerate(reports):
            require(report["node"] == [21, 0, stage] and report["stage"] == stage
                    and report["policy_id"] == gates.POLICY_ID
                    and report["residual_state_lineage"] == report["kv_lineage"] == "PASS",
                    "producer report policy/lineage mismatch")
            if stage < 18:
                require(report["local_operator_fp16"]["passed"] is True,
                        "producer local gate failure")
        gate = reports[-1]["binary64_v1"]
        expected_failure = {
            "index": 62,
            **{key: value for key, value in accepted_margin["baseline"].items()
               if key not in ("signed_error", "excess_over_budget",
                              "representation_floor_already_subtracted")},
        }
        require(gate["passed"] is False and gate["failure_count"] == 1
                and gate["failures"] == [expected_failure],
                "producer global failure changed")


def check_suffix(document, accepted_margin):
    require(document["diagnostic_id"] == suffix.ID
            and document["status"] == "BOUNDED_COUNTERFACTUAL_PASS"
            and document["audit"] == {"native_layers": [22, 23]}
            and document["native_layer_invocations"] == 2
            and document["first_native_layer"] == 22 and document["first_failure"] is None
            and document["position"] == 0 and document["history"] == [9707]
            and document["retained_status"] == "DOWNSTREAM_GATE_FAIL",
            "wrong frozen adjusted suffix result")
    for key, value in suffix.FLAGS.items():
        require(type(document[key]) is type(value) and document[key] == value,
                f"changed suffix boundary: {key}")
    require(document["reference_policy"] == suffix.REFERENCE_POLICY
            and document["margin_sha256"] == suffix.PINS
            and document["suffix_sha256"] == margin.PINS
            and document["margin_dependency"] == retained.record(suffix.INPUT / "result.json")
            and document["suffix_dependency"] == accepted_margin["upstream_result"]
            and document["minimal_residual_cut"] == accepted_margin["minimal_residual_cut"]
            and document["frozen_first_failure"] == accepted_margin["first_failure"]
            and document["frozen_lineage"] == producer.LINEAGE
            and document["projected_L21_global_gate"] == accepted_margin["projected_full_vector_gate"],
            "adjusted suffix dependency/reference mismatch")


def check_frozen_suffix(inputs, document, frozen, accepted_margin):
    arrays = inputs.archive(frozen["layers"][-1]["actual_stages"])
    parent = suffix.minimal_cut(retained.state_from(arrays, "output", "stage18"),
                                accepted_margin["minimal_residual_cut"])
    upstream = margin.upstream.upstream
    freeze = inputs.read(retained.record(upstream.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    suffix.check_boundary(parent, {k: np.empty((0, 128), dtype="<u2") for k in ("k", "v")},
                          extension)
    parent_record = retained.record(suffix.OUTPUT / "counterfactual_L21_state.npz")
    retained.verify_parent(inputs.archive(parent_record), parent)
    require([entry["layer"] for entry in document["layers"]] == [22, 23],
            "wrong frozen suffix sequence")
    for entry in document["layers"]:
        layer = entry["layer"]
        directory = suffix.OUTPUT / f"layer{layer}"
        item = extension["layers"][str(layer)]
        margin.check_reference(entry, item)
        require(entry["status"] == "PASS" and entry["position"] == 0
                and entry["prior_kv"] == "own empty P0"
                and entry["prior_layer_kv_consumed"] is False
                and entry["input_state_evidence"] == parent_record,
                "frozen suffix state/KV lineage mismatch")
        for key, name in (("actual_stages", "actual_stages.npz"),
                          ("local_references", "local_references.npz"),
                          ("reports", "reports.json"),
                          ("output_state_evidence", "counterfactual_state.npz"),
                          ("own_kv_evidence", "own_kv.npz")):
            require(entry[key] == retained.record(directory / name),
                    "frozen suffix artifact substitution")
        arrays = inputs.archive(entry["actual_stages"])
        locals_ = inputs.archive(entry["local_references"])
        reports = inputs.read(entry["reports"])
        trajectory = inputs.archive(item["fp16"])
        reference = upstream.global_reference(inputs, item)
        margin.margin.check_layer(arrays, parent)
        require(set(locals_) == {f"stage{s:02d}" for s in range(18)}
                and len(reports) == 19, "incomplete frozen suffix reports")
        for stage in range(19):
            expected = locals_[f"stage{stage:02d}"] if stage < 18 else None
            if stage == 12:
                require(np.array_equal(expected, retained.transition_reference(
                    parent, arrays["stage11"])["h"]), "changed exact S12 reference")
            report = gates.evaluate_decoder_stage(
                stage=stage, actual=arrays[f"stage{stage:02d}"],
                reference=trajectory[f"stage{stage:02d}"], policy=gates.POLICY_ID,
                local_reference=expected, reference_binary64=reference if stage == 18 else None)
            report.update(node=[layer, 0, stage], residual_state_lineage="PASS",
                          kv_lineage="PASS", local_reference_independent=stage < 18)
            require(report == reports[stage] and report["status"] == "PASS",
                    f"frozen suffix gate mismatch L{layer}/S{stage}")
        parent_record = entry["output_state_evidence"]
        parent = inputs.archive(parent_record)
        retained.verify_parent(parent, retained.state_from(arrays, "output", "stage18"))
        kv = inputs.archive(entry["own_kv_evidence"])
        require(set(kv) == {"k", "v"}, "invalid frozen KV keys")
        for kind in ("k", "v"):
            suffix.local.finite_words(kv[kind], (1, 128))
            require(np.array_equal(kv[kind], arrays["output_cache_" + kind]),
                    "frozen suffix KV producer mismatch")


def authenticate():
    with no_execution():
        inputs, accepted_margin, frozen = producer.authenticate()
        document = margin.scalar_math.pinned(inputs, PRODUCER_INPUT, PRODUCER_SHA256)
        check_producer(document, frozen, accepted_margin)
        check_validation(document["validation"], producer.EXPECTED_TESTS)
        for record in document["authenticated_inputs"]:
            inputs.bind(record)
        bind_validation(inputs, document["validation"], producer.SOURCE, producer.CONTRACT)
        require(document["controls"][0]["reports"] == inputs.read(frozen["layers"][-1]["reports"]),
                "frozen retained producer baseline mismatch")
        documents = {name: margin.scalar_math.pinned(inputs, suffix.OUTPUT / name, digest)
                     for name, digest in SUFFIX_PINS.items()}
        adjusted, validation, command = (documents[key] for key in
                                         ("result.json", "validation.json", "command.json"))
        check_suffix(adjusted, accepted_margin)
        check_validation(validation, suffix.EXPECTED_TESTS)
        require(adjusted["tests_executed"] == suffix.EXPECTED_TESTS
                and adjusted["validation"] == retained.record(suffix.OUTPUT / "validation.json")
                and adjusted["command"] == retained.record(suffix.OUTPUT / "command.json"),
                "frozen suffix validation linkage mismatch")
        for key in ("cwd", "executable", "PYTHONPATH", "dont_write_bytecode"):
            require(command[key] == validation[key], "frozen suffix command context mismatch")
        require(command["argv"] == [str(suffix.SOURCE), "--out",
                                   str(suffix.OUTPUT.relative_to(ROOT))],
                "frozen suffix command substitution")
        for record in adjusted["authenticated_inputs"]:
            inputs.bind(record)
        bind_validation(inputs, validation, suffix.SOURCE, suffix.CONTRACT)
        check_frozen_suffix(inputs, adjusted, frozen, accepted_margin)
        parent_binding = frozen["layers"][-1]["input_state_evidence"]
        parent = inputs.archive(parent_binding)
        arrays = inputs.archive(frozen["layers"][-1]["actual_stages"])
        retained.verify_parent(retained.state_from(arrays, "input", "input_hidden"), parent)
        return inputs, parent_binding, frozen["first_failure"]


def search_points():
    return (0, *(sign * (1 << exponent) for exponent in range(25) for sign in (1, -1)))


def perturb(parent, delta):
    require(type(delta) is int and abs(delta) <= SEARCH["radius_Q24_units"],
            "bounded integer Q24 delta required")
    retained.verify_parent(parent, parent)
    value = int(parent["i"][62]) + delta
    require(-(1 << 63) <= value < (1 << 63), "Q24 int64 overflow")
    changed = {key: array.copy() for key, array in parent.items()}
    changed["i"][62] = value
    changed["h"][62] = margin.rational.project(value, int(changed["z"][62]))
    retained.verify_parent(changed, changed)
    require(np.array_equal(changed["z"], parent["z"])
            and np.flatnonzero(changed["i"] != parent["i"]).tolist() == ([62] if delta else [])
            and set(np.flatnonzero(changed["h"] != parent["h"]).tolist()) <= {62},
            "perturbation changed another coordinate or zero-sign tag")
    return changed


def validate():
    command = producer.context()
    tests = importlib.import_module(TEST_MODULE)
    importlib.import_module(LEGACY_TEST)
    records = origins()
    contract_record = retained.record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    compiled = []
    for path in (SOURCE, ROOT.joinpath(*TEST_MODULE.split(".")).with_suffix(".py")):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(retained.record(path))
    with no_execution():
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        count = suite.countTestCases()
        require(count == EXPECTED_TESTS, "focused test collection mismatch")
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(result.testsRun == EXPECTED_TESTS and result.wasSuccessful()
            and not result.skipped, "focused tests failed")
    require(origins() == records and retained.record(CONTRACT) == contract_record,
            "source/contract changed during validation")
    return {**command, "origins": records, "compiled": compiled, "contract": contract_record,
            "contract_parsed": True, "collected": count, "executed": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "legacy_19_tests_executed": False,
            "upstream_tests_executed": False}


def preflight():
    validation = validate()
    inputs, parent_binding, failure = authenticate()
    for record in (*validation["origins"].values(), validation["contract"]):
        inputs.bind(record)
    require(origins() == validation["origins"], "module origins changed during preflight")
    for record in inputs.records.values():
        require(retained.record(record["path"]) == record,
                f"input/source changed during preflight: {record['path']}")
    return {
        "diagnostic_id": ID, "status": "PREFLIGHT_VALIDATED", **FLAGS,
        "validation": validation, "native_layers": [], "search_executed": False,
        "search": SEARCH, "search_points_Q24_units": list(search_points()),
        "producer_sha256": PRODUCER_SHA256, "suffix_sha256": SUFFIX_PINS,
        "margin_sha256": suffix.PINS, "L15_cut_suffix_sha256": margin.PINS,
        "authenticated_inputs": list(inputs.records.values()), "parent_binding": parent_binding,
        "frozen_first_failure": failure, "producer_status": "INCONCLUSIVE",
        "adjusted_suffix_status": "BOUNDED_COUNTERFACTUAL_PASS",
        "reference_policy": suffix.REFERENCE_POLICY,
        "claim_boundary": (
            "Read-only CPU-software search preflight, not numerical sensitivity or producer "
            "attribution. Historical failures and bounded Reviewer acceptances remain unchanged. "
            "Q24 state is wider than FP16; INT4 weights and FP16 operators/KV are unchanged. "
            "No native layer, RTL, GPU, FPGA, hardware, simulation, synthesis, PPA, "
            "strict-FP16-state W4A16, new-token or full-model PASS."
        ),
    }


@contextmanager
def native_only(audit, layer):
    require(type(layer) is int and layer in (21, 22, 23), "only native L21-L23 permitted")
    raw = producer.native.candidate._stages

    def forbidden(*args, **kwargs):
        raise RuntimeError("native replay, external execution or publication forbidden")

    def guarded(tensors, requested, parent, arrays):
        require(type(requested) is int and requested == layer, "native dispatch mismatch")
        require(len(audit) < EXECUTION["maximum_native_layer_invocations"],
                "native execution budget exceeded")
        audit.append(requested)
        yield from raw(tensors, requested, parent, arrays)

    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.model.candidates."):
                for attribute in ("stages", "continuation_stages", "native_layer",
                                  "run", "_stages", "save"):
                    if callable(getattr(module, attribute, None)):
                        replacement = (guarded if module is producer.native.candidate
                                       and attribute == "_stages" else forbidden)
                        stack.enter_context(patch.object(module, attribute, replacement))
        stack.enter_context(patch.object(subprocess, "Popen", forbidden))
        stack.enter_context(patch.object(os, "system", forbidden))
        yield


def parent_gate(parent, trajectory, reference):
    retained.verify_parent(parent, parent)
    return gates.evaluate_decoder_stage(
        stage=18, actual=parent["h"], reference=trajectory["stage18"],
        policy=gates.POLICY_ID, reference_binary64=reference)


def search(run_probe, run_suffix):
    samples = []
    result = {
        "status": "INCONCLUSIVE", "reason": "SEARCH_EXHAUSTED",
        "samples": samples, "search_exhausted": False, "minimality_proven": False,
    }
    for delta in search_points():
        row, output = run_probe(delta)
        samples.append(row)
        boundary = row["boundary"]
        if boundary is None:
            continue
        if boundary != "PASS_BOUNDARY":
            result["reason"] = boundary
            return result
        require(delta != 0, "retained baseline unexpectedly passed")
        neighbor_delta = delta - (1 if delta > 0 else -1)
        existing = next((item for item in samples
                         if item["delta_Q24_units"] == neighbor_delta), None)
        if existing is None:
            neighbor, _ = run_probe(neighbor_delta)
        else:
            neighbor = existing
        result.update(passing_delta_Q24_units=delta, one_unit_less=neighbor,
                      suffix=run_suffix(output), reason="SAMPLED_SUFFICIENCY_MINIMALITY_UNRESOLVED")
        # Adjacent failure alone does not establish a minimum in a nonmonotone search.
        if (abs(delta) == 1 and neighbor["boundary"] is None
                and result["suffix"]["status"] == "PASS"):
            result.update(status="BOUNDED_INPUT_ADJUSTMENT_PASS",
                          reason="MINIMUM_ABSOLUTE_NONZERO_DELTA", minimality_proven=True)
        return result
    result["search_exhausted"] = True
    return result


def execute(result):
    started = time.monotonic()
    result.update(preflight())
    result.update(status="BLOCKED", search_executed=False, samples=[],
                  native_layers=[], executor=EXECUTION, phase_seconds={})
    result["claim_boundary"] = (
        "Bounded non-admitting CPU input sensitivity only. Frozen historical failures and "
        "acceptances remain unchanged. Sampled failure does not establish non-local causation. "
        "Q24 state is wider than FP16; no strict-FP16-state W4A16, new-token, full-model, "
        "RTL, GPU, FPGA, hardware, simulation, synthesis or PPA claim."
    )
    inputs = margin.margin.prior.BoundInputs()
    for record in result["authenticated_inputs"]:
        inputs.bind(record)
    frozen = inputs.read(retained.record(margin.INPUT / "result.json"))
    entry = frozen["layers"][-1]
    require(entry["layer"] == 21 and entry["input_state_evidence"] == result["parent_binding"],
            "frozen L21 input substitution")
    parent = inputs.archive(result["parent_binding"])
    original = inputs.archive(entry["actual_stages"])
    expected_reports = inputs.read(entry["reports"])
    expected_locals = inputs.archive(entry["local_references"])
    upstream = margin.upstream.upstream
    freeze = inputs.read(retained.record(upstream.margin.INPUT / "freeze.json"))
    extension = inputs.read(freeze["reference_extension"])
    for layer in (20, 21):
        frozen_entry = next(item for item in frozen["layers"] if item["layer"] == layer)
        margin.check_reference(frozen_entry, extension["layers"][str(layer)])
    trajectories = {layer: inputs.archive(extension["layers"][str(layer)]["fp16"])
                    for layer in (20, 21, 22, 23)}
    references = {layer: upstream.global_reference(inputs, extension["layers"][str(layer)])
                  for layer in (20, 21, 22, 23)}
    require(parent_gate(parent, trajectories[20], references[20])["status"] == "PASS",
            "authenticated frozen L20 parent global gate failed")
    result["phase_seconds"]["compile_tests_authentication"] = time.monotonic() - started
    producer.native.candidate.torch.set_num_threads(1)
    audit = result["native_layers"]
    with upstream.safe_open(str(inputs.bind(extension["checkpoint"])), framework="numpy") as model:
        started = time.monotonic()
        tensors21 = {name: model.get_tensor(name) for name in producer.local.tensor_shapes(21)}
        producer.local.authenticate_tensors(tensors21, extension["layers"]["21"]["canonical"], 21)
        result["phase_seconds"]["L21_operand_loading"] = time.monotonic() - started

        def run_probe(delta):
            changed = perturb(parent, delta)
            gate = parent_gate(changed, trajectories[20], references[20])
            row = {"delta_Q24_units": delta, "L20_global_gate": gate,
                   "boundary": "L20_GLOBAL_GATE_BOUNDARY", "reports": []}
            if gate["status"] != "PASS":
                require(gate["status"] == "FAIL", "L20 gate blocked")
                print(json.dumps({"probe": row}, sort_keys=True), flush=True)
                return row, None
            with native_only(audit, 21):
                arrays, reports, changes, timings = producer.drive_control(
                    tensors21, changed, trajectories[21], references[21], ())
            require(not changes, "native operand substitution")
            if delta == 0:
                producer.same_arrays(arrays, original)
                require(reports == expected_reports, "retained baseline reports changed")
                for stage in range(18):
                    expected = (retained.transition_reference(parent, arrays["stage11"])["h"]
                                if stage == 12 else producer.local.local_reference(
                                    stage, {key: arrays[key].copy()
                                            for key in producer.local.OPERANDS[stage]}, tensors21, 21))
                    require(np.array_equal(expected, expected_locals[f"stage{stage:02d}"]),
                            "retained baseline local reference changed")
            row.update(boundary=producer.decision(reports), reports=reports,
                       phase_timings=timings, **producer.sample(arrays, reports, references[21]))
            print(json.dumps({"probe": row}, sort_keys=True), flush=True)
            output = (retained.state_from(arrays, "output", "stage18")
                      if row["boundary"] == "PASS_BOUNDARY" else None)
            return row, output

        def run_suffix(output):
            layers = []
            for layer in (22, 23):
                retained.verify_parent(output, output)
                tensors = {name: model.get_tensor(name)
                           for name in producer.local.tensor_shapes(layer)}
                producer.local.authenticate_tensors(
                    tensors, extension["layers"][str(layer)]["canonical"], layer)
                arrays, local_refs, reports = {}, {}, []
                with native_only(audit, layer):
                    failure, timings = suffix.drive_layer(
                        tensors, layer, output, trajectories[layer], references[layer],
                        arrays, local_refs, reports)
                layers.append({"layer": layer, "reports": reports, "timings": timings,
                               "original_reference": extension["layers"][str(layer)],
                               "prior_kv": "own empty P0", "prior_layer_kv_consumed": False})
                if failure:
                    return {"status": "FAIL", "failure": failure, "layers": layers}
                output = retained.state_from(arrays, "output", "stage18")
            return {"status": "PASS", "layers": layers}

        result["search_executed"] = True
        result.update(search(run_probe, run_suffix))
    require(origins() == result["validation"]["origins"], "module origins changed during search")
    for record in inputs.records.values():
        require(retained.record(record["path"]) == record,
                f"input/source changed during search: {record['path']}")
    result["authenticated_inputs"] = list(inputs.records.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = {"diagnostic_id": ID, "status": "BLOCKED", **FLAGS, "native_layers": []}
    try:
        if args.execute:
            execute(result)
        else:
            result = preflight()
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
            RuntimeError, SyntaxError, ImportError) as exc:
        result.update(status="BLOCKED", error=f"{type(exc).__name__}: {exc}")
        print(result["error"], file=sys.stderr)
    result["native_invocations"] = result["native_layer_invocations"] = len(result["native_layers"])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in (
        "PREFLIGHT_VALIDATED", "INCONCLUSIVE", "BOUNDED_INPUT_ADJUSTMENT_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
