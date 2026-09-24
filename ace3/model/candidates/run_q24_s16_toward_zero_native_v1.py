"""Freeze and execute one fresh, independently checked L0-L2/P0 CPU candidate."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np
import safetensors
from safetensors import safe_open
import torch

from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_s16_toward_zero_native_v1 as candidate
from ace3.model.candidates import q24_software_root_v3 as retained


ROOT = retained.ROOT
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_toward_zero_native_v1.json"
HISTORY = ROOT / "build/q24_software_root_e37713d52f39_attempt001"
require = local.require


def execute(out, preserve_attempt=None, *, resume_parent=None):
    pattern = (r"q24_software_l3_l8_[a-z0-9]+_attempt[0-9]+" if resume_parent is not None
               else r"q24_software_root_[a-z0-9]+_attempt[0-9]+")
    require(out.parent == ROOT / "build"
            and re.fullmatch(pattern, out.name),
            "fresh ignored Q24 build attempt required")
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return run(out, preserve_attempt, resume_parent=resume_parent)


def run(out, preserve_attempt=None, *, resume_parent=None):
    started = time.monotonic()
    bound, layers, phases = {}, [], []
    phase, node, failure = "authentication", None, None
    status = "BLOCKED"
    selected_layers = range(3, 9) if resume_parent is not None else range(3)
    contract_path = (ROOT / "ace3/contracts/candidates/q24_s16_toward_zero_l3_l8_v1.json"
                     if resume_parent is not None else CONTRACT)
    module = ("ace3.model.candidates.run_q24_s16_toward_zero_l3_l8_v1"
              if resume_parent is not None
              else "ace3.model.candidates.run_q24_s16_toward_zero_native_v1")
    continuation = None

    def bind(rec):
        signature = {key: rec[key] for key in ("path", "bytes", "sha256")}
        path = signature["path"]
        if path not in bound:
            require(retained.record(path) == signature, f"binding mismatch: {path}")
            bound[path] = signature
        require(bound[path] == signature, f"conflicting binding: {path}")
        return Path(path)

    def read(rec):
        return json.loads(bind(rec).read_text())

    def bind_tree(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                bind(value)
            else:
                for child in value.values():
                    bind_tree(child)
        elif isinstance(value, list):
            for child in value:
                bind_tree(child)

    try:
        contract = read(retained.record(contract_path))
        require(contract["policy_id"] == gates.POLICY_ID
                and contract["candidate_id"] == "ace3-q24-s16-toward-zero-native-v1"
                and contract["scope"] == {"layers": list(selected_layers), "position": 0, "history": [9707]},
                "candidate contract mismatch")
        if resume_parent is not None:
            from ace3.model.candidates.run_q24_s16_toward_zero_l3_l8_v1 import authenticate_resume
            require(preserve_attempt is None, "continuation cannot replace a root attempt")
            continuation = authenticate_resume(resume_parent, bind, read, bind_tree)
        prior_record = None
        if preserve_attempt is not None:
            require(preserve_attempt.parent == ROOT / "build"
                    and re.fullmatch(r"q24_software_root_[a-z0-9]+_attempt[0-9]+", preserve_attempt.name)
                    and preserve_attempt != out, "invalid prior candidate attempt")
            prior_record = retained.record(preserve_attempt / "result.json")
            prior_result = read(prior_record)
            require(prior_result["status"] == "FAIL"
                    and prior_result["candidate_id"] == contract["candidate_id"],
                    "repair must preserve an actual failed candidate")
            prior_freeze = read(retained.record(preserve_attempt / "freeze.json"))
            for source in prior_freeze["sources"]:
                bind(source["snapshot"])
            bind_tree(prior_result["layers"])
            bind(retained.record(preserve_attempt / "command.log"))
        review = read(retained.record(retained.HANDOFFS / "e37713d52f39/round-0001.json"))
        require(review["kind"] == "round_reviewed_handoff"
                and review["producer_role"] == "reviewer"
                and review["mission_id"] == "e37713d52f39"
                and review["review"]["status"] == "done", "missing genuine selection review")
        historical = read(retained.record(HISTORY / "result.json"))
        bind(retained.record(HISTORY / "freeze.json"))
        bind_tree(historical["arms"])
        arms = {arm["rounding"]: arm for arm in historical["arms"]}
        require(set(arms) == {"rne", "toward_zero", "away_zero"}
                and arms["rne"]["status"] == "FAIL"
                and arms["rne"]["witness"]["excess_error"] == "1/2"
                and arms["rne"]["witness"]["actual_fp16_bits"] == "616e"
                and all(not arm["candidate_admitted"] for arm in arms.values()),
                "historical diagnostic/admission boundary changed")
        old = ROOT / "build/q24_software_root_995d0ba22311_attempt002"
        old_result = read(retained.record(old / "result.json"))
        require(old_result["status"] == "FAIL", "original native Q24 failure changed")
        for name in ("freeze.json", "layer02/reports.json", "layer02/actual_stages.npz"):
            bind(retained.record(old / name))

        policy = read(retained.record(gates.CONTRACT))
        b_record = retained.record(retained.B_FREEZE)
        require(b_record["sha256"] == policy["trusted_independent_freeze_sha256"],
                "wrong independent model freeze")
        b = read(b_record)
        control_record = retained.record(retained.CONTROL)
        require(control_record["sha256"] == retained.CONTROL_SHA, "wrong original global control")
        control = read(control_record)
        require(b["checkpoint"] == control["checkpoint"] and b["embeddings"] == control["embeddings"]
                and b["reference_source"] == control["fp16_reference_root"],
                "original root/reference lineage mismatch")
        spec = read(b["reference_source"])
        require(spec["checkpoint"] == b["checkpoint"]
                and spec["checkpoint_tensors"] == b["checkpoint_tensors"],
                "canonical tensor specification mismatch")
        read(control["binary64_reference_freeze"])
        read(control["binary64_array_binding_freeze"])
        with bind(control["binary64_csv"]).open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream)
                    if int(r["position"]) == 0 and int(r["layer"]) in selected_layers]
        originals = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"] for r in rows}
        require(len(originals) == len(rows) == 896 * len(selected_layers),
                "original global coordinate coverage")
        references, trajectories = {}, {}
        for key, case in zip(control["ordered_cases"], control["cases"], strict=True):
            layer, position = key
            if position == 0 and layer in selected_layers:
                require(layer not in references, "duplicate original global layer")
                values = np.load(bind(case["binary64_reference_array"]), allow_pickle=False)
                require(values.shape == (896,) and values.dtype == np.dtype("<f8")
                        and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504)
                        and all(float(v).hex() == originals[layer, i] for i, v in enumerate(values)),
                        "original global array differs from frozen CSV")
                references[layer] = values
        for transaction in b["reference_transactions"]:
            layer = transaction["layer"]
            if transaction["position"] == 0 and layer in selected_layers:
                require(layer not in trajectories
                        and set(transaction["stages"]) == {str(s) for s in range(19)},
                        "FP16 trajectory coverage/order")
                trajectories[layer] = {}
                for stage, rec in transaction["stages"].items():
                    words = retained.read_words(bind(rec), local.SIZES[int(stage)])
                    require(hashlib.sha256(words.tobytes()).hexdigest() == rec["semantic_sha256"],
                            "trajectory semantic identity")
                    trajectories[layer][int(stage)] = words
        require(set(references) == set(trajectories) == set(selected_layers), "missing mandatory references")
        embedding_record = b["embeddings"][0]
        require(embedding_record["token"] == 9707, "wrong root token")
        embedding = retained.read_words(bind(embedding_record["input"]), 896, indexed=True)
        require(hashlib.sha256(embedding.tobytes()).hexdigest() == embedding_record["semantic_sha256"],
                "embedding semantic identity")
        canonical = {r["name"]: r for r in b["checkpoint_tensors"]}
        require(len(canonical) == len(b["checkpoint_tensors"]), "duplicate canonical tensors")
        tensors = {}
        with safe_open(str(bind(b["checkpoint"])), framework="numpy") as model:
            official = model.get_slice("model.embed_tokens.weight")[9707:9708]
            require(official.dtype == np.dtype("<f2") and official.shape == (1, 896)
                    and np.array_equal(official.reshape(-1).view("<u2"), embedding),
                    "root not official checkpoint embedding9707")
            for layer in selected_layers:
                tensors[layer] = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors[layer], canonical, layer)

        phase = "freeze"
        paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
                 if getattr(m, "__file__", None) and str(m.__file__).endswith(".py")
                 and str(Path(m.__file__).resolve()).startswith(str(ROOT / "ace3/model/"))}
        paths.update((CONTRACT, contract_path, gates.CONTRACT,
                      ROOT / "ace3/contracts/candidates/binary64_fp16_excess_v1.json",
                      ROOT / "tests/test_q24_s16_toward_zero_native_v1.py",
                      ROOT / "ace3/model/tests/test_q24_software_root_v3.py",
                      ROOT / "ace3/model/tests/test_decoder_gate_policy_v3.py"))
        if continuation is not None:
            paths.add(ROOT / "tests/test_q24_s16_toward_zero_l3_l8_v1.py")
        source_dir = out / "source"
        source_dir.mkdir()
        sources = []
        for index, path in enumerate(sorted(paths)):
            rec = retained.record(path)
            bind(rec)
            copy = source_dir / f"{index:02d}_{path.name}"
            shutil.copyfile(path, copy)
            snapshot = retained.record(copy)
            require(snapshot["sha256"] == rec["sha256"], "source snapshot drift")
            sources.append({"original": rec, "snapshot": snapshot})
        retained.write(out / "freeze.json", {
            "contract": contract, "sources": sources, "input_bindings": list(bound.values()),
            "command": [sys.executable, "-m", module,
                        "--out", str(out)] + (["--preserve-attempt", str(preserve_attempt)]
                                             if preserve_attempt is not None else [])
                        + (["--resume-parent", str(resume_parent)] if resume_parent is not None else []),
            "prior_native_attempt": prior_record,
            "cwd": str(ROOT), "tools": {"python": sys.version, "numpy": np.__version__,
                "torch": torch.__version__, "safetensors": safetensors.__version__,
                "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                "thread_environment": {k: os.environ.get(k) for k in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
            "arithmetic_lineage": "fresh CPU native canonical implementation; RTZ only at S16",
            "state_lineage": {"root": embedding_record,
                              "prior_residual": continuation, "prior_kv": "own empty P0"},
            "global_reference_lineage": {"model": b_record, "control": control_record,
                                         "candidate_data_used": False},
            "historical_evidence": {"result": retained.record(HISTORY / "result.json"),
                                    "review": retained.record(retained.HANDOFFS / "e37713d52f39/round-0001.json")},
            "rtl_backlog": "preserved and excluded; no RTL/scheduler operations",
            "normal_host_review": "REQUIRED"})
        phases.append({"phase": "authentication_and_freeze", "seconds": time.monotonic() - started})
        if continuation is None:
            parent = candidate.lift(embedding)
            retained.verify_parent(parent, parent, embedding=embedding)
            parent_record = retained.save(out / "root_state.npz", parent)
        else:
            parent_record = continuation["parent"]["state"]
            with np.load(bind(parent_record), allow_pickle=False) as saved:
                parent = {key: saved[key].copy() for key in saved.files}
            retained.verify_parent(parent, parent)
        for layer in selected_layers:
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            with np.load(bind(parent_record), allow_pickle=False) as saved:
                loaded = {key: saved[key].copy() for key in saved.files}
            retained.verify_parent(loaded, parent, embedding=embedding if layer == 0 else None)
            parent = loaded
            arrays, local_refs, reports = {}, {}, []
            compute_seconds, oracle_seconds = 0.0, 0.0
            stage_producer = candidate.stages if continuation is None else candidate.continuation_stages
            producer = stage_producer(tensors[layer], layer, parent, arrays)
            try:
                for stage in range(19):
                    node, phase = [layer, 0, stage], "candidate_arithmetic"
                    stamp = time.monotonic()
                    require(next(producer) == stage, "out-of-order native stage")
                    compute_seconds += time.monotonic() - stamp
                    stamp = time.monotonic()
                    phase = "state_lineage"
                    expected = retained.check_stage_state(stage, arrays, parent)
                    phase = "independent_local_reference"
                    if stage < 18 and stage != 12:
                        expected = local.local_reference(stage, {
                            key: arrays[key].copy() for key in local.OPERANDS[stage]}, tensors[layer], layer)
                    if stage < 18:
                        local_refs[f"stage{stage:02d}"] = expected
                    phase = "global_numerical" if stage == 18 else "local_operator_numerical"
                    report = gates.evaluate_decoder_stage(
                        stage=stage, actual=arrays[f"stage{stage:02d}"],
                        reference=trajectories[layer][stage], policy=gates.POLICY_ID,
                        local_reference=expected,
                        reference_binary64=references[layer] if stage == 18 else None)
                    report.update(node=node, residual_state_lineage="PASS", kv_lineage="PASS",
                                  local_reference_independent=stage < 18,
                                  operand_names=["input_i", "input_z", "stage11"] if stage == 12
                                  else list(local.OPERANDS[stage]) if stage < 18 else None)
                    reports.append(report)
                    oracle_seconds += time.monotonic() - stamp
                    if report["status"] != "PASS":
                        failure = {"node": node, "failure_taxonomy": phase,
                            "root_cause_hypothesis": "Native RTZ cone misses this unchanged mandatory gate; "
                            "canonical counterfactual success alone did not establish this implementation.",
                            "regression": str(directory / "reports.json")}
                        break
            finally:
                producer.close()
                archive = retained.save(directory / "actual_stages.npz", arrays)
                local_record = retained.save(directory / "local_references.npz", local_refs)
                retained.write(directory / "reports.json", reports)
            entry = {"layer": layer, "position": 0, "status": "FAIL" if failure else "PASS",
                     "input_parent": parent_record, "actual_stages": archive,
                     "local_references": local_record, "reports": retained.record(directory / "reports.json"),
                     "stage_outcomes": {str(r["stage"]): r["status"] for r in reports},
                     "local_vectors": sum(r["stage"] < 18 for r in reports),
                     "local_coordinates": sum(local.SIZES[r["stage"]] for r in reports if r["stage"] < 18),
                     "global_coordinates": 896 if len(reports) == 19 else 0,
                     "state_and_kv_identity": "PASS", "candidate_seconds": compute_seconds,
                     "oracle_seconds": oracle_seconds}
            if len(reports) == 19:
                entry["global_witness_index62"] = reports[18]["binary64_v1"]["rows"][62]
            layers.append(entry)
            print(json.dumps(entry), flush=True)
            if failure:
                status = "FAIL"
                break
            parent = retained.state_from(arrays, "output", "stage18")
            parent_record = retained.save(directory / "parent_state.npz", parent)
            entry["output_parent"] = parent_record
            entry["kv_state"] = retained.save(directory / "kv_state.npz",
                {"k": arrays["output_cache_k"], "v": arrays["output_cache_v"]})
            retained.write(directory / "software_parent.json", {
                "candidate_id": contract["candidate_id"], "state_id": contract["state_id"],
                "policy_id": gates.POLICY_ID, "model": contract["model"], "history": [9707],
                "position": 0, "next_layer": layer + 1, "state": parent_record, "kv": entry["kv_state"],
                "state_lineage_parent": entry["input_parent"], "arithmetic_lineage": retained.record(out / "freeze.json"),
                "numerical_report": entry["reports"], "evidence_kind": "cpu_software_q24",
                "rtl_admissible": False, "normal_host_review": "REQUIRED"})
        else:
            status = "PASS"
        phase = "final_binding_integrity"
        for rec in bound.values():
            require(retained.record(rec["path"]) == rec, f"bound artifact changed: {rec['path']}")
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError, StopIteration, RuntimeError) as exc:
        status = "BLOCKED"
        failure = {"node": node, "failure_taxonomy": phase, "reason": f"{type(exc).__name__}: {exc}",
                   "root_cause_hypothesis": f"Unverifiable {phase} dependency; no numerical admission",
                   "regression": "captured command and source/input freeze"}
        print(json.dumps(failure), flush=True)
    result = {"status": status, "candidate_admitted": status == "PASS",
              "candidate_id": "ace3-q24-s16-toward-zero-native-v1",
              "candidate_admission_scope": "bounded CPU numerical/state gates only; Host review still required",
              "evidence_kind": "cpu_software_q24", "policy_id": gates.POLICY_ID,
              "layers": layers, "first_failure": failure, "phase_times": phases,
              "seconds": time.monotonic() - started, "normal_host_review": "REQUIRED",
              "historical_fail_preserved": phase == "final_binding_integrity" and status != "BLOCKED",
              "rtl_invocations": 0, "l9_invocations": 0, "policy_adopted": False,
              "scope": {"layers": list(selected_layers), "position": 0, "history": [9707]},
              "claim_boundary": ("L3-L8/P0 from reviewed L2 software parent only; "
                                 if continuation is not None else "L0-L2/P0 token9707 only; ")
                                + "no production, strict-FP16-state, RTL, token or full-model admission"}
    retained.write(out / "result.json", result)
    print(json.dumps({"status": status, "candidate_admitted": result["candidate_admitted"],
                      "seconds": result["seconds"]}), flush=True)
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preserve-attempt", type=Path)
    args = parser.parse_args()
    return execute(args.out.resolve(), args.preserve_attempt.resolve() if args.preserve_attempt else None)


if __name__ == "__main__":
    raise SystemExit(main())
