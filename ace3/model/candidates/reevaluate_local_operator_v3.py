"""Re-adjudicate the pinned retained P0/L5-L8 software cone; never execute RTL."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import decoder_gate_policy as legacy
from ace3.model.candidates import decoder_gate_policy_v3 as policy
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates.run_single_round_residual_rtl import retained_record, retained_write


ROOT = Path(__file__).resolve().parents[3]
RETAINED = ROOT / "build/single_round_residual_rtl_execution_2f35dc72423f_policy_attempt001"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/2f35dc72423f/round-0001.json")


def read_words(path, indexed=False):
    rows = Path(path).read_text().splitlines()
    width = 10 if indexed else 4
    local.require(all(re.fullmatch(f"[0-9a-fA-F]{{{width}}}", row) for row in rows),
                  "invalid FP16 hex ABI")
    if indexed:
        local.require([int(row[:-4], 16) for row in rows] == list(range(len(rows))),
                      "indexed hidden coordinate identity mismatch")
    return np.asarray([int(row[-4:], 16) for row in rows], dtype="<u2")


def run(out, selected):
    local.require(selected == policy.POLICY_ID, "explicit v3 opt-in required")
    out.mkdir(parents=True, exist_ok=False)
    bound, reports, provenance, timings = {}, [], [], []
    phase, active = "authentication", None
    started = time.monotonic()

    def bind(rec):
        signature = {key: rec[key] for key in ("path", "bytes", "sha256")}
        path = signature["path"]
        if path in bound:
            local.require(bound[path] == signature, f"conflicting artifact bindings: {path}")
        else:
            local.require(retained_record(path) == signature, f"artifact binding drift: {path}")
            bound[path] = signature
        return Path(path)

    def read(rec):
        return json.loads(bind(rec).read_text())

    try:
        contract = json.loads(policy.CONTRACT.read_text())
        manifest_rec = retained_record(RETAINED / "output_manifest.json")
        local.require(manifest_rec["sha256"] == contract["trusted_retained_manifest_sha256"],
                      "unreviewed retained manifest")
        manifest = read(manifest_rec)
        members = {rec["path"]: rec for rec in manifest["artifacts"]}
        local.require(len(members) == len(manifest["artifacts"]), "duplicate manifest member")
        old_result = read(members[str(RETAINED / "result.json")])
        local.require(members[str(RETAINED / "result.json")]["sha256"] ==
                      contract["trusted_retained_result_sha256"], "unreviewed retained result")
        frozen = read(old_result["freeze"])
        review_rec = retained_record(REVIEW)
        review = read(review_rec)
        local.require(review["kind"] == "round_reviewed_handoff"
                      and review["producer_role"] == "reviewer"
                      and review["mission_id"] == "2f35dc72423f"
                      and review["review"]["status"] == "done", "retained independent review missing")
        local.require(frozen["policy_id"] == legacy.POLICY_ID
                      and frozen["scope"] == {"layers": [5, 6, 7, 8], "position": 0, "history": [9707]}
                      and frozen["semantics"] ==
                      "RNE16(H + O + D), exact three-term sum, only one rounding"
                      and frozen["input_state"] is None
                      and old_result["decoder_rtl_invocations"] == 0, "retained candidate lineage mismatch")
        for rec in frozen["input_bindings"] + frozen["sources"]:
            bind(rec)
        inputs = {r["path"]: r for r in frozen["input_bindings"]}

        def ending(suffix):
            matches = [r for p, r in inputs.items() if p.endswith(suffix)]
            local.require(len(matches) == 1, f"missing/ambiguous frozen input: {suffix}")
            return matches[0]

        independent_rec = ending("/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010/frozen.json")
        local.require(independent_rec["sha256"] == contract["trusted_independent_freeze_sha256"],
                      "wrong independent model/control freeze")
        independent = read(independent_rec)
        specification = read(independent["reference_source"])
        local.require(independent["checkpoint"] == specification["checkpoint"]
                      and independent["checkpoint_tensors"] == specification["checkpoint_tensors"],
                      "canonical model metadata disagreement with original reference source")
        checkpoint = bind(specification["checkpoint"])
        canonical = {r["name"]: r for r in specification["checkpoint_tensors"]}
        local.require(len(canonical) == len(specification["checkpoint_tensors"]),
                      "duplicate canonical tensors")
        specification_sources = [r for r in specification["sources"]
                                 if Path(r["path"]).name in
                                 ("official_single_decoder_layer.py", "official_model24_next_token.py")]
        local.require({Path(r["path"]).name for r in specification_sources} ==
                      {"official_single_decoder_layer.py", "official_model24_next_token.py"},
                      "missing independently frozen operator specification")
        for rec in specification_sources:
            bind(rec)
        hidden = read_words(bind(frozen["input_hidden"]), indexed=True)
        local.require(hashlib.sha256(hidden.tobytes()).hexdigest() ==
                      frozen["input_hidden"]["semantic_sha256"], "accepted L4 hidden identity mismatch")
        reference_records = {layer: ending(f"/layer{layer:02d}_position000_reference.npy")
                             for layer in range(5, 9)}
        reference_root = Path(reference_records[5]["path"]).parent
        reference_result = read(inputs[str(reference_root / "result.json")])
        with bind(reference_result["full_binary64_deltas"]).open(newline="") as stream:
            original_rows = [r for r in csv.DictReader(stream)
                             if int(r["position"]) == 0 and 5 <= int(r["layer"]) <= 8]
        original64 = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"]
                      for r in original_rows}
        local.require(len(original64) == len(original_rows) == 4 * 896,
                      "original global reference coverage/identity mismatch")
        global_references = {}
        for layer, rec in reference_records.items():
            values = np.load(bind(rec), allow_pickle=False)
            local.require(values.dtype == np.dtype("<f8") and values.shape == (896,)
                          and all(float(v).hex() == original64[layer, i]
                                  for i, v in enumerate(values)),
                          "original global trajectory changed")
            global_references[layer] = values
        compiler = shutil.which("iverilog")
        local.require(compiler is not None, "public-contract compiler unavailable; no RTL conclusion")
        contract_source = out / "public_contract.sv"
        shutil.copyfile(bind(frozen["public_contract_source"]), contract_source)
        command = [compiler, "-g2012", "-s", "frozen_contract", "-o",
                   str(out / "public_contract.vvp"), str(contract_source)]
        sources = [retained_record(p) for p in (
            Path(__file__), Path(policy.__file__), Path(local.__file__), policy.CONTRACT,
            Path(legacy.__file__), Path(legacy.binary64.__file__),
            ROOT / "ace3/model/fp16_adaptation_oracle.py",
            ROOT / "ace3/model/awq_bit_oracle.py")]
        retained_write(out / "freeze.json", {
            "policy_id": selected, "contract": retained_record(policy.CONTRACT), "sources": sources,
            "retained_manifest": manifest_rec, "retained_review": review_rec,
            "actual_input_root": frozen["input_hidden"], "candidate_semantics": frozen["semantics"],
            "input_state": None, "scope": frozen["scope"],
            "canonical_model": independent_rec, "operator_specification": independent["reference_source"],
            "global_reference_source": inputs[str(reference_root / "result.json")],
            "global_reference_inputs": reference_records,
            "global_reference_policy": legacy.binary64.REFERENCE_POLICY,
            "local_reference_input_policy": "Only declared actual producer DATA, never candidate output",
            "input_bindings": list(bound.values()), "public_contract": frozen["public_contract"],
            "parameters": frozen["parameters"], "public_contract_source": retained_record(contract_source),
            "compile_command": command, "tools": {"python": sys.version, "numpy": np.__version__,
                "platform": platform.platform(), "iverilog": retained_record(compiler)},
            "evaluation_order": "Ordered L5-L8; S0-S17 local mandatory then S18 global mandatory",
            "decoder_rtl_invocations": 0, "normal_host_review": "REQUIRED"})
        phase = "public_contract_compile"
        retained_write(out / "public_contract_compile.command.json", command)
        with (out / "public_contract_compile.log").open("xb") as stream:
            compiled = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        retained_write(out / "public_contract_compile.result.json", {"returncode": compiled.returncode,
                                                                   "rtl_execution": False})
        local.require(compiled.returncode == 0, "public-contract compilation failed")
        first_failure = None
        for layer in range(5, 9):
            begin = time.monotonic()
            active, phase = [layer, 0], "local_reference_input_authentication"
            rows = [r for r in old_result["software_provenance"] if r["layer"] == layer]
            local.require(len(rows) == 1 and rows[0]["input_state"] is None
                          and rows[0]["kv_policy"] == "own empty prior P0 KV",
                          "missing compatible retained software lineage")
            rec = rows[0]["stages"]
            local.require(rec == (members if layer == 8 else inputs)[rec["path"]],
                          "unmanifested software archive")
            with np.load(bind(rec), allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            with safe_open(str(checkpoint), framework="numpy") as model:
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
            transactions = [r for r in independent["reference_transactions"]
                            if r["layer"] == layer and r["position"] == 0]
            local.require(len(transactions) == 1 and set(transactions[0]["stages"]) ==
                          {str(s) for s in range(19)}, "independent trajectory coverage mismatch")
            trajectory, trajectory_records = {}, {}
            for stage, record in transactions[0]["stages"].items():
                values = read_words(bind(record))
                local.require(hashlib.sha256(values.tobytes()).hexdigest() ==
                              record["semantic_sha256"], "trajectory tensor identity drift")
                trajectory[int(stage)] = values
                trajectory_records[stage] = record
            directory = out / f"layer{layer:02d}"
            directory.mkdir()

            def emit(report):
                nonlocal active
                active = report["node"]
                report.update(evidence_kind="retained_software", actual_archive=rec,
                              original_trajectory=trajectory_records[str(report["stage"])])
                if report["stage"] == 18:
                    report["global_reference"] = reference_records[layer]
                old_path = str(RETAINED / f"layer{layer:02d}/stage{report['stage']:02d}_evaluation.json")
                if old_path in members:
                    previous = read(members[old_path])
                    local.require(report["fp16"] == previous["fp16"], "legacy diagnostic changed")
                retained_write(directory / f"stage{report['stage']:02d}_evaluation.json", report)

            phase = "ordered_v3_evaluation"
            transaction = policy.evaluate_p0_transaction(
                arrays=arrays, tensors=tensors, canonical_records=canonical, expected_hidden=hidden,
                trajectory=trajectory, reference_binary64=global_references[layer], layer=layer,
                position=0, history=[9707], policy=selected, emit=emit)
            with (directory / "local_references.npz").open("xb") as stream:
                np.savez(stream, **transaction.pop("local_references"))
            retained_write(directory / "lineage.json", transaction["lineage"])
            reports.extend(transaction["reports"])
            provenance.append({"layer": layer, "actual_archive": rec, "source": rows[0],
                               "lineage": transaction["lineage"],
                               "local_references": retained_record(directory / "local_references.npz"),
                               "canonical_tensors": [canonical[name] for name in tensors]})
            timings.append({"layer": layer, "seconds": time.monotonic() - begin})
            if transaction["status"] != "PASS":
                first_failure = transaction["reports"][-1]
                break
            hidden = arrays["stage18"].copy()
        result = {
            "status": first_failure["status"] if first_failure else "SUPPORTED_SOFTWARE_ONLY",
            "policy_id": selected, "first_failure": first_failure, "software_provenance": provenance,
            "mandatory_local_operator_fp16": {
                "gates": sum("local_operator_fp16" in r for r in reports),
                "coordinates": sum(r.get("local_operator_fp16", {}).get("comparisons", 0) for r in reports),
                "failures": sum(r.get("local_operator_fp16", {}).get("failure_count", 0) for r in reports)},
            "mandatory_global_binary64_v1": [
                {"node": r["node"], "coordinates": r["binary64_v1"]["coordinates"],
                 "failure_count": r["binary64_v1"]["failure_count"]}
                for r in reports if "binary64_v1" in r],
            "legacy_trajectory_diagnostics": [
                {"node": r["node"], "comparisons": r["fp16"]["comparisons"],
                 "failure_count": r["fp16"]["failure_count"], "failures": r["fp16"]["failures"]}
                for r in reports],
            "independent_lineage_state": [p["lineage"] for p in provenance],
            "failure_taxonomy": "candidate_numerical_gate" if first_failure else None,
            "root_cause_hypothesis": (
                "Local operator implementation error or global accumulated drift at the named gate; "
                "use its authenticated operand archive and frozen canonical tensors for a same-input control."
                if first_failure else None),
            "regression": "Keep the first failed full-vector gate and original diagnostic; no coordinate patch.",
            "phase_times": timings, "elapsed_seconds": time.monotonic() - started,
            "decoder_rtl_invocations": 0, "ancestors_replayed": [], "normal_host_review": "REQUIRED",
            "no_full_model_or_generation_admission": True,
            "rtl_result_admission": "Not established: runtime source/input/saved-state adapter remains required"}
        retained_write(out / "result.json", result)
        return 1 if first_failure else 0
    except (OSError, ValueError, KeyError, FloatingPointError, subprocess.SubprocessError) as error:
        retained_write(out / "result.json", {
            "status": "BLOCKED", "policy_id": selected, "phase": phase, "active": active,
            "error": str(error), "failure_taxonomy": "evaluator_no_completion",
            "root_cause_hypothesis": "The named mandatory input/source/reference boundary is unavailable or invalid.",
            "regression": "Reject missing, corrupt or unverifiable evidence; never substitute candidate outputs.",
            "decoder_rtl_invocations": 0, "no_rtl_correctness_conclusion": True,
            "normal_host_review": "REQUIRED"})
        raise
    finally:
        retained_write(out / "output_manifest.json", {
            "artifacts": [retained_record(p) for p in sorted(out.rglob("*")) if p.is_file()]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate-policy", choices=[policy.POLICY_ID], required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    local.require(out.is_relative_to(ROOT / "build"), "evidence must remain under ignored build")
    return run(out, args.gate_policy)


if __name__ == "__main__":
    raise SystemExit(main())
