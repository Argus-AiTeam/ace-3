"""Read-only final RMSNorm/tied-head preparation for nine reviewed raw L23 states.

Run with PYTHONDONTWRITEBYTECODE=1 and --check [--out build/<version>_attemptNNN].
Only JSON is emitted to stdout. This version accepts final references only from
the exact reviewed original-input attempt003 bundle, never another token or candidate.
A missing final reference is a blocked preparation, not execution authority.
Asset-binding revision 3 binds that isolated bundle and its terminal Reviewer
record without changing the retained original-input manifest.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from ace3.model.candidates import q24_s16_l23_from_l22_coordinate62_suffix_execution_v1 as parent


preflight = parent.preflight
matrix = preflight.parent.matrix
census = preflight.census
require, same, read_bound, record = (
    preflight.require, preflight.same, preflight.read_bound, preflight.record)
ROOT = parent.ROOT
NAME = "q24_s16_final_head_from_l23_coordinate62_suffix_preflight_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
BASE = parent.OUTPUT
MISSION = "a1d2195ce52f"
ASSET_BINDING_REVISION = 3
FINAL_REFERENCE_NAME = "q24_s16_final_original_reference_v1"
FINAL_REFERENCE_MISSION = "eff468ba33e5"
FINAL_REFERENCE_OUTPUT = ROOT / "build" / (FINAL_REFERENCE_NAME + "_attempt003")
FINAL_REFERENCE_PINS = {
    "review": {
        "path": str(preflight.HANDOFFS / FINAL_REFERENCE_MISSION / "round-0001.json"),
        "sha256": "8750b7ef41a46f33b6d1f1dc9750b4abfa580cd8423e8695162f0210aaf3c38f",
        "bytes": 690,
    },
    "mission": {
        "path": str(preflight.HANDOFFS / FINAL_REFERENCE_MISSION / "mission.json"),
        "sha256": "155722776e22cd67faa2396bc7c23325aa637e817ba10443a396d8c8bd1e99b4",
        "bytes": 2247,
    },
    "manifest": {
        "path": str(FINAL_REFERENCE_OUTPUT / "freeze.json"),
        "sha256": "87b4746306ecaba3c106f0d6f3ec6c112c2ad3856386458a23ef463c04a98feb",
        "bytes": 316192,
    },
}
TOKENIZER_DIR = ROOT / "build/host_dialogue_audit_20260829T182819Z/tokenizer"
FINAL_ARRAYS = (
    ("rmsnorm_binary64", (896,), "<f8"), ("rmsnorm_fp16", (896,), "<u2"),
    ("logits_binary64", (151936,), "<f8"), ("logits_fp16", (151936,), "<u2"),
)
CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62",
    "mapped_all", "inherited_native",
)
PINS = {
    "review": {
        "path": str(preflight.HANDOFFS / MISSION / "round-0002.json"),
        "sha256": "4860fc28b5d5ec614e9a522193d0490fc66940ca4b884882df17be26e7b86763",
    },
    "mission": {
        "path": str(preflight.HANDOFFS / MISSION / "mission.json"),
        "sha256": "527d07cf1b64adea19f14e57c9901b71a11ca064d39ee7f0b0d27cb528a3d4f8",
    },
    "result": {
        "path": str(BASE / "result.json"),
        "sha256": "e0f23a1d333ce4389b936a900e2e2881def2ddea07a9a10fa50a554f5d70aac9",
    },
    "validation": {
        "path": str(BASE / "validation.json"),
        "sha256": "3b465c2d28c459351832c0a978aaa2622e0d836e9f4183ca136a4ebfa2add2fd",
    },
}
FLAGS = {
    **preflight.FLAGS, "native_L0_L23_invocations": 0,
    "prefix_invocations": 0, "admission_invocations": 0,
    "final_rmsnorm_invocations": 0, "lm_head_invocations": 0, "top_k_invocations": 0,
    "tokenizer_decode_invocations": 0,
    "reference_rmsnorm_invocations": 0, "reference_lm_head_invocations": 0,
    "reference_suffix_computation": False, "retained_evidence_writes": 0,
}
FINAL_CONTRACT = {
    "history": [9707], "position": 0, "parent_node": [23, 0, 18],
    "state": {**preflight.STATE,
              "transformation": "none; entire raw L23 output I/Z/H, not a coordinate cut"},
    "rmsnorm": {
        "input": "stage18; authenticated FP16 projection of retained Q24 I/Z",
        "tensor": "model.norm.weight", "shape": [896], "dtype": "float16",
        "epsilon": "1/1000000", "output": "896 finite FP16 words",
    },
    "head": {
        "tensor": "lm_head.weight", "tied_to": "model.embed_tokens.weight",
        "shape": [151936, 896], "dtype": "float16",
        "tied_value_sha256": "d74257dc547b48be5ae7b93f1c9af072c0c42dbbb85503078e25c59cd09e68d0",
        "arithmetic": "exact FP16 products; Q47.48 accumulation; one FP16 RNE rounding",
        "top_k": 10, "order": "descending finite FP16 logit, ascending token_id on ties",
    },
    "kv": "retained own P0 FP16 KV lineage; final RMSNorm/head do not read or write KV",
    "weights": "unchanged native G128 asymmetric packed INT4 GEMM ordering; no qzero plus-one",
    "boundaries": "FP16 scales, operators and KV unchanged; tied head is official F16, not INT4",
}
BOUNDARY = (
    "Preparation only for nine isolated failed-parent Q24 CPU-software suffixes. "
    "Q24 residual state is wider than FP16. No final-head execution, original or "
    "accepted prefix/admission replay, reference recomputation/reanchoring, evidence "
    "writes, successor publication, RTL, hardware, GPU or simulation. No "
    "strict-FP16-state W4A16, new-token or full-model admission claim. Retain every "
    "historical failure and exact threshold. A separately reviewed executor must "
    "repeat source/identity/role-model/account/budget/access/lock/concurrency gates. "
    "Normal independent Host Reviewer validation of this new preflight is required."
)


def output_path(value):
    path = Path(value)
    out = path if path.is_absolute() else ROOT / path
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(NAME + r"_attempt[0-9]{3,}", out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0,
            "output must be a canonical versioned direct child of build")
    require(not out.exists() and not out.is_symlink(), "output occupied or symlinked")
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored: " +
            ignored.stderr.decode(errors="replace"))
    return out


def check_review(review, latest, backlog, *, mission=MISSION, round_number=2, pin=None):
    pin = PINS["review"] if pin is None else pin
    matrix.check_review(review, mission, round_number)
    require(latest["kind"] == "handoff_ref"
            and latest["handoff"]["path"] == pin["path"],
            mission + " review is not the resolved terminal review")
    rows = [row for row in backlog if row["id"] == mission]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            mission + " review lacks terminal backlog corroboration")


def check_reports(row, reports, arrays, reference):
    label = row["control"]
    failures = [62] if label == "mapped_all" else [62, 241]
    same(row["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"],
         "retained L23 stage failures changed")
    same(row["S18_failure_indices"], failures, "retained L23 failure coordinates changed")
    require(len(reports) == 19 and row["L23_status"] == "FAIL"
            and row["source_operand_state_KV_RTZ_checks"] == "PASS",
            "L23 stage or lineage gate missing")
    preflight.parent.check_state(arrays)
    for stage, report in enumerate(reports):
        require(report["stage"] == stage and report["node"] == [23, 0, stage]
                and report["policy_id"] == census.POLICY
                and report["status"] == row["mandatory_statuses"][stage]
                and report["kv_lineage"] == report["residual_state_lineage"] == "PASS",
                "L23 source/state/KV/lineage gate changed")
        if stage < 18:
            gate = report["local_operator_fp16"]
            require(gate["role"] == "mandatory" and gate["passed"] is True
                    and gate["failure_count"] == 0 and gate["failures"] == [],
                    "L23 local operand gate changed")
    matrix.check_gate(reports[18], 23, failures)
    sources = {}
    nearest = np.asarray(reference, dtype="<f2").view("<u2")
    for index, metric in enumerate(reports[18]["binary64_v1"]["rows"]):
        require(metric["reference_binary64_hex"] == float(reference[index]).hex()
                and int(metric["actual_fp16_bits"], 16) == int(arrays["stage18"][index])
                and int(metric["nearest_fp16_bits"], 16) == int(nearest[index]),
                "L23 original-reference/raw-state splice")
        for pin in metric["sources"]:
            if pin["path"] in sources:
                same(sources[pin["path"]], pin, "conflicting metric source")
            sources[pin["path"]] = pin
    for pin in sources.values():
        read_bound(pin)
    return [matrix.coordinate(metric) for metric in reports[18]["binary64_v1"]["failures"]]


def bind_assets(checkpoint):
    path = Path(checkpoint["path"])
    require(path == ROOT / "build/model24_rtl_cascade/checkpoint/model.safetensors"
            and path.resolve() == path, "checkpoint path changed")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    require(digest.hexdigest() == checkpoint["sha256"]
            == "c50d807b7bed7ff314308972e0f4bcf4e5a70bc60ad88fc7df53940831ed0c1b"
            and size == checkpoint["bytes"] == 730652248, "checkpoint identity changed")
    tensors = {}
    with parent.producer.legacy.safe_open(str(path), framework="numpy") as model:
        for name, shape in (("model.norm.weight", (896,)),
                            ("model.embed_tokens.weight", (151936, 896)),
                            ("lm_head.weight", (151936, 896))):
            tensor = model.get_tensor(name)
            require(tensor.shape == shape and tensor.dtype.str == "<f2"
                    and np.all(np.isfinite(tensor)), "invalid final operator tensor: " + name)
            value_hash = hashlib.sha256(tensor.tobytes()).hexdigest()
            if name != "model.norm.weight":
                require(value_hash == FINAL_CONTRACT["head"]["tied_value_sha256"],
                        "official tied head identity changed")
            tensors[name] = {"shape": list(shape), "dtype": "float16", "sha256": value_hash}
    return {"checkpoint": checkpoint, "tensors": tensors, **bind_tokenizer()}


def bind_tokenizer():
    # This archived official copy has the same fixed-revision byte identities.
    # Do not follow the unavailable execution-vectors symlink or search fallbacks.
    tokenizer = []
    missing = []
    for name, digest in (
        ("tokenizer.json", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
        ("tokenizer_config.json", "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    ):
        pin = {"path": str(TOKENIZER_DIR / name), "sha256": digest}
        try:
            read_bound(pin)
        except FileNotFoundError:
            missing.append(pin["path"])
        tokenizer.append(pin)
    return {"tokenizer": tokenizer,
            "tokenizer_scope": "official fixed-revision byte identity only; no decode",
            "tokenizer_status": "BLOCKED_MISSING_TOKENIZER_BINDING" if missing else "BOUND",
            "missing_tokenizer": missing}


def bind_final_reference(binding, assets):
    require(binding["manifest"]["path"] == str(preflight.REFERENCE_MANIFEST),
            "original reference manifest substituted")
    manifest = json.loads(read_bound(binding["manifest"]))
    same(manifest["layers"]["23"], binding["reference"], "original L23 reference changed")
    same(manifest["checkpoint"], assets["checkpoint"], "reference checkpoint splice")
    require(manifest["scope"]["history"] == FINAL_CONTRACT["history"]
            and manifest["scope"]["position"] == FINAL_CONTRACT["position"]
            and manifest["global_reference_policy"] == census.REFERENCE_POLICY
            and manifest["reference_only"] is True,
            "original final-reference history or policy changed")
    require("final" not in manifest, "retained original reference manifest was extended")
    authority = {key: json.loads(read_bound(FINAL_REFERENCE_PINS[key]))
                 for key in ("review", "mission")}
    latest = json.loads((preflight.HANDOFFS / FINAL_REFERENCE_MISSION / "latest.json").read_bytes())
    backlog = [json.loads(line) for line in preflight.BACKLOG.read_text().splitlines() if line.strip()]
    check_review(authority["review"], latest, backlog, mission=FINAL_REFERENCE_MISSION,
                 round_number=1, pin=FINAL_REFERENCE_PINS["review"])
    mission = authority["mission"]
    require(mission["mission_id"] == FINAL_REFERENCE_MISSION
            and mission["node_key"] == "materialize-final-original-reference-binding"
            and mission["execution_workdir"] == str(ROOT), "final reference reviewed scope changed")
    terminal = {
        "mission_id": FINAL_REFERENCE_MISSION, "round": 1, "producer_role": "reviewer",
        "review_status": "done", "backlog_status": "done", "outcome_review_status": "done",
    }
    pin = FINAL_REFERENCE_PINS["manifest"]
    require(pin["path"] == str(FINAL_REFERENCE_OUTPUT / "freeze.json"),
            "reviewed final reference manifest substituted")
    try:
        reviewed = json.loads(read_bound(pin))
    except FileNotFoundError:
        arrays = []
        for key, shape, dtype in FINAL_ARRAYS:
            path = FINAL_REFERENCE_OUTPUT / (key + ".npy")
            require(path.resolve() == path, "noncanonical final reference path: " + str(path))
            arrays.append({"role": key, "path": str(path), "shape": list(shape), "dtype": dtype,
                           "status": "PRESENT_UNBOUND" if path.exists() else "MISSING"})
        return {"status": "BLOCKED_MISSING_FINAL_REFERENCE",
                "manifest": pin, "authority": FINAL_REFERENCE_PINS, "terminal_review": terminal,
                "missing": [pin["path"] + "#final"] + [
                    row["path"] for row in arrays if row["status"] == "MISSING"],
                "required_arrays": arrays,
                "required_binding": {
                    "checkpoint": assets["checkpoint"], "tensors": assets["tensors"],
                    "tokenizer": assets["tokenizer"], "history": FINAL_CONTRACT["history"],
                    "position": FINAL_CONTRACT["position"],
                    "reference_policy": census.REFERENCE_POLICY, "reference_only": True,
                },
                "reason": "The exact reviewed attempt003 final binding is missing. Unbound files, "
                          "stale default attempts and other histories cannot substitute. Extending "
                          "or replacing the retained manifest is not authorized by this check.",
                "input_binary64": binding["reference"]["binary64"],
                "input_fp16": binding["reference"]["fp16"]}
    require(reviewed["diagnostic_id"] == FINAL_REFERENCE_NAME
            and reviewed["schema_version"] == 1 and reviewed["implementation_revision"] == 2
            and reviewed["status"] == "REFERENCE_ONLY_FINAL_BOUND_PENDING_REVIEW"
            and reviewed["final_binding"] == pin["path"] + "#final",
            "reviewed final reference identity changed")
    same(reviewed["output"], {
        "path": str(FINAL_REFERENCE_OUTPUT), "fresh_creation_only": True, "ignored": True,
        "direct_child": True, "overwrite": False,
    }, "reviewed final reference output changed")
    same(reviewed["original_reference_manifest"], binding["manifest"],
         "final original reference manifest reanchored")
    retained = reviewed["retained_parent"]
    same(retained["L23_original_reference"], binding, "final L23 input lineage changed")
    same(retained["evidence"], PINS, "final reviewed L23 parents changed")
    same(retained["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9},
         "final retained failures changed")
    sources = reviewed["source_bindings"]
    require(len(sources) == 3, "final reference source bindings changed")
    for source, path in zip(sources[:2], (
            ROOT / f"ace3/model/candidates/{FINAL_REFERENCE_NAME}.py",
            ROOT / f"tests/test_{FINAL_REFERENCE_NAME}.py")):
        require(source["path"] == str(path), "final reference producer source substituted")
        read_bound(source)
    # The reviewed manifest authenticates the historical revision-2 consumer;
    # comparing that snapshot with this revision-3 source would invalidate history.
    same(sources[2], {
        "path": str(SOURCE), "bytes": 20917,
        "sha256": "b6718573434dc9c16424c239fa9b120a912d3d7639f0d3b0fc724c7cbf4eb05f",
    }, "historical final reference consumer binding changed")
    final = reviewed["final"]
    require(isinstance(final, dict), "invalid final reference binding")
    same(final["input_binary64"], binding["reference"]["binary64"], "final reference reanchored")
    same(final["input_fp16"], binding["reference"]["fp16"], "final FP16 reference reanchored")
    same(final["checkpoint"], assets["checkpoint"], "final checkpoint changed")
    same(final["tensors"], assets["tensors"], "final operand binding changed")
    same(final["tokenizer"], assets["tokenizer"], "final tokenizer binding changed")
    require(final["history"] == [9707] and final["position"] == 0
            and final["reference_policy"] == census.REFERENCE_POLICY
            and final["reference_only"] is True, "final original-input lineage changed")
    missing = []
    for key, shape, dtype in FINAL_ARRAYS:
        pin = final[key]
        require(pin["path"] == str(FINAL_REFERENCE_OUTPUT / (key + ".npy"))
                and pin["shape"] == list(shape) and pin["dtype"] == dtype,
                "final reference archive substituted")
        try:
            payload = read_bound(pin)
        except FileNotFoundError:
            missing.append(pin["path"])
            continue
        require(payload.startswith(b"\x93NUMPY"), "final reference must be an NPY array")
        stream = io.BytesIO(payload)
        array = np.load(stream, allow_pickle=False)
        require(array.shape == shape and array.dtype.str == dtype
                and stream.tell() == len(payload)
                and np.all(np.isfinite(array.view("<f2") if dtype == "<u2" else array)),
                "invalid final reference array")
        canonical = io.BytesIO()
        np.save(canonical, array, allow_pickle=False)
        require(canonical.getvalue() == payload, "noncanonical final reference NPY encoding")
    return {"status": "BLOCKED_MISSING_FINAL_REFERENCE" if missing else "BOUND_ORIGINAL_INPUT_FINAL",
            "manifest": FINAL_REFERENCE_PINS["manifest"], "reference": final, "missing": missing,
            "authority": FINAL_REFERENCE_PINS, "terminal_review": terminal,
            "implementation_revision": reviewed["implementation_revision"],
            "output": str(FINAL_REFERENCE_OUTPUT),
            "original_reference_manifest": binding["manifest"]}


def authenticate(out=OUTPUT):
    out = output_path(out)
    pinned = {key: json.loads(read_bound(pin)) for key, pin in PINS.items()}
    latest = json.loads((preflight.HANDOFFS / MISSION / "latest.json").read_bytes())
    backlog = [json.loads(line) for line in preflight.BACKLOG.read_text().splitlines() if line.strip()]
    check_review(pinned["review"], latest, backlog)
    mission = pinned["mission"]
    require(mission["mission_id"] == MISSION
            and mission["node_key"] == "execute-l23-from-l22-failed-parent-suffix"
            and mission["execution_workdir"] == str(ROOT), "L23 reviewed scope changed")
    evidence = parent.authenticate()
    result, validation = pinned["result"], pinned["validation"]
    parent.check_result(result, evidence, BASE)
    same(list(CONTROLS), list(parent.CONTROLS), "nine-control ordering changed")
    same(validation["result"], record(BASE / "result.json"), "validation/result splice")
    require(validation["status"] == "VALIDATED_EXECUTION_EVIDENCE"
            and validation["stage_reports_verified"] == 171
            and validation["source_operand_state_KV_lineage_checks"] == "PASS"
            and validation["native_layer_invocations"] == validation["forbidden_calls"] == 0,
            "reviewed L23 validation gates changed")
    same(validation["controls"], list(CONTROLS), "validation controls changed")
    same(validation["flags"], parent.FLAGS, "validation boundary changed")
    binding = evidence["summary"]["L23_original_reference"]
    same(validation["original_input_L23_reference_binding"], binding, "L23 reference lineage changed")
    for pin in list(result["origins"].values()) + list(result["preflight_pins"].values()):
        read_bound(pin)
    tests = result["tests"]
    require(tests["collected"] == tests["executed"] == parent.EXPECTED_TESTS
            and all(tests[key] == 0 for key in ("failures", "errors", "skipped",
                                               "native_layer_invocations")),
            "L23 compile/test receipt changed")
    for pin in tests["compiled"]:
        read_bound(pin)
    bound = {pin["path"]: pin for pin in result["artifacts"]}
    expected = {str(BASE / name) for name in ("command.json", "interventions.json")}
    expected.update(str(BASE / f"{label}_L23{suffix}") for label in CONTROLS
                    for suffix in (".npz", "_parent.npz", "_local_references.npz", "_gates.json"))
    require(len(result["artifacts"]) == len(bound) == len(expected)
            and set(bound) == expected, "L23 artifact set changed")
    for pin in bound.values():
        read_bound(pin)
    same(json.loads(read_bound(bound[str(BASE / "interventions.json")])),
         result["controls"], "L23 intervention rows changed")
    reference = np.load(io.BytesIO(read_bound(binding["reference"]["binary64"])), allow_pickle=False)
    states, plans = {}, []
    for row in result["controls"]:
        label = row["control"]
        pin = bound[str(BASE / f"{label}_L23.npz")]
        arrays = preflight.archive(pin)
        stored_parent = preflight.archive(bound[str(BASE / f"{label}_L23_parent.npz")])
        for key, field in preflight.STATE["fields"].items():
            previous = evidence["states"][label][field]
            require(np.array_equal(stored_parent[key], previous)
                    and stored_parent[key].dtype == previous.dtype, "L22/L23 parent-state splice")
        same(row["gates"], bound[str(BASE / f"{label}_L23_gates.json")], "L23 gate splice")
        failures = check_reports(row, json.loads(read_bound(row["gates"])), arrays, reference)
        states[label] = arrays
        plans.append({"control": label, "terminal_archive": pin, "retained_L23": row,
                      "L23_failures": failures, "final_head_status": "NOT_EXECUTED"})
    assets = bind_assets(result["checkpoint"])
    final = bind_final_reference(binding, assets)
    summary = {
        "diagnostic_id": NAME, "schema_version": 1,
        "asset_binding_revision": ASSET_BINDING_REVISION,
        "status": final["status"] if final["missing"] else (
            "BLOCKED_MISSING_TOKENIZER_BINDING" if assets["missing_tokenizer"] else "PREFLIGHT_READY"),
        "terminal_parent_review": {"mission_id": MISSION, "round": 2, "producer_role": "reviewer",
                                  "review_status": "done", "backlog_status": "done",
                                  "outcome_review_status": "done"},
        "evidence": PINS, "parent_output": str(BASE), "control_count": 9, "controls": plans,
        "final_contract": FINAL_CONTRACT, "assets": assets, "final_reference": final,
        "L23_original_reference": binding,
        "retained_L21_L22_history": evidence["summary"],
        "retained_L23_stage_reports": {"PASS": 162, "FAIL": 9},
        "thresholds": evidence["summary"]["thresholds"],
        "historical_failures_preserved": True, "original_global_reference_unchanged": True,
        "gate_evidence_scope": "AUTHENTICATED_RETAINED_ONLY; raw I/Z/H and exact S18 metrics",
        "future_output": {"selected": str(out), "fresh": True, "ignored": True,
                          "direct_child": True, "created": False, "overwrite": False},
        "final_head_status": "NOT_EXECUTED", "normal_host_review": "REQUIRED",
        "claim_boundary": BOUNDARY, **FLAGS,
    }
    return {"summary": summary, "states": states, "result": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)
    try:
        require(Path(__file__).resolve() == SOURCE and Path.cwd().resolve() == ROOT
                and sys.dont_write_bytecode and not sys.flags.optimize
                and os.getuid() == 1000, "source/cwd/bytecode/optimization/account gate failed")
        summary = authenticate(args.out)["summary"]
    except (ValueError, OSError, KeyError, TypeError, EOFError) as error:
        print(json.dumps({"diagnostic_id": NAME, "status": "CHECK_FAILED",
                          "error": str(error), **FLAGS}, sort_keys=True))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
