"""Isolated ordered CPU experiment; no RTL dispatch or production ABI adoption."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import decoder_gate_policy_v3 as gates
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import q24_software_candidate_v1 as candidate
from ace3.model.candidates import residual_exact_grid_q24_reference_v1 as rational


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "ace3/contracts/candidates/q24_software_root_v3.json"
HANDOFFS = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")
B_FREEZE = ROOT / "build/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010/frozen.json"
CONTROL = ROOT / "build/fp16_reference_binary64_control_attempt002/freeze.json"
CONTROL_SHA = "96430bae5dca0dc8a21484384bc902a9efb6159a68336f7912106af0b2732d08"
STATE_ID = "ace3-q24-software-paired-state-v1"
EXPERIMENT_ID = "ace3-q24-software-root-p0-v1"
require = local.require


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def write(path, document):
    with Path(path).open("x", encoding="ascii") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
    return record(path)


def read_words(path, count, indexed=False):
    rows = Path(path).read_text("ascii").splitlines()
    width = 10 if indexed else 4
    require(len(rows) == count and all(re.fullmatch(f"[0-9a-fA-F]{{{width}}}", r)
                                      for r in rows), "invalid FP16 text layout")
    if indexed:
        require([int(r[:-4], 16) for r in rows] == list(range(count)),
                "embedding token/coordinate order")
    words = np.asarray([int(r[-4:], 16) for r in rows], dtype="<u2")
    local.finite_words(words, (count,))
    return words


def verify_parent(state, expected, *, embedding=None):
    require(set(state) == set(expected) == {"i", "z", "h"},
            "missing paired Q24 parent; FP16-only state forbidden")
    for key, dtype in (("i", "<i8"), ("z", "u1"), ("h", "<u2")):
        require(state[key].dtype == np.dtype(dtype) and state[key].shape == (896,),
                f"invalid Q24 parent {key} shape/dtype")
        require(np.array_equal(state[key], expected[key]),
                f"invented/spliced Q24 parent {key}")
    for index, (integer, tag, word) in enumerate(zip(
            state["i"], state["z"], state["h"], strict=True)):
        require(rational.project(int(integer), int(tag)) == int(word),
                f"Q24 parent/view mismatch at {index}")
        if embedding is not None:
            require(rational.root(int(embedding[index])) == (int(integer), int(tag))
                    and int(word) == int(embedding[index]),
                    f"invented root state at {index}")


def transition_reference(state, words):
    rows = [rational.add(int(i), int(z), int(w)) for i, z, w in
            zip(state["i"], state["z"], words, strict=True)]
    return {"i": np.asarray([r[0] for r in rows], dtype="<i8"),
            "z": np.asarray([r[1] for r in rows], dtype="u1"),
            "h": np.asarray([r[2] for r in rows], dtype="<u2")}


def state_from(arrays, prefix, hidden):
    return {"i": arrays[prefix + "_i"], "z": arrays[prefix + "_z"], "h": arrays[hidden]}


def check_stage_state(stage, arrays, parent):
    local.finite_words(arrays[f"stage{stage:02d}"], (local.SIZES[stage],))
    require(np.array_equal(arrays["input_hidden"], parent["h"])
            and np.array_equal(arrays["input_i"], parent["i"])
            and np.array_equal(arrays["input_z"], parent["z"]),
            "producer-to-consumer Q24 identity mismatch")
    for kind in ("k", "v"):
        local.finite_words(arrays["input_cache_" + kind], (0, 128))
    if stage in (6, 7):
        kind, source = ("k", 5) if stage == 6 else ("v", 3)
        cache = arrays["output_cache_" + kind]
        local.finite_words(cache, (1, 128))
        require(np.array_equal(cache[0], arrays[f"stage{source:02d}"])
                and np.array_equal(cache[0], arrays[f"stage{stage:02d}"]),
                "own-layer P0 KV write/read identity mismatch")
    if stage == 12:
        expected = transition_reference(parent, arrays["stage11"])
        verify_parent(state_from(arrays, "scratch", "stage12"), expected)
        return expected["h"]
    if stage == 18:
        scratch = transition_reference(parent, arrays["stage11"])
        verify_parent(state_from(arrays, "scratch", "stage12"), scratch)
        expected = transition_reference(scratch, arrays["stage17"])
        verify_parent(state_from(arrays, "output", "stage18"), expected)
    return None


def run(out, selected):
    require(selected == gates.POLICY_ID, "explicit v3 opt-in required")
    require(out.parent == ROOT / "build" and
            re.fullmatch(r"q24_software_root_[a-z0-9]+_attempt[0-9]+", out.name),
            "output must be a fresh ignored software-only build attempt")
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    bound, layers, timings = {}, [], []
    phase, node, first_failure = "authority", None, None
    status, parent_receipt = "BLOCKED", None

    def bind(rec):
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        path = signature["path"]
        if path not in bound:
            require(record(path) == signature, f"source/tensor/reference binding drift: {path}")
            bound[path] = signature
        require(bound[path] == signature, f"conflicting input identities: {path}")
        return Path(path)

    def read(rec):
        return json.loads(bind(rec).read_text())

    try:
        contract = read(record(CONTRACT))
        require(contract["policy_id"] == selected
                and contract["experiment_id"] == EXPERIMENT_ID
                and contract["state_id"] == STATE_ID
                and contract["authority"]["directive"] == "resume_argus_software_only"
                and contract["evidence_kind"] == "cpu_software_q24"
                and contract["rtl_admissible"] is False, "software-only policy mismatch")
        mission = read(record(HANDOFFS / "995d0ba22311/mission.json"))
        require("software" in json.dumps(mission).lower(), "software mission authority missing")
        reviews = []
        for task, round_number in (("5df93323345b", 1), ("229c5f00e558", 2),
                                   ("0106645d010b", 2)):
            rec = record(HANDOFFS / task / f"round-{round_number:04d}.json")
            review = read(rec)
            require(review["kind"] == "round_reviewed_handoff"
                    and review["producer_role"] == "reviewer"
                    and review["mission_id"] == task
                    and review["review"]["status"] == "done",
                    f"missing genuine prerequisite review: {task}")
            reviews.append(rec)
        phase = "reference_authentication"
        v3 = read(record(gates.CONTRACT))
        b_record = record(B_FREEZE)
        require(b_record["sha256"] == v3["trusted_independent_freeze_sha256"],
                "wrong original model/reference freeze")
        b = read(b_record)
        control_record = record(CONTROL)
        require(control_record["sha256"] == CONTROL_SHA, "wrong reviewed original-input control")
        control = read(control_record)
        require(b["checkpoint"] == control["checkpoint"]
                and b["embeddings"] == control["embeddings"]
                and b["reference_source"] == control["fp16_reference_root"],
                "global/local original input lineage mismatch")
        specification = read(b["reference_source"])
        require(specification["checkpoint"] == b["checkpoint"]
                and specification["checkpoint_tensors"] == b["checkpoint_tensors"],
                "canonical tensors disagree with independent specification")
        read(control["binary64_reference_freeze"])
        read(control["binary64_array_binding_freeze"])
        global_refs, trajectories = {}, {}
        with bind(control["binary64_csv"]).open(newline="") as stream:
            rows = [r for r in csv.DictReader(stream)
                    if int(r["position"]) == 0 and 0 <= int(r["layer"]) <= 8]
        original = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"] for r in rows}
        require(len(original) == len(rows) == 9 * 896, "original global reference coverage")
        for key, case in zip(control["ordered_cases"], control["cases"], strict=True):
            layer, position = key
            if position != 0:
                continue
            values = np.load(bind(case["binary64_reference_array"]), allow_pickle=False)
            require(values.dtype == np.dtype("<f8") and values.shape == (896,)
                    and np.all(np.isfinite(values)) and np.all(np.abs(values) <= 65504)
                    and all(float(v).hex() == original[layer, i] for i, v in enumerate(values)),
                    f"invalid original global reference L{layer}")
            global_refs[layer] = values
        for transaction in b["reference_transactions"]:
            layer = transaction["layer"]
            if transaction["position"] != 0 or not 0 <= layer <= 8:
                continue
            require(layer not in trajectories and set(transaction["stages"]) ==
                    {str(s) for s in range(19)}, "ambiguous original FP16 trajectory")
            trajectories[layer] = {}
            for stage, rec in transaction["stages"].items():
                words = read_words(bind(rec), local.SIZES[int(stage)])
                require(hashlib.sha256(words.tobytes()).hexdigest() == rec["semantic_sha256"],
                        "FP16 trajectory semantic identity mismatch")
                trajectories[layer][int(stage)] = words
        require(set(global_refs) == set(trajectories) == set(range(9)),
                "missing original L0-L8 references")
        phase = "tensor_authentication"
        checkpoint = bind(b["checkpoint"])
        embedding_record = b["embeddings"][0]
        require(embedding_record["token"] == 9707, "wrong model-root token")
        embedding = read_words(bind(embedding_record["input"]), 896, indexed=True)
        require(hashlib.sha256(embedding.tobytes()).hexdigest() ==
                embedding_record["semantic_sha256"], "embedding semantic identity mismatch")
        canonical = {r["name"]: r for r in b["checkpoint_tensors"]}
        require(len(canonical) == len(b["checkpoint_tensors"]), "duplicate canonical tensor identity")
        tensors_by_layer = {}
        with safe_open(str(checkpoint), framework="numpy") as model:
            official_embedding = model.get_slice("model.embed_tokens.weight")[9707:9708]
            require(official_embedding.dtype == np.dtype("<f2")
                    and official_embedding.shape == (1, 896)
                    and np.array_equal(official_embedding.reshape(-1).view("<u2"), embedding),
                    "root differs from official checkpoint embedding")
            for layer in range(9):
                tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
                local.authenticate_tensors(tensors, canonical, layer)
                tensors_by_layer[layer] = tensors
        phase = "source_freeze"
        prior = read(record(ROOT / "build/single_round_residual_binary64_excess_55c2b0a6b67a_attempt001/freeze.json"))
        for module in (candidate.fixed, candidate.awq):
            paths = [r for r in prior["bindings"] if r["path"] == str(Path(module.__file__).resolve())]
            require(len(paths) == 1, "missing retained non-residual arithmetic source")
            bind(paths[0])
        source_paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
                        if getattr(m, "__file__", None)
                        and str(Path(m.__file__).resolve()).startswith(str(ROOT / "ace3/model/"))
                        and str(m.__file__).endswith(".py")}
        source_paths.update((CONTRACT, gates.CONTRACT,
                            ROOT / "ace3/contracts/candidates/binary64_fp16_excess_v1.json",
                            ROOT / "ace3/contracts/candidates/residual_exact_grid_q24_semantics_51efa8e34614_attempt001.md",
                            ROOT / "ace3/model/tests/test_q24_software_root_v3.py"))
        source_dir = out / "source"
        source_dir.mkdir()
        sources = []
        for index, path in enumerate(sorted(source_paths)):
            rec = record(path)
            copy = source_dir / f"{index:02d}_{path.name}"
            shutil.copyfile(path, copy)
            copied = record(copy)
            require(copied["sha256"] == rec["sha256"], "source snapshot changed")
            sources.append({"original": rec, "snapshot": copied})
        write(out / "freeze.json", {
            "attempt": out.name, "contract": contract, "sources": sources,
            "input_bindings": list(bound.values()), "prerequisite_reviews": reviews,
            "tools": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
            "arithmetic_lineage": {"candidate": record(candidate.__file__),
                                   "non_residual_helpers": [record(m.__file__) for m in
                                                            (candidate.fixed, candidate.awq)]},
            "state_lineage": {"root": embedding_record, "prior_residual": None,
                              "prior_kv": "own empty P0 per layer; no opaque state import"},
            "global_reference_lineage": {"control": control_record, "model_freeze": b_record,
                                         "candidate_data_used": False},
            "rtl_backlog_disposition": "preserved and excluded; no scheduler or RTL operation",
            "scope": contract["scope"], "normal_host_review": "REQUIRED",
            "rtl_invocations": 0, "public_rtl_contract": "not applicable; software only"})
        timings.append({"phase": "authentication_and_freeze", "seconds": time.monotonic() - started})
        phase = "root_state"
        parent = candidate.lift(embedding)
        verify_parent(parent, parent, embedding=embedding)
        root_state = save(out / "root_state.npz", parent)
        parent_receipt = {"evidence_kind": "cpu_software_q24", "rtl_admissible": False,
                          "state_id": STATE_ID, "policy_id": selected,
                          "model": contract["model"], "history": [9707], "position": 0,
                          "next_layer": 0, "root": embedding_record["semantic_sha256"],
                          "state": root_state, "producer": "official_embedding_root",
                          "arithmetic_lineage": record(out / "freeze.json")}
        write(out / "root_parent.json", parent_receipt)
        for layer in range(9):
            begin = time.monotonic()
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            phase, node = "state", [layer, 0, None]
            require(parent_receipt["next_layer"] == layer
                    and parent_receipt["state_id"] == STATE_ID
                    and parent_receipt["policy_id"] == selected
                    and parent_receipt["evidence_kind"] == "cpu_software_q24"
                    and parent_receipt["rtl_admissible"] is False
                    and parent_receipt["history"] == [9707]
                    and parent_receipt["position"] == 0, "incompatible software parent owner")
            with np.load(bind(parent_receipt["state"]), allow_pickle=False) as data:
                loaded = {k: data[k].copy() for k in data.files}
            verify_parent(loaded, parent, embedding=embedding if layer == 0 else None)
            parent = loaded
            arrays, reports, local_refs = {}, [], {}
            compute_seconds, oracle_seconds = 0.0, 0.0
            producer = candidate.stages(tensors_by_layer[layer], layer, parent, arrays)
            try:
                for stage in range(19):
                    phase, node = "candidate_arithmetic", [layer, 0, stage]
                    stamp = time.monotonic()
                    require(next(producer) == stage, "out-of-order stage execution")
                    compute_seconds += time.monotonic() - stamp
                    stamp = time.monotonic()
                    phase = "state"
                    expected = check_stage_state(stage, arrays, parent)
                    phase = "local_reference"
                    if stage < 18 and stage != 12:
                        operands = {key: arrays[key] for key in local.OPERANDS[stage]}
                        expected = local.local_reference(stage, operands, tensors_by_layer[layer], layer)
                    if stage < 18:
                        local_refs[f"stage{stage:02d}"] = expected
                    phase = "global_numerical" if stage == 18 else "local_numerical"
                    report = gates.evaluate_decoder_stage(
                        stage=stage, actual=arrays[f"stage{stage:02d}"],
                        reference=trajectories[layer][stage], policy=selected,
                        local_reference=expected,
                        reference_binary64=global_refs[layer] if stage == 18 else None)
                    report.update(node=node, residual_state_lineage="PASS", kv_lineage="PASS",
                                  operand_names=["input_i", "input_z", "stage11"] if stage == 12
                                  else list(local.OPERANDS[stage]) if stage < 18 else None)
                    reports.append(report)
                    oracle_seconds += time.monotonic() - stamp
                    if report["status"] != "PASS":
                        first_failure = {"node": node, "failure_taxonomy": phase,
                                         "report": str(directory / "reports.json"),
                                         "root_cause_hypothesis": "candidate trajectory misses unchanged mandatory gate",
                                         "regression": "same frozen actual operands and mandatory reference in layer archive"}
                        break
            finally:
                producer.close()
                archive = save(directory / "actual_stages.npz", arrays)
                save(directory / "local_references.npz", local_refs)
                write(directory / "reports.json", reports)
            layer_result = {"layer": layer, "position": 0, "input_parent": parent_receipt,
                            "actual_stages": archive, "reports": record(directory / "reports.json"),
                            "status": "FAIL" if first_failure else "PASS",
                            "candidate_seconds": compute_seconds, "oracle_seconds": oracle_seconds,
                            "total_seconds": time.monotonic() - begin}
            layers.append(layer_result)
            write(directory / "result.json", layer_result)
            print(f"L{layer}/P0 {layer_result['status']} stages={len(reports)} "
                  f"cpu={compute_seconds:.3f}s oracle={oracle_seconds:.3f}s", flush=True)
            if first_failure:
                status = "FAIL"
                break
            parent = state_from(arrays, "output", "stage18")
            state_record = save(directory / "parent_state.npz", parent)
            kv_record = save(directory / "kv_state.npz", {
                "k": arrays["output_cache_k"], "v": arrays["output_cache_v"]})
            parent_receipt = dict(parent_receipt, next_layer=layer + 1, state=state_record,
                                  producer=archive, kv=kv_record,
                                  state_lineage_parent=layer_result["input_parent"]["state"],
                                  residual_state_lineage="PASS", kv_lineage="PASS",
                                  numerical_report=record(directory / "reports.json"))
            write(directory / "software_parent.json", parent_receipt)
        else:
            status = "PASS"
    except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError, StopIteration) as exc:
        first_failure = {"node": node, "failure_taxonomy": phase,
                         "reason": f"{type(exc).__name__}: {exc}",
                         "root_cause_hypothesis": f"first incompatible {phase} dependency",
                         "regression": "reproduce the captured command with the frozen dependency"}
        print(json.dumps(first_failure), file=sys.stderr, flush=True)
    result = {"status": status, "evidence_kind": "cpu_software_q24",
              "policy_id": selected, "experiment_id": EXPERIMENT_ID,
              "rtl_admissible": False, "rtl_invocations": 0,
              "first_failure": first_failure, "layers": layers, "phase_times": timings,
              "seconds": time.monotonic() - started, "normal_host_review": "REQUIRED",
              "software_l8_parent": record(out / "layer08/software_parent.json")
              if status == "PASS" else None,
              "claim_boundary": "P0 fixed token9707 CPU Q24 residual only; no RTL, tokens, full model or hardware"}
    write(out / "result.json", result)
    print(f"Q24_SOFTWARE_ROOT_{status} layers={len(layers)} result={out / 'result.json'}", flush=True)
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    return run(args.out.resolve(), args.policy)


if __name__ == "__main__":
    raise SystemExit(main())
