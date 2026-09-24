"""Continue the reviewed Q24 L2 CPU parent through L3-L8/P0, stopping on failure."""

import argparse
import ast
from pathlib import Path

import numpy as np

from ace3.model.candidates import q24_s16_toward_zero_native_v1 as candidate
from ace3.model.candidates import run_q24_s16_toward_zero_native_v1 as runner


ROOT = runner.ROOT
ACCEPTED = ROOT / "build/q24_software_root_572169d94b1d_attempt002"
SELECTED_PARENT = ACCEPTED / "layer02/software_parent.json"
REVIEW = runner.retained.HANDOFFS / "572169d94b1d/round-0001.json"
ACCEPTED_FREEZE_SHA = "786fea5ef6743659c9e6d87589da0847f1be4fc3aa35fcb95b44a70e9b4f54e6"
require = runner.require


def validate_parent_scope(parent):
    require(parent["candidate_id"] == "ace3-q24-s16-toward-zero-native-v1"
            and parent["state_id"] == "ace3-q24-software-paired-state-v1"
            and parent["policy_id"] == runner.gates.POLICY_ID
            and parent["model"] == "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
            and parent["history"] == [9707]
            and type(parent["position"]) is int and parent["position"] == 0
            and type(parent["next_layer"]) is int and parent["next_layer"] == 3
            and parent["evidence_kind"] == "cpu_software_q24"
            and parent["rtl_admissible"] is False
            and parent["normal_host_review"] == "REQUIRED",
            "wrong reviewed Q24 continuation scope")


def verify_native_compatibility(historical_source, current_source):
    old, new = ast.parse(historical_source), ast.parse(current_source)
    old_stages = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "stages")
    new_stages = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "_stages")
    require(ast.dump(ast.Module(body=old_stages.body[1:], type_ignores=[]))
            == ast.dump(ast.Module(body=new_stages.body, type_ignores=[])),
            "accepted native stage arithmetic changed")
    old.body = [n for n in old.body if n is not old_stages]
    new.body = [n for n in new.body if not isinstance(n, ast.FunctionDef)
                or n.name not in {"stages", "continuation_stages", "_stages"}]
    require(ast.dump(old) == ast.dump(new), "accepted native helper arithmetic changed")


def authenticate_resume(path, bind, read, bind_tree):
    require(path == SELECTED_PARENT, "only the Planner-selected accepted L2 parent is supported")
    parent_record = runner.retained.record(path)
    parent = read(parent_record)
    validate_parent_scope(parent)
    review_record = runner.retained.record(REVIEW)
    review = read(review_record)
    require(review["kind"] == "round_reviewed_handoff"
            and review["producer_role"] == "reviewer"
            and review["mission_id"] == "572169d94b1d"
            and review["review"]["status"] == "done", "missing independent accepted-root review")
    require(parent["arithmetic_lineage"]["path"] == str(ACCEPTED / "freeze.json")
            and parent["arithmetic_lineage"]["sha256"] == ACCEPTED_FREEZE_SHA,
            "wrong accepted arithmetic freeze")
    freeze = read(parent["arithmetic_lineage"])
    require(freeze["contract"]["scope"] == {"layers": [0, 1, 2], "position": 0, "history": [9707]},
            "wrong accepted root scope")
    extended_sources = {str(Path(candidate.__file__).resolve()), str(Path(runner.__file__).resolve())}
    for record in freeze["input_bindings"]:
        if record["path"] not in extended_sources:
            bind(record)
    native_sources = []
    for source in freeze["sources"]:
        snapshot = bind(source["snapshot"])
        require(source["snapshot"]["sha256"] == source["original"]["sha256"],
                "accepted source snapshot mismatch")
        if source["original"]["path"] == str(Path(candidate.__file__).resolve()):
            native_sources.append(snapshot)
    require(len(native_sources) == 1, "missing accepted native arithmetic source")
    verify_native_compatibility(native_sources[0].read_text(), Path(candidate.__file__).read_text())
    state_sources = [s for s in freeze["sources"]
                     if s["original"]["path"] == str(Path(candidate.state.__file__).resolve())]
    require(len(state_sources) == 1, "missing accepted Q24 state implementation")
    bind(state_sources[0]["original"])

    result_record = runner.retained.record(ACCEPTED / "result.json")
    result = read(result_record)
    require(result["status"] == "PASS" and result["candidate_admitted"] is True
            and result["candidate_id"] == parent["candidate_id"]
            and result["policy_id"] == parent["policy_id"]
            and result["normal_host_review"] == "REQUIRED"
            and result["rtl_invocations"] == result["l9_invocations"] == 0
            and result["first_failure"] is None
            and [entry["layer"] for entry in result["layers"]] == [0, 1, 2],
            "accepted root result is not a complete CPU L0-L2 PASS")
    bind_tree(result["layers"])
    bind_tree(parent)
    previous = None
    last_state = None
    for layer, entry in enumerate(result["layers"]):
        reports = read(entry["reports"])
        require(entry["status"] == "PASS" and entry["position"] == 0
                and len(reports) == 19
                and [report["stage"] for report in reports] == list(range(19))
                and all(report["status"] == "PASS" and report["node"] == [layer, 0, stage]
                        for stage, report in enumerate(reports)),
                "accepted prefix has a missing or failing mandatory stage")
        if previous is not None:
            require(entry["input_parent"] == previous, "accepted residual chain is spliced")
        with np.load(bind(entry["input_parent"]), allow_pickle=False) as archive:
            incoming = {key: archive[key].copy() for key in archive.files}
        with np.load(bind(entry["actual_stages"]), allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        runner.retained.verify_parent(runner.retained.state_from(arrays, "input", "input_hidden"), incoming)
        with np.load(bind(entry["output_parent"]), allow_pickle=False) as archive:
            last_state = {key: archive[key].copy() for key in archive.files}
        runner.retained.verify_parent(last_state, runner.retained.state_from(arrays, "output", "stage18"))
        previous = entry["output_parent"]
    final = result["layers"][-1]
    require(parent["state"] == final["output_parent"]
            and parent["state_lineage_parent"] == final["input_parent"]
            and parent["numerical_report"] == final["reports"]
            and parent["kv"] == final["kv_state"], "L2 receipt differs from reviewed output")
    with np.load(bind(parent["state"]), allow_pickle=False) as archive:
        runner.retained.verify_parent({key: archive[key].copy() for key in archive.files}, last_state)
    with np.load(bind(parent["kv"]), allow_pickle=False) as cache:
        require(set(cache.files) == {"k", "v"}, "accepted L2 cache keys changed")
        for name in ("k", "v"):
            require(cache[name].dtype == np.dtype("<u2") and cache[name].shape == (1, 128)
                    and np.array_equal(cache[name], arrays["output_cache_" + name]),
                    "accepted L2 cache differs from its own output")
    bind(runner.retained.record(ACCEPTED / "command.log"))
    return {"receipt": parent_record, "parent": parent, "review": review_record,
            "result": result_record, "native_arithmetic_compatibility": "unchanged",
            "prior_layer_kv_consumed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-parent", type=Path, required=True)
    args = parser.parse_args()
    return runner.execute(args.out.resolve(), resume_parent=args.resume_parent.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
