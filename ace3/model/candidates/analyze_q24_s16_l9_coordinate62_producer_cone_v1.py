"""Read-only classification of the reviewed L9 coordinate-62 producer cone."""

import argparse
from fractions import Fraction
import hashlib
from importlib.machinery import PathFinder
import json
from pathlib import Path
import sys


ROOT = Path("/home/argustest/ace3-argus")
PREFIX = "q24_s16_l9_coordinate62_producer_cone_"
REVIEWED = ROOT / "build" / (PREFIX + "a38731cbbd8f_attempt001")
RESULT_SHA = "c36ed0cb236b06b2fd72e2f73502f38f022e55c7fb67bf0f0b2fc705915539d4"
DIAGNOSTIC = "ace3-q24-s16-l9-coordinate62-entry-producer-cone-v1"
POLICY = "ace3-w4a16-local-operator-global-binary64-authority-v3"
PROFILE = "ace3-w4a16-layer-final-binary64-fp16-excess-v1"
REFERENCE = "legacy-binary64-AWQ-fully-independent-propagation"
MODULE = "ace3.model.candidates.analyze_q24_s16_l9_coordinate62_producer_cone_v1"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_producer_cone_analyzer_v1"
SOURCE_MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1"
SOURCE_TEST = "tests.test_q24_s16_l9_coordinate62_entry_producer_cone_v1"
CONTROLS = (
    "actual", "frozen_inherited", "frozen_o", "frozen_down",
    "frozen_inherited_o", "frozen_inherited_down", "frozen_o_down",
    "frozen_inherited_o_down", "inherited_native", "mapped62", "mapped_all",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def repository_path(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT),
            f"non-repository source/input: {value}")
    return path.resolve()


def module_origin(name):
    """Resolve packages without importing diagnostic or native implementations."""
    paths = [str(ROOT)]
    for index, part in enumerate(name.split(".")):
        qualified = ".".join(name.split(".")[:index + 1])
        spec = PathFinder.find_spec(qualified, paths)
        require(spec is not None, f"missing repository module: {qualified}")
        expected = ROOT.joinpath(*qualified.split("."))
        if spec.origin is None:
            locations = list(spec.submodule_search_locations or ())
            require(locations and all(Path(p).resolve() == expected for p in locations),
                    f"namespace origin mismatch: {qualified}")
        else:
            origin = repository_path(spec.origin)
            require(origin in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {qualified}")
        paths = list(spec.submodule_search_locations or ())
    if spec.origin is None:
        return {"namespace_paths": paths}
    loaded = sys.modules.get(name)
    if loaded is not None:
        require(Path(loaded.__file__).resolve() == origin, f"loaded origin mismatch: {name}")
    return {"path": str(origin)}


class Bindings:
    def __init__(self):
        self.verified = {}

    def bind(self, item):
        path = repository_path(item["path"])
        require(str(path) == item["path"], f"noncanonical binding: {item['path']}")
        if path in self.verified:
            require(self.verified[path] == item, f"conflicting binding: {path}")
            return path
        require(path.stat().st_size == item["bytes"], f"size mismatch: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(digest == item["sha256"], f"hash mismatch: {path}")
        self.verified[path] = item
        return path


def check_root(root):
    resolved = root.resolve()
    require(resolved.parent == ROOT / "build" and resolved.name.startswith(PREFIX),
            "result root must use the corrected producer_cone prefix directly under build")
    require(root.absolute() == resolved, "result root must not be a symlink")
    require(resolved.is_dir(), "missing saved result root")
    return resolved


def check_metadata(result):
    expected = {
        "diagnostic_id": DIAGNOSTIC, "status": "DIAGNOSED", "node": [9, 0, 18],
        "index": 62, "candidate_admitted": False, "policy_adopted": False,
        "successor_published": False, "rtl_invocations": 0,
        "accepted_L0_L8_execution": False, "normal_host_review": "REQUIRED",
        "retained_status": "FAIL", "control_count": 11,
        "native_retained_bitwise_reproduction": True,
        "reviewed_cut9_mapped_suffix_bitwise_reproduction": True,
        "original_L9_S18_bitwise_reproduction": True, "unchanged_baseline_gates": True,
        "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "unique_upstream_producer_attributed": False,
    }
    for key, value in expected.items():
        require(type(result[key]) is type(value) and result[key] == value,
                f"reviewed metadata mismatch: {key}")


def check_summary(summary, count_key):
    count = summary[count_key]
    failures = summary["failures"]
    require(type(count) is int and count > 0 and
            summary["failure_count"] == len(failures) <= count and
            summary["passed"] is (len(failures) == 0), "inconsistent gate summary")


def check_reports(reports, layer):
    require(layer in range(9, 14) and len(reports) == 19, "gate layer/stage scope mismatch")
    for stage, gate in enumerate(reports):
        require(gate["node"] == [layer, 0, stage] and gate["stage"] == stage,
                "reordered or out-of-scope saved gate")
        require(gate["policy_id"] == POLICY and gate["kv_lineage"] == "PASS" and
                gate["residual_state_lineage"] == "PASS" and
                gate["fp16_role"] == "independent-whole-FP16-trajectory-diagnostic",
                "gate policy/state/KV mismatch")
        check_summary(gate["fp16"], "comparisons")
        mandatory = gate["binary64_v1"] if stage == 18 else gate["local_operator_fp16"]
        require(mandatory["role"] == "mandatory", "mandatory gate role changed")
        check_summary(mandatory, "coordinates" if stage == 18 else "comparisons")
        require(gate["status"] == ("PASS" if mandatory["passed"] else "FAIL"),
                "mandatory gate status mismatch")
        if stage != 18:
            continue
        rows = mandatory["rows"]
        require(mandatory["profile_id"] == PROFILE and len(rows) == 896 and
                mandatory["coordinates"] == 896, "global reference shape/profile mismatch")
        failures = []
        for index, row in enumerate(rows):
            require(row["index"] == index and row["profile_id"] == PROFILE and
                    row["reference_policy"] == REFERENCE and
                    row["excess_budget"] == "1/8", "global reference/threshold changed")
            error, floor = Fraction(row["actual_error"]), Fraction(row["q"])
            excess = Fraction(row["excess_error"])
            require(error >= 0 and floor >= 0 and excess == max(Fraction(0), error - floor),
                    "inconsistent saved excess")
            require(row["accepted"] is (excess <= Fraction(1, 8)),
                    "saved excess acceptance mismatch")
            if not row["accepted"]:
                failures.append(row)
        require(failures == mandatory["failures"], "global failure records changed")
    return reports


def first_divergences(actual):
    diagnostic = None
    mandatory = None
    for layer in range(9, 14):
        for gate in actual[layer]:
            if diagnostic is None:
                matches = [f for f in gate["fp16"]["failures"] if f["index"] == 62]
                if matches:
                    diagnostic = {"node": gate["node"], "index": 62,
                                  "role": "diagnostic_only", "saved_failure": matches[0]}
            component = gate["binary64_v1"] if gate["stage"] == 18 else gate["local_operator_fp16"]
            if mandatory is None:
                matches = [f for f in component["failures"] if f["index"] == 62]
                if matches:
                    mandatory = {"node": gate["node"], "index": 62,
                                 "role": "mandatory", "saved_failure": matches[0]}
    return {"first_material_divergence": diagnostic, "first_mandatory_failure": mandatory}


def rank_producers(decomposition, controls, actual):
    parts = decomposition["signed_parts"]
    require(sum((Fraction(v) for v in parts.values()), Fraction(0)) ==
            Fraction(decomposition["signed_error"]), "signed decomposition does not close")
    definitions = (
        ("inherited", "incoming_Q24_minus_original", [8, 0, 18], "frozen_input_only"),
        ("o", "S11_minus_original", [9, 0, 11], "saved_L9_gate"),
        ("down", "S17_minus_original", [9, 0, 17], "saved_L9_gate"),
    )
    rows = []
    for branch, key, node, kind in definitions:
        value = Fraction(parts[key])
        cut = controls["frozen_" + branch]
        rows.append({
            "producer": branch, "node": node, "evidence_kind": kind,
            "signed_Q24_error_contribution": str(value),
            "absolute_Q24_error_contribution": str(abs(value)),
            "local_gate_status": None if branch == "inherited" else actual[9][node[2]]["status"],
            "frozen_singleton_coordinate62_rescue": cut["index62"]["accepted"],
            "frozen_singleton_suffix_all_gates_pass": cut["all_L10_L13_gates_pass"],
        })
    rows.sort(key=lambda row: (-Fraction(row["absolute_Q24_error_contribution"]), row["producer"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def analyze(root):
    root = check_root(root)
    raw = (root / "result.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RESULT_SHA, "not the pinned reviewed result")
    result = json.loads(raw)
    check_metadata(result)
    bindings = Bindings()
    validation = read_json(bindings.bind(result["validation"]))
    require(validation["cwd"] == str(ROOT) and validation["PYTHONPATH"] == str(ROOT) and
            validation["executable"] == "/home/argustest/miniconda3/bin/python" and
            validation["collected"] == validation["executed"] == 18 and
            validation["failures"] == validation["errors"] == validation["skipped"] == 0 and
            validation["accepted_L0_L8_tests_executed"] is False,
            "saved validation metadata mismatch")
    origins = result["origins_after_execution"]
    require(origins == validation["origins"] and SOURCE_MODULE in origins and SOURCE_TEST in origins,
            "saved source origins mismatch")
    for name, item in origins.items():
        live = module_origin(name)
        if "path" in item:
            require(live["path"] == item["path"], f"saved module origin mismatch: {name}")
            bindings.bind(item)
        else:
            require(live == item, f"saved namespace origin mismatch: {name}")
    live_origins = {name: module_origin(name) for name in (MODULE, TEST_MODULE)}
    contract = read_json(bindings.bind(validation["contract"]))
    require(contract["diagnostic_id"] == DIAGNOSTIC and contract["policy_id"] == POLICY and
            contract["controls"] == list(CONTROLS) and
            contract["claim_boundary"] == result["claim_boundary"], "contract metadata mismatch")
    for item in result["input_bindings"] + validation["compiled"]:
        bindings.bind(item)
    artifacts = {}
    for item in result["artifacts"]:
        path = bindings.bind(item)
        require(path.parent == root and path.name not in artifacts, "artifact origin/duplicate mismatch")
        artifacts[path.name] = path
    require(bindings.bind(result["validation"]).parent == root, "foreign saved validation")
    expected_gates = {
        f"{label}_L{layer}_gates.json"
        for label in CONTROLS for layer in range(10, 14)
    } | {"actual_L9_gates.json", "inherited_native_L9_gates.json"}
    require({n for n in artifacts if n.endswith("_gates.json")} == expected_gates and
            {p.name for p in root.glob("*_gates.json")} == expected_gates,
            "unexpected or missing saved gate files")
    gates = {}
    for label in CONTROLS:
        gates[label] = {}
        for layer in range(9 if label in ("actual", "inherited_native") else 10, 14):
            gates[label][layer] = check_reports(
                read_json(artifacts[f"{label}_L{layer}_gates.json"]), layer)
    original_references = {
        layer: [(r["reference_binary64_hex"], r["q"], r["sources"])
                for r in gates["actual"][layer][18]["binary64_v1"]["rows"]]
        for layer in range(9, 14)
    }
    for layers in gates.values():
        for layer, reports in layers.items():
            rows = reports[18]["binary64_v1"]["rows"]
            require([(r["reference_binary64_hex"], r["q"], r["sources"]) for r in rows] ==
                    original_references[layer], "counterfactual reference substitution")
            for source in rows[0]["sources"]:
                path = repository_path(str(ROOT / source["path"]))
                require(path in bindings.verified and
                        bindings.verified[path]["sha256"] == source["sha256"],
                        "unbound numerical policy source")
    interventions = read_json(artifacts["interventions.json"])
    require([r["label"] for r in interventions] == list(CONTROLS), "control order/coverage changed")
    controls = {r["label"]: r for r in interventions}
    for label, row in controls.items():
        require(row["candidate_admitted"] is False and
                [s["layer"] for s in row["suffix"]] == list(range(10, 14)),
                "control admission/suffix scope changed")
        require(bindings.bind(row["parent"]) == artifacts[label + "_L9_parent.npz"],
                "control parent binding mismatch")
        statuses9 = [g["status"] for g in gates[label][9]] if 9 in gates[label] else None
        require(row["L9_operator_gate_statuses"] == statuses9, "fabricated L9 gates")
        for saved in row["suffix"]:
            layer = saved["layer"]
            reports = gates[label][layer]
            require(bindings.bind(saved["gates"]) == artifacts[f"{label}_L{layer}_gates.json"] and
                    bindings.bind(saved["vectors"]) == artifacts[f"{label}_L{layer}.npz"] and
                    saved["source_operand_state_KV_RTZ_checks"] == "PASS" and
                    saved["mandatory_statuses"] == [g["status"] for g in reports] and
                    saved["all_gates_pass"] is all(g["status"] == "PASS" for g in reports) and
                    saved["S18_failure_indices"] ==
                    [r["index"] for r in reports[18]["binary64_v1"]["failures"]],
                    "saved suffix source/state/KV/gate linkage mismatch")
            scalar = reports[18]["binary64_v1"]["rows"][62]
            require(all(saved["index62"][key] == scalar[key] for key in
                        ("accepted", "actual_fp16_bits", "reference_binary64_hex", "excess_error")),
                    "coordinate62 scalar/gate mismatch")
        require(row["index62"] == row["suffix"][-1]["index62"] and
                row["all_L10_L13_gates_pass"] is all(s["all_gates_pass"] for s in row["suffix"]),
                "control endpoint mismatch")
    decomposition = read_json(artifacts["L9_coordinate62_decomposition.json"])
    ranking = rank_producers(decomposition, controls, gates["actual"])
    sufficient = [r["producer"] for r in ranking if r["frozen_singleton_coordinate62_rescue"]]
    classification = ("conditional_inherited_L8_branch_sufficiency" if sufficient == ["inherited"]
                      else "conditional_local_L9_branch_sufficiency" if len(sufficient) == 1
                      else "unresolved")
    require(classification == result["classification"] and
            sorted(sufficient) == sorted(result["sufficient_frozen_single_branches"]),
            "saved classification mismatch")
    return {
        "status": "ANALYZED", "reviewed_result": str(root / "result.json"),
        "reviewed_result_sha256": RESULT_SHA, "corrected_prefix": PREFIX,
        "source_origins_validated": True, "saved_module_count": len(origins),
        "analyzer_origins": live_origins, "authenticated_file_count": len(bindings.verified),
        "unchanged_policy_metadata": {k: contract[k] for k in
                                      ("policy_id", "thresholds", "reference", "mapping", "authentication")},
        "source_operand_state_KV_lineage_checks": "PASS",
        "validation_boundary": "Authenticated saved evidence; no new numerical or native-layer evaluation.",
        "saved_gate_layers": list(range(9, 14)), "saved_gate_file_count": len(expected_gates),
        "saved_gate_record_count": len(expected_gates) * 19,
        "native_layer_executions": 0, "accepted_L0_L8_execution": False, "rtl_invocations": 0,
        **first_divergences(gates["actual"]),
        "materiality_definition": "First saved coordinate62 FP16-trajectory predicate failure in L9-L13; "
                                  "not a new threshold and not a mandatory-gate failure.",
        "entry_coordinate62": read_json(artifacts["L8_entry_coordinate62.json"]),
        "ranked_producer_gates": ranking, "classification": classification,
        "ranking_basis": "Descending absolute signed Q24 branch error at L9; not causal necessity.",
        "nonproducer_signed_terms": {k: v for k, v in decomposition["signed_parts"].items()
                                     if k in ("final_FP16_projection", "negative_original_addition_roundoff")},
        "inherited_native_rescue": controls["inherited_native"]["index62"]["accepted"],
        "mapped_output_coordinate62_rescue": controls["mapped62"]["index62"]["accepted"],
        "unique_upstream_producer_attributed": False,
        "retained_status": result["retained_status"], "candidate_admitted": False,
        "policy_adopted": False, "successor_published": False, "normal_host_review": "REQUIRED",
        "missing_evidence": result["missing_evidence"], "claim_boundary": result["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=REVIEWED)
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.result_root), indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"producer-cone analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
