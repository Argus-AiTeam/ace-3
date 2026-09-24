"""Versioned L10/P0 producer-cone inputs from the selected frozen L9 parent.

The adjacent contract publishes the repository-bound --check and --out commands.
--check authenticates and tests only, with zero native dispatch. --out validates
once before running the fixed L10-L13/P0 controls in a fresh bounded directory.
Producer arithmetic is reused unchanged from v1 in an isolated namespace.
The historical v1 result is not a source-compatible input to this path.
"""

import argparse
from contextlib import redirect_stderr
import importlib
import json
from pathlib import Path
import sys
import traceback
from types import FunctionType
import unittest
from unittest.mock import patch

from ace3.model.candidates import analyze_q24_s16_l9_coordinate62_producer_cone_v1 as selected
from ace3.model.candidates import diagnose_q24_s16_l10_coordinate62_producer_cone_v1 as legacy
from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_input_threshold_v1 as binding


entry = legacy.entry
ROOT = entry.ROOT
ID = "ace3-q24-s16-l10-coordinate62-producer-cone-v2"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l10_coordinate62_producer_cone_v2"
TEST_MODULE = "tests.test_q24_s16_l10_coordinate62_producer_cone_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l10_coordinate62_producer_cone_v2.json"
SELECTED = binding.REVIEWED[0]
HISTORICAL = binding.REVIEWED[1]
require = entry.require
plan = legacy.plan
cut_parent = legacy.cut_parent
original_branches = legacy.original_branches
BRANCH_DEFINITIONS = legacy.BRANCH_DEFINITIONS
OUTPUT_PREFIX = "q24_s16_l10_coordinate62_producer_cone_"
NATIVE_LAYERS = (10, 11, 12, 13)


def read_result(pin, bound):
    directory, digest, diagnostic_id = pin
    item = entry.record(ROOT / "build" / directory / "result.json")
    require(item["sha256"] == digest, "wrong pinned producer result")
    reviewed = json.loads(entry.upstream.bind_input(item, bound).read_text())
    require(reviewed["diagnostic_id"] == diagnostic_id, "producer result identity mismatch")
    return item, reviewed


def selected_parent(bound):
    result_binding, reviewed = read_result(SELECTED, bound)
    selected.check_metadata(reviewed)
    bind = lambda item: entry.upstream.bind_input(item, bound)
    origin = reviewed["origins_after_execution"][entry.MODULE]
    require(origin["path"] == str(Path(entry.__file__).resolve()),
            "selected L9 entry source origin mismatch")
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    policy = binding.authenticate_policy(reviewed, bound)

    def artifact(name):
        path = str(ROOT / "build" / SELECTED[0] / name)
        matches = [item for item in reviewed["artifacts"] if item["path"] == path]
        require(len(matches) == 1, f"missing or duplicate selected artifact: {name}")
        return matches[0]

    controls = json.loads(bind(artifact("interventions.json")).read_text())
    require([row["label"] for row in controls] == [label for label, _ in entry.plan()],
            "selected L9 controls changed")
    parent_binding = artifact("actual_L9_parent.npz")
    require(controls[0]["parent"] == parent_binding, "selected L9 actual-parent linkage mismatch")
    parent = entry.native.load_state(parent_binding, bind)
    arrays = entry.native.load_state(artifact("actual_L9.npz"), bind)
    entry.prior.retained.verify_parent(
        parent, entry.prior.retained.state_from(arrays, "output", "stage18"))
    return {
        "parent": parent, "arrays": arrays,
        "L10_arrays": entry.native.load_state(artifact("actual_L10.npz"), bind),
        "L10_reports": json.loads(bind(artifact("actual_L10_gates.json")).read_text()),
        "evidence": {
            "result": result_binding, "parent": parent_binding,
            "entry_source": origin, "policy_binding": policy, "read_only": True,
        },
    }


def authenticate():
    # Keep the complete accepted-parent, operand, reference and KV authentication.
    data = entry.authenticate()
    current = selected_parent(data["bound"])
    entry.paired.same_arrays(current["arrays"], data["layers"][9]["arrays"])
    entry.paired.same_arrays(current["parent"], data["layers"][9]["output"])
    entry.paired.same_arrays(current["parent"], data["layers"][10]["parent"])
    entry.paired.same_arrays(current["L10_arrays"], data["layers"][10]["arrays"])
    entry.coordinate.check_reports(current["L10_reports"], data["layers"][10]["reports"])
    data["layers"][10]["parent"] = current["parent"]
    data["selected_L9_evidence"] = current["evidence"]
    item, historical = read_result(HISTORICAL, data["bound"])
    old_source = historical["origins_after_execution"][entry.MODULE]
    require(old_source["path"] == current["evidence"]["entry_source"]["path"]
            and old_source["sha256"] != current["evidence"]["entry_source"]["sha256"],
            "historical L10 source incompatibility changed")
    data["historical"] = [{
        "result": item, "entry_source": old_source, "read_only": True,
        "source_compatible": False, "execution_authorized_by_this_diagnostic": False,
    }]
    return data


def source_context():
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "L10 v2 source origin mismatch")
    importlib.import_module(TEST_MODULE)
    origins = binding.runtime.source_context()
    require(MODULE in origins and TEST_MODULE in origins, "missing L10 v2 source origins")
    return origins


def validate():
    origins = source_context()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(origins[name])
    contract = json.loads(CONTRACT.read_text())
    require(contract["diagnostic_id"] == ID
            and contract["policy_id"] == entry.prior.gates.POLICY_ID
            and contract["selected_L9"] == list(SELECTED)
            and contract["historical_L10"] == list(HISTORICAL)
            and contract["controls"] == [label for label, _ in plan()]
            and contract["branch_definitions"] == BRANCH_DEFINITIONS
            and contract["normal_host_review"] == "REQUIRED"
            and contract["native_layer_invocations"] == 0
            and contract["rtl_invocations"] == 0
            and contract["execution"]["output_prefix"] == OUTPUT_PREFIX
            and contract["execution"]["native_layers"] == list(NATIVE_LAYERS)
            and contract["execution"]["native_layer_invocations"] == 41
            and contract["execution"]["position"] == 0
            and contract["execution"]["fresh_exclusive_directory"] is True
            and contract["execution"]["rtl_invocations"] == 0
            and all(contract[key] is False for key in (
                "candidate_admitted", "policy_adopted", "successor_published",
                "scientific_result_claim", "accepted_L0_L8_execution")),
            "L10 v2 contract mismatch")
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    with patch.object(entry.upstream, "execute_layer",
                      side_effect=AssertionError("native dispatch forbidden during preflight")) as dispatch, \
            patch.object(entry.native, "stages",
                         side_effect=AssertionError("native stages forbidden during preflight")) as stages:
        result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
        require(not dispatch.called and not stages.called, "native execution attempted during preflight")
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped, "focused validation failed")
    require(source_context() == origins, "source changed during validation")
    return {
        "status": "VALIDATED_SOFTWARE_ONLY", "diagnostic_id": ID,
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": str(ROOT),
        "origins": origins, "compiled": compiled, "contract": entry.record(CONTRACT),
        "collected": count, "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "selected_L9": entry.record(ROOT / "build" / SELECTED[0] / "result.json"),
        "selected_L9_entry_source": origins[entry.MODULE],
        "historical_L10": entry.record(ROOT / "build" / HISTORICAL[0] / "result.json"),
        "historical_L10_source_mismatch_rejected": True,
        "authentication_scope": "Full entry authentication and selected frozen L9-to-L10 linkage",
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "native_layer_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "normal_host_review": "REQUIRED",
    }


def output_path(value):
    require(not value.is_symlink(), "output must not be a symlink")
    out = legacy.output_path(value)
    require(len(out.name) > len(OUTPUT_PREFIX), "output needs a fresh name")
    return out


def diagnose(out, log, validation):
    require(source_context() == validation["origins"]
            and entry.record(CONTRACT) == validation["contract"],
            "source/contract changed before diagnostic")
    evidence, native_layers = {}, []

    def selected_inputs():
        data = authenticate()
        evidence.update(data["selected_L9_evidence"])
        return data

    # v1 is frozen evidence: bind only its orchestration globals, not its module.
    runner = FunctionType(legacy.diagnose.__code__, {
        **legacy.diagnose.__globals__, "authenticate": selected_inputs,
        "origins": source_context, "ID": ID,
    }, legacy.diagnose.__name__)
    native_stages = entry.native.stages

    def guarded_stages(tensors, layer, parent, arrays):
        require(type(layer) is int and layer in NATIVE_LAYERS,
                "native stage execution outside L10-L13; L0-L8 forbidden")
        native_layers.append(layer)
        return native_stages(tensors, layer, parent, arrays)

    with patch.object(entry.native, "stages", side_effect=guarded_stages):
        result = runner(out, log)
    expected = [10]
    for label, _ in plan():
        if label == "inherited_native":
            expected.append(10)
        expected.extend((11, 12, 13))
    require(native_layers == result["native_layer_dispatches"] == expected,
            "native stage/dispatch accounting mismatch")
    require(source_context() == validation["origins"]
            and entry.record(CONTRACT) == validation["contract"],
            "source/contract changed during diagnostic")
    result["selected_L9_evidence"] = evidence
    result["native_stage_dispatches"] = native_layers
    result["contract"] = validation["contract"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(validate()))
        return 0
    out = output_path(args.out)
    out.mkdir(exist_ok=False)
    with (out / "command.log").open("x") as log:
        log.write(json.dumps({
            "cwd": str(Path.cwd()), "executable": sys.executable, "argv": sys.argv,
            "PYTHONPATH": entry.os.environ.get("PYTHONPATH"),
            "PYTHONDONTWRITEBYTECODE": entry.os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "rtl_invocations": 0,
        }) + "\n")
        log.flush()
        try:
            with redirect_stderr(log):
                validation = validate()
            entry.write(out / "validation.json", validation)
            result = diagnose(out, log, validation)
            result["validation"] = entry.record(out / "validation.json")
            entry.write(out / "result.json", result)
        except (ValueError, OSError, KeyError, TypeError, IndexError, ArithmeticError,
                RuntimeError, AssertionError):
            traceback.print_exc(file=log)
            raise
        summary = {"status": result["status"], "result": str(out / "result.json")}
        log.write(json.dumps(summary) + "\n")
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
