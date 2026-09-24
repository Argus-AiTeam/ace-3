"""Check-only v2 authentication for the unchanged L12-parent coordinate diagnostic.

The adjacent contract publishes the repository-bound command. Historical
sources remain bound to their frozen snapshots, separately from current loaded
sources. No native controls, layer dispatch or scientific result is available
through this entry point.
"""

import argparse
from contextlib import ExitStack
import importlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from ace3.model.candidates import diagnose_q24_s16_l13_l12_parent_coordinate_cut_v1 as legacy


ROOT = legacy.ROOT
ID = "ace3-q24-s16-l13-l12-parent-coordinate-cut-v2"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l13_l12_parent_coordinate_cut_v2"
TEST_MODULE = "tests.test_q24_s16_l13_l12_parent_coordinate_cut_v2"
CONTRACT = ROOT / "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v2.json"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 "
           f"/home/argustest/miniconda3/bin/python -B -m {MODULE} --check")
BLOCKED = ROOT / "build/q24_s16_l13_l12_parent_coordinate_cut_7e4a72f81a3e_attempt001/result.json"
BLOCKED_SHA = "9c035e6c8d0a22659bc27f9c9a53f24086fc0badc11c3551e702febba67eac29"
require = legacy.require
record = legacy.record
prior = legacy.prior
upstream = legacy.upstream
native = legacy.native
paired = legacy.paired
interventions = legacy.interventions
intervene = legacy.intervene
classify = legacy.classify
compare_parents = legacy.compare_parents
check_reports = legacy.check_reports

# These are authentication changes, not substitutions for the executed sources.
SOURCE_UPDATES = {
    "ace3/contracts/candidates/q24_s16_toward_zero_l9_l23_v1.json": {
        "snapshot": "002_q24_s16_toward_zero_l9_l23_v1.json",
        "retained_bytes": 2865,
        "retained_sha256": "f9c2234c5d0c40cddacb94c20abc36abb79ebbf851098b0fdb0ab0da3ea4d6bf",
        "current_bytes": 3005,
        "current_sha256": "ae5985b558d365a4967949535658a243f63b754062d18af2f83f9a3a4100b2b3",
    },
    "ace3/model/candidates/run_q24_s16_toward_zero_l9_l23_v1.py": {
        "snapshot": "019_run_q24_s16_toward_zero_l9_l23_v1.py",
        "retained_bytes": 25964,
        "retained_sha256": "2019d1ad0472eb2090ec7881e44644efcf9203c0dd428fdedc5b20395ab5098a",
        "current_bytes": 26065,
        "current_sha256": "08e19ca69e8db6b3875478188d6febada9c18f231b13ba089af023ac7fcddaa6",
    },
    "tests/test_q24_s16_toward_zero_l9_l23_v1.py": {
        "snapshot": "024_test_q24_s16_toward_zero_l9_l23_v1.py",
        "retained_bytes": 12878,
        "retained_sha256": "9727cf0ca7137e4175960449a30e0573b580e99b0f839c7548fd9d1c480cc2af",
        "current_bytes": 12883,
        "current_sha256": "4a5aacbeda7455b6333d737cbd97b9bb3a8ee84e01f8f988ca18f18605d98e61",
    },
}
PRESERVED_FIELDS = (
    "policy_id", "candidate_admitted", "policy_adopted", "successor_published",
    "rtl_invocations", "normal_host_review", "reviewed_result_sha256", "width",
    "shard_size", "controls", "position", "history", "suffix", "interventions",
    "mapping", "thresholds", "gates", "classification", "claim_boundary",
)


def source_records(relative, update):
    path = ROOT / relative
    retained = {"path": str(path), "bytes": update["retained_bytes"],
                "sha256": update["retained_sha256"]}
    current = {"path": str(path), "bytes": update["current_bytes"],
               "sha256": update["current_sha256"]}
    snapshot = {**retained, "path": str(prior.INPUT / "source" / update["snapshot"])}
    return retained, current, snapshot


def bind_input(item, bound, updates):
    signature = {key: item[key] for key in ("path", "bytes", "sha256")}
    path = Path(signature["path"])
    require(path.is_absolute() and path.resolve().is_relative_to(ROOT),
            f"input/source outside isolated repository: {path}")
    relative = str(path.relative_to(ROOT))
    if relative not in SOURCE_UPDATES:
        return upstream.bind_input(item, bound)
    retained, current, snapshot = source_records(relative, SOURCE_UPDATES[relative])
    require(signature == retained, f"unexpected historical source binding: {path}")
    upstream.bind_input(snapshot, bound)
    upstream.bind_input(current, bound)
    updates[relative] = {"retained_original": retained, "retained_snapshot": snapshot,
                         "current": current, "historical_source_relabelled": False}
    return Path(snapshot["path"])


def check_contract(contract):
    original = json.loads(legacy.CONTRACT.read_text())
    for key in PRESERVED_FIELDS:
        require(type(contract[key]) is type(original[key]) and contract[key] == original[key],
                f"v1 scientific contract changed: {key}")
    expected = {
        "diagnostic_id": ID, "cwd": str(ROOT), "command": COMMAND,
        "interface": ["--check"], "native_layer_invocations": 0,
        "scientific_result_claim": False, "source_updates": SOURCE_UPDATES,
        "historical_blocked_result": str(BLOCKED), "historical_blocked_sha256": BLOCKED_SHA,
        "retained_freeze_sha256": prior.FREEZE_SHA,
    }
    for key, value in expected.items():
        require(type(contract[key]) is type(value) and contract[key] == value,
                f"v2 check contract mismatch: {key}")


def authenticate():
    bound, updates = {}, {}
    bind = lambda item: bind_input(item, bound, updates)
    freeze_record = record(prior.INPUT / "freeze.json")
    require(freeze_record["sha256"] == prior.FREEZE_SHA, "wrong retained attempt002 freeze")
    freeze = json.loads(bind(freeze_record).read_text())
    for source in freeze["sources"]:
        require(source["original"]["sha256"] == source["snapshot"]["sha256"]
                and source["original"]["bytes"] == source["snapshot"]["bytes"],
                "retained source snapshot mismatch")
        bind(source["original"])
        bind(source["snapshot"])
    require(set(updates) == set(SOURCE_UPDATES), "missing selected source authentication")
    reviewed_record = record(legacy.REVIEWED / "result.json")
    require(reviewed_record["sha256"] == legacy.REVIEWED_SHA, "wrong reviewed parent-cut result")
    reviewed = json.loads(bind(reviewed_record).read_text())
    require(reviewed["diagnostic_id"] == upstream.ID and reviewed["status"] == "DIAGNOSED"
            and reviewed["reviewed_L12_cut_reproduced"] is True
            and reviewed["native_retained_bitwise_reproduction"] is True
            and reviewed["unchanged_baseline_gates"] is True
            and reviewed["original_global_reference_unchanged"] is True
            and reviewed["source_operand_state_KV_lineage_checks"] == "PASS"
            and reviewed["rtl_invocations"] == 0
            and all(reviewed[k] is False for k in
                    ("candidate_admitted", "policy_adopted", "successor_published")),
            "reviewed parent-cut scope mismatch")
    for item in reviewed["input_bindings"] + reviewed["artifacts"] + [reviewed["validation"]]:
        bind(item)
    for item in reviewed["origins_after_execution"].values():
        if "path" in item:
            bind(item)
    blocked_record = record(BLOCKED)
    require(blocked_record["sha256"] == BLOCKED_SHA, "historical blocked result changed")
    blocked = json.loads(bind(blocked_record).read_text())
    require(blocked["diagnostic_id"] == legacy.ID and blocked["status"] == "BLOCKED",
            "historical attempt001 is not BLOCKED")
    return {
        "retained_freeze": freeze_record, "reviewed_parent_cut": reviewed_record,
        "historical_attempt001": blocked_record, "historical_status": blocked["status"],
        "source_updates": updates, "input_bindings": list(bound.values()),
        "authentication_scope": "Read-only source/artifact binding closure. Existing retained "
        "operand/state/KV/lineage and gate evidence is authenticated, not numerically rerun.",
    }


def origins():
    require(Path.cwd() == ROOT and Path(sys.executable) == prior.PYTHON
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "entry source origin mismatch")
    require(CONTRACT.resolve() == ROOT / "ace3/contracts/candidates/q24_s16_l13_l12_parent_coordinate_cut_v2.json",
            "contract origin mismatch")
    importlib.import_module(TEST_MODULE)
    result = {}
    for name, module in sorted(list(sys.modules.items())):
        if name not in ("ace3", "tests") and not name.startswith(("ace3.", "tests.")):
            continue
        expected = ROOT.joinpath(*name.split("."))
        filename = getattr(module, "__file__", None)
        if filename is None:
            locations = list(getattr(module, "__path__", ()))
            require(locations and all(Path(p).resolve() == expected for p in locations),
                    f"namespace origin mismatch: {name}")
            result[name] = {"namespace_paths": locations}
        else:
            path = Path(filename).resolve()
            require(path.is_relative_to(ROOT)
                    and path in (expected.with_suffix(".py"), expected / "__init__.py"),
                    f"module origin mismatch: {name}")
            result[name] = record(path)
    require(MODULE in result and TEST_MODULE in result, "missing candidate/test origins")
    return result


def validate():
    source_origins = origins()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(source_origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(source_origins[name])
    contract_record = record(CONTRACT)
    check_contract(json.loads(CONTRACT.read_text()))
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    log = io.StringIO()
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
            "native controls and layer dispatch forbidden in v2 check")))
            for owner, name in ((legacy, "execute"), (legacy, "diagnose"),
                                (upstream, "execute_layer"), (native, "stages"),
                                (native, "execute_layers"))]
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        require(count == tests.EXPECTED_TESTS and result.testsRun == count
                and result.wasSuccessful() and not result.skipped,
                f"focused validation failed:\n{log.getvalue()}")
        authentication = authenticate()
        require(not any(guard.called for guard in guards), "native dispatch attempted")
    require(origins() == source_origins and record(CONTRACT) == contract_record,
            "source/contract changed during check")
    return {
        "status": "VALIDATED_SOFTWARE_ONLY", "diagnostic_id": ID,
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": os.environ["PYTHONPATH"],
        "command": COMMAND, "origins": source_origins, "compiled": compiled,
        "contract": contract_record, "collected": count, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "unittest_log": log.getvalue(),
        "authentication": authentication,
        "native_layer_invocations": 0, "rtl_invocations": 0,
        "accepted_L0_L8_tests_executed": False, "scientific_result_claim": False,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "normal_host_review": "REQUIRED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(validate(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
