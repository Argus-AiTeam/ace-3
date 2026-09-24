"""Repository-bound, non-admitting L9/P0 coordinate-62 CPU diagnostic.

From /home/argustest/ace3-argus, validate once with:
PYTHONPATH=/home/argustest/ace3-argus PYTHONDONTWRITEBYTECODE=1 \
/home/argustest/miniconda3/bin/python -B -m \
ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_runtime_v1 --check

For a scientific run replace --check with --out followed by a fresh
build/q24_s16_l9_coordinate62_runtime_... directory; use the durable runner
when the run exceeds two minutes. This version executes only L9, not the
older diagnostic's L10-L13 suffix. Frozen branch cuts are arithmetic
diagnostics, not admitted native parents or complete operator-gate passes.
Q24 residual state remains wider than FP16. No RTL, token or full-model claim.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import unittest

from ace3.model.candidates import diagnose_q24_s16_l9_coordinate62_entry_producer_cone_v1 as entry


ROOT = Path("/home/argustest/ace3-argus")
ID = "ace3-q24-s16-l9-coordinate62-runtime-v1"
MODULE = "ace3.model.candidates.diagnose_q24_s16_l9_coordinate62_runtime_v1"
TEST_MODULE = "tests.test_q24_s16_l9_coordinate62_runtime_v1"
ORIGIN_ONLY_TEST = "tests.test_q24_s16_toward_zero_l3_l8_v1"
require = entry.require


def source_context():
    require(entry.ROOT == ROOT and Path.cwd() == ROOT
            and Path(sys.executable) == Path("/home/argustest/miniconda3/bin/python")
            and entry.os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "use the published repository-bound command")
    require(Path(__file__).resolve() == ROOT.joinpath(*MODULE.split(".")).with_suffix(".py"),
            "runtime source origin mismatch")
    for name in (TEST_MODULE, entry.TEST_MODULE, ORIGIN_ONLY_TEST):
        importlib.import_module(name)
    origins = entry.origins()
    require(all(name in origins for name in (MODULE, TEST_MODULE, ORIGIN_ONLY_TEST)),
            "missing repository-bound source origins")
    return origins


def validate():
    origins = source_context()
    compiled = []
    for name in (MODULE, TEST_MODULE):
        path = Path(origins[name]["path"])
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(origins[name])
    tests = sys.modules[TEST_MODULE]
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    count = suite.countTestCases()
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(count == tests.EXPECTED_TESTS and result.testsRun == count
            and result.wasSuccessful() and not result.skipped,
            "focused test collection or execution failed")
    require(source_context() == origins, "source changed during validation")
    return {
        "cwd": str(ROOT), "executable": sys.executable, "PYTHONPATH": str(ROOT),
        "origins": origins, "compiled": compiled, "collected": count,
        "executed": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "accepted_L0_L8_tests_executed": False, "rtl_invocations": 0,
    }


def execute_layer(layer, position, parent, data):
    require(type(layer) is int and layer == 9 and type(position) is int and position == 0,
            "runtime permits only native L9/P0; native L0-L8 execution is forbidden")
    item = data["layers"][layer]
    return entry.upstream.execute_layer(
        item["tensors"], layer, parent, item["trajectory"], item["reference"])


def frozen_controls(parent, arrays, original):
    rows = []
    for label, parts in entry.plan()[:8]:
        _, _, _, scratch, output = entry.producer.frozen_parent(parent, arrays, original, parts)
        rows.append({
            "label": label, "parts": list(parts),
            "S12_Q24_units62": int(scratch["i"][62]),
            "S18_Q24_units62": int(output["i"][62]),
            "S18_FP16_bits62": f"{int(output['h'][62]):04x}",
            "index62": entry.prior.measure(int(output["h"][62]), float(original["s18"][62])),
            "complete_native_operator_gates_evaluated": False,
            "candidate_admitted": False,
        })
    by_label = {row["label"]: row for row in rows}
    base = by_label["actual"]["S18_Q24_units62"]
    for row in rows:
        require(row["S18_Q24_units62"] - base == sum(
            by_label["frozen_" + part]["S18_Q24_units62"] - base for part in row["parts"]),
            "frozen Q24 additive decomposition did not close")
    return rows


def reauthenticate(data):
    for item in data["bound"].values():
        require(entry.record(Path(item["path"])) == item,
                f"authenticated input changed: {item['path']}")
    entry.check_review(data["L8_review_binding"])


def diagnose(validation):
    started = time.monotonic()
    require(source_context() == validation["origins"], "source changed before diagnostic")
    entry.torch.set_num_threads(1)
    data = entry.authenticate()
    timing = {"authentication": time.monotonic() - started}
    began = time.monotonic()
    original = entry.original_branches(data)
    timing["original_input_L9_reference"] = time.monotonic() - began
    item = data["layers"][9]
    began = time.monotonic()
    arrays, locals_, reports = execute_layer(9, 0, item["parent"], data)
    entry.paired.same_arrays(arrays, item["arrays"])
    entry.paired.same_arrays(locals_, item["locals"])
    entry.coordinate.check_reports(reports, item["reports"])
    timing["native_L9_baseline_and_gates"] = time.monotonic() - began
    decomposition = entry.paired.parts(item["parent"], arrays, original, 62)
    controls = frozen_controls(item["parent"], arrays, original)
    mapped = entry.paired.mapped_parent(original["input"])
    incoming = entry.coordinate.intervene(item["parent"], mapped, [62])
    began = time.monotonic()
    generated, _, inherited_reports = execute_layer(9, 0, incoming, data)
    timing["native_L9_inherited_coordinate62_and_gates"] = time.monotonic() - began
    inherited = entry.prior.measure(int(generated["stage18"][62]), float(item["reference"][62]))
    reauthenticate(data)
    require(source_context() == validation["origins"], "source changed during diagnostic")
    timing["total"] = time.monotonic() - started
    return {
        "diagnostic_id": ID, "status": "DIAGNOSED", "node": [9, 0, 18], "index": 62,
        "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
        "rtl_invocations": 0, "normal_host_review": "REQUIRED",
        "accepted_L0_L8_execution": False, "native_layer_invocations": [9, 9],
        "unchanged_baseline_gates": True, "original_global_reference_unchanged": True,
        "source_operand_state_KV_lineage_checks": "PASS",
        "native_L9_retained_bitwise_reproduction": True,
        "policy_id": entry.prior.gates.POLICY_ID,
        "retained_authentication": data["authentication"],
        "input_bindings": list(data["bound"].values()),
        "metadata_only_external_review": data["L8_review_binding"],
        "L9_coordinate62_decomposition": decomposition,
        "frozen_controls": controls, "actual_L9_gates": reports,
        "inherited_native_L9_gates": inherited_reports, "inherited_native_index62": inherited,
        "classification": "L9_only_conditional_branch_diagnostic",
        "claim_boundary": (
            "No L0-L8 execution or L10+ suffix execution. Frozen scalar gate results do not "
            "certify omitted operators; all native gate failures remain in the reports. "
            "No unique producer or performance cause attributed. Q24 is wider than FP16; "
            "INT4 weights and FP16 operator boundaries/KV are unchanged. No policy, "
            "successor, strict-FP16-state W4A16, RTL, new-token or full-model PASS."
        ),
        "timing_seconds": timing, "validation": validation,
    }


def output_path(value):
    out = value.resolve()
    require(out.parent == ROOT / "build"
            and out.name.startswith("q24_s16_l9_coordinate62_runtime_"),
            "output outside bounded build scope")
    require(not out.exists(), "output already exists")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = output_path(args.out) if args.out is not None else None
    validation = validate()
    if out is None:
        print(json.dumps({"status": "VALIDATED", **validation}))
        return 0
    out.mkdir(exist_ok=False)
    # Publish only after all checks; an exception leaves validation, never a PASS.
    entry.write(out / "validation.json", validation)
    result = diagnose(validation)
    entry.write(out / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(out / "result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
