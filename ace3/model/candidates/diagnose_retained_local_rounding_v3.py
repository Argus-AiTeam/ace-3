"""Read-only P0 canonical-suffix controls, not RTL replay or admission."""

import argparse
import hashlib
import json
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates.binary64_fp16_excess_v1 import evaluate_layer_final_output


POLICY = "ace3-w4a16-local-operator-global-binary64-authority-v3"
# P0 has one visible value: Q/K cannot affect its unit softmax probability.
SUFFIXES = {
    "canonical_s16_suffix": (16, 17),
    "canonical_s13_suffix": (13, 14, 15, 16, 17),
    "canonical_s3_suffix": (3, 7, 10, 11, 12, 13, 14, 15, 16, 17),
    "canonical_s0_suffix": (0, 3, 7, 10, 11, 12, 13, 14, 15, 16, 17),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def bound(item):
    actual = record(item["path"])
    require(all(actual[key] == item[key] for key in ("bytes", "sha256")),
            f"retained binding mismatch: {item['path']}")
    return Path(item["path"])


def load_words(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def value(word):
    return Fraction.from_float(float(np.asarray(word, dtype="<u2").view("<f2")))


def residual(arrays):
    return sum(local.finite_words(arrays[name], (896,))
               for name in ("input_hidden", "stage11", "stage17"))


def witness(arrays, index, reference):
    operands = {name: value(arrays[name][index])
                for name in ("input_hidden", "stage11", "stage17")}
    total = sum(operands.values())
    bits = int(local.rounded(residual(arrays))[index])
    gate = evaluate_layer_final_output(
        actual_fp16_bits=bits, reference_binary64_hex=reference)
    return {
        "index": index,
        "operand_bits": {name: f"{int(arrays[name][index]):04x}" for name in operands},
        "operand_exact": {name: str(v) for name, v in operands.items()},
        "exact_sum": str(total), "sum_decimal": float(total),
        "final_rounding_delta": str(value(bits) - total),
        "binary64_v1": gate,
    }


def run(root, out):
    started = time.monotonic()
    admission = root / "runtime/layer09/admission/result.json"
    # The root is the explicitly requested retained L9 transaction, not a new run.
    result = read_json(admission)
    require(result["status"] == result["numerical_status"] == "FAIL",
            "expected a retained numerical FAIL")
    require(result["policy_id"] == POLICY
            and result["evidence_kind"] == "actual_rtl_numerical_evaluation",
            "wrong retained policy or evidence kind")
    invocation = read_json(bound(result["admission_invocation"]))
    manifest_path = bound(invocation["manifest"])
    require(invocation["manifest"]["sha256"] == invocation["trusted_manifest_sha256"],
            "retained Host invocation manifest binding mismatch")
    manifest = read_json(manifest_path)
    layer, position = result["reports"][-1]["node"][:2]
    require(layer == 9 and position == 0 and manifest["history"] == [9707],
            "this bounded diagnostic requires the retained L9/P0 transaction")
    require(result["lineage"]["status"] == "PASS", "retained lineage was not admitted")
    require(all(r["local_operator_fp16"]["passed"] for r in result["reports"][:-1]),
            "an earlier mandatory local gate failed")
    reports = result["reports"][-1]["binary64_v1"]
    failures = reports["failures"]
    require(reports["coordinates"] == 896 and len(failures) == 1,
            "expected the single retained global failure")
    index = failures[0]["index"]
    reference_hex = failures[0]["reference_binary64_hex"]
    reference = Fraction.from_float(float.fromhex(reference_hex))

    extension = read_json(bound(manifest["reference_extension"]))
    refs = extension["layers"][str(layer)]
    arrays_path = bound(manifest["actual_operands"])
    locals_path = bound(result["local_references"])
    arrays, retained_local = load_words(arrays_path), load_words(locals_path)
    input64_path, final64_path = bound(refs["input_binary64"]), bound(refs["binary64"])
    input64, final64 = (np.load(p, allow_pickle=False) for p in (input64_path, final64_path))
    require(input64.shape == final64.shape == (896,)
            and input64.dtype == final64.dtype == np.dtype("float64")
            and np.all(np.isfinite(input64)) and np.all(np.isfinite(final64)),
            "invalid original-global arrays")
    require(float(final64[index]).hex() == reference_hex, "global witness mismatch")
    original_fp16 = load_words(bound(refs["fp16"]))
    for stage, count in local.SIZES.items():
        local.finite_words(arrays[f"stage{stage:02d}"], (count,))
    local.finite_words(arrays["input_hidden"], (896,))
    require(arrays["input_cache_k"].shape == arrays["input_cache_v"].shape == (0, 128),
            "P0 cache is not empty")
    require(np.array_equal(arrays["stage09"], local.rounded(np.ones(14))),
            "P0 probability is not one")
    require(np.array_equal(local.rounded(residual(arrays)), arrays["stage18"]),
            "retained final residual does not match exact own-input rounding")
    parent = read_json(bound(manifest["parent"]))
    require(parent["status"] == parent["numerical_status"] == "PASS",
            "retained L8 parent was not accepted")
    prepared = read_json(bound(manifest["prepared"]))
    hidden_file = bound(prepared["input_hidden"])
    hidden = np.asarray([int(line.split()[-1], 16)
                         for line in hidden_file.read_text().splitlines() if line.strip()],
                        dtype="<u2")
    local.validate_lineage(arrays, hidden, position=position, history=manifest["history"])

    source_records = manifest["source_closure"] + manifest["evaluator_extension"]["sources"]
    for item in source_records:
        bound(item)
    canonical = refs["canonical"]
    with safe_open(extension["checkpoint"]["path"], framework="np") as model:
        tensors = {name: model.get_tensor(name) for name in local.tensor_shapes(layer)}
    local.authenticate_tensors(tensors, canonical, layer)
    setup_seconds = time.monotonic() - started
    differences = {}
    for stage in range(18):
        name = f"stage{stage:02d}"
        positions = np.flatnonzero(arrays[name] != retained_local[name])
        if len(positions):
            differences[str(stage)] = [
                {"index": int(i), "actual": f"{int(arrays[name][i]):04x}",
                 "same_input_canonical": f"{int(retained_local[name][i]):04x}"}
                for i in positions]
    require(set(differences) == {"0", "3", "13", "16"},
            "retained primitive deviation scope changed")

    baseline = witness(arrays, index, reference_hex)
    require(all(baseline["binary64_v1"][key] == failures[0][key] for key in
                ("actual_fp16_bits", "reference_binary64_hex", "nearest_fp16_bits",
                 "q", "actual_error", "excess_error", "accepted")),
            "retained failing scalar did not reproduce")
    control_arrays = {}
    controls = []
    for name, stages in SUFFIXES.items():
        phase = time.monotonic()
        data = arrays.copy()
        for stage in stages:
            operands = {key: data[key] for key in local.OPERANDS[stage]}
            computed = local.local_reference(stage, operands, tensors, layer)
            if stage == stages[0]:
                require(np.array_equal(computed, retained_local[f"stage{stage:02d}"]),
                        f"fresh same-input oracle disagrees with retained S{stage}")
            data[f"stage{stage:02d}"] = computed
        data["stage18"] = local.rounded(residual(data))
        measured = witness(data, index, reference_hex)
        measured.update(
            name=name, canonical_stages=list(stages),
            elapsed_seconds=time.monotonic() - phase,
            changed_words={f"stage{s:02d}": int(np.count_nonzero(
                data[f"stage{s:02d}"] != arrays[f"stage{s:02d}"])) for s in (*stages, 18)})
        controls.append(measured)
        for stage in (*stages, 18):
            control_arrays[f"{name}_stage{stage:02d}"] = data[f"stage{stage:02d}"]

    inherited = value(arrays["input_hidden"][index]) - Fraction.from_float(float(input64[index]))
    increment = value(arrays["stage11"][index]) + value(arrays["stage17"][index])
    original_increment = reference - Fraction.from_float(float(input64[index]))
    nearest = value(int(failures[0]["nearest_fp16_bits"], 16))
    radius = Fraction(failures[0]["q"]) + Fraction(1, 8)
    nearest_bits = int(failures[0]["nearest_fp16_bits"], 16)
    neighbour_values = sorted((value(nearest_bits - 1), value(nearest_bits + 1)))
    unique_passing = (neighbour_values[0] < reference - radius <= nearest
                      <= reference + radius < neighbour_values[1])
    require(unique_passing, "witness requires a different passing-set analysis")
    passing_upper_midpoint = (nearest + neighbour_values[1]) / 2
    for item in [baseline, *controls]:
        item["sum_above_passing_upper_midpoint"] = str(
            Fraction(item["exact_sum"]) - passing_upper_midpoint)
    decomposition = {
        "original_global_input_hex": float(input64[index]).hex(),
        "original_global_input_exact": str(Fraction.from_float(float(input64[index]))),
        "inherited_input_error": str(inherited),
        "actual_increment": str(increment), "original_global_increment": str(original_increment),
        "increment_difference": str(increment - original_increment),
        "pre_round_total_error": str(inherited + increment - original_increment),
        "rounding_delta": baseline["final_rounding_delta"],
        "passing_output_interval": [str(reference - radius), str(reference + radius)],
        "unique_passing_fp16": failures[0]["nearest_fp16_bits"],
        "passing_upper_midpoint": str(passing_upper_midpoint),
        "original_fp16_witness_bits": {
            key: f"{int(val[index]):04x}" for key, val in original_fp16.items()
            if key in ("stage11", "stage12", "stage13", "stage17", "stage18")},
    }
    require(inherited + increment - original_increment
            + Fraction(baseline["final_rounding_delta"])
            == value(arrays["stage18"][index]) - reference,
            "exact error decomposition does not close")
    report = {
        "status": "DIAGNOSTIC_COMPLETE_NOT_ADMISSION",
        "retained_l9_status": result["status"], "policy_id": POLICY,
        "evidence_kind": "retained_same_input_canonical_suffix_software_counterfactual",
        "scope": "L9/P0 final-output dependency cone, actual L8 hidden fixed",
        "boundary": (
            "Each arm canonicalizes its named cut and reachable suffix. These are nested "
            "controls, not isolated RTL patches or a proof that every locally passing "
            "implementation must fail. Q/K are outside the P0 layer-final value cone; "
            "their future cached-state effects are not evaluated."),
        "retained_manifest": record(manifest_path), "retained_result": record(admission),
        "inputs": [record(p) for p in (arrays_path, locals_path, input64_path, final64_path)],
        "canonical_tensor_authentication": "selected L9 tensors match frozen canonical hashes",
        "source_bindings": source_records,
        "diagnostic_source": record(__file__), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "primitive_differences": differences, "baseline": baseline,
        "error_decomposition": decomposition, "controls": controls,
        "canonical_suffix_repair_demonstrated": any(
            c["binary64_v1"]["accepted"] for c in controls),
        "setup_seconds": setup_seconds, "elapsed_seconds": time.monotonic() - started,
        "new_decoder_rtl_invocations": 0, "capture_or_admission_invocations": 0,
        "l10_l23_executions": 0, "global_reference_recomputed_or_seeded": False,
        "runtime_directories_observed": sorted(p.name for p in (root / "runtime").iterdir()
                                              if p.is_dir()),
    }
    np.savez(out / "canonical_suffixes.npz", **control_arrays)
    with (out / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(out / "result.json"),
        "primitive_difference_counts": {k: len(v) for k, v in differences.items()},
        "baseline": baseline, "error_decomposition": decomposition,
        "controls": controls, "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.out.is_dir() and not (args.out / "result.json").exists()
            and not (args.out / "canonical_suffixes.npz").exists(),
            "use a fresh existing ignored attempt directory")
    run(args.root.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
