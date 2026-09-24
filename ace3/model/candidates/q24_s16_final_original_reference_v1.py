"""Original-input L23 -> final reference extension, CPU software only.

Use PYTHONDONTWRITEBYTECODE=1 python -B -m
ace3.model.candidates.q24_s16_final_original_reference_v1 --create
[--out build/q24_s16_final_original_reference_v1_attemptNNN].
--verify authenticates and recomputes the same reference-only extension.
Both modes emit one JSON document to stdout. Occupied attempts are never reused.

The retained freeze.json is not extended in place. The new freeze.json#final
binds its unchanged L23 binary64 and FP16 trajectories, official final operands
and tokenizer. FP16 diagnostics round at operator boundaries, not by rounding
the binary64 trajectory. No candidate state enters reference arithmetic.
Existing final-head preparation remains unchanged and does not accept this new
binding automatically: its future integration requires separate review.
The frozen FP16 stage uses the PyTorch CPU float64-to-float16 conversion
(via float32 in the bound runtime); it is not NumPy's direct float64-to-half
conversion. Both final FP16 boundaries retain that original diagnostic policy.

Focused tests accept ACE3_FINAL_REFERENCE_TEST_OUT to select an attempt.
Repeating them verifies that attempt afresh, without overwriting it.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

from ace3.model.candidates import remaining_layers_v3 as original
from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_preflight_v1 as parent


ROOT = parent.ROOT
NAME = "q24_s16_final_original_reference_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
require, same, read_bound, record = parent.require, parent.same, parent.read_bound, parent.record
torch = parent.parent.producer.legacy.torch
safe_open = parent.parent.producer.legacy.safe_open
ARRAYS = parent.FINAL_ARRAYS
FILES = {"freeze.json"} | {key + ".npy" for key, _, _ in ARRAYS}
FLAGS = {
    **parent.FLAGS, "evidence_writes": 5, "retained_evidence_writes": 0,
    "original_reference_recomputation": False, "reference_suffix_computation": True,
    "reference_rmsnorm_invocations": 2, "reference_lm_head_invocations": 2,
}
BOUNDARY = (
    "Reference-only original-input final suffix, not candidate final-head execution. "
    "All nine L23 failures, historical failures, exact thresholds and source/operand/"
    "state/KV/lineage gates remain unchanged. No original prefix/admission/native "
    "layer replay or reference reanchoring. Q24 residual state is wider than FP16; "
    "native G128 asymmetric INT4 GEMM packing, no qzero plus-one, FP16 scales, "
    "operator boundaries and KV are unchanged. No RTL, hardware, GPU, simulation, "
    "strict-FP16-state W4A16, new-token, successor or full-model admission claim. "
    "Host-managed role/model/account/budget/access/lock/concurrency controls are "
    "unchanged; no service or scheduler operations. Normal independent Reviewer "
    "validation is required before acceptance."
)


def output_path(value, *, existing=False):
    path = Path(value)
    out = path if path.is_absolute() else ROOT / path
    require(out.resolve() == out and out.parent == ROOT / "build"
            and re.fullmatch(NAME + r"_attempt[0-9]{3,}", out.name) is not None
            and int(out.name.rsplit("attempt", 1)[1]) > 0,
            "output must be a canonical versioned direct child of build")
    require(not out.is_symlink(), "output symlinked")
    require(out.is_dir() if existing else not out.exists(), "output missing or occupied")
    ignored = parent.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(out)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored: " +
            ignored.stderr.decode(errors="replace"))
    return out


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def checked_array(payload, shape, dtype):
    require(payload.startswith(b"\x93NUMPY"), "reference must be an NPY array")
    stream = io.BytesIO(payload)
    array = np.load(stream, allow_pickle=False)
    require(array.shape == shape and array.dtype.str == dtype
            and stream.tell() == len(payload)
            and np.all(np.isfinite(array.view("<f2") if dtype == "<u2" else array)),
            "invalid reference shape/dtype/finite values/trailing bytes")
    require(npy_bytes(array) == payload, "noncanonical NPY encoding")
    return array


def fp16_input(payload):
    # np.savez's zero-comment EOCD must terminate the authenticated archive.
    require(len(payload) >= 22 and payload[-22:-18] == b"PK\x05\x06"
            and payload[-2:] == b"\0\0", "trailing or noncanonical FP16 archive")
    sizes = (896, 896, 128, 128, 896, 128, 128, 128, 14, 14,
             896, 896, 896, 896, 4864, 4864, 4864, 896, 896)
    expected = {f"stage{s:02d}.npy": (size,) for s, size in enumerate(sizes)}
    expected["input_hidden.npy"] = (896,)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(len(archive.namelist()) == len(expected)
                and set(archive.namelist()) == set(expected), "FP16 archive members changed")
        arrays = {key: checked_array(archive.read(key), shape, "<u2")
                  for key, shape in expected.items()}
    return arrays["stage18.npy"].view("<f2").astype("<f8")


def authenticate():
    summary = parent.authenticate()["summary"]
    require(summary["status"] == "BLOCKED_MISSING_FINAL_REFERENCE"
            and summary["assets"]["tokenizer_status"] == "BOUND"
            and not summary["assets"]["missing_tokenizer"],
            "expected missing final binding or bound tokenizer changed")
    same(summary["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9},
         "nine retained failures changed")
    return summary


def reference_arrays(summary):
    binding, assets = summary["L23_original_reference"], summary["assets"]
    manifest = json.loads(read_bound(binding["manifest"]))
    same(manifest["layers"]["23"], binding["reference"], "L23 input reanchored")
    same(manifest["checkpoint"], assets["checkpoint"], "checkpoint splice")
    require("final" not in manifest and manifest["reference_only"] is True
            and manifest["scope"]["history"] == [9707] and manifest["scope"]["position"] == 0
            and manifest["global_reference_policy"] == parent.census.REFERENCE_POLICY,
            "original reference policy/history changed")
    h64 = checked_array(read_bound(binding["reference"]["binary64"]), (896,), "<f8")
    h16 = fp16_input(read_bound(binding["reference"]["fp16"]))
    generators = {Path(pin["path"]).name: pin for pin in manifest["generators"]}
    same(record(Path(original.__file__).resolve()), generators["remaining_layers_v3.py"],
         "frozen reference helper changed")
    namespace = {"np": np, "torch": torch}
    original.definitions(generators["official_single_decoder_layer.py"],
                         ("_torch_rmsnorm",), namespace)
    torch.set_num_threads(1)
    require(str(torch.tensor(0, device="cpu").device) == "cpu", "CPU reference required")
    result = {}
    with safe_open(assets["checkpoint"]["path"], framework="numpy") as model:
        weight = model.get_tensor("model.norm.weight")
        require(weight.shape == (896,) and weight.dtype.str == "<f2"
                and np.all(np.isfinite(weight))
                and hashlib.sha256(weight.tobytes()).hexdigest()
                == assets["tensors"]["model.norm.weight"]["sha256"],
                "final norm operand changed after authentication")
        rmsnorm = namespace["_torch_rmsnorm"]
        n64 = rmsnorm(torch.from_numpy(h64.reshape(1, 896)), weight)[0]
        n16 = rmsnorm(torch.from_numpy(h16.reshape(1, 896)), weight)[0].to(torch.float16)
        result["rmsnorm_binary64"] = n64.numpy().astype("<f8")
        result["rmsnorm_fp16"] = n16.numpy().astype("<f2").view("<u2")
        result["logits_binary64"] = np.empty(151936, dtype="<f8")
        result["logits_fp16"] = np.empty(151936, dtype="<u2")
        head = model.get_slice("lm_head.weight")
        require(head.get_shape() == [151936, 896], "head shape changed")
        digest = hashlib.sha256()
        for start in range(0, 151936, 2048):
            end = min(start + 2048, 151936)
            chunk = np.asarray(head[start:end])
            require(chunk.dtype.str == "<f2" and np.all(np.isfinite(chunk)),
                    "invalid head operand")
            digest.update(chunk.tobytes())
            weights = torch.from_numpy(chunk.astype("<f8"))
            result["logits_binary64"][start:end] = (weights @ n64).numpy()
            result["logits_fp16"][start:end] = (
                weights @ n16.to(torch.float64)).to(torch.float16).numpy().view("<u2")
        require(digest.hexdigest() == assets["tensors"]["lm_head.weight"]["sha256"]
                == parent.FINAL_CONTRACT["head"]["tied_value_sha256"],
                "head operand changed after authentication")
    for key, shape, dtype in ARRAYS:
        checked_array(npy_bytes(result[key]), shape, dtype)
    return result


def manifest_for(out, summary, arrays):
    binding, assets = summary["L23_original_reference"], summary["assets"]
    final = {
        "input_binary64": binding["reference"]["binary64"],
        "input_fp16": binding["reference"]["fp16"],
        "checkpoint": assets["checkpoint"], "tensors": assets["tensors"],
        "tokenizer": assets["tokenizer"], "history": [9707], "position": 0,
        "reference_policy": parent.census.REFERENCE_POLICY, "reference_only": True,
    }
    for key, shape, dtype in ARRAYS:
        payload = npy_bytes(arrays[key])
        final[key] = {"path": str(out / (key + ".npy")),
                      "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                      "shape": list(shape), "dtype": dtype}
    return {
        "diagnostic_id": NAME, "schema_version": 1, "implementation_revision": 2,
        "status": "REFERENCE_ONLY_FINAL_BOUND_PENDING_REVIEW",
        "final_binding": str(out / "freeze.json") + "#final", "final": final,
        "original_reference_manifest": binding["manifest"],
        "retained_parent": summary, "flags": FLAGS,
        "source_bindings": [record(path) for path in (SOURCE, TEST, parent.SOURCE)],
        "arithmetic": {
            "binary64": "frozen _torch_rmsnorm; CPU float64 head matvec, 2048-row chunks",
            "fp16": "own original FP16 L23 stage18; float64 RMSNorm and head matvec; "
                    "each boundary uses frozen propagated_reference stage's "
                    "PyTorch CPU float64.to(float16), via float32 in bound runtime",
            "runtime": {"torch": torch.__version__, "numpy": np.__version__},
            "epsilon": "1/1000000", "candidate_inputs": False,
            "original_reference_recomputation": "none; only missing final suffix computed",
        },
        "output": {"path": str(out), "fresh_creation_only": True, "ignored": True,
                   "direct_child": True, "overwrite": False},
        "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
    }


def check_output(out, expected):
    require({path.name for path in out.iterdir()} == FILES, "output artifact set changed")
    path = out / "freeze.json"
    require(path.resolve() == path, "noncanonical final manifest")
    same(json.loads(path.read_bytes()), expected, "final binding changed or reanchored")
    for key, shape, dtype in ARRAYS:
        checked_array(read_bound(expected["final"][key]), shape, dtype)
    return record(path)


def run(value, *, verify=False):
    out = output_path(value, existing=verify)
    if not verify:
        out.mkdir(exist_ok=False)
    try:
        summary = authenticate()
        arrays = reference_arrays(summary)
        expected = manifest_for(out, summary, arrays)
        if not verify:
            for key, _, _ in ARRAYS:
                with (out / (key + ".npy")).open("xb") as stream:
                    stream.write(npy_bytes(arrays[key]))
            with (out / "freeze.json").open("x") as stream:
                json.dump(expected, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        pin = check_output(out, expected)
    except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        if not verify:
            with (out / "failure.json").open("x") as stream:
                json.dump({"status": "FAILED", "error": str(error),
                           "candidate_admitted": False, "normal_host_review": "REQUIRED"}, stream)
        raise
    return {"diagnostic_id": NAME, "status": "VERIFIED" if verify else "CREATED_AND_VERIFIED",
            "manifest": pin, "final_binding": str(out / "freeze.json") + "#final",
            "arrays": {key: expected["final"][key] for key, _, _ in ARRAYS},
            "retained_L23_stage_reports": summary["retained_L23_stage_reports"],
            "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
            **{**FLAGS, "evidence_writes": 0 if verify else 5}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)
    try:
        require(Path(__file__).resolve() == SOURCE and Path.cwd().resolve() == ROOT
                and Path(parent.__file__).resolve() == parent.SOURCE
                and sys.dont_write_bytecode and not sys.flags.optimize
                and os.getuid() == 1000, "source/cwd/bytecode/optimization/account gate failed")
        result = run(args.out, verify=args.verify)
    except (ValueError, OSError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        result = {"diagnostic_id": NAME, "status": "FAILED", "error": str(error),
                  "candidate_admitted": False, "normal_host_review": "REQUIRED"}
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
