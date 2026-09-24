"""Read-only admission of externally authenticated P0 runtime artifacts.

The manifest digest is a trust input from the owning controller/Host, not a
field to copy from the candidate's result. This module neither runs a simulator
nor issues an execution authorization or an independent Reviewer verdict.
"""

import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import capture_harness_v3 as harness


ROOT = Path(__file__).resolve().parents[3]
TOP = "ace3_decoder_layer0_token_engine"
SCHEMA = "ace3-v3-actual-p0-runtime-v1"
RETAINED = ROOT / "build/single_round_residual_rtl_execution_2f35dc72423f_policy_attempt001"
SOURCE_FREEZE_SHA256 = "faa8c612c6cd2f94a5c55e591ca48053ac1921f1e3506537cf0f0955e0b7b9bb"
TRACE_LAYOUT = {
    "path": str(ROOT / "build/single_round_residual_rtl_execution_fb06353c5f53_attempt001"
                / "layer04/runtime/raw/trace.hex"),
    "bytes": 396508,
    "sha256": "e18e81745b7af8009bdbbaa6eba8c049c7057d46b14108741eb79dce874f4296",
}


def artifact(record):
    local.require(isinstance(record, dict)
                  and isinstance(record.get("path"), str)
                  and Path(record["path"]).is_absolute()
                  and type(record.get("bytes")) is int and record["bytes"] >= 0
                  and isinstance(record.get("sha256"), str)
                  and re.fullmatch("[0-9a-f]{64}", record["sha256"]) is not None,
                  "missing or invalid artifact receipt")
    data = Path(record["path"]).read_bytes()
    local.require(len(data) == record["bytes"]
                  and hashlib.sha256(data).hexdigest() == record["sha256"],
                  f"artifact receipt mismatch: {record['path']}")
    return data


def document(record):
    value = json.loads(artifact(record))
    local.require(isinstance(value, dict), "receipt document must be an object")
    return value


def unique(records, predicate, label):
    matches = [r for r in records if predicate(r)]
    local.require(len(matches) == 1, f"missing/ambiguous {label}")
    return matches[0]


def hex_words(data, indexed=False):
    rows = data.decode("ascii").splitlines()
    width = 10 if indexed else 4
    local.require(all(re.fullmatch(f"[0-9a-fA-F]{{{width}}}", r) for r in rows),
                  "invalid raw FP16 hex ABI")
    if indexed:
        local.require([int(r[:-4], 16) for r in rows] == list(range(len(rows))),
                      "raw hidden coordinate identity mismatch")
    return np.asarray([int(r[-4:], 16) for r in rows], dtype="<u2")


def trace_rows(data):
    rows = data.decode("ascii").splitlines()
    local.require(all(re.fullmatch("[0-9a-fA-F]{16}", r) for r in rows),
                  "invalid actual trace ABI")
    local.require(all(r[:2] == "00" for r in rows), "unexpected transaction token ordinal")
    # The retained C++ writer emits token, position, stage, index, FP16.
    return [(int(r[6:8], 16), int(r[2:6], 16), int(r[8:12], 16), int(r[12:], 16))
            for r in rows]


def canonical_stages(rows, coordinates):
    """Decode P0 coordinates, retaining the independently bound causal layout."""
    local.require([r[:3] for r in rows] == coordinates, "actual trace order/coverage mismatch")
    stages = {s: {} for s in local.SIZES}
    attention = []
    for stage, position, index, word in rows:
        local.require(stage in stages and position == 0 and 0 <= word <= 0xffff,
                      "invalid actual trace stage/position/value")
        if stage in (8, 9, 10):
            attention.append((stage, index))
        if stage in (8, 9):
            local.require(index == 0, "P0 attention context index must be zero")
            index = len(stages[stage])
        local.require(0 <= index < local.SIZES[stage] and index not in stages[stage],
                      f"duplicate/out-of-range actual coordinate: S{stage}/{index}")
        stages[stage][index] = word
    local.require(all(len(stages[s]) == count for s, count in local.SIZES.items()),
                  "missing actual trace coordinates")
    expected_attention = []
    for head in range(14):
        expected_attention.extend([(8, 0), (9, 0)])
        expected_attention.extend((10, head * 64 + dim) for dim in range(64))
    local.require(attention == expected_attention, "P0 attention head/occurrence causal order mismatch")
    return {f"stage{s:02d}": np.asarray([stages[s][i] for i in range(count)], dtype="<u2")
            for s, count in local.SIZES.items()}


def validate_derivation(manifest):
    """A retained re-decoding may replace operands, never execution receipts."""
    if "derivation" not in manifest:
        return
    boundary = document(manifest["derivation"])
    local.require(boundary["schema"] == "ace3-v3-retained-coordinate-decoding-v1"
                  and boundary["decoder_rtl_invocations"] == 0,
                  "invalid retained decoding boundary")
    freeze = document(boundary["freeze"])
    original = document(freeze["original_manifest"])
    document(freeze["original_plan"])
    local.require(freeze["schema"] == "ace3-v3-retained-recovery-plan-v1"
                  and freeze["scope"] == {"layers": [5], "position": 0, "history": [9707]}
                  and original["node"] == manifest["node"] == [5, 0],
                  "retained recovery scope mismatch")
    local.require({k: v for k, v in manifest.items() if k not in ("actual_operands", "derivation")}
                  == {k: v for k, v in original.items() if k != "actual_operands"},
                  "retained recovery changed original execution identity")
    artifact(original["actual_operands"])
    transaction = document(original["transaction"])
    local.require(boundary["original_manifest"] == freeze["original_manifest"]
                  and boundary["actual_operands"] == manifest["actual_operands"]
                  and boundary["raw_trace"] == transaction["raw"]["trace"]
                  and boundary["final"] == transaction["output"]
                  and boundary["state"] == transaction["output_state"]
                  and boundary["decoder_sources"] == freeze["decoder_sources"],
                  "retained decoding producer boundary mismatch")
    expected_paths = {str(Path(__file__).resolve()),
                      str(Path(__file__).resolve().with_name("host_capture_v3.py"))}
    sources = boundary["decoder_sources"]
    local.require(len(sources) == len(expected_paths)
                  and {r["path"] for r in sources} == expected_paths,
                  "retained decoder source coverage mismatch")
    for rec in sources:
        artifact(rec)
    for rec in freeze["bindings"]:
        artifact(rec)


def p0_rope_bytes():
    return "".join(f"0000{pair:02x}3c000000\n" for pair in range(32)).encode("ascii")


def source_map(records):
    result = {}
    for record in records:
        name = Path(record["path"]).name
        local.require(name not in result, "duplicate runtime source")
        artifact(record)
        result[name] = record
    return result


def _trusted_context(contract, layer):
    """Follow only the previously pinned retained/reference graph."""
    local.require(type(layer) is int and 5 <= layer <= 23, "unsupported runtime layer")
    if layer >= 9:
        from ace3.model.candidates.remaining_layers_v3 import reference_context
        return reference_context(contract, layer)
    manifest_path = RETAINED / "output_manifest.json"
    data = manifest_path.read_bytes()
    local.require(hashlib.sha256(data).hexdigest() ==
                  contract["trusted_retained_manifest_sha256"], "retained authority drift")
    members = json.loads(data)["artifacts"]
    result = document(unique(members, lambda r: r["sha256"] ==
                             contract["trusted_retained_result_sha256"], "retained result"))
    frozen = document(result["freeze"])
    bindings = frozen["input_bindings"]
    source = document(unique(bindings, lambda r: r["sha256"] == SOURCE_FREEZE_SHA256,
                             "accepted single-round source freeze"))
    independent = document(unique(bindings, lambda r: r["sha256"] ==
                                  contract["trusted_independent_freeze_sha256"],
                                  "original independent reference freeze"))
    specification = document(independent["reference_source"])
    local.require(specification["checkpoint"] == independent["checkpoint"]
                  and specification["checkpoint_tensors"] == independent["checkpoint_tensors"],
                  "independent canonical metadata disagreement")
    canonical = {r["name"]: r for r in specification["checkpoint_tensors"]}
    local.require(len(canonical) == len(specification["checkpoint_tensors"]),
                  "duplicate canonical reference metadata")
    reference = unique(independent["reference_transactions"],
                       lambda r: r["layer"] == layer and r["position"] == 0,
                       "original FP16 reference transaction")
    trajectory = {int(s): hex_words(artifact(r)) for s, r in reference["stages"].items()}
    binary64_record = unique(
        bindings, lambda r: r["path"].endswith(f"/layer{layer:02d}_position000_reference.npy"),
        "original binary64 reference")
    binary64 = np.load(io.BytesIO(artifact(binary64_record)), allow_pickle=False)
    return {
        "source": source, "root_hidden": frozen["input_hidden"],
        "fp16_final": reference["stages"]["18"],
        "canonical": {name: canonical[name] for name in local.tensor_shapes(layer)},
        "trajectory": trajectory, "binary64": binary64,
        "trace_coordinates": [r[:3] for r in trace_rows(artifact(TRACE_LAYOUT))],
        "reference_bindings": {"independent": independent["reference_source"],
                               "binary64": binary64_record},
    }


def saved_state(data, header, slow, symbols, layer, arrays):
    """Decode the existing Verilator save ABI as data, without restore/eval."""
    declarations = {}
    for kind, name, dimensions in re.findall(
            r"\b(CData|SData|IData|QData|WData)/\*[^*]+\*/\s+(\w+)((?:\[\d+\])*)\s*;",
            header):
        declarations[name] = ({"CData": 1, "SData": 2, "IData": 4,
                               "QData": 8, "WData": 4}[kind],
                              [int(n) for n in re.findall(r"\d+", dimensions)])
    for width, name in re.findall(r"\bVL_(?:IN|OUT)(8|16|64)?\((\w+),\d+,\d+\);", header):
        declarations[name] = (int(width or 32) // 8, [])
    marker = "::__Vserialize(VerilatedSerialize& os) {"
    local.require(slow.count(marker) == 1 and symbols.count(marker) == 1,
                  "unsupported generated saved-state ABI")
    body = slow.split(marker)[1].split("\n}\n", 1)[0]
    check = re.search(r"vluint64_t __Vcheckval = (0x[0-9a-f]+)ULL;", body)
    local.require(check is not None, "missing saved-state model checksum")
    checksum = int(check[1], 16).to_bytes(8, "little")
    local.require(data.startswith(b"verilatorsave01\n") and data.endswith(b"vltsaved")
                  and data.count(checksum) == 1, "saved-state header/checksum mismatch")
    offset = data.index(checksum)
    declarations["__Vcheckval"] = (8, [])
    prefix = TOP + "__DOT__cache__DOT__"
    wanted = {"busy_o", "done_valid_o", "phase_o", "layer_index_o",
              prefix + "k_mem", prefix + "v_mem", prefix + "valid_mem"}
    loops, seen, fields = [], set(), {}
    for statement in body.splitlines():
        statement = statement.strip()
        if not statement or statement.startswith("vluint64_t __Vcheckval"):
            continue
        loop = re.fullmatch(r"\{ int __Vi(\d+)=0; for \(; __Vi\1<(\d+); \+\+__Vi\1\) \{",
                            statement)
        if loop:
            local.require(int(loop[1]) == len(loops), "unsupported saved-state loop nesting")
            loops.append(int(loop[2]))
            continue
        if statement == "}}":
            local.require(bool(loops), "unmatched saved-state loop")
            loops.pop()
            continue
        if statement == "__VlSymsp->__Vserialize(os);":
            symbol_body = symbols.split(marker)[1].split("\n}", 1)[0]
            local.require(not loops and re.findall(r"os<<(\w+);", symbol_body) == ["__Vm_didInit"]
                          and data[offset:offset + 1] == b"\x01",
                          "uninitialized/incompatible saved symbol state")
            offset += 1
            continue
        field = re.fullmatch(r"os<<(\w+)((?:\[__Vi\d+\])*)\s*;", statement)
        local.require(field is not None, f"unsupported saved-state statement: {statement}")
        name = field[1]
        local.require(name in declarations and name not in seen,
                      "unknown/duplicate saved-state field")
        seen.add(name)
        width, dimensions = declarations[name]
        local.require(dimensions == loops and field[2].count("[") == len(loops),
                      "saved-state layout geometry mismatch")
        count = math.prod(dimensions)
        end = offset + count * width
        local.require(end <= len(data) - 8, "truncated saved state")
        if name in wanted:
            fields[name] = list(struct.unpack(
                f"<{count}{ {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[width] }", data[offset:end]))
        offset = end
    local.require(not loops and offset + 8 == len(data) and set(fields) == wanted,
                  "saved-state extent/required field mismatch")
    local.require(all(fields[k] == [0] for k in ("busy_o", "done_valid_o", "phase_o"))
                  and fields["layer_index_o"] == [layer], "saved state is not own-layer idle")
    local.require(fields[prefix + "valid_mem"] == [1] * 128 + [0] * 32640,
                  "saved state is not exactly own slot0/P0 KV")
    for kind in ("k", "v"):
        local.require(len(fields[prefix + kind + "_mem"]) == 32768
                      and fields[prefix + kind + "_mem"][:128] ==
                      arrays["output_cache_" + kind][0].tolist(),
                      "saved cache differs from actual output KV")
    return {"status": "PASS", "idle": True, "layer": layer, "valid_entries": 128,
            "state_restore_or_eval_performed": False}


def validate_runtime(*, receipt, trusted_manifest_sha256, contract, arrays, tensors,
                     canonical_records, expected_hidden, trajectory, reference_binary64,
                     layer, position, history, policy):
    local.require(isinstance(trusted_manifest_sha256, str)
                  and re.fullmatch("[0-9a-f]{64}", trusted_manifest_sha256) is not None,
                  "missing independent runtime manifest digest")
    local.require(isinstance(receipt, dict)
                  and receipt.get("sha256") == trusted_manifest_sha256,
                  "runtime manifest differs from independent receipt")
    manifest = document(receipt)
    validate_derivation(manifest)
    local.require(type(layer) is int and 5 <= layer <= 23 and type(position) is int
                  and position == 0 and history == [9707], "unsupported runtime layer/history")
    local.require(manifest["schema"] == SCHEMA and manifest["policy_id"] == policy
                  and policy == contract["policy_id"] and manifest["evidence_kind"] == "actual_rtl"
                  and manifest["node"] == [layer, 0] and manifest["history"] == history,
                  "runtime identity/policy mismatch")
    if layer >= 9:
        local.require(manifest.get("reference_extension") == contract.get("reference_extension")
                      and manifest.get("reference_extension") is not None,
                      "missing/mismatched original reference extension")
        from ace3.model.candidates.remaining_decoder_gate_policy_v3 import validate_binding
        validate_binding(contract.get("evaluator_extension"))
        local.require(manifest.get("evaluator_extension") == contract["evaluator_extension"],
                      "runtime remaining evaluator extension mismatch")
    context = _trusted_context(contract, layer)
    source = context["source"]
    local.require(manifest["public_contract"] == source["public_contract"]
                  and manifest["parameters"] == {"LAYER_INDEX": layer, "ACCURATE_SILU": 1}
                  and manifest["semantics"] == source["semantics"],
                  "runtime public interface/parameters/arithmetic profile mismatch")
    actual_sources = source_map(manifest["source_closure"])
    accepted_sources = source_map(source["source_closure"])
    capture_only = harness.compatible_sources(actual_sources, accepted_sources, artifact)
    top = artifact(actual_sources[TOP + ".sv"]).decode("ascii")
    header = re.search(r"\bmodule\s+" + TOP + r"\s*#\(.*?\);", top, re.S)
    local.require(header is not None and header[0] == source["public_contract"],
                  "compiled public module header mismatch")
    compile_receipt = document(manifest["compile"])
    local.require(type(compile_receipt["returncode"]) is int
                  and compile_receipt["returncode"] == 0
                  and compile_receipt["natural_exit"] is True, "no completed runtime compilation")
    artifact(compile_receipt["tool"])
    artifact(compile_receipt["log"])
    artifact(compile_receipt["binary"])
    expected_command = []
    original_command = source["command"]
    for index, argument in enumerate(original_command):
        if index == 0:
            expected_command.append(compile_receipt["tool"]["path"])
        elif index and original_command[index - 1] == "--Mdir":
            expected_command.append(compile_receipt["object_dir"])
        elif argument == "-GLAYER_INDEX=4":
            expected_command.append(f"-GLAYER_INDEX={layer}")
        elif argument in [r["path"] for r in accepted_sources.values()]:
            expected_command.append(actual_sources[Path(argument).name]["path"])
        else:
            expected_command.append(argument)
    local.require(compile_receipt["argv"] == expected_command
                  and compile_receipt["binary"]["path"] ==
                  str(Path(compile_receipt["object_dir"]) / ("V" + TOP)),
                  "runtime compiler command/binary mismatch")
    generated = compile_receipt["generated"]
    local.require(set(generated) == {"header", "slow", "symbols"},
                  "missing generated state ABI receipts")
    for key, suffix in (("header", ".h"), ("slow", "__Slow.cpp"), ("symbols", "__Syms.cpp")):
        local.require(generated[key]["path"] ==
                      str(Path(compile_receipt["object_dir"]) / ("V" + TOP + suffix)),
                      "generated ABI belongs to another runtime")
    launch = document(manifest["launch"])
    execution = document(manifest["execution"])
    prepared = document(manifest["prepared"])
    transaction = document(manifest["transaction"])
    local.require(launch["binary"] == prepared["binary"] == execution["binary"] ==
                  compile_receipt["binary"] and launch["prepared"] == manifest["prepared"]
                  and execution["launch"] == manifest["launch"]
                  and execution["transaction"] == manifest["transaction"]
                  and launch["node"] == [layer, 0] and launch["history"] == history
                  and launch["parameters"] == manifest["parameters"],
                  "runtime binary/launch/transaction binding mismatch")
    local.require(isinstance(launch["argv"], list) and len(launch["argv"]) > 0
                  and all(isinstance(a, str) for a in launch["argv"])
                  and launch["argv"][0] == prepared["binary"]["path"]
                  and execution["argv"] == launch["argv"]
                  and type(execution["returncode"]) is int and execution["returncode"] == 0
                  and execution["natural_exit"] is True and execution["executed"] is True,
                  "runtime evaluator no-execution or unsuccessful execution")
    local.require((launch["argv"].count(harness.CAPTURE_FLAG) == 1) if capture_only
                  else harness.CAPTURE_FLAG not in launch["argv"],
                  "runtime capture mode/source mismatch")
    artifact(execution["log"])
    local.require(launch["input_state"] is None and prepared["input_state"] is None
                  and launch["input_hidden"] == prepared["input_hidden"]
                  and launch["cache_slot"] == 0
                  and transaction["layer_index"] == layer and transaction["position"] == 0,
                  "runtime input/position/empty P0 state mismatch")
    parent = manifest["parent"]
    if layer == 5:
        local.require(parent == context["root_hidden"], "runtime hidden is not accepted actual L4")
        parent_output = parent
    else:
        parent_result = document(parent)
        local.require(parent_result["status"] == parent_result["numerical_status"] == "PASS"
                      and parent_result["evidence_kind"] == "actual_rtl_numerical_evaluation"
                      and parent_result["policy_id"] == policy
                      and parent_result["runtime_admission"]["node"] == [layer - 1, 0]
                      and parent_result["runtime_admission"]["history"] == history
                      and parent_result["runtime_admission"]["source_sha256"] ==
                      {n: r["sha256"] for n, r in actual_sources.items()},
                      "unadmitted/spliced runtime hidden parent")
        parent_output = parent_result["runtime_admission"]["output"]
    local.require(prepared["input_hidden"] == parent_output, "runtime producer/consumer parent mismatch")
    hidden = hex_words(artifact(parent_output), indexed=True)
    local.require(np.array_equal(hidden, expected_hidden), "runtime expected hidden parent mismatch")
    with np.load(io.BytesIO(artifact(manifest["actual_operands"])), allow_pickle=False) as archive:
        local.require(set(archive.files) == set(arrays), "runtime operand archive coverage mismatch")
        for name in archive.files:
            value = archive[name]
            local.require(value.dtype == arrays[name].dtype and value.shape == arrays[name].shape
                          and np.array_equal(value, arrays[name]), f"runtime operand mismatch: {name}")
    required = {"input_hidden", "input_cache_k", "input_cache_v",
                "output_cache_k", "output_cache_v", *(f"stage{s:02d}" for s in range(19))}
    local.require(set(arrays) == required, "unexpected/missing runtime operand arrays")
    lineage = local.validate_lineage(arrays, hidden, position=position, history=history)
    raw = transaction["raw"]
    rows = trace_rows(artifact(raw["trace"]))
    local.require([r[:3] for r in rows] == context["trace_coordinates"]
                  and raw["trace_count"] == len(rows) == sum(local.SIZES.values())
                  and raw["done_count"] == 1 and raw["final_count"] == 896,
                  "actual trace coverage/layout/causal order mismatch")
    decoded = canonical_stages(rows, context["trace_coordinates"])
    for stage in range(19):
        words = decoded[f"stage{stage:02d}"]
        local.require(np.array_equal(words, arrays[f"stage{stage:02d}"]),
                      f"actual trace/operand identity mismatch: S{stage}")
    final = hex_words(artifact(transaction["output"]), indexed=True)
    local.require(np.array_equal(final, arrays["stage18"]), "actual final/trace mismatch")
    artifact(raw["terminal"])
    artifact(transaction["simulation_log"])
    local.require(transaction["input"]["sha256"] ==
                  hashlib.sha256(hidden.tobytes()).hexdigest(), "actual transaction input mismatch")
    vectors = prepared["vectors"]
    local.require(vectors == transaction["vectors"], "prepared/executed vector identity mismatch")
    local.require(np.array_equal(hex_words(artifact(vectors["input"]), indexed=True), hidden),
                  "runtime serialized input mismatch")
    local.require(canonical_records == context["canonical"], "untrusted canonical reference metadata")
    local.authenticate_tensors(tensors, context["canonical"], layer)
    selected = {r["checkpoint_tensor"]["name"]: r for r in vectors["tensors"]}
    local.require(len(selected) == len(vectors["tensors"]) and set(selected) == set(tensors),
                  "runtime canonical tensor selection mismatch")
    for name, value in tensors.items():
        record = selected[name]
        metadata = record["checkpoint_tensor"]
        canonical = context["canonical"][name]
        local.require(metadata["shape"] == canonical["shape"]
                      and metadata["dtype"] == {"float16": "F16", "int32": "I32"}[canonical["dtype"]]
                      and metadata["sha256"] == canonical["sha256"],
                      "runtime canonical weight/control metadata mismatch")
        width = value.dtype.itemsize * 2
        serialized = artifact(record["serialized"]).decode("ascii").splitlines()
        local.require(all(re.fullmatch(f"[0-9a-fA-F]{{{width}}}", r) for r in serialized),
                      "runtime tensor serialization ABI mismatch")
        words = np.asarray([int(r, 16) for r in serialized], dtype=f"<u{value.dtype.itemsize}")
        local.require(np.array_equal(words, value.view(words.dtype).reshape(-1)),
                      "runtime canonical tensor data mismatch")
    # P0 half-split RoPE is exactly cos=1, sin=0 for every pair. Do not accept
    # candidate-selected control bytes just because two receipts agree.
    rope = artifact(vectors["rope_coefficients"])
    local.require(rope == artifact(manifest["independent_p0_rope"]) == p0_rope_bytes()
                  and manifest["independent_p0_rope"]["sha256"] ==
                  launch["independent_p0_rope_sha256"], "runtime canonical RoPE control mismatch")
    local.require(set(trajectory) == set(context["trajectory"])
                  and all(trajectory[s].dtype == context["trajectory"][s].dtype
                          and np.array_equal(trajectory[s], context["trajectory"][s]) for s in trajectory),
                  "runtime original FP16 trajectory reference mismatch")
    original64 = context["binary64"]
    local.require(isinstance(reference_binary64, np.ndarray)
                  and reference_binary64.dtype == original64.dtype == np.dtype("<f8")
                  and reference_binary64.shape == original64.shape == (896,)
                  and reference_binary64.tobytes() == original64.tobytes(),
                  "runtime original global binary64 reference mismatch")
    local.require(execution["output_state"] == transaction["output_state"],
                  "saved state is spliced from another runtime")
    state = saved_state(
        artifact(transaction["output_state"]), artifact(generated["header"]).decode("ascii"),
        artifact(generated["slow"]).decode("ascii"), artifact(generated["symbols"]).decode("ascii"),
        layer, arrays)
    return {"status": "PASS", "node": [layer, 0], "history": history, "manifest": receipt,
            "source_sha256": {n: r["sha256"] for n, r in actual_sources.items()},
            "output": transaction["output"], "output_state": transaction["output_state"],
            "parent": parent, "lineage": lineage, "saved_state": state,
            "reference_bindings": context["reference_bindings"]}
