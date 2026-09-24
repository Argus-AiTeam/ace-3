#!/usr/bin/env python3
"""Execute a reviewed single-round residual cone with independent ordered gates."""
import argparse
import ast
import bisect
import csv
import encodings.cp437
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
PREFLIGHT = BUILD / "single_round_residual_rtl_f0c34aee3654_attempt001"
SOFTWARE = BUILD / "single_round_residual_binary64_excess_55c2b0a6b67a_attempt001"
B = BUILD / "layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010"
REVIEW = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs/f0c34aee3654/round-0001.json")
TOP = "ace3_decoder_layer0_token_engine"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module unavailable: {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def retained_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def retained_write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def retained_require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def retained_modules():
    return sorted({str(Path(m.__file__).resolve()) for m in tuple(sys.modules.values())
                   if getattr(m, "__file__", None) and Path(m.__file__).is_file()})


def prepare_retained_verification(out, attempt):
    """Freeze a new stdlib-only reader; never import the historical replay helper."""
    attempt = attempt.resolve()
    retained_require(attempt.parent == BUILD and attempt.name.startswith(
        "single_round_residual_rtl_execution_"), "retained attempt outside execution namespace")
    retained_require(attempt != out, "verification must not replace its retained attempt")
    out.mkdir(exist_ok=False)
    (out / "source").mkdir()
    source = out / "source/runner.py"
    shutil.copyfile(__file__, source)
    documents = ["output_manifest.json", "freeze.json", "result.json",
                 "layer04/prepared.json", "layer04/transaction.json",
                 "layer04/result.json", "rtl_binary64_v1.json"]
    historical = {}

    def collect(value):
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                rec = {key: value[key] for key in ("path", "bytes", "sha256")}
                old = historical.setdefault(rec["path"], rec)
                retained_require(old == rec, f"conflicting historical signature: {rec['path']}")
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for name in documents:
        collect(json.loads((attempt / name).read_text()))
    launch = BUILD / "single_round_residual_rtl_execution_fb06353c5f53_launch_attempt001"
    receipt_path = ROOT / ".argus_subagents/fb06353c5f53-l4-attempt001.json"
    receipt = json.loads(receipt_path.read_text())
    external = [receipt_path, launch / "submit.receipt.json",
                launch / "terminal_verification.command", launch / "terminal_verification.log",
                ROOT / receipt["stdout_log"], ROOT / receipt["stderr_log"],
                ROOT / (".argus_subagents/fb06353c5f53-l4-attempt001_logs/exit_code."
                        + receipt["run_id"])]
    freeze = {
        "schema": "ace3_single_round_retained_l4_verification_v1",
        "attempt": str(attempt), "validator": retained_record(source),
        "interpreter": retained_record(sys.executable), "python_version": sys.version,
        "executable_helpers": [],
        "runtime_modules": [retained_record(p) for p in retained_modules()
                            if Path(p) != Path(__file__).resolve()],
        "historical_bindings": list(historical.values()),
        "retained_tree": [retained_record(p) for p in sorted(attempt.rglob("*")) if p.is_file()],
        "external_evidence": [retained_record(p) for p in external],
        "evaluation_order": ["raw_trace_final_and_state", "19_unchanged_FP16_gates",
                             "eligible_original_binary64_v1"],
        "scope": {"layer": 4, "position": 0, "history": [9707], "rtl_replay": False},
        "historical_helper_boundary": (
            "No historical identity is inferred for unbound replay.py. The new validator "
            "imports no project helpers and independently reads retained data."),
        "normal_host_review": "REQUIRED",
    }
    retained_write(out / "freeze.json", freeze)
    sums = [freeze["validator"], freeze["interpreter"], retained_record(out / "freeze.json")]
    with (out / "preexecution.sha256").open("x") as stream:
        for rec in sums:
            stream.write(f"{rec['sha256']}  {rec['path']}\n")
    argv = [sys.executable, "-I", "-S", "-B", str(source), "--out", str(out),
            "--verify-retained"]
    command = (
        "set -euC\ncd " + shlex.quote(str(ROOT)) + "\n"
        "sha256sum --check " + shlex.quote(str(out / "preexecution.sha256"))
        + " > " + shlex.quote(str(out / "preexecution.log")) + " 2>&1\n"
        + "set +e\n" + shlex.join(argv) + " > "
        + shlex.quote(str(out / "verification.log")) + " 2>&1\n"
        + "status=$?\nset -e\nprintf '%s\\n' \"$status\" > "
        + shlex.quote(str(out / "verification.exit_code")) + "\n"
        + "sha256sum " + shlex.join([str(out / name) for name in (
            "verification.log", "verification.exit_code", "output_manifest.json")])
        + " > " + shlex.quote(str(out / "completion.sha256")) + "\nexit \"$status\"\n")
    with (out / "verify.command").open("x") as stream:
        stream.write(command)
    with (out / "verification.invocation").open("x") as stream:
        stream.write("bash " + shlex.quote(str(out / "verify.command")) + "\n")
    print(f"Prepared retained-only verification: {out}", flush=True)


def retained_npy(data):
    retained_require(data[:8] in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00"),
                     "unsupported NPY format")
    width = 2 if data[6] == 1 else 4
    length = int.from_bytes(data[8:8 + width], "little")
    end = 8 + width + length
    header = ast.literal_eval(data[8 + width:end].decode("ascii"))
    retained_require(set(header) == {"descr", "fortran_order", "shape"}
                     and header["fortran_order"] is False
                     and header["descr"] in ("<u2", "<f8"), "unsupported NPY ABI")
    shape = header["shape"]
    retained_require(isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape),
                     "invalid NPY geometry")
    count = math.prod(shape)
    code = "H" if header["descr"] == "<u2" else "d"
    retained_require(len(data) - end == count * struct.calcsize(code), "NPY payload length")
    return header["descr"], shape, list(struct.unpack(f"<{count}{code}", data[end:]))


def retained_npz(path, names):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        retained_require(len(members) == len(set(members)), "duplicate NPZ members")
        return {name: retained_npy(archive.read(name + ".npy")) for name in names}


def retained_units(bits):
    retained_require(type(bits) is int and 0 <= bits <= 65535 and bits & 0x7c00 != 0x7c00,
                     f"invalid/nonfinite FP16 operand: {bits}")
    exponent, fraction = (bits >> 10) & 31, bits & 1023
    magnitude = fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)
    return -magnitude if bits & 0x8000 else magnitude


def retained_state(attempt, stages):
    """Decode the frozen generated save layout as data, without restore/eval or compilation."""
    obj = attempt / "layer04/obj"
    base = obj / f"V{TOP}"
    header = base.with_suffix(".h").read_text()
    generated = Path(str(base) + "__Slow.cpp").read_text()
    body = generated.split("::__Vserialize(VerilatedSerialize& os) {", 1)[1].split(
        "\n}\n", 1)[0]
    declarations = {}
    for kind, name, dimensions in re.findall(
            r"\b(CData|SData|IData|QData|WData)/\*[^*]+\*/\s+(\w+)((?:\[\d+\])*)\s*;",
            header):
        declarations[name] = ({"CData": 1, "SData": 2, "IData": 4,
                               "QData": 8, "WData": 4}[kind],
                              [int(n) for n in re.findall(r"\d+", dimensions)])
    for width, name in re.findall(r"\bVL_(?:IN|OUT)(8|16|64)?\((\w+),\d+,\d+\);", header):
        declarations[name] = (int(width or 32) // 8, [])
    check = re.search(r"vluint64_t __Vcheckval = (0x[0-9a-f]+)ULL;", body)
    retained_require(check is not None, "missing model serialization checksum")
    checksum = int(check[1], 16).to_bytes(8, "little")
    data = (attempt / "layer04/candidate.state").read_bytes()
    retained_require(data.startswith(b"verilatorsave01\n") and data.endswith(b"vltsaved"),
                     "invalid Verilator state header/trailer")
    retained_require(data.count(checksum) == 1, "ambiguous saved model checksum")
    offset = data.index(checksum)
    header_bytes = offset
    declarations["__Vcheckval"] = (8, [])
    loops, layout, fields = [], {}, {}
    prefix = TOP + "__DOT__"
    wanted = {"busy_o", "done_valid_o", "phase_o", "layer_index_o",
              prefix + "cache__DOT__k_mem", prefix + "cache__DOT__v_mem",
              prefix + "cache__DOT__valid_mem"}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("vluint64_t __Vcheckval"):
            continue
        loop = re.fullmatch(r"\{ int __Vi(\d+)=0; for \(; __Vi\1<(\d+); \+\+__Vi\1\) \{", line)
        if loop:
            retained_require(int(loop[1]) == len(loops), "unsupported serializer loop nesting")
            loops.append(int(loop[2]))
            continue
        if line == "}}":
            retained_require(bool(loops), "unmatched serializer loop")
            loops.pop()
            continue
        if line == "__VlSymsp->__Vserialize(os);":
            retained_require(not loops, "symbol serializer nested in loop")
            symbols = Path(str(base) + "__Syms.cpp").read_text().split(
                "::__Vserialize(VerilatedSerialize& os) {", 1)[1].split("\n}", 1)[0]
            retained_require(re.findall(r"os<<(\w+);", symbols) == ["__Vm_didInit"]
                             and data[offset:offset + 1] == b"\x01",
                             "unexpected symbol state ABI or uninitialized model")
            offset += 1
            continue
        field = re.fullmatch(r"os<<(\w+)((?:\[__Vi\d+\])*)\s*;", line)
        retained_require(field is not None, f"unsupported serializer statement: {line}")
        name = field[1]
        retained_require(name in declarations and name not in layout,
                         f"unknown/duplicate serializer field: {name}")
        size, dimensions = declarations[name]
        retained_require(dimensions == loops and len(re.findall(r"\[", field[2])) == len(loops),
                         f"serializer/declaration geometry mismatch: {name}")
        count = math.prod(dimensions)
        length = count * size
        layout[name] = {"offset": offset, "element_bytes": size, "elements": count}
        if name in wanted:
            fields[name] = list(struct.unpack(
                f"<{count}{ {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[size] }",
                data[offset:offset + length]))
        offset += length
    retained_require(not loops and offset + 8 == len(data), "saved-state extent mismatch")
    retained_require(set(fields) == wanted, "saved-state required fields missing")
    retained_require(all(fields[name] == [0] for name in ("busy_o", "done_valid_o", "phase_o"))
                     and fields["layer_index_o"] == [4], "saved L4 model is not idle")
    valid = fields[prefix + "cache__DOT__valid_mem"]
    retained_require(valid == [1] * 128 + [0] * (32768 - 128),
                     "saved K/V valid map is not exactly slot0/position0")
    for name, stage in (("k", 6), ("v", 7)):
        retained_require(fields[prefix + f"cache__DOT__{name}_mem"][:128] == stages[stage],
                         f"saved {name.upper()} does not match actual S{stage}")
    return {"idle": True, "valid_entries": 128, "invalid_entries": 32640,
            "actual_s6_s7_exact": True, "serialized_fields": len(layout),
            "header_bytes": header_bytes, "bytes": len(data),
            "layout": {name: layout[name] for name in sorted(wanted)},
            "state_restore_or_eval_performed": False,
            "layout_source_boundary": "retained generated C++ read as data; no new executable"}


def verify_retained(out):
    freeze = json.loads((out / "freeze.json").read_text())
    attempt = Path(freeze["attempt"])
    layer = attempt / "layer04"
    authenticated, reports = {}, []
    phase, witness = "source_and_input_authentication", None
    started = time.monotonic()

    def authenticate(rec):
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        if rec["path"] not in authenticated:
            actual = retained_record(rec["path"])
            retained_require(actual == signature, f"artifact binding mismatch: {rec['path']}")
            authenticated[rec["path"]] = actual
        retained_require(authenticated[rec["path"]] == signature,
                         f"conflicting artifact binding: {rec['path']}")

    def load(path):
        retained_require(str(path) in authenticated, f"unbound JSON input: {path}")
        return json.loads(path.read_text())

    def array_digest(bits):
        return hashlib.sha256(struct.pack(f"<{len(bits)}H", *bits)).hexdigest()

    try:
        retained_require(freeze["schema"] == "ace3_single_round_retained_l4_verification_v1"
                         and freeze["executable_helpers"] == [], "verification contract mismatch")
        retained_require(freeze["validator"]["path"] == str(Path(__file__).resolve())
                         and sys.version == freeze["python_version"], "validator/runtime identity")
        for rec in [freeze["validator"], freeze["interpreter"], *freeze["runtime_modules"],
                    *freeze["retained_tree"], *freeze["external_evidence"],
                    *freeze["historical_bindings"]]:
            authenticate(rec)
        retained_require(set(retained_modules()) <= set(authenticated),
                         "unbound executable Python module")
        original = load(attempt / "freeze.json")
        result = load(attempt / "result.json")
        tx = load(layer / "transaction.json")
        prepared = load(layer / "prepared.json")
        layer_result = load(layer / "result.json")
        retained_require(retained_record(attempt / "freeze.json")["sha256"] ==
                         "faa8c612c6cd2f94a5c55e591ca48053ac1921f1e3506537cf0f0955e0b7b9bb",
                         "attempt001 freeze differs from retained reviewed-round evidence")
        retained_require(original["semantics"] ==
                         "RNE16(H + O + D), exact three-term sum, only one rounding"
                         and original["token_history"] == [9707], "arithmetic/history mismatch")
        review = load(Path(original["review"]["path"]))
        retained_require((review["kind"], review["producer_role"], review["mission_id"],
                          review["review"]["status"]) ==
                         ("round_reviewed_handoff", "reviewer", "34220ce45e75", "done"),
                         "accepted parent review missing")
        parent = load(Path(original["parent_result"]["path"]))
        parent_rows = [r for r in parent["transactions"] if Path(r["path"]).parent.name == "layer03"]
        retained_require(len(parent_rows) == 1, "ambiguous actual L3 parent")
        parent_tx = load(Path(parent_rows[0]["path"]))["transaction"]
        retained_require(parent_tx["output"] == original["input_hidden"] == prepared["input_hidden"]
                         and parent_tx["output_state"] == original["parent_state_not_imported"],
                         "actual L3 lineage mismatch")
        retained_require(original["input_state"] is None and prepared["input_state"] is None
                         and (tx["layer_index"], tx["position"]) == (4, 0),
                         "L4 must own empty input K/V")
        command = load(layer / "runtime/simulation.command.json")
        argv = command["argv"]
        retained_require("--state-in" not in argv and argv[0] == prepared["binary"]["path"]
                         and argv[argv.index("--state-out") + 1] == tx["output_state"]["path"]
                         and argv[argv.index("--transaction-position") + 1] == "0"
                         and argv[argv.index("--layer-index") + 1] == "4",
                         "runtime binary/state/position contract mismatch")
        receipt_path = ROOT / ".argus_subagents/fb06353c5f53-l4-attempt001.json"
        receipt = load(receipt_path)
        launch = BUILD / "single_round_residual_rtl_execution_fb06353c5f53_launch_attempt001"
        submit = load(launch / "submit.receipt.json")
        retained_require(receipt["state"] == "done" and receipt["exit_code"] == 0
                         and receipt["task_id"] == submit["task_id"] ==
                         "fb06353c5f53-l4-attempt001"
                         and receipt["run_id"] == submit["run_id"], "natural durable completion")
        exit_path = ROOT / (".argus_subagents/fb06353c5f53-l4-attempt001_logs/exit_code."
                            + receipt["run_id"])
        retained_require(exit_path.read_bytes() == b"0\n", "durable exit is not zero")
        for path in (attempt / "public_contract_compile.execution.json",
                     layer / "compile.execution.json", layer / "runtime/simulation.execution.json"):
            retained_require(load(path)["exit_code"] == 0, f"retained execution failure: {path}")
        terminal_items = [s.split("=", 1) for s in Path(tx["raw"]["terminal"]["path"]).read_text().split()]
        terminal = dict(terminal_items)
        retained_require(len(terminal) == len(terminal_items) and terminal == {
            "schema": "ace3_decoder_token_transaction_v1", "layer_index": "4", "position": "0",
            "natural_terminal": "1", "exit_code": "0", "trace_count": "23324",
            "final_count": "896", "done_count": "1"}, "natural terminal mismatch")
        phase = "raw_trace_final_and_state"
        counts = [896, 896, 128, 128, 896, 128, 128, 128, 14, 14,
                  896, 896, 896, 896, 4864, 4864, 4864, 896, 896]
        indexed_stages = [{} for _ in counts]
        for number, line in enumerate(Path(tx["raw"]["trace"]["path"]).read_text().splitlines()):
            retained_require(re.fullmatch("[0-9a-f]{16}", line) is not None,
                             f"malformed raw trace at line {number + 1}")
            token, position, stage, index, bits = (int(line[a:b], 16) for a, b in
                                                  ((0, 2), (2, 6), (6, 8), (8, 12), (12, 16)))
            retained_require(token == position == 0 and stage < 19, "raw token/position/stage")
            # RoPE emits low/high pairs, not ascending coordinates. At P0 the
            # score/probability trace index is zero for each successive head.
            if stage in (8, 9):
                retained_require(index == 0, "P0 attention trace context index")
                index = len(indexed_stages[stage])
            retained_require(index < counts[stage] and index not in indexed_stages[stage],
                             f"raw duplicate/out-of-range coordinate: stage {stage}, index {index}")
            indexed_stages[stage][index] = bits
        retained_require([len(s) for s in indexed_stages] == counts, "raw trace stage counts")
        stages = [[values[i] for i in range(count)] for values, count in
                  zip(indexed_stages, counts, strict=True)]

        def hidden(path):
            values = []
            for index, line in enumerate(path.read_text().splitlines()):
                retained_require(re.fullmatch("[0-9a-f]{10}", line) is not None
                                 and int(line[:2], 16) == 0 and int(line[2:6], 16) == index,
                                 f"malformed/duplicate hidden record: {path}:{index + 1}")
                values.append(int(line[6:], 16))
            retained_require(len(values) == 896, "hidden vector width")
            return values

        final = hidden(Path(tx["output"]["path"]))
        incoming = hidden(Path(original["input_hidden"]["path"]))
        retained_require(final == stages[18] and array_digest(final) ==
                         tx["output"]["semantic_sha256"], "raw final/trace/semantic hash mismatch")
        retained_require(incoming == hidden(Path(tx["vectors"]["input"]["path"]))
                         and array_digest(incoming) == tx["input"]["sha256"],
                         "actual L3 hidden was not the L4 vector input")
        names = ["input_hidden", *[f"stage{s:02d}" for s in range(19)]]
        archived = retained_npz(layer / "actual_stages.npz", names)
        software = retained_npz(attempt / "software_stages.npz",
                                [*names, "input_cache_k", "input_cache_v"])
        retained_require(archived["input_hidden"][2] == software["input_hidden"][2] == incoming,
                         "archived actual/software hidden mismatch")
        retained_require(all(software[name] == ("<u2", (0, 128), [])
                             for name in ("input_cache_k", "input_cache_v")),
                         "software input K/V is not empty")
        for stage in range(19):
            retained_require(archived[f"stage{stage:02d}"][0] ==
                             software[f"stage{stage:02d}"][0] == "<u2"
                             and archived[f"stage{stage:02d}"][2] ==
                             software[f"stage{stage:02d}"][2] == stages[stage],
                             f"raw/archived/same-input stage mismatch: S{stage}")
        for name, stage in (("k", 6), ("v", 7)):
            retained_require(layer_result["actual_kv"][name] == stages[stage],
                             f"retained actual K/V mismatch: {name}")
        state = retained_state(attempt, stages)
        retained_write(out / "state_verification.json", state)
        phase = "unchanged_fp16_interstage"
        graph_paths = [Path(p) for p in authenticated
                       if p.endswith("general_b_hidden_drift_d7f5d9101813_attempt003/retained_graph.npz")]
        retained_require(len(graph_paths) == 1, "independent FP16 reference graph ambiguity")
        refs = retained_npz(graph_paths[0], [f"l4_p0_s{s}_reference" for s in range(19)])
        for stage, actual in enumerate(stages):
            dtype, _, reference = refs[f"l4_p0_s{stage}_reference"]
            retained_require(dtype == "<u2" and len(reference) == len(actual), "FP16 reference ABI")
            failures = []
            for index, (a, b) in enumerate(zip(actual, reference, strict=True)):
                au, bu = retained_units(a), retained_units(b)
                delta = abs(au - bu)
                ordered = lambda h: 32768 - (h & 32767) if h & 32768 else 32768 + h
                ulp = abs(ordered(a) - ordered(b))
                accepted = delta <= 1 << 21 or (delta * 1000 < max(abs(bu), 1024) and ulp <= 1)
                if not accepted:
                    failures.append({"index": index, "actual_bits": a, "reference_bits": b,
                                     "absolute_error": str(Fraction(delta, 1 << 24)),
                                     "relative_error": str(Fraction(delta, max(abs(bu), 1024))),
                                     "ordered_fp16_ulp": ulp})
            row = {"node": [4, 0, stage], "comparisons": len(actual),
                   "finite_comparisons": len(actual), "failure_count": len(failures),
                   "failures": failures}
            reports.append(row)
            retained_write(out / f"stage{stage:02d}_gate.json", row)
            if failures:
                witness = row
                raise RuntimeError("retained raw RTL fails unchanged FP16 gate")
            for label in ("software", "rtl"):
                old = load(attempt / f"{label}_stage{stage:02d}_gate.json")
                retained_require(all(old[key] == row[key] for key in row), "old FP16 report disagreement")
        retained_require(sum(counts) == result["finite_scalar_comparisons"] == 23324
                         and result["affected_fp16_gates"] == 19, "original gate totals disagree")
        phase = "eligible_original_binary64_v1"
        v1 = load(attempt / "rtl_binary64_v1.json")
        retained_require(v1["profile"] == original["binary64_profile"] ==
                         "ace3-w4a16-layer-final-binary64-fp16-excess-v1", "v1 policy identity")
        dtype, shape, ref64 = retained_npy(Path(v1["reference"]["path"]).read_bytes())
        retained_require(dtype == "<f8" and shape == (896,), "binary64 reference ABI")
        csv_paths = [Path(p) for p in authenticated if Path(p).name == "full_binary64_deltas.csv"]
        retained_require(len(csv_paths) == 1, "original binary64 reference ambiguity")
        with csv_paths[0].open(newline="") as stream:
            original_rows = [r for r in csv.DictReader(stream)
                             if int(r["layer"]) == 4 and int(r["position"]) == 0]
        originals = {int(r["index"]): r["reference_binary64_hex"] for r in original_rows}
        retained_require(len(original_rows) == 896 and set(originals) == set(range(896)),
                         "original independently propagated reference coordinates")
        finite_halves = sorted({Fraction(retained_units(b), 1 << 24)
                                for b in range(65536) if b & 0x7c00 != 0x7c00})
        rows = []
        for index, (bits, reference) in enumerate(zip(final, ref64, strict=True)):
            retained_require(math.isfinite(reference) and abs(reference) <= 65504
                             and reference.hex() == originals[index], "invalid/drifted binary64 reference")
            r = Fraction.from_float(reference)
            insertion = bisect.bisect_left(finite_halves, r)
            adjacent = finite_halves[max(0, insertion - 1):insertion + 1]
            q = min(abs(h - r) for h in adjacent)
            error = abs(Fraction(retained_units(bits), 1 << 24) - r)
            excess = error - q
            retained_require(excess >= 0, "negative excess is an evaluator defect")
            row = {"index": index, "actual_fp16_bits": bits, "reference_binary64_hex": reference.hex(),
                   "q": str(q), "actual_error": str(error), "excess_error": str(excess),
                   "accepted": excess <= Fraction(1, 8),
                   "legacy_absolute_accepted": error <= Fraction(1, 8)}
            rows.append(row)
        failures = [r for r in rows if not r["accepted"]]
        retained_write(out / "binary64_v1.json", {
            "profile": v1["profile"], "reference": v1["reference"], "rows": rows,
            "coordinates": len(rows), "failure_count": len(failures),
            "method": "exact Fraction exhaustive finite-binary16 set; no project evaluator imports"})
        if failures:
            witness = {"node": [4, 0, 18], "first": failures[0], "failure_count": len(failures)}
            raise RuntimeError("retained raw RTL fails unchanged binary64-v1")
        retained_require(len(v1["rows"]) == v1["coordinates"] == 896 and v1["failure_count"] == 0,
                         "original v1 totals disagree")
        for new, old in zip(rows, v1["rows"], strict=True):
            retained_require(all(new[k] == old[k] for k in (
                "index", "q", "actual_error", "excess_error", "accepted", "legacy_absolute_accepted")),
                "independent exact arithmetic disagrees with original v1 report")
        phase = "immutable_artifact_preservation"
        retained_require(set(retained_modules()) <= set(authenticated),
                         "verification loaded an unbound executable Python module")
        current = [retained_record(p) for p in sorted(attempt.rglob("*")) if p.is_file()]
        retained_require(current == freeze["retained_tree"], "attempt001 mutated during verification")
        retained_write(out / "result.json", {
            "status": "RETAINED_L4_VERIFIED_PENDING_NORMAL_HOST_REVIEW",
            "rtl_replayed": False, "software_propagation_replayed": False,
            "public_contract_recompiled": False, "attempt001_unchanged": True,
            "retained_files_unchanged": len(current), "authenticated_artifacts": len(authenticated),
            "fp16": {"stage_gates": len(reports), "finite_comparisons": sum(counts), "failures": 0},
            "binary64_v1": {"coordinates": len(rows), "failures": 0,
                            "legacy_absolute_failures_preserved":
                            sum(not r["legacy_absolute_accepted"] for r in rows)},
            "input_hidden": original["input_hidden"], "input_state": None,
            "parent_state_not_imported": original["parent_state_not_imported"],
            "output": tx["output"], "output_state": tx["output_state"], "state": state,
            "natural_terminal": terminal, "durable_run_id": receipt["run_id"],
            "original_result": retained_record(attempt / "result.json"),
            "original_manifest": retained_record(attempt / "output_manifest.json"),
            "freeze": retained_record(out / "freeze.json"),
            "parent_review": original["review"], "elapsed_seconds": time.monotonic() - started,
            "normal_host_review": "REQUIRED", "frontier": "L4/P0 only; L5+ and P1+ unproved",
            "historical_helper_boundary": freeze["historical_helper_boundary"]})
        print("Retained L4/P0: 19 FP16 gates / 23324 scalars; 896 binary64-v1 coordinates; "
              "saved native K/V matches raw S6/S7. No RTL replay. Host review required.", flush=True)
    except (RuntimeError, OSError, ValueError, KeyError, IndexError, struct.error,
            zipfile.BadZipFile) as error:
        taxonomy = ("retained_numerical_gate" if witness else
                    "source_provenance_gap" if phase == "source_and_input_authentication" else
                    "retained_evidence_contract")
        retained_write(out / "failure.json", {
            "phase": phase, "error": str(error), "failure_taxonomy": taxonomy,
            "root_cause_hypothesis": "The named retained-data boundary is inconsistent; "
                                     "no unique RTL arithmetic root cause is established.",
            "regression": "Bind every executable before use; independently decode unchanged "
                          "raw outputs/state; preserve FP16-before-v1 ordering and no replay.",
            "first_failure": witness, "completed_fp16_gates": len(reports),
            "rtl_replayed": False, "normal_host_review": "REQUIRED"})
        raise
    finally:
        retained_write(out / "output_manifest.json", {
            "artifacts": [retained_record(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.name not in ("verification.log", "output_manifest.json")],
            "normal_host_review": "PENDING"})


def continue_cone(out, layers, gate_policy=None):
    """Screen the whole authorized suffix before its actual-output-fed RTL chain."""
    from ace3.model.candidates import decoder_gate_policy as policy
    policy.fp16_is_mandatory(18, gate_policy)
    retained_require(layers in ((4,), (5, 6, 7, 8)), "unsupported continuation scope")
    parent_layer = layers[0] - 1
    parent_mission, review_round = (
        ("34220ce45e75", 3) if parent_layer == 3 else ("fb06353c5f53", 5))
    parent = BUILD / f"single_round_residual_rtl_execution_{parent_mission}_attempt001"
    review_path = REVIEW.parent.parent / parent_mission / f"round-{review_round:04d}.json"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "runner.py")
    helper = retained_record(BUILD /
        "layer08_position3_softmax_exp_replay_attempt003/dependency_closure/replay.py")
    d = module("single_round_continuation_helpers", Path(helper["path"]))
    bound, timings, gates = {}, [], []
    phase, active, first_failure = "authentication", [layers[0], 0], None
    started = time.monotonic()
    rtl_started = False
    transactions, software, references, rtl_results = [], {}, {}, []

    def bind(rec):
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        if rec["path"] not in bound:
            d.authenticate(signature)
            bound[rec["path"]] = signature
        else:
            d.require(bound[rec["path"]] == signature, "conflicting artifact signatures")
        return Path(rec["path"])

    def read(rec):
        return d.load(bind(rec))

    def timed(argv, directory, label, timeout=1800):
        begin = time.monotonic()
        d.run_command(list(argv), directory, label, timeout)
        timings.append({"phase": label, "seconds": time.monotonic() - begin})

    def compare(stages, label):
        nonlocal active, first_failure
        rows = []
        for stage in range(19):
            active = [layer, 0, stage]
            actual = stages[stage].reshape(-1)
            reference = graph[f"l{layer}_p0_s{stage}_reference"].reshape(-1)
            d.require(actual.shape == reference.shape, "independent FP16 stage geometry")
            d.require(actual.dtype == reference.dtype == np.dtype("<u2")
                      and np.all((actual & 0x7c00) != 0x7c00)
                      and np.all((reference & 0x7c00) != 0x7c00),
                      "invalid/nonfinite FP16 stage operand")
            # Q24 integers make the frozen strict relative boundary exact.
            a = np.asarray([sw.units(x) for x in actual], dtype=np.int64)
            b = np.asarray([sw.units(x) for x in reference], dtype=np.int64)
            ordered = lambda x: np.where(
                x & 0x8000, 0x8000 - (x & 0x7fff).astype(np.int64),
                0x8000 + x.astype(np.int64))
            delta = np.abs(a - b)
            ulp = np.abs(ordered(actual) - ordered(reference))
            accepted = ((delta <= (1 << 21)) |
                        ((delta * 1000 < np.maximum(np.abs(b), 1024)) & (ulp <= 1)))
            failures = [{"index": int(i), **scalar.scalar(actual[i], reference[i])}
                        for i in np.flatnonzero(~accepted)]
            row = {"node": active, "evidence_kind": label, "comparisons": int(actual.size),
                   "finite_comparisons": int(actual.size), "failure_count": len(failures),
                   "failures": failures}
            if gate_policy is not None:
                row.update(policy_id=gate_policy, fp16_role=(
                    "mandatory" if policy.fp16_is_mandatory(stage, gate_policy)
                    else "layer-final-trajectory-conformance-diagnostic"))
            d.write(directory / f"{label}_stage{stage:02d}_gate.json", row)
            rows.append(row)
            gates.append(row)
            if failures and policy.fp16_is_mandatory(stage, gate_policy):
                first_failure = row
                raise RuntimeError("unchanged FP16 interstage gate failed")
        return rows

    def binary64(actual, label):
        nonlocal first_failure, active
        active = [layer, 0, 18]
        reference_record, reference64 = references[layer]
        rows = []
        for index, (bits, reference) in enumerate(zip(actual, reference64, strict=True)):
            r = Fraction.from_float(float(reference))
            row = profile.evaluate_layer_final_output(
                actual_fp16_bits=int(bits), reference_binary64_hex=float(reference).hex())
            nearest = scalar.rne(r)
            q = abs(scalar.half(nearest) - r)
            error = abs(scalar.half(int(bits)) - r)
            d.require(row["q"] == str(q) and row["actual_error"] == str(error)
                      and row["excess_error"] == str(error - q)
                      and row["accepted"] == (error - q <= Fraction(1, 8)),
                      "independent Fraction/v1 evaluator disagreement")
            rows.append({"index": index, "legacy_absolute_accepted": error <= Fraction(1, 8),
                         **row})
        failures = [r for r in rows if not r["accepted"]]
        result = {"node": active, "evidence_kind": label, "reference": reference_record,
                  "profile": profile.PROFILE_ID, "coordinates": len(rows),
                  "failure_count": len(failures), "rows": rows}
        if gate_policy is not None:
            result.update(policy_id=gate_policy, role="mandatory")
        d.write(directory / f"{label}_binary64_v1.json", result)
        if failures:
            first_failure = {"node": active, "evidence_kind": label,
                             "failure_count": len(failures), "first": failures[0]}
            raise RuntimeError("unchanged binary64-v1 gate failed")
        return {"coordinates": len(rows), "failures": len(failures)}

    try:
        bind(helper)
        review = read(d.record(review_path))
        d.require(review["kind"] == "round_reviewed_handoff" and
                  review["producer_role"] == "reviewer" and
                  review["mission_id"] == parent_mission and
                  review["review"]["status"] == "done", "accepted actual parent missing")
        if parent_layer == 4:
            admission = read(d.record(BUILD /
                "single_round_residual_rtl_execution_a0825679a5e3_parent_binding_attempt001/parent_binding.json"))
            bind(admission["reviewed_parent"]["review"])
            for name in ("freeze", "result", "output_manifest"):
                d.require(d.record(parent / f"{name}.json")["sha256"] ==
                          admission["original_execution"][f"{name}_sha256"],
                          f"reviewed L4 {name} binding drift")
            bind(admission["sole_next_cone_hidden_parent"])
            bind(admission["state_lineage"]["output_state"])
        manifest = read(d.record(parent / "output_manifest.json"))
        members = {r["path"]: r for r in manifest["artifacts"]}
        pr = read(members[str(parent / "result.json")])
        pf = read(pr["freeze"])
        parent_v1 = pr["binary64_v1"]
        d.require(pr["status"] == "PASS_BOUNDED_RTL_CANDIDATE_PENDING_HOST_REVIEW"
                  and pr["affected_fp16_gates"] == (58 if parent_layer == 3 else 19)
                  and all(x["failures"] == 0 for x in
                          (parent_v1 if isinstance(parent_v1, list) else [parent_v1])),
                  "parent bounded numerical result not eligible")
        d.require(pf["semantics"] == "RNE16(H + O + D), exact three-term sum, only one rounding"
                  and pf["token_history"] == [9707], "parent arithmetic/history mismatch")
        parent_rows = [r for r in pr["transactions"]
                       if Path(r["path"]).parent.name == f"layer{parent_layer:02d}"]
        d.require(len(parent_rows) == 1, "ambiguous actual hidden parent")
        tx = read(parent_rows[0])["transaction"]
        hidden_path = bind(tx["output"])
        bind(tx["output_state"])
        payload = bind(tx["raw"]["trace"]).read_bytes()
        inherited = {r["path"]: r for r in pf["input_bindings"]}
        sf = read(inherited[str(SOFTWARE / "freeze.json")])
        inherited.update({r["path"]: r for r in sf["bindings"]})
        for rec in sf["bindings"]:
            if rec["path"].endswith(".py"):
                bind(rec)
        sw = module("single_round_software_operator", bind(sf["source"]))
        split = module("single_round_l4_split", bind(inherited[str(
            BUILD / "layer08_position2_rope_split_86938809f0a2_attempt007/experiment.py")]))
        native = module("single_round_l4_native", bind(inherited[str(
            BUILD / "layer08_position3_softmax_exp_replay_attempt003/dependency_closure/candidate_oracle.py")]))
        scalar = module("single_round_l4_fraction", bind(inherited[str(
            BUILD / "general_b_l8p3s1_index223_5a41bc443f05_attempt002/run.py")]))
        prior, _ = d.imports()
        np, traversal, profile = prior.np, prior.traversal, sw.profile
        preflight = read(inherited[str(PREFLIGHT / "result.json")])
        preflight_freeze = read(preflight["freeze"])
        exact = module("single_round_l4_residual_oracle", bind(preflight_freeze["runner"]))
        hidden = traversal.load_hidden_bits(hidden_path)
        d.require(hidden.shape == (896,) and np.array_equal(
            hidden, prior.frontier.trace_stage(payload, 0, 18, 896)),
            "parent actual final/trace mismatch")
        graph_path = BUILD / "general_b_hidden_drift_d7f5d9101813_attempt003/retained_graph.npz"
        with np.load(bind(inherited[str(graph_path)]), allow_pickle=False) as archive:
            graph = {k: archive[k].copy() for k in archive.files
                     if any(k.startswith(f"l{n}_p0_") for n in layers)}
        reference_root = next(Path(r["path"]).parent for r in sf["bindings"]
                              if Path(r["path"]).name == "layer03_position000_reference.npy")
        reference_result = read(d.record(reference_root / "result.json"))
        read(d.record(reference_root / "frozen.json"))
        with bind(reference_result["full_binary64_deltas"]).open(newline="") as stream:
            original = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"]
                        for r in csv.DictReader(stream)
                        if int(r["layer"]) in layers and int(r["position"]) == 0}
        for layer in layers:
            rec = d.record(reference_root / f"layer{layer:02d}_position000_reference.npy")
            reference64 = np.load(bind(rec), allow_pickle=False)
            d.require(reference64.dtype == np.dtype("<f8") and reference64.shape == (896,)
                      and all(float(r).hex() == original[layer, i]
                              for i, r in enumerate(reference64)),
                      "array differs from original independently propagated binary64 evidence")
            references[layer] = (rec, reference64)
        bf = read(inherited[str(B / "frozen.json")])
        read(bf["reference_source"])
        # The older graph omits L8/P0. Use the same original trajectory already
        # hash-bound by the accepted candidate archive, never a candidate output.
        if 8 in layers:
            refs = [r for r in bf["reference_transactions"]
                    if r["layer"] == 8 and r["position"] == 0]
            d.require(len(refs) == 1 and set(refs[0]["stages"]) ==
                      {str(s) for s in range(19)}, "incomplete original L8 reference")
            read(refs[0]["parent"])
            original_root = BUILD / "independent_fp16_trajectory_20260906_1133/recovery001"
            original_receipt = read(d.record(original_root / "layer08_generation0.json"))
            d.require(original_receipt["layer"] == 8 and original_receipt["history"][0] == 9707
                      and original_receipt["positions"] == [0, 1], "L8 reference root mismatch")
            originals = {r["path"]: r for r in original_receipt["stages"]}
            for stage in range(19):
                rec = refs[0]["stages"][str(stage)]
                words = bind(rec).read_text().split()
                d.require(len(words) == rec["records"] and
                          all(re.fullmatch(r"[0-9a-fA-F]{4}", w) for w in words),
                          "independent reference hex ABI")
                values = np.asarray([int(w, 16) for w in words], dtype="<u2")
                d.require(hashlib.sha256(values.tobytes()).hexdigest() ==
                          rec["semantic_sha256"], "L8 historical semantic hash mismatch")
                path = original_root / f"trajectory/layer08/position000/stage{stage:02d}.npy"
                original_rec = originals[str(path)]
                raw_rec = d.record(path)
                d.require(raw_rec["sha256"] == original_rec["file_sha256"]
                          and original_rec["semantic_sha256"] == rec["semantic_sha256"]
                          and np.array_equal(np.load(bind(raw_rec), allow_pickle=False), values),
                          "L8 independent original trajectory mismatch")
                graph[f"l8_p0_s{stage}_reference"] = values
        d.require(all(f"l{n}_p0_s{s}_reference" in graph for n in layers for s in range(19)),
                  "authorized reference stage coverage incomplete")
        checkpoint = bind(bf["checkpoint"])
        metadata = {r["name"]: r for r in bf["checkpoint_tensors"]}
        source = out / "source"
        source.mkdir()
        for rec in pf["source_closure"]:
            path = bind(rec)
            shutil.copyfile(path, source / path.name)
        d.require(d.header((source / f"{TOP}.sv").read_text()) == pf["public_contract"],
                  "exact public module/port/parameter contract changed")
        (out / "public_contract.sv").write_text(pf["public_contract"] + "\nendmodule\n")
        tools = {name: d.tool(name, flag) for name, flag in
                 (("verilator", "--version"), ("iverilog", "-V"),
                  ("make", "--version"), ("g++", "--version"))}
        d.SOURCE = source
        commands = {}
        for layer in layers:
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            commands[layer] = d.decoder_command(tools, layer, directory / "obj", True)
            commands[layer].insert(-1, str(source / "ace3_fp16_single_round_residual_core.sv"))
        from safetensors import safe_open
        executable_modules = []
        for path in retained_modules():
            if Path(path).is_relative_to(ROOT):
                executable_modules.append(d.record(bind(d.record(path))))
        d.write(out / "freeze.json", {
            **({"gate_policy": gate_policy, "gate_policy_contract": d.record(policy.CONTRACT),
                "gate_policy_evaluator": d.record(Path(policy.__file__))}
               if gate_policy is not None else {}),
            "review": d.record(review_path), "parent_result": members[str(parent / "result.json")],
            "semantics": pf["semantics"], "token_history": [9707],
            "scope": {"layers": list(layers), "position": 0, "stages": list(range(19))},
            "public_contract": pf["public_contract"],
            "parameters": [{"LAYER_INDEX": n, "ACCURATE_SILU": 1} for n in layers],
            "input_hidden": tx["output"], "input_state": None,
            "parent_state_not_imported": tx["output_state"],
            "kv_policy": "Each layer starts its own empty history; retain actual S6/S7 and state",
            "gate": sf["stage_gate"], "reference_policy": sf["reference_policy"],
            "binary64_profile": profile.PROFILE_ID,
            "evaluation_order": (
                "Ordered suffix software: S0-S17 mandatory FP16; S18 FP16 diagnostic and mandatory binary64-v1; then ordered actual RTL with the same policy"
                if gate_policy is not None else
                "Ordered suffix software: 19 FP16 gates then v1 per layer; only if all supported, ordered actual RTL with the same gates"),
            "tools": tools, "commands": commands, "input_bindings": list(bound.values()),
            "executable_modules": executable_modules,
            "helper_binding_boundary": "Fresh current helper identity, not inferred historical identity",
            "source_closure": [d.record(p) for p in sorted(source.iterdir())],
            "runner": d.record(out / "runner.py"),
            "python": sys.version, "numpy": np.__version__,
            "official_model_attempt": False, "normal_host_review": "REQUIRED",
            "excluded": [f"L{layers[-1] + 1}+", "P1+", "generation", "tail", "production promotion", "hardware"]})
        phase = "public_contract_compile"
        timed([tools["iverilog"]["executable"]["path"], "-g2012", "-s", TOP,
               "-o", str(out / "public_contract.vvp"), str(out / "public_contract.sv")],
              out, phase, 60)
        parent_hidden = hidden.copy()
        parent_output = tx["output"]
        with safe_open(str(checkpoint), framework="numpy") as model:
            for layer in layers:
                directory = out / f"layer{layer:02d}"
                phase, active = "software_propagation", [layer, 0]
                begin = time.monotonic()
                values = {}
                prefix = f"model.layers.{layer}."
                for name, rec in metadata.items():
                    if name.startswith(prefix):
                        tensor = model.get_tensor(name)
                        d.require(hashlib.sha256(tensor.tobytes()).hexdigest() == rec["sha256"],
                                  f"official checkpoint tensor drift: {name}")
                        values[name.replace(prefix, "model.layers.0.") + ":"] = (
                            tensor.view("<u2").reshape(-1).tolist() if tensor.dtype.itemsize == 2
                            else tensor.view("<u4").reshape(-1))
                calls = {"q": 0, "k": 0}
                runner = native.bind(native.run_token, {
                    "_module": split.operators.projection(values, False),
                    "_rope": split.split_rope(native.run_token.__globals__["_rope"],
                                             14, 2, True, False, calls)})
                keys, vals = [], []
                final, records = runner(values, hidden.tolist(), 0, keys, vals, accurate_silu=True)
                groups = {}
                for stage, index, value, position in records:
                    d.require(position == 0, "native position ABI drift")
                    groups.setdefault(stage, []).append((index, value))
                stages = {s: split.d.decode_stage_records(rows, s, 0) for s, rows in groups.items()}
                d.require(set(stages) == set(range(19)) and calls == {"q": 1, "k": 1}
                          and np.array_equal(final, stages[18]) and len(keys) == len(vals) == 1
                          and np.array_equal(keys[0], stages[6]) and np.array_equal(vals[0], stages[7]),
                          "software native trace/KV ABI mismatch")
                residual = []
                for h, o, down in zip(hidden, stages[11], stages[17], strict=True):
                    bits, invalid, saturation = exact.oracle(int(h), int(o), int(down))
                    d.require(not invalid and not saturation and
                              bits == sw.operator(int(h), int(o), int(down), "single_round")[0],
                              "independent exact residual oracle mismatch")
                    residual.append(bits)
                stages[18] = np.asarray(residual, dtype="<u2")
                with (directory / "software_stages.npz").open("xb") as stream:
                    np.savez(stream, input_hidden=hidden,
                             input_cache_k=np.empty((0, 128), dtype="<u2"),
                             input_cache_v=np.empty((0, 128), dtype="<u2"),
                             **{f"stage{s:02d}": a for s, a in stages.items()})
                software[layer] = (hidden.copy(), stages, records)
                timings.append({"phase": phase, "layer": layer, "seconds": time.monotonic() - begin})
                phase = "software_fp16_comparison"
                software_gates = compare(stages, "software")
                phase = "software_binary64_v1_comparison"
                software_v1 = binary64(stages[18], "software")
                d.write(directory / "software_screen.json", {
                    **({"gate_policy": gate_policy} if gate_policy is not None else {}),
                    "status": "SUPPORTED_SOFTWARE_ONLY", "stage_gates": software_gates,
                    "binary64_v1": software_v1, "stages": d.record(directory / "software_stages.npz"),
                    "input_lineage": "actual retained parent then software-only suffix",
                    "actual_root": parent_output, "input_state": None})
                hidden = stages[18].copy()
            print(f"L{layers[0]}-L{layers[-1]}/P0 software screen supported; starting actual RTL", flush=True)
            hidden, actual_parent = parent_hidden, parent_output
            for layer in layers:
                directory = out / f"layer{layer:02d}"
                expected_input, stages, records = software[layer]
                d.require(np.array_equal(hidden, expected_input), "actual/software input divergence")
                phase, active = "decoder_compile", [layer, 0]
                timed(commands[layer], directory, "compile")
                binary = directory / f"obj/V{TOP}"
                phase = "vector_preparation"
                vectors = traversal.materialize_transaction_vectors(
                    model, layer, 0, hidden, directory / "vectors")
                for tensor in vectors["tensors"]:
                    rec = tensor["checkpoint_tensor"]
                    d.require(rec["sha256"] == metadata[rec["name"]]["sha256"],
                              "official vector tensor drift")
                counters, trace = {8: 0, 9: 0}, []
                for stage, index, _, position in records:
                    offset = counters[stage] if stage in counters else index
                    trace.append((stage, index, int(stages[stage][offset]), position))
                    if stage in counters:
                        counters[stage] += 1
                d.require(len(trace) == sum(a.size for a in stages.values()), "trace coverage")
                vectors.update(traversal.materialize_runtime_vector_contract(
                    layer, 0, stages[18], trace, directory / "vectors"))
                d.write(directory / "prepared.json", {
                    "binary": d.record(binary), "input_hidden": actual_parent, "input_state": None,
                    "kv_policy": "own empty prior K/V, no preceding layer state import",
                    "vectors": vectors, "software_oracle": d.record(directory / "software_stages.npz")})
                traversal.run_logged = lambda argv, log: timed(argv, log.parent, log.stem)
                phase, rtl_started = "decoder_simulation", True
                result, final = traversal.execute_transaction(
                    binary, layer, 0, hidden, vectors, directory / "vectors",
                    directory / "runtime", directory / "candidate.state", None)
                d.write(directory / "transaction.json", result)
                payload = bind(result["raw"]["trace"]).read_bytes()
                decoded = {s: prior.frontier.trace_stage(
                    payload, 0, s, stages[s].size, 1 if s in (8, 9) else None) for s in range(19)}
                with (directory / "actual_stages.npz").open("xb") as stream:
                    np.savez(stream, input_hidden=hidden,
                             **{f"stage{s:02d}": a for s, a in decoded.items()})
                phase = "rtl_fp16_comparison"
                rtl_gates = compare(decoded, "rtl")
                phase = "same_input_oracle_comparison"
                d.require(all(np.array_equal(decoded[s], stages[s]) for s in range(19))
                          and np.array_equal(final, decoded[18]), "actual/same-input oracle disagreement")
                actual_path = bind(result["output"])
                bind(result["output_state"])
                phase = "rtl_binary64_v1_comparison"
                rtl_v1 = binary64(final, "rtl")
                d.write(directory / "result.json", {
                    **({"gate_policy": gate_policy} if gate_policy is not None else {}),
                    "transaction": result, "stage_gates": rtl_gates, "binary64_v1": rtl_v1,
                    "actual_kv": {"k": decoded[6].tolist(), "v": decoded[7].tolist()},
                    "same_input_oracle_exact": True})
                transactions.append(d.record(directory / "result.json"))
                rtl_results.append(rtl_v1)
                hidden = traversal.load_hidden_bits(actual_path)
                actual_parent = result["output"]
        for rec in executable_modules:
            d.authenticate(rec)
        d.write(out / "result.json", {
            **({"gate_policy": gate_policy,
                "legacy_fp16_layer_final_diagnostics": [
                    r for r in gates if r["node"][2] == 18 and r["evidence_kind"] == "rtl"]}
               if gate_policy is not None else {}),
            "status": "PASS_BOUNDED_RTL_CANDIDATE_PENDING_HOST_REVIEW",
            "freeze": d.record(out / "freeze.json"),
            "affected_fp16_gates": (18 if gate_policy is not None else 19) * len(layers),
            "finite_scalar_comparisons": sum(r["finite_comparisons"] for r in gates
                                             if r["evidence_kind"] == "rtl"),
            "binary64_v1": rtl_results, "transactions": transactions,
            "phase_times": timings, "elapsed_seconds": time.monotonic() - started,
            "no_full_model_or_generation_admission": True, "normal_host_review": "REQUIRED"})
        print(f"L{layers[0]}-L{layers[-1]}/P0 actual RTL complete; independent Host review required", flush=True)
    except (RuntimeError, OSError, ValueError, KeyError, ArithmeticError,
            subprocess.SubprocessError) as error:
        numerical = first_failure is not None
        d.write(out / "failure.json", {
            **({"gate_policy": gate_policy} if gate_policy is not None else {}),
            "phase": phase, "active": active, "error": str(error),
            "first_failure": first_failure, "rtl_started": rtl_started,
            "failure_taxonomy": "candidate_numerical_gate" if numerical else "evaluator_no_completion",
            "root_cause_hypothesis": (
                "The propagated candidate exceeds the unchanged gate; unique arithmetic causality is unestablished."
                if numerical else "The named binding/execution boundary did not complete; no RTL correctness conclusion."),
            "regression": "Preserve the reviewed actual parent, own empty per-layer K/V, all gates and first failure; no unchanged replay.",
            "gates": gates, "input_bindings": list(bound.values()), "phase_times": timings,
            "elapsed_seconds": time.monotonic() - started,
            "normal_host_review": "REQUIRED", "no_expansion": True})
        raise
    finally:
        d.write(out / "output_manifest.json", {
            "artifacts": [d.record(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and "obj" not in p.parts],
            "normal_host_review": "PENDING"})


def reevaluate_software(out, attempt, gate_policy, manifest_sha256, review_path):
    """Re-adjudicate retained P0 software; compute only a missing admitted suffix."""
    from ace3.model.candidates import decoder_gate_policy as policy
    policy.fp16_is_mandatory(18, gate_policy)
    retained_require(gate_policy == policy.POLICY_ID, "explicit v2 opt-in required")
    retained_require(attempt.parent == BUILD and attempt != out, "retained attempt namespace")
    out.mkdir(exist_ok=False)
    bound, reports, provenance, timings = {}, [], [], []
    phase, active = "authentication", None
    started = time.monotonic()

    def bind(rec):
        path = rec["path"]
        signature = {k: rec[k] for k in ("path", "bytes", "sha256")}
        if path not in bound:
            retained_require(retained_record(path) == signature, f"artifact binding drift: {path}")
            bound[path] = signature
        else:
            retained_require(bound[path] == signature, f"conflicting binding: {path}")
        return Path(path)

    def read(rec):
        return json.loads(bind(rec).read_text())

    try:
        manifest_rec = retained_record(attempt / "output_manifest.json")
        retained_require(manifest_rec["sha256"] == manifest_sha256,
                         "reviewed retained manifest changed")
        manifest = read(manifest_rec)
        members = {r["path"]: r for r in manifest["artifacts"]}
        retained_require(len(members) == len(manifest["artifacts"]), "duplicate manifest members")
        for rec in members.values():
            bind(rec)
        review_rec = retained_record(review_path)
        review = read(review_rec)
        retained_require(review["kind"] == "round_reviewed_handoff"
                         and review["producer_role"] == "reviewer"
                         and review["review"]["status"] == "done"
                         and attempt.name.startswith(
                             f"single_round_residual_rtl_execution_{review['mission_id']}_"),
                         "genuine mission-scoped retained review missing")
        frozen = read(members[str(attempt / "freeze.json")])
        legacy_result = read(members[str(attempt / "result.json")])
        retained_require(legacy_result["software_only"]
                         and legacy_result["decoder_rtl_invocations"] == 0
                         and frozen["history"] == [9707] and frozen["position"] == 0
                         and frozen["layers"] == [5, 6, 7, 8]
                         and frozen["semantics"] ==
                         "RNE16(H + O + D), exact three-term sum, only one rounding",
                         "retained software arithmetic/history/scope mismatch")
        for rec in frozen["inputs"] + frozen["sources"]:
            bind(rec)
        inputs = {r["path"]: r for r in frozen["inputs"]}

        def input_ending(suffix):
            matches = [r for p, r in inputs.items() if p.endswith(suffix)]
            retained_require(len(matches) == 1, f"missing/ambiguous required input: {suffix}")
            return matches[0]

        historical = module("retained_policy_software", bind(members[str(attempt / "run.py")]))
        np = historical.np
        historical.torch.set_num_threads(1)
        hidden = historical.bits_array(bind(frozen["parent"]))
        retained_require(hidden.shape == (896,) and historical.semantic(hidden) ==
                         frozen["parent"]["semantic_sha256"], "actual L4 hidden parent mismatch")
        bind(frozen["parent_state_not_imported"])
        parent_review = read(frozen["review"])
        retained_require(parent_review["producer_role"] == "reviewer"
                         and parent_review["review"]["status"] == "done", "L4 parent review missing")
        admission = read(input_ending("/parent_binding.json"))
        parent_dir = Path(frozen["parent"]["path"]).parents[3]
        for name in ("result", "freeze", "output_manifest"):
            rec = retained_record(parent_dir / f"{name}.json")
            retained_require(rec["sha256"] == admission["original_execution"][f"{name}_sha256"],
                             f"accepted L4 {name} changed")
            bind(rec)
        with np.load(bind(input_ending("/retained_graph.npz")), allow_pickle=False) as archive:
            graph = {key: archive[key].copy() for key in archive.files}
        bf = read(input_ending(
            "/layer08_position2_q_only_rope_rtl_86938809f0a2_attempt010/frozen.json"))
        l8 = [r for r in bf["reference_transactions"] if r["layer"] == 8 and r["position"] == 0]
        retained_require(len(l8) == 1 and set(l8[0]["stages"]) == {str(s) for s in range(19)},
                         "incomplete original L8 FP16 references")
        for stage, rec in l8[0]["stages"].items():
            array = historical.bits_array(bind(rec))
            retained_require(historical.semantic(array) == rec["semantic_sha256"],
                             "original L8 reference semantic drift")
            graph[f"l8_p0_s{stage}_reference"] = array
        references, reference_records = {}, {}
        for layer in range(5, 9):
            rec = input_ending(f"/layer{layer:02d}_position000_reference.npy")
            reference_records[layer] = rec
            references[layer] = np.load(bind(rec), allow_pickle=False)
        reference_root = Path(reference_records[5]["path"]).parent
        reference_result = read(inputs[str(reference_root / "result.json")])
        with bind(reference_result["full_binary64_deltas"]).open(newline="") as stream:
            originals = {(int(r["layer"]), int(r["index"])): r["reference_binary64_hex"]
                         for r in csv.DictReader(stream)
                         if int(r["position"]) == 0 and 5 <= int(r["layer"]) <= 8}
        for layer, values in references.items():
            retained_require(values.dtype == np.dtype("<f8") and values.shape == (896,)
                             and all(float(v).hex() == originals[layer, i]
                                     for i, v in enumerate(values)),
                             "original independently propagated binary64 reference drift")
        retained_require(all(f"l{n}_p0_s{s}_reference" in graph
                             for n in range(5, 9) for s in range(19)), "FP16 reference coverage")
        checkpoint = bind(bf["checkpoint"])
        compiler = shutil.which("iverilog")
        retained_require(compiler is not None, "public-contract compiler unavailable; no execution")
        contract_path = out / "public_contract.sv"
        shutil.copyfile(bind(members[str(attempt / "public_contract.sv")]), contract_path)
        command = [compiler, "-g2012", "-s", "frozen_contract", "-o",
                   str(out / "public_contract.vvp"), str(contract_path)]
        sources = [retained_record(p) for p in
                   (Path(__file__), Path(policy.__file__), policy.CONTRACT)]
        retained_write(out / "freeze.json", {
            "policy_id": gate_policy, "policy_contract": sources[-1], "sources": sources,
            "binary64_profile": policy.binary64.PROFILE_ID, "retained_manifest": manifest_rec,
            "retained_result": members[str(attempt / "result.json")], "retained_review": review_rec,
            "input_bindings": list(bound.values()), "input_hidden": frozen["parent"],
            "input_state": None, "parent_state_not_imported": frozen["parent_state_not_imported"],
            "public_contract": frozen["public_contract"], "parameters": frozen["parameters"],
            "public_contract_source": retained_record(contract_path), "compile_command": command,
            "tools": {"python": sys.version, "numpy": np.__version__,
                      "torch": historical.torch.__version__, "iverilog": retained_record(compiler),
                      "iverilog_version": subprocess.run(
                          [compiler, "-V"], capture_output=True, text=True, check=True).stdout},
            "scope": {"layers": [5, 6, 7, 8], "position": 0, "history": [9707]},
            "semantics": frozen["semantics"], "reference_policy": policy.binary64.REFERENCE_POLICY,
            "evaluation_order": "S0-S17 mandatory FP16; S18 FP16 diagnostic then mandatory binary64-v1",
            "execution": "Re-evaluate retained compatible software first; only missing suffix arithmetic",
            "decoder_rtl_invocations": 0, "normal_host_review": "REQUIRED"})
        phase = "public_contract_compile"
        with (out / "public_contract_compile.log").open("xb") as stream:
            compiled = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        retained_write(out / "public_contract_compile.json", {"command": command,
                       "returncode": compiled.returncode, "rtl_execution": False})
        retained_require(compiled.returncode == 0, "public-contract compilation failed")
        timings.append({"phase": "authentication_and_contract", "seconds": time.monotonic() - started})
        first_failure = None
        for layer in range(5, 9):
            begin = time.monotonic()
            active, phase = [layer, 0], "software_input_authentication"
            directory = out / f"layer{layer:02d}"
            directory.mkdir()
            retained_path = attempt / f"layer{layer:02d}/software_stages.npz"
            if str(retained_path) in members:
                rec = members[str(retained_path)]
                with np.load(bind(rec), allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                evidence_kind = "retained_software"
            else:
                retained_require(not retained_path.exists(), "unmanifested retained software exists")
                phase = "missing_suffix_software"
                prefix, tensors, tensor_records = f"model.layers.{layer}.", {}, []
                with historical.safe_open(str(checkpoint), framework="numpy") as model:
                    for rec in bf["checkpoint_tensors"]:
                        name = rec["name"]
                        if name.startswith(prefix):
                            tensor = model.get_tensor(name)
                            retained_require(hashlib.sha256(tensor.tobytes()).hexdigest() == rec["sha256"]
                                             and list(tensor.shape) == rec["shape"],
                                             "official tensor hash/shape drift")
                            tensors[name.replace(prefix, "model.layers.0.") + ":"] = tensor.view(
                                "<u2" if tensor.dtype.itemsize == 2 else "<u4").reshape(-1)
                            tensor_records.append(rec)
                retained_require(bool(tensors), "missing layer tensors")
                retained_write(directory / "tensor_bindings.json", tensor_records)
                stages, keys, vals = historical.single_round_position0_software(tensors, hidden)
                arrays = {"input_hidden": hidden, "input_cache_k": np.empty((0, 128), dtype="<u2"),
                          "input_cache_v": np.empty((0, 128), dtype="<u2"),
                          "output_cache_k": np.asarray(keys, dtype="<u2"),
                          "output_cache_v": np.asarray(vals, dtype="<u2"),
                          **{f"stage{s:02d}": a for s, a in stages.items()}}
                with (directory / "software_stages.npz").open("xb") as stream:
                    np.savez(stream, **arrays)
                rec = retained_record(directory / "software_stages.npz")
                evidence_kind = "fresh_missing_suffix_software"
            retained_require(all(a.dtype == np.dtype("<u2") for a in arrays.values())
                             and np.array_equal(arrays["input_hidden"], hidden)
                             and arrays["input_hidden"].shape == (896,)
                             and all(arrays[f"input_cache_{k}"].shape == (0, 128) for k in ("k", "v"))
                             and all(arrays[f"output_cache_{k}"].shape == (1, 128)
                                     and np.array_equal(arrays[f"output_cache_{k}"][0], arrays[f"stage{s:02d}"])
                                     for k, s in (("k", 6), ("v", 7))), "own-layer P0 hidden/KV lineage drift")
            for h, o, d, a in zip(hidden, arrays["stage11"], arrays["stage17"], arrays["stage18"], strict=True):
                retained_require(historical.rne(historical.half(h) + historical.half(o) +
                                               historical.half(d)) == int(a),
                                 "independent exact single-round residual mismatch")
            provenance.append({"layer": layer, "stages": rec, "evidence_kind": evidence_kind,
                               "input_state": None, "kv_policy": "own empty prior P0 KV"})
            phase = "policy_evaluation"
            for stage in range(19):
                active = [layer, 0, stage]
                report = policy.evaluate_decoder_stage(
                    stage=stage, actual=arrays[f"stage{stage:02d}"],
                    reference=graph[f"l{layer}_p0_s{stage}_reference"], policy=gate_policy,
                    reference_binary64=references[layer] if stage == 18 else None)
                report.update(node=active, evidence_kind=evidence_kind)
                if stage == 18:
                    report["binary64_reference"] = reference_records[layer]
                if evidence_kind == "retained_software":
                    old = read(members[str(attempt / f"layer{layer:02d}/software_stage{stage:02d}_gate.json")])
                    retained_require(all(report["fp16"][k] == old[k]
                                         for k in ("comparisons", "failure_count", "failures")),
                                     "original FP16 comparison was not preserved")
                retained_write(directory / f"stage{stage:02d}_evaluation.json", report)
                reports.append(report)
                if report["status"] != "PASS":
                    first_failure = report
                    break
            timings.append({"phase": "layer_policy_evaluation", "layer": layer,
                            "evidence_kind": evidence_kind, "seconds": time.monotonic() - begin})
            if first_failure is not None:
                break
            hidden = arrays["stage18"].copy()
        for rec in sources:
            bind(rec)
        retained_write(out / "result.json", {
            "status": (first_failure["status"] if first_failure else "SUPPORTED_SOFTWARE_ONLY"),
            "policy_id": gate_policy, "freeze": retained_record(out / "freeze.json"),
            "first_failure": first_failure, "software_provenance": provenance,
            "internal_fp16": {"gates": sum(r["stage"] < 18 for r in reports),
                              "failures": sum(r["fp16"]["failure_count"] for r in reports if r["stage"] < 18)},
            "layer_final_binary64_v1": [
                {"node": r["node"], "coordinates": r["binary64_v1"]["coordinates"],
                 "failure_count": r["binary64_v1"]["failure_count"]}
                for r in reports if "binary64_v1" in r],
            "legacy_fp16_layer_final_diagnostics": [
                {"node": r["node"], **r["fp16"]} for r in reports if r["stage"] == 18],
            "failure_taxonomy": "candidate_numerical_gate" if first_failure else None,
            "root_cause_hypothesis": (
                "Candidate trajectory exceeds a mandatory gate; upstream arithmetic causality remains unlocalized."
                if first_failure else None),
            "regression": "Keep internal FP16 mandatory, S18 v1 unchanged and old S18 FP16 failures diagnostic.",
            "phase_times": timings, "elapsed_seconds": time.monotonic() - started,
            "decoder_rtl_invocations": 0, "ancestors_replayed": [], "normal_host_review": "REQUIRED",
            "no_full_model_or_generation_admission": True})
        print(f"v2 software re-adjudication: {'FAIL' if first_failure else 'SUPPORTED_SOFTWARE_ONLY'}; no RTL")
    except (OSError, ValueError, KeyError, RuntimeError, ArithmeticError,
            subprocess.SubprocessError) as error:
        retained_write(out / "result.json", {
            "status": "BLOCKED", "policy_id": gate_policy, "phase": phase, "active": active,
            "error": str(error), "failure_taxonomy": "evaluator_no_completion",
            "root_cause_hypothesis": "The named required binding/operand/execution boundary did not complete.",
            "regression": "Missing references and incompatible evidence must block, never fall back to FP16.",
            "input_bindings": list(bound.values()), "decoder_rtl_invocations": 0,
            "normal_host_review": "REQUIRED", "no_rtl_correctness_conclusion": True})
        raise
    finally:
        retained_write(out / "output_manifest.json", {
            "artifacts": [retained_record(p) for p in sorted(out.rglob("*")) if p.is_file()],
            "normal_host_review": "PENDING"})


def main():
    from ace3.model.candidates import decoder_gate_policy as policy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continuation-layer", type=int, choices=[4, 5])
    retained = parser.add_mutually_exclusive_group()
    retained.add_argument("--prepare-retained-verification", type=Path)
    retained.add_argument("--verify-retained", action="store_true")
    retained.add_argument("--reevaluate-software", type=Path)
    parser.add_argument("--gate-policy", choices=[policy.POLICY_ID])
    parser.add_argument("--retained-manifest-sha256")
    parser.add_argument("--retained-review", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != BUILD or not out.name.startswith("single_round_residual_rtl_execution_"):
        raise RuntimeError("output outside isolated execution namespace")
    if args.reevaluate_software is not None:
        if (args.continuation_layer is not None or args.gate_policy is None
                or args.retained_manifest_sha256 is None or args.retained_review is None):
            parser.error("software re-adjudication requires policy, manifest and review; no RTL continuation")
        reevaluate_software(out, args.reevaluate_software.resolve(), args.gate_policy,
                            args.retained_manifest_sha256, args.retained_review.resolve())
        return
    if args.gate_policy is not None and args.continuation_layer is None:
        parser.error("policy opt-in requires a supported continuation or software re-adjudication")
    if args.prepare_retained_verification is not None or args.verify_retained:
        if args.continuation_layer is not None:
            parser.error("retained verification cannot launch an RTL continuation")
        if args.prepare_retained_verification is not None:
            prepare_retained_verification(out, args.prepare_retained_verification)
        else:
            verify_retained(out)
        return
    if args.continuation_layer is not None:
        continue_cone(out, (4,) if args.continuation_layer == 4 else (5, 6, 7, 8),
                      args.gate_policy)
        return
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "runner.py")
    d = module("single_round_execution_helpers", BUILD /
               "layer08_position3_softmax_exp_replay_attempt003/dependency_closure/replay.py")
    bound, gates, transactions, timings, finals = {}, [], [], [], {}
    phase, active = "authentication", None
    started = time.monotonic()

    def bind(rec):
        if rec["path"] not in bound:
            d.authenticate(rec)
            bound[rec["path"]] = rec
        else:
            d.require(bound[rec["path"]] == rec, "conflicting artifact signatures")
        return Path(rec["path"])

    def read(rec):
        return d.load(bind(rec))

    def timed(argv, directory, label, timeout=1800):
        begin = time.monotonic()
        d.run_command(list(argv), directory, label, timeout)
        timings.append({"phase": label, "node": active,
                        "seconds": time.monotonic() - begin})

    def compare(layer, stage, actual, graph, prior):
        reference = graph[f"l{layer}_p0_s{stage}_reference"].reshape(-1)
        d.require(actual.shape == reference.shape, "independent stage geometry")
        row = {"layer": layer, "position": 0, "stage": stage,
               **prior.comparison.comparison(actual, reference)}
        gates.append(row)
        d.write(out / f"gate_{len(gates):03d}.json", row)
        d.require(row["failure_count"] == 0, "unchanged FP16 interstage gate failed")

    try:
        review = read(d.record(REVIEW))
        d.require(review["producer_role"] == "reviewer" and
                  review["kind"] == "round_reviewed_handoff" and
                  review["mission_id"] == "f0c34aee3654" and
                  review["review"]["status"] == "done", "reviewed dependency missing")
        preflight_result = read(d.record(PREFLIGHT / "result.json"))
        pf = read(preflight_result["freeze"])
        d.require(pf["semantics"] == "RNE16(H + O + D), exact three-term sum, only one rounding",
                  "single-round semantics changed")
        inherited = {r["path"]: r for r in pf["bindings"]}
        manifest = read(inherited[str(SOFTWARE / "output_manifest.json")])
        artifacts = {r["path"]: r for r in manifest["artifacts"]}
        sf = read(artifacts[str(SOFTWARE / "freeze.json")])
        software_result = read(artifacts[str(SOFTWARE / "result.json")])
        arm = next(x for x in software_result["outcomes"] if x["arm"] == "single_round")
        d.require(arm["first_failure"] is None, "software candidate has an unresolved failure")
        inherited.update({r["path"]: r for r in sf["bindings"]})
        bf = read(inherited[str(B / "frozen.json")])
        checkpoint = bind(bf["checkpoint"])
        prior, _ = d.imports()
        np, traversal = prior.np, prior.traversal
        exact = module("single_round_independent_fraction", bind(pf["runner"]))
        sys.path.insert(0, str(ROOT))
        from ace3.model.candidates import binary64_fp16_excess_v1 as profile
        for path in profile.SOURCE_PATHS:
            bind(inherited[str(ROOT / path)])
        graph_path = BUILD / "general_b_hidden_drift_d7f5d9101813_attempt003/retained_graph.npz"
        with np.load(bind(inherited[str(graph_path)]), allow_pickle=False) as archive:
            graph = {k: archive[k].copy() for k in archive.files
                     if any(k.startswith(f"l{layer}_p0_") for layer in range(4))}
        software, references = {}, {}
        for layer in range(4):
            path = SOFTWARE / f"single_round/layer{layer:02d}_position0.npz"
            with np.load(bind(artifacts[str(path)]), allow_pickle=False) as archive:
                software[layer] = {k: archive[k].copy() for k in archive.files}
            refs = [r for r in inherited.values()
                    if Path(r["path"]).name == f"layer{layer:02d}_position000_reference.npy"]
            d.require(len(refs) == 1, f"missing independent binary64 reference L{layer}/P0")
            references[layer] = (refs[0], np.load(bind(refs[0]), allow_pickle=False))
            d.require(references[layer][1].shape == (896,) and
                      references[layer][1].dtype == np.dtype("<f8"), "binary64 reference ABI")
        old = read(d.record(B / "result.json"))
        d.require(old["status"] == "PASS_BOUNDED_RTL_CANDIDATE", "unchanged RTL prefix not accepted")
        rows = [r for r in old["transactions"] if (r["layer"], r["position"]) == (0, 0)]
        d.require(len(rows) == 1, "ambiguous actual L0/P0 parent")
        retained = read(rows[0]["result"])
        prepared = read(retained["prepared"])
        tx = retained["transaction"]
        bind(prepared["binary"])
        bind(tx["output_state"])
        payload = bind(tx["raw"]["trace"]).read_bytes()
        reference_source = read(bf["reference_source"])
        embedding = bind(reference_source["embeddings"][0]["input"])
        hidden = traversal.load_hidden_bits(embedding)
        d.require(np.array_equal(hidden, software[0]["input_hidden"]), "L0 embedding lineage")
        l0 = {}
        for stage in range(18):
            l0[stage] = prior.frontier.trace_stage(
                payload, 0, stage, software[0][f"stage{stage:02d}"].size,
                1 if stage in (8, 9) else None)
            d.require(np.array_equal(l0[stage], software[0][f"stage{stage:02d}"]),
                      f"retained actual RTL L0/S{stage} differs from reviewed operands")
        schedule = [(int(line[6:8], 16), int(line[8:12], 16))
                    for line in payload.decode("ascii").splitlines()]
        source = out / "source"
        source.mkdir()
        for rec in pf["source_closure"]:
            path = bind(rec)
            shutil.copyfile(path, source / path.name)
        d.require(d.header((source / f"{TOP}.sv").read_text()) == pf["public_contract"],
                  "exact public top/port/parameter contract changed")
        (out / "public_contract.sv").write_text(pf["public_contract"] + "\nendmodule\n")
        tools = {name: d.tool(name, flag) for name, flag in
                 (("verilator", "--version"), ("iverilog", "-V"), ("vvp", "-V"),
                  ("make", "--version"), ("g++", "--version"))}
        d.SOURCE = source
        commands = {}
        for layer in range(1, 4):
            command = d.decoder_command(tools, layer, out / f"layer{layer:02d}/obj", True)
            command.insert(-1, str(source / "ace3_fp16_single_round_residual_core.sv"))
            commands[layer] = command
        vectors = out / "l0_actual_operand_vectors.txt"
        with vectors.open("x", encoding="ascii") as stream:
            stream.write("896\n")
            for h, o, down in zip(hidden, l0[11], l0[17], strict=True):
                value, invalid, saturation = exact.oracle(int(h), int(o), int(down))
                d.require(not invalid and not saturation, "L0 exact oracle invalid result")
                stream.write(f"{int(h):04x} {int(o):04x} {int(down):04x} "
                             f"{value:04x} {invalid} {saturation}\n")
        bind(d.record(vectors))
        d.write(out / "freeze.json", {
            "review": d.record(REVIEW), "semantics": pf["semantics"],
            "public_contract": pf["public_contract"], "parameters": {
                "LAYER_INDEX": [1, 2, 3], "ACCURATE_SILU": 1},
            "scope": "L0/P0/S18 actual RTL; L1-3/P0/S0-18 actual-output-fed RTL",
            "unchanged": sf["unchanged"], "token_history": [9707],
            "l0_prefix": rows[0], "l0_state": tx["output_state"],
            "l0_state_scope": "Actual unchanged K/V only; old residual must not be used as candidate hidden",
            "gate": sf["stage_gate"], "reference_policy": sf["reference_policy"],
            "binary64_profile": profile.PROFILE_ID,
            "evaluation_order": "All 58 affected FP16 gates first; then four eligible binary64-v1 outputs",
            "input_bindings": list(bound.values()), "tools": tools,
            "source_closure": [d.record(p) for p in sorted(source.iterdir())],
            "runner": d.record(out / "runner.py"), "commands": commands,
            "imported_python": [d.record(Path(m.__file__)) for m in list(sys.modules.values())
                                if getattr(m, "__file__", None) and
                                Path(m.__file__).suffix == ".py" and
                                Path(m.__file__).is_relative_to(ROOT)],
            "python": sys.version, "numpy": np.__version__, "official_model_attempt": False,
            "excluded": ["P1+", "L4+", "production promotion", "greedy generation",
                         "tail", "synthesis", "PPA", "hardware"]})
        phase = "public_contract_compile"
        timed([tools["iverilog"]["executable"]["path"], "-g2012", "-s", TOP,
               "-o", str(out / "public_contract.vvp"), str(out / "public_contract.sv")],
              out, phase, 60)
        active, phase = [0, 0, 18], "l0_residual_compile"
        primitive = out / "residual.vvp"
        timed([tools["iverilog"]["executable"]["path"], "-g2012", "-s",
               "ace3_single_round_residual_tb", "-o", str(primitive),
               *[str(source / name) for name in ("ace3_fp16_fixed.sv",
                 "ace3_fp16_single_round_residual_core.sv", "ace3_single_round_residual_tb.sv")]],
              out, phase, 60)
        phase = "l0_residual_simulation"
        actual_path = out / "l0_actual.txt"
        timed([tools["vvp"]["executable"]["path"], str(primitive), f"+INPUT={vectors}",
               f"+RAW={actual_path}", f"+WAVE={out / 'l0_residual.vcd'}"], out, phase, 60)
        raw = [line.split() for line in actual_path.read_text().splitlines()]
        d.require(len(raw) == 896 and all(
            len(r) == 4 and int(r[0]) == i and r[2:] == ["0", "0"]
            for i, r in enumerate(raw)), "L0 actual stream cardinality/flags")
        hidden = np.asarray([int(r[1], 16) for r in raw], dtype="<u2")
        compare(0, 18, hidden, graph, prior)
        finals[0] = hidden.copy()
        parent = d.record(actual_path)
        d.write(out / "l0_result.json", {
            "output": parent, "operands": d.record(vectors), "gate": gates[-1],
            "unchanged_actual_kv": tx["output_state"],
            "candidate_hidden_must_be_consumed_separately": True})
        traversal.run_logged = lambda argv, log: timed(argv, log.parent, log.stem)
        tensor_metadata = {r["name"]: r for r in bf["checkpoint_tensors"]}
        from safetensors import safe_open
        with safe_open(str(checkpoint), framework="np") as model:
            for layer in range(1, 4):
                active, phase = [layer, 0], "decoder_compile"
                directory = out / f"layer{layer:02d}"
                directory.mkdir()
                timed(commands[layer], directory, "compile")
                binary = directory / f"obj/V{TOP}"
                arrays = {s: software[layer][f"stage{s:02d}"] for s in range(19)}
                d.require(np.array_equal(hidden, software[layer]["input_hidden"]),
                          "actual hidden differs from reviewed same-input oracle; no software seed substitution")
                d.require(software[layer]["input_cache_k"].size == 0 and
                          software[layer]["input_cache_v"].size == 0, "P0 requires empty prior KV")
                counters = {8: 0, 9: 0}
                trace = []
                for stage, index in schedule:
                    offset = counters[stage] if stage in counters else index
                    trace.append((stage, index, int(arrays[stage][offset]), 0))
                    if stage in counters:
                        counters[stage] += 1
                d.require(len(trace) == sum(a.size for a in arrays.values()), "trace schedule coverage")
                phase = "vector_preparation"
                vectors_info = traversal.materialize_transaction_vectors(
                    model, layer, 0, hidden, directory / "vectors")
                for tensor in vectors_info["tensors"]:
                    rec = tensor["checkpoint_tensor"]
                    d.require(rec["sha256"] == tensor_metadata[rec["name"]]["sha256"],
                              "official checkpoint tensor changed")
                vectors_info.update(traversal.materialize_runtime_vector_contract(
                    layer, 0, arrays[18], trace, directory / "vectors"))
                d.write(directory / "prepared.json", {
                    "binary": d.record(binary), "input_hidden": parent, "input_state": None,
                    "kv_policy": "empty prior history; retain only this layer's actual P0 K/V",
                    "vectors": vectors_info, "oracle": artifacts[str(
                        SOFTWARE / f"single_round/layer{layer:02d}_position0.npz")]})
                phase = "decoder_simulation"
                result, final = traversal.execute_transaction(
                    binary, layer, 0, hidden, vectors_info, directory / "vectors",
                    directory / "runtime", directory / "candidate.state", None)
                payload = bind(result["raw"]["trace"]).read_bytes()
                phase = "fp16_comparison"
                decoded = {s: prior.frontier.trace_stage(
                    payload, 0, s, arrays[s].size, 1 if s in (8, 9) else None)
                    for s in range(19)}
                with (directory / "actual_stages.npz").open("xb") as stream:
                    np.savez(stream, input_hidden=hidden,
                             **{f"stage{s:02d}": a for s, a in decoded.items()})
                for stage in range(19):
                    compare(layer, stage, decoded[stage], graph, prior)
                    d.require(np.array_equal(decoded[stage], arrays[stage]),
                              f"same-input oracle mismatch L{layer}/S{stage}")
                d.require(np.array_equal(final, decoded[18]), "final/trace disagreement")
                bind(result["output_state"])
                d.write(directory / "result.json", {
                    "transaction": result, "stage_gates": gates[-19:],
                    "actual_kv": {k: decoded[s].tolist() for k, s in (("k", 6), ("v", 7))},
                    "same_input_oracle_exact": True})
                transactions.append(d.record(directory / "result.json"))
                hidden, parent = final, result["output"]
                finals[layer] = final.copy()
                print(f"completed L{layer}/P0 affected_fp16_gates={len(gates)}", flush=True)
        phase = "binary64_v1_comparison"
        d.require(len(gates) == 58, "affected FP16 gate denominator")
        binary64 = []
        for layer in range(4):
            active = [layer, 0, 18]
            reference_record, reference = references[layer]
            rows = [dict(index=i, **profile.evaluate_layer_final_output(
                actual_fp16_bits=int(value), reference_binary64_hex=float(reference[i]).hex()))
                for i, value in enumerate(finals[layer])]
            failures = [r for r in rows if not r["accepted"]]
            d.write(out / f"binary64_layer{layer:02d}.json", {
                "reference": reference_record, "rows": rows, "failure_count": len(failures)})
            binary64.append({"layer": layer, "coordinates": len(rows), "failures": len(failures)})
            d.require(not failures, "unchanged binary64-v1 gate failed")
        d.write(out / "result.json", {
            "status": "PASS_BOUNDED_RTL_CANDIDATE_PENDING_HOST_REVIEW",
            "freeze": d.record(out / "freeze.json"), "transactions": transactions,
            "l0_result": d.record(out / "l0_result.json"), "affected_fp16_gates": len(gates),
            "binary64_v1": binary64, "phase_times": timings,
            "elapsed_seconds": time.monotonic() - started,
            "next_cone": "Eligible for independent Reviewer expansion decision only",
            "no_full_model_or_generation_admission": True})
    except (RuntimeError, OSError, ValueError, KeyError, ArithmeticError,
            subprocess.SubprocessError) as error:
        numerical = phase in ("fp16_comparison", "binary64_v1_comparison")
        d.write(out / "failure.json", {
            "phase": phase, "active": active, "error": str(error),
            "failure_taxonomy": "candidate_numerical_gate" if numerical else "evaluator_no_completion",
            "root_cause_hypothesis": (
                "The actual propagated candidate exceeds an unchanged numerical gate; local arithmetic causality is not established."
                if numerical else "The named execution or binding boundary did not complete; no RTL correctness conclusion follows."),
            "regression": "Retain the first failing boundary and original gates; repair only a justified changed dependency in a fresh attempt.",
            "gates": gates, "transactions": transactions, "phase_times": timings,
            "elapsed_seconds": time.monotonic() - started,
            "normal_host_review": "REQUIRED", "no_expansion": True})
        raise
    finally:
        d.write(out / "output_manifest.json", {
            "artifacts": [d.record(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and "obj" not in p.parts],
            "normal_host_review": "PENDING"})


if __name__ == "__main__":
    main()
