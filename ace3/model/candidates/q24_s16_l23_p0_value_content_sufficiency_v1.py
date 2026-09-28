"""One non-admitting L23/P0 value substitution; only S10-S18 and final head run."""

import argparse
from contextlib import contextmanager, redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_impact_census_v1 as retained
from ace3.model.candidates import q24_s16_final_head_from_l23_coordinate62_suffix_execution_v1 as final
from ace3.model.candidates import q24_s16_l23_from_l22_coordinate62_suffix_execution_v1 as layer
from ace3.model.candidates import diagnostic_capture_v1 as capture
from ace3.model import stage11_current_runtime_release as release


ROOT = final.ROOT
NAME = "q24_s16_l23_p0_value_content_sufficiency_v1"
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
OUTPUT = ROOT / "build" / (NAME + "_attempt001")
MISSION = "8ee0b3aaeecb"
AUTHORITY = ROOT / ".argus/live/manager-authority" / f"stage11-science-{MISSION}-attempt001"
CONTROLS = final.CONTROLS
native = layer.producer.native.candidate
require, same, record, read_bound = final.require, final.same, final.record, final.read_bound
PROTECTED = (
    "input_hidden", "input_i", "input_z", "input_cache_k", "input_cache_v",
    "output_cache_k", "output_cache_v",
) + tuple(f"stage{s:02d}" for s in range(10) if s != 7)
PAIR = (34319, 13)
FLAGS = {
    "candidate_admitted": False, "policy_adopted": False, "successor_published": False,
    "strict_FP16_state_claim": False, "new_token_claim": False, "full_model_claim": False,
    "historical_failures_preserved": True, "original_global_reference_unchanged": True,
    "reference_recomputation": False, "stage07_native_operator_validated": False,
    "full_vocabulary_head_recomputation": False, "token_selection": False,
}
BOUNDARY = (
    "Conditional rescue requires all nine substituted S10-S18 gates and strict "
    "pair[34319,13] movement toward both original references. Only those two head "
    "rows run; this directional test is not an admission threshold. "
    "Historical L0-L22 and unmodified L23 FAIL remain unchanged. Q24 "
    "residual state is wider than FP16; native INT4/FP16 operators and retained FP16 "
    "KV remain fixed. Stage07 is an explicit non-cache-writing intervention, not "
    "validation of an unchanged native V operator. No unique-root-cause or "
    "failure-creation sufficiency claim. No prefix/admission/reference replay, "
    "hardware, GPU, RTL, token publication or full-model admission."
)


def binding_requirements():
    """Describe the existing issuer inputs and this candidate's launch interface."""
    pin = {"path": "canonical absolute file path", "bytes": "integer byte count",
           "sha256": "SHA256 of actual bytes"}
    return {
        "authority_path": str(AUTHORITY / "build-authorization.json"),
        "prepared_owner_path": str(AUTHORITY / "manager-owner-prepared.json"),
        "envelope_path": str(AUTHORITY / "issued/envelope.json"),
        "science_input_path": str(ROOT / ".argus/live/manager-inputs"
                                  / f"stage11-science-{MISSION}.json"),
        "pin_schema": pin,
        "authority_schema": {
            "mission_id": MISSION, "authority_id": "matching Manager authority identity",
            "life_dir": str(release.LIFE), "issuer": "manager", "expected_attempt": 1,
            "authorized_science": True, "scientific_run_budget": 1,
            "authorized_root": "science_input.capture_root",
            "manager_inputs": {"issuer": pin, "science_input": pin},
        },
        "prepared_owner_schema": {
            "mission_id": MISSION, "authority_id": "authority.authority_id",
            "state": "PREPARED_BEFORE_NATIVE_HANDOFF", "build_authorization": pin,
        },
        "science_input_schema": {
            "mission_id": MISSION, "authority_id": "authority.authority_id",
            "attempt": 1, "scientific_run_budget": 1,
            "candidate": pin, "candidate_tests": pin, "runner": pin,
            "capture_implementation": [pin], "release_source": pin,
            "capture_root": "fresh ignored build capture root",
            "contract": pin, "seal": pin, "operational_release_selection": pin,
            "selected_release_mission": "independently accepted release mission",
            "runner_review": {"mission_id": "independent Reviewer mission",
                             "latest": pin, "review": pin, "checkpoint": pin, "mission": pin},
            "contract_review": {"mission_id": "independent Reviewer mission",
                               "latest": pin, "review": pin, "checkpoint": pin, "mission": pin},
            "accepted_review": {"mission_id": "independent Reviewer mission",
                               "latest": pin, "review": pin, "checkpoint": pin},
            "accepted_identity": {key: pin for key in (
                "authority_bundle", "launcher_capture", "launcher_expected_identity",
                "launcher_identity", "proposal", "receipt_index", "terminal_receipt",
            )},
        },
        "envelope_schema": {
            "schema_version": 1, "mission_id": MISSION, "attempt": 1,
            "scientific_run_budget": 1, "declaration": pin, "issuance": pin,
            "review": "accepted_review", "current_release_contract": pin,
            "current_release_review": "contract_review",
            "operational_release_selection": pin, "audit_limit": release.AUDIT_LIMIT,
        },
        "declaration_launch_argv": [
            sys.executable, "-B", "-m", "ace3.model.candidates." + NAME,
            "--check", "--output", "<fresh absolute ignored build output directory>",
        ],
        "issuer_contract": str(ROOT / ".argus/live/manager-tools"
                               / "publish-stage11-science-issuance-v1.py"),
        "output_rule": "exact issued launch --output; no historical/default output reuse",
    }


def manager_binding(output=None, *, live=True):
    pins = []

    def document(pin):
        value = json.loads(read_bound(pin))
        pins.append(pin)
        return value

    authority = document(record(AUTHORITY / "build-authorization.json"))
    prepared = document(record(AUTHORITY / "manager-owner-prepared.json"))
    require(authority["mission_id"] == MISSION and authority["issuer"] == "manager"
            and authority["life_dir"] == str(release.LIFE)
            and type(authority["expected_attempt"]) is int and authority["expected_attempt"] == 1
            and authority["authorized_science"] is True
            and type(authority["scientific_run_budget"]) is int
            and authority["scientific_run_budget"] == 1, "Manager authority identity/budget differs")
    same(prepared["build_authorization"], pins[0], "prepared authority bytes differ")
    require(prepared["mission_id"] == MISSION
            and prepared["authority_id"] == authority["authority_id"]
            and prepared["state"] == "PREPARED_BEFORE_NATIVE_HANDOFF",
            "prepared Manager owner differs")
    issuer = authority["manager_inputs"]["issuer"]
    same(issuer, record(ROOT / ".argus/live/manager-tools"
                       / "publish-stage11-science-issuance-v1.py"), "Manager issuer bytes differ")
    pins.append(issuer)
    inputs = document(authority["manager_inputs"]["science_input"])
    require(inputs["mission_id"] == MISSION
            and inputs["authority_id"] == authority["authority_id"]
            and type(inputs["attempt"]) is int and inputs["attempt"] == 1
            and type(inputs["scientific_run_budget"]) is int
            and inputs["scientific_run_budget"] == 1
            and inputs["capture_root"] == authority["authorized_root"],
            "science input identity/budget/root differs")
    for key, path in (("candidate", SOURCE), ("candidate_tests", TEST),
                      ("release_source", Path(release.__file__)),
                      ("runner", ROOT / ".argus/live/manager-tools/run-stage11-science-once-v1.py")):
        same(inputs[key], record(path), "Manager source binding differs: " + key)
        pins.append(inputs[key])
    same(inputs["capture_implementation"], capture.implementation_pins(),
         "capture implementation differs")
    pins.extend(inputs["capture_implementation"])
    envelope = document(record(AUTHORITY / "issued/envelope.json"))
    require(envelope["mission_id"] == MISSION
            and type(envelope["attempt"]) is int and envelope["attempt"] == 1
            and type(envelope["scientific_run_budget"]) is int
            and envelope["scientific_run_budget"] == 1, "issued identity/budget differs")
    declaration = document(envelope["declaration"])
    issuance = document(envelope["issuance"])
    same(issuance["declaration"], envelope["declaration"], "issuance/declaration splice")
    for key, input_key in (("current_release_contract", "contract"),
                           ("current_release_review", "contract_review"),
                           ("operational_release_selection", "operational_release_selection"),
                           ("review", "accepted_review")):
        expected = inputs[input_key]
        if key == "current_release_review":
            expected = {k: expected[k] for k in ("mission_id", "latest", "review", "checkpoint")}
        same(envelope[key], expected, "Manager envelope/input splice: " + key)
    if live:
        same(release.validate_execution_envelope(envelope), declaration,
             "Manager envelope validation differs")
    launch = declaration["launch"]
    argv = launch["argv"]
    require(len(argv) == 7 and argv[:6] == [
        sys.executable, "-B", "-m", "ace3.model.candidates." + NAME, "--check", "--output",
    ], "issued launch must explicitly bind this candidate and --output")
    bound_output = Path(argv[6])
    require(bound_output.is_absolute() and bound_output.resolve() == bound_output
            and bound_output.parent == ROOT / "build" and bound_output != OUTPUT,
            "noncanonical or historical Manager output")
    if output is not None:
        same(str(output), str(bound_output), "requested output differs from Manager launch")
    same(declaration["contract"]["sources"]["current_diagnostic"], record(SOURCE),
         "declaration candidate source differs")
    same(declaration["contract"]["sources"]["current_tests"], record(TEST),
         "declaration test source differs")
    if live:
        require(not bound_output.exists() and not bound_output.is_symlink(),
                "occupied output; zero dispatch")
    return {"output": bound_output, "pins": pins, "envelope": envelope,
            "declaration": declaration, "capture_root": ROOT / inputs["capture_root"]}


def captured_launch(binding):
    declaration = binding["declaration"]
    launch = declaration["launch"]
    require(Path.cwd() == ROOT and sys.dont_write_bytecode and not sys.flags.optimize
            and launch["cwd"] == str(ROOT) and launch["uid"] == os.getuid() == 1000,
            "source/interpreter/account gate")
    same(launch["environment"], dict(os.environ), "live launch environment differs")
    out = Path(os.readlink("/proc/self/fd/1"))
    require(out == binding["capture_root"] / "run/check.stdout" and out.resolve() == out
            and os.readlink("/proc/self/fd/2") == str(out.with_name("check.stderr")),
            "Manager shared byte-exact capture required")
    preflight = json.loads(out.with_name("check.environment.json").read_bytes())
    same(preflight["execution_authorization"], binding["envelope"], "capture authority differs")
    same(preflight["sources"], [record(SOURCE), record(TEST)], "capture sources differ")
    same(preflight["launch_constraints"], launch["constraints"], "capture constraints differ")
    claim = capture.retained_document(declaration["claim"]["pin"])
    same(release.stable_claim(preflight["normal_running_claim"]),
         release.stable_claim(claim["native_record"]), "capture native claim differs")
    same(preflight["native_backlog"], str(release.LIFE / "backlog.jsonl"),
         "capture backlog differs")
    same(preflight["capture_implementation"], capture.implementation_pins(),
         "capture implementation differs")
    require(preflight["role"] == "engineer" and preflight["independent_host_review"] == "REQUIRED"
            and preflight["model_or_service_calls_authorized"] == 0
            and preflight["command_budget"] == {"compile": 1, "pytest": 1, "check": 1}
            and preflight["mission_id"] == MISSION, "capture role/review/budget differs")
    same(preflight["environment"], dict(os.environ), "capture environment differs")
    capture.verify_command(
        capture.decode_command_file(out.with_name("check.command.txt").read_bytes()),
        json.loads(out.with_name("check.argv.json").read_bytes()),
        preflight["environment"], launch["argv"], launch["environment"])
    capture.verify_command(launch["command"],
                           [sys.executable, "-B", "-m", "ace3.model.candidates." + NAME, *sys.argv[1:]],
                           dict(os.environ), launch["argv"], launch["environment"])


def archive(pin):
    with np.load(io.BytesIO(read_bound(pin)), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate archive fields")
        return {key: saved[key] for key in saved.files}


def prepare(actual, reference):
    for arrays in (actual, reference):
        for stage, size in ((3, 128), (6, 128), (7, 128), (9, 14)):
            layer.prior.local.finite_words(arrays[f"stage{stage:02d}"], (size,))
        require(np.array_equal(arrays["stage03"], arrays["stage07"]),
                "retained/reference V source lineage changed")
        require(np.array_equal(arrays["stage09"], np.full(14, 0x3c00, dtype="<u2")),
                "P0 probabilities are not all exactly one")
    layer.preflight.parent.check_state(actual)
    for kind, stage in (("k", 6), ("v", 7)):
        require(actual["input_cache_" + kind].shape == (0, 128)
                and actual["output_cache_" + kind].shape == (1, 128)
                and np.array_equal(actual["output_cache_" + kind][0],
                                   actual[f"stage{stage:02d}"]),
                "selected P0 KV/source identity changed")
    parent = {key: actual[field].copy() for key, field in
              (("h", "input_hidden"), ("i", "input_i"), ("z", "input_z"))}
    arrays = {key: actual[key].copy() for key in PROTECTED}
    arrays["stage07"] = reference["stage07"].copy()
    for value in arrays.values():
        value.flags.writeable = False
    return parent, arrays


def protected(actual, arrays, reference):
    for key in PROTECTED:
        require(actual[key].dtype == arrays[key].dtype
                and np.array_equal(actual[key], arrays[key]), "protected field changed: " + key)
    require(np.array_equal(arrays["stage07"], reference["stage07"]),
            "substituted value changed")


def suffix_stages(tensors, parent, arrays):
    """Keep the pinned native producer untouched; reuse its arithmetic primitives."""
    prefix = "model.layers.23."
    scratch = None
    for stage in range(10, 19):
        if stage == 10:
            value = native.decoded(arrays["stage07"]).reshape(2, 64).repeat_interleave(7, dim=0)
            word = native.rne((native.decoded(arrays["stage09"])[:, None] * value).reshape(-1))
        elif stage in (11, 14, 15, 17):
            name, source = {
                11: ("self_attn.o_proj", 10), 14: ("mlp.gate_proj", 13),
                15: ("mlp.up_proj", 13), 17: ("mlp.down_proj", 16),
            }[stage]
            word = native.projection(tensors, prefix + name, arrays[f"stage{source:02d}"])
        elif stage == 12:
            scratch = native.state.add(parent, arrays["stage11"])
            arrays["scratch_i"], arrays["scratch_z"] = scratch["i"], scratch["z"]
            word = scratch["h"]
        elif stage == 13:
            value = native.decoded(arrays["stage12"])
            gamma = native.torch.from_numpy(
                tensors[prefix + "post_attention_layernorm.weight"].astype("<f8"))
            word = native.rne(
                (value * native.torch.rsqrt(value.square().mean() + 1e-6)) * gamma)
        elif stage == 16:
            value = (native.functional.silu(native.decoded(arrays["stage14"]))
                     * native.decoded(arrays["stage15"]))
            arrays["s16_unrounded_binary64"] = value.numpy().copy()
            word = native.toward_zero(arrays["s16_unrounded_binary64"])
        else:
            require(scratch is not None, "missing intervened residual scratch")
            successor = native.state.add(scratch, arrays["stage17"])
            arrays["output_i"], arrays["output_z"] = successor["i"], successor["z"]
            word = successor["h"]
        arrays[f"stage{stage:02d}"] = word
        yield stage


def pair_logits(words, weights):
    layer.prior.local.finite_words(words, (896,))
    require(weights.shape == (2, 896) and weights.dtype.str == "<f2"
            and np.all(np.isfinite(weights)), "only the two finite pair head rows may run")
    return final.logits(words, weights)


def margin_rows(baseline, substituted, references):
    layer.prior.local.finite_words(substituted, (2,))
    actual = baseline[list(PAIR)].view("<f2").astype("<f8")
    changed = substituted.view("<f2").astype("<f8")
    rows = []
    for branch in ("fp16", "binary64"):
        reference = references["logits_" + branch][list(PAIR)]
        if branch == "fp16":
            reference = reference.view("<f2").astype("<f8")
        require(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)),
                "nonfinite retained pair")
        exact = lambda x: Fraction.from_float(float(x))
        a = exact(actual[0]) - exact(actual[1])
        s = exact(changed[0]) - exact(changed[1])
        r = exact(reference[0]) - exact(reference[1])
        rows.append({
            "pair": list(PAIR), "reference_branch": branch,
            "retained_actual_margin": str(a), "retained_reference_margin": str(r),
            "retained_margin_change": str(a - r), "substituted_actual_margin": str(s),
            "substituted_margin_change": str(s - r), "paired_intervention_delta": str(s - a),
        })
    return rows


def scientific_status(rows):
    same([row["control"] for row in rows], list(CONTROLS), "incomplete/reordered controls")
    directions = []
    for row in rows:
        same([r["stage"] for r in row["reports"]], list(range(10, 19)),
             "incomplete affected cone")
        require(all(r["status"] in ("PASS", "FAIL") for r in row["reports"]),
                "invalid acceptance result")
        margins = row["paired_final_margins"]
        same([(r["pair"], r["reference_branch"]) for r in margins],
             [(list(PAIR), branch) for branch in ("fp16", "binary64")],
             "incomplete/reordered terminal references")
        for margin in margins:
            actual = Fraction(margin["retained_actual_margin"])
            reference = Fraction(margin["retained_reference_margin"])
            changed = Fraction(margin["substituted_actual_margin"])
            for key, value in (("retained_margin_change", actual - reference),
                               ("substituted_margin_change", changed - reference),
                               ("paired_intervention_delta", changed - actual)):
                require(Fraction(margin[key]) == value, "inconsistent terminal margin")
            directions.append((changed - actual) * (reference - actual) > 0)
    return ("SUPPORTED" if all(directions)
            and all(r["status"] == "PASS" for row in rows for r in row["reports"])
            else "REJECTED")


@contextmanager
def write_scope(out, audit):
    import builtins

    def permitted(path, writing):
        if writing:
            require(not isinstance(path, int), "descriptor write not authorized")
            path = Path(path).absolute()
            require(path.parent == out and path.resolve() == path,
                    "write outside fresh output root")
            audit["file_write_opens"] += 1

    def wrap(original):
        def opening(path, mode="r", *args, **kwargs):
            permitted(path, isinstance(mode, str) and any(c in mode for c in "wax+"))
            return original(path, mode, *args, **kwargs)
        return opening

    original_os_open = os.open

    def os_open(path, flags, *args, **kwargs):
        permitted(path, flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        return original_os_open(path, flags, *args, **kwargs)

    with patch.object(builtins, "open", wrap(builtins.open)), \
            patch.object(io, "open", wrap(io.open)), patch.object(os, "open", os_open):
        yield


def predecessor_review():
    directory = layer.preflight.HANDOFFS / "9b2a69ef5a4e"
    latest = json.loads((directory / "latest.json").read_bytes())
    require(latest["kind"] == "handoff_ref", "predecessor is not reviewed")
    path = Path(latest["handoff"]["path"])
    require(path.parent == directory and path.resolve() == path, "review path escaped")
    review = json.loads(path.read_bytes())
    layer.preflight.parent.matrix.check_review(review, "9b2a69ef5a4e", review["round"])
    backlog = [json.loads(line) for line in layer.preflight.BACKLOG.read_text().splitlines()
               if line.strip()]
    rows = [row for row in backlog if row["id"] == "9b2a69ef5a4e"]
    require(len(rows) == 1 and rows[0]["status"] == "done"
            and rows[0]["outcome"]["review_status"] == "done"
            and rows[0]["finished_ts"] >= review["created_at"],
            "predecessor review lacks terminal corroboration")
    return record(path)


def focused_tests():
    import importlib.util

    spec = importlib.util.spec_from_file_location(NAME + "_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    loaded = suite.countTestCases()
    require(loaded > 0, "focused tests loaded an empty suite")
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    report = {
        "loaded": loaded, "executed": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "unexpected_successes": len(result.unexpectedSuccesses),
        "synthetic_stage10_invocations": 1, "retained_control_dispatches": 0,
        "raw_output": output.getvalue(),
    }
    require(result.testsRun == loaded and result.wasSuccessful()
            and all(report[key] == 0 for key in (
                "failures", "errors", "skipped", "unexpected_successes")),
            "focused tests failed:\n" + json.dumps(report, sort_keys=True))
    return report


def runtime_sources(declaration, *, message="Manager runtime source binding differs"):
    """Authenticate the same complete census before and after execution."""
    origins = final.source_context()
    origins["value_sufficiency_source"], origins["value_sufficiency_test"] = record(SOURCE), record(TEST)
    for role in ("stage13_suffix", "stage13_tests"):
        origins[role] = record(Path(declaration["runtime_sources"][role]["path"]))
    expected = declaration["runtime_sources"]
    changes = {
        role: {"expected": expected.get(role), "actual": origins.get(role)}
        for role in sorted(expected.keys() | origins.keys())
        if role not in expected or role not in origins or expected[role] != origins[role]
    }
    require(not changes, message + ": " + json.dumps(changes, sort_keys=True))
    for role in ("stage13_suffix", "stage13_tests"):
        read_bound(origins[role])
    return origins


def execute(tests=None, output=None):
    binding = manager_binding(output)
    output = binding["output"]
    require(not output.exists() and not output.is_symlink(), "occupied output; zero dispatch")
    captured_launch(binding)
    require(output.resolve() == output and output.parent == ROOT / "build"
            and Path(__file__).resolve() == SOURCE, "noncanonical source/output")
    ignored = final.subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--", str(output)],
        capture_output=True, check=False)
    require(ignored.returncode == 0, "output not confirmed ignored")
    native.torch.set_num_threads(1)
    require(str(native.torch.tensor(0).device) == "cpu", "non-CPU default device")
    origins = runtime_sources(binding["declaration"])
    output.mkdir(exist_ok=False)
    audit = {"controls": [], "stage_dispatches": [], "final_rmsnorm_invocations": 0,
             "lm_head_invocations": 0, "top_k_invocations": 0, "forbidden_calls": 0,
             "file_write_opens": 0, "native_L0_L22_invocations": 0,
             "native_S0_S9_invocations": 0, "retained_evidence_writes": 0,
             "reference_producer_invocations": 0, "external_invocations": 0}
    rows, artifacts = [], []

    def save_json(name, payload):
        with (output / name).open("x") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return record(output / name)

    try:
        with write_scope(output, audit):
            artifacts.append(save_json("command.json", {
                "argv": sys.argv, "cwd": str(ROOT), "uid": os.getuid(),
                "executable": sys.executable, "origins": origins,
                "manager_binding": binding["pins"],
                "host_managed_controls": "unchanged role/model/account/budget/access/concurrency locks",
            }))
            compiled = []
            for pin in {p["path"]: p for p in origins.values()}.values():
                compile(read_bound(pin), pin["path"], "exec")
                compiled.append(pin)
            if tests is None:
                tests = focused_tests()
            tests["compiled"] = compiled
            artifacts.append(save_json("compile_tests.json", tests))
            predecessor = predecessor_review()
            started = time.monotonic()
            baseline, baseline_arrays, references, pins = retained.authenticate()
            summary = baseline["preflight"]
            contract = binding["declaration"]["contract"]
            same(contract["controls"], list(CONTROLS), "Manager controls differ")
            same(contract["thresholds"], summary["thresholds"], "Manager thresholds differ")
            for row in baseline["controls"]:
                same(contract["inputs"][row["control"]], row["parent"]["terminal_archive"],
                     "Manager retained control differs")
            for branch in ("fp16", "binary64"):
                same(contract["references"]["original_input_L23_" + branch],
                     summary["L23_original_reference"]["reference"][branch],
                     "Manager original L23 reference differs")
                require(contract["references"]["original_input_final_" + branch] in pins,
                        "Manager original final reference differs")
            tensors, trajectory, reference, checkpoint = layer.load_inputs({"summary": summary})
            operands = final.load_operands(summary)
            operands[1] = operands[1][list(PAIR)].copy()
            operands[1].flags.writeable = False
            # Rebind checkpoint/tokenizer/tied-tensor identities without running their producers.
            same(final.preflight.bind_assets(checkpoint), summary["assets"], "asset identity changed")
            require(summary["L23_original_reference"]["reference"]["prior_kv"] == "own empty P0",
                    "reference KV/source scope changed")
            same(summary["L23_original_reference"]["reference"]["fp16"],
                 summary["final_reference"]["reference"]["input_fp16"],
                 "reference input lineage splice")
            for tensor in tensors.values():
                tensor.flags.writeable = False
            prepared = []
            for row in baseline["controls"]:
                actual = archive(row["parent"]["terminal_archive"])
                parent, arrays = prepare(actual, trajectory)
                prepared.append((row, actual, parent, arrays))
            same([r["control"] for r, _, _, _ in prepared], list(CONTROLS),
                 "nine-control source order changed")
            authentication_seconds = time.monotonic() - started
            with final.no_dispatch(audit):
                for row, actual, parent, arrays in prepared:
                    label = row["control"]
                    audit["controls"].append(label)
                    reports, timings = [], {"native_seconds": 0.0, "oracle_seconds": 0.0}
                    stages = suffix_stages(tensors, parent, arrays)
                    for expected_stage in range(10, 19):
                        began = time.monotonic()
                        audit["stage_dispatches"].append([label, 23, 0, expected_stage])
                        stage = next(stages)
                        require(stage == expected_stage, "out-of-cone dispatch")
                        timings["native_seconds"] += time.monotonic() - began
                        began = time.monotonic()
                        expected = layer.expected_stage(stage, arrays, parent, tensors)
                        reports.append(layer.stage_report(stage, arrays, trajectory, reference, expected))
                        if expected is not None:
                            arrays[f"local_reference_stage{stage:02d}"] = expected
                        timings["oracle_seconds"] += time.monotonic() - began
                    require(next(stages, None) is None, "extra suffix stage")
                    protected(actual, arrays, trajectory)
                    require(np.array_equal(layer.prior.rtz_reference(arrays["s16_unrounded_binary64"]),
                                           arrays["stage16"]), "S16 RTZ boundary changed")
                    began = time.monotonic()
                    audit["final_rmsnorm_invocations"] += 1
                    arrays["final_rmsnorm"], details = final.rmsnorm(arrays["stage18"], operands[0])
                    timings["final_rmsnorm_seconds"] = time.monotonic() - began
                    began = time.monotonic()
                    audit["lm_head_invocations"] += 1
                    arrays["final_logits"] = pair_logits(arrays["final_rmsnorm"], operands[1])
                    timings["lm_head_seconds"] = time.monotonic() - began
                    margins = margin_rows(baseline_arrays[label]["logits"], arrays["final_logits"], references)
                    payload = output / f"{label}.npz"
                    with payload.open("xb") as stream:
                        np.savez(stream, **arrays, retained_stage07=actual["stage07"])
                    pin = record(payload)
                    artifacts.append(pin)
                    entry = {
                        "control": label, "reports": reports, "head_pair": list(PAIR),
                        "paired_final_margins": margins, "parent": row["parent"], "raw_payload": pin,
                        "substitution": {
                            "source": summary["L23_original_reference"]["reference"]["fp16"],
                            "field": "stage07", "shape": [128], "dtype": "<u2",
                            "changed_components": int(np.count_nonzero(actual["stage07"] != arrays["stage07"])),
                            "retained_KV_unchanged": True, "retained_state_unchanged": True,
                            "stage03_and_cache_not_substituted": True,
                        },
                        "rmsnorm_integer_details": details, "timing_seconds": timings,
                        "status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL",
                    }
                    rows.append(entry)
                    print(json.dumps({k: entry[k] for k in
                                      ("control", "status", "paired_final_margins",
                                       "substitution", "timing_seconds")}), flush=True)
            runtime_sources({"runtime_sources": origins}, message="source changed during execution")
            for pin in pins:
                read_bound(pin)
            for pin in binding["pins"]:
                read_bound(pin)
            for pin in (record(SOURCE), record(TEST)):
                same(pin, origins["value_sufficiency_source" if pin["path"] == str(SOURCE)
                                  else "value_sufficiency_test"], "new source changed")
            status = scientific_status(rows)
            require(len(audit["stage_dispatches"]) == 81
                    and audit["final_rmsnorm_invocations"] == audit["lm_head_invocations"] == 9
                    and audit["top_k_invocations"] == 0
                    and audit["forbidden_calls"] == 0 and audit["file_write_opens"] == 11,
                    "dispatch/write census mismatch")
            result = {
                "diagnostic_id": NAME, "status": status, "controls": rows, "origins": origins,
                "predecessor_review": predecessor, "input_pins": pins, "preflight": summary,
                "thresholds": summary["thresholds"], "final_reference": summary["final_reference"],
                "audit": audit, "artifacts": artifacts, "output": str(output),
                "manager_binding": binding["pins"],
                "expected_final_file_count": 12, "authentication_seconds": authentication_seconds,
                "flags": FLAGS, "claim_boundary": BOUNDARY, "normal_host_review": "REQUIRED",
                "decision": ("conditional all-nine directional rescue; not unique cause"
                             if status == "SUPPORTED" else
                             "reject all-nine directional rescue; no root-cause claim"),
            }
            save_json("result.json", result)
            require(len(list(output.iterdir())) == 12 and audit["file_write_opens"] == 12,
                    "final write census mismatch")
            print(json.dumps({"status": status, "output": str(output), "audit": audit,
                              "dispatch_and_write_audit": {"scientific_invocations": 1},
                              "decision": result["decision"], "flags": FLAGS,
                              "claim_boundary": BOUNDARY}), flush=True)
    except Exception as error:
        with write_scope(output, audit):
            save_json("failure.json", {"status": "UNKNOWN", "error_type": type(error).__name__,
                                      "error": str(error), "audit": audit, "controls": rows,
                                      "flags": FLAGS, "automatic_replay_authorized": False})
        raise


def check(output=None):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec")
        compiled.append(record(path))
    tests = focused_tests()
    print(json.dumps({"phase": "compile_tests", "compiled": compiled, **tests}), flush=True)
    execute(tests, output)


def validate(output=None):
    """Authenticate stored evidence without rerunning any numerical producer."""
    binding = manager_binding(output, live=False)
    output = binding["output"]
    result = json.loads((output / "result.json").read_bytes())
    same(result["manager_binding"], binding["pins"], "stored Manager binding differs")
    for key, path in (("value_sufficiency_source", SOURCE), ("value_sufficiency_test", TEST)):
        same(result["origins"][key], record(path),
             "retained source differs; historical result cannot validate this candidate")
    same(result["flags"], FLAGS, "claim boundary flags changed")
    same(result["output"], str(output), "output binding changed")
    names = {"result.json"}
    for pin in result["artifacts"]:
        path = Path(pin["path"])
        require(path.parent == output and path.resolve() == path, "artifact outside output")
        require(path.name not in names, "duplicate artifact")
        names.add(path.name)
        read_bound(pin)
    same({path.name for path in output.iterdir()}, names, "output file census changed")
    for pin in result["origins"].values():
        read_bound(pin)
    for pin in result["input_pins"]:
        read_bound(pin)
    read_bound(result["predecessor_review"])
    summary = result["preflight"]
    trajectory = archive(summary["L23_original_reference"]["reference"]["fp16"])
    reference = np.load(io.BytesIO(read_bound(
        summary["L23_original_reference"]["reference"]["binary64"])), allow_pickle=False)
    for row in result["controls"]:
        arrays = archive(row["raw_payload"])
        actual = archive(row["parent"]["terminal_archive"])
        prepare(actual, trajectory)
        protected(actual, arrays, trajectory)
        same(row["head_pair"], list(PAIR), "head pair changed")
        layer.prior.local.finite_words(arrays["final_logits"], (2,))
        for stage, report in zip(range(10, 19), row["reports"], strict=True):
            expected = arrays.get(f"local_reference_stage{stage:02d}")
            same(layer.stage_report(stage, arrays, trajectory, reference, expected),
                 report, "stored numerical gate changed")
    status = scientific_status(result["controls"])
    same(result["status"], status, "stored scientific classification changed")
    print(json.dumps({"status": status, "output": str(output),
                      "normal_host_review": "REQUIRED", "flags": FLAGS}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="must equal the fresh path in the issued Manager launch; no default")
    args = parser.parse_args()
    events = io.StringIO()
    try:
        with redirect_stdout(events):
            if args.validate:
                validate(args.output)
            elif args.check:
                check(args.output)
            else:
                execute(output=args.output)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "UNKNOWN", "error_type": type(error).__name__,
                          "error": str(error), "normal_host_review": "REQUIRED",
                          "expected_binding": binding_requirements(),
                          "events": [json.loads(line) for line in events.getvalue().splitlines()],
                          "automatic_replay_authorized": False, "flags": FLAGS}), flush=True)
        raise SystemExit(1) from error
    records = [json.loads(line) for line in events.getvalue().splitlines()]
    print(json.dumps({**records[-1], "events": records[:-1]}), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("ace3.model.candidates." + NAME).main()
