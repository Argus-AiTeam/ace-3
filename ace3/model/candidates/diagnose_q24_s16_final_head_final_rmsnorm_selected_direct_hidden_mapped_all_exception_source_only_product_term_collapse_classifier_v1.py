"""Stdout-only exact source-only classification of 35 reviewed product differences."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_source_only_product_term_collapse_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_component_delta_factorizer_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-component-deltas-19029437e84e-attempt001"
REVIEW_ROOT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(CAPTURE_ROOT / "check.stdout", 20112157,
                  "08a87e30fac1924bbf48a510f3587b99f35aa3951c533e995d08f533cf7195b8"),
    "capture": pin(CAPTURE_ROOT / "capture.json", 31277,
                   "f1d00e9ce420bdcce129bf5719d64df563f6283e5562cf57e608ac5e3d145fc4"),
    "review": pin(REVIEW_ROOT / "19029437e84e/round-0001.json", 690,
                  "6b30a9a3ecfed1d7a7aec3c460e441ea1ff2654ee791fecf6b8def0f8092c8c5"),
}
SOURCE_PINS = [
    pin(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py"), 29750,
        "f3f42815abdb8c2a4e014106259977c9c98045a62e3d2128e415da389d3c5179"),
    pin(ROOT / "tests" / ("test_" + PARENT_NAME + ".py"), 19073,
        "e8ef102b2928827db3a1c85c1c07e59674d7cc9425344edc5ed7d58ccae98090"),
]
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
TERMS = (
    "modal_weight_times_source_delta",
    "modal_source_times_weight_delta",
    "source_delta_times_weight_delta",
)
SIDES = ("modal_class", "mapped_all", "delta")
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch")
SCOPES = (
    ("coordinate", 34319, 319, "binary64"),
    ("coordinate", 319, 34319, "binary64"),
    ("coordinate_component", 34319, 13, "binary64"),
)
MODAL_CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "inherited_native",
)
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout",
            ".stderr", ".whole-command.log")
SOURCE_ONLY = "source_delta_only"
ZERO_MOVEMENT = "zero_source_no_weight_movement"
BOUNDARY = (
    "Descriptive exact retained-rational product-term collapse only, for mapped_all "
    "minus frozen_inherited in the same three reviewed unstable partitions. "
    "Source-delta-only is an algebraic classification, not a causal attribution, "
    "intervention, performance diagnosis, repair or admission claim. Comparator "
    "equivalence does not imply component/source equivalence of the other seven "
    "modal controls. Only reviewed 190 stdout/capture/source/test/review and capture "
    "sidecars are opened; no producer is imported or replayed. Nested original-input "
    "independently propagated global references, exact thresholds, source/operand/"
    "state/KV/lineage gates, accepted evidence and historical failures are unchanged. "
    "No prefix/admission/reference replay, native decoder, tensor decoding, RMSNorm, "
    "head/row-dot or other model operator; no closed-branch reopening or re-review. "
    "The row-319 missing 896-element pre-round producer and "
    "NOT_RETAINED_NO_RECONSTRUCTION binary64 internal stages remain limitations, "
    "not availability work. Q24 residual state is wider than FP16; native S16 RTZ, "
    "G128 asymmetric packed INT4, native GEMM nibble ordering, no qzero plus-one, "
    "FP16 scales/operator boundaries/KV remain unchanged. No strict-FP16-state "
    "W4A16, new-token/full-model admission, GPU/RTL/FPGA/hardware, precision/scale "
    "expansion or ACE2 changes. Host independent Reviewer REQUIRED."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def bound_bytes(binding):
    data = Path(binding["path"]).read_bytes()
    same(len(data), binding["bytes"], "retained byte count changed: " + binding["path"])
    same(hashlib.sha256(data).hexdigest(), binding["sha256"],
         "retained hash changed: " + binding["path"])
    return data


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


@contextmanager
def read_only(audit):
    active = True
    allowed = {
        str(SOURCE), str(TEST), *(p["path"] for p in PINS.values()),
        *(p["path"] for p in SOURCE_PINS),
        *(str(CAPTURE_ROOT / (label + suffix)) for label in LABELS for suffix in SUFFIXES),
    }

    def guard(event, args):
        if not active:
            return
        forbidden = False
        if event == "open":
            path, mode, flags = args
            forbidden = (not isinstance(path, (str, bytes, os.PathLike))
                         or os.fsdecode(path) not in allowed
                         or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT |
                                          os.O_TRUNC | os.O_APPEND)))
        elif event == "import":
            forbidden = True
        elif event.startswith(("subprocess.", "socket.", "ctypes.", "shutil.")):
            forbidden = True
        elif event in (
            "os.system", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn",
            "os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link",
            "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.utime",
        ):
            forbidden = True
        if forbidden:
            audit["forbidden_calls"] += 1
            raise RuntimeError("retained-only product collapse classifier forbids " + event)

    sys.addaudithook(guard)
    try:
        yield
    finally:
        active = False


def zero_counters(values):
    require(bool(values) and all(v is False or type(v) is int and v == 0
                                for v in values.values()), "forbidden retained dispatch/write/claim")


def validate_retained(retained):
    same((retained["diagnostic_id"], retained["status"]),
         (PARENT_NAME, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MAPPED_ALL_EXCEPTION_COMPONENT_DELTA_FACTORIZER"),
         "wrong retained producer/boundary")
    require(type(retained["version"]) is int and retained["version"] == 1, "version changed")
    same(retained["compiled_sources"], SOURCE_PINS, "source/test splice")
    node = retained
    for child in ("retained_667", "retained_3fd", "retained_0146", "retained_6184",
                  "retained_e795", "retained_449b", "retained_50f"):
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
        node = node[child]
    zero_counters(node["dispatch_and_write_audit"])
    zero_counters(node["flags"])
    report = retained["report"]
    require(report["descriptive_accounting_only"] is True
            and report["exact_rational_closure"] is True, "accounting boundary changed")
    same((report["unstable_partition_count"], report["coordinate_account_count"],
          report["weighted_component_account_count"]), (3, 5, 35), "retained census changed")
    same([tuple(p[k] for k in SCOPE_FIELDS) for p in report["partitions"]],
         list(SCOPES), "exact three scopes required")


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    same((review["kind"], review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("round_reviewed_handoff", "19029437e84e", "reviewer", "done"),
         "independent terminal 190 review required")
    capture = decode(bound_bytes(PINS["capture"]))
    require(capture["success"] is True and bool(capture["checks"])
            and all(v is True for v in capture["checks"].values()), "retained capture failed")
    same(capture["capture_directory"], str(CAPTURE_ROOT), "capture directory splice")
    preflight = capture["preflight"]
    same((preflight["cwd"], preflight["uid"], preflight["python"]),
         (str(ROOT), 1000, PYTHON), "retained workdir/account/interpreter gate")
    same(capture["sources_after"], preflight["sources"], "retained source drift")
    same(capture["sources_after"], SOURCE_PINS, "retained source/test census")
    same(capture["accepted_artifacts_after"], preflight["accepted_artifacts"],
         "retained accepted-artifact overwrite")
    for binding in SOURCE_PINS:
        bound_bytes(binding)
    same([r["label"] for r in capture["results"]], list(LABELS), "retained command census")
    stdout = None
    for result in capture["results"]:
        require(type(result["exit_status"]) is int and result["exit_status"] == 0
                and result["timed_out"] is False, "retained command failure")
        label = result["label"]
        same([p["path"] for p in result["files"]],
             [str(CAPTURE_ROOT / (label + s)) for s in SUFFIXES], "capture member splice")
        command, argv, environment, output, error, whole = [
            bound_bytes(p) for p in result["files"]]
        same(command, (result["command"] + "\n").encode(), "command binding")
        same(decode(argv), result["argv"], "argv binding")
        same(decode(environment), {k: preflight[k] for k in
                                  ("cwd", "uid", "python", "python_version", "environment")},
             "environment binding")
        same(error, b"", "retained stderr")
        same(whole, b"COMMAND\n" + command + b"ENVIRONMENT\n" + environment
             + b"\nSTDOUT\n" + output + b"\nSTDERR\n" + error
             + b"\nEXIT_STATUS=0\nTIMED_OUT=False\n", "whole-command binding")
        if label == "branch":
            same(output, b"argus/full-projection\n", "retained branch")
        if label == "check":
            module = "ace3.model.candidates." + PARENT_NAME
            same(result["command"],
                 f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {module} --check",
                 "retained check command")
            same(decode(argv), [PYTHON, "-B", "-m", module, "--check"], "retained check argv")
            same(result["files"][3], PINS["stdout"], "capture/stdout splice")
            stdout = output
    require(stdout is not None, "missing retained stdout")
    retained = decode(stdout)
    same(retained["command"], capture["results"][-1]["command"], "stdout command splice")
    validate_retained(retained)
    return retained, capture, review


def rational(text):
    require(type(text) is str, "rational must be an exact string")
    value = Fraction(text)
    same(str(value), text, "noncanonical rational")
    return value


def movement(value):
    same(set(value), set(SIDES), "movement field census")
    parsed = {side: rational(value[side]) for side in SIDES}
    same(parsed["mapped_all"] - parsed["modal_class"], parsed["delta"], "movement delta closure")
    return parsed


def classify_coordinate(account):
    weight = movement(account["weight"])
    same(weight["delta"], 0, "source-only collapse requires zero coordinate weight delta")
    same([c["component"] for c in account["components"]], list(COMPONENTS), "seven component census")
    components = []
    for component in account["components"]:
        source, weighted = [movement(component[k]) for k in ("source", "weighted")]
        for side in SIDES[:2]:
            same(weight[side] * source[side], weighted[side], "source/weight product closure")
        expected = (
            weight["modal_class"] * source["delta"],
            source["modal_class"] * weight["delta"],
            source["delta"] * weight["delta"],
        )
        same(set(component["product_delta_terms"]), set(TERMS), "product-term census")
        terms = [rational(component["product_delta_terms"][name]) for name in TERMS]
        same(terms, list(expected), "retained product-term identity")
        same(sum(terms, Fraction(0)), weighted["delta"], "product-difference closure")
        same(terms[1:], [0, 0], "two weight-delta terms must vanish")
        same(terms[0], weighted["delta"], "source-only weighted movement closure")
        require(component["exact_product_delta_closure"] is True, "parent product closure required")
        if weighted["delta"]:
            require(source["delta"] != 0 and weight["modal_class"] != 0, "nonzero source movement required")
            classification = SOURCE_ONLY
        else:
            same(source["delta"], 0, "zero weighted movement must not mask a source movement")
            classification = ZERO_MOVEMENT
        components.append({
            **component, "classification": classification,
            "exact_source_only_collapse": True, "zero_weight_delta_product_terms": 2,
        })
    signed = movement(account["signed"])
    for side in SIDES:
        same(sum((rational(c["weighted"][side]) for c in components), Fraction(0)),
             signed[side], "seven-component coordinate closure")
    require(account["exact_seven_component_closure"] is True, "parent coordinate closure required")
    return {**account, "components": components, "exact_zero_weight_delta": True}


def verify_partition(partition, parent):
    same(tuple(partition[k] for k in SCOPE_FIELDS), tuple(parent[k] for k in SCOPE_FIELDS),
         "parent partition identity")
    modal = {"canonical_control": MODAL_CONTROLS[0], "controls": list(MODAL_CONTROLS), "count": 8}
    for item in (partition, parent):
        same(item["modal_class"], modal, "canonical modal comparator class")
        same(item["exception_class"], {"controls": ["mapped_all"], "count": 1}, "exception class")
    coordinate_table = partition["table"] == "coordinate"
    identities = ({"coordinate": 62}, {"coordinate": 241}) if coordinate_table else (
        {"coordinate": 241, "component": COMPONENTS[-1]},
        {"coordinate": 241, "component": "mlp_stage17"},
    )
    same([a["coordinate"] for a in partition["coordinates"]],
         [62, 241] if coordinate_table else [241], "coordinate identity census")
    selected = {}
    same(set(partition["candidates"]), {"exception", "modal"}, "candidate census")
    for label, identity in zip(("exception", "modal"), identities, strict=True):
        candidate = partition["candidates"][label]
        same(candidate["identity"], identity, "candidate identity")
        for side in SIDES[:2]:
            same(parent["identity_membership"][side][label], identity, "parent candidate identity")
        account = next(a for a in partition["coordinates"] if a["coordinate"] == identity["coordinate"])
        selected[label] = {
            c["component"]: c["weighted"] for c in account["components"]
            if coordinate_table or c["component"] == identity["component"]
        }
        signed = {side: str(sum((rational(v[side]) for v in selected[label].values()), Fraction(0)))
                  for side in SIDES}
        same(candidate["signed"], signed, "candidate signed closure")
        require(candidate["exact_parent_closure"] is True, "parent candidate closure required")
        same(signed, {k: parent["field_comparisons"][label + ".signed"][k] for k in SIDES},
             "archetype candidate closure")
        same(signed["delta"], parent["contrast_vector"][label + "_signed_delta"], "parent candidate contrast")
    expected = {
        name: {side: str(rational(selected["exception"].get(name, {}).get(side, "0"))
                        - rational(selected["modal"].get(name, {}).get(side, "0")))
               for side in SIDES}
        for name in COMPONENTS
    }
    same(partition["comparator_components"], expected, "comparator component closure")
    comparator = {side: str(sum((rational(c[side]) for c in expected.values()), Fraction(0)))
                  for side in SIDES}
    same(partition["signed_comparator"], comparator, "signed comparator closure")
    same(comparator, {k: parent["field_comparisons"]["signed_comparator"][k] for k in SIDES},
         "archetype comparator closure")
    same(comparator["delta"], parent["contrast_vector"]["signed_comparator_delta"], "parent comparator contrast")
    require(partition["exact_parent_comparator_closure"] is True, "parent comparator closure required")
    same(partition["nonlinear_parent_fields"], {
        k: v for k, v in parent["field_comparisons"].items()
        if k not in ("exception.signed", "modal.signed", "signed_comparator")
    }, "nonlinear parent fields preserved")
    context = partition["selected_component_context"]
    if coordinate_table:
        same(context, None, "coordinate partition has no selected-component context")
    else:
        selected_names = [i["component"] for i in identities]
        remaining = [c for c in COMPONENTS if c not in selected_names]
        same(context["selected_components"], selected_names, "selected component identities")
        same(context["remaining_components"], remaining, "context component identities")
        account = partition["coordinates"][0]
        for field, names in (("selected_sum", selected_names), ("context_sum", remaining)):
            same(context[field], {
                side: str(sum((rational(c["weighted"][side]) for c in account["components"]
                               if c["component"] in names), Fraction(0))) for side in SIDES
            }, "selected/context component sum")
        same(context["coordinate_signed"], account["signed"], "context coordinate binding")
        for side in SIDES:
            same(rational(context["selected_sum"][side]) + rational(context["context_sum"][side]),
                 rational(account["signed"][side]), "selected-plus-context closure")
        require(context["exact_selected_context_closure"] is True, "parent context closure required")


def orientation(forward, reverse):
    weighted_checks = term_checks = 0
    for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
        same(left["coordinate"], right["coordinate"], "reversed coordinate identity")
        for field in ("weight", "signed"):
            for side in SIDES:
                same(rational(left[field][side]), -rational(right[field][side]), "coordinate/weight orientation")
        for a, b in zip(left["components"], right["components"], strict=True):
            same(a["component"], b["component"], "reversed component identity")
            same(a["source"], b["source"], "reversed source invariance")
            same(a["classification"], b["classification"], "orientation-invariant classification")
            for side in SIDES:
                same(rational(a["weighted"][side]), -rational(b["weighted"][side]), "weighted orientation")
                weighted_checks += 1
            for term in TERMS:
                same(rational(a["product_delta_terms"][term]), -rational(b["product_delta_terms"][term]),
                     "product-term orientation")
                term_checks += 1
    for name in COMPONENTS:
        for side in SIDES:
            same(rational(forward["comparator_components"][name][side]),
                 -rational(reverse["comparator_components"][name][side]), "comparator component orientation")
    for side in SIDES:
        same(rational(forward["signed_comparator"][side]), -rational(reverse["signed_comparator"][side]),
             "signed comparator orientation")
    same(forward["nonlinear_parent_fields"], reverse["nonlinear_parent_fields"], "nonlinear orientation invariance")
    return {"ordered_pairs": [[34319, 319], [319, 34319]],
            "component_signed_orientation_checks": weighted_checks,
            "product_term_signed_orientation_checks": term_checks,
            "exact_signed_negation": True, "source_deltas_identical": True,
            "classifications_identical": True, "nonlinear_parent_fields_identical": True}


def report(retained):
    validate_retained(retained)
    parent = retained["report"]
    archetype = retained["retained_667"]["report"]
    same(len(archetype["partitions"]), 3, "archetype partition census")
    partitions = []
    counts = {SOURCE_ONLY: 0, ZERO_MOVEMENT: 0}
    for partition, ancestor in zip(parent["partitions"], archetype["partitions"], strict=True):
        classified = {**partition, "coordinates": [classify_coordinate(a) for a in partition["coordinates"]]}
        verify_partition(classified, ancestor)
        census = {label: sum(c["classification"] == label for a in classified["coordinates"]
                            for c in a["components"]) for label in counts}
        same(tuple(census.values()), (8, 6) if partition["table"] == "coordinate" else (4, 3),
             "partition classification census")
        for label in counts:
            counts[label] += census[label]
        partitions.append({**classified, "classification_counts": census})
    coordinates = [a for p in partitions for a in p["coordinates"]]
    same((len(coordinates), sum(len(a["components"]) for a in coordinates)), (5, 35), "account census")
    same(tuple(counts.values()), (20, 15), "20 source-only and 15 zero-source accounts required")
    return {
        "unstable_partition_count": 3, "coordinate_account_count": 5,
        "weighted_component_account_count": 35, "zero_coordinate_weight_delta_count": 5,
        "zero_weight_delta_product_term_count": 70, "classification_counts": counts,
        "partitions": partitions, "orientation_symmetry": orientation(*partitions[:2]),
        **{k: parent[k] for k in ("delta_rule", "source_rule", "nonlinear_boundary",
                                 "reference_scope", "lineage_separation")},
        "claim_boundary": BOUNDARY, "exact_rational_closure": True,
        "descriptive_accounting_only": True,
    }


def check():
    require(ROOT == Path("/home/argustest/ace3-argus") and Path.cwd() == ROOT
            and Path(__file__).resolve() == SOURCE and sys.executable == PYTHON
            and os.getuid() == 1000 and os.environ.get("PYTHONPATH") == str(ROOT)
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        compiled = []
        for path in (SOURCE, TEST):
            data = path.read_bytes()
            compile(data, str(path), "exec", dont_inherit=True)
            compiled.append(pin(path, len(data), hashlib.sha256(data).hexdigest()))
        retained, _, _ = authenticate()
        result = report(retained)
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch/write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MAPPED_ALL_EXCEPTION_SOURCE_ONLY_PRODUCT_TERM_COLLAPSE_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "retained_source_pins": SOURCE_PINS, "normal_host_review": "REQUIRED",
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
