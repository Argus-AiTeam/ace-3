"""Stdout-only exact component/source deltas for three retained exception partitions."""

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_component_delta_factorizer_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
PARENT_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_archetype_classifier_v1"
CAPTURE_ROOT = ROOT / "build/selected-direct-hidden-archetypes-667adfd00007-attempt001"
REVIEW_ROOT = Path("/home/argustest/.argus-skill-ace3/projects/s-62150b05/handoffs")


def pin(path, size, digest):
    return {"path": str(path), "bytes": size, "sha256": digest}


PINS = {
    "stdout": pin(CAPTURE_ROOT / "check.stdout", 20031453,
                  "e748c1c34d6349cbe450459286e7f0ac74cf6b6c291c189c6f118f738172741a"),
    "capture": pin(CAPTURE_ROOT / "capture.json", 31932,
                   "f77adc863f0dc4ba7f110176fec28c1234939eae0d3ee1e32478d9d6feb61e6c"),
    "review": pin(REVIEW_ROOT / "667adfd00007/round-0001.json", 690,
                  "2b32f30ac2957d45cee03650f1f79cb348f53619b09a010cff8dcf4481f53bd7"),
}
SOURCE_PINS = [
    pin(ROOT / "ace3/model/candidates" / (PARENT_NAME + ".py"), 29544,
        "287a8c2d1fb08e65a3427d9dafe5fbedfab74b233c181104dd5fd5aa503ad10a"),
    pin(ROOT / "tests" / ("test_" + PARENT_NAME + ".py"), 17299,
        "f78173d99db72a3e555e3df0ab430b80c021806993bb7d303daef11acdb5e1e2"),
]
NESTED_PINS = (
    ("retained_3fd", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-modal-equivalence-3fdafbcebc0f-uyn6nk5a/check.stdout", 19969930,
                      "16d13d84b23b43670fbd3222216b285b2672e3140d49b0a489890dccd3890200"),
        "capture": pin(ROOT / "build/selected-direct-hidden-modal-equivalence-3fdafbcebc0f-uyn6nk5a/capture.json", 32594,
                       "6f6edd6d456eff74190539bce7c4f090a71f39f7e2e297f76e7ec7ab63f044d5"),
        "review": pin(REVIEW_ROOT / "3fdafbcebc0f/round-0001.json", 690,
                      "f68bdd6b6ee5f2e40632fadce463264e3e39b937b70e67aaeb8cf8925278e67c"),
    }),
    ("retained_0146", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-modal-contrast-0146c951608d-pt2zlbcz/check.stdout", 19618193,
                      "233e274d21ddd53f80912ff83e666cc4d3e3db5a8388cde18dd8eb8603b1294f"),
        "capture": pin(ROOT / "build/selected-direct-hidden-modal-contrast-0146c951608d-pt2zlbcz/capture.json", 32049,
                       "9ce9e56b089f5d0351f1e40dd40820e894615b86325d408241b258cc2a97c2d9"),
        "review": pin(REVIEW_ROOT / "0146c951608d/round-0001.json", 690,
                      "b8cd60e1bf22c51be87175bbe88912e3b67bc90568c66e2916592384aa665ba8"),
    }),
    ("retained_6184", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/check.stdout", 19526954,
                      "75c4cd644ad2c264801de7a364b0dc4af9ff1d2c1960a95951e4752a0835a204"),
        "capture": pin(ROOT / "build/selected-direct-hidden-partition-6184a5063c9d-completion-b6volua0/capture.json", 27328,
                       "913dcb43aabbe46dd4124f88d416dce3e193d31a0ef46b1f3b727ca8e0dedbae"),
        "review": pin(REVIEW_ROOT / "6184a5063c9d/round-0001.json", 690,
                      "2deb0d33524599c6082d77ecc35847a5a85855a119704b21cdee77a2f1dd8b6e"),
    }),
    ("retained_e795", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/check.stdout", 18840034,
                      "831c13630634ca0b8f6521aea271667851a8311f455e008af813a66c07ff7bb6"),
        "capture": pin(ROOT / "build/selected-direct-hidden-stability-e795493d08bc-fumvbe57/capture.json", 17021,
                       "247e09a954f046b2b2854522260e2cb1ff054a7e3e5171648b447a906f9385cf"),
        "review": pin(REVIEW_ROOT / "e795493d08bc/round-0001.json", 767,
                      "2cd5adea4c912fd5b648782dbcd605d6b9a59eaa9c1c5b3b14886c2438d9ca8e"),
    }),
    ("retained_449b", {
        "stdout": pin(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/check.stdout", 14452633,
                      "67318f659a54c5a8e05fbe8acd0f7e209b7fcdbf670183d1e863b13e1a67f9a2"),
        "capture": pin(ROOT / "build/selected-direct-hidden-hotspots-449b1c24ea06-apfh_msz/capture.json", 5327,
                       "21dc61bea5786d8674434780599cd12d565ab2cf6cadd9ba465c3b596c73ffc3"),
        "review": pin(REVIEW_ROOT / "449b1c24ea06/round-0001.json", 688,
                      "566ae7c16df4d327bd028394d112e365fd7822540aed0b4e15bc0a556f0ebc41"),
    }),
    ("retained_50f", {
        "stdout": pin(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/stdout.json", 6737063,
                      "30790957680088caecfa2539e10e672f0d5ff373573d4c80a71cb3be2d65cdeb"),
        "capture": pin(ROOT / "build/direct-hidden-residual-50f5bb2062c9-bc457u29/capture.json", 7652,
                       "f29d226f2bc4e99d967b89b9f8baeeba7c71c555be742f58367eefcda3dad464"),
        "review": pin(REVIEW_ROOT / "50f5bb2062c9/round-0001.json", 689,
                      "feb58e79b284e652e9bb028a2ea79b3f4b48a6b9fd18a58f0465367a6a204bb8"),
    }),
)
COMPONENTS = (
    "input_hidden", "attention_stage11", "mlp_stage17",
    "actual_residual_boundary", "negative_reference_residual_boundary",
    "q24_to_fp16_conversion", "fp16_to_branch_terminal_remainder",
)
MODAL_CONTROLS = (
    "frozen_inherited", "frozen_inherited_o", "frozen_inherited_down",
    "frozen_inherited_o_down", "scratch", "scratch_down", "mapped62", "inherited_native",
)
SCOPES = (
    ("coordinate", 34319, 319, "binary64"),
    ("coordinate", 319, 34319, "binary64"),
    ("coordinate_component", 34319, 13, "binary64"),
)
SCOPE_FIELDS = ("table", "left_id", "right_id", "branch")
SIDES = ("modal_class", "mapped_all", "delta")
LABELS = ("branch", "ignored-build", "compile", "pytest", "check")
SUFFIXES = (".command.txt", ".argv.json", ".environment.json", ".stdout",
            ".stderr", ".whole-command.log")
NONLINEAR = (
    "Only signed quantities have additive component closure. abs(mapped)-abs(modal) "
    "is not abs(mapped-modal) or a sum of component absolute changes. Competition "
    "ranks, winner identities, ties and dominance gaps depend non-linearly on the "
    "whole retained candidate set; no component contribution to those fields is assigned. "
    "Dominant means tied largest absolute signed component delta, not a causal source "
    "or runtime bottleneck. Opposing means opposite to the net signed delta; when "
    "the net is zero, nonzero components are explicitly cancelling, not aligned."
)
BOUNDARY = (
    "Descriptive retained-rational mapped_all-minus-canonical-modal component/source "
    "accounting only for three reviewed unstable partitions. No causal, intervention, "
    "performance, repair or admission claim. Only reviewed 667 stdout/capture/source/"
    "test/review and capture sidecars are opened; nested 3fd/0146/6184/e795/449b/50f "
    "pins and retained data are authenticated without reopening or executing producers. "
    "All accepted evidence and historical failures are preserved. Original-input "
    "independently propagated global references, exact thresholds and source/operand/"
    "state/KV/lineage gates remain unchanged. No prefix/admission/reference/producer "
    "replay, tensor/native decoding, RMSNorm, head/row-dot or model operators, closed "
    "480/50f/f0/6184/e795/0146/3fd/667 re-review, or row319 availability work. "
    "The row-319 missing 896-element pre-round producer and "
    "NOT_RETAINED_NO_RECONSTRUCTION binary64 internal stages remain limitations. "
    "Q24 residual state is wider than FP16; native S16 RTZ, G128 asymmetric packed "
    "INT4, native GEMM nibble ordering, no qzero plus-one, FP16 scales/operator "
    "boundaries/KV remain unchanged. No strict-FP16-state W4A16, new-token/full-model "
    "admission, GPU/RTL/FPGA/hardware, precision/scale expansion or ACE2 changes. "
    "Host independent Reviewer REQUIRED."
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
            forbidden = args[0].startswith(("ace3.", "numpy", "torch", "safetensors", "ctypes"))
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
            raise RuntimeError("retained-only component delta factorizer forbids " + event)

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
         (PARENT_NAME, "READ_ONLY_SELECTED_DIRECT_HIDDEN_MAPPED_ALL_EXCEPTION_ARCHETYPE_CLASSIFIER"),
         "wrong retained producer/boundary")
    require(type(retained["version"]) is int and retained["version"] == 1, "version changed")
    same(retained["compiled_sources"], SOURCE_PINS, "source/test splice")
    node = retained
    for child, pins in NESTED_PINS:
        same(node["input_pins"], pins, "nested pins changed: " + child)
        zero_counters(node["dispatch_and_write_audit"])
        zero_counters(node["flags"])
        node = node[child]
    zero_counters(node["dispatch_and_write_audit"])
    zero_counters(node["flags"])
    parent = retained["report"]
    require(parent["descriptive_accounting_only"] is True
            and parent["exact_rational_closure"] is True, "accounting boundary changed")
    same((parent["unstable_partition_count"], parent["modal_class_count"],
          parent["singleton_exception_count"]), (3, 3, 3), "reviewed census changed")
    same([tuple(p[k] for k in SCOPE_FIELDS) for p in parent["partitions"]],
         list(SCOPES), "exact three scopes required")


def authenticate():
    review = decode(bound_bytes(PINS["review"]))
    same((review["kind"], review["mission_id"], review["producer_role"], review["review"]["status"]),
         ("round_reviewed_handoff", "667adfd00007", "reviewer", "done"),
         "independent terminal 667 review required")
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


def movement(modal, mapped):
    left, right = rational(modal), rational(mapped)
    return {"modal_class": modal, "mapped_all": mapped, "delta": str(right - left)}


def classes(deltas):
    values = {k: rational(v) for k, v in deltas.items()}
    require(bool(values), "empty component class")
    total = sum(values.values(), Fraction(0))
    maximum = max(abs(v) for v in values.values())
    return {
        "signed_sum": str(total),
        "absolute_mass": str(sum((abs(v) for v in values.values()), Fraction(0))),
        "dominant": [k for k, v in values.items() if v and abs(v) == maximum],
        "aligned": [k for k, v in values.items() if v * total > 0],
        "opposing": [k for k, v in values.items() if v * total < 0],
        "zero": [k for k, v in values.items() if not v],
        "cancelling_nonzero": [k for k, v in values.items() if v and not total],
        "net_zero": not total,
    }


def one(items, predicate, message):
    selected = [item for item in items if predicate(item)]
    same(len(selected), 1, message)
    return selected[0]


def selected_row(retained, partition, control, coordinate):
    account = one(retained["report"]["controls"], lambda a: a["control"] == control,
                  "unique control required")
    pair = one(account["pairs"], lambda p: (p["left_id"], p["right_id"]) ==
               (partition["left_id"], partition["right_id"]), "unique ordered pair required")
    return one(pair["branches"][partition["branch"]]["selected_coordinates"],
               lambda r: r["coordinate"] == coordinate, "unique selected coordinate required")


def coordinate_account(modal, mapped):
    same(modal["coordinate"], mapped["coordinate"], "coordinate splice")
    weights = [rational(row["weight_times_reference_anchor_times_row_difference"])
               for row in (modal, mapped)]
    for row, weight in zip((modal, mapped), weights, strict=True):
        same(set(row["weighted_components"]), set(COMPONENTS), "seven weighted components required")
        same(set(row["hidden_components"]), set(COMPONENTS), "seven source components required")
        require(row["exact_direct_hidden_identity"] is True, "retained coordinate closure required")
        for component in COMPONENTS:
            same(rational(row["weighted_components"][component]),
                 weight * rational(row["hidden_components"][component]), "source/weight closure")
        same(sum((rational(v) for v in row["weighted_components"].values()), Fraction(0)),
             rational(row["direct_hidden_weighted_term"]), "coordinate signed closure")
    components = []
    dw = weights[1] - weights[0]
    for component in COMPONENTS:
        hidden = movement(modal["hidden_components"][component], mapped["hidden_components"][component])
        weighted = movement(modal["weighted_components"][component], mapped["weighted_components"][component])
        dh = rational(hidden["delta"])
        terms = {
            "modal_weight_times_source_delta": str(weights[0] * dh),
            "modal_source_times_weight_delta": str(rational(hidden["modal_class"]) * dw),
            "source_delta_times_weight_delta": str(dh * dw),
        }
        same(sum((rational(v) for v in terms.values()), Fraction(0)),
             rational(weighted["delta"]), "signed product-difference closure")
        components.append({"component": component, "source": hidden, "weighted": weighted,
                           "product_delta_terms": terms, "exact_product_delta_closure": True})
    signed = movement(modal["direct_hidden_weighted_term"], mapped["direct_hidden_weighted_term"])
    for side in SIDES:
        same(sum((rational(c["weighted"][side]) for c in components), Fraction(0)),
             rational(signed[side]), "seven-component signed movement closure")
    return {
        "coordinate": modal["coordinate"], "components": components,
        "weight": movement(str(weights[0]), str(weights[1])),
        "signed": signed, "exact_seven_component_closure": True,
        "component_delta_classes": classes({c["component"]: c["weighted"]["delta"] for c in components}),
        "source_delta_classes": classes({c["component"]: c["source"]["delta"] for c in components}),
    }


def parent_binding(actual, parent, message):
    same(actual, {k: parent[k] for k in SIDES}, message)


def factor_partition(parent, hotspots, source):
    same(tuple(parent[k] for k in SCOPE_FIELDS) in SCOPES, True, "unsupported partition")
    same(parent["modal_class"],
         {"canonical_control": MODAL_CONTROLS[0], "controls": list(MODAL_CONTROLS), "count": 8},
         "authenticated canonical modal class required")
    same(parent["exception_class"], {"controls": ["mapped_all"], "count": 1}, "exception class")
    ids = ({"coordinate": 62}, {"coordinate": 241})
    if parent["table"] == "coordinate_component":
        ids = ({"coordinate": 241, "component": COMPONENTS[-1]},
               {"coordinate": 241, "component": "mlp_stage17"})
    for side in ("modal_class", "mapped_all"):
        same(tuple(parent["identity_membership"][side][c] for c in ("exception", "modal")),
             ids, "reviewed candidate identities")
    coordinates = []
    for coordinate in dict.fromkeys(i["coordinate"] for i in ids):
        rows = []
        for control in (MODAL_CONTROLS[0], "mapped_all"):
            row = selected_row(hotspots, parent, control, coordinate)
            same(row, selected_row(source, parent, control, coordinate), "449b/50f retained source splice")
            rows.append(row)
        coordinates.append(coordinate_account(*rows))
    candidates = {}
    selected_components = {}
    for label, identity in zip(("exception", "modal"), ids, strict=True):
        account = one(coordinates, lambda r: r["coordinate"] == identity["coordinate"], "candidate coordinate")
        if parent["table"] == "coordinate":
            selected = {c["component"]: c["weighted"] for c in account["components"]}
            signed = account["signed"]
        else:
            component = one(account["components"], lambda c: c["component"] == identity["component"],
                            "selected component")
            selected = {component["component"]: component["weighted"]}
            signed = component["weighted"]
        parent_binding(signed, parent["field_comparisons"][label + ".signed"], "parent candidate signed closure")
        same(signed["delta"], parent["contrast_vector"][label + "_signed_delta"], "parent candidate contrast")
        candidates[label] = {"identity": identity, "signed": signed, "exact_parent_closure": True}
        selected_components[label] = selected
    comparator_components = {}
    for name in COMPONENTS:
        comparator_components[name] = {
            side: str(rational(selected_components["exception"].get(name, {}).get(side, "0"))
                      - rational(selected_components["modal"].get(name, {}).get(side, "0")))
            for side in SIDES
        }
    comparator = {side: str(sum((rational(c[side]) for c in comparator_components.values()), Fraction(0)))
                  for side in SIDES}
    parent_binding(comparator, parent["field_comparisons"]["signed_comparator"], "parent signed comparator closure")
    same(comparator["delta"], parent["contrast_vector"]["signed_comparator_delta"], "parent comparator contrast")
    context = None
    if parent["table"] == "coordinate_component":
        selected_names = [i["component"] for i in ids]
        context_names = [c for c in COMPONENTS if c not in selected_names]
        components = {c["component"]: c["weighted"] for c in coordinates[0]["components"]}
        sums = {
            label: {side: str(sum((rational(components[c][side]) for c in names), Fraction(0)))
                    for side in SIDES}
            for label, names in (("selected_sum", selected_names), ("context_sum", context_names))
        }
        for side in SIDES:
            same(rational(sums["selected_sum"][side]) + rational(sums["context_sum"][side]),
                 rational(coordinates[0]["signed"][side]), "selected/context coordinate closure")
        context = {
            "selected_components": selected_names, "remaining_components": context_names, **sums,
            "coordinate_signed": coordinates[0]["signed"], "exact_selected_context_closure": True,
            "boundary": "Selected sum is not the exception-minus-modal comparator; remaining context is not assigned to either selected candidate.",
        }
    return {
        **{k: parent[k] for k in SCOPE_FIELDS},
        "modal_class": parent["modal_class"], "exception_class": parent["exception_class"],
        "coordinates": coordinates, "candidates": candidates,
        "signed_comparator": comparator, "comparator_components": comparator_components,
        "comparator_delta_classes": classes({k: v["delta"] for k, v in comparator_components.items()}),
        "exact_parent_comparator_closure": True, "selected_component_context": context,
        "nonlinear_parent_fields": {k: v for k, v in parent["field_comparisons"].items()
                                    if k not in ("exception.signed", "modal.signed", "signed_comparator")},
    }


def orientation(forward, reverse):
    checked = 0
    for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
        same(left["coordinate"], right["coordinate"], "reversed coordinate identity")
        for a, b in zip(left["components"], right["components"], strict=True):
            same(a["component"], b["component"], "reversed component identity")
            same(a["source"], b["source"], "reversed source invariance")
            for side in SIDES:
                same(rational(a["weighted"][side]), -rational(b["weighted"][side]), "component orientation")
                checked += 1
            for term in a["product_delta_terms"]:
                same(rational(a["product_delta_terms"][term]), -rational(b["product_delta_terms"][term]),
                     "source factor orientation")
        for field in ("signed", "weight"):
            for side in SIDES:
                same(rational(left[field][side]), -rational(right[field][side]), "coordinate/weight orientation")
        same(left["component_delta_classes"], {
            **right["component_delta_classes"],
            "signed_sum": str(-rational(right["component_delta_classes"]["signed_sum"])),
        }, "orientation-invariant component classes")
    for component in COMPONENTS:
        for side in SIDES:
            same(rational(forward["comparator_components"][component][side]),
                 -rational(reverse["comparator_components"][component][side]), "comparator component orientation")
    for side in SIDES:
        same(rational(forward["signed_comparator"][side]), -rational(reverse["signed_comparator"][side]),
             "signed comparator orientation")
    same(forward["nonlinear_parent_fields"], reverse["nonlinear_parent_fields"], "nonlinear orientation invariance")
    return {"ordered_pairs": [[34319, 319], [319, 34319]],
            "component_signed_orientation_checks": checked,
            "exact_signed_negation": True, "source_deltas_identical": True,
            "nonlinear_parent_fields_identical": True}


def report(retained):
    validate_retained(retained)
    hotspots = retained
    for child, _ in NESTED_PINS[:-1]:
        hotspots = hotspots[child]
    partitions = [factor_partition(p, hotspots, hotspots["retained_50f"])
                  for p in retained["report"]["partitions"]]
    return {
        "unstable_partition_count": 3, "coordinate_account_count": 5,
        "weighted_component_account_count": 35, "partitions": partitions,
        "orientation_symmetry": orientation(*partitions[:2]),
        "delta_rule": "mapped_all minus frozen_inherited, the authenticated canonical modal comparator representative; component/source equality of the other seven controls is not assumed.",
        "source_rule": "Retained hidden components times retained weight/reference-anchor/row-difference factor. Product differences are exact descriptive algebra, not interventions or reconstructed binary64 internal stages.",
        "nonlinear_boundary": NONLINEAR, "claim_boundary": BOUNDARY,
        "reference_scope": retained["report"]["reference_scope"],
        "lineage_separation": retained["report"]["lineage_separation"],
        "exact_rational_closure": True, "descriptive_accounting_only": True,
    }


def check():
    require(ROOT == Path("/home/argustest/ace3-argus") and Path.cwd() == ROOT
            and Path(__file__).resolve() == SOURCE and sys.executable == PYTHON
            and os.getuid() == 1000 and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        compiled = []
        for path in (SOURCE, TEST):
            data = path.read_bytes()
            compile(data, str(path), "exec", dont_inherit=True)
            compiled.append(pin(path, len(data), hashlib.sha256(data).hexdigest()))
        retained, capture, review = authenticate()
        result = report(retained)
    same(audit, {"forbidden_calls": 0}, "forbidden dispatch/write")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MAPPED_ALL_EXCEPTION_COMPONENT_DELTA_FACTORIZER",
        "command": COMMAND, "compiled_sources": compiled, "input_pins": PINS,
        "retained_source_pins": SOURCE_PINS, "normal_host_review": "REQUIRED",
        "dispatch_and_write_audit": {**retained["dispatch_and_write_audit"], **audit},
        "flags": retained["flags"], "claim_boundary": BOUNDARY,
        "retained_667": retained, "retained_667_capture": capture,
        "retained_667_review": review, "report": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
