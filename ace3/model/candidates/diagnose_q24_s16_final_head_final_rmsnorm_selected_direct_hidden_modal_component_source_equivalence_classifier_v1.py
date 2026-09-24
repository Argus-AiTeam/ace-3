"""Read-only modal source/component equivalence at five retained coordinate accounts."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_modal_component_source_equivalence_classifier_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / "ace3/model/candidates" / (NAME + ".py")
TEST = ROOT / "tests" / ("test_" + NAME + ".py")
PYTHON = "/home/argustest/miniconda3/bin/python"
COMMAND = f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m {MODULE} --check"
HELPER_NAME = "diagnose_q24_s16_final_head_final_rmsnorm_selected_direct_hidden_mapped_all_exception_source_only_product_term_collapse_classifier_v1"
HELPER_PINS = (
    {"path": str(ROOT / "ace3/model/candidates" / (HELPER_NAME + ".py")), "bytes": 24296,
     "sha256": "8dd5094d9eeb882ee62504ac4d97b8d95533696fe16f211672ee03d0a5197693"},
    {"path": str(ROOT / "tests" / ("test_" + HELPER_NAME + ".py")), "bytes": 18012,
     "sha256": "86e4df9206ff9be93588e6cf6b87ed7d74f24db5575d2c210239bb1ccb0b94a0"},
)


def load_retained_io():
    snapshots = []
    for binding in HELPER_PINS:
        data = Path(binding["path"]).read_bytes()
        if len(data) != binding["bytes"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise ValueError("retained authentication helper changed: " + binding["path"])
        snapshots.append(data)
    module = ModuleType("ace3.model.candidates." + HELPER_NAME)
    module.__file__ = HELPER_PINS[0]["path"]
    # Execute the authenticated snapshot, not a second loader read of a mutable file.
    exec(compile(snapshots[0], module.__file__, "exec"), module.__dict__)
    return module


retained_io = load_retained_io()
COMPONENTS = retained_io.COMPONENTS
CONTROLS = retained_io.MODAL_CONTROLS
CANONICAL = "frozen_inherited"
SCOPES = retained_io.SCOPES
SCOPE_FIELDS = retained_io.SCOPE_FIELDS
FACTOR = "weight_times_reference_anchor_times_row_difference"
ANCHOR = "reference_inverse_norm_anchor"
BOUNDARY = (
    "Descriptive exact retained-scalar equality only, not causal, performance, repair, "
    "intervention or admission evidence. No producer, operator, native decoder, tensor, "
    "RMSNorm, head or row-dot replay; no prefix/admission or original-reference replay. "
    "Original-input global references, exact thresholds and source/operand/state/KV/lineage "
    "gates remain unchanged. Native S16 RTZ and Q24-wide residual state are not strict-FP16 "
    "state W4A16; G128 asymmetric packed INT4 native nibble/qzero semantics, FP16 scales, "
    "operator boundaries and KV are unchanged. No new-token/full-model admission, "
    "precision/scale/hardware/ACE2 expansion, row319 availability work or closed-branch reopening."
)

require = retained_io.require
same = retained_io.same
rational = retained_io.rational


def scope(item):
    return tuple(item[k] for k in SCOPE_FIELDS)


def one(items, predicate, message):
    matches = [item for item in items if predicate(item)]
    require(len(matches) == 1, message)
    return matches[0]


def selected_row(document, partition, control, coordinate):
    account = one(document["report"]["controls"], lambda r: r["control"] == control,
                  "unique retained modal control required")
    pair = one(account["pairs"], lambda r: (r["left_id"], r["right_id"]) ==
               (partition["left_id"], partition["right_id"]), "unique retained ordered pair required")
    return one(pair["branches"][partition["branch"]]["selected_coordinates"],
               lambda r: r["coordinate"] == coordinate, "unique retained coordinate required")


def status(mismatches, unknowns):
    return "REJECTED" if mismatches else "UNKNOWN" if unknowns else "SUPPORTED"


def spread(values):
    same(tuple(values), CONTROLS, "exact ordered eight-control census required")
    parsed = {control: None if value is None else rational(value)
              for control, value in values.items()}
    known = {c: v for c, v in parsed.items() if v is not None}
    unknowns = [c for c, v in parsed.items() if v is None]
    canonical = parsed[CANONICAL]
    deltas = {c: None if v is None or canonical is None else str(v - canonical)
              for c, v in parsed.items()}
    mismatches = [
        {"control": c, "canonical_control": CANONICAL, "canonical": values[CANONICAL],
         "retained": values[c], "delta": deltas[c]}
        for c, v in known.items() if canonical is not None and v != canonical
    ]
    low, high = (min(known.values()), max(known.values())) if known else (None, None)
    return {
        "values": values, "deltas_from_canonical": deltas,
        "minimum": None if low is None else str(low),
        "maximum": None if high is None else str(high),
        "spread": None if low is None else str(high - low),
        "minimum_controls": [c for c, v in known.items() if v == low],
        "maximum_controls": [c for c, v in known.items() if v == high],
        "extrema_cover_all_controls": not unknowns,
        "mismatch_rows": mismatches, "unknown_controls": unknowns,
        "integrity_status": status(mismatches, unknowns),
    }


def relation(label, actual, expected, **identity):
    missing = actual is None or expected is None
    return {
        **identity, "field": label, "actual": actual, "expected": expected,
        "integrity_status": "UNKNOWN" if missing else
        "SUPPORTED" if actual == expected else "REJECTED",
    }


def total(values):
    return None if any(v is None for v in values) else str(
        sum((rational(v) for v in values), Fraction(0)))


def product(left, right):
    return None if left is None or right is None else str(rational(left) * rational(right))


def difference(left, right):
    return None if left is None or right is None else str(rational(left) - rational(right))


def magnitude(value):
    return None if value is None else str(abs(rational(value)))


def coordinate_fields(row):
    fields = {FACTOR: row.get(FACTOR), ANCHOR: row.get(ANCHOR),
              "direct_hidden_weighted_term": row.get("direct_hidden_weighted_term")}
    for family in ("hidden", "weighted"):
        components = row.get(family + "_components", {})
        require(set(components) <= set(COMPONENTS), "foreign retained component")
        fields.update({family + "." + c: components.get(c) for c in COMPONENTS})
        fields[family + ".sum"] = total([components.get(c) for c in COMPONENTS])
        mass = row.get(family + "_component_mass", {})
        for name in ("signed", "absolute", "cancellation_absolute_mass"):
            fields[family + ".mass." + name] = mass.get(name)
    for value in fields.values():
        if value is not None:
            rational(value)
    return fields


def coordinate_closures(row, fields, control, coordinate):
    checks = []

    def bind(label, actual, expected):
        checks.append(relation(label, actual, expected, control=control, coordinate=coordinate))

    bind("retained_exact_direct_hidden_identity", row.get("exact_direct_hidden_identity"), True)
    for component in COMPONENTS:
        bind("product." + component, fields["weighted." + component],
             product(fields[FACTOR], fields["hidden." + component]))
    bind("seven_weighted_sum", fields["weighted.sum"], fields["direct_hidden_weighted_term"])
    bind("weight_times_seven_hidden_sum", product(fields[FACTOR], fields["hidden.sum"]),
         fields["direct_hidden_weighted_term"])
    for family in ("hidden", "weighted"):
        absolute = total([magnitude(fields[family + "." + c]) for c in COMPONENTS])
        bind(family + ".signed_mass", fields[family + ".mass.signed"], fields[family + ".sum"])
        bind(family + ".absolute_mass", fields[family + ".mass.absolute"], absolute)
        bind(family + ".cancellation_mass", fields[family + ".mass.cancellation_absolute_mass"],
             difference(absolute, magnitude(fields[family + ".sum"])))
    return checks


def summarize(fields, checks):
    mismatch_rows = [
        {"field": name, **row}
        for name, account in fields.items() for row in account["mismatch_rows"]
    ]
    unknown_rows = [
        {"field": name, "control": control, "reason": "retained scalar missing"}
        for name, account in fields.items() for control in account["unknown_controls"]
    ]
    mismatch_rows.extend(c for c in checks if c["integrity_status"] == "REJECTED")
    unknown_rows.extend(c for c in checks if c["integrity_status"] == "UNKNOWN")
    return {"integrity_status": status(mismatch_rows, unknown_rows),
            "mismatch_rows": mismatch_rows, "unknown_rows": unknown_rows}


def classify_partition(parent, comparator, hotspots, source):
    require(scope(parent) in SCOPES, "unsupported partition")
    same(scope(comparator), scope(parent), "parent comparator partition splice")
    same(parent["modal_class"],
         {"canonical_control": CANONICAL, "controls": list(CONTROLS), "count": 8},
         "canonical parent modal class")
    same(comparator["modal_controls"], list(CONTROLS), "parent eight-control membership")
    same([p["control"] for p in comparator["control_profiles"]], list(CONTROLS),
         "unique ordered parent comparator profiles")
    identities = ({"coordinate": 62}, {"coordinate": 241})
    if parent["table"] == "coordinate_component":
        identities = ({"coordinate": 241, "component": COMPONENTS[-1]},
                      {"coordinate": 241, "component": "mlp_stage17"})
    same(tuple(parent["candidates"][k]["identity"] for k in ("exception", "modal")),
         identities, "fixed candidate identities")
    coordinates, checks, all_fields = [], [], {}
    by_coordinate = {}
    for coordinate in dict.fromkeys(i["coordinate"] for i in identities):
        values = {}
        for control in CONTROLS:
            row = selected_row(source, parent, control, coordinate)
            copied = selected_row(hotspots, parent, control, coordinate)
            checks.append(relation("449b_50f_selected_row_identity", copied, row,
                                   control=control, coordinate=coordinate))
            values[control] = coordinate_fields(row)
            checks.extend(coordinate_closures(row, values[control], control, coordinate))
        by_coordinate[coordinate] = values
        fields = {name: spread({c: values[c][name] for c in CONTROLS})
                  for name in values[CANONICAL]}
        coordinates.append({"coordinate": coordinate, "field_spreads": fields})
        all_fields.update({f"coordinate.{coordinate}.{k}": v for k, v in fields.items()})

    candidate_values = {}
    profiles = comparator["control_profiles"]
    for profile in profiles:
        control = profile["control"]
        fields = {}
        for label, identity in zip(("exception", "modal"), identities, strict=True):
            selected = by_coordinate[identity["coordinate"]][control]
            key = "weighted." + identity["component"] if "component" in identity else "weighted.sum"
            fields[label + ".signed"] = selected[key]
            fields[label + ".absolute"] = magnitude(selected[key])
            checks.append(relation(label + ".identity", profile["identities"][label], identity,
                                   control=control))
            checks.append(relation(label + ".factorizer_canonical_closure", selected[key],
                                   parent["candidates"][label]["signed"]["modal_class"], control=control))
        fields["signed_comparator"] = difference(fields["exception.signed"], fields["modal.signed"])
        fields["absolute_comparator"] = difference(fields["exception.absolute"], fields["modal.absolute"])
        if parent["table"] == "coordinate_component":
            selected = by_coordinate[241][control]
            names = [i["component"] for i in identities]
            fields["selected_sum"] = total([selected["weighted." + c] for c in names])
            fields["context_sum"] = total([selected["weighted." + c] for c in COMPONENTS if c not in names])
            checks.append(relation("selected_plus_context_closure",
                                   total([fields["selected_sum"], fields["context_sum"]]),
                                   selected["weighted.sum"], control=control))
        for name in ("exception.signed", "modal.signed", "exception.absolute", "modal.absolute",
                     "signed_comparator", "absolute_comparator"):
            checks.append(relation("parent." + name, fields[name],
                                   profile["fields"].get("comparator." + name), control=control))
        checks.append(relation("factorizer_canonical_comparator", fields["signed_comparator"],
                               parent["signed_comparator"]["modal_class"], control=control))
        for name in ("identities", "ranks", "ties", "signs", "roles", "contrast_membership"):
            checks.append(relation("parent_canonical." + name, profile[name], profiles[0][name],
                                   control=control))
        candidate_values[control] = fields
    candidates = {name: spread({c: candidate_values[c][name] for c in CONTROLS})
                  for name in candidate_values[CANONICAL]}
    profile_names = set(profiles[0]["fields"])
    require(all(set(p["fields"]) == profile_names for p in profiles), "parent field census")
    parent_spreads = {name: spread({p["control"]: p["fields"][name] for p in profiles})
                      for name in sorted(profile_names)}
    all_fields.update({"candidate." + k: v for k, v in candidates.items()})
    all_fields.update({"parent." + k: v for k, v in parent_spreads.items()})
    return {
        **{k: parent[k] for k in SCOPE_FIELDS}, "canonical_control": CANONICAL,
        "modal_controls": list(CONTROLS), "coordinates": coordinates,
        "candidate_identities": dict(zip(("exception", "modal"), identities, strict=True)),
        "candidate_spreads": candidates, "parent_comparator_spreads": parent_spreads,
        "closure_checks": checks, **summarize(all_fields, checks),
    }


def orientation(forward, reverse):
    same((scope(forward), scope(reverse)), SCOPES[:2], "fixed reversed-pair orientation")
    checks = []
    for left, right in zip(forward["coordinates"], reverse["coordinates"], strict=True):
        same(left["coordinate"], right["coordinate"], "reversed coordinate identity")
        for name, account in left["field_spreads"].items():
            negated = (name == FACTOR or name == "direct_hidden_weighted_term"
                       or name.startswith("weighted.") and name not in
                       ("weighted.mass.absolute", "weighted.mass.cancellation_absolute_mass"))
            for control in CONTROLS:
                value = account["values"][control]
                expected = difference("0", value) if negated else value
                checks.append(relation(name, right["field_spreads"][name]["values"][control],
                                       expected, control=control, coordinate=left["coordinate"],
                                       orientation="negated" if negated else "identical"))
    for family in ("candidate_spreads", "parent_comparator_spreads"):
        for name, account in forward[family].items():
            negated = "signed" in name and "absolute" not in name
            for control in CONTROLS:
                value = account["values"][control]
                checks.append(relation(family + "." + name,
                                       reverse[family][name]["values"][control],
                                       difference("0", value) if negated else value, control=control,
                                       orientation="negated" if negated else "identical"))
    return {"checks": checks, **summarize({}, checks)}


def report(retained):
    retained_io.validate_retained(retained)
    archetypes = retained["retained_667"]
    comparators = archetypes["retained_3fd"]
    hotspots = comparators["retained_0146"]["retained_6184"]["retained_e795"]["retained_449b"]
    source = hotspots["retained_50f"]
    parents = retained["report"]["partitions"]
    same([scope(p) for p in comparators["report"]["partitions"]], list(SCOPES),
         "same three retained comparator scopes")
    partitions = [classify_partition(p, q, hotspots, source)
                  for p, q in zip(parents, comparators["report"]["partitions"], strict=True)]
    reversed_pair = orientation(*partitions[:2])
    mismatches, unknowns = [], []
    for item in (*partitions, reversed_pair):
        location = {k: item[k] for k in SCOPE_FIELDS if k in item}
        if not location:
            location = {"account": "reversed_pair"}
        mismatches.extend({**location, **r} for r in item["mismatch_rows"])
        unknowns.extend({**location, **r} for r in item["unknown_rows"])
    return {
        "integrity_status": status(mismatches, unknowns),
        "unstable_partition_count": len(partitions), "modal_control_count": len(CONTROLS),
        "coordinate_account_count": sum(len(p["coordinates"]) for p in partitions),
        "control_coordinate_account_count": sum(len(p["coordinates"]) for p in partitions) * len(CONTROLS),
        "control_component_account_count": sum(len(p["coordinates"]) for p in partitions)
        * len(CONTROLS) * len(COMPONENTS),
        "canonical_control": CANONICAL, "partitions": partitions, "reversed_pair": reversed_pair,
        "mismatch_rows": mismatches, "unknown_rows": unknowns,
        "spread_rule": "Exact max-minus-min over all eight controls, with all tied extrema; missing values disclose partial extrema.",
        "factor_scope": "Retained combined weight*reference_inverse_norm_anchor*row_difference and separately retained inverse-norm anchor only.",
        "separate_weight_and_row_difference_equality": "UNKNOWN",
        "separate_factor_boundary": "Individual weight and row-difference scalars are not retained in these selected rows; do not infer them from product equality.",
        "descriptive_accounting_only": True, "arithmetic": "canonical integer-rational strings; no float",
        "lineage_separation": source["report"]["lineage_separation"],
        "reference_scope": source["report"]["reference_scope"],
        "claim_boundary": BOUNDARY,
    }


def source_binding(path):
    data = path.read_bytes()
    compile(data, str(path), "exec")
    return retained_io.pin(path, len(data), hashlib.sha256(data).hexdigest())


def check():
    for binding in HELPER_PINS:
        retained_io.bound_bytes(binding)
    compiled = [source_binding(p) for p in (SOURCE, TEST)]
    audit = {"forbidden_calls": 0}
    # Only the helper's authenticated parent files can be opened inside this guard.
    with retained_io.read_only(audit):
        retained, capture, review = retained_io.authenticate()
        result = report(retained)
        flags = dict(retained["flags"])
        audit = {**retained["dispatch_and_write_audit"], **audit}
        retained_io.zero_counters(audit)
        retained_io.zero_counters(flags)
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_SELECTED_DIRECT_HIDDEN_MODAL_COMPONENT_SOURCE_EQUIVALENCE_CLASSIFIER",
        "command": COMMAND, "compiled_sources": compiled, "helper_source_pins": list(HELPER_PINS),
        "input_pins": retained_io.PINS, "retained_source_pins": retained_io.SOURCE_PINS,
        "retained_review": review, "retained_capture_checks": capture["checks"],
        "report": result, "dispatch_and_write_audit": audit, "flags": flags,
        "normal_host_review": "Independent Reviewer completion required; not asserted by Engineer.",
        "claim_boundary": BOUNDARY,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
