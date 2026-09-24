"""Read-only retained stage18-to-final-RMSNorm delta accounting; stdout JSON only."""

import argparse
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_l23_residual_branch_delta_v1 as residual


hotspots = residual.channels.hotspots
contributions = hotspots.contributions
base, parent = hotspots.base, hotspots.parent
ROOT = base.ROOT
NAME = "diagnose_q24_s16_final_head_final_rmsnorm_hidden_delta_bridge_v1"
MODULE = "ace3.model.candidates." + NAME
SOURCE = ROOT / f"ace3/model/candidates/{NAME}.py"
TEST = ROOT / f"tests/test_{NAME}.py"
COMMAND = (f"PYTHONPATH={ROOT} PYTHONDONTWRITEBYTECODE=1 {parent.PYTHON}"
           f" -B -m {MODULE} --check")
WIDTH, LIMIT, EXPECTED_TESTS = 896, 8, 18
BRANCHES = ("fp16", "binary64")
EPSILON = Fraction(1, 1000000)
require, same = base.require, base.same
FLAGS = {
    **residual.FLAGS,
    "rmsnorm_operator_replay": 0, "head_operator_replay": 0,
    "reference_producer_replay": 0, "evidence_writes": 0,
    "precision_or_scale_expansion": False,
    "upstream_causality_claimed": False, "rounding_only_attribution": False,
}
BOUNDARY = (
    "Read-only CPU-software accounting from authenticated retained L23 stage18 "
    "hidden vectors to final RMSNorm hotspots on reviewed final-head attempt001 "
    "and independently propagated original-input final attempt003. FP16 and "
    "binary64 hidden/reference outputs remain separate; no reference is cast "
    "from another branch or reanchored. Exact rational accounting uses explicitly "
    "binary64 scalar inverse-norm anchors, not operator outputs or replay. "
    "Boundary/conversion remainders include implementation differences and "
    "scalar-anchor approximation; they are not rounding-only attributions. "
    "Q24-to-FP16 input conversion is disclosed but raw Q24 state is not propagated "
    "through RMSNorm. No upstream causality, dominant-stage or performance claim. "
    "All historical failures, exact thresholds, source/operand/state/KV/lineage "
    "gates and original global references remain unchanged. Old sparse-cut PASS "
    "cannot certify, substitute for or propagate these nine parents. Q24 residual "
    "state is wider than FP16; native S16 RTZ, G128 asymmetric packed INT4 native "
    "GEMM ordering without qzero plus-one, FP16 scales/operator boundaries/KV "
    "and tokenizer/tied-head identities remain unchanged. No native, decoder, "
    "prefix, admission, reference-producer, RMSNorm, head or other operator replay, "
    "earlier checks, evidence writes, accepted-artifact overwrite, hardware, GPU, "
    "RTL, simulation, ACE2 work, precision or scale expansion. No admission, "
    "policy adoption, successor publication, strict-FP16-state W4A16, new-token "
    "or full-model capability. Normal independent Host Reviewer REQUIRED."
)
TERMS = ("direct_hidden", "global_scale", "interaction", "boundary_remainder_delta")


@contextmanager
def read_only(audit):
    def forbidden(*args, **kwargs):
        audit["forbidden_calls"] += 1
        raise RuntimeError("hidden bridge forbids operator/reference replay and writes")

    with residual.read_only(audit), ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("ace3.") and name != MODULE:
                for attribute in ("rmsnorm", "_torch_rmsnorm", "logits",
                                  "decode_array_q24", "load_operands",
                                  "reference_suffix", "compute"):
                    if callable(getattr(module, attribute, None)):
                        stack.enter_context(patch.object(module, attribute, forbidden))
        yield


def validate_weight(array, pin):
    require(isinstance(array, np.ndarray) and array.shape == (WIDTH,)
            and array.dtype.str == "<f2" and np.all(np.isfinite(array)),
            "invalid final RMSNorm weight")
    same(pin, {"shape": [WIDTH], "dtype": "float16",
               "sha256": hashlib.sha256(array.tobytes()).hexdigest()},
         "final RMSNorm tensor binding changed")
    return contributions.exact_vector(array)


def authenticate():
    result, arrays, references, files, assets = hotspots.authenticate()
    same(parent.preflight.FINAL_CONTRACT["rmsnorm"], {
        "input": "stage18; authenticated FP16 projection of retained Q24 I/Z",
        "tensor": "model.norm.weight", "shape": [WIDTH], "dtype": "float16",
        "epsilon": str(EPSILON), "output": "896 finite FP16 words",
    }, "final RMSNorm epsilon/tensor/input contract changed")
    same(parent.norm.EPSILON_Q48, 281474977, "native RMSNorm epsilon changed")
    same(result["arithmetic"], parent.ARITHMETIC, "retained final arithmetic changed")
    actual, reference, archives, reference_archive, pins = residual.load_residuals(result)
    final = result["preflight"]["final_reference"]["reference"]
    same(final["input_binary64"],
         result["preflight"]["L23_original_reference"]["reference"]["binary64"],
         "binary64 terminal input reanchored")
    binary64 = parent.checked_array(base.read_bound(final["input_binary64"]),
                                   (WIDTH,), "<f8")
    with parent.preflight.parent.producer.legacy.safe_open(
            assets["checkpoint"]["path"], framework="numpy") as model:
        weight = model.get_tensor("model.norm.weight")
    weights = validate_weight(weight, assets["tensors"]["model.norm.weight"])
    hidden = {"fp16": reference["stage18"],
              "binary64": contributions.exact_vector(binary64)}
    return {
        "result": result, "arrays": arrays, "references": references,
        "actual": actual, "hidden": hidden, "weights": weights,
        "archives": archives, "reference_archive": reference_archive,
        "binary64": binary64, "weight_array": weight,
        "assets": assets, "files": files,
        "hidden_pins": [*pins, final["input_binary64"]],
    }


def norm_summary(hidden):
    require(len(hidden) == WIDTH and all(isinstance(v, Fraction) for v in hidden),
            "invalid hidden accounting vector")
    mean = sum((v * v for v in hidden), Fraction()) / WIDTH
    radicand = mean + EPSILON
    rounded = float(radicand)
    require(math.isfinite(rounded) and rounded > 0, "invalid norm radicand")
    root = math.sqrt(rounded)
    inverse = Fraction.from_float(1.0 / root)
    return {
        "mean_square": str(mean), "epsilon": str(EPSILON),
        "radicand": str(radicand),
        "radicand_binary64_conversion_delta": str(Fraction(rounded) - radicand),
        "root_binary64_hex": root.hex(),
        "inverse_norm_anchor": str(inverse),
        "inverse_norm_anchor_hex": float(inverse).hex(),
        "inverse_square_identity_defect": str(inverse * inverse * radicand - 1),
    }


def energy_delta(actual, reference):
    delta = [a - r for a, r in zip(actual, reference, strict=True)]
    linear = [2 * r * d / WIDTH for r, d in zip(reference, delta, strict=True)]
    quadratic = [d * d / WIDTH for d in delta]
    return linear, quadratic


def account(a, r, ya, yr, weight, sa, sr, coordinate, linear, quadratic,
            q24):
    dh, ds = a - r, sa - sr
    direct, scale, interaction = weight * dh * sr, weight * r * ds, weight * dh * ds
    ba, br = ya - weight * a * sa, yr - weight * r * sr
    terms = dict(zip(TERMS, (direct, scale, interaction, ba - br), strict=True))
    observed = ya - yr
    require(sum(terms.values(), Fraction()) == observed, "hidden bridge closure failed")
    energy = sum(linear, Fraction()) + sum(quadratic, Fraction())
    selected = linear[coordinate] + quadratic[coordinate]
    return {
        "coordinate": coordinate,
        "actual_hidden": str(a), "reference_hidden": str(r), "hidden_delta": str(dh),
        "actual_rmsnorm": str(ya), "reference_rmsnorm": str(yr),
        "retained_rmsnorm_delta": str(observed), "weight": str(weight),
        "inverse_norm_anchor_delta": str(ds),
        **{key: str(value) for key, value in terms.items()},
        "actual_boundary_conversion_remainder": str(ba),
        "reference_boundary_conversion_remainder": str(br),
        "signed_term_sum": str(sum(terms.values(), Fraction())),
        "absolute_term_sum": str(sum(map(abs, terms.values()), Fraction())),
        "cancellation_absolute_mass": str(sum(map(abs, terms.values()), Fraction()) - abs(observed)),
        "selected_mean_square_linear_change": str(linear[coordinate]),
        "selected_mean_square_quadratic_change": str(quadratic[coordinate]),
        "other_coordinates_mean_square_change": str(energy - selected),
        "actual_raw_q24_hidden": str(q24),
        "actual_q24_to_stage18_conversion": str(a - q24),
        "exact_accounting_identity": True,
    }


def report(evidence):
    base.check_history(evidence["result"])
    same(list(evidence["arrays"]), list(parent.CONTROLS), "hidden bridge control census changed")
    refs = evidence["references"]
    output_refs = {"fp16": residual.words(refs["rmsnorm_fp16"]),
                   "binary64": contributions.exact_vector(refs["rmsnorm_binary64"])}
    reference_norms = {b: norm_summary(evidence["hidden"][b]) for b in BRANCHES}
    controls = []
    for label in parent.CONTROLS:
        a = evidence["actual"][label]["stage18"]
        ya = residual.words(evidence["arrays"][label]["rmsnorm"])
        an = norm_summary(a)
        sa = Fraction(an["inverse_norm_anchor"])
        branches = {}
        for branch in BRANCHES:
            r, yr = evidence["hidden"][branch], output_refs[branch]
            rn = reference_norms[branch]
            sr = Fraction(rn["inverse_norm_anchor"])
            linear, quadratic = energy_delta(a, r)
            change = Fraction(an["mean_square"]) - Fraction(rn["mean_square"])
            require(change == sum(linear, Fraction()) + sum(quadratic, Fraction()),
                    "global mean-square delta identity failed")
            deltas = [x - y for x, y in zip(ya, yr, strict=True)]
            order = sorted(range(WIDTH), key=lambda i: (-abs(deltas[i]), i))

            def selected(i):
                return account(a[i], r[i], ya[i], yr[i], evidence["weights"][i],
                               sa, sr, i, linear, quadratic,
                               evidence["actual"][label]["output_q24"][i])

            branches[branch] = {
                "hidden_reference": "original_input_L23_" + branch,
                "rmsnorm_reference": "original_input_final_attempt003_" + branch,
                "reference_norm": rn,
                "global_mean_square_delta": str(change),
                "global_mean_square_linear_change": str(sum(linear, Fraction())),
                "global_mean_square_quadratic_change": str(sum(quadratic, Fraction())),
                "hotspots": [{"absolute_rank": rank, **selected(i)}
                             for rank, i in enumerate(order[:LIMIT], 1)],
                "forced_coordinate62": {"absolute_rank": order.index(62) + 1, **selected(62)},
            }
        controls.append({"control": label, "actual_norm": an, "branches": branches})
    return {
        "selection": "top eight absolute retained final RMSNorm deltas, coordinate ascending on ties; coordinate62 separately",
        "scalar_anchor": "s=1/math.sqrt(float(exact_mean_square+1/1000000)); binary64 only, not an RMSNorm operator",
        "identity": "delta_y=w*delta_h*s_ref+w*h_ref*delta_s+w*delta_h*delta_s+delta_b; b=y-w*h*s",
        "energy_identity": "delta_mean_square=mean(2*h_ref*delta_h)+mean(delta_h**2)",
        "remainder_scope": "combined implementation, epsilon, normalization, multiply/conversion and scalar-anchor effects; not separately identifiable or rounding-only",
        "native_epsilon_q48": 281474977,
        "native_epsilon_minus_contract": str(Fraction(281474977, 1 << 48) - EPSILON),
        "logical_hotspot_accounts": len(controls) * len(BRANCHES) * LIMIT,
        "forced_coordinate_accounts": len(controls) * len(BRANCHES),
        "controls": controls,
    }


def measure():
    evidence = authenticate()
    evidence["report"] = report(evidence)
    return evidence


def focused_tests(evidence):
    compiled = []
    for path in (SOURCE, TEST):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        compiled.append(parent.record(path))
    spec = importlib.util.spec_from_file_location("final_rmsnorm_hidden_bridge_tests", TEST)
    require(spec is not None and spec.loader is not None, "focused test loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = evidence
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    same(suite.countTestCases(), EXPECTED_TESTS, "focused test census changed")
    outcome = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    require(outcome.wasSuccessful() and outcome.testsRun == EXPECTED_TESTS and not outcome.skipped,
            "hidden bridge focused tests failed, errored or skipped")
    return {"compiled": compiled, "executed": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors),
            "skipped": len(outcome.skipped)}


def check():
    require(Path.cwd() == ROOT and Path(__file__).resolve() == SOURCE
            and sys.executable == parent.PYTHON and os.getuid() == 1000
            and os.environ.get("PYTHONPATH") == str(ROOT)
            and sys.dont_write_bytecode and not sys.flags.optimize,
            "isolated source/workdir/account/interpreter/bytecode gate failed")
    origins = {**parent.source_context(), MODULE: parent.record(SOURCE),
               "hidden_bridge_test": parent.record(TEST)}
    audit = {"forbidden_calls": 0}
    with read_only(audit):
        evidence = measure()
        tests = focused_tests(evidence)
        for pin in (*origins.values(), *hotspots.PINS.values(),
                    *evidence["hidden_pins"]):
            base.read_bound(pin)
        same(audit, {"forbidden_calls": 0}, "hidden bridge attempted forbidden work")
    return {
        "diagnostic_id": NAME, "version": 1,
        "status": "READ_ONLY_FINAL_RMSNORM_HIDDEN_DELTA_BRIDGE",
        "command": COMMAND, "diagnostic_sources": origins,
        "pins": {**base.PINS, **hotspots.PINS},
        "original_execution_sources": parent.RETAINED_SOURCES,
        "authenticated_files": evidence["files"], "hidden_pins": evidence["hidden_pins"],
        "assets": evidence["assets"],
        "final_reference_authority": evidence["result"]["preflight"]["final_reference"],
        "retained_controls_and_failure_gates": evidence["result"]["controls"],
        "dispatch_and_write_audit": {**audit, **FLAGS}, "flags": FLAGS,
        "tests": tests, "normal_host_review": "REQUIRED", "claim_boundary": BOUNDARY,
        "report": evidence["report"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(check(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
