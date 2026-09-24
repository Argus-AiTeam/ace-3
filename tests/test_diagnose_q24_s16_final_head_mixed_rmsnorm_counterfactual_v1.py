"""Independent Fraction/sorting oracles and fail-closed counterfactual checks."""

from copy import deepcopy
import fcntl
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import diagnose_q24_s16_final_head_mixed_rmsnorm_counterfactual_v1 as d
from ace3.model.candidates import diagnostic_capture_v1 as capture


EVIDENCE = None


def words(values):
    return np.asarray(values, dtype="<f2").view("<u2")


def oracle_dot(weight, hidden):
    return sum((Fraction(float(a)) * Fraction(float(b))
                for a, b in zip(weight, hidden, strict=True)), Fraction())


class MixedRMSNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EVIDENCE is None:
            raise RuntimeError("live tests require the one captured --check evidence; "
                               "standalone pytest selects SyntheticMixedRMSNormTests")
        (cls.result, cls.arrays, cls.references, cls.inputs, cls.aliases,
         cls.outputs, cls.weights, cls.report) = EVIDENCE

    def test_nine_controls_and_bounded_inputs(self):
        self.assertEqual(self.report["control_count"], 9)
        self.assertEqual([r["control"] for r in self.report["controls"]], list(d.parent.CONTROLS))
        self.assertEqual(self.report["counterfactual_input_count"], 3)
        self.assertEqual(len(self.report["counterfactual_branches"]), 3)

    def test_every_actual_logit_reproduced(self):
        for label in d.parent.CONTROLS:
            self.assertEqual(self.outputs[self.aliases[label]]["exact_rne"].tobytes(),
                             self.arrays[label]["logits"].tobytes())

    def test_every_independent_fp16_logit_reproduced(self):
        self.assertEqual(self.outputs["independent_fp16"]["torch_forward"].tobytes(),
                         self.references["logits_fp16"].tobytes())

    def test_live_exact_dots_independent_fraction_oracle(self):
        for name, hidden in self.inputs.items():
            for index, weight in self.weights.items():
                expected = oracle_dot(weight, hidden.view("<f2"))
                self.assertEqual(Fraction(int(self.outputs[name]["q48"][index]), 1 << 48), expected)
                self.assertEqual(
                    int(self.outputs[name]["exact_rne"][index]),
                    int(words([float(expected)])[0]))

    def test_live_top_k_independent_sorting_oracle(self):
        for name, output in self.outputs.items():
            for policy in (*d.POLICIES, "forward_direct_rne", "reverse_direct_rne"):
                values = output[policy].view("<f2")
                order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
                summary = self.report["counterfactual_branches"][name]["policies"][policy]
                self.assertEqual(summary["top_k_ids_diagnostic_only"], order[:10])
                self.assertEqual(summary["highest_excluded"]["token_id"], order[10])
                self.assertEqual(Fraction(summary["cutoff_gap"]),
                                 Fraction(float(values[order[9]])) - Fraction(float(values[order[10]])))

    def test_live_pair_decomposition_independent_oracle(self):
        reversed_pairs = 0
        for row in self.report["controls"]:
            actual = self.inputs[self.aliases[row["control"]]].view("<f2")
            reference = self.inputs["independent_fp16"].view("<f2")
            for pair in row["pair_effects"]:
                a, b = pair["actual_only_id"], pair["reference_only_id"]
                ea = oracle_dot(self.weights[a], actual) - oracle_dot(self.weights[b], actual)
                er = oracle_dot(self.weights[a], reference) - oracle_dot(self.weights[b], reference)
                self.assertEqual(Fraction(pair["rmsnorm_vector_effect_fixed_exact_head"]), ea - er)
                self.assertEqual(Fraction(pair["actual_exact_margin"]), ea)
                self.assertEqual(Fraction(pair["reference_exact_margin"]), er)
                self.assertEqual(
                    Fraction(pair["total_retained_relative_change"]),
                    Fraction(pair["rmsnorm_vector_effect_fixed_exact_head"])
                    + Fraction(pair["combined_head_effect"]))
                self.assertEqual(pair["vector_swap_reverses_pair_under_exact_head"], ea > 0 > er)
                reversed_pairs += ea > 0 > er
        mechanism = self.report["mechanism"]
        self.assertTrue(mechanism["complete_nine_control_three_branch_pair_census"])
        self.assertEqual(mechanism["exchanged_pair_count"], 9)
        self.assertEqual(mechanism["closed_additive_pair_count"], 9)
        self.assertEqual(mechanism["exact_head_vector_swap_reversal_count"], reversed_pairs)
        self.assertEqual(mechanism["classification"],
                         "SUPPORTED" if reversed_pairs == 9 else "REJECTED")

    def test_live_rounding_and_order_counts(self):
        for name, output in self.outputs.items():
            summary = self.report["counterfactual_branches"][name]
            self.assertEqual(summary["order_effect_fp16_word_count"],
                             int(np.count_nonzero(output["torch_forward"] != output["torch_reverse"])))
            self.assertEqual(summary["forward_torch_vs_direct_rounding_word_count"],
                             int(np.count_nonzero(output["torch_forward"] != output["forward_direct_rne"])))

    def test_live_rmsnorm_error_fraction_oracle(self):
        for row in self.report["controls"]:
            actual = self.arrays[row["control"]]["rmsnorm"].view("<f2")
            expected = max(abs(Fraction(float(a)) - Fraction(float(b)))
                           for a, b in zip(actual, self.references["rmsnorm_binary64"], strict=True))
            self.assertEqual(Fraction(row["rmsnorm"]["binary64_max_absolute_error"]), expected)

    def test_histories_and_reference_authority_preserved(self):
        self.assertEqual(self.report["thresholds"], self.result["preflight"]["thresholds"])
        self.assertEqual(self.report["original_global_reference"],
                         self.result["preflight"]["final_reference"])
        self.assertEqual(self.report["retained_flags"], self.result["flags"])
        self.assertEqual(self.report["retained_L23_stage_reports"], {"PASS": 162, "FAIL": 9})
        for row in self.report["controls"]:
            self.assertEqual(row["L21_L22_L23_status"], ["FAIL"] * 3)
            self.assertEqual(set(row["retained_failures"]), {"L21", "L22", "L23"})
            self.assertEqual(row["retained_top_k"]["comparisons"]["binary64"]["overlap_count"], 10)
            self.assertEqual(row["retained_top_k"]["comparisons"]["fp16"]["overlap_count"], 9)
            history = row["retained_L23_and_ancestral_lineage"]
            for item in (history, history["retained_L21"], history["retained_L22"]):
                self.assertEqual(item["mandatory_statuses"], ["PASS"] * 18 + ["FAIL"])
                for flag in ("candidate_admitted", "policy_adopted", "successor_published"):
                    self.assertIs(item[flag], False)

    def test_history_gate_and_threshold_mutations_refused(self):
        for key, value in (("candidate_admitted", True),
                           ("source_operand_state_KV_RTZ_checks", "FAIL"),
                           ("S18_failure_indices", [])):
            changed = deepcopy(self.result)
            changed["controls"][0]["parent"]["retained_L23"][key] = value
            with self.assertRaises(ValueError):
                d.base.check_history(changed)
        changed = deepcopy(self.result)
        changed["controls"][0]["parent"]["L23_failures"][0]["excess_budget"] = "1/4"
        with self.assertRaises(ValueError):
            d.base.check_history(changed)

    def test_source_pin_mutation_refused(self):
        with self.assertRaises(ValueError):
            d.base.read_bound({**d.CUTOFF_PIN, "sha256": "0" * 64})

    def test_reference_and_tokenizer_binding_mutations_refused(self):
        summary = self.result["preflight"]
        changed = deepcopy(summary["assets"])
        changed["tokenizer"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.preflight.bind_final_reference(summary["L23_original_reference"], changed)

    def test_retained_logit_policy_counterexamples_rejected(self):
        for branch, policy in ((self.aliases[d.parent.CONTROLS[0]], "exact_rne"),
                               ("independent_fp16", "torch_forward")):
            changed = deepcopy(self.outputs)
            changed[branch][policy][0] ^= 1
            report = d.report(self.result, self.arrays, self.references,
                              self.inputs, self.aliases, changed)
            self.assertEqual(report["mechanism"]["classification"], "REJECTED")
            self.assertTrue(report["mechanism"]["logit_policy_counterexamples"])
            self.assertTrue(report["lane_terminated"])
            self.assertEqual(report["retained_flags"], self.result["flags"])


class SyntheticMixedRMSNormTests(unittest.TestCase):
    def test_exact_cancellation_and_subnormal_products(self):
        hidden = words([1, 1, 1, 2 ** -24])
        weight = np.asarray([[32, 2 ** -12, -32, 2 ** -24],
                             [0, 0, 0, -2 ** -24]], dtype="<f2")
        sums, rounded = d.exact_head(hidden, weight)
        for i in range(2):
            expected = oracle_dot(weight[i], hidden.view("<f2"))
            self.assertEqual(Fraction(int(sums[i]), 1 << 48), expected)
            self.assertEqual(int(rounded[i]), int(words([float(expected)])[0]))

    def test_exact_rne_ties(self):
        hidden = words([1, 2 ** -11])
        weight = np.asarray([[1, 1], [1, 3]], dtype="<f2")
        _, actual = d.exact_head(hidden, weight)
        self.assertEqual(actual.tobytes(), words([1, 1 + 2 ** -9]).tobytes())

    def test_double_rounding_discriminator(self):
        hidden = words([1, 2 ** -11, 2 ** -24])
        weight = np.asarray([[1, 1, 2 ** -10]], dtype="<f2")
        _, exact = d.exact_head(hidden, weight)
        values, rounded, direct = d.float_head(hidden, weight)
        self.assertEqual(Fraction(float(values[0])), oracle_dot(weight[0], hidden.view("<f2")))
        self.assertEqual(int(exact[0]), 0x3C01)
        self.assertEqual(int(direct[0]), 0x3C01)
        self.assertEqual(int(rounded[0]), 0x3C00)

    def test_nonfinite_wrong_shape_and_overflow_refused(self):
        for hidden, weight in (
                (words([float("nan")]), np.ones((1, 1), dtype="<f2")),
                (words([1]), np.asarray([[float("inf")]], dtype="<f2")),
                (words([1, 2]), np.ones((1, 1), dtype="<f2")),
                (words([65504]), np.asarray([[65504]], dtype="<f2"))):
            with self.assertRaises(ValueError):
                d.exact_head(hidden, weight)

    def test_pair_zero_tie_no_invented_attribution(self):
        output = {"q48": np.zeros(2, dtype="<i8")}
        effect = d.pair_effect(0, 1, output, output, words([0, 0]), words([0, 0]))
        self.assertIsNone(effect["vector_share_of_signed_change"])
        self.assertTrue(effect["exact_margin_tie"])
        self.assertFalse(effect["vector_swap_reverses_pair_under_exact_head"])
        rows = [{"control": label, "pair_effects": [effect]} for label in d.parent.CONTROLS]
        self.assertEqual(d.classify_mechanism(rows, 3)["classification"], "REJECTED")
        for changed, branches in (([], 3), (rows, 2),
                                  ([{**row, "pair_effects": []} for row in rows], 3)):
            self.assertEqual(d.classify_mechanism(changed, branches)["classification"], "REJECTED")
        for change in ({"vector_swap_reverses_pair_under_exact_head": True},
                       {"total_retained_relative_change": "1"},
                       {"reference_only_id": 0}):
            altered = deepcopy(rows)
            altered[0]["pair_effects"][0].update(change)
            with self.assertRaises(ValueError):
                d.classify_mechanism(altered, 3)
        malformed = deepcopy(rows)
        del malformed[0]["pair_effects"][0]["actual_exact_margin"]
        with self.assertRaises(ValueError):
            d.classify_mechanism(malformed, 3)

    def test_numeric_tie_order(self):
        p = d.cutoff.profile(np.asarray([0, -0.0, -1], dtype="<f8"), 1)
        self.assertEqual(p["summary"]["top_k_ids_diagnostic_only"], [0])
        self.assertTrue(p["summary"]["cutoff_tie"])

    def test_logit_reproduction_exact_words_and_counterexample(self):
        retained = words([0, 1])
        actual = words([-0.0, 1])
        self.assertEqual(d.logit_reproduction(retained, retained), {
            "word_count": 2, "word_mismatch_count": 0, "first_counterexample": None})
        self.assertEqual(d.logit_reproduction(actual, retained), {
            "word_count": 2, "word_mismatch_count": 1,
            "first_counterexample": {"index": 0, "computed_word": 32768, "retained_word": 0}})
        with self.assertRaises(ValueError):
            d.logit_reproduction(actual[:1], retained)

    def test_dispatch_and_filesystem_mutations_blocked(self):
        audit = {"forbidden_calls": 0}
        calls = [
            lambda: d.parent.execute("unused"),
            lambda: d.parent.logits(None, None),
            lambda: d.parent.norm.rmsnorm(None, None),
            lambda: subprocess.Popen(["false"]),
            lambda: open(d.SOURCE, "wb"),
            lambda: os.open(d.SOURCE, os.O_WRONLY),
            lambda: os.rename(d.SOURCE, d.TEST),
        ]
        with d.base.read_only(audit):
            for call in calls:
                with self.assertRaises(RuntimeError):
                    call()
        self.assertEqual(audit["forbidden_calls"], len(calls))

    def test_read_access_and_non_admission_flags(self):
        with d.SOURCE.open("rb") as stream:
            self.assertTrue(stream.read(3))
        for name in ("native_layer_invocations", "decoder_invocations", "evidence_writes",
                     "reference_rmsnorm_invocations", "reference_lm_head_invocations"):
            self.assertEqual(d.FLAGS[name], 0)
        for name in ("token_published", "token_selection", "token_selected_for_feedback",
                     "rmsnorm_recomputation", "original_reference_recomputation", "upstream_cause_identified"):
            self.assertIs(d.FLAGS[name], False)
        self.assertIn("independent Reviewer validation is REQUIRED", d.BOUNDARY)

    def test_cli_single_json_document(self):
        stdout = io.StringIO()
        with patch.object(d, "check", return_value={"controls": 9}), patch("sys.stdout", stdout):
            d.main(["--check"])
        self.assertEqual(json.loads(stdout.getvalue()), {"controls": 9})
        self.assertEqual(stdout.getvalue().count("\n"), 1)
        for error in (ValueError("source pin changed"), RuntimeError("forbidden dispatch"),
                      OSError("reference unavailable")):
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(d, "check", side_effect=error), patch("sys.stdout", stdout), \
                    patch("sys.stderr", stderr), self.assertRaises(SystemExit) as exited:
                d.main(["--check"])
            self.assertEqual(exited.exception.code, 1)
            failure = json.loads(stdout.getvalue())
            self.assertEqual(failure["status"], "UNKNOWN")
            self.assertEqual(failure["mechanism"]["classification"], "UNKNOWN")
            self.assertEqual(failure["failure"]["message"], str(error))
            self.assertEqual(failure["normal_host_review"], "REQUIRED")
            self.assertNotIn("dispatch_and_write_audit", failure)
            self.assertIn("UNKNOWN", stderr.getvalue())
            self.assertEqual(stdout.getvalue().count("\n"), 1)

    def test_cli_execute_and_output_options_refused(self):
        for args in ([], ["--execute"], ["--check", "--out", "unused"],
                     ["--check", "--decode"], ["--check", "--admit"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                d.main(args)


ENVIRONMENT = {
    "PATH": str(Path(d.parent.PYTHON).parent) + ":/usr/bin:/bin",
    "PYTHONPATH": str(d.ROOT), "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def capture_run(directory):
    directory = Path(directory).absolute()
    d.require(directory.name == "run" and directory.parent.parent == d.ROOT / "build"
              and directory.parent.name.startswith("mixed-rmsnorm-final-head-")
              and directory.resolve() == directory, "isolated capture run path required")
    d.require((Path.cwd(), os.getuid(), sys.executable) == (d.ROOT, 1000, d.parent.PYTHON),
              "capture account/workdir/interpreter gate")
    directory.mkdir()
    sources = [capture.binding(d.SOURCE), capture.binding(d.TEST)]
    preflight = {
        "cwd": str(d.ROOT), "uid": os.getuid(), "python": sys.executable,
        "python_version": sys.version, "environment": ENVIRONMENT, "sources": sources,
        "executable": capture.binding(Path(sys.executable).resolve()),
        "independent_host_review": "REQUIRED", "scope": d.MODULE,
        "mission_id": "a399e916489f", "role": "engineer",
        "host_managed_controls": "unchanged role/model/account/budget/access; no service calls",
        "model_or_service_calls_authorized": 0,
        "command_budget": {"compile": 1, "pytest": 1, "check": 1},
        "counterfactual_head_policy_budget": 9,
        "capture_implementation": capture.implementation_pins(),
    }
    results, failure, classification = [], None, "UNKNOWN"

    def run(label, argv):
        return capture.run_command(directory, label, argv, preflight, results, timeout=90)

    try:
        d.require(run("branch", ["git", "branch", "--show-current"])
                  == b"argus/full-projection\n", "capture branch mismatch")
        run("ignored-build", ["git", "check-ignore", str(directory)])
        probe = (
            "import json,os,subprocess; "
            "rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines(); "
            f"module={d.MODULE!r}; test={str(d.TEST)!r}; "
            "matches=[r for r in rows if int(r.split(None,1)[0]) != os.getpid() "
            "and ((' -m '+module+' ') in r or (' -m pytest ' in r and test in r))]; "
            "print(json.dumps(matches))"
        )
        d.same(json.loads(run("concurrency", [sys.executable, "-B", "-c", probe])),
               [], "same-scope diagnostic/test already running")
        compile_code = (
            "import py_compile; "
            f"paths={[str(d.SOURCE), str(d.TEST)]!r}; out={str(directory)!r}; "
            "[py_compile.compile(p,cfile=out+'/compiled-'+str(i)+'.pyc',doraise=True) "
            "for i,p in enumerate(paths)]; print('py_compile: 2 files')"
        )
        run("compile", [sys.executable, "-B", "-c", compile_code])
        run("pytest", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--basetemp", str(directory / "pytest-tmp"),
                       "--junitxml", str(directory / "pytest.xml"),
                       str(d.TEST) + "::SyntheticMixedRMSNormTests"])
        result = json.loads(run("check", [sys.executable, "-B", "-m", d.MODULE, "--check"]))
        classification = result["mechanism"]["classification"]
        d.require(classification in ("SUPPORTED", "REJECTED") and result["lane_terminated"],
                  "missing terminal scientific classification")
        d.same(result["flags"], d.FLAGS, "non-admission flags changed")
        d.same(result["tests"]["executed"], d.EXPECTED_TESTS, "focused test census changed")
        d.same(result["dispatch_and_write_audit"]["forbidden_calls"], 0, "forbidden dispatch")
        d.same(result["dispatch_and_write_audit"]["counterfactual_final_head_invocations"],
               9, "bounded final-head policy budget changed")
        d.same([capture.binding(d.SOURCE), capture.binding(d.TEST)], sources, "source drift")
        d.same(capture.implementation_pins(), preflight["capture_implementation"],
               "capture implementation drift")
        for command in results:
            d.same([capture.binding(pin["path"]) for pin in command["files"]],
                   command["files"], "captured command bytes changed")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        classification = "UNKNOWN"
    receipt = {
        "preflight": preflight, "results": results, "failure": failure,
        "success": failure is None, "classification": classification,
        "sources_after": [capture.binding(d.SOURCE), capture.binding(d.TEST)],
    }
    pin = capture.save(directory, "capture.json", capture.encoded(receipt))
    print(json.dumps({"capture": pin, "success": failure is None,
                      "classification": classification, "failure": failure}))
    return int(failure is not None)


def launch(directory):
    directory = Path(directory).absolute()
    d.require(directory.parent == d.ROOT / "build"
              and directory.name.startswith("mixed-rmsnorm-final-head-")
              and directory.resolve() == directory, "fresh isolated capture attempt required")
    d.require((Path.cwd(), os.getuid(), sys.executable) == (d.ROOT, 1000, d.parent.PYTHON),
              "launcher account/workdir/interpreter gate")
    # A stable in-scope lock covers compile, tests and capture without touching sibling locks.
    descriptor = os.open(d.ROOT / "build/mixed-rmsnorm-final-head-capture.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory.mkdir(mode=0o700)
        argv = [sys.executable, "-B", str(d.TEST), "--run", str(directory / "run")]
        identity = {
            "argv": argv, "command": capture.environment_command(
                dict(sorted(ENVIRONMENT.items())), argv),
            "cwd": str(d.ROOT), "uid": os.getuid(), "environment": ENVIRONMENT,
            "launcher": capture.binding(d.TEST), "source": capture.binding(d.SOURCE),
            "executable": capture.binding(Path(sys.executable).resolve()),
            "invocation_argv": sys.orig_argv, "scope": d.MODULE,
            "independent_host_review": "REQUIRED", "model_or_service_calls_authorized": 0,
            "capture_implementation": capture.implementation_pins(),
        }
        output, outer = capture.seal_launcher(directory, identity, timeout=115)
        receipt = capture.binding(directory / "launcher.capture.json")
        capture.verify_launcher(directory, identity, receipt, require_success=False)
        print(json.dumps({"capture": receipt, "success": outer["success"],
                          "exit_status": outer["exit_status"],
                          "inner_summary": output.decode(),
                          "independent_host_review": "REQUIRED"}))
        return outer["exit_status"] or int(not outer["success"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("--launch", "--run"):
        raise SystemExit(launch(sys.argv[2]) if sys.argv[1] == "--launch"
                         else capture_run(sys.argv[2]))
    unittest.main()
