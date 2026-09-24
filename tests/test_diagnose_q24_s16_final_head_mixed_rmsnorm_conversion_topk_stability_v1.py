import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from ace3.model.candidates import (
    diagnose_q24_s16_final_head_mixed_rmsnorm_conversion_topk_stability_v1 as d,
)


def profile(ids, excluded, gap=1):
    return {
        "elements": 151936, "all_finite": True, "top_k_ids_diagnostic_only": ids,
        "lowest_included": {"token_id": ids[-1], "rank": 10, "value_hex": float(8 + gap).hex()},
        "highest_excluded": {"token_id": excluded, "rank": 11, "value_hex": float(8).hex()},
        "cutoff_gap": str(gap), "cutoff_tie": gap == 0,
    }


def comparison(a, r, distance):
    gained = sorted(set(a["top_k_ids_diagnostic_only"]) - set(r["top_k_ids_diagnostic_only"]))
    lost = sorted(set(r["top_k_ids_diagnostic_only"]) - set(a["top_k_ids_diagnostic_only"]))
    return {
        "actual_only_ids": gained, "reference_only_ids": lost,
        "overlap_count": 10 - len(gained),
        "same_order": a["top_k_ids_diagnostic_only"] == r["top_k_ids_diagnostic_only"],
        "cutoff_reversals": [{
            "actual_only_id": g, "reference_only_id": l, "actual_preference_margin": "1",
            "reference_preference_margin": "1", "actual_only_value_change": "2",
            "reference_only_value_change": "0", "relative_value_change": "2",
            "relative_change_minus_reference_margin": "1", "exact_reversal_identity": True,
            "tie_at_either_boundary": False, "adjacent_cutoff_in_both": False,
            "max_distance_from_cutoff_rank": distance,
        } for g in gained for l in lost],
    }


@pytest.fixture
def document():
    refs = {"fp16": profile(list(range(1, 10)) + [319], 34319),
            "binary64": profile(list(range(1, 10)) + [34319], 13)}
    branches = {}
    for branch in d.BRANCHES:
        policies = {}
        for policy in d.POLICIES:
            p = copy.deepcopy(refs["fp16" if branch == "independent_fp16" else "binary64"])
            p["vs_retained_references"] = {
                name: comparison(p, r, 4 if name == "binary64" else 5)
                for name, r in refs.items()}
            p["word_mismatches_vs_exact_rne"] = 2 if policy.startswith("torch") else 0
            policies[policy] = p
        branches[branch] = {
            "policies": policies, "forward_torch_vs_direct_rounding_word_count": 2,
            "reverse_torch_vs_direct_rounding_word_count": 2}
    doc = {
        **{key: {"retained": key} for key in d.PRESERVED},
        "flags": copy.deepcopy(d.PARENT_FLAGS), "retained_flags": copy.deepcopy(d.PARENT_FLAGS),
        "dispatch_and_write_audit": {
            **d.PARENT_FLAGS, "counterfactual_final_head_invocations": 9, "forbidden_calls": 0},
        "k": 10, "counterfactual_input_count": 3, "control_count": 9,
        "retained_L23_failures": 9, "normal_host_review": "REQUIRED",
        "reference_top_k": refs, "counterfactual_branches": branches,
        "controls": [{"control": str(i), "L21_L22_L23_status": ["FAIL"] * 3,
                      "retained_failures": {"L21": ["FAIL"]},
                      "retained_L23_and_ancestral_lineage": {"original": i}} for i in range(9)]}
    return doc


def test_supported_and_preserves_every_history(document):
    before = copy.deepcopy(document)
    result = d.classify(document)
    assert result["classification"] == "SUPPORTED" and result["lane_terminated"]
    assert len(result["torch_vs_direct"]) == 6 and len(result["vs_exact_rne_diagnostic_only"]) == 12
    assert not result["counterexamples"] and document == before
    assert result["retained_context"] == {key: document[key] for key in d.PRESERVED}
    for branch, policies in result["branches"].items():
        assert set(policies) == set(d.POLICIES)
        for p in policies.values():
            expected = (10, 11) if branch == "independent_fp16" else (15, 10)
            assert tuple(p["selected_ids"][str(i)]["rank"] for i in (319, 34319)) == expected
            assert p["selected_ids"]["319"]["present"] == (branch == "independent_fp16")


def test_rejects_order_only_change_with_exact_rank_difference(document):
    p = document["counterfactual_branches"]["frozen_inherited"]["policies"]["forward_direct_rne"]
    p["top_k_ids_diagnostic_only"][:2] = [2, 1]
    p["vs_retained_references"]["binary64"]["same_order"] = False
    result = d.classify(document)
    assert result["classification"] == "REJECTED" and result["lane_terminated"]
    counterexample, = result["counterexamples"]
    assert counterexample["top_k_membership_equal"] and not counterexample["top_k_order_equal"]
    assert counterexample["branch"] == "frozen_inherited"
    assert counterexample["right_policy"] == "forward_direct_rne"
    assert counterexample["order_differences"] == [
        {"rank": 1, "left_id": 1, "right_id": 2}, {"rank": 2, "left_id": 2, "right_id": 1}]


def test_rejects_exchanged_membership(document):
    policies = document["counterfactual_branches"]
    policies["mapped_all"]["policies"]["reverse_direct_rne"] = copy.deepcopy(
        policies["independent_fp16"]["policies"]["reverse_direct_rne"])
    result = d.classify(document)
    assert result["classification"] == "REJECTED"
    row, = result["counterexamples"]
    assert row["left_only_ids"] == [34319] and row["right_only_ids"] == [319]
    assert row["classification_difference"] is not None


def test_rejects_cutoff_tie_without_top_k_change(document):
    p = document["counterfactual_branches"]["mapped_all"]["policies"]["forward_direct_rne"]
    p["lowest_included"]["value_hex"] = p["highest_excluded"]["value_hex"]
    p["highest_excluded"]["token_id"] = 40000
    p["cutoff_gap"], p["cutoff_tie"] = "0", True
    row, = d.classify(document)["counterexamples"]
    assert row["top_k_order_equal"] and not row["classification_equal"]
    assert row["cutoff"]["right"]["cutoff_tie"]


def test_positive_gap_change_is_not_a_classification_change(document):
    p = document["counterfactual_branches"]["mapped_all"]["policies"]["forward_direct_rne"]
    p["lowest_included"]["value_hex"], p["cutoff_gap"] = float(10).hex(), "2"
    result = d.classify(document)
    assert result["classification"] == "SUPPORTED"
    assert any(not row["cutoff_gap_equal"] for row in result["torch_vs_direct"])


def test_rejects_pair_classification_change_without_top_k_change(document):
    p = document["counterfactual_branches"]["frozen_inherited"]["policies"]["torch_reverse"]
    pair = p["vs_retained_references"]["fp16"]["cutoff_reversals"][0]
    pair.update(actual_preference_margin="0", actual_only_value_change="1",
                relative_value_change="1", relative_change_minus_reference_margin="0",
                tie_at_either_boundary=True)
    row, = d.classify(document)["counterexamples"]
    assert row["top_k_order_equal"] and not row["classification_equal"]
    assert row["left_policy"] == "torch_reverse"


@pytest.mark.parametrize("mutation", [
    lambda x: x["flags"].update(candidate_admitted=True),
    lambda x: x["flags"].update(evidence_writes=False),
    lambda x: x["retained_flags"].pop("new_token_claim"),
    lambda x: x["controls"][0].update(L21_L22_L23_status=["PASS"] * 3),
    lambda x: x["counterfactual_branches"].pop("mapped_all"),
    lambda x: x["counterfactual_branches"]["mapped_all"]["policies"].pop("torch_reverse"),
    lambda x: x["counterfactual_branches"]["mapped_all"].update(
        forward_torch_vs_direct_rounding_word_count=0),
    lambda x: x.pop("thresholds"),
])
def test_unknown_contains_only_missing_fact_not_partial_science(document, mutation, monkeypatch):
    mutation(document)
    monkeypatch.setattr(d, "load_capture", lambda: (document, {}))
    result = d.check()
    assert result["classification"] == "UNKNOWN" and result["missing_binding"]
    assert result["lane_terminated"] is False and "branches" not in result
    assert result["flags"]["candidate_admitted"] is False


def test_ambiguous_rank_not_invented(document):
    p = document["counterfactual_branches"]["mapped_all"]["policies"]["exact_rne"]
    p["vs_retained_references"]["fp16"]["cutoff_reversals"][0]["max_distance_from_cutoff_rank"] = 1
    with pytest.raises(d.BindingError, match="rank"):
        d.classify(document)


def test_numeric_tie_and_fraction_checks():
    p = profile(list(range(1, 10)) + [319], 34319, gap=0)
    d.validate_profile(p, "synthetic")
    p["highest_excluded"]["token_id"] = 13
    with pytest.raises(d.BindingError, match="numeric tie order"):
        d.validate_profile(p, "synthetic")
    with pytest.raises(d.BindingError, match="fraction"):
        d.fraction("1/0", "fraction")


def test_binding_tamper_and_wrong_path(tmp_path):
    path = tmp_path / "member"
    path.write_bytes(b"original")
    pin = d.fixed_pin(path, 8, hashlib.sha256(b"original").hexdigest())
    assert d.read_pin(pin, path) == b"original"
    path.write_bytes(b"tampered")
    with pytest.raises(d.BindingError, match="byte binding"):
        d.read_pin(pin, path)
    with pytest.raises(d.BindingError, match="member path"):
        d.read_pin(pin, tmp_path / "other")


def test_command_identity_survives_json_key_sorting():
    environment = {"B": "", "A": "x y", "C": "a=b"}
    argv = ["python", "-c", "print('x y')"]
    command = d.capture.environment_command(environment, argv)
    persisted = json.loads(d.capture.encoded(environment))
    assert list(environment) != list(persisted)
    d.validate_command(command, argv, persisted, "command binding")


@pytest.mark.parametrize("command", [
    "A=x A=y python -B",
    "A=x B=changed python -B",
    "A=x B=y python --execute",
])
def test_command_binding_refuses_changed_environment_or_argv(command):
    with pytest.raises(d.BindingError, match="command binding"):
        d.validate_command(command, ["python", "-B"], {"A": "x", "B": "y"}, "command binding")


@pytest.mark.parametrize("error", [
    FileNotFoundError("retained capture absent"),
    d.BindingError("source/test pin mismatch"),
    d.BindingError("launcher/run command/environment binding"),
    json.JSONDecodeError("retained check.stdout JSON", "", 0),
])
def test_authentication_failures_do_not_classify(error, monkeypatch):
    def unavailable():
        raise error

    def forbidden(*args):
        raise AssertionError("classification reached without authenticated input")

    monkeypatch.setattr(d, "load_capture", unavailable)
    monkeypatch.setattr(d, "classify", forbidden)
    result = d.check()
    assert result["classification"] == "UNKNOWN" and result["missing_binding"] == str(error)
    assert "retained_context" not in result and "capture_bindings" not in result


def test_no_producer_import_or_dispatch_route(document, monkeypatch):
    tree = ast.parse(Path(d.__file__).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {n.module if isinstance(n, ast.ImportFrom) else alias.name
             for n in imports for alias in n.names}
    assert names <= {"argparse", "hashlib", "json", "math", "shlex", "fractions", "pathlib",
                     "ace3.model.candidates"}
    assert [a.name for n in imports if isinstance(n, ast.ImportFrom)
            and n.module == "ace3.model.candidates" for a in n.names] == ["diagnostic_capture_v1"]
    monkeypatch.setattr(d, "load_capture", lambda: (document, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden dispatch")

    monkeypatch.setattr(d.capture, "run_command", forbidden)
    monkeypatch.setattr(d.capture, "seal_launcher", forbidden)
    monkeypatch.setattr(d.capture, "_execute", forbidden)
    assert d.check()["classification"] == "SUPPORTED"


def test_cli_single_document_and_forbidden_options(document, monkeypatch, capsys):
    monkeypatch.setattr(d, "load_capture", lambda: (document, {}))
    assert d.main(["--check"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["classification"] == "SUPPORTED" and not output.err
    with pytest.raises(SystemExit):
        d.main(["--check", "--execute"])
    monkeypatch.setattr(d, "load_capture", lambda: (_ for _ in ()).throw(
        d.BindingError("missing retained check.stdout binding")))
    assert d.main(["--check"]) == 1
