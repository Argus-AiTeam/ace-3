"""Synthetic resume authentication and dispatch checks, not numerical admission."""

import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ace3.model.candidates import q24_s16_toward_zero_native_v1 as candidate
from ace3.model.candidates import run_q24_s16_toward_zero_l3_l8_v1 as continuation


def historical_native_source(current):
    old = ast.parse(current)
    shared = next(node for node in old.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_stages")
    legacy = next(node for node in old.body
                  if isinstance(node, ast.FunctionDef) and node.name == "stages")
    shared.name = "stages"
    shared.body.insert(0, copy.deepcopy(legacy.body[0]))
    old.body = [node for node in old.body if node is shared
                or not isinstance(node, ast.FunctionDef)
                or node.name not in {"stages", "continuation_stages"}]
    return ast.unparse(old)


class ContinuationTests(unittest.TestCase):
    def test_dispatch_is_explicit_and_bounded(self):
        for layer in (0, 2, 9, 23, True, 3.0):
            with self.assertRaises(ValueError):
                next(candidate.continuation_stages({}, layer, {}, {}))
        with self.assertRaises(ValueError):
            next(candidate.stages({}, 3, {}, {}))
        for layer in range(3, 9):
            with patch.object(candidate, "_stages", return_value=iter(range(19))) as execute:
                self.assertEqual(list(candidate.continuation_stages({}, layer, {}, {})), list(range(19)))
                execute.assert_called_once_with({}, layer, {}, {})

    def test_wrong_parent_rejected_before_any_read(self):
        with self.assertRaisesRegex(ValueError, "Planner-selected"):
            continuation.authenticate_resume(Path("/wrong/parent.json"), None, None, None)

    def test_parent_scope_rejects_stale_and_runtime_receipts(self):
        parent = {
            "candidate_id": "ace3-q24-s16-toward-zero-native-v1",
            "state_id": "ace3-q24-software-paired-state-v1",
            "policy_id": continuation.runner.gates.POLICY_ID,
            "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "history": [9707], "position": 0, "next_layer": 3,
            "evidence_kind": "cpu_software_q24", "rtl_admissible": False,
            "normal_host_review": "REQUIRED"}
        continuation.validate_parent_scope(parent)
        for key, value in (("next_layer", 2), ("next_layer", 9), ("next_layer", 3.0),
                           ("position", False), ("position", 1), ("history", [9707, 0]),
                           ("rtl_admissible", True), ("evidence_kind", "rtl"),
                           ("normal_host_review", "PASS"), ("state_id", "fp16"),
                           ("policy_id", "legacy")):
            bad = dict(parent, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                continuation.validate_parent_scope(bad)

    def test_compatibility_does_not_hide_arithmetic_changes(self):
        current = Path(candidate.__file__).read_text()
        historical = historical_native_source(current)
        continuation.verify_native_compatibility(historical, current)
        for bad in (current.replace("/ 8)", "/ 4)"),
                    current.replace("torch.mv(", "torch.mm("),
                    current.replace("state.add(scratch,", "state.add(parent,")):
            with self.assertRaisesRegex(ValueError, "arithmetic changed"):
                continuation.verify_native_compatibility(historical, bad)


class ResumeAuthenticationFixture:
    @staticmethod
    def rewrite_record(record, value):
        path = Path(record["path"])
        if isinstance(value, str):
            path.write_text(value)
        elif path.suffix == ".npz":
            with path.open("wb") as stream:
                np.savez(stream, **value)
        else:
            path.write_text(json.dumps(value))
        record.update(continuation.runner.retained.record(path))

    @contextlib.contextmanager
    def fixture(self):
        retained = continuation.runner.retained
        build = continuation.ROOT / "build"
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="q24_resume_auth_", dir=build) as temporary:
            root = Path(temporary)
            out = root / "blocked"
            out.mkdir()
            native = root / "native.py"
            native.write_text(historical_native_source(Path(candidate.__file__).read_text()))
            state_source = root / "state.py"
            state_source.write_text(Path(candidate.state.__file__).read_text())
            native_record = retained.record(native)
            native_original = dict(native_record, path=str(Path(candidate.__file__).resolve()))
            freeze = {
                "contract": {"scope": {"layers": [0, 1, 2], "position": 0, "history": [9707]}},
                "input_bindings": [
                    native_original, retained.record(continuation.runner.__file__),
                    retained.record(candidate.state.__file__)],
                "sources": [
                    {"snapshot": native_record, "original": native_original},
                    {"snapshot": retained.record(state_source),
                     "original": retained.record(candidate.state.__file__)}]}
            retained.write(root / "freeze.json", freeze)
            freeze_record = retained.record(root / "freeze.json")
            states = [
                {"i": np.full(896, index << 24, dtype="<i8"),
                 "z": np.zeros(896, dtype="u1"),
                 "h": np.full(896, word, dtype="<u2")}
                for index, word in enumerate((0, 0x3c00, 0x4000, 0x4200))]
            state_records = [retained.save(root / f"state{index}.npz", state)
                             for index, state in enumerate(states)]
            actual, caches, reports, layers = [], [], [], []
            for layer in range(3):
                incoming, outgoing = states[layer:layer + 2]
                cache = {name: np.full((1, 128), 0x3c00 + layer * 16 + offset, dtype="<u2")
                         for offset, name in enumerate(("k", "v"))}
                arrays = {
                    "input_i": incoming["i"].copy(), "input_z": incoming["z"].copy(),
                    "input_hidden": incoming["h"].copy(),
                    "output_i": outgoing["i"].copy(), "output_z": outgoing["z"].copy(),
                    "stage18": outgoing["h"].copy(),
                    "output_cache_k": cache["k"].copy(), "output_cache_v": cache["v"].copy()}
                report = [{"stage": stage, "node": [layer, 0, stage], "status": "PASS"}
                          for stage in range(19)]
                report_path = root / f"reports{layer}.json"
                retained.write(report_path, report)
                layers.append({
                    "layer": layer, "position": 0, "status": "PASS",
                    "input_parent": state_records[layer], "output_parent": state_records[layer + 1],
                    "actual_stages": retained.save(root / f"actual{layer}.npz", arrays),
                    "reports": retained.record(report_path),
                    "kv_state": retained.save(root / f"kv{layer}.npz", cache)})
                actual.append(arrays)
                caches.append(cache)
                reports.append(report)
            result = {
                "status": "PASS", "candidate_admitted": True,
                "candidate_id": "ace3-q24-s16-toward-zero-native-v1",
                "policy_id": continuation.runner.gates.POLICY_ID,
                "normal_host_review": "REQUIRED", "rtl_invocations": 0,
                "l9_invocations": 0, "first_failure": None, "layers": layers}
            parent = {
                "candidate_id": result["candidate_id"], "policy_id": result["policy_id"],
                "state_id": "ace3-q24-software-paired-state-v1",
                "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "history": [9707],
                "position": 0, "next_layer": 3, "evidence_kind": "cpu_software_q24",
                "rtl_admissible": False, "normal_host_review": "REQUIRED",
                "arithmetic_lineage": freeze_record, "state": state_records[-1],
                "state_lineage_parent": state_records[-2],
                "numerical_report": layers[-1]["reports"], "kv": layers[-1]["kv_state"]}
            review = {"kind": "round_reviewed_handoff", "producer_role": "reviewer",
                      "mission_id": "572169d94b1d", "review": {"status": "done"}}
            (root / "command.log").write_text("synthetic fixture; no candidate execution\n")
            with patch.multiple(continuation, ACCEPTED=root, SELECTED_PARENT=root / "parent.json",
                                REVIEW=root / "review.json"):
                yield {
                    "root": root, "out": out, "parent": parent, "review": review,
                    "freeze": freeze, "freeze_record": freeze_record, "result": result,
                    "states": states, "state_records": state_records, "actual": actual,
                    "caches": caches, "reports": reports}


class ResumeAuthenticationTests(ResumeAuthenticationFixture, unittest.TestCase):
    def run_authentication(self, fixture, reason, *, accepted=False):
        retained = continuation.runner.retained
        self.rewrite_record(fixture["freeze_record"], fixture["freeze"])
        for name in ("parent", "review", "result"):
            retained.write(fixture["root"] / f"{name}.json", fixture[name])
        authenticated = []
        authenticate = continuation.authenticate_resume

        def stop_after_authentication(*args):
            authenticated.append(authenticate(*args))
            # Never proceed to historical evidence, model inputs or numerical execution.
            raise ValueError("synthetic authentication boundary reached")

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                continuation, "ACCEPTED_FREEZE_SHA", fixture["freeze_record"]["sha256"]))
            auth = stack.enter_context(patch.object(
                continuation, "authenticate_resume", side_effect=stop_after_authentication))
            prohibited = [
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError(
                    f"{name} must not run in resume-authentication regressions")))
                for owner, name in (
                    (candidate, "stages"), (candidate, "continuation_stages"),
                    (candidate, "lift"), (continuation.runner, "safe_open"),
                    (continuation.runner.local, "local_reference"),
                    (continuation.runner.gates, "evaluate_decoder_stage"))]
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = continuation.runner.run(fixture["out"], resume_parent=continuation.SELECTED_PARENT)
            auth.assert_called_once()
            for operation in prohibited:
                operation.assert_not_called()
        self.assertEqual(code, 1)
        result = json.loads((fixture["out"] / "result.json").read_text())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIs(result["candidate_admitted"], False)
        self.assertEqual(result["first_failure"]["failure_taxonomy"], "authentication")
        self.assertIsNone(result["first_failure"]["node"])
        self.assertIn(reason, result["first_failure"]["reason"])
        self.assertEqual(result["layers"], [])
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertEqual(result["l9_invocations"], 0)
        self.assertEqual(list(fixture["out"].iterdir()), [fixture["out"] / "result.json"])
        self.assertEqual(len(authenticated), int(accepted))
        return authenticated

    def test_valid_local_fixture_reaches_authentication_boundary_without_dispatch(self):
        with self.fixture() as fixture:
            authenticated, = self.run_authentication(
                fixture, "synthetic authentication boundary reached", accepted=True)
            self.assertEqual(authenticated["parent"], fixture["parent"])
            self.assertEqual(authenticated["native_arithmetic_compatibility"], "unchanged")
            self.assertIs(authenticated["prior_layer_kv_consumed"], False)

    def test_wrong_review_rejected_before_dispatch(self):
        for key, value in (("kind", "mission_context"), ("producer_role", "engineer"),
                           ("mission_id", "another-root"), ("review", {"status": "blocked"})):
            with self.subTest(key=key), self.fixture() as fixture:
                fixture["review"][key] = value
                self.run_authentication(fixture, "missing independent accepted-root review")

    def test_wrong_arithmetic_freeze_rejected_before_dispatch(self):
        for key, value in (("path", "/wrong/freeze.json"), ("sha256", "0" * 64)):
            with self.subTest(key=key), self.fixture() as fixture:
                fixture["parent"]["arithmetic_lineage"] = dict(
                    fixture["freeze_record"], **{key: value})
                self.run_authentication(fixture, "wrong accepted arithmetic freeze")
        with self.fixture() as fixture:
            fixture["freeze"]["contract"]["scope"]["layers"] = [0, 1]
            self.run_authentication(fixture, "wrong accepted root scope")

    def test_wrong_source_bindings_rejected_before_dispatch(self):
        for fault, reason in (
                ("snapshot", "accepted source snapshot mismatch"),
                ("missing_native", "missing accepted native arithmetic source"),
                ("duplicate_native", "missing accepted native arithmetic source"),
                ("missing_state", "missing accepted Q24 state implementation"),
                ("state_binding", "conflicting binding"),
                ("input_binding", "binding mismatch")):
            with self.subTest(fault=fault), self.fixture() as fixture:
                sources = fixture["freeze"]["sources"]
                if fault == "snapshot":
                    sources[0]["original"]["sha256"] = "0" * 64
                elif fault == "missing_native":
                    sources.pop(0)
                elif fault == "duplicate_native":
                    sources.append(copy.deepcopy(sources[0]))
                elif fault == "missing_state":
                    sources.pop(1)
                elif fault == "state_binding":
                    sources[1]["original"]["bytes"] += 1
                else:
                    fixture["freeze"]["input_bindings"][-1]["sha256"] = "0" * 64
                self.run_authentication(fixture, reason)

    def test_changed_native_arithmetic_rejected_before_dispatch(self):
        for old, new in (("/ 8)", "/ 4)"), ("torch.mv(", "torch.mm(")):
            with self.subTest(change=old), self.fixture() as fixture:
                source = fixture["freeze"]["sources"][0]
                historical = Path(source["snapshot"]["path"]).read_text()
                changed = historical.replace(old, new)
                self.assertNotEqual(changed, historical)
                self.rewrite_record(source["snapshot"], changed)
                source["original"]["sha256"] = source["snapshot"]["sha256"]
                self.run_authentication(fixture, "arithmetic changed")

    def test_incomplete_or_wrong_result_rejected_before_dispatch(self):
        for key, value in (
                ("status", "FAIL"), ("candidate_admitted", False),
                ("candidate_id", "another-candidate"), ("policy_id", "legacy"),
                ("normal_host_review", "PASS"), ("rtl_invocations", 1),
                ("l9_invocations", 1), ("first_failure", {"node": [2, 0, 18]})):
            with self.subTest(key=key), self.fixture() as fixture:
                fixture["result"][key] = value
                self.run_authentication(fixture, "not a complete CPU L0-L2 PASS")
        with self.fixture() as fixture:
            fixture["result"]["layers"].pop()
            self.run_authentication(fixture, "not a complete CPU L0-L2 PASS")

    def test_missing_or_failing_stage_rejected_before_dispatch(self):
        for fault in ("entry_status", "position", "missing", "order", "node", "status"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                entry = fixture["result"]["layers"][1]
                reports = fixture["reports"][1]
                if fault == "entry_status":
                    entry["status"] = "FAIL"
                elif fault == "position":
                    entry["position"] = 1
                elif fault == "missing":
                    reports.pop()
                elif fault == "order":
                    reports[0], reports[1] = reports[1], reports[0]
                elif fault == "node":
                    reports[0]["node"] = [0, 0, 0]
                else:
                    reports[0]["status"] = "FAIL"
                self.rewrite_record(entry["reports"], reports)
                self.run_authentication(fixture, "missing or failing mandatory stage")

    def test_spliced_residual_chain_rejected_before_dispatch(self):
        with self.fixture() as fixture:
            fixture["result"]["layers"][1]["input_parent"] = fixture["state_records"][0]
            self.run_authentication(fixture, "accepted residual chain is spliced")

    def test_actual_paired_state_mismatch_rejected_before_dispatch(self):
        for key in ("input_i", "input_z", "input_hidden", "output_i", "output_z", "stage18"):
            with self.subTest(key=key), self.fixture() as fixture:
                fixture["actual"][1][key][0] += 1
                self.rewrite_record(fixture["result"]["layers"][1]["actual_stages"],
                                    fixture["actual"][1])
                self.run_authentication(fixture, "invented/spliced Q24 parent")

    def test_invalid_paired_state_rejected_before_dispatch(self):
        for fault, reason in (("missing", "missing paired Q24 parent"),
                              ("dtype", "shape/dtype"), ("shape", "shape/dtype"),
                              ("view", "parent/view mismatch")):
            with self.subTest(fault=fault), self.fixture() as fixture:
                state = fixture["states"][0]
                if fault == "missing":
                    del state["z"]
                elif fault == "dtype":
                    state["i"] = state["i"].astype("<i4")
                elif fault == "shape":
                    state["i"] = state["i"][:-1]
                else:
                    state["i"][0] = 1 << 24
                fixture["actual"][0]["input_i"] = state["i"].copy()
                self.rewrite_record(fixture["state_records"][0], state)
                self.rewrite_record(fixture["result"]["layers"][0]["actual_stages"],
                                    fixture["actual"][0])
                self.run_authentication(fixture, reason)

    def test_receipt_lineage_mismatch_rejected_before_dispatch(self):
        for field, source in (("state", "output_parent"),
                              ("state_lineage_parent", "input_parent"),
                              ("numerical_report", "reports"), ("kv", "kv_state")):
            with self.subTest(field=field), self.fixture() as fixture:
                fixture["parent"][field] = fixture["result"]["layers"][0][source]
                self.run_authentication(fixture, "L2 receipt differs from reviewed output")

    def test_wrong_kv_contents_shape_dtype_or_keys_rejected_before_dispatch(self):
        for name in ("k", "v"):
            for fault in ("contents", "shape", "dtype", "keys"):
                with self.subTest(name=name, fault=fault), self.fixture() as fixture:
                    cache = fixture["caches"][2]
                    if fault == "contents":
                        cache[name][0, 0] += 1
                    elif fault == "shape":
                        cache[name] = cache[name].reshape(128)
                    elif fault == "dtype":
                        cache[name] = cache[name].astype("<u4")
                    else:
                        del cache[name]
                    self.rewrite_record(fixture["parent"]["kv"], cache)
                    self.run_authentication(
                        fixture, "cache keys changed" if fault == "keys"
                        else "cache differs from its own output")

    def test_changed_retained_archive_rejected_before_dispatch(self):
        with self.fixture() as fixture:
            record = fixture["result"]["layers"][0]["actual_stages"]
            with Path(record["path"]).open("ab") as stream:
                stream.write(b"changed")
            self.run_authentication(fixture, "binding mismatch")


class ContinuationDispatchTests(ResumeAuthenticationFixture, unittest.TestCase):
    @contextlib.contextmanager
    def dispatch_fixture(self):
        runner = continuation.runner
        retained = runner.retained
        with self.fixture() as fixture, contextlib.ExitStack() as stack:
            root = fixture["root"]

            def document(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                retained.write(path, value)
                return retained.record(path)

            def words(name, values, *, indexed=False):
                path = root / name
                path.write_text("".join(
                    (f"{index:06x}" if indexed else "") + f"{int(word):04x}\n"
                    for index, word in enumerate(values)))
                return dict(retained.record(path),
                            semantic_sha256=hashlib.sha256(values.tobytes()).hexdigest())

            self.rewrite_record(fixture["freeze_record"], fixture["freeze"])
            for name in ("parent", "review", "result"):
                document(f"{name}.json", fixture[name])
            document("reviews/e37713d52f39/round-0001.json", {
                "kind": "round_reviewed_handoff", "producer_role": "reviewer",
                "mission_id": "e37713d52f39", "review": {"status": "done"}})
            document("history/result.json", {"arms": [
                {"rounding": rounding, "status": "FAIL", "candidate_admitted": False,
                 "witness": {"excess_error": "1/2", "actual_fp16_bits": "616e"}}
                for rounding in ("rne", "toward_zero", "away_zero")]})
            document("history/freeze.json", {"synthetic": True})
            old_records = {
                "result.json": document("old/result.json", {"status": "FAIL"}),
                "freeze.json": document("old/freeze.json", {"synthetic": True}),
                "layer02/reports.json": document("old/reports.json", []),
                "layer02/actual_stages.npz": retained.save(root / "old/actual.npz", {})}
            old = runner.ROOT / "build/q24_software_root_995d0ba22311_attempt002"
            redirects = {old / name: record for name, record in old_records.items()}
            record = retained.record
            stack.enter_context(patch.object(
                retained, "record", side_effect=lambda path: (
                    redirects[Path(path)] if Path(path) in redirects else record(path))))

            embedding = np.zeros(896, dtype="<u2")
            embeddings = [{"token": 9707,
                           "input": words("embedding.txt", embedding, indexed=True),
                           "semantic_sha256": hashlib.sha256(embedding.tobytes()).hexdigest()}]
            checkpoint = document("mock_checkpoint.json", {"synthetic": True})
            canonical = [{"name": f"layer{layer}"} for layer in range(3, 9)]
            spec = document("spec.json", {
                "checkpoint": checkpoint, "checkpoint_tensors": canonical})
            trajectories = {
                stage: np.full(size, stage, dtype="<u2")
                for stage, size in runner.local.SIZES.items()}
            trajectory_records = {
                str(stage): words(f"trajectory{stage:02d}.txt", values)
                for stage, values in trajectories.items()}
            model_freeze = document("model_freeze.json", {
                "checkpoint": checkpoint, "checkpoint_tensors": canonical,
                "embeddings": embeddings, "reference_source": spec,
                "reference_transactions": [
                    {"layer": layer, "position": 0, "stages": trajectory_records}
                    for layer in range(3, 9)]})
            # Independent fixture data, deliberately distinct from actual residuals.
            global_reference = np.full(896, 3.0625, dtype="<f8")
            with (root / "global.npy").open("xb") as stream:
                np.save(stream, global_reference)
            with (root / "global.csv").open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("layer", "position", "index", "reference_binary64_hex"))
                writer.writerows((layer, 0, index, float(value).hex())
                                 for layer in range(3, 9)
                                 for index, value in enumerate(global_reference))
            control = document("control.json", {
                "checkpoint": checkpoint, "embeddings": embeddings,
                "fp16_reference_root": spec,
                "binary64_reference_freeze": document("binary64_freeze.json", {}),
                "binary64_array_binding_freeze": document("array_freeze.json", {}),
                "binary64_csv": retained.record(root / "global.csv"),
                "ordered_cases": [[layer, 0] for layer in range(3, 9)],
                "cases": [{"binary64_reference_array": retained.record(root / "global.npy")}
                          for _ in range(3, 9)]})
            policy = json.loads(Path(runner.gates.CONTRACT).read_text())
            policy["trusted_independent_freeze_sha256"] = model_freeze["sha256"]
            policy_record = document("policy.json", policy)
            stack.enter_context(patch.object(runner.gates, "CONTRACT", Path(policy_record["path"])))
            stack.enter_context(patch.object(runner, "HISTORY", root / "history"))
            stack.enter_context(patch.multiple(
                retained, HANDOFFS=root / "reviews", B_FREEZE=Path(model_freeze["path"]),
                CONTROL=Path(control["path"]), CONTROL_SHA=control["sha256"]))
            stack.enter_context(patch.object(
                continuation, "ACCEPTED_FREEZE_SHA", fixture["freeze_record"]["sha256"]))
            model = MagicMock()
            model.get_slice.return_value = np.zeros((9708, 896), dtype="<f2")
            model.get_tensor.side_effect = lambda name: np.asarray([int(name[5:])])
            opened = stack.enter_context(patch.object(runner, "safe_open"))
            opened.return_value.__enter__.return_value = model
            stack.enter_context(patch.object(
                runner.local, "tensor_shapes", side_effect=lambda layer: {f"layer{layer}": None}))
            tensors_authenticated = stack.enter_context(patch.object(runner.local, "authenticate_tensors"))
            fixture.update(trajectories=trajectories, global_reference=global_reference,
                           opened=opened, checkpoint=checkpoint,
                           tensors_authenticated=tensors_authenticated, canonical=canonical)
            yield fixture

    @staticmethod
    def local_words(layer, stage):
        word = layer if stage in (3, 5, 6, 7) else 1 if stage in (11, 17) else 0
        return np.full(continuation.runner.local.SIZES[stage], word, dtype="<u2")

    def run_dispatch(self, fixture, failing_node=None):
        runner = continuation.runner
        dispatched, evaluated, closed = [], [], []
        active = {}

        def stages(tensors, layer, parent, arrays):
            self.assertIn(layer, range(3, 9))
            np.testing.assert_array_equal(tensors[f"layer{layer}"], [layer])
            active.update(layer=layer, arrays=arrays)
            arrays.update(input_hidden=parent["h"].copy(), input_i=parent["i"].copy(),
                          input_z=parent["z"].copy(),
                          input_cache_k=np.empty((0, 128), dtype="<u2"),
                          input_cache_v=np.empty((0, 128), dtype="<u2"))
            try:
                for stage in range(19):
                    self.assertEqual(dispatched, evaluated, "consumer ran before mandatory gate")
                    if stage == 12:
                        scratch = candidate.state.add(parent, arrays["stage11"])
                        arrays.update(scratch_i=scratch["i"], scratch_z=scratch["z"])
                        value = scratch["h"]
                    elif stage == 18:
                        output = candidate.state.add(scratch, arrays["stage17"])
                        arrays.update(output_i=output["i"], output_z=output["z"])
                        value = output["h"]
                    else:
                        value = self.local_words(layer, stage)
                    arrays[f"stage{stage:02d}"] = value
                    if stage in (6, 7):
                        arrays["output_cache_" + ("k" if stage == 6 else "v")] = value.reshape(1, 128).copy()
                    dispatched.append((layer, stage))
                    yield stage
            finally:
                closed.append(layer)

        def local_reference(stage, operands, tensors, layer):
            self.assertEqual(set(operands), set(runner.local.OPERANDS[stage]))
            for name, value in operands.items():
                np.testing.assert_array_equal(value, active["arrays"][name])
                self.assertFalse(np.shares_memory(value, active["arrays"][name]))
            np.testing.assert_array_equal(tensors[f"layer{layer}"], [layer])
            return self.local_words(layer, stage)

        def evaluate(*, stage, actual, reference, policy, local_reference, reference_binary64):
            layer = active["layer"]
            self.assertEqual(policy, runner.gates.POLICY_ID)
            self.assertIs(actual, active["arrays"][f"stage{stage:02d}"])
            np.testing.assert_array_equal(reference, fixture["trajectories"][stage])
            if stage == 18:
                self.assertIsNone(local_reference)
                np.testing.assert_array_equal(reference_binary64, fixture["global_reference"])
                self.assertFalse(np.shares_memory(reference_binary64, actual))
            else:
                self.assertIsNone(reference_binary64)
                np.testing.assert_array_equal(local_reference, actual)
            evaluated.append((layer, stage))
            report = {"stage": stage, "status": "FAIL" if (layer, stage) == failing_node else "PASS"}
            if stage == 18:
                report["binary64_v1"] = {"rows": [
                    {"index": index, "status": report["status"]} for index in range(896)]}
            return report

        with contextlib.ExitStack() as stack:
            auth = stack.enter_context(patch.object(
                continuation, "authenticate_resume", wraps=continuation.authenticate_resume))
            stack.enter_context(patch.object(candidate, "continuation_stages", side_effect=stages))
            root_operations = [
                stack.enter_context(patch.object(candidate, name, side_effect=AssertionError(
                    "accepted prefix must not execute"))) for name in ("stages", "lift")]
            check_state = stack.enter_context(patch.object(
                runner.retained, "check_stage_state", wraps=runner.retained.check_stage_state))
            oracle = stack.enter_context(patch.object(
                runner.local, "local_reference", side_effect=local_reference))
            gate = stack.enter_context(patch.object(runner.gates, "evaluate_decoder_stage", side_effect=evaluate))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = runner.run(fixture["out"], resume_parent=continuation.SELECTED_PARENT)
            auth.assert_called_once()
            for operation in root_operations:
                operation.assert_not_called()
            self.assertEqual(check_state.call_count, len(dispatched))
            self.assertEqual(gate.call_count, len(dispatched))
            self.assertEqual(oracle.call_count, sum(stage not in (12, 18) for _, stage in dispatched))
        fixture["opened"].assert_called_once_with(fixture["checkpoint"]["path"], framework="numpy")
        self.assertEqual(fixture["tensors_authenticated"].call_count, 6)
        for call, layer in zip(fixture["tensors_authenticated"].call_args_list, range(3, 9), strict=True):
            tensors, canonical, actual_layer = call.args
            self.assertEqual(actual_layer, layer)
            self.assertEqual(canonical, {record["name"]: record for record in fixture["canonical"]})
            np.testing.assert_array_equal(tensors[f"layer{layer}"], [layer])
        result = json.loads((fixture["out"] / "result.json").read_text())
        self.assertEqual(code, 0 if failing_node is None else 1)
        self.assertEqual(result["status"], "PASS" if failing_node is None else "FAIL")
        self.assertIs(result["candidate_admitted"], failing_node is None)
        self.assertEqual(result["scope"], {"layers": list(range(3, 9)), "position": 0, "history": [9707]})
        self.assertEqual(result["candidate_id"], "ace3-q24-s16-toward-zero-native-v1")
        self.assertEqual(result["policy_id"], runner.gates.POLICY_ID)
        self.assertEqual(result["evidence_kind"], "cpu_software_q24")
        self.assertEqual(result["normal_host_review"], "REQUIRED")
        self.assertEqual(result["candidate_admission_scope"],
                         "bounded CPU numerical/state gates only; Host review still required")
        self.assertEqual(result["claim_boundary"],
                         "L3-L8/P0 from reviewed L2 software parent only; "
                         "no production, strict-FP16-state, RTL, token or full-model admission")
        self.assertEqual(result["rtl_invocations"], 0)
        self.assertEqual(result["l9_invocations"], 0)
        self.assertIs(result["policy_adopted"], False)
        self.assertIs(result["historical_fail_preserved"], True)
        expected = [(layer, stage) for layer in range(3, 9) for stage in range(19)]
        if failing_node is not None:
            expected = expected[:expected.index(failing_node) + 1]
        self.assertEqual(dispatched, expected)
        self.assertEqual(evaluated, expected)
        self.assertEqual(closed, sorted({layer for layer, _ in expected}))
        return result

    def test_mandatory_failure_stops_dispatch_before_successor_publication(self):
        for layer, stage in ((3, 0), (3, 12), (3, 16), (3, 18), (4, 18)):
            with self.subTest(layer=layer, stage=stage), self.dispatch_fixture() as fixture:
                result = self.run_dispatch(fixture, (layer, stage))
                self.assertEqual(result["first_failure"]["node"], [layer, 0, stage])
                self.assertEqual(result["first_failure"]["failure_taxonomy"],
                                 "global_numerical" if stage == 18 else "local_operator_numerical")
                self.assertEqual([entry["layer"] for entry in result["layers"]], list(range(3, layer + 1)))
                failed = result["layers"][-1]
                self.assertEqual(failed["status"], "FAIL")
                self.assertNotIn("output_parent", failed)
                self.assertNotIn("kv_state", failed)
                reports = json.loads(Path(failed["reports"]["path"]).read_text())
                self.assertEqual([report["stage"] for report in reports], list(range(stage + 1)))
                self.assertEqual([report["status"] for report in reports], ["PASS"] * stage + ["FAIL"])
                directory = fixture["out"] / f"layer{layer:02d}"
                self.assertEqual({path.name for path in directory.iterdir()},
                                 {"actual_stages.npz", "local_references.npz", "reports.json"})
                for later in range(layer + 1, 10):
                    self.assertFalse((fixture["out"] / f"layer{later:02d}").exists())
                self.assertEqual(
                    sorted(path.parent.name for path in fixture["out"].glob("layer*/software_parent.json")),
                    [f"layer{prior:02d}" for prior in range(3, layer)])

    def test_passing_mocked_layers_publish_only_own_software_parents(self):
        with self.dispatch_fixture() as fixture:
            result = self.run_dispatch(fixture)
            self.assertIsNone(result["first_failure"])
            self.assertEqual([entry["layer"] for entry in result["layers"]], list(range(3, 9)))
            previous = fixture["parent"]["state"]
            expected_i = fixture["states"][-1]["i"].copy()
            for entry in result["layers"]:
                layer = entry["layer"]
                directory = fixture["out"] / f"layer{layer:02d}"
                parent = json.loads((directory / "software_parent.json").read_text())
                self.assertEqual(entry["status"], "PASS")
                self.assertEqual(entry["input_parent"], previous)
                self.assertEqual(parent, {
                    "candidate_id": "ace3-q24-s16-toward-zero-native-v1",
                    "state_id": "ace3-q24-software-paired-state-v1",
                    "policy_id": continuation.runner.gates.POLICY_ID,
                    "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "history": [9707],
                    "position": 0, "next_layer": layer + 1,
                    "state": entry["output_parent"], "kv": entry["kv_state"],
                    "state_lineage_parent": previous,
                    "arithmetic_lineage": continuation.runner.retained.record(fixture["out"] / "freeze.json"),
                    "numerical_report": entry["reports"], "evidence_kind": "cpu_software_q24",
                    "rtl_admissible": False, "normal_host_review": "REQUIRED"})
                with np.load(entry["actual_stages"]["path"], allow_pickle=False) as arrays, \
                        np.load(previous["path"], allow_pickle=False) as incoming, \
                        np.load(parent["state"]["path"], allow_pickle=False) as outgoing, \
                        np.load(parent["kv"]["path"], allow_pickle=False) as cache:
                    for key, actual_key in (("i", "input_i"), ("z", "input_z"), ("h", "input_hidden")):
                        np.testing.assert_array_equal(arrays[actual_key], incoming[key])
                    expected_i += 2
                    np.testing.assert_array_equal(outgoing["i"], expected_i)
                    for key, actual_key in (("i", "output_i"), ("z", "output_z"), ("h", "stage18")):
                        np.testing.assert_array_equal(outgoing[key], arrays[actual_key])
                    for kind, stage in (("k", 6), ("v", 7)):
                        self.assertEqual(arrays["input_cache_" + kind].shape, (0, 128))
                        self.assertEqual(arrays["input_cache_" + kind].dtype, np.dtype("<u2"))
                        self.assertEqual(cache[kind].shape, (1, 128))
                        self.assertEqual(cache[kind].dtype, np.dtype("<u2"))
                        np.testing.assert_array_equal(cache[kind], arrays["output_cache_" + kind])
                        np.testing.assert_array_equal(cache[kind][0], self.local_words(layer, stage))
                previous = parent["state"]
            self.assertFalse((fixture["out"] / "layer09").exists())
            self.assertFalse((fixture["out"] / "root_state.npz").exists())
            freeze = json.loads((fixture["out"] / "freeze.json").read_text())
            self.assertIs(freeze["global_reference_lineage"]["candidate_data_used"], False)
            self.assertEqual(freeze["state_lineage"]["prior_residual"]["parent"], fixture["parent"])
            self.assertEqual(freeze["state_lineage"]["prior_kv"], "own empty P0")


if __name__ == "__main__":
    unittest.main()
