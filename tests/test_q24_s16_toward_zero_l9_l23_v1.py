"""Synthetic L9-L23 orchestration regressions, never numerical admission."""

import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ace3.model.candidates import run_q24_s16_toward_zero_l9_l23_v1 as continuation


EXPECTED_TESTS = 8


class ContinuationTests(unittest.TestCase):
    def test_dispatch_bounds_and_unchanged_prefix_guards(self):
        for layer in (*range(9), 24, True, 9.0):
            with self.assertRaises(ValueError):
                next(continuation.stages({}, layer, {}, {}))
        for layer in continuation.LAYERS:
            with patch.object(continuation.candidate, "_stages", return_value=iter(range(19))) as call:
                self.assertEqual(list(continuation.stages({}, layer, {}, {})), list(range(19)))
                call.assert_called_once_with({}, layer, {}, {})
        for dispatch in (continuation.candidate.stages, continuation.candidate.continuation_stages):
            with self.assertRaises(ValueError):
                next(dispatch({}, 9, {}, {}))

    def test_wrong_resume_path_blocks_before_read(self):
        with self.assertRaisesRegex(ValueError, "Planner-selected"):
            continuation.authenticate_resume(Path("/wrong"), None, None)

    def test_scope_rejects_runtime_and_stale_parents(self):
        parent = {
            "candidate_id": "ace3-q24-s16-toward-zero-native-v1",
            "state_id": "ace3-q24-software-paired-state-v1",
            "policy_id": continuation.gates.POLICY_ID,
            "model": "Qwen/Qwen2.5-0.5B-Instruct-AWQ", "history": [9707],
            "position": 0, "next_layer": 9, "evidence_kind": "cpu_software_q24",
            "rtl_admissible": False, "normal_host_review": "REQUIRED"}
        continuation.validate_parent_scope(parent)
        for key, value in (("next_layer", 8), ("next_layer", 9.0), ("position", False),
                           ("history", [9707, 0]), ("rtl_admissible", True),
                           ("state_id", "fp16"), ("normal_host_review", "PASS")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_parent_scope(dict(parent, **{key: value}))

    def test_binding_drift_blocks(self):
        with tempfile.TemporaryDirectory(prefix="q24_software_l9_l23_bind_",
                                         dir=continuation.ROOT / "build") as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            record = continuation.retained.record(path)
            path.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                continuation.bind_record(record, {})

    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory(prefix="q24_software_l9_l23_tests_",
                                         dir=continuation.ROOT / "build") as directory:
            out = Path(directory)
            retained = continuation.retained
            contract = json.loads(continuation.CONTRACT.read_text())
            retained.write(out / "freeze.json", {"synthetic": True})
            parent = continuation.candidate.lift(np.zeros(896, dtype="<u2"))
            parent_record = retained.save(out / "input.npz", parent)
            resume = {"parent": {"state": parent_record}}
            trajectory = {f"stage{stage:02d}": np.zeros(size, dtype="<u2")
                          for stage, size in continuation.local.SIZES.items()}
            fp16 = retained.save(out / "reference.npz", trajectory)
            with (out / "global.npy").open("xb") as stream:
                np.save(stream, np.full(896, 0.0625, dtype="<f8"))
            extension = {
                "checkpoint": retained.record(out / "freeze.json"),
                "layers": {str(layer): {"canonical": {}, "fp16": fp16,
                    "binary64": retained.record(out / "global.npy")} for layer in continuation.LAYERS}}
            yield out, contract, resume, extension

    def dispatch(self, fixture, failing_node=None, corrupt_state=False, binding_failure=False):
        out, contract, resume, extension = fixture
        calls, evaluated, closed = [], [], []
        active = {}

        def producer(tensors, layer, parent, arrays):
            active.update(layer=layer, arrays=arrays)
            arrays.update(input_i=parent["i"].copy(), input_z=parent["z"].copy(),
                          input_hidden=parent["h"].copy(),
                          input_cache_k=np.empty((0, 128), dtype="<u2"),
                          input_cache_v=np.empty((0, 128), dtype="<u2"))
            if corrupt_state:
                arrays["input_i"][0] += 1
            try:
                for stage in range(19):
                    self.assertEqual(calls, evaluated, "consumer ran before its gate")
                    if stage == 12:
                        scratch = continuation.candidate.state.add(parent, arrays["stage11"])
                        arrays.update(scratch_i=scratch["i"], scratch_z=scratch["z"])
                        words = scratch["h"]
                    elif stage == 18:
                        output = continuation.candidate.state.add(scratch, arrays["stage17"])
                        arrays.update(output_i=output["i"], output_z=output["z"])
                        words = output["h"]
                    else:
                        words = np.full(continuation.local.SIZES[stage],
                                        1 if stage in (11, 17) else 0, dtype="<u2")
                    arrays[f"stage{stage:02d}"] = words
                    if stage in (6, 7):
                        arrays["output_cache_" + ("k" if stage == 6 else "v")] = words.reshape(1, 128)
                    calls.append((layer, stage))
                    yield stage
            finally:
                closed.append(layer)

        def oracle(stage, operands, tensors, layer):
            self.assertEqual(set(operands), set(continuation.local.OPERANDS[stage]))
            for key, value in operands.items():
                np.testing.assert_array_equal(value, active["arrays"][key])
                self.assertFalse(np.shares_memory(value, active["arrays"][key]))
            return active["arrays"][f"stage{stage:02d}"].copy()

        def evaluate(**kwargs):
            layer, stage = active["layer"], kwargs["stage"]
            self.assertEqual(kwargs["policy"], continuation.gates.POLICY_ID)
            if stage == 18:
                np.testing.assert_array_equal(kwargs["reference_binary64"], np.full(896, 0.0625))
                self.assertIsNone(kwargs["local_reference"])
            else:
                np.testing.assert_array_equal(kwargs["actual"], kwargs["local_reference"])
                self.assertIsNone(kwargs["reference_binary64"])
            evaluated.append((layer, stage))
            failed = (layer, stage) == failing_node
            detail = {"passed": not failed, "failures": [{"index": 7}] if failed else [],
                      "rows": [{"index": 0, "accepted": True}, {"index": 7, "accepted": not failed}]}
            return {"stage": stage, "status": "FAIL" if failed else "PASS",
                    "binary64_v1" if stage == 18 else "local_operator_fp16": detail,
                    "fp16": {"passed": False, "failures": [{"index": 2}]}}

        def before_publish():
            if binding_failure:
                raise ValueError("synthetic prepublication source drift")

        with contextlib.ExitStack() as stack:
            model = MagicMock()
            opened = stack.enter_context(patch.object(continuation, "safe_open"))
            opened.return_value.__enter__.return_value = model
            stack.enter_context(patch.object(continuation.local, "tensor_shapes", return_value={}))
            stack.enter_context(patch.object(continuation.local, "authenticate_tensors"))
            stack.enter_context(patch.object(continuation, "stages", side_effect=producer))
            stack.enter_context(patch.object(continuation.local, "local_reference", side_effect=oracle))
            stack.enter_context(patch.object(continuation.gates, "evaluate_decoder_stage", side_effect=evaluate))
            root = stack.enter_context(patch.object(continuation.candidate, "lift",
                                                    side_effect=AssertionError("prefix replay")))
            result = continuation.execute_layers(
                out, contract, resume, extension,
                lambda record: continuation.bind_record(record, {}), before_publish)
            root.assert_not_called()
        self.assertEqual(closed, sorted({layer for layer, _ in calls}))
        if not corrupt_state:
            self.assertEqual(calls, evaluated)
        return result

    def test_first_failure_stops_without_successor(self):
        for node in ((9, 0), (9, 12), (9, 16), (9, 18), (10, 18)):
            with self.subTest(node=node), self.fixture() as fixture:
                status, entries, failure = self.dispatch(fixture, node)
                self.assertEqual(status, "FAIL")
                self.assertEqual(failure["node"], [node[0], 0, node[1]])
                self.assertEqual(failure["index"], 7)
                self.assertNotIn("output_parent", entries[-1])
                for layer in range(node[0], 24):
                    self.assertFalse((fixture[0] / f"layer{layer:02d}/software_parent.json").exists())
                for layer in range(node[0] + 1, 24):
                    self.assertFalse((fixture[0] / f"layer{layer:02d}").exists())

    def test_state_failure_stops_before_numerical_gate(self):
        with self.fixture() as fixture:
            status, entries, failure = self.dispatch(fixture, corrupt_state=True)
            self.assertEqual(status, "BLOCKED")
            self.assertEqual(failure["node"], [9, 0, 0])
            self.assertEqual(failure["failure_taxonomy"], "state_lineage")
            self.assertEqual(len(entries), 1)
            self.assertFalse(list(fixture[0].glob("layer*/software_parent.json")))

    def test_binding_failure_prevents_parent_publication(self):
        with self.fixture() as fixture:
            status, _, failure = self.dispatch(fixture, binding_failure=True)
            self.assertEqual(status, "BLOCKED")
            self.assertEqual(failure["failure_taxonomy"], "prepublication_binding_integrity")
            self.assertFalse(list(fixture[0].glob("layer*/software_parent.json")))

    def test_passing_lineage_empty_kv_and_metadata(self):
        with self.fixture() as fixture:
            status, entries, failure = self.dispatch(fixture)
            self.assertEqual(status, "PASS")
            self.assertIsNone(failure)
            self.assertEqual([entry["layer"] for entry in entries], list(range(9, 24)))
            previous = fixture[2]["parent"]["state"]
            for entry in entries:
                directory = fixture[0] / f"layer{entry['layer']:02d}"
                receipt = json.loads((directory / "software_parent.json").read_text())
                self.assertEqual(receipt["state_lineage_parent"], previous)
                self.assertEqual(entry["input_parent"], previous)
                self.assertEqual(receipt["state"], entry["output_parent"])
                self.assertEqual(receipt["next_layer"], entry["layer"] + 1)
                self.assertEqual(receipt["history"], [9707])
                self.assertEqual(receipt["position"], 0)
                self.assertEqual(receipt["evidence_kind"], "cpu_software_q24")
                self.assertIs(receipt["rtl_admissible"], False)
                self.assertEqual(receipt["normal_host_review"], "REQUIRED")
                with np.load(entry["actual_stages"]["path"]) as actual, \
                        np.load(receipt["kv"]["path"]) as cache, np.load(previous["path"]) as incoming:
                    for kind in ("k", "v"):
                        self.assertEqual(actual["input_cache_" + kind].shape, (0, 128))
                        self.assertEqual(actual["input_cache_" + kind].dtype, np.dtype("<u2"))
                        np.testing.assert_array_equal(cache[kind], actual["output_cache_" + kind])
                    for kind, name in (("i", "input_i"), ("z", "input_z"), ("h", "input_hidden")):
                        np.testing.assert_array_equal(actual[name], incoming[kind])
                previous = receipt["state"]
            self.assertFalse((fixture[0] / "layer24").exists())
            contract = fixture[1]
            self.assertEqual(contract["scope"], continuation.SCOPE)
            self.assertIs(contract["policy_adopted"], False)
            self.assertIs(contract["rtl_admissible"], False)
            self.assertEqual(contract["normal_host_review"], "REQUIRED")


if __name__ == "__main__":
    unittest.main()
