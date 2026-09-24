"""Synthetic receipt regressions, not simulator or RTL correctness evidence."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import decoder_gate_policy_v3 as policy
from ace3.model.candidates import local_operator_reference_v3 as local
from ace3.model.candidates import runtime_admission_v3 as runtime


class RuntimeAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="v3-runtime-unit-", dir=runtime.ROOT / "build")
        cls.root = Path(cls.temporary.name)
        cls.sequence = 0
        cls.arrays = {f"stage{s:02d}": np.zeros(n, dtype="<u2") for s, n in local.SIZES.items()}
        cls.arrays["stage09"][:] = np.float16(1).view("<u2")
        cls.arrays["input_hidden"] = np.zeros(896, dtype="<u2")
        for kind in ("k", "v"):
            cls.arrays["input_cache_" + kind] = np.empty((0, 128), dtype="<u2")
            cls.arrays["output_cache_" + kind] = np.zeros((1, 128), dtype="<u2")
        cls.tensors, canonical, vectors = {}, {}, []
        for name, (shape, dtype) in local.tensor_shapes(5).items():
            tensor = np.zeros(shape, dtype=dtype)
            cls.tensors[name] = tensor
            canonical[name] = {"name": name, "shape": list(shape), "dtype": dtype,
                               "sha256": hashlib.sha256(tensor.tobytes()).hexdigest()}
            width = tensor.dtype.itemsize * 2
            data = ((("0" * width) + "\n") * tensor.size).encode()
            vectors.append({"checkpoint_tensor": dict(canonical[name], dtype=
                            {"float16": "F16", "int32": "I32"}[dtype]),
                            "serialized": cls.store("tensor.hex", data)})
        cls.canonical = canonical
        final_bytes = "".join(f"{i:06x}0000\n" for i in range(896)).encode()
        hidden = cls.store("parent.hex", final_bytes)
        coordinates = [r[:3] for r in runtime.trace_rows(runtime.artifact(runtime.TRACE_LAYOUT))]
        trace = []
        heads = {8: 0, 9: 0}
        for s, position, index in coordinates:
            offset = heads[s] if s in heads else index
            trace.append(f"00{position:04x}{s:02x}{index:04x}{int(cls.arrays[f'stage{s:02d}'][offset]):04x}\n")
            if s in heads:
                heads[s] += 1
        raw_trace = cls.store("trace.hex", "".join(trace).encode())
        public = "module " + runtime.TOP + " #(parameter integer LAYER_INDEX=0) ();"
        top = cls.store(runtime.TOP + ".sv", (public + "\nendmodule\n").encode(), exact=True)
        tool = cls.store("verilator", b"synthetic tool; never executed")
        binary = cls.store("obj/V" + runtime.TOP, b"synthetic binary; never executed", exact=True)
        log = cls.store("log.txt", b"synthetic receipt fixture; no RTL execution\n")
        command = [tool["path"], "--cc", "--savable", "-GLAYER_INDEX=4",
                   "--Mdir", str(cls.root / "old-obj"), top["path"], "--build"]
        source = {"source_closure": [top], "public_contract": public, "command": command,
                  "semantics": "RNE16(H + O + D), exact three-term sum, only one rounding"}
        header, slow, symbols, state = cls.state_fixture()
        generated = {}
        for name, suffix, data in (("header", ".h", header), ("slow", "__Slow.cpp", slow),
                                   ("symbols", "__Syms.cpp", symbols)):
            generated[name] = cls.store("obj/V" + runtime.TOP + suffix, data.encode(), exact=True)
        state_record = cls.store("candidate.state", state)
        cls.state_bytes = state
        compile_command = command.copy()
        compile_command[3] = "-GLAYER_INDEX=5"
        compile_command[5] = str(cls.root / "obj")
        compile_record = cls.store_json("compile.json", {
            "returncode": 0, "natural_exit": True, "tool": tool, "log": log, "binary": binary,
            "argv": compile_command, "object_dir": str(cls.root / "obj"), "generated": generated})
        rope = cls.store("rope.hex", b"".join(
            f"{0:04x}{i:02x}{15360:04x}{0:04x}\n".encode() for i in range(32)))
        vector_record = {"input": hidden, "tensors": vectors, "rope_coefficients": rope}
        prepared = cls.store_json("prepared.json", {"binary": binary, "input_hidden": hidden,
                    "input_state": None, "vectors": vector_record})
        parameters = {"LAYER_INDEX": 5, "ACCURATE_SILU": 1}
        launch_command = [binary["path"], "+fixture-only"]
        launch = cls.store_json("launch.json", {"binary": binary, "prepared": prepared,
                "node": [5, 0], "history": [9707], "parameters": parameters,
                "argv": launch_command, "input_hidden": hidden, "input_state": None,
                "cache_slot": 0, "independent_p0_rope_sha256": rope["sha256"]})
        transaction = cls.store_json("transaction.json", {
            "layer_index": 5, "position": 0, "output": cls.store("final.hex", final_bytes),
            "output_state": state_record, "input": {"sha256": hashlib.sha256(bytes(1792)).hexdigest()},
            "raw": {"trace": raw_trace, "trace_count": sum(local.SIZES.values()),
                    "done_count": 1, "final_count": 896, "terminal": log},
            "simulation_log": log, "vectors": vector_record})
        execution = cls.store_json("execution.json", {"binary": binary, "launch": launch,
                    "transaction": transaction, "argv": launch_command, "returncode": 0,
                    "natural_exit": True, "executed": True, "output_state": state_record, "log": log})
        archive_path = cls.root / "actual.npz"
        with archive_path.open("xb") as stream:
            np.savez(stream, **cls.arrays)
        archive = cls.record(archive_path)
        cls.manifest = {"schema": runtime.SCHEMA, "policy_id": policy.POLICY_ID,
            "evidence_kind": "actual_rtl", "node": [5, 0], "history": [9707],
            "public_contract": public, "parameters": parameters, "semantics": source["semantics"],
            "source_closure": [top], "compile": compile_record, "launch": launch,
            "prepared": prepared, "execution": execution, "transaction": transaction,
            "actual_operands": archive, "parent": hidden, "independent_p0_rope": rope}
        cls.context = {"source": source, "root_hidden": hidden, "canonical": canonical,
            "trajectory": {s: cls.arrays[f"stage{s:02d}"].copy() for s in range(19)},
            "binary64": np.zeros(896, dtype="<f8"), "trace_coordinates": coordinates,
            "reference_bindings": {"fixture": "synthetic, not an admission authority"}}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def record(cls, path):
        data = path.read_bytes()
        return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    @classmethod
    def store(cls, name, data, exact=False):
        cls.sequence += 1
        path = cls.root / (name if exact else f"{cls.sequence}-{name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return cls.record(path)

    @classmethod
    def store_json(cls, name, value):
        return cls.store(name, json.dumps(value).encode())

    @staticmethod
    def state_fixture():
        prefix = runtime.TOP + "__DOT__cache__DOT__"
        header = "\n".join(f"CData/*7:0*/ {n};" for n in
                           ("busy_o", "done_valid_o", "phase_o", "layer_index_o"))
        body = ["vluint64_t __Vcheckval = 0x123456789abcdef0ULL;", "os<<__Vcheckval;"]
        body.extend(f"os<<{n};" for n in ("busy_o", "done_valid_o", "phase_o", "layer_index_o"))
        data = b"verilatorsave01\n" + struct.pack("<Q", 0x123456789abcdef0) + bytes([0, 0, 0, 5])
        for kind in ("k", "v", "valid"):
            name = prefix + kind + "_mem"
            header += f"\n{'CData/*7:0*/' if kind == 'valid' else 'SData/*15:0*/'} {name}[32768];"
            body.extend(("{ int __Vi0=0; for (; __Vi0<32768; ++__Vi0) {",
                         f"os<<{name}[__Vi0];", "}}"))
            data += bytes([1] * 128 + [0] * 32640) if kind == "valid" else bytes(65536)
        body.append("__VlSymsp->__Vserialize(os);")
        marker = "::__Vserialize(VerilatedSerialize& os) {\n"
        return (header, marker + "\n".join(body) + "\n}\n",
                marker + "os<<__Vm_didInit;\n}\n", data + b"\x01vltsaved")

    def evaluate(self, manifest=None, *, digest=None, **changes):
        receipt = self.store_json("manifest.json", self.manifest if manifest is None else manifest)
        arguments = {"arrays": self.arrays, "tensors": self.tensors,
                     "canonical_records": self.canonical, "expected_hidden": self.arrays["input_hidden"],
                     "trajectory": self.context["trajectory"],
                     "reference_binary64": self.context["binary64"], "layer": 5,
                     "position": 0, "history": [9707], "policy": policy.POLICY_ID, "emit": lambda r: None}
        arguments.update(changes)
        with patch.object(runtime, "_trusted_context", return_value=self.context):
            return policy.evaluate_actual_rtl_result(
                runtime_admission=receipt, trusted_runtime_manifest_sha256=digest or receipt["sha256"],
                **arguments)

    def assert_blocked(self, result):
        self.assertEqual(result["status"], "BLOCKED", result)
        self.assertEqual(result["numerical_status"], "NOT_EVALUATED", result)
        self.assertIn("reason", result)

    def test_complete_synthetic_receipt_reaches_real_numerical_evaluator(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "PASS", result.get("reason"))
        self.assertEqual(len(result["reports"]), 19)
        self.assertEqual(result["runtime_admission"]["saved_state"]["valid_entries"], 128)
        self.assertFalse(result["runtime_admission"]["saved_state"]["state_restore_or_eval_performed"])

    def test_real_retained_context_and_p0_control_binding(self):
        contract = json.loads(policy.CONTRACT.read_text())
        context = runtime._trusted_context(contract, 5)
        self.assertEqual(set(context["trajectory"]), set(range(19)))
        self.assertEqual(context["binary64"].shape, (896,))
        self.assertEqual(len(context["trace_coordinates"]), sum(local.SIZES.values()))
        self.assertEqual({row[0] for row in context["trace_coordinates"]}, set(range(19)))
        self.assertEqual(hashlib.sha256(runtime.p0_rope_bytes()).hexdigest(),
                         "703b8c6715e1b03341af300ed3bbd4a2d571b6bd4d78dd004af9a634001bf284")

    def derived_manifest(self):
        manifest = copy.deepcopy(self.manifest)
        sources = [self.record(Path(runtime.__file__).resolve()),
                   self.record(Path(runtime.__file__).resolve().with_name("host_capture_v3.py"))]
        original = self.store_json("original-manifest.json", manifest)
        freeze = self.store_json("recovery-freeze.json", {
            "schema": "ace3-v3-retained-recovery-plan-v1",
            "scope": {"layers": [5], "position": 0, "history": [9707]},
            "original_manifest": original,
            "original_plan": self.store_json("original-plan.json", {"fixture": "synthetic"}),
            "decoder_sources": sources, "bindings": sources})
        transaction = runtime.document(manifest["transaction"])
        boundary = {
            "schema": "ace3-v3-retained-coordinate-decoding-v1", "freeze": freeze,
            "original_manifest": original, "decoder_sources": sources,
            "raw_trace": transaction["raw"]["trace"], "final": transaction["output"],
            "state": transaction["output_state"], "actual_operands": manifest["actual_operands"],
            "decoder_rtl_invocations": 0}
        manifest["derivation"] = self.store_json("boundary.json", boundary)
        return manifest

    def test_synthetic_retained_derivation_reaches_unchanged_gates(self):
        result = self.evaluate(self.derived_manifest())
        self.assertEqual(result["status"], "PASS", result.get("reason"))
        self.assertEqual(len(result["reports"]), 19)

    def test_derived_manifest_cannot_change_producer_or_execution(self):
        for field in ("actual_operands", "decoder_sources", "state", "freeze", "original_manifest"):
            manifest = self.derived_manifest()
            boundary = runtime.document(manifest["derivation"])
            if field == "decoder_sources":
                boundary[field][0]["sha256"] = "0" * 64
            else:
                boundary[field] = manifest["parent"]
            manifest["derivation"] = self.store_json("bad-boundary.json", boundary)
            with self.subTest(field=field):
                self.assert_blocked(self.evaluate(manifest))
        manifest = self.derived_manifest()
        manifest["execution"] = manifest["launch"]
        self.assert_blocked(self.evaluate(manifest))
        self.assert_blocked(self.evaluate(self.derived_manifest(), digest="0" * 64))

    def test_nonuniform_rope_is_admitted_by_coordinate_not_emission_order(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays["stage04"] = np.arange(896, dtype="<u2")
        manifest = copy.deepcopy(self.manifest)
        archive = self.root / f"nonuniform-{self.sequence}.npz"
        with archive.open("xb") as stream:
            np.savez(stream, **arrays)
        manifest["actual_operands"] = self.record(archive)
        transaction = runtime.document(manifest["transaction"])
        rows = runtime.trace_rows(runtime.artifact(transaction["raw"]["trace"]))
        trace = "".join(f"00{p:04x}{s:02x}{i:04x}"
                        f"{int(arrays['stage04'][i]) if s == 4 else word:04x}\n"
                        for s, p, i, word in rows)
        transaction["raw"]["trace"] = self.store("nonuniform-trace.hex", trace.encode())
        manifest["transaction"] = self.store_json("nonuniform-transaction.json", transaction)
        execution = runtime.document(manifest["execution"])
        execution["transaction"] = manifest["transaction"]
        manifest["execution"] = self.store_json("nonuniform-execution.json", execution)
        result = self.evaluate(manifest, arrays=arrays)
        self.assertEqual(result["status"], "PASS", result.get("reason"))

    def test_missing_or_self_asserted_receipts_never_run_numerics(self):
        for receipt in (None, {}, {"status": "PASS"}, {"sha256": "0" * 64}):
            with patch.object(policy, "evaluate_p0_transaction") as numerical:
                self.assert_blocked(policy.evaluate_actual_rtl_result(runtime_admission=receipt))
                numerical.assert_not_called()
        self.assert_blocked(self.evaluate(digest="0" * 64))
        for field in ("compile", "launch", "execution", "prepared", "transaction",
                      "actual_operands", "parent", "source_closure", "independent_p0_rope"):
            manifest = copy.deepcopy(self.manifest)
            del manifest[field]
            with self.subTest(field=field):
                self.assert_blocked(self.evaluate(manifest))

    def test_source_interface_parameters_and_receipt_rejection(self):
        for field, replacement in (("evidence_kind", "software"), ("node", [6, 0]),
                                   ("history", [358]), ("public_contract", "alias module"),
                                   ("parameters", {"LAYER_INDEX": 5, "ACCURATE_SILU": 0}),
                                   ("semantics", "two-round")):
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = replacement
            with self.subTest(field=field):
                self.assert_blocked(self.evaluate(manifest))
        for key in ("compile", "actual_operands", "parent", "transaction"):
            manifest = copy.deepcopy(self.manifest)
            manifest[key]["sha256"] = "0" * 64
            self.assert_blocked(self.evaluate(manifest))
        manifest = copy.deepcopy(self.manifest)
        manifest["source_closure"][0]["sha256"] = "0" * 64
        self.assert_blocked(self.evaluate(manifest))

    def test_operand_and_reference_identity_rejection(self):
        for name in ("input_hidden", "stage01", "stage06", "stage18", "output_cache_v"):
            arrays = {key: value.copy() for key, value in self.arrays.items()}
            arrays[name].flat[0] = 1
            with self.subTest(name=name):
                self.assert_blocked(self.evaluate(arrays=arrays))
        arrays = dict(self.arrays, input_cache_k=np.zeros((1, 128), dtype="<u2"))
        self.assert_blocked(self.evaluate(arrays=arrays))
        self.assert_blocked(self.evaluate(expected_hidden=np.ones(896, dtype="<u2")))
        canonical = copy.deepcopy(self.canonical)
        canonical[next(iter(canonical))]["sha256"] = "0" * 64
        self.assert_blocked(self.evaluate(canonical_records=canonical))
        self.assert_blocked(self.evaluate(reference_binary64=np.ones(896, dtype="<f8")))
        trajectory = dict(self.context["trajectory"], **{})
        trajectory[0] = np.ones(896, dtype="<u2")
        self.assert_blocked(self.evaluate(trajectory=trajectory))

    def test_no_execution_binary_and_state_splice_rejection(self):
        for field, replacement in (("executed", False), ("natural_exit", False),
                                   ("returncode", 1), ("binary", self.manifest["parent"]),
                                   ("output_state", self.manifest["parent"])):
            manifest = copy.deepcopy(self.manifest)
            execution = runtime.document(manifest["execution"])
            execution[field] = replacement
            manifest["execution"] = self.store_json("changed-execution.json", execution)
            with self.subTest(field=field):
                self.assert_blocked(self.evaluate(manifest))
        for suffix in (b"", b"corrupt"):
            manifest = copy.deepcopy(self.manifest)
            transaction = runtime.document(manifest["transaction"])
            state = self.store("bad.state", self.state_bytes[:-9] + suffix)
            transaction["output_state"] = state
            manifest["transaction"] = self.store_json("changed-transaction.json", transaction)
            execution = runtime.document(manifest["execution"])
            execution.update(transaction=manifest["transaction"], output_state=state)
            manifest["execution"] = self.store_json("changed-execution.json", execution)
            self.assert_blocked(self.evaluate(manifest))

    def test_matching_but_wrong_canonical_control_receipts_fail(self):
        manifest = copy.deepcopy(self.manifest)
        rope = self.store("wrong-rope.hex", runtime.p0_rope_bytes().replace(b"3c00", b"4000"))
        manifest["independent_p0_rope"] = rope
        prepared = runtime.document(manifest["prepared"])
        prepared["vectors"]["rope_coefficients"] = rope
        manifest["prepared"] = self.store_json("changed-prepared.json", prepared)
        transaction = runtime.document(manifest["transaction"])
        transaction["vectors"] = prepared["vectors"]
        manifest["transaction"] = self.store_json("changed-transaction.json", transaction)
        launch = runtime.document(manifest["launch"])
        launch.update(prepared=manifest["prepared"], independent_p0_rope_sha256=rope["sha256"])
        manifest["launch"] = self.store_json("changed-launch.json", launch)
        execution = runtime.document(manifest["execution"])
        execution.update(launch=manifest["launch"], transaction=manifest["transaction"])
        manifest["execution"] = self.store_json("changed-execution.json", execution)
        result = self.evaluate(manifest)
        self.assert_blocked(result)
        self.assertIn("RoPE control mismatch", result["reason"])

    def test_saved_state_semantics_not_just_receipt_hash(self):
        header, slow, symbols, data = self.state_fixture()
        for offset, replacement in ((len(b"verilatorsave01\n") + 8, 1),
                                     (len(b"verilatorsave01\n") + 11, 6),
                                     (len(b"verilatorsave01\n") + 12, 1),
                                     (len(b"verilatorsave01\n") + 12 + 131072 + 128, 1)):
            changed = bytearray(data)
            changed[offset] = replacement
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                runtime.saved_state(bytes(changed), header, slow, symbols, 5, self.arrays)


if __name__ == "__main__":
    unittest.main()
