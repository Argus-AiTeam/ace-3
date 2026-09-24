"""Executable C++ observer/terminal controls with synthetic pins/state, NOT decoder RTL."""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from ace3.model.candidates import capture_harness_v3 as harness
from ace3.model.candidates import host_capture_v3 as capture
from ace3.model.candidates import runtime_admission_v3 as runtime
from ace3.model.tests import test_runtime_admission_v3 as fixtures


def observer_program(source, header):
    """Compile the production observer and terminal block, not a rewritten comparator."""
    preamble = source[source.index("#include <array>"):source.index("static std::vector")]
    observer = source[source.index("    void mismatch("):source.index("    bool tick()")]
    close = source[source.index("    void close_raw()"):source.index("\n};\n\nstatic void write_terminal")]
    terminal = source[source.index("static void write_terminal"):source.index("\nint main(")]
    finish = source[source.index("            if(trace_count!=h.expected_trace.size()"):
                    source.index("\n        }\n        h.idle();")]
    pins = sorted(set(re.findall(r"top\.(\w+)", observer)))
    return preamble + f'\n#include "{header}"\n' + r"""
#include <sstream>
struct Pins {
""" + "\n".join(f"    uint64_t {pin} = 0;" for pin in pins) + r"""
    void final() {}
};
static std::string fixture_state;
struct VerilatedSave {
    std::ofstream out;
    void open(const char* path) { out.open(path, std::ios::binary); }
    VerilatedSave& operator<<(const Pins&) {
        std::ifstream in(fixture_state, std::ios::binary);
        out << in.rdbuf();
        if (!in || !out) throw std::runtime_error("fixture state write failed");
        return *this;
    }
    void close() { out.close(); }
};
struct Harness {
    Pins top;
    std::vector<Trace> expected_trace;
    std::vector<Final> expected_final;
    std::ofstream raw_trace, raw_final;
    bool fail_after_raw = false;
    bool trace_hold = false, final_hold = false, done_hold = false;
    std::array<uint64_t, 4> trace_held{}, final_held{}, done_held{};
    unsigned expected_done_position = 0;
""" + observer + close + "\n};\n" + terminal + r"""
static std::vector<Trace> read_trace(const char* path) {
    std::ifstream in(path);
    std::vector<Trace> rows;
    std::string line;
    while (std::getline(in, line))
        rows.push_back({unsigned(std::stoul(line.substr(0,2),nullptr,16)),
                        unsigned(std::stoul(line.substr(2,4),nullptr,16)),
                        unsigned(std::stoul(line.substr(6,2),nullptr,16)),
                        unsigned(std::stoul(line.substr(8,4),nullptr,16)),
                        unsigned(std::stoul(line.substr(12,4),nullptr,16))});
    return rows;
}
int main(int argc, char** argv) {
    if (argc != 6) return 9;
    std::string raw_dir=argv[1], mode=argv[5], state_out=raw_dir+"/candidate.state";
    fixture_state=argv[4];
    capture_only=mode!="strict";
    transaction_mode=true; active_layer_index=5; checking=true;
    try {
        Harness h;
        h.expected_trace=read_trace(argv[2]);
        for (const auto& row : h.expected_trace)
            if (row.stage==18) h.expected_final.push_back({row.token,row.index,row.value});
        h.raw_trace.open(raw_dir+"/trace.hex"); h.raw_final.open(raw_dir+"/final.hex");
        h.fail_after_raw=mode=="partial";
        auto rows=read_trace(argv[3]);
        if (mode=="missing") rows.pop_back();
        for (const auto& row : rows) {
            h.top.trace_valid_o=h.top.trace_ready_i=h.top.final_ready_i=1;
            h.top.trace_stage_o=row.stage; h.top.trace_position_o=row.position;
            h.top.trace_index_o=row.index; h.top.trace_f16_o=row.value;
            h.top.final_valid_o=row.stage==18; h.top.final_index_o=row.index;
            h.top.final_f16_o=row.value; h.top.final_last_o=row.index==895;
            if (mode=="index" && trace_count==0) ++h.top.trace_index_o;
            if (mode=="last" && row.stage==18) h.top.final_last_o=!h.top.final_last_o;
            if (mode=="stall" && trace_count==0) {
                h.top.trace_ready_i=0; h.observe();
                h.top.trace_ready_i=1; ++h.top.trace_f16_o;
            }
            h.observe();
        }
        h.top.trace_valid_o=h.top.final_valid_o=0;
        h.top.done_valid_o=h.top.done_ready_i=1; h.top.done_cycles_o=2;
        h.top.done_cache_slot_o=mode=="done" ? 1 : 0;
        h.observe();
""" + finish + r"""
    } catch (const std::exception& error) {
        write_terminal(raw_dir,false,2);
        std::cerr << error.what() << "\n";
        return 2;
    }
}
"""


class CaptureHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RuntimeAdmissionTests.setUpClass()
        cls.fixture = fixtures.RuntimeAdmissionTests()
        cls.root = cls.fixture.root

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get("ACE3_CAPTURE_TEST_ARTIFACT_DIR")
        if destination:
            target = capture.fresh(Path(destination), capture.BUILD)
            target.mkdir()
            for path in cls.root.iterdir():
                if path.is_dir() and path.name in (
                        "capture_source", "capture", "strict", "index", "last",
                        "stall", "done", "missing", "partial"):
                    shutil.copytree(path, target / path.name)
                elif path.name.startswith("observer"):
                    shutil.copyfile(path, target / path.name)
        fixtures.RuntimeAdmissionTests.tearDownClass()

    def test_non_bit_exact_capture_reaches_v3_and_integrity_still_fails(self):
        f = self.fixture
        context = runtime._trusted_context(json.loads(capture.policy.CONTRACT.read_text()), 5)
        accepted = runtime.source_map(context["source"]["source_closure"])
        original = runtime.artifact(accepted[harness.HARNESS])
        derived = harness.capture_source(original)
        source_dir = self.root / "capture_source"
        source_dir.mkdir()
        cpp = source_dir / harness.HARNESS
        cpp.write_bytes(derived)
        derived_record = f.record(cpp)
        sources = dict(accepted, **{harness.HARNESS: derived_record})
        self.assertTrue(harness.compatible_sources(sources, accepted, runtime.artifact))
        self.assertFalse(harness.compatible_sources(accepted, accepted, runtime.artifact))
        altered = dict(sources)
        altered[harness.HARNESS] = f.store(harness.HARNESS, derived + b"\n// changed\n")
        with self.assertRaisesRegex(ValueError, "source closure"):
            harness.compatible_sources(altered, accepted, runtime.artifact)
        with self.assertRaisesRegex(ValueError, "unsupported capture harness"):
            harness.capture_source(derived)

        program = source_dir / "observer_fixture.cpp"
        program.write_text(observer_program(
            derived.decode("ascii"), accepted["ace3_layer0_trace_capture_policy.h"]["path"]))
        tool = shutil.which("g++")
        self.assertIsNotNone(tool, "C++ compiler required for capture regression")
        binary = self.root / "observer_fixture"
        capture.observe([tool, "-std=c++17", "-Wall", "-Wextra", str(program), "-o", str(binary)],
                        self.root, "observer_compile")
        arrays = {name: value.copy() for name, value in f.arrays.items()}
        for stage in (0, 17, 18):
            arrays[f"stage{stage:02d}"][0] = 1
        monitor = runtime.document(f.manifest["transaction"])["raw"]["trace"]
        actual = f.store("pins.hex", "".join(
            f"000000{s:02x}{i:04x}{int(arrays[f'stage{s:02d}'][i]):04x}\n"
            for s, _, i in f.context["trace_coordinates"]).encode())
        fixture_state = f.store("synthetic.state", f.state_bytes)
        outputs = {}
        for mode in ("capture", "strict", "index", "last", "stall", "done", "missing", "partial"):
            directory = self.root / mode
            directory.mkdir()
            argv = [str(binary), str(directory), monitor["path"], actual["path"],
                    fixture_state["path"], mode]
            if mode == "capture":
                capture.observe(argv, directory, "observer")
                self.assertIn("TRANSACTION_CAPTURED", (directory / "observer.log").read_text())
                self.assertNotIn("TRANSACTION_PASS", (directory / "observer.log").read_text())
                self.assertEqual((directory / "trace.hex").read_bytes(), runtime.artifact(actual))
                self.assertEqual((directory / "candidate.state").read_bytes(), f.state_bytes)
                outputs = {name: f.record(directory / name)
                           for name in ("trace.hex", "final.hex", "terminal.txt", "candidate.state")}
            else:
                with self.assertRaises(capture.subprocess.CalledProcessError, msg=mode):
                    capture.observe(argv, directory, "observer")
                self.assertFalse((directory / "candidate.state").exists(), mode)
                self.assertIn("natural_terminal=0", (directory / "terminal.txt").read_text())
        self.assertNotEqual(runtime.artifact(outputs["trace.hex"]), runtime.artifact(monitor))
        self.assertEqual(capture.adapter().parse_natural_terminal(
            Path(outputs["terminal.txt"]["path"]), 5, 0)["final_count"], 896)

        # Existing explicitly synthetic receipt context, now fed by the executed C++ observer.
        manifest, trusted = copy.deepcopy(f.manifest), copy.deepcopy(f.context)
        trusted["source"]["source_closure"].append(accepted[harness.HARNESS])
        trusted["source"]["command"].insert(-1, accepted[harness.HARNESS]["path"])
        manifest["source_closure"].append(derived_record)
        compiled = runtime.document(manifest["compile"])
        compiled["argv"].insert(-1, derived_record["path"])
        manifest["compile"] = f.store_json("compile.json", compiled)
        transaction = runtime.document(manifest["transaction"])
        transaction["output"] = outputs["final.hex"]
        transaction["output_state"] = outputs["candidate.state"]
        transaction["raw"].update(trace=outputs["trace.hex"], terminal=outputs["terminal.txt"])
        manifest["transaction"] = f.store_json("transaction.json", transaction)
        extracted = capture.actual_arrays(transaction, arrays["input_hidden"],
                                          trusted["trace_coordinates"])
        for name in arrays:
            np.testing.assert_array_equal(extracted[name], arrays[name])
        archive = self.root / "captured_arrays.npz"
        with archive.open("xb") as stream:
            np.savez(stream, **extracted)
        manifest["actual_operands"] = f.record(archive)
        launch = runtime.document(manifest["launch"])
        launch["argv"].append(harness.CAPTURE_FLAG)
        manifest["launch"] = f.store_json("launch.json", launch)
        execution = runtime.document(manifest["execution"])
        execution.update(launch=manifest["launch"], transaction=manifest["transaction"],
                         argv=launch["argv"], output_state=transaction["output_state"])
        manifest["execution"] = f.store_json("execution.json", execution)
        with patch.object(f, "context", trusted):
            result = f.evaluate(manifest, arrays=extracted)
        self.assertEqual(result["status"], "PASS", result.get("reason"))
        self.assertEqual(len(result["reports"]), 19)
        self.assertEqual(result["runtime_admission"]["saved_state"]["valid_entries"], 128)
        launch["argv"].remove(harness.CAPTURE_FLAG)
        manifest["launch"] = f.store_json("no_capture_flag.json", launch)
        execution.update(launch=manifest["launch"], argv=launch["argv"])
        manifest["execution"] = f.store_json("no_capture_execution.json", execution)
        with patch.object(f, "context", trusted):
            blocked = f.evaluate(manifest, arrays=extracted)
        f.assert_blocked(blocked)
        self.assertIn("capture mode/source mismatch", blocked["reason"])


if __name__ == "__main__":
    unittest.main()
