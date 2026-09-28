"""No-claim CLI integration through real source, archive, state and output gates."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shlex
import unittest
from unittest import mock

from ace3.model.candidates import stage11_exact_integer_o_projection_execution_v1 as w


d = w.library
fixtures = d.retained.module_from(d.TEST, "stage11_execution_fixtures")


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.native.torch.set_num_threads(1)

    def setUp(self):
        self.fixture = fixtures.AuthenticatedFixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.handoffs = self.root / "handoffs"
        self.mission_id = "abcdef012345"
        owner = self.handoffs / self.mission_id
        owner.mkdir(parents=True)
        self.out = self.root / f"{w.NAME}-{self.mission_id}-attempt001"
        self.out.mkdir()
        self.mission_path = owner / "mission.json"
        self.mission = {
            "mission_id": self.mission_id, "scope": "bounded", "node_key": w.NODE_KEY,
            "execution_workdir": str(w.ROOT), "deps": [], "tags": ["review:required"],
        }
        self.mission_path.write_text(json.dumps(self.mission))
        self.review_path = self.handoffs / w.SOFTWARE_MISSION / "round-0001.json"
        self.review_path.parent.mkdir()
        self.review = {"mission_id": w.SOFTWARE_MISSION, "producer_role": "reviewer",
                       "review": {"status": "done"}}
        self.review_path.write_text(json.dumps(self.review))
        self.handoff = owner / "stage11-exact-integer-execution.json"
        self.document = {
            "schema_version": 1, "mission_id": self.mission_id, "science_budget": 1,
            "mission": w.pin(self.mission_path), "review": w.pin(self.review_path),
            "output": str(self.out), "output_claim": str(self.out / "command.json"),
            "source_pins": [w.pin(Path(p)) for p in sorted(w.project_sources())],
        }
        self.publish()
        patch = mock.patch.object(w, "HANDOFFS", self.handoffs)
        patch.start()
        self.addCleanup(patch.stop)
        for module, name in ((d, "evaluate"), (d, "compute_suffix"), (w, "execute"),
                             (d.gate_up, "check"), (d.base, "execute"), (d.prior, "execute")):
            patch = mock.patch.object(module, name, side_effect=AssertionError("production science"))
            guard = patch.start()
            self.addCleanup(guard.assert_not_called)
            self.addCleanup(patch.stop)
        self.addCleanup(self.assert_no_claim)

    def assert_no_claim(self):
        self.assertFalse((self.out / "command.json").exists())
        self.assertFalse((self.out / "result.json").exists())

    def publish(self):
        self.handoff.write_text(json.dumps(self.document))
        self.sha256 = w.pin(self.handoff)["sha256"]
        (self.out / "command.sh").write_text(w.command(self.handoff, self.sha256))

    def cli(self):
        args = shlex.split(w.command(self.handoff, self.sha256))[4:]
        self.assertEqual(args[0], "--execute")
        args[0] = "--preflight"
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(w.main(args), 0)
        return json.loads(output.getvalue())

    def test_full_preclaim_entry_authenticates_nine_payloads_and_is_read_only(self):
        before = {str(p): w.pin(p) for p in self.root.rglob("*") if p.is_file()}
        receipt = self.cli()
        self.assertEqual(receipt["status"], "PREFLIGHT_ONLY")
        self.assertEqual(receipt["science_invocations"], 0)
        self.assertFalse(receipt["output_claim_acquired"])
        self.assertEqual(receipt["command_bytes"], (self.out / "command.sh").read_text())
        self.assertEqual(receipt["source_pins"], self.document["source_pins"])
        self.assertEqual(len(receipt["source_pins"]), 64)
        self.assertEqual(len(d.project_sources()), 62)
        self.assertEqual(set(receipt["retained_inputs"]["stage10"]), set(d.CONTROLS))
        self.assertEqual(receipt["retained_inputs"]["baseline"], self.fixture.result_pin)
        self.assertEqual(receipt["retained_inputs"]["review"], self.fixture.review_pin)
        self.assertIn(self.fixture.result_pin, receipt["input_pins"])
        self.assertIn(self.fixture.review_pin, receipt["input_pins"])
        for row in self.fixture.document["controls"]:
            self.assertIn(row["payload"], receipt["input_pins"])
        self.assertEqual(before, {str(p): w.pin(p) for p in self.root.rglob("*") if p.is_file()})

    def test_payload_failure_traverses_real_preclaim_consumer(self):
        path = Path(self.fixture.document["controls"][-1]["payload"]["path"])
        with path.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            self.cli()

    def test_missing_fields_and_wrong_source_command_claim_bindings(self):
        pristine = copy.deepcopy(self.document)
        for field in ("mission", "mission_id", "review", "source_pins", "output", "output_claim"):
            self.document = copy.deepcopy(pristine)
            del self.document[field]
            self.publish()
            with self.subTest(field=field), self.assertRaises((KeyError, ValueError)):
                self.cli()
        for defect in ("budget", "bool_budget", "schema", "source_missing", "source_extra",
                       "source_order", "source_hash", "source_size", "review_hash", "mission_hash",
                       "output", "claim", "command", "handoff_hash"):
            self.document = copy.deepcopy(pristine)
            if defect in ("budget", "bool_budget", "schema"):
                self.document["schema_version" if defect == "schema" else "science_budget"] = (
                    True if defect != "budget" else 2)
            elif defect == "source_missing":
                self.document["source_pins"].pop()
            elif defect == "source_extra":
                self.document["source_pins"].append(w.pin(self.mission_path))
            elif defect == "source_order":
                self.document["source_pins"].reverse()
            elif defect.startswith("source_"):
                self.document["source_pins"][0]["sha256" if defect == "source_hash" else "bytes"] = (
                    "0" * 64 if defect == "source_hash" else 1)
            elif defect in ("review_hash", "mission_hash"):
                self.document[defect.removesuffix("_hash")]["sha256"] = "0" * 64
            elif defect == "output":
                self.document["output"] = str(w.ROOT)
            elif defect == "claim":
                self.document["output_claim"] = str(self.root / "command.json")
            self.publish()
            if defect == "command":
                (self.out / "command.sh").write_text("wrong command\n")
            elif defect == "handoff_hash":
                self.sha256 = "0" * 64
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.cli()

    def test_owner_and_independent_review_gates(self):
        for field, value in (("producer_role", "engineer"), ("mission_id", self.mission_id),
                             ("review", {"status": "blocked"})):
            self.review_path.write_text(json.dumps({**self.review, field: value}))
            self.document["review"] = w.pin(self.review_path)
            self.publish()
            with self.assertRaisesRegex(ValueError, "independent software acceptance"):
                self.cli()
        self.review_path.write_text(json.dumps(self.review))
        self.document["review"] = w.pin(self.review_path)
        for field, value in (("mission_id", w.SOFTWARE_MISSION), ("scope", "unbounded"),
                             ("node_key", "stage16-silu-product-execution"),
                             ("deps", ["unfinished"]), ("execution_workdir", "/tmp"),
                             ("tags", []), ("authorization_id", "fabricated"), ("action", "run")):
            self.mission_path.write_text(json.dumps({**self.mission, field: value}))
            self.document["mission"] = w.pin(self.mission_path)
            self.publish()
            with self.assertRaisesRegex(ValueError, "owner/dependency"):
                self.cli()

    def test_missing_occupied_aliased_and_noncanonical_roots(self):
        for kind in ("missing", "occupied", "symlink", "noncanonical"):
            target = self.root / kind / self.out.name
            target.parent.mkdir()
            if kind == "occupied":
                target.mkdir()
                (target / "command.json").write_text("historical UNKNOWN claim")
            elif kind == "symlink":
                target.symlink_to(self.out, target_is_directory=True)
            elif kind == "noncanonical":
                target = self.root / "other-attempt"
            self.document["output"] = str(target)
            self.document["output_claim"] = str(target / "command.json")
            self.publish()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.cli()
        self.assertEqual((self.root / "occupied" / self.out.name / "command.json").read_text(),
                         "historical UNKNOWN claim")

    def test_payload_and_write_gates(self):
        with self.assertRaisesRegex(ValueError, "payload control census"):
            w.emit_payloads(self.out, {}, {})
        for path, mode in ((self.root / "outside", "x"), (self.out / "command.sh", "w")):
            audit = d.base.new_audit()
            with self.assertRaises(ValueError), w.writing(self.out, audit):
                path.open(mode)
            self.assertTrue(audit["forbidden_events"])


if __name__ == "__main__":
    unittest.main()
