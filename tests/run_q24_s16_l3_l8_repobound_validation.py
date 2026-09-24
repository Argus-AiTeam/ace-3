"""Validate only the published repository-bound Q24 regression plan.

Run from /home/argustest/ace3-argus:
    /home/argustest/miniconda3/bin/python -I -B \
        /home/argustest/ace3-argus/tests/run_q24_s16_l3_l8_repobound_validation.py

-I ignores inherited Python search-path settings and the user site; this
process explicitly prepends the repository root. -B suppresses implicit
bytecode writes. A fresh build-local cache prefix prevents repository imports
from reading old bytecode; explicit compilation also stays under build/.
Neither the environment nor Host/global configuration is modified.

This does not certify the historical bare command, which failed at import
and collected zero tests. These synthetic regressions do not confer new
numerical, strict-FP16-state, RTL, token, or full-model admission.
"""

import importlib
import importlib.util
from pathlib import Path
import py_compile
import sys
import tempfile
import unittest


ROOT = Path("/home/argustest/ace3-argus")
PYTHON = Path("/home/argustest/miniconda3/bin/python")
TARGET = "tests.test_q24_s16_toward_zero_l3_l8_v1"
COMPILE_FILES = (
    "ace3/model/candidates/run_q24_s16_toward_zero_native_v1.py",
    "ace3/model/candidates/run_q24_s16_toward_zero_l3_l8_v1.py",
    "tests/test_q24_s16_toward_zero_l3_l8_v1.py",
    "tests/__init__.py",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_origin(name):
    spec = importlib.util.find_spec(name)
    require(spec is not None, f"unresolved repository module: {name}")
    expected = ROOT.joinpath(*name.split("."))
    if spec.submodule_search_locations is not None:
        locations = [Path(path).resolve() for path in spec.submodule_search_locations]
        require(locations == [expected], f"package search-origin mismatch: {name}: {locations}")
        expected = expected / "__init__.py"
    else:
        expected = expected.with_suffix(".py")
    if spec.origin is None:
        require(spec.submodule_search_locations is not None, f"missing source origin: {name}")
        print(f"ORIGIN {name}: namespace {locations}", flush=True)
        return
    origin = Path(spec.origin).resolve()
    require(origin.is_relative_to(ROOT) and origin == expected and origin.is_file(),
            f"source-origin mismatch: {name}: {origin}; expected {expected}")
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == origin,
                f"loaded source-origin mismatch: {name}")
    print(f"ORIGIN {name}: {origin}", flush=True)


def main():
    require(Path.cwd() == ROOT, f"working directory must be {ROOT}")
    require(Path(sys.executable) == PYTHON, f"interpreter must be {PYTHON}")
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,
            "run with the published -I -B flags and assertions enabled")
    sys.path.insert(0, str(ROOT))
    print(f"CONTEXT cwd={Path.cwd()} executable={sys.executable}", flush=True)
    print(f"CONTEXT isolated={sys.flags.isolated} dont_write_bytecode={sys.dont_write_bytecode}"
          f" optimize={sys.flags.optimize} sys.path={sys.path!r}", flush=True)
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q24_repobound_compile_", dir=ROOT / "build") as cache:
        sys.pycache_prefix = cache
        print(f"CONTEXT fresh command-local pycache_prefix={cache}", flush=True)
        for name in (
            "ace3", "ace3.model", "ace3.model.candidates", "tests", TARGET,
            "ace3.model.candidates.q24_s16_toward_zero_native_v1",
            "ace3.model.candidates.run_q24_s16_toward_zero_native_v1",
            "ace3.model.candidates.run_q24_s16_toward_zero_l3_l8_v1",
        ):
            check_origin(name)
        importlib.import_module(TARGET)
        repository_modules = sorted(
            name for name in sys.modules
            if name == "ace3" or name.startswith("ace3.") or name in ("tests", TARGET)
        )
        for name in repository_modules:
            check_origin(name)
        for relative in COMPILE_FILES:
            source = ROOT / relative
            require(source.resolve() == source, f"compile source is redirected: {source}")
            py_compile.compile(str(source), cfile=str(Path(cache) / relative.replace("/", "_")),
                               doraise=True)
            print(f"COMPILED {source}", flush=True)
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(TARGET)
        require(not loader.errors, f"test collection errors: {loader.errors}")
        collected = suite.countTestCases()
        print(f"COLLECTED {collected}", flush=True)
        require(collected == 19, f"expected 19 collected tests, got {collected}")
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        require(result.testsRun == 19, f"expected 19 executed tests, got {result.testsRun}")
        require(not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses,
                "skipped tests or expected/unexpected failure outcomes are not accepted")
        require(result.wasSuccessful(), "target test suite failed")
        for name in sorted(set(sys.modules) - set(repository_modules)):
            if name.startswith("ace3."):
                check_origin(name)
        print("REPO_BOUND_OK collected=19 executed=19 failures=0 errors=0 skipped=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
