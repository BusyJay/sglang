"""Run the registered startup failure probe with only test import glue isolated."""

import ast
import sys
import types
import unittest
import os

from common import OUT, ROOT, load

if os.environ.get("WARMUP_MUTANT") == "lazy":
    path = ROOT / "python/sglang/srt/hardware_backend/mlx/warmup.py"
    mutant = types.ModuleType("sglang.srt.hardware_backend.mlx.warmup")
    exec(compile(path.read_text().replace("mx.eval(block(x, indices))", "block(x, indices)"),
                 "lazy_warmup_mutant", "exec"), mutant.__dict__)
    sys.modules[mutant.__name__] = mutant

for name in ("sglang.test", "sglang.test.ci"):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
    sys.modules[name] = pkg
tree = ast.parse((ROOT / "python/sglang/test/test_utils.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CustomTestCase")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init_subclass__")
node = ast.ClassDef(
    name="CustomTestCase",
    bases=[ast.Attribute(value=ast.Name(id="unittest", ctx=ast.Load()), attr="TestCase", ctx=ast.Load())],
    keywords=[], body=[method], decorator_list=[],
)
module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
namespace = {"unittest": unittest, "_wait_for_gpu_idle_in_ci": lambda: None, "is_in_ci": lambda: False}
exec(compile(module, "extracted_CustomTestCase", "exec"), namespace)
utils = types.ModuleType("sglang.test.test_utils")
utils.CustomTestCase = namespace["CustomTestCase"]
sys.modules[utils.__name__] = utils
load("sglang.test.ci.ci_register", ROOT / "python/sglang/test/ci/ci_register.py")
test = load("device_startup_test", ROOT / "test/registered/unit/hardware_backend/mlx/test_mlx_startup_warmup.py")
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
raise SystemExit(0 if result.wasSuccessful() else 1)
