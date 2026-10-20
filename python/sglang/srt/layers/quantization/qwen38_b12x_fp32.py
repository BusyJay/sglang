# SPDX-License-Identifier: Apache-2.0
"""Derive an FP32 epilogue from installed B12X; never mutate upstream classes.

Only scatter-output casts and the output pointer compile type change. Weight
rounding, SwiGLU rounding, scheduling, barriers and prepared bindings stay B12X.
Source guards reject incompatible upstream definitions instead of using BF16.
"""

import ast
import inspect
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def fp32_scatter_method(source):
    tree = ast.parse(textwrap.dedent(source))
    changed = 0

    class Transform(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal changed
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "scatter_output"
                for target in node.targets
            ):
                if not (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "BFloat16"
                ):
                    raise RuntimeError("unsupported B12X scatter epilogue")
                node.value.func.id = "Float32"
                changed += 1
            return node

    tree = Transform().visit(tree)
    return ast.fix_missing_locations(tree), changed


def fp32_compile_function(source):
    tree = ast.parse(textwrap.dedent(source))
    changed = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "b12x_compile"
        ):
            # Native ABI: kernel + x + 12 tensor pointers; output is argument 13.
            if len(node.args) < 14:
                raise RuntimeError("unsupported B12X direct compile ABI")
            output = node.args[13]
            if not (
                isinstance(output, ast.Call)
                and isinstance(output.func, ast.Name)
                and output.func.id == "dummy"
                and len(output.args) == 1
                and isinstance(output.args[0], ast.Attribute)
                and output.args[0].attr == "BFloat16"
            ):
                raise RuntimeError("unsupported B12X output pointer type")
            output.args[0].attr = "Float32"
            changed += 1
    if changed != 1:
        raise RuntimeError("B12X direct compilation definition changed")
    return ast.fix_missing_locations(tree)


@lru_cache(maxsize=1)
def fp32_kernel_namespace():
    from b12x.moe._shared.kernels import micro
    from b12x.moe._shared.kernels.w4a16 import kernel

    overrides = {}
    generated = Path(tempfile.mkdtemp(prefix="qwen38-b12x-fp32-"))
    for name, method in vars(micro.MoEMicroKernelBackend).items():
        if not callable(method):
            continue
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError):
            continue
        if "scatter_output[" not in source:
            continue
        tree, count = fp32_scatter_method(source)
        if count:
            namespace = dict(vars(micro))
            # CuTe inspects the Python source again when lowering a @cute.jit
            # method. Keep an actual file matching the generated code/line map.
            path = generated / (name + ".py")
            path.write_text(ast.unparse(tree) + "\n")
            exec(compile(path.read_text(), str(path), "exec"), namespace)
            overrides[name] = namespace[name]
    if not overrides:
        raise RuntimeError("cannot find guarded B12X FP32 epilogues")
    fp32_class = type(
        "Qwen38SmallMDirectFp32", (kernel.MoEMicroKernelW4A16SmallMDirect,), overrides
    )
    namespace = dict(vars(kernel))
    namespace["MoEMicroKernelW4A16SmallMDirect"] = fp32_class
    namespace["_SMALL_M_DIRECT_CACHE"] = {}  # separate dtype-specialized cache
    tree = fp32_compile_function(
        inspect.getsource(kernel._compile_w4a16_small_m_direct)
    )
    # Compiler cache must distinguish output ABI even with identical shape.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in (
            "w4a16_small_m_direct",
            "moe.w4a16.small_m_direct",
        ):
            node.value += ".qwen38_fp32"
    exec(
        compile(
            tree, inspect.getsourcefile(kernel._compile_w4a16_small_m_direct), "exec"
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)
