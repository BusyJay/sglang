# SPDX-License-Identifier: Apache-2.0
"""B12X FC2 epilogue with device-selected CPU contribution and no merge launch."""

import ast
import inspect
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

# Only these FC2 writers and their three callers may gain hybrid arguments.
_WRITERS = (
    "_m1_fc2_rowpair_narrow",
    "_m1_fc2_rowpair_wide",
    "_m2_fc2_rowquad_narrow",
    "_m2_fc2_rowquad_wide",
    "_m2_fc2_rowpair_narrow",
    "_run_fc2_trellis",
)
_CALLERS = ("__call__", "kernel", "_run_fc2")
_ARGUMENTS = (
    "hybrid_plan",
    "hybrid_completed",
    "hybrid_result",
    "hybrid_failed",
    "hybrid_sequence",
)


def hybrid_method(source, name):
    tree = ast.parse(textwrap.dedent(source))
    method = tree.body[0]
    if not isinstance(method, ast.FunctionDef) or method.name != name:
        raise RuntimeError(f"unexpected B12X method {name}")
    if method.args.kwonlyargs or method.args.vararg or method.args.kwarg:
        raise RuntimeError("unsupported B12X method signature")
    # __call__ has two optional trellis pointers following the CUDA stream.
    insert = len(method.args.args) - (3 if name == "__call__" else 0)
    if name == "__call__" and [a.arg for a in method.args.args[-3:]] != [
        "stream",
        "trellis_lut_ptr",
        "trellis_rot_ptr",
    ]:
        raise RuntimeError("B12X launch ABI changed")
    pointer_type = "cute.Pointer" if name in ("__call__", "kernel") else "cute.Tensor"
    annotations = [pointer_type] * 4 + ["Int32"]
    for offset, (arg, annotation) in enumerate(zip(_ARGUMENTS, annotations)):
        method.args.args.insert(
            insert + offset,
            ast.arg(arg=arg, annotation=ast.parse(annotation, mode="eval").body),
        )

    forwarded = 0
    scattered = 0

    class Transform(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal forwarded
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in _WRITERS + ("_run_fc2", "kernel")
            ):
                forwarded_names = (
                    tuple(f"{arg}_tensor" for arg in _ARGUMENTS[:4]) + (_ARGUMENTS[4],)
                    if name == "kernel"
                    else _ARGUMENTS
                )
                node.args.extend(
                    ast.Name(id=arg, ctx=ast.Load()) for arg in forwarded_names
                )
                forwarded += 1
            return node

        def visit_Assign(self, node):
            nonlocal scattered
            self.generic_visit(node)
            if name not in _WRITERS:
                return node
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
                return node
            target = node.targets[0]
            if (
                not isinstance(target.value, ast.Name)
                or target.value.id != "scatter_output"
            ):
                return node
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "BFloat16"
                and len(value.args) == 1
            ):
                raise RuntimeError("unsupported B12X FC2 scatter")
            # Keep the native FP32 accumulation, add CPU before the sole BF16 cast.
            node.value = ast.Call(
                func=ast.Name(id="BFloat16", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id="_hybrid_add", ctx=ast.Load()),
                        args=[value.args[0], target.slice]
                        + [ast.Name(id=arg, ctx=ast.Load()) for arg in _ARGUMENTS],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
            scattered += 1
            return node

    tree = Transform().visit(tree)
    if name == "kernel":
        if forwarded != 2:
            raise RuntimeError("B12X kernel FC2 call sites changed")
        tensors = ast.parse(
            "hybrid_plan_tensor = cute.make_tensor(hybrid_plan, cute.make_layout(Int32(4)))\n"
            "hybrid_completed_tensor = cute.make_tensor(hybrid_completed, cute.make_layout(Int32(1)))\n"
            "hybrid_result_tensor = cute.make_tensor(hybrid_result, cute.make_layout(Int32(3 * cfg.k_dim)))\n"
            "hybrid_failed_tensor = cute.make_tensor(hybrid_failed, cute.make_layout(Int32(1)))\n"
        ).body
        cfg_index = next(
            i
            for i, statement in enumerate(method.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "cfg"
                for target in statement.targets
            )
        )
        method.body[cfg_index + 1 : cfg_index + 1] = tensors
    if (name in _WRITERS and not scattered) or (name in _CALLERS and not forwarded):
        raise RuntimeError(f"B12X {name} no longer has its guarded FC2 calls")
    return ast.fix_missing_locations(tree), scattered


def hybrid_compile_function(source):
    tree = ast.parse(textwrap.dedent(source))
    found = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "b12x_compile"
        ):
            if len(node.args) < 20:
                raise RuntimeError("B12X direct compile ABI changed")
            stream = node.args[-1]
            if (
                not isinstance(stream, ast.Call)
                or not isinstance(stream.func, ast.Name)
                or stream.func.id != "current_cuda_stream"
            ):
                raise RuntimeError("B12X compile stream changed")
            extra = ["Int32", "Int32", "Float32", "Int32"]
            node.args[-1:-1] = [
                ast.parse(f"dummy(cutlass.{dt})", mode="eval").body for dt in extra
            ] + [ast.parse("cutlass.Int32(0)", mode="eval").body]
            found += 1
        if isinstance(node, ast.Constant) and node.value in (
            "w4a16_small_m_direct",
            "moe.w4a16.small_m_direct",
        ):
            node.value += ".qwen38_hybrid"
    if found != 1:
        raise RuntimeError("B12X direct compile entry changed")
    return ast.fix_missing_locations(tree)


@lru_cache(maxsize=1)
def hybrid_kernel_namespace():
    from b12x.moe._shared.kernels import micro
    from b12x.moe._shared.kernels.w4a16 import kernel
    from cutlass import Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def load_cv_f32(address: Int64, *, loc=None, ip=None) -> Float32:
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Int64(address).ir_value(loc=loc, ip=ip)],
                "ld.global.cv.f32 $0, [$1];",
                "=f,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def load_cv_i32(address: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int64(address).ir_value(loc=loc, ip=ip)],
                "ld.global.cv.s32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def fail_trap(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "trap;",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    @micro.cute.jit
    def _hybrid_add(
        value: Float32,
        offset: Int32,
        hybrid_plan: micro.cute.Tensor,
        hybrid_completed: micro.cute.Tensor,
        hybrid_result: micro.cute.Tensor,
        hybrid_failed: micro.cute.Tensor,
        hybrid_sequence: Int32,
    ) -> Float32:
        if Int32(hybrid_plan[2]) > Int32(0):
            completed_address = micro.get_ptr_as_int64(hybrid_completed, Int32(0))
            failed_address = micro.get_ptr_as_int64(hybrid_failed, Int32(0))
            completed = load_cv_i32(completed_address)
            while completed < hybrid_sequence:
                if load_cv_i32(failed_address) != Int32(0):
                    fail_trap()
                completed = load_cv_i32(completed_address)
            # The CPU publishes completion on both success and failure. A
            # failure can therefore arrive with the final counter value.
            if load_cv_i32(failed_address) != Int32(0):
                fail_trap()
            value += load_cv_f32(micro.get_ptr_as_int64(hybrid_result, offset))
        return value

    generated = Path(tempfile.mkdtemp(prefix="qwen38-b12x-hybrid-"))
    overrides = {}
    namespace = dict(vars(micro), _hybrid_add=_hybrid_add)
    for name in _WRITERS + _CALLERS:
        source = inspect.getsource(getattr(micro.MoEMicroKernelBackend, name))
        tree, _ = hybrid_method(source, name)
        path = generated / (name + ".py")
        path.write_text(ast.unparse(tree) + "\n")
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        overrides[name] = namespace[name]
    hybrid_class = type(
        "Qwen38SmallMDirectHybrid", (kernel.MoEMicroKernelW4A16SmallMDirect,), overrides
    )
    compiled_namespace = dict(vars(kernel))
    compiled_namespace["MoEMicroKernelW4A16SmallMDirect"] = hybrid_class
    compiled_namespace["_SMALL_M_DIRECT_CACHE"] = {}
    tree = hybrid_compile_function(
        inspect.getsource(kernel._compile_w4a16_small_m_direct)
    )
    exec(
        compile(
            tree, inspect.getsourcefile(kernel._compile_w4a16_small_m_direct), "exec"
        ),
        compiled_namespace,
    )
    compiled_namespace["_qwen38_helpers"] = {"_hybrid_add": _hybrid_add}
    compiled_namespace["_qwen38_compile_tree"] = tree
    return SimpleNamespace(**compiled_namespace)
