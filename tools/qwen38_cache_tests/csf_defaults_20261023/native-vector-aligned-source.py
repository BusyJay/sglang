# SPDX-License-Identifier: Apache-2.0
"""Native Qwen38 FC1 vector loads, derived without changing installed B12X."""

import ast
import copy
import hashlib
import inspect
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def vector_method(source, bits):
    if bits not in (64, 128):
        raise ValueError("native vector loads require 64 or 128 bits")
    tree = ast.parse(textwrap.dedent(source))
    changed = 0

    class Transform(ast.NodeTransformer):
        def visit_For(self, node):
            nonlocal changed
            # Only the aligned generic FC1 loop has these unconditional loads.
            # Qwen38 has five K16 blocks per lane (40 packed weight bytes).
            pairs = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        pairs[target.id] = stmt
            if not all(name in pairs for name in ("gw0", "gw1")):
                return self.generic_visit(node)
            for name in ("gw0", "gw1"):
                call = pairs[name].value
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "ld_global_nc_u32"):
                    return self.generic_visit(node)
            changed += 1
            prefix = []
            if bits == 128:
                prefix = ast.parse("""
_qwen_gw = _qwen_load40(gate_byte_addr)
if cutlass.const_expr(self.is_gated):
    _qwen_uw = _qwen_load40(up_byte_addr)
""").body

            class Loads(ast.NodeTransformer):
                def visit_Assign(self, stmt):
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        return stmt
                    name = stmt.targets[0].id
                    if name not in ("gw0", "gw1", "uw0", "uw1"):
                        return stmt
                    call = stmt.value
                    if not (isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Name)
                            and call.func.id == "ld_global_nc_u32"):
                        raise RuntimeError("B12X FC1 vector load definition changed")
                    if bits == 128:
                        stmt.value = ast.parse(
                            f"_qwen_{name[:2]}[seg * 2 + {name[-1]}]", mode="eval"
                        ).body
                        return stmt
                    if name.endswith("1"):
                        return None
                    addr = ast.unparse(call.args[0])
                    return ast.parse(
                        f"{name}, {name[:2]}1 = _qwen_load64({addr})"
                    ).body[0]

            node = Loads().visit(node)
            return prefix + [node]

    tree = Transform().visit(tree)
    if changed != 1:
        raise RuntimeError("B12X aligned FC1 loop definition changed")
    return ast.fix_missing_locations(tree)


def vector_kernel_namespace(native, bits):
    return _vector_kernel_namespace(
        native.MoEMicroKernelW4A16SmallMDirect,
        native._compile_w4a16_small_m_direct,
        bits,
    )


@lru_cache(maxsize=6)
def _vector_kernel_namespace(cls, compile_function, bits):
    native = SimpleNamespace(**compile_function.__globals__)
    from b12x.moe._shared.kernels import micro
    from cutlass import Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def _qwen_load64(address: Int64, *, loc=None, ip=None):
        value = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32()]),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.v2.u32 {$0, $1}, [$2];", "=r,=r,l",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
        return tuple(Uint32(llvm.extractvalue(T.i32(), value, [i], loc=loc, ip=ip)) for i in range(2))

    @dsl_user_op
    def _qwen_load128(address: Int64, *, loc=None, ip=None):
        value = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 4),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
        return tuple(Uint32(llvm.extractvalue(T.i32(), value, [i], loc=loc, ip=ip)) for i in range(4))

    @micro.cute.jit
    def _qwen_load40(address: Int64):
        # Forty bytes per lane alternate between 16- and 8-byte alignment.
        # Read three aligned vectors and select the useful ten words. The
        # extra eight bytes overlap a neighbor; the final lane ends exactly
        # at the 1280-byte row boundary, so no read crosses an expert extent.
        aligned = address & ~Int64(15)
        a, b, c, d = _qwen_load128(aligned)
        e, f, g, h = _qwen_load128(aligned + Int64(16))
        i, j, k, l = _qwen_load128(aligned + Int64(32))
        if (address & Int64(8)) != Int64(0):
            a, b, c, d, e, f, g, h, i, j = c, d, e, f, g, h, i, j, k, l
        return a, b, c, d, e, f, g, h, i, j

    tree = vector_method(inspect.getsource(cls.kernel), bits)
    generated = Path(tempfile.mkdtemp(prefix="qwen38-b12x-vector-")) / "kernel.py"
    generated.write_text(ast.unparse(tree) + "\n")
    ns = dict(vars(micro), _qwen_load64=_qwen_load64, _qwen_load40=_qwen_load40)
    ns.update(getattr(native, "_qwen38_helpers", {}))
    exec(compile(generated.read_text(), str(generated), "exec"), ns)
    derived = type(f"Qwen38NativeLoad{bits}", (cls,), {"kernel": ns["kernel"]})
    namespace = dict(vars(native))
    namespace["MoEMicroKernelW4A16SmallMDirect"] = derived
    namespace["_SMALL_M_DIRECT_CACHE"] = {}
    tree = (copy.deepcopy(native._qwen38_compile_tree)
            if hasattr(native, "_qwen38_compile_tree") else
            ast.parse(textwrap.dedent(inspect.getsource(native._compile_w4a16_small_m_direct))))
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(
            ("w4a16_small_m_direct", "moe.w4a16.small_m_direct")
        ):
            node.value += f".qwen38_load{bits}.{digest}"
    exec(compile(tree, inspect.getsourcefile(native._compile_w4a16_small_m_direct), "exec"), namespace)
    namespace["_qwen38_compile_tree"] = tree
    return SimpleNamespace(**namespace)
