# SPDX-License-Identifier: Apache-2.0
"""Guarded native B12X scale-load substitution, including FP32/hybrid variants.

Native scale address arithmetic remains unchanged. Only its uint32 loads turn
into CSF loads; no expanded GPU buffer or separate decompression launch exists.
"""

import ast
import copy
import hashlib
import inspect
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace


def csf_method(source, specs, *, direct_expert=False):
    tree = ast.parse(textwrap.dedent(source))
    origins = {
        "w1s_base_addr": "w13_blockscale_swizzled",
        "w2s_base_addr": "w2_blockscale_swizzled",
    }

    def origin(node):
        found = {
            origins[n.id]
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and n.id in origins
        }
        if len(found) > 1:
            raise RuntimeError("ambiguous B12X scale address")
        return next(iter(found), None)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            plane = origin(node.value)
            if plane:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        origins[target.id] = plane
    changed = 0

    class Transform(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "prefetch_global_l2"
                and origin(node.value)
            ):
                # Native virtual offsets must never prefetch beyond a CSF slab.
                return ast.Pass()
            return self.generic_visit(node)

        def visit_Call(self, node):
            nonlocal changed
            self.generic_visit(node)
            if (
                direct_expert
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "_resolve_route_expert"
            ):
                node.args.append(ast.Name(id="route_expert_limit", ctx=ast.Load()))
                return node
            if not isinstance(node.func, ast.Name) or not node.func.id.startswith(
                "ld_global_nc_"
            ):
                return node
            plane = origin(node.args[0]) if node.args else None
            if plane not in specs:
                return node
            if node.func.id != "ld_global_nc_u32" or len(node.args) != 1:
                raise RuntimeError("unsupported B12X native scale load")
            base = ast.Name(
                id="w1s_base_addr" if plane.startswith("w13") else "w2s_base_addr",
                ctx=ast.Load(),
            )
            spec = specs[plane]
            changed += 1
            offset = ast.BinOp(left=node.args[0], op=ast.Sub(), right=base)
            args = [base, offset]
            if direct_expert:
                # Native route resolution already produced this expert ID.
                # Keep its local scale offset instead of recovering the ID
                # with a 64-bit division for every four decoded scale bytes.
                expert = ast.Name(id="eid", ctx=ast.Load())
                args = [
                    base,
                    expert,
                    ast.BinOp(
                        left=offset,
                        op=ast.Sub(),
                        right=ast.BinOp(
                            left=ast.Call(
                                func=ast.Name(id="Int64", ctx=ast.Load()),
                                args=[expert],
                                keywords=[],
                            ),
                            op=ast.Mult(),
                            right=ast.Constant(spec["rows"] * spec["columns"]),
                        ),
                    ),
                ]
            return ast.Call(
                func=ast.Name(
                    id="_csf_word_at" if direct_expert else "_csf_word",
                    ctx=ast.Load(),
                ),
                args=[
                    *args,
                    *[ast.Constant(spec[k]) for k in ("rows", "columns", "stride")],
                    *(
                        [ast.Constant(True)]
                        if spec.get("encoding") == "b12x_csf_tiered_v1"
                        else []
                    ),
                ],
                keywords=[],
            )

    tree = Transform().visit(tree)
    return ast.fix_missing_locations(tree), changed


def csf_kernel_namespace(native, specs):
    """Called once at adapter startup; derive from the installed/pinned class."""
    specs = {
        name: {
            "encoding": spec.get("encoding", "b12x_csf_bitmap_v1"),
            **{key: spec[key] for key in ("rows", "columns", "stride")},
        }
        for name, spec in specs.items()
    }
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    if any(
        s.get("encoding", "b12x_csf_bitmap_v1")
        not in ("b12x_csf_bitmap_v1", "b12x_csf_tiered_v1")
        for s in specs.values()
    ):
        raise ValueError("B12X requires bitmap/rank CSF; regenerate the pack offline")
    from b12x.moe._shared.kernels import micro
    from cutlass import Int32, Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def _csf_u8(address: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Int64(address).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u8 $0, [$1];",
                "=r,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _csf_codes(address: Int64, *, loc=None, ip=None) -> Uint32:
        # Native scale words are four-byte aligned, so their two packed
        # nibble bytes are two-byte aligned as well.
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Int64(address).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u16 $0, [$1];",
                "=r,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @micro.cute.jit
    def _csf_word_at(
        base: Int64,
        expert: Int32,
        offset: Int64,
        rows: micro.cutlass.Constexpr,
        columns: micro.cutlass.Constexpr,
        stride: micro.cutlass.Constexpr,
        tiered: micro.cutlass.Constexpr = False,
    ) -> Uint32:
        pos = Int32(offset)
        slab = base + expert.to(Int64) * Int64(stride)
        # Bitmap/rank is shared by A4 and A16.
        tile = pos // Int32(512)
        lane = (pos % Int32(512)) // Int32(4)
        fixed = Int64(1024) + Int64(pos // Int32(columns * 128)) * Int64(
            128 * (1 + columns // 2)
        )
        b = _csf_u8(slab + fixed + Int64(lane))
        codes = (
            slab + fixed + Int64(128) + Int64((pos % Int32(columns * 128)) // Int32(2))
        )
        packed = _csf_codes(codes)
        deltas = (
            (packed & Uint32(0xF))
            | ((packed & Uint32(0xF0)) << Uint32(4))
            | ((packed & Uint32(0xF00)) << Uint32(8))
            | ((packed & Uint32(0xF000)) << Uint32(12))
        )
        word = deltas + b * Uint32(0x01010101)
        metadata = (
            slab + Int64(1024 + rows * (1 + columns // 2)) + Int64(tile) * Int64(32)
        )
        group, bit = lane // Int32(32), lane % Int32(32)
        mask = micro.ld_global_nc_u32(metadata + Int64(8) + Int64(group) * Int64(4))
        flag = Uint32(1) << bit.to(Uint32)
        if (mask & flag) != Uint32(0):
            start = micro.ld_global_nc_u32(metadata)
            prefix = micro.ld_global_nc_u32(metadata + Int64(4))
            rank = (prefix >> (group.to(Uint32) * Uint32(8))) & Uint32(255)
            rank += micro.cute.arch.popc(mask & (flag - Uint32(1)))
            records = slab + Int64(
                1024 + rows * (1 + columns // 2) + rows * columns // 16
            )
            if micro.cutlass.const_expr(tiered):
                external = (
                    micro.ld_global_nc_u32(slab + Int64(992)).to(Int64)
                    & Int64(0xFFFFFFFF)
                ) | (micro.ld_global_nc_u32(slab + Int64(996)).to(Int64) << Int64(32))
                if external != Int64(0):
                    records = external
            word = micro.ld_global_nc_u32(records + (start + rank).to(Int64) * Int64(4))
        return word

    cls = native.MoEMicroKernelW4A16SmallMDirect

    @micro.cute.jit
    def _resolve_route_expert(self, route_eid: Int64, route_expert_limit: Int32):
        eid, active = Int32(0), Int32(0)
        if route_eid >= Int64(0) and route_eid < Int64(route_expert_limit):
            eid, active = Int32(route_eid), Int32(1)
        return eid, active

    @micro.cute.jit
    def _fc2_route(self, topk_ids, topk_weights, eid_addr, route_expert_limit):
        eid, active = self._resolve_route_expert(
            Int64(topk_ids[eid_addr]), route_expert_limit
        )
        weight = micro.Float32(0.0)
        if active > Int32(0):
            weight = micro.Float32(topk_weights[eid_addr])
        return eid, weight, active

    namespace = dict(vars(micro), _csf_word_at=_csf_word_at)
    namespace.update(getattr(native, "_qwen38_helpers", {}))
    # Hybrid/FP32 derived methods refer to generated helpers in their globals.
    generated = Path(tempfile.mkdtemp(prefix="qwen38-b12x-csf-"))
    overrides, total = {}, 0
    for name in (
        "kernel",
        "_m1_fc2_rowpair_narrow",
        "_m1_fc2_rowpair_wide",
        "_m2_fc2_rowquad_narrow",
        "_m2_fc2_rowpair_narrow",
        "_m2_fc2_rowquad_wide",
    ):
        method = getattr(cls, name)
        tree, count = csf_method(inspect.getsource(method), specs, direct_expert=True)
        if count:
            ns = dict(
                namespace,
                **{
                    k: v
                    for k, v in method.__globals__.items()
                    if k.startswith("_hybrid")
                },
            )
            path = generated / (name + ".py")
            path.write_text(ast.unparse(tree) + "\n")
            exec(compile(path.read_text(), str(path), "exec"), ns)
            overrides[name] = ns[name]
            total += count
    if total == 0 or "kernel" not in overrides:
        raise RuntimeError("B12X native scale load definitions changed")
    overrides.update(_resolve_route_expert=_resolve_route_expert, _fc2_route=_fc2_route)
    derived = type("Qwen38SmallMDirectCsf", (cls,), overrides)
    compiled = dict(vars(native))
    compiled["MoEMicroKernelW4A16SmallMDirect"] = derived
    compiled["_SMALL_M_DIRECT_CACHE"] = {}
    # Only the route bound depends on the physical slot count. Both FC1 and
    # FC2 now check the runtime extent; reuse programs across KV/cache resize.
    compiled["_qwen38_capacity_bucket"] = True
    tree = (
        copy.deepcopy(native._qwen38_compile_tree)
        if hasattr(native, "_qwen38_compile_tree")
        else ast.parse(
            textwrap.dedent(inspect.getsource(native._compile_w4a16_small_m_direct))
        )
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(
                ("w4a16_small_m_direct", "moe.w4a16.small_m_direct")
            )
        ):
            node.value += (
                ".csf."
                + source_hash
                + repr(sorted((n, tuple(sorted(s.items()))) for n, s in specs.items()))
            )
    exec(
        compile(
            tree, inspect.getsourcefile(native._compile_w4a16_small_m_direct), "exec"
        ),
        compiled,
    )
    return SimpleNamespace(**compiled)
