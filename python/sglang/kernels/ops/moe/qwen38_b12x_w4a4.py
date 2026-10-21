# SPDX-License-Identifier: Apache-2.0
"""B12X W4A4 MoE with slot-strided bitmap/rank scale operands.

Only operand addressing is adapted. B12X performs scale reconstruction inside
its persistent MoE kernel; there is no expansion allocation or decode launch.
"""

import ast
import inspect
import tempfile
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def _derive(method, namespace, transform, directory):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    tree = transform(tree)
    path = directory / (method.__name__ + ".py")
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace[method.__name__]


def prepare_dynamic_launch(namespace):
    """Reuse the pinned vendor policy and ABI without launching during binding.

    External route planning retains the ordinary vendor launch path. Prepared
    bindings own the barrier reset and update only invocation-specific pointers.
    B12X launch policy environment settings must be fixed before adapter creation.
    """
    method = namespace._launch_dynamic_flat

    def transform(tree):
        body = tree.body[0].body
        resets = planners = launches = 0
        for i, node in enumerate(body):
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                if condition == "volatile_launch_state and (not external_route_plan)":
                    body[i] = ast.Pass()
                    resets += 1
                elif condition == "external_route_plan":
                    node.body = [ast.Return(value=ast.Constant(None))]
                    planners += 1
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "compiled"
            ):
                body[i] = ast.Return(
                    value=ast.Tuple(
                        elts=[
                            ast.Name(id="compiled", ctx=ast.Load()),
                            ast.Tuple(elts=node.value.args, ctx=ast.Load()),
                        ],
                        ctx=ast.Load(),
                    )
                )
                launches += 1
        if (resets, planners, launches) != (1, 1, 1):
            raise RuntimeError("B12X dynamic launch preparation changed")
        return tree

    directory = Path(tempfile.mkdtemp(prefix="qwen38-b12x-bind-"))
    return _derive(method, dict(method.__globals__), transform, directory)


def dynamic_kernel_namespace(specs):
    """Derive from FlashInfer 0.7.2 without mutating vendor module globals."""
    import flashinfer
    from b12x._lib.quant import nvfp4_csf_inline as reader
    from b12x.moe._shared.kernels import dynamic, silu
    from b12x.moe.fused_moe import _impl as impl

    if flashinfer.__version__ != "0.7.2":
        raise RuntimeError("Qwen38 B12X W4A4 requires FlashInfer 0.7.2")
    directory = Path(tempfile.mkdtemp(prefix="qwen38-b12x-w4a4-"))

    # Keep the upstream register decoder. Exceptions read their original
    # replacement words directly: no optional shared replacement payload.
    def global_records(tree):
        changed = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                old = "setp.ge.u32 absent, rank, 80;"
                if old in node.value:
                    node.value = node.value.replace(old, "setp.ge.u32 absent, rank, 0;")
                    changed += 1
        if changed != 1:
            raise RuntimeError("B12X shared scale decoder changed")
        return tree

    ns = dict(vars(reader))
    _derive(reader._shared_scale_word, ns, global_records, directory)
    from b12x._lib.intrinsics import ld_shared_v2_u32, st_shared_u32

    ns.update(ld_shared_v2_u32=ld_shared_v2_u32, st_shared_u32=st_shared_u32)

    def make_reader(rows, columns):
        name = (
            "w13_blockscale_swizzled"
            if rows == specs["_w13_rows"]
            else "w2_blockscale_swizzled"
        )
        spec = specs.get(name)
        if columns % 8:
            raise ValueError("Qwen38 W4A4 requires full K128 scale operands")
        stride = spec["stride"] if spec else 0
        if spec and spec.get("encoding", "b12x_csf_bitmap_v1") != "b12x_csf_bitmap_v1":
            raise ValueError("W4A4 requires a bitmap/rank pack; convert it offline")
        local = dict(ns, _slot_stride=stride)
        reader_directory = directory / name
        reader_directory.mkdir(exist_ok=True)

        def stage_tree(tree):
            source = ast.unparse(tree)
            required = [
                "zero = cute.recast_ptr(storage, dtype=cutlass.Uint8)",
                "slab = Int64(expert) * Int64(self.rows // 128) + Int64(row_tile)",
                "meta = Int64(experts) * Int64(self.fixed_bytes) + tile * Int64(32)",
                "records = Int64(experts) * Int64(self.fixed_bytes + self.tiles * 32)",
            ]
            if not all(x in source for x in required):
                raise RuntimeError("B12X scale stage addressing changed")
            source = source.replace(
                required[0], required[0] + " + Int64(expert) * Int64(_slot_stride)"
            )
            source = source.replace(required[1], "slab = Int64(row_tile)")
            source = source.replace(
                required[2], "meta = Int64(self.fixed_bytes) + tile * Int64(32)"
            )
            source = source.replace(
                required[3], "records = Int64(self.fixed_bytes + self.tiles * 32)"
            )
            tree = ast.parse(source)

            class RemovePayload(ast.NodeTransformer):
                def visit_If(self, node):
                    if ast.unparse(node.test) == "count > Int32(0)":
                        return ast.parse("""address = (origin + records).toint()
st_shared_u32((target + Int32(704)).toint(), address.to(Uint32))
st_shared_u32((target + Int32(708)).toint(), (address >> Int64(32)).to(Uint32))
cute.arch.fence_proxy("async.shared", space="cta")""").body
                    return self.generic_visit(node)

            return RemovePayload().visit(tree)

        def expansion_tree(tree):
            # Each stage remembers its own slot's replacement base. Retain
            # B12X's two barriers so no thread overwrites another lane's codes.
            source = ast.unparse(tree)
            start = (
                source.index("    records_address = (")
                if "    records_address = (" in source
                else source.index("    records_address = storage.toint()")
            )
            end = source.index("\n    values = ", start)
            source = (
                source[:start]
                + """    pair = ld_shared_v2_u32((target + Int32(704)).toint())
    records_address = pair[0].to(Int64) | (pair[1].to(Int64) << Int64(32))"""
                + source[end:]
            )
            old = "second_values = self._read_shared_operand(second_target, records_address,"
            if old not in source:
                raise RuntimeError("B12X dual scale stage changed")
            source = source.replace(
                old,
                """pair2 = ld_shared_v2_u32((second_target + Int32(704)).toint())
        second_records = pair2[0].to(Int64) | (pair2[1].to(Int64) << Int64(32))
        second_values = self._read_shared_operand(second_target, second_records,""",
            )
            return ast.parse(source)

        if stride:
            stage = _derive(
                reader.InlineNvfp4Reader.stage, local, stage_tree, reader_directory
            )
            expand = _derive(
                reader.InlineNvfp4Reader.expand_shared,
                local,
                expansion_tree,
                reader_directory,
            )
            shared_word = _derive(
                reader.InlineNvfp4Reader.shared_word,
                local,
                lambda t: t,
                reader_directory,
            )
        else:
            # A poorly compressible plane remains native in the same pack.
            # Supply the full operand through the same B12X pipeline barrier.
            @reader.cute.jit
            def stage(
                self,
                storage,
                experts,
                expert,
                row_tile,
                k_tile,
                shared,
                stage,
                barrier,
                bounds=None,
                bound_index=0,
            ):
                lane = reader.Int32(reader.cute.arch.lane_idx())
                if lane == reader.Int32(0):
                    target = reader.cute.recast_ptr(
                        shared, dtype=reader.cutlass.Uint8
                    ) + stage * reader.Int32(1024)
                    origin = reader.cute.recast_ptr(storage, dtype=reader.cutlass.Uint8)
                    offset = (
                        reader.Int64(expert) * reader.Int64(rows * columns)
                        + reader.Int64(row_tile) * reader.Int64(128 * columns)
                        + reader.Int64(k_tile) * reader.Int64(1024)
                    )
                    reader._expect_payload(barrier, reader.Int32(320))
                    reader.cp_async_bulk_g2s_mbar(
                        target.toint(),
                        (origin + offset).toint(),
                        reader.Int32(1024),
                        barrier.toint(),
                    )

            @reader.cute.jit
            def expand(
                self,
                storage,
                experts,
                shared,
                stage,
                threads,
                barrier,
                second_shared=None,
                second_stage=0,
                row_half=-1,
                second_row_half=-1,
            ):
                barrier.arrive_and_wait()
                barrier.arrive_and_wait()

            shared_word = reader.InlineNvfp4Reader.shared_word

        @reader.cute.jit
        def prefetch(self, storage, experts, expert, outer, *, columns):
            return None

        cls = type(
            "Qwen38SlotScaleReader",
            (reader.InlineNvfp4Reader,),
            dict(
                stage=stage,
                expand_shared=expand,
                shared_word=shared_word,
                prefetch_bounds=prefetch,
            ),
        )
        return cls(rows, columns)

    dynamic_ns = dict(vars(dynamic), InlineNvfp4Reader=make_reader)
    call = _derive(
        dynamic.MoEDynamicKernelBackend.__call__, dynamic_ns, lambda t: t, directory
    )
    backend = type(
        "Qwen38DynamicBackend", (dynamic.MoEDynamicKernelBackend,), {"__call__": call}
    )
    silu_ns = dict(vars(silu), MoEDynamicKernelBackend=backend)
    _derive(silu.MoEDynamicKernelSilu.__init__, silu_ns, lambda t: t, directory)
    # Zero-argument super needs the original class cell; use an explicit base.
    source = (
        (directory / "__init__.py")
        .read_text()
        .replace("super().__init__(", "MoEDynamicKernelBackend.__init__(self, ")
    )
    (directory / "__init__.py").write_text(source)
    exec(compile(source, str(directory / "__init__.py"), "exec"), silu_ns)
    cls = type("Qwen38DynamicSilu", (backend,), {"__init__": silu_ns["__init__"]})
    host = dict(vars(impl), _DYNAMIC_KERNEL_CACHE={})
    original_spec = impl._get_activation_kernel_spec
    host["_get_activation_kernel_spec"] = lambda *a, **kw: replace(
        original_spec(*a, **kw), dynamic_kernel_cls=cls
    )
    signature = repr(
        sorted(
            (n, tuple(sorted(s.items())))
            for n, s in specs.items()
            if isinstance(s, dict)
        )
    )

    def compiler_tree(tree):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value == "integration.tp_moe.dynamic"
            ):
                node.value += ".qwen38.bitmap." + signature
        return tree

    _derive(impl._get_dynamic_kernel, host, compiler_tree, directory)
    _derive(impl._launch_dynamic_flat, host, lambda t: t, directory)
    return SimpleNamespace(**host)


def cached_b12x_w4a4_moe(adapter, x, ids, router, tensors):
    """Run using caller-owned scratch and borrowed compressed expert slots."""
    return adapter.run(x, ids, router, tensors)
