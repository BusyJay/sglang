# SPDX-License-Identifier: Apache-2.0
"""Guarded native FC1 staging; route positions and FC2 accumulation stay intact."""

import ast
import copy
import hashlib
import inspect
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace


def masked_fc1_method(source):
    tree = ast.parse(textwrap.dedent(source))
    changed = 0
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.While)
            or ast.unparse(node.test) != "fc1_task < fc1_task_count"
        ):
            continue
        split = next(
            (
                i + 1
                for i, stmt in enumerate(node.body)
                if isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "_resolve_route_expert"
            ),
            None,
        )
        if split is None or ast.unparse(node.body[-1]) != "fc1_task += Int32(gdim_x)":
            raise RuntimeError("B12X FC1 task definition changed")
        # Every thread in this CTA has the same route predicate. Unselected
        # tasks leave intermediate untouched; the second FC1 fills its complement.
        guarded = ast.If(
            test=ast.parse("route_active > Int32(0)", mode="eval").body,
            body=node.body[split:-1],
            orelse=[],
        )
        node.body[split:-1] = [guarded]
        changed += 1

    class RuntimeBound(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "_resolve_route_expert"
            ):
                if len(node.args) != 1:
                    raise RuntimeError("B12X FC1 route bound ABI changed")
                node.args.append(ast.Name(id="route_expert_limit", ctx=ast.Load()))
            return node

    tree = RuntimeBound().visit(tree)
    if changed != 1:
        raise RuntimeError("cannot find unique native FC1 task loop")
    return ast.fix_missing_locations(tree)


def phase_kernel_namespace(native, phase):
    if phase not in (1, 2):
        raise ValueError("expected FC1 or FC2 phase")
    from b12x.moe._shared.kernels import micro

    cls = native.MoEMicroKernelW4A16SmallMDirect
    overrides = {}
    generated = Path(tempfile.mkdtemp(prefix="qwen38-b12x-phase-"))
    if phase == 1:
        method = cls.kernel
        tree = masked_fc1_method(inspect.getsource(method))
        ns = dict(vars(micro), **method.__globals__)
        path = generated / "kernel.py"
        path.write_text(ast.unparse(tree) + "\n")
        exec(compile(path.read_text(), str(path), "exec"), ns)
        overrides["kernel"] = ns["kernel"]
        route_tree = ast.parse("""@cute.jit
def _resolve_route_expert(self, route_eid: Int64, route_expert_limit: Int32):
    eid, active = Int32(0), Int32(0)
    if route_eid >= Int64(0) and route_eid < Int64(route_expert_limit):
        eid, active = Int32(route_eid), Int32(1)
    return eid, active
""")
        route_path = generated / "_resolve_route_expert.py"
        route_path.write_text(ast.unparse(route_tree) + "\n")
        exec(compile(route_path.read_text(), str(route_path), "exec"), ns)
        overrides["_resolve_route_expert"] = ns["_resolve_route_expert"]

    if phase == 2:
        # The installed public FC2 endpoint accepts contiguous BF16. We retain
        # FC1's native pair-swizzled FP16 bits, with the original arithmetic.
        class InternalIntermediate(ast.NodeTransformer):
            def visit_Compare(self, node):
                if ast.unparse(node) == "self.compile_time_phase == 2":
                    return ast.Constant(False)
                return self.generic_visit(node)

        for name in (
            "_fp4_dot4_for_math",
            "_m2_fc2_rowquad_narrow",
            "_m2_fc2_rowpair_narrow",
        ):
            method = getattr(cls, name)
            tree = InternalIntermediate().visit(
                ast.parse(textwrap.dedent(inspect.getsource(method)))
            )
            ns = dict(vars(micro), **method.__globals__)
            path = generated / (name + ".py")
            path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
            exec(compile(path.read_text(), str(path), "exec"), ns)
            overrides[name] = ns[name]

    def configure(self, *args, **kwargs):
        cls.configure(self, *args, **kwargs)
        cfg = self._cfg
        if (cfg.k_dim, cfg.n) != (2560, 640) or cfg.k_segments == 2:
            raise ValueError("FC1 overlap requires validated Qwen38 geometry")

    overrides["configure"] = configure
    derived = type(f"Qwen38SmallMPhase{phase}", (cls,), overrides)
    namespace = dict(vars(native))
    namespace["MoEMicroKernelW4A16SmallMDirect"] = derived
    namespace["_SMALL_M_DIRECT_CACHE"] = {}
    # Runtime route bounds protect actual allocations. Expert count only sizes
    # fake tensors; share compiled phases across adaptive capacity changes.
    namespace["_qwen38_capacity_bucket"] = True
    tree = (
        copy.deepcopy(native._qwen38_compile_tree)
        if hasattr(native, "_qwen38_compile_tree")
        else ast.parse(
            textwrap.dedent(inspect.getsource(native._compile_w4a16_small_m_direct))
        )
    )
    count = 0
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MoEMicroKernelW4A16SmallMDirect"
        ):
            node.keywords.append(
                ast.keyword(arg="compile_time_phase", value=ast.Constant(phase))
            )
            count += 1
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(
                ("w4a16_small_m_direct", "moe.w4a16.small_m_direct")
            )
        ):
            node.value += f".qwen38_phase{phase}.{digest}"
    if count != 1:
        raise RuntimeError("B12X direct constructor changed")
    exec(
        compile(
            ast.fix_missing_locations(tree),
            inspect.getsourcefile(native._compile_w4a16_small_m_direct),
            "exec",
        ),
        namespace,
    )
    namespace["_qwen38_compile_tree"] = tree
    namespace["_small_m_direct_host_barrier_reset_enabled"] = lambda: False
    return SimpleNamespace(**namespace)


class NativeFc1Stage:
    """Two reusable graphs; no input activation references survive a call.

    The mailbox owns plan/fill ordering. Its snapshot writes ``early`` before
    plan mutation; CPU placeholders remain in late FC1, preserving initialized
    intermediates even for safe-mask zero-weight routes.
    """

    def __init__(self, adapter, tokens, topk, tensors, weights, resident_alphas=None):
        import torch
        from sglang.kernels.ops.moe.qwen38_route_alphas import (
            prepare_fc1_routes,
            refresh_route_alphas,
        )
        from sglang.srt.layers.quantization.qwen38_cached_b12x import (
            _PreparedSmallMLaunch,
        )

        self.adapter = adapter
        self.x = torch.zeros(
            (tokens, adapter.hidden), dtype=torch.bfloat16, device=weights.device
        )
        self.remapped = torch.zeros_like(weights, dtype=torch.int32)
        self.early = torch.full_like(self.remapped, -1)
        self.late = torch.empty_like(self.remapped)
        self.arena = torch.empty_like(self.remapped)
        self.output = torch.empty_like(self.x, dtype=torch.float32)
        capacity = tensors["w13_weight"].shape[0]
        g1, g2 = adapter.globals[:, :capacity].unbind()
        self.fc1 = _PreparedSmallMLaunch(
            adapter.fc1_namespace,
            adapter.workspace,
            adapter.intermediate,
            torch.float32,
        )
        self.fc2 = _PreparedSmallMLaunch(
            adapter.fc2_namespace,
            adapter.workspace,
            adapter.intermediate,
            torch.float32,
        )

        warm_resident = (
            (torch.zeros_like(resident_alphas[0]), *resident_alphas[1:])
            if resident_alphas is not None
            else None
        )

        def begin(warmup=False):
            prepare_fc1_routes(
                self.early,
                self.remapped,
                self.late,
                tensors["g1_alphas"],
                tensors["g2_alphas"],
                tensors["w13_input_scale_quant"],
                tensors["w2_input_scale_quant"],
                g1,
                g2,
                resident_alphas=warm_resident if warmup else resident_alphas,
            )
            self.fc1.run(
                self.x,
                self.early,
                weights,
                tensors,
                g1,
                g2,
                self.output,
                torch.cuda.current_stream(),
            )

        def finish(warmup=False):
            if resident_alphas is None or warmup:
                refresh_route_alphas(
                    self.remapped,
                    tensors["g1_alphas"],
                    tensors["g2_alphas"],
                    tensors["w13_input_scale_quant"],
                    tensors["w2_input_scale_quant"],
                    g1,
                    g2,
                )
            self.fc1.run(
                self.x,
                self.late,
                weights,
                tensors,
                g1,
                g2,
                self.output,
                torch.cuda.current_stream(),
            )
            self.fc2.run(
                self.x,
                self.remapped,
                weights,
                tensors,
                g1,
                g2,
                self.output,
                torch.cuda.current_stream(),
            )

        # Precompile outside capture. Closures are local and discarded: graph
        # objects retain addresses, not cache tensor views or source activations.
        # A new binding can see a previous plan with obsolete slot indices.
        # Warm the resident kernel with an empty plan, before capture can
        # trigger its first device load. Never scatter a stale previous plan.
        begin(warmup=True)
        finish(warmup=True)
        self.begin_graph, self.finish_graph = (
            torch.cuda.CUDAGraph(),
            torch.cuda.CUDAGraph(),
        )
        # Capture after the current stream's warmup work. The graph
        # context manager synchronizes the entire device and empties allocator
        # caches, which would stall serving when a new binding first appears.
        # Stream ordering already protects initialization and prior readers.
        current = torch.cuda.current_stream(self.x.device)
        capture_stream = torch.cuda.Stream(device=self.x.device)
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream):
            for graph, launch in (
                (self.begin_graph, begin),
                (self.finish_graph, finish),
            ):
                # Other serving threads may retire samples on unrelated
                # streams. Keep this thread's capture checks without rejecting
                # their independent CUDA calls under global capture mode.
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    launch()
                except BaseException:
                    try:
                        graph.capture_end()
                    except Exception:
                        pass
                    raise
                else:
                    graph.capture_end()
        current.wait_stream(capture_stream)

    def begin(self, x):
        import torch

        stream = torch.cuda.current_stream(self.x.device)
        if self.adapter.used:
            stream.wait_event(self.adapter.done)
        self.x.copy_(x)
        self.begin_graph.replay()

    def finish(self):
        import torch

        try:
            self.finish_graph.replay()
            return self.output
        finally:
            self.adapter.done.record(torch.cuda.current_stream(self.x.device))
            self.adapter.used = True
