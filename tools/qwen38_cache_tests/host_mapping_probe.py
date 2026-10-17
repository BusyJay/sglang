#!/usr/bin/env python3
"""Small mapped-host byte parity/hugepage probe; never changes system knobs."""
import argparse
import json
import mmap
import time
from pathlib import Path


def backing(address):
    selected = False
    result = {}
    with open('/proc/self/smaps') as f:
        for line in f:
            first = line.split()[0]
            if '-' in first and ':' not in first:
                lo, hi = (int(x, 16) for x in first.split('-'))
                selected = lo <= address < hi
            elif selected and ':' in line:
                key, value = line.split(':', 1)
                if key in ('Size', 'Rss', 'AnonHugePages', 'Private_Hugetlb',
                           'Shared_Hugetlb', 'KernelPageSize', 'MMUPageSize'):
                    result[key + '_bytes'] = int(value.split()[0]) * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['base', 'thp', 'hugetlb'], required=True)
    parser.add_argument('--mib', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.mib <= 1024:
        parser.error('--mib must be in [2,1024]')
    import torch
    import triton
    import triton.language as tl
    from sglang.kernels.ops.moe.qwen38_expert_cache import _module
    from sglang.srt.layers.moe.qwen38_observation import snapshot

    @triton.jit
    def gather(pointer, indices, output, n: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        offset = tl.load(indices + i, i < n, 0)
        source = pointer.to(tl.pointer_type(tl.uint8))
        value = tl.load(source + offset, i < n, 0)
        tl.store(output + i, value, i < n)

    torch.cuda.init()
    module = _module()  # compile outside registration timing
    nbytes = ((args.mib << 20) + (2 << 20)-1) // (2 << 20) * (2 << 20)
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    if args.mode == 'hugetlb':
        flags |= 0x40000 | (21 << 26)
    result = dict(mode=args.mode, bytes=nbytes, gpu=torch.cuda.get_device_name(),
                  success=False, hugepage_path_confirmed=False)
    mapping = None
    tensor = None
    registered = False
    try:
        mapping = mmap.mmap(-1, nbytes, flags=flags,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        if args.mode == 'thp':
            mapping.madvise(mmap.MADV_HUGEPAGE)
        elif args.mode == 'base':
            mapping.madvise(mmap.MADV_NOHUGEPAGE)
        # Materialize every page, with a distinctive value at page boundaries.
        tensor = torch.frombuffer(mapping, dtype=torch.uint8)
        tensor.zero_()
        offsets = torch.arange(0, nbytes, 4096, dtype=torch.int64)
        values = ((offsets // 4096) * 37 % 251).to(torch.uint8)
        tensor[offsets] = values
        result['before_backing'] = backing(tensor.data_ptr())
        result['before_memory'] = snapshot(torch.device('cuda'))
        started = time.perf_counter()
        address = module.mailbox_register(tensor)
        registered = True
        result['register_seconds'] = time.perf_counter() - started
        result['after_memory'] = snapshot(torch.device('cuda'))
        result['after_backing'] = backing(tensor.data_ptr())
        # Random order plus nonboundary bytes test row-offset interpretation.
        torch.manual_seed(42)
        indices = torch.cat((offsets, offsets + 159, offsets + 4095))
        indices = indices[torch.randperm(indices.numel())]
        expected = tensor[indices].clone()
        ids = indices.cuda()
        output = torch.empty(ids.numel(), dtype=torch.uint8, device='cuda')
        gather[(triton.cdiv(ids.numel(), 256),)](address, ids, output, ids.numel(), 256)
        actual = output.cpu()
        result['byte_parity'] = torch.equal(actual, expected)
        observed = result['after_backing']
        huge = sum(observed.get(k + '_bytes', 0) for k in
                   ('AnonHugePages', 'Private_Hugetlb', 'Shared_Hugetlb'))
        result['huge_backed_bytes'] = huge
        result['hugepage_path_confirmed'] = result['byte_parity'] and huge >= nbytes * .95
        result['success'] = result['byte_parity']
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if registered:
            torch.cuda.synchronize()
            module.mailbox_unregister(tensor)
        tensor = None
        if mapping is not None:
            mapping.close()
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
