"""Measure full-sized source preparation/registration and verify sampled bytes.

Run with the service stopped and enough host RAM. The shared_thp cache must be
bound from host tmpfs to survive container removal. No global caches are dropped.
"""

import argparse
import ctypes
import json
import logging
import os
import random
import time
from pathlib import Path


def mapping_stats(mapping):
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    selected = False
    result = {}
    for line in Path("/proc/self/smaps").read_text().splitlines():
        columns = line.split()
        if "-" in columns[0]:
            begin, end = (int(value, 16) for value in columns[0].split("-"))
            selected = begin <= address < end
        elif selected and columns[0].rstrip(":") in (
            "Size",
            "Rss",
            "Anonymous",
            "AnonHugePages",
            "ShmemPmdMapped",
            "Shared_Dirty",
            "Private_Dirty",
        ):
            result[columns[0].rstrip(":") + "_bytes"] = int(columns[1]) * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--mode", choices=("anonymous_thp", "shared_thp"), required=True
    )
    parser.add_argument("--cache-dir", default="/dev/shm/sglang-qwen38-experts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    os.environ["SGLANG_QWEN38_SOURCE_MODE"] = args.mode
    os.environ["SGLANG_QWEN38_SOURCE_CACHE_DIR"] = args.cache_dir

    import torch

    from sglang.kernels.ops.moe.qwen38_expert_cache import _module
    from sglang.srt.model_loader.qwen38_nvfp4_pack.format import (
        Qwen38ExpertPack,
        load_manifest,
    )

    manifest = load_manifest(args.manifest)
    torch.cuda.init()
    module = _module()
    with Qwen38ExpertPack(manifest) as pack:
        started = time.perf_counter()
        mapping = pack.cuda_source_mapping()
        prepare_seconds = time.perf_counter() - started
        view = torch.frombuffer(mapping, dtype=torch.uint8)
        try:
            started = time.perf_counter()
            address = module.mailbox_register(view)
            register_seconds = time.perf_counter() - started
            try:
                # Compare boundaries and repeatable random pages against the
                # immutable source file, including its unaligned final bytes.
                rng = random.Random(20261018)
                offsets = [0, manifest.payload_nbytes - 4096]
                offsets += [
                    rng.randrange(manifest.payload_nbytes - 4096) for _ in range(256)
                ]
                with manifest.payload_path.open("rb", buffering=0) as source:
                    for offset in offsets:
                        expected = os.pread(source.fileno(), 4096, offset)
                        assert mapping[offset : offset + 4096] == expected, offset
                result = dict(
                    mode=args.mode,
                    source_bytes=manifest.payload_nbytes,
                    cache_hit=pack._source_cache_hit,
                    prepare_seconds=prepare_seconds,
                    register_seconds=register_seconds,
                    device_address_valid=address != 0,
                    checked_pages=len(offsets),
                    mapping_stats=mapping_stats(mapping),
                    status="PASS",
                )
                Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
            finally:
                module.mailbox_unregister(view)
        finally:
            del view


if __name__ == "__main__":
    main()
