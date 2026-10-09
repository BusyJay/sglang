#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

try:
    from .format import load_manifest
except ImportError:
    from format import load_manifest  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--sglang-commit")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    manifest.assert_runtime_compatible(
        checkpoint_revision=args.checkpoint_revision,
        sglang_commit=args.sglang_commit,
    )
    print(
        json.dumps(
            {
                "valid": True,
                "payload": str(manifest.payload_path),
                "payload_nbytes": manifest.payload_nbytes,
                "num_layers": manifest.num_layers,
                "num_experts": manifest.num_experts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
