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
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    value = manifest.to_dict(manifest_path=args.manifest)
    if args.layer is not None:
        value["layers"] = {str(args.layer): value["layers"][str(args.layer)]}
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
