"""Restore complete PyTorch memory snapshots from size-limited, SHA256-verified Git parts."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "profiles.json").read_text())
    for row in manifest:
        data = b"".join((root / part).read_bytes() for part in row["parts"])
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("profile checksum mismatch: " + row["file"])
        (root / row["file"]).write_bytes(data)
        print(root / row["file"])


if __name__ == "__main__":
    main()
