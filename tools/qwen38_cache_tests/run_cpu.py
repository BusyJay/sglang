#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import unittest


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        str(root), pattern="test_cpu_*.py", top_level_dir=str(root.parent.parent)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
