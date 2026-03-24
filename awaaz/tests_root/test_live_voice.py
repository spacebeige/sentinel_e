#!/usr/bin/env python3
"""Compatibility launcher for the integrated AWAAZ live voice test.

Keeps the existing root command working:
    /path/to/.venv/bin/python test_live_voice.py

Actual implementation lives in:
    awaaz/test_live_voice.py
"""

import os
import runpy
import sys


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(repo_root, "awaaz", "test_live_voice.py")
    if not os.path.exists(target):
        raise FileNotFoundError(f"Integrated test script not found: {target}")

    target_dir = os.path.dirname(target)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    sys.argv[0] = target
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
