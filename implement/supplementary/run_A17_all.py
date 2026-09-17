# -*- coding: utf-8 -*-
"""Run the original and strict-inductive configurations for Table A17.

The strict variant is selected in the child process via
``A17_STRICT_MASKING=1``.  Training then hides test-gene edges and features;
the unchanged model is evaluated with the complete graph and frozen features.
"""

import os
from pathlib import Path
import subprocess
import sys


MAIN = Path(__file__).with_name("main_A17.py")
CONFIGURATIONS = (
    ("cpdb", False),
    ("cpdb", True),
    ("string", False),
    ("string", True),
)


def main():
    for dataset, strict_masking in CONFIGURATIONS:
        variant = "masking" if strict_masking else "original"
        print(f"\n===== A17: {dataset.upper()} / {variant} =====", flush=True)
        environment = os.environ.copy()
        environment["A17_DATASET"] = dataset
        environment["A17_STRICT_MASKING"] = "1" if strict_masking else "0"
        subprocess.run(
            [sys.executable, str(MAIN)],
            env=environment,
            cwd=MAIN.parent,
            check=True,
        )


if __name__ == "__main__":
    main()
