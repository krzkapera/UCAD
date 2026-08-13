"""Entry point that makes the released code run on a current Python stack.

Usage: python scripts/launch.py <the arguments run_ucad.py expects>

Two things stand between this repository and a 2026 environment. NumPy 2 removed `np.sctypes` and the
`np.float_`-style aliases that `imgaug` and parts of `patchcore` still reach for, and an unbounded
allocation early in training can take a node down before Slurm notices. Both are handled here rather
than by editing the model code, so `run_ucad.py` stays as close to the published version as possible.

    UCAD_MEM_LIMIT_GB   address-space ceiling for the process (default 12)
"""

import os
import resource
import runpy
import sys
from pathlib import Path

import numpy as np

LIMIT_GB = int(os.environ.get("UCAD_MEM_LIMIT_GB", "12"))
resource.setrlimit(resource.RLIMIT_DATA, (LIMIT_GB * 2**30, LIMIT_GB * 2**30))

np.sctypes = {
    "int": [np.int8, np.int16, np.int32, np.int64],
    "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    "float": [np.float16, np.float32, np.float64],
    "complex": [np.complex64, np.complex128],
    "others": [bool, object, bytes, str, np.void],
}
for alias, dtype in (
    ("bool8", np.bool_), ("float_", np.float64), ("int_", np.int64),
    ("unicode_", np.str_), ("object_", np.object_), ("str_", np.str_),
):
    if not hasattr(np, alias):
        setattr(np, alias, dtype)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.argv = ["run_ucad.py"] + sys.argv[1:]
runpy.run_path(str(ROOT / "run_ucad.py"), run_name="__main__")
