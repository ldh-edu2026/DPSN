"""Train Si3N4 with the local custom descriptor on DeepMD-kit 3."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# The host has no usable NVIDIA driver.  Hiding CUDA prevents TensorFlow from
# probing a stale accelerator during neighbour-statistics batch tuning.  Set
# SI3N4_USE_GPU=1 on a correctly configured GPU host to keep CUDA visible.
if os.environ.get("SI3N4_USE_GPU") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DP_INFER_BATCH_SIZE", "1024")
sys.path.insert(0, str(PROJECT_DIR))

# Importing the package registers both the descriptor implementation and its
# DeepMD input-schema plugin before the command-line entry point validates the
# JSON configuration.
import descriptor  # noqa: F401, E402
from deepmd.main import main  # noqa: E402


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input_my.json"
    main(["--tf", "train", input_file, "--output", "out_my_stress.json"])
