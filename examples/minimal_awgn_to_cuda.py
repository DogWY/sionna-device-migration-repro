"""Minimal AWGN repro for Sionna device migration.

Run from the repository root after installing the package:

    python examples/minimal_awgn_to_cuda.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_runtime_cache_dirs() -> None:
    cache_root = Path(".cache")
    defaults = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for name, path in defaults.items():
        if name not in os.environ:
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path.resolve())


_prepare_runtime_cache_dirs()

import torch
from sionna.phy.channel import AWGN


def main() -> int:
    device = "cuda:0"
    if not torch.cuda.is_available():
        print("CUDA is not available; this minimal repro needs a CUDA device.")
        return 2

    channel = AWGN()
    print(f"before .to(): channel.device={channel.device!r}")

    channel.to(device)
    print(f"after .to({device!r}): channel.device={channel.device!r}")

    x = torch.ones((4, 8), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    y = channel(x, no)

    print(f"input device:  {x.device}")
    print(f"output device: {y.device}")

    if y.device != torch.device(device):
        print("BUG: output is not on the requested target device.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
