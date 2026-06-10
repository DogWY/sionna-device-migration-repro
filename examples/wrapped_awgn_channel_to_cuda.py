"""Minimal wrapper repro for the Sionna AWGN `.to(cuda:x)` issue.

Run after installing the package:

    python examples/wrapped_awgn_channel_to_cuda.py cuda:0
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from sionna_device_migration_repro.repros import AWGNChannel


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    target = torch.device(device)

    if target.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; this repro needs a CUDA device.")
        return 2
    if target.type == "cuda" and target.index is not None and target.index >= torch.cuda.device_count():
        print(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        return 2

    channel = AWGNChannel(snr=10)
    print(f"before .to(): channel.awgn.device={channel.awgn.device!r}")

    channel.to(device)
    print(f"after .to({device!r}): channel.awgn.device={channel.awgn.device!r}")

    x = torch.randn(2, 4, 8, device=device)
    y = channel(x)

    print(f"input device:  {x.device}")
    print(f"output device: {y.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
