"""Environment collection for reproducibility reports."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from typing import Any


def collect_env() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "packages": {},
    }

    for package_name in ("torch", "sionna", "sionna-rt", "sionna-no-rt"):
        try:
            data["packages"][package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            data["packages"][package_name] = None

    try:
        import torch

        cuda: dict[str, Any] = {
            "is_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda": torch.version.cuda,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "devices": [],
        }
        for index in range(torch.cuda.device_count()):
            cuda["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
            )
        data["torch"] = {
            "version": torch.__version__,
            "cuda": cuda,
            "mps_is_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except Exception as exc:
        data["torch_error"] = f"{type(exc).__name__}: {exc}"

    return data


def format_env(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
