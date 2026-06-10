"""Recursive device audit helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceIssue:
    """A single device mismatch found inside an object tree."""

    path: str
    kind: str
    actual: str
    expected: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_TORCH_MODULE_INTERNALS = {
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
    "training",
}


def audit_device_tree(
    root: Any,
    expected_device: str | torch.device,
    *,
    include_private: bool = True,
    max_depth: int = 8,
) -> list[DeviceIssue]:
    """Find tensors and logical device fields that are not on expected_device.

    The audit intentionally looks beyond registered PyTorch parameters/buffers.
    Sionna objects can keep important device state in ordinary attributes such as
    ``_device_str``; PyTorch's default ``nn.Module.to`` does not know about those.
    """

    expected = torch.device(expected_device)
    issues: list[DeviceIssue] = []
    seen: set[int] = set()

    def add_issue(path: str, kind: str, actual: Any, detail: str) -> None:
        actual_text = str(actual)
        if _device_matches(actual_text, expected):
            return
        issues.append(
            DeviceIssue(
                path=path,
                kind=kind,
                actual=actual_text,
                expected=str(expected),
                detail=detail,
            )
        )

    def walk(value: Any, path: str, depth: int) -> None:
        if depth < 0 or value is None:
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, torch.Tensor):
            add_issue(path, "tensor", value.device, f"dtype={value.dtype}, shape={tuple(value.shape)}")
            return

        if isinstance(value, torch.nn.Module):
            _audit_module(value, path, depth)
            return

        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}", depth - 1)
            return

        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}[{key!r}]", depth - 1)
            return

        if isinstance(value, tuple):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth - 1)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth - 1)
            return

        if hasattr(value, "__dict__") and not isinstance(value, (str, bytes, bytearray)):
            for name, item in vars(value).items():
                if not include_private and name.startswith("_"):
                    continue
                walk(item, f"{path}.{name}", depth - 1)

    def _audit_module(module: torch.nn.Module, path: str, depth: int) -> None:
        logical_device = getattr(module, "_device_str", None)
        if logical_device is not None:
            add_issue(
                f"{path}._device_str",
                "logical-device",
                logical_device,
                "Sionna logical device field",
            )
        elif hasattr(type(module), "device"):
            try:
                add_issue(
                    f"{path}.device",
                    "logical-device",
                    getattr(module, "device"),
                    "module device property",
                )
            except Exception as exc:  # pragma: no cover - defensive only
                issues.append(
                    DeviceIssue(
                        path=f"{path}.device",
                        kind="logical-device-error",
                        actual=type(exc).__name__,
                        expected=str(expected),
                        detail=str(exc),
                    )
                )

        registered_names: set[str] = set()

        for name, parameter in module.named_parameters(recurse=False):
            registered_names.add(name)
            walk(parameter, f"{path}.{name}", depth - 1)

        for name, buffer in module.named_buffers(recurse=False):
            registered_names.add(name)
            walk(buffer, f"{path}.{name}", depth - 1)

        for name, child in module.named_children():
            registered_names.add(name)
            walk(child, f"{path}.{name}", depth - 1)

        for name, item in vars(module).items():
            if name in _TORCH_MODULE_INTERNALS or name in registered_names:
                continue
            if not include_private and name.startswith("_"):
                continue
            walk(item, f"{path}.{name}", depth - 1)

    walk(root, "root", max_depth)
    return issues


def format_issues(issues: Sequence[DeviceIssue]) -> str:
    """Render issues as compact text for CLI output."""

    if not issues:
        return "No device migration issues found."

    lines = [f"Found {len(issues)} device migration issue(s):"]
    for issue in issues:
        lines.append(
            f"- {issue.path}: expected={issue.expected}, actual={issue.actual}, "
            f"kind={issue.kind}, {issue.detail}"
        )
    return "\n".join(lines)


def _device_matches(actual: str, expected: torch.device) -> bool:
    try:
        actual_device = torch.device(actual)
    except (RuntimeError, ValueError):
        return False

    if actual_device.type != expected.type:
        return False
    if expected.index is None:
        return True
    return actual_device.index == expected.index
