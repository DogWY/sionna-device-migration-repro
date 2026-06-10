"""Case execution for Sionna device migration repros."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .audit import DeviceIssue, audit_device_tree
from .cases import CaseSpec


@dataclass
class CaseResult:
    name: str
    status: str
    description: str
    build_device: str
    target_device: str
    issues: list[DeviceIssue]
    forward_issues: list[DeviceIssue]
    forward_error: str | None = None
    skip_reason: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        data["forward_issues"] = [issue.to_dict() for issue in self.forward_issues]
        return data


def validate_device_available(device: str) -> None:
    target = torch.device(device)

    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    elif target.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")


def run_case(
    case: CaseSpec,
    *,
    device: str,
    build_device: str | None = "cpu",
    probe_forward: bool = True,
    include_private: bool = True,
    max_depth: int = 8,
) -> CaseResult:
    build_device_label = "default" if build_device is None else build_device
    try:
        obj = case.build(build_device)
    except Exception as exc:
        return CaseResult(
            name=case.name,
            status="skipped",
            description=case.description,
            build_device=build_device_label,
            target_device=device,
            issues=[],
            forward_issues=[],
            skip_reason=f"build failed: {type(exc).__name__}: {exc}",
        )

    try:
        obj.to(device)
    except Exception as exc:
        return CaseResult(
            name=case.name,
            status="failed",
            description=case.description,
            build_device=build_device_label,
            target_device=device,
            issues=[],
            forward_issues=[],
            forward_error=f".to({device!r}) failed: {type(exc).__name__}: {exc}",
        )

    issues = audit_device_tree(
        obj,
        device,
        include_private=include_private,
        max_depth=max_depth,
    )

    forward_issues: list[DeviceIssue] = []
    forward_error: str | None = None

    if probe_forward and case.make_inputs is not None:
        try:
            args, kwargs = case.make_inputs(device)
            output = obj(*args, **kwargs)
            forward_issues = audit_device_tree(
                output,
                device,
                include_private=True,
                max_depth=max_depth,
            )
        except Exception as exc:
            forward_error = f"{type(exc).__name__}: {exc}"

    status = "failed" if issues or forward_issues or forward_error else "passed"
    return CaseResult(
        name=case.name,
        status=status,
        description=case.description,
        build_device=build_device_label,
        target_device=device,
        issues=issues,
        forward_issues=forward_issues,
        forward_error=forward_error,
    )
