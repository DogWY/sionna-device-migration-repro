"""Inventory stateful classes under ``sionna.phy``.

This script is intentionally repository-local and does not require installing
this repository as a package. Run it in the environment that has Sionna
installed, for example:

    python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


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


@dataclass(frozen=True)
class ImportErrorRecord:
    module: str
    error_type: str
    error: str


@dataclass(frozen=True)
class ClassInventoryRecord:
    module: str
    class_name: str
    qualname: str
    source_file: str | None
    top_level_area: str
    base_classes: list[str]
    signature: str | None
    signature_error: str | None
    required_constructor_params: list[str]
    can_construct_without_args: bool
    is_abstract: bool
    is_torch_module: bool
    is_sionna_object: bool
    is_sionna_block: bool
    constructor_accepts_device: bool
    has_device_property: bool
    source_contains: dict[str, bool]
    risk: str
    risk_reasons: list[str]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect classes under sionna.phy for device-migration risk."
    )
    parser.add_argument(
        "--root-package",
        default="sionna.phy",
        help="Package namespace to inspect. Defaults to sionna.phy.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--include-import-errors",
        action="store_true",
        help="Print module import errors in the terminal summary.",
    )
    args = parser.parse_args(argv)

    try:
        payload = inspect_package(args.root_package)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}")
        return 2

    print_summary(payload, include_import_errors=args.include_import_errors)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nWrote JSON report: {args.json_report}")

    return 0


def inspect_package(root_package: str) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    sionna_phy = importlib.import_module("sionna.phy")
    root = importlib.import_module(root_package)

    sionna_object = getattr(sionna_phy, "Object", None)
    sionna_block = getattr(sionna_phy, "Block", None)

    modules, import_errors = _import_modules(root)
    records: dict[tuple[str, str], ClassInventoryRecord] = {}

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            class_module = getattr(cls, "__module__", "")
            if not class_module.startswith(root_package):
                continue

            key = (class_module, getattr(cls, "__qualname__", cls.__name__))
            if key in records:
                continue

            records[key] = inspect_class(
                cls,
                root_package=root_package,
                torch_module=torch.nn.Module,
                sionna_object=sionna_object,
                sionna_block=sionna_block,
            )

    class_records = sorted(
        records.values(),
        key=lambda item: (item.risk, item.module, item.qualname),
    )

    summary = summarize(class_records, import_errors)
    return {
        "root_package": root_package,
        "python": sys.version.replace("\n", " "),
        "summary": summary,
        "classes": [asdict(record) for record in class_records],
        "import_errors": [asdict(error) for error in import_errors],
    }


def _import_modules(root: ModuleType) -> tuple[list[ModuleType], list[ImportErrorRecord]]:
    modules = [root]
    errors: list[ImportErrorRecord] = []

    if not hasattr(root, "__path__"):
        return modules, errors

    for module_info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        module_name = module_info.name
        if "rt" in module_name.split("."):
            continue
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:  # noqa: BLE001 - inventory must continue
            errors.append(
                ImportErrorRecord(
                    module=module_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )

    return modules, errors


def inspect_class(
    cls: type,
    *,
    root_package: str,
    torch_module: type,
    sionna_object: type | None,
    sionna_block: type | None,
) -> ClassInventoryRecord:
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", cls.__name__)
    source_file = inspect.getsourcefile(cls)
    signature, signature_error, required_params, accepts_device = _inspect_signature(cls)
    source = _get_source(cls)
    source_contains = _source_features(source)

    is_torch_module = _safe_issubclass(cls, torch_module)
    is_sionna_object = sionna_object is not None and _safe_issubclass(cls, sionna_object)
    is_sionna_block = sionna_block is not None and _safe_issubclass(cls, sionna_block)
    has_device_property = isinstance(getattr(cls, "device", None), property)
    is_abstract = inspect.isabstract(cls)
    risk, reasons = classify_risk(
        is_abstract=is_abstract,
        is_torch_module=is_torch_module,
        is_sionna_object=is_sionna_object,
        is_sionna_block=is_sionna_block,
        constructor_accepts_device=accepts_device,
        has_device_property=has_device_property,
        source_contains=source_contains,
    )

    return ClassInventoryRecord(
        module=module,
        class_name=cls.__name__,
        qualname=qualname,
        source_file=source_file,
        top_level_area=_top_level_area(module, root_package),
        base_classes=[
            f"{base.__module__}.{base.__qualname__}"
            for base in getattr(cls, "__bases__", ())
        ],
        signature=signature,
        signature_error=signature_error,
        required_constructor_params=required_params,
        can_construct_without_args=not required_params and signature_error is None,
        is_abstract=is_abstract,
        is_torch_module=is_torch_module,
        is_sionna_object=is_sionna_object,
        is_sionna_block=is_sionna_block,
        constructor_accepts_device=accepts_device,
        has_device_property=has_device_property,
        source_contains=source_contains,
        risk=risk,
        risk_reasons=reasons,
    )


def _inspect_signature(cls: type) -> tuple[str | None, str | None, list[str], bool]:
    try:
        signature = inspect.signature(cls)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", [], False

    required: list[str] = []
    accepts_device = False
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if name == "device":
            accepts_device = True
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return str(signature), None, required, accepts_device


def _get_source(cls: type) -> str:
    try:
        return inspect.getsource(cls)
    except Exception:  # noqa: BLE001
        return ""


def _source_features(source: str) -> dict[str, bool]:
    checks = {
        "contains_device_str": "_device_str" in source,
        "contains_register_buffer": "register_buffer" in source,
        "contains_self_device": "self.device" in source,
        "contains_device_kwarg": "device=" in source,
        "contains_torch_tensor_factory": any(
            token in source
            for token in (
                "torch.tensor",
                "torch.zeros",
                "torch.ones",
                "torch.empty",
                "torch.randn",
                "torch.rand",
                "torch.arange",
                "torch.eye",
                "torch.full",
                "torch.as_tensor",
            )
        ),
        "contains_random_factory": any(
            token in source
            for token in (
                "normal(",
                "rand(",
                "complex_normal(",
                "torch_rng",
            )
        ),
    }
    return checks


def classify_risk(
    *,
    is_abstract: bool,
    is_torch_module: bool,
    is_sionna_object: bool,
    is_sionna_block: bool,
    constructor_accepts_device: bool,
    has_device_property: bool,
    source_contains: dict[str, bool],
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if is_abstract:
        reasons.append("abstract class")
        return "P2", reasons

    if is_sionna_block:
        reasons.append("inherits Sionna Block")
    elif is_sionna_object:
        reasons.append("inherits Sionna Object")
    elif is_torch_module:
        reasons.append("inherits torch.nn.Module")

    if constructor_accepts_device:
        reasons.append("constructor accepts device")
    if has_device_property:
        reasons.append("has device property")
    if source_contains["contains_device_str"]:
        reasons.append("source references _device_str")
    if source_contains["contains_self_device"]:
        reasons.append("source uses self.device")
    if source_contains["contains_register_buffer"]:
        reasons.append("source registers buffers")
    if source_contains["contains_device_kwarg"]:
        reasons.append("source passes device=")
    if source_contains["contains_torch_tensor_factory"]:
        reasons.append("source uses torch tensor factories")
    if source_contains["contains_random_factory"]:
        reasons.append("source uses random tensor factories")

    if (
        (is_sionna_object or is_sionna_block)
        and (
            constructor_accepts_device
            or has_device_property
            or source_contains["contains_self_device"]
            or source_contains["contains_register_buffer"]
            or source_contains["contains_device_kwarg"]
            or source_contains["contains_random_factory"]
        )
    ):
        return "P0", reasons

    if is_torch_module or any(source_contains.values()):
        return "P1", reasons

    if not reasons:
        reasons.append("no stateful device features detected")
    return "P2", reasons


def _top_level_area(module_name: str, root_package: str) -> str:
    suffix = module_name.removeprefix(root_package).lstrip(".")
    if not suffix:
        return "core"
    return suffix.split(".", 1)[0]


def _safe_issubclass(cls: type, base: type) -> bool:
    try:
        return issubclass(cls, base)
    except TypeError:
        return False


def summarize(
    records: list[ClassInventoryRecord],
    import_errors: list[ImportErrorRecord],
) -> dict[str, Any]:
    by_risk = Counter(record.risk for record in records)
    by_area = Counter(record.top_level_area for record in records)
    p0_by_area = Counter(
        record.top_level_area for record in records if record.risk == "P0"
    )
    return {
        "total_classes": len(records),
        "import_error_count": len(import_errors),
        "by_risk": dict(sorted(by_risk.items())),
        "by_area": dict(sorted(by_area.items())),
        "p0_by_area": dict(sorted(p0_by_area.items())),
    }


def print_summary(payload: dict[str, Any], *, include_import_errors: bool) -> None:
    summary = payload["summary"]
    print(f"Root package: {payload['root_package']}")
    print(f"Classes: {summary['total_classes']}")
    print(f"Import errors: {summary['import_error_count']}")
    print("Risk counts:")
    for risk, count in summary["by_risk"].items():
        print(f"  {risk}: {count}")
    print("P0 classes by area:")
    for area, count in summary["p0_by_area"].items():
        print(f"  {area}: {count}")

    if include_import_errors and payload["import_errors"]:
        print("\nImport errors:")
        for error in payload["import_errors"]:
            print(f"  {error['module']}: {error['error_type']}: {error['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
