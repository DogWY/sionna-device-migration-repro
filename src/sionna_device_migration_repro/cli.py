"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .cases import get_case, get_cases_by_category, iter_cases, iter_categories
from .env import collect_env, format_env


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-cases":
        return _cmd_list_cases()
    if args.command == "env":
        print(format_env(collect_env()))
        return 0
    if args.command == "run":
        return _cmd_run(args)

    parser.error("missing command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_repro.py",
        description="Reproduce Sionna PHY device migration issues after .to(device).",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-cases", help="List built-in Sionna repro cases.")
    subparsers.add_parser("env", help="Print environment information as JSON.")

    run_parser = subparsers.add_parser("run", help="Run one or more repro cases.")
    run_parser.add_argument("--device", default="cuda:0", help="Target device, e.g. cuda:0.")
    run_parser.add_argument(
        "--build-device",
        default="cpu",
        help=(
            "Device used when constructing repro objects before calling .to(device). "
            "Use 'default' to keep Sionna's global default device."
        ),
    )
    run_parser.add_argument(
        "--case",
        default="all",
        help="Case name to run, or 'all'. Use list-cases to inspect available cases.",
    )
    run_parser.add_argument(
        "--category",
        help=(
            "Run all cases in a category when --case=all. "
            f"Available categories: {', '.join(iter_categories())}."
        ),
    )
    run_parser.add_argument(
        "--no-probe-forward",
        action="store_true",
        help="Only audit object state after .to(device); skip forward execution.",
    )
    run_parser.add_argument(
        "--public-only",
        action="store_true",
        help="Do not traverse private non-PyTorch attributes such as _device_str.",
    )
    run_parser.add_argument("--max-depth", type=int, default=8, help="Maximum recursive audit depth.")
    run_parser.add_argument("--json-report", type=Path, help="Optional path for a JSON report.")
    run_parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Return exit code 0 even when a migration issue is found.",
    )

    return parser


def _cmd_list_cases() -> int:
    for case in iter_cases():
        categories = ",".join(case.categories)
        print(f"{case.name} [{categories}]: {case.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _prepare_runtime_cache_dirs()

    try:
        from .runner import run_case, validate_device_available
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print(
                "Missing runtime dependency: torch. "
                "Install this project in an environment with PyTorch and Sionna before running repro cases."
            )
            return 2
        raise

    device = args.device
    build_device = None if args.build_device == "default" else args.build_device
    try:
        validate_device_available(device)
    except RuntimeError as exc:
        print(f"Device validation failed: {exc}")
        return 2
    if build_device is not None:
        try:
            validate_device_available(build_device)
        except RuntimeError as exc:
            print(f"Build-device validation failed: {exc}")
            return 2

    if args.case == "all":
        if args.category:
            try:
                cases = get_cases_by_category(args.category)
            except KeyError as exc:
                print(exc)
                return 2
        else:
            cases = iter_cases()
    else:
        try:
            cases = (get_case(args.case),)
        except KeyError as exc:
            print(exc)
            return 2
        if args.category:
            cases = tuple(case for case in cases if args.category in case.categories)
            if not cases:
                print(f"Case {args.case!r} is not in category {args.category!r}.")
                return 2

    results = [
        run_case(
            case,
            device=device,
            build_device=build_device,
            probe_forward=not args.no_probe_forward,
            include_private=not args.public_only,
            max_depth=args.max_depth,
        )
        for case in cases
    ]

    _print_device_results(device, args.build_device, results)

    if args.json_report:
        _write_json_report(args.json_report, device, args.build_device, results)

    has_failure = any(result.failed for result in results)
    return 1 if has_failure and not args.no_fail else 0


def _print_device_results(device: str, build_device: str, results: Sequence[Any]) -> None:
    print(f"\n# Build device {build_device} -> target device {device}")
    _print_results(results)


def _print_results(results: Sequence[Any]) -> None:
    from .audit import format_issues

    for result in results:
        print(f"\n== {result.name} [{result.status}] ==")
        print(result.description)

        if result.skip_reason:
            print(result.skip_reason)
            continue

        if result.forward_error:
            print(f"forward error: {result.forward_error}")

        print(format_issues(result.issues))

        if result.forward_issues:
            print("Forward output device issue(s):")
            print(format_issues(result.forward_issues))


def _write_json_report(
    path: Path,
    device: str,
    build_device: str,
    results: Sequence[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "build_device": build_device,
        "device": device,
        "env": collect_env(),
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote JSON report: {path}")


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
