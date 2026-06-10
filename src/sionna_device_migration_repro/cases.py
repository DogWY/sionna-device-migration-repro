"""Sionna repro case registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


InputFactory = Callable[[str], tuple[tuple[Any, ...], dict[str, Any]]]
BuildFactory = Callable[[str | None], Any]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    build: BuildFactory
    make_inputs: InputFactory | None = None
    categories: tuple[str, ...] = ("channel",)


def iter_cases(category: str | None = None) -> tuple[CaseSpec, ...]:
    if category is None:
        return _CASES
    return tuple(case for case in _CASES if category in case.categories)


def iter_categories() -> tuple[str, ...]:
    return tuple(sorted({category for case in _CASES for category in case.categories}))


def get_cases_by_category(category: str) -> tuple[CaseSpec, ...]:
    cases = iter_cases(category)
    if not cases:
        categories = ", ".join(iter_categories())
        raise KeyError(f"Unknown category {category!r}. Available categories: {categories}")
    return cases


def get_case(name: str) -> CaseSpec:
    for case in _CASES:
        if case.name == name:
            return case
    names = ", ".join(case.name for case in _CASES)
    raise KeyError(f"Unknown case {name!r}. Available cases: {names}")


def _torch():
    import torch

    return torch


def _device_kwargs(build_device: str | None) -> dict[str, str]:
    return {} if build_device is None else {"device": build_device}


def _build_awgn(build_device: str | None):
    from sionna.phy.channel import AWGN

    return AWGN(**_device_kwargs(build_device))


def _inputs_awgn(device: str):
    torch = _torch()
    x = torch.ones((4, 8), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, no), {}


def _build_generate_flat_fading_channel(build_device: str | None):
    from sionna.phy.channel import GenerateFlatFadingChannel

    return GenerateFlatFadingChannel(num_tx_ant=2, num_rx_ant=3, **_device_kwargs(build_device))


def _inputs_generate_flat_fading_channel(device: str):
    _ = device
    return (4,), {}


def _build_wrapped_awgn_channel(build_device: str | None):
    from .repros import AWGNChannel

    return AWGNChannel(snr=10, device=build_device)


def _inputs_wrapped_awgn_channel(device: str):
    torch = _torch()
    x = torch.randn(2, 4, 8, dtype=torch.float32, device=device)
    return (x,), {}


def _build_apply_flat_fading_channel(build_device: str | None):
    from sionna.phy.channel import ApplyFlatFadingChannel

    return ApplyFlatFadingChannel(**_device_kwargs(build_device))


def _inputs_apply_flat_fading_channel(device: str):
    torch = _torch()
    x = torch.ones((4, 2), dtype=torch.complex64, device=device)
    h = torch.ones((4, 3, 2), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, h, no), {}


def _build_apply_ofdm_channel(build_device: str | None):
    from sionna.phy.channel import ApplyOFDMChannel

    return ApplyOFDMChannel(**_device_kwargs(build_device))


def _inputs_apply_ofdm_channel(device: str):
    torch = _torch()
    x = torch.ones((2, 1, 2, 3, 4), dtype=torch.complex64, device=device)
    h_freq = torch.ones((2, 1, 2, 1, 2, 3, 4), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, h_freq, no), {}


def _build_apply_time_channel(build_device: str | None):
    from sionna.phy.channel import ApplyTimeChannel

    return ApplyTimeChannel(num_time_samples=8, l_tot=3, **_device_kwargs(build_device))


def _inputs_apply_time_channel(device: str):
    torch = _torch()
    x = torch.ones((2, 1, 2, 8), dtype=torch.complex64, device=device)
    h_time = torch.ones((2, 1, 2, 1, 2, 10, 3), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, h_time, no), {}


def _build_flat_fading_channel(build_device: str | None):
    from sionna.phy.channel import FlatFadingChannel

    return FlatFadingChannel(
        num_tx_ant=2,
        num_rx_ant=3,
        return_channel=True,
        **_device_kwargs(build_device),
    )


def _inputs_flat_fading_channel(device: str):
    torch = _torch()
    x = torch.ones((4, 2), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, no), {}


def _build_kronecker_flat_fading_channel(build_device: str | None):
    import torch
    from sionna.phy.channel import FlatFadingChannel, KroneckerModel

    r_tx = torch.eye(2, dtype=torch.complex64)
    r_rx = torch.eye(3, dtype=torch.complex64)
    spatial_corr = KroneckerModel(r_tx=r_tx, r_rx=r_rx, **_device_kwargs(build_device))
    return FlatFadingChannel(
        num_tx_ant=2,
        num_rx_ant=3,
        spatial_corr=spatial_corr,
        return_channel=True,
        **_device_kwargs(build_device),
    )


def _build_kronecker_model(build_device: str | None):
    import torch
    from sionna.phy.channel import KroneckerModel

    r_tx = torch.eye(2, dtype=torch.complex64)
    r_rx = torch.eye(3, dtype=torch.complex64)
    return KroneckerModel(r_tx=r_tx, r_rx=r_rx, **_device_kwargs(build_device))


def _inputs_kronecker_model(device: str):
    torch = _torch()
    h = torch.ones((4, 3, 2), dtype=torch.complex64, device=device)
    return (h,), {}


def _build_per_column_model(build_device: str | None):
    import torch
    from sionna.phy.channel import PerColumnModel

    r_rx = torch.eye(3, dtype=torch.complex64).repeat(2, 1, 1)
    return PerColumnModel(r_rx=r_rx, **_device_kwargs(build_device))


def _inputs_per_column_model(device: str):
    torch = _torch()
    h = torch.ones((4, 3, 2), dtype=torch.complex64, device=device)
    return (h,), {}


def _build_binary_memoryless_channel(build_device: str | None):
    from sionna.phy.channel import BinaryMemorylessChannel

    return BinaryMemorylessChannel(**_device_kwargs(build_device))


def _inputs_binary_memoryless_channel(device: str):
    torch = _torch()
    x = torch.randint(0, 2, (2, 8), dtype=torch.int64, device=device)
    pb = torch.tensor([[0.1, 0.2]], dtype=torch.float32, device=device)
    return (x, pb), {}


def _build_binary_symmetric_channel(build_device: str | None):
    from sionna.phy.channel import BinarySymmetricChannel

    return BinarySymmetricChannel(**_device_kwargs(build_device))


def _build_binary_erasure_channel(build_device: str | None):
    from sionna.phy.channel import BinaryErasureChannel

    return BinaryErasureChannel(**_device_kwargs(build_device))


def _build_binary_z_channel(build_device: str | None):
    from sionna.phy.channel import BinaryZChannel

    return BinaryZChannel(**_device_kwargs(build_device))


def _inputs_binary_channel(device: str):
    torch = _torch()
    x = torch.randint(0, 2, (2, 8), dtype=torch.int64, device=device)
    pb = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (x, pb), {}


def _build_rayleigh_block_fading(build_device: str | None):
    from sionna.phy.channel import RayleighBlockFading

    return RayleighBlockFading(
        num_rx=1,
        num_rx_ant=2,
        num_tx=1,
        num_tx_ant=2,
        **_device_kwargs(build_device),
    )


def _inputs_rayleigh_block_fading(device: str):
    _ = device
    return (2, 4), {}


def _build_edfa(build_device: str | None):
    from sionna.phy.channel import EDFA

    return EDFA(g=4.0, f=2.0, dt=1.0e-12, **_device_kwargs(build_device))


def _inputs_edfa(device: str):
    torch = _torch()
    x = torch.ones((2, 8), dtype=torch.complex64, device=device)
    return (x,), {}


def _build_ssfm(build_device: str | None):
    from sionna.phy.channel import SSFM

    return SSFM(
        length=1.0,
        n_ssfm=1,
        with_amplification=False,
        with_attenuation=True,
        with_dispersion=False,
        with_nonlinearity=False,
        **_device_kwargs(build_device),
    )


def _inputs_ssfm(device: str):
    torch = _torch()
    x = torch.ones((2, 16), dtype=torch.complex64, device=device)
    return (x,), {}


_CASES = (
    CaseSpec(
        name="awgn",
        description="AWGN channel; exposes stale Sionna logical device even without registered tensors.",
        build=_build_awgn,
        make_inputs=_inputs_awgn,
        categories=("channel", "noise"),
    ),
    CaseSpec(
        name="wrapped-awgn-channel",
        description=(
            "User-style nn.Module wrapper around Sionna AWGN; reproduces mixed-device "
            "forward failures after wrapper.to(cuda:x)."
        ),
        build=_build_wrapped_awgn_channel,
        make_inputs=_inputs_wrapped_awgn_channel,
        categories=("channel", "wrapper", "noise"),
    ),
    CaseSpec(
        name="generate-flat-fading",
        description="GenerateFlatFadingChannel; random flat-fading matrix generator.",
        build=_build_generate_flat_fading_channel,
        make_inputs=_inputs_generate_flat_fading_channel,
        categories=("channel", "fading"),
    ),
    CaseSpec(
        name="apply-flat-fading",
        description="ApplyFlatFadingChannel; contains an internal AWGN child block.",
        build=_build_apply_flat_fading_channel,
        make_inputs=_inputs_apply_flat_fading_channel,
        categories=("channel", "fading", "apply", "noise"),
    ),
    CaseSpec(
        name="apply-ofdm",
        description="ApplyOFDMChannel; frequency-domain channel application with an internal AWGN child block.",
        build=_build_apply_ofdm_channel,
        make_inputs=_inputs_apply_ofdm_channel,
        categories=("channel", "ofdm", "apply", "noise"),
    ),
    CaseSpec(
        name="apply-time",
        description="ApplyTimeChannel; time-domain channel application with gather buffer and AWGN child block.",
        build=_build_apply_time_channel,
        make_inputs=_inputs_apply_time_channel,
        categories=("channel", "time", "apply", "noise"),
    ),
    CaseSpec(
        name="flat-fading",
        description="FlatFadingChannel; combines generator, applier, and AWGN child blocks.",
        build=_build_flat_fading_channel,
        make_inputs=_inputs_flat_fading_channel,
        categories=("channel", "fading", "noise"),
    ),
    CaseSpec(
        name="kronecker-flat-fading",
        description="FlatFadingChannel with KroneckerModel buffers in the spatial correlation path.",
        build=_build_kronecker_flat_fading_channel,
        make_inputs=_inputs_flat_fading_channel,
        categories=("channel", "fading", "correlation", "noise"),
    ),
    CaseSpec(
        name="kronecker-model",
        description="KroneckerModel; spatial correlation object with registered correlation buffers.",
        build=_build_kronecker_model,
        make_inputs=_inputs_kronecker_model,
        categories=("channel", "correlation"),
    ),
    CaseSpec(
        name="per-column-model",
        description="PerColumnModel; directly audits registered correlation buffers after .to(device).",
        build=_build_per_column_model,
        make_inputs=_inputs_per_column_model,
        categories=("channel", "correlation"),
    ),
    CaseSpec(
        name="binary-memoryless",
        description="BinaryMemorylessChannel; discrete asymmetric binary channel.",
        build=_build_binary_memoryless_channel,
        make_inputs=_inputs_binary_memoryless_channel,
        categories=("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-symmetric",
        description="BinarySymmetricChannel; discrete bit-flip channel.",
        build=_build_binary_symmetric_channel,
        make_inputs=_inputs_binary_channel,
        categories=("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-erasure",
        description="BinaryErasureChannel; discrete erasure channel.",
        build=_build_binary_erasure_channel,
        make_inputs=_inputs_binary_channel,
        categories=("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-z",
        description="BinaryZChannel; discrete Z-channel.",
        build=_build_binary_z_channel,
        make_inputs=_inputs_binary_channel,
        categories=("channel", "discrete"),
    ),
    CaseSpec(
        name="rayleigh-block-fading",
        description="RayleighBlockFading; stochastic channel impulse response model.",
        build=_build_rayleigh_block_fading,
        make_inputs=_inputs_rayleigh_block_fading,
        categories=("channel", "fading", "model"),
    ),
    CaseSpec(
        name="edfa",
        description="EDFA optical channel block; includes registered buffers and derived tensor state.",
        build=_build_edfa,
        make_inputs=_inputs_edfa,
        categories=("channel", "optical"),
    ),
    CaseSpec(
        name="ssfm",
        description="SSFM optical fiber channel block; includes registered buffers and derived tensor state.",
        build=_build_ssfm,
        make_inputs=_inputs_ssfm,
        categories=("channel", "optical"),
    ),
)
