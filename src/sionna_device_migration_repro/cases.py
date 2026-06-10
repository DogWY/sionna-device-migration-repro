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
    categories: tuple[str, ...] = ("phy",)


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


def _categories(*names: str) -> tuple[str, ...]:
    return ("phy", *names)


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


def _build_binary_source(build_device: str | None):
    from sionna.phy.mapping import BinarySource

    return BinarySource(**_device_kwargs(build_device))


def _inputs_source_shape(device: str):
    _ = device
    return ((2, 8),), {}


def _build_constellation_qam(build_device: str | None):
    from sionna.phy.mapping import Constellation

    return Constellation("qam", 2, **_device_kwargs(build_device))


def _inputs_no_args(device: str):
    _ = device
    return (), {}


def _build_mapper_qam(build_device: str | None):
    from sionna.phy.mapping import Mapper

    return Mapper("qam", 2, **_device_kwargs(build_device))


def _inputs_mapper_qam(device: str):
    torch = _torch()
    bits = torch.randint(0, 2, (2, 8), dtype=torch.int64, device=device)
    return (bits,), {}


def _build_demapper_qam(build_device: str | None):
    from sionna.phy.mapping import Demapper

    return Demapper("app", "qam", 2, **_device_kwargs(build_device))


def _inputs_demapper_qam(device: str):
    torch = _torch()
    y = torch.ones((2, 4), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (y, no), {}


def _build_symbol_demapper_qam(build_device: str | None):
    from sionna.phy.mapping import SymbolDemapper

    return SymbolDemapper("qam", 2, **_device_kwargs(build_device))


def _build_llrs2symbol_logits(build_device: str | None):
    from sionna.phy.mapping import LLRs2SymbolLogits

    return LLRs2SymbolLogits(2, **_device_kwargs(build_device))


def _inputs_llrs2symbol_logits(device: str):
    torch = _torch()
    llrs = torch.zeros((2, 4, 2), dtype=torch.float32, device=device)
    return (llrs,), {}


def _build_symbol_logits2llrs(build_device: str | None):
    from sionna.phy.mapping import SymbolLogits2LLRs

    return SymbolLogits2LLRs("app", 2, **_device_kwargs(build_device))


def _inputs_symbol_logits(device: str):
    torch = _torch()
    logits = torch.zeros((2, 4), dtype=torch.float32, device=device)
    return (logits,), {}


def _build_symbol_inds2bits(build_device: str | None):
    from sionna.phy.mapping import SymbolInds2Bits

    return SymbolInds2Bits(2, **_device_kwargs(build_device))


def _inputs_symbol_inds2bits(device: str):
    torch = _torch()
    symbol_ind = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
    return (symbol_ind,), {}


def _build_symbol_logits2moments(build_device: str | None):
    from sionna.phy.mapping import SymbolLogits2Moments

    return SymbolLogits2Moments("qam", 2, **_device_kwargs(build_device))


def _build_pam2qam(build_device: str | None):
    from sionna.phy.mapping import PAM2QAM

    return PAM2QAM(4, **_device_kwargs(build_device))


def _inputs_pam2qam(device: str):
    torch = _torch()
    pam1 = torch.tensor([0, 1], dtype=torch.int64, device=device)
    pam2 = torch.tensor([2, 3], dtype=torch.int64, device=device)
    return (pam1, pam2), {}


def _build_qam2pam(build_device: str | None):
    from sionna.phy.mapping import QAM2PAM

    return QAM2PAM(4, **_device_kwargs(build_device))


def _inputs_qam2pam(device: str):
    torch = _torch()
    ind_qam = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
    return (ind_qam,), {}


def _build_pam_source(build_device: str | None):
    from sionna.phy.mapping import PAMSource

    return PAMSource(2, **_device_kwargs(build_device))


def _build_qam_source(build_device: str | None):
    from sionna.phy.mapping import QAMSource

    return QAMSource(2, **_device_kwargs(build_device))


def _build_symbol_source(build_device: str | None):
    from sionna.phy.mapping import SymbolSource

    return SymbolSource("qam", 2, **_device_kwargs(build_device))


def _build_upsampling(build_device: str | None):
    from sionna.phy.signal import Upsampling

    return Upsampling(2, **_device_kwargs(build_device))


def _inputs_upsampling(device: str):
    torch = _torch()
    x = torch.ones((2, 4), dtype=torch.float32, device=device)
    return (x,), {}


def _build_downsampling(build_device: str | None):
    from sionna.phy.signal import Downsampling

    return Downsampling(2, **_device_kwargs(build_device))


def _inputs_signal_vector(device: str):
    torch = _torch()
    x = torch.ones((2, 8), dtype=torch.float32, device=device)
    return (x,), {}


def _build_window_base(build_device: str | None):
    from sionna.phy.signal import Window

    return Window(**_device_kwargs(build_device))


def _build_custom_window(build_device: str | None):
    import torch
    from sionna.phy.signal import CustomWindow

    coefficients = torch.ones(8, dtype=torch.float32)
    return CustomWindow(coefficients, **_device_kwargs(build_device))


def _build_hamming_window(build_device: str | None):
    from sionna.phy.signal import HammingWindow

    return HammingWindow(**_device_kwargs(build_device))


def _build_hann_window(build_device: str | None):
    from sionna.phy.signal import HannWindow

    return HannWindow(**_device_kwargs(build_device))


def _build_blackman_window(build_device: str | None):
    from sionna.phy.signal import BlackmanWindow

    return BlackmanWindow(**_device_kwargs(build_device))


def _build_filter_base(build_device: str | None):
    from sionna.phy.signal import Filter

    return Filter(4, 2, **_device_kwargs(build_device))


def _build_custom_filter(build_device: str | None):
    import torch
    from sionna.phy.signal import CustomFilter

    coefficients = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
    return CustomFilter(2, coefficients, **_device_kwargs(build_device))


def _build_sinc_filter(build_device: str | None):
    from sionna.phy.signal import SincFilter

    return SincFilter(4, 2, **_device_kwargs(build_device))


def _build_raised_cosine_filter(build_device: str | None):
    from sionna.phy.signal import RaisedCosineFilter

    return RaisedCosineFilter(4, 2, 0.25, **_device_kwargs(build_device))


def _build_root_raised_cosine_filter(build_device: str | None):
    from sionna.phy.signal import RootRaisedCosineFilter

    return RootRaisedCosineFilter(4, 2, 0.25, **_device_kwargs(build_device))


def _inputs_filter(device: str):
    torch = _torch()
    x = torch.ones((2, 8), dtype=torch.float32, device=device)
    return (x,), {"padding": "same"}


_CASES = (
    CaseSpec(
        name="awgn",
        description="AWGN channel; exposes stale Sionna logical device even without registered tensors.",
        build=_build_awgn,
        make_inputs=_inputs_awgn,
        categories=_categories("channel", "noise"),
    ),
    CaseSpec(
        name="wrapped-awgn-channel",
        description=(
            "User-style nn.Module wrapper around Sionna AWGN; reproduces mixed-device "
            "forward failures after wrapper.to(cuda:x)."
        ),
        build=_build_wrapped_awgn_channel,
        make_inputs=_inputs_wrapped_awgn_channel,
        categories=_categories("channel", "wrapper", "noise"),
    ),
    CaseSpec(
        name="generate-flat-fading",
        description="GenerateFlatFadingChannel; random flat-fading matrix generator.",
        build=_build_generate_flat_fading_channel,
        make_inputs=_inputs_generate_flat_fading_channel,
        categories=_categories("channel", "fading"),
    ),
    CaseSpec(
        name="apply-flat-fading",
        description="ApplyFlatFadingChannel; contains an internal AWGN child block.",
        build=_build_apply_flat_fading_channel,
        make_inputs=_inputs_apply_flat_fading_channel,
        categories=_categories("channel", "fading", "apply", "noise"),
    ),
    CaseSpec(
        name="apply-ofdm",
        description="ApplyOFDMChannel; frequency-domain channel application with an internal AWGN child block.",
        build=_build_apply_ofdm_channel,
        make_inputs=_inputs_apply_ofdm_channel,
        categories=_categories("channel", "ofdm", "apply", "noise"),
    ),
    CaseSpec(
        name="apply-time",
        description="ApplyTimeChannel; time-domain channel application with gather buffer and AWGN child block.",
        build=_build_apply_time_channel,
        make_inputs=_inputs_apply_time_channel,
        categories=_categories("channel", "time", "apply", "noise"),
    ),
    CaseSpec(
        name="flat-fading",
        description="FlatFadingChannel; combines generator, applier, and AWGN child blocks.",
        build=_build_flat_fading_channel,
        make_inputs=_inputs_flat_fading_channel,
        categories=_categories("channel", "fading", "noise"),
    ),
    CaseSpec(
        name="kronecker-flat-fading",
        description="FlatFadingChannel with KroneckerModel buffers in the spatial correlation path.",
        build=_build_kronecker_flat_fading_channel,
        make_inputs=_inputs_flat_fading_channel,
        categories=_categories("channel", "fading", "correlation", "noise"),
    ),
    CaseSpec(
        name="kronecker-model",
        description="KroneckerModel; spatial correlation object with registered correlation buffers.",
        build=_build_kronecker_model,
        make_inputs=_inputs_kronecker_model,
        categories=_categories("channel", "correlation"),
    ),
    CaseSpec(
        name="per-column-model",
        description="PerColumnModel; directly audits registered correlation buffers after .to(device).",
        build=_build_per_column_model,
        make_inputs=_inputs_per_column_model,
        categories=_categories("channel", "correlation"),
    ),
    CaseSpec(
        name="binary-memoryless",
        description="BinaryMemorylessChannel; discrete asymmetric binary channel.",
        build=_build_binary_memoryless_channel,
        make_inputs=_inputs_binary_memoryless_channel,
        categories=_categories("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-symmetric",
        description="BinarySymmetricChannel; discrete bit-flip channel.",
        build=_build_binary_symmetric_channel,
        make_inputs=_inputs_binary_channel,
        categories=_categories("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-erasure",
        description="BinaryErasureChannel; discrete erasure channel.",
        build=_build_binary_erasure_channel,
        make_inputs=_inputs_binary_channel,
        categories=_categories("channel", "discrete"),
    ),
    CaseSpec(
        name="binary-z",
        description="BinaryZChannel; discrete Z-channel.",
        build=_build_binary_z_channel,
        make_inputs=_inputs_binary_channel,
        categories=_categories("channel", "discrete"),
    ),
    CaseSpec(
        name="rayleigh-block-fading",
        description="RayleighBlockFading; stochastic channel impulse response model.",
        build=_build_rayleigh_block_fading,
        make_inputs=_inputs_rayleigh_block_fading,
        categories=_categories("channel", "fading", "model"),
    ),
    CaseSpec(
        name="edfa",
        description="EDFA optical channel block; includes registered buffers and derived tensor state.",
        build=_build_edfa,
        make_inputs=_inputs_edfa,
        categories=_categories("channel", "optical"),
    ),
    CaseSpec(
        name="ssfm",
        description="SSFM optical fiber channel block; includes registered buffers and derived tensor state.",
        build=_build_ssfm,
        make_inputs=_inputs_ssfm,
        categories=_categories("channel", "optical"),
    ),
    CaseSpec(
        name="binary-source",
        description="BinarySource; random bit source using Sionna logical device state.",
        build=_build_binary_source,
        make_inputs=_inputs_source_shape,
        categories=_categories("mapping", "source"),
    ),
    CaseSpec(
        name="constellation-qam",
        description="Constellation(QAM); exposes mapping lookup state and generated points.",
        build=_build_constellation_qam,
        make_inputs=_inputs_no_args,
        categories=_categories("mapping", "constellation"),
    ),
    CaseSpec(
        name="mapper-qam",
        description="Mapper(QAM); bit-to-symbol mapper with internal constellation state.",
        build=_build_mapper_qam,
        make_inputs=_inputs_mapper_qam,
        categories=_categories("mapping", "constellation"),
    ),
    CaseSpec(
        name="demapper-qam-app",
        description="Demapper(APP, QAM); symbol-to-bit LLR demapper with constellation state.",
        build=_build_demapper_qam,
        make_inputs=_inputs_demapper_qam,
        categories=_categories("mapping", "constellation"),
    ),
    CaseSpec(
        name="symbol-demapper-qam",
        description="SymbolDemapper(QAM); symbol posterior demapper with constellation state.",
        build=_build_symbol_demapper_qam,
        make_inputs=_inputs_demapper_qam,
        categories=_categories("mapping", "constellation"),
    ),
    CaseSpec(
        name="llrs2symbol-logits",
        description="LLRs2SymbolLogits; converts bit LLRs into symbol logits.",
        build=_build_llrs2symbol_logits,
        make_inputs=_inputs_llrs2symbol_logits,
        categories=_categories("mapping", "logits"),
    ),
    CaseSpec(
        name="symbol-logits2llrs",
        description="SymbolLogits2LLRs; converts symbol logits into bit LLRs.",
        build=_build_symbol_logits2llrs,
        make_inputs=_inputs_symbol_logits,
        categories=_categories("mapping", "logits"),
    ),
    CaseSpec(
        name="symbol-inds2bits",
        description="SymbolInds2Bits; converts symbol indices into bit labels.",
        build=_build_symbol_inds2bits,
        make_inputs=_inputs_symbol_inds2bits,
        categories=_categories("mapping", "indexing"),
    ),
    CaseSpec(
        name="symbol-logits2moments",
        description="SymbolLogits2Moments(QAM); computes symbol moments from logits.",
        build=_build_symbol_logits2moments,
        make_inputs=_inputs_symbol_logits,
        categories=_categories("mapping", "logits", "constellation"),
    ),
    CaseSpec(
        name="pam2qam",
        description="PAM2QAM; converts two PAM indices into one QAM index.",
        build=_build_pam2qam,
        make_inputs=_inputs_pam2qam,
        categories=_categories("mapping", "indexing"),
    ),
    CaseSpec(
        name="qam2pam",
        description="QAM2PAM; converts QAM indices into two PAM indices.",
        build=_build_qam2pam,
        make_inputs=_inputs_qam2pam,
        categories=_categories("mapping", "indexing"),
    ),
    CaseSpec(
        name="pam-source",
        description="PAMSource; random PAM symbol source built from mapping primitives.",
        build=_build_pam_source,
        make_inputs=_inputs_source_shape,
        categories=_categories("mapping", "source"),
    ),
    CaseSpec(
        name="qam-source",
        description="QAMSource; random QAM symbol source built from mapping primitives.",
        build=_build_qam_source,
        make_inputs=_inputs_source_shape,
        categories=_categories("mapping", "source"),
    ),
    CaseSpec(
        name="symbol-source",
        description="SymbolSource(QAM); random constellation symbol source.",
        build=_build_symbol_source,
        make_inputs=_inputs_source_shape,
        categories=_categories("mapping", "source", "constellation"),
    ),
    CaseSpec(
        name="upsampling",
        description="Upsampling; inserts zeros according to samples-per-symbol state.",
        build=_build_upsampling,
        make_inputs=_inputs_upsampling,
        categories=_categories("signal", "resampling"),
    ),
    CaseSpec(
        name="downsampling",
        description="Downsampling; keeps samples according to samples-per-symbol state.",
        build=_build_downsampling,
        make_inputs=_inputs_signal_vector,
        categories=_categories("signal", "resampling"),
    ),
    CaseSpec(
        name="window-base",
        description="Window base class; audit-only logical device state check.",
        build=_build_window_base,
        make_inputs=None,
        categories=_categories("signal", "window", "audit-only"),
    ),
    CaseSpec(
        name="custom-window",
        description="CustomWindow; user-provided window coefficients should migrate with the module.",
        build=_build_custom_window,
        make_inputs=_inputs_signal_vector,
        categories=_categories("signal", "window"),
    ),
    CaseSpec(
        name="hamming-window",
        description="HammingWindow; generated window coefficients depend on Sionna device state.",
        build=_build_hamming_window,
        make_inputs=_inputs_signal_vector,
        categories=_categories("signal", "window"),
    ),
    CaseSpec(
        name="hann-window",
        description="HannWindow; generated window coefficients depend on Sionna device state.",
        build=_build_hann_window,
        make_inputs=_inputs_signal_vector,
        categories=_categories("signal", "window"),
    ),
    CaseSpec(
        name="blackman-window",
        description="BlackmanWindow; generated window coefficients depend on Sionna device state.",
        build=_build_blackman_window,
        make_inputs=_inputs_signal_vector,
        categories=_categories("signal", "window"),
    ),
    CaseSpec(
        name="filter-base",
        description="Filter base class; audit-only logical device state check.",
        build=_build_filter_base,
        make_inputs=None,
        categories=_categories("signal", "filter", "audit-only"),
    ),
    CaseSpec(
        name="custom-filter",
        description="CustomFilter; user-provided filter coefficients should migrate with the module.",
        build=_build_custom_filter,
        make_inputs=_inputs_filter,
        categories=_categories("signal", "filter"),
    ),
    CaseSpec(
        name="sinc-filter",
        description="SincFilter; generated filter coefficients depend on Sionna device state.",
        build=_build_sinc_filter,
        make_inputs=_inputs_filter,
        categories=_categories("signal", "filter"),
    ),
    CaseSpec(
        name="raised-cosine-filter",
        description="RaisedCosineFilter; generated filter coefficients depend on Sionna device state.",
        build=_build_raised_cosine_filter,
        make_inputs=_inputs_filter,
        categories=_categories("signal", "filter"),
    ),
    CaseSpec(
        name="root-raised-cosine-filter",
        description="RootRaisedCosineFilter; generated filter coefficients depend on Sionna device state.",
        build=_build_root_raised_cosine_filter,
        make_inputs=_inputs_filter,
        categories=_categories("signal", "filter"),
    ),
)
