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


def _fec_linear_generator_matrix():
    import numpy as np

    return np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)


def _fec_ldpc_pcm():
    import numpy as np

    return np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.int32)


def _fec_polar_frozen_positions():
    import numpy as np

    return np.array([0, 1], dtype=int)


def _build_fec_gaussian_prior_source(build_device: str | None):
    from sionna.phy.fec import GaussianPriorSource

    return GaussianPriorSource(**_device_kwargs(build_device))


def _build_fec_crc_encoder(build_device: str | None):
    from sionna.phy.fec import CRCEncoder

    return CRCEncoder("CRC6", k=8, **_device_kwargs(build_device))


def _build_fec_crc_decoder(build_device: str | None):
    from sionna.phy.fec import CRCDecoder

    return CRCDecoder(_build_fec_crc_encoder(build_device), **_device_kwargs(build_device))


def _build_fec_trellis(build_device: str | None):
    from sionna.phy.fec.conv.utils import Trellis

    return Trellis(("101", "111"), **_device_kwargs(build_device))


def _build_fec_conv_encoder(build_device: str | None):
    from sionna.phy.fec import ConvEncoder

    return ConvEncoder(**_device_kwargs(build_device))


def _build_fec_viterbi_decoder(build_device: str | None):
    from sionna.phy.fec import ViterbiDecoder

    return ViterbiDecoder(**_device_kwargs(build_device))


def _build_fec_bcjr_decoder(build_device: str | None):
    from sionna.phy.fec import BCJRDecoder

    return BCJRDecoder(**_device_kwargs(build_device))


def _build_fec_row_column_interleaver(build_device: str | None):
    from sionna.phy.fec import RowColumnInterleaver

    return RowColumnInterleaver(4, **_device_kwargs(build_device))


def _build_fec_random_interleaver(build_device: str | None):
    from sionna.phy.fec import RandomInterleaver

    return RandomInterleaver(seed=1, **_device_kwargs(build_device))


def _build_fec_turbo_3gpp_interleaver(build_device: str | None):
    from sionna.phy.fec import Turbo3GPPInterleaver

    return Turbo3GPPInterleaver(**_device_kwargs(build_device))


def _build_fec_deinterleaver(build_device: str | None):
    from sionna.phy.fec import Deinterleaver

    return Deinterleaver(
        _build_fec_random_interleaver(build_device),
        **_device_kwargs(build_device),
    )


def _build_fec_scrambler(build_device: str | None):
    from sionna.phy.fec import Scrambler

    return Scrambler(seed=1, keep_batch_constant=True, **_device_kwargs(build_device))


def _build_fec_tb5g_scrambler(build_device: str | None):
    from sionna.phy.fec import TB5GScrambler

    return TB5GScrambler(**_device_kwargs(build_device))


def _build_fec_descrambler(build_device: str | None):
    from sionna.phy.fec import Descrambler

    return Descrambler(_build_fec_scrambler(build_device), **_device_kwargs(build_device))


def _build_fec_linear_encoder(build_device: str | None):
    from sionna.phy.fec import LinearEncoder

    return LinearEncoder(_fec_linear_generator_matrix(), **_device_kwargs(build_device))


def _build_fec_os_decoder(build_device: str | None):
    from sionna.phy.fec import OSDecoder

    return OSDecoder(
        enc_mat=_fec_linear_generator_matrix(),
        **_device_kwargs(build_device),
    )


def _build_fec_ldpc_bp_decoder(build_device: str | None):
    from sionna.phy.fec import LDPCBPDecoder

    return LDPCBPDecoder(
        _fec_ldpc_pcm(),
        num_iter=1,
        **_device_kwargs(build_device),
    )


def _build_fec_ldpc_5g_encoder(build_device: str | None):
    from sionna.phy.fec import LDPC5GEncoder

    return LDPC5GEncoder(12, 24, **_device_kwargs(build_device))


def _build_fec_ldpc_5g_decoder(build_device: str | None):
    from sionna.phy.fec import LDPC5GDecoder

    return LDPC5GDecoder(
        _build_fec_ldpc_5g_encoder(build_device),
        num_iter=1,
        **_device_kwargs(build_device),
    )


def _build_fec_exit_callback(build_device: str | None):
    from sionna.phy.fec.ldpc.utils import EXITCallback

    return EXITCallback(1, **_device_kwargs(build_device))


def _build_fec_decoder_statistics_callback(build_device: str | None):
    from sionna.phy.fec.ldpc.utils import DecoderStatisticsCallback

    return DecoderStatisticsCallback(1, **_device_kwargs(build_device))


def _build_fec_weighted_bp_callback(build_device: str | None):
    from sionna.phy.fec.ldpc.utils import WeightedBPCallback

    return WeightedBPCallback(4, **_device_kwargs(build_device))


def _build_fec_polar_encoder(build_device: str | None):
    from sionna.phy.fec import PolarEncoder

    return PolarEncoder(
        _fec_polar_frozen_positions(),
        4,
        **_device_kwargs(build_device),
    )


def _build_fec_polar_sc_decoder(build_device: str | None):
    from sionna.phy.fec import PolarSCDecoder

    return PolarSCDecoder(
        _fec_polar_frozen_positions(),
        4,
        **_device_kwargs(build_device),
    )


def _build_fec_polar_scl_decoder(build_device: str | None):
    from sionna.phy.fec import PolarSCLDecoder

    return PolarSCLDecoder(
        _fec_polar_frozen_positions(),
        4,
        list_size=2,
        **_device_kwargs(build_device),
    )


def _build_fec_polar_bp_decoder(build_device: str | None):
    from sionna.phy.fec import PolarBPDecoder

    return PolarBPDecoder(
        _fec_polar_frozen_positions(),
        4,
        num_iter=1,
        **_device_kwargs(build_device),
    )


def _build_fec_polar_5g_encoder(build_device: str | None):
    from sionna.phy.fec import Polar5GEncoder

    return Polar5GEncoder(20, 40, **_device_kwargs(build_device))


def _build_fec_polar_5g_decoder(build_device: str | None):
    from sionna.phy.fec import Polar5GDecoder

    return Polar5GDecoder(
        _build_fec_polar_5g_encoder(build_device),
        dec_type="SC",
        **_device_kwargs(build_device),
    )


def _build_fec_turbo_termination(build_device: str | None):
    from sionna.phy.fec.turbo.utils import TurboTermination

    return TurboTermination(3, **_device_kwargs(build_device))


def _build_fec_turbo_encoder(build_device: str | None):
    from sionna.phy.fec import TurboEncoder

    return TurboEncoder(**_device_kwargs(build_device))


def _build_fec_turbo_decoder(build_device: str | None):
    from sionna.phy.fec import TurboDecoder

    return TurboDecoder(
        constraint_length=3,
        num_iter=1,
        **_device_kwargs(build_device),
    )


def _nr_pusch_config():
    from sionna.phy.nr import PUSCHConfig

    config = PUSCHConfig()
    config.carrier.n_size_grid = 1
    config.symbol_allocation = [0, 4]
    config.tb.mcs_index = 0
    return config


def _build_nr_layer_mapper(build_device: str | None):
    from sionna.phy.nr import LayerMapper

    return LayerMapper(num_layers=2, **_device_kwargs(build_device))


def _build_nr_layer_demapper(build_device: str | None):
    from sionna.phy.nr import LayerDemapper

    return LayerDemapper(
        _build_nr_layer_mapper(build_device),
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_nr_tb_encoder(build_device: str | None):
    from sionna.phy.nr import TBEncoder

    return TBEncoder(
        target_tb_size=64,
        num_coded_bits=192,
        target_coderate=0.5,
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_nr_tb_decoder(build_device: str | None):
    from sionna.phy.nr import TBDecoder

    return TBDecoder(
        _build_nr_tb_encoder(build_device),
        num_bp_iter=1,
        **_device_kwargs(build_device),
    )


def _build_nr_pusch_pilot_pattern(build_device: str | None):
    from sionna.phy.nr import PUSCHPilotPattern

    _ = build_device
    return PUSCHPilotPattern(_nr_pusch_config())


def _build_nr_pusch_precoder(build_device: str | None):
    import numpy as np
    from sionna.phy.nr import PUSCHPrecoder

    precoding_matrices = [np.array([[1.0 + 0.0j]], dtype=np.complex64)]
    return PUSCHPrecoder(precoding_matrices, **_device_kwargs(build_device))


def _build_nr_pusch_transmitter(build_device: str | None):
    from sionna.phy.nr import PUSCHTransmitter

    return PUSCHTransmitter(_nr_pusch_config(), **_device_kwargs(build_device))


def _build_nr_pusch_ls_channel_estimator(build_device: str | None):
    from sionna.phy.nr import PUSCHLSChannelEstimator

    transmitter = _build_nr_pusch_transmitter(build_device)
    return PUSCHLSChannelEstimator(
        transmitter.resource_grid,
        transmitter._dmrs_length,
        transmitter._dmrs_additional_position,
        transmitter._num_cdm_groups_without_data,
        **_device_kwargs(build_device),
    )


def _build_nr_pusch_receiver(build_device: str | None):
    from sionna.phy.nr import PUSCHReceiver

    return PUSCHReceiver(
        _build_nr_pusch_transmitter(build_device),
        **_device_kwargs(build_device),
    )


def _build_nr_coded_awgn_channel(build_device: str | None):
    from sionna.phy.nr import CodedAWGNChannelNR

    return CodedAWGNChannelNR(
        num_bits_per_symbol=2,
        num_info_bits=64,
        target_coderate=0.5,
        num_iter_decoder=1,
        **_device_kwargs(build_device),
    )


def _build_nr_mcs_decoder(build_device: str | None):
    from sionna.phy.nr import MCSDecoderNR

    return MCSDecoderNR(**_device_kwargs(build_device))


def _build_nr_transport_block(build_device: str | None):
    from sionna.phy.nr import TransportBlockNR

    return TransportBlockNR(**_device_kwargs(build_device))


def _stream_management_1x1():
    import numpy as np
    from sionna.phy.mimo import StreamManagement

    return StreamManagement(np.array([[1]], dtype=np.int32), 1)


def _build_mimo_stream_management(build_device: str | None):
    _ = build_device
    return _stream_management_1x1()


def _build_mimo_list2llr(build_device: str | None):
    from sionna.phy.mimo import List2LLR

    return List2LLR(**_device_kwargs(build_device))


def _build_mimo_list2llr_simple(build_device: str | None):
    from sionna.phy.mimo import List2LLRSimple

    return List2LLRSimple(2, **_device_kwargs(build_device))


def _build_mimo_linear_detector(build_device: str | None):
    from sionna.phy.mimo import LinearDetector

    return LinearDetector(
        "lmmse",
        "bit",
        "app",
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_mimo_maximum_likelihood_detector(build_device: str | None):
    from sionna.phy.mimo import MaximumLikelihoodDetector

    return MaximumLikelihoodDetector(
        "bit",
        "app",
        1,
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_mimo_k_best_detector(build_device: str | None):
    from sionna.phy.mimo import KBestDetector

    return KBestDetector(
        "bit",
        1,
        4,
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_mimo_ep_detector(build_device: str | None):
    from sionna.phy.mimo import EPDetector

    return EPDetector(
        "bit",
        2,
        **_device_kwargs(build_device),
    )


def _build_mimo_mmse_pic_detector(build_device: str | None):
    from sionna.phy.mimo import MMSEPICDetector

    return MMSEPICDetector(
        "bit",
        "app",
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _resource_grid_empty(build_device: str | None):
    from sionna.phy.ofdm import ResourceGrid

    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=8,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=2,
        num_guard_carriers=(1, 1),
        dc_null=True,
        pilot_pattern="empty",
        **_device_kwargs(build_device),
    )


def _resource_grid_kronecker(build_device: str | None):
    from sionna.phy.ofdm import ResourceGrid

    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=8,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=2,
        num_guard_carriers=(1, 1),
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[1],
        **_device_kwargs(build_device),
    )


def _build_resource_grid_empty(build_device: str | None):
    return _resource_grid_empty(build_device)


def _build_resource_grid_kronecker(build_device: str | None):
    return _resource_grid_kronecker(build_device)


def _build_empty_pilot_pattern(build_device: str | None):
    from sionna.phy.ofdm import EmptyPilotPattern

    return EmptyPilotPattern(1, 1, 4, 5, **_device_kwargs(build_device))


def _build_pilot_pattern(build_device: str | None):
    import torch
    from sionna.phy.ofdm import PilotPattern

    tensor_kwargs = _device_kwargs(build_device)
    mask = torch.zeros((1, 1, 4, 5), dtype=torch.bool, **tensor_kwargs)
    pilots = torch.zeros((1, 1, 0), dtype=torch.complex64, **tensor_kwargs)
    return PilotPattern(mask, pilots, **_device_kwargs(build_device))


def _build_kronecker_pilot_pattern(build_device: str | None):
    from sionna.phy.ofdm import KroneckerPilotPattern

    return KroneckerPilotPattern(
        _resource_grid_empty(build_device),
        [1],
        **_device_kwargs(build_device),
    )


def _build_resource_grid_mapper(build_device: str | None):
    from sionna.phy.ofdm import ResourceGridMapper

    return ResourceGridMapper(_resource_grid_empty(build_device), **_device_kwargs(build_device))


def _inputs_resource_grid_mapper(device: str):
    torch = _torch()
    x = torch.ones((2, 1, 1, 20), dtype=torch.complex64, device=device)
    return (x,), {}


def _build_resource_grid_demapper(build_device: str | None):
    from sionna.phy.ofdm import ResourceGridDemapper

    return ResourceGridDemapper(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _inputs_resource_grid_demapper(device: str):
    torch = _torch()
    y = torch.ones((2, 1, 1, 4, 8), dtype=torch.complex64, device=device)
    return (y,), {}


def _build_remove_nulled_subcarriers(build_device: str | None):
    from sionna.phy.ofdm import RemoveNulledSubcarriers

    return RemoveNulledSubcarriers(_resource_grid_empty(build_device), **_device_kwargs(build_device))


def _inputs_remove_nulled_subcarriers(device: str):
    torch = _torch()
    y = torch.ones((2, 1, 1, 4, 8), dtype=torch.complex64, device=device)
    return (y,), {}


def _build_ofdm_modulator(build_device: str | None):
    from sionna.phy.ofdm import OFDMModulator

    return OFDMModulator(cyclic_prefix_length=2, **_device_kwargs(build_device))


def _inputs_ofdm_modulator(device: str):
    torch = _torch()
    x = torch.ones((2, 4, 8), dtype=torch.complex64, device=device)
    return (x,), {}


def _build_ofdm_demodulator(build_device: str | None):
    from sionna.phy.ofdm import OFDMDemodulator

    return OFDMDemodulator(
        fft_size=8,
        l_min=0,
        cyclic_prefix_length=2,
        **_device_kwargs(build_device),
    )


def _inputs_ofdm_demodulator(device: str):
    torch = _torch()
    x = torch.ones((2, 40), dtype=torch.complex64, device=device)
    return (x,), {}


def _build_base_channel_interpolator(build_device: str | None):
    from sionna.phy.ofdm import BaseChannelInterpolator

    return BaseChannelInterpolator(**_device_kwargs(build_device))


def _build_base_channel_estimator(build_device: str | None):
    from sionna.phy.ofdm import BaseChannelEstimator

    return BaseChannelEstimator(
        _resource_grid_kronecker(build_device),
        interpolation_type=None,
        **_device_kwargs(build_device),
    )


def _build_ls_channel_estimator(build_device: str | None):
    from sionna.phy.ofdm import LSChannelEstimator

    return LSChannelEstimator(
        _resource_grid_kronecker(build_device),
        interpolation_type=None,
        **_device_kwargs(build_device),
    )


def _inputs_ls_channel_estimator(device: str):
    torch = _torch()
    y = torch.ones((2, 1, 1, 4, 8), dtype=torch.complex64, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (y, no), {}


def _build_lmmse_equalizer(build_device: str | None):
    from sionna.phy.ofdm import LMMSEEqualizer

    return LMMSEEqualizer(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_zf_equalizer(build_device: str | None):
    from sionna.phy.ofdm import ZFEqualizer

    return ZFEqualizer(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_mf_equalizer(build_device: str | None):
    from sionna.phy.ofdm import MFEqualizer

    return MFEqualizer(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _inputs_ofdm_equalizer(device: str):
    torch = _torch()
    y = torch.ones((2, 1, 1, 4, 8), dtype=torch.complex64, device=device)
    h_hat = torch.ones((2, 1, 1, 1, 1, 4, 5), dtype=torch.complex64, device=device)
    err_var = torch.zeros((2, 1, 1, 1, 1, 4, 5), dtype=torch.float32, device=device)
    no = torch.tensor(0.1, dtype=torch.float32, device=device)
    return (y, h_hat, err_var, no), {}


def _build_lmmse_post_equalization_sinr(build_device: str | None):
    from sionna.phy.ofdm import LMMSEPostEqualizationSINR

    return LMMSEPostEqualizationSINR(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_post_equalization_sinr(build_device: str | None):
    from sionna.phy.ofdm import PostEqualizationSINR

    return PostEqualizationSINR(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _dummy_equalizer(y, h_hat, err_var, no):
    _ = h_hat, err_var, no
    return y, y.real


def _dummy_detector(y, h_hat, err_var, no):
    _ = h_hat, err_var, no
    return y.real


def _dummy_detector_with_prior(y, h_hat, prior, err_var, no):
    _ = h_hat, prior, err_var, no
    return y.real


def _build_ofdm_equalizer(build_device: str | None):
    from sionna.phy.ofdm import OFDMEqualizer

    return OFDMEqualizer(
        _dummy_equalizer,
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_linear_detector(build_device: str | None):
    from sionna.phy.ofdm import LinearDetector

    return LinearDetector(
        "lmmse",
        "bit",
        "app",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_maximum_likelihood_detector(build_device: str | None):
    from sionna.phy.ofdm import MaximumLikelihoodDetector

    return MaximumLikelihoodDetector(
        "bit",
        "app",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_maximum_likelihood_detector_with_prior(build_device: str | None):
    from sionna.phy.ofdm import MaximumLikelihoodDetectorWithPrior

    return MaximumLikelihoodDetectorWithPrior(
        "bit",
        "app",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_k_best_detector(build_device: str | None):
    from sionna.phy.ofdm import KBestDetector

    return KBestDetector(
        "bit",
        1,
        4,
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_ep_detector(build_device: str | None):
    from sionna.phy.ofdm import EPDetector

    return EPDetector(
        "bit",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_mmse_pic_detector(build_device: str | None):
    from sionna.phy.ofdm import MMSEPICDetector

    return MMSEPICDetector(
        "bit",
        "app",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_ofdm_detector(build_device: str | None):
    from sionna.phy.ofdm import OFDMDetector

    return OFDMDetector(
        _dummy_detector,
        "bit",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_ofdm_detector_with_prior(build_device: str | None):
    from sionna.phy.ofdm import OFDMDetectorWithPrior

    return OFDMDetectorWithPrior(
        _dummy_detector_with_prior,
        "bit",
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        constellation_type="qam",
        num_bits_per_symbol=2,
        **_device_kwargs(build_device),
    )


def _build_rzf_precoder(build_device: str | None):
    from sionna.phy.ofdm import RZFPrecoder

    return RZFPrecoder(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_precoded_channel(build_device: str | None):
    from sionna.phy.ofdm import PrecodedChannel

    return PrecodedChannel(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_cbf_precoded_channel(build_device: str | None):
    from sionna.phy.ofdm import CBFPrecodedChannel

    return CBFPrecodedChannel(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_eye_precoded_channel(build_device: str | None):
    from sionna.phy.ofdm import EyePrecodedChannel

    return EyePrecodedChannel(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


def _build_rzf_precoded_channel(build_device: str | None):
    from sionna.phy.ofdm import RZFPrecodedChannel

    return RZFPrecodedChannel(
        _resource_grid_empty(build_device),
        _stream_management_1x1(),
        **_device_kwargs(build_device),
    )


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
    CaseSpec(
        name="fec-gaussian-prior-source",
        description="GaussianPriorSource; FEC helper that samples Gaussian LLR priors using Sionna device state.",
        build=_build_fec_gaussian_prior_source,
        make_inputs=None,
        categories=_categories("fec", "source", "audit-only"),
    ),
    CaseSpec(
        name="fec-crc-encoder",
        description="CRCEncoder; FEC CRC encoder with generated CRC matrix state.",
        build=_build_fec_crc_encoder,
        make_inputs=None,
        categories=_categories("fec", "crc", "audit-only"),
    ),
    CaseSpec(
        name="fec-crc-decoder",
        description="CRCDecoder; FEC CRC decoder that owns a CRCEncoder child block.",
        build=_build_fec_crc_decoder,
        make_inputs=None,
        categories=_categories("fec", "crc", "audit-only"),
    ),
    CaseSpec(
        name="fec-trellis",
        description="Trellis; convolutional-code helper with explicit device-managed tensor state.",
        build=_build_fec_trellis,
        make_inputs=None,
        categories=_categories("fec", "conv", "audit-only"),
    ),
    CaseSpec(
        name="fec-conv-encoder",
        description="ConvEncoder; convolutional FEC encoder with nested Trellis state.",
        build=_build_fec_conv_encoder,
        make_inputs=None,
        categories=_categories("fec", "conv", "audit-only"),
    ),
    CaseSpec(
        name="fec-viterbi-decoder",
        description="ViterbiDecoder; convolutional FEC decoder with trellis-derived state.",
        build=_build_fec_viterbi_decoder,
        make_inputs=None,
        categories=_categories("fec", "conv", "audit-only"),
    ),
    CaseSpec(
        name="fec-bcjr-decoder",
        description="BCJRDecoder; convolutional FEC decoder with trellis-derived state.",
        build=_build_fec_bcjr_decoder,
        make_inputs=None,
        categories=_categories("fec", "conv", "audit-only"),
    ),
    CaseSpec(
        name="fec-row-column-interleaver",
        description="RowColumnInterleaver; FEC interleaver with generated permutation state.",
        build=_build_fec_row_column_interleaver,
        make_inputs=None,
        categories=_categories("fec", "interleaver", "audit-only"),
    ),
    CaseSpec(
        name="fec-random-interleaver",
        description="RandomInterleaver; FEC random interleaver with seed and device state.",
        build=_build_fec_random_interleaver,
        make_inputs=None,
        categories=_categories("fec", "interleaver", "audit-only"),
    ),
    CaseSpec(
        name="fec-turbo-3gpp-interleaver",
        description="Turbo3GPPInterleaver; 3GPP turbo-code interleaver helper.",
        build=_build_fec_turbo_3gpp_interleaver,
        make_inputs=None,
        categories=_categories("fec", "interleaver", "turbo", "audit-only"),
    ),
    CaseSpec(
        name="fec-deinterleaver",
        description="Deinterleaver; FEC deinterleaver wrapping a RandomInterleaver child block.",
        build=_build_fec_deinterleaver,
        make_inputs=None,
        categories=_categories("fec", "interleaver", "audit-only"),
    ),
    CaseSpec(
        name="fec-scrambler",
        description="Scrambler; FEC scrambler with random sequence generation state.",
        build=_build_fec_scrambler,
        make_inputs=None,
        categories=_categories("fec", "scrambling", "audit-only"),
    ),
    CaseSpec(
        name="fec-tb5g-scrambler",
        description="TB5GScrambler; 5G transport-block scrambler with generated sequence state.",
        build=_build_fec_tb5g_scrambler,
        make_inputs=None,
        categories=_categories("fec", "scrambling", "audit-only"),
    ),
    CaseSpec(
        name="fec-descrambler",
        description="Descrambler; FEC descrambler wrapping a Scrambler child block.",
        build=_build_fec_descrambler,
        make_inputs=None,
        categories=_categories("fec", "scrambling", "audit-only"),
    ),
    CaseSpec(
        name="fec-linear-encoder",
        description="LinearEncoder; generic linear block-code encoder with generator-matrix state.",
        build=_build_fec_linear_encoder,
        make_inputs=None,
        categories=_categories("fec", "linear", "audit-only"),
    ),
    CaseSpec(
        name="fec-os-decoder",
        description="OSDecoder; ordered-statistics decoder with generator-matrix state.",
        build=_build_fec_os_decoder,
        make_inputs=None,
        categories=_categories("fec", "linear", "audit-only"),
    ),
    CaseSpec(
        name="fec-ldpc-bp-decoder",
        description="LDPCBPDecoder; belief-propagation decoder with parity-check graph tensors.",
        build=_build_fec_ldpc_bp_decoder,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "audit-only"),
    ),
    CaseSpec(
        name="fec-ldpc-5g-encoder",
        description="LDPC5GEncoder; 5G LDPC encoder with base-graph lookup tensors.",
        build=_build_fec_ldpc_5g_encoder,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "audit-only"),
    ),
    CaseSpec(
        name="fec-ldpc-5g-decoder",
        description="LDPC5GDecoder; 5G LDPC decoder with encoder and BP decoder state.",
        build=_build_fec_ldpc_5g_decoder,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "audit-only"),
    ),
    CaseSpec(
        name="fec-exit-callback",
        description="EXITCallback; LDPC decoder callback with registered metric buffers.",
        build=_build_fec_exit_callback,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "callback", "audit-only"),
    ),
    CaseSpec(
        name="fec-decoder-statistics-callback",
        description="DecoderStatisticsCallback; LDPC decoder callback with registered statistic buffers.",
        build=_build_fec_decoder_statistics_callback,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "callback", "audit-only"),
    ),
    CaseSpec(
        name="fec-weighted-bp-callback",
        description="WeightedBPCallback; LDPC weighted-BP callback with trainable edge weights.",
        build=_build_fec_weighted_bp_callback,
        make_inputs=None,
        categories=_categories("fec", "ldpc", "callback", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-encoder",
        description="PolarEncoder; polar encoder with frozen-bit and gather-index tensors.",
        build=_build_fec_polar_encoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-sc-decoder",
        description="PolarSCDecoder; polar successive-cancellation decoder state.",
        build=_build_fec_polar_sc_decoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-scl-decoder",
        description="PolarSCLDecoder; polar list decoder with frozen-bit and CRC child state.",
        build=_build_fec_polar_scl_decoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-bp-decoder",
        description="PolarBPDecoder; polar belief-propagation decoder state.",
        build=_build_fec_polar_bp_decoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-5g-encoder",
        description="Polar5GEncoder; 5G polar encoder with CRC, rate-matching, and interleaver state.",
        build=_build_fec_polar_5g_encoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-polar-5g-decoder",
        description="Polar5GDecoder; 5G polar decoder with nested encoder and decoder state.",
        build=_build_fec_polar_5g_decoder,
        make_inputs=None,
        categories=_categories("fec", "polar", "audit-only"),
    ),
    CaseSpec(
        name="fec-turbo-termination",
        description="TurboTermination; turbo-code termination helper with device-managed state.",
        build=_build_fec_turbo_termination,
        make_inputs=None,
        categories=_categories("fec", "turbo", "audit-only"),
    ),
    CaseSpec(
        name="fec-turbo-encoder",
        description="TurboEncoder; turbo-code encoder with trellis, interleaver, and termination state.",
        build=_build_fec_turbo_encoder,
        make_inputs=None,
        categories=_categories("fec", "turbo", "audit-only"),
    ),
    CaseSpec(
        name="fec-turbo-decoder",
        description="TurboDecoder; turbo-code decoder with BCJR and interleaver child state.",
        build=_build_fec_turbo_decoder,
        make_inputs=None,
        categories=_categories("fec", "turbo", "audit-only"),
    ),
    CaseSpec(
        name="nr-layer-mapper",
        description="LayerMapper; NR MIMO layer mapping block with Sionna device state.",
        build=_build_nr_layer_mapper,
        make_inputs=None,
        categories=_categories("nr", "layer-mapping", "audit-only"),
    ),
    CaseSpec(
        name="nr-layer-demapper",
        description="LayerDemapper; NR inverse layer mapping block tied to a LayerMapper child.",
        build=_build_nr_layer_demapper,
        make_inputs=None,
        categories=_categories("nr", "layer-mapping", "audit-only"),
    ),
    CaseSpec(
        name="nr-tb-encoder",
        description="TBEncoder; 5G NR transport-block encoder with CRC, LDPC, and scrambling state.",
        build=_build_nr_tb_encoder,
        make_inputs=None,
        categories=_categories("nr", "transport-block", "audit-only"),
    ),
    CaseSpec(
        name="nr-tb-decoder",
        description="TBDecoder; 5G NR transport-block decoder wrapping LDPC, descrambling, and CRC state.",
        build=_build_nr_tb_decoder,
        make_inputs=None,
        categories=_categories("nr", "transport-block", "audit-only"),
    ),
    CaseSpec(
        name="nr-pusch-pilot-pattern",
        description="PUSCHPilotPattern; NR pilot pattern generated from a PUSCHConfig.",
        build=_build_nr_pusch_pilot_pattern,
        make_inputs=None,
        categories=_categories("nr", "pusch", "pilot", "audit-only"),
    ),
    CaseSpec(
        name="nr-pusch-precoder",
        description="PUSCHPrecoder; NR PUSCH precoder with registered precoding matrix buffer.",
        build=_build_nr_pusch_precoder,
        make_inputs=None,
        categories=_categories("nr", "pusch", "precoding", "audit-only"),
    ),
    CaseSpec(
        name="nr-pusch-transmitter",
        description="PUSCHTransmitter; composite NR transmitter with FEC, mapping, OFDM, and pilot state.",
        build=_build_nr_pusch_transmitter,
        make_inputs=None,
        categories=_categories("nr", "pusch", "transmitter", "audit-only"),
    ),
    CaseSpec(
        name="nr-pusch-ls-channel-estimator",
        description="PUSCHLSChannelEstimator; NR LS estimator over a PUSCH resource grid.",
        build=_build_nr_pusch_ls_channel_estimator,
        make_inputs=None,
        categories=_categories("nr", "pusch", "channel-estimation", "audit-only"),
    ),
    CaseSpec(
        name="nr-pusch-receiver",
        description="PUSCHReceiver; composite NR receiver with estimator, detector, demapper, and decoder state.",
        build=_build_nr_pusch_receiver,
        make_inputs=None,
        categories=_categories("nr", "pusch", "receiver", "audit-only"),
    ),
    CaseSpec(
        name="nr-coded-awgn-channel",
        description="CodedAWGNChannelNR; NR coded AWGN link helper with source, mapper, channel, and decoder state.",
        build=_build_nr_coded_awgn_channel,
        make_inputs=None,
        categories=_categories("nr", "utils", "audit-only"),
    ),
    CaseSpec(
        name="nr-mcs-decoder",
        description="MCSDecoderNR; NR MCS decoder helper with tensor lookup state.",
        build=_build_nr_mcs_decoder,
        make_inputs=None,
        categories=_categories("nr", "utils", "audit-only"),
    ),
    CaseSpec(
        name="nr-transport-block",
        description="TransportBlockNR; NR transport-block helper with dynamic encoding state.",
        build=_build_nr_transport_block,
        make_inputs=None,
        categories=_categories("nr", "utils", "audit-only"),
    ),
    CaseSpec(
        name="mimo-stream-management",
        description="StreamManagement; MIMO stream association metadata shared by OFDM MIMO blocks.",
        build=_build_mimo_stream_management,
        make_inputs=None,
        categories=_categories("mimo", "stream-management", "audit-only"),
    ),
    CaseSpec(
        name="mimo-list2llr",
        description="List2LLR; audit-only MIMO candidate-list to LLR base block.",
        build=_build_mimo_list2llr,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-list2llr-simple",
        description="List2LLRSimple; MIMO candidate-list to LLR helper with lookup tensors.",
        build=_build_mimo_list2llr_simple,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-linear-detector",
        description="LinearDetector; standalone MIMO linear detector with mapping child blocks.",
        build=_build_mimo_linear_detector,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-maximum-likelihood-detector",
        description="MaximumLikelihoodDetector; standalone MIMO ML detector with lookup tensors.",
        build=_build_mimo_maximum_likelihood_detector,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-k-best-detector",
        description="KBestDetector; standalone MIMO K-best detector with precomputed tensor state.",
        build=_build_mimo_k_best_detector,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-ep-detector",
        description="EPDetector; standalone MIMO expectation-propagation detector.",
        build=_build_mimo_ep_detector,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mimo-mmse-pic-detector",
        description="MMSEPICDetector; standalone MIMO MMSE-PIC detector with mapping child blocks.",
        build=_build_mimo_mmse_pic_detector,
        make_inputs=None,
        categories=_categories("mimo", "detector", "audit-only"),
    ),
    CaseSpec(
        name="resource-grid-empty",
        description="ResourceGrid with an empty pilot pattern; central OFDM grid metadata object.",
        build=_build_resource_grid_empty,
        make_inputs=None,
        categories=_categories("ofdm", "resource-grid", "audit-only"),
    ),
    CaseSpec(
        name="resource-grid-kronecker",
        description="ResourceGrid with a Kronecker pilot pattern; owns nested pilot pattern state.",
        build=_build_resource_grid_kronecker,
        make_inputs=None,
        categories=_categories("ofdm", "resource-grid", "pilot"),
    ),
    CaseSpec(
        name="empty-pilot-pattern",
        description="EmptyPilotPattern; OFDM pilot metadata without pilot tensors.",
        build=_build_empty_pilot_pattern,
        make_inputs=None,
        categories=_categories("ofdm", "pilot", "audit-only"),
    ),
    CaseSpec(
        name="pilot-pattern",
        description="PilotPattern; explicit pilot mask and pilot tensors.",
        build=_build_pilot_pattern,
        make_inputs=None,
        categories=_categories("ofdm", "pilot", "audit-only"),
    ),
    CaseSpec(
        name="kronecker-pilot-pattern",
        description="KroneckerPilotPattern; generated pilot tensors tied to a ResourceGrid.",
        build=_build_kronecker_pilot_pattern,
        make_inputs=None,
        categories=_categories("ofdm", "pilot", "audit-only"),
    ),
    CaseSpec(
        name="resource-grid-mapper",
        description="ResourceGridMapper; maps stream symbols into an OFDM resource grid.",
        build=_build_resource_grid_mapper,
        make_inputs=_inputs_resource_grid_mapper,
        categories=_categories("ofdm", "resource-grid", "mapper"),
    ),
    CaseSpec(
        name="resource-grid-demapper",
        description="ResourceGridDemapper; extracts stream symbols from an OFDM resource grid.",
        build=_build_resource_grid_demapper,
        make_inputs=_inputs_resource_grid_demapper,
        categories=_categories("ofdm", "resource-grid", "demapper"),
    ),
    CaseSpec(
        name="remove-nulled-subcarriers",
        description="RemoveNulledSubcarriers; removes guard and DC subcarriers from an OFDM grid.",
        build=_build_remove_nulled_subcarriers,
        make_inputs=_inputs_remove_nulled_subcarriers,
        categories=_categories("ofdm", "resource-grid"),
    ),
    CaseSpec(
        name="ofdm-modulator",
        description="OFDMModulator; converts frequency-domain OFDM symbols to time-domain samples.",
        build=_build_ofdm_modulator,
        make_inputs=_inputs_ofdm_modulator,
        categories=_categories("ofdm", "modem"),
    ),
    CaseSpec(
        name="ofdm-demodulator",
        description="OFDMDemodulator; converts time-domain OFDM samples to frequency-domain symbols.",
        build=_build_ofdm_demodulator,
        make_inputs=_inputs_ofdm_demodulator,
        categories=_categories("ofdm", "modem"),
    ),
    CaseSpec(
        name="base-channel-interpolator",
        description="BaseChannelInterpolator; audit-only base OFDM channel interpolation object.",
        build=_build_base_channel_interpolator,
        make_inputs=None,
        categories=_categories("ofdm", "channel-estimation", "audit-only"),
    ),
    CaseSpec(
        name="base-channel-estimator",
        description=(
            "BaseChannelEstimator without default interpolator; audit-only base "
            "OFDM channel estimator object."
        ),
        build=_build_base_channel_estimator,
        make_inputs=None,
        categories=_categories("ofdm", "channel-estimation", "audit-only"),
    ),
    CaseSpec(
        name="ls-channel-estimator",
        description=(
            "LSChannelEstimator without default interpolator; pilot-only least-squares "
            "channel estimator."
        ),
        build=_build_ls_channel_estimator,
        make_inputs=_inputs_ls_channel_estimator,
        categories=_categories("ofdm", "channel-estimation"),
    ),
    CaseSpec(
        name="lmmse-equalizer",
        description="LMMSEEqualizer; OFDM linear MMSE equalizer with resource-grid metadata.",
        build=_build_lmmse_equalizer,
        make_inputs=_inputs_ofdm_equalizer,
        categories=_categories("ofdm", "equalizer"),
    ),
    CaseSpec(
        name="zf-equalizer",
        description="ZFEqualizer; OFDM zero-forcing equalizer with resource-grid metadata.",
        build=_build_zf_equalizer,
        make_inputs=_inputs_ofdm_equalizer,
        categories=_categories("ofdm", "equalizer"),
    ),
    CaseSpec(
        name="mf-equalizer",
        description="MFEqualizer; OFDM matched-filter equalizer with resource-grid metadata.",
        build=_build_mf_equalizer,
        make_inputs=_inputs_ofdm_equalizer,
        categories=_categories("ofdm", "equalizer"),
    ),
    CaseSpec(
        name="lmmse-post-equalization-sinr",
        description="LMMSEPostEqualizationSINR; audit-only post-equalization SINR helper.",
        build=_build_lmmse_post_equalization_sinr,
        make_inputs=None,
        categories=_categories("ofdm", "equalizer", "audit-only"),
    ),
    CaseSpec(
        name="post-equalization-sinr",
        description="PostEqualizationSINR; audit-only post-equalization SINR helper.",
        build=_build_post_equalization_sinr,
        make_inputs=None,
        categories=_categories("ofdm", "equalizer", "audit-only"),
    ),
    CaseSpec(
        name="ofdm-equalizer",
        description="OFDMEqualizer; audit-only wrapper around a user-provided equalizer callable.",
        build=_build_ofdm_equalizer,
        make_inputs=None,
        categories=_categories("ofdm", "equalizer", "audit-only"),
    ),
    CaseSpec(
        name="linear-detector",
        description="LinearDetector; OFDM linear detector with constellation metadata.",
        build=_build_linear_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="maximum-likelihood-detector",
        description="MaximumLikelihoodDetector; OFDM ML detector with constellation metadata.",
        build=_build_maximum_likelihood_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="maximum-likelihood-detector-with-prior",
        description="MaximumLikelihoodDetectorWithPrior; OFDM ML detector with prior metadata.",
        build=_build_maximum_likelihood_detector_with_prior,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="k-best-detector",
        description="KBestDetector; OFDM K-best detector with constellation metadata.",
        build=_build_k_best_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="ep-detector",
        description="EPDetector; OFDM expectation-propagation detector.",
        build=_build_ep_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="mmse-pic-detector",
        description="MMSEPICDetector; OFDM MMSE-PIC detector with constellation metadata.",
        build=_build_mmse_pic_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="ofdm-detector",
        description="OFDMDetector; audit-only wrapper around a user-provided detector callable.",
        build=_build_ofdm_detector,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="ofdm-detector-with-prior",
        description="OFDMDetectorWithPrior; audit-only detector wrapper with prior metadata.",
        build=_build_ofdm_detector_with_prior,
        make_inputs=None,
        categories=_categories("ofdm", "detector", "audit-only"),
    ),
    CaseSpec(
        name="rzf-precoder",
        description="RZFPrecoder; audit-only OFDM regularized zero-forcing precoder.",
        build=_build_rzf_precoder,
        make_inputs=None,
        categories=_categories("ofdm", "precoding", "audit-only"),
    ),
    CaseSpec(
        name="precoded-channel",
        description="PrecodedChannel; audit-only base precoded OFDM channel helper.",
        build=_build_precoded_channel,
        make_inputs=None,
        categories=_categories("ofdm", "precoding", "audit-only"),
    ),
    CaseSpec(
        name="cbf-precoded-channel",
        description="CBFPrecodedChannel; audit-only conjugate beamforming channel helper.",
        build=_build_cbf_precoded_channel,
        make_inputs=None,
        categories=_categories("ofdm", "precoding", "audit-only"),
    ),
    CaseSpec(
        name="eye-precoded-channel",
        description="EyePrecodedChannel; audit-only identity precoding channel helper.",
        build=_build_eye_precoded_channel,
        make_inputs=None,
        categories=_categories("ofdm", "precoding", "audit-only"),
    ),
    CaseSpec(
        name="rzf-precoded-channel",
        description="RZFPrecodedChannel; audit-only RZF-precoded OFDM channel helper.",
        build=_build_rzf_precoded_channel,
        make_inputs=None,
        categories=_categories("ofdm", "precoding", "audit-only"),
    ),
)
