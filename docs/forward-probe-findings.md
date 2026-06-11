# Forward-probe findings

## Context

This document records CUDA forward-probe evidence for the current 126-case
dynamic PHY case set. Unlike the audit-only sweeps, this run kept forward
probes enabled for cases that already have safe minimal inputs.

The purpose is to show runtime impact after CPU construction followed by normal
PyTorch `.to(cuda_device)`, not only stale object state.

## Command

```bash
CUDA_DEVICE=cuda:0
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-fail --json-report reports/phy-forward-cuda.json
```

The collected report used `cuda:1`; any visible CUDA device can be used.

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Target device in collected report: `cuda:1`.
- Build device: `cpu`.
- Forward probes: enabled where the case defines safe minimal inputs.

Result:

- Total PHY dynamic cases: 126.
- Failed cases: 125.
- Passed cases: 1.
- Skipped cases: 0.
- Only passed case: standalone `fec-trellis`.
- Forward errors: 18 cases.
- Wrong-device forward outputs: 30 cases, covering 33 returned tensors.

## Runtime failure categories

### Forward exceptions

The broad run counted 18 forward errors. One was a server-side CUDA allocation
failure in `wrapped-awgn-channel` during the broad sweep and is not used as
device-migration evidence here. Representative device-related exceptions are:

| Case | Error class | Main symptom |
| --- | --- | --- |
| `apply-time` | `RuntimeError` | CPU index tensor used with CUDA input tensor |
| `kronecker-flat-fading` | `RuntimeError` | matrix multiplication mixed CPU and CUDA tensors |
| `binary-memoryless` | `RuntimeError` | CPU/CUDA tensor mix |
| `binary-symmetric` | `RuntimeError` | CPU/CUDA tensor mix |
| `binary-erasure` | `RuntimeError` | CPU/CUDA tensor mix |
| `binary-z` | `RuntimeError` | CPU/CUDA tensor mix |
| `edfa` | `RuntimeError` | CPU/CUDA tensor mix |
| `ssfm` | `RuntimeError` | CPU/CUDA tensor mix |
| `pam2qam` | `RuntimeError` | CPU lookup indices used with CUDA input |
| `qam2pam` | `RuntimeError` | CPU lookup indices used with CUDA input |
| `resource-grid-demapper` | `RuntimeError` | CPU gather/index state used with CUDA input |
| `remove-nulled-subcarriers` | `RuntimeError` | CPU index-select state used with CUDA input |
| `ofdm-demodulator` | `RuntimeError` | CPU/CUDA tensor mix |
| `ls-channel-estimator` | `RuntimeError` | CPU index-select state used with CUDA input |
| `lmmse-equalizer` | `RuntimeError` | CPU index-select state used with CUDA input |
| `zf-equalizer` | `RuntimeError` | CPU index-select state used with CUDA input |
| `mf-equalizer` | `RuntimeError` | CPU index-select state used with CUDA input |

### Forward outputs left on CPU

The following cases completed their forward probe but returned one or more CPU
tensors when the target was a CUDA device:

| Case | Wrong-device output tensors |
| --- | ---: |
| `awgn` | 1 |
| `generate-flat-fading` | 1 |
| `apply-flat-fading` | 1 |
| `apply-ofdm` | 1 |
| `flat-fading` | 2 |
| `rayleigh-block-fading` | 2 |
| `binary-source` | 1 |
| `constellation-qam` | 1 |
| `mapper-qam` | 1 |
| `demapper-qam-app` | 1 |
| `symbol-demapper-qam` | 1 |
| `llrs2symbol-logits` | 1 |
| `symbol-logits2llrs` | 1 |
| `symbol-inds2bits` | 1 |
| `symbol-logits2moments` | 2 |
| `pam-source` | 1 |
| `qam-source` | 1 |
| `symbol-source` | 1 |
| `upsampling` | 1 |
| `downsampling` | 1 |
| `custom-window` | 1 |
| `hamming-window` | 1 |
| `hann-window` | 1 |
| `blackman-window` | 1 |
| `custom-filter` | 1 |
| `sinc-filter` | 1 |
| `raised-cosine-filter` | 1 |
| `root-raised-cosine-filter` | 1 |
| `resource-grid-mapper` | 1 |
| `ofdm-modulator` | 1 |

## Interpretation

The forward sweep confirms that the stale object-state issues are not merely
inspection artifacts:

- Some blocks silently return tensors on CPU after `.to(device)`.
- Some blocks fail at runtime because stale CPU tensors are used with CUDA
  inputs.
- Some composite cases only expose object-state issues because their forward
  probes are intentionally disabled until safe minimal inputs are available.

The audit-only and forward-probe evidence together show that Sionna PHY objects
do not consistently satisfy expected PyTorch `.to(device)` semantics.
