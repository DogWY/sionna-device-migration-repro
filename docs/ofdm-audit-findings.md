# OFDM audit findings

## Context

This document records CUDA audit evidence for the `sionna.phy.ofdm` dynamic
case set. The clean OFDM sweep shows the same device-migration pattern already
seen in channel, mapping, and signal objects.

## Command

The clean sweep used:

```bash
CUDA_DEVICE=cuda:0
python run_repro.py run --category ofdm --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/ofdm-audit-cuda.json
```

The collected report used `cuda:1`; any visible CUDA device can be used.

## Clean sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Target device in collected report: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total OFDM-category cases: 33.
- Failed audit cases: 33.
- Skipped cases: 0.
- All OFDM-category cases include stale Sionna logical device state.
- Many OFDM cases also include ordinary tensor attributes that remain on CPU
  after `.to(device)`.

## Case-level findings

| Case | Issue count | Main finding |
| --- | ---: | --- |
| `apply-ofdm` | 2 | parent and internal `AWGN` logical devices remain `cpu` |
| `resource-grid-empty` | 2 | resource grid and empty pilot pattern logical devices remain `cpu` |
| `resource-grid-kronecker` | 2 | resource grid and Kronecker pilot pattern logical devices remain `cpu` |
| `empty-pilot-pattern` | 1 | `root._device_str` remains `cpu` |
| `pilot-pattern` | 1 | `root._device_str` remains `cpu` |
| `kronecker-pilot-pattern` | 1 | `root._device_str` remains `cpu` |
| `resource-grid-mapper` | 4 | resource-grid child state and `_rg_type` remain `cpu` |
| `resource-grid-demapper` | 4 | stream management and resource-grid child logical devices remain `cpu` |
| `remove-nulled-subcarriers` | 1 | `root._device_str` remains `cpu` |
| `ofdm-modulator` | 1 | `root._device_str` remains `cpu` |
| `ofdm-demodulator` | 1 | `root._device_str` remains `cpu` |
| `base-channel-interpolator` | 1 | `root._device_str` remains `cpu` |
| `base-channel-estimator` | 4 | estimator, pilot pattern, nulled-subcarrier remover, and `_pilot_ind` remain `cpu` |
| `ls-channel-estimator` | 4 | estimator, pilot pattern, nulled-subcarrier remover, and `_pilot_ind` remain `cpu` |
| `lmmse-equalizer` | 9 | child logical devices and equalizer index tensors remain `cpu` |
| `zf-equalizer` | 9 | child logical devices and equalizer index tensors remain `cpu` |
| `mf-equalizer` | 9 | child logical devices and equalizer index tensors remain `cpu` |
| `lmmse-post-equalization-sinr` | 4 | helper, resource-grid, pilot-pattern, and stream-management devices remain `cpu` |
| `post-equalization-sinr` | 4 | helper, resource-grid, pilot-pattern, and stream-management devices remain `cpu` |
| `ofdm-equalizer` | 9 | wrapper child devices and OFDM data/detection index tensors remain `cpu` |
| `linear-detector` | 18 | nested detector, demapper, constellation, and OFDM index state remain `cpu` |
| `maximum-likelihood-detector` | 21 | nested ML detector lookup tensors and OFDM index state remain `cpu` |
| `maximum-likelihood-detector-with-prior` | 23 | nested prior detector, constellation, and OFDM index state remain `cpu` |
| `k-best-detector` | 17 | nested list-to-LLR and K-best lookup tensors remain `cpu` |
| `ep-detector` | 17 | nested EP detector tensors and OFDM index state remain `cpu` |
| `mmse-pic-detector` | 23 | nested MMSE-PIC detector, demapper, constellation, and OFDM index state remain `cpu` |
| `ofdm-detector` | 9 | wrapper child devices and OFDM data/detection index tensors remain `cpu` |
| `ofdm-detector-with-prior` | 11 | wrapper child devices, constellation, and OFDM index tensors remain `cpu` |
| `rzf-precoder` | 5 | precoder, resource-grid, stream-management, and nulled-subcarrier remover devices remain `cpu` |
| `precoded-channel` | 5 | precoded-channel helper child devices remain `cpu` |
| `cbf-precoded-channel` | 5 | conjugate-beamforming helper child devices remain `cpu` |
| `eye-precoded-channel` | 5 | identity-precoding helper child devices remain `cpu` |
| `rzf-precoded-channel` | 5 | RZF-precoded helper child devices remain `cpu` |

## Failure categories observed

### Stale logical device state

Every OFDM-category case includes stale logical device state. Examples:

- `ResourceGrid._device_str`
- `ResourceGrid._pilot_pattern._device_str`
- `ResourceGridMapper._resource_grid._device_str`
- `StreamManagement._device_str`
- `RemoveNulledSubcarriers._device_str`
- `OFDMModulator._device_str`
- `OFDMDemodulator._device_str`
- Detector and precoder wrapper `_device_str` fields

### Ordinary tensor attributes not migrated

OFDM objects also keep ordinary tensor attributes that PyTorch `.to()` does not
migrate. Examples:

- Resource-grid tensor state: `_rg_type`.
- Channel-estimator tensor state: `_pilot_ind`.
- Equalizer and detector index tensors: `_data_ind`, `_detection_desired_ind`,
  `_detection_undesired_ind`, and `_stream_ind`.
- Mapping/detector lookup tensors nested inside detector objects:
  `_points`, `_c0`, `_c1`, `_a`, `_vecs`, `_vecs_ind`, `_c`, `_indices`,
  `_sym_pattern`, `_ind_pattern`, `_es`, and `_no`.

### Nested child state

Composite OFDM objects recursively own ResourceGrid, StreamManagement,
RemoveNulledSubcarriers, mapping, detector, and constellation helpers. The
audit shows stale state across these child objects, confirming that PyTorch
module recursion does not fix Sionna-specific logical device fields or ordinary
tensor attributes.

## Interpretation

The clean OFDM sweep is strong evidence that `.to(device)` migration issues are
systematic across standalone `sionna.phy.ofdm` objects, not only across channel,
mapping, and signal objects.

Combined audit-only CUDA evidence:

- `sionna.phy.channel`: 17/17 cases failed audit.
- `sionna.phy.mapping`: 14/14 cases failed audit.
- `sionna.phy.signal`: 12/12 cases failed audit.
- `sionna.phy.ofdm`: 33/33 OFDM-category cases failed audit.

## Related evidence

The umbrella PHY audit and forward-probe summaries are in
[`phy-audit-findings.md`](phy-audit-findings.md) and
[`forward-probe-findings.md`](forward-probe-findings.md).
