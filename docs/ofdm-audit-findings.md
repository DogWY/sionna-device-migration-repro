# OFDM audit findings

## Context

This document records CUDA audit evidence for the `sionna.phy.ofdm` dynamic
case set. The clean OFDM sweep shows the same device-migration pattern already
seen in channel, mapping, and signal objects.

An earlier OFDM sweep exposed a repro-harness issue: two channel-estimator cases
were skipped during construction because Sionna's default channel interpolator
uses the global `sionna.phy.config.device`, bypassing the case-level
`build_device`. The runner now temporarily aligns Sionna's global construction
device with `--build-device`, and the estimator cases avoid the default
interpolator so that `--build-device cpu` remains clean.

## Command

The clean sweep used:

```bash
python run_repro.py run --category ofdm --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/ofdm-audit-cuda1.json
```

## Clean sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Sionna environment: `sdm`.
- Target device: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total OFDM-category cases: 33.
- Failed audit cases: 33.
- Skipped cases: 0.
- All OFDM-category cases include stale Sionna logical device state.
- Many OFDM cases also include ordinary tensor attributes that remain on CPU
  after `.to(cuda:1)`.

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

## Harness fix confirmed

The clean sweep confirms that the construction-device fix worked:

- `base-channel-estimator` and `ls-channel-estimator` no longer skip during
  construction.
- `StreamManagement._device_str` is now constructed on `cpu` under
  `--build-device cpu`, rather than leaking Sionna's previous global default
  such as `cuda:0`.

These fields still fail the audit after `.to(cuda:1)`, which is the intended
device-migration evidence.

## Interpretation

The clean OFDM sweep is strong evidence that `.to(device)` migration issues are
systematic across standalone `sionna.phy.ofdm` objects, not only across channel,
mapping, and signal objects.

Combined audit-only CUDA evidence so far:

- `sionna.phy.channel`: 17/17 current cases failed audit.
- `sionna.phy.mapping`: 14/14 current cases failed audit.
- `sionna.phy.signal`: 12/12 current cases failed audit.
- `sionna.phy.ofdm`: 33/33 OFDM-category cases failed audit.

## Next step

The updated umbrella PHY sweep after the OFDM expansion also found 75/75
current cases failed and 0 skipped. Standalone `sionna.phy.mimo` cases have
now been added and audited. The MIMO sweep found 8/8 failed cases and 0
skipped. The updated umbrella PHY sweep after the MIMO expansion found 83/83
then-current cases failed and 0 skipped. See
[`mimo-audit-findings.md`](mimo-audit-findings.md) and
[`phy-audit-findings.md`](phy-audit-findings.md). Standalone
`sionna.phy.fec` cases have now been added and audited. The FEC sweep found
30/31 failed cases, one passed standalone Trellis case, and 0 skipped. See
[`fec-audit-findings.md`](fec-audit-findings.md). The updated umbrella PHY
sweep across all 114 then-current dynamic cases found 113 failed cases, one passed
standalone Trellis case, and 0 skipped. Standalone NR dynamic cases have now
been added and audited: 12/12 failed and 0 skipped. The updated umbrella PHY
sweep across all 126 current dynamic cases found 125 failed cases, one passed
standalone Trellis case, and 0 skipped.
