# NR audit findings

## Context

This document records CUDA audit evidence for the standalone `sionna.phy.nr`
dynamic cases. These cases cover all 12 P0 `sionna.phy.nr` classes identified
by the local inventory.

The goal is to isolate post-`.to(device)` object state for NR blocks, including
composite PUSCH and transport-block workflows.

## Command

The sweep constructed objects on CPU before calling `.to(device)` and disabled
forward probes:

```bash
CUDA_DEVICE=cuda:0
python run_repro.py run --category nr --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/nr-audit-cuda.json
```

The collected report used `cuda:1`; any visible CUDA device can be used.

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Target device in collected report: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total NR-category cases: 12.
- Failed audit cases: 12.
- Passed audit cases: 0.
- Skipped cases: 0.

## Case summary

| Case | Status | Issue count | Main finding |
| --- | --- | ---: | --- |
| `nr-layer-mapper` | failed | 1 | stale logical device state |
| `nr-layer-demapper` | failed | 2 | stale parent and child mapper logical device state |
| `nr-tb-encoder` | failed | 4 | stale parent, CRC, scrambler, and LDPC encoder logical device state |
| `nr-tb-decoder` | failed | 15 | stale TB encoder, LDPC decoder graph tensors, descrambler, CRC decoder, and output permutation |
| `nr-pusch-pilot-pattern` | failed | 1 | stale inherited pilot-pattern logical device state |
| `nr-pusch-precoder` | failed | 1 | stale logical device state despite registered precoding matrix buffer |
| `nr-pusch-transmitter` | failed | 17 | stale nested source, TB encoder, mapper, pilot, resource-grid, and grid-mapper state |
| `nr-pusch-ls-channel-estimator` | failed | 6 | stale estimator child blocks plus interpolation and pilot index tensors |
| `nr-pusch-receiver` | failed | 41 | stale nested OFDM, MIMO, layer-demapping, and TB decoding state |
| `nr-coded-awgn-channel` | failed | 1 | stale logical device state |
| `nr-mcs-decoder` | failed | 1 | stale logical device state |
| `nr-transport-block` | failed | 1 | stale logical device state |

## Failure categories observed

### Stale logical device state

All standalone NR cases include stale Sionna logical device state after CPU
construction followed by a normal PyTorch `.to(device)` call:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

This affects both simple helper blocks, such as `LayerMapper`, and composite
objects, such as `PUSCHTransmitter` and `PUSCHReceiver`.

### Ordinary tensor attributes not migrated

Several NR objects keep non-buffer tensor attributes that stay on CPU after
`.to(device)`. Representative examples include:

- `TBDecoder._decoder._cn_gather_idx`
- `TBDecoder._decoder._vn_mask`
- `TBDecoder._output_perm_inv`
- `PUSCHTransmitter._tb_encoder._target_coderate`
- `PUSCHTransmitter._mapper._constellation._points`
- `PUSCHTransmitter._resource_grid_mapper._rg_type`
- `PUSCHLSChannelEstimator._interpol._gather_ind`
- `PUSCHLSChannelEstimator._pilot_ind`
- `PUSCHReceiver._mimo_detector._data_ind`
- `PUSCHReceiver._tb_decoder._decoder._cn_scatter_idx`

### Composite child state

The strongest NR evidence comes from composite blocks. `PUSCHTransmitter` and
`PUSCHReceiver` recursively own mapping, OFDM, MIMO, FEC, and NR child blocks.
PyTorch recursion is not sufficient because those child blocks keep stale
Sionna logical device fields and ordinary tensor attributes.

Representative nested paths include:

- `PUSCHTransmitter._tb_encoder._encoder._device_str`
- `PUSCHTransmitter._mapper._constellation._points`
- `PUSCHTransmitter._resource_grid_mapper._rg_type`
- `PUSCHReceiver._channel_estimator._interpol._device_str`
- `PUSCHReceiver._mimo_detector._detector._constellation._points`
- `PUSCHReceiver._tb_decoder._decoder._vn_gather_idx`

## Interpretation

The NR sweep confirms that the same `.to(device)` migration issue appears in
the 5G NR subpackage. The problem is not limited to low-level channel,
mapping, OFDM, MIMO, signal, or FEC objects.

The updated umbrella PHY sweep now covers 126 dynamic cases. It found 125
failed cases, one passed standalone `fec-trellis` boundary case, and no
skipped cases.
