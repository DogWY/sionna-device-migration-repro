# PHY audit findings

## Context

This document records the latest collected umbrella CUDA audit evidence for the
dynamic `sionna.phy` case set. It combines the channel, mapping, signal,
standalone MIMO, standalone OFDM, and standalone FEC cases currently
implemented in this repository.

The important point is that 113 of 114 audited dynamic PHY cases show device
migration issues after a normal PyTorch `.to(cuda:1)` call when constructed on
CPU first. The only passing case is the standalone `fec-trellis` case, which is
a boundary case because it is not itself a Sionna `Block` with stale logical
device state.

## Command

The sweep constructed objects on CPU before calling `.to(cuda:1)` and disabled
forward probes so that the report focuses on post-migration object state:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Sionna environment: `sdm`.
- Target device: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total audited PHY dynamic cases: 114.
- Failed audit cases: 113.
- Passed audit cases: 1.
- Skipped cases: 0.
- Channel category cases: 17/17 failed.
- Mapping category cases: 14/14 failed.
- Signal category cases: 12/12 failed.
- MIMO category cases: 8/8 failed.
- OFDM category cases: 33/33 failed.
- FEC category cases: 30/31 failed; standalone `fec-trellis` passed.

The category counts overlap by one case: `apply-ofdm` belongs to both
`channel` and `ofdm`.

## Area summary

| Area | Failed cases | Main finding |
| --- | ---: | --- |
| `sionna.phy.channel` | 17/17 | stale `_device_str` across all cases; optical blocks also keep ordinary tensor attributes on CPU |
| `sionna.phy.mapping` | 14/14 | stale `_device_str`, child-module device state, and mapping lookup tensors |
| `sionna.phy.signal` | 12/12 | stale `_device_str`; filters and custom windows also keep `_coefficients` on CPU |
| `sionna.phy.mimo` | 8/8 | stale logical devices, child mapping state, and detector/list-to-LLR lookup tensors |
| `sionna.phy.ofdm` | 33/33 | stale logical devices across ResourceGrid, pilot, estimator, equalizer, detector, and precoding helpers; many OFDM index and lookup tensors remain on CPU |
| `sionna.phy.fec` | 30/31 | stale logical devices across Sionna FEC blocks; convolutional, LDPC, polar, turbo, and helper tensors remain on CPU |

## Failure categories observed

### Stale logical device state

Every current dynamic PHY case includes at least one stale Sionna logical device
field:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

This confirms that PyTorch `.to(cuda:1)` does not synchronize Sionna's separate
logical device state such as `_device_str`.

### Ordinary tensor attributes not migrated

Several Sionna objects store tensor state in ordinary attributes rather than
registered buffers or parameters. PyTorch `.to()` does not migrate these fields.
Examples include:

- Channel tensors: `_rho_n_ase`, `_p_n_ase`, `_dz`, and `_rho_n`.
- Mapping tensors: `_points`, `_bit_positions`, `_c0`, `_c1`, `_a`,
  `_bit_labels`, `_qam_ind`, `_pam1_ind`, `_pam2_ind`, and `_no_threshold`.
- Signal tensors: `_coefficients`.
- MIMO tensors: `_c0`, `_c1`, `_vecs`, `_vecs_ind`, `_c`, `_indices`,
  `_sym_pattern`, `_ind_pattern`, `_points`, `_es`, and `_no`.
- OFDM tensors: `_rg_type`, `_pilot_ind`, `_data_ind`,
  `_detection_desired_ind`, `_detection_undesired_ind`, `_stream_ind`,
  detector lookup tensors, and nested constellation points.
- FEC tensors: CRC lookup state, convolutional encoder generator polynomials,
  Trellis-derived next-state and output tables, interleaver indices,
  scrambler seeds, LDPC and polar code metadata, callback tracking state, and
  turbo encoder/decoder child state.

### Stale child module state

Composite PHY blocks keep stale state inside child Sionna objects. Examples
include:

- `FlatFadingChannel._gen_chn._device_str`
- `FlatFadingChannel._app_chn._awgn._device_str`
- `Mapper._constellation._points`
- `Demapper._logits2llrs._a`
- `QAMSource._mapper._bit_positions`
- `KBestDetector._list2llr._c0`
- `MMSEPICDetector._bit_demapper._logits2llrs._a`
- `ResourceGrid._pilot_pattern._device_str`
- `LMMSEEqualizer._removed_nulled_scs._device_str`
- `MaximumLikelihoodDetector._detector._vecs`
- `ConvEncoder._trellis.to_nodes`
- `LDPC5GDecoder._decoder._cn_schedule`
- `TurboEncoder._encoder1._trellis.op_by_tonode`

This shows that PyTorch recursion into child modules is insufficient when the
child module keeps Sionna-specific logical device fields and ordinary tensor
attributes.

## Interpretation

The current evidence supports a broader project conclusion:

`sionna.phy` objects do not reliably implement PyTorch `.to(device)` semantics
when they keep device-related state outside registered parameters and buffers.

The original wrapped `AWGN` failure is therefore a representative symptom, not
an isolated corner case.

## Current status

This document summarizes the latest collected 114-case umbrella PHY CUDA audit.
Focused area reports are available for channel, mapping/signal, MIMO, OFDM,
and FEC. The next coverage expansion target is standalone `sionna.phy.nr`
cases.

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```
