# Mapping and signal audit findings

## Context

This document records CUDA audit evidence for the current
`sionna.phy.mapping` and `sionna.phy.signal` dynamic case sets. It complements
the channel findings in [`channel-audit-findings.md`](channel-audit-findings.md)
and the broader plan in [`phy-audit-plan.md`](phy-audit-plan.md).

The important point is that the stale-device pattern is not limited to channel
objects. All currently implemented mapping and signal cases show device
migration issues after a normal PyTorch `.to(cuda:1)` call.

## Commands

Both sweeps constructed objects on CPU before calling `.to(cuda:1)` and disabled
forward probes so that the reports focus on post-migration object state:

```bash
python run_repro.py run --category mapping --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/mapping-audit-cuda1.json
python run_repro.py run --category signal --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/signal-audit-cuda1.json
```

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Sionna environment: `sdm`.
- Target device: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total mapping cases: 14.
- Failed mapping audit cases: 14.
- Total signal cases: 12.
- Failed signal audit cases: 12.
- All mapping and signal cases include stale Sionna logical device state.
- Many mapping and signal cases also include ordinary tensor attributes that
  remain on CPU after `.to(cuda:1)`.

## Mapping case-level findings

| Case | Issue count | Main finding |
| --- | ---: | --- |
| `binary-source` | 1 | `root._device_str` remains `cpu` |
| `constellation-qam` | 2 | `_device_str` and `_points` remain `cpu` |
| `mapper-qam` | 4 | parent, constellation, points, and bit positions remain `cpu` |
| `demapper-qam-app` | 8 | parent, constellation, logits converter, lookup tensors, and threshold remain `cpu` |
| `symbol-demapper-qam` | 3 | parent, constellation, and points remain `cpu` |
| `llrs2symbol-logits` | 2 | `_device_str` and `_a` remain `cpu` |
| `symbol-logits2llrs` | 4 | `_device_str`, `_c0`, `_c1`, and `_a` remain `cpu` |
| `symbol-inds2bits` | 2 | `_device_str` and `_bit_labels` remain `cpu` |
| `symbol-logits2moments` | 3 | parent, constellation, and points remain `cpu` |
| `pam2qam` | 2 | `_device_str` and `_qam_ind` remain `cpu` |
| `qam2pam` | 3 | `_device_str`, `_pam1_ind`, and `_pam2_ind` remain `cpu` |
| `pam-source` | 6 | source, child source, mapper, constellation, points, and bit positions remain `cpu` |
| `qam-source` | 6 | source, child source, mapper, constellation, points, and bit positions remain `cpu` |
| `symbol-source` | 6 | source, child source, mapper, constellation, points, and bit positions remain `cpu` |

## Signal case-level findings

| Case | Issue count | Main finding |
| --- | ---: | --- |
| `upsampling` | 1 | `root._device_str` remains `cpu` |
| `downsampling` | 1 | `root._device_str` remains `cpu` |
| `window-base` | 1 | `root._device_str` remains `cpu` |
| `custom-window` | 2 | `_device_str` and `_coefficients` remain `cpu` |
| `hamming-window` | 1 | `root._device_str` remains `cpu` |
| `hann-window` | 1 | `root._device_str` remains `cpu` |
| `blackman-window` | 1 | `root._device_str` remains `cpu` |
| `filter-base` | 1 | `root._device_str` remains `cpu` |
| `custom-filter` | 2 | `_device_str` and `_coefficients` remain `cpu` |
| `sinc-filter` | 2 | `_device_str` and `_coefficients` remain `cpu` |
| `raised-cosine-filter` | 2 | `_device_str` and `_coefficients` remain `cpu` |
| `root-raised-cosine-filter` | 2 | `_device_str` and `_coefficients` remain `cpu` |

## Failure categories observed

### Stale logical device state

Every mapping and signal case shows stale Sionna logical device state:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

This is the same logical-device failure pattern already observed for
`sionna.phy.channel`.

### Ordinary tensor attributes not migrated

Several objects store tensor state in ordinary attributes rather than registered
buffers or parameters. PyTorch `.to()` does not migrate these fields. Examples:

- Mapping tensors: `_points`, `_bit_positions`, `_c0`, `_c1`, `_a`,
  `_bit_labels`, `_qam_ind`, `_pam1_ind`, `_pam2_ind`, and `_no_threshold`.
- Signal tensors: `_coefficients`.

### Stale child module state

Composite mapping blocks keep stale state inside child Sionna objects. Examples:

- `Mapper._constellation._device_str`
- `Demapper._constellation._points`
- `Demapper._logits2llrs._a`
- `PAMSource._binary_source._device_str`
- `QAMSource._mapper._bit_positions`
- `SymbolSource._mapper._constellation._points`

This confirms that PyTorch recursion into child modules is insufficient when
the child modules keep Sionna-specific logical device fields and ordinary tensor
attributes.

## Interpretation

The clean mapping and signal sweeps are strong evidence that the `.to(device)`
migration problem is systematic across multiple `sionna.phy` areas:

- `sionna.phy.channel`: 17/17 current cases failed audit.
- `sionna.phy.mapping`: 14/14 current cases failed audit.
- `sionna.phy.signal`: 12/12 current cases failed audit.

The issue is therefore broader than the original wrapped `AWGN` failure.

## Next step

The umbrella PHY audit report has now also been collected. See
[`phy-audit-findings.md`](phy-audit-findings.md).

Standalone `sionna.phy.ofdm` dynamic cases have now been added and audited. The
clean OFDM sweep found 33/33 OFDM-category cases failed and 0 skipped. The
updated umbrella PHY sweep also found 75/75 current cases failed and 0 skipped.
Standalone `sionna.phy.mimo` cases have now been added and audited. The MIMO
sweep found 8/8 failed cases and 0 skipped. The updated umbrella PHY sweep
after the MIMO expansion found 83/83 current cases failed and 0 skipped. See
[`mimo-audit-findings.md`](mimo-audit-findings.md) and
[`phy-audit-findings.md`](phy-audit-findings.md). The next coverage target
should be selected from standalone `sionna.phy.fec` or `sionna.phy.nr`.

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```
