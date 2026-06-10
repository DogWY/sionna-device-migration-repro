# Channel audit findings

## Context

This document records the current CUDA audit evidence for
`sionna.phy.channel` objects. It complements the broader PHY-wide plan in
[`docs/phy-audit-plan.md`](phy-audit-plan.md).

The important point is that the issue is not limited to the user-defined
`AWGNChannel` wrapper. Every currently implemented channel case shows stale
Sionna logical device state after a normal PyTorch `.to(cuda:1)` call.

## Commands

The first sweep used Sionna's global default construction behavior:

```bash
python run_repro.py run --category channel --device cuda:1 --no-probe-forward --json-report reports/channel-audit-cuda1.json
```

This captured several valid stale-device findings, but some cases failed during
construction with CUDA out-of-memory errors because Sionna's default device was
`cuda:0`.

The clean sweep constructed every object on CPU before calling `.to(cuda:1)`:

```bash
python run_repro.py run --category channel --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/channel-audit-cuda1-clean.json
```

This removed the construction-time OOM noise and produced a clean audit result
for all channel cases.

## Clean sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Sionna environment: `sdm`.
- Target device: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total channel cases: 17.
- Failed audit cases: 17.
- Skipped cases: 0.
- All cases include at least one stale logical device field.
- Optical channel cases also include non-registered derived tensor fields that
  remain on CPU.

## Case-level findings

| Case | Issue count | Main finding |
| --- | ---: | --- |
| `awgn` | 1 | `root._device_str` remains `cpu` |
| `wrapped-awgn-channel` | 1 | `root.awgn._device_str` remains `cpu` |
| `generate-flat-fading` | 1 | `root._device_str` remains `cpu` |
| `apply-flat-fading` | 2 | parent and internal `AWGN` logical devices remain `cpu` |
| `apply-ofdm` | 2 | parent and internal `AWGN` logical devices remain `cpu` |
| `apply-time` | 2 | parent and internal `AWGN` logical devices remain `cpu` |
| `flat-fading` | 4 | parent, generator, applier, and internal `AWGN` logical devices remain `cpu` |
| `kronecker-flat-fading` | 5 | parent, generator, spatial correlation, applier, and internal `AWGN` logical devices remain `cpu` |
| `kronecker-model` | 1 | `root._device_str` remains `cpu` |
| `per-column-model` | 1 | `root._device_str` remains `cpu` |
| `binary-memoryless` | 1 | `root._device_str` remains `cpu` |
| `binary-symmetric` | 1 | `root._device_str` remains `cpu` |
| `binary-erasure` | 1 | `root._device_str` remains `cpu` |
| `binary-z` | 1 | `root._device_str` remains `cpu` |
| `rayleigh-block-fading` | 1 | `root._device_str` remains `cpu` |
| `edfa` | 3 | `_device_str`, `_rho_n_ase`, and `_p_n_ase` remain `cpu` |
| `ssfm` | 4 | `_device_str`, `_dz`, `_rho_n`, and `_p_n_ase` remain `cpu` |

## Failure categories observed

### Stale logical device state

All channel cases show stale Sionna logical device state. This is the broad
pattern:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

This indicates that PyTorch `.to(cuda:1)` does not update Sionna's `_device_str`
field.

### Stale child module state

Composite blocks keep stale child state. Examples:

- `ApplyFlatFadingChannel._awgn._device_str`
- `ApplyOFDMChannel._awgn._device_str`
- `ApplyTimeChannel._awgn._device_str`
- `FlatFadingChannel._gen_chn._device_str`
- `FlatFadingChannel._app_chn._awgn._device_str`

This shows that PyTorch recursion into child modules is not enough when the
child module has Sionna-specific logical device state.

### Derived ordinary tensors not migrated

The optical blocks show a second issue type beyond `_device_str`:

- `EDFA._rho_n_ase`
- `EDFA._p_n_ase`
- `SSFM._dz`
- `SSFM._rho_n`
- `SSFM._p_n_ase`

These are ordinary tensor attributes derived from registered buffers. PyTorch
`.to()` does not migrate them because they are not registered buffers or
parameters.

## Interpretation

The clean channel sweep is strong evidence that the `.to(device)` migration
problem is systematic across `sionna.phy.channel`, not isolated to `AWGN`.

Two separate migration gaps are now documented:

- Sionna logical device state is not synchronized with PyTorch `.to()`.
- Derived tensor attributes that are not registered as buffers are not moved by
  PyTorch `.to()`.

## Next step

Dynamic repro coverage has now been extended beyond `sionna.phy.channel` for:

- `sionna.phy.mapping`
- `sionna.phy.signal`

The next CUDA-server step is to collect audit-only reports for those categories:

```bash
python run_repro.py run --category mapping --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/mapping-audit-cuda1.json
python run_repro.py run --category signal --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/signal-audit-cuda1.json
```

After that, extend dynamic coverage to standalone `sionna.phy.ofdm` classes.
