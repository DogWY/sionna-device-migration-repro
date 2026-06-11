# Upstream repro note

## Suggested title

`sionna.phy` modules do not consistently migrate logical device and ordinary
tensor state with `.to(device)`

## Summary

Some `sionna.phy` objects appear to keep Sionna-specific logical device state
and ordinary tensor attributes outside PyTorch's normal parameter and buffer
migration path.

After constructing an object on CPU and then calling `.to(cuda_device)`, fields
such as `_device_str` can remain on CPU. In forward probes, this can either
return CPU tensors for CUDA inputs or raise mixed CPU/CUDA runtime errors.

The issue was first observed in a user-defined `torch.nn.Module` wrapper around
`sionna.phy.channel.AWGN`. A broader sweep of dynamic `sionna.phy` cases shows
the same pattern across channel, mapping, signal, MIMO, OFDM, FEC, and NR
objects.

## Reproduction repository

```text
https://github.com/DogWY/sionna-device-migration-repro
```

The repository does not patch Sionna and does not reimplement communication
models. It constructs small Sionna PHY objects, calls PyTorch `.to(device)`,
recursively audits device state, and optionally runs small forward probes.

## Minimal AWGN repro

Run on any visible CUDA device:

```bash
CUDA_DEVICE=cuda:0
python run_repro.py run --case awgn --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

Observed result from the collected `cuda:1` run:

```text
== awgn [failed] ==
AWGN channel; exposes stale Sionna logical device even without registered tensors.
Found 1 device migration issue(s):
- root._device_str: expected=cuda:1, actual=cpu, kind=logical-device, Sionna logical device field
Forward output device issue(s):
Found 1 device migration issue(s):
- root: expected=cuda:1, actual=cpu, kind=tensor, dtype=torch.complex64, shape=(4, 8)
```

Expected behavior:

- After `awgn.to("cuda:x")`, the object should use the selected CUDA device as
  its logical device.
- If the forward input is on `cuda:x`, the forward output should also be on
  that selected CUDA device.

Actual behavior:

- `root._device_str` remains `cpu`.
- The forward output is returned on CPU.

## User-style wrapper repro

The same pattern appears when `AWGN` is used as a child module inside a normal
PyTorch wrapper:

```python
import torch
from torch import nn
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no


def power_normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    power = torch.mean(x**2, dim=(1, 2), keepdim=True)
    return x / torch.sqrt(power), power


class AWGNChannel(nn.Module):
    def __init__(self, snr: float | None = None):
        super().__init__()
        self.snr = snr
        self.awgn = AWGN()

    def forward(self, inputs: torch.Tensor, snr: float | None = None) -> torch.Tensor:
        if snr is None and self.snr is None:
            raise ValueError("SNR must be provided.")
        if snr is not None:
            self.snr = snr

        batch, length, dim = inputs.shape
        x, pwr = power_normalize(inputs)
        x = x.reshape(batch, length, -1, 2)
        x_complex = x[..., 0] + 1j * x[..., 1]
        noise_std = ebnodb2no(self.snr, 1, 1)
        y_complex = self.awgn(x_complex, noise_std)
        y = torch.stack((y_complex.real, y_complex.imag), dim=-1)
        return y.reshape(batch, length, dim) * torch.sqrt(pwr)


channel = AWGNChannel(snr=10)
device = "cuda:0"  # choose any visible CUDA device
channel.to(device)
x = torch.randn(2, 4, 8, device=device)
y = channel(x)
```

Repository command:

```bash
python run_repro.py run --case wrapped-awgn-channel --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

The controlled audit from the collected `cuda:1` run reports stale nested child
state:

```text
root.awgn._device_str: expected=cuda:1, actual=cpu, kind=logical-device
```

When Sionna's global default device is left unchanged, the same failure mode can
also appear as `actual=cuda:0` while the wrapper is moved to `cuda:1`. The key
problem is not the specific source device. The key problem is that the child
Sionna module's logical device is not synchronized by PyTorch `.to(target)`.

## Broader PHY sweep

Object-state audit across the current dynamic PHY case set:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda.json
```

Forward-probe sweep across cases with safe minimal inputs:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-fail --json-report reports/phy-forward-cuda.json
```

Audit-only result:

- Total dynamic PHY cases: 126.
- Failed audit cases: 125.
- Passed audit cases: 1.
- Skipped cases: 0.
- Only passed case: standalone `fec-trellis`.

Category-level results:

| Area | Result |
| --- | ---: |
| `sionna.phy.channel` | 17/17 failed |
| `sionna.phy.mapping` | 14/14 failed |
| `sionna.phy.signal` | 12/12 failed |
| `sionna.phy.mimo` | 8/8 failed |
| `sionna.phy.ofdm` | 33/33 failed |
| `sionna.phy.fec` | 30/31 failed |
| `sionna.phy.nr` | 12/12 failed |

Forward-probe result:

- Forward exceptions: 18 cases.
- Wrong-device forward outputs: 30 cases.
- Wrong-device returned tensors: 33 tensors.

Representative forward exceptions:

| Case | Symptom |
| --- | --- |
| `apply-time` | CPU index tensor used with CUDA input tensor |
| `kronecker-flat-fading` | matrix multiplication mixed CPU and CUDA tensors |
| `binary-memoryless` | CPU/CUDA tensor mix |
| `pam2qam` | CPU lookup indices used with CUDA input |
| `resource-grid-demapper` | CPU gather/index state used with CUDA input |
| `ls-channel-estimator` | CPU index-select state used with CUDA input |
| `lmmse-equalizer` | CPU index-select state used with CUDA input |

Representative wrong-device outputs:

| Case | Wrong-device output |
| --- | --- |
| `awgn` | returned CPU tensor |
| `generate-flat-fading` | returned CPU tensor |
| `mapper-qam` | returned CPU tensor |
| `symbol-logits2moments` | returned two CPU tensors |
| `upsampling` | returned CPU tensor |
| `custom-filter` | returned CPU tensor |
| `resource-grid-mapper` | returned CPU tensor |
| `ofdm-modulator` | returned CPU tensor |

## Failure categories

The failures observed in the sweep fall into these categories:

- stale Sionna logical device state, for example `_device_str` remains `cpu`
  after `.to("cuda:x")`;
- stale logical device state in child Sionna modules after a parent PyTorch
  wrapper is moved;
- ordinary tensor attributes not migrated because they are not registered
  buffers or parameters;
- forward-created tensors using stale `self.device`;
- successful forward execution with returned tensors on the wrong device;
- runtime errors from operations that combine stale CPU tensors with CUDA
  inputs.

## Environment used for collected evidence

- OS/runtime: Ubuntu server, Linux `5.4.0-81-generic`, x86_64.
- GPU topology visible to PyTorch: 4 x NVIDIA GeForce RTX 4090.
- CUDA visibility environment: `CUDA_VISIBLE_DEVICES` unset,
  `NVIDIA_VISIBLE_DEVICES` unset.
- Python: `3.12.13`.
- Sionna: `2.0.1`.
- Sionna RT: `2.0.1`.
- PyTorch: `2.11.0+cu128`.
- PyTorch CUDA runtime: `12.8`.
- Target CUDA device used in the collected reports: `cuda:1`.
- Construction device used in the controlled sweeps: `cpu`.

The `cuda:1` index is not required. It was used only because that GPU was idle
on the test server. Readers can replace it with any visible CUDA device.

For a full version dump on a repro machine, run:

```bash
python run_repro.py env
```

## Inventory summary

The `sionna.phy` inventory was generated on the same server:

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

Inventory summary:

- Classes under `sionna.phy`: 176.
- Import errors: 0.
- Risk counts: P0 = 157, P1 = 2, P2 = 17.
- P0 classes by area: `channel` = 40, `ofdm` = 36, `fec` = 30,
  `mapping` = 14, `signal` = 12, `nr` = 12, `mimo` = 8, `utils` = 3,
  plus core `Object` and `Block`.

## Why this appears to be a `.to(device)` integration issue

PyTorch `.to(device)` migrates parameters and registered buffers and recursively
visits child modules. It does not know how to update separate device metadata
stored by a library unless that metadata is wired into the module's migration
logic.

The reports suggest that many Sionna PHY objects keep device state in fields
such as `_device_str`, and many also keep tensor state in ordinary attributes.
These fields can remain on the construction device after normal PyTorch
migration, even though users naturally expect a `torch.nn.Module` to be usable
on the requested target device after `.to(target)`.

## Expected maintainer-facing outcome

The expected fix is not tied to this repro repository. Any upstream solution
that makes Sionna PHY objects honor PyTorch module migration semantics would
address the issue. Possible directions include:

- updating Sionna logical device state when `.to(device)` is called;
- registering persistent tensor state as buffers where appropriate;
- deriving runtime tensors from input tensor devices where possible;
- adding regression tests for CPU construction followed by `.to("cuda:x")` on
  CUDA inputs.

## Related local evidence documents

- [`phy-audit-findings.md`](phy-audit-findings.md)
- [`forward-probe-findings.md`](forward-probe-findings.md)
- [`channel-audit-findings.md`](channel-audit-findings.md)
- [`mapping-signal-audit-findings.md`](mapping-signal-audit-findings.md)
- [`ofdm-audit-findings.md`](ofdm-audit-findings.md)
- [`mimo-audit-findings.md`](mimo-audit-findings.md)
- [`fec-audit-findings.md`](fec-audit-findings.md)
- [`nr-audit-findings.md`](nr-audit-findings.md)
