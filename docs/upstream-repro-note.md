# Upstream repro note

## Suggested title

`sionna.phy` PyTorch modules do not consistently migrate logical device and
ordinary tensor state with `.to(device)`

## Summary

Sionna PHY objects are PyTorch modules, but several of them keep device-related
state outside PyTorch parameters and registered buffers. After constructing a
PHY object on CPU and then calling the normal PyTorch `.to("cuda:1")`, internal
Sionna logical device fields such as `_device_str` can remain on CPU. Ordinary
tensor attributes can also remain on CPU.

This leads to three user-visible outcomes:

- post-migration object state still reports `cpu` instead of the requested CUDA
  device;
- forward execution returns tensors on CPU even though the input tensors and
  target module device are CUDA;
- forward execution raises mixed-device errors when stale CPU state is used
  together with CUDA inputs.

The issue was first observed in a user-defined `torch.nn.Module` wrapper around
`sionna.phy.channel.AWGN`, but a broader sweep shows the same pattern across
many `sionna.phy` areas.

## Reproduction repository

Repository:

```text
https://github.com/DogWY/sionna-device-migration-repro
```

The repository does not patch Sionna and does not reimplement communication
models. It only constructs small Sionna PHY objects, calls PyTorch `.to(device)`,
recursively audits device state, and optionally runs small forward probes.

## Environment used for collected evidence

- OS/runtime: Ubuntu server, Linux `5.4.0-81-generic`, x86_64.
- GPU topology visible to PyTorch: 4 x NVIDIA GeForce RTX 4090.
- CUDA visibility environment: `CUDA_VISIBLE_DEVICES` unset,
  `NVIDIA_VISIBLE_DEVICES` unset.
- Conda environment: `sdm`.
- Python: `3.12.13`.
- Sionna: `2.0.1`.
- Sionna RT: `2.0.1`.
- PyTorch: `2.11.0+cu128`.
- PyTorch CUDA runtime: `12.8`.
- Target CUDA device used in the reports: `cuda:1`.
- Construction device used in the controlled sweeps: `cpu`.

For a full version dump on the CUDA machine, run:

```bash
python run_repro.py env
```

The `sionna.phy` inventory was also generated on the same server:

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

## Minimal AWGN object-state repro

From a fresh checkout with Sionna installed:

```bash
conda activate sdm
python run_repro.py run --case awgn --device cuda:1 --build-device cpu --no-fail
```

Observed result:

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

- After `awgn.to("cuda:1")`, the object should use `cuda:1` as its logical
  device.
- If the forward input is on `cuda:1`, the forward output should also be on
  `cuda:1`.

Actual behavior:

- The logical Sionna device remains `cpu`.
- The forward output is returned on CPU.

## User-style wrapper repro

The original user pattern is a normal PyTorch wrapper that owns a Sionna AWGN
child module:

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
channel.to("cuda:1")
x = torch.randn(2, 4, 8, device="cuda:1")
y = channel(x)
```

In the repro repository, this wrapper is available as:

```bash
python run_repro.py run --case wrapped-awgn-channel --device cuda:1 --build-device cpu --no-fail
```

The controlled audit reports the stale nested child state:

```text
root.awgn._device_str: expected=cuda:1, actual=cpu, kind=logical-device
```

When Sionna's global default device is left unchanged, the same failure mode can
also appear as `actual=cuda:0` while the wrapper is moved to `cuda:1`. The key
problem is not the specific source device. The key problem is that the child
Sionna module's logical device is not synchronized by PyTorch `.to(target)`.

## Full PHY audit commands

Object-state audit across the current 126-case dynamic PHY case set:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

Forward-probe sweep across cases with safe minimal inputs:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-fail --json-report reports/phy-forward-cuda1.json
```

## Full PHY audit result summary

The audit-only sweep found:

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

The forward-probe sweep found:

- Forward exceptions: 18 cases.
- Wrong-device forward outputs: 30 cases.
- Wrong-device returned tensors: 33 tensors.

Representative forward failures include:

| Case | Symptom |
| --- | --- |
| `apply-time` | CPU index tensor used with CUDA input tensor |
| `kronecker-flat-fading` | matrix multiplication mixed CPU and CUDA tensors |
| `binary-memoryless` | CPU/CUDA tensor mix |
| `pam2qam` | CPU lookup indices used with CUDA input |
| `resource-grid-demapper` | CPU gather/index state used with CUDA input |
| `ls-channel-estimator` | CPU index-select state used with CUDA input |
| `lmmse-equalizer` | CPU index-select state used with CUDA input |

Representative wrong-device outputs include:

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
  after `.to("cuda:1")`;
- stale logical device state in child Sionna modules after a parent PyTorch
  wrapper is moved;
- ordinary tensor attributes not migrated because they are not registered
  buffers or parameters;
- forward-created tensors using stale `self.device`;
- successful forward execution with returned tensors on the wrong device;
- runtime errors from operations that combine stale CPU tensors with CUDA
  inputs.

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

The expected fix is not necessarily tied to this repro repository. Any upstream
solution that makes Sionna PHY objects honor PyTorch module migration semantics
would address the issue. Possible directions include:

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
