# Sionna device migration repro

This repository supports an upstream Sionna issue about PyTorch device
migration semantics in `sionna.phy`.

The core repro is:

1. construct a Sionna PHY object on CPU;
2. call the normal PyTorch `.to(cuda_device)`;
3. inspect logical device fields and tensor attributes;
4. optionally run a small forward probe with CUDA inputs.

The repository does not patch Sionna and does not reimplement communication
models. It only collects reproducible evidence for whether Sionna PHY objects
honor the selected target device after `.to(device)`.

## Issue Summary

Some `sionna.phy` objects keep Sionna-specific logical device state and ordinary
tensor attributes outside PyTorch's normal parameter and buffer migration path.
After CPU construction followed by `.to("cuda:x")`, fields such as
`_device_str` can remain on CPU. Forward probes can then either return CPU
tensors for CUDA inputs or raise mixed CPU/CUDA runtime errors.

The minimal observed `AWGN` case shows both symptoms:

```text
== awgn [failed] ==
AWGN channel; exposes stale Sionna logical device even without registered tensors.
Found 1 device migration issue(s):
- root._device_str: expected=cuda:1, actual=cpu, kind=logical-device, Sionna logical device field
Forward output device issue(s):
Found 1 device migration issue(s):
- root: expected=cuda:1, actual=cpu, kind=tensor, dtype=torch.complex64, shape=(4, 8)
```

`cuda:1` is only the device used in the collected evidence. Any visible CUDA
device, such as `cuda:0`, can be used when rerunning the repro.

## Quick Start

Install Sionna and use a CUDA-enabled PyTorch build that matches your system:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "sionna==2.0.1" pytest
```

No package installation is required for this repository itself.

Choose any visible CUDA device:

```bash
CUDA_DEVICE=cuda:0
```

Collect environment diagnostics:

```bash
python run_repro.py env
```

Run the minimal AWGN repro:

```bash
python run_repro.py run --case awgn --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

Run the user-style wrapper repro:

```bash
python run_repro.py run --case wrapped-awgn-channel --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

Run the full PHY object-state audit:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda.json
```

Run the PHY forward-probe sweep:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-fail --json-report reports/phy-forward-cuda.json
```

Build a static inventory of `sionna.phy` classes:

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

For a single-file reproduction of the wrapper pattern:

```bash
python examples/wrapped_awgn_channel_to_cuda.py "$CUDA_DEVICE"
```

## Expected Behavior

After calling `.to(cuda_device)` on a Sionna PHY module:

- internal logical device state should match the selected CUDA device;
- persistent tensor state should be migrated or otherwise follow the selected
  CUDA device;
- forward outputs for CUDA inputs should stay on the selected CUDA device.

## Actual Behavior Captured

Collected evidence from an Ubuntu multi-GPU server with Sionna 2.0.1,
Sionna RT 2.0.1, Python 3.12.13, and PyTorch 2.11.0+cu128 shows:

- Minimal `awgn`: `_device_str` remains `cpu` after `.to("cuda:1")`, and
  forward returns a CPU tensor for a CUDA input.
- Wrapped `AWGNChannel`: nested `root.awgn._device_str` remains `cpu` after the
  parent PyTorch module is moved.
- Audit-only PHY sweep: 125 of 126 dynamic PHY cases failed; the only passing
  case was standalone `fec-trellis`.
- Forward-probe PHY sweep: 18 cases raised forward exceptions; 30 cases
  completed forward execution but returned 33 tensors on CPU.

Affected areas include:

- `sionna.phy.channel`
- `sionna.phy.mapping`
- `sionna.phy.signal`
- `sionna.phy.mimo`
- `sionna.phy.ofdm`
- `sionna.phy.fec`
- `sionna.phy.nr`

## Evidence Documents

- [docs/upstream-repro-note.md](docs/upstream-repro-note.md): issue-ready
  evidence note.
- [docs/phy-audit-findings.md](docs/phy-audit-findings.md): umbrella
  object-state audit summary.
- [docs/forward-probe-findings.md](docs/forward-probe-findings.md): runtime
  impact summary.
- [docs/channel-audit-findings.md](docs/channel-audit-findings.md): channel
  case details.
- [docs/mapping-signal-audit-findings.md](docs/mapping-signal-audit-findings.md):
  mapping and signal case details.
- [docs/ofdm-audit-findings.md](docs/ofdm-audit-findings.md): OFDM case
  details.
- [docs/mimo-audit-findings.md](docs/mimo-audit-findings.md): MIMO case
  details.
- [docs/fec-audit-findings.md](docs/fec-audit-findings.md): FEC case details.
- [docs/nr-audit-findings.md](docs/nr-audit-findings.md): NR case details.
- [docs/phy-audit-plan.md](docs/phy-audit-plan.md): audit methodology.
- [docs/project-plan.md](docs/project-plan.md): project scope.

## Project Layout

```text
.
+-- run_repro.py
+-- docs/
+-- examples/
+-- scripts/
+-- src/sionna_device_migration_repro/
+-- tests/
`-- tools/
```

## Notes

By default, repro objects are constructed on CPU before PyTorch `.to(device)` is
called. This avoids unrelated failures caused by Sionna's global default device
being a different CUDA device during construction. To intentionally preserve
Sionna's global default construction behavior, use `--build-device default`.

Use `--no-fail` when collecting JSON reports from broad sweeps. Device-migration
failures are expected evidence, and without `--no-fail` the command exits with a
non-zero status before later chained commands can run.
