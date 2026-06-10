# Sionna PHY device migration audit plan

## Objective

Audit `sionna.phy` for PyTorch `.to(device)` migration problems. The concrete
bug already observed is that a user-defined wrapper moved to `cuda:1` still
contains an internal Sionna `AWGN` block whose logical device state remains
`cuda:0`.

This project should determine whether that behavior is isolated to `AWGN` and
channel wrappers, or whether it is a broader pattern across stateful objects in
`sionna.phy`.

## Scope

In scope:

- Python package namespace: `sionna.phy`.
- Classes that inherit from `torch.nn.Module`, `sionna.phy.Object`, or
  `sionna.phy.Block`.
- Classes with a `device` constructor argument or a `device` property.
- Classes that store `_device_str` or similar logical device state.
- Classes that call `register_buffer()` or hold ordinary `torch.Tensor`
  attributes.
- Classes that create tensors during `forward`, `call`, or `__call__` using
  `self.device`.
- Composite PHY blocks that own child Sionna blocks.

Out of scope:

- `sionna.rt` and all RT-specific functionality.
- Pure functions unless they are needed to build a minimal repro case.
- Rewriting or patching Sionna source code.
- Benchmarking performance or numerical accuracy.

## Key hypothesis

PyTorch's default `nn.Module.to(device)` migrates parameters and registered
buffers, and it recursively visits child modules. It does not know how to update
Sionna's separate logical device state such as `_device_str`.

If a Sionna PHY object later creates tensors from `self.device`, those tensors
may be created on the stale logical device instead of the requested target
device. This can produce:

- audit-only mismatches, such as `expected=cuda:1, actual=cuda:0`;
- mixed-device forward failures;
- output tensors on the wrong device;
- failures only when the target device differs from Sionna's default or global
  config device.

## Evidence already captured

The user-side wrapper `AWGNChannel` was moved with:

```bash
python run_repro.py run --case wrapped-awgn-channel --device cuda:1
```

Observed audit issue:

```text
root.awgn._device_str: expected=cuda:1, actual=cuda:0
```

This is enough to justify a broader `sionna.phy` audit. Further work should not
focus on comparing different GPU indices. It should focus on which PHY classes
exhibit the same stale logical device pattern.

The first Ubuntu channel sweep also showed that some cases failed during object
construction with CUDA out-of-memory errors because Sionna's global default
device was `cuda:0`. The repro runner now defaults to constructing cases on
CPU before calling `.to(target_device)`. Use `--build-device default` only when
the goal is to reproduce Sionna's global default construction behavior.

A clean channel sweep with `--build-device cpu` found stale device state in all
17 implemented channel cases. See
[`docs/channel-audit-findings.md`](channel-audit-findings.md) for the case-level
summary.

Clean mapping and signal sweeps with `--build-device cpu` found stale device
state in all 26 implemented non-channel cases. See
[`docs/mapping-signal-audit-findings.md`](mapping-signal-audit-findings.md) for
the case-level summary.

The first umbrella PHY sweep before standalone OFDM expansion found stale
device state across all 43 dynamic cases. The current umbrella PHY sweep after
standalone MIMO expansion found stale device state across all 83 dynamic cases.
See [`docs/phy-audit-findings.md`](phy-audit-findings.md) for the current
PHY-level summary.

The clean standalone OFDM sweep found stale device state in all 33
OFDM-category cases. See
[`docs/ofdm-audit-findings.md`](ofdm-audit-findings.md).

The clean standalone MIMO sweep found stale device state in all 8 MIMO-category
cases. See [`docs/mimo-audit-findings.md`](mimo-audit-findings.md).

## Audit workflow

### Phase 1: inventory `sionna.phy`

Build an inventory script that imports and scans `sionna.phy` recursively.

Proposed command:

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

For each discovered class, record:

- module name;
- class name;
- source file path;
- base classes;
- constructor signature;
- whether it is a `torch.nn.Module`;
- whether it inherits Sionna `Object` or `Block`;
- whether the constructor accepts `device`;
- whether source text contains `_device_str`;
- whether source text contains `register_buffer`;
- whether source text contains `self.device`;
- whether source text contains `torch.tensor`, `torch.zeros`, `torch.ones`,
  random tensor factories, or tensor conversions using `device=`;
- whether the class is abstract or directly instantiable.

### Phase 2: risk classification

Classify inventory entries into priority levels:

- P0: must test dynamically.
  Stateful PHY objects with Sionna logical device state, child modules,
  registered buffers, or tensor creation through `self.device`.
- P1: audit-only first.
  Objects that are likely relevant but require complex domain-specific inputs
  for forward execution.
- P2: static inspection only.
  Abstract classes, pure configuration/data holders, or classes without device
  state or tensor creation.
- Excluded: RT-only objects and non-PHY objects.

Expected high-risk areas:

- `sionna.phy.channel`
- `sionna.phy.ofdm`
- `sionna.phy.mimo`
- `sionna.phy.mapping`
- `sionna.phy.signal`
- `sionna.phy.fec`
- `sionna.phy.nr`

### Phase 3: generate minimal dynamic repro cases

For P0 classes, add minimal cases to the existing `run_repro.py` workflow.

Initial command for clean state migration evidence:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/phy-audit-cuda1.json
```

Then run forward probes only for objects with small, reliable inputs:

```bash
python run_repro.py run --category phy --device cuda:1 --json-report reports/phy-forward-cuda1.json
```

Dynamic cases should:

- construct the object on a controlled build device, CPU by default;
- call `.to(target_device)` through PyTorch;
- audit registered tensors, ordinary tensors, child modules, and logical device
  fields;
- optionally run one forward/call probe using small inputs already placed on
  `target_device`;
- emit JSON records that distinguish object-state issues, forward exceptions,
  and output-device mismatches.

### Phase 4: failure taxonomy

Group failures by cause:

- stale logical device state, e.g. `_device_str` remains on another device;
- child module stale state after parent `.to()`;
- registered buffers migrated but derived ordinary tensor state did not;
- forward creates new tensors on stale `self.device`;
- output is on the wrong device without an exception;
- forward fails because operands are on different devices.

### Phase 5: communication artifacts

Produce artifacts that can be shared with maintainers or collaborators:

- `reports/phy-inventory.json`
- `reports/phy-audit-cuda1.json`
- `reports/phy-forward-cuda1.json`
- a concise summary table listing affected classes, issue paths, and failure
  mode;
- a minimal standalone snippet for the most representative failure.

## Implementation checklist

- [x] Add `tools/inspect_phy_inventory.py`.
- [ ] Generate `reports/phy-inventory.json` on the Ubuntu CUDA server.
- [x] Add category support for `phy` once non-channel PHY cases are included.
- [x] Expand dynamic cases beyond `sionna.phy.channel` for
  `sionna.phy.mapping` and `sionna.phy.signal`.
- [x] Keep `channel` cases as a subset category for focused reruns.
- [x] Run audit-only channel sweep on the CUDA server.
- [x] Run audit-only PHY sweep on the CUDA server.
- [x] Run audit-only mapping and signal sweeps on the CUDA server.
- [x] Add standalone `sionna.phy.ofdm` dynamic cases.
- [x] Run first audit-only OFDM sweep on the CUDA server.
- [x] Add construction-time Sionna `config.device` guard for `--build-device`.
- [x] Rerun audit-only OFDM sweep after the construction-device fix.
- [x] Run updated umbrella PHY sweep after the standalone OFDM expansion.
- [x] Add standalone `sionna.phy.mimo` dynamic cases.
- [x] Run audit-only MIMO sweep on the CUDA server.
- [x] Rerun umbrella PHY sweep after the standalone MIMO expansion.
- [ ] Run forward PHY sweep for safe cases.
- [x] Summarize affected classes and failure modes for the current dynamic case
  set.
- [ ] Prepare a short upstream-facing repro note.

## Current repository status

Implemented already:

- repository-local entrypoint: `python run_repro.py`;
- static PHY inventory script: `tools/inspect_phy_inventory.py`;
- recursive device audit helper;
- JSON report output;
- `channel` category with multiple `sionna.phy.channel` cases;
- `mapping` category with source, constellation, mapper, demapper, logits, and
  symbol-index conversion cases;
- `signal` category with resampling, window, and filter cases;
- `mimo` category with stream management, list-to-LLR helpers, and standalone
  detector cases;
- `ofdm` category with resource-grid, pilot, mapper/demapper, modem, channel
  estimator, equalizer, detector, and precoding cases;
- `phy` umbrella category across the current dynamic case set;
- primary user wrapper repro: `wrapped-awgn-channel`;
- CPU smoke validation for the current channel, mapping, signal, and PHY case
  sets.
- CPU smoke validation for the current OFDM case set.
- CPU smoke validation for the current MIMO case set.
- CUDA audit-only evidence for `channel`, `mapping`, and `signal` dynamic
  cases.
- CUDA audit-only evidence for the first umbrella `phy` dynamic case set before
  standalone OFDM cases were added.
- Clean CUDA audit-only evidence for standalone `ofdm` cases.
- CUDA audit-only evidence for the current umbrella `phy` dynamic case set
  after standalone OFDM cases were added.
- Clean CUDA audit-only evidence for standalone `mimo` cases.
- CUDA audit-only evidence for the current umbrella `phy` dynamic case set
  after standalone MIMO cases were added.

Local inventory smoke result with Sionna 2.0.1 in the `sdm` environment:

- total classes under `sionna.phy`: 176;
- import errors: 0;
- risk counts: P0 = 157, P1 = 2, P2 = 17;
- P0 classes by area: `channel` = 40, `ofdm` = 36, `fec` = 30,
  `mapping` = 14, `signal` = 12, `nr` = 12, `mimo` = 8, `utils` = 3,
  plus core `Object`/`Block`.

Not implemented yet:

- dynamic repro cases for standalone `fec` and `nr` classes;
- forward-probe CUDA reports for safe cases.

## Recommended next step

Choose the next dynamic coverage area. The strongest remaining candidates from
the local inventory are standalone `sionna.phy.fec` and `sionna.phy.nr`.

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

Use the inventory to decide whether `fec` or `nr` should be implemented next,
then add a focused dynamic category before rerunning the umbrella `phy` audit.
