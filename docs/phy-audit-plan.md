# Sionna PHY device migration audit methodology

## Objective

Audit `sionna.phy` for PyTorch `.to(device)` migration problems. The concrete
bug is that a Sionna PHY object can be constructed on CPU, moved with
`.to(cuda_device)`, and still keep logical device state or ordinary tensor
attributes on CPU.

The original symptom was a user-defined PyTorch wrapper containing
`sionna.phy.channel.AWGN`. The broader audit checks whether the same pattern is
systematic across `sionna.phy`.

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

- `sionna.rt` and RT-specific functionality.
- Pure functions unless they are needed to build a minimal repro case.
- Rewriting or patching Sionna source code.
- Performance or numerical-accuracy benchmarking.

## Hypothesis

PyTorch's default `nn.Module.to(device)` migrates parameters and registered
buffers, and it recursively visits child modules. It does not automatically
update library-specific device metadata such as Sionna's `_device_str`.

If a Sionna PHY object later creates tensors from stale `self.device`, or keeps
ordinary tensor attributes that are not registered buffers, `.to(device)` can
leave part of the object graph on the construction device. This can produce:

- audit-only mismatches, such as `expected=cuda:x, actual=cpu`;
- mixed-device forward failures;
- output tensors on CPU for CUDA inputs;
- stale state inside child modules even after a parent PyTorch wrapper is moved.

## Audit workflow

### 1. Inventory `sionna.phy`

Generate a static class inventory:

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

For each discovered class, record module path, class name, base classes,
constructor signature, device-related fields, buffer usage, tensor attributes,
and whether the class is directly instantiable.

### 2. Classify risk

Use the inventory to identify dynamic repro candidates:

- P0: stateful PHY objects with Sionna logical device state, child modules,
  registered buffers, ordinary tensor attributes, or tensor creation through
  `self.device`.
- P1: relevant objects that require complex domain-specific forward inputs.
- P2: abstract classes, pure configuration/data holders, or classes without
  device state or tensor creation.

High-risk areas are `channel`, `mapping`, `signal`, `mimo`, `ofdm`, `fec`, and
`nr`.

### 3. Run dynamic repro cases

Choose any visible CUDA device:

```bash
CUDA_DEVICE=cuda:0
```

Run the object-state audit:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda.json
```

Run the forward-probe sweep:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-fail --json-report reports/phy-forward-cuda.json
```

Each dynamic case should:

- construct the object on a controlled build device, CPU by default;
- call `.to(target_device)` through PyTorch;
- audit registered tensors, ordinary tensors, child modules, and logical device
  fields;
- optionally run one forward or call probe using small inputs already placed on
  `target_device`;
- emit JSON records that distinguish object-state issues, forward exceptions,
  and output-device mismatches.

## Current evidence

Collected audit-only CUDA evidence:

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

Collected forward-probe CUDA evidence:

- Forward exceptions: 18 cases.
- Wrong-device forward outputs: 30 cases.
- Wrong-device returned tensors: 33 tensors.

## Failure taxonomy

- Stale logical device state, for example `_device_str` remains `cpu` after
  `.to("cuda:x")`.
- Stale child module state after parent `.to(device)`.
- Ordinary tensor attributes not migrated because they are not registered
  buffers or parameters.
- Forward-created tensors using stale `self.device`.
- Forward execution succeeds but returns tensors on the wrong device.
- Forward execution fails because operands are split across CPU and CUDA.

## Related documents

- [`upstream-repro-note.md`](upstream-repro-note.md)
- [`phy-audit-findings.md`](phy-audit-findings.md)
- [`forward-probe-findings.md`](forward-probe-findings.md)
- [`channel-audit-findings.md`](channel-audit-findings.md)
- [`mapping-signal-audit-findings.md`](mapping-signal-audit-findings.md)
- [`ofdm-audit-findings.md`](ofdm-audit-findings.md)
- [`mimo-audit-findings.md`](mimo-audit-findings.md)
- [`fec-audit-findings.md`](fec-audit-findings.md)
- [`nr-audit-findings.md`](nr-audit-findings.md)
