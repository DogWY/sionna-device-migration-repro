# Project scope

## Goal

Build a narrow repro project for one upstream question:

Does a `sionna.phy` object constructed on CPU behave like a normal PyTorch
module after `.to(cuda_device)`?

The repository should make the answer reproducible through small commands,
machine-readable JSON reports, and concise evidence documents.

## Scope boundaries

- Do not modify Sionna source code.
- Do not reimplement communication models.
- Do not benchmark performance or numerical accuracy.
- Do not rely on one error message only; collect both post-migration object
  state and forward behavior.
- Construct objects on CPU by default, then call `.to(target_device)`, because
  that is the PyTorch behavior users naturally expect to work.

## Evidence to support the issue

The project collects three kinds of evidence:

- Environment and inventory data: Python, PyTorch, Sionna, CUDA, GPU topology,
  and a static inventory of `sionna.phy` classes.
- Object-state audits: logical device fields, parameters, buffers, ordinary
  tensor attributes, and child module state after `.to(device)`.
- Forward probes: runtime exceptions and output tensor devices for CUDA inputs.

## Current coverage

Dynamic cases currently cover these `sionna.phy` areas:

- `sionna.phy.channel`
- `sionna.phy.mapping`
- `sionna.phy.signal`
- `sionna.phy.mimo`
- `sionna.phy.ofdm`
- `sionna.phy.fec`
- `sionna.phy.nr`

Latest collected CUDA evidence:

- Audit-only PHY sweep: 126 total dynamic cases, 125 failed audit cases, one
  passed case, and zero skipped cases.
- Only passed audit case: standalone `fec-trellis`.
- Forward-probe PHY sweep: 18 forward exceptions and 30 cases with wrong-device
  forward outputs, covering 33 returned tensors.

## Primary commands

Choose any visible CUDA device:

```bash
CUDA_DEVICE=cuda:0
```

Minimal AWGN repro:

```bash
python run_repro.py run --case awgn --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

User-style wrapper repro:

```bash
python run_repro.py run --case wrapped-awgn-channel --device "$CUDA_DEVICE" --build-device cpu --no-fail
```

Full object-state audit:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda.json
```

Full forward-probe sweep:

```bash
python run_repro.py run --category phy --device "$CUDA_DEVICE" --build-device cpu --no-fail --json-report reports/phy-forward-cuda.json
```

## Verification strategy

- Repository tests should not require Sionna or CUDA; they validate the device
  auditor and case registry.
- Real Sionna repros should be executed through the CLI in a CUDA-enabled
  environment.
- Without CUDA, `python run_repro.py env` should still provide diagnostics, and
  `pytest` should still be able to run.
