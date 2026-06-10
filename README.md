# Sionna device migration repro

This repository reproduces and diagnoses device migration issues in Sionna PHY
channel objects when they are used as PyTorch modules and moved with
`.to(device)`.

The goal is not to reimplement communication channel models. The goal is to
turn the observed behavior into small, reproducible evidence that can be used
for upstream issue reports or regression tests:

- Construct an object on CPU, then call `.to("cuda:0")`.
- Recursively inspect tensors, buffers, parameters, child modules, and Sionna
  logical device fields.
- Optionally run one forward pass and verify that outputs are on the target
  device.
- Emit both text output and JSON reports.

The upstream Sionna 2.0 README states that Sionna PHY/SYS are PyTorch-based and
require Python 3.11+ and PyTorch 2.9+. See:
https://github.com/NVlabs/sionna

## Project layout

```text
.
+-- run_repro.py
+-- docs/
|   `-- project-plan.md
+-- examples/
|   `-- minimal_awgn_to_cuda.py
+-- reports/
|   `-- .gitkeep
+-- scripts/
|   `-- run_cuda_repro.sh
+-- src/
|   `-- sionna_device_migration_repro/
|       +-- audit.py
|       +-- cases.py
|       +-- cli.py
|       +-- env.py
|       +-- repros.py
|       `-- runner.py
`-- tests/
    +-- conftest.py
    +-- test_audit.py
    `-- test_cases.py
```

## Environment

Use the prepared Conda environment:

```bash
conda activate sdm
```

If you want to avoid Sionna RT, follow upstream installation guidance and use
`sionna-no-rt` instead before running the repros. No package installation is
required for this repository itself.

The locally verified environment path is:

```bash
/opt/anaconda3/envs/sdm/bin/python
```

## Target runtime

The final target environment is an Ubuntu server with multiple NVIDIA GPUs.
Device-selection behavior, CUDA validation, and repro reports should therefore
be designed for multi-GPU CUDA systems, not only for a single local GPU.

## Audit plan

The detailed plan for auditing all relevant `sionna.phy` objects is maintained
in [docs/phy-audit-plan.md](docs/phy-audit-plan.md).

## Quick start

List built-in cases:

```bash
python run_repro.py list-cases
```

Collect environment information:

```bash
python run_repro.py env
```

Run all built-in CUDA repros and write a JSON report:

```bash
python run_repro.py run --device cuda:0 --json-report reports/cuda0.json
```

Run all channel-related cases on the target CUDA device:

```bash
python run_repro.py run --category channel --device cuda:1 --json-report reports/channel-cuda1.json
```

Run only the post-`.to()` object-state audit for all channel-related cases:

```bash
python run_repro.py run --category channel --device cuda:1 --no-probe-forward --json-report reports/channel-audit-cuda1.json
```

Run only the minimal AWGN case:

```bash
python run_repro.py run --case awgn --device cuda:0
```

Run the user-style wrapper that reproduces the mixed-device forward failure:

```bash
python run_repro.py run --case wrapped-awgn-channel --device cuda:0
```

You can also run the wrapper script:

```bash
./scripts/run_cuda_repro.sh cuda:0
```

If the current machine has no visible CUDA device, the command exits during
device validation. Use CPU for a framework smoke test:

```bash
python run_repro.py run --device cpu --json-report reports/cpu.json --no-fail
```

For a single-file reproduction of the wrapper issue:

```bash
python examples/wrapped_awgn_channel_to_cuda.py cuda:0
```

## Expected observations

If a Sionna object does not correctly support PyTorch `.to()` semantics, common
symptoms include:

- Internal fields such as `_device_str` or the `device` property remain `cpu`
  after `.to("cuda:0")`.
- Registered buffers move to CUDA, but Sionna's logical device remains CPU,
  causing inputs to be converted back to CPU or causing mixed-device errors
  during forward execution.
- Forward execution succeeds, but output tensors still live on CPU.

## Primary wrapped AWGN repro

The `wrapped-awgn-channel` case mirrors the user-side pattern:

```python
channel = AWGNChannel(snr=10)
channel.to("cuda:0")
x = torch.randn(2, 4, 8, device="cuda:0")
y = channel(x)
```

The expected behavior is that the wrapper and its internal `sionna.phy.channel.AWGN`
block both use `cuda:0`, and the forward output is also on `cuda:0`.

The suspected failure path is:

- PyTorch recursively visits the child module during `AWGNChannel.to("cuda:0")`.
- Sionna's `AWGN` object keeps logical device state in fields such as
  `_device_str`.
- PyTorch's default `.to()` does not update that logical Sionna state.
- During forward execution, Sionna may convert tensors according to its stale
  logical device, while the wrapper still has CUDA tensors such as the saved
  normalization power.
- The final operation can then mix CPU and CUDA tensors and fail.

## Broader channel sweep

The AWGN wrapper is only the first concrete failure. The more useful next step
is to check whether the same stale-device pattern appears across other Sionna
channel-related objects.

The current case set covers:

- Noise blocks: `AWGN`, wrapped `AWGNChannel`.
- Flat fading blocks: `GenerateFlatFadingChannel`, `ApplyFlatFadingChannel`,
  `FlatFadingChannel`, and a Kronecker-correlated flat-fading setup.
- OFDM/time-domain apply blocks: `ApplyOFDMChannel`, `ApplyTimeChannel`.
- Spatial correlation models: `KroneckerModel`, `PerColumnModel`.
- Discrete channels: `BinaryMemorylessChannel`, `BinarySymmetricChannel`,
  `BinaryErasureChannel`, `BinaryZChannel`.
- Channel model: `RayleighBlockFading`.
- Optical channel blocks: `EDFA`, `SSFM`.

Use the same target CUDA device that exposed the original bug and run an
audit-only sweep first:

```bash
python run_repro.py run --category channel --device cuda:1 --no-probe-forward --json-report reports/channel-audit-cuda1.json
```

Then run the forward probes for the same case set:

```bash
python run_repro.py run --category channel --device cuda:1 --json-report reports/channel-forward-cuda1.json
```

The key signal is not whether `cuda:1` differs from another GPU. The key signal
is whether each object still reports stale internal Sionna device state or
fails forward execution after a normal PyTorch `.to(target_device)` call.

## Adding repro cases

Add a new `CaseSpec` in `src/sionna_device_migration_repro/cases.py`:

```python
CaseSpec(
    name="my-channel",
    description="short description",
    build=build_my_channel,
    make_inputs=make_my_channel_inputs,
)
```

`build` should construct the object on CPU by default. The runner calls
`.to(args.device)` consistently. `make_inputs(device)` should return inputs
already placed on the target device so stale Sionna logical device state can be
observed directly.
