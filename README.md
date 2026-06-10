# Sionna device migration repro

This repository reproduces and diagnoses device migration issues in Sionna PHY
objects when they are used as PyTorch modules and moved with `.to(device)`.

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
|   +-- channel-audit-findings.md
|   +-- fec-audit-findings.md
|   +-- mapping-signal-audit-findings.md
|   +-- mimo-audit-findings.md
|   +-- ofdm-audit-findings.md
|   +-- phy-audit-findings.md
|   +-- phy-audit-plan.md
|   `-- project-plan.md
+-- examples/
|   +-- minimal_awgn_to_cuda.py
|   `-- wrapped_awgn_channel_to_cuda.py
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
+-- tools/
|   `-- inspect_phy_inventory.py
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
Current channel-specific CUDA findings are summarized in
[docs/channel-audit-findings.md](docs/channel-audit-findings.md).
Current FEC CUDA findings are tracked in
[docs/fec-audit-findings.md](docs/fec-audit-findings.md).
Current mapping and signal CUDA findings are summarized in
[docs/mapping-signal-audit-findings.md](docs/mapping-signal-audit-findings.md).
Current MIMO CUDA findings are tracked in
[docs/mimo-audit-findings.md](docs/mimo-audit-findings.md).
The current umbrella PHY CUDA summary is maintained in
[docs/phy-audit-findings.md](docs/phy-audit-findings.md).
Current OFDM CUDA findings are tracked in
[docs/ofdm-audit-findings.md](docs/ofdm-audit-findings.md).

## Quick start

List built-in cases:

```bash
python run_repro.py list-cases
```

Collect environment information:

```bash
python run_repro.py env
```

Build a static inventory of `sionna.phy` classes:

```bash
python tools/inspect_phy_inventory.py --json-report reports/phy-inventory.json
```

Run all built-in CUDA repros and write a JSON report:

```bash
python run_repro.py run --device cuda:0 --json-report reports/cuda0.json
```

Run all current PHY dynamic cases on the target CUDA device:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

Run focused category sweeps on the target CUDA device:

```bash
python run_repro.py run --category channel --device cuda:1 --json-report reports/channel-cuda1.json
python run_repro.py run --category mapping --device cuda:1 --json-report reports/mapping-cuda1.json
python run_repro.py run --category signal --device cuda:1 --json-report reports/signal-cuda1.json
```

Run only the post-`.to()` object-state audit for the current PHY case set:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

By default, repro objects are constructed on CPU before PyTorch `.to(device)` is
called. This avoids accidental construction on Sionna's global default device
such as `cuda:0`, which can cause unrelated out-of-memory failures before the
audit even starts. To intentionally keep Sionna's global default construction
behavior, use:

```bash
python run_repro.py run --case wrapped-awgn-channel --device cuda:1 --build-device default
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

## Broader PHY sweep

The AWGN wrapper is only the first concrete failure. The more useful next step
is to check whether the same stale-device pattern appears across other
`sionna.phy` objects.

Collected audit-only CUDA evidence so far:

- Latest collected umbrella PHY sweep: 113/114 current cases failed audit; the
  standalone `fec-trellis` case passed.
- `sionna.phy.channel`: 17/17 current cases failed audit.
- `sionna.phy.mapping`: 14/14 current cases failed audit.
- `sionna.phy.signal`: 12/12 current cases failed audit.
- `sionna.phy.mimo`: 8/8 current cases failed audit.
- Clean `sionna.phy.ofdm` sweep: 33/33 OFDM-category cases failed audit.
- `sionna.phy.fec`: 30/31 current cases failed audit; standalone
  `fec-trellis` passed.

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
- Mapping blocks: `BinarySource`, `Constellation`, `Mapper`, `Demapper`,
  `SymbolDemapper`, `LLRs2SymbolLogits`, `SymbolLogits2LLRs`,
  `SymbolInds2Bits`, `SymbolLogits2Moments`, `PAM2QAM`, `QAM2PAM`,
  `PAMSource`, `QAMSource`, and `SymbolSource`.
- Signal blocks: `Upsampling`, `Downsampling`, window classes, and filter
  classes. Base `Window` and base `Filter` are audit-only cases because they
  do not have usable coefficients for a forward probe by themselves.
- Standalone FEC blocks: CRC, convolutional, interleaver, scrambler, linear
  block code, LDPC, polar, turbo, callback, and Gaussian-prior helper objects.
  These are audit-only cases until safe forward probes are added.
- Standalone MIMO blocks: `StreamManagement`, `List2LLR`,
  `List2LLRSimple`, `LinearDetector`, `MaximumLikelihoodDetector`,
  `KBestDetector`, `EPDetector`, and `MMSEPICDetector`. These are audit-only
  cases until safe forward probes are added.
- Standalone OFDM blocks: resource grids, pilot patterns, grid mappers and
  demappers, OFDM modulation/demodulation, channel estimators, equalizers,
  detector wrappers, detectors, and precoding helpers. Complex detector and
  precoding cases are audit-only until safe forward probes are added.

Use the same target CUDA device that exposed the original bug and run the
current umbrella audit-only sweep:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

The command above constructs objects on CPU by default and then calls
`.to(cuda:1)`. The runner also temporarily aligns `sionna.phy.config.device`
with `--build-device` during construction. This keeps the audit focused on
PyTorch migration behavior rather than on Sionna's global default device.

Use focused category sweeps when rechecking one area:

```bash
python run_repro.py run --category fec --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/fec-audit-cuda1.json
```

The focused FEC CUDA report and the updated umbrella PHY CUDA report have both
been collected. Use `--no-fail` when rerunning the umbrella sweep so the command
still writes a complete report even though failed audit cases are expected:

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```

The next coverage expansion target after FEC is standalone `sionna.phy.nr`.

If multiple audit commands are chained in one shell command, pass `--no-fail`
to earlier commands. Failed audit cases are expected and otherwise stop `&&`
chains before later reports are collected.

Then run focused forward probes:

```bash
python run_repro.py run --category mapping --device cuda:1 --json-report reports/mapping-forward-cuda1.json
python run_repro.py run --category signal --device cuda:1 --json-report reports/signal-forward-cuda1.json
python run_repro.py run --category phy --device cuda:1 --json-report reports/phy-forward-cuda1.json
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
    categories=_categories("channel", "my-area"),
)
```

`build` should construct the object on CPU by default. The runner calls
`.to(args.device)` consistently. `make_inputs(device)` should return inputs
already placed on the target device so stale Sionna logical device state can be
observed directly. `categories` should include the relevant focused area
(`channel`, `mapping`, `signal`, and so on); `_categories(...)` automatically
adds the umbrella `phy` category.
