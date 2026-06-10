# Project plan

## Goal

Build a narrow, reproducible, and extensible project that demonstrates how
Sionna PHY objects, despite being `torch.nn.Module` instances, may fail to
migrate all internal state when `.to("cuda:0")` is called directly.

## Scope boundaries

- Do not modify Sionna source code.
- Do not implement communication channel algorithms in this repository.
- Do not rely only on one object's error message; record both object-state
  audits and forward-pass behavior.
- Construct objects on CPU by default, then call `.to(target_device)`, because
  that is the behavior PyTorch users naturally expect to work.

## Phase 1: minimal repro scaffold

- Provide a CLI for listing cases, running cases, and collecting environment
  information.
- Include built-in Sionna PHY channel cases.
- Print device audit results and forward probe results.
- Write JSON reports for upstream issue reports.

## Phase 2: broader object coverage

Prioritize these object categories:

- Objects with no internal tensors but with Sionna logical `device` state, such
  as `AWGN`.
- Composite objects with child blocks, such as `FlatFadingChannel`.
- Objects with registered buffers, such as `KroneckerModel` and
  `PerColumnModel`.
- Mapping objects with lookup tensors, constellations, and child source or
  mapper blocks.
- Signal objects with generated or user-provided coefficient tensors.
- Objects that generate or apply OFDM and time-domain channel behavior.
- TR 38.901 objects, especially those with internal caches, topology state,
  random state, or many tensor attributes.

Current dynamic coverage includes `sionna.phy.channel`,
`sionna.phy.mapping`, `sionna.phy.signal`, standalone `sionna.phy.fec`,
standalone `sionna.phy.mimo`, and standalone `sionna.phy.ofdm` cases. The
first 43 audit-only CUDA cases covering channel, mapping, and signal all failed
after CPU construction followed by `.to(cuda:1)`. The clean OFDM CUDA audit
found 33 failed OFDM-category cases and no skips. The clean MIMO CUDA audit
found 8 failed MIMO-category cases and no skips. The clean FEC CUDA audit found
30 failed FEC-category cases, one passed standalone Trellis case, and no skips.
The latest collected umbrella PHY audit across the current 114-case dynamic set
failed 113 cases, passed only `fec-trellis`, and had no skips. The next
coverage target is standalone `sionna.phy.nr`.

## Phase 3: upstream report material

Prepare the material needed for an upstream issue:

- Python, PyTorch, Sionna, CUDA, and GPU environment information.
- Minimal runnable commands.
- Expected behavior: after `.to("cuda:0")`, both internal object state and
  forward outputs should live on `cuda:0`.
- Actual behavior: mismatched audit paths, forward exceptions, or output-device
  mismatches.
- Any temporary workaround should be recorded separately instead of being mixed
  into the repro scripts.

## Verification strategy

- Repository tests should not require Sionna or CUDA; they should validate only
  the device auditor and case registry.
- Real Sionna repros should be executed through the CLI in a CUDA-enabled
  environment.
- Without CUDA, `python run_repro.py env` should still provide environment
  diagnostics, and `pytest` should still be able to run.
