# MIMO audit findings

## Context

This document records CUDA audit evidence for the standalone `sionna.phy.mimo`
dynamic case set. The sweep shows the same device-migration pattern already
seen in channel, mapping, signal, and OFDM objects.

## Command

The focused sweep used:

```bash
CUDA_DEVICE=cuda:0
python run_repro.py run --category mimo --device "$CUDA_DEVICE" --build-device cpu --no-probe-forward --no-fail --json-report reports/mimo-audit-cuda.json
```

The collected report used `cuda:1`; any visible CUDA device can be used.

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Target device in collected report: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total MIMO-category cases: 8.
- Failed audit cases: 8.
- Skipped cases: 0.
- All standalone MIMO cases include stale Sionna logical device state.
- Detector and list-to-LLR helpers also keep ordinary tensor attributes on CPU
  after `.to(device)`.

## Case-level findings

| Case | Issue count | Main finding |
| --- | ---: | --- |
| `mimo-stream-management` | 1 | `root._device_str` remains `cpu` |
| `mimo-list2llr` | 1 | `root._device_str` remains `cpu` |
| `mimo-list2llr-simple` | 3 | `root._device_str`, `_c0`, and `_c1` remain `cpu` |
| `mimo-linear-detector` | 9 | detector, constellation, demapper, and demapper lookup tensors remain `cpu` |
| `mimo-maximum-likelihood-detector` | 12 | ML lookup tensors, constellation, and logits/LLR child state remain `cpu` |
| `mimo-k-best-detector` | 8 | list-to-LLR child state, constellation tensor, and K-best lookup tensors remain `cpu` |
| `mimo-ep-detector` | 8 | EP detector points, scalar state, and nested symbol-logits-to-LLR state remain `cpu` |
| `mimo-mmse-pic-detector` | 12 | constellation, LLR/logits helpers, demapper, and demapper lookup tensors remain `cpu` |

## Failure categories observed

### Stale logical device state

Every standalone MIMO case includes stale logical device state:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

Examples include:

- `StreamManagement._device_str`
- `List2LLR._device_str`
- `List2LLRSimple._device_str`
- detector `_device_str` fields
- nested mapping helper `_device_str` fields

### Ordinary tensor attributes not migrated

Several MIMO objects keep tensor state in ordinary attributes that PyTorch
`.to()` does not migrate. Examples include:

- `List2LLRSimple._c0` and `_c1`
- `MaximumLikelihoodDetector._vecs`, `_vecs_ind`, and `_c`
- `KBestDetector._constellation`, `_indices`, `_sym_pattern`, and
  `_ind_pattern`
- `EPDetector._points`, `_es`, and `_no`
- nested mapping lookup tensors such as `_points`, `_c0`, `_c1`, `_a`, and
  `_no_threshold`

### Nested child state

Composite MIMO detector blocks recursively own mapping, demapper, constellation,
and list-to-LLR helpers. The audit shows stale state across these child objects,
confirming that PyTorch module recursion does not fix Sionna-specific logical
device fields or ordinary tensor attributes.

## Interpretation

The focused MIMO sweep confirms that `.to(device)` migration issues are also
systematic across standalone `sionna.phy.mimo` objects. The issue is therefore
not limited to channel wrappers, mapping utilities, signal filters, or OFDM
wrappers.

Combined audit-only CUDA evidence:

- `sionna.phy.channel`: 17/17 cases failed audit.
- `sionna.phy.mapping`: 14/14 cases failed audit.
- `sionna.phy.signal`: 12/12 cases failed audit.
- `sionna.phy.mimo`: 8/8 cases failed audit.
- `sionna.phy.ofdm`: 33/33 OFDM-category cases failed audit.

The current umbrella PHY sweep covers 126 dynamic cases and found 125 failed
audit cases, one passed standalone `fec-trellis` boundary case, and zero
skipped cases. Runtime-impact evidence is summarized in
[`forward-probe-findings.md`](forward-probe-findings.md).
