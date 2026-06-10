# FEC audit findings

## Context

This document records CUDA audit evidence for the standalone `sionna.phy.fec`
dynamic case set. The focused sweep shows the same device-migration pattern
already seen in channel, mapping, signal, MIMO, and OFDM objects.

## Command

The focused sweep used:

```bash
python run_repro.py run --category fec --device cuda:1 --build-device cpu --no-probe-forward --json-report reports/fec-audit-cuda1.json
```

The command wrote `reports/fec-audit-cuda1.json`. Because failed audit cases
return a non-zero process status by default, commands chained with `&&` after
this sweep will not run unless `--no-fail` is also passed.

## Sweep summary

Environment:

- Runtime: Ubuntu server with NVIDIA CUDA GPUs.
- Sionna environment: `sdm`.
- Target device: `cuda:1`.
- Build device: `cpu`.
- Forward probes: disabled.

Result:

- Total FEC-category cases: 31.
- Failed audit cases: 30.
- Passed audit cases: 1.
- Skipped cases: 0.
- The only passed case was standalone `fec-trellis`.
- All Sionna `Block` or `Object` FEC cases include stale Sionna logical device
  state after `.to(cuda:1)`.

## Case-level findings

| Case | Status | Issue count | Main finding |
| --- | --- | ---: | --- |
| `fec-gaussian-prior-source` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-crc-encoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-crc-decoder` | failed | 2 | decoder and nested CRC encoder logical devices remain `cpu` |
| `fec-trellis` | passed | 0 | standalone Trellis `.to(cuda:1)` migrates its explicit tensor state |
| `fec-conv-encoder` | failed | 7 | encoder logical device and nested Trellis tensors remain `cpu` |
| `fec-viterbi-decoder` | failed | 7 | decoder logical device and nested Trellis tensors remain `cpu` |
| `fec-bcjr-decoder` | failed | 7 | decoder logical device and nested Trellis tensors remain `cpu` |
| `fec-row-column-interleaver` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-random-interleaver` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-turbo-3gpp-interleaver` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-deinterleaver` | failed | 2 | deinterleaver and child interleaver logical devices remain `cpu` |
| `fec-scrambler` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-tb5g-scrambler` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-descrambler` | failed | 2 | descrambler and child scrambler logical devices remain `cpu` |
| `fec-linear-encoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-os-decoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-ldpc-bp-decoder` | failed | 7 | LDPC graph gather, mask, and scatter tensors remain `cpu` |
| `fec-ldpc-5g-encoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-ldpc-5g-decoder` | failed | 8 | nested encoder state and LDPC graph tensors remain `cpu` |
| `fec-exit-callback` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-decoder-statistics-callback` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-weighted-bp-callback` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-polar-encoder` | failed | 2 | polar encoder logical device and `_ind_gather` remain `cpu` |
| `fec-polar-sc-decoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-polar-scl-decoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-polar-bp-decoder` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-polar-5g-encoder` | failed | 3 | encoder, CRC child, and `_ind_gather` remain `cpu` |
| `fec-polar-5g-decoder` | failed | 5 | nested encoder, CRC child, gather tensor, and decoder state remain `cpu` |
| `fec-turbo-termination` | failed | 1 | `root._device_str` remains `cpu` |
| `fec-turbo-encoder` | failed | 16 | nested interleaver, encoder, Trellis tensors, and puncturing tensor remain `cpu` |
| `fec-turbo-decoder` | failed | 16 | nested interleaver, BCJR decoder, Trellis tensors, and puncturing tensor remain `cpu` |

## Failure categories observed

### Stale logical device state

All Sionna `Block` or `Object` FEC cases include stale logical device state:

```text
expected=cuda:1, actual=cpu, kind=logical-device
```

Examples include CRC blocks, convolutional encoders/decoders, interleavers,
scramblers, LDPC blocks, polar blocks, turbo blocks, and LDPC callbacks.

### Ordinary tensor attributes not migrated

Several FEC objects keep tensor state in ordinary attributes that PyTorch
`.to()` does not migrate. Examples include:

- Trellis tensors nested in convolutional and turbo blocks: `to_nodes`,
  `from_nodes`, `op_mat`, `op_by_tonode`, `ip_by_tonode`, and
  `op_by_fromnode`.
- LDPC graph tensors: `_cn_gather_idx`, `_cn_mask`, `_vn_gather_idx`,
  `_vn_mask`, `_cn_scatter_idx`, and `_vn_scatter_idx`.
- Polar gather tensors: `_ind_gather`.
- Turbo puncturing tensors: `_punct_pattern`.

### Nested child state

Composite FEC objects recursively own child blocks such as CRC encoders,
interleavers, scramblers, convolutional encoders, BCJR decoders, LDPC encoders,
polar encoders, and polar decoders. The audit shows stale state across these
child objects, confirming that PyTorch module recursion does not fix
Sionna-specific logical device fields or ordinary tensor attributes.

## Trellis note

The standalone `Trellis` case passed the audit. This is useful boundary
evidence: `Trellis.to(cuda:1)` can migrate its explicit tensor state when called
directly. However, FEC blocks that own a Trellis still failed because their
parent Sionna logical device state and nested ordinary tensor attributes remain
on CPU after parent `.to(cuda:1)`.

## Interpretation

The focused FEC sweep confirms that `.to(device)` migration issues are also
systematic across standalone `sionna.phy.fec` blocks. The updated umbrella PHY
sweep across all 114 then-current dynamic cases found 113 failed cases, one passed
standalone Trellis case, and 0 skipped. Standalone NR dynamic cases have now
been added and audited: 12/12 failed and 0 skipped. The updated umbrella PHY
sweep across all 126 current dynamic cases found 125 failed cases, one passed
standalone Trellis case, and 0 skipped.

```bash
python run_repro.py run --category phy --device cuda:1 --build-device cpu --no-probe-forward --no-fail --json-report reports/phy-audit-cuda1.json
```
