# Aligned memory windows for the compiled RISC-V CPU

This experiment makes RAM and DROM movable, expands their bank selectors, and proves actual compact-circuit generation at two useful Doom memory sizes. It does not run Doom or produce a Doom frame.

All maps keep the existing 256 KiB code ROM at `0x00040000`. RAM starts at `0x01000000`. DROM starts at `0x02000000`. This preserves instruction and jump-token geometry. The experiment supports the standalone RV32IM+Zba CPU. The text bridge is explicitly rejected. Custom decimal memory instructions are not covered and retain upstream assumptions about low-address memory.

## Recorded proofs

| Evidence | RAM | DROM | Actual chip layers | Generated tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `evidence/small` | 64 KiB | 256 KiB | 14 | 296 | All 12 cases match the reference. All 65,536 recurrent RAM bytes match after every case. |
| `evidence/low-memory` | 64 KiB | 4 MiB | 44 | 296 | Same complete suite and RAM extraction checks. |
| `evidence/scale` | 8 MiB | 8 MiB | 584 | 121 | Earlier first/middle/last-bank and width/control-flow workload matches. This run predates the boot-sentinel and RAM-extraction additions. |

The complete suite exercises SW, SB, SH, LW, signed and unsigned byte/halfword loads, both region boundaries, writes to read-only data, old-address aliases, misaligned accesses, AUIPC, call, and return. It also loads three nonzero boot snapshot words before modifying memory. The success case outputs `ABC`. Fault cases generate the exact fault trace through HALT.

The compact constructor was checked against the original upstream constructor on the small map. All 188 sparse parameter/state tensors and their metadata matched exactly. The generated traces then matched independently executed reference traces. `sparse_exec.py` retains upstream `Executor.step`, `load_init`, `reset`, and `embed`. Only matrix construction changes. It assembles nonzero coordinates directly instead of creating large dense NumPy matrices before discarding their zeros. Output coefficients preserve the original constructor's assignment behavior.

The 4 MiB DROM configuration can hold a converted data set below that size. The unmodified 4,196,020-byte original shareware IWAD exceeds 4 MiB before engine read-only data. No game data is part of this export.

## Reproduce

Place this directory at `experiments/memory-map` in the repository and use its configured environment. Each invocation needs a fresh scratch directory. The script refuses to overwrite an existing package. It checks the clean pinned upstream revision before copying it into scratch.

```sh
.venv/bin/python experiments/memory-map/run.py \
  --scratch .cache/memory-small \
  --out artifacts/memory-small \
  --direct-sparse --check-constructor

.venv/bin/python experiments/memory-map/guard.py \
  --seconds 300 --rss-gib 4 \
  --log artifacts/memory-low.log --status artifacts/memory-low-status.json -- \
  .venv/bin/python experiments/memory-map/run.py \
  --scratch .cache/memory-low --out artifacts/memory-low \
  --ram-kib 64 --drom-kib 4096 --direct-sparse
```

The optional constructor comparison deliberately creates the old dense temporary matrices. Do it only on a small map. The low-memory run uses the direct sparse constructor alone. Its recorded watchdog peak was 993,771,520 resident bytes, including the Python graph and runtime. This is one observed memory footprint, not a device-memory guarantee or speed measurement.

The archived full-scale workload can be reproduced with `run_scale_v1.py`, `--ram-kib 8192 --drom-kib 8192 --scale-smoke --direct-sparse`. It completed under its original 300-second wall limit and 8 GiB sampled resident-memory cap. Its observed watchdog peak was 6,885,097,472 bytes. `run.py` now adds snapshot and extraction checks. No full-scale repeat of those added checks is claimed.

`guard.py` uses POSIX process groups and `ps`. The recorded CPU driver reports macOS peak RSS. The core preparation, sparse constructor, and state decoder do not require these watchdog APIs.

## Memory representation and cost

`prepare.py` adds `region_literals(bits, base, size)` to the isolated CPU source. The function requires a power-of-two window aligned to its size and returns its fixed upper address-bit literals. Both load-range predicates use it. RAM bank width already derives from `RAM_WORDS`; DROM bank width now derives from `DROM_WORDS` instead of the fixed `EA[13:17]` slice. `DATA_BASE` moves with RAM, and the stack top follows the existing `SP_INIT` expression.

RAM snapshots now make one pass over RAM words and assign each nonzero word to its bank and row. The original nested loop scanned all RAM words once per bank. The three nonzero snapshot loads and complete final-state comparisons verify the replacement on both 64 KiB configurations.

| Actual circuit | Logical value heads | Actual fp32 recurrent-state bytes |
| --- | ---: | ---: |
| 64 KiB RAM, 256 KiB DROM | 287 | 18,808,832 |
| 64 KiB RAM, 4 MiB DROM | 1,247 | 81,723,392 |
| 8 MiB RAM, 8 MiB DROM | 18,527 | 1,214,185,472 |

The standalone dense HF layout previously measured 2,113,019,200 parameters for 11 chip layers. Each additional chip layer with the same dimensions adds 124,212,928 parameters. Applying that formula to the actual 44-layer circuit gives an estimated 6,212,045,824 parameters, or 12,424,091,648 bf16 parameter bytes. That excludes extra runtime allocations and is not a promise of fitting a 16 GB GPU. The 584-layer estimate remains 146,574,053,888 bf16 parameter bytes. Neither dense model was allocated or emitted here. `cost-estimates.json` preserves the calculation.

## Extract bytes from real recurrent RAM

Use `state.read_ram(executor, address, count, batch_row=0)`. It reads the compact executor's `S` tensors and head metadata. It does not consult the reference machine, its output, or generated token contents. The complete 64 KiB image extracted after every small-map and low-memory case matched the expected final RAM bytes.

For a byte offset relative to RAM, let `wi = offset >> 2`. Its key head is `(wi & 15) | ((wi >> 12) << 4)`, its value head within that key head is `(wi >> 4) & 1`, and its row is `(wi >> 5) & 127`. Snapshot/SW bits occupy columns 2 through 33. Merged SB/SH bits occupy columns 34 through 65. Each data bit is the sum of its two columns divided by `A_BIT`. The decoder validates exact bits and ballast before creating bytes.

The same function accepts the `S`/`heads` layout of upstream `Executor2` and skips its FFN dictionaries. That compatibility follows the inspected data structure. GPU extraction itself has not been executed in this experiment.

For a Doom frame, after the generated DG callback marker, read the exported framebuffer address and length symbols from recurrent RAM with this API, then read the indicated bytes. The decoder only converts bytes already produced by the compiled CPU. It performs no Doom rendering.
