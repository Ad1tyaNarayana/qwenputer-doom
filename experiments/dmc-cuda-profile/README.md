# DMC compact CUDA performance profile

This experiment measures bounded token windows on the RTX 5070 Ti. It implements
step 2 of [the Doom handoff](../../docs/doom-24fps-handoff.md). It makes no
executor optimization, circuit-weight change, or interactive-frame-rate claim.
The circuit uses fp32 arithmetic through the existing compact Triton executor.
It is separate from stock Transformers execution and loads no pretrained Qwen weights.

## Windows and reference scope

`boot` generates the first 65,536 tokens of the historical first-frame ELF from
its normal boot state. Its comparison trace is the frozen, independently
generated reference used by `experiments/dmc-cuda`. The historical prefix took
31.3896 seconds in eight 8,192-token chunks.

`frame2_render` uses the continuing ten-frame walk ELF. The software reference
advances to a record boundary at least 100,000 tokens after frame 2 enters
`D_Display`. `core.snapshot` installs that reference state in the existing
circuit's initial recurrent tables. `FreeRun.load(GO)` resets lag histories.
The GPU then generates 65,536 self-fed tokens. The reference supplies no token
inputs during that window. Every generated token is compared afterward with
a separate trace from `Machine.record()` starting at the same reference state.
Reference preparation occurs before measurement. No host guest calculation
occurs in the timed GPU loop.

**The frame-2 window is reference-seeded and is only a performance profile.**
It does not prove neural execution of startup, frame 1, or the path to this
state. Matching a seeded window does not establish that its recurrent state
is identical to an uninterrupted neural run. Reset lag histories can affect
the beginning of the window. Both comparison scopes are bounded.

The existing actual-GPU checkpoint belongs to the legacy first-frame ELF.
Its metadata is in `evidence/inventory.json`. That ELF differs from the walk
ELF, so the checkpoint cannot establish the continuing workload's state.
Checkpoints, ELF files, reference token arrays, and guest state stay private.

## Measurement method

Five alternating boot and rendering runs use the upstream `FreeRun.run` path,
with one 65,536-token call per run. The timer includes trace-buffer allocation,
one CUDA graph replay per token, GPU token decoding and next-token embedding,
one device trace copy per token, the terminal CUDA synchronization, and
conversion of the trace to a CPU NumPy array. It excludes imports, circuit
construction, compilation, graph capture, reference preparation, state reset,
reference comparison, file output, checkpoints, pixel extraction, and display.
The historical prefix used eight chunks, so the allocation and synchronization
counts differ. These are token-window timings, not full-frame wall times.

Three separate diagnostics follow the throughput runs.

- A CUDA event pair spans 8,192 normal replays and trace copies. Host timers
  separately measure replay submission, trace-copy submission, terminal
  synchronization wait, and D2H conversion. Host submission overlaps device
  execution. The terminal wait is the outstanding GPU work, not an additional
  device bottleneck. These measurements must not be summed with GPU time.
- `torch.profiler` records CPU and CUDA activity for 256 tokens on the original
  graph. The reducer checks all 108 circuit kernels per token against the
  scheduled head or FFN kernel followed by its apply kernel. It attributes
  every scheduled sublayer separately, groups RAM, code-ROM and data-ROM
  heads, and separates feedback kernels and trace copies. Gaps are the
  remaining elapsed device-stream time between the first graph kernel and
  the last trace copy. They include graph-node scheduling and between-replay
  gaps. They are not a direct measurement of Python launch overhead.
- A separate graph brackets each unchanged scheduled sublayer with external
  CUDA events. The wrapper calls the original `Executor2.step` on each
  scheduled entry, in order. The feedback path remains `FreeRun._step`.
  Each replay synchronizes before reading events. These numbers include
  event-induced scheduling changes and are retained as a diagnostic only.
  They are not used to predict normal throughput or allocate its time.

The profiler's per-token GPU shares use its own device-stream span as their
denominator. They do not divide profiler durations by uninstrumented wall
time. Profiling can change WDDM scheduling and clock behavior. Raw profiler
traces remain in the GPU artifact directory. `profile-summary.json` records
their hashes, sizes, exact per-sublayer means, and resource metadata.

The runtime applies an 8 GiB PyTorch allocator cap before construction. The
process watchdog defaults to 1,100 seconds. SSH foreground runs are bounded
but are not detached; an SSH drop aborts the attempt. GPU utilization, memory
utilization, clocks, power, temperature and memory usage are sampled with
`nvidia-smi` about once per second. Sampling continues across resets, so only
samples labelled with a window belong to throughput measurement. This is a
Windows WDDM desktop with other graphics applications. Occupancy fields from
Kineto are resource/grid estimates, not measured active-SM occupancy. Neither
`nsys` nor `ncu` was available on PATH. This experiment installs no system tools.

## Reproduction

The working GPU environment is native Windows Python 3.14.2,
PyTorch 2.14.1+cu130, CUDA runtime 13.0, and triton-windows 3.8.0.post29.
The pinned upstream is `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.
`inventory.json` checks that the prepared and upstream `tritonexec.py` files
are byte-identical and records the clean upstream status. `runtime.py` retains
the existing standalone token-family adaptation. No upstream file is edited.

The private layout and initial prefix inputs are those in
[the existing runner](../dmc-cuda/README.md). The continuing walk build and
matching `workload.json` must already exist under
`.cache/dmc-continuing/walk-build`, as described in
[the continuing workload](../dmc-rv32/CONTINUING.md).
Run from the Mac repository root to sync the scripts:

```sh
tar -cf - experiments/dmc-cuda-profile | ssh gpu "tar -xf - -C C:/Users/Admin/qwenputer-games-work/repo"
```

The recorded main command was:

```sh
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-cuda-profile/profile_gpu.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/dmc-cuda-profile-20261029 --tokens 65536 --repeats 5 --diagnostic-tokens 256 --max-seconds 1100"
```

A preceding smoke run used `--tokens 256 --repeats 1 --diagnostic-tokens 8
--max-seconds 180` and a separate `dmc-cuda-profile-smoke` output directory.
It passed token comparisons in both windows. The smoke run is excluded from
the reported rates. `summarize.py --input ARTIFACT_DIR --out EVIDENCE_DIR`
reduces the traces. `inventory_gpu.py --work PRIVATE_WORK --out inventory.json`
collects source hashes and checkpoint metadata. Copy only `report.json`,
`profile-summary.json`, and `inventory.json` back to this directory's evidence.

## Evidence and limits

`evidence/report.json` contains every timing run, match counts and hashes,
CUDA event diagnostics, version information, reference-seed offsets, and GPU
samples. `evidence/profile-summary.json` contains kernel attribution and
per-sublayer resource hints. `evidence/inventory.json` identifies the exact
runtime and arithmetic sources and the inspected legacy checkpoint.

This experiment measures no pixel extraction or presentation. Neither window
finishes a displayed frame. It does not test full-state equivalence, long-run
state dependence, interactive input latency, frame pacing, or stock-model
performance. Optimization candidates belong in the conclusions only.

The Mac independently recomputes the seeded reference window with:

```sh
.venv/bin/python experiments/dmc-cuda-profile/verify_reference.py --package .cache/dmc-continuing/walk-machine --elf .cache/dmc-continuing/walk-build/doom.elf --report experiments/dmc-cuda-profile/evidence/report.json --out experiments/dmc-cuda-profile/evidence/reference-recheck.json
```

That check uses the same independent RV32 semantics on a different host. It
checks the seed's token offset, instruction offset and PC, then checks every
recorded GPU window hash. It does not turn reference seeding into a proof
of neural execution before the window.

## Recorded results on 2026-10-29

There is no measured difference between the two window rates. The median
changes by 0.13%, within the overlapping run ranges. Five fresh resets of
each window produced the same token hash every time. This result does not
exclude state-dependent cost in other instructions or later frames.

| Metric | Boot | Frame-2 rendering, reference-seeded |
| --- | ---: | ---: |
| Median tokens/s | 2,073.96 | 2,076.58 |
| Tokens/s range | 2,003.01 to 2,081.80 | 2,017.61 to 2,080.92 |
| Median 65,536-token seconds | 31.5995 | 31.5596 |

The following GPU durations are microseconds per token in the separate
256-token profiler captures. Shares use that capture's stream span.
Graph replay spans overlap the listed components. D2H occurs after that span.

| GPU activity | Boot µs/token | Frame-2 µs/token | Frame-2 share |
| --- | ---: | ---: | ---: |
| Data-ROM heads | 262.270 | 260.136 | 51.34% |
| RAM heads | 41.375 | 41.289 | 8.15% |
| Code-ROM heads | 9.958 | 9.959 | 1.97% |
| ALU/decode FFNs | 56.309 | 55.628 | 10.98% |
| Other heads | 15.014 | 14.971 | 2.95% |
| Residual/history apply | 65.406 | 65.372 | 12.90% |
| Token feedback/embedding | 29.691 | 29.565 | 5.83% |
| Trace D2D copy | 0.836 | 0.836 | 0.17% |
| Device-stream gaps | 28.960 | 28.937 | 5.71% |
| Graph replay span | 491.185 | 488.187 | overlaps above |
| Device-stream span | 509.820 | 506.693 | overlaps above |

The host diagnostics use 8,192 tokens. The times below include driver waits
inside submission calls. They overlap GPU execution and are not an additive
breakdown of the uninstrumented wall time.

| Host or event measurement | Boot | Frame-2 rendering |
| --- | ---: | ---: |
| Replay submission, µs/token | 5.632 µs | 15.420 µs |
| Trace-copy submission, µs/token | 453.701 µs | 443.215 µs |
| Terminal synchronization wait, total | 0.232 s | 0.235 s |
| D2H conversion, total | 719.100 µs | 121.900 µs |
| CUDA-event stream elapsed, total | 3997.189 ms | 3993.592 ms |

The largest three individual device groups in rendering are data-ROM heads
at 51.34%, residual/history apply at 12.90%, and ALU/decode FFNs at 10.98%.
All memory heads together account for 61.45%. The 32 data-ROM sublayers run
on every token in both windows. This identifies a circuit evaluation cost,
not a measured VRAM-bandwidth limit. The profile contains 133 graph kernel
nodes per token, including 108 circuit kernels and 25 feedback kernels.

The normal CUDA-event diagnostic takes about 488 µs/token. The external-event
graph takes about 740 µs/token at boot and 725 µs/token in rendering. That
perturbation is why its per-sublayer event means are excluded from the
bottleneck percentages. The original graph's profiler stream spans are
about 507 to 510 µs/token, also slower than the uninstrumented path.

Across throughput-labelled samples, median GPU utilization is 100% and
median memory utilization is 1% in both windows. Median SM clock is 2,887 MHz,
with about 80 W board power. Data-ROM kernels launch only 32 blocks of 128
threads, with 153 registers per thread. Kineto labels their block occupancy
limit as registers. A busy device does not establish high active-SM occupancy
or memory-bandwidth saturation. RAM kernels use 218 registers per thread.
Peak PyTorch allocation is 278,903,808 bytes, with 295,698,432 bytes reserved.
The initial Windows CPU load query returned null; no CPU-idle claim follows
from it. The machine had desktop graphics applications running.

All 672,768 generated tokens in the main timing and diagnostic windows
match their independent references. That count includes 655,360 throughput
tokens, 16,384 submission-diagnostic tokens, 512 profiler tokens, and 512
external-event tokens. The Mac reference recheck reproduces the rendering
window hash `26ad5f553b567aa9b8b4144207d9b20647cf46a1c6f033b9069c7abecad51de7`.
The seed is at token 6,470,215, instruction 3,966,856, PC `0x63864`,
after frame 2 enters rendering at token 6,370,214.

Potential follow-up experiments are fewer boundaries between unchanged
memory sublayers, fusion of residual/history updates, and fewer boundaries
in token feedback. Each needs circuit-equivalence and recurrent-state checks.
No such optimization is implemented here. Full continuing neural execution,
pixel extraction and presentation costs, achieved occupancy, and the behavior
of other rendering locations remain open.

## Host queue diagnostic

The follow-up `queue_probe.py` keeps the original boot graph and trace-copy
path. It changes only the bounded window length and records each host call.
All 8,768 tokens match the frozen boot reference.

| Tokens | Copies taking over 100 µs | First 64 copy-call median | Terminal sync wait | Device stream |
| --- | ---: | ---: | ---: | ---: |
| 64 | 0 | 6.15 µs | 30.93 ms | 31.70 ms |
| 512 | 0 | 5.90 µs | 244.12 ms | 249.65 ms |
| 8,192 | 247 | 6.05 µs | 237.24 ms | 4,029.19 ms |

The long window spends 3.751 seconds inside host trace-copy calls, while the
short windows submit copies in about 6 µs each. The long window's maximum
copy call lasts 15.70 ms. Blocking moves from the final synchronization into
submission as more work is queued. This supports a driver-queue backpressure
interpretation. It does not identify the exact WDDM scheduler mechanism.
The device copy itself is about 0.836 µs/token in the profiler. Treating the
453 µs/token average host copy-call duration as additional copy computation
would double-count waiting for the already-measured device work.

The exact follow-up command was:

```sh
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-cuda-profile/queue_probe.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --out C:/Users/Admin/qwenputer-games-work/artifacts/dmc-cuda-profile-20261029/queue-probe.json"
```

`evidence/queue-probe.json` preserves every measured host call. It has its
own 120-second watchdog and uses the same runtime allocator cap. This is one
run per queue length, used to locate blocking rather than claim a speedup.
The inventory confirms 70 SMs and triton-windows distribution 3.8.0.post29.
The data-ROM launch's 32 blocks cannot occupy all 70 SMs simultaneously.
This is a launch-geometry limit, not measured achieved occupancy.

Source attribution uses pinned upstream `experimental/riscv/chipformer/rv/`.
`tritonexec.py:272` schedules each circuit sublayer and its apply kernel.
`fastexec.py:298` decodes the output and embeds the next self-fed token.
`fastexec.py:329` replays the graph, copies each token, synchronizes, and
transfers the trace to the CPU. `core.py:1218` implements reference snapshot
initialization used only for the labelled performance window.
