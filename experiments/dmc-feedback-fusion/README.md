# DMC token-feedback fusion

This experiment stacks token-feedback launch fusion on the verified revised
data-ROM fusion. It preserves the circuit, every head, weights, memory
encoding, and fp32 state. The exact-equivalence gates and five-repeat timing pass. The cumulative
speedup is 2.102 times at boot and 2.103 times in reference-seeded rendering.

## Dependency analysis and design

The [fresh profile](evidence/rom-profile.json) uses the separate 256-token
captures from the revised-ROM five-repeat run. `profile_summary.py` checks
48 circuit kernels and 25 feedback kernels per token against the original
schedule. Shares use the captured stream span, not uninstrumented wall time.
The largest groups in rendering are FFNs at 55.779 microseconds per token,
data-ROM heads at 53.190, RAM heads at 41.333, feedback at 29.879, other apply
at 23.750, and ROM apply at 20.464. Gaps are 17.781.

The circuit completes before feedback at pinned upstream
[fastexec.py:298](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L298).
The residual dimensions used by family and payload decoding are fixed GPU
index arrays constructed at [fastexec.py:255](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L255). Family decoding rounds
five values, multiplies by powers of two, sums, converts to int64, and clamps
the family index. Payload decoding rounds 16 values and forms the int64
payload. The token is the selected family's token ID plus that payload.
Small families select a clamped table row. Big families add their base
embedding to the payload-bit matrix product.

The [GPU geometry inspection](evidence/feedback-geometry.json) finds 20
families, 16 payload bits, 2,679 residual dimensions, and 948 small-family
table rows. Each family and embedding coordinate has at most one nonzero
payload coefficient. Coefficients are 0, 1, or 256, and base coordinates
are -128, 0, 1, or 128. The installer must assert these properties. For finite
rounded bit inputs, each dot product has at most one nonzero term. No sum
of multiple nonzero fp32 contributions is reassociated. Valid circuit
family and payload bits make weighted decode sums exactly representable.
Rounding retains ties-to-even. Family and row clamping retain int64 order.

All feedback tables are immutable and private to FreeRun. Only the token
and next embedding are persistent outputs. Feedback does not read or write
S, histories, or ordinary circuit output buffers. The same residual buffer
is read for decoding and overwritten by the next embedding. A single block
must finish all cross-warp residual reads before any embedding stores, using
an explicit `tl.debug_barrier()`. No layout-conversion barrier is assumed.
There are no concurrent producers or cross-block dependencies in that design.

Two structures were compared inline. One block can decode and embed with no
intermediate buffer. Two kernels can decode into a small scratch tensor and
then embed across blocks, using the launch boundary for ordering. The first
is selected for this 2,679-dimensional batch-one path because it removes all
24 extra launches and avoids scratch state. The public adapter is
`install(FreeRun)`, returning the existing ROM-fusion metadata. It installs
ROM fusion on `fr.ex`, then replaces only that instance's `_step` with
circuit evaluation followed by `_decode_embed`. The original executor remains
unmodified. A separate `install_feedback` supports standalone stress tests.

Folding all other applies into following kernels is not immediately safe.
[tritonexec.py:272](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L272) launches multi-block head or FFN work, then applies
all residual updates before the next sublayer reads arbitrary sparse residual
indices. A block barrier cannot establish that global dependency. Batching
non-ROM heads also requires a separate read/write dependency proof. Feedback
is the largest measured launch-fusion candidate without those circuit-state
dependencies. FFNs and RAM remain larger total compute groups.

## Verification plan

First compare the standalone kernel with the original PyTorch feedback under
thousands of queued graph replays. Then compare all recurrent state against
the original executor in isolated processes with raw-bit per-token digests,
65,536 tokens in each window. Run direct word acceptance with one-bit controls.
Only after those gates pass, time five alternating repeats of original,
ROM-only, and ROM-plus-feedback executors in both windows.

Frame-2 rendering is reference-seeded for equivalence and performance only.
Every remote process uses a watchdog and the existing 8 GiB allocator cap.
There are no upstream edits, commits, installations, or skipped heads.

## Standalone stress result

[feedback-race.json](evidence/feedback-race.json) records 8,192 graph replays
queued before synchronization, with zero embedding-bit or int64 token
differences against the original PyTorch feedback. Cases span all 32 encoded
family values, random 16-bit payloads, signed half-integer rounding, and
family and table-row clamping. Peak allocation is 447,611,392 bytes.
The kernel source SHA256 is
`943e5e4bed66317955dd76f76c94be3c625d1438b34478747fa9b6e1ccd15258`.
The first launch attempt failed before kernel execution because the circuit
package path is established by `runtime.build()`. The stress runner now
imports fusion after that setup. No numerical failure was observed.

The full isolated comparison reuses the fresh original-executor records from
step 0, `artifacts/rom-revision-base`. The original executor and initial
fixtures are unchanged. The new executor runs in its own process and graph
under `artifacts/feedback-isolation`, with 4,096 queued tokens per chunk.

## Isolated full-state result

[isolation-comparison.json](evidence/isolation-comparison.json) reports zero
per-token full-state digest mismatches against the original executor across
65,536 boot tokens and 65,536 reference-seeded rendering tokens.
[isolation.json](evidence/isolation.json) reports zero reference-token
mismatches, successful one-bit recorder controls, exact final S
reconstruction, and zero private output after every token. Peak allocation
is 7,296,323,584 bytes. Both JSONL files are also byte-identical to the
revised ROM-only records. Digests include raw S bytes, all lag histories,
pre-feedback residual, next embedding, both token words, and output buffers.

## Reproduction

Run from the Mac repository root. The private PC fixtures and continuing ELF
are those used by the ROM experiment. Sync both experiment directories with
tar over `ssh gpu`. The commands below run through cmd, not PowerShell.

```sh
tar -cf - experiments/dmc-rom-fusion experiments/dmc-feedback-fusion | ssh gpu "tar -xf - -C C:/Users/Admin/qwenputer-games-work/repo"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-feedback-fusion/feedback_race.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --out C:/Users/Admin/qwenputer-games-work/artifacts/feedback-race.json --tokens 8192"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-feedback-fusion/isolation.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/feedback-isolation --side fused --tokens 65536 --render-tokens 65536 --chunk-tokens 4096 --compare-dir C:/Users/Admin/qwenputer-games-work/artifacts/rom-revision-base --max-seconds 1100"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-rom-fusion/compare_isolation.py --left C:/Users/Admin/qwenputer-games-work/artifacts/rom-revision-base --right C:/Users/Admin/qwenputer-games-work/artifacts/feedback-isolation --out C:/Users/Admin/qwenputer-games-work/artifacts/feedback-isolation-comparison.json"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-feedback-fusion/run_gpu.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/feedback-gate --mode verify --tokens 65536 --max-seconds 1100"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-feedback-fusion/run_gpu.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/feedback-gate --mode bench --tokens 65536 --repeats 5 --max-seconds 1100"
```

Each run uses its own watchdog. Run them sequentially, with no other GPU
experiment active. Bench requires a full passing direct-word report with the
same script and ROM hashes. The direct verifier records one-bit controls and
the actual CUDA graph topology. The benchmark creates separate original,
ROM-only, and stacked instances, resets all histories explicitly, and clears
scratch outputs before each call. Timers include normal FreeRun trace-buffer
allocation, graph replay, feedback, token copies, synchronization, and CPU
trace conversion. Construction, capture, reset, reference generation,
comparison, profiling, and evidence writes are excluded.

Private states and JSONL records stay on the PC. Export only report JSON.
Reduce the original ROM capture with `profile_summary.py --input
artifacts/rom-revision-gate --schedule evidence/schedule.json --out REPORT`.
For the stacked capture, use `--input artifacts/feedback-gate` and
`--feedback-kernels 1`. Run `check_evidence.py` locally after exporting the
reports under their documented names. `check_diagnosis.py` in the ROM
experiment separately checks step 0.

## Direct word result

[verify.json](evidence/verify.json) passes all 86 state tensors after every
token in both 65,536-token windows, with zero differing 32-bit words and
zero reference-token mismatches. All six one-bit controls detect exactly
one changed word. The comparison graph has 186 nodes and 185 edges.
It contains the original circuit and feedback, stacked circuit and one
feedback kernel, both residual copies, word comparison, and record advance.
Peak allocation is below the 8 GiB cap. The kernel and harness hashes in
this report bind the benchmark gate.

## Fresh ROM-only profile ranking

These are device microseconds per token in the separate profiler captures.
They are not additive components of the normal-path wall-time table.

| Cost | Boot | Rendering |
| --- | ---: | ---: |
| ALU/decode FFNs | 55.794 | 55.779 |
| Data-ROM heads | 53.221 | 53.190 |
| RAM heads | 41.344 | 41.333 |
| Feedback and embedding, 25 kernels | 29.891 | 29.879 |
| Other residual/history apply | 23.751 | 23.750 |
| Data-ROM apply | 20.478 | 20.464 |
| Device-stream gaps | 17.600 | 17.781 |
| Other heads | 15.200 | 15.201 |
| Code-ROM heads | 10.005 | 10.013 |
| Trace copy | 0.821 | 0.818 |

The profiler stream spans are 268.104 and 268.207 microseconds per token.
The [stacked capture](evidence/new-profile.json) confirms 48 circuit kernels
and one feedback kernel per token. Feedback falls to 5.736 microseconds at
boot and 5.782 in rendering. Stream spans fall to 239.149 and 239.208.
FFNs, data-ROM heads, and RAM heads remain the largest compute costs.

## Accepted three-way timing

Five alternating repeats per side and window each generate 65,536 tokens
through normal `FreeRun.run`. The table reports medians and the full rate
range. See [bench.json](evidence/bench.json),
[timing-summary.json](evidence/timing-summary.json), and
[the standalone chart](timing.html).

| Window | Executor | Median µs/token | Median tokens/s | Tokens/s range |
| --- | --- | ---: | ---: | ---: |
| Boot | Original | 479.777 | 2084.30 | 2083.78 to 2084.38 |
| Boot | Revised ROM fusion | 251.717 | 3972.72 | 3968.74 to 3973.32 |
| Boot | ROM plus feedback | 228.291 | 4380.38 | 4361.30 to 4381.25 |
| Rendering, reference-seeded | Original | 479.705 | 2084.61 | 2077.61 to 2084.85 |
| Rendering, reference-seeded | Revised ROM fusion | 251.663 | 3973.57 | 3967.75 to 3973.79 |
| Rendering, reference-seeded | ROM plus feedback | 228.155 | 4382.98 | 4361.54 to 4383.56 |

Cumulative speedups are 2.1016 times at boot and 2.1025 in rendering.
Incremental speedups over revised ROM fusion are 1.1026 and 1.1030.
The stacked path saves 23.426 and 23.507 microseconds per token relative
to ROM alone, consistent with the measured feedback cost reduction.
All 1,966,080 timed tokens and 1,536 profiler tokens match their references.
The gain exceeds the observed run ranges. The three-way ROM timing is
slightly different from step 0 because it is a separate measurement.

The [host snapshot](evidence/bench-host.json) records 12 logical CPUs and
0% CPU load before timing. Desktop graphics applications remain present.
During labelled samples, median GPU utilization is 100% for all sides.
Median SM clocks are 2,887 MHz for original and 2,880 MHz for both fused
sides. Clock differences do not explain the incremental gain. Peak benchmark
allocation is 611,496,448 bytes. Every GPU process completes below its
watchdog and allocation cap.

[checks.json](evidence/checks.json) binds the standalone stress result, both
full isolated windows, direct words, one-bit controls, all timing repeats,
profiles, current scripts, and accepted ROM hash. No tolerance is used.

## Limits and remaining work

These are bounded boot and reference-seeded rendering windows. There is no
full-frame timing, continuing neural gameplay, input-latency test, or 24 fps
claim. Invalid nonfinite residuals and other circuit geometries are outside
this experiment. The installer rejects batch sizes other than one and
embedding matrices with overlapping nonzero bit coefficients.

The next launch-fusion candidate is other residual/history apply, now about
23.7 microseconds per token. Folding it into a multi-block successor requires
a global ordering design; a block barrier alone is insufficient. FFN and
RAM kernel improvements require their own arithmetic and dependency analysis.
No upstream code, circuit, weights, memory encoding, or heads were changed.
No commits or system installations were made. The pstack Sequence Work into
Verifiable Units principle kept timing behind exact acceptance. Model the
Domain kept the adapter organized around immutable feedback tables and two
persistent outputs. Benchmark Checklist kept profiler and wall time separate.
