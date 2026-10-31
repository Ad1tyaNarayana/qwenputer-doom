# DMC data-ROM launch fusion

This experiment evaluates fewer launches for the unchanged 64 KiB RAM and
4 MiB data-ROM compact circuit. It changes no weights, memory encoding, or
guest computation. It evaluates every head. It is separate from stock
Transformers execution and interactive Doom.

## Dependency analysis

The pinned upstream is `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.
`analyze.py` rebuilds the graph on the Mac with the frozen memory-map
preparation. [dependencies.json](evidence/dependencies.json) records every
group's complete residual read set, taps, outputs, state shapes, source hashes,
and agreement with the profile schedule. The existing
[graph.json](../memory-map/evidence/low-memory/graph.json) records geometry,
not dependency edges.

There are 32 data-ROM sublayers at levels `21 + 2*g`, each containing key
heads `drom[16*g:16*g+16]` and 32 value heads. Their schedule indices are
18 for group 0 and `19 + g` for groups 1 through 31. An FFN at schedule
index 19, level 22, separates the first two groups. Its seven nodes are
`bn.dig0` through `bn.dig3` and `exit.fam.b2` through `exit.fam.b4`.
Group 0 and that FFN remain unchanged.

Every group reads the same residual dimensions through different fixed
weights. Queries read dimension 0, 850 through 855, 857 through 858,
988 through 993, and 995 through 1003. These are ONE and the effective-address
expressions. Gates read dimension 0, 846 through 849, 856, 984 through 987,
and 994. Every nonconstant producer is at level 12. All query taps are
`(1,0,0,0)`. Keys, values, decay expressions, and beta expressions have empty
read sets. Beta defaults to one. No data-ROM head reads another data-ROM
output, any lag history, or an intervening FFN result.

The construction is at upstream
[`core.py:857-881`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/core.py#L857).
The memory-map preparation widens `EA[13:17]` to `EA[13:22]` at
[`prepare.py:66`](../memory-map/prepare.py#L66).
Placement fills 16 key-head slots per odd level rather than expressing a
data dependency at
[`ir.py:156-161`](../../.upstream/qwenputer/experimental/riscv/chipformer/q35/ir.py#L156)
and [`ir.py:198-214`](../../.upstream/qwenputer/experimental/riscv/chipformer/q35/ir.py#L198).
Executor2 traverses those levels in order at
[`tritonexec.py:185-209`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L185).

Each sublayer owns `S[batch,32,128,128]`. All 32 value heads read their
own query-selected rows. They never write or decay S because keys and decay
expressions are empty. The kernel's conditional state updates and row reads
are at [`tritonexec.py:117-145`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L117).
The tables' initial byte encoding is at
[`core.py:1194-1211`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/core.py#L1194).

Outputs overlap. Every group adds to the same four `drom.b` dimensions,
2495 through 2498,
which start at zero, rather than writing disjoint residual slices.
Each sublayer also writes its own entire `hist[batch,3,2679]`. The apply
kernel shifts the two older histories, saves the full pre-add residual as
lag 1, adds the output buffer to the residual, and clears the buffer at
[`tritonexec.py:152-170`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L152).
Consequently later histories include earlier groups' additions. The head
evaluations are independent, but collapsing all history snapshots to the
group-entry residual would violate full-state equivalence.

At the schedule level, exact residual equality is achievable if each
original sublayer's output buffer is bit-identical. Retaining separate
buffers permits the same 31 fp32 additions in order.
Each intermediate residual must also be saved to its original history.
This preserves the cross-sublayer addition order without a tolerance.
Within each sublayer, the original kernel already uses atomic additions at
[`tritonexec.py:146-148`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L146).
For valid address bits, only one of 512 key-head queries and one value-head
gate can select the byte columns. The byte contribution is an integer in
0 through 255 scaled by 1/256. The valid-state byte sum needs at most eight
significand bits. This supports exact addition for the circuit's states,
but does not replace the required token-by-token bit comparison. Arbitrary
invalid residuals can activate several selectors, so arbitrary fp32
reassociation is not assumed safe.

FreeRun requests a row history reset on `load` and feeds back its generated token through
[`fastexec.py:292-307`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L292).
Its normal timing path is
[`fastexec.py:329-340`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L329).

## Prototype design

The data shape is 31 original sublayers, each with 32 value heads, independent
S tables, a separate output buffer, and a full three-position history.
`install(executor)` packs the existing sparse weights and state storage and
replaces only those 31 scheduled evaluations. Other sublayers still call
Executor2.step unchanged.

Two designs were considered inline. A single shared output buffer would
change atomic accumulation across sublayers and lose intermediate history
values. The selected design uses one parallel head launch with 992 blocks
and one ordered apply launch. The head launch calls the upstream Triton
head function with each original sublayer's private output buffer. The apply
launch reproduces every residual addition and history snapshot in order.
The first data-ROM sublayer stays before the intervening FFN.

Phase A passes the independent-evaluation gate. Phase B fails the required
bit-exact gate. The prototype remains experimental and is not accepted.
No speedup is claimed.

## Equivalence results

The GPU used Python 3.14.2, PyTorch 2.14.1+cu130, CUDA 13.0, and
triton-windows 3.8.0.post29 on the 70-SM RTX 5070 Ti. Mac analysis used
Python 3.13.4 and PyTorch 2.15.0. The GPU runtime's prepared `tritonexec.py`
was byte-identical to its clean pinned checkout. Cross-host source checks
normalize Windows CRLF line endings only.

The verifier captures both self-fed executors and a comparison kernel in
one graph. It compares every 32-bit word of every S tensor and full lag
history after every token. It also compares the residual before feedback,
the next-token embedding, both halves of the int64 token, and the executor
output buffers. It uses direct bit comparisons, without hashes or a
tolerance. Six one-bit mutations independently exercise S, histories,
embedding, token, output buffer, and residual detection. Each detects
exactly one changed word. The fused temporary output buffer is checked
separately at the end of successful runs.

The requested 65,536-token attempt stopped at its first 4,096-token boot
chunk. [equivalence-failure.json](evidence/equivalence-failure.json) records
the first mismatch at token 2,382, using one-based positions. The captured
residual differs in seven words. Histories at schedule indices 28 through
50 each differ in one word at that position. These are the histories after
the `drom112` through `drom127` sublayer. Across the chunk, 366 word
differences were observed and the maximum absolute fp32 difference was
1.0. States match again at the end of the chunk. Both 4,096-token traces
have zero independent-reference mismatches. That token match does not
satisfy state equivalence.

The earlier [256-token smoke run](evidence/smoke-detail.json) passed boot
state and reference comparisons, then failed the reference-seeded rendering
state check at token 28. Its maximum absolute difference was also 1.0,
with zero reference-token mismatches on both 256-token rendering traces.
The seed is token 6,470,215, instruction 3,966,856, PC `0x63864`.
That window is reference-seeded for equivalence and performance experiments
only. It does not establish neural execution before the seed.

The discrepancy depends on execution conditions. A
[512-token run with synchronization after every token](evidence/synchronized-smoke.json)
passes both windows. A separate [28-token run](evidence/short-28.json)
also passes. These short passes do not erase the failures. An
[unchanged-versus-unchanged control](evidence/baseline-control-4096.json)
passes state and reference checks for 4,096 tokens in each window. This
control is bounded and does not prove universal baseline determinism.

The first boot mismatch appears within the fused data-ROM sequence.
Preserving cross-sublayer addition order did not establish bit equivalence.
Within-sublayer atomic accumulation order and changed Triton compilation
are candidate causes. Neither cause was isolated. The maximum difference
of 1.0 includes downstream residual effects and is not a measured raw
data-ROM byte error. No claim that exact fusion is impossible is made.
Further fusion runs and all timing were stopped at this gate. No
tolerance-based acceptance was attempted.

## Reset finding

The initial-state check found that `Executor2.reset([row])` calls
`hist[rows].zero_()` at
[`tritonexec.py:260-263`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L260).
PyTorch advanced indexing returns a copy, so that expression does not clear
the original history. [reset-check.json](evidence/reset-check.json) reproduces
the behavior on the Mac. The GPU verifier's initial-state check also found
different histories after the original warm-up and load sequences.
This harness explicitly calls `reset()` without row indices on both sides
and zeros the stale output token before each window. The circuit and
upstream source are unchanged.

This differs from the profile harness's reset setup, although the seed and
timed FreeRun path are the same. The old profile's claim that every `load`
clears lag histories needs correction. The impact on historical timings
and window prefixes is not established here. No old performance evidence
was modified.

## Timing status

No before/after benchmark ran. The table retains the historical profile
only as context. Total time is the inverse of uninstrumented median
throughput. Data-ROM time comes from the separate instrumented profile and
must not be subtracted directly from that wall time.

| Window | Executor | Data-ROM µs/token | Total µs/token | Tokens/s |
| --- | --- | ---: | ---: | ---: |
| Boot | Historical profile | 262.270 | 482.170 | 2,073.96 |
| Boot | Fusion | Not measured | Not measured | Not measured |
| Frame-2 rendering, reference-seeded | Historical profile | 260.136 | 481.562 | 2,076.58 |
| Frame-2 rendering, reference-seeded | Fusion | Not measured | Not measured | Not measured |

The gated `--mode bench` path is provided but unexecuted. It requires a
successful full 65,536-token verification report from the same scripts.
It alternates five repeats per executor and window through `FreeRun.run`,
then uses separate 256-token profiler captures to attribute data-ROM heads
and apply kernels. It is not a validated measurement path yet.

Each remote process has a watchdog, defaults to 1,100 seconds, and inherits
the profile runtime's 8 GiB PyTorch allocation cap. No system tools were
installed. Private ELF files, full traces, snapshots, and tensors stay on
the GPU machine. Evidence contains counts, hashes, and tensor metadata.

Recorded harnesses evolved during diagnosis. The final harness adds
per-tensor mismatch counts and clears that diagnostic buffer between
windows. Each report records its script hashes. `fusion.py` remained
unchanged across the recorded GPU runs. Failed attempts used no benchmark
timings. No commits were made.

## Reproduce the CPU analysis

```sh
.venv/bin/python experiments/dmc-rom-fusion/analyze.py --out experiments/dmc-rom-fusion/evidence/dependencies.json
```

Sync and run the bounded GPU verifier from the Mac repository root.

```sh
tar -cf - experiments/dmc-rom-fusion | ssh gpu "tar -xf - -C C:/Users/Admin/qwenputer-games-work/repo"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-rom-fusion/run_gpu.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/dmc-rom-fusion-equivalence --tokens 65536 --max-seconds 1100"
```

The expected acceptance result remains failure or unestablished until a new
full run passes. `--baseline-control --tokens 4096` compares two unchanged
executors. `--chunk-tokens 1` adds a host synchronization after every token
for diagnosis and changes execution conditions. Neither replaces the
required queued-window comparison.

## Diagnosis, 2026-10-29

The original two-launch prototype has a read-before-clear race inside
`_ordered_apply`. This supersedes the earlier unisolated atomic-order and
floating-point-compilation candidates. The discrepancy also occurs when the
fused executor runs alone. It is not caused by running two executors in the
verification graph.

The generated PTX uses different lane layouts for the output load and zero
store. In the first half of a 1,024-word tile, thread `t` loads words
`4*t` through `4*t+3`, but clears words `t`, `t+128`, `t+256`, and `t+384`.
For example, warp 0 can clear word 128 before warp 1 loads it. Source order
within one thread does not order these accesses across warps. The loop's
history-layout conversions contain barriers before the output load, but
there was no barrier between that load and the clear.
[The standalone reproducer](evidence/diagnosis-apply-race.json) fills every
output word with one and expects 31 after the 31 additions. The original
kernel produces 873,089 wrong words across 8,192 queued replays. A block
barrier before clearing the output produces zero wrong words under the same
conditions. The fix adds that barrier at
[`fusion.py`](fusion.py#L37). It preserves every head, addition, and history
snapshot. It changes no circuit, weight, or upstream file.

### Buffer and capture audit

[The GPU allocation and capture audit](evidence/diagnosis-audit.json)
records 2,310 tensor references. It finds zero overlapping byte ranges
between executors. The shared `base.c` object contains circuit descriptions
and CPU initialization data. Each executor allocates its GPU tensors through
[`Executor2.__init__`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L176)
and [`FreeRun.__init__`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L240).
`install()` changes only the alternate executor's step dictionaries.

| Buffer | Access in fused sequence | Ownership and reuse |
| --- | --- | --- |
| `x`, shape `[1,2679]` | Prefix writes it; parallel heads read it; ordered apply reads and writes it; suffix and feedback update it | Private to each `FreeRun`; intentionally reused on later tokens |
| Packed `q/k/v/z/a/b` CSR pointers, indices, weights | Read by head launch | New `cat` allocations at [`fusion.py:60-77`](fusion.py#L60); empty arrays have private dummy storage; no baseline aliases |
| Packed `bconst/odim/ow/nrows/khead` | Read by head launch | New allocations at [`fusion.py:78-80`](fusion.py#L78); immutable across tokens |
| Packed `S`, shape `[1,992,128,128]` | Head launch reads selected rows; no writes or decay for these data-ROM heads | New allocation at [`fusion.py:81`](fusion.py#L81); alternate step views cover disjoint 32-head slices; baseline remains separate |
| Packed `hist`, shape `[31,3,2679]` | Apply shifts both older lags and saves each intermediate residual | New allocation at [`fusion.py:82-86`](fusion.py#L82); disjoint alternate step views; reused recurrently across tokens |
| Private `y`, shape `[31,2679]` | Heads atomically add within each private group; apply loads and clears it | New zero allocation at [`fusion.py:83`](fusion.py#L83); disjoint group slices; the proven race is between its load and clear within one apply block |
| `ex.y`, other sublayer S, histories, and weights | Original prefix and suffix kernels read or update them | Private original executor allocations; selected groups no longer use `ex.y` |
| Feedback `fam_idx/pay_idx/fam_w/pay_w/base/bits/ids/small/row0/table`, `tok` | Decode reads residual and tables, writes token and next `x` | Private allocations in [`FreeRun`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L254); no baseline aliases |
| Verifier residual copies, pointer tables, counts, differences, per-tensor counts, position, token records | Residual wrappers write copies; comparison reads both states and writes diagnostics; advance writes records and position | Separate allocations at [`run_gpu.py:122-160`](run_gpu.py#L122); pointer tables intentionally reference state; no executor uses diagnostic storage |
| Kernel registers and shared memory | Private execution scratch | Both fused kernels report zero global scratch, zero profile scratch, and zero spills; shared memory is per block, 512 bytes for heads and 2,048 bytes for original apply |

The unchanged prefix and suffix FFNs read their `a/b` CSR arrays, `coef`,
and `out`, atomically accumulate into private `ex.y`, then add and clear it.
Their head sublayers read the same named CSR and output-mapping arrays as
the packed heads, and update only their own S and histories. The complete
schedule dispatch is at
[`tritonexec.py:272-288`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L272).
The fusion module has no global GPU tensor allocation. Its retained tensors
are the returned packed-storage dictionary and the alternate executor's
state views. PyTorch feedback temporaries are scoped to the graph's ordered
operations; they are not additional persistent executor state.

Although a packed history pointer is passed to the head function, the
asserted query and gate indices are below `D`, and key, value, decay, and
beta expressions are empty. These heads therefore load no lagged values.
See [`fusion.py:51-59`](fusion.py#L51) and
[`_lag_load`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py#L75).
The stale-history and residual-boundary candidates do not explain the
standalone apply reproducer.

Every circuit launch in the audited joint capture uses the same nondefault
capture stream. The baseline contributes 108 circuit launches and the fused
executor contributes 48. Feedback and comparison use that same stream.
The CUDA driver reports 207 graph nodes and 206 edges, with maximum incoming
and outgoing degree one. The graph is one ordered chain. Its relevant
sequence is baseline prefix through feedback, alternate prefix through
parallel heads, ordered apply, alternate suffix through feedback, then
comparison. The actual verifier additionally advances the record position
after comparison at [`run_gpu.py:208-214`](run_gpu.py#L208).
There is no missing launch dependency between the fused head and apply.
Graph replays are submitted serially on the current stream at
[`run_gpu.py:225-228`](run_gpu.py#L225). The separate capture warm-up stream
has waits in both directions at
[`fastexec.py:313-318`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L313)
and does not remain a concurrent producer during verification.

### Exact isolated records

[`isolation.py`](isolation.py) constructs one executor per process, with its
own graph. Its main runs queue 4,096 tokens before synchronization. A GPU
kernel compares every raw 32-bit S word with the previous token and records
all changed words. Overflow aborts the run. The CPU reconstructs the exact S
bytes and updates SHA256 leaves of 4,096 words each. The S root hashes every
leaf in order, including unchanged leaves. The recorder separately hashes
the full raw bytes of every lag history, pre-feedback residual, next
embedding, both int64 token words, and ordinary output buffer after every
token. The full-state root combines these ordered hashes. The private fused
output buffer is additionally required to be all zero after every token.
There is no numerical tolerance or arithmetic checksum. One-bit controls
exercise the S delta recorder and every dynamic tensor, and final S
reconstruction is checked against the GPU bytes.

Per-token JSONL records, initial states, raw mismatch snapshots, and PTX stay
under `C:\Users\Admin\qwenputer-games-work\artifacts\rom-*` on the PC.
Only counts, hashes, allocation metadata, and comparison reports are exported.
A 256-token-chunk baseline run is a supplemental control, separate from the
main 4,096-token-chunk comparisons.

The original isolated fused run differs at boot token 3,200 and seeded
rendering token 1,107. It has 45 and three per-token digest mismatches,
respectively, across 65,536 boot and 4,096 rendering tokens. All S tensors,
next embeddings, tokens, and cleared output buffers match. The first boot
difference affects the residual and histories 28 through 50; rendering
affects the residual and histories 35 through 50. Both traces still match
their independent token references. See
[the isolated comparison](evidence/diagnosis-isolation-original.json).

Splitting the path gives bounded passes for parallel heads with upstream
per-sublayer applies and for original separate head launches with ordered
apply, each over 4,096 tokens in both windows. See
[heads only](evidence/diagnosis-heads-only.json) and
[apply only](evidence/diagnosis-apply-only.json).
Those passes did not exonerate apply. The standalone reproducer supplies
nonzero outputs in every lane and exposes the within-block race directly.

A retained unpatched control localizes an actual queued circuit failure at
boot token 6,119. Residual words 2495, 2499, 2502, and 2503 differ. Histories
28 through 50 each differ only at lag-1 word 2495, by exactly `25/256`.
All S words match. The largest residual difference is 1.0. The full raw
differences remain on the PC; [the exported word report](evidence/diagnosis-unsafe-words.json)
contains indices and counts. Word 2495 belongs to tile 2, where warp 1 can
clear it before warp 3 loads it. This directly connects the lane-layout
race to a dropped valid-state byte contribution and its downstream flags.

The earlier attribution of the first affected history range to the
`drom112` through `drom127` sublayer is also superseded. Schedule index 27
contains `drom128` through `drom143`; history 28 is the first pre-add
snapshot after that sublayer. The [dependency inventory](evidence/dependencies.json)
records those names and indices.


### Baseline determinism

Three main baseline processes use 4,096-token queue chunks. The first
records 65,536 boot and 4,096 rendering tokens. Two repeats record 65,536
tokens in each window. The first repeat matches every digest over the
shared 65,536-token boot and 4,096-token rendering windows. The second
repeat matches both complete 65,536-token windows, including byte-identical
JSONL files. See [repeat 1](evidence/diagnosis-baseline-repeat-1.json) and
[repeat 2](evidence/diagnosis-baseline-repeat-2.json).
The supplemental 256-token-chunk run matches the same boot and rendering
prefixes. These are bounded determinism checks, not a universal claim.

### Fixed isolated equivalence

With the barrier, the isolated fused executor matches the baseline at every
token in both complete 65,536-token windows. Every full-state digest matches,
all independent reference tokens match, every fused output buffer is zero
after every token, and final reconstructed S bytes match the GPU. See
[the fixed isolated comparison](evidence/diagnosis-isolation-fixed.json)
and [the fixed run metadata](evidence/diagnosis-fixed-b.json).
Peak PyTorch allocation is below 6.86 GiB, within the existing 8 GiB cap.

### Direct bit-by-bit acceptance

The corrected original verifier also passes every 32-bit word of all 86
state tensors over both complete 65,536-token windows. Both reference traces
match. Each of the six independent one-bit controls detects exactly one
changed word. See [the acceptance report](evidence/diagnosis-verify.json).
Its actual graph has 210 nodes and 209 edges in one chain, including the two
residual-copy nodes, comparison, and position advance. This records the real
verifier topology in addition to the separate allocation audit.

The earlier unaccepted status is superseded for this barrier-patched
prototype and these bounded windows. The original failure reports remain
valid evidence for the unpatched source. `check_evidence.py` checks that
historical source hash and writes `historical-checks.json` without replacing
the original `checks.json`. The new `check_diagnosis.py` checks the diagnosis,
fixed isolation, direct gate, and timing evidence together.

### Gated before-and-after timing

The earlier no-timing status and unvalidated benchmark-path status are
superseded by this fixed-source run. Five alternating repeats per executor
and window use 65,536 tokens per call through the normal `FreeRun.run` path.
The baseline uses the same explicit full reset as fusion. Verification,
reference preparation, state reset, compilation, and profiling are outside
the timer. Allocation, graph replay, feedback, token copies, final
synchronization, and CPU trace conversion are inside it.

| Window | Executor | Median µs/token | Median tokens/s | Tokens/s range |
| --- | --- | ---: | ---: | ---: |
| Boot | Original | 479.334 | 2,086.23 | 2,085.81 to 2,086.33 |
| Boot | Fixed fusion | 253.759 | 3,940.75 | 3,940.39 to 3,940.87 |
| Frame-2 rendering, reference-seeded | Original | 479.283 | 2,086.45 | 2,086.35 to 2,086.46 |
| Frame-2 rendering, reference-seeded | Fixed fusion | 253.711 | 3,941.50 | 3,941.16 to 3,941.75 |

The measured speedup is 1.889× in each window, with about 47.06% less
elapsed time per token. The gain is much larger than the five-run ranges.
All 1,310,720 normal-path tokens match their independent references. All
1,024 profiler tokens also match. See [the full benchmark](evidence/diagnosis-bench.json)
and [the reduced timing evidence](evidence/diagnosis-timing-summary.json).

The separate 256-token profiler captures retain every scheduled circuit
kernel. They attribute the change to data-ROM work while the remaining
circuit and feedback still execute. The grid changes from 32 blocks per
original head launch to 992 in the batched launch. This increases available
parallelism on 70 SMs; achieved occupancy is not measured.

| Window | Executor | Data-ROM heads µs/token | Data-ROM apply µs/token | Circuit kernels/token |
| --- | --- | ---: | ---: | ---: |
| Boot | Original | 259.738 | 41.391 | 108 |
| Boot | Fixed fusion | 53.223 | 21.647 | 48 |
| Frame-2 rendering, reference-seeded | Original | 259.681 | 41.356 | 108 |
| Frame-2 rendering, reference-seeded | Fixed fusion | 53.217 | 21.665 | 48 |

These profiler durations are separate measurements, not additive parts of
the uninstrumented wall-time table. The remaining sublayers, feedback,
history traffic, and synchronization limit further gains. The experiment
does not identify a VRAM-bandwidth limit or measure full-frame latency.

The host has 12 logical CPUs and measured 6% total CPU use immediately
before timing. Desktop graphics applications remained running. During
labelled timing samples, median GPU utilization is 100%; median SM clocks
are 2,887 MHz for the baseline and 2,880 MHz for fusion. This clock
difference cannot explain the rate gain. The host snapshot is in
[diagnosis-bench-host.json](evidence/diagnosis-bench-host.json). Every remote
run stays below its watchdog and PyTorch allocation cap. Raw profiler
traces stay on the PC.

[The final checker](evidence/diagnosis-checks.json) validates the historical
failure, race reproducer, baseline repeats, fixed isolation, direct
acceptance, and all five timing repeats. No tolerance is used.

### Historical reset impact

The historical first-frame proof used freshly allocated zero histories at
construction, but its retained prefix began after one generated warm-up
token and a reload. `runtime.build()` constructs and loads `FreeRun` at
[`runtime.py:28`](../dmc-cuda/source/runtime.py#L28), then executes
`fr.run(1)` and `fr.load(...)` at
[`runtime.py:31`](../dmc-cuda/source/runtime.py#L31).
`capture()` clears histories after its internal warm-up at
[`fastexec.py:326`](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py#L326),
but `run(1)` subsequently generates another token. The following `load`
calls the ineffective `reset([row])`. The GPU audit measures 10,833 nonzero
history words on return from the unchanged `runtime.build()` and zero after
an explicit full `reset()`.

The prefix worker immediately starts generation from that returned state at
[`prefix_probe.py:10-14`](../dmc-cuda/source/prefix_probe.py#L10).
Its reload requested the intended row reset, which did not occur. It did
not start its retained prefix with fresh zero history. This supersedes any implication
that the original prefix's reload guaranteed zero histories. The later
full-frame worker restores actual S, histories, embedding, and token from
the saved GPU checkpoint at
[`full_worker_original.py:43-45`](../dmc-cuda/source/full_worker_original.py#L43)
and [`runtime.py:43-48`](../dmc-cuda/source/runtime.py#L43).

The bug qualifies the startup-state claim. It does not erase the recorded
zero token mismatches or the independent frame and final-RAM byte
comparisons for the execution that actually occurred. It also does not
establish full-state equality to a clean zero-history startup. The profile's
statement that `FreeRun.load(GO)` clears lag histories is superseded.
No historical timing is silently replaced, and no upstream source is edited.

### Reproduce the diagnosis

Run from the Mac `qwenputer-games` repository. All private states and digests
remain on the PC. Use separate output directories for baseline repeats and
fusion. The isolated command records both full windows; change `--side`
and `--out` for the fused run, and add `--compare-dir` pointing to a completed
baseline output directory to retain the first raw mismatch automatically.

```sh
tar -cf - experiments/dmc-rom-fusion | ssh gpu "tar -xf - -C C:/Users/Admin/qwenputer-games-work/repo"
ssh gpu "C:/Users/Admin/qwenputer-games-work/.venv/Scripts/python.exe C:/Users/Admin/qwenputer-games-work/repo/experiments/dmc-rom-fusion/isolation.py --probe C:/Users/Admin/qwenputer-games-work/probes/dmc-frame-20261029 --repo C:/Users/Admin/qwenputer-games-work/repo --out C:/Users/Admin/qwenputer-games-work/artifacts/rom-isolation-base-new --side baseline --tokens 65536 --render-tokens 65536 --chunk-tokens 4096 --max-seconds 1100"
```

`compare_isolation.py --left BASE --right FUSED --out REPORT.json` compares
every recorded full-state digest. `--common-prefix` checks the common window
when the rendering lengths differ. `apply_race.py --probe PROBE --out REPORT`
runs the standalone unpatched and fixed apply kernels under a 120-second
watchdog. `--part unsafe` on the isolated fused runner deliberately restores
the pre-barrier apply kernel for diagnosis. It is not an accepted executor.

The normal `run_gpu.py --mode verify` followed by `--mode bench --repeats 5`
uses the same `--out` directory for its source-hash gate. The recorded fixed
gate and timing directory is
`C:\Users\Admin\qwenputer-games-work\artifacts\rom-fixed-gate`.
Each process is bounded to 1,100 seconds unless a smaller limit is supplied.
No commits, upstream edits, circuit changes, head skipping, or system
installations are part of these commands.

### Barrier placement revision, 2026-10-29

After the gated run, `_ordered_apply` was revised to load each group's
histories and output before one block barrier, then perform all of that
group's stores. The history shift previously relied on barriers inserted by
Triton layout conversions. The revision keeps the same additions in the same
order. The recorded script hashes and the results above apply to the earlier
source; the revised source needs a new full equivalence run before reuse.

### Revised barrier re-verification, 2026-10-29

The revised `_ordered_apply` source has SHA256
`ccb8abe273c55f100d93369be8938a031620d243af6bd8a645b04e791dceb678`.
Fresh isolated original and fused processes each generated 65,536 tokens
in both boot and reference-seeded frame-2 rendering windows. Every per-token
full-state raw-bit digest matched, and every reference token matched.
The one-bit recorder controls and final S reconstruction passed.
The fused private output was zero after every token. Peak fused allocation
was 7,363,432,448 bytes, below the 8 GiB cap.
See [the comparison](evidence/revision-isolation.json),
[original metadata](evidence/revision-base.json), and
[revised fusion metadata](evidence/revision-fused.json).

These runs use separate private directories `artifacts/rom-revision-base`
and `artifacts/rom-revision-fused`, with 4,096 queued tokens per chunk and
a 1,100-second watchdog per process. Earlier diagnosis reports are preserved.
The [direct word gate](evidence/revision-verify.json) also passes every raw
32-bit word of all 86 tensors after every token in both full windows, with
zero reference-token mismatches. All six one-bit controls detect exactly
one changed word. The [standalone reproducer](evidence/revision-apply-race.json)
queues 8,192 graph replays. The revised kernel produces zero wrong words;
the unsafe control produces 960,715. The [new five-repeat timing](evidence/revision-bench.json) and
[reduced timing evidence](evidence/revision-timing-summary.json) use the same
verified scripts. Median original to revised fusion time is 479.704 to
252.170 microseconds per token at boot, and 479.640 to 252.109 in rendering.
The speedup is 1.902 times in both windows. Original rate ranges are
2,083.97 to 2,084.76 tokens/s at boot and 2,070.63 to 2,084.95 in rendering.
Fusion ranges are 3,965.48 to 3,965.85 and 3,928.06 to 3,966.63 respectively.
All 1,310,720 timing tokens and 1,024 profiler tokens match references.
The [host snapshot](evidence/revision-bench-host.json) records 12 logical CPUs
and 0% CPU load before timing, with desktop graphics applications present.
`check_diagnosis.py` now requires all revision-specific acceptance and timing
reports and writes [revision-checks.json](evidence/revision-checks.json).
The earlier `diagnosis-checks.json` remains historical evidence.
