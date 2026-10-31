# One RV32 instruction per neural transition

Design investigation, 2026-10-29. Pinned compiler:
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c`.

## Recommendation and status

Do not approve a full CPU rewrite on the assumption that replacing value tokens
with a constant STEP token is sufficient. The pinned stock-layer mapping has a
missing return path: a computed result near the end of the network cannot update
an earlier layer's recurrent table on the next transition merely by adding a
convolution tap. This applies to registers, RAM, and especially the branch-to-PC
path. More layers do not repair that direction of dependency.

Recommend a gated investigation of a **full-word commit packet**, with word-based
RAM and a combinational divider. First resolve how that packet returns to the
front of the network in stock execution. The two concrete transports below have
serious limitations: ordinary token transport is representable but impractically
large; tensor feedback is practical but adds a recurrent connection outside the
current stock model. Neither is an accepted complete implementation under all
the hard constraints. No custom-Triton-only design is recommended for production.

The useful result here is a measured opportunity, a checked stock masked-write
primitive, and an explicit architecture gate. The present evidence does **not**
establish a practical, compliant one-transition CPU. If stock greedy generation
with the current layer graph is mandatory, retain the current CPU until that gate
is solved. Do not describe tensor feedback as satisfying it by default.

At unchanged transition cost, one instruction per transition would save 38.4% of
transitions on the short continuing walk workload, or 1.624x throughput. A rough
56–64-layer design gives only 1.12–1.28x under uniform layer-cost scaling; a wider
or deeper divider can erase the gain. These are conditional estimates, not GPU
measurements. This step alone is nowhere near 24 fps.

## Constraints and scope

The [handoff](../../docs/doom-24fps-handoff.md), including its Boundaries table
and acceptance criteria, is the contract. A single set of fixed weights must
decode, execute, choose branches, and commit for every supported program.
Program instructions, initial data, and architectural state are recurrent tensor
state. Host code may load the initial image, supply external input, and present
guest-produced pixels. No host or native backend may compute guest results.

Keep the current standalone memory map: 256 KiB immutable code at `0x00040000`,
64 KiB writable RAM at `0x01000000`, and 4 MiB read-only data at `0x02000000`.
“Any RV32 program” here must mean any program within the supported ISA, memory,
alignment, and I/O contract; it cannot mean arbitrary privileged RV32 software
with unrestricted memory. Preserve RV32IM and Zba, current CSR read stubs and
fault behavior. No interrupts, writable executable memory, or functional machine
CSRs have been established. Custom decimal instructions and the merged text
bridge are already outside the expanded-memory proof; expanding their scope is
a separate decision. A one-transition claim for RV32IM must include DIV/REM even
though they are rare in Doom.

The existing code state is predecoded by `ref.rom_word`, including static branch
facts and CSR stubs. Weights remain program-independent. Whether the project
requires raw instruction decoding inside weights rather than this inherited
program-state encoding needs an explicit ruling. This proposal must not silently
strengthen or weaken that interpretation of “decode.”

## Grounding: what executes today

The source of truth is the pinned
[spec.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/spec.py),
[core.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/core.py),
[q35/ir.py](../../.upstream/qwenputer/experimental/riscv/chipformer/q35/ir.py), and
[q35/emit.py](../../.upstream/qwenputer/experimental/riscv/chipformer/q35/emit.py).

One record describes an instruction, but a record can occupy many positions.
The current input token commits the previous position's predicted operation;
the rest of that position computes the next output token. Register keys use the
previous position's decoded destination via convolution. Register values come
from F0/F1/F2 payloads and V history, except links, which are available before
the register layer and can use a local lag. RAM writes similarly combine a
lagged address with store-control tokens and either lagged SW data or serialized
SB/SH merged words. Inside an unfinished record, execution is largely repeated.

The PC is an anchor plus a commit count. J supplies the new anchor; R refreshes
it after 63 commits. The count uses halving/thermometer state rather than an
unbounded exact integer. A result in signed range [-16384,16383] takes F0;
[-2^29,2^29-1] otherwise takes V,F1; other values take V,V,F2. Loads with rd=x0
still perform access checks. SB/SH always emit three V chunks and SS, even for
small data. SW, branches, and jumps already take one token. JAL/JALR links are
computed in weights. Faults take FLT, eight HEX digits, HALT: ten tokens.
DIV/REM emit QS, zero through eight QD digits, then a result record; rd=x0
suppresses quotient work. Custom digit instructions have other iterative records.

The expanded graph built locally has 44 chip layers, 2,679 residual dimensions,
627 key heads and 1,247 logical value heads, 42,014 FFN units, a maximum of 7,876
units in a sublayer, and 16 key heads per sublayer. It uses hidden width 2,816
under `pick_layout`, with 11 ballast dimensions. There are 64 RAM key heads and
512 DROM key heads; code ROM uses 32. Logical recurrent state is 81,723,392 fp32
bytes, excluding histories, scratch, and unused stock slots. Most depth comes
from the DROM's 32 attention sublayers. Division's current named FFNs span
sublayers 10–20 and contain 8,685 units, but compute just one quotient digit at
a time. The multiplier has 9,212 named units across sublayers 10–18.

[fastexec.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/fastexec.py)
evaluates sparse/compact projections, recurrent head updates, and FFNs;
`FreeRun` decodes output bits and builds the next embedding on the GPU.
[tritonexec.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/tritonexec.py)
implements the same scheduled sublayers with sparse gathers and apply kernels.
Its overwrite/gather implementation is justified only for the IR's one-hot keys
and permitted queries, decay and ballast states. It is not a general stock
DeltaNet kernel. Any new beta or key scheme must re-establish that equivalence.

The measured [ROM fusion](../dmc-rom-fusion/README.md) and
[feedback fusion](../dmc-feedback-fusion/README.md) preserve this circuit. Their
race fixes, explicit whole-history resets, and exact state gates matter to any
new benchmark. `reset([row])` does not reliably clear upstream stored histories;
do not reuse the historical warm-up/reload startup as a clean-state proof.

## Measured protocol accounting

`count_protocol.py` replays the software reference and classifies every record
using the original instruction at its pre-record PC. It checks total instructions
and tokens against recorded reports, counts R separately, and saves opcode-level
record-length histograms and token-family counts. It does not use a neural
executor, and is not an independent ISA oracle. No game data is exported.

Both new replays passed. [counts-legacy.json](counts-legacy.json) binds the exact
historical ELF hash; [counts-walk.json](counts-walk.json) binds the ten-frame walk
ELF hash. The table's ALU row includes Zba and excludes upper-immediate and
multiply/divide rows. Counts include rd=x0 behavior. Branch rows include taken
and untaken branches; both have length one in this workload.

| Instruction class | First-frame instructions | First-frame tokens | Tokens/instruction | Walk instructions | Walk tokens |
|---|---:|---:|---:|---:|---:|
| LB/LBU | 300,621 | 300,621 | 1.0000 | 2,191,327 | 2,191,327 |
| LH/LHU | 48,356 | 49,779 | 1.0294 | 330,094 | 339,124 |
| LW | 383,030 | 619,670 | 1.6178 | 3,155,731 | 5,006,357 |
| SB | 224,427 | 897,708 | 4.0000 | 1,408,047 | 5,632,188 |
| SH | 15,834 | 63,336 | 4.0000 | 130,638 | 522,552 |
| SW | 211,199 | 211,199 | 1.0000 | 1,643,360 | 1,643,360 |
| Integer ALU / Zba | 1,797,619 | 3,017,124 | 1.6784 | 13,621,410 | 22,707,483 |
| LUI/AUIPC | 226,411 | 455,240 | 2.0107 | 1,943,621 | 3,905,622 |
| MUL variants | 54,518 | 95,178 | 1.7458 | 535,496 | 938,653 |
| DIV/REM variants | 1,163 | 7,292 | 6.2700 | 11,069 | 70,718 |
| Branches | 443,817 | 443,817 | 1.0000 | 2,749,296 | 2,749,296 |
| JAL/JALR | 178,477 | 178,477 | 1.0000 | 1,117,998 | 1,117,998 |
| ECALL | 72 | 72 | 1.0000 | 252 | 252 |
| PC refresh, no instruction | 0 | 3,420 | — | 0 | 33,711 |
| Total | 3,885,544 | 6,342,933 | 1.6324 | 28,838,339 | 46,858,641 |

First-frame overhead is 2,457,389 transitions: 720,783 from partial stores,
6,129 from division, 3,420 from refresh, and 1,727,057 from all other value
records. Partial stores account for 29.3% of overhead; division only 0.25%.
Removing division iterations has little workload benefit but substantial fixed
network cost, because all instructions traverse the compiled network.

For frames 2–10, the corrected interval report at local
`.cache/dmc-continuing/final-run/report.json` gives **2,772,503.111 instructions
and 4,501,708.333 old tokens per frame**, ratio 1.623698. Thus the target protocol
would generate 2,772,503.111 STEP/event tokens per frame, excluding one-time
initialization and any terminal drain. The first displayed frame would require
3,885,544 transitions for the historical ELF. These are exact workload-count
projections, not evidence of continuing neural gameplay or long-run distribution.

Some older local walk reports store cumulative values in `frames[].instructions`
and `frames[].tokens`; averaging those fields gives a wrong answer. The new
reference replay agrees on global totals. Use the corrected interval report or
differences of cumulative offsets, preserving the final tail convention in
[CONTINUING.md](../dmc-rv32/CONTINUING.md). The corrected report's last interval
ends before `qp_frame_ready`; final termination instructions belong to the tail.

## Stock-layer contract and the feedback obstacle

“Stock” concretely means the construction in
[tools/hf_smoke.py](../../tools/hf_smoke.py),
[rv/hf.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/hf.py), and
[rv/firmware.py](../../.upstream/qwenputer/experimental/riscv/chipformer/rv/firmware.py):
`Qwen3_5ForCausalLM`, compiled fixed weights, a zero PAD prefill, initial snapshot
injection into DeltaNet cache, then ordinary cached greedy generation from GO.
It does not load pretrained language-model weights. The model config has chip
`linear_attention` layers and one appended zero-weight `full_attention` layer
for cache position accounting. Head dimensions are 128, the convolution has four
taps, and this circuit uses 16 key / 32 value heads and intermediate width 9,216.
Configurable depth/hidden width is not evidence of fitting the parameters of a
particular pretrained checkpoint.

| Design operation | Stock feature used | Limitation that must remain visible |
|---|---|---|
| Decode, ALU, compare, mux, mask and full-word merge | Fixed linear projections and SwiGLU, with scaled SiLU acting as exact ReLU on supported inputs | New units must obey gate bounds and sublayer capacity; arbitrary 32-bit scalar arithmetic is inexact |
| Register/RAM storage | DeltaNet fp32 matrix state, one-hot keys, bit columns and ballast | Read ports need copies; beta is scalar per value head, not per column |
| Write enable | `in_proj_b`, sigmoid(256*enable−128) | Exact saturated 0/1; gate is computed before state read |
| Keep/clear/halve | `in_proj_a`, A_log=0, dt_bias=−110 | Supported IR selectors 0/238/110; arbitrary decay is a new numerical contract |
| Prior local values | Depthwise conv of projected q/k/v, taps t..t−3 | Only sees inputs to that layer, never the output of later layers |
| Read output | q/k L2 normalization, gated RMSNorm, z gate, out_proj | Ballast and query cardinality are essential, including bf16 output rounding |
| Output/input transport today | Final code dimensions, tied embedding/lm_head, argmax, embedding lookup | One finite token ID must carry all data that needs to return upstream |
| Proposed tensor packet return | Fixed output-bit projection and routing into next `inputs_embeds` | No equivalent connection exists in the current stock greedy model; requires a mapping decision |

The installed Transformers 5.21.0 source was also inspected directly:
`Qwen3_5GatedDeltaNet.forward` projects hidden input to q/k/v, updates that
layer's convolution and recurrent cache, and emits an output. Its own new output
does not enter its q/k/v cache. FFNs have no recurrent cache. A full-attention
cache likewise stores that layer's input-derived keys/values; the appended zero
layer is not a return connection. `emit_layers` implements exactly that topology.

For example, register reads are at sublayer 9, while an ADD result is computed
after them. A lagged value expression at that register head would have captured
the **pre-head** residual, where that result does not exist. Moving the writer
later allows the write, but the earlier reader then holds a different table.
Duplicating tables needs a synchronization path too. A late PC writer cannot
drive the early instruction fetch on the next position. Hidden width, head
count, or a byte mask does not resolve this.

This is a concrete failure of the obvious construction in this IR, not a proof
that every possible stock Qwen network is incapable of simulating a CPU. A novel
encoding in the stock recurrent rule would need an explicit construction and
checks. No such construction was established here. In particular, a single-head
“gated delta CPU” cannot be assumed: its q/k/v/gates are functions of the input
before it reads its state, and the current IR restricts it to overwrite tables.

## Alternatives and synthesis decision

Two distinct feedback architectures were examined inline, as requested. The
recommended primitive is shared, but their stock compliance is different.

### A. One ordinary token contains a full commit payload

Keep normal stock `generate` and let fixed neural output code choose one token
representing the full result, next PC, and control. Fixed embedding features
decode the packet into bits. Early heads commit it, using locally lagged decode
for the destination and address where available. This preserves the existing
neural/host boundary and provides the missing return path. A new PC token field
on every step eliminates the refresh record.

A straightforward encoding has at least 16 next-PC word bits and 32 data bits,
plus tags; routing some mutually exclusive cases separately can reduce this.
Even just 2^32 full-result embeddings at width 2,816 take **24.2 TB** in bf16,
before other weights; the straightforward 48-bit product would take about
1.59 EB. These are construction sizes, not a universal information lower bound
on all possible encodings. Arithmetic/sparse embedding and an analytical decoder
in the custom executor do not make the stock dense embedding/lm_head affordable.
Factoring a token into several separately embedded components changes the stock
input/output interface or returns to several tokens.

Status: mathematically expressible using stock layer kinds, unusable at the
project's resource scale. It hides CPU internals behind a token interface, but
exposes an infeasible vocabulary to the stock runtime. A tiny restricted-value
demo would not establish a universal 32-bit design.

### B. Fixed neural tensor feedback carries a full-word packet

Have the circuit emit a STEP/event token for observers and, independently, a
binary packet tensor. A fixed tensor projection/router supplies those neural
bits as next-position embedding features. All semantic decisions, masks and
values are generated in the circuit. The routing is a fixed neural connection,
not a host switch over opcodes or guest addresses. The token can be constant for
ordinary instructions; optional BYTE, FAULT, HALT tokens expose I/O boundaries.

This has a practical small interface and avoids the vocabulary explosion. Each
body evaluation can consist entirely of unmodified stock HF layers, but the
closed loop is **not** the current stock `Qwen3_5ForCausalLM.generate` graph.
The packet is an additional neural recurrent register outside the layer caches.
Checking a supplied packet sequence through stock layers is insufficient:
bounded autonomous tensor feedback must also be checked and described accurately.

Status: preferred research architecture **only if** the orchestrator accepts
that extra recurrent connection as part of the stock-model mapping. It is not
claimed compliant under the existing hard constraint. Otherwise reject it.
Calling the router “just feedback fusion” does not establish acceptance: the
current fusion merely implements the existing finite token embedding lookup.

Synthesis: preserve word RAM and use a merged-word packet to minimize head and
state growth; use byte-lane heads as a verified alternative if measurements
justify the cost. Gate all CPU work on the return-path decision. Avoid a design
with independent host-owned commit, register, or RAM implementations. Keep one
packet schema from which compiler, observer, and extraction metadata are derived.

## Conditional execution and commit design

This section defines the circuit contract to implement after transport is solved;
it does not waive the stock gate.

Caller usage should be `load_image(image, snapshot)`, then
`transition(external_input) -> event, architectural_view`. The transition
interface conceals opcode decode, PC update, register forwarding, RAM banking,
and packet transport. External input is the only caller-provided variable data
during execution. State extraction for verification is observational.

Conceptual bit-vector types (no packed 32-bit fp32 scalar):

```
CommitPacket = {
  valid: Bit, next_pc: Bits[32],
  rd_enable: Bit, rd: Bits[5], rd_value: Bits[32],
  store_enable: Bit, word_address: Bits[14], byte_mask: Bits[4],
  merged_store_word: Bits[32],
  terminal: Bit, fault_cause: Bits[3], fault_pc: Bits[32],
  io_enable: Bit, io_payload: Bits[9], exit_code: Bits[32]
}
NeuralState = { layer_caches, pending_packet }
compile_cpu(map, isa_contract, target) -> FixedWeights, StateSchema
evaluate_transition(weights, state, input_bits) -> state, event
architectural_view(state) -> PC, x[32], RAM, fault, I/O
```

Fields are mutually gated; the exact layout can be reduced after correctness.
The pending packet is neural state, never a host-computed delta. Host extraction
may decode bits but must not supply extracted results back to execution. The
implementation modules should own (1) ISA bit circuits and packet generation,
(2) the recurrent-state schema and snapshot/extractor, (3) stock emission and
transport, and (4) independent observation/testing. Avoid duplicating semantic
opcode tables in executors or transport adapters.

At entry t, apply the previous pending packet to the state tables before reads.
Fetch the current PC, read operands, execute, validate, and form packet t for
the next instruction boundary. Next PC selection is neural: PC+4, taken target,
or JALR target with its low-bit and range checks. Links and all 32 value bits
are selected before packet emission. x0 never has an enabled write.

There is an important boundary convention: the physical early tables have
materialized the previous packet; the newest architectural state is the base
tables plus the single pending packet. That overlay is part of recurrent neural
state and must be included in checkpoints and comparisons. It cannot be omitted
when reporting the completed instruction. If acceptance requires every effect
physically in the main tables at the end of the same transition, this design
does not meet it. Late physical mirrors would add heads and still require the
return connection to synchronize early reads. An extra unreported drain step
cannot be charged as zero work. On HALT the observer must read the pending final
state, or explicitly report a drain transition as outside instruction execution.

### Full-word stores and load ordering

Recommended RAM encoding retains 64 key / 128 value heads: each selected row
stores a word as 32 separate A_BIT columns plus two A_REF ballast columns.
Remove the old disjoint SW/SS payload columns; one full-word representation
handles all widths. The current instruction's RAM read provides the old word.
FFN Boolean masks compute each merged bit:

`merged[i] = selected_byte[i//8] ? shifted_store_data[i] : old_word[i]`.

Neural address/width logic generates the mask, including aligned SH lanes. SW
selects all bytes. The packet returns all merged bits. On the next transition,
the memory head uses that packet's address and bits with beta=1 only for its
selected bank/value head; other rows and banks have beta=0. The state rule is
`S_new = S_old + beta*k*(v - k^T*S_old)` for one-hot normalized k and keep decay.
Thus an enabled write replaces one whole row, including ballast. A read follows
the write in that head, so a following LB/LH/LW observes it. The exact HF
normalization path must still pass the numerical gate; the ideal overwrite
equation alone is not proof of fp32/bf16 implementation exactness.

No instruction here requires two architectural stores. The next instruction's
effective address may depend on the just-written register: commit the previous
rd before operand reads. For a store followed by a store, the second word merge
must read the first update. For a fault, disable rd/store/I/O packet bits together
and preserve the faulting PC; do not let an early speculative store escape.

### Alternative masked byte-lane writes

A value head's beta gates a **whole value vector**; it is not a four-bit byte
mask. Splitting a word across four value heads gives four independent betas.
With the present 2:1 value/key ratio, that needs two key heads per word slot
group. Keeping the original two word slots per RAM bank gives four key / eight
value heads per old bank: 256 key / 512 value heads total, versus 64/128 today.
It adds 192 key heads, 384 value heads, 25,165,824 logical fp32 state bytes and
at least 12 attention layers at 16 key heads/layer. It avoids read-modify-write
for stores, but not the return-path problem. Masking v alone in an ordinary
word head would zero the other bytes and is incorrect.

The [masked_write_poc.py](masked_write_poc.py) check used exactly the pinned IR
and emitter. Two key heads / four value heads held one word, sharing row keys
within each pair and using separate byte betas. Fourteen supplied packets covered
SW, SB, aligned SH, disabled writes, zero overwrites and high-bit values.
IR exactness and HF-kernel rounding checks passed; stock bf16 cached decode and
chunked prefill each matched all IR residual dimensions at all 14 positions.
[masked-write-result.json](masked-write-result.json) records versions and words.
This proves only the masked-write primitive with supplied packets. It does not
prove neural address generation, universal data transport, final cache equality,
or an autonomous instruction loop. No system package was installed.

### Division and exactness

Unroll eight radix-16 digit stages in fixed weights. Each stage computes quotient
digit comparisons from A, B and the previous stage's partial quotient, with
signed magnitude conversion and final signed quotient/remainder selection.
Use bit/nibble adders and the existing bounded product/threshold gadgets; no
floating-point `/`, scalar 32-bit multiply, host quotient, or variable transition
count. DIV by zero returns all-one quotient; REM by zero returns the dividend;
INT_MIN/−1 returns INT_MIN and remainder zero. Signed division truncates toward
zero. rd=x0 still completes in one transition without a register write.

Replicating the current digit logic eight times suggests roughly 60–70k division
FFN units before simplification, compared with 8,685 currently. It is not enough
to multiply the existing total layer count by eight: the divider can execute in
parallel with DROM layers, but successive quotient digits depend on each other.
Allow approximately 5–7 FFN stages per digit plus operand/sign preparation and
selection, giving a provisional 56–80-layer full CPU envelope with word RAM.
The exact schedule has not been built. High fan-in and unit capacity can push
it beyond this range. A 32-stage binary restoring divider is simpler to specify
but likely deeper; it is a fallback to compare by emitted graph, not an assumed
improvement. Skipping division because it is rare would change the ISA claim.

Every data bit remains 0/1, stored as 0/A_BIT; addresses and PCs are bit vectors.
Bound nibble sums, carry terms, negative gates, residual magnitude and ballast
ratios. Use `ir.evaluate(check=True, kernel=True)` wherever its supported
numerical contract applies. Check stock output and recurrent tensors separately.
Disable TF32 in compact fp32 checks and retain exact state-bit comparisons;
arbitrary tolerances must not turn a wrong architectural bit into a pass.

## Cost model and limits

Use the accepted reference-seeded rendering median **228.155 microseconds per
old token** from feedback fusion (boot: 228.291). It includes normal replay,
feedback, trace copies, synchronization and CPU trace conversion, but excludes
construction, reset, verification, pixel extraction and display. No new GPU
timing was performed here. The old continuing instruction rate inferred from
this window and the frame distribution is about 2,699 instructions/s.

First sensitivity model: `C(L) = 228.155 * L/44` microseconds per transition;
new instructions/s = `1e6/C`; projected frame generation =
`2,772,503.111*C/1e6` seconds. Apply it only to schedules with comparable width,
head distribution and execution strategy. Break-even is about 71.4 such layers.

| Chip layers | Assumed microseconds/instruction | Instructions/s | Gain over old steady-state estimate | Seconds/frame |
|---:|---:|---:|---:|---:|
| 44, optimistic floor | 228.2 | 4,383 | 1.624x | 632.6 |
| 56 | 290.4 | 3,444 | 1.276x | 805.1 |
| 64 | 331.9 | 3,013 | 1.116x | 920.1 |
| 80 | 414.8 | 2,411 | 0.893x | 1,150.1 |
| 96, stress scenario | 497.8 | 2,009 | 0.744x | 1,380.1 |

The old extrapolated frame generation time is 1,027.1 seconds at this measured
window rate. Even the optimistic new rate needs about 15,181x more throughput
to reach 41.7 ms/frame. Actual whole-frame and interactive timings remain
unmeasured. No stock-HF performance estimate follows from these sparse-executor
numbers. [cost-model.html](cost-model.html) visualizes only this sensitivity.

A better second model must use category work rather than depth alone. The
separate stacked profiler reports, in rendering microseconds/token, FFNs 55.779,
DROM heads 53.190, DROM apply 20.464, RAM 41.333, other heads 15.201, code ROM
10.013, other residual/history apply 23.750, and feedback about 5.782, plus trace
and stream gaps. These durations are instrumented and **must not be added to
228.155**. Normalize changed category ratios by the old profiler span, then
apply that ratio to wall time, leaving unmodeled overhead explicit.

For example, doubling total FFN work alone adds roughly 55.8 microseconds of
profiled work, or about 23% of the 239.2-microsecond stacked profiler span.
With all other categories unchanged this suggests ~281 microseconds/transition,
~3,556 instructions/s and ~1.32x gain. A byte-head design can quadruple logical
RAM value-head work, so even a rough extra 3*41.3 microseconds would exceed the
protocol saving before division. Neither example predicts an actual kernel
schedule or makes profiler and wall durations additive. They show why retaining
word RAM and measuring divider FFNs matter more than an attractive layer count.

Head/state budget for the word design starts around today's 627/1,247 heads
and 81.7 MB logical state, removing obsolete counter/quotient/protocol heads
and adding only small status/packet storage after transport is chosen. Budget
under 650 key / 1,300 logical value heads provisionally, not as a compiled
measurement. Byte RAM adds 192/384 heads to that baseline, approaching 819/1,631
before removals. Stock layers allocate all configured value slots even if
unused: at 32 value heads and fp32 128x128 state, about 2 MiB per chip layer
for batch one, plus conv/KV caches and tensors.

The 56–80-layer word envelope is a planning range, not a “44-layer optimized
CPU.” Unrolled divider intermediates can add thousands of residual dimensions;
hidden width may grow from 2,816 toward 8–10k without verified slot reuse.
The existing dense 44-layer estimate is already about 6.21 billion parameters /
12.42 GB bf16 weights, excluding runtime allocations. Wider/deeper variants
could exceed 16 GB by a large margin. Report actual dims, liveness, selected
layout, parameter bytes and peak memory before any full-map stock allocation.
Bounded stock checks should start with small memory maps using the **same**
generic circuit construction, with no program-specialized weights. Those checks
do not prove that the full Doom map fits a 5070 Ti stock-model run.

## Observable boundaries and risks

- **Commit definition:** compare the full architectural view, including pending
  packet, after each instruction. Raw early-table bytes alone are one commit
  behind in the conditional packet design. Checkpoint/reset must preserve or
  initialize the packet as well as every S and convolution cache.
- **Load after store:** the prior write must occur before the current query in
  the same RAM head, across all banks and word-slot selectors. A late mirror
  without an early update does not satisfy this. Test store-to-store merging too.
- **Code:** preserve immutable predecoded ROM and rejection of stores to it.
  Self-modifying code is unsupported today. Supporting it requires writable
  instruction memory, raw decode or coherent neural predecode updates, and
  instruction/data ordering; it is a separate design decision, not FENCE alone.
- **Faults and I/O:** one faulting instruction emits a single fault event and
  stores fault PC/cause in neural state; no ten-token fault serialization is
  needed. Halted state must be absorbing. Preserve BYTE's nine-bit payload,
  special strip marker 256, ECALL a7=93 exit code and exact event order. Precise
  faults suppress every architectural side effect other than fault reporting.
- **CSR scope:** retain the current supported pure-read constants, including
  MISA, and illegal faults for unsupported accesses. Do not silently add a host
  instruction counter as CSR state. True CSR writes/interrupts need neural
  state and precise priority rules.
- **Input:** continuing fixtures use recorded schedules in guest data, not live
  input transport. An eventual host key latch may supply raw external key bits
  at a specified instruction boundary; guest instructions still generate events
  and simulate. Neural logic must sample/acknowledge it. Specify polling and
  chunk bounds before batching. At ~3,000 instructions/s even 128 instructions
  per unchecked chunk add about 43 ms; constant feedback does not fix guest
  polling or presentation latency. Preserve 35 simulation tics / 24 displays.
- **Exactness:** fp32 has only 24 significant bits and bf16 only eight; a full
  RV32 word must never be one ordinary residual scalar. Normalize/re-quantize
  Boolean decisions in circuit where needed and test long chains, not just
  final output. Existing source-level exactness assertions do not by themselves
  certify all stock recurrent kernels.

## Verifiable implementation milestones

All initial work stays in this experiment. No upstream modifications or changes
to historical fixtures are needed to start.

| Milestone | Work | Acceptance criterion |
|---|---|---|
| M0: protocol and primitive evidence | Reproduce counts and the byte-head check | Exact legacy and walk totals; all class sums reconcile; 14 stock decode/prefill residual checks pass. Completed here, with the narrow scopes above. |
| M1: autonomous stock return path | Choose transport; build a tiny general CPU loop with ADD/XOR, dependent addresses, load/store, taken branch and JALR | One evaluated token position per retired instruction for at least 1,024 instructions, two programs and changed data using identical weights; no supplied expected results/semantic tokens; compare PC/registers/RAM each step. Unmodified stock modules must execute the loop. For B, obtain explicit mapping approval first and label the wrapper. Stop if this fails. |
| M2: state/commit contract | Formalize packet validity, fault suppression, x0, snapshot, architectural overlay and absorbing halt | Local independent scoreboard agrees at every boundary, including last instruction before halt/fault; checkpoint/resume identical at every possible packet phase; full and row reset controls catch nonzero history. No hidden drain. |
| M3: generic word memory | Add existing map parameterization, word merge, range/alignment checks and bank selectors | All memory-map cases reused, all RAM bytes checked; SB at every lane, SH at both aligned lanes, alternating partial stores, same/different-bank dependencies, final/first rows and read-only/old-alias faults. Signed/unsigned loads after each write match an independent reference. Same circuit weights across programs for a fixed map. |
| M4: complete ISA datapath | Add all ALU, MUL high variants, Zba and eight-stage division | Directed boundary cases plus seeded random programs with independent ISA execution; DIV/REM all sign combinations, zero divisor, INT_MIN/−1, quotient digit boundaries, rd=x0 and dependency chains. Exactly one transition per instruction, including faults. Export graph schedule/width/head and exactness report. |
| M5: stock/circuit/backend triangulation | Compare ideal IR, compact CPU, stock cached decode and stock chunk mode; then GPU | Inspect logical architectural state and recurrent words, not only output tokens. Bounded stock checks exercise every operation and multi-instruction chain. Compact/Triton comparisons include histories/packet and one-bit recorder controls. Weight/code hashes bind results. No altered native opcode implementation in executor. |
| M6: Doom regression and bounded performance | Reuse legacy ELF, corrected continuing idle/walk manifests, original strips and frame hashes | Legacy PC/register/store/event stream matches independently at every instruction, 25 strips, 76,800 pixels and all final 64 KiB RAM bytes match; continuing frame hashes/gametics/35:24 progress match. Reference-seeded rendering is labelled separately from uninterrupted neural execution. Old token identity is deliberately not required. |
| M7: full timing scope | Only after exact correctness gates, repeat alternating bounded and full-frame runs | Publish instruction/s, transitions/frame, layer/category costs, medians/ranges, cold costs separately, true frame and input/presentation latency, peak memory, fresh-zero startup and resume coverage. Reuse accepted ROM/feedback race fixes only after new-geometry equivalence checks. No 24-fps claim from prefix timing. |

The independent ISA gate must not compare the new CPU only with itself. The
existing `Machine.record()` is useful for trace counts and fixtures but shares
decode/arithmetic helpers with the compiler. Use a separately implemented RV32
reference or an established external RV32 emulator for architectural arithmetic,
and a small independent memory-map/fault/I/O adapter for this platform contract.
The upstream flat `Rv32` reference is useful corroboration but shares helpers
and has different fault/CSR semantics; it is not sufficient independence alone.
Deliberately corrupt one bit of result, PC, store mask, address, fault code and
input latch to prove each observer detects the earliest affected boundary.

Reuse [verification.md](../../docs/verification.md) without inflating its scope:
the existing stock hello/prefix/frame proofs concern earlier circuits, and the
DMC full-frame proof has the documented history-reset qualification. Small-map
stock proofs and a large-map compact Doom proof are separate results. Report
that distinction again after this redesign.

## Decisions needed from the orchestrator/user

1. **Stock recurrence mapping:** must the CPU run under ordinary stock
   `Qwen3_5ForCausalLM.generate` with token-only feedback, or is a fixed neural
   output-to-`inputs_embeds` recurrent connection acceptable? B changes the
   current model graph and recurrent-state boundary. No approval is presumed.
   If token-only is mandatory, M1 needs a new construction; the STEP proposal
   is not ready to implement.
2. **Instruction boundary:** is a neural pending-commit packet plus tables an
   acceptable architectural-state representation? Requiring all main tables
   physically updated at transition end rules out the stated boundary contract.
3. **Decode claim and ISA scope:** explicitly retain or replace the inherited
   predecoded program-state representation, current CSR stubs, immutable code,
   and the standalone RV32IM+Zba scope. Raw decode, interrupts, self-modification,
   decimal extensions or text-bridge compatibility each need further design.
4. **Stock geometry/resource envelope:** confirm whether increased chip depth,
   width and bounded small-map stock checks satisfy the intended evidence, or
   whether the entire Doom-map stock model must fit the 16 GB GPU. Head shape
   128, 16/32 grouping and conv length four are retained in this proposal. Any
   change to them is a separate mapping decision.
5. **Memory map:** no address/capacity change is proposed. Any move, capacity
   reduction, writable code, or new input MMIO region must be approved explicitly.
   Byte-lane state encoding is conditional and would need its own equivalence
   proof even without an address-map change.

No decisions above were resolved by this design session. No full CPU was
implemented, GPU run launched, upstream edited, commit made, or system tool
installed. Next concrete step is M1, after the stock return-path ruling.

## Reproduction and evidence provenance

Run from the `qwenputer-games` root with its existing virtual environment. Scratch
packages are ignored and must be fresh. They are isolated copies prepared by the
existing memory-map experiment, never modifications to pinned upstream.

```sh
.venv/bin/python experiments/one-instr-design/count_protocol.py \
  --elf .cache/dmc-continuing/legacy-build/doom.elf \
  --expected-report experiments/dmc-rv32/report.json \
  --scratch experiments/one-instr-design/.scratch-legacy-new \
  --out experiments/one-instr-design/counts-legacy-new.json
.venv/bin/python experiments/one-instr-design/count_protocol.py \
  --elf .cache/dmc-continuing/walk-build/doom.elf \
  --expected-report .cache/dmc-continuing/final-run/report.json \
  --scratch experiments/one-instr-design/.scratch-walk-new \
  --out experiments/one-instr-design/counts-walk-new.json
.venv/bin/python experiments/one-instr-design/masked_write_poc.py
```

The recorded walk accounting used the older `walk-run/report.json` for global
total checks; it has the same ELF/global totals as the corrected `final-run`.
Only the corrected interval report was used for per-frame arithmetic. POC
weights are compiled from the clean pinned IR, with no pretrained model.
Evidence is local CPU work; all GPU costs are explicitly imported from the
existing bounded-window experiments. Design alternatives were compared inline
under the user's no-delegation instruction. Implementation reconciliation is
limited to M0: the masked byte-head primitive works; autonomous feedback and
full instruction commit remain unimplemented.
