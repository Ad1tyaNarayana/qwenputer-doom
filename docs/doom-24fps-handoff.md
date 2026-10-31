# Handoff for Doom at 24 fps on a neural RISC-V CPU

Date: 2026-10-29.

## Goal and agreed constraint

Target interactive Doom at 24 displayed frames per second on an RTX 5070 Ti while preserving a RISC-V CPU implemented in neural weights and recurrent state.

The project constraint is:

> One fixed neural CPU runs different guest programs by changing program and data state. Its weights define instruction execution. The GPU backend only evaluates that network.

Changing the CPU design and compiling new fixed weights is allowed. Specializing those weights for each guest program is outside this strict target. The implementation can use the GPU for inference, host input, and presentation of guest-rendered pixels. The host must not calculate guest instruction results, branch decisions, or Doom geometry.

RISC-V semantic correctness and neural execution are separate requirements. Matching registers or pixels proves neither requirement by itself.

The 24 fps target is unproven. This handoff records an investigation and proposed experiments. No proposed architecture change or new GPU benchmark was implemented during that investigation.

## Existing evidence and its limits

The verified Doom run uses the DMC strip-rendering port through the compact Triton circuit executor. It uses fp32 circuit arithmetic, self-generated trace tokens, and neural recurrent RAM. It is separate from execution through stock Hugging Face model layers. The pretrained Qwen language model was not loaded for this Doom proof.

| Existing first-frame evidence | Value |
| --- | ---: |
| GPU | RTX 5070 Ti |
| RISC-V instructions | 3,885,544 |
| Generated trace tokens | 6,342,933 |
| Token mismatches | 0 |
| Retained-path generation time | 3,044.5744 seconds |
| Output | 25 strips covering 320 by 240 pixels |
| Circuit layers | 44 |
| Writable guest RAM | 64 KiB |
| Read-only guest data window | 4 MiB |
| Data-ROM key heads | 512 |

The first frame includes startup and level initialization. The run exits after that frame. Continuing gameplay, input responsiveness, and steady-state frame cost remain unverified.

The generation time sums successful retained chunks across checkpoint recovery. It includes token decoding, self-fed embeddings, trace copies, and CPU transfer. It excludes setup, capture, comparisons, file I/O, checkpoint handling, pixel extraction, discarded work, and recovery downtime. It is not an uninterrupted end-to-end timing.

At 24 fps, the displayed-frame budget is 41.7 milliseconds. Dividing the historical first-frame time by that budget gives approximately 73,070. This is a first-frame comparison, not an established steady-state speedup requirement.

The first-frame trace averages 1.6324 tokens per instruction. Collapsing that trace to one transition per instruction reduces transition count by about 1.63 times. That calculation assumes equal transition cost and is not a speedup measurement.

The evidence lives in these files:

- [GPU execution and reproduction](../experiments/dmc-cuda/README.md).
- [GPU execution report](../experiments/dmc-cuda/evidence/full-report.json).
- [Checkpoint recovery and timing scope](../experiments/dmc-cuda/evidence/recovery.json).
- [Reference instruction counts](../experiments/dmc-rv32/report.json).
- [Expanded circuit geometry](../experiments/memory-map/evidence/low-memory/graph.json).
- [Current verification scope](verification.md).

## Boundaries for proposed optimizations

| Proposal | Status under the strict neural CPU constraint | Required evidence |
| --- | --- | --- |
| Faster Triton kernels for the same circuit | Allowed | The backend evaluates the same circuit, including recurrent updates. |
| Layer fusion and persistent execution | Allowed | Fusion preserves dependencies, lag history, memory updates, and numerical behavior. |
| Multiple existing transitions per CUDA graph | Allowed | Every transition uses its own generated result as the next input. |
| One RISC-V instruction per neural transition | Allowed as a CPU redesign | The fixed network implements decode, execution, and commit. |
| Multiple instructions per neural transition | Allowed as a CPU redesign | The fixed network implements dependencies, forwarding, memory ordering, and precise observable behavior. |
| Generic neural SIMD or vector instructions | Allowed as an ISA extension | The neural circuit implements each operation and its memory semantics. |
| Doom changes that use the generic ISA | Allowed | Guest code still performs simulation and rendering through the neural CPU. |
| Skipping inactive neural memory heads | Conditional | Prove the skipped contributions and state updates are inert for all supported states where the optimization applies. |
| Different neural memory encoding | Conditional CPU redesign | Memory remains representable in neural state and accessed through the fixed neural circuit. |
| Guest basic blocks compiled into new neural circuits | Deferred outside the strict target | This supports a separate claim of program-specialized neural execution. |
| Guest basic blocks compiled into native CUDA instruction operations | Excluded | Semantic correctness would establish GPU emulation, not a CPU defined by neural weights. |
| Native RISC-V interpreter inside a persistent kernel | Excluded | Persistence does not make an interpreter neural. |
| Host renderer or host-computed guest state | Excluded | The host may present guest-rendered pixels and provide external input. |

All software inference backends use native GPU instructions. The distinction is whether those instructions evaluate the neural circuit or independently implement guest instruction semantics.

A specialized array access that replaces a neural memory calculation needs a circuit-equivalence argument. Ordinary GPU storage backing a recurrent tensor is expected. An independently implemented guest memory system is a different design.

## Findings that determine experiment order

### Memory capacity adds circuit depth

The original CPU has 11 circuit layers. The expanded Doom circuit has 44 layers and 512 data-ROM key heads. The current executor traverses scheduled sublayers on each token.

This makes memory organization a candidate bottleneck. The source structure alone does not establish its share of runtime. Adding execution lanes before profiling memory could duplicate an expensive operation.

Compact-executor scheduling improvements and changes that fit the stock model layer layout need separate results. Equivalent circuit output does not establish stock-model execution performance.

### Execution still crosses many kernel boundaries

`FreeRun` in upstream `experimental/riscv/chipformer/rv/fastexec.py` captures one transition as a CUDA graph. Its `run` method replays that graph and copies the output token for each transition.

`Executor2` in upstream `experimental/riscv/chipformer/rv/tritonexec.py` evaluates scheduled sublayers through separate Triton kernels and apply kernels. These files are in the pinned checkout under `.upstream/qwenputer/` after setup.

Multi-transition graph capture and persistent circuit evaluation are distinct experiments. Neither may substitute a native RISC-V interpreter.

### The token protocol serializes internal state changes

Upstream `experimental/riscv/chipformer/rv/spec.py` defines multi-token value records. A 32-bit value can take three tokens. A byte or halfword store takes four tokens.

A redesigned CPU could commit values, store masks, and the PC directly through neural state. Optional trace output could observe those updates. Hiding trace output alone does not remove the existing internal transitions.

### Wider execution needs memory bandwidth and correct dependencies

Multiple instructions in one transition still require sequential circuit depth when their results depend on one another. Replicating the circuit can increase work without reducing elapsed time.

Vector arithmetic needs corresponding neural memory access capacity. A wide arithmetic unit with serial lane loads and stores may offer little benefit. The current memory-saving strip renderer also warrants profiling before choosing vector operations or increasing guest RAM.

## Proposed execution sequence

1. Establish a continuing-frame reference workload. Separate startup, level initialization, simulation, and rendering. Add a deterministic input schedule. Preserve Doom's 35 simulation tics per second while targeting 24 displayed frames per second.
2. Capture a representative profile on the 5070 Ti. Separate host submission, circuit kernels, memory heads, token feedback, trace handling, and pixel extraction. Record the software versions and measurement scope.
3. Optimize the unchanged circuit. Compare the existing executor with multi-transition graphs, fused token feedback, and bounded persistent execution. Require identical traces and recurrent state before accepting a result.
4. Investigate neural memory scheduling and encoding. Compare selected-bank evaluation and independent-head scheduling with the existing circuit. Treat a new encoding as a separately verified CPU design.
5. Prototype one instruction per neural transition. Keep full internal values in neural state. Verify architectural state at instruction boundaries against an independent reference.
6. Prototype multiple-instruction neural execution and generic SIMD separately. Choose hot instruction sequences and loops from the profile. Measure useful work per second rather than transitions per second alone.
7. Adapt Doom only after a generic CPU improvement demonstrates value. Keep the existing first-frame fixture as a regression case and add continuing gameplay fixtures.

Guest-block JIT compilation is not part of this sequence under the agreed strict constraint. Any later exploration needs an explicit change to the project claim.

The next concrete deliverable is a continuing-frame workload and a repeatable GPU profile. It is not a promised 24 fps implementation.

## Acceptance criteria

Every candidate needs both correctness evidence and neural-execution evidence.

For unchanged-circuit optimizations, compare generated traces, recurrent state, and guest output. Include supported memory operations, lag histories, and checkpoint continuation. Add targeted cases for any new scheduling assumptions.

For redesigned CPUs, compare the PC, registers, memory effects, faults, and I/O against an independent RISC-V reference. Include partial stores, signed operations, division edge cases, dependency chains, and branch boundaries. Trace formats may change, so old token identity is not required for a redesigned CPU.

Document how each redesigned operation maps to neural weights and recurrent updates. A test that compares only the native backend with its own implementation does not establish neural execution. Use circuit-level comparison and bounded stock-model checks to support the claimed mapping. Report their scope separately from full Doom execution.

Input sampling, I/O, faults, and any supported interrupts are observable boundaries. Batching must preserve those boundaries and allow bounded input latency. A matching final register file is insufficient.

Report repeated steady-state frame timings, timing spread, simulation progress, output resolution, and input latency. Separate cold startup and compilation from gameplay. Include host input and presentation costs in the interactive result. Replaying recorded frames or slowing simulation does not satisfy 24 fps gameplay.

## Research references

These sources motivate experiments. Their reported performance does not predict Qwenputer performance.

- [NVIDIA CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) describes graph execution and submission overhead.
- [Triton persistent kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html) describes device-resident work scheduling.
- [MPK compiler and runtime paper](https://arxiv.org/abs/2512.22219) investigates persistent execution of tensor programs.
- [QEMU translator internals](https://www.qemu.org/docs/master/devel/tcg.html) describes translation blocks, state assumptions, and observable exits. It is a semantic reference for the deferred JIT idea.
- [RISC-V vector specification](https://docs.riscv.org/reference/isa/v20260120/unpriv/v-st-ext.html) defines the standard vector extension. A custom subset must not be presented as full RVV support.

The investigation applied pstack's Explain the Number principle by separating historical first-frame timing from the unmeasured steady-state target. That distinction determines why measurement precedes an architecture overhaul.
