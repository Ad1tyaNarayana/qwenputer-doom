# Continuing DMC Doom reference workload

This workload runs the existing DMC E1M1 engine for a build-time number of
displayed frames in the software RV32 reference machine. The engine sources,
linker memory regions, CPU implementation, trace token format, and strip pump
are unchanged. Guest instructions perform all simulation and rendering.
The observer presents the guest's indexed strips using its RGB565 palette.

## Run a workload

Use the pinned clean source, Zig toolchain, generated tables, and private NOR
image described in [README.md](README.md). From the repository root, build a
10-frame workload with the supplied walk-forward schedule:

```sh
.venv/bin/python experiments/dmc-rv32/build.py \
  --zig .venv/lib/python3.13/site-packages/ziglang/zig \
  --source .upstream/vape-doom/PixoAspire_DOOM_Port \
  --gen /path/to/private/data/gen --nor /path/to/private/data/nor.img \
  --libc experiments/doom-baremetal --out .cache/dmc-walk-build \
  --frames 10 --input experiments/dmc-rv32/walk-forward.json

.venv/bin/python experiments/dmc-rv32/run_reference.py \
  --prepare experiments/memory-map/prepare.py --upstream .upstream/qwenputer \
  --elf .cache/dmc-walk-build/doom.elf --scratch .cache/dmc-walk-machine \
  --out .cache/dmc-walk-run --frames 10 --max-instructions 40000000
```

Choose a fresh `--scratch` directory for each run. `--frames` accepts 1 through
1,000,000 at build time. The runner's `--frames` is an expected-result gate;
it must match the build manifest. Omitting `--input` gives idle input.
A 100-frame build requires a larger instruction budget, for example
`--max-instructions 400000000`. A budget exhaustion writes a failed report
and exits unsuccessfully. Frame count support is not a claim that all such
run lengths have been verified.

`workload.json` beside the ELF records the frame count, rates, schedule, and
ELF SHA-256. Keep it with the executable. The runner rejects a different ELF
or frame count. Game data and executables remain private and ignored.

To reproduce the historical first-frame fixture, omit both `--frames` and
`--input` from the build command. Run that ELF with `--check-first-frame`.
That option requires exactly 3,885,544 instructions, 6,342,933 trace tokens,
and RGB565 SHA-256
`6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2`.
Opt-in `--frames 1` uses the new glue and has different instruction counts.

## Simulation time and external input

The port retains `singletics=true`. The linker wraps calls to `TryRunTics`
without editing the engine. The boot call runs one tic. Displayed frame `f`,
numbered from 1, then runs
`floor(35*f/24) - floor(35*(f-1)/24)` additional tics. An integer remainder
implements this schedule. Every 24 displayed frames advance exactly 35 tics.
The first ten displayed frames finish at gametics
`2, 3, 5, 6, 8, 9, 11, 12, 14, 15`, including the boot tic.
Each engine simulation step retains Doom's 1/35-second semantics.
No simulation tic is interpolated, shortened, or skipped.

In continuing mode `I_StartTic` advances the deterministic clock once per
built ticcmd. `I_GetTime` reads that clock, and `I_GetTimeMS` returns its
integer millisecond conversion. Neither read advances time. This is a
virtual 35 Hz simulation and 24 Hz display schedule. It does not pace the
software reference to wall time or establish interactive frame rates.
The legacy clock and exit path remain behind the default build mode.

The input file is a JSON list of strictly increasing nonnegative tic states:

```json
[
  {"tic": 2, "keys": ["forward"]},
  {"tic": 12, "keys": []}
]
```

Each entry replaces the full held-key state before building that tic's
command. An empty list releases all keys. The initial state is idle; the
last state persists. Tic 0 is the boot simulation tic. Supported names are
`forward`, `back`, `left`, `right`, `fire`, `use`, `run`, `strafe_left`, and
`strafe_right`. Input tic indices are limited to 2,000,000.

The builder puts only this external schedule in guest read-only data.
`I_StartTic` compares held states and calls `D_PostEvent` with normal keydown
and keyup events. The unchanged engine calls `D_ProcessEvents`,
`G_BuildTiccmd`, and `G_Ticker`. The host supplies no ticcmds, positions,
geometry, simulation results, or rendered pixels. This is recorded input,
not a live keyboard transport or a test of the meter's button resolver.

## Counts and frame boundaries

The observer recognizes entry PCs from the exact ELF symbols and the guest's
return address register. It observes each return before executing the next
instruction. `TryRunTics` supplies simulation boundaries, `P_SetupLevel`
supplies level-initialization boundaries, and `D_Display` supplies rendering
boundaries. Nested level initialization is charged only to that phase.
Startup covers execution before `D_DoomLoop` outside those functions.
Other loop, wrapper, sound-update, and termination instructions are overhead.
This function-boundary method includes callees and does not sample PCs.
It depends on the current non-LTO build and these retained symbols.

`qp_frame_ready` is a guest function with an observable entry PC after
`R_PumpIdle` and frame-memory metadata updates. The observer snapshots a
continuing frame immediately before that function executes. The marker uses
ordinary guest instructions and introduces no MMIO address or token family.
Each frame's interval begins at the previous snapshot, so a prior display's
short return epilogue belongs to the next interval. The final marker,
success message, and exit form `tail_phases`. Legacy mode snapshots at HALT,
so its frame includes termination. These boundary conventions are explicit
rather than silently discarding glue costs.

Every `Machine.record()` contributes its emitted token count. PC-refresh
records contribute tokens but zero instructions. Entry and return hooks run
only before actual instruction records, preventing duplicate events at a
refresh. Counts exclude prefill and GO. Per-frame phases plus tail phases
must sum exactly to both global counts; the runner checks this identity.
Per-tic records cover each `TryRunTics` invocation, including input handling,
and record guest gametics and the held-key mask at its return.

The first frame fills 76,800 pixels. Later frames normally update the
64,000-pixel view while the status band retains its last guest-rendered
pixels. The observer maintains that display surface. Reports distinguish
`updated_pixels`, `retained_pixels`, and cumulative `written_pixels`.
All pixels must have been supplied by guest strips, and each frame must
update at least the full view's pixel count. This gate is scoped to the
normal E1M1 view, not menus or automap.

`report.json` contains per-frame and per-phase instructions and tokens,
per-tic counts, frame hashes, guest game state, memory bounds, the workload,
and success gates. Frame `instructions` and `tokens` are interval counts;
`instruction_offset` and `generated_token_offset` are cumulative.
`strips.json` retains cumulative strip stop offsets and frame numbers.
`frame-NNNN.rgb565` and `frame-NNNN.png` contain each displayed frame;
`frame.rgb565` and `frame.png` contain the last one. `ram.bin` is final guest
RAM. Do not publish game-bearing outputs without the appropriate rights.

`steady_state_extrapolation.tokens_per_frame` is the arithmetic mean of
frames 2 through N, including their overhead. It is null for a one-frame
run. It is a reference-machine token estimate for this short workload,
not a measured neural timing or proof of long-run steady state.

## Verification scope

Local verification uses only the software reference on a Mac. No CUDA,
Triton, GPU inference, compiled neural execution, or real-time performance
is established. Long sessions, later levels, menus, automap, sound, save
and load, live input latency, and recovery are not verified by this work.
The current observer needs retained ELF symbols and the matching build
manifest. Input events are tested through the engine keyboard path; the
original hardware controller glue remains outside this workload.

## Recorded local check

A fresh default build passed the exact historical first-frame regression.
Opt-in one-frame idle mode passed at 3,885,608 instructions and 6,343,019
tokens with the same image hash. Both 10-frame idle and scripted walk runs
passed with gametic and leveltime 15. The walk schedule held forward during
tics 2 through 11 and released it at tic 12. Every displayed image after
frame 1 differed from the corresponding idle image. A checker recomputed
all raw hashes and compared every decoded PNG byte with its RGB565 source.
A separate rebuild reproduced the walk's instruction and token totals,
phase counts, and every frame hash exactly.
A 100-instruction budget exhaustion produced a failed report and exit code 1.
Configuration generation was checked for 1, 10, and 100 frames; a 100-frame
gameplay run was not performed.

The 10-frame walk reference counts, including the final tail, were:

| Phase | Instructions | Trace tokens |
| --- | ---: | ---: |
| Startup | 134,313 | 237,404 |
| Level initialization | 187,232 | 309,655 |
| Simulation | 285,056 | 443,715 |
| Rendering | 28,230,196 | 45,865,842 |
| Overhead | 1,542 | 2,025 |
| Total | 28,838,339 | 46,858,641 |

The arithmetic mean over frames 2 through 10 was 4,501,708.33 reference
tokens per displayed frame. This short estimate includes alternating one-
and two-tic frames and moving viewpoints. It does not establish a long-run
steady-state distribution or any neural execution rate.

The report checker accepts the legacy and two 10-frame report paths:

```sh
.venv/bin/python experiments/dmc-rv32/check_continuing.py \
  --first-frame /path/to/legacy-run/report.json \
  --walk /path/to/walk-run/report.json \
  --idle /path/to/idle-run/report.json \
  --repeat /path/to/previous-walk-run/report.json
```

`--repeat` is optional and compares instruction and token totals, phase
counts, and every frame hash across two walk runs. The checker expects the
supplied `walk-forward.json` behavior and exactly ten continuing frames.
It reads local game-bearing frame files; it does not run the guest itself.
