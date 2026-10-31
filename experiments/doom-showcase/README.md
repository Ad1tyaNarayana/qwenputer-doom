# Run the private Doom showcase

This builds a fresh, self-fed fused FreeRun from ELF boot and checks every
generated token. Pixels, traces and RAM stay private; only JSON evidence is kept.

## Results

| Run | Network | Tokens | Mismatches | Generation (s) | Uninterrupted wall (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| `ff-full` | Original 44-layer circuit, ROM + feedback fusion | 6,342,933 | 0 | 1,446.28 | 1,469.41 |
| `unified-full` | Unified W=4 (`0e515bbb…`), graph-derived fusion | 6,342,933 | 0 | 1,693.11 | 1,715.49 |

The historical run took 3,044.57 retained seconds on the original circuit, so
`ff-full` gives an observed 2.105× historical generation-time ratio on the same
ELF, circuit and token trace. The historical run was interrupted and began from
warm-up histories. This is consistent with the [controlled alternating-window
experiment](../dmc-feedback-fusion/README.md), the better evidence for the fusion effect. `unified-full` uses a
different, larger network and costs 17.07% more per token than `ff-full`.
Both reach HALT and match the historical trace, frame and final RAM hashes.
Run 1 has no final RAM or register export; unified Doom additionally passes
literal Mac RAM and all three register-export comparisons.
A 4,096-token stock `generate` prefix of the same ELF on the shared weights also
matches. Doom's full frame was not run in stock `generate`.

`verify_runs.py` recompares exported outputs with independent Mac references and
writes a new evidence directory supplied with `--out`. `unified_worker.py` is separate so `ff-full` stays bound to the
`showcase_worker.py` hash in its report. The walk harness and inputs are ready, but
continuing frames 2 and 3 have not been run on the GPU.

The detached example below uses `Start-Process`, which dies with the SSH session.
Actual runs used `Invoke-CimMethod Win32_Process Create` under `C:\Users\Admin\gpu-lock`.

## Prepare and check on the Mac

From the repository root, use the existing environment. No installs are needed.

```sh
.venv/bin/python -B experiments/doom-showcase/prepare_walk_reference.py
.venv/bin/python -m py_compile experiments/doom-showcase/*.py
.venv/bin/python -B -m unittest discover -s experiments/doom-showcase -p test_local.py -v
```

The reference command requires a fresh `.cache/doom-showcase/walk-inputs/`.
Use `--out .cache/doom-showcase/walk-inputs-repeat` for a repeat. It executes
`Machine.record()` from ELF boot through entry to `qp_frame_ready` for frame 3.
It uses memory-map `prepare.py`, `elf_info.py`, and workload key definitions.
It checks every strip descriptor and offset against the existing
`walk-run-v2/strips.json`, and checks each fresh display byte against that
run's RGB565 observer output. The older observer report stores cumulative
`tokens` and `instructions`; interval counts come from its `phases`.
All private outputs stay under gitignored `.cache/`.

The trace ends before the frame-ready function executes, after any required
PC-refresh record. It excludes GO and prefill. BYTE256 strip offsets and
frame-end offsets are separate stops. The 10-frame ELF is unchanged; stopping
at frame 3 does not require it to HALT. `--frames 1` or `--frames 2` can use the
same three-frame inputs on the GPU. More frames require fresh reference inputs.

## Sync into a new Windows artifact directory

Run these commands only after the GPU operator schedules the showcase.
These are instructions, not commands executed during implementation.
Set a unique name. This example uses `verified-01`. Both sync operations below
write into the new artifact directory and never overwrite the PC repo/probe.
The existing interpreter and upstream checkout paths below follow the original
private-work layout. Check these paths with read-only `dir` before launch.

PowerShell is sent as UTF-16LE base64 because the SSH login shell is `cmd`.
This creates the fresh destination or fails if it already exists.

```sh
ps=$(cat <<'PS'
$ErrorActionPreference = 'Stop'
$root = 'C:/Users/Admin/qwenputer-games-work/artifacts/doom-showcase-verified-01'
if (Test-Path $root) { throw 'Choose a new artifact directory' }
New-Item -ItemType Directory -Path $root | Out-Null
PS
)
encoded=$(printf '%s' "$ps" | .venv/bin/python -c 'import sys,base64; print(base64.b64encode(sys.stdin.read().encode("utf-16le")).decode())')
ssh gpu "powershell -NoProfile -EncodedCommand $encoded"

tar -cf - \
  experiments/doom-showcase/showcase_worker.py \
  experiments/dmc-feedback-fusion/fusion.py \
  experiments/dmc-feedback-fusion/evidence/checks.json \
  experiments/dmc-feedback-fusion/evidence/verify.json \
  experiments/dmc-rom-fusion/fusion.py \
  experiments/dmc-rom-fusion/analyze.py \
  experiments/dmc-rom-fusion/evidence/dependencies.json \
  experiments/dmc-cuda/evidence/full-report.json \
  | ssh gpu 'tar -xf - -C C:/Users/Admin/qwenputer-games-work/artifacts/doom-showcase-verified-01'

tar -C .cache/doom-showcase -cf - \
  walk-inputs/doom.elf walk-inputs/workload.json \
  walk-inputs/reference.tokens.u32 walk-inputs/reference-trace.json walk-inputs/strips.json \
  | ssh gpu 'tar -xf - -C C:/Users/Admin/qwenputer-games-work/artifacts/doom-showcase-verified-01'
```

The first-frame mode reads the historical probe's `inputs/`. Walk mode reads
`--inputs` in the new artifact directory. The worker makes ordinary copies of
runtime.py, helpers, and the four inputs under `OUT/runtime/probe/`. Its
`runtime.build()` writes only there. An injected prepare wrapper selects the
original clean upstream checkout instead of the runtime's relocated default.
There are no symlinks, writes to the probe's `private-run`, or modified runtime
source. Triton and Inductor cache paths are also placed under `--out`. Walk overrides only the loaded module's ELF/trace hash and token-count
constants with the independently validated manifest values. The original
first-frame constants remain mandatory for first-frame mode.

## Smoke, launch detached, and resume

Use this PowerShell payload for a detached 65,536-token walk smoke. Encode and
send it with the same `encoded=...` and `ssh gpu ...` lines above. `Start-Process`
returns a PID without keeping SSH open. stdout and stderr go to separate files.

```powershell
$ErrorActionPreference = 'Stop'
$homeDir = 'C:/Users/Admin/qwenputer-games-work'
$root = "$homeDir/artifacts/doom-showcase-verified-01"
$python = "$homeDir/.venv/Scripts/python.exe"
$argsList = @(
  '-B', "$root/experiments/doom-showcase/showcase_worker.py",
  '--repo', $root,
  '--probe', "$homeDir/probes/dmc-frame-20261029",
  '--upstream', "$homeDir/.upstream/qwenputer",
  '--inputs', "$root/walk-inputs",
  '--workload', 'walk', '--frames', '3',
  '--out', "$root/smoke-walk", '--max-seconds', '300',
  '--stop-after-tokens', '65536'
)
Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $root `
  -RedirectStandardOutput "$root/smoke-walk.stdout.log" `
  -RedirectStandardError "$root/smoke-walk.stderr.log" -PassThru | Select-Object Id
```

Wait for that PID to exit before scheduling another GPU process. To test resume,
use the same output, add `--resume`, change the limit to `65792`, and use new
stdout/stderr log names. Inspect `report.json` for zero mismatches, the restored
offset, and a fresh checkpoint at 65,792. This smoke does not exercise strips.

For an uninterrupted three-frame timing run, remove `--stop-after-tokens`, use
`--out "$root/walk-full"`, increase `--max-seconds` to `14400`, and use new log
names. Do not resume the smoke when claiming uninterrupted end-to-end time.
For the historical first-frame fixture, remove `--inputs`, set `--workload first-frame --frames 1`, and use a different fresh output directory.

For recovery of a full run, repeat its arguments with `--resume`, the same
output directory, and new stdout/stderr filenames. Do not change the worker,
fusion sources, frame limit, or inputs between attempts. Resume asserts their
hashes. Run directory names must remain separate from the historical probe.

## Read timing and recovery evidence

`chunks.jsonl` records each successful chunk's range, hash, own seconds, and
retained cumulative seconds. Normal chunks contain 8,192 tokens. Strip stops,
frame stops, and an explicit smoke limit can shorten a chunk. `perf_counter`
around `fr.run()` matches the historical timing scope, including output copies
and CPU transfer. Comparisons, extraction, disk IO, checkpointing, setup, and
capture are outside this sum. `report.json` compares a completed first-frame
chunk sum with the historical 3,044.574422900019 seconds. Walk comparisons are
marked as different workloads and have no speedup field value.

The wall clocks start at module entry and at post-setup respectively. Setup
includes imports, validation, input copies, construction, and baseline capture;
the fused capture/reset interval is also reported separately. Restore time is
included in `setup_and_restore_seconds`. Wall measurements end immediately
before final report serialization, after checkpoints and end-clock sampling.
That last report write is outside the wall measurement. A resumed run records
per-process wall time, `interruptions`, and null uninterrupted-wall fields.
`sessions.json` preserves prior process reports. Retained chunk seconds cover
only the checkpointed path after discarded work has been truncated on resume.
No field adds recovery downtime into a fabricated uninterrupted wall time.

Checkpoints contain every head's S and lag histories, x, tok, zero fusion
scratch, the trace digest, offset, chunk sum, completed strips/frames, display,
per-frame coverage, and chunk-journal length. They are fsynced and replaced
atomically with bounded Windows sharing retries. The trace and journal are
fsynced first. Resume checks all retained actual tokens against the reference
and the saved digest, then truncates discarded trace and chunk-journal tails.
It restores state into the captured buffers, including the fusion's shared S
and history views. No reference execution snapshot supplies continuation state.
The initial checkpoint is at boot. Later checkpoints occur about every 262,144
tokens, at every strip/frame boundary, and at a requested smoke stop.

`--max-seconds` is a hard process watchdog with exit code 124. It can kill a
write; resume uses the last successfully replaced checkpoint. An interrupted
initial setup without a checkpoint needs a new output directory. The runtime
sets the existing 8 GiB allocator cap. The report includes GPU identity, peak
allocation/reservation, current and accepted fusion hashes, and start/end
`nvidia-smi` SM clock samples. These samples are observations, not a clock lock.

## Export only the non-asset evidence

Keep PNGs, RGB565, ELF, traces, checkpoints, runtime copies, and RAM private.
Export only the report/progress/session JSON and per-chunk hashes. The following
copies evidence into ignored local storage. Adjust the source run name once.

```sh
mkdir -p .cache/doom-showcase/export-walk-full
ssh gpu 'tar -cf - -C C:/Users/Admin/qwenputer-games-work/artifacts/doom-showcase-verified-01/walk-full report.json progress.json sessions.json chunks.jsonl' \
  | tar -xf - -C .cache/doom-showcase/export-walk-full
```

## 2026-10-31 blind-review correction

The historical 2.105× ratio above does not isolate the executor effect. Narrow
chunk ranges show stable within-run throughput, not absence of GPU contention.
Inspected execution data flow and output validation jointly support execution;
hashes cannot prove provenance against someone replacing all the evidence.

`verify_runs.py` now requires unified final RAM and register exports. Run 1's
historically narrower coverage remains explicit. Historical reports and validation
files are unchanged. Commit `9815ee1` holds the old verifier; source bindings
remain tied to their recorded revisions, as described in
Historical source bindings.
Fresh retained-artifact results are in `evidence/recheck-20261031/`:

```sh
.venv/bin/python -B experiments/doom-showcase/verify_runs.py --out experiments/doom-showcase/evidence/recheck-20261031
```
