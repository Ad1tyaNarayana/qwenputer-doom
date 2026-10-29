# DMC compiled-circuit frame runner

This is the source and non-asset evidence for the actual DMC RV32 game running through Qwenputer's compact Triton circuit executor. It uses fp32 circuit arithmetic and self-fed generated tokens. It is not the stock Transformers model. The safe export excludes the WAD-bearing ELF, NOR data, full reference/generated traces, RAM snapshots, checkpoints, palette data and strip buffers. The actual frame PNG is a separate local deliverable.

The completed GPU run generated all 6,342,933 tokens with zero mismatches, captured all 25 strips from actual recurrent RAM, and covered all 76,800 pixels of the 320 by 240 E1M1 frame. Independent local comparison found identical full token bytes, RGB565 frame bytes, PNG pixels and all 65,536 final RAM bytes compared with the separately executed Mac reference. The frame has 147 RGB565 colors. Reports and per-strip proof hashes are under `evidence/`.

Successful retained-path generation chunks totalled 3,044.5744 seconds, about 50.7 minutes. This excludes setup, capture, comparison, IO and extraction. A Windows status-file sharing violation interrupted the first attempt near completion. The actual strip-24 checkpoint was restored, a 17-token continuation proof passed, and the final 112,867 tokens completed in 54.31 seconds of worker wall time. Discarded but already validated work was regenerated, so the retained-path sum is not an uninterrupted end-to-end wall-time claim. `evidence/recovery.json` records that distinction.

## Source layout and inputs

Copy the contents of `source/` into a fresh private work directory with this layout:

```text
private-work/
  .upstream/qwenputer/        clean pinned upstream checkout
  .venv/                     isolated Python environment
  probes/dmc-frame/
    runtime.py
    prefix_probe.py
    full_worker.py
    reference_trace.py
    helpers/prepare.py
    helpers/state.py
    inputs/                  private, never publish
      doom.elf
      strips.json
      reference.tokens.u32
      reference-trace.json
```

`runtime.py` locates upstream two directories above its own probe directory. It prepares the 64 KiB RAM / 4 MiB DROM map into a private `scratch/` copy and puts that copy first on `sys.path`. RAM starts at `0x01000000`, DROM at `0x02000000`, and the original 256 KiB code ROM remains at `0x00040000`. It checks the pinned ELF/reference hashes before constructing the runtime. The original upstream checkout stays unchanged.

The tested environment used Python 3.14.2, torch 2.14.1+cu130, CUDA runtime 13.0, and `triton-windows==3.8.0.post29` on an RTX 5070 Ti. The Windows Triton wheel was pinned by SHA-256 `1d0fd6a6cf36ba5cced997177068e328d015e42cd4300f6995fff5a6f8707aff`. Installation added only that package to the already-tested CUDA environment. No pretrained model weights, system CUDA toolkit or WSL are needed by this execution path.

The upstream revision is `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. Preparation and RAM extraction helpers are copied unchanged from the frozen memory-map experiment. The only FreeRun adaptation omits the merged-chat-only Q family when constructing standalone embedding tables. Q is last, so existing standalone family IDs retain their indices. Executor2's arithmetic kernels are unchanged.

## Generate private comparison inputs

Build the private ELF and generate `strips.json` with the sibling DMC port's `run_reference.py`. That observer also creates reference pixels and RAM; do not copy those into the GPU runner inputs. The GPU runner needs only the ELF, stop offsets/ABI addresses and token trace. Input generation and the GPU run remain separate phases.

`reference_trace_original.py` is the exact original local script that generated the independent Mac trace. Its `/tmp` fixture paths are archived for provenance. `reference_trace.py` is the portable CLI adaptation. It was rerun locally against the same prepared package and ELF and reproduced all 6,342,933 tokens with SHA-256 `23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38`.

Use the prepared reference package made by the sibling observer:

```sh
python probes/dmc-frame/reference_trace.py \
  --package /path/to/private/prepared-reference-package \
  --elf /path/to/private/doom.elf \
  --trace probes/dmc-frame/inputs/reference.tokens.u32 \
  --report probes/dmc-frame/inputs/reference-trace.json
```

The output files must be fresh. The portable writer makes little-endian uint32 records explicit. It streams from `ref.Machine.record()` with cached instruction decoding, writes hashes per 65,536-token block, and checks clean termination. This is software reference generation, not neural execution. It never supplies execution state to the GPU runner.

The captured fixture's ELF SHA-256 is `9effb70dfe4cad579f0381a657ac3711564da0278a1410f59789e61fa26ca340`. Build paths or a different local data source may change that ELF hash. Such a fixture needs a fresh independent reference and an explicit update of the expected constants/marker manifest. Do not disable those checks.

## Exact execution commands

Run from `private-work/`, using the isolated environment's Python. On native Windows the tested interpreter command is `.venv\Scripts\python.exe`.

```sh
python probes/dmc-frame/prefix_probe.py
python probes/dmc-frame/full_worker.py --max-seconds 120 --stop-after-tokens 65792
python probes/dmc-frame/full_worker.py --max-seconds 4200
```

The first command has a 120-second process watchdog and an 8 GiB CUDA allocator limit. It constructs the circuit, captures the graph, resets to the ELF boot state and generates eight 8,192-token chunks. Every actual output token is compared with the independently generated trace afterward. The timed prefix was 31.3896 seconds, excluding setup, capture, comparison, file IO and checkpoint handling. It saved a private 83,076,085-byte actual-GPU-state checkpoint at token 65,536, reset the runtime, restored that checkpoint, and generated 256 further exact tokens.

The second command tested the full worker's resume/controller path by advancing the actual checkpoint to token 65,792. The third is the full run. It starts from that actual generated checkpoint, never from a later reference machine snapshot.

## Checkpoints and strip observation

A private checkpoint stores the exact recurrent `S` tensors, lag histories, next self-generated token embedding, last generated token and trace offset. The full worker also stores accumulated generation time. Restoring copies these values into the reconstructed runtime's existing buffers after graph capture. It does not execute RV instructions on the host to reconstruct state. Checkpoints are atomic replacements, saved about every 262,144 tokens and at each strip marker. On restart, the actual trace is truncated to the saved offset and its retained prefix is verified before generation continues.

At each reference-derived stop offset the worker first checks that its own final emitted token is BYTE256. It then reads the descriptor, indexed strip pixels and palette from its own GPU recurrent RAM with `state.read_ram`. Descriptor geometry is compared with the manifest only as validation. Image assembly uses the descriptor, palette and indices extracted from GPU state. It arranges already-rendered column-major pixels and converts RGB565 to PNG; it does not render Doom geometry or import reference pixels. Synthetic normal/doubled-column cases were checked separately.

The terminal gate requires all 25 strips, complete 320 by 240 coverage, HALT, zero trace mismatches, full trace SHA-256 `23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38`, RGB565 frame SHA-256 `6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2`, and final GPU RAM SHA-256 `b9ea6e4ad1a05162f571970604e1a085078d8cf6ae21da48749b47f4dd80b3ef`.

All runtime products live under private `private-run/`. Those include reference-dependent traces and asset-bearing state; keep them outside public artifacts. Live preview unknown pixels are shown as checkerboard by a separate read-only viewer fetcher. The raw GPU preview and original evidence remain unchanged. This source package makes no live-interactive-play claim.

## Observed IO recovery

The original failed source is archived as `source/full_worker_original.py`; `recovery-io.diff` shows the entire correction. The circuit, kernels and GPU-state format did not change. The corrected file boundary retries Windows `PermissionError` for up to five seconds. A real held-reader test verified replacement succeeds after the reader releases its handle. The read-only status fetcher was also changed to allow delete sharing.

The captured recovery commands, after preserving the failed source/report/trace, were:

```sh
python probes/dmc-frame/io_retry_probe.py
python probes/dmc-frame/full_worker.py --max-seconds 120 --stop-after-tokens 6230066
python probes/dmc-frame/full_worker.py --max-seconds 300
python probes/dmc-frame/collect_proof.py
```

The original actual checkpoint was at token 6,230,049, with 24 already-extracted GPU strips. Recovery did not restart the game or inject reference state. `collect_proof.py` exports only hashes and tensor shapes; private checkpoint/trace/RAM contents stay in place. The full actual trace was independently downloaded to a private directory and compared byte-for-byte with the independent Mac trace. Live polling continued beyond completion for the recording tail. No generation processes remain running.
