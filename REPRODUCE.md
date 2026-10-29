# Reproduce the Doom frame

This guide runs the whole experiment in order: expand Qwenputer's memory, build Doom as RISC-V firmware, record a reference trace, then generate the frame on a GPU through the compiled CPU circuit. Each step links to its experiment README, which has the full detail and recorded evidence.

**No game data is included.** Supply your own shareware `DOOM1.WAD`. Keep the WAD, the converted data, the ELF, traces and checkpoints in a private, ignored directory: they all carry game assets. `private/` is already in `.gitignore`.

## What you need

| | Tested with |
| --- | --- |
| Reference and build machine | macOS arm64, Python 3.13.4 |
| GPU machine | Native Windows, RTX 5070 Ti (16 GB), Python 3.14.2, torch 2.14.1+cu130, `triton-windows==3.8.0.post29` |
| Compiler | Zig 0.17, installed by `requirements.txt` as the `ziglang` package |
| Qwenputer | [divagr18/qwenputer](https://github.com/divagr18/qwenputer) at `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` |
| Doom port | [atc1441/Vape_DOOM_ScreenShare](https://github.com/atc1441/Vape_DOOM_ScreenShare) at `710f452b59decb07e8b34bad9a66875657d63484`, directory `PixoAspire_DOOM_Port` |
| Game data | Your own Doom 1.9 shareware `DOOM1.WAD` |

No pretrained model weights are needed. The run uses only the compiled CPU layers.

## 1. Set up

```sh
python3 tools/setup.py
```

This clones Qwenputer into `.upstream/qwenputer` at the pinned revision, creates `.venv`, and installs the pinned dependencies. The upstream checkout is never modified. Each experiment copies it into its own scratch directory before patching.

Clone the Doom port at its pinned commit:

```sh
git clone https://github.com/atc1441/Vape_DOOM_ScreenShare.git private/vape-doom
git -C private/vape-doom checkout 710f452b59decb07e8b34bad9a66875657d63484
```

## 2. Expand the CPU's memory

The released CPU has 32 KiB of RAM, and Doom's framebuffer alone is 64,000 bytes. [`experiments/memory-map`](experiments/memory-map/README.md) makes RAM and DROM movable and larger, and builds the circuit's sparse matrices directly so a big map fits in memory. Prove the 64 KiB RAM / 4 MiB DROM map Doom uses:

```sh
.venv/bin/python experiments/memory-map/guard.py \
  --seconds 300 --rss-gib 4 \
  --log private/memory-low.log --status private/memory-low-status.json -- \
  .venv/bin/python experiments/memory-map/run.py \
  --scratch .cache/memory-low --out private/memory-low \
  --ram-kib 64 --drom-kib 4096 --direct-sparse
```

All 12 memory cases should match the reference, and all 65,536 RAM bytes read back from the circuit state should match after every case. Compare with [`evidence/low-memory/report.json`](experiments/memory-map/evidence/low-memory/report.json).

Optionally, repeat the check on the GPU with [`experiments/memory-map-cuda`](experiments/memory-map-cuda/README.md), after setting up the [Windows GPU environment](experiments/gpu-setup/README.md).

## 3. Build Doom as RISC-V firmware

[`experiments/dmc-rv32`](experiments/dmc-rv32/README.md) ports DMC100-Doom's strip renderer to Qwenputer. Convert your WAD into the port's compact tables and NOR image. The script never downloads a WAD:

```sh
.venv/bin/python experiments/dmc-rv32/prepare_data.py \
  --source private/vape-doom/PixoAspire_DOOM_Port \
  --wad /path/to/DOOM1.WAD --out private/data
```

Build the bare-metal RV32IM firmware. It links against the small libc subset in [`experiments/doom-baremetal`](experiments/doom-baremetal/README.md):

```sh
ZIG=$(.venv/bin/python -c "import ziglang, pathlib; print(pathlib.Path(ziglang.__file__).parent / 'zig')")
.venv/bin/python experiments/dmc-rv32/build.py --zig "$ZIG" \
  --source private/vape-doom/PixoAspire_DOOM_Port \
  --gen private/data/gen --nor private/data/nor.img \
  --libc experiments/doom-baremetal --out private/rv-build
```

## 4. Run it on the reference CPU

The software reference machine runs the firmware to the first E1M1 frame. It records where each of the 25 strips lands in the token stream:

```sh
.venv/bin/python experiments/dmc-rv32/run_reference.py \
  --prepare experiments/memory-map/prepare.py \
  --upstream .upstream/qwenputer \
  --elf private/rv-build/doom.elf \
  --scratch private/reference-package --out private/run \
  --max-instructions 10000000
```

Expect 3,885,544 instructions, 6,342,933 trace tokens, exit status 0, and a 320 × 240 frame covering all 76,800 pixels. Compare with [`report.json`](experiments/dmc-rv32/report.json) and [`strips.json`](experiments/dmc-rv32/strips.json). Build paths can change the ELF hash. The frame pixel hash is the behavioural check.

Then write the full reference token trace that the GPU run is checked against:

```sh
.venv/bin/python experiments/dmc-cuda/source/reference_trace.py \
  --package private/reference-package \
  --elf private/rv-build/doom.elf \
  --trace private/reference.tokens.u32 \
  --report private/reference-trace.json
```

This is software reference generation. It is only used afterwards to check the GPU's tokens, and it never feeds the GPU run.

## 5. Generate the frame on the GPU

[`experiments/dmc-cuda`](experiments/dmc-cuda/README.md) runs the firmware through Qwenputer's compact Triton circuit executor. Lay out a private work directory as that README describes. Put the ELF, `strips.json` and the reference trace under `probes/dmc-frame/inputs/`, then run on the Windows GPU machine:

```powershell
.venv\Scripts\python.exe probes\dmc-frame\prefix_probe.py
.venv\Scripts\python.exe probes\dmc-frame\full_worker.py --max-seconds 120 --stop-after-tokens 65792
.venv\Scripts\python.exe probes\dmc-frame\full_worker.py --max-seconds 4200
```

1. **The prefix probe** generates 65,536 tokens, saves a real GPU checkpoint, restores it, and generates 256 more. All of them are checked against the reference.
2. **The second command** tests the resume path from that checkpoint.
3. **The third command** is the full run. It saves checkpoints of the actual recurrent state as it goes. At each strip, it reads the pixels and palette out of the circuit's own RAM.

On an RTX 5070 Ti, generation took about 51 minutes. If the worker is interrupted, rerun the last command: it resumes from the newest checkpoint.

## 6. Check the result

A successful run must hit all of these:

| Check | Expected |
| --- | --- |
| Generated tokens | 6,342,933, with 0 mismatches |
| Full trace SHA-256 | `23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38` |
| Strips captured from GPU RAM | 25 / 25, covering 320 × 240 |
| RGB565 frame SHA-256 | `6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2` |
| Final GPU RAM SHA-256 (65,536 bytes) | `b9ea6e4ad1a05162f571970604e1a085078d8cf6ae21da48749b47f4dd80b3ef` |

The recorded run's reports, per-strip proof hashes and recovery record are in [`experiments/dmc-cuda/evidence`](experiments/dmc-cuda/evidence).

## Licences

The DMC platform adaptation in `experiments/dmc-rv32` is GPL-2.0-or-later, as is any Doom executable you build. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and each experiment's `NOTICE`. The rest of the code is MIT.
