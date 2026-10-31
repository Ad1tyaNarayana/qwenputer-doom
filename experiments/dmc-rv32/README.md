# DMC Doom on Qwenputer: 64 KiB first-frame port

The user-supplied vape port exposes a practical memory target. Its engine is
Artur Bieniek's DMC100-Doom, adapted to stream game records and render narrow
strips. The Qwenputer glue here runs that real engine to E1M1 in a software
RV32 reference machine with 64 KiB writable RAM. This result does not by
itself establish compiled neural execution.

An opt-in [continuing reference workload](CONTINUING.md) adds deterministic
35 Hz simulation, a 24 Hz display schedule, recorded keyboard input, and
per-frame phase counts. The default build retains this first-frame fixture.

The pinned source is atc1441/Vape_DOOM_ScreenShare commit
`710f452b59decb07e8b34bad9a66875657d63484`, directory
`PixoAspire_DOOM_Port`. Its converter uses Doomgeneric commit
`dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284` as the original table source.
Neither upstream engine source nor any generated game data is included here.

The reference run rendered the original shareware E1M1 view and status bar,
320 by 240 pixels, in 3,885,544 RV instructions and 6,342,933 trace tokens.
It exited with status 0 and no CPU fault, at episode 1, map 1, tic 2. Its 25
strip events wrote every one of 76,800 screen pixels. The little-endian
RGB565 image has 147 distinct colors and SHA-256
`6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2`.
The decoded image was visually inspected. These are reference workload
counts, not a neural performance measurement.

The linked memory and observed reservations are:

| Region | Bytes |
|---|---:|
| Actual code, in the existing 256 KiB ROM | 149,072 |
| Engine read-only data | 31,328 |
| Local sparse NOR image in DROM | 2,453,504 |
| Static writable sections | 13,764 |
| Actual frame scratch region | 15,200 |
| E1M1 plan's level arena | 3,848 |
| Reserved stack at the top of RAM | 8,192 |
| Observed stack use through this frame | 912 |

RAM is `0x01000000 + 64 KiB`, DROM is `0x02000000 + 4 MiB`, and code ROM is
`0x00040000 + 256 KiB`. The frame scratch starts at `0x010036d0`; the level
arena ends at `0x0100e000`, immediately below the reserved stack. Static
caches are included in the writable-section count. There is no full frame
buffer in guest RAM, no DMC_HOST build, and no host renderer.

## Actual execution and strip capture

The engine sources are unmodified. `platform.c` derives from the upstream
size-build platform and replaces the necessary functions with working RV
startup, deterministic time, error/exit handling, a small sound-channel
allocation pool, and real NOR reads. Input is idle and audio is disabled.
The front-end menu drawer is called only when a menu is active. Menu,
intermission, save, flash-write, and long-running gameplay behavior have not
been verified; this is an E1M1 first-frame experiment.

The upstream converter generates compact E1M1 tables and DOOM.DAT from a
local IWAD. Its mknor.py puts the pack into the meter's FAT12 NOR image.
The builder retains the image through its final nonzero 4 KiB sector: the
unused trailing sectors are not loaded into DROM. The real physical NOR
addresses remain unchanged. Requests past the retained payload return
`0xff`; all actual record copies execute RV byte loads and stores against
DROM and RAM. No host callback supplies individual reads.

`pump_submit` receives the renderer's actual indexed strips. It records
pointer, x/y position, height, number of columns, duplication flag, and
serial number in RAM, then emits the existing ecall 1 BYTE token with
payload 256. The observer decodes the strip bytes and the guest's own
RGB565 palette from RAM immediately after that event, before either strip
buffer can be reused. It only arranges already-rendered pixels and converts
their format for a PNG. It never renders geometry or supplies CPU state.

`strips.json` records cumulative generated-token offsets and descriptor
addresses for the exact local ELF SHA-256. Offsets exclude model prefill
and GO; they count tokens emitted by reference instruction records. They
can set independent generation stop budgets. A compiled run must generate
its own tokens, check each marker and the full trace, and decode its own
recurrent RAM at those stops. Reference pixels or recurrent state must not
be substituted. The WAD-bearing ELF and full reference token stream remain
private local files; neither is part of this source export.

## Reproduce

Use a private ignored work directory for the converter outputs and ELF.
Copy the source tree to that directory before running the upstream converter,
since its native table dumper writes under the source's build/ directory.
Supply your own local IWAD; do not publish any converted tables or NOR data.

`prepare_data.py` verifies the source revision and clean working tree, copies
it into the private output directory, fetches the converter's pinned original
Doom source there, and invokes the converter and upstream image builder with
an explicit local WAD. It creates an ignore-all file in its output directory.
The supplied source checkout is read only. A sparse checkout's missing
mknor.py is read from the already-verified local Git revision.

```sh
python3 prepare_data.py \
  --source /path/to/clean/pinned/PixoAspire_DOOM_Port \
  --wad /path/to/local/DOOM1.WAD --out /path/to/private/data
```

The helper locates native llvm-nm/nm and llvm-dwarfdump. On macOS it also asks
`xcrun` for the Command Line Tools llvm-dwarfdump. If needed, give their paths
explicitly:

```sh
python3 prepare_data.py \
  --source /path/to/clean/pinned/PixoAspire_DOOM_Port \
  --wad /path/to/local/DOOM1.WAD --out /path/to/fresh/private/data \
  --nm /usr/bin/nm \
  --dwarfdump /Library/Developer/CommandLineTools/usr/bin/llvm-dwarfdump
```

The included `nm_action_names.py` wrapper runs the selected native nm and
removes the Mach-O leading underscore from `A_*` action names. The helper
sets `NM` to that wrapper, `DMC_REAL_NM` to the selected executable, and
`DWARFDUMP` to the selected llvm-dwarfdump. No hand-written wrapper is needed.
It preserves compiler diagnostics in prepare.log and fails on any unsuccessful
step. It never downloads a WAD.

Then build this port with the new libc subset from the sibling
`doom-baremetal` export. It uses Zig 0.17, RV32IM+Zba, ILP32, no compressed
instructions, and no Linux libc or syscalls:

```sh
python3 build.py --zig /path/to/zig \
  --source /path/to/pinned/PixoAspire_DOOM_Port \
  --gen /path/to/private/data/gen --nor /path/to/private/data/nor.img \
  --libc /path/to/experiments/doom-baremetal --out /path/to/private/rv-build

python3 run_reference.py --prepare /path/to/experiments/memory-map/prepare.py \
  --upstream /path/to/clean/pinned/qwenputer \
  --elf /path/to/private/rv-build/doom.elf \
  --scratch /path/to/fresh/machine-package --out /path/to/private/run \
  --max-instructions 10000000
```

The high-address preparation script is exported separately and pins
Qwenputer `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. The reference observer
needs Pillow for the decoded PNG and exits unsuccessfully unless the real
E1M1 frame covers all 76,800 pixels, has multiple colors, and the CPU exits
without a fault. Build hashes may change with debug paths;
the frame pixel hash and state provide the behavioral reproduction check.

The original engine and this derived platform are GPL-2.0-or-later. New pump,
linker and observer glue is MIT as marked. The combined executable remains
subject to the GPL; preserve the upstream source and notices. See COPYING,
the sibling Doom and Zig notices, and upstream CREDITS.md. Code licenses do
not grant rights to the game data. No WAD, converted tables, NOR image, ELF,
RAM snapshot, or extracted pixels are included in this directory.
