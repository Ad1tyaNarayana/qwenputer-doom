# Doomgeneric bare-metal RV32 port

This prototype runs the pinned original Doom engine to a genuine E1M1 frame in
Qwenputer's software reference machine. It has **not** run Doom through the
compiled neural CPU. The full-size compact circuit and its trace remain the next gate.

The first-frame report records 31,506,621 RV instructions and 51,386,454 trace
tokens, exit status 0, no fault, episode 1, map 1, and game tic 1. The 320x200
XRGB8888 framebuffer contains 146 distinct colors and has SHA-256
`7fc65d25045d2218905c3c05e62cac6698330db0d2e8780c3c7e5f53cb84fda6`.
The decoded image was visually inspected: it shows E1M1 and the status bar.
These counts are a workload, not a neural speed measurement.

The memory map is 8 MiB RAM at `0x01000000`, 8 MiB DROM at `0x02000000`,
and the existing 256 KiB ROM at `0x00040000`. The final executable uses 204,160
bytes of code, 98,688 bytes of read-only engine data, 4,196,020 bytes of local
IWAD payload, and 299,584 bytes of static writable data. The bump allocator
used 6,583,200 bytes through the first frame, including the 6 MiB zone and
256,000-byte output buffer. The linker reserves 64 KiB above its heap limit
for the stack; this is a reservation, not a measured peak stack depth.

The engine sources are unmodified Doomgeneric commit
`e257a09527062bfb986b6e5e0780839b6cbd5b71`. The build uses Zig 0.17's
`generic_rv32+m+zba`, ILP32, and its compiler runtime for software floating
point. The ELF flags are zero: no compressed or hardware floating-point ABI.
`baseline_rv32` is unsuitable here because this Zig release enables the C
extension for that CPU. Linux musl headers provide declarations only. No
musl library or Linux startup code is linked.

## Build and reproduce

Keep the WAD, ELF, memory dumps and extracted pixels in a private ignored
output directory. This directory exports source and textual evidence only.
Provide your own local Doom shareware IWAD; do not commit it.

```sh
python3 build.py --zig /path/to/zig \
  --doomgeneric /path/to/pinned/doomgeneric \
  --wad /path/to/local/DOOM1.WAD --out /path/to/private/build

python3 run_reference.py \
  --prepare /path/to/experiments/memory-map/prepare.py \
  --upstream /path/to/clean/pinned/qwenputer \
  --scratch /path/to/fresh/scratch \
  --elf /path/to/private/build/doom.elf \
  --out /path/to/private/reference --max-instructions 50000000
```

The preparation script is the separately exported high-address memory-map
prototype. It copies clean Qwenputer commit
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c` and applies the RAM/DROM window
changes before any package imports. Use a fresh scratch path per run. The
reference runner needs Pillow only to write its decoded PNG. It streams token
counts without retaining the multi-million-token trace in memory.

Omit `--wad` to build the no-IWAD diagnostic variant. The WAD-bearing build
uses a `.wad` DROM section. Its custom `fread` copies from those bytes with
compiled RV load instructions; the host does not supply individual load
answers or rendered pixels. The host supplies the initial memory snapshot.

The platform provides deterministic time, no input and no audio. In
`DG_SetWindowTitle`, called after the first game tic, it makes the existing
wipe-state match the game-state to suppress the initial transition. Otherwise
the first output callback contains the beginning of the screen wipe. The
normal original-engine render and palette conversion produce the pixels.
`DG_DrawFrame` records its buffer pointer and size. After the first tick returns,
main prints `QWENPUTER_DOOM_FRAME_READY` and exits through ecall 93. Character
output uses ecall 1. No other host syscall is implemented.

The libc subset is intentionally limited to this first-frame experiment:
`free` does not reclaim bump allocations, output formatting supports only the
formats needed here, floating diagnostic output is a placeholder, and `atof`
accepts only integer values. Config and save files are absent. Do not treat
this as a complete libc or a verified long-running game port. The original
Doom zone allocator still manages its own 6 MiB region normally.

See NOTICE and the included upstream licenses. Never redistribute a
WAD-bearing executable or derived game data as part of the source result.
