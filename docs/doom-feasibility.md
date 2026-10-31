# Stock Doomgeneric exceeds the released Qwenputer memory map

The goal is to run the original Doom engine on Qwenputer and leave an open source result. The released machine fails the writable memory requirement before a platform port can run. No Doom game ran during this audit. Game data was unnecessary for this source audit and is not included in the repository.

Subsequent work verified a standalone 64 KiB RAM extension through stock-model generation. It does not resolve the engine's larger memory requirement. A Doom 1.9 shareware IWAD has since been obtained locally for the port; it remains excluded from the repository.

[GZDoom](https://github.com/ZDoom/gzdoom) is another open-source engine port, with an OpenGL renderer and substantial scripting support. We chose Doomgeneric for this experiment because its small platform callback interface is a narrower porting target. That is an engineering choice, not a demonstrated GZDoom compatibility result. Both still need game data; the [original id Software source release](https://github.com/id-Software/DOOM) also identifies that requirement.

The user subsequently identified [the native Aspire PIXO Doom port](https://github.com/atc1441/Vape_DOOM_ScreenShare/tree/main/PixoAspire_DOOM_Port). It builds on [DMC100-Doom](https://github.com/ArturBieniek4/DMC100-Doom), which reports a 32 KiB RAM target using a strip renderer and streamed map records. That architecture avoids the full-framebuffer and large-zone assumptions in this stock Doomgeneric audit. It is now a candidate for a smaller Qwenputer port. The board's flash and display drivers still need replacement, and the converted game data must fit an expanded compiled memory region; we have not yet executed that engine through the circuit.

The audit pins [Qwenputer ca13310e2f9f788d1dd3feeea8213c1b0c23913c](https://github.com/divagr18/qwenputer/tree/ca13310e2f9f788d1dd3feeea8213c1b0c23913c) and [Doomgeneric e257a09527062bfb986b6e5e0780839b6cbd5b71](https://github.com/ozkl/doomgeneric/tree/e257a09527062bfb986b6e5e0780839b6cbd5b71). The local checkout contents and source hashes are recorded in [`artifacts/doom/audit.json`](../artifacts/doom/audit.json).

Qwenputer's `experimental/riscv/chipformer/rv/spec.py` defines 32,768 bytes of writable RAM, 131,072 bytes of read-only DROM, and 262,144 bytes of code ROM. Its `env/rom.ld` starts writable allocations at `0x100`, reserves the upper 4,096 RAM bytes for the stack, and ends the heap at `0x7000`. This leaves at most 28,416 bytes for the heap before static data, TLS, and BSS. DROM cannot substitute for writable framebuffer or zone storage.

The pinned engine has these source allocations.

| Allocation | Bytes | Evidence |
| --- | ---: | --- |
| Doom indexed framebuffer, 320 by 200 | 64,000 | `doomgeneric/i_video.h` defines the dimensions. `i_video.c` allocates `SCREENWIDTH * SCREENHEIGHT` with `Z_Malloc`. |
| Doomgeneric output buffer, default 640 by 400 at four bytes per pixel | 1,024,000 | `doomgeneric.h` defines the output dimensions. `doomgeneric.c` multiplies them by four in `malloc`. |
| Default Doomgeneric zone | 6,291,456 | `i_system.c` sets `DEFAULT_RAM` and `MIN_RAM` to 6 MiB and allocates the zone with `malloc`. |
| Default zone plus output buffer | 7,315,456 | The indexed framebuffer is inside the zone and is not added again. This excludes other allocations and the stack. |
| Zone plus output configured to 320 by 200 | 6,547,456 | The output buffer remains four bytes per pixel. This also excludes other allocations and the stack. |

The 64,000 byte engine framebuffer alone is 31,232 bytes larger than all current writable RAM. Changing only Doomgeneric's output size, removing its output conversion, or lowering the zone cannot make this renderer's framebuffer fit. The 6 MiB zone is a default of this pinned Doomgeneric port. It is not a measurement of the minimum memory of every Doom implementation. A different renderer or memory system requires a separate feasibility test.

The optional crosscompile successfully compiled all 80 engine translation units listed in Doomgeneric's Makefile, excluding `doomgeneric_xlib.c`. It used the local Zig 0.17 compiler, RV32IM plus Zba, ILP32, `-Os`, and Linux musl headers. It did not link musl or any other libc. The ELF32 section parser counted these unpruned object totals.

| Object sections | Bytes |
| --- | ---: |
| Executable allocated sections | 215,444 |
| Read-only allocated sections | 136,105 |
| Writable allocated sections, including BSS | 297,309 |

The target explicitly uses `generic_rv32+m+zba`; all object ELF flags are checked to exclude compressed instructions and incompatible ABIs. An earlier audit used `baseline_rv32`, which enabled compressed instructions and understated code size. The corrected objects above supersede that measurement. These sums include unused engine functions and data. A final link with garbage collection can remove some of them, while libc, startup code, and a platform backend can add others. The sums do not establish a live code or static data requirement. A relocatable combined-link attempt stopped because this Zig driver rejects linker `-r`. No runnable ELF or final linker map was produced. The independent source framebuffer calculation is sufficient for the no-go.

The runtime API is another port requirement. Qwenputer `env/rt.h` and `ref.py` implement byte output through ecall 1 and process exit through ecall 93. Doomgeneric needs `DG_Init`, `DG_DrawFrame`, `DG_SleepMs`, `DG_GetTicksMs`, `DG_GetKey`, and `DG_SetWindowTitle`. Its default WAD reader uses `fopen`, `fseek`, and `fread`. The successful object compilation does not supply those APIs or a heap allocator.

The smallest useful next gate is a full bare-metal RV32 link that reaches the missing-IWAD diagnostic, with no game data. Implement the Doomgeneric backend and libc subset against an isolated software RV32 reference runner with an explicitly larger memory map. Record the final code, read-only data, static writable data, requested zone, output buffer, and stack budgets. Success means an actual linked ELF starts, initializes its memory and platform code, prints the expected missing-IWAD diagnostic, and exits without an illegal instruction, unsupported ecall, or out-of-range memory access. This would establish a port and an exact memory budget. It would not establish neural Qwenputer execution. Any required memory-map expansion must then be implemented and verified in the Qwenputer circuit before claiming Doom on Qwenputer.

Rerun the source gate with Python 3. The script writes JSON and exits successfully when the audit itself succeeds, including the expected `NO_GO` result.

```sh
python3 tools/audit_doom.py \
  --qwenputer /path/to/qwenputer \
  --doomgeneric /path/to/doomgeneric \
  --output /path/to/audit-output
```

Add `--zig /path/to/zig` to repeat the 80 object compilations. `artifacts/doom/compile-commands.json` records the compiler arguments with local executable and working-directory paths replaced by `<ZIG>` and `<AUDIT_WORKDIR>`. `artifacts/doom/compile.log` records compiler output, and `artifacts/doom/object-sections.json` records each ELF section. Clone the two upstream repositories and check out the pinned revisions before running. The script rejects other HEAD revisions.

The new audit script is marked MIT. It does not vendor Doom source or game data. Doomgeneric's `LICENSE` contains GPL version 2, and engine file notices permit GPL version 2 or later. Preserve the relevant upstream notices if engine source or binaries are incorporated later. The audit records source facts rather than supplying a game-data license.

This is a source and build audit, with no performance claim.
