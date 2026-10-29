/* SPDX-License-Identifier: MIT */
ENTRY(_start)
MEMORY {
 ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
 RAM (rw) : ORIGIN = 0x01000100, LENGTH = 0xdf00
 DROM (r) : ORIGIN = 0x02000000, LENGTH = 4M
}
SECTIONS {
 .text : { *(.text.init) *(.text*) } > ROM
 .rodata : { *(.rodata* .srodata*) } > DROM
 .nor : ALIGN(16) { __nor_start = .; KEEP(*(.nor)) __nor_end = .; } > DROM
 .data : { *(.data* .sdata*) } > RAM
 .bss : { *(.bss* .sbss*) *(COMMON) . = ALIGN(16); __heap_start = .; } > RAM
 __frame_start = __heap_start;
 __frame_max = 15208;
 __ram_end = 0x0100e000;
 __stack_top = 0x01010000;
 __heap_end = __heap_start;
 __tls_base = 0x01000100;
 __wad_start = 0; __wad_end = 0;
 PROVIDE(dmc_tic = __frame_start + 6912);
 /DISCARD/ : { *(.eh_frame*) *(.comment) }
}
