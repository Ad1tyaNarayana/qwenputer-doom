/* SPDX-License-Identifier: MIT */
ENTRY(_start)
MEMORY {
 RAM (rw) : ORIGIN = 0x01000100, LENGTH = 0x7fff00
 DROM (r) : ORIGIN = 0x02000000, LENGTH = 8M
 ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K
}
SECTIONS {
 .text : { *(.text.init) *(.text .text.*) } > ROM
 .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > DROM
 .wad : ALIGN(16) { __wad_start = .; KEEP(*(.wad)) __wad_end = .; } > DROM
 .data : { *(.data .data.* .sdata .sdata.*) } > RAM
 .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
 .tbss : { *(.tbss .tbss.*) } > RAM
 .bss : { . += SIZEOF(.tbss); *(.bss .bss.* .sbss .sbss.*) *(COMMON) . = ALIGN(16); __heap_start = .; } > RAM
 __heap_end = 0x017f0000;
 __stack_top = 0x01800000;
 /DISCARD/ : { *(.eh_frame .eh_frame_hdr .comment) }
}
