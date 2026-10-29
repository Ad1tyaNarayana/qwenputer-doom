/* SPDX-License-Identifier: MIT */
.section .text.init,"ax",@progbits
.global _start
_start:
  .option push
  .option norelax
  la sp, __stack_top
  la tp, __tls_base
  .option pop
  call dmc_size_entry
  li a7,93
  ecall
1: j 1b
