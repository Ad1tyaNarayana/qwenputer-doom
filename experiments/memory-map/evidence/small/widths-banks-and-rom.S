_start:
li t0, 16777220
lw t2, 0(t0)
li t0, 16810048
lw t3, 0(t0)
li t0, 16842744
lw t4, 0(t0)
li a7, 1
li t0, 16777216
li t1, 0x11223344
sw t1, 0(t0)
li t1, 0x80
sb t1, 1(t0)
li t1, 0x8001
sh t1, 2(t0)
lw s0, 0(t0)
lb s1, 1(t0)
lbu s2, 1(t0)
lh s3, 2(t0)
lhu s4, 2(t0)
li t0, 16809984
li t1, 0xdeadbeef
sw t1, 0(t0)
lw s5, 0(t0)
li t0, 16842748
li t1, 0x817f80ff
sw t1, 0(t0)
lb s6, 3(t0)
lbu s7, 3(t0)
lh s8, 2(t0)
lhu s9, 2(t0)
lw s10, 0(t0)
li t0, 16777216
lw s11, 0(t0)
li a0, 65
ecall
li t0, 33554432
lb s0, 0(t0)
lbu s1, 0(t0)
lh s2, 2(t0)
lhu s3, 2(t0)
lw s4, 0(t0)
li t0, 33685504
lw s5, 0(t0)
li t0, 33816572
lw s6, 0(t0)
lbu s7, 3(t0)
li t0, 33554432
lw s8, 0(t0)
auipc s9, 0
call callee
li a0, 67
ecall
li a7, 93
ecall
callee:
li a0, 66
ecall
ret
