def workloads(ram_base, ram_bytes, drom_base, drom_bytes):
    end = ram_base + ram_bytes
    success = f'''_start:
li t0, {ram_base + 4}
lw t2, 0(t0)
li t0, {ram_base + ram_bytes // 2 + 64}
lw t3, 0(t0)
li t0, {end - 8}
lw t4, 0(t0)
li a7, 1
li t0, {ram_base}
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
li t0, {ram_base + ram_bytes // 2}
li t1, 0xdeadbeef
sw t1, 0(t0)
lw s5, 0(t0)
li t0, {end - 4}
li t1, 0x817f80ff
sw t1, 0(t0)
lb s6, 3(t0)
lbu s7, 3(t0)
lh s8, 2(t0)
lhu s9, 2(t0)
lw s10, 0(t0)
li t0, {ram_base}
lw s11, 0(t0)
li a0, 65
ecall
li t0, {drom_base}
lb s0, 0(t0)
lbu s1, 0(t0)
lh s2, 2(t0)
lhu s3, 2(t0)
lw s4, 0(t0)
li t0, {drom_base + drom_bytes // 2}
lw s5, 0(t0)
li t0, {drom_base + drom_bytes - 4}
lw s6, 0(t0)
lbu s7, 3(t0)
li t0, {drom_base}
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
'''
    cases = [('widths-banks-and-rom', success, None, 'ABC')]
    faults = [
        ('ram-before-load', ram_base - 1, 'lb', 'load out of range'),
        ('ram-after-load', end, 'lw', 'load out of range'),
        ('ram-before-store', ram_base - 1, 'sb', 'store out of range'),
        ('ram-after-store', end, 'sw', 'store out of range'),
        ('drom-before-load', drom_base - 1, 'lb', 'load out of range'),
        ('drom-after-load', drom_base + drom_bytes, 'lw', 'load out of range'),
        ('drom-store', drom_base, 'sw', 'store out of range'),
        ('old-ram-alias', ram_bytes // 2, 'lw', 'load out of range'),
        ('old-drom-alias', 0x20000, 'lw', 'load out of range'),
        ('misaligned-half-load', ram_base + 1, 'lh', 'misaligned load'),
        ('misaligned-half-store', ram_base + 1, 'sh', 'misaligned store'),
    ]
    for name, address, operation, fault in faults:
        operands = f'a0, 0(t0)' if operation[0] == 'l' else 'zero, 0(t0)'
        cases.append((name, f'_start:\nli t0, {address}\n{operation} {operands}\n', fault, ''))
    return cases
