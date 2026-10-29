"""Generate real compact-circuit traces against a high-address reference map."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys

from prepare import prepare


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ram-kib', type=int, default=64)
    parser.add_argument('--drom-kib', type=int, default=256)
    parser.add_argument('--scale-smoke', action='store_true')
    parser.add_argument('--direct-sparse', action='store_true')
    parser.add_argument('--check-constructor', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    upstream = (args.upstream or args.repo / '.upstream/qwenputer').resolve()
    manifest = prepare(upstream, args.scratch.resolve(), args.ram_kib, args.drom_kib)
    (args.out / 'patch-manifest.json').write_text(json.dumps(manifest, indent=2))
    sys.path.insert(0, str(args.scratch.resolve()))
    import numpy as np
    import torch
    from chipformer.q35.ir import Head, HD
    from chipformer.rv import core, ref, spec
    from chipformer.rv.asm import assemble
    from chipformer.rv.fastexec import Executor
    torch.set_num_threads(1)
    print('Building circuit graph', flush=True)
    circuit = core.build()
    graph = {
        'chip_layers': circuit.n_layers(), 'dims': len(circuit.dims),
        'key_heads': sum(isinstance(n, Head) for n in circuit.nodes),
        'logical_value_heads': sum(len(n.vheads()) for n in circuit.nodes if isinstance(n, Head)),
        'ram_key_heads': sum(k.startswith('ram') for k in circuit.tables),
        'drom_key_heads': sum(k.startswith('drom') for k in circuit.tables),
    }
    graph['logical_recurrent_fp32_bytes'] = graph['logical_value_heads'] * HD * HD * 4
    (args.out / 'graph.json').write_text(json.dumps(graph, indent=2))
    print(json.dumps(graph), flush=True)
    print('Building compact executor', flush=True)
    constructor_tensors = None
    if args.direct_sparse:
        from sparse_exec import DirectSparseExecutor, compare_constructors
        executor = DirectSparseExecutor(circuit, device='cpu')
        if args.check_constructor:
            original = Executor(circuit, device='cpu', sparse=True)
            constructor_tensors = compare_constructors(executor, original)
            del original
            print(f'Constructor equivalence passed for {constructor_tensors} tensors', flush=True)
    else:
        executor = Executor(circuit, device='cpu', sparse=True)
    tensor_bytes = 0
    for step in executor.steps:
        if isinstance(step, dict):
            tensor_bytes += step['S'].numel() * step['S'].element_size()
    assert tensor_bytes == graph['logical_recurrent_fp32_bytes']
    rodata = bytearray(args.drom_kib * 1024)
    for offset, word in [(0, 0x80017f80), (len(rodata) // 2, 0xaabbccdd), (len(rodata) - 4, 0x12345678)]:
        rodata[offset:offset + 4] = word.to_bytes(4, 'little')
    cases = workloads(spec.RAM_BASE, 4 * spec.RAM_WORDS, spec.DROM_BASE, 4 * spec.DROM_WORDS)
    if args.scale_smoke:
        cases = cases[:1]
    results = []
    for name, assembly, expected_fault, expected_output in cases:
        print(f'Generating {name}', flush=True)
        program = assemble(assembly)
        program.rodata = bytes(rodata)
        def machine():
            result = ref.Machine.from_program(program)
            for offset, word in [(4, 0x1234abcd), (2 * spec.RAM_WORDS + 64, 0x89abcdef),
                                 (4 * spec.RAM_WORDS - 8, 0x55aa1234)]:
                result.ram[offset // 4] = word
            return result
        reference = machine()
        wanted = reference.run(1000)[1:]
        assert reference.halted
        actual_fault = None if reference.fault is None else spec.FAULTS[reference.fault[0]]
        assert actual_fault == expected_fault, (name, actual_fault, expected_fault)
        core.snapshot(circuit, machine())
        executor.load_init()
        executor.reset()
        current = spec.tok('GO')
        generated = []
        embeddings = {}
        with torch.no_grad():
            for _ in range(len(wanted) + 2):
                if current not in embeddings:
                    embeddings[current] = executor.embed([current])
                residual = executor.step(embeddings[current])
                current = core.predictions(circuit, residual.numpy())[0]
                generated.append(current)
                if current == spec.tok('HALT'):
                    break
        assert generated == wanted, (name, generated, wanted)
        output = bytes(spec.family(t)[1] for t in generated if spec.family(t)[0] == 'BYTE').decode('latin-1')
        assert output == expected_output
        from state import read_ram
        extracted = read_ram(executor, spec.RAM_BASE, 4 * spec.RAM_WORDS)
        expected_ram = np.asarray(reference.ram, dtype='<u4').tobytes()
        assert extracted == expected_ram, f'{name}: recurrent RAM extraction differs'
        np.asarray(generated, dtype='<u4').tofile(args.out / f'{name}.u32')
        (args.out / f'{name}.S').write_text(assembly)
        results.append({'name': name, 'generated_tokens': len(generated), 'mismatches': 0,
                        'output': output, 'fault': actual_fault,
                        'trace_sha256': hashlib.sha256(np.asarray(generated, dtype='<u4').tobytes()).hexdigest(),
                        'extracted_ram_bytes_compared': len(extracted),
                        'extracted_ram_sha256': hashlib.sha256(extracted).hexdigest(),
                        'registers_after_reference': reference.x})
        print(json.dumps(results[-1]), flush=True)
    report = {
        'mode': 'Compact circuit tensor execution, CPU fp32, free-running generated tokens; not stock HF',
        'is_doom': False, 'standalone_only': True, 'ram_base': spec.RAM_BASE,
        'ram_bytes': 4 * spec.RAM_WORDS, 'drom_base': spec.DROM_BASE,
        'drom_bytes': 4 * spec.DROM_WORDS, 'rom_base': spec.ROM_BASE,
        'rom_bytes': 4 * spec.ROM_WORDS, 'graph': graph, 'cases': results,
        'generated_tokens': sum(r['generated_tokens'] for r in results), 'mismatches': 0,
        'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'snapshot_packer': 'One pass over RAM words, assigning each nonzero word to its bank and row',
        'torch': torch.__version__, 'numpy': np.__version__,
        'direct_sparse_constructor': args.direct_sparse,
        'equivalent_constructor_tensors': constructor_tensors,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
