#!/usr/bin/env python3
"""Generate private walk tokens from boot, observing the guest strip ABI."""
import argparse
import array
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--elf', type=Path, default=REPO / '.cache/dmc-continuing/walk-build/doom.elf')
    p.add_argument('--out', type=Path, default=REPO / '.cache/doom-showcase/walk-inputs')
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--max-instructions', type=int, default=40000000)
    p.add_argument('--observer', type=Path, default=REPO / '.cache/dmc-continuing/walk-run-v2')
    a = p.parse_args()
    assert not a.out.exists(), 'Choose a fresh output directory.'
    assert subprocess.run(['git', '-C', str(REPO), 'check-ignore', '-q', str(a.out / 'doom.elf')]).returncode == 0
    sys.path.insert(0, str(REPO / 'experiments/dmc-rv32'))
    from elf_info import inspect
    from workload import KEYS
    info = inspect(a.elf)
    workload = json.loads(a.elf.with_name('workload.json').read_text())
    assert info['sha256'] == workload['elf_sha256']
    assert 1 <= a.frames <= workload['frames']
    assert KEYS and workload['mode'] == 'continuing'
    a.out.mkdir(parents=True)
    scratch = a.out / 'reference-machine'
    prep_spec = importlib.util.spec_from_file_location('showcase_prepare', REPO / 'experiments/memory-map/prepare.py')
    prep = importlib.util.module_from_spec(prep_spec)
    prep_spec.loader.exec_module(prep)
    prep.prepare(REPO / '.upstream/qwenputer', scratch, 64, 4096)
    sys.path.insert(0, str(scratch))
    from chipformer.rv import elf, ref, spec as S
    from showcase_worker import draw_strip
    import numpy as np
    ref.decode = functools.lru_cache(None)(ref.decode)
    m = ref.Machine.from_program(elf.program(a.elf))
    symbols = info['symbols']
    addresses = {n: v for n, v in symbols.items() if n.startswith('qp_strip_') or n == 'pump_lut'}
    observer = json.loads((a.observer / 'report.json').read_text())
    observer_strips = json.loads((a.observer / 'strips.json').read_text())
    assert observer['passed'] and observer['elf']['sha256'] == info['sha256']
    surface = np.zeros((240, 320), dtype='<u2')
    coverage = np.zeros((240, 320), dtype=bool)
    strips, frames, chunks = [], [], []
    offset = 0
    previous = 0
    buffer = array.array('I')
    digest = hashlib.sha256()

    def word(name):
        return m.ram[(symbols[name] - S.RAM_BASE) // 4]

    def ram_bytes(address, count):
        assert S.RAM_BASE <= address <= S.RAM_BASE + 4 * S.RAM_WORDS - count
        return bytes((m.ram[(v - S.RAM_BASE) // 4] >> (8 * (v & 3))) & 255
                     for v in range(address, address + count))

    def flush(output):
        if not buffer:
            return
        block = array.array('I', buffer)
        assert block.itemsize == 4
        if sys.byteorder != 'little':
            block.byteswap()
        raw = block.tobytes()
        output.write(raw)
        digest.update(raw)
        chunks.append({'end_token': offset, 'tokens': len(block), 'sha256': hashlib.sha256(raw).hexdigest()})
        del buffer[:]

    with (a.out / 'reference.tokens.u32').open('wb') as output:
        while not m.halted and m.steps < a.max_instructions:
            if m.n != S.N_MAX and m.pc == symbols['qp_frame_ready']:
                number = len(frames) + 1
                raw = surface.tobytes()
                frame = {'frame': number, 'generated_token_offset': offset,
                         'tokens': offset - previous, 'instruction_offset': m.steps,
                         'frame_sha256': hashlib.sha256(raw).hexdigest(),
                         'updated_pixels': int(coverage.sum())}
                expected = dict(observer['frames'][number - 1])
                expected['generated_token_offset'] = expected.get('generated_token_offset', expected['tokens'])
                expected['instruction_offset'] = expected.get('instruction_offset', expected['instructions'])
                expected['tokens'] = sum(phase['tokens'] for phase in expected['phases'].values())
                for key in ('generated_token_offset', 'tokens', 'instruction_offset', 'frame_sha256', 'updated_pixels'):
                    assert frame[key] == expected[key], (key, frame, expected)
                assert raw == (a.observer / f'frame-{number:04d}.rgb565').read_bytes()
                frames.append(frame)
                previous = offset
                coverage.fill(False)
                print(json.dumps(frame), flush=True)
                if number == a.frames:
                    break
            record = m.record()
            assert m.fault is None
            offset += len(record)
            buffer.extend(record)
            if record == [S.tok('BYTE', 256)]:
                d = {k: word('qp_strip_' + k) for k in ('ptr', 'x', 'y', 'h', 'n', 'flags', 'serial')}
                draw_strip(surface, coverage, d, ram_bytes(symbols['pump_lut'], 512), ram_bytes(d['ptr'], d['n'] * d['h']))
                marker = dict(d, generated_token_offset=offset, instruction_offset=m.steps, frame=len(frames) + 1)
                assert marker == observer_strips['strips'][len(strips)]
                strips.append(marker)
            if len(buffer) >= 65536:
                flush(output)
        flush(output)
    assert len(frames) == a.frames and not m.halted
    shutil.copyfile(a.elf, a.out / 'doom.elf')
    shutil.copyfile(a.elf.with_name('workload.json'), a.out / 'workload.json')
    manifest = {'kind': 'Independent software RV reference; offsets exclude GO and prefill',
                'elf_sha256': info['sha256'], 'metadata_addresses': addresses,
                'frame_marker_pc': symbols['qp_frame_ready'], 'strips': strips, 'frames': frames}
    report = {'kind': 'Private little-endian uint32 software reference; no neural execution',
              'tokens': offset, 'sha256': digest.hexdigest(), 'elf_sha256': info['sha256'],
              'frames': frames, 'chunks': chunks, 'observer_report_sha256': hashlib.sha256((a.observer / 'report.json').read_bytes()).hexdigest(),
              'frame_hash_source': 'Fresh strip observer, checked byte-for-byte against existing run_reference.py outputs',
              'instructions': m.steps, 'halted': m.halted}
    for name, value in [('strips.json', manifest), ('reference-trace.json', report)]:
        (a.out / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'chunks'}), flush=True)


if __name__ == '__main__':
    main()
