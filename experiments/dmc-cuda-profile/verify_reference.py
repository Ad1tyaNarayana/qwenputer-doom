# SPDX-License-Identifier: MIT
"""Recompute the seeded window's independent reference hash on another host."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import struct
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--elf', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    assert hashlib.sha256(a.elf.read_bytes()).hexdigest() == report['walk_elf_sha256']
    sys.path.insert(0, str(a.package.resolve()))
    from chipformer.rv import elf, ref
    ref.decode = functools.lru_cache(None)(ref.decode)
    m = ref.Machine.from_program(elf.program(a.elf))
    offset = 0
    while offset < report['seed']['token_offset']:
        offset += len(m.record())
        assert not m.halted
    assert offset == report['seed']['token_offset']
    assert m.pc == report['seed']['pc'] and m.steps == report['seed']['instruction_offset']
    runs = report['windows']['frame2_render']['runs']
    count = runs[0]['match']['tokens']
    tokens = []
    while len(tokens) < count:
        tokens.extend(m.record())
    sha = hashlib.sha256(struct.pack(f'<{count}I', *tokens[:count])).hexdigest()
    assert all(run['match']['sha256'] == sha and run['match']['mismatches'] == 0 for run in runs)
    result = {'kind': 'Independent software-reference recomputation, no GPU execution',
              'platform': sys.platform, 'python': sys.version, 'seed': report['seed'],
              'tokens': count, 'sha256': sha, 'all_recorded_gpu_window_hashes_match': True}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
