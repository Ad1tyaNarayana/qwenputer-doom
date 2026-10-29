"""Prepare a pinned standalone CPU with aligned RAM and DROM windows."""
import hashlib
from pathlib import Path
import shutil
import subprocess

REVISION = 'ca13310e2f9f788d1dd3feeea8213c1b0c23913c'

REGION_HELPER = '''def region_literals(bits, base, size):
    """Upper address literals for an aligned power-of-two memory window."""
    if size <= 0 or size & (size - 1) or base % size or not 0 <= base < base + size <= 1 << 32:
        raise ValueError((base, size))
    width = size.bit_length() - 1
    return [bits[i] if base >> i & 1 else 1 - bits[i] for i in range(width, 32)]


'''

OLD_SNAPSHOT = '''    for h in range(S.RAM_WORDS >> 8):
        r0, r1 = {}, {}
        for wi in range(S.RAM_WORDS):
            if (wi & 15 | (wi >> 12) << 4) != h or not m.ram[wi]:
                continue
            row = np.zeros(HD)
            row[list(REF_COLS)] = A_REF
            for i in range(32):
                row[2 + i] = A_BIT * (m.ram[wi] >> i & 1)
            (r1 if wi >> 4 & 1 else r0)[wi >> 5 & 127] = row
        c.tables[f"ram{h}"].init = [r0, r1]'''

NEW_SNAPSHOT = '''    for h in range(S.RAM_WORDS >> 8):
        c.tables[f"ram{h}"].init = [{}, {}]
    for wi, value in enumerate(m.ram):
        if not value:
            continue
        h = wi & 15 | (wi >> 12) << 4
        row = np.zeros(HD)
        row[list(REF_COLS)] = A_REF
        for i in range(32):
            row[2 + i] = A_BIT * (value >> i & 1)
        c.tables[f"ram{h}"].init[wi >> 4 & 1][wi >> 5 & 127] = row'''


def prepare(upstream: Path, scratch: Path, ram_kib: int, drom_kib: int):
    for size, minimum in ((ram_kib, 16), (drom_kib, 8)):
        if size < minimum or size > 8192 or size & (size - 1):
            raise ValueError('Windows require power-of-two KiB sizes within their supported ranges.')
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected {REVISION}, got {revision}')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True):
        raise ValueError('Use a clean pinned upstream checkout.')
    source, destination = upstream / 'experimental/riscv/chipformer', scratch / 'chipformer'
    if destination.exists():
        raise ValueError(f'Choose a fresh scratch directory. Package already exists at {destination}.')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patches = {
        'rv/spec.py': [
            ('DROM_BASE, DROM_WORDS = 0x20000, 1 << 15', f'DROM_BASE, DROM_WORDS = 0x02000000, {drom_kib * 256}'),
            ('RAM_BASE, RAM_WORDS = 0x0, 1 << 13', f'RAM_BASE, RAM_WORDS = 0x01000000, {ram_kib * 256}'),
            ('DATA_BASE = 0x100', 'DATA_BASE = RAM_BASE + 0x100'),
        ],
        'rv/core.py': [
            ('def build(target:', REGION_HELPER + 'def build(target:'),
            ('    c = RvCore(target, text)', '    if text is not None:\n        raise ValueError("This memory-map prototype supports the standalone CPU only.")\n    c = RvCore(target, text)'),
            ('hl = lits_of(EA[13:17], h)', 'hl = lits_of(EA[13:(4 * S.DROM_WORDS).bit_length() - 1], h)'),
            ('in_ram = c.and_([1 - b for b in EA[15:32]], "in_ram")', 'in_ram = c.and_(region_literals(EA, S.RAM_BASE, 4 * S.RAM_WORDS), "in_ram")'),
            ('in_drom = c.and_([1 - b for b in EA[18:32]] + [EA[17]], "in_drom")', 'in_drom = c.and_(region_literals(EA, S.DROM_BASE, 4 * S.DROM_WORDS), "in_drom")'),
            (OLD_SNAPSHOT, NEW_SNAPSHOT),
        ],
    }
    manifest = {}
    for relative, replacements in patches.items():
        original = (source / relative).read_text()
        updated = original
        for before, after in replacements:
            if updated.count(before) != 1:
                raise ValueError(f'Expected one patch site in {relative}: {before[:70]}')
            updated = updated.replace(before, after)
        (destination / relative).write_text(updated)
        manifest[relative] = {
            'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'scratch_sha256': hashlib.sha256(updated.encode()).hexdigest(),
            'replacements': replacements,
        }
    return manifest
