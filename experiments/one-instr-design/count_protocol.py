"""Count existing reference records by opcode; no neural-performance claim."""
import argparse
from collections import Counter, defaultdict
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--elf', type=Path, required=True)
p.add_argument('--expected-report', type=Path, required=True)
p.add_argument('--scratch', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
s = importlib.util.spec_from_file_location('prepare', ROOT / 'experiments/memory-map/prepare.py')
prep = importlib.util.module_from_spec(s)
s.loader.exec_module(prep)
prep.prepare(ROOT / '.upstream/qwenputer', a.scratch, 64, 4096)
sys.path.insert(0, str(a.scratch))
from chipformer.rv import elf, ref, spec
ref.decode = functools.lru_cache(None)(ref.decode)
m = ref.Machine.from_program(elf.program(a.elf))
expected = json.loads(a.expected_report.read_text())
by_op = defaultdict(lambda: {'instructions': 0, 'tokens': 0, 'lengths': Counter()})
families, refresh = Counter(), 0
while not m.halted and m.steps < expected['instructions'] + 1:
    op = 'REFRESH' if m.n == spec.N_MAX else ref.decode(m.rom[(m.pc-spec.ROM_BASE)//4]).op
    rec = m.record()
    row = by_op[op]
    row['instructions'] += op != 'REFRESH'
    row['tokens'] += len(rec)
    row['lengths'][len(rec)] += 1
    families.update(spec.family(t)[0] for t in rec)
total = sum(r['tokens'] for r in by_op.values())
assert m.halted and not m.fault
assert (m.steps, total) == (expected['instructions'], expected['tokens'])
report = {'scope': 'software reference protocol accounting, not neural execution',
          'upstream_revision': prep.REVISION,
          'elf_sha256': hashlib.sha256(a.elf.read_bytes()).hexdigest(),
          'instructions': m.steps, 'tokens': total, 'tokens_per_instruction': total/m.steps,
          'by_op': dict(sorted(by_op.items())), 'token_families': dict(sorted(families.items()))}
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('instructions', 'tokens', 'tokens_per_instruction')}))
