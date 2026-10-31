#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run DMC Doom in the software RV32 reference and observe guest output."""
import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time

from PIL import Image

from elf_info import inspect
from phase_profile import PhaseProfile
from workload import KEYS

p = argparse.ArgumentParser(description=__doc__)
for name in ('prepare', 'upstream', 'elf', 'scratch', 'out'):
    p.add_argument('--' + name, type=Path, required=True)
p.add_argument('--max-instructions', type=int, default=30000000)
p.add_argument('--frames', type=int, default=1, help='Expected displayed frames')
p.add_argument('--check-first-frame', action='store_true')
a = p.parse_args()
if a.frames < 1 or a.max_instructions < 1:
    p.error('--frames and --max-instructions must be positive')

info = inspect(a.elf)
syms = info['symbols']
continuing = 'qp_frame_ready' in syms
if not continuing and a.frames != 1:
    p.error('Multiple frames require an ELF built with --frames')
if continuing and a.check_first_frame:
    p.error('--check-first-frame requires a legacy first-frame ELF')
workload = None
if continuing:
    workload = json.loads(a.elf.with_name('workload.json').read_text())
    if workload['elf_sha256'] != info['sha256'] or workload['frames'] != a.frames:
        p.error('workload.json must match the ELF hash and --frames')

sp = importlib.util.spec_from_file_location('prep', a.prepare)
mod = importlib.util.module_from_spec(sp)
sp.loader.exec_module(mod)
mod.prepare(a.upstream, a.scratch, 64, 4096)
sys.path.insert(0, str(a.scratch))
from chipformer.rv import elf, ref, spec as S

ref.decode = functools.lru_cache(None)(ref.decode)
m = ref.Machine.from_program(elf.program(a.elf))
tokens, last = 0, 0
started = time.time()
strips, frames = [], []
pixels = [0] * (320 * 240)
written = set()
covered = set()
min_sp = S.SP_INIT
profile = PhaseProfile(syms)
frame_strip_start = 0
a.out.mkdir(parents=True, exist_ok=True)


def word(name):
    return m.ram[(syms[name] - S.RAM_BASE) // 4]


def byte(addr):
    assert S.RAM_BASE <= addr < S.RAM_BASE + 4 * S.RAM_WORDS, hex(addr)
    return (m.ram[(addr - S.RAM_BASE) // 4] >> ((addr & 3) * 8)) & 255


def strip():
    d = {k: word('qp_strip_' + k) for k in ('ptr', 'x', 'y', 'h', 'n', 'flags', 'serial')}
    assert d['x'] < 320 and d['y'] + d['h'] <= 240, d
    mul = 2 if d['flags'] & 1 else 1
    assert d['x'] + d['n'] * mul <= 320, d
    for c in range(d['n']):
        for y in range(d['h']):
            idx = byte(d['ptr'] + c * d['h'] + y)
            addr = syms['pump_lut'] + 2 * idx
            rgb = byte(addr) | (byte(addr + 1) << 8)
            for k in range(mul):
                pos = (d['y'] + y) * 320 + d['x'] + c * mul + k
                pixels[pos] = rgb
                written.add(pos)
                covered.add(pos)
    d.update(generated_token_offset=tokens, instruction_offset=m.steps,
             frame=len(frames) + 1)
    strips.append(d)


def finish_frame():
    global frame_strip_start
    raw = struct.pack('<76800H', *pixels)
    number = len(frames) + 1
    frame = profile.finish_frame()
    frame.update(frame=number, gametic=word('gametic'), leveltime=word('leveltime'),
                 instruction_offset=m.steps, generated_token_offset=tokens,
                 instructions=sum(v['instructions'] for v in frame['phases'].values()),
                 tokens=sum(v['tokens'] for v in frame['phases'].values()),
                 strips=len(strips) - frame_strip_start,
                 written_pixels=len(covered), updated_pixels=len(written),
                 retained_pixels=len(covered - written), unique_colors=len(set(pixels)),
                 frame_sha256=hashlib.sha256(raw).hexdigest())
    frames.append(frame)
    (a.out / f'frame-{number:04d}.rgb565').write_bytes(raw)
    rgb = bytes(v for c in pixels for v in (
        ((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31))
    Image.frombytes('RGB', (320, 240), rgb).save(a.out / f'frame-{number:04d}.png')
    frame_strip_start = len(strips)
    written.clear()


while not m.halted and m.steps < a.max_instructions:
    if m.n != S.N_MAX:
        key_mask = word('qp_keys') if 'qp_keys' in syms else (0 if continuing else None)
        profile.boundary(m.pc, m.x[1], word('gametic'), key_mask)
        if continuing and m.pc == syms['qp_frame_ready']:
            finish_frame()
    before = m.steps
    rec = m.record()
    tokens += len(rec)
    min_sp = min(min_sp, m.x[2])
    profile.record(m.steps - before, len(rec))
    if rec == [S.tok('BYTE', 256)]:
        strip()
    if m.steps - last >= 1000000:
        last = m.steps
        print(m.steps, hex(m.pc), 'frames', len(frames), 'strips', len(strips), flush=True)

if not continuing:
    finish_frame()
raw = struct.pack('<76800H', *pixels)
(a.out / 'frame.rgb565').write_bytes(raw)
if frames:
    (a.out / 'frame.png').write_bytes((a.out / f'frame-{len(frames):04d}.png').read_bytes())
steady = frames[1:]
report = {
    'kind': 'DMC E1M1 software reference execution; no compiled neural execution',
    'mode': 'continuing' if continuing else 'first_frame',
    'instructions': m.steps, 'tokens': tokens, 'halted': m.halted,
    'exit_code': m.exit_code, 'fault': m.fault, 'pc': m.pc,
    'elapsed_seconds': time.time() - started,
    'output': m.out.decode(errors='replace'),
    'memory': {
        'frame_base': word('qp_frame_base'), 'frame_bytes': word('qp_frame_bytes'),
        'plan_arena_bytes': word('qp_plan_arena'), 'arena_end': syms['__ram_end'],
        'stack_top': S.SP_INIT, 'minimum_sp': min_sp, 'stack_used': S.SP_INIT - min_sp,
    },
    'strips': len(strips), 'written_pixels': frames[-1]['written_pixels'] if frames else 0,
    'frame_sha256': hashlib.sha256(raw).hexdigest(), 'unique_colors': len(set(pixels)),
    'game_state': {n: word(n) for n in ('gamestate', 'gameepisode', 'gamemap', 'gametic', 'leveltime')},
    'elf': {k: v for k, v in info.items() if k != 'symbols'},
    'workload': workload,
    'input_key_bits': {name: 1 << i for i, name in enumerate(KEYS)},
    'count_scope': 'Software RV32 reference instruction records; tokens exclude prefill and GO, include refresh records. No neural timings.',
    'phase_method': 'ELF entry PCs and guest ra returns for TryRunTics, P_SetupLevel, D_Display; nested level initialization is excluded from simulation.',
    'phase_entry_pcs': {n: syms[n] for n in ('D_DoomLoop', 'TryRunTics', 'P_SetupLevel', 'D_Display')},
    'frames': frames, 'phase_totals': profile.total, 'tail_phases': profile.frame,
    'steady_state_extrapolation': {
        'kind': 'Reference-machine arithmetic mean after frame 1; short workload estimate, not a proven long-run steady state',
        'sample_frames': len(steady),
        'tokens_per_frame': sum(sum(v['tokens'] for v in f['phases'].values()) for f in steady) / len(steady) if steady else None,
    },
    'timing': {
        'simulation_hz': 35 if continuing else None, 'display_hz': 24 if continuing else None,
        'boot_tics': 1,
        'display_tic_formula': '1 + floor(frame_number * 35 / 24)' if continuing else None,
    },
}
if a.check_first_frame:
    report['first_frame_regression'] = {'passed': (
        m.steps == 3885544 and tokens == 6342933 and report['frame_sha256'] ==
        '6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2')}

accounted = [f['phases'] for f in frames] + [profile.frame]
report['accounting_passed'] = all(
    sum(sum(phase[unit] for phase in part.values()) for part in accounted) == value
    and sum(phase[unit] for phase in profile.total.values()) == value
    for unit, value in (('instructions', m.steps), ('tokens', tokens)))
if continuing:
    def scheduled_keys(tic):
        held = []
        for state in workload['input']:
            if state['tic'] > tic:
                break
            held = state['keys']
        return sum(1 << KEYS.index(key) for key in held)

    report['input_schedule_passed'] = all(
        t['guest_key_mask'] == scheduled_keys(t['gametic_before']) for t in profile.tics)
expected = (
    m.halted and m.exit_code == 0 and m.fault is None
    and b'QWENPUTER_DMC_FRAME_READY\n' in m.out
    and len(frames) == a.frames and report['accounting_passed']
    and all(f['written_pixels'] == 320 * 240 and f['updated_pixels'] >= 320 * 200
            and f['unique_colors'] > 1 for f in frames)
    and (not continuing or all(f['gametic'] == 1 + f['frame'] * 35 // 24 for f in frames))
    and (not continuing or report['input_schedule_passed'])
    and all(t.get('gametic_after') == t['gametic_before'] + 1 for t in profile.tics)
    and (not a.check_first_frame or report['first_frame_regression']['passed'])
    and report['game_state']['gamestate'] == 0
    and report['game_state']['gameepisode'] == 1 and report['game_state']['gamemap'] == 1
    and report['memory']['frame_base'] + report['memory']['frame_bytes']
        <= report['memory']['arena_end'] - report['memory']['plan_arena_bytes']
    and min_sp >= report['memory']['arena_end'])
report['passed'] = bool(expected)
(a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(a.out / 'strips.json').write_text(json.dumps({
    'kind': 'Reference-derived stop offsets only; no pixels or model state',
    'token_offsets': 'Cumulative record tokens; excludes prefill and GO',
    'elf_sha256': info['sha256'],
    'metadata_addresses': {n: v for n, v in syms.items() if n.startswith('qp_strip_') or n == 'pump_lut'},
    'strips': strips,
}, indent=2) + '\n')
(a.out / 'ram.bin').write_bytes(struct.pack('<%dI' % len(m.ram), *m.ram))
print(json.dumps(report, indent=2))
if not expected:
    raise SystemExit('DMC reference gate failed; inspect report.json')
