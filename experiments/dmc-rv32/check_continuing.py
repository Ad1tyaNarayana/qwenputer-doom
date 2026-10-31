#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check real reference reports and their guest-rendered frame files."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--first-frame', type=Path, required=True, help='Legacy report.json')
p.add_argument('--walk', type=Path, required=True, help='10-frame walk-forward report.json')
p.add_argument('--idle', type=Path, required=True, help='10-frame idle report.json')
p.add_argument('--repeat', type=Path, help='Previous walk report for deterministic count and pixel comparison')
a = p.parse_args()


def read(path):
    report = json.loads(path.read_text())
    assert report['passed'] and report['accounting_passed'], path
    assert report['halted'] and report['exit_code'] == 0 and report['fault'] is None, path
    for frame in report['frames']:
        stem = path.parent / f"frame-{frame['frame']:04d}"
        raw = stem.with_suffix('.rgb565').read_bytes()
        assert len(raw) == 153600
        assert hashlib.sha256(raw).hexdigest() == frame['frame_sha256']
        pixels = struct.unpack('<76800H', raw)
        rgb = bytes(v for c in pixels for v in (
            ((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31))
        with Image.open(stem.with_suffix('.png')) as image:
            assert image.size == (320, 240) and image.convert('RGB').tobytes() == rgb
    for unit in ('instructions', 'tokens'):
        parts = [f['phases'] for f in report['frames']] + [report['tail_phases']]
        assert sum(sum(v[unit] for v in part.values()) for part in parts) == report[unit]
    return report


legacy, walk, idle = (read(path) for path in (a.first_frame, a.walk, a.idle))
assert legacy['first_frame_regression']['passed']
assert (legacy['instructions'], legacy['tokens'], legacy['frame_sha256']) == (
    3885544, 6342933, '6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2')
for report in (walk, idle):
    assert len(report['frames']) == 10
    assert [f['gametic'] for f in report['frames']] == [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]
    tics = [t for f in report['frames'] for t in f['tics']]
    assert [t['gametic_before'] for t in tics] == list(range(15))
    assert [t['gametic_after'] for t in tics] == list(range(1, 16))
    expected = [int(2 <= tic < 12) for tic in range(15)] if report is walk else [0] * 15
    assert [t['guest_key_mask'] for t in tics] == expected
    assert report['frames'][0]['frame_sha256'] == legacy['frame_sha256']
assert all(w['frame_sha256'] != i['frame_sha256']
           for w, i in zip(walk['frames'][1:], idle['frames'][1:]))
if a.repeat:
    repeat = read(a.repeat)
    for key in ('instructions', 'tokens', 'phase_totals', 'frame_sha256'):
        assert walk[key] == repeat[key], key
    for current, previous in zip(walk['frames'], repeat['frames']):
        assert current['frame_sha256'] == previous['frame_sha256']
        assert current['phases'] == previous['phases']
print('PASS: exact legacy fixture, 10-frame tic progression, guest key press/release,')
print('phase accounting, raw frame hashes, decoded PNGs, and distinct walk/idle output.')
if a.repeat:
    print('PASS: repeat walk instruction/token counts, phases, and all frame hashes.')
