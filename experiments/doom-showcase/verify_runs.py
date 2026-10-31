#!/usr/bin/env python3
"""Compare exported private GPU outputs with independent Mac references and write non-asset evidence."""
import hashlib
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/doom-showcase'
TRACE = ROOT / '.cache/neural-accelerator/legacy-neural-inputs-20261030/doom/reference.tokens.u32'
OBSERVER = ROOT / '.cache/dmc-continuing/legacy-run-v2'
HISTORICAL = json.loads((ROOT / 'experiments/dmc-cuda/evidence/full-report.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(name):
    assert name in ('ff-full', 'unified-full'), name
    run = CACHE / f'export-{name}'
    report = json.loads((run / 'report.json').read_text())
    assert report['complete'] and report['halted'] and report['mismatches'] == 0 and 'error' not in report
    assert report['interruptions'] == 0 and report['uninterrupted_wall_claim_valid']
    actual = np.fromfile(run / 'actual.tokens.u32', dtype='<u4')
    expected = np.fromfile(TRACE, dtype='<u4')
    assert len(actual) == len(expected) == HISTORICAL['generated_tokens']
    differing = int(np.count_nonzero(actual != expected))
    assert differing == 0
    assert (run / 'frame-0001.rgb565').read_bytes() == (OBSERVER / 'frame.rgb565').read_bytes()
    chunks = [json.loads(line) for line in (run / 'chunks.jsonl').read_text().splitlines()]
    position = 0
    for chunk in chunks:
        assert chunk['begin_offset'] == position
        position = chunk['end_offset']
        raw = actual[chunk['begin_offset']:position].astype('<u4').tobytes()
        assert hashlib.sha256(raw).hexdigest() == chunk['sha256']
    assert position == len(actual)
    seconds = sum(c['seconds'] for c in chunks)
    assert abs(seconds - report['retained_chunk_seconds']) < 1e-6
    full = [1e6 * c['seconds'] / c['amount'] for c in chunks if c['amount'] == 8192]
    result = dict(run=name, tokens_compared=len(actual), differing_tokens=differing,
                  independent_trace_sha256=sha(TRACE), actual_trace_sha256=sha(run / 'actual.tokens.u32'),
                  frame_rgb565_sha256=sha(run / 'frame-0001.rgb565'), frame_bytes_equal_mac_observer=True,
                  chunks=len(chunks), full_size_chunks=len(full), chunk_hashes_verified=True,
                  microseconds_per_token_full_chunks=dict(min=min(full), median=statistics.median(full), max=max(full)),
                  retained_chunk_seconds=report['retained_chunk_seconds'],
                  uninterrupted_end_to_end_seconds=report['uninterrupted_end_to_end_seconds'],
                  speedup_vs_historical_retained=HISTORICAL['compute_seconds_total'] / report['retained_chunk_seconds'],
                  historical_retained_seconds=HISTORICAL['compute_seconds_total'])
    if name == 'unified-full':
        assert (run / 'final-gpu-ram.bin').read_bytes() == (OBSERVER / 'ram.bin').read_bytes()
        result['final_ram_bytes_equal_mac_observer'] = True
        registers = json.loads((run / 'actual-registers.json').read_text())
        reference = json.loads((ROOT / 'experiments/neural-accelerator/legacy-doom.json').read_text())['original']['regs']
        assert set(registers) == {'regs.rs1', 'regs.rs2', 'regs.rd'}
        assert all(values == reference for values in registers.values()), 'Final registers differ from reference'
        result['all_three_register_copies_equal_reference'] = True
        result['final_state_coverage'] = 'Literal final RAM and all 96 register values compared with Mac reference'
    else:
        result['final_state_coverage'] = 'Historical ff-full has no final RAM or register export; worker-reported RAM hash only'
        if (run / 'final-gpu-ram.bin').exists():
            assert (run / 'final-gpu-ram.bin').read_bytes() == (OBSERVER / 'ram.bin').read_bytes()
            result['final_ram_bytes_equal_mac_observer'] = True
    result['verifier_sha256'] = sha(Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True, help='New evidence directory; historical evidence is never overwritten')
    evidence = parser.parse_args().out
    evidence.mkdir(parents=True, exist_ok=False)
    for name in ('ff-full', 'unified-full'):
        result = verify(name)
        (evidence / f'{name}-validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
