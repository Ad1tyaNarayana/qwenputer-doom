#!/usr/bin/env python3
"""First E1M1 frame on the shared unified W=4 network, self-fed and token-checked."""
import time
PROCESS_START = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
import traceback
import types
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from showcase_worker import (FINAL_RAM_SHA, atomic_json, clock_sample, draw_strip, heads, restore_checkpoint,
                             save_checkpoint, sha, TRACE_SHA, validate_manifest, verify_prefix, write_frame)

WEIGHT_SHA = '0e515bbb3109997a130c903deb40dce3447f8991e15b6637e0071c71f8b6e07d'
CONSTRUCTOR_SHA = '5e156bdb8fa6c543d96f0760a1a4d18dfea4cba3e558673a909bed1be0fde7da'
HISTORICAL_SECONDS = 3044.574422900019


def fused_acceptance(repo):
    fused = repo / 'experiments/tiny-llm-ndot-fused'
    checks = json.loads((fused / 'evidence/checks.json').read_text())
    assert checks['status'] == 'pass' and not checks['full_state_digest_mismatches']
    assert not checks['reference_mismatches'] and not checks['timing_reference_mismatches']
    hashes = {name: sha(fused / name) for name in ('fusion.py', 'runtime.py')}
    assert all(hashes[name] == checks['source_sha256'][name] for name in hashes)
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'repo', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-seconds', type=float, default=7200)
    p.add_argument('--stop-after-tokens', type=int)
    a = p.parse_args()
    a.out, a.probe, a.repo = a.out.resolve(), a.probe.resolve(), a.repo.resolve()
    assert a.out != a.probe and a.probe not in a.out.parents, 'Output must be separate from the historical probe'
    inputs = a.probe / 'inputs'
    if a.resume:
        assert (a.out / 'checkpoint.pt').is_file()
    else:
        a.out.mkdir(parents=True, exist_ok=False)
    timer = threading.Timer(max(0.01, a.max_seconds - (time.perf_counter() - PROCESS_START)), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    report = {'workload': 'first-frame', 'complete': False, 'reference_tokens_used_as_inputs': False,
              'reference_pixels_used': False, 'reference_state_used_for_resume': False, 'mismatches': 0,
              'timed_work': 'perf_counter around successful fr.run chunks including self-feedback, output copies and CPU transfer',
              'wall_scope': 'Module entry until final report serialization begins; includes setup, capture, comparison, extraction, IO and checkpoints'}
    post_setup = None
    os.environ['TRITON_CACHE_DIR'] = str(a.out / 'kernel-cache/triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(a.out / 'kernel-cache/inductor')
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    try:
        import numpy as np
        import torch
        manifest = json.loads((inputs / 'strips.json').read_text())
        trace_info = json.loads((inputs / 'reference-trace.json').read_text())
        markers, frame_ends = validate_manifest(manifest, trace_info, 'first-frame', 1)
        input_hashes = {name: sha(inputs / name) for name in ('doom.elf', 'reference.tokens.u32', 'strips.json', 'reference-trace.json')}
        assert input_hashes['doom.elf'] == manifest['elf_sha256']
        assert input_hashes['reference.tokens.u32'] == trace_info['sha256'] == TRACE_SHA
        expected = np.memmap(inputs / 'reference.tokens.u32', mode='r', dtype='<u4')
        assert len(expected) == trace_info['tokens']
        fused_hashes = fused_acceptance(a.repo)
        legacy = json.loads((a.repo / 'experiments/neural-accelerator/legacy-doom.json').read_text())['original']
        assert legacy['trace_sha256'] == TRACE_SHA and legacy['final_ram_sha256'] == FINAL_RAM_SHA
        if a.resume:
            shutil.rmtree(a.out / 'runtime')
        fixture = a.out / 'runtime/fixture'
        fixture.mkdir(parents=True)
        shutil.copyfile(inputs / 'doom.elf', fixture / 'engine.elf')
        report['gpu_start'] = clock_sample()
        sys.path.insert(0, str(a.repo / 'experiments/showcase'))
        import gpu_worker
        args = types.SimpleNamespace(repo=a.repo, upstream=a.upstream.resolve(), out=a.out / 'runtime',
                                     inputs=fixture, original_circuit=False)
        fr, fused, spec, metadata = gpu_worker.build(args, {})
        report['runtime'] = metadata
        assert metadata['network_weight_sha256'] == WEIGHT_SHA
        assert fused['circuit'].source_manifest['generated_sha256'] == CONSTRUCTOR_SHA
        assert all(not torch.count_nonzero(s['hist']).item() for s in heads(fr))
        assert not torch.count_nonzero(fr.ex.y).item() and not torch.count_nonzero(fr.tok).item()
        assert all(not torch.count_nonzero(y).item() for y in fused['scratch_buffers'])
        identity = {'inputs': input_hashes, 'fused_acceptance': fused_hashes, 'scripts': metadata['scripts'],
                    'worker': sha(Path(__file__)), 'showcase_worker': sha(HERE / 'showcase_worker.py'),
                    'workload': 'first-frame', 'network_weight_sha256': WEIGHT_SHA}
        report.update(identity)
        report['startup_state'] = 'ELF boot snapshot in the unified circuit; every lag history, residual and fusion scratch buffer asserted zero'
        report['setup_seconds'] = time.perf_counter() - PROCESS_START
        surface = np.zeros((240, 320), dtype='<u2')
        coverage = np.zeros((240, 320), dtype=bool)
        state = {'identity': identity, 'offset': 0, 'chunk_seconds': 0.0, 'strips': [], 'frames': [],
                 'chunks_bytes': 0, 'trace_sha256': hashlib.sha256(b'').hexdigest()}
        trace_path = a.out / 'actual.tokens.u32'
        chunks_path = a.out / 'chunks.jsonl'
        if a.resume:
            state, surface, coverage = restore_checkpoint(fr, fused, a.out / 'checkpoint.pt', identity)
            assert state['offset'] == 0 or int(fr.tok[0].item()) == int(expected[state['offset'] - 1])
            assert verify_prefix(trace_path, expected, state['offset']) == state['trace_sha256']
            with chunks_path.open('r+b') as stream:
                assert stream.seek(0, 2) >= state['chunks_bytes']
                stream.truncate(state['chunks_bytes'])
        else:
            trace_path.touch()
            chunks_path.touch()
        sessions_path = a.out / 'sessions.json'
        sessions = json.loads(sessions_path.read_text()) if a.resume else []
        sessions.append({'pid': os.getpid(), 'started_unix': time.time(), 'resume': a.resume,
                         'checkpoint_offset': state['offset'], 'setup_seconds': report['setup_seconds'],
                         'prior_report': json.loads((a.out / 'report.json').read_text()) if a.resume and (a.out / 'report.json').exists() else None})
        atomic_json(sessions_path, sessions)
        report['interruptions'] = len(sessions) - 1
        report['started_from_checkpoint_offset'] = state['offset']
        report['uninterrupted_wall_claim_valid'] = not a.resume and len(sessions) == 1
        report['gpu'] = torch.cuda.get_device_name()
        sys.path.insert(0, str(a.repo / 'experiments/dmc-cuda/source/helpers'))
        from state import read_ram
        stop = frame_ends[-1]['generated_token_offset']
        targets = sorted({m['generated_token_offset'] for m in markers} | {stop})
        marker_map = {m['generated_token_offset']: m for m in markers}
        frame_map = {f['generated_token_offset']: f for f in frame_ends}
        last_checkpoint = state['offset']
        post_setup = time.perf_counter()
        report['setup_and_restore_seconds'] = post_setup - PROCESS_START
        trace_digest = hashlib.sha256()
        with trace_path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                trace_digest.update(block)
        with trace_path.open('ab') as trace, chunks_path.open('ab') as chunks, torch.no_grad():
            def checkpoint():
                nonlocal last_checkpoint
                for stream in (trace, chunks):
                    stream.flush()
                    os.fsync(stream.fileno())
                state['chunks_bytes'] = chunks.tell()
                state['trace_sha256'] = trace_digest.hexdigest()
                save_checkpoint(fr, fused, a.out / 'checkpoint.pt', state, surface, coverage)
                last_checkpoint = state['offset']

            if not a.resume:
                checkpoint()
            for target in targets:
                if target <= state['offset']:
                    continue
                while state['offset'] < target:
                    amount = min(8192, target - state['offset'])
                    if a.stop_after_tokens is not None:
                        amount = min(amount, a.stop_after_tokens - state['offset'])
                    if amount <= 0:
                        break
                    begin = state['offset']
                    started = time.perf_counter()
                    generated = fr.run(amount)[0]
                    seconds = time.perf_counter() - started
                    bad = np.flatnonzero(generated != expected[begin:begin + amount])
                    if len(bad):
                        i = int(bad[0])
                        report['mismatches'] = int(len(bad))
                        report['first_mismatch'] = {'position': begin + i, 'actual': int(generated[i]), 'expected': int(expected[begin + i])}
                        raise AssertionError(report['first_mismatch'])
                    raw = generated.astype('<u4').tobytes()
                    trace.write(raw)
                    trace.flush()
                    trace_digest.update(raw)
                    state['offset'] += amount
                    state['chunk_seconds'] += seconds
                    chunk = {'begin_offset': begin, 'end_offset': state['offset'], 'amount': amount, 'seconds': seconds,
                             'compute_seconds_total': state['chunk_seconds'], 'sha256': hashlib.sha256(raw).hexdigest()}
                    chunks.write((json.dumps(chunk) + '\n').encode())
                    chunks.flush()
                    progress = dict(chunk, generated_tokens=state['offset'], total_tokens=stop, mismatches=0,
                                    completed_strips=len(state['strips']), checkpoint_offset=last_checkpoint,
                                    process_wall_seconds=time.perf_counter() - PROCESS_START,
                                    post_setup_wall_seconds=time.perf_counter() - post_setup)
                    atomic_json(a.out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    if state['offset'] - last_checkpoint >= 262144 and state['offset'] != target:
                        checkpoint()
                if state['offset'] != target:
                    break
                if target in marker_map:
                    marker = marker_map[target]
                    assert int(fr.tok[0].item()) == spec.tok('BYTE', 256)
                    ram = read_ram(fr.ex, spec.RAM_BASE, 4 * spec.RAM_WORDS)
                    addresses = manifest['metadata_addresses']
                    d = {k: struct.unpack_from('<I', ram, addresses['qp_strip_' + k] - spec.RAM_BASE)[0]
                         for k in ('ptr', 'x', 'y', 'h', 'n', 'flags', 'serial')}
                    assert d == {k: marker[k] for k in d}, (d, marker)
                    start, count = d['ptr'] - spec.RAM_BASE, d['n'] * d['h']
                    palette = addresses['pump_lut'] - spec.RAM_BASE
                    assert 0 <= start <= len(ram) - count and 0 <= palette <= len(ram) - 512
                    draw_strip(surface, coverage, d, ram[palette:palette + 512], ram[start:start + count])
                    state['strips'].append(dict(d, generated_token_offset=target, frame=1))
                if target in frame_map:
                    end = frame_map[target]
                    assert coverage.all()
                    digest = hashlib.sha256(surface.tobytes()).hexdigest()
                    assert digest == end['frame_sha256'], (digest, end['frame_sha256'])
                    write_frame(surface, a.out / 'frame-0001.png')
                    state['frames'].append({'frame': 1, 'generated_token_offset': target, 'frame_sha256': digest,
                                            'updated_pixels': int(coverage.sum()), 'unique_rgb565_colors': int(len(np.unique(surface)))})
                checkpoint()
            checkpoint()
        report.update(generated_tokens=state['offset'], tokens=state['offset'], frame_hashes=state['frames'],
                      completed_strips=len(state['strips']), retained_chunk_seconds=state['chunk_seconds'],
                      tokens_per_second=state['offset'] / state['chunk_seconds'] if state['chunk_seconds'] else None,
                      microseconds_per_token=1e6 * state['chunk_seconds'] / state['offset'] if state['offset'] else None,
                      actual_trace_sha256=trace_digest.hexdigest(), checkpoint_offset=state['offset'])
        if state['offset'] == stop:
            assert len(state['frames']) == 1 and len(state['strips']) == 25
            assert int(fr.tok[0].item()) == spec.tok('HALT')
            assert trace_digest.hexdigest() == TRACE_SHA
            ram = read_ram(fr.ex, spec.RAM_BASE, 4 * spec.RAM_WORDS)
            (a.out / 'final-gpu-ram.bin').write_bytes(ram)
            assert hashlib.sha256(ram).hexdigest() == FINAL_RAM_SHA
            registers = sys.modules['worker'].read_registers(fr, fused['circuit'])
            atomic_json(a.out / 'actual-registers.json', registers)
            assert registers['regs.rs1'] == legacy['regs']
            report.update(halted=True, final_gpu_ram_sha256=FINAL_RAM_SHA, final_registers_match=True, complete=True)
            report['historical_comparison'] = {
                'retained_chunk_seconds': HISTORICAL_SECONDS,
                'scope': 'Historical first frame on the original circuit, unfused, interrupted and resumed; different network',
                'chunk_seconds_ratio': HISTORICAL_SECONDS / state['chunk_seconds']}
        else:
            report['stop_reason'] = 'requested_token_boundary'
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if 'state' in locals():
            report.update(validated_tokens_this_attempt_end=state['offset'], validated_chunk_seconds=state['chunk_seconds'])
        print(report['traceback'], flush=True)
    finally:
        if 'state' in locals():
            atomic_json(a.out / 'progress.json', {
                'generated_tokens': state['offset'], 'mismatches': report['mismatches'],
                'retained_chunk_seconds': state['chunk_seconds'], 'completed_strips': len(state['strips']),
                'checkpoint_offset': locals().get('last_checkpoint'), 'complete': report['complete'], 'error': report.get('error')})
        report['gpu_end'] = clock_sample()
        report['process_wall_seconds'] = time.perf_counter() - PROCESS_START
        report['post_setup_wall_seconds'] = time.perf_counter() - post_setup if post_setup is not None else None
        valid = report.get('uninterrupted_wall_claim_valid')
        report['uninterrupted_end_to_end_seconds'] = report['process_wall_seconds'] if valid else None
        report['uninterrupted_post_setup_seconds'] = report['post_setup_wall_seconds'] if valid else None
        atomic_json(a.out / 'report.json', report)
        timer.cancel()
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
