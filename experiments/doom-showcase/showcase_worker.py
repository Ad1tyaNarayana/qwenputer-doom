#!/usr/bin/env python3
"""Self-fed, token-checked Doom generation with private atomic checkpoints."""
import time
PROCESS_START = time.perf_counter()
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tiny-llm-ndot'))
from checkpoint_integrity import seal, verify

import threading
sys.dont_write_bytecode = True
import traceback

FRAME_SHA = '6d60d53e30603c046bc61d6fdc1818cc9318f227064ec9b77285b1bda15d0db2'
FINAL_RAM_SHA = 'b9ea6e4ad1a05162f571970604e1a085078d8cf6ae21da48749b47f4dd80b3ef'
ELF_SHA = '9effb70dfe4cad579f0381a657ac3711564da0278a1410f59789e61fa26ca340'
TRACE_SHA = '23d6d7138e2612dfd971f3d86ac3f711c98a0cd2ac90e77410bcc38240edde38'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def replace_shared(temporary, path):
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    replace_shared(temporary, path)


def load_module(name, path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def draw_strip(frame, coverage, descriptor, palette, indices):
    import numpy as np
    d = descriptor
    mul = 2 if d['flags'] & 1 else 1
    assert 0 <= d['x'] < 320 and 0 <= d['y'] < 240 and 0 < d['h'] <= 240 - d['y']
    assert 0 < d['n'] <= 320 and d['x'] + d['n'] * mul <= 320 and d['flags'] in (0, 1)
    assert len(indices) == d['n'] * d['h'] and len(palette) == 512
    values = np.frombuffer(palette, dtype='<u2')[np.frombuffer(indices, dtype='u1')].reshape(d['n'], d['h']).T
    target = (slice(d['y'], d['y'] + d['h']), slice(d['x'], d['x'] + d['n'] * mul))
    assert not coverage[target].any(), 'Overlapping strips within one displayed frame'
    frame[target] = np.repeat(values, mul, axis=1)
    coverage[target] = True


def write_frame(frame, path):
    import numpy as np
    from PIL import Image
    raw_path = path.with_suffix('.rgb565')
    temporary = raw_path.with_suffix('.tmp')
    temporary.write_bytes(frame.astype('<u2').tobytes())
    replace_shared(temporary, raw_path)
    rgb = np.empty((240, 320, 3), dtype='u1')
    for i, shift, mask in ((0, 11, 31), (1, 5, 63), (2, 0, 31)):
        rgb[:, :, i] = ((frame >> shift) & mask) * 255 // mask
    temporary = path.with_suffix('.tmp')
    Image.fromarray(rgb).save(temporary, format='PNG')
    replace_shared(temporary, path)


def validate_manifest(manifest, trace, workload, frames):
    total = trace['tokens']
    assert total > 0 and manifest['elf_sha256']
    addresses = manifest['metadata_addresses']
    for key in ('ptr', 'x', 'y', 'h', 'n', 'flags', 'serial'):
        assert isinstance(addresses['qp_strip_' + key], int)
    assert isinstance(addresses['pump_lut'], int)
    markers = manifest['strips']
    offsets = [m['generated_token_offset'] for m in markers]
    assert offsets == sorted(set(offsets)) and all(0 < o <= total for o in offsets)
    if workload == 'first-frame':
        assert frames == 1 and total == 6342933 and len(markers) == 25
        assert manifest['elf_sha256'] == ELF_SHA and trace['sha256'] == TRACE_SHA
        frame_ends = [{'frame': 1, 'generated_token_offset': total, 'frame_sha256': FRAME_SHA}]
    else:
        frame_ends = manifest['frames'][:frames]
        assert len(frame_ends) == frames and frames >= 1
        previous = 0
        for number, end in enumerate(frame_ends, 1):
            stop = end['generated_token_offset']
            assert end['frame'] == number and previous < stop <= total
            selected = [m for m in markers if previous < m['generated_token_offset'] <= stop]
            assert selected and all(m['frame'] == number for m in selected)
            previous = stop
    return markers, frame_ends


def fusion_acceptance(repo):
    feedback = repo / 'experiments/dmc-feedback-fusion'
    checks = json.loads((feedback / 'evidence/checks.json').read_text())
    verify = json.loads((feedback / 'evidence/verify.json').read_text())
    hashes = {'feedback': sha(feedback / 'fusion.py'),
              'rom': sha(repo / 'experiments/dmc-rom-fusion/fusion.py')}
    assert checks['accepted'] and not checks['upstream_modified']
    assert not verify['baseline_control'] and verify['tokens'] >= 65536
    assert set(verify['windows']) == {'boot', 'frame2_render'}
    for window in verify['windows'].values():
        assert window['state_equal'] and window['tokens_checked'] >= 65536
        assert window['mismatched_words'] == 0
    for evidence in (checks, verify):
        assert hashes['feedback'] == evidence['scripts']['fusion.py']
        assert hashes['rom'] == evidence['rom_fusion_sha256']
    assert sha(feedback / 'evidence/verify.json') == checks['evidence_sha256']['verify.json']
    return hashes


def isolated_runtime(probe, inputs, out, upstream):
    root = out / 'runtime/probe'
    root.mkdir(parents=True)
    shutil.copyfile(probe / 'runtime.py', root / 'runtime.py')
    shutil.copytree(probe / 'helpers', root / 'helpers', symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (root / 'inputs').mkdir()
    for name in ('doom.elf', 'reference.tokens.u32', 'strips.json', 'reference-trace.json'):
        shutil.copyfile(inputs / name, root / 'inputs' / name, follow_symlinks=True)
    runtime = load_module('showcase_runtime', root / 'runtime.py')
    original_prepare = runtime.prepare
    runtime.prepare = lambda ignored, scratch, ram, drom: original_prepare(upstream, scratch, ram, drom)
    return runtime


def heads(fr):
    return [s for s in fr.ex.steps if s['kind'] == 'head']


def save_checkpoint(fr, fusion, path, metadata, surface, coverage):
    import torch
    assert not torch.count_nonzero(fr.ex.y).item()
    assert not torch.count_nonzero(fusion['scratch_y']).item()
    data = {'metadata': metadata, 'x': fr.x.detach().cpu().clone(), 'tok': fr.tok.detach().cpu().clone(),
            'heads': [{k: s[k].detach().cpu().clone() for k in ('S', 'hist')} for s in heads(fr)],
            'surface': torch.from_numpy(surface.astype('int32')), 'coverage': torch.from_numpy(coverage.copy()),
            'scratch_y': fusion['scratch_y'].detach().cpu().clone()}
    seal(data)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_shared(temporary, path)


def restore_checkpoint(fr, fusion, path, identity):
    import torch
    data = torch.load(path, map_location='cpu', weights_only=True)
    verify(data)
    assert data['metadata']['identity'] == identity, 'Checkpoint inputs or executable sources changed'
    assert len(heads(fr)) == len(data['heads'])
    for state, saved in zip(heads(fr), data['heads']):
        for key in ('S', 'hist'):
            assert state[key].shape == saved[key].shape and state[key].dtype == saved[key].dtype
            state[key].copy_(saved[key])
    for key in ('x', 'tok'):
        target = getattr(fr, key)
        assert target.shape == data[key].shape and target.dtype == data[key].dtype
        target.copy_(data[key])
    assert fusion['scratch_y'].shape == data['scratch_y'].shape
    assert not torch.count_nonzero(data['scratch_y']).item()
    fusion['scratch_y'].zero_()
    fr.ex.y.zero_()
    return data['metadata'], data['surface'].numpy().astype('<u2'), data['coverage'].numpy().copy()


def verify_prefix(path, expected, offset):
    import numpy as np
    assert path.stat().st_size >= offset * 4
    digest = hashlib.sha256()
    with path.open('r+b') as stream:
        for begin in range(0, offset, 65536):
            raw = stream.read(min(65536, offset - begin) * 4)
            assert np.array_equal(np.frombuffer(raw, dtype='<u4'), expected[begin:begin + len(raw) // 4])
            digest.update(raw)
        stream.truncate(offset * 4)
    return digest.hexdigest()


def clock_sample():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.sm,utilization.gpu,power.draw,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        return {'values': result.stdout.strip(), 'returncode': result.returncode, 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': repr(error)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'repo', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--inputs', type=Path, help='Walk reference input directory; defaults to probe/inputs')
    p.add_argument('--upstream', type=Path, help='Clean pinned checkout; defaults to probe grandparent/.upstream/qwenputer')
    p.add_argument('--workload', choices=['first-frame', 'walk'], required=True)
    p.add_argument('--frames', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-seconds', type=float, default=14400)
    p.add_argument('--stop-after-tokens', type=int)
    a = p.parse_args()
    if a.max_seconds <= 0 or a.frames < 1 or (a.stop_after_tokens is not None and a.stop_after_tokens < 1):
        p.error('Budgets and frames must be positive')
    a.out = a.out.resolve()
    a.probe = a.probe.resolve()
    assert a.out != a.probe and a.probe not in a.out.parents, 'Output must be separate from the historical probe'
    inputs = (a.inputs or a.probe / 'inputs').resolve()
    upstream = (a.upstream or a.probe.parents[1] / '.upstream/qwenputer').resolve()
    if a.resume:
        assert (a.out / 'checkpoint.pt').is_file()
    else:
        a.out.mkdir(parents=True, exist_ok=False)
    timer = threading.Timer(max(0.01, a.max_seconds - (time.perf_counter() - PROCESS_START)), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    report = {'workload': a.workload, 'requested_frames': a.frames, 'complete': False,
              'reference_tokens_used_as_inputs': False, 'reference_pixels_used': False,
              'reference_state_used_for_resume': False, 'mismatches': 0,
              'timed_work': 'perf_counter around successful fr.run chunks including self-feedback, output copies and CPU transfer',
              'wall_scope': 'Module entry until final report serialization begins; includes setup, capture, comparison, extraction, IO and checkpoints'}
    post_setup = None
    os.environ['TRITON_CACHE_DIR'] = str(a.out / 'kernel-cache/triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(a.out / 'kernel-cache/inductor')
    try:
        import numpy as np
        import torch
        manifest = json.loads((inputs / 'strips.json').read_text())
        trace_info = json.loads((inputs / 'reference-trace.json').read_text())
        markers, frame_ends = validate_manifest(manifest, trace_info, a.workload, a.frames)
        input_hashes = {name: sha(inputs / name) for name in ('doom.elf', 'reference.tokens.u32', 'strips.json', 'reference-trace.json')}
        assert input_hashes['doom.elf'] == manifest['elf_sha256']
        assert input_hashes['reference.tokens.u32'] == trace_info['sha256']
        assert (inputs / 'reference.tokens.u32').stat().st_size == trace_info['tokens'] * 4
        hashes = fusion_acceptance(a.repo)
        source_hashes = {'runtime.py': sha(a.probe / 'runtime.py'), 'worker': sha(Path(__file__))}
        source_hashes.update({'helpers/' + str(path.relative_to(a.probe / 'helpers')): sha(path)
                              for path in sorted((a.probe / 'helpers').rglob('*.py'))})
        identity = {'inputs': input_hashes, 'fusion': hashes, 'sources': source_hashes,
                    'workload': a.workload, 'frames': a.frames}
        report.update(identity)
        acceptance_dir = a.repo / 'experiments/dmc-feedback-fusion/evidence'
        report['fusion_acceptance'] = {
            name: {'sha256': sha(acceptance_dir / name),
                   'feedback_sha256': json.loads((acceptance_dir / name).read_text())['scripts']['fusion.py'],
                   'rom_sha256': json.loads((acceptance_dir / name).read_text())['rom_fusion_sha256']}
            for name in ('checks.json', 'verify.json')}
        if a.resume:
            # Rebuild disposable preparation, never the probe's historical scratch or private-run.
            shutil.rmtree(a.out / 'runtime')
        runtime = isolated_runtime(a.probe, inputs, a.out, upstream)
        runtime.ELF_SHA = manifest['elf_sha256']
        runtime.TRACE_SHA = trace_info['sha256']
        runtime.TOTAL_TOKENS = trace_info['tokens']
        report['gpu_start'] = clock_sample()
        base, expected, spec, base_report = runtime.build()
        report['runtime'] = base_report
        sys.path.insert(0, str(a.repo / 'experiments/dmc-rom-fusion'))
        from analyze import analyze
        dependencies = json.loads((a.repo / 'experiments/dmc-rom-fusion/evidence/dependencies.json').read_text())
        assert analyze(base.ex.c)['groups'] == dependencies['groups']
        assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
        for name, digest in dependencies['source_sha256'].items():
            assert hashlib.sha256((upstream / 'experimental/riscv/chipformer' / name).read_text().encode()).hexdigest() == digest
        arithmetic = Path(sys.modules['chipformer.rv.tritonexec'].__file__)
        assert arithmetic.read_bytes() == (upstream / 'experimental/riscv/chipformer/rv/tritonexec.py').read_bytes()
        fr = type(base)(base.c, 1, 'cuda', compile_=False, v2=True)
        del base
        fusion_module = load_module('showcase_feedback_fusion', a.repo / 'experiments/dmc-feedback-fusion/fusion.py')
        fusion = fusion_module.install(fr)

        def reset():
            fr.load(0, spec.tok('GO'))
            fr.ex.reset()
            fr.tok.zero_()
            fr.ex.y.zero_()
            fusion['scratch_y'].zero_()
            assert all(not torch.count_nonzero(s['hist']).item() for s in heads(fr))
            torch.cuda.synchronize()

        with torch.no_grad():
            reset()
            capture_start = time.perf_counter()
            fr.capture()
            reset()
        report['fused_capture_and_reset_seconds'] = time.perf_counter() - capture_start
        report['setup_seconds'] = time.perf_counter() - PROCESS_START
        surface = np.zeros((240, 320), dtype='<u2')
        coverage = np.zeros((240, 320), dtype=bool)
        state = {'identity': identity, 'offset': 0, 'chunk_seconds': 0.0,
                 'strips': [], 'frames': [], 'chunks_bytes': 0, 'trace_sha256': hashlib.sha256(b'').hexdigest()}
        trace_path = a.out / 'actual.tokens.u32'
        chunks_path = a.out / 'chunks.jsonl'
        if a.resume:
            state, surface, coverage = restore_checkpoint(fr, fusion, a.out / 'checkpoint.pt', identity)
            assert 0 <= state['offset'] <= frame_ends[-1]['generated_token_offset']
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
        from state import read_ram
        stop = frame_ends[-1]['generated_token_offset']
        targets = sorted({m['generated_token_offset'] for m in markers if m['generated_token_offset'] <= stop}
                         | {f['generated_token_offset'] for f in frame_ends})
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
                save_checkpoint(fr, fusion, a.out / 'checkpoint.pt', state, surface, coverage)
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
                    chunk = {'begin_offset': begin, 'end_offset': state['offset'], 'amount': amount,
                             'seconds': seconds, 'compute_seconds_total': state['chunk_seconds'], 'sha256': hashlib.sha256(raw).hexdigest()}
                    chunks.write((json.dumps(chunk) + '\n').encode())
                    chunks.flush()
                    progress = dict(chunk, generated_tokens=state['offset'], total_tokens=stop,
                                    mismatches=0, completed_frames=len(state['frames']),
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
                    state['strips'].append(dict(d, generated_token_offset=target, frame=len(state['frames']) + 1))
                if target in frame_map:
                    end = frame_map[target]
                    assert coverage.all() if end['frame'] == 1 else coverage.sum() >= 64000
                    digest = hashlib.sha256(surface.tobytes()).hexdigest()
                    assert digest == end['frame_sha256'], (end['frame'], digest, end['frame_sha256'])
                    write_frame(surface, a.out / f"frame-{end['frame']:04d}.png")
                    state['frames'].append({'frame': end['frame'], 'generated_token_offset': target,
                                            'frame_sha256': digest, 'updated_pixels': int(coverage.sum())})
                    coverage.fill(False)
                checkpoint()
            checkpoint()
        report.update(generated_tokens=state['offset'], tokens=state['offset'], frame_hashes=state['frames'],
                      completed_strips=len(state['strips']), retained_chunk_seconds=state['chunk_seconds'],
                      tokens_per_second=state['offset'] / state['chunk_seconds'] if state['chunk_seconds'] else None,
                      microseconds_per_token=1e6 * state['chunk_seconds'] / state['offset'] if state['offset'] else None,
                      actual_trace_sha256=trace_digest.hexdigest(), checkpoint_offset=state['offset'])
        if state['offset'] == stop:
            assert len(state['frames']) == a.frames
            if a.workload == 'first-frame':
                assert int(fr.tok[0].item()) == spec.tok('HALT')
                assert trace_digest.hexdigest() == TRACE_SHA
                final_ram_sha = hashlib.sha256(read_ram(fr.ex, spec.RAM_BASE, 4 * spec.RAM_WORDS)).hexdigest()
                assert final_ram_sha == FINAL_RAM_SHA
                report.update(halted=True, final_gpu_ram_sha256=final_ram_sha)
            elif stop == trace_info['tokens']:
                assert trace_digest.hexdigest() == trace_info['sha256']
            report['complete'] = True
        else:
            report['stop_reason'] = 'requested_token_boundary'
        history = json.loads((a.repo / 'experiments/dmc-cuda/evidence/full-report.json').read_text())
        historical = history['compute_seconds_total']
        report['historical_comparison'] = {'retained_chunk_seconds': historical,
            'scope': 'Historical first-frame retained generation only, interrupted and resumed',
            'same_workload': a.workload == 'first-frame',
            'chunk_sum_speedup': historical / state['chunk_seconds'] if report['complete'] and a.workload == 'first-frame' else None,
            'chunk_seconds_delta': state['chunk_seconds'] - historical if report['complete'] and a.workload == 'first-frame' else None}
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if 'state' in locals():
            report.update(validated_tokens_this_attempt_end=state['offset'],
                          validated_chunk_seconds=state['chunk_seconds'],
                          last_checkpoint_offset=locals().get('last_checkpoint'))
        print(report['traceback'], flush=True)
    finally:
        if 'state' in locals():
            atomic_json(a.out / 'progress.json', {
                'generated_tokens': state['offset'], 'mismatches': report['mismatches'],
                'retained_chunk_seconds': state['chunk_seconds'],
                'completed_frames': len(state['frames']), 'completed_strips': len(state['strips']),
                'checkpoint_offset': locals().get('last_checkpoint'),
                'complete': report['complete'], 'error': report.get('error')})
        report['gpu_end'] = clock_sample()
        report['process_wall_seconds'] = time.perf_counter() - PROCESS_START
        report['post_setup_wall_seconds'] = time.perf_counter() - post_setup if post_setup is not None else None
        report['uninterrupted_end_to_end_seconds'] = report['process_wall_seconds'] if report.get('uninterrupted_wall_claim_valid') else None
        report['uninterrupted_post_setup_seconds'] = report['post_setup_wall_seconds'] if report.get('uninterrupted_wall_claim_valid') else None
        atomic_json(a.out / 'report.json', report)
        timer.cancel()
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
