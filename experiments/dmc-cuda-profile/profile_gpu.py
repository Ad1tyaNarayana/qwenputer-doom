# SPDX-License-Identifier: MIT
"""Bounded performance diagnostics. Reference seeding is not a correctness proof."""
import argparse
import base64
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=65536)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--diagnostic-tokens', type=int, default=64)
    p.add_argument('--max-seconds', type=int, default=1100)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    watchdog = threading.Timer(a.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(a.probe.resolve()))
    import runtime
    import numpy as np
    import torch
    fr, boot_expected, spec, base = runtime.build()
    from chipformer.rv import core, elf, ref
    ref.decode = functools.lru_cache(None)(ref.decode)
    boot = ref.Machine.from_program(elf.program(a.probe / 'inputs/doom.elf'))
    firmware = a.repo / '.cache/dmc-continuing/walk-build/doom.elf'
    manifest = json.loads((firmware.parent / 'workload.json').read_text())
    assert digest(firmware) == manifest['elf_sha256']
    sys.path.insert(0, str(a.repo / 'experiments/dmc-rv32'))
    from elf_info import inspect
    info = inspect(firmware)
    symbols = info['symbols']
    machine = ref.Machine.from_program(elf.program(firmware))
    frame = 1
    offset = 0
    render_offset = None
    while machine.steps < 6000000:
        if machine.n != spec.N_MAX:
            if machine.pc == symbols['qp_frame_ready']:
                frame += 1
            if frame == 2 and machine.pc == symbols['D_Display']:
                render_offset = offset
            if render_offset is not None and offset >= render_offset + 100000:
                break
        offset += len(machine.record())
        assert not machine.halted
    assert frame == 2 and render_offset is not None
    seed = copy.deepcopy(machine)
    expected = []
    while len(expected) < a.tokens:
        expected.extend(machine.record())
    steady_expected = np.asarray(expected[:a.tokens], dtype='<u4')
    report = {
        'scope': 'Unchanged compact circuit performance only. Frame 2 is reference-seeded.',
        'base': base, 'python': sys.version, 'platform': platform.platform(),
        'cuda_runtime': torch.version.cuda, 'upstream': subprocess.check_output(
            ['git', '-C', str(a.probe.parents[1] / '.upstream/qwenputer'), 'rev-parse', 'HEAD'], text=True).strip(),
        'nsys': shutil.which('nsys'), 'ncu': shutil.which('ncu'),
        'logical_cpus': os.cpu_count(), 'walk_elf_sha256': digest(firmware),
        'seed': {'frame': frame, 'token_offset': offset, 'instruction_offset': seed.steps,
                 'pc': seed.pc, 'render_entry_token_offset': render_offset,
                 'reference_seeded': True, 'lag_history': 'reset by snapshot/load at record boundary'},
        'windows': {}, 'script_sha256': digest(Path(__file__)),
    }
    cpu_command = 'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfLogicalProcessors,LoadPercentage | ConvertTo-Json'
    report['cpu_before'] = subprocess.check_output(
        ['powershell', '-NoProfile', '-EncodedCommand', base64.b64encode(cpu_command.encode('utf-16le')).decode()], text=True)
    try:
        report['system_before'] = subprocess.check_output(['nvidia-smi'], text=True)
    except OSError as e:
        report['system_before'] = repr(e)
    heads = [st for st in fr.ex.steps if st['kind'] == 'head']
    schedule = []
    for i, st in enumerate(fr.ex.steps):
        names = [h.name for h in st.get('heads', [])]
        group = 'other_head'
        if any(n.startswith(('ram', 'rom', 'drom')) for n in names):
            group = 'memory_head'
        if st['kind'] == 'ffn':
            group = 'alu_decode_ffn'
        schedule.append({'index': i, 'kind': st['kind'], 'group': group,
                         'heads': names, 'value_heads': st.get('V'), 'units': st.get('U')})
    report['schedule'] = schedule

    def reset(m):
        core.snapshot(fr.ex.c, m)
        fr.load(0, spec.tok('GO'))
        fr.ex.y.zero_()
        torch.cuda.synchronize()

    def check(got, want):
        bad = np.flatnonzero(got != want[:len(got)])
        if len(bad):
            raise AssertionError({'mismatches': len(bad), 'first': int(bad[0]),
                                  'actual': int(got[bad[0]]), 'expected': int(want[bad[0]])})
        return {'tokens': len(got), 'mismatches': 0,
                'sha256': hashlib.sha256(got.astype('<u4').tobytes()).hexdigest()}

    samples = []
    stopped = threading.Event()
    active = {'window': None}

    def sample_gpu():
        while not stopped.is_set():
            values = subprocess.run([
                'nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'window': active['window'], 'values': values.stdout.strip(), 'returncode': values.returncode})
            stopped.wait(1)

    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    fixtures = [('boot', boot, boot_expected), ('frame2_render', seed, steady_expected)]
    with torch.no_grad():
        for name, m, want in fixtures:
            reset(m)
            report['windows'][name] = {'reference_seeded': name != 'boot', 'runs': []}
        for repeat in range(a.repeats):
            for name, m, want in fixtures:
                reset(m)
                active['window'] = name
                start = time.perf_counter()
                generated = fr.run(a.tokens)[0]
                seconds = time.perf_counter() - start
                active['window'] = None
                match = check(generated, want)
                run = {'seconds': seconds, 'tokens_per_second': a.tokens / seconds, 'match': match}
                report['windows'][name]['runs'].append(run)
                print(json.dumps({'window': name, 'repeat': repeat, **run}), flush=True)
                (a.out / 'progress.json').write_text(json.dumps(report, indent=2))
        for name, m, want in fixtures:
            w = report['windows'][name]
            rates = [r['tokens_per_second'] for r in w['runs']]
            w['rate_median'] = statistics.median(rates)
            w['rate_range'] = [min(rates), max(rates)]
            reset(m)
            n = min(a.tokens, 8192)
            out = torch.zeros(1, n, dtype=torch.long, device='cuda')
            start_ev, end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            host_replay = host_copy = 0.0
            start_ev.record()
            for t in range(n):
                t0 = time.perf_counter()
                fr.graph.replay()
                host_replay += time.perf_counter() - t0
                t0 = time.perf_counter()
                out[:, t].copy_(fr.tok)
                host_copy += time.perf_counter() - t0
            end_ev.record()
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            sync = time.perf_counter() - t0
            t0 = time.perf_counter()
            got = out.cpu().numpy()[0]
            d2h = time.perf_counter() - t0
            w['submission_diagnostic'] = {
                'tokens': n, 'host_graph_replay_seconds': host_replay,
                'host_trace_copy_submission_seconds': host_copy,
                'device_stream_elapsed_ms': start_ev.elapsed_time(end_ev),
                'terminal_sync_wait_seconds': sync, 'trace_d2h_seconds': d2h,
                'match': check(got, want)}
            reset(m)
            n = a.diagnostic_tokens
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    generated = fr.run(n)[0]
                w['profiler_match'] = check(generated, want)
                prof.export_chrome_trace(str(a.out / f'{name}-trace.json'))
                w['profiler_summary'] = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25)
            except Exception as e:
                w['profiler_error'] = repr(e)
            # Events bracket the exact upstream step calls; no arithmetic is copied or replaced.
            reset(m)
            original_step = fr.ex.step
            all_steps = fr.ex.steps
            pairs = [(torch.cuda.Event(enable_timing=True, external=True),
                      torch.cuda.Event(enable_timing=True, external=True)) for _ in all_steps]
            feedback_end = torch.cuda.Event(enable_timing=True, external=True)

            def measured_step(x):
                for st, (begin, end) in zip(all_steps, pairs):
                    begin.record()
                    fr.ex.steps = [st]
                    try:
                        original_step(x)
                    finally:
                        fr.ex.steps = all_steps
                    end.record()
                return x

            fr.ex.step = measured_step
            instrumented = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(instrumented):
                    fr._step()
                    feedback_end.record()
            finally:
                fr.ex.step = original_step
            reset(m)
            times = []
            feedback = []
            generated = []
            for _ in range(n):
                instrumented.replay()
                torch.cuda.synchronize()
                generated.append(int(fr.tok.item()))
                times.append([begin.elapsed_time(end) for begin, end in pairs])
                feedback.append(pairs[-1][1].elapsed_time(feedback_end))
            w['event_instrumented'] = {
                'tokens': n, 'sublayer_mean_ms': np.mean(times, axis=0).tolist(),
                'feedback_mean_ms': statistics.mean(feedback),
                'match': check(np.asarray(generated), want),
                'scope': 'Separate event-instrumented graph, synchronized per token. Events perturb scheduling; not production throughput.'}
            print(json.dumps({'diagnostic_complete': name}), flush=True)
    stopped.set()
    sampler.join(timeout=5)
    report['gpu_samples'] = samples
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    watchdog.cancel()


if __name__ == '__main__':
    main()
