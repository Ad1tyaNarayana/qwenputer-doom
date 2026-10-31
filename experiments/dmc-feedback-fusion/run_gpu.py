"""Bounded full-state comparison and alternating throughput of ROM fusion."""
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['verify', 'bench'], default='verify')
    p.add_argument('--tokens', type=int, default=65536)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--max-seconds', type=int, default=1100)
    p.add_argument('--baseline-control', action='store_true')
    p.add_argument('--chunk-tokens', type=int, default=4096)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    watchdog = threading.Timer(a.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'dmc-rom-fusion'))
    sys.path.insert(0, str(a.probe.resolve()))
    import runtime
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    base, boot_expected, spec, metadata = runtime.build()
    from chipformer.rv import core, elf, ref
    from fusion import install
    from analyze import analyze
    analysis = analyze(base.ex.c)
    expected_analysis = json.loads((a.repo / 'experiments/dmc-rom-fusion/evidence/dependencies.json').read_text())
    assert analysis['groups'] == expected_analysis['groups']
    alt = type(base)(base.c, 1, 'cuda', compile_=False, v2=True)
    fusion = install(alt) if not a.baseline_control else {'scratch_y': torch.zeros(1, device='cuda')}
    boot = ref.Machine.from_program(elf.program(a.probe / 'inputs/doom.elf'))
    firmware = a.repo / '.cache/dmc-continuing/walk-build/doom.elf'
    manifest = json.loads((firmware.parent / 'workload.json').read_text())
    assert sha(firmware) == manifest['elf_sha256']
    sys.path.insert(0, str(a.repo / 'experiments/dmc-rv32'))
    from elf_info import inspect
    symbols = inspect(firmware)['symbols']
    ref.decode = functools.lru_cache(None)(ref.decode)
    m = ref.Machine.from_program(elf.program(firmware))
    offset, frame, render_offset = 0, 1, None
    while m.steps < 6000000:
        if m.n != spec.N_MAX:
            if m.pc == symbols['qp_frame_ready']:
                frame += 1
            if frame == 2 and m.pc == symbols['D_Display']:
                render_offset = offset
            if render_offset is not None and offset >= render_offset + 100000:
                break
        offset += len(m.record())
        assert not m.halted
    assert (offset, m.steps, m.pc) == (6470215, 3966856, 0x63864)
    seed = copy.deepcopy(m)
    expected = []
    while len(expected) < a.tokens:
        expected.extend(m.record())
    fixtures = [('boot', boot, boot_expected[:a.tokens]),
                ('frame2_render', seed, np.asarray(expected[:a.tokens], dtype='<u4'))]
    upstream = a.probe.parents[1] / '.upstream/qwenputer'
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
    for name, digest in expected_analysis['source_sha256'].items():
        path = upstream / 'experimental/riscv/chipformer' / name
        normalized = hashlib.sha256(path.read_text().encode()).hexdigest()
        assert normalized == digest, (name, normalized, digest)
    arithmetic = Path(sys.modules['chipformer.rv.tritonexec'].__file__)
    assert arithmetic.read_bytes() == (upstream / 'experimental/riscv/chipformer/rv/tritonexec.py').read_bytes()
    report = {'mode': a.mode, 'baseline_control': a.baseline_control, 'tokens': a.tokens, 'base': metadata,
              'seed': {'reference_seeded': True, 'token_offset': offset,
                       'instruction_offset': seed.steps, 'pc': seed.pc,
                       'lag_histories': 'reset at reference record boundary'},
              'scripts': {name: sha(Path(__file__).parent / name)
                          for name in ('run_gpu.py', 'fusion.py')},
              'rom_fusion_sha256': sha(Path(__file__).resolve().parents[1] / 'dmc-rom-fusion/fusion.py'),
              'python': sys.version, 'torch': torch.__version__, 'triton': triton.__version__,
              'cuda': torch.version.cuda, 'windows': {}, 'controls': [],
              'gpu_before': subprocess.check_output(['nvidia-smi'], text=True)}
    print(json.dumps({'ready': True, 'mode': a.mode, 'tokens': a.tokens}), flush=True)

    def save():
        (a.out / f'{a.mode}.json').write_text(json.dumps(report, indent=2) + '\n')

    rom = None
    rom_info = None

    def reset(fr, machine):
        core.snapshot(fr.ex.c, machine)
        fr.load(0, spec.tok('GO'))
        fr.ex.reset()
        fr.tok.zero_()
        fr.ex.y.zero_()
        if fr is alt:
            fusion['scratch_y'].zero_()
        if fr is rom:
            rom_info['scratch_y'].zero_()
        torch.cuda.synchronize()

    def check(got, want):
        bad = np.flatnonzero(got != want)
        result = {'tokens': len(got), 'mismatches': len(bad),
                  'sha256': hashlib.sha256(got.astype('<u4').tobytes()).hexdigest()}
        if len(bad):
            result['first'] = int(bad[0])
        assert not len(bad), result
        return result

    reset(alt, boot)
    alt.capture()
    if a.mode == 'verify':
        residuals = [torch.zeros_like(fr.x) for fr in (base, alt)]
        methods = [fr.ex.step for fr in (base, alt)]
        for fr, method, residual in zip((base, alt), methods, residuals):
            def wrapped(x, method=method, residual=residual):
                out = method(x)
                residual.copy_(out)
                return out
            fr.ex.step = wrapped
        pairs = [('x', base.x, alt.x), ('tok', base.tok, alt.tok),
                 ('y', base.ex.y, alt.ex.y), ('residual', *residuals)]
        for i, (left, right) in enumerate(zip(base.ex.steps, alt.ex.steps)):
            if left['kind'] == 'head':
                for name in ('S', 'hist'):
                    pairs.append((f'{name}:{i}', left[name], right[name]))
        assert all(x.is_contiguous() and y.is_contiguous() and x.shape == y.shape
                   for _, x, y in pairs)
        pointers_a, pointers_b, sizes, kinds, tensor_ids = [], [], [], [], []
        block = 4096
        for tensor_id, (_, x, y) in enumerate(pairs):
            item_count = x.numel() * x.element_size() // 4
            for begin in range(0, item_count, block):
                pointers_a.append(x.data_ptr() + 4 * begin)
                pointers_b.append(y.data_ptr() + 4 * begin)
                sizes.append(min(block, item_count - begin))
                kinds.append(x.dtype == torch.int64)
                tensor_ids.append(tensor_id)
        ptr_a = torch.tensor(pointers_a, device='cuda', dtype=torch.uint64)
        ptr_b = torch.tensor(pointers_b, device='cuda', dtype=torch.uint64)
        ns = torch.tensor(sizes, device='cuda', dtype=torch.int32)
        is_int = torch.tensor(kinds, device='cuda', dtype=torch.int32)
        tensor_ids = torch.tensor(tensor_ids, device='cuda', dtype=torch.int32)
        by_tensor = torch.zeros(a.tokens, len(pairs), device='cuda', dtype=torch.int32)
        counts = torch.zeros(a.tokens, device='cuda', dtype=torch.int32)
        differences = torch.zeros(a.tokens, device='cuda')
        position = torch.zeros(1, device='cuda', dtype=torch.int32)
        tokens = torch.zeros(2, a.tokens, device='cuda', dtype=torch.int64)

        @triton.jit
        def compare(pa, pb, ns, is_int, ids, position, counts, differences,
                    by_tensor, TENSORS: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            d = tl.arange(0, BLOCK)
            n = tl.load(ns + pid)
            ap = tl.load(pa + pid).to(tl.pointer_type(tl.int32))
            bp = tl.load(pb + pid).to(tl.pointer_type(tl.int32))
            av = tl.load(ap + d, d < n, other=0)
            bv = tl.load(bp + d, d < n, other=0)
            mismatches = tl.sum((av != bv).to(tl.int32), 0)
            if mismatches > 0:
                t = tl.load(position)
                tl.atomic_add(counts + t, mismatches)
                tl.atomic_add(by_tensor + t * TENSORS + tl.load(ids + pid), mismatches)
                diff = tl.abs(av.to(tl.float32, bitcast=True) - bv.to(tl.float32, bitcast=True))
                diff = tl.where((d < n) & (tl.load(is_int + pid) == 0), diff, 0.0)
                tl.atomic_max(differences + t, tl.max(diff, 0))

        @triton.jit
        def advance(position, tokens, left, right, N: tl.constexpr):
            t = tl.load(position)
            tl.store(tokens + t, tl.load(left))
            tl.store(tokens + N + t, tl.load(right))
            tl.store(position, t + 1)

        def compare_now():
            compare[(len(sizes),)](ptr_a, ptr_b, ns, is_int, tensor_ids, position, counts,
                                   differences, by_tensor, len(pairs), BLOCK=block)

        reset(base, boot)
        reset(alt, boot)
        compare_now()
        assert counts[0].item() == 0, [name for name, left, right in pairs
                                      if not torch.equal(left.view(torch.int32), right.view(torch.int32))]
        for category in ('S', 'hist', 'x', 'tok', 'y', 'residual'):
            name, left, right = next(p for p in pairs if p[0].split(':')[0] == category)
            saved = right.flatten()[0].clone()
            right.view(torch.int32).flatten()[0].bitwise_xor_(1)
            counts.zero_()
            differences.zero_()
            by_tensor.zero_()
            compare_now()
            mismatches = counts[0].item()
            assert mismatches > 0, category
            report['controls'].append({'mutation': name, 'mismatched_words': mismatches})
            right.flatten()[0].copy_(saved)
        counts.zero_()
        differences.zero_()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            report['capture_stream'] = int(torch.cuda.current_stream().cuda_stream)
            base._step()
            alt._step()
            compare_now()
            advance[(1,)](position, tokens, base.tok, alt.tok, a.tokens)
        from audit_gpu import graph_topology
        report['graph_topology'] = graph_topology(graph)
        for name, machine, want in fixtures:
            reset(base, machine)
            reset(alt, machine)
            position.zero_()
            counts.zero_()
            differences.zero_()
            by_tensor.zero_()
            started = time.perf_counter()
            for begin in range(0, a.tokens, a.chunk_tokens):
                for _ in range(min(a.chunk_tokens, a.tokens - begin)):
                    graph.replay()
                torch.cuda.synchronize()
                mismatch_counts = counts.cpu().numpy()
                bad = np.flatnonzero(mismatch_counts)
                if len(bad):
                    report['windows'][name] = {'state_equal': False,
                        'first_bad_token': int(bad[0]) + 1,
                        'mismatched_words': int(mismatch_counts.sum()),
                        'max_abs_difference': float(differences.max().item())}
                    first_counts = by_tensor[int(bad[0])].cpu().tolist()
                    report['windows'][name]['first_token_differences'] = [
                        {'tensor': pairs[i][0], 'mismatched_words': count}
                        for i, count in enumerate(first_counts) if count]
                    report['windows'][name]['end_of_chunk_differences'] = [
                        {'tensor': label,
                         'mismatched_words': int(torch.count_nonzero(left.view(torch.int32) != right.view(torch.int32)).item()),
                         'max_abs_difference': float((left - right).abs().max().item()),
                         'first_flat_word_indices': torch.nonzero((left.view(torch.int32) != right.view(torch.int32)).flatten())[:16, 0].cpu().tolist()}
                        for label, left, right in pairs
                        if not torch.equal(left.view(torch.int32), right.view(torch.int32))]
                    got = tokens[:, :min(begin + a.chunk_tokens, a.tokens)].cpu().numpy()
                    report['windows'][name]['baseline_token_mismatches'] = int(np.count_nonzero(got[0] != want[:got.shape[1]]))
                    report['windows'][name]['fused_token_mismatches'] = int(np.count_nonzero(got[1] != want[:got.shape[1]]))
                    save()
                    raise AssertionError(report['windows'][name])
                if a.chunk_tokens > 1 or begin % 256 == 0:
                    print(json.dumps({'verify': name, 'checked': min(begin + a.chunk_tokens, a.tokens)}), flush=True)
            got = tokens.cpu().numpy()
            report['windows'][name] = {'state_equal': True, 'tokens_checked': a.tokens,
                'mismatched_words': 0, 'max_abs_difference': 0.0,
                'baseline_reference': check(got[0], want),
                'fused_reference': check(got[1], want),
                'verification_seconds': time.perf_counter() - started}
            save()
        report['comparison'] = {'state_tensors': len(pairs),
            'bytes_per_executor_per_token': sum(x.numel() * x.element_size() for _, x, _ in pairs),
            'method': 'Every 32-bit word, including both halves of int64 tokens; no hashes or tolerance.',
            'residual': 'Compared before token feedback',
            'scratch': 'Baseline y compared each token; fused private y checked zero after each window.'}
        assert not torch.count_nonzero(fusion['scratch_y']).item()
    else:
        from fusion import _rom
        rom = type(base)(base.c, 1, 'cuda', compile_=False, v2=True)
        rom_info = _rom.install(rom.ex)
        rom.capture()
        acceptance = json.loads((a.out / 'verify.json').read_text())
        assert not acceptance['baseline_control']
        assert set(acceptance['windows']) == {'boot', 'frame2_render'}
        assert acceptance['tokens'] >= 65536
        assert acceptance['scripts'] == report['scripts']
        assert acceptance['rom_fusion_sha256'] == report['rom_fusion_sha256']
        assert all(w['state_equal'] and w['tokens_checked'] >= 65536
                   for w in acceptance['windows'].values())
        samples = []
        stopped = threading.Event()
        active = {'window': None}

        def sample():
            while not stopped.is_set():
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                samples.append({'window': active['window'], 'values': result.stdout.strip(),
                                'returncode': result.returncode})
                stopped.wait(1)

        threading.Thread(target=sample, daemon=True).start()
        for name, _, _ in fixtures:
            report['windows'][name] = {side: {'runs': []} for side in ('baseline', 'rom', 'fused')}
        for repeat in range(a.repeats):
            for name, machine, want in fixtures:
                sides = [('baseline', base), ('rom', rom), ('fused', alt)]
                if repeat % 2:
                    sides.reverse()
                for side, fr in sides:
                    reset(fr, machine)
                    active['window'] = f'{name}/{side}'
                    start = time.perf_counter()
                    generated = fr.run(a.tokens)[0]
                    seconds = time.perf_counter() - start
                    active['window'] = None
                    result = {'seconds': seconds, 'tokens_per_second': a.tokens / seconds,
                              'match': check(generated, want)}
                    report['windows'][name][side]['runs'].append(result)
                    print(json.dumps({'window': name, 'side': side, 'repeat': repeat, **result}), flush=True)
                    save()
        for name, machine, want in fixtures:
            for side, fr in (('baseline', base), ('rom', rom), ('fused', alt)):
                result = report['windows'][name][side]
                rates = [r['tokens_per_second'] for r in result['runs']]
                result['tokens_per_second_median'] = statistics.median(rates)
                result['tokens_per_second_range'] = [min(rates), max(rates)]
                result['total_us_per_token'] = 1e6 / statistics.median(rates)
                reset(fr, machine)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    generated = fr.run(256)[0]
                result['profile_match'] = check(generated, want[:256])
                path = a.out / f'{name}-{side}-trace.json'
                prof.export_chrome_trace(str(path))
                trace = json.loads(path.read_text())
                kernels = sorted([e for e in trace['traceEvents'] if e.get('cat') == 'kernel'
                                  and any(k in e['name'] for k in ('_head_kernel', '_ffn_kernel', '_apply_kernel', '_parallel_heads', '_ordered_apply'))], key=lambda e: e['ts'])
                pattern = []
                for i, st in enumerate(fr.ex.steps):
                    if side != 'baseline' and 20 <= i <= 50:
                        if i == 20:
                            pattern.extend([('_parallel_heads', 'drom'), ('_ordered_apply', 'drom_apply')])
                        continue
                    kind = '_head_kernel' if st['kind'] == 'head' else '_ffn_kernel'
                    group = 'drom' if st['kind'] == 'head' and all(h.name.startswith('drom') for h in st['heads']) else 'other'
                    pattern.extend([(kind, group), ('_apply_kernel', 'drom_apply' if group == 'drom' else 'other_apply')])
                assert len(kernels) == len(pattern) * 256, (len(kernels), len(pattern))
                sums = {'drom': 0.0, 'drom_apply': 0.0}
                for i, event in enumerate(kernels):
                    kernel, group = pattern[i % len(pattern)]
                    assert kernel in event['name'], (i, kernel, event['name'])
                    if group in sums:
                        sums[group] += event['dur'] / 256
                result['profile'] = {'tokens': 256, 'circuit_kernels_per_token': len(pattern),
                                     'data_rom_us_per_token': sums['drom'],
                                     'data_rom_apply_us_per_token': sums['drom_apply'],
                                     'trace_sha256': sha(path), 'trace_bytes': path.stat().st_size}
                save()
        stopped.set()
        report['samples'] = samples
    report['peak_allocated'] = torch.cuda.max_memory_allocated()
    report['peak_reserved'] = torch.cuda.max_memory_reserved()
    save()
    watchdog.cancel()
    print(json.dumps({'complete': a.mode, 'report': str(a.out / f'{a.mode}.json')}), flush=True)


if __name__ == '__main__':
    main()
