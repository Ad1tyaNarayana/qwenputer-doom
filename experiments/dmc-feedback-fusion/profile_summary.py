"""Attribute fresh ROM-fused profiler captures using the original schedule."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--feedback-kernels', type=int, default=25)
    a = p.parse_args()
    bench = json.loads((a.input / 'bench.json').read_text())
    schedule = json.loads(a.schedule.read_text())['schedule']
    pattern = []
    for st in schedule:
        i = st['index']
        if 20 <= i <= 50:
            if i == 20:
                pattern.extend([('_parallel_heads', 'data_rom_heads'), ('_ordered_apply', 'data_rom_apply')])
            continue
        names = st['heads']
        group = 'other_heads'
        for prefix, category in [('drom', 'data_rom_heads'), ('rom', 'code_rom_heads'), ('ram', 'ram_heads')]:
            if names and all(n.startswith(prefix) for n in names):
                group = category
                break
        if st['kind'] == 'ffn':
            group = 'alu_decode_ffn'
        pattern.extend([('_head_kernel' if st['kind'] == 'head' else '_ffn_kernel', group),
                        ('_apply_kernel', 'data_rom_apply' if group == 'data_rom_heads' else 'other_residual_history_apply')])
    result = {'scripts': bench['scripts'], 'scope': 'Separate 256-token profiler captures; shares use captured stream span.', 'expected_feedback_kernels': a.feedback_kernels, 'windows': {}}
    for window in bench['windows']:
        path = a.input / f'{window}-fused-trace.json'
        events = json.loads(path.read_text())['traceEvents']
        kernels = sorted([e for e in events if e.get('cat') == 'kernel'], key=lambda e:e['ts'])
        graph = [e for e in kernels if e['args'].get('graph id', 0)]
        circuit = [e for e in graph if any(k in e['name'] for k, _ in pattern)]
        n = 256
        assert len(circuit) == n * len(pattern)
        groups = defaultdict(float)
        for i, e in enumerate(circuit):
            kernel, category = pattern[i % len(pattern)]
            assert kernel in e['name']
            groups[category] += e['dur'] / n
        feedback = [e for e in graph if e not in circuit]
        assert len(feedback) == a.feedback_kernels * n
        groups['feedback_embedding'] = sum(e['dur'] for e in feedback) / n
        copies = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'DtoD' in e['name']]
        assert len(copies) == n
        groups['trace_copy'] = sum(e['dur'] for e in copies) / n
        span = (max(e['ts'] + e['dur'] for e in copies) - min(e['ts'] for e in graph)) / n
        groups['gaps'] = span - sum(groups.values())
        assert groups['gaps'] >= 0
        result['windows'][window] = {'tokens': n, 'stream_us_per_token': span,
            'circuit_kernels_per_token': len(pattern), 'feedback_kernels_per_token': len(feedback)//n,
            'groups_us_per_token': dict(groups),
            'ranking': [{'group': k, 'us_per_token': v, 'stream_percent': 100*v/span} for k,v in sorted(groups.items(), key=lambda x:-x[1])],
            'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'trace_bytes': path.stat().st_size}
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['windows'],indent=2))


if __name__ == '__main__':
    main()
