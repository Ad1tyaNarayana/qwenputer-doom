# SPDX-License-Identifier: MIT
"""Reduce torch.profiler CUDA graph traces without exporting guest state."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.input / 'report.json').read_text())
    summary = {'scope': 'GPU activity attribution from a separate profiled run. Host and device times overlap.',
               'windows': {}}
    for name, window in report['windows'].items():
        path = a.input / f'{name}-trace.json'
        events = json.loads(path.read_text())['traceEvents']
        kernels = sorted([e for e in events if e.get('cat') == 'kernel'], key=lambda e: e['ts'])
        graph = [e for e in kernels if e['args'].get('graph id', 0)]
        circuit = [e for e in graph if e['name'] in ('_head_kernel', '_ffn_kernel', '_apply_kernel')]
        schedule = report['schedule']
        n = window['profiler_match']['tokens']
        assert len(circuit) == n * len(schedule) * 2
        sublayers = []
        groups = defaultdict(float)
        for index, step in enumerate(schedule):
            compute, apply = [], []
            for token in range(n):
                at = (token * len(schedule) + index) * 2
                first, second = circuit[at:at + 2]
                assert first['name'] == ('_head_kernel' if step['kind'] == 'head' else '_ffn_kernel')
                assert second['name'] == '_apply_kernel'
                compute.append(first['dur'])
                apply.append(second['dur'])
            head_names = step['heads']
            category = step['group']
            if head_names and all(h.startswith('drom') for h in head_names):
                category = 'data_rom_heads'
            elif head_names and all(h.startswith('rom') for h in head_names):
                category = 'code_rom_heads'
            elif head_names and all(h.startswith('ram') for h in head_names):
                category = 'ram_heads'
            entry = {**step, 'category': category, 'compute_us_per_token': statistics.mean(compute),
                     'apply_us_per_token': statistics.mean(apply),
                     'occupancy_hint': circuit[index * 2]['args']}
            sublayers.append(entry)
            groups[category] += sum(compute) / n
            groups['apply_residual_history'] += sum(apply) / n
        feedback = [e for e in graph if e['name'] not in ('_head_kernel', '_ffn_kernel', '_apply_kernel')]
        groups['feedback_embedding'] = sum(e['dur'] for e in feedback) / n
        copies = [e for e in events if e.get('cat') == 'gpu_memcpy']
        dto_d = [e for e in copies if 'DtoD' in e['name']]
        dto_h = [e for e in copies if 'DtoH' in e['name']]
        assert len(dto_d) == n and len(dto_h) == 1
        groups['trace_device_copy'] = sum(e['dur'] for e in dto_d) / n
        span = max(e['ts'] + e['dur'] for e in dto_d) - min(e['ts'] for e in graph)
        groups['device_gaps'] = span / n - sum(groups.values())
        assert groups['device_gaps'] >= -0.01
        graph_nodes = len(graph) // n
        assert len(graph) % n == 0
        graph_spans = []
        for i in range(n):
            part = graph[i * graph_nodes:(i + 1) * graph_nodes]
            graph_spans.append(part[-1]['ts'] + part[-1]['dur'] - part[0]['ts'])
        runtime = defaultdict(float)
        for e in events:
            if e.get('cat') == 'cuda_runtime':
                runtime[e['name']] += e.get('dur', 0)
        summary['windows'][name] = {
            'rate_median': window['rate_median'], 'rate_range': window['rate_range'],
            'profiler_tokens': n, 'graph_kernel_nodes_per_token': graph_nodes,
            'graph_replay_span_us_per_token': statistics.mean(graph_spans),
            'device_stream_span_us_per_token': span / n,
            'device_groups_us_per_token': dict(groups),
            'device_groups_percent_of_stream_span': {k: v * n / span * 100 for k, v in groups.items()},
            'trace_d2h_device_us_total': sum(e['dur'] for e in dto_h),
            'host_cuda_runtime_us_total': dict(runtime),
            'sublayers': sublayers,
            'raw_trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'raw_trace_bytes': path.stat().st_size,
            'raw_trace_location': str(path),
            'submission_diagnostic': window['submission_diagnostic'],
        }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'profile-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, w in summary['windows'].items():
        print(name, json.dumps({k: v for k, v in w.items() if k not in ('sublayers', 'submission_diagnostic')}))


if __name__ == '__main__':
    main()
