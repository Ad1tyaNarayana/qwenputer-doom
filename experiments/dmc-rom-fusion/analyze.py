"""Rebuild the pinned memory-map graph and export data-ROM dependencies on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


def analyze(c):
    from chipformer.q35.ir import Head
    levels = sorted({n.level for n in c.nodes})
    groups = []
    prior_writes = set()
    for index, level in enumerate(levels):
        nodes = [n for n in c.nodes if n.level == level]
        if not any(n.name.startswith('drom') and isinstance(n, Head) for n in nodes):
            continue
        assert all(isinstance(n, Head) and n.name.startswith('drom') for n in nodes)
        reads = {field: set() for field in ('q', 'k', 'v', 'z', 'a', 'beta')}
        taps = set()
        outputs = set()
        for n in nodes:
            for field in ('q', 'k'):
                for ch in getattr(n, field).values():
                    taps.add(ch.taps)
                    reads[field].update((lag, d) for lag, tap in enumerate(ch.taps) if tap
                                        for d, cf in ch.expr.terms.items() if cf)
            for vh in n.vheads():
                for ch in vh.v.values():
                    taps.add(ch.taps)
                    reads['v'].update((lag, d) for lag, tap in enumerate(ch.taps) if tap
                                      for d, cf in ch.expr.terms.items() if cf)
                for e in vh.z.values():
                    reads['z'].update((0, d) for d, cf in e.terms.items() if cf)
                reads['a'].update((0, d) for d, cf in vh.a.terms.items() if cf)
                if vh.beta is not None:
                    reads['beta'].update((0, d) for d, cf in vh.beta.terms.items() if cf)
                outputs.update(d for _, d, w in vh.out if w)
        current_reads = {d for rs in reads.values() for lag, d in rs if lag == 0}
        groups.append({
            'schedule_index': index, 'level': level, 'heads': [n.name for n in nodes],
            'value_heads': sum(len(n.vheads()) for n in nodes),
            'reads': {field: [{'lag': lag, 'dim': d, 'name': c.dims[d].name,
                               'producer_level': c.dims[d].level} for lag, d in sorted(rs)]
                      for field, rs in reads.items()},
            'taps': [list(t) for t in sorted(taps)], 'output_dims': sorted(outputs),
            'earlier_drom_output_reads': sorted(current_reads & prior_writes),
            'S_shape_per_batch': [sum(len(n.vheads()) for n in nodes), 128, 128],
            'S_writes': [], 'history_shape_per_batch': [3, len(c.dims)],
            'history_update': 'h[2]=old h[1]; h[1]=old h[0]; h[0]=full pre-add residual',
        })
        assert all(not n.k and not n.a.terms and all(not v.a.terms and not v.v
                   for v in n.vheads()) for n in nodes)
        prior_writes.update(outputs)
    assert len(groups) == 32 and sum(len(g['heads']) for g in groups) == 512
    assert all(not g['earlier_drom_output_reads'] for g in groups)
    assert all(not any(r['lag'] for rs in g['reads'].values() for r in rs) for g in groups)
    assert all(g['output_dims'] == groups[0]['output_dims'] for g in groups)
    start, end = groups[0]['schedule_index'], groups[-1]['schedule_index']
    intervening = []
    for index in range(start, end + 1):
        if index not in {g['schedule_index'] for g in groups}:
            ns = [n for n in c.nodes if n.level == levels[index]]
            intervening.append({'schedule_index': index, 'level': levels[index],
                                'nodes': [n.name for n in ns]})
    return {'dims': len(c.dims), 'layers': c.n_layers(), 'groups': groups,
            'intervening': intervening,
            'verdict': 'Head evaluations independent; shared additive outputs and ordered full histories. '
                       'Keep group 0 and intervening FFN unchanged; fuse groups 1..31 with ordered apply.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo / 'experiments/memory-map'))
    from prepare import prepare, REVISION
    upstream = repo / '.upstream/qwenputer'
    with tempfile.TemporaryDirectory(prefix='dmc-rom-analysis-') as scratch:
        manifest = prepare(upstream, Path(scratch), 64, 4096)
        sys.path.insert(0, scratch)
        from chipformer.rv import core
        result = analyze(core.build())
        result.update(upstream=REVISION, python=platform.python_version(), patches=manifest)
    source = upstream / 'experimental/riscv/chipformer'
    result['source_sha256'] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                               for name in ('rv/core.py', 'rv/tritonexec.py', 'rv/fastexec.py', 'q35/ir.py')}
    graph = json.loads((repo / 'experiments/memory-map/evidence/low-memory/graph.json').read_text())
    assert graph['dims'] == result['dims'] and graph['chip_layers'] == result['layers']
    report = json.loads((repo / 'experiments/dmc-cuda-profile/evidence/report.json').read_text())
    assert [s['index'] for s in report['schedule'] if any(h.startswith('drom') for h in s['heads'])] == [g['schedule_index'] for g in result['groups']]
    result['matches_profile_schedule'] = True
    result['upstream_status'] = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['verdict'])


if __name__ == '__main__':
    main()
