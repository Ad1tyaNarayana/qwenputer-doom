"""Validate exported diagnosis, exact-equivalence gates, and five-run timing evidence."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    evidence = root / 'evidence'
    load = lambda name: json.loads((evidence / name).read_text())
    race = load('revision-apply-race.json')
    assert race['kernels']['original']['mismatched_words'] > 0
    assert race['kernels']['barrier']['mismatched_words'] == 0
    audit = load('diagnosis-audit.json')
    assert not audit['cross_executor_overlaps']
    assert len({x['stream'] for x in audit['launches']}) == 1
    assert audit['comparison_stream'] == audit['launches'][0]['stream']
    assert len(audit['graph_edges']) == audit['graph_nodes'] - 1
    assert audit['provenance']['upstream_status'] == ''
    assert audit['provenance']['arithmetic_matches_upstream']
    assert sum(audit['runtime_build_nonzero_history_words'].values()) == 10833
    assert audit['full_reset_nonzero_history_words'] == 0
    historical = load('equivalence-failure.json')['scripts']['fusion.py']
    source = (root / 'fusion.py').read_text()
    bisect = (root / 'bisect_fusion.py').read_text()
    unsafe = bisect[bisect.index('@triton.jit\ndef _unsafe_apply'):].replace('_unsafe_apply', '_ordered_apply').rstrip()
    original_source = source[:source.index('@triton.jit\ndef _ordered_apply')] + unsafe + source[source.index('\n\n\ndef install'):]
    assert hashlib.sha256(original_source.encode()).hexdigest() == historical
    old = load('diagnosis-isolation-original.json')
    assert old['windows']['boot']['first']['token'] == 3200
    assert old['windows']['frame2_render']['first']['token'] == 1107
    for name in ('diagnosis-baseline-repeat-1.json', 'diagnosis-baseline-repeat-2.json', 'revision-isolation.json'):
        r = load(name)
        assert all(w['digest_mismatches'] == 0 for w in r['windows'].values()), name
    fixed = load('revision-isolation.json')
    assert all(w['tokens'] >= 65536 for w in fixed['windows'].values())
    for side in ('base', 'fused'):
        isolated = load(f'revision-{side}.json')
        assert isolated['scripts']['fusion.py'] == hashlib.sha256((root / 'fusion.py').read_bytes()).hexdigest()
        assert isolated['tokens'] == isolated['render_tokens'] == 65536
        assert isolated['peak_allocated'] < 8 * 1024**3
        assert isolated['one_bit_recording_control']
        for name, window in isolated['windows'].items():
            assert window['tokens'] >= 65536 and window['reference_mismatches'] == 0
            assert window['first_state_mismatch'] is None and window['final_S_reconstruction_equal']
            index = 0 if side == 'base' else 1
            assert fixed['windows'][name]['files'][index]['sha256'] == window['digests_sha256']
    assert race['scripts']['fusion.py'] == hashlib.sha256((root / 'fusion.py').read_bytes()).hexdigest()
    verify = load('revision-verify.json')
    topology = verify['graph_topology']
    assert topology['nodes'] == 210 and len(topology['edges']) == 209
    assert verify['comparison']['state_tensors'] == 86
    assert verify['comparison']['bytes_per_executor_per_token'] == 83073616
    assert verify['tokens'] >= 65536 and not verify['baseline_control']
    assert verify['scripts']['fusion.py'] == hashlib.sha256((root/'fusion.py').read_bytes()).hexdigest()
    assert len(verify['controls']) == 6
    assert all(c['mismatched_words'] == 1 for c in verify['controls'])
    for w in verify['windows'].values():
        assert w['state_equal'] and w['tokens_checked'] >= 65536
        assert w['mismatched_words'] == 0
        assert w['baseline_reference']['mismatches'] == w['fused_reference']['mismatches'] == 0
    bench = load('revision-bench.json')
    assert bench['scripts'] == verify['scripts']
    assert bench['tokens'] == 65536
    for window in bench['windows'].values():
        for result in window.values():
            assert len(result['runs']) == 5
            assert all(r['match']['mismatches'] == 0 for r in result['runs'])
            assert result['profile_match']['mismatches'] == 0
    result = {'root_cause': 'Cross-warp read-before-clear race from different Triton lane layouts in ordered apply.',
              'accepted': True, 'timing_ran': True, 'upstream_modified': False,
              'fusion_sha256': verify['scripts']['fusion.py'],
              'current_scripts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))},
              'evidence_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*evidence.glob('diagnosis-*.json'), *evidence.glob('revision-*.json')]) if p.name not in ('diagnosis-checks.json', 'revision-checks.json')}}
    (evidence/'revision-checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('accepted', 'timing_ran', 'upstream_modified')}))


if __name__ == '__main__':
    main()
