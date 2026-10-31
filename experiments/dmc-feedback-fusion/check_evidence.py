"""Check exact acceptance and three-way five-repeat timing provenance."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    evidence = root / 'evidence'
    rom = root.parent / 'dmc-rom-fusion'
    load = lambda name: json.loads((evidence / name).read_text())
    fusion_hash = sha(root / 'fusion.py')
    rom_hash = sha(rom / 'fusion.py')
    accepted_rom = json.loads((rom / 'evidence/revision-checks.json').read_text())
    assert accepted_rom['accepted'] and accepted_rom['fusion_sha256'] == rom_hash
    race = load('feedback-race.json')
    assert race['scripts']['fusion.py'] == fusion_hash
    assert race['scripts']['feedback_race.py'] == sha(root / 'feedback_race.py')
    assert race['queued_replays'] >= 8192 and race['mismatched_values'] == race['bad_replays'] == 0
    isolated = load('isolation.json')
    comparison = load('isolation-comparison.json')
    baseline = json.loads((rom / 'evidence/revision-base.json').read_text())
    assert isolated['scripts']['fusion.py'] == fusion_hash
    assert isolated['scripts']['isolation.py'] == sha(root / 'isolation.py')
    assert isolated['rom_fusion_sha256'] == rom_hash
    assert isolated['one_bit_recording_control']
    assert set(isolated['windows']) == set(comparison['windows']) == {'boot', 'frame2_render'}
    assert isolated['peak_allocated'] < 8 * 1024**3
    for name, w in isolated['windows'].items():
        assert w['tokens'] >= 65536 and w['reference_mismatches'] == 0
        assert w['first_state_mismatch'] is None and w['final_S_reconstruction_equal']
        c = comparison['windows'][name]
        assert c['tokens'] >= 65536 and c['digest_mismatches'] == 0
        assert c['files'][0]['sha256'] == baseline['windows'][name]['digests_sha256']
        assert c['files'][1]['sha256'] == w['digests_sha256']
    verify, bench = load('verify.json'), load('bench.json')
    for report in (verify, bench):
        assert report['rom_fusion_sha256'] == rom_hash
        assert report['scripts'] == {n: sha(root / n) for n in ('run_gpu.py', 'fusion.py')}
        assert report['tokens'] >= 65536 and not report['baseline_control']
        assert report['peak_allocated'] < 8 * 1024**3
        assert set(report['windows']) == {'boot', 'frame2_render'}
    assert verify['comparison']['state_tensors'] == 86
    assert verify['graph_topology']['nodes'] == 186
    assert len(verify['graph_topology']['edges']) == 185
    assert len(verify['controls']) == 6 and all(c['mismatched_words'] == 1 for c in verify['controls'])
    summary = {}
    for name, w in verify['windows'].items():
        assert w['tokens_checked'] >= 65536 and w['state_equal'] and w['mismatched_words'] == 0
        assert w['baseline_reference']['mismatches'] == w['fused_reference']['mismatches'] == 0
        window = bench['windows'][name]
        assert set(window) == {'baseline', 'rom', 'fused'}
        for side, result in window.items():
            assert len(result['runs']) == 5
            assert all(r['match']['tokens'] >= 65536 and r['match']['mismatches'] == 0 for r in result['runs'])
            assert result['profile_match']['mismatches'] == 0
            assert all(r['match']['sha256'] == w['baseline_reference']['sha256'] for r in result['runs'])
        summary[name] = {side: {k:v for k,v in x.items() if k != 'runs'} for side,x in window.items()}
        summary[name]['cumulative_speedup'] = window['baseline']['total_us_per_token'] / window['fused']['total_us_per_token']
        summary[name]['incremental_speedup'] = window['rom']['total_us_per_token'] / window['fused']['total_us_per_token']
    (evidence / 'timing-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    profile = load('new-profile.json')
    assert profile['scripts'] == bench['scripts']
    assert load('rom-profile.json')['scripts']['fusion.py'] == rom_hash
    assert all(w['feedback_kernels_per_token'] == 1 and w['circuit_kernels_per_token'] == 48 for w in profile['windows'].values())
    result = {'accepted': True, 'tolerance': None, 'upstream_modified': False,
              'scripts': {p.name: sha(p) for p in sorted(root.glob('*.py'))},
              'rom_fusion_sha256': rom_hash,
              'evidence_sha256': {p.name: sha(p) for p in sorted(evidence.glob('*.json')) if p.name != 'checks.json'},
              'timing': summary}
    (evidence / 'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'accepted': True, 'timing': summary}, indent=2))


if __name__ == '__main__':
    main()
