"""Check historical bounded evidence for the original unpatched prototype."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    evidence = root / 'evidence'
    load = lambda name: json.loads((evidence / name).read_text())
    deps = load('dependencies.json')
    assert len(deps['groups']) == 32
    assert deps['matches_profile_schedule'] and deps['upstream_status'] == ''
    assert all(not g['earlier_drom_output_reads'] for g in deps['groups'])
    fusion_sha = load('equivalence-failure.json')['scripts']['fusion.py']
    names = ['smoke-failure.json', 'smoke-detail.json', 'short-28.json',
             'synchronized-smoke.json', 'equivalence-failure.json',
             'baseline-control-256.json', 'baseline-control-4096.json']
    for name in names:
        r = load(name)
        assert r['scripts']['fusion.py'] == fusion_sha
        assert len(r['controls']) == 6
        assert all(c['mismatched_words'] == 1 for c in r['controls'])
        for w in r['windows'].values():
            if w['state_equal']:
                assert w['mismatched_words'] == 0
                assert w['baseline_reference']['mismatches'] == 0
                assert w['fused_reference']['mismatches'] == 0
            elif 'baseline_token_mismatches' in w:
                assert w['baseline_token_mismatches'] == w['fused_token_mismatches'] == 0
    fail = load('equivalence-failure.json')['windows']['boot']
    smoke = load('smoke-detail.json')['windows']['frame2_render']
    assert fail['first_bad_token'] == 2382 and fail['max_abs_difference'] == 1.0
    assert smoke['first_bad_token'] == 28 and smoke['max_abs_difference'] == 1.0
    control = load('baseline-control-4096.json')
    assert control['baseline_control']
    assert set(control['windows']) == {'boot', 'frame2_render'}
    assert all(w['tokens_checked'] == 4096 and w['state_equal']
               for w in control['windows'].values())
    result = {'phase_a': 'independent head math; ordered additive residual and history updates',
              'accepted': False, 'timing_ran': False, 'full_65536_token_gate_passed': False,
              'boot_first_state_mismatch': fail['first_bad_token'],
              'reference_seeded_rendering_first_state_mismatch': smoke['first_bad_token'],
              'maximum_observed_abs_difference': 1.0,
              'numerical_root_cause': 'not isolated',
              'upstream_modified': False, 'fusion_sha256': fusion_sha,
              'files': {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
                        for name in ['dependencies.json', 'reset-check.json', *names]},
              'current_scripts': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ('analyze.py', 'fusion.py', 'run_gpu.py')},
              'limits': ['No timing or speedup claim.', 'No full 65536-token state-equivalence pass.',
                         'Frame-2 window is reference-seeded.',
                         'Unchanged-executor determinism control is bounded to 4096 tokens.',
                         'Explicit full reset differs from profile reset setup.']}
    (evidence / 'historical-checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('accepted', 'timing_ran', 'maximum_observed_abs_difference')}))


if __name__ == '__main__':
    main()
