"""Export word indices and counts while retaining raw state differences on the PC."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--fused', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.fused/'isolation.json').read_text())
    layout = [t for t in report['state_layout'] if not t['name'].startswith('S:')]
    result = {'windows': {}}
    for window, window_report in report['windows'].items():
        first = window_report['first_state_mismatch']
        if first is None:
            result['windows'][window] = {'first': None}
            continue
        token = first['token']
        left = np.load(a.baseline/f'{window}-token-{token}.npz')
        right = np.load(a.fused/f'{window}-first-mismatch.npz')
        output = {'token': token, 'S_mismatched_words': int(np.count_nonzero(left['S'] != right['S'])), 'tensors': []}
        private = {}
        for i, tensor in enumerate(layout):
            name = tensor['name']
            if name == 'scratch_y':
                continue
            start, end = map(int, left['offsets'][i:i+2])
            x, y = left['dynamic'][start:end], right['dynamic'][start:end]
            bad = np.flatnonzero(x != y)
            if len(bad):
                private[name] = np.stack([bad, x[bad], y[bad]], axis=1)
                output['tensors'].append({'name': name, 'mismatched_words': len(bad), 'flat_word_indices': bad.tolist(), 'max_abs_difference': float(np.max(np.abs(x.view(np.float32)[bad] - y.view(np.float32)[bad])))})
        np.savez_compressed(a.fused/f'{window}-first-full-diffs.npz', **private)
        result['windows'][window] = output
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
