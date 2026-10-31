"""Compare isolated state digests and any matching private raw-state snapshots."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--common-prefix', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = {'left': str(a.left), 'right': str(a.right), 'windows': {}}
    for window in ('boot', 'frame2_render'):
        mismatches, first, count = 0, None, 0
        paths = [directory / f'{window}-digests.jsonl' for directory in (a.left, a.right)]
        with paths[0].open() as left, paths[1].open() as right:
            for count, pair in enumerate(itertools.zip_longest(left, right), 1):
                if a.limit and count > a.limit:
                    count = a.limit
                    break
                if not all(pair) and a.common_prefix:
                    count -= 1
                    break
                assert all(pair), ('unequal lengths', count)
                x, y = map(json.loads, pair)
                assert x['token_index'] == y['token_index'] == count
                if x['digest'] != y['digest']:
                    mismatches += 1
                    if first is None:
                        first = {'token': count, 'tensors': [name for name, digest in x['tensors'].items() if y['tensors'].get(name) != digest], 'left_token': x['token'], 'right_token': y['token']}
        result = {'tokens': count, 'digest_mismatches': mismatches, 'first': first, 'files': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
        snapshots = sorted(a.left.glob(f'{window}-token-*.npz'))
        for left in snapshots:
            right = a.right / left.name
            if not right.exists():
                continue
            import numpy as np
            x, y = np.load(left), np.load(right)
            diffs = {}
            for name in ('S', 'dynamic'):
                if x[name].shape != y[name].shape:
                    if name == 'dynamic':
                        width = min(x[name].size, y[name].size)
                        xx, yy = x[name][:width], y[name][:width]
                    else:
                        raise AssertionError(name)
                else:
                    xx, yy = x[name], y[name]
                bad = np.flatnonzero(xx != yy)
                diffs[name] = {'words': len(bad), 'first_words': [{'index': int(i), 'left_bits': int(xx[i]) & 0xffffffff, 'right_bits': int(yy[i]) & 0xffffffff} for i in bad[:64]]}
            result.setdefault('snapshots', {})[left.name] = diffs
        report['windows'][window] = result
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
