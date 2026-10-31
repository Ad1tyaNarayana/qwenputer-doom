"""Exercise load-before-clear across Triton lane layouts in ordered apply."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=8192)
    a = p.parse_args()
    watchdog = threading.Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(a.probe))
    import runtime
    fr, _, _, _ = runtime.build()
    del fr
    import torch
    import triton
    import triton.language as tl
    from fusion import _ordered_apply
    from bisect_fusion import _unsafe_apply
    D, groups = 2679, 31
    x = torch.zeros(D, device='cuda')
    y = torch.ones(groups, D, device='cuda')
    hist = torch.zeros(groups, 3, D, device='cuda')
    records = torch.zeros(a.tokens, 3, 3, device='cuda', dtype=torch.int32)
    pos = torch.zeros(1, device='cuda', dtype=torch.int32)

    @triton.jit
    def record(x, hist, records, pos, D: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        d = pid * BLOCK + tl.arange(0, BLOCK)
        val = tl.load(x+d, d < D, other=31.0)
        bad = (d < D) & (val != 31.0)
        first = tl.min(tl.where(bad, d, D), 0)
        bits = tl.load(x+first, first < D, other=31.0).to(tl.int32, bitcast=True)
        t = tl.load(pos)
        tl.store(records+t*9+pid*3, tl.sum(bad.to(tl.int32), 0))
        tl.store(records+t*9+pid*3+1, first)
        tl.store(records+t*9+pid*3+2, bits)

    @triton.jit
    def advance(pos):
        tl.store(pos, tl.load(pos)+1)

    report = {'tokens': a.tokens, 'kernels': {}, 'scripts': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('fusion.py', 'bisect_fusion.py', 'apply_race.py')}}
    for name, kernel in [('original', _unsafe_apply), ('barrier', _ordered_apply)]:
        def step():
            x.zero_()
            y.fill_(1.0)
            hist.zero_()
            kernel[(3,)](x, y, hist, D, groups, BLOCK=1024)
            record[(3,)](x, hist, records, pos, D, BLOCK=1024)
            advance[(1,)](pos)
        step()
        torch.cuda.synchronize()
        pos.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        pos.zero_()
        records.zero_()
        for _ in range(a.tokens):
            graph.replay()
        torch.cuda.synchronize()
        raw = records.cpu().numpy()
        import numpy as np
        bad = np.argwhere(raw[:, :, 0] != 0)
        first = None
        if len(bad):
            t, block = map(int, bad[0])
            first = {'token': t+1, 'word': int(raw[t, block, 1]), 'actual_bits': int(raw[t, block, 2]) & 0xffffffff, 'expected_bits': 0x41f80000}
        report['kernels'][name] = {'bad_token_blocks': len(bad), 'mismatched_words': int(raw[:, :, 0].sum()), 'first': first}
        print(json.dumps({name: report['kernels'][name]}), flush=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    watchdog.cancel()


if __name__ == '__main__':
    main()
