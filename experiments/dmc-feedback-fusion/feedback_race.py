"""Stress exact standalone feedback under thousands of queued graph replays."""
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
    timer = threading.Timer(120, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    sys.path.insert(0, str(a.probe))
    import runtime
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    left, _, _, _ = runtime.build()
    from fusion import install_feedback
    right = type(left)(left.c, 1, 'cuda', compile_=False, v2=True)
    left.ex.step = lambda x: x
    right.ex.step = lambda x: x
    install_feedback(right)
    rng = np.random.default_rng(20261029)
    data = np.zeros((a.tokens, left.ex.D), dtype=np.float32)
    families = np.arange(a.tokens, dtype=np.int64) % 32
    payloads = rng.integers(0, 65536, a.tokens)
    for j, d in enumerate(left.fam_idx.cpu().tolist()):
        data[:, d] = ((families >> j) & 1)
    for j, d in enumerate(left.pay_idx.cpu().tolist()):
        data[:, d] = ((payloads >> j) & 1)
    data[::7, left.pay_idx.cpu().numpy()] = np.resize(np.array([-1.5,-0.5,0.5,1.5],dtype=np.float32), (len(data[::7]),16))
    inputs = torch.tensor(data,device='cuda')
    position = torch.zeros(1,device='cuda',dtype=torch.int32)
    counts = torch.zeros(a.tokens,device='cuda',dtype=torch.int32)

    @triton.jit
    def prepare(inputs, pos, left, right, D: tl.constexpr, BLOCK: tl.constexpr):
        d = tl.arange(0,BLOCK)
        v = tl.load(inputs + tl.load(pos)*D + d,d<D,other=0.0)
        tl.store(left+d,v,d<D)
        tl.store(right+d,v,d<D)

    @triton.jit
    def compare(left, right, lt, rt, pos, counts, D: tl.constexpr, BLOCK: tl.constexpr):
        d=tl.arange(0,BLOCK)
        a=tl.load(left+d,d<D,other=0)
        b=tl.load(right+d,d<D,other=0)
        bad=tl.sum((a!=b).to(tl.int32),0)
        bad += (tl.load(lt)!=tl.load(rt)).to(tl.int32)
        t=tl.load(pos)
        tl.store(counts+t,bad)
        tl.store(pos,t+1)

    def step():
        prepare[(1,)](inputs,position,left.x,right.x,left.ex.D,BLOCK=4096)
        left._step()
        right._step()
        compare[(1,)](left.x.view(torch.int32),right.x.view(torch.int32),left.tok,right.tok,position,counts,left.ex.D,BLOCK=4096)
    step()
    torch.cuda.synchronize()
    position.zero_()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    position.zero_()
    counts.zero_()
    for _ in range(a.tokens):
        graph.replay()
    torch.cuda.synchronize()
    bad=counts.cpu().numpy()
    report={'tokens':a.tokens,'queued_replays':a.tokens,'mismatched_values':int(bad.sum()),
            'bad_replays':int(np.count_nonzero(bad)), 'first_bad_replay':int(np.flatnonzero(bad)[0])+1 if np.any(bad) else None,
            'cases':'All 32 family codes, random 16-bit payloads, signed half-integer rounding controls every seventh case; family and row clamp.',
            'peak_allocated':torch.cuda.max_memory_allocated(),
            'scripts':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ('fusion.py','feedback_race.py')}}
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    timer.cancel()
    assert not np.any(bad),report


if __name__=='__main__':
    main()
