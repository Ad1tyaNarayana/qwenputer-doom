"""Batch independent ROM heads while retaining ordered residual/history updates."""
import torch
import triton
import triton.language as tl
from chipformer.rv.tritonexec import _head_kernel


@triton.jit
def _parallel_heads(x, hist, S, y, D: tl.constexpr, V: tl.constexpr,
                    qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                    arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
                    KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr,
                    KZ: tl.constexpr, QHOT: tl.constexpr):
    group = tl.program_id(0) // 32
    _head_kernel(x, hist, S, y + group * D, D, V,
                 qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
                 arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows,
                 KQ, KK, KV, KZ, QHOT)


@triton.jit
def _ordered_apply(x, y, hist, D: tl.constexpr, GROUPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = d < D
    value = tl.load(x + d, mask, other=0.0)
    for group in range(GROUPS):
        h = hist + group * 3 * D + d
        out = y + group * D + d
        h0 = tl.load(h, mask, other=0.0)
        h1 = tl.load(h + D, mask, other=0.0)
        added = tl.load(out, mask, other=0.0)
        # Loads and stores can use different lane layouts across warps, so
        # finish every read of this group's history and output before writing.
        tl.debug_barrier()
        tl.store(h + 2 * D, h1, mask)
        tl.store(h + D, h0, mask)
        tl.store(h, value, mask)
        tl.store(out, 0.0, mask)
        value = value + added
    tl.store(x + d, value, mask)


def install(ex):
    if ex.B != 1:
        raise ValueError('This experiment supports batch one only.')
    original = ex.step
    steps = ex.steps
    indices = [i for i, st in enumerate(steps) if st['kind'] == 'head'
               and all(h.name.startswith('drom') for h in st['heads'])]
    assert indices == [18, *range(20, 51)]
    selected = [steps[i] for i in indices[1:]]
    for st in selected:
        assert st['V'] == 32 and len(st['heads']) == 16
        assert not torch.count_nonzero(st['nrows']).item()
        for name in ('k', 'v', 'a', 'b'):
            assert st[name][0][-1].item() == 0
        for name in ('q', 'z'):
            assert torch.all(st[name][1] < ex.D).item()
        for name in ('KQ', 'KK', 'KV', 'KZ', 'QHOT'):
            assert st[name] == selected[0][name]
    packed = {}
    for name in ('q', 'k', 'v', 'z', 'a', 'b'):
        pointers, indices_, weights = [], [], []
        offset = 0
        for st in selected:
            rp, ix, w = st[name]
            count = int(rp[-1].item())
            pointers.append(rp[:-1] + offset)
            indices_.append(ix[:count])
            weights.append(w[:count])
            offset += count
        pointers.append(torch.tensor([offset], device=ex.dev, dtype=torch.int32))
        ix = torch.cat(indices_)
        w = torch.cat(weights)
        if not ix.numel():
            ix = torch.zeros(1, device=ex.dev, dtype=torch.int32)
            w = torch.zeros(1, device=ex.dev)
        packed[name] = [torch.cat(pointers), ix, w]
    for name in ('bconst', 'odim', 'ow', 'nrows'):
        packed[name] = torch.cat([st[name] for st in selected])
    packed['khead'] = torch.cat([st['khead'] + 16 * g for g, st in enumerate(selected)])
    S = torch.cat([st['S'] for st in selected], dim=1)
    hist = torch.stack([st['hist'][0] for st in selected])
    y = torch.zeros(len(selected), ex.D, device=ex.dev)
    for g, st in enumerate(selected):
        st['S'] = S[:, g * 32:(g + 1) * 32]
        st['hist'] = hist[g:g + 1]
    first = selected[0]

    def fused_step(x):
        ex.steps = steps[:20]
        try:
            original(x)
            _parallel_heads[(len(selected) * 32, 1)](
                x, hist, S, y, ex.D, len(selected) * 32,
                *packed['q'], *packed['k'], *packed['v'], *packed['z'],
                *packed['a'], *packed['b'], packed['bconst'], packed['odim'],
                packed['ow'], packed['khead'], packed['nrows'],
                KQ=first['KQ'], KK=first['KK'], KV=first['KV'], KZ=first['KZ'],
                QHOT=first['QHOT'])
            _ordered_apply[(triton.cdiv(ex.D, 1024),)](
                x, y, hist, ex.D, len(selected), BLOCK=1024)
            ex.steps = steps[51:]
            return original(x)
        finally:
            ex.steps = steps

    ex.step = fused_step
    return {'groups': len(selected), 'value_heads': len(selected) * 32,
            'scratch_y': y, 'packed': packed, 'S': S, 'hist': hist}
