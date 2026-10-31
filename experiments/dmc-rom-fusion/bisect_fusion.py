"""Separate head batching from ordered apply without changing either kernel."""
import triton
import triton.language as tl
from chipformer.rv.tritonexec import _apply_kernel, _head_kernel
from fusion import install, _parallel_heads, _ordered_apply


def install_variant(ex, part):
    original = ex.step
    info = install(ex)
    steps = ex.steps
    selected = steps[20:51]
    packed = info['packed']
    first = selected[0]
    y, hist, S = info['scratch_y'], info['hist'], info['S']

    def step(x):
        ex.steps = steps[:20]
        try:
            original(x)
            if part in ('heads', 'unsafe'):
                _parallel_heads[(992, 1)](x, hist, S, y, ex.D, 992,
                    *packed['q'], *packed['k'], *packed['v'], *packed['z'],
                    *packed['a'], *packed['b'], packed['bconst'], packed['odim'],
                    packed['ow'], packed['khead'], packed['nrows'],
                    KQ=first['KQ'], KK=first['KK'], KV=first['KV'], KZ=first['KZ'], QHOT=first['QHOT'])
            else:
                for group, st in enumerate(selected):
                    _head_kernel[(32, 1)](x, st['hist'], st['S'], y[group], ex.D, 32,
                        *st['q'], *st['k'], *st['v'], *st['z'], *st['a'], *st['b'],
                        st['bconst'], st['odim'], st['ow'], st['khead'], st['nrows'],
                        KQ=st['KQ'], KK=st['KK'], KV=st['KV'], KZ=st['KZ'], QHOT=st['QHOT'])
            if part in ('apply', 'unsafe'):
                apply = _unsafe_apply if part == 'unsafe' else _ordered_apply
                apply[(triton.cdiv(ex.D, 1024),)](x, y, hist, ex.D, 31, BLOCK=1024)
            else:
                for group, st in enumerate(selected):
                    _apply_kernel[(triton.cdiv(ex.D, 1024), 1)](x, y[group], st['hist'], ex.D, HIST=True, BLOCK=1024)
            ex.steps = steps[51:]
            return original(x)
        finally:
            ex.steps = steps

    ex.step = step
    return info


@triton.jit
def _unsafe_apply(x, y, hist, D: tl.constexpr, GROUPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = d < D
    value = tl.load(x + d, mask, other=0.0)
    for group in range(GROUPS):
        h = hist + group * 3 * D + d
        h0 = tl.load(h, mask, other=0.0)
        h1 = tl.load(h + D, mask, other=0.0)
        tl.store(h + 2 * D, h1, mask)
        tl.store(h + D, h0, mask)
        tl.store(h, value, mask)
        out = y + group * D + d
        value = value + tl.load(out, mask, other=0.0)
        tl.store(out, 0.0, mask)
    tl.store(x + d, value, mask)

