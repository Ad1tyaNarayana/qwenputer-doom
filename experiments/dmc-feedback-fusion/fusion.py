"""Fuse batch-one feedback after the unchanged ROM-fused circuit evaluation."""
import importlib.util
from pathlib import Path
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_rom_path = Path(__file__).resolve().parents[1] / 'dmc-rom-fusion/fusion.py'
_spec = importlib.util.spec_from_file_location('verified_rom_fusion', _rom_path)
_rom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rom)


@triton.jit
def _decode_embed(x, tok, fam_idx, pay_idx, ids, small, row0, table, base, bits,
                  D: tl.constexpr, NF: tl.constexpr, NB: tl.constexpr,
                  NROWS: tl.constexpr, NFB: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, 32)
    fi = tl.load(fam_idx + lane, lane < NFB, other=0)
    family_bits = libdevice.nearbyint(tl.load(x + fi, lane < NFB, other=0.0))
    family = tl.sum(family_bits * (1 << lane).to(tl.float32), 0).to(tl.int64)
    family = tl.minimum(tl.maximum(family, 0), NF - 1)
    pi = tl.load(pay_idx + lane, lane < NB, other=0)
    pay = libdevice.nearbyint(tl.load(x + pi, lane < NB, other=0.0))
    payload = tl.sum(pay * (1 << lane).to(tl.float32), 0).to(tl.int64)
    token = tl.load(ids + family) + payload
    row = tl.minimum(tl.maximum(tl.load(row0 + family) + payload, 0), NROWS - 1)
    is_small = tl.load(small + family)
    d = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0.0, tl.float32)
    for bit in tl.static_range(NB):
        digit = tl.sum(tl.where(lane == bit, pay, 0.0), 0)
        coefficient = tl.load(bits + (family * NB + bit) * D + d, d < D, other=0.0)
        value = value + digit * coefficient
    big = tl.load(base + family * D + d, d < D, other=0.0) + value
    little = tl.load(table + row * D + d, d < D, other=0.0)
    result = tl.where(is_small, little, big)
    tl.debug_barrier()
    tl.store(tok, token)
    tl.store(x + d, result, d < D)


def install_feedback(fr):
    if fr.B != 1:
        raise ValueError('Feedback fusion requires batch one.')
    assert fr.bits.shape == (len(fr.ids), len(fr.pay_w), fr.ex.D)
    assert len(fr.fam_w) <= 32 and len(fr.pay_w) <= 32
    assert torch.all((fr.bits != 0).sum(1) <= 1).item()
    assert torch.all((fr.bits == 0) | (fr.bits == 1) | (fr.bits == 256)).item()
    assert torch.all(fr.fam_w == 2.0 ** torch.arange(len(fr.fam_w), device=fr.dev)).item()
    assert torch.all(fr.pay_w == 2.0 ** torch.arange(len(fr.pay_w), device=fr.dev)).item()

    def feedback():
        _decode_embed[(1,)](fr.x, fr.tok, fr.fam_idx, fr.pay_idx, fr.ids, fr.small,
            fr.row0, fr.table, fr.base, fr.bits, D=fr.ex.D, NF=len(fr.ids),
            NB=len(fr.pay_w), NROWS=len(fr.table), NFB=len(fr.fam_w),
            BLOCK=triton.next_power_of_2(fr.ex.D), enable_fp_fusion=False)

    def step():
        fr.ex.step(fr.x)
        feedback()

    fr._step = step
    fr.graph = None
    return feedback


def install(fr):
    info = _rom.install(fr.ex)
    install_feedback(fr)
    return info
