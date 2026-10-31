"""Bounded stock-layer masked-write check, NOT an autonomous CPU prototype."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.upstream/qwenputer/experimental/riscv'))
import torch
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS, Target, evaluate
from chipformer.q35.prims import Circ, thead
from chipformer.rv import hf

torch.set_num_threads(4)
# Each token is a supplied commit packet for this primitive test only.
packets = [(0xFFFFFFFF, 15), (0x12345678, 1), (0x89ABCDEF, 12),
           (0x01020304, 0), (0x01020304, 2), (0, 15), (0x80000001, 15)]
c = Circ(Target(key_heads=16, value_heads=32, intermediate=9216))
data = [c.token_feature(f'd{i}', lambda t, i=i: packets[t][0] >> i & 1) for i in range(32)]
mask = [c.token_feature(f'm{b}', lambda t, b=b: packets[t][1] >> b & 1) for b in range(4)]
outs = []
for pair in range(2):
    vhs = []
    for b in range(2*pair, 2*pair+2):
        vhs.append({'data': {2+i: data[8*b+i] for i in range(8)},
                    'ref': c.one, 'beta': mask[b]})
    result = thead(c, f'bytes{pair}', {0: c.one}, {0: c.one}, vhs)
    outs.extend(bit for vh in result for bit in vh)

toks = list(range(len(packets))) * 2
carry = {}
matrix = evaluate(c, toks, carry=carry, kernel=True)
expected, words = 0, []
for t, (value, mask4) in enumerate(packets * 2):
    for b in range(4):
        if mask4 >> b & 1:
            expected = (expected & ~(255 << (8*b))) | (value & (255 << (8*b)))
    bits = [matrix[t, next(iter(e.terms))] for e in outs]
    got = sum(int(v) << i for i, v in enumerate(bits))
    assert got == expected, (t, got, expected)
    words.append(got)
model, layout, ids = hf.build_model(c, toks)
decode_positions = hf.check(c, model, layout, ids, toks, mode='decode')
prefill_positions = hf.check(c, model, layout, ids, toks, mode='prefill')
report = {'scope': 'supplied masked-write packets, one word, four byte value heads; no CPU loop',
          'packets': len(toks), 'chip_layers': c.n_layers(), 'key_heads_used': 2,
          'value_heads_used': 4, 'configured_key_heads': 16, 'configured_value_heads': 32,
          'hidden_size': layout.hidden, 'dtype': 'bf16 weights and residual, fp32 delta state',
          'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
          'ir_exactness_and_kernel_check': True, 'stock_decode_positions': decode_positions,
          'stock_prefill_positions': prefill_positions, 'words': words}
Path(__file__).with_name('masked-write-result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
