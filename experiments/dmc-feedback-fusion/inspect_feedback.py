"""Inspect feedback tensor geometry and exact embedding reduction constraints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

p = argparse.ArgumentParser()
p.add_argument('--probe', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
timer = threading.Timer(120, lambda: os._exit(124))
timer.daemon = True
timer.start()
sys.path.insert(0, str(a.probe))
import runtime
import torch
fr, _, _, _ = runtime.build()
bits = fr.bits.cpu()
base = fr.base.cpu()
report = {'D': fr.ex.D, 'families': len(fr.ids), 'payload_bits': len(fr.pay_w),
          'shapes': {name: list(getattr(fr, name).shape) for name in ('fam_idx','pay_idx','base','bits','ids','small','row0','table')},
          'max_nonzero_bit_terms_per_coordinate': int((bits != 0).sum(1).max()),
          'bit_values': torch.unique(bits).tolist(),
          'base_values': torch.unique(base).tolist(),
          'small_families': fr.small.cpu().tolist(),
          'runtime_sha256': hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()}
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
timer.cancel()
