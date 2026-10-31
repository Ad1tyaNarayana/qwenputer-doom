# SPDX-License-Identifier: MIT
"""Export hashes and checkpoint metadata, never checkpoint tensors or guest bytes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    upstream = a.work / '.upstream/qwenputer'
    probe = a.work / 'probes/dmc-frame-20261029'
    paths = [upstream / 'experimental/riscv/chipformer/rv/tritonexec.py',
             upstream / 'experimental/riscv/chipformer/rv/fastexec.py',
             probe / 'scratch/chipformer/rv/tritonexec.py',
             probe / 'runtime.py', probe / 'helpers/prepare.py',
             probe / 'inputs/doom.elf', probe / 'inputs/reference.tokens.u32']
    report = {'files': {str(path): {'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths},
              'upstream_status': subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True)}
    assert paths[0].read_bytes() == paths[2].read_bytes()
    import torch
    properties = torch.cuda.get_device_properties(0)
    report['gpu_properties'] = {'name': properties.name, 'multiprocessors': properties.multi_processor_count,
                                'total_memory': properties.total_memory}
    report['triton_windows_distribution'] = importlib.metadata.version('triton-windows')
    checkpoint_path = probe / 'private-run/checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    report['existing_checkpoint'] = {
        'path': str(checkpoint_path), 'bytes': checkpoint_path.stat().st_size,
        'elf_sha256': checkpoint['elf_sha256'], 'offset': checkpoint['offset'],
        'use_for_walk': False,
        'reason': 'Legacy first-frame ELF differs from the continuing walk ELF.'}
    report['processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv'], text=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
