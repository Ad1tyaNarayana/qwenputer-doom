"""Record allocation ownership and actual capture-stream launch order."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import threading


def graph_topology(graph):
    import ctypes
    driver = ctypes.WinDLL('nvcuda.dll')
    handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    assert driver.cuGraphGetNodes(handle, None, ctypes.byref(count)) == 0
    nodes = (ctypes.c_void_p * count.value)()
    assert driver.cuGraphGetNodes(handle, nodes, ctypes.byref(count)) == 0
    edges = ctypes.c_size_t()
    assert driver.cuGraphGetEdges(handle, None, None, ctypes.byref(edges)) == 0
    sources = (ctypes.c_void_p * edges.value)()
    targets = (ctypes.c_void_p * edges.value)()
    assert driver.cuGraphGetEdges(handle, sources, targets, ctypes.byref(edges)) == 0
    index = {node: i for i, node in enumerate(nodes)}
    graph_edges = [[index[x], index[y]] for x, y in zip(sources, targets)]
    return {'nodes': count.value, 'edges': graph_edges}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    watchdog = threading.Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(a.probe))
    import runtime
    import torch
    base, _, spec, metadata = runtime.build()
    import fusion
    from chipformer.rv import tritonexec
    reset_before = {f'hist:{i}': int(torch.count_nonzero(st['hist']).item()) for i, st in enumerate(base.ex.steps) if st['kind'] == 'head'}
    base.ex.reset()
    reset_after = sum(int(torch.count_nonzero(st['hist']).item()) for st in base.ex.steps if st['kind'] == 'head')
    alt = type(base)(base.c, 1, 'cuda', compile_=False, v2=True)
    packed = fusion.install(alt.ex)
    tensors = []

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            tensors.append({'name': name, 'pointer': value.data_ptr(), 'bytes': value.numel()*value.element_size(), 'storage': value.untyped_storage().data_ptr(), 'storage_bytes': value.untyped_storage().nbytes()})
        elif isinstance(value, dict):
            for key, val in value.items():
                visit(f'{name}.{key}', val)
        elif isinstance(value, (tuple, list)):
            for i, val in enumerate(value):
                visit(f'{name}.{i}', val)

    for name, fr in [('base', base), ('alt', alt)]:
        for key, value in vars(fr).items():
            if key not in ('ex', 'c', 'graph'):
                visit(f'{name}.{key}', value)
        visit(f'{name}.ex.y', fr.ex.y)
        visit(f'{name}.steps', fr.ex.steps)
    visit('fusion', packed)
    cross = []
    for left in [t for t in tensors if t['name'].startswith('base.')]:
        for right in [t for t in tensors if not t['name'].startswith('base.')]:
            if left['pointer'] < right['pointer']+right['bytes'] and right['pointer'] < left['pointer']+left['bytes']:
                cross.append([left['name'], right['name']])
    launches = []
    resources = {}

    class Launch:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def call(*args, **kwargs):
                launches.append({'kernel': self.name, 'stream': int(torch.cuda.current_stream().cuda_stream), 'grid': list(grid)})
                result = launch(*args, **kwargs)
                if self.name in ('_parallel_heads', '_ordered_apply') and self.name not in resources:
                    resources[self.name] = {'metadata': str(result.metadata), 'n_regs': result.n_regs, 'n_spills': result.n_spills}
                    a.out.with_name(self.name + '.ptx').write_text(result.asm['ptx'])
                return result

            return call

    for module, names in [(tritonexec, ('_head_kernel', '_ffn_kernel', '_apply_kernel')), (fusion, ('_parallel_heads', '_ordered_apply'))]:
        for name in names:
            setattr(module, name, Launch(getattr(module, name), name))
    base.load(0, spec.tok('GO'))
    alt.load(0, spec.tok('GO'))
    base.ex.reset()
    alt.ex.reset()
    base._step()
    alt._step()
    torch.cuda.synchronize()
    launches.clear()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        base._step()
        boundary = len(launches)
        alt._step()
        comparison_stream = int(torch.cuda.current_stream().cuda_stream)
        torch.eq(base.x, alt.x)
    topology = graph_topology(graph)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    upstream = a.probe.parents[1] / '.upstream/qwenputer'
    arithmetic = Path(tritonexec.__file__)
    provenance = {'upstream_status': subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True), 'arithmetic_sha256': hashlib.sha256(arithmetic.read_bytes()).hexdigest(), 'arithmetic_matches_upstream': arithmetic.read_bytes() == (upstream / 'experimental/riscv/chipformer/rv/tritonexec.py').read_bytes()}
    report = {'provenance': provenance, 'resources': resources, 'graph_nodes': topology['nodes'], 'graph_edges': topology['edges'], 'cross_executor_overlaps': cross, 'tensors': tensors, 'launches': launches, 'baseline_launches': boundary, 'comparison_stream': comparison_stream, 'runtime_build_nonzero_history_words': reset_before, 'full_reset_nonzero_history_words': reset_after, 'scripts': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('audit_gpu.py', 'fusion.py', 'run_gpu.py')}}
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'cross_executor_overlaps': cross, 'capture_streams': sorted({x['stream'] for x in launches}), 'baseline_launches': boundary, 'fused_launches': len(launches)-boundary, 'stale_history_words': sum(reset_before.values())}), flush=True)
    watchdog.cancel()


if __name__ == '__main__':
    main()
