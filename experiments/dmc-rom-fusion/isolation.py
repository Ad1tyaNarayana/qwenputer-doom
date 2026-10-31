"""Single-executor queued raw-bit state recording with SHA256 Merkle digests."""
import argparse
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--side', choices=['baseline', 'fused'], required=True)
    p.add_argument('--part', choices=['both', 'heads', 'apply', 'unsafe'], default='both')
    p.add_argument('--tokens', type=int, default=65536)
    p.add_argument('--render-tokens', type=int, default=4096)
    p.add_argument('--chunk-tokens', type=int, default=4096)
    p.add_argument('--max-seconds', type=int, default=1100)
    p.add_argument('--dump-tokens', default='')
    p.add_argument('--compare-dir', type=Path)
    p.add_argument('--dump-token', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    watchdog = threading.Timer(a.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(a.probe.resolve()))
    import runtime
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    base, boot_expected, spec, metadata = runtime.build()
    from chipformer.rv import core, elf, ref
    from fusion import install
    fusion = None
    if a.side == 'fused':
        if a.part == 'both':
            fusion = install(base.ex)
        else:
            from bisect_fusion import install_variant
            fusion = install_variant(base.ex, a.part)
    boot = ref.Machine.from_program(elf.program(a.probe / 'inputs/doom.elf'))
    firmware = a.repo / '.cache/dmc-continuing/walk-build/doom.elf'
    manifest = json.loads((firmware.parent / 'workload.json').read_text())
    assert sha(firmware) == manifest['elf_sha256']
    sys.path.insert(0, str(a.repo / 'experiments/dmc-rv32'))
    from elf_info import inspect
    symbols = inspect(firmware)['symbols']
    ref.decode = functools.lru_cache(None)(ref.decode)
    m = ref.Machine.from_program(elf.program(firmware))
    offset, frame, render_offset = 0, 1, None
    while m.steps < 6000000:
        if m.n != spec.N_MAX:
            if m.pc == symbols['qp_frame_ready']:
                frame += 1
            if frame == 2 and m.pc == symbols['D_Display']:
                render_offset = offset
            if render_offset is not None and offset >= render_offset + 100000:
                break
        offset += len(m.record())
        assert not m.halted
    assert (offset, m.steps, m.pc) == (6470215, 3966856, 0x63864)
    seed = copy.deepcopy(m)
    expected = []
    while len(expected) < a.tokens:
        expected.extend(m.record())
    fixtures = [('boot', boot, boot_expected[:a.tokens]),
                ('frame2_render', seed, np.asarray(expected[:a.tokens], dtype='<u4'))]
    fixtures[1] = ('frame2_render', seed, np.asarray(expected[:a.render_tokens], dtype='<u4'))
    residual = torch.zeros_like(base.x)
    method = base.ex.step

    def step(x):
        out = method(x)
        residual.copy_(out)
        return out

    base.ex.step = step
    states = [(f'S:{i}', st['S']) for i, st in enumerate(base.ex.steps) if st['kind'] == 'head']
    dynamic = [('residual', residual), ('x', base.x), ('tok', base.tok), ('y', base.ex.y)]
    dynamic += [(f'hist:{i}', st['hist']) for i, st in enumerate(base.ex.steps) if st['kind'] == 'head']
    if fusion:
        dynamic.append(('scratch_y', fusion['scratch_y']))
    common = [(name, t) for name, t in dynamic if name != 'scratch_y']
    offsets = np.cumsum([0] + [t.numel() * t.element_size() // 4 for _, t in dynamic])
    snapshot = torch.zeros(a.chunk_tokens, int(offsets[-1]), device='cuda', dtype=torch.int32)
    capacity = 4096
    changes = torch.zeros(a.chunk_tokens, capacity, 2, device='cuda', dtype=torch.int32)
    counts = torch.zeros(a.chunk_tokens, device='cuda', dtype=torch.int32)
    position = torch.zeros(1, device='cuda', dtype=torch.int32)
    previous = [torch.empty_like(t) for _, t in states]
    sizes, pointers, oldptrs, starts = [], [], [], []
    state_offsets = np.cumsum([0] + [t.numel() for _, t in states])
    block = 1024
    for (_, t), old, start in zip(states, previous, state_offsets):
        for begin in range(0, t.numel(), block):
            pointers.append(t.data_ptr() + 4 * begin)
            oldptrs.append(old.data_ptr() + 4 * begin)
            sizes.append(min(block, t.numel() - begin))
            starts.append(int(start) + begin)
    pa = torch.tensor(pointers, device='cuda', dtype=torch.uint64)
    pb = torch.tensor(oldptrs, device='cuda', dtype=torch.uint64)
    ns = torch.tensor(sizes, device='cuda', dtype=torch.int32)
    ix = torch.tensor(starts, device='cuda', dtype=torch.int32)

    @triton.jit
    def delta(pa, pb, ns, ix, pos, counts, changes, CAP: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        d = tl.arange(0, BLOCK)
        n = tl.load(ns + pid)
        ap = tl.load(pa + pid).to(tl.pointer_type(tl.int32))
        bp = tl.load(pb + pid).to(tl.pointer_type(tl.int32))
        av = tl.load(ap + d, d < n, other=0)
        bv = tl.load(bp + d, d < n, other=0)
        changed = (d < n) & (av != bv)
        total = tl.sum(changed.to(tl.int32), 0)
        if total > 0:
            t = tl.load(pos)
            begin = tl.atomic_add(counts + t, total)
            rank = tl.cumsum(changed.to(tl.int32), 0) - 1
            slot = begin + rank
            target = changes + (t * CAP + slot) * 2
            tl.store(target, tl.load(ix + pid) + d, changed & (slot < CAP))
            tl.store(target + 1, av, changed & (slot < CAP))
        tl.store(bp + d, av, d < n)

    @triton.jit
    def capture_dynamic(src, dst, pos, N: tl.constexpr, WIDTH: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
        d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        t = tl.load(pos)
        tl.store(dst + t * WIDTH + OFFSET + d, tl.load(src + d, d < N, other=0), d < N)

    @triton.jit
    def advance(pos):
        tl.store(pos, tl.load(pos) + 1)

    def record():
        delta[(len(sizes),)](pa, pb, ns, ix, position, counts, changes, capacity, BLOCK=block)
        for j, (_, t) in enumerate(dynamic):
            n = t.numel() * t.element_size() // 4
            capture_dynamic[(triton.cdiv(n, block),)](t.view(torch.int32), snapshot, position, n, int(offsets[-1]), int(offsets[j]), BLOCK=block)
        advance[(1,)](position)

    def reset(machine):
        core.snapshot(base.ex.c, machine)
        base.load(0, spec.tok('GO'))
        base.ex.reset()
        base.tok.zero_()
        base.ex.y.zero_()
        residual.zero_()
        if fusion:
            fusion['scratch_y'].zero_()
        for (_, t), old in zip(states, previous):
            old.copy_(t)
        position.zero_()
        counts.zero_()
        torch.cuda.synchronize()

    reset(boot)
    base._step()
    record()
    torch.cuda.synchronize()
    reset(boot)
    states[0][1].view(torch.int32).flatten()[0].bitwise_xor_(1)
    for _, tensor in dynamic:
        tensor.view(torch.int32).flatten()[0].bitwise_xor_(1)
    record()
    torch.cuda.synchronize()
    assert counts[0].item() == 1
    assert changes[0, 0, 0].item() == 0
    assert changes[0, 0, 1].item() == states[0][1].view(torch.int32).flatten()[0].item()
    for j, (_, tensor) in enumerate(dynamic):
        assert torch.equal(snapshot[0, int(offsets[j]):int(offsets[j+1])], tensor.view(torch.int32).flatten())
    states[0][1].view(torch.int32).flatten()[0].bitwise_xor_(1)
    reset(boot)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        base._step()
        record()
    try:
        graph.debug_dump(str(a.out / 'graph.dot'))
        graph_debug = 'graph.dot'
    except RuntimeError as error:
        graph_debug = str(error)
    report = {'side': a.side, 'part': a.part, 'tokens': a.tokens, 'render_tokens': a.render_tokens,
              'chunk_tokens': a.chunk_tokens, 'one_bit_recording_control': True, 'windows': {}, 'base': metadata,
              'scripts': {name: sha(Path(__file__).parent / name) for name in ('isolation.py', 'fusion.py')},
              'method': 'SHA256 Merkle digest over raw int32 bits. Every S word compared to previous token on GPU; changed words update CPU 4096-word leaf hashes. Full raw bytes of every dynamic tensor hashed each token. Root hashes ordered tensor digests. No tolerance or lossy reduction.',
              'state_layout': [{'name': name, 'shape': list(t.shape), 'dtype': str(t.dtype), 'bytes': t.numel()*t.element_size(), 'pointer': t.data_ptr(), 'storage_pointer': t.untyped_storage().data_ptr()} for name, t in states + dynamic],
              'graph_debug': graph_debug, 'stream': int(torch.cuda.current_stream().cuda_stream)}
    leaf_words = 4096
    for window, machine, want in fixtures:
        reset(machine)
        cpu = np.concatenate([t.cpu().numpy().view(np.int32).ravel() for _, t in states])
        leaves = [hashlib.sha256(cpu[j:j+leaf_words].tobytes()).digest() for j in range(0, len(cpu), leaf_words)]
        initial = a.out / f'{window}-initial.npz'
        np.savez_compressed(initial, S=cpu, **{name.replace(':', '_'): t.cpu().numpy() for name, t in dynamic})
        expected_digests = [json.loads(line) for line in (a.compare_dir / f'{window}-digests.jsonl').open()] if a.compare_dir else None
        first_state_mismatch = None
        dump_tokens = {int(value) for value in a.dump_tokens.split(',') if value}
        mismatches = 0
        maximum_delta = 0
        started = time.perf_counter()
        with (a.out / f'{window}-digests.jsonl').open('w') as output:
            for begin in range(0, len(want), a.chunk_tokens):
                amount = min(a.chunk_tokens, len(want) - begin)
                position.zero_()
                counts.zero_()
                for _ in range(amount):
                    graph.replay()
                torch.cuda.synchronize()
                cc = counts[:amount].cpu().numpy()
                assert int(cc.max()) <= capacity, ('delta overflow', int(cc.max()))
                maximum_delta = max(maximum_delta, int(cc.max()))
                dd = changes[:amount, :int(cc.max())].cpu().numpy()
                raw = snapshot[:amount].cpu().numpy()
                for t in range(amount):
                    changed = dd[t, :int(cc[t])]
                    if len(changed):
                        cpu[changed[:, 0]] = changed[:, 1]
                        for leaf in np.unique(changed[:, 0] // leaf_words):
                            start = int(leaf) * leaf_words
                            leaves[int(leaf)] = hashlib.sha256(cpu[start:start+leaf_words].tobytes()).digest()
                    state_digest = hashlib.sha256(b''.join(leaves)).hexdigest()
                    hashes = {'S': state_digest}
                    for j, (name, _) in enumerate(states):
                        hashes[name] = hashlib.sha256(b''.join(leaves[int(state_offsets[j])//leaf_words:int(state_offsets[j+1])//leaf_words])).hexdigest()
                    for j, (name, _) in enumerate(dynamic):
                        hashes[name] = hashlib.sha256(raw[t, offsets[j]:offsets[j+1]].tobytes()).hexdigest()
                    if fusion:
                        assert not np.any(raw[t, offsets[-2]:offsets[-1]]), 'scratch not cleared'
                    tok_index = next(j for j, (name, _) in enumerate(dynamic) if name == 'tok')
                    token = int(raw[t, offsets[tok_index]:offsets[tok_index+1]].view(np.int64)[0])
                    mismatches += int(token != int(want[begin+t]))
                    root = hashlib.sha256(bytes.fromhex(state_digest) + b''.join(bytes.fromhex(hashes[name]) for name, _ in common)).hexdigest()
                    if expected_digests and root != expected_digests[begin+t]['digest'] and first_state_mismatch is None:
                        first_state_mismatch = {'token': begin+t+1, 'tensors': [name for name in expected_digests[begin+t]['tensors'] if hashes.get(name) != expected_digests[begin+t]['tensors'][name]]}
                        np.savez_compressed(a.out / f'{window}-first-mismatch.npz', S=cpu, dynamic=raw[t], offsets=offsets)
                    output.write(json.dumps({'token_index': begin+t+1, 'digest': root, 'tensors': hashes, 'token': token}) + '\n')
                    if begin+t+1 in dump_tokens or (a.dump_token and begin+t+1 in (28, 2382, 3200, a.dump_token)):
                        np.savez_compressed(a.out / f'{window}-token-{begin+t+1}.npz', S=cpu, dynamic=raw[t], offsets=offsets)
                if begin % 4096 == 0:
                    print(json.dumps({'window': window, 'tokens': begin+amount, 'seconds': time.perf_counter()-started, 'max_delta_words': maximum_delta}), flush=True)
        actual = np.concatenate([t.cpu().numpy().view(np.int32).ravel() for _, t in states])
        assert np.array_equal(cpu, actual), 'S delta reconstruction failed'
        report['windows'][window] = {'tokens': len(want), 'reference_mismatches': mismatches, 'first_state_mismatch': first_state_mismatch, 'final_S_reconstruction_equal': True, 'max_delta_words': maximum_delta, 'seconds': time.perf_counter()-started, 'digests_sha256': sha(a.out / f'{window}-digests.jsonl'), 'initial_sha256': sha(initial)}
        (a.out / 'isolation.json').write_text(json.dumps(report, indent=2) + '\n')
    report['peak_allocated'] = torch.cuda.max_memory_allocated()
    (a.out / 'isolation.json').write_text(json.dumps(report, indent=2) + '\n')
    watchdog.cancel()
    print(json.dumps({'complete': True, 'out': str(a.out)}), flush=True)


if __name__ == '__main__':
    main()
