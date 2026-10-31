# SPDX-License-Identifier: MIT
"""Locate host-call blocking as the unchanged replay/copy queue fills."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    watchdog = threading.Timer(120, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    sys.path.insert(0, str(a.probe.resolve()))
    import numpy as np
    import torch
    from runtime import build
    fr, expected, spec, _ = build()
    report = {'scope': 'Separate diagnostic of host blocking. Normal self-fed graph and trace copies.', 'runs': []}
    with torch.no_grad():
        for n in (64, 512, 8192):
            fr.load(0, spec.tok('GO'))
            torch.cuda.synchronize()
            out = torch.zeros(1, n, dtype=torch.long, device='cuda')
            begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            replay, trace_copy = [], []
            begin.record()
            for t in range(n):
                t0 = time.perf_counter()
                fr.graph.replay()
                replay.append((time.perf_counter() - t0) * 1e6)
                t0 = time.perf_counter()
                out[:, t].copy_(fr.tok)
                trace_copy.append((time.perf_counter() - t0) * 1e6)
            end.record()
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            sync = time.perf_counter() - t0
            got = out.cpu().numpy()[0]
            bad = int(np.count_nonzero(got != expected[:n]))
            assert bad == 0
            report['runs'].append({'tokens': n, 'mismatches': bad,
                'host_replay_us': replay, 'host_trace_copy_us': trace_copy,
                'host_trace_copy_first64_median_us': statistics.median(trace_copy[:64]),
                'host_trace_copy_last64_median_us': statistics.median(trace_copy[-64:]),
                'host_trace_copy_over100us': sum(t > 100 for t in trace_copy),
                'terminal_sync_seconds': sync, 'device_stream_ms': begin.elapsed_time(end)})
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    watchdog.cancel()


if __name__ == '__main__':
    main()
