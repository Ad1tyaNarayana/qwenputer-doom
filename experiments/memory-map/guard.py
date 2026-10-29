"""Bound a prototype subprocess by wall time and sampled resident memory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--rss-gib', type=float, default=8)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply a command after --')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started, peak, reason = time.monotonic(), 0, None
    with args.log.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            result = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)], capture_output=True, text=True)
            sample = result.stdout.strip()
            if sample:
                peak = max(peak, int(sample) * 1024)
            if time.monotonic() - started > args.seconds:
                reason = 'wall_time_limit'
            if peak > args.rss_gib * 2**30:
                reason = 'sampled_rss_limit'
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(0.5)
    status = {'exit_code': child.returncode, 'stop_reason': reason,
              'peak_sampled_rss_bytes': peak, 'wall_limit_seconds': args.seconds,
              'rss_limit_bytes': int(args.rss_gib * 2**30)}
    args.status.write_text(json.dumps(status, indent=2))
    print(json.dumps(status))
    raise SystemExit(0 if child.returncode == 0 else 1)


if __name__ == '__main__':
    main()
