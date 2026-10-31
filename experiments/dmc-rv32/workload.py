# SPDX-License-Identifier: MIT
"""Build-time external key states, consumed by guest I_StartTic."""
import json

KEYS = ('forward', 'back', 'left', 'right', 'fire', 'use', 'run',
        'strafe_left', 'strafe_right')


def configure(frames, schedule_path, out):
    if not 1 <= frames <= 1000000:
        raise ValueError('--frames must be between 1 and 1000000')
    schedule = json.loads(schedule_path.read_text()) if schedule_path else []
    if not isinstance(schedule, list):
        raise ValueError('Input schedule must be a list of {tic, keys} states')
    rows = []
    previous = -1
    for state in schedule:
        if not isinstance(state, dict) or set(state) != {'tic', 'keys'}:
            raise ValueError('Each input state must contain exactly tic and keys')
        tic, keys = state['tic'], state['keys']
        if type(tic) is not int or not previous < tic <= 2000000:
            raise ValueError('Input tics must be nonnegative, strictly increasing integers <= 2000000')
        if not isinstance(keys, list) or any(k not in KEYS for k in keys):
            raise ValueError(f'Input keys must be a list drawn from {KEYS}')
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate input key')
        rows.append((tic, sum(1 << KEYS.index(k) for k in keys)))
        previous = tic
    header = (f'#define QP_DISPLAY_FRAMES {frames}u\n'
              f'#define QP_INPUT_COUNT {len(rows)}u\n'
              'static const struct { unsigned tic, keys; } qp_input[] = {\n' +
              ''.join(f'    {{{tic}u, {keys}u}},\n' for tic, keys in rows or [(0, 0)]) +
              '};\n')
    (out / 'qp_workload.h').write_text(header)
    (out / 'workload.json').write_text(json.dumps({
        'mode': 'continuing', 'frames': frames, 'simulation_hz': 35,
        'display_hz': 24, 'boot_tics': 1, 'input': schedule,
    }, indent=2) + '\n')
