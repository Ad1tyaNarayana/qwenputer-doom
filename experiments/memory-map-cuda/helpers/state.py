"""Read compiled CPU RAM bytes directly from compact recurrent state."""
import numpy as np

from chipformer.q35.ir import A_BIT, A_REF, REF_COLS
from chipformer.rv import spec


def read_ram(executor, address, count, batch_row=0):
    """Return bytes from Executor.S without consulting the reference machine or trace."""
    offset = address - spec.RAM_BASE
    if count < 0 or offset < 0 or offset + count > 4 * spec.RAM_WORDS:
        raise ValueError((address, count))
    if not count:
        return b''
    locations = {}
    for step in executor.steps:
        if isinstance(step, tuple) or 'S' not in step:
            continue
        if not any(head.name.startswith('ram') for head in step['heads']):
            continue
        value_offset = 0
        state = step['S'][batch_row].detach().cpu().numpy()
        for head in step['heads']:
            if head.name.startswith('ram'):
                locations[int(head.name[3:])] = state, value_offset
            value_offset += len(head.vheads())
    words = np.arange(offset // 4, (offset + count + 3) // 4, dtype=np.int64)
    heads = (words & 15) | ((words >> 12) << 4)
    order = np.argsort(heads, kind='stable')
    breaks = np.flatnonzero(np.diff(heads[order])) + 1
    values = np.zeros(len(words), dtype='<u4')
    powers = np.left_shift(np.uint64(1), np.arange(32, dtype=np.uint64))
    for positions in np.split(order, breaks):
        word_indices = words[positions]
        state, value_offset = locations[int(heads[positions[0]])]
        rows = state[value_offset + ((word_indices >> 4) & 1), (word_indices >> 5) & 127]
        refs = rows[:, list(REF_COLS)]
        if not np.isin(refs, [0, A_REF]).all():
            raise ValueError('RAM ballast is not a snapshot/write row.')
        bits = (rows[:, 2:34] + rows[:, 34:66]) / A_BIT
        if not np.isin(bits, [0, 1]).all():
            raise ValueError('RAM data columns are not exact bits.')
        values[positions] = (bits.astype(np.uint64) * powers).sum(axis=1).astype('<u4')
    raw = values.tobytes()
    return raw[offset % 4:offset % 4 + count]
