"""Compile existing compact-executor matrices directly from their nonzero entries."""
import torch

from chipformer.q35.ir import FFN, HD, Head
from chipformer.rv.fastexec import Executor, SparseLin


class Entries:
    def __init__(self):
        self.values = {}

    def add(self, lag, dim, column, weight):
        if weight:
            key = lag, dim, column
            self.values[key] = self.values.get(key, 0.0) + float(weight)

    def expression(self, expression, column, lag=0, weight=1):
        for dim, coefficient in expression.terms.items():
            self.add(lag, dim, column, float(coefficient) * weight)

    def assign(self, lag, dim, column, weight):
        self.values[lag, dim, column] = float(weight)

    def channel(self, channel, column):
        for lag, tap in enumerate(channel.taps):
            if tap:
                self.expression(channel.expr, column, lag, tap)

    def linear(self, dimensions, columns, device):
        pairs = [(key, value) for key, value in sorted(self.values.items()) if value]
        linear = object.__new__(SparseLin)
        linear.n, linear.D = columns, dimensions
        linear.lags = sorted({key[0] for key, _ in pairs})
        linear.empty = not pairs
        linear.src = torch.tensor([lag * dimensions + dim for (lag, dim, _), _ in pairs], dtype=torch.long, device=device)
        linear.dst = torch.tensor([column for (_, _, column), _ in pairs], dtype=torch.long, device=device)
        linear.w = torch.tensor([value for _, value in pairs], dtype=torch.float32, device=device)
        linear.nl = max(linear.lags) + 1 if linear.lags else 1
        return linear


class DirectSparseExecutor(Executor):
    """Use upstream step/load/embed semantics with a sparse-only constructor."""
    def __init__(self, circuit, device='cpu', batch=1):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.c, self.dev, self.B = circuit, device, batch
        dimensions = self.D = len(circuit.dims)
        self.steps = []
        grouped = {}
        for node in circuit.nodes:
            grouped.setdefault(node.level, []).append(node)
        for level, nodes in sorted(grouped.items()):
            if isinstance(nodes[0], FFN):
                terms = [(coefficient, a, b, node.out) for node in nodes for coefficient, a, b in node.terms]
                units = len(terms)
                inputs, outputs = Entries(), Entries()
                for unit, (coefficient, a, b, out) in enumerate(terms):
                    inputs.expression(a, unit)
                    inputs.expression(b, units + unit)
                    outputs.add(0, unit, out, coefficient)
                self.steps.append(('ffn', inputs.linear(dimensions, 2 * units, device), units,
                                   outputs.linear(units, dimensions, device)))
                continue
            heads = [node for node in nodes if isinstance(node, Head)]
            value_heads = [(head, value) for head, node in enumerate(heads) for value in node.vheads()]
            h, v = len(heads), len(value_heads)
            decays = [index for index, (_, value) in enumerate(value_heads) if any(value.a.terms.values())]
            nq, nv, nd = h * HD, v * HD, len(decays)
            split = [nq, nq, nv, nd, v, nv]
            qoff, koff, voff, aoff, boff, zoff = 0, nq, 2 * nq, 2 * nq + nv, 2 * nq + nv + nd, 2 * nq + nv + nd + v
            projection, output = Entries(), Entries()
            for index, node in enumerate(heads):
                for channels, offset in ((node.q, qoff), (node.k, koff)):
                    for slot, channel in channels.items():
                        projection.channel(channel, offset + index * HD + slot)
            decay_slots = {index: slot for slot, index in enumerate(decays)}
            constants = []
            for index, (_, value) in enumerate(value_heads):
                for column, channel in value.v.items():
                    projection.channel(channel, voff + index * HD + column)
                if index in decay_slots:
                    projection.expression(value.a, aoff + decay_slots[index])
                constants.append(float(value.beta is None))
                if value.beta is not None:
                    projection.expression(value.beta, boff + index)
                for column, expression in value.z.items():
                    projection.expression(expression, zoff + index * HD + column)
                for column, dim, weight in value.out:
                    output.assign(0, index * HD + column, dim, weight)
            self.steps.append({
                'proj': projection.linear(dimensions, sum(split), device), 'split': split,
                'decays': torch.tensor(decays, dtype=torch.long, device=device),
                'bconst': torch.tensor(constants, dtype=torch.float32, device=device),
                'o': output.linear(nv, dimensions, device),
                'kh': torch.tensor([head for head, _ in value_heads], dtype=torch.long, device=device),
                'H': h, 'V': v, 'heads': heads,
                'qhot': max(getattr(node, 'qhot', 2) for node in heads),
                'S': torch.zeros(batch, v, HD, HD, device=device),
            })
        self.feat_ids = sorted(circuit.token_fns)
        self.bi = torch.arange(batch, device=device)
        self.hist = [None if isinstance(step, tuple) else torch.zeros(batch, 3, dimensions, device=device)
                     for step in self.steps]


def compare_constructors(direct, original):
    """Require identical sparse tensors and metadata from both constructors."""
    assert len(direct.steps) == len(original.steps)
    tensors = 0
    def linear(left, right):
        nonlocal tensors
        for name in ('n', 'D', 'lags', 'empty', 'nl'):
            assert getattr(left, name) == getattr(right, name), name
        for name in ('src', 'dst', 'w'):
            assert torch.equal(getattr(left, name), getattr(right, name)), name
            tensors += 1
    for left, right in zip(direct.steps, original.steps):
        if isinstance(left, tuple):
            assert left[0] == right[0] and left[2] == right[2]
            linear(left[1], right[1])
            linear(left[3], right[3])
        else:
            for name in ('split', 'H', 'V', 'qhot'):
                assert left[name] == right[name], name
            for name in ('decays', 'bconst', 'kh', 'S'):
                assert torch.equal(left[name], right[name]), name
                tensors += 1
            linear(left['proj'], right['proj'])
            linear(left['o'], right['o'])
    return tensors
