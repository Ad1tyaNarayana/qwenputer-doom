# SPDX-License-Identifier: MIT
"""Observe guest function boundaries without changing instructions or records."""
PHASES = ('startup', 'level_initialization', 'simulation', 'rendering', 'overhead')


def counts():
    return {name: {'instructions': 0, 'tokens': 0} for name in PHASES}


class PhaseProfile:
    def __init__(self, symbols):
        self.entries = {symbols[name]: (name, phase) for name, phase in (
            ('TryRunTics', 'simulation'), ('P_SetupLevel', 'level_initialization'),
            ('D_Display', 'rendering'))}
        self.loop = symbols['D_DoomLoop']
        self.base = 'startup'
        self.stack = []
        self.total = counts()
        self.frame = counts()
        self.tics = []
        self.frame_tics = []

    def boundary(self, pc, ra, gametic, input_keys=None):
        while self.stack and pc == self.stack[-1]['return_pc']:
            context = self.stack.pop()
            if 'tic' in context:
                context['tic']['gametic_after'] = gametic
                context['tic']['guest_key_mask'] = input_keys
        if pc == self.loop:
            self.base = 'overhead'
        if pc in self.entries:
            name, phase = self.entries[pc]
            context = {'name': name, 'phase': phase, 'return_pc': ra}
            if name == 'TryRunTics':
                tic = {'gametic_before': gametic, 'phases': counts()}
                self.tics.append(tic)
                self.frame_tics.append(tic)
                context['tic'] = tic
            self.stack.append(context)

    def record(self, instructions, tokens):
        phase = self.stack[-1]['phase'] if self.stack else self.base
        total, frame = self.total[phase], self.frame[phase]
        total['instructions'] += instructions
        total['tokens'] += tokens
        frame['instructions'] += instructions
        frame['tokens'] += tokens
        for context in self.stack:
            if 'tic' in context:
                tic = context['tic']['phases'][phase]
                tic['instructions'] += instructions
                tic['tokens'] += tokens

    def finish_frame(self):
        result = {'phases': self.frame, 'tics': self.frame_tics}
        self.frame, self.frame_tics = counts(), []
        return result
