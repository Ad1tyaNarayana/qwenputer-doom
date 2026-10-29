"""Synthesize the launch video soundtrack (no samples, deterministic).

Usage: python tools/soundtrack.py public/soundtrack.wav
Cue times match the scene boundaries in src/theme.ts at 30 fps.
"""
import sys
import wave

import numpy as np

SR = 48000
DUR = 760 / 30
BPM = 120
BEAT = 60 / BPM
t = np.arange(int(SR * DUR)) / SR
out = np.zeros((len(t), 2))

rng = np.random.default_rng(1993)


def env(n, attack, decay):
	a = int(SR * attack)
	e = np.exp(-np.arange(n) / (SR * decay))
	e[:a] *= np.linspace(0, 1, a) if a else 1
	return e


def place(sig, at, gain=1.0, pan=0.0):
	i = int(at * SR)
	j = min(len(out), i + len(sig))
	if i >= len(out):
		return
	s = sig[: j - i] * gain
	out[i:j, 0] += s * (1 - pan / 2)
	out[i:j, 1] += s * (1 + pan / 2)


def kick():
	n = int(SR * 0.45)
	tt = np.arange(n) / SR
	f = 45 + 110 * np.exp(-tt * 28)
	ph = 2 * np.pi * np.cumsum(f) / SR
	return np.tanh(2.2 * np.sin(ph) * env(n, 0.001, 0.16))


def hat(decay=0.03):
	n = int(SR * 0.12)
	noise = rng.standard_normal(n)
	noise = np.diff(noise, prepend=0)  # crude high-pass
	return noise * env(n, 0.0005, decay) * 0.25


def tick():
	n = int(SR * 0.02)
	return rng.standard_normal(n) * env(n, 0.0002, 0.003) * 0.35


def impact(length=2.5):
	n = int(SR * length)
	tt = np.arange(n) / SR
	boom = np.sin(2 * np.pi * (38 + 60 * np.exp(-tt * 9)) * tt) * env(n, 0.001, 0.7)
	crash = rng.standard_normal(n) * env(n, 0.001, 0.35) * 0.5
	return np.tanh(1.8 * (boom + crash))


def riser(length):
	n = int(SR * length)
	tt = np.arange(n) / SR
	noise = rng.standard_normal(n) * (tt / length) ** 2 * 0.35
	tone = np.sin(2 * np.pi * np.cumsum(200 + 1400 * (tt / length) ** 2) / SR) * (tt / length) ** 3 * 0.2
	return noise + tone


def bass_note(freq, length, drive=3.0):
	n = int(SR * length)
	tt = np.arange(n) / SR
	saw = 2 * ((freq * tt) % 1) - 1
	sub = np.sin(2 * np.pi * freq / 2 * tt)
	return np.tanh(drive * (0.6 * saw + 0.6 * sub)) * env(n, 0.004, length * 0.6) * 0.45


# Scene starts in seconds (keep in sync with S in src/theme.ts)
QWEN, PIPE, FOOT, FRAME, END = 60 / 30, 150 / 30, 255 / 30, 555 / 30, 640 / 30

# hook: keyboard ticks, hit on "NOT BY ASKING IT."
for k in range(30):
	place(tick(), 2 / 30 + k / 42 + rng.uniform(0, 0.008), 0.8, rng.uniform(-0.4, 0.4))
place(impact(1.4), 30 / 30, 0.8)

# Low drone under everything until the end card
drone = np.tanh(2 * (np.sin(2 * np.pi * 41.2 * t) + 0.3 * np.sin(2 * np.pi * 61.7 * t)))
drone *= np.clip((t - QWEN + 0.2) / 1.0, 0, 1) * np.clip((END - t) / 0.3, 0, 1) * 0.10
out += drone[:, None]

# qwenputer + pipeline: kick on quarters, hats on off-beats
for b in np.arange(QWEN, FOOT - 1e-6, BEAT):
	place(kick(), b, 0.9)
	place(hat(), b + BEAT / 2, 0.8, 0.3)
for k in range(5):  # pipeline cards pop in
	place(tick(), PIPE + (14 + k * 11) / 30, 1.2)
place(riser(2.0), FOOT - 2.0, 0.9)

# footage: driving 8th-note bass line, full beat
riff = [41.2, 41.2, 49.0, 41.2, 55.0, 41.2, 49.0, 46.2]
for i, b in enumerate(np.arange(FOOT, FRAME - 1e-6, BEAT / 2)):
	place(bass_note(riff[i % len(riff)], BEAT / 2), b, 0.9)
	if i % 2 == 0:
		place(kick(), b, 1.0)
	place(hat(0.02 if i % 2 else 0.04), b + BEAT / 4, 0.7, -0.3 if i % 2 else 0.3)
place(impact(2.0), FOOT, 0.9)
place(riser(1.5), FRAME - 1.5, 1.0)

# final frame: big impact with stat pops
place(impact(2.5), FRAME, 1.1)
for d in (4, 12, 20, 28):
	place(tick(), FRAME + d / 30, 1.4)

# end card: last hit, tail rings out
place(impact(4.0), END, 1.0)

# Master: gentle saturation, normalize, fade
out = np.tanh(out * 1.2)
out /= np.max(np.abs(out)) / 0.7
out *= np.clip((DUR - t) / 1.2, 0, 1)[:, None]

with wave.open(sys.argv[1], 'wb') as w:
	w.setnchannels(2)
	w.setsampwidth(2)
	w.setframerate(SR)
	w.writeframes((out * 32767).astype('<i2').tobytes())
