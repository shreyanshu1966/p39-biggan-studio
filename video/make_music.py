"""Synthesizes the soundtrack (no downloaded audio): 120 BPM, A minor, Am-F-C-G.

python video/make_music.py intro <out.wav>          20 s brag intro: drums, bass, pad, riser + hits on cuts
python video/make_music.py bed <out.wav> <seconds>   soft bed for the demo walkthrough (pad + light pulse)
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 120
BEAT = 60 / BPM
rng = np.random.default_rng(7)


def note(n):  # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G (one per bar)
ROOTS = [45, 41, 36, 43]


def env(n, a, d):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / d)
    return e


def place(buf, sig, t):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def hat():
    n = int(0.06 * SR)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)  # crude high-pass
    x = np.convolve(x, np.ones(4) / 4, mode="same")  # take the fizz off
    return x * np.exp(-np.arange(n) / SR * 90) * 0.08


def clap():
    n = int(0.18 * SR)
    x = rng.standard_normal(n)
    x = np.convolve(x, np.ones(6) / 6, mode="same")
    return x * np.exp(-np.arange(n) / SR * 26) * 0.3


def bass_note(midi, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = note(midi)
    s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    return s * env(n, 0.01, dur * 0.9) * 0.9


def pad_chord(notes, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for m in notes:
        for det in (-0.12, 0.0, 0.12):
            f = note(m + 12) * 2 ** (det / 12)
            s += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    a = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.4).clip(0)
    return s / 9 * a


def pluck(midi, dur=0.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = note(midi)
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.3 + np.sin(2 * np.pi * f * t)
    return s * env(n, 0.003, 0.12) * 0.35


def lowpass(x, alpha):
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # one-pole, fine for short buffers
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def reverb(x, mix=0.25):
    out = x.copy()
    for d, g in ((0.031, 0.5), (0.047, 0.42), (0.071, 0.33), (0.113, 0.25), (0.167, 0.18)):
        k = int(d * SR)
        out[k:] += x[:-k] * g * mix * 2
    return out


def master(x, peak=0.89):
    x = x / (np.abs(x).max() + 1e-9)
    x = np.tanh(x * 1.1) / np.tanh(1.1)  # gentle glue, not a squash
    return x * peak


def write(path, x):
    st = np.stack([x, x], axis=1)
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def intro(dur=20.0):
    n = int(dur * SR)
    drums, bass, pad, lead, fx = (np.zeros(n) for _ in range(5))
    beats = int(dur / BEAT)
    for b in range(beats):
        t = b * BEAT
        bar = int(t // 2) % 4
        if t >= 2.0:  # drums enter after the first bar (hook is typing)
            place(drums, kick() * 0.9, t)
            if b % 4 in (1, 3):
                place(drums, clap() * 0.35, t)
        place(drums, hat() * (0.5 if t >= 2 else 0.3), t + BEAT / 2)
        place(bass, bass_note(ROOTS[bar], BEAT * 0.9), t)
    for bar_i in range(int(dur / 2)):
        place(pad, pad_chord(CHORDS[bar_i % 4], 2.0), bar_i * 2.0)
    # typing ticks in the hook (on 8th notes, quiet)
    for k in range(int(2.2 / (BEAT / 2))):
        place(fx, hat() * 0.5, 0.8 + k * BEAT / 2)
    # reveal: riser into 4.0, plucks on each card (4.5, 5.0, 5.5, 6.0)
    r = np.arange(int(1.5 * SR)) / SR
    riser = rng.standard_normal(len(r)) * (r / 1.5) ** 2 * 0.12
    place(fx, lowpass(lowpass(riser, 0.12), 0.2), 2.5)
    for i, m in enumerate([69, 72, 76, 81]):
        place(lead, pluck(m), 4.5 + i * 0.5)
    # accuracy bars: rising arpeggio
    for i, m in enumerate([57, 60, 64, 69, 72, 76]):
        place(lead, pluck(m, 0.3) * 0.8, 13.0 + i * 0.25)
    # outro hit on 16.0 + tail
    place(fx, kick() * 1.2 + np.pad(pad_chord([45, 52, 57], 3.5), (0, 0)) [: int(0.35 * SR)] * 0, 16.0)
    place(pad, pad_chord([57, 60, 64, 69], 4.0) * 1.4, 16.0)
    mix = 0.8 * drums + 0.55 * bass + 0.45 * reverb(pad) + 0.5 * reverb(lead) + 0.4 * fx
    fade = np.ones(n)
    fade[-int(1.2 * SR):] = np.linspace(1, 0, int(1.2 * SR))
    return master(mix * fade)


def bed(dur):
    n = int(dur * SR)
    pad, pulse = np.zeros(n), np.zeros(n)
    for bar_i in range(int(dur / 2) + 1):
        place(pad, pad_chord(CHORDS[bar_i % 4], 2.0), bar_i * 2.0)
    for b in range(int(dur / BEAT) + 1):
        bar = int((b * BEAT) // 2) % 4
        place(pulse, bass_note(ROOTS[bar] + 12, BEAT * 0.5) * 0.35, b * BEAT)
        place(pulse, hat() * 0.25, b * BEAT + BEAT / 2)
    mix = reverb(pad) * 0.6 + pulse * 0.5
    fade = np.ones(n)
    k = int(2.0 * SR)
    fade[:k] = np.linspace(0, 1, k)
    fade[-k:] = np.linspace(1, 0, k)
    return master(mix * fade, peak=0.6)


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    if kind == "intro":
        write(out, intro())
    else:
        write(out, bed(float(sys.argv[3])))
    print("wrote", out)
