"""
Coded Cosmos — genre_engine.py
------------------------------
Original, code-generated music in different genres for the daily Shorts.
Nothing is sampled or copied: every sound is synthesised here from sine waves,
noise and simple physics (FM, additive partials, filtered noise). Genres follow
standard production conventions (tempo, swing, harmony, drum pattern), not any
particular song.

Use from a Short:
    from genre_engine import Song, pick_genre, log_genre
    g = pick_genre(MUSIC_LOG)                       # least-recently-used genre
    song = Song(g, bpm=GENRES[g]["bpm"], total_beats=84, key="A", seed=7)
    song.sections = [("intro", 0, 8, 0.5), ("main", 8, 59, 1.0), ("riser", 59, 63, 0.8),
                     ("payoff", 64, 84, 1.0)]       # (name, beat0, beat1, energy 0..1)
    song.silent_beats = [63]
    song.events = [dict(beat=4.0, kind="hit", idx=0), dict(beat=32, kind="big"), ...]
    stereo = song.render()                           # float32 (N, 2), seamless loop, peak 0.5
Event kinds: "hit" (lead note, idx = scale degree), "big" (chord stab + crash),
"tick" (small percussion), "rise" (riser that ends on that beat), "chord" (harmony stab only).

Run this file directly to render a sampler of every genre:  python genre_engine.py
"""

import os
import math
import json
import numpy as np
from scipy.signal import lfilter, fftconvolve, butter, sosfilt

SR = 44100
A4 = 440.0
NOTE = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1,
        "A": 0, "A#": 1, "B": 2}

# =====================================================================
# GENRES  (all knobs here; each is a style, not a song)
# =====================================================================
GENRES = {
    "classical": dict(bpm=96,  swing=0.0,  mode="major",  scale=[0, 2, 4, 5, 7, 9, 11],
                      prog=[[0, 4, 7], [9, 0, 4], [5, 9, 0], [7, 11, 2]],               # I vi IV V
                      drums=None, bass="alberti", pad="strings", lead="harp", reverb=0.35, lowpass=None,
                      desc="string pad, Alberti piano bass, harp lead, no drums, hall reverb"),
    "jazz":      dict(bpm=120, swing=0.62, mode="major",  scale=[0, 2, 4, 5, 7, 9, 10, 11],
                      prog=[[2, 5, 9, 12], [7, 11, 14, 17], [0, 4, 7, 11], [9, 13, 16, 19]],  # ii7 V7 Imaj7 VI7
                      drums="swing", bass="walking", pad="rhodes_comp", lead="vibes", reverb=0.22, lowpass=None,
                      desc="swing ride + brushes, walking bass, Rhodes comping, vibraphone lead"),
    "hiphop":    dict(bpm=90,  swing=0.58, mode="minor",  scale=[0, 2, 3, 5, 7, 8, 10],
                      prog=[[0, 3, 7, 10], [8, 12, 15], [5, 8, 12], [7, 10, 14]],         # i7 VI iv v
                      drums="boombap", bass="808", pad="rhodes", lead="vox_chop", reverb=0.12, lowpass=None,
                      desc="boom-bap kick/snare, swung hats, 808 sub, dusty Rhodes, vowel chops"),
    "lofi":      dict(bpm=80,  swing=0.6,  mode="major",  scale=[0, 2, 4, 7, 9],
                      prog=[[5, 9, 12, 16], [4, 7, 11, 14], [9, 12, 16, 19], [7, 11, 14, 17]],  # IVmaj7 iii7 vi9 V9
                      drums="lofi", bass="soft", pad="rhodes_warm", lead="kalimba", reverb=0.25, lowpass=3200,
                      desc="lazy drums, vinyl crackle, warped Rhodes, kalimba lead, tape lowpass"),
    "raga":      dict(bpm=100, swing=0.0,  mode="yaman",  scale=[0, 2, 4, 6, 7, 9, 11],
                      prog=[[0, 7, 12]],                                                  # tanpura Sa-Pa drone
                      drums="tabla", bass=None, pad="tanpura", lead="santoor", reverb=0.2, lowpass=None,
                      desc="tanpura drone (Sa Pa), tabla teentaal-style pulse, santoor in raag Yaman scale"),
    "synthwave": dict(bpm=108, swing=0.0,  mode="minor",  scale=[0, 2, 3, 5, 7, 8, 10],
                      prog=[[0, 3, 7], [8, 12, 15], [3, 7, 10], [10, 14, 17]],            # i VI III VII
                      drums="gated", bass="octave", pad="saw_pad", lead="saw_lead", reverb=0.3, lowpass=None,
                      desc="gated snare, octave bass, 80s saw pads + arps"),
    "ambient":   dict(bpm=72,  swing=0.0,  mode="major",  scale=[0, 2, 4, 7, 9],
                      prog=[[0, 7, 16], [9, 16, 19], [5, 12, 16], [7, 14, 19]],
                      drums=None, bass="sub", pad="glass", lead="piano", reverb=0.5, lowpass=None,
                      desc="slow glass pads, felt piano, wide reverb, no drums"),
    # --- 28 Oct owner pool: trendy, fast (>=120 BPM except asmr). Only these are picked by pick_genre(). ---
    "hardtechno": dict(bpm=140, swing=0.0, mode="phrygian", scale=[0, 1, 3, 5, 7, 8, 10],
                       prog=[[0, 3, 7], [1, 5, 8], [0, 3, 7], [10, 13, 17]],            # i bII i bVII
                       drums="hardtechno", bass="rumble", pad=None, lead="glass", reverb=0.18, lowpass=None,
                       desc="distorted 4/4 kick, rumble bass on the off-16ths, clap 2/4, open hats, glass-ping lead"),
    "asmr":        dict(bpm=120, swing=0.0, mode="major", scale=[0, 2, 4, 7, 9],
                       prog=[[0, 7, 16]],
                       drums=None, bass=None, pad=None, lead="glass", reverb=0.3, lowpass=None,
                       desc="no beat wall: crisp glass pings, soft sub, every event its own tiny sound"),
}
TRENDY = ["phonk", "funk", "hardtechno", "hyperpop", "asmr"]   # phonk/funk/hyperpop: add presets before first use
ROTATION_GAP = 3          # a genre can't repeat within the last 3 Shorts
TARGET_PEAK = 0.5


# =====================================================================
# HELPERS
# =====================================================================
def hz(semi_from_a4):
    return A4 * 2 ** (semi_from_a4 / 12)


def env_adsr(n, a, d, s, r, sustain_n=None):
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    sustain_n = max(0, (sustain_n if sustain_n is not None else n) - a_n - d_n - r_n)
    e = np.concatenate([np.linspace(0, 1, max(a_n, 1)), np.linspace(1, s, max(d_n, 1)),
                        np.full(sustain_n, s), np.linspace(s, 0, max(r_n, 1))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def lowpass(x, fc, order=2):
    sos = butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def highpass(x, fc, order=2):
    sos = butter(order, fc, "high", fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def bandpass(x, f0, q=4.0):
    lo, hi = f0 / (1 + 1 / (2 * q)), f0 * (1 + 1 / (2 * q))
    sos = butter(2, [lo, min(hi, SR * 0.45)], "band", fs=SR, output="sos")
    return sosfilt(sos, x)


def reverb_ir(seconds, rng):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t / (seconds / 6.9))[:, None]
    ir[:, 0] = lowpass(ir[:, 0], 6000)
    ir[:, 1] = lowpass(ir[:, 1], 6000)
    return ir / np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))


# =====================================================================
# INSTRUMENTS  (each returns mono np.array)
# =====================================================================
def osc_additive(f, dur, partials, decays, rng=None, inharm=0.0, detune=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    w = np.zeros(n)
    for k, (amp, tau) in enumerate(zip(partials, decays), start=1):
        fk = f * k * math.sqrt(1 + inharm * k * k)
        if fk > SR * 0.45:
            break
        w += amp * np.sin(2 * np.pi * fk * t + (rng.uniform(0, 6.28) if rng is not None else 0)) * np.exp(-t / tau)
        if detune:
            w += 0.5 * amp * np.sin(2 * np.pi * fk * (1 + detune) * t) * np.exp(-t / tau)
    return w


def inst(name, f, dur, vel, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    if name == "harp":
        w = osc_additive(f, dur, [1, .5, .3, .15, .08], [1.2, .7, .4, .25, .15], rng)
        w *= np.clip(t / 0.002, 0, 1)
    elif name == "piano":
        w = osc_additive(f, dur, [1, .45, .25, .14, .08, .05], [2.2, 1.2, .8, .5, .35, .25], rng, inharm=0.0004)
        w = lowpass(w * np.clip(t / 0.004, 0, 1), 2400)
    elif name in ("rhodes", "rhodes_comp", "rhodes_warm"):
        idx = 1.6 * np.exp(-t / 0.25)                                   # FM electric piano
        w = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t)) * np.exp(-t / 1.3)
        w += 0.12 * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.05)  # tine click
        w *= np.clip(t / 0.003, 0, 1)
        if name == "rhodes_warm":
            w = lowpass(w, 1800)
    elif name == "vibes":
        w = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t / 0.3)) * np.exp(-t / 1.6)
        w *= 1 + 0.25 * np.sin(2 * np.pi * 5.5 * t)                     # motor tremolo
        w *= np.clip(t / 0.002, 0, 1)
    elif name == "kalimba":
        w = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.04)) * np.exp(-t / 0.7)
        w *= np.clip(t / 0.001, 0, 1)
    elif name == "santoor":
        w = osc_additive(f, dur, [1, .6, .4, .3, .2, .12], [1.0, .6, .45, .3, .2, .15], rng, detune=0.003)
        w *= np.clip(t / 0.0015, 0, 1)
    elif name == "vox_chop":                                           # vowel "ah"/"oh" formant chop
        src = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 30) if f * k < SR * 0.45)
        fm = (800, 1150) if vel > 0.5 else (500, 900)
        w = bandpass(src, fm[0], 6) + 0.6 * bandpass(src, fm[1], 8)
        w *= env_adsr(n, 0.005, 0.08, 0.6, 0.06, int(0.22 * SR))
    elif name == "saw_lead":
        w = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 16) if f * k < SR * 0.45)
        w = lowpass(w, 2600) * env_adsr(n, 0.005, 0.15, 0.5, 0.1, int(0.3 * SR))
    elif name == "glass":                                              # struck glass: inharmonic partials, fast decay
        w = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.35) + 0.45 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.12)
             + 0.25 * np.sin(2 * np.pi * f * 5.40 * t) * np.exp(-t / 0.05))
        w *= np.clip(t / 0.0007, 0, 1)
    elif name == "bell":
        w = osc_additive(f, dur, [1, .5, .3], [1.8, 1.0, .6], rng)
    else:
        raise ValueError(name)
    return vel * w


def pad_voice(kind, freqs, dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    w = np.zeros(n)
    for f in freqs:
        if kind == "strings":
            for det in (-0.004, 0.0, 0.005):
                w += sum(np.sin(2 * np.pi * f * (1 + det) * k * t + rng.uniform(0, 6)) / k ** 1.3 for k in range(1, 9))
        elif kind == "saw_pad":
            for det in (-0.006, 0.006):
                w += sum(np.sin(2 * np.pi * f * (1 + det) * k * t) / k for k in range(1, 12))
        elif kind == "glass":
            w += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.01 * t) + 0.15 * np.sin(2 * np.pi * f * 3 * t)
    w = lowpass(w, 2200 if kind != "glass" else 4000)
    return w * env_adsr(n, min(0.6, dur * 0.3), 0.2, 0.85, min(0.5, dur * 0.3))


def tanpura(f_sa, dur, rng):
    """Jawari buzz: many slowly-beating harmonics, plucked Pa-Sa-Sa-sa cycle."""
    n = int(dur * SR)
    out = np.zeros(n)
    cyc = [f_sa * 1.5 / 2, f_sa, f_sa, f_sa / 2]          # Pa(low) Sa Sa Sa(low)
    step = 1.1
    k = 0
    while k * step < dur:
        f = cyc[k % 4]
        m = int(min(4.0, dur - k * step) * SR)
        t = np.arange(m) / SR
        w = np.zeros(m)
        for h in range(1, 24):
            amp = (1 / h ** 0.8) * (0.6 + 0.4 * np.sin(2 * np.pi * (0.15 * h) * t + h))   # jawari shimmer
            w += amp * np.sin(2 * np.pi * f * h * t)
        w *= np.exp(-t / 2.5) * np.clip(t / 0.01, 0, 1)
        i = int(k * step * SR)
        out[i:i + m] += w[:n - i]
        k += 1
    return lowpass(out, 5000) * 0.25


# --- drums ---
def kick(rng, punch=1.0, tail=0.15):
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    f = 48 + 110 * punch * np.exp(-t / 0.03)
    w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tail)
    w[:int(0.002 * SR)] += np.linspace(0.4, 0, int(0.002 * SR))
    return w


def sub808(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    fr = f * (1 + 0.6 * np.exp(-t / 0.02))
    return np.tanh(1.5 * np.sin(2 * np.pi * np.cumsum(fr) / SR)) * np.exp(-t / (dur * 0.6))


def snare(rng, tone=190, snap=1.0, dur=0.25):
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = highpass(rng.standard_normal(n), 1500) * np.exp(-t / (0.06 * snap))
    body = np.sin(2 * np.pi * tone * t) * np.exp(-t / 0.05)
    return 0.7 * noise + 0.5 * body


def hat(rng, open_=False):
    n = int((0.3 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    return highpass(rng.standard_normal(n), 7000) * np.exp(-t / (0.12 if open_ else 0.012))


def ride(rng):
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    w = highpass(rng.standard_normal(n), 5000) * np.exp(-t / 0.35) * 0.4
    w += sum(np.sin(2 * np.pi * f * t) for f in (3200, 4870, 6100)) * 0.05 * np.exp(-t / 0.5)
    return w


def brush(rng, dur=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return bandpass(rng.standard_normal(n), 3000, 1.2) * np.sin(np.pi * np.clip(t / dur, 0, 1)) * 0.5


def crash(rng):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    return highpass(rng.standard_normal(n), 3500) * np.exp(-t / 0.7) * 0.6


def tabla(rng, bol, f_sa):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    if bol == "dha":   # dayan + bayan
        f = 80 * (1 + 0.8 * np.exp(-t / 0.08))
        w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
        w += 0.7 * np.sin(2 * np.pi * f_sa * 2 * t) * np.exp(-t / 0.25)
    elif bol == "na":  # ringing dayan
        w = (np.sin(2 * np.pi * f_sa * 2 * t) + 0.4 * np.sin(2 * np.pi * f_sa * 2 * 2.9 * t)) * np.exp(-t / 0.22)
    elif bol == "ge":  # bayan bend
        f = 75 * (1 + 0.5 * np.exp(-t / 0.15))
        w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.4)
    else:              # "ti" dry slap
        w = bandpass(rng.standard_normal(n), 2200, 2) * np.exp(-t / 0.02) * 2
    return w * np.clip(t / 0.001, 0, 1)


def vinyl(n, rng):
    hiss = lowpass(rng.standard_normal(n), 5000) * 0.02
    pops = np.zeros(n)
    idx = rng.integers(0, n, size=int(n / SR * 6))
    pops[idx] = rng.uniform(0.1, 0.4, len(idx)) * rng.choice([-1, 1], len(idx))
    return hiss + highpass(pops, 1000)


# =====================================================================
# SONG
# =====================================================================
class Song:
    def __init__(self, genre, bpm=None, total_beats=84, key="A", seed=0):
        self.g = GENRES[genre]
        self.genre = genre
        self.bpm = bpm or self.g["bpm"]
        self.beat_s = 60.0 / self.bpm
        self.total_beats = total_beats
        self.dur = total_beats * self.beat_s
        self.n = int(round(self.dur * SR))
        self.root = NOTE[key] - 24            # semitones from A4: root two octaves down (A2 for key A)
        self.rng = np.random.default_rng(seed)
        self.sections = [("main", 0, total_beats, 1.0)]
        self.silent_beats = []
        self.events = []
        self.chord_beats = 8 if self.bpm < 100 else 8

    # --- timing ---
    def t_of(self, beat, sub=0.0):
        """beat + swung subdivision (sub in 8ths: 0 or 0.5)."""
        sw = self.g["swing"]
        off = (sw if sub == 0.5 and sw else sub) * self.beat_s
        return beat * self.beat_s + off

    def energy(self, b):
        for _, b0, b1, e in self.sections:
            if b0 <= b < b1:
                return e
        return 1.0

    def chord_at(self, b):
        p = self.g["prog"]
        return p[int(b // self.chord_beats) % len(p)]

    def scale_note(self, idx, octave=2):
        sc = self.g["scale"]
        return self.root + 12 * octave + sc[idx % len(sc)] + 12 * (idx // len(sc))

    def put(self, buf, t, w, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i >= len(buf) or i < -len(w):
            return
        s = max(i, 0)
        e = min(i + len(w), len(buf))
        g = gain * np.array([math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)]) * 1.414
        buf[s:e] += w[s - i:e - i, None] * g[None, :]

    # --- layers ---
    def layer_harmony(self, buf):
        g, rng = self.g, self.rng
        cb = self.chord_beats
        if not g["pad"]:
            return
        for b0 in range(0, self.total_beats, cb):
            ch = self.chord_at(b0)
            freqs = [hz(self.root + 12 + c) for c in ch]
            dur = cb * self.beat_s + 0.4
            if g["pad"] in ("strings", "saw_pad", "glass"):
                self.put(buf, b0 * self.beat_s, pad_voice(g["pad"], freqs, dur, rng), 0.10, 0)
            elif g["pad"] == "tanpura":
                pass
            else:                                               # Rhodes family: comp rhythm
                hits = {"rhodes_comp": [(0, 0.5), (1, 0.5), (3, 0), (4, 0.5), (6, 0)],
                        "rhodes": [(0, 0), (2, 0.5), (4, 0), (6, 0.5)],
                        "rhodes_warm": [(0, 0), (3, 0.5), (4, 0)]}[g["pad"]]
                for hb, sub in hits:
                    if b0 + hb >= self.total_beats:
                        continue
                    for j, f in enumerate(freqs):
                        self.put(buf, self.t_of(b0 + hb, sub) + 0.012 * j,
                                 inst(g["pad"], f * 2, 1.6, 0.11, rng), 1.0, -0.3 + 0.2 * j)
        if g["pad"] == "tanpura":
            self.put(buf, 0, tanpura(hz(self.root + 12), self.dur + 1, rng), 1.0, 0)

    def layer_bass(self, buf):
        g, rng = self.g, self.rng
        kind = g["bass"]
        if not kind:
            return
        for b in range(self.total_beats):
            if self.energy(b) < 0.3:
                continue
            ch = self.chord_at(b)
            rootf = hz(self.root + ch[0] - (12 if ch[0] > 6 else 0))
            pos = b % self.chord_beats
            if kind == "alberti":                               # low-high-mid-high in 8ths
                order = [0, 2, 1, 2]
                for s in range(2):
                    c = ch[order[(2 * pos + s) % 4] % len(ch)]
                    self.put(buf, self.t_of(b, s * 0.5), inst("piano", hz(self.root + c), 1.0, 0.18, rng), 1.0, -0.2)
            elif kind == "walking":
                nxt = self.chord_at(b + 1)[0]
                steps = [ch[0], ch[1] % 12 if len(ch) > 1 else ch[0] + 4, ch[2] % 12 if len(ch) > 2 else ch[0] + 7, nxt + 1]
                note = steps[pos % 4] if pos % 4 < 3 else (nxt - 1)
                f = hz(self.root + (note % 12))
                w = osc_additive(f, 0.6, [1, .4, .15], [.5, .3, .2], rng) * env_adsr(int(0.6 * SR), .005, .1, .7, .1)
                self.put(buf, b * self.beat_s, w, 0.32, 0.1)
            elif kind == "808" and pos in (0, 3, 6):
                self.put(buf, self.t_of(b, 0.5 if pos == 3 else 0), sub808(rootf, 1.2), 0.30, 0)
            elif kind == "soft" and pos in (0, 4):
                self.put(buf, b * self.beat_s, osc_additive(rootf, 1.8, [1, .3], [1.0, .5], rng), 0.30, 0)
            elif kind == "octave":
                for s in range(2):
                    f = rootf * (2 if s else 1)
                    w = lowpass(sum(np.sin(2 * np.pi * f * k * np.arange(int(.22 * SR)) / SR) / k for k in range(1, 8)), 900)
                    self.put(buf, self.t_of(b, s * 0.5), w * env_adsr(len(w), .003, .05, .6, .05), 0.16, 0)
            elif kind == "rumble":                               # reverb-y distorted sub on the 3 off-16ths
                for s16 in (0.25, 0.5, 0.75):
                    n = int(0.11 * SR); tt = np.arange(n) / SR
                    w = np.tanh(3 * np.sin(2 * np.pi * rootf * tt)) * np.exp(-tt / 0.05)
                    self.put(buf, (b + s16) * self.beat_s, lowpass(w, 260), 0.22 * self.energy(b), 0)
            elif kind == "sub" and pos == 0:
                self.put(buf, b * self.beat_s, osc_adsub(rootf, self.chord_beats * self.beat_s), 0.22, 0)

    def layer_drums(self, buf):
        g, rng = self.g, self.rng
        kind = g["drums"]
        if not kind:
            return
        K, S, Hc, Ho = kick(rng), snare(rng), hat(rng), hat(rng, True)
        R, Br = ride(rng), brush(rng)
        f_sa = hz(self.root + 12)
        teentaal = ["dha", "dhin", "dhin", "dha", "dha", "dhin", "dhin", "dha",
                    "dha", "tin", "tin", "ta", "ta", "dhin", "dhin", "dha"]
        bolmap = {"dha": "dha", "dhin": "dha", "tin": "na", "ta": "ti"}
        for b in range(self.total_beats):
            e = self.energy(b)
            if b in self.silent_beats or e < 0.3:
                continue
            bar = b % 4
            if kind == "boombap":
                if bar == 0 or (bar == 2 and b % 8 >= 4):
                    self.put(buf, self.t_of(b), K, 0.42)
                if bar == 1 and b % 2 == 1:
                    self.put(buf, self.t_of(b, 0.5), K, 0.3)
                if bar in (1, 3):
                    self.put(buf, self.t_of(b), S, 0.30, 0.1)
                for s in (0, 0.5):
                    self.put(buf, self.t_of(b, s), Hc, 0.10 * e, 0.3)
            elif kind == "lofi":
                if bar == 0 or (bar == 2 and b % 8 == 6):
                    self.put(buf, self.t_of(b), lowpass(K, 900), 0.40)
                if bar in (1, 3):
                    self.put(buf, self.t_of(b) + 0.012, lowpass(S, 3500), 0.20, 0.1)
                for s in (0, 0.5):
                    self.put(buf, self.t_of(b, s), lowpass(Hc, 9000), 0.06 * e, 0.3)
            elif kind == "swing":
                self.put(buf, self.t_of(b), R, 0.10, 0.4)
                if bar in (1, 3):
                    self.put(buf, self.t_of(b, 0.5), R, 0.07, 0.4)
                    self.put(buf, self.t_of(b), Br, 0.20, -0.2)
                if bar == 0 and e > 0.6:
                    self.put(buf, self.t_of(b), kick(rng, 0.5, 0.1), 0.20)
            elif kind == "gated":
                self.put(buf, self.t_of(b), K, 0.42)
                if bar in (1, 3):
                    gs = snare(rng, 200, 2.5, 0.35)
                    gs[int(0.18 * SR):] = 0                     # gate
                    self.put(buf, self.t_of(b), gs, 0.34)
                for s in (0, 0.5):
                    self.put(buf, self.t_of(b, s), Ho if s else Hc, 0.07 * e, 0.3)
            elif kind == "hardtechno":
                if not hasattr(self, "_hk"):
                    self._hk = np.tanh(2.5 * kick(rng, 1.4, 0.22))
                self.put(buf, self.t_of(b), self._hk, 0.50)
                if bar in (1, 3) and e > 0.6:
                    self.put(buf, self.t_of(b), snare(rng, 900, 3.0, 0.18), 0.22, -0.1)
                self.put(buf, self.t_of(b, 0.5), Ho, 0.10 * e, 0.25)
                if e > 0.8:
                    for s16 in (0.25, 0.75):
                        self.put(buf, (b + s16) * self.beat_s, Hc, 0.05, -0.3)
            elif kind == "tabla":
                for s in (0, 0.5):
                    bol = teentaal[(2 * b + int(2 * s)) % 16]
                    self.put(buf, self.t_of(b, s), tabla(rng, bolmap[bol], f_sa), 0.22 * (1.2 if s == 0 else 0.8), 0.15)

    def layer_events(self, buf):
        g, rng = self.g, self.rng
        for ev in self.events:
            b, kind = ev["beat"], ev.get("kind", "hit")
            t = b * self.beat_s
            if int(b) in self.silent_beats:
                continue
            if kind == "hit":
                f = hz(self.scale_note(ev.get("idx", 0), 3))
                self.put(buf, t, inst(g["lead"], f, 1.8, 0.30 * ev.get("vel", 1.0), rng), 1.0, ev.get("pan", 0.15))
            elif kind in ("big", "chord"):
                ch = self.chord_at(b)
                lead = g["lead"]
                for j, c in enumerate(ch):
                    f = hz(self.root + 24 + c)
                    self.put(buf, t + 0.01 * j, inst(lead, f, 2.5, 0.16, rng), 1.0, -0.4 + 0.25 * j)
                if kind == "big":
                    if g["drums"]:
                        self.put(buf, t, crash(rng), 0.18, 0.3)
                        self.put(buf, t, kick(rng), 0.40)
                    else:
                        self.put(buf, t, inst("bell", hz(self.root + 36), 3.0, 0.2, rng), 1.0, 0)
            elif kind == "tick":
                if g["drums"] == "tabla":
                    self.put(buf, t, tabla(rng, "ti", hz(self.root + 12)), 0.25, 0.4)
                else:
                    self.put(buf, t, hat(rng), 0.12, 0.5)
            elif kind == "rise":
                L = ev.get("len", 4) * self.beat_s
                n = int(L * SR)
                x = np.linspace(0, 1, n)
                if g["pad"] in ("strings", "glass"):
                    w = pad_voice(g["pad"], [hz(self.root + 24 + c) for c in self.chord_at(b - 1)], L, rng) * x ** 2
                else:
                    w = highpass(rng.standard_normal(n), 400) * x ** 2.5 * 0.5
                    w += 0.4 * np.sin(2 * np.pi * np.cumsum(200 * 2 ** (3 * x)) / SR) * x ** 2
                self.put(buf, t - L, w, 0.14, 0)

    # --- master ---
    def render(self):
        tail = int(4 * SR)
        buf = np.zeros((self.n + tail, 2))
        self.layer_harmony(buf)
        rb = np.zeros_like(buf)
        self.layer_bass(rb)
        self.layer_drums(rb)
        br = getattr(self, "brightness", None)            # optional fn(t_s)->0..1: 0 = muffled, 1 = open
        if br is not None:
            w = np.array([br(i / SR) for i in range(0, len(rb), 256)])
            w = np.interp(np.arange(len(rb)), np.arange(0, len(rb), 256), w)[:, None]
            rb = lowpass(rb, 380) * (1 - w) + rb * w
        buf += rb
        self.layer_events(buf)
        g = self.g
        if g["lowpass"]:
            buf = lowpass(buf, g["lowpass"])
        if g["drums"] == "lofi":                                # tape wow: slow pitch wobble
            tt = np.arange(len(buf)) / SR
            warp = tt + 0.0018 * np.sin(2 * np.pi * 0.5 * tt)
            buf = np.stack([np.interp(warp, tt, buf[:, c]) for c in range(2)], axis=1)
            v = vinyl(len(buf), self.rng)
            buf += np.stack([v, np.roll(v, 50)], axis=1)
        wet = np.stack([fftconvolve(buf[:, c], reverb_ir(2.2, self.rng)[:, c])[:len(buf)] for c in range(2)], axis=1)
        buf = buf + g["reverb"] * wet
        out = buf[:self.n].copy()
        out[:tail] += buf[self.n:self.n + tail]                 # wrap the tail -> seamless loop
        for sb in self.silent_beats:                            # true digital silence, 4 ms fades
            s0, s1 = int(sb * self.beat_s * SR), int((sb + 1) * self.beat_s * SR)
            nf = int(0.004 * SR)
            out[s0 - nf:s0] *= np.linspace(1, 0, nf)[:, None]
            out[s0:s1] = 0
            out[s1:s1 + nf] *= np.linspace(0, 1, nf)[:, None]
        out = np.tanh(out / (np.max(np.abs(out)) + 1e-9) * 1.2) / math.tanh(1.2)
        return (out * TARGET_PEAK).astype(np.float32)


def osc_adsub(f, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * env_adsr(n, 0.3, 0.2, 0.8, 0.6)


# =====================================================================
# ROTATION
# =====================================================================
def read_log(path):
    if not os.path.exists(path):
        return []
    rows = []
    for ln in open(path, encoding="utf-8"):
        p = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(p) >= 3 and p[2] in GENRES:
            rows.append(p)
    return rows


def pick_genre(log_path, prefer=None, avoid=()):
    """prefer: genres that suit the topic (e.g. ['raga'] for Ancient Code); never one used in the last ROTATION_GAP."""
    used = [r[2] for r in read_log(log_path)]
    recent = set(used[-ROTATION_GAP:]) | set(avoid)
    trendy = [g for g in TRENDY if g in GENRES]          # owner 28 Oct: old presets stay but are never picked
    pool = [g for g in (prefer or trendy) if g not in recent and g in trendy] or [g for g in trendy if g not in recent]
    last_use = {g: (len(used) - 1 - used[::-1].index(g)) if g in used else -1 for g in pool}
    return min(pool, key=lambda g: last_use[g])


def log_genre(log_path, date, short, genre, bpm, key, note=""):
    new = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if new:
            f.write("# MUSIC_LOG — one row per Short (genre rotation: no repeat within 3)\n"
                    "| Date | Short | Genre | BPM | Key | Note |\n|---|---|---|---|---|---|\n")
        f.write(f"| {date} | {short} | {genre} | {bpm} | {key} | {note} |\n")


# =====================================================================
# SAMPLER
# =====================================================================
if __name__ == "__main__":
    import wave
    out_dir = os.path.dirname(os.path.abspath(__file__))
    parts, marks, t = [], [], 0.0
    for gi, name in enumerate(GENRES):
        s = Song(name, total_beats=16, key="A", seed=gi + 1)
        s.sections = [("main", 0, 16, 1.0)]
        s.events = ([dict(beat=b, kind="hit", idx=i) for i, b in enumerate([1, 2.5, 4, 5.5, 7, 9, 10.5, 12])]
                    + [dict(beat=8, kind="big"), dict(beat=14, kind="rise", len=2), dict(beat=15.5, kind="tick")])
        a = s.render()
        parts += [a, np.zeros((int(0.6 * SR), 2), np.float32)]
        marks.append(dict(genre=name, t0=round(t, 2), t1=round(t + len(a) / SR, 2), bpm=s.bpm, desc=GENRES[name]["desc"]))
        t += len(a) / SR + 0.6
    st = np.concatenate(parts)
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(os.path.join(out_dir, "genre_sampler.wav"), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())
    json.dump(marks, open(os.path.join(out_dir, "genre_sampler.json"), "w"), indent=1)
    print(json.dumps(marks, indent=1))
