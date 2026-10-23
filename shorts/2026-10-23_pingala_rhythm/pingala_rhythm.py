"""
Coded Cosmos — "Poets found this before Fibonacci."  (Pingala-tradition rhythm counting) — v3
------------------------------------------------------------------------------------------
ONE idea, shown with its WHY:
  Indian prosody counts rhythms made of short sounds (1 beat) and long sounds (2 beats).
  How many rhythms last n beats?   1, 2, 3, 5, 8, 13 ...
  Why: every n-beat rhythm ends in a short or a long sound, so it is
        (an (n-1)-beat rhythm + short)  or  (an (n-2)-beat rhythm + long).
  We build the 4-beat column exactly that way (3 + 2 = 5), then 5 beats (5 + 3 = 8).
  Virahanka (~700 CE) wrote this rule ~500 years before Fibonacci (1202).

Visual language: plain blocks — short = small cyan block, long = double-width gold
block — and a drum "ta"/"taa" plays each block as it lights.  The drum plays exactly
the rhythms on screen; plucks/chimes take their pitch from the counts.  100% original.

Run:  python pingala_rhythm.py          (add --sheet for the preview sheet only)
  -> pingala_rhythm.mp4, pingala_rhythm_upload.mp4, preview_frames.jpg,
     audio_levels.txt, beats.json
"""

import os
import io
import sys
import json
import math
import wave
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

# =====================================================================
# GLOBAL PARAMETERS
# =====================================================================
OUT_DIR          = os.path.dirname(os.path.abspath(__file__))
SLUG             = "pingala_rhythm"
OUT_VIDEO        = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD       = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET        = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_LEVELS       = os.path.join(OUT_DIR, "audio_levels.txt")
OUT_BEATS        = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO        = os.path.join(OUT_DIR, "_pingala_audio.wav")
MAKE_UPLOAD      = True
UPLOAD_TARGET_MB = 9.0
UPLOAD_AUDIO_K   = 160

W, H             = 1080, 1920
FPS              = 30
BG_COLOR         = "#05060a"
BG_RGB           = (5, 6, 10)
plt.rcParams["figure.facecolor"] = BG_COLOR
plt.rcParams["savefig.facecolor"] = BG_COLOR

# --- music grid (everything below is in BEATS from t = 0; 1 beat = 0.5 s) ---
BPM              = 120
BEATS_PER_BAR    = 4
SR               = 44100
TOTAL_BEATS      = 116             # 29 bars = 58 s

# --- colours ---
C_SHORT          = "#4fd1ff"       # short sound: small cyan block
C_LONG           = "#ffd166"       # long sound: double-width gold block
C_FROM1          = "#c79bff"       # rows copied from the (n-1)-beat column
C_FROM2          = "#ff9f5a"       # rows copied from the (n-2)-beat column
C_ASK            = "#ff6bd6"
C_DIM            = "#8a93a6"

# --- music sections (bars on bar lines): (key, start beat, end beat) ---
SECTIONS = [
    ("intro", 0, 20), ("small", 20, 40), ("why4", 40, 72), ("why5", 72, 80),
    ("fib", 80, 92), ("poets", 92, 104), ("ask", 104, 116),
]

# --- story beats (visual change at beat B; caption follows CAPTION_LAG_BEATS later) ---
STORY = [
    (0,   "Poets found this\nbefore Fibonacci."),
    (10,  "Short sound = 1 beat.\nLong = 2 beats."),
    (20,  "1 beat: 1 way.\n2 beats: 2 ways."),
    (30,  "3 beats: 3 ways."),
    (40,  "4 beats? Take the 3 ways,\nadd a short."),
    (51,  "Then the 2 ways,\nadd a long."),
    (61,  "So 4 beats:\n3 + 2 = 5 ways."),
    (72,  "5 beats:\n5 + 3 = 8 ways."),
    (82,  "This is the\nFibonacci sequence."),
    (92,  "Indian poets wrote it ~500\nyears before Fibonacci."),
    (102, "How many ways\nfor 8 beats?"),
]
CAPTION_LAG_BEATS = 0.5
MIN_CAPTION_S    = 5.0

# --- intro (hook) ---
HOOK_PATTERN     = "LG"            # ta  taa  (then 1 beat rest) every bar
INTRO_LABEL_BEAT = 10
INTRO_SHORT_TALK = (10, 15)        # highlight short while "Short sound = 1 beat"
INTRO_LONG_TALK  = (15, 20)
END_HOOK_BEAT    = 112             # last bar morphs back into the hook

# --- columns ---
# rows that play fully: (column n, row i, onset beat, unit beats per short)
PLAYS = [(1, 0, 20, 1.0), (2, 0, 21, 1.0), (2, 1, 23, 1.0),
         (3, 0, 30, 1.0), (3, 1, 33, 1.0), (3, 2, 36, 1.0),
         (4, 0, 45, 0.5), (4, 1, 47, 0.5), (4, 2, 49, 0.5), (4, 3, 55, 0.5), (4, 4, 57, 0.5)]
BADGE_BEAT       = {1: 21, 2: 25, 3: 31, 4: 61, 5: 72}
# the WHY: build column n from (n-1)+short and (n-2)+long
WHY = {
    4: dict(sA=40, dA=1.5, tA=[42, 43, 44], sB=51, dB=1.5, tB=[53, 54]),
    5: dict(sA=72, dA=1.0, tA=[73.5, 73.75, 74.0, 74.25, 74.5], sB=75, dB=1.0, tB=[76.5, 76.75, 77.0]),
}
# column layouts: (start beat, visible columns, unit px)
LAYOUTS          = [(0, [], 100, 0), (20, [1, 2], 100, 0), (30, [1, 2, 3], 90, 0), (40, [2, 3, 4], 70, 130),
                    (72, [3, 4, 5], 54, 130)]    # (beat, columns, unit px, right reserve for brackets)
GROUP_GAP        = 30              # gap between "from n-1" rows and "from n-2" rows
OUTLINE_W        = 5
BRACKET_W        = 6
BRACKET_DX       = 22
BRACKET_LABEL_SIZE = 30
LAYOUT_MOVE_BEATS = 1.0
COLS_OUT_BEAT    = 82              # columns fade, number strip arrives

# --- strip / riser / silence / timeline / ask ---
STRIP_VALUES     = [1, 2, 3, 5, 8, 13]
STRIP_POP0       = 82.0
STRIP_POP_STEP   = 0.5
STRIP_DOTS_BEAT  = 85.0
RISER_BEATS      = (86, 91)
SILENT_BEAT      = 91
TL_BEATS         = [92.0, 93.0, 94.0]   # dot 1, dot 2 + arrow, "~500 years" bracket
TIMELINE         = [("Virahāṅka", "~700 CE", C_LONG), ("Fibonacci", "1202 CE", C_SHORT)]
SOURCE_CHIP      = "VIRAHĀṄKA · HEMACHANDRA"
ASK_BEAT         = 102

# --- layout (pixels) ---
SIGN_Y           = 46
EQ_Y_TOP         = 120
EQ_BLOCK_H       = 190
CHIP_Y           = 330
SIM_CY_FRAC      = 0.46
SIM_W, SIM_H     = 960, 900
SIM_X0           = (W - SIM_W) // 2
SIM_Y0           = int(H * SIM_CY_FRAC) - SIM_H // 2
STORY_Y          = SIM_Y0 + SIM_H + 12
STORY_MAX_W      = 880
SAFE_BOTTOM      = int(H * 0.80)

# intro blocks
IN_UNIT_PX       = 190
IN_BLOCK_H       = 120
IN_GAP           = 50
IN_Y             = 400
IN_SYL_Y         = 255
IN_RULER_Y       = 505
IN_LABEL_Y       = 575
# columns
COL_MIN_SLOT     = 150
COL_GAP          = 50
COL_HEAD_Y       = 120
ROW_Y0           = 200
ROW_PITCH_K      = 0.75
ROW_PITCH_MIN    = 56
ROW_PITCH_MAX    = 72
BLOCK_H_K        = 0.70
BADGE_Y          = 745
# strip
ST_Y             = 380
ST_DX            = 132
ST_BOX           = 108
# timeline
TL_Y             = 560
TL_X             = [230, 730]

# --- drawing ---
GLOW_DOWNSCALE   = 4
GLOW_RADIUS      = 5
GLOW_GAIN        = 1.5
KICK_FLASH       = 0.22
KICK_FLASH_DECAY = 8.0
HIT_FLASH_DECAY  = 7.0
SILENT_DIM       = 0.28
DIM_OTHER        = 0.35

# --- text ---
FONT_BOLD        = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG         = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
STORY_FONT_SIZE  = 64
CHIP_FONT_SIZE   = 34
SIGN_FONT_SIZE   = 32
TEXT_STROKE      = 5
TEXT_FADE_S      = 0.30
EQ_FONTSIZE      = 13
EQ_DPI           = 330
EQ_COLOR         = "#ffffffdd"
EQ_STROKE_W      = 3
EQ_FADE_S        = 0.15

# --- audio mix ---
DRONE_GAIN       = 0.11
SUB_GAIN         = 0.10
TABLA_GAIN       = 0.62
KICK_GAIN        = 0.34
HAT_GAIN         = 0.04
CLAP_GAIN        = 0.08
CHIME_GAIN       = 0.16
POP_GAIN         = 0.18
RISER_GAIN       = 0.14
PUMP_DEPTH       = 0.30
PUMP_DECAY_S     = 0.12
SECTION_XFADE_S  = 0.02
DRONE_ROOT       = 110.0
TARGET_LUFS      = -14.0
TARGET_TP        = -1.0
LIMIT_CEIL_DB    = -2.2
SIGN_OFF         = "@cosmoscoded"

# =====================================================================
# DERIVED SCHEDULE
# =====================================================================
BEAT_S   = 60.0 / BPM
BAR_S    = BEAT_S * BEATS_PER_BAR
DURATION = TOTAL_BEATS * BEAT_S
N_FRAMES = int(round(DURATION * FPS))
N_SAMP   = int(round(DURATION * SR))
SCHED    = [dict(key=k, t0=b0 * BEAT_S, t1=b1 * BEAT_S) for k, b0, b1 in SECTIONS]
SILENT_T0 = SILENT_BEAT * BEAT_S
SILENT_T1 = SILENT_T0 + BEAT_S
UNITS    = {"L": 1, "G": 2}


def section_at(time_s):
    for s in SCHED:
        if s["t0"] <= time_s < s["t1"]:
            return s
    return SCHED[-1]


TEXT_EVENTS = []
prev = -1e9
for b, txt in STORY:
    trig = (b + (CAPTION_LAG_BEATS if b > 0 else 0)) * BEAT_S
    start = max(trig, prev + MIN_CAPTION_S) if TEXT_EVENTS else trig
    TEXT_EVENTS.append([start, txt])
    prev = start
for i in range(len(TEXT_EVENTS)):
    TEXT_EVENTS[i].append(TEXT_EVENTS[i + 1][0] if i + 1 < len(TEXT_EVENTS) else DURATION + 1)


def compositions(n):
    """Rhythms of n beats: the (n-1)-beat ones + short, then the (n-2)-beat ones + long."""
    if n == 0:
        return [""]
    if n < 0:
        return []
    return [c + "L" for c in compositions(n - 1)] + [c + "G" for c in compositions(n - 2)]


COMPS = {n: compositions(n) for n in range(1, 6)}
COUNT = {n: len(COMPS[n]) for n in COMPS}


def pattern_onsets(pat, unit_beats):
    out, u = [], 0
    for s in pat:
        out.append((u * unit_beats, UNITS[s] * unit_beats, s))
        u += UNITS[s]
    return out


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def intro_hits():
    """(onset beat, dur beats, syllable) of the hook drum: ta-taa each bar."""
    ev = []
    for bar in list(range(5)) + [END_HOOK_BEAT // 4]:
        for on, du, s in pattern_onsets(HOOK_PATTERN, 1.0):
            ev.append((bar * 4 + on, du, s))
    return ev


INTRO_HITS = intro_hits()


def append_events():
    """(beat, column n, row i, syllable) — the block appended in the WHY animation."""
    ev = []
    for n, w in WHY.items():
        for i, tb in enumerate(w["tA"]):
            ev.append((tb, n, i, "L"))
        for j, tb in enumerate(w["tB"]):
            ev.append((tb, n, COUNT[n - 1] + j, "G"))
    return ev


APPENDS = append_events()


def kick_on(b):
    if SILENT_BEAT <= b < SILENT_BEAT + 1:
        return False
    if 72 <= b < SILENT_BEAT:
        return True
    if 40 <= b < 72 or 92 <= b < 104:
        return b % 4 in (0, 2)
    return b % 4 == 0


KICK_TIMES = np.array([b * BEAT_S for b in range(TOTAL_BEATS) if kick_on(b)])

# =====================================================================
# AUDIO
# =====================================================================
RNG = np.random.default_rng(14)


def add_at(buf, t_s, w):
    i = int(round(t_s * SR))
    if i >= len(buf):
        return
    e = min(i + len(w), len(buf))
    buf[i:e] += w[:e - i]


def tone_env(n, attack_s, tau):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack_s, 1e-4), 0, 1)
    return a * np.exp(-t / tau), t


def stroke_ta(dur):
    """short stroke (dayan 'na'): bright, short."""
    n = int(0.35 * SR)
    env, t = tone_env(n, 0.001, 0.035 + 0.22 * min(dur, 0.5))
    f = 600.0
    w = (0.75 * np.sin(2 * np.pi * f * t) + 0.30 * np.sin(2 * np.pi * 2.01 * f * t) * np.exp(-t / 0.02)
         + 0.20 * np.sin(2 * np.pi * 2.98 * f * t) * np.exp(-t / 0.012)) * env
    return w + RNG.standard_normal(n) * np.exp(-t / 0.0015) * 0.5


def stroke_taa(dur):
    """long stroke (bayan 'ge' glide + ring): sustains through its 2 beats."""
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    fb = 82 + 38 * (1 - np.exp(-t / 0.10))
    pb = 2 * np.pi * np.cumsum(fb) / SR
    tau = 0.10 + 0.5 * dur
    bay = np.sin(pb) * np.clip(t / 0.003, 0, 1) * np.exp(-t / tau)
    ring = 0.35 * np.sin(2 * np.pi * 300 * t) * np.exp(-t / (0.06 + 0.4 * dur))
    ring += 0.15 * np.sin(2 * np.pi * 600 * t) * np.exp(-t / 0.05)
    return bay + ring + RNG.standard_normal(n) * np.exp(-t / 0.002) * 0.35


def bell(f0, harms, dur=1.8):
    n = int(dur * SR)
    t = np.arange(n) / SR
    w = np.zeros(n)
    for h in harms:
        w += (1 / math.sqrt(h)) * np.sin(2 * np.pi * f0 * h * t) * np.exp(-t / (1.1 / (1 + 0.12 * h)))
    w *= np.clip(t / 0.003, 0, 1)
    return w / max(1.0, math.sqrt(len(harms)))


def pluck(f, tau=0.28, dur=1.0):
    n = int(dur * SR)
    env, t = tone_env(n, 0.002, tau)
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.08)) * env


def kick_wave():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 46 + 100 * np.exp(-t / 0.03)
    w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.15)
    w[:int(0.003 * SR)] += np.linspace(0.5, 0, int(0.003 * SR))
    return w


def ebur_measure(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    js = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])
    return dict(I=float(js["input_i"]), TP=float(js["input_tp"]), LRA=float(js["input_lra"]))


def write_wav(path, st):
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def limiter(st, ceil_db):
    """Look-ahead brickwall on a 4x oversampled peak estimate (true-peak-ish)."""
    ceil = 10 ** (ceil_db / 20)
    for _ in range(2):
        os4 = np.max(np.abs(resample_poly(st, 4, 1, axis=0)), axis=1)
        pk = os4.reshape(-1, 4).max(axis=1)[:len(st)]
        pk = np.pad(pk, (0, len(st) - len(pk)))
        gr = 1 - np.minimum(1, ceil / np.maximum(pk, 1e-9))
        L = int(0.004 * SR)
        gr = maximum_filter1d(gr, size=2 * L + 1)
        gr = uniform_filter1d(gr, size=L + 1)
        rel = uniform_filter1d(maximum_filter1d(gr, size=int(0.05 * SR)), size=int(0.05 * SR))
        gr = np.maximum(gr, rel * 0.9)
        st = st * (1 - gr)[:, None]
    return np.clip(st, -ceil, ceil)



# =====================================================================
# TEXT / EQUATION LAYERS (cached)
# =====================================================================
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


_FONTS = {}


def font(size, bold=True):
    k = (int(size), bold)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, int(size))
    return _FONTS[k]


def mathtext_png(line):
    fig = plt.figure(figsize=(W / EQ_DPI, EQ_BLOCK_H / EQ_DPI), dpi=EQ_DPI)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, line, ha="center", va="center", fontsize=EQ_FONTSIZE, color=EQ_COLOR,
             path_effects=[pe.withStroke(linewidth=EQ_STROKE_W, foreground=BG_COLOR)])
    buf = io.BytesIO()
    fig.savefig(buf, dpi=EQ_DPI, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").resize((W, EQ_BLOCK_H), Image.LANCZOS)


# ONE simple line per section

def wrap(draw, text, fnt, max_w):
    lines = []
    for para in text.split("\n"):
        words = para.split()
        if draw.textlength(para, font=fnt) <= max_w:
            lines.append(para)
            continue
        best = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            m = max(draw.textlength(a, font=fnt), draw.textlength(b, font=fnt))
            if best is None or m < best[0]:
                best = (m, a, b)
        assert best and best[0] <= max_w, f"text needs >2 lines: {para!r}"
        lines += [best[1], best[2]]
    return lines


def text_layer(text, size, color, bold, max_w):
    fnt = font(size, bold)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = wrap(tmp, text, fnt, max_w)
    lh = int(size * 1.18)
    img = Image.new("RGBA", (W, lh * len(lines) + 2 * TEXT_STROKE + 12), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((W // 2, TEXT_STROKE + i * lh), ln, font=fnt, fill=color, anchor="ma",
               stroke_width=TEXT_STROKE, stroke_fill=BG_RGB)
    return img, len(lines)


LAYERS = {}



# =====================================================================
# DRAWING HELPERS (sim-local, additive on black)
# =====================================================================
def scol(c, a):
    r, g, b = hex_rgb(c) if isinstance(c, str) else c
    a = max(0.0, a)
    return (int(min(255, r * a)), int(min(255, g * a)), int(min(255, b * a)))


def glyph(d, ch, cx, cy, size, color, a=1.0):
    if a > 0.01:
        d.text((cx, cy), ch, font=font(size), fill=scol(color, a), anchor="mm")


def pill(d, x0, y0, x1, y1, color, a, outline=False, width=3):
    if a <= 0.01:
        return
    r = (y1 - y0) / 2
    if outline:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=scol(color, a), width=width)
    else:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=scol(color, a))


def syl_color(s):
    return C_SHORT if s == "L" else C_LONG


def hit_level(local_beats, onsets):
    for j, (on, du, _) in enumerate(onsets):
        if on <= local_beats < on + du:
            return j, math.exp(-(local_beats - on) * BEAT_S * HIT_FLASH_DECAY)
    return -1, 0.0


def lerp_col(c1, c2, f):
    a, b = np.array(hex_rgb(c1)), np.array(hex_rgb(c2))
    return tuple(int(v) for v in a + (b - a) * f)


def alpha_paste(base, lay, top_y, alpha):
    if alpha <= 0:
        return
    h = lay.shape[0]
    y0, y1 = max(top_y, 0), min(top_y + h, H)
    if y1 <= y0:
        return
    lay = lay[y0 - top_y:y1 - top_y]
    a = (lay[..., 3:4] / 255.0) * alpha
    base[y0:y1] = base[y0:y1] * (1 - a) + lay[..., :3] * a


def new_layer():
    L = Image.new("RGB", (SIM_W, SIM_H), (0, 0, 0))
    return L, ImageDraw.Draw(L)


def synth_audio():
    tt = np.arange(N_SAMP) / SR
    PHI = (1 + math.sqrt(5)) / 2
    fA = np.full(N_SAMP, DRONE_ROOT)
    fB = np.full(N_SAMP, DRONE_ROOT * 1.5)
    dr_env = np.ones(N_SAMP)
    drums_on = np.zeros(N_SAMP)
    tabla = np.zeros(N_SAMP)
    music = np.zeros(N_SAMP)

    def seg(b0, b1):
        return slice(int(b0 * BEAT_S * SR), int(b1 * BEAT_S * SR))

    # drone second note: ratio of consecutive counts as they are revealed
    for b0, b1, r in ((61, 78, 5 / 3), (78, 82, 8 / 5), (82, 86, 13 / 8), (86, END_HOOK_BEAT, PHI)):
        fB[seg(b0, b1)] = DRONE_ROOT * r
    drums_on[seg(72, SILENT_BEAT)] = 1.0
    pA = 2 * np.pi * np.cumsum(fA) / SR
    pB = 2 * np.pi * np.cumsum(fB) / SR
    for s in SCHED[1:]:
        i, n = int(s["t0"] * SR), int(SECTION_XFADE_S * SR)
        dr_env[i - n:i + n] *= 0.35 + 0.65 * np.abs(np.linspace(-1, 1, 2 * n))
    stack = lambda p: sum(np.sin(h * p) / h ** 1.4 for h in range(1, 6))
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * tt)
    drone_L = (stack(pA) + 0.72 * stack(pB)) * lfo
    drone_R = (stack(pA) * 0.9 + 0.8 * stack(pB)) * lfo
    sub = np.sin(pA / 2)

    def stroke(syl, dur_beats):
        return stroke_ta(dur_beats * BEAT_S) if syl == "L" else stroke_taa(dur_beats * BEAT_S)

    # the drum plays exactly what lights up
    for on, du, s in INTRO_HITS:
        add_at(tabla, on * BEAT_S, stroke(s, du))
    for n, i, on, unit in PLAYS:
        for o, du, s in pattern_onsets(COMPS[n][i], unit):
            add_at(tabla, (on + o) * BEAT_S, stroke(s, du))
    for b, n, i, s in APPENDS:
        add_at(tabla, b * BEAT_S, stroke(s, 0.5 if n == 4 else 0.25) * (1.0 if n == 4 else 0.8))
    # counts: pluck at 220 Hz x count; the sum chord uses the three counts as harmonics
    for n, b in BADGE_BEAT.items():
        add_at(music, b * BEAT_S, POP_GAIN * pluck(220.0 * COUNT[n] / (2 if COUNT[n] > 4 else 1), tau=0.3))
        if n >= 4:
            add_at(music, b * BEAT_S, CHIME_GAIN * bell(DRONE_ROOT, [COUNT[n - 2], COUNT[n - 1], COUNT[n]]))
    for k, v in enumerate(STRIP_VALUES):
        add_at(music, (STRIP_POP0 + k * STRIP_POP_STEP) * BEAT_S, POP_GAIN * pluck(DRONE_ROOT * v, tau=0.22))
    for b in TL_BEATS[:2]:
        add_at(music, b * BEAT_S, CHIME_GAIN * bell(4 * DRONE_ROOT, [1, 2, 3]))
    for b in (ASK_BEAT, ASK_BEAT + 4):
        add_at(music, b * BEAT_S, 0.8 * CHIME_GAIN * bell(2 * DRONE_ROOT, [1, 3, 8]))

    # groove
    kick = np.zeros(N_SAMP)
    kw = kick_wave()
    for tk in KICK_TIMES:
        add_at(kick, tk, kw)
    hlen = int(0.05 * SR)
    hat = np.diff(RNG.standard_normal(hlen + 1)) * np.exp(-np.arange(hlen) / SR / 0.010)
    clen = int(0.2 * SR)
    cn = RNG.standard_normal(clen)
    cn = cn - np.convolve(cn, np.ones(12) / 12, mode="same")
    clap = cn * np.exp(-np.arange(clen) / SR / 0.05)
    hats, claps = np.zeros(N_SAMP), np.zeros(N_SAMP)
    for b in range(72, SILENT_BEAT):
        add_at(hats, (b + 0.5) * BEAT_S, hat)
        if b >= 80 and b % 4 in (1, 3):
            add_at(claps, b * BEAT_S, clap)

    riser = np.zeros(N_SAMP)
    r0, r1 = int(RISER_BEATS[0] * BEAT_S * SR), int(RISER_BEATS[1] * BEAT_S * SR)
    rn = RNG.standard_normal(r1 - r0)
    rn = rn - np.convolve(rn, np.ones(24) / 24, mode="same")
    x = np.linspace(0, 1, r1 - r0)
    sweep = np.sin(2 * np.pi * np.cumsum(220 * 2 ** (3 * x)) / SR)
    riser[r0:r1] = (0.8 * rn + 0.6 * sweep) * x ** 2.5

    beat_pos = np.full(N_SAMP, 10.0)
    idx = np.searchsorted(KICK_TIMES, tt, side="right") - 1
    ok = idx >= 0
    beat_pos[ok] = tt[ok] - KICK_TIMES[idx[ok]]
    pump = 1.0 - PUMP_DEPTH * np.exp(-beat_pos / PUMP_DECAY_S) * drums_on
    bed = dr_env * pump
    common = (SUB_GAIN * sub * bed * drums_on + TABLA_GAIN * tabla + music
              + KICK_GAIN * kick + HAT_GAIN * hats + CLAP_GAIN * claps + RISER_GAIN * riser)
    st = np.stack([DRONE_GAIN * drone_L * bed + common, DRONE_GAIN * drone_R * bed + common], axis=1)

    m = np.ones(N_SAMP)
    s0, s1 = int(SILENT_T0 * SR), int(SILENT_T1 * SR)
    nf = int(0.004 * SR)
    m[s0 - nf:s0] = np.linspace(1, 0, nf)
    m[s0:s1] = 0.0
    m[s1:s1 + nf] = np.linspace(0, 1, nf)
    st *= m[:, None]

    st /= np.max(np.abs(st)) / 0.5
    for _ in range(4):
        write_wav(TMP_AUDIO, st)
        meas = ebur_measure(TMP_AUDIO)
        print(f"  audio pass: I={meas['I']:.2f} LUFS TP={meas['TP']:.2f} dBTP")
        if abs(meas["I"] - TARGET_LUFS) < 0.3 and meas["TP"] <= TARGET_TP - 0.8:
            break
        st = st * 10 ** ((TARGET_LUFS - meas["I"]) / 20)
        st = limiter(st, LIMIT_CEIL_DB)
        st *= m[:, None]
    write_wav(TMP_AUDIO, st)


# =====================================================================
# TOP BAND / CHIP / CAPTION CACHES
# =====================================================================
EQ_LINES = {
    "beats": r"$\mathrm{short} = 1\ \mathrm{beat}\ \ \cdot\ \ \mathrm{long} = 2\ \mathrm{beats}$",
    "s4q":   r"$3 + 2 = \ ?$",
    "s4":    r"$3 + 2 = 5$",
    "s5q":   r"$5 + 3 = \ ?$",
    "s5":    r"$5 + 3 = 8$",
    "seq":   r"$1,\ 2,\ 3,\ 5,\ 8,\ 13\ \ldots$",
    "ask":   r"$1,\ 2,\ 3,\ 5,\ 8,\ 13,\ ?,\ ?$",
}


def eq_key_at(time_s):
    b = time_s / BEAT_S
    if b < 40 or b >= END_HOOK_BEAT:
        return "beats"
    if b < BADGE_BEAT[4]:
        return "s4q"
    if b < WHY[5]["sA"]:
        return "s4"
    if b < COLS_OUT_BEAT:
        return "s5"
    if b < ASK_BEAT:
        return "seq"
    return "ask"


EQ_CHANGES = [fi / FPS for fi in range(1, N_FRAMES) if eq_key_at(fi / FPS) != eq_key_at((fi - 1) / FPS)]


def chip_at(time_s):
    b = time_s / BEAT_S
    return SOURCE_CHIP if TL_BEATS[0] <= b < ASK_BEAT else None


LAYERS = {}


def build_caches():
    eq = {k: np.asarray(mathtext_png(v), dtype=np.float32) for k, v in EQ_LINES.items()}
    chip = np.asarray(text_layer(SOURCE_CHIP, CHIP_FONT_SIZE, hex_rgb(C_LONG), True, STORY_MAX_W)[0],
                      dtype=np.float32)
    story = {}
    for st, txt, _ in TEXT_EVENTS:
        img, nl = text_layer(txt, STORY_FONT_SIZE, (255, 255, 255), True, STORY_MAX_W)
        assert nl <= 2 and STORY_Y + img.size[1] <= SAFE_BOTTOM + 4, f"caption too tall: {txt!r}"
        story[txt] = np.asarray(img, dtype=np.float32)
    sign = np.asarray(text_layer(SIGN_OFF, SIGN_FONT_SIZE, (255, 255, 255), False, STORY_MAX_W)[0],
                      dtype=np.float32)
    LAYERS.update(eq=eq, chip=chip, story=story, sign=sign)


# =====================================================================
# VISUALS (sim-local, additive on black; tb = time in beats)
# =====================================================================
def syl_color(s):
    return C_SHORT if s == "L" else C_LONG


def block(d, x0, x1, yc, h, s, a, flash=0.0):
    if a <= 0.01:
        return
    col = lerp_col(syl_color(s), "#ffffff", min(0.6, 0.6 * flash)) if flash > 0 else hex_rgb(syl_color(s))
    d.rounded_rectangle([x0, yc - h / 2, x1, yc + h / 2], radius=min(14, h / 3),
                        fill=scol(col, a * (1 + 0.5 * flash)))


def draw_intro(d, tb, a=1.0, labels=True):
    U = IN_UNIT_PX
    left = (SIM_W - (3 * U + IN_GAP)) / 2
    xs = {"L": (left, left + U), "G": (left + U + IN_GAP, left + 3 * U + IN_GAP)}
    lit, fl = None, 0.0
    for on, du, s in INTRO_HITS:
        if on <= tb < on + du:
            lit, fl = s, math.exp(-(tb - on) * BEAT_S * HIT_FLASH_DECAY)
    talk = None
    if labels and INTRO_SHORT_TALK[0] <= tb < INTRO_SHORT_TALK[1]:
        talk = "L"
    elif labels and INTRO_LONG_TALK[0] <= tb < INTRO_LONG_TALK[1]:
        talk = "G"
    la = ease((tb - (INTRO_LABEL_BEAT - 0.5)) / 0.5) if labels else 0.0
    for s in "LG":
        x0, x1 = xs[s]
        em = 1.0 if talk in (None, s) else DIM_OTHER
        on = lit == s
        base = 0.72 + 0.28 * on
        block(d, x0 + 6, x1 - 6, IN_Y, IN_BLOCK_H, s, a * em * base, fl if on else 0.0)
        glyph(d, "ta" if s == "L" else "taa", (x0 + x1) / 2, IN_SYL_Y, 70,
              syl_color(s) if on else C_DIM, a * em * (1.0 if on else 0.75))
        if la > 0:
            n = UNITS[s]
            yr = IN_RULER_Y
            d.line([x0, yr, x1, yr], fill=scol(C_DIM, 0.6 * la * a * em), width=3)
            for k in range(n + 1):
                xx = x0 + k * (x1 - x0) / n
                d.line([xx, yr - 12, xx, yr + 12], fill=scol(C_DIM, 0.9 * la * a * em), width=3)
            glyph(d, "1 beat" if s == "L" else "2 beats", (x0 + x1) / 2, IN_LABEL_Y, 56, (255, 255, 255),
                  la * a * em)


def layout(cols, U, reserve=0):
    widths = [max(n * U, COL_MIN_SLOT) for n in cols]
    total = sum(widths) + COL_GAP * max(len(cols) - 1, 0)
    x = (SIM_W - total - reserve) / 2
    out = {}
    for n, w in zip(cols, widths):
        out[n] = x + w / 2
        x += w + COL_GAP
    return out


def col_geom(tb):
    p = max(i for i, L in enumerate(LAYOUTS) if L[0] <= tb)
    cur = LAYOUTS[p]
    prv = LAYOUTS[p - 1] if p > 0 else cur
    f = ease((tb - cur[0]) / LAYOUT_MOVE_BEATS)
    lc, lp = layout(cur[1], cur[2], cur[3]), layout(prv[1], prv[2], prv[3])
    U = prv[2] + (cur[2] - prv[2]) * f
    geom = {}
    for n in sorted(set(cur[1]) | set(prv[1])):
        xp = lp.get(n, lc.get(n))
        xc = lc.get(n, xp)
        a = 1.0 if (n in cur[1] and n in prv[1]) else (f if n in cur[1] else 1 - f)
        geom[n] = (xp + (xc - xp) * f, a)
    return geom, U


def row_pitch(U):
    return min(max(U * ROW_PITCH_K, ROW_PITCH_MIN), ROW_PITCH_MAX)


def emph(n, part, tb):
    """how strongly a column (or a WHY part of it) is talked about right now"""
    if 30 <= tb < 40:
        return 1.0 if n == 3 else 0.6
    if 40 <= tb < 51:
        return 1.0 if (n == 3 or (n == 4 and part == "A")) else 0.4
    if 51 <= tb < 61:
        return 1.0 if (n == 2 or (n == 4 and part == "B")) else 0.5
    if 61 <= tb < 72:
        return 1.0 if n == 4 else 0.5
    if 72 <= tb < 75:
        return 1.0 if (n == 4 or (n == 5 and part == "A")) else 0.45
    if 75 <= tb < 78:
        return 1.0 if (n == 3 or (n == 5 and part == "B")) else 0.6
    if 78 <= tb < 82:
        return 1.0 if n == 5 else 0.55
    return 1.0


def outline_for(n, i, tb):
    """origin outline colour (rows copied from n-1: violet, from n-2: orange)"""
    if n == 3 and WHY[4]["sA"] <= tb < WHY[5]["sA"]:
        return C_FROM1
    if n == 2 and WHY[4]["sB"] <= tb < WHY[5]["sA"]:
        return C_FROM2
    if n == 4 and tb >= WHY[5]["sA"]:
        return C_FROM1
    if n == 3 and tb >= WHY[5]["sB"]:
        return C_FROM2
    if n in WHY:
        return C_FROM1 if i < COUNT[n - 1] else C_FROM2
    return None


def play_state(n, i, tb):
    for m, r, on, unit in PLAYS:
        if m == n and r == i:
            ons = pattern_onsets(COMPS[n][i], unit)
            for j, (o, du, s) in enumerate(ons):
                if on + o <= tb < on + o + du:
                    return j, math.exp(-(tb - on - o) * BEAT_S * HIT_FLASH_DECAY)
    return -1, 0.0


def first_play(n, i):
    return min(on for m, r, on, _ in PLAYS if m == n and r == i)


def draw_row(d, left, yc, pat, U, bh, a, lit=-1, fl=0.0, outline=None, pop=None):
    u = 0
    for j, s in enumerate(pat):
        ff = fl if j == lit else 0.0
        if pop is not None and j == len(pat) - 1:
            ff = max(ff, pop)
        block(d, left + u * U + 4, left + (u + UNITS[s]) * U - 4, yc, bh, s, a * (1.0 if j == lit else 0.85), ff)
        u += UNITS[s]
    if outline and a > 0.02:
        w = u * U
        d.rounded_rectangle([left - 8, yc - bh / 2 - 8, left + w + 8, yc + bh / 2 + 8], radius=12,
                            outline=scol(outline, a), width=OUTLINE_W)


def badge(d, n, cx, tb, a):
    b = BADGE_BEAT[n]
    if tb < b:
        return
    ba = ease((tb - b) / 0.25) * a
    fl = math.exp(-(tb - b) * BEAT_S * 4)
    if n < 4 or (n == 4 and tb >= WHY[5]["sA"]):     # plain count once it is a source column
        glyph(d, str(COUNT[n]), cx, BADGE_Y, 76, C_LONG, ba * (1 + 0.5 * fl))
    else:
        parts = [(str(COUNT[n - 1]), C_FROM1), ("+", (255, 255, 255)), (str(COUNT[n - 2]), C_FROM2),
                 ("=", (255, 255, 255)), (str(COUNT[n]), C_LONG)]
        f = font(64)
        widths = [f.getlength(p) for p, _ in parts]
        total = sum(widths) + 18 * (len(parts) - 1)
        x = cx - total / 2
        for (p, c), w in zip(parts, widths):
            glyph(d, p, x + w / 2, BADGE_Y, 64, c, ba * (1 + 0.4 * fl))
            x += w + 18
    glyph(d, "way" if COUNT[n] == 1 else "ways", cx, BADGE_Y + 58, 32, C_DIM, ba)


def draw_columns(d, tb, a_all=1.0):
    geom, U = col_geom(tb)
    P = row_pitch(U)
    bh = P * BLOCK_H_K
    for n, (cx, ca) in geom.items():
        a = ca * a_all
        if a <= 0.01:
            continue
        if n in WHY and tb < WHY[n]["sA"]:
            continue
        e = max(emph(n, "A", tb), emph(n, "B", tb)) if n in WHY else emph(n, None, tb)
        glyph(d, f"{n} beat{'s' if n > 1 else ''}", cx, COL_HEAD_Y, 40, (255, 255, 255), a * (0.5 + 0.5 * e))
        left = cx - n * U / 2
        for i, pat in enumerate(COMPS[n]):
            yc = row_y(n, i, P)
            if n in WHY:
                w = WHY[n]
                part = "A" if i < COUNT[n - 1] else "B"
                src_n, src_i = (n - 1, i) if part == "A" else (n - 2, i - COUNT[n - 1])
                s0, ds = (w["sA"], w["dA"]) if part == "A" else (w["sB"], w["dB"])
                t_app = (w["tA"] + w["tB"])[i]
                if tb < s0:
                    continue
                f = (tb - s0) / ds
                if src_n in geom:
                    scx = geom[src_n][0]
                    sleft = scx - src_n * U / 2
                    sy = row_y(src_n, src_i, P)
                else:
                    sleft, sy = left, yc
                # route: first down/up to the target row height, then across (never diagonally
                # through other columns); while moving, drawn faint and without glow
                fy, fx = ease(2 * f), ease(2 * f - 1)
                x = sleft + (left - sleft) * fx
                y = sy + (yc - sy) * fy
                moving = 0 < f < 1
                shown = pat if tb >= t_app else pat[:-1]
                pop = math.exp(-(tb - t_app) * BEAT_S * 4) if tb >= t_app else None
                lit, fl = play_state(n, i, tb)
                ee = emph(n, part, tb)
                if moving and NOGLOW["d"] is not None:
                    draw_row(NOGLOW["d"], x, y, shown, U, bh, MOVING_ALPHA * a * ee, -1, 0.0,
                             outline_for(n, i, tb), None)
                else:
                    draw_row(d, x, y, shown, U, bh, a * ee, lit, fl, outline_for(n, i, tb), pop)
            else:
                if tb < first_play(n, i):
                    continue
                ra = ease((tb - first_play(n, i)) / 0.2)
                lit, fl = play_state(n, i, tb)
                draw_row(d, left, yc, pat, U, bh, a * ra * e, lit, fl, outline_for(n, i, tb))
        if n in WHY and tb < (WHY[5]["sA"] if n == 4 else TOTAL_BEATS):
            brackets(d, n, left + n * U, P, bh, tb, a)
        badge(d, n, cx, tb, a * (0.6 + 0.4 * e))


def brackets(d, n, right, P, bh, tb, a):
    """bold coloured bracket + label to the right of each origin group"""
    w = WHY[n]
    groups = [(0, COUNT[n - 1] - 1, w["tA"][-1], C_FROM1, "+ short"),
              (COUNT[n - 1], COUNT[n] - 1, w["tB"][-1], C_FROM2, "+ long")]
    for i0, i1, t_on, col, label in groups:
        if tb < t_on:
            continue
        ba = ease((tb - t_on) / 0.4) * a
        y0 = row_y(n, i0, P) - bh / 2 - 6
        y1 = row_y(n, i1, P) + bh / 2 + 6
        x = right + BRACKET_DX
        d.line([x - 12, y0, x, y0, x, y1, x - 12, y1], fill=scol(col, ba), width=BRACKET_W, joint="curve")
        lw = font(BRACKET_LABEL_SIZE).getlength(label)
        glyph(d, label, x + 14 + lw / 2, (y0 + y1) / 2, BRACKET_LABEL_SIZE, col, ba)


def row_y(n, i, P):
    gap = GROUP_GAP if (n in WHY and i >= COUNT[n - 1]) else 0
    return ROW_Y0 + i * P + gap


def draw_strip(d, tb, y, a, ask=False):
    vals = [str(v) for v in STRIP_VALUES] + (["?", "?"] if ask else [])
    nbox = len(vals)
    dx = ST_DX if not ask else 112
    bw = ST_BOX if not ask else 96
    x0 = SIM_W / 2 - (nbox - 1) * dx / 2 - (0 if ask else dx / 2 * 0.6)
    for k, v in enumerate(vals):
        t_in = STRIP_POP0 + k * STRIP_POP_STEP if not ask else ASK_BEAT
        if tb < t_in:
            continue
        ka = ease((tb - t_in) / 0.25) * a
        fl = math.exp(-(tb - t_in) * BEAT_S * 4) if not ask else 0.0
        x = x0 + k * dx
        last = ask and k == nbox - 1
        pulse = math.exp(-((tb % 2.0) * BEAT_S) * 4) if last else 0.0
        col = C_ASK if last else (C_DIM if v == "?" else C_LONG)
        d.rounded_rectangle([x - bw / 2, y - bw / 2, x + bw / 2, y + bw / 2], radius=16,
                            outline=scol(col, ka * (0.85 + 0.5 * fl + 0.5 * pulse)), width=4 if last else 3)
        glyph(d, v, x, y, 58 if len(v) < 2 else 50, (255, 255, 255) if v != "?" else col,
              ka * (0.95 + 0.6 * fl + 0.5 * pulse))
        glyph(d, f"{k + 1} beat{'s' if k else ''}", x, y + bw / 2 + 28, 24, C_ASK if last else C_DIM, 0.85 * ka)
    if not ask and tb >= STRIP_DOTS_BEAT:
        x = x0 + nbox * dx - dx * 0.35
        glyph(d, "…", x, y, 60, (255, 255, 255), ease((tb - STRIP_DOTS_BEAT) / 0.3) * a)


def draw_timeline(d, tb, a=1.0):
    la = ease((tb - (TL_BEATS[0] - 0.2)) / 0.3) * a
    d.line([TL_X[0] - 70, TL_Y, TL_X[1] + 70, TL_Y], fill=scol(C_DIM, 0.6 * la), width=3)
    for j, (name, date, col) in enumerate(TIMELINE):
        bt = TL_BEATS[j]
        if tb < bt:
            continue
        aa = ease((tb - bt) / 0.3) * a
        fl = math.exp(-(tb - bt) * BEAT_S * 5)
        x = TL_X[j]
        r = 16 + 6 * fl
        d.ellipse([x - r, TL_Y - r, x + r, TL_Y + r], fill=scol(col, aa))
        glyph(d, name, x, TL_Y - 66, 50, col, aa)
        glyph(d, date, x, TL_Y + 62, 44, (255, 255, 255), 0.9 * aa)
        if j == 1:
            xa, xb = TL_X[0] + 34, x - 34
            d.line([xa, TL_Y, xb, TL_Y], fill=scol(col, aa), width=5)
            d.polygon([(xb, TL_Y), (xb - 18, TL_Y - 11), (xb - 18, TL_Y + 11)], fill=scol(col, aa))
    if tb >= TL_BEATS[2]:
        aa = ease((tb - TL_BEATS[2]) / 0.4) * a
        yb = TL_Y + 130
        d.line([TL_X[0], yb - 16, TL_X[0], yb, TL_X[1], yb, TL_X[1], yb - 16], fill=scol(C_ASK, aa), width=4)
        glyph(d, "~500 years", SIM_W / 2, yb + 44, 52, C_ASK, aa)


def draw_ask(d, tb, a=1.0):
    draw_strip(d, tb, 330, a, ask=True)
    x0 = SIM_W / 2 - 4 * 100
    for k in range(8):
        cx = x0 + k * 100 + 50
        d.rounded_rectangle([cx - 40, 570 - 24, cx + 40, 570 + 24], radius=12, outline=scol(C_ASK, a * 0.85), width=3)
    glyph(d, "8 beats  =  ? ways", SIM_W / 2, 670, 54, C_ASK, a)


# =====================================================================
# FRAME RENDERING
# =====================================================================
NOGLOW = {"d": None, "L": None}
MOVING_ALPHA = 0.5


def sim_image(time_s):
    tb = time_s / BEAT_S
    layers = []
    NOGLOW["L"], NOGLOW["d"] = new_layer()

    def lay(fn, *args):
        L, d = new_layer()
        fn(d, *args)
        layers.append(L)

    if tb < 20:
        lay(draw_intro, tb)
    elif tb < COLS_OUT_BEAT + 1:
        fo = 1 - ease((tb - COLS_OUT_BEAT) / 0.8)
        if tb < 20.6:   # intro blocks fade as the columns arrive
            lay(draw_intro, tb, 1 - ease((tb - 20) / 0.6))
        lay(draw_columns, tb, fo)
        if tb >= STRIP_POP0:
            lay(draw_strip, tb, ST_Y, 1.0)
    elif tb < TL_BEATS[0]:
        lay(draw_strip, tb, ST_Y, 1.0)
    elif tb < ASK_BEAT:
        f = ease((tb - TL_BEATS[0]) / 1.0)
        lay(draw_strip, tb, ST_Y + (200 - ST_Y) * f, 1 - 0.5 * f)
        lay(draw_timeline, tb)
    else:
        f = ease((tb - END_HOOK_BEAT) / 1.0)
        fa = ease((tb - ASK_BEAT) / 0.6)
        if f < 1:
            lay(draw_ask, tb, fa * (1 - f))
            if tb < ASK_BEAT + 0.6:
                lay(draw_timeline, tb, 1 - fa)
        if f > 0:
            lay(draw_intro, tb, f, False)
    arr = None
    for L in layers:
        a = np.asarray(L, dtype=np.float32)
        arr = a if arr is None else arr + a
    ng = np.asarray(NOGLOW["L"], dtype=np.float32)
    NOGLOW["d"] = None
    return arr, ng


def render_frame(fi):
    time_s = fi / FPS
    frame = np.empty((H, W, 3), dtype=np.float32)
    frame[:] = BG_RGB
    silent = SILENT_T0 <= time_s < SILENT_T1
    i = np.searchsorted(KICK_TIMES, time_s + 1e-9, side="right") - 1
    flash = KICK_FLASH * math.exp(-(time_s - KICK_TIMES[i]) * KICK_FLASH_DECAY) if i >= 0 and not silent else 0.0
    intensity = (1.0 + flash) * (SILENT_DIM if silent else 1.0)
    arr, ng = sim_image(time_s)
    if arr is not None:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        small = img.resize((SIM_W // GLOW_DOWNSCALE, SIM_H // GLOW_DOWNSCALE), Image.BILINEAR)
        glow = small.filter(ImageFilter.GaussianBlur(GLOW_RADIUS)).resize((SIM_W, SIM_H), Image.BILINEAR)
        frame[SIM_Y0:SIM_Y0 + SIM_H, SIM_X0:SIM_X0 + SIM_W] += (arr + GLOW_GAIN * np.asarray(glow, np.float32)) * intensity
    frame[SIM_Y0:SIM_Y0 + SIM_H, SIM_X0:SIM_X0 + SIM_W] += ng * intensity
    fa = 1.0
    for tc in EQ_CHANGES:
        if tc - EQ_FADE_S <= time_s < tc:
            fa = (tc - time_s) / EQ_FADE_S
        elif tc <= time_s < tc + EQ_FADE_S:
            fa = (time_s - tc) / EQ_FADE_S
    alpha_paste(frame, LAYERS["eq"][eq_key_at(time_s)], EQ_Y_TOP, fa * (0.45 if silent else 1.0))
    if chip_at(time_s):
        alpha_paste(frame, LAYERS["chip"], CHIP_Y, 1.0)
    for st, txt, en in TEXT_EVENTS:
        if st <= time_s < en:
            a_in = 1.0 if st == 0 else np.clip((time_s - st) / TEXT_FADE_S, 0, 1)
            a_out = np.clip((en - time_s) / TEXT_FADE_S, 0, 1) if en <= DURATION else 1.0
            alpha_paste(frame, LAYERS["story"][txt], STORY_Y, float(min(a_in, a_out)))
    alpha_paste(frame, LAYERS["sign"], SIGN_Y, 0.6)
    return np.clip(frame, 0, 255).astype(np.uint8)


# =====================================================================
# REVIEW OUTPUTS
# =====================================================================
SHEET_TIMES = [("hook", 0.0), ("1-2 beats", 13.5), ("why +short", 24.6), ("3+2=5", 34.0),
               ("sequence", 44.0), ("ending", DURATION - 1.0 / FPS)]

def contact_sheet(times=SHEET_TIMES, path=OUT_SHEET):
    tiles = []
    for name, ts in times:
        im = Image.fromarray(render_frame(int(round(ts * FPS))))
        d = ImageDraw.Draw(im)
        d.rectangle([0, int(H * 0.8), W, int(H * 0.8) + 2], fill=(90, 30, 30))
        d.text((20, H - 80), f"{name}  t={ts:.2f}s", font=font(48), fill=(255, 80, 80))
        tiles.append(im.resize((W // 2, H // 2), Image.LANCZOS))
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W // 2, rows * H // 2), BG_RGB)
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * W // 2, (i // cols) * H // 2))
    sheet.save(path, quality=90)


def write_beats_json():
    out = dict(duration_s=round(DURATION, 3), bpm=BPM, sections=[], captions=[], top_band=[],
               silent_beat=[round(SILENT_T0, 3), round(SILENT_T1, 3)])
    for s in SCHED:
        out["sections"].append(dict(key=s["key"], t0=round(s["t0"], 3), t1=round(s["t1"], 3)))
    for st, txt, en in TEXT_EVENTS:
        en = min(en, DURATION)
        words = len(txt.split())
        out["captions"].append(dict(t0=round(st, 3), t1=round(en, 3), seconds=round(en - st, 2),
                                    words=words, words_per_s=round(words / (en - st), 2),
                                    text=txt.replace("\n", " ")))
    last, t0 = None, 0.0
    for fi in range(N_FRAMES + 1):
        v = eq_key_at(fi / FPS) if fi < N_FRAMES else None
        if v != last:
            if last is not None:
                out["top_band"].append(dict(t0=round(t0, 3), t1=round(fi / FPS, 3), text=EQ_LINES[last]))
            last, t0 = v, fi / FPS
    with open(OUT_BEATS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def audio_report(video):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-map", "0:a",
                        "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    summ = r.stderr[r.stderr.rfind("Summary:"):]
    meas = ebur_measure(video)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-map", "0:a", "-f", "f32le", "-ac", "2",
                          "-ar", str(SR), "-"], capture_output=True).stdout
    a = np.frombuffer(dec, dtype=np.float32).reshape(-1, 2)

    def rms_db(x):
        v = np.sqrt(np.mean(x.astype(np.float64) ** 2)) if len(x) else 0
        return 20 * math.log10(v) if v > 0 else -float("inf")
    lines = [f"File: {os.path.basename(video)}",
             f"Integrated loudness: {meas['I']:.2f} LUFS   (target {TARGET_LUFS})",
             f"True peak:           {meas['TP']:.2f} dBTP  (target <= {TARGET_TP})",
             f"Loudness range:      {meas['LRA']:.2f} LU", "", "Per-section RMS (dBFS):"]
    for s in SCHED:
        seg = a[int(s["t0"] * SR):int(s["t1"] * SR)]
        lines.append(f"  {s['key']:<8} {s['t0']:6.2f}-{s['t1']:6.2f}s  RMS {rms_db(seg):7.2f}  "
                     f"peak {20 * math.log10(max(np.max(np.abs(seg)), 1e-12)):7.2f}")
    m = int(0.01 * SR)
    seg = a[int(SILENT_T0 * SR) + m:int(SILENT_T1 * SR) - m]
    pk = np.max(np.abs(seg)) if len(seg) else 0
    lines += ["", f"Silent beat {SILENT_T0:.2f}-{SILENT_T1:.2f}s (inner, 10 ms margins): "
                  f"RMS {rms_db(seg):.1f} dBFS, peak {20 * math.log10(max(pk, 1e-12)):.1f} dBFS",
              "", "ffmpeg ebur128 summary:", summ.strip()]
    with open(OUT_LEVELS, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


def encode_upload():
    kbps = int(UPLOAD_TARGET_MB * 1e6 * 8 / DURATION / 1000) - UPLOAD_AUDIO_K
    plog = os.path.join(OUT_DIR, "_x264pass")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_VIDEO, "-c:v", "libx264", "-preset", "medium",
            "-tune", "animation", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-passlogfile", plog]
    subprocess.run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull], check=True)
    subprocess.run(base + ["-pass", "2", "-c:a", "aac", "-b:a", f"{UPLOAD_AUDIO_K}k",
                           "-movflags", "+faststart", OUT_UPLOAD], check=True)
    for f in os.listdir(OUT_DIR):
        if f.startswith("_x264pass"):
            os.remove(os.path.join(OUT_DIR, f))
    print(f"upload: {kbps}k video -> {os.path.getsize(OUT_UPLOAD) / 1e6:.2f} MB")


def main():
    print(f"Duration {DURATION:.2f}s, {N_FRAMES} frames, BPM {BPM}")
    build_caches()
    write_beats_json()
    if "--sheet" in sys.argv:
        contact_sheet()
        return
    synth_audio()
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", TMP_AUDIO,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT_VIDEO]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(N_FRAMES):
        proc.stdin.write(render_frame(fi).tobytes())
        if fi % 300 == 0:
            print(f"frame {fi}/{N_FRAMES}", flush=True)
    proc.stdin.close()
    proc.wait()
    os.remove(TMP_AUDIO)
    print("Saved", OUT_VIDEO)
    contact_sheet()
    audio_report(OUT_VIDEO)
    if MAKE_UPLOAD:
        encode_upload()


if __name__ == "__main__":
    main()
