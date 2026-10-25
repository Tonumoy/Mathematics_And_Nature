"""
Coded Cosmos — "Count the petals."  (Ancient Code: the 7 chakras as a mandala)
------------------------------------------------------------------------------
In the Sat-cakra-nirupana (Purnananda, 1577; tr. Woodroffe, "The Serpent Power", 1919)
the six lotuses up the spine have 4, 6, 10, 12, 16 and 2 petals, and every petal
carries one Sanskrit letter:  4+6+10+12+16+2 = 50 letters (the matrika).
The crown lotus (Sahasrara) has 1000 petals: the same 50 letters, 20 times.
Colours follow the verses exactly (petals / letters): crimson / gold, vermilion / lightning,
rain-cloud / blue lotus, bandhuka red / vermilion, smoky purple / crimson, moon white / white;
crown whiter than the full Moon with young-Sun filaments. Not the modern rainbow scheme.

Every ring is one polar curve:  r(theta) = R + a |sin(n theta / 2)|^p  (p = 0.6, Ajna wings p = 3)
Music comes from the same numbers: a ring with n petals plays n x 55 Hz
(the harmonic series of A1), one pluck per petal.  100% original audio.

Run:  python chakra_mandala.py          (add --sheet for the preview sheet only)
  -> chakra_mandala.mp4, chakra_mandala_upload.mp4, preview_frames.jpg,
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
# GLOBALS
# =====================================================================
OUT_DIR          = os.path.dirname(os.path.abspath(__file__))
SLUG             = "chakra_mandala"
OUT_VIDEO        = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD       = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET        = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_LEVELS       = os.path.join(OUT_DIR, "audio_levels.txt")
OUT_BEATS        = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO        = os.path.join(OUT_DIR, "_chakra_audio.wav")
MAKE_UPLOAD      = True
UPLOAD_TARGET_MB = 9.0
UPLOAD_AUDIO_K   = 160

W, H             = 1080, 1920
FPS              = 30
BG_COLOR         = "#06050a"
BG_RGB           = (6, 5, 10)

# --- the lotuses (name, petals, colour named in the text) ---
# colours exactly as the verses give them (Avalon's translation, wisdomlib.org):
#   lotus/petals                                   letters on the petals
RINGS = [
    ("Mūlādhāra",    4,  "#c8143c", "#ffc93c"),   # v4: crimson of the red lotus       | shining gold
    ("Svādhiṣṭhāna", 6,  "#f0522a", "#fff6b0"),   # v14: beautiful vermilion           | colour of lightning
    ("Maṇipūra",     10, "#4e6496", "#7fb2ff"),   # v19: heavy-laden rain-clouds (dark)| blue lotus
    ("Anāhata",      12, "#ff2430", "#ff6a3a"),   # v22: bandhuka flower (deep red)    | vermilion
    ("Viśuddha",     16, "#8c7099", "#ff3a5a"),   # v28: smoky purple                  | crimson vowels
    ("Ājñā",         2,  "#f2f4ff", "#ffffff"),   # v32: white like the moon           | white
]
CROWN_NAME       = "Sahasrāra"
CROWN_N          = 1000
CROWN_COL        = "#f4f6ff"      # v40: lustrous, whiter than the full Moon
CROWN_FIL_COL    = "#ffb347"      # v40: filaments tinged with the colour of the young Sun
RING_R           = [(58, 118), (122, 178), (182, 236), (240, 292), (296, 344), (300, 412)]   # Ajna wings start behind the throat lotus
CROWN_R          = (416, 452)
GEOM_SCALE       = 0.88           # keeps the crown glow inside the sim box
PETAL_POW        = 0.6            # r = R + a |sin(n th / 2)|^p : p = 0.6 for round lotus petals
WING_POW         = 3.0            # p = 3 for the two narrow Ajna wings
ECHO_FRAC        = 0.55           # inner echo curve at R + ECHO_FRAC * a
PTS_PER_PETAL    = 28
CROWN_PTS        = 6
SPIN_TURNS       = 2              # base drift in petal-widths; cancels the hook/loop wind-up (-2) so letters return to the same petal
CROWN_SPIN       = 20
SS               = 2              # supersample
LINE_W           = 5              # at SS
CROWN_LINE_W     = 2
FILL_A           = 0.16
SQUARE_COL       = "#ffd166"      # earth square (yantra) inside the root lotus
HOOK_TURNS       = 1              # hook/loop counter-spin: each ring turns this many petal-widths (integer = seamless)
HOOK_PULSE       = 0.10           # petals breathe on every kick in the hook and the loop
# --- letters on the petals (Sat-cakra-nirupana: 50 letters, root -> brow) ---
LETTERS = [
    ["व", "श", "ष", "स"],
    ["ब", "भ", "म", "य", "र", "ल"],
    ["ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ"],
    ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ"],
    ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ", "अं", "अः"],
    ["ह", "क्ष"],
]
FONT_DEVA        = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"   # OFL
LETTER_PX        = 30             # at 1x
LETTER_COL       = (255, 238, 200)
LETTER_R_FRAC    = 0.40           # letter sits at R + 0.40 a
AJNA_LETTER_R    = 380            # brow wings: letter on the visible part of the wing

# --- timing (beats) ---
BPM              = 120
BEATS_PER_BAR    = 4
SR               = 44100
TOTAL_BEATS      = 84             # 42 s
SECTIONS = [
    ("hook", 0, 8), ("collapse", 8, 10), ("build", 10, 46), ("riser", 46, 50),
    ("silence", 50, 51), ("payoff", 51, 52), ("crown", 52, 60), ("hold", 60, 72), ("loop", 72, 84),
]
COLLAPSE         = (8, 10)
BUILD0           = 10
RING_BEATS       = 6              # one lotus every 6 beats
REVEAL_BEATS     = 4              # petals pop in over the first 4 beats of a lotus
RISER_BEATS      = (46, 50)
SILENT_BEAT      = 50
PAYOFF_BEAT      = 51
CROWN_BEATS      = (52, 60)

STORY = [
    (0,  "Count the petals."),
    (8,  "6 lotuses up the spine.\nOne letter on each petal."),
    (22, "Keep counting,\nit adds up."),
    (34, "Throat lotus: 16 petals,\nall 16 vowels."),
    (51, "50 petals.\n50 letters."),
    (58, "Crown: 1000 petals.\n50 letters, 20 times."),
    (66, "Colours from the text.\nNot a rainbow."),
    (74, "Ancient Code.\nOld texts, real maths."),
]
MIN_CAPTION_S    = 3.0
SERIES_CHIP      = "ANCIENT CODE"
SOURCE_CHIP      = "Ṣaṭ-cakra-nirūpaṇa, 1577"

C_HERO           = "#ffd166"

# --- layout (pixels) ---
SIGN_Y           = 46
EQ_Y_TOP         = 120
EQ_BLOCK_H       = 190
CHIP_Y           = 318
SIM_CY_FRAC      = 0.47
SIM_W, SIM_H     = 960, 900
SIM_X0           = (W - SIM_W) // 2
SIM_Y0           = int(H * SIM_CY_FRAC) - SIM_H // 2
STORY_Y          = SIM_Y0 + SIM_H + 8
STORY_MAX_W      = 900
SAFE_BOTTOM      = int(H * 0.80)
CX, CY           = SIM_W // 2, SIM_H // 2

GLOW_DOWNSCALE   = 4
GLOW_RADIUS      = 5
GLOW_GAIN        = 1.2
KICK_FLASH       = 0.16
KICK_FLASH_DECAY = 8.0
SILENT_DIM       = 0.28

FONT_BOLD        = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG         = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
STORY_FONT_SIZE  = 64
CHIP_FONT_SIZE   = 38
SIGN_FONT_SIZE   = 32
TEXT_STROKE      = 5
TEXT_FADE_S      = 0.30
EQ_FONTSIZE      = 12
EQ_DPI           = 330
EQ_COLOR         = "#ffffffdd"
EQ_STROKE_W      = 3
EQ_FADE_S        = 0.15

# --- audio ---
BASE_HZ          = 55.0           # ring with n petals plays n * BASE_HZ
PLUCK_GAIN       = 0.24
BELL_GAIN        = 0.18
PAD_GAIN         = 0.10
SUB_GAIN         = 0.12
KICK_GAIN        = 0.34
HAT_GAIN         = 0.04
CLAP_GAIN        = 0.07
RISER_GAIN       = 0.14
SHIMMER_GAIN     = 0.05
PUMP_DEPTH       = 0.30
PUMP_DECAY_S     = 0.12
TARGET_LUFS      = -14.0
TARGET_TP        = -1.0
LIMIT_CEIL_DB    = -2.2
SIGN_OFF         = "@cosmoscoded"

# =====================================================================
# DERIVED
# =====================================================================
BEAT_S   = 60.0 / BPM
DURATION = TOTAL_BEATS * BEAT_S
N_FRAMES = int(round(DURATION * FPS))
N_SAMP   = int(round(DURATION * SR))
SCHED    = [dict(key=k, t0=b0 * BEAT_S, t1=b1 * BEAT_S) for k, b0, b1 in SECTIONS]
SILENT_T0 = SILENT_BEAT * BEAT_S
SILENT_T1 = SILENT_T0 + BEAT_S
PETALS   = [r[1] for r in RINGS]
TOTAL    = sum(PETALS)
assert TOTAL == 50 and CROWN_N == TOTAL * 20
assert [len(x) for x in LETTERS] == PETALS
CUM      = list(np.cumsum([0] + PETALS))
RING_START = [BUILD0 + k * RING_BEATS for k in range(len(RINGS))]
assert RING_START[-1] + RING_BEATS == RISER_BEATS[0]
PETAL_BEATS = [[RING_START[k] + j * REVEAL_BEATS / n for j in range(n)] for k, n in enumerate(PETALS)]

TEXT_EVENTS = []
prev = -1e9
for b, txt in STORY:
    st = max(b * BEAT_S, prev + MIN_CAPTION_S)
    TEXT_EVENTS.append([st, txt])
    prev = st
for i in range(len(TEXT_EVENTS)):
    TEXT_EVENTS[i].append(TEXT_EVENTS[i + 1][0] if i + 1 < len(TEXT_EVENTS) else DURATION + 1)


def kick_on(b):
    if SILENT_BEAT <= b < PAYOFF_BEAT:
        return False
    if b < 8 or 10 <= b < 46 or 51 <= b < 72:
        return True
    if 72 <= b < 84:
        return b % 2 == 0
    return False


KICK_TIMES = np.array([b * BEAT_S for b in range(TOTAL_BEATS) if kick_on(b)])
RNG = np.random.default_rng(50)
_FONTS = {}


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)

# =====================================================================
# HELPERS (reused from collatz_coral.py)
# =====================================================================
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


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


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


def scol(c, a):
    r, g, b = hex_rgb(c) if isinstance(c, str) else c
    a = max(0.0, a)
    return (int(min(255, r * a)), int(min(255, g * a)), int(min(255, b * a)))


def pill(d, x0, y0, x1, y1, color, a, outline=False, width=3):
    if a <= 0.01:
        return
    r = (y1 - y0) / 2
    if outline:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=scol(color, a), width=width)
    else:
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=scol(color, a))


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


def contact_sheet(times=None, path=OUT_SHEET):
    times = times or SHEET_TIMES
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



# =====================================================================
# GEOMETRY  (sim-local, drawn at SS x)
# =====================================================================
def phase0(n):
    return -math.pi / 2 if n == 2 else -math.pi / 2 - math.pi / n     # petal 0 points up (Ajna: left/right wings)


def hook_f(tb):
    """1 -> 0 over the hook, 0 -> -1 over the last 8 beats; integer jump = invisible, so the loop is seamless"""
    if tb < COLLAPSE[0]:
        return (1 - tb / COLLAPSE[0]) ** 2
    if tb >= TOTAL_BEATS - COLLAPSE[0]:
        return -((tb - (TOTAL_BEATS - COLLAPSE[0])) / COLLAPSE[0]) ** 2
    return 0.0


def hook_pulse(tb):
    if tb < COLLAPSE[0] or tb >= TOTAL_BEATS - COLLAPSE[0]:
        return 1.0 + HOOK_PULSE * math.exp(-(tb - math.floor(tb)) * 5)
    return 1.0


def spin(n, time_s, k, turns):
    if n == 2:
        return 0.0                                             # Ajna wings stay level
    d = 1 if k % 2 == 0 else -1
    tb = time_s / BEAT_S
    extra = d * HOOK_TURNS * (2 * math.pi / n) * hook_f(tb) * (CROWN_SPIN if n == CROWN_N else 1)
    return extra + d * turns * (2 * math.pi / n) * time_s / DURATION


def petal_pts(n, j, r0, r1, phi, scale, frac=1.0, pts=PTS_PER_PETAL):
    pw = WING_POW if n == 2 else PETAL_POW
    u = np.linspace(0, 1, pts)
    th = phase0(n) + phi + (j + u) * 2 * math.pi / n
    r = (r0 + frac * pulse_now * (r1 - r0) * np.abs(np.sin(math.pi * u)) ** pw) * scale * GEOM_SCALE
    x = CX * SS + SS * r * np.cos(th)
    y = CY * SS + SS * r * np.sin(th)
    return np.stack([x, y], 1)


def draw_ring(d, k, time_s, scale, alpha, n_vis, pop=None, bright=1.0):
    name, n, col, lcol = RINGS[k]
    r0, r1 = RING_R[k]
    phi = spin(n, time_s, k, SPIN_TURNS)
    rgb = hex_rgb(col)
    for j in range(n_vis):
        a = alpha * (pop[j] if pop is not None else 1.0)
        if a <= 0.01:
            continue
        outer = petal_pts(n, j, r0, r1, phi, scale)
        base = petal_pts(n, j, r0, r0, phi, scale)[::-1]
        poly = [tuple(p) for p in np.vstack([outer, base])]
        d.polygon(poly, fill=scol(rgb, FILL_A * a * bright))
        d.line([tuple(p) for p in outer], fill=scol(rgb, a * bright), width=LINE_W, joint="curve")
        echo = petal_pts(n, j, r0, r1, phi, scale, frac=ECHO_FRAC)
        d.line([tuple(p) for p in echo], fill=scol(rgb, 0.55 * a * bright), width=max(2, LINE_W - 2))
        la = a * min(1.0, max(0.0, (scale - 0.75) / 0.25))
        if la > 0.02:
            rc = AJNA_LETTER_R if n == 2 else r0 + LETTER_R_FRAC * (r1 - r0) * pulse_now
            th = phase0(n) + phi + (j + 0.5) * 2 * math.pi / n
            paste_letter(d._image, LETTERS[k][j], CX + rc * scale * GEOM_SCALE * math.cos(th),
                         CY + rc * scale * GEOM_SCALE * math.sin(th), la * min(bright, 1.3), hex_rgb(lcol))


LETTER_CACHE = {}


def paste_letter(img, ch, x, y, a, rgb):
    if ch not in LETTER_CACHE:
        f = ImageFont.truetype(FONT_DEVA, LETTER_PX * SS, layout_engine=ImageFont.Layout.RAQM)
        m = Image.new("L", (LETTER_PX * SS * 3, LETTER_PX * SS * 3), 0)
        ImageDraw.Draw(m).text((m.size[0] // 2, m.size[1] // 2), ch, font=f, fill=255, anchor="mm")
        LETTER_CACHE[ch] = m.crop(m.getbbox())
    m = LETTER_CACHE[ch]
    mm = m.point(lambda v: int(v * min(1.0, a)))
    col = Image.new("RGB", m.size, rgb)
    img.paste(col, (int(x * SS - m.size[0] / 2), int(y * SS - m.size[1] / 2)), mm)


pulse_now = 1.0


def draw_crown(d, time_s, scale, alpha, sweep=1.0, bright=1.0):
    if alpha <= 0.01 or sweep <= 0:
        return
    r0, r1 = CROWN_R
    phi = spin(CROWN_N, time_s, 1, CROWN_SPIN)
    m = int(math.ceil(CROWN_N * sweep))
    pts = np.vstack([petal_pts(CROWN_N, j, r0, r1, phi, scale, pts=CROWN_PTS)[:-1] for j in range(m)])
    d.line([tuple(p) for p in pts], fill=scol(CROWN_COL, alpha * bright), width=CROWN_LINE_W)
    rr = r0 * scale * SS * GEOM_SCALE
    a0 = math.degrees(phase0(CROWN_N) + phi)
    d.arc([CX * SS - rr, CY * SS - rr, CX * SS + rr, CY * SS + rr], a0, a0 + 360 * min(1.0, m / CROWN_N),
          fill=scol(CROWN_FIL_COL, 0.9 * alpha * bright), width=CROWN_LINE_W + 2)


def draw_center(d, time_s, scale, alpha, bright=1.0):
    if alpha <= 0.01:
        return
    phi = spin(4, time_s, 0, SPIN_TURNS)
    s = 40 * scale * SS * GEOM_SCALE
    pts = [(CX * SS + s * math.cos(phi + math.pi / 4 + q * math.pi / 2),
            CY * SS + s * math.sin(phi + math.pi / 4 + q * math.pi / 2)) for q in range(5)]
    d.line(pts, fill=scol(SQUARE_COL, 0.9 * alpha * bright), width=LINE_W - 1)
    b = 7 * SS * max(scale, 0.4)
    d.ellipse([CX * SS - b, CY * SS - b, CX * SS + b, CY * SS + b], fill=scol(SQUARE_COL, alpha * bright))


def revealed(k, tb):
    """number of petals of ring k visible at beat tb during the build, and their pop-in alphas"""
    n = PETALS[k]
    pops = [ease((tb - pb) / 0.25) for pb in PETAL_BEATS[k]]
    return sum(1 for pb in PETAL_BEATS[k] if tb >= pb), pops


def counted(tb):
    if tb < BUILD0:
        return 0
    c = 0
    for k in range(len(RINGS)):
        c += sum(1 for pb in PETAL_BEATS[k] if tb >= pb)
    return c


def sim_image(time_s):
    global pulse_now
    tb = time_s / BEAT_S
    pulse_now = hook_pulse(tb)
    big = Image.new("RGB", (SIM_W * SS, SIM_H * SS), (0, 0, 0))
    d = ImageDraw.Draw(big)
    full = tb < COLLAPSE[0] or tb >= RISER_BEATS[0]
    if tb < COLLAPSE[1]:                                  # hook: complete mandala, then it folds in
        sc = 1 - 0.92 * ease((tb - COLLAPSE[0]) / (COLLAPSE[1] - COLLAPSE[0]))
        a = 1 - ease((tb - COLLAPSE[0] - 1.0) / 1.0)
        draw_crown(d, time_s, sc, a)
        for k in reversed(range(len(RINGS))):
            draw_ring(d, k, time_s, sc, a, PETALS[k])
        draw_center(d, time_s, sc, max(a, 0.5))
    elif not full:                                        # build: petal by petal
        draw_center(d, time_s, 1.0, 1.0)
        for k in reversed(range(len(RINGS))):
            if tb >= RING_START[k]:
                nv, pops = revealed(k, tb)
                done = RING_START[k] + REVEAL_BEATS
                bright = 1.0 + 0.6 * math.exp(-max(0.0, tb - done) * 1.5) if tb >= done else 1.0
                draw_ring(d, k, time_s, 1.0, 1.0, nv, pops, bright)
    else:                                                 # riser, payoff, crown, hold, loop
        bright = 1.0
        if tb < SILENT_BEAT:
            bright = 1.0 + 0.5 * ease((tb - RISER_BEATS[0]) / (RISER_BEATS[1] - RISER_BEATS[0]))
        elif tb >= PAYOFF_BEAT:
            bright = 1.0 + 0.8 * math.exp(-(tb - PAYOFF_BEAT) * 1.2)
        sweep = 1.0 if tb >= CROWN_BEATS[1] else ease((tb - CROWN_BEATS[0]) / (CROWN_BEATS[1] - CROWN_BEATS[0]))
        draw_crown(d, time_s, 1.0, 1.0, sweep, bright)
        for k in reversed(range(len(RINGS))):
            draw_ring(d, k, time_s, 1.0, 1.0, PETALS[k], None, bright)
        draw_center(d, time_s, 1.0, 1.0, bright)
    img = big.reduce(SS)
    return np.asarray(img, dtype=np.float32)


# =====================================================================
# AUDIO
# =====================================================================
PAD_CHORDS = [(0, 10, (2, 3, 5)), (10, 28, (2, 3, 4)), (28, 46, (2, 3, 5)), (46, 51, (4, 5, 6)),
              (51, 72, (2, 3, 5)), (72, 84, (2, 3, 4))]


def synth_audio():
    tt = np.arange(N_SAMP) / SR
    music = np.zeros(N_SAMP)
    # hook: the full mandala chord, one pluck per lotus, rolled
    for k, n in enumerate(PETALS):
        add_at(music, k * 0.5 * BEAT_S, 0.8 * PLUCK_GAIN * pluck(n * BASE_HZ, tau=0.5, dur=1.6))
    # build: one pluck per petal, pitch = n * 55 Hz
    for k, n in enumerate(PETALS):
        f = n * BASE_HZ
        for pb in PETAL_BEATS[k]:
            add_at(music, pb * BEAT_S, PLUCK_GAIN * pluck(f, tau=0.22 if n >= 10 else 0.35, dur=1.0))
        add_at(music, (RING_START[k] + REVEAL_BEATS) * BEAT_S, BELL_GAIN * bell(f / 2 if f > 600 else f, [1, 2, 3]))
    # payoff: all six lotuses at once
    for n in PETALS:
        add_at(music, PAYOFF_BEAT * BEAT_S, 0.6 * BELL_GAIN * bell(n * BASE_HZ, [1, 2, 3], dur=3.0))
    # crown: 1000 petals -> a shimmer of high harmonics of 55 Hz
    c0, c1 = CROWN_BEATS[0] * BEAT_S, CROWN_BEATS[1] * BEAT_S
    for g in range(160):
        tg = c0 + (c1 - c0) * (g / 160) + RNG.uniform(0, 0.03)
        h = RNG.choice([16, 20, 24, 30, 32, 40])
        add_at(music, tg, SHIMMER_GAIN * pluck(h * BASE_HZ, tau=0.12, dur=0.4))
    # loop: the hook roll again so the replay is seamless
    for k, n in enumerate(PETALS):
        add_at(music, (80 + k * 0.5) * BEAT_S, 0.6 * PLUCK_GAIN * pluck(n * BASE_HZ, tau=0.4, dur=1.2))

    pad = np.zeros(N_SAMP)
    sub = np.zeros(N_SAMP)
    for v in range(3):
        fr = np.zeros(N_SAMP)
        for b0, b1, ch in PAD_CHORDS:
            fr[int(b0 * BEAT_S * SR):int(b1 * BEAT_S * SR)] = ch[v] * BASE_HZ
        fr[fr == 0] = PAD_CHORDS[-1][2][v] * BASE_HZ
        ph = 2 * np.pi * np.cumsum(fr) / SR
        pad += sum(np.sin(h * ph) / h ** 1.6 for h in range(1, 5)) * (1.0 if v == 0 else 0.7)
        if v == 0:
            sub = np.sin(ph / 2)
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.25 * tt)
    for b0, _, _ in PAD_CHORDS[1:]:
        i, n = int(b0 * BEAT_S * SR), int(0.02 * SR)
        pad[i - n:i + n] *= 0.5 + 0.5 * np.abs(np.linspace(-1, 1, 2 * n))
    pad *= lfo

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
    for b in list(range(22, 46)) + list(range(52, 72)):
        add_at(hats, (b + 0.5) * BEAT_S, hat)
        if (b >= 34) and b % 4 in (1, 3):
            add_at(claps, b * BEAT_S, clap)
    riser = np.zeros(N_SAMP)
    r0, r1 = int(RISER_BEATS[0] * BEAT_S * SR), int(RISER_BEATS[1] * BEAT_S * SR)
    rn = RNG.standard_normal(r1 - r0)
    rn = rn - np.convolve(rn, np.ones(24) / 24, mode="same")
    x = np.linspace(0, 1, r1 - r0)
    riser[r0:r1] = (0.8 * rn + 0.6 * np.sin(2 * np.pi * np.cumsum(220 * 2 ** (3 * x)) / SR)) * x ** 2.5

    drums_on = np.array([1.0 if kick_on(int(t / BEAT_S)) else 0.0 for t in tt[::441]])
    drums_on = np.repeat(drums_on, 441)[:N_SAMP]
    beat_pos = np.full(N_SAMP, 10.0)
    idx = np.searchsorted(KICK_TIMES, tt, side="right") - 1
    ok = idx >= 0
    beat_pos[ok] = tt[ok] - KICK_TIMES[idx[ok]]
    pump = 1.0 - PUMP_DEPTH * np.exp(-beat_pos / PUMP_DECAY_S) * drums_on
    common = (SUB_GAIN * sub * pump * drums_on + music + KICK_GAIN * kick
              + HAT_GAIN * hats + CLAP_GAIN * claps + RISER_GAIN * riser)
    st = np.stack([PAD_GAIN * pad * pump + common, PAD_GAIN * 0.92 * pad * pump + common], axis=1)

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
    "petal": r"$r(\theta) = R + a\,|\sin(n\theta/2)|^{p}$",
    "sum":   r"$4+6+10+12+16+2 = 50$",
    "crown": r"$1000 = 50 \times 20$",
}


def eq_key_at(time_s):
    b = time_s / BEAT_S
    if b < 4 or b >= 82:
        return None
    if b < 46 or b >= 72:
        return "petal"
    if b < 56:
        return "sum"
    return "crown"


EQ_CHANGES = [fi / FPS for fi in range(1, N_FRAMES) if eq_key_at(fi / FPS) != eq_key_at((fi - 1) / FPS)]


def chip_at(time_s):
    b = time_s / BEAT_S
    if BUILD0 <= b < RISER_BEATS[0]:
        k = min(len(RINGS) - 1, int((b - BUILD0) // RING_BEATS))
        name, n = RINGS[k][:2]
        return f"{name}  {n} petals   ·   total {counted(b)}"
    if RISER_BEATS[0] <= b < 58:
        return f"total {TOTAL}"
    if 58 <= b < 64:
        return f"{CROWN_NAME}  {CROWN_N} petals"
    if 64 <= b < 74:
        return SOURCE_CHIP
    if b >= 74:
        return SERIES_CHIP
    return None


LAYERS = {}
CHIP_CACHE = {}


def chip_layer(txt):
    if txt not in CHIP_CACHE:
        col = hex_rgb(C_HERO) if txt in (SERIES_CHIP, f"total {TOTAL}") else (230, 236, 245)
        CHIP_CACHE[txt] = np.asarray(text_layer(txt, CHIP_FONT_SIZE, col, True, STORY_MAX_W)[0], dtype=np.float32)
    return CHIP_CACHE[txt]


def build_caches():
    eq = {k: np.asarray(mathtext_png(v), dtype=np.float32) for k, v in EQ_LINES.items()}
    story = {}
    for st, txt, _ in TEXT_EVENTS:
        img, nl = text_layer(txt, STORY_FONT_SIZE, (255, 255, 255), True, STORY_MAX_W)
        assert nl <= 2 and STORY_Y + img.size[1] <= SAFE_BOTTOM + 4, f"caption too tall: {txt!r}"
        story[txt] = np.asarray(img, dtype=np.float32)
    sign = np.asarray(text_layer(SIGN_OFF, SIGN_FONT_SIZE, (255, 255, 255), False, STORY_MAX_W)[0],
                      dtype=np.float32)
    LAYERS.update(eq=eq, story=story, sign=sign)


def render_frame(fi):
    time_s = fi / FPS
    frame = np.empty((H, W, 3), dtype=np.float32)
    frame[:] = BG_RGB
    silent = SILENT_T0 <= time_s < SILENT_T1
    i = np.searchsorted(KICK_TIMES, time_s + 1e-9, side="right") - 1
    flash = KICK_FLASH * math.exp(-(time_s - KICK_TIMES[i]) * KICK_FLASH_DECAY) if i >= 0 and not silent else 0.0
    intensity = (1.0 + flash) * (SILENT_DIM if silent else 1.0)
    arr = sim_image(time_s)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    small = img.resize((SIM_W // GLOW_DOWNSCALE, SIM_H // GLOW_DOWNSCALE), Image.BILINEAR)
    glow = small.filter(ImageFilter.GaussianBlur(GLOW_RADIUS)).resize((SIM_W, SIM_H), Image.BILINEAR)
    frame[SIM_Y0:SIM_Y0 + SIM_H, SIM_X0:SIM_X0 + SIM_W] += (arr + GLOW_GAIN * np.asarray(glow, np.float32)) * intensity
    key = eq_key_at(time_s)
    if key:
        fa = 1.0
        for tc in EQ_CHANGES:
            if tc - EQ_FADE_S <= time_s < tc:
                fa = (tc - time_s) / EQ_FADE_S
            elif tc <= time_s < tc + EQ_FADE_S:
                fa = (time_s - tc) / EQ_FADE_S
        alpha_paste(frame, LAYERS["eq"][key], EQ_Y_TOP, fa * (0.45 if silent else 1.0))
    c = chip_at(time_s)
    if c:
        alpha_paste(frame, chip_layer(c), CHIP_Y, 0.95)
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
SHEET_TIMES = [("hook", 0.0), ("build: 10 petals", 11.6), ("build: throat 16", 19.2),
               ("payoff 50", PAYOFF_BEAT * BEAT_S + 0.3), ("crown 1000", 29.5), ("ending", DURATION - 1.0 / FPS)]


def write_beats_json():
    out = dict(duration_s=round(DURATION, 3), bpm=BPM, petals=PETALS, total=TOTAL, crown=CROWN_N,
               ring_start_s=[round(b * BEAT_S, 3) for b in RING_START],
               sections=[dict(key=s["key"], t0=round(s["t0"], 3), t1=round(s["t1"], 3)) for s in SCHED],
               silent=[round(SILENT_T0, 3), round(SILENT_T1, 3)], captions=[])
    for st, txt, en in TEXT_EVENTS:
        en = min(en, DURATION)
        out["captions"].append(dict(t0=round(st, 3), t1=round(en, 3), seconds=round(en - st, 2),
                                    words=len(txt.split()), text=txt.replace("\n", " ")))
    with open(OUT_BEATS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    print(f"Duration {DURATION:.2f}s, {N_FRAMES} frames, BPM {BPM}; petals {PETALS} = {TOTAL}; crown {CROWN_N}")
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
