"""
Coded Cosmos — "Which ball wins?"  (Weird Physics: the brachistochrone + tautochrone)
-----------------------------------------------------------------------------------
Three balls slide with no friction from A to B (r = 1 m, g = 9.81 m/s^2):
  straight line, a deep steep dip, and the cycloid  x = r(th - sin th), y = r(1 - cos th).
Times come from integrating t = INT ds / sqrt(2 g y) along each track (no hand-typed numbers):
  cycloid 1.003 s  (= pi sqrt(r/g)),  dip 1.152 s,  line 1.189 s.
The cycloid is the brachistochrone (Johann Bernoulli's 1696 challenge; solved 1697).
On the cycloid the parameter runs uniformly, th = t sqrt(g/r): the ball IS a point on a rolling wheel.
Tautochrone (Huygens, Horologium Oscillatorium, 1673): released from rest at ANY height on the
cycloid, a ball reaches the bottom after T = pi sqrt(r/g); arc length s(t) = s0 cos(t sqrt(g/4r)).
Shown in slow motion so T lands on 4 beats at 120 BPM.  100% original code-generated music.

Run:  python cycloid_race.py          (add --sheet for the preview sheet only)
  -> cycloid_race.mp4, cycloid_race_upload.mp4, preview_frames.jpg, audio_levels.txt, beats.json
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
SLUG             = "cycloid_race"
OUT_VIDEO        = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD       = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET        = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_LEVELS       = os.path.join(OUT_DIR, "audio_levels.txt")
OUT_BEATS        = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO        = os.path.join(OUT_DIR, "_cycloid_audio.wav")
MAKE_UPLOAD      = True
UPLOAD_TARGET_MB = 9.0
UPLOAD_AUDIO_K   = 160

W, H             = 1080, 1920
FPS              = 30
BG_COLOR         = "#06050a"
BG_RGB           = (6, 5, 10)

# --- physics ---
G                = 9.81           # m/s^2
R_WHEEL          = 1.0            # m, cycloid radius
DIP_D            = 2.0            # dip track: y = 2x/pi + D sin x
N_TRACK          = 20001          # samples per track for the time integral
T_TAUTO          = math.pi * math.sqrt(R_WHEEL / G)   # 1.003 s

# --- timing (beats) ---
BPM              = 120
SR               = 44100
TOTAL_BEATS      = 84             # 42 s
SLOWMO_BEATS     = 4              # T_TAUTO is shown as exactly 4 beats (slow-mo x ~2)
RACE_STARTS      = [-1, 6, 13, 20, 76, 83]    # period 7; 83 - 84 == -1 so the loop is seamless
RACE_FADE        = (4.9, 5.5)     # beats after start: balls fade at B
RACE_PREP        = 1.5            # beats before start: balls fade in at A
WHEEL_RACE       = 20             # race that shows the rolling wheel
SCENE_A_END      = 26
XFADE            = 2.0            # beats
TAUTO_RELEASE    = 28
TAUTO_END        = 74             # bowls fade out 72-74
SILENT_BEAT      = 63
PAYOFF_BEAT      = 64
RISER_BEATS      = (59, 63)
SECTIONS = [
    ("hook", 0, 8), ("race", 8, 26), ("xfade", 26, 28), ("tauto", 28, 59), ("riser", 59, 63),
    ("silence", 63, 64), ("payoff", 64, 72), ("loop", 72, 84),
]

STORY = [
    (0,  "Which ball wins?"),
    (5,  "The straight line\nis the shortest..."),
    (11, "...but the curve\nis faster."),
    (20, "A rolling wheel\ndraws that curve."),
    (26, "Now drop 5 balls\nfrom any height."),
    (32, "They all hit the\nbottom together."),
    (42, "Huygens, 1673.\nHe built clocks on it."),
    (50, "Higher ones just\ngo faster."),
    (57, "Wait for it..."),
    (64, "Same time.\nEvery time."),
    (70, "Shortest isn't fastest."),
    (76, "Which ball wins?"),
]
MIN_CAPTION_S    = 3.0
SERIES_CHIP      = "WEIRD PHYSICS"
PHYS_CHIP        = "no friction  ·  r = 1 m  ·  slow-mo"
SOURCE_CHIP      = "Horologium Oscillatorium, 1673"

C_HERO           = "#ffd166"      # cycloid
C_LINE           = "#5ad1ff"
C_DIP            = "#ff5fa2"
BOWL_COLS        = ["#ffd166", "#ffa94d", "#ff6b6b", "#c77dff", "#5ad1ff"]
BOWL_S0          = [0.95, 0.76, 0.57, 0.38, 0.19]   # start arc length as a fraction of 4r
TRACK_W          = 4              # at SS
BALL_R           = 17             # px at 1x
WHEEL_COL        = "#8a8fa8"

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
RACE_PX_M        = 180            # px per metre, race
RACE_X0, RACE_Y0 = 197, 150
LEGEND_Y         = 760
BOWL_PX_M        = 128
BOWL_Y0          = 130            # top of the first (back) bowl
BOWL_DY          = 120            # stack offset

GLOW_DOWNSCALE   = 4
GLOW_RADIUS      = 5
GLOW_GAIN        = 1.2
KICK_FLASH       = 0.12
KICK_FLASH_DECAY = 8.0
SILENT_DIM       = 0.28

FONT_BOLD        = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG         = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
STORY_FONT_SIZE  = 64
CHIP_FONT_SIZE   = 38
SIGN_FONT_SIZE   = 32
LABEL_FONT_SIZE  = 34
TEXT_STROKE      = 5
TEXT_FADE_S      = 0.30
EQ_FONTSIZE      = 12
EQ_DPI           = 330
EQ_COLOR         = "#ffffffdd"
EQ_STROKE_W      = 3
EQ_FADE_S        = 0.15

# --- audio ---
ROOT_HZ          = 110.0          # A2
PENTA            = [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3]   # one note per bowl
PLUCK_GAIN       = 0.24
BELL_GAIN        = 0.20
PAD_GAIN         = 0.10
SUB_GAIN         = 0.12
KICK_GAIN        = 0.34
HAT_GAIN         = 0.04
CLAP_GAIN        = 0.07
RISER_GAIN       = 0.14
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
SLOWMO   = SLOWMO_BEATS * BEAT_S / T_TAUTO          # display seconds per real second
SCHED    = [dict(key=k, t0=b0 * BEAT_S, t1=b1 * BEAT_S) for k, b0, b1 in SECTIONS]
SILENT_T0 = SILENT_BEAT * BEAT_S
SILENT_T1 = SILENT_T0 + BEAT_S


def track_time(xs, ys):
    """t(s) along a track from rest at y=0: dt = ds / v, v = sqrt(2 g y)."""
    ds = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    v = np.sqrt(2 * G * np.maximum(ys, 0))
    t = np.zeros_like(s)
    t[1] = 2 * ds[0] / max(v[1], 1e-12)                 # first step from rest: uniform acceleration
    t[2:] = t[1] + np.cumsum(2 * ds[1:] / (v[2:] + v[1:-1]))
    return t


_th = np.linspace(0, math.pi, N_TRACK)
_x = np.linspace(0, math.pi * R_WHEEL, N_TRACK)
TRACKS = {
    "cyc":  (R_WHEEL * (_th - np.sin(_th)), R_WHEEL * (1 - np.cos(_th)), C_HERO, "curve"),
    "dip":  (_x, 2 * R_WHEEL * _x / (math.pi * R_WHEEL) + DIP_D * np.sin(_x / R_WHEEL), C_DIP, "dip"),
    "line": (_x, 2 * R_WHEEL * _x / (math.pi * R_WHEEL), C_LINE, "line"),
}
TRACK_T = {k: track_time(v[0], v[1]) for k, v in TRACKS.items()}
FINISH = {k: float(t[-1]) for k, t in TRACK_T.items()}
assert abs(FINISH["cyc"] - T_TAUTO) / T_TAUTO < 0.005, FINISH
assert FINISH["cyc"] < FINISH["dip"] < FINISH["line"], FINISH
FINISH_BEATS = {k: v * SLOWMO / BEAT_S for k, v in FINISH.items()}
assert max(FINISH_BEATS.values()) < RACE_FADE[0]
assert abs(FINISH_BEATS["cyc"] - SLOWMO_BEATS) < 1e-6

TEXT_EVENTS = []
prev = -1e9
for b, txt in STORY:
    st = max(b * BEAT_S, prev + MIN_CAPTION_S)
    TEXT_EVENTS.append([st, txt])
    prev = st
for i in range(len(TEXT_EVENTS)):
    TEXT_EVENTS[i].append(TEXT_EVENTS[i + 1][0] if i + 1 < len(TEXT_EVENTS) else DURATION + 1)


def kick_on(b):
    if b == SILENT_BEAT:
        return False
    if RISER_BEATS[0] <= b < RISER_BEATS[1]:
        return b % 2 == 1
    return True


KICK_TIMES = np.array([b * BEAT_S for b in range(TOTAL_BEATS) if kick_on(b)])
RNG = np.random.default_rng(1696)
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
# SIM
# =====================================================================
SS = 2
OMEGA = math.sqrt(G / (4 * R_WHEEL))
BOWL_X0 = (SIM_W - 2 * math.pi * R_WHEEL * BOWL_PX_M) / 2
BOWL_HITS = [TAUTO_RELEASE + SLOWMO_BEATS * (2 * m + 1) for m in range(12)]
BOWL_HITS = [b for b in BOWL_HITS if b < TOTAL_BEATS]
BOWL_TURNS = [TAUTO_RELEASE + SLOWMO_BEATS * 2 * m for m in range(1, 12)]
BOWL_TURNS = [b for b in BOWL_TURNS if b < TAUTO_END - 2]


def race_px(x, y):
    return (RACE_X0 + x * RACE_PX_M) * SS, (RACE_Y0 + y * RACE_PX_M) * SS


def race_vis(tb):
    if tb < SCENE_A_END:
        return 1.0
    if tb < SCENE_A_END + XFADE:
        return 1 - ease((tb - SCENE_A_END) / XFADE)
    if tb >= TAUTO_END - XFADE:
        return ease((tb - (TAUTO_END - XFADE)) / XFADE)
    return 0.0


def bowl_vis(tb):
    if tb < SCENE_A_END:
        return 0.0
    if tb < SCENE_A_END + XFADE:
        return ease((tb - SCENE_A_END) / XFADE)
    if tb >= TAUTO_END - XFADE:
        return 1 - ease((tb - (TAUTO_END - XFADE)) / XFADE)
    return 1.0


def race_now(tb):
    """(start beat, tau beats, ball alpha) of the race that owns time tb."""
    cands = [s for s in RACE_STARTS if s - RACE_PREP <= tb]
    s = max(cands) if cands else RACE_STARTS[0]
    tau = tb - s
    if tau < 0:
        a = ease((tau + RACE_PREP) / RACE_PREP)
    elif tau < RACE_FADE[0]:
        a = 1.0
    else:
        a = 1 - ease((tau - RACE_FADE[0]) / (RACE_FADE[1] - RACE_FADE[0]))
    return s, tau, a


def ball_on_track(k, tau):
    t_real = max(0.0, tau) * BEAT_S / SLOWMO
    x, y = TRACKS[k][0], TRACKS[k][1]
    i = np.searchsorted(TRACK_T[k], t_real)
    if i >= len(x):
        return x[-1], y[-1], len(x) - 1
    if i == 0:
        return x[0], y[0], 0
    t0, t1 = TRACK_T[k][i - 1], TRACK_T[k][i]
    f = (t_real - t0) / max(t1 - t0, 1e-12)
    return x[i - 1] + f * (x[i] - x[i - 1]), y[i - 1] + f * (y[i] - y[i - 1]), i


def dot(d, cx, cy, rad, col, a):
    if a <= 0.01:
        return
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=scol(col, a))
    rc = rad * 0.45
    d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=scol((255, 255, 255), a))


TRACK_PX = {k: [race_px(x, y) for x, y in zip(v[0][::100], v[1][::100])] + [race_px(v[0][-1], v[1][-1])]
            for k, v in TRACKS.items()}


def draw_race(d, tb, vis):
    s, tau, ba = race_now(tb)
    for k in ("dip", "line", "cyc"):
        col = TRACKS[k][2]
        d.line(TRACK_PX[k], fill=scol(col, 0.32 * vis), width=TRACK_W, joint="curve")
        if tau > 0 and ba > 0:
            _, _, i = ball_on_track(k, tau)
            pts = TRACK_PX[k][:max(2, i // 100 + 1)]
            d.line(pts, fill=scol(col, 0.95 * vis * ba), width=TRACK_W + 2, joint="curve")
    ax, ay = race_px(0, 0)
    bx, by = race_px(*(TRACKS["cyc"][0][-1], TRACKS["cyc"][1][-1]))
    for (px, py), lab in (((ax, ay), "A"), ((bx, by), "B")):
        d.ellipse([px - 7 * SS, py - 7 * SS, px + 7 * SS, py + 7 * SS], outline=scol((220, 220, 230), 0.8 * vis), width=SS * 2)
        d.text((px + (-34 if lab == "A" else 22) * SS, py - 18 * SS), lab, font=font(30 * SS), fill=scol((220, 220, 230), 0.8 * vis))
    if s == WHEEL_RACE and tau > -RACE_PREP:
        th = min(math.pi, max(0.0, tau) * BEAT_S / SLOWMO * math.sqrt(G / R_WHEEL))
        cx, cy = race_px(R_WHEEL * th, R_WHEEL)
        rr = R_WHEEL * RACE_PX_M * SS
        wa = 0.8 * vis * ba
        d.line([race_px(-R_WHEEL, 0), race_px(math.pi * R_WHEEL + R_WHEEL, 0)], fill=scol(WHEEL_COL, 0.6 * wa), width=SS * 2)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=scol(WHEEL_COL, wa), width=SS * 3)
        px, py = race_px(R_WHEEL * (th - math.sin(th)), R_WHEEL * (1 - math.cos(th)))
        d.line([(cx, cy), (px, py)], fill=scol(WHEEL_COL, wa), width=SS * 3)
        d.ellipse([cx - 6 * SS, cy - 6 * SS, cx + 6 * SS, cy + 6 * SS], fill=scol(WHEEL_COL, wa))
    for k in ("line", "dip", "cyc"):
        for gi, lag in enumerate((0.24, 0.16, 0.08, 0.0)):        # short motion ghosts
            x, y, _ = ball_on_track(k, tau - lag)
            px, py = race_px(x, y)
            dot(d, px, py, BALL_R * SS * (0.6 + 0.1 * gi), TRACKS[k][2], vis * ba * (0.25 + 0.25 * gi))
    # finish board
    slots = {"cyc": 0, "dip": 1, "line": 2}
    s_leg = max([x for x in RACE_STARTS if x <= tb] or [RACE_STARTS[0]])
    tau_l = tb - s_leg                                   # board stays up until the next race starts
    for k, j in slots.items():
        if tau_l >= FINISH_BEATS[k]:
            a = vis * ease((tau_l - FINISH_BEATS[k]) / 0.3)
            tau = tau_l
            pop = 1 + 0.5 * math.exp(-(tau - FINISH_BEATS[k]) * 4)
            txt = f"{TRACKS[k][3]}  {FINISH[k]:.2f} s"
            x0 = (40 + j * 300) * SS
            y0 = LEGEND_Y * SS
            d.ellipse([x0, y0 + 10 * SS, x0 + 22 * SS, y0 + 32 * SS], fill=scol(TRACKS[k][2], a * min(pop, 1.4)))
            d.text((x0 + 32 * SS, y0 + 4 * SS), txt, font=font(LABEL_FONT_SIZE * SS),
                   fill=scol((255, 255, 255) if k == "cyc" else (200, 205, 215), a))


def bowl_px(k, th):
    x = BOWL_X0 + R_WHEEL * (th - np.sin(th)) * BOWL_PX_M
    y = BOWL_Y0 + k * BOWL_DY + R_WHEEL * (1 - np.cos(th)) * BOWL_PX_M
    return x * SS, y * SS


_BTH = np.linspace(0, 2 * math.pi, 400)
BOWL_PTS = [list(zip(*bowl_px(k, _BTH))) for k in range(len(BOWL_S0))]


def bowl_theta(s):
    q = np.clip(abs(s) / (4 * R_WHEEL), 0, 1)
    th = 2 * math.acos(q)
    return th if s >= 0 else 2 * math.pi - th


def draw_bowls(d, tb, vis):
    hit = 0.0
    for hb in BOWL_HITS:
        if hb <= tb:
            hit = max(hit, math.exp(-(tb - hb) * 2.5))
    cx = (BOWL_X0 + math.pi * R_WHEEL * BOWL_PX_M) * SS
    if hit > 0.02:
        y0 = (BOWL_Y0 + 2 * R_WHEEL * BOWL_PX_M - 30) * SS
        y1 = (BOWL_Y0 + (len(BOWL_S0) - 1) * BOWL_DY + 2 * R_WHEEL * BOWL_PX_M + 30) * SS
        d.line([(cx, y0), (cx, y1)], fill=scol((255, 255, 255), 0.7 * hit * vis), width=SS * 4)
    for k in range(len(BOWL_S0)):
        col = BOWL_COLS[k]
        d.line(BOWL_PTS[k], fill=scol(col, 0.38 * vis), width=TRACK_W, joint="curve")
        for gi, lag in enumerate((0.24, 0.16, 0.08, 0.0)):
            tr = max(0.0, tb - lag - TAUTO_RELEASE) * BEAT_S / SLOWMO
            s = BOWL_S0[k] * 4 * R_WHEEL * math.cos(OMEGA * tr)
            px, py = bowl_px(k, bowl_theta(s))
            dot(d, px, py, BALL_R * SS * (0.6 + 0.1 * gi) * (1 + 0.35 * hit * (gi == 3)), col,
                vis * (0.25 + 0.25 * gi))


def sim_image(time_s):
    tb = time_s / BEAT_S
    big = Image.new("RGB", (SIM_W * SS, SIM_H * SS), (0, 0, 0))
    d = ImageDraw.Draw(big)
    rv, bv = race_vis(tb), bowl_vis(tb)
    if bv > 0:
        draw_bowls(d, tb, bv)
    if rv > 0:
        draw_race(d, tb, rv)
    img = big.reduce(SS)
    return np.asarray(img, dtype=np.float32)


# =====================================================================
# AUDIO
# =====================================================================
BASE_HZ = 55.0
PAD_CHORDS = [(0, 26, (2, 3, 5)), (26, 59, (2, 3, 4)), (59, 63, (4, 5, 6)), (63, 84, (2, 3, 5))]


def synth_audio():
    tt = np.arange(N_SAMP) / SR
    music = np.zeros(N_SAMP)
    for s in RACE_STARTS:
        if s >= 0:
            add_at(music, s * BEAT_S, 0.5 * PLUCK_GAIN * pluck(ROOT_HZ * 8, tau=0.06, dur=0.3))
        for k in ("cyc", "dip", "line"):
            ta = (s + FINISH_BEATS[k]) * BEAT_S
            if 0 <= ta < DURATION:
                if k == "cyc":
                    add_at(music, ta, BELL_GAIN * bell(ROOT_HZ * 4, [1, 2, 3], dur=2.0))
                    add_at(music, ta, PLUCK_GAIN * pluck(ROOT_HZ * 8, tau=0.3))
                else:
                    f = ROOT_HZ * (2 * PENTA[3] if k == "dip" else 2 * PENTA[1])
                    add_at(music, ta, 0.6 * PLUCK_GAIN * pluck(f, tau=0.12, dur=0.5))
    for j, p in enumerate(PENTA):                                  # release: one pluck per bowl
        add_at(music, (TAUTO_RELEASE + j * 0.12) * BEAT_S, 0.7 * PLUCK_GAIN * pluck(ROOT_HZ * 2 * p, tau=0.4))
    for hb in BOWL_HITS:                                           # all 5 land together: one chord
        if TAUTO_RELEASE < hb < TAUTO_END:
            g = 1.5 if hb == PAYOFF_BEAT else 1.0
            for p in PENTA:
                add_at(music, hb * BEAT_S, g * 0.55 * BELL_GAIN * bell(ROOT_HZ * 4 * p, [1, 2, 3], dur=2.5))
    for tb_ in BOWL_TURNS:
        add_at(music, tb_ * BEAT_S, 0.4 * PLUCK_GAIN * pluck(ROOT_HZ * PENTA[2], tau=0.3))

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
    for b in list(range(8, 59)) + list(range(64, 72)):
        add_at(hats, (b + 0.5) * BEAT_S, hat)
        if 33 <= b < 59 and b % 4 in (1, 3):
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
    "cyc":   r"$x = r(\theta - \sin\theta),\quad y = r(1 - \cos\theta)$",
    "time":  r"$t = \int \frac{ds}{\sqrt{2gy}}\quad \mathrm{smallest\ on\ the\ cycloid}$",
    "tauto": r"$T = \pi\sqrt{r/g}\quad \mathrm{from\ any\ height}$",
}


def eq_key_at(time_s):
    b = time_s / BEAT_S
    if b < 4 or b >= 82:
        return None
    if b < 12 or 20 <= b < 26 or b >= 64:
        return "cyc"
    if b < 26:
        return "time"
    return "tauto"


EQ_CHANGES = [fi / FPS for fi in range(1, N_FRAMES) if eq_key_at(fi / FPS) != eq_key_at((fi - 1) / FPS)]


def chip_at(time_s):
    b = time_s / BEAT_S
    if 42 <= b < 50:
        return SOURCE_CHIP
    if 66 <= b < 75:
        return SERIES_CHIP
    if b < 4 or b >= 82:
        return None
    return PHYS_CHIP


LAYERS = {}
CHIP_CACHE = {}


def chip_layer(txt):
    if txt not in CHIP_CACHE:
        col = hex_rgb(C_HERO) if txt == SERIES_CHIP else (230, 236, 245)
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
SHEET_TIMES = [("hook", 0.0), ("race result", 6.0), ("wheel", 12.2),
               ("bowls hit", 32 * BEAT_S + 0.05), ("payoff", PAYOFF_BEAT * BEAT_S + 0.1), ("ending", DURATION - 1.0 / FPS)]


def write_beats_json():
    out = dict(duration_s=round(DURATION, 3), bpm=BPM, slowmo=round(SLOWMO, 4),
               finish_s={k: round(v, 4) for k, v in FINISH.items()}, t_tauto=round(T_TAUTO, 4),
               race_starts_s=[round(b * BEAT_S, 3) for b in RACE_STARTS], bowl_hits_s=[round(b * BEAT_S, 3) for b in BOWL_HITS],
               sections=[dict(key=s["key"], t0=round(s["t0"], 3), t1=round(s["t1"], 3)) for s in SCHED],
               silent=[round(SILENT_T0, 3), round(SILENT_T1, 3)], captions=[])
    for st, txt, en in TEXT_EVENTS:
        en = min(en, DURATION)
        out["captions"].append(dict(t0=round(st, 3), t1=round(en, 3), seconds=round(en - st, 2),
                                    words=len(txt.split()), text=txt.replace("\n", " ")))
    with open(OUT_BEATS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main():
    print(f"Duration {DURATION:.2f}s, {N_FRAMES} frames; slow-mo x{SLOWMO:.3f}; finish {FINISH}")
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
