"""
Coded Cosmos — "POV: you're a raindrop"  (Weird Physics #2, monsoon over Kolkata rooftops)
------------------------------------------------------------------------------------------
Four drops leave a cloud 2 km up at the same moment:
  ghost  = a drop with NO air:      z = H0 - g t^2 / 2           -> hits at sqrt(2 H0 / g) = 20.2 s, 198 m/s = 713 km/h
  1/2/5 mm drops WITH air (quadratic drag, m dv/dt = mg - 1/2 rho Cd A v^2):
      v(t) = vt tanh(g t / vt),  z(t) = H0 - (vt^2 / g) ln cosh(g t / vt)
  terminal speeds vt measured by Gunn & Kinzer (1949): 1 mm 4.03, 2 mm 6.49, 5 mm 9.09 m/s.
  drop shape: axis ratio b/a = 1.03 - 0.062 d[mm]  (Pruppacher & Beard 1970) -> big drops are buns, not tears.
Time is warped (x3 while the ghost falls, then x14); the on-screen clock always shows REAL time.
Music: raag Megh notes (S R M P n), tanpura + tabla + santoor, all synthesised; every landing,
speed threshold and altitude mark is a note at the exact frame the sim reaches it.
Run:  python monsoon_raindrop.py [--sheet]
"""
import os, io, sys, json, math, wave, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

sys.path.insert(0, os.environ.get("MUSIC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_music")))
import genre_engine as ge

# =====================================================================
# GLOBALS
# =====================================================================
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
SLUG       = "monsoon_raindrop"
OUT_VIDEO  = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET  = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_LEVELS = os.path.join(OUT_DIR, "audio_levels.txt")
OUT_BEATS  = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO  = os.path.join(OUT_DIR, "_audio.wav")
UPLOAD_TARGET_MB, UPLOAD_AUDIO_K = 8.5, 160

W, H, FPS  = 1080, 1920, 30
BG_COLOR   = "#0c1a24"               # look L3: monsoon night (one BG value everywhere)
BG_RGB     = (12, 26, 36)
SR         = ge.SR
BPM        = 100
BEAT_S     = 60.0 / BPM
TOTAL_BEATS = 74
DURATION   = TOTAL_BEATS * BEAT_S      # 44.4 s, loops
N_FRAMES   = int(round(DURATION * FPS))
TARGET_LUFS, TARGET_TP = -14.0, -1.5

# --- physics ---
G          = 9.81
H0         = 2000.0                    # m, cloud base in this scene
DROPS      = [("1 mm", 1.0, 4.03), ("2 mm", 2.0, 6.49), ("5 mm", 5.0, 9.09)]   # Gunn & Kinzer 1949
T_GHOST    = math.sqrt(2 * H0 / G)     # 20.19 s
V_GHOST    = G * T_GHOST               # 198.1 m/s
WARP1, WARP2 = 2.4, 14.0              # x2.4 -> frame 0 opens 4.4 s into the fall (ghost already at ~150 km/h)
IMPACT_BEAT = 11                       # ghost impact lands exactly on beat 11
T_IMPACT_V = IMPACT_BEAT * BEAT_S
S0         = T_GHOST - WARP1 * T_IMPACT_V   # real time at frame 0
RAMP_V0, RAMP_V1 = T_IMPACT_V + 0.3, T_IMPACT_V + 1.8
LOOP_FADE_V = 1.4                      # last seconds: new drops form in the cloud

# --- layout (px, top = 0) ---
Y_TOP, Y_GROUND = 470, 1250            # 2 km -> 0 m
PX_M       = (Y_GROUND - Y_TOP) / H0
COL_X      = {"ghost": 175, "1 mm": 385, "2 mm": 595, "5 mm": 805}
FONT_BOLD  = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG   = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
C_TITLE_BOX = (242, 177, 52)           # marigold
C_TITLE_TXT = (20, 22, 28)
C_STICKER  = (255, 214, 64)
C_DROP     = (170, 225, 255)
C_YOU      = (255, 236, 160)
C_GHOST    = (200, 205, 230)
C_WINDOW   = (255, 170, 70)
EQ_DPI, EQ_FONTSIZE, EQ_BLOCK_H, EQ_STROKE_W = 330, 11, 120, 3
EQ_Y       = 1290
LEGEND_Y   = 1405
STICKER_Y  = 1455
TEXT_STROKE = 4
RAIN_N     = 230
SEED       = 27
_FONTS, LAYERS = {}, {}

# =====================================================================
# TIME + PHYSICS
# =====================================================================
_tv = np.arange(N_FRAMES + 1) / FPS
def _warp(tv):
    if tv < RAMP_V0: return WARP1
    if tv > RAMP_V1: return WARP2
    x = (tv - RAMP_V0) / (RAMP_V1 - RAMP_V0)
    return WARP1 + (WARP2 - WARP1) * (3 * x * x - 2 * x ** 3)
_fine = np.linspace(0, DURATION, 200001)
_wv = np.array([_warp(t) for t in _fine])
_sim = S0 + np.concatenate([[0], np.cumsum(0.5 * (_wv[1:] + _wv[:-1]) * np.diff(_fine))])

def sim_t(tv):  return float(np.interp(tv, _fine, _sim))
def vid_t(ts):  return float(np.interp(ts, _sim, _fine))

def drop_state(vt, ts):
    a = G * ts / vt
    v = vt * math.tanh(a)
    z = H0 - (vt * vt / G) * (a + math.log1p(math.exp(-2 * a)) - math.log(2))   # ln cosh, overflow-safe
    return max(z, 0.0), v

def land_time(vt):
    lo, hi = 0.0, 5000.0
    for _ in range(80):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if drop_state(vt, m)[0] > 0 else (lo, m)
    return hi

LAND = {name: land_time(vt) for name, _, vt in DROPS}
LAND_V = {k: vid_t(v) for k, v in LAND.items()}
LAND_V["ghost"] = T_IMPACT_V
assert all(v < DURATION - LOOP_FADE_V - 0.5 for v in LAND_V.values()), LAND_V

def ghost_state(ts):
    if ts >= T_GHOST: return 0.0, V_GHOST
    return H0 - 0.5 * G * ts * ts, G * ts

def axis_ratio(d):  return 1.03 - 0.062 * d          # Pruppacher & Beard 1970

def fmt_clock(ts):
    m, s = divmod(int(round(ts)), 60)
    return f"{m}:{s:02d}"

# event times (video seconds), all from the sim
SPEED_MARKS = [100, 200, 300, 400, 500, 600, 700]     # km/h crossed by the ghost
SPEED_EV = [vid_t((k / 3.6) / G) for k in SPEED_MARKS if (k / 3.6) / G > S0 + 0.3]
ALT_MARKS = [1750, 1500, 1250, 1000, 750, 500, 250]    # m crossed by "you" (5 mm)
def _cross(vt, alt):
    lo, hi = 0.0, LAND["5 mm"]
    for _ in range(80):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if drop_state(vt, m)[0] > alt else (lo, m)
    return hi
ALT_EV = [vid_t(_cross(9.09, a)) for a in ALT_MARKS]

# =====================================================================
# HELPERS (from cycloid_race.py)
# =====================================================================
def font(size, bold=True):
    k = (int(size), bold)
    if k not in _FONTS:
        _FONTS[k] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, int(size))
    return _FONTS[k]

def mathtext_png(line):
    fig = plt.figure(figsize=(W / EQ_DPI, EQ_BLOCK_H / EQ_DPI), dpi=EQ_DPI)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, line, ha="center", va="center", fontsize=EQ_FONTSIZE, color="#e8f4ff",
             path_effects=[pe.withStroke(linewidth=EQ_STROKE_W, foreground=BG_COLOR)])
    buf = io.BytesIO()
    fig.savefig(buf, dpi=EQ_DPI, transparent=True, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").resize((W, EQ_BLOCK_H), Image.LANCZOS)

def alpha_comp(base, lay, x, y, alpha=1.0):
    if alpha <= 0.01: return
    base.alpha_composite(lay if alpha >= 0.999 else _fade(lay, alpha), (int(x), int(y)))

def _fade(lay, a):
    l = lay.copy()
    l.putalpha(l.getchannel("A").point(lambda p: int(p * a)))
    return l

def box_text(text, size, fg, bg, pad=(26, 14), bold=True, radius=14):
    f = font(size, bold)
    tmp = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    lines = text.split("\n")
    wmax = max(tmp.textlength(l, font=f) for l in lines)
    lh = int(size * 1.2)
    im = Image.new("RGBA", (int(wmax + 2 * pad[0]), lh * len(lines) + 2 * pad[1]), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], radius=radius, fill=bg + (255,))
    for i, l in enumerate(lines):
        d.text((im.size[0] / 2, pad[1] + i * lh), l, font=f, fill=fg, anchor="ma")
    assert im.size[0] <= W - 60, f"box too wide: {text!r}"
    return im

def plain_text(text, size, fill, bold=True, stroke=TEXT_STROKE):
    f = font(size, bold)
    tmp = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    w = int(tmp.textlength(text, font=f)) + 2 * stroke + 4
    im = Image.new("RGBA", (w, int(size * 1.3) + 2 * stroke), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((stroke + 2, stroke), text, font=f, fill=fill,
                            stroke_width=stroke, stroke_fill=BG_RGB + (255,))
    return im

def ebur_measure(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    t = r.stderr
    js = json.loads(t[t.rindex("{"):t.rindex("}") + 1])
    return dict(I=float(js["input_i"]), TP=float(js["input_tp"]), LRA=float(js["input_lra"]))

def write_wav(path, st):
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())

def limiter(st, ceil_db):
    ceil = 10 ** (ceil_db / 20)
    for _ in range(2):
        os4 = np.max(np.abs(resample_poly(st, 4, 1, axis=0)), axis=1)
        pk = os4.reshape(-1, 4).max(axis=1)[:len(st)]
        pk = np.pad(pk, (0, len(st) - len(pk)))
        gr = 1 - np.minimum(1, ceil / np.maximum(pk, 1e-9))
        L = int(0.004 * SR)
        gr = uniform_filter1d(maximum_filter1d(gr, size=2 * L + 1), size=L + 1)
        rel = uniform_filter1d(maximum_filter1d(gr, size=int(0.05 * SR)), size=int(0.05 * SR))
        st = st * (1 - np.maximum(gr, rel * 0.9))[:, None]
    return np.clip(st, -ceil, ceil)

# =====================================================================
# WORLD (static layers, built once)
# =====================================================================
def build_world():
    rng = np.random.default_rng(SEED)
    bg = Image.new("RGBA", (W, H), BG_RGB + (255,))
    # clouds: soft blobs, brighter tops, two layers
    cl = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(46):
        cx, cy = rng.uniform(-80, W + 80), rng.uniform(330, 470)
        r = rng.uniform(60, 150)
        cl += np.exp(-(((xx - cx) / (r * 1.6)) ** 2 + ((yy - cy) / r) ** 2)) * rng.uniform(0.4, 1.0)
    cl = np.clip(cl, 0, 1.6) / 1.6
    shade = np.clip(1.0 - (yy - 330) / 260.0, 0.35, 1.0)
    rgb = np.array(BG_RGB, np.float32)[None, None, :] * (1 - cl[..., None]) + \
          np.array([58, 74, 92], np.float32)[None, None, :] * cl[..., None] * shade[..., None] + \
          np.array([22, 32, 44], np.float32)[None, None, :] * cl[..., None] * (1 - shade[..., None])
    ramp = np.clip((yy - 250) / 110.0, 0, 1)[..., None]
    rgb = np.array(BG_RGB, np.float32)[None, None, :] * (1 - ramp) + rgb * ramp
    cl = cl * ramp[..., 0]
    LAYERS["cloud_mask"] = cl
    bg = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(bg)
    # skyline: Kolkata-style rooftops with water tanks, a few warm windows
    x = -10
    sky = (20, 38, 50)
    while x < W + 10:
        w = int(rng.uniform(60, 140)); h = int(rng.uniform(40, 120))
        d.rectangle([x, Y_GROUND - h, x + w, Y_GROUND], fill=sky)
        if rng.random() < 0.55:                                   # water tank
            tx = x + int(rng.uniform(8, w - 38))
            d.rectangle([tx, Y_GROUND - h - 26, tx + 30, Y_GROUND - h], fill=sky)
            d.rectangle([tx + 4, Y_GROUND - h - 30, tx + 26, Y_GROUND - h - 26], fill=sky)
        for _ in range(int(rng.integers(0, 4))):
            wx = x + int(rng.uniform(8, max(9, w - 16))); wy = Y_GROUND - int(rng.uniform(14, max(15, h - 10)))
            d.rectangle([wx, wy, wx + 9, wy + 12], fill=C_WINDOW if rng.random() < 0.7 else (160, 200, 230))
        x += w + int(rng.uniform(0, 14))
    # wet street with reflection strip
    d.rectangle([0, Y_GROUND, W, Y_GROUND + 22], fill=(26, 48, 62))
    d.line([0, Y_GROUND, W, Y_GROUND], fill=(90, 140, 170), width=2)
    # altitude axis
    for alt in range(0, 2001, 250):
        y = Y_TOP + (H0 - alt) * PX_M
        L = 22 if alt % 1000 == 0 else 11
        d.line([52, y, 52 + L, y], fill=(120, 160, 190), width=2)
    for alt, lab in [(2000, "2 km"), (1000, "1 km")]:
        y = Y_TOP + (H0 - alt) * PX_M
        d.text((80, y), lab, font=font(24, False), fill=(150, 185, 210), anchor="lm")
    LAYERS["world"] = bg
    # rain streaks (loop-perfect: each streak travels k * span over the Short)
    span = H + 200
    LAYERS["rain"] = [(rng.uniform(0, W), rng.uniform(0, span), int(rng.integers(2, 5)) * span / DURATION,
                       rng.uniform(18, 46), rng.uniform(0.10, 0.32)) for _ in range(RAIN_N)]
    # text
    LAYERS["title"] = box_text("POV: YOU'RE A RAINDROP", 60, C_TITLE_TXT, C_TITLE_BOX, pad=(30, 16))
    LAYERS["hook"] = box_text("how fast do you hit?", 40, (20, 22, 28), (245, 247, 250), pad=(22, 10))
    LAYERS["wm"] = plain_text("@cosmoscoded", 28, (220, 230, 240, 200), False, 2)
    LAYERS["series"] = plain_text("Weird Physics #2", 26, (255, 214, 64, 230), True, 2)
    LAYERS["eq1"] = mathtext_png(r"$m\,\frac{dv}{dt} = mg - \frac{1}{2}\rho\, C_d A\, v^2$")
    LAYERS["eq2"] = mathtext_png(r"$v_t = \sqrt{2mg\,/\,\rho C_d A}\qquad z_{no\ air} = H - \frac{1}{2}g t^2$")
    LAYERS["legend1"] = plain_text("ghost = no air", 30, (205, 225, 240, 255), False, 3)
    LAYERS["legend2"] = plain_text("drag grows with speed, so you cruise", 30, (205, 225, 240, 255), False, 3)
    for k, txt in STICKERS:
        LAYERS["st_" + k] = box_text(txt, 40, (20, 22, 28), C_STICKER, pad=(24, 12), radius=10)
    LAYERS["src"] = plain_text("speeds: Gunn & Kinzer 1949 · shape: Pruppacher & Beard 1970", 21, (150, 175, 195, 255), False, 2)

L5 = LAND["5 mm"]; L2 = LAND["2 mm"]; L1 = LAND["1 mm"]
def _mmss(t):
    m, s = divmod(int(round(t)), 60)
    return f"{m} min {s} s"
STICKERS = [
    ("a", "no air: it just keeps speeding up"),
    ("b", f"{V_GHOST * 3.6:.0f} km/h. almost 5x a fast bowler"),
    ("c", f"real air caps you at {9.09 * 3.6:.0f} km/h"),
    ("d", "you're not a tear. you're a bun"),
    ("e", f"you landed after ~{_mmss(L5)}"),
    ("f", f"1 mm drops: {4.03 * 3.6:.1f} km/h, still up there"),
    ("g", "air is why rain doesn't hurt\npick your size: 1, 2 or 5 mm?"),
]
STICKER_T = [("a", 1.6, T_IMPACT_V), ("b", T_IMPACT_V, T_IMPACT_V + 4.6), ("c", T_IMPACT_V + 4.6, 16.0),
             ("d", 16.0, LAND_V["5 mm"]), ("e", LAND_V["5 mm"], LAND_V["2 mm"]),
             ("f", LAND_V["2 mm"], LAND_V["1 mm"]), ("g", LAND_V["1 mm"], DURATION + 1)]

# =====================================================================
# DRAW
# =====================================================================
def y_of(z):  return Y_TOP + (H0 - z) * PX_M

DROP_SCALE = 1.8                     # drops drawn magnified (shape true to scale between sizes)
def draw_drop(d, cx, cy, d_mm, col, a, scale=DROP_SCALE):
    a_ax = (5 + 3.2 * d_mm) * scale
    q = axis_ratio(d_mm)
    th = np.linspace(0, 2 * np.pi, 60)
    xs = a_ax * np.cos(th)
    ys = -a_ax * q * np.sin(th)
    flat = max(0.0, (d_mm - 2.5) / 2.5)                     # big drops: flattened base
    ys = np.where(ys > 0, ys * (1 - 0.55 * flat), ys)
    pts = list(zip(cx + xs, cy + ys))
    c = tuple(int(v * a) for v in col)
    d.polygon(pts, fill=c)
    d.ellipse([cx - a_ax * 0.45, cy - a_ax * q * 0.65, cx - a_ax * 0.1, cy - a_ax * q * 0.3],
              fill=tuple(min(255, int(v * a + 60 * a)) for v in col))

def render_frame(fi):
    tv = fi / FPS
    ts = sim_t(tv)
    im = LAYERS["world"].copy()
    fx = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(fx)
    # rain
    span = H + 200
    for x0, y0, sp, ln, al in LAYERS["rain"]:
        y = (y0 + sp * tv) % span - 100
        if y < 300: continue
        x = x0 - 0.12 * (y - 300)
        x %= W
        d.line([x, y, x - 0.12 * ln, y + ln], fill=(150, 190, 220, int(255 * al)), width=2)
    # loop fade: last seconds new drops form at the cloud (frame 0 state)
    loop_a = float(np.clip((tv - (DURATION - LOOP_FADE_V)) / LOOP_FADE_V, 0, 1))
    ts0 = sim_t(0.0)
    # impact flash + ripple rings at landings
    for key, tl in LAND_V.items():
        if tl <= tv < tl + 2.5:
            k = (tv - tl) / 2.5
            cx = COL_X[key]
            big = key == "ghost"
            for j in range(3 if big else 2):
                kk = k - 0.12 * j
                if kk <= 0: continue
                rx = (40 + (300 if big else 110) * kk); ry = rx * 0.14
                d.ellipse([cx - rx, Y_GROUND + 8 - ry, cx + rx, Y_GROUND + 8 + ry],
                          outline=(200, 230, 255, int(220 * (1 - kk))), width=3)
    # ghost
    zg, vg = ghost_state(ts)
    gx = COL_X["ghost"]
    if ts < T_GHOST:
        yg = y_of(zg)
        trail = min(320, 60 + vg * 1.4)
        for j in range(14):
            f0 = j / 14
            d.line([gx, yg - trail * f0, gx, yg - trail * (f0 + 1 / 14)],
                   fill=(200, 205, 230, int(150 * (1 - f0))), width=int(10 * (1 - f0)) + 2)
        draw_drop(d, gx, yg, 5.0, C_GHOST, 0.75)
        lab = plain_text(f"{vg * 3.6:.0f} km/h", 34, (235, 238, 250, 255), True, 3)
        fx.alpha_composite(lab, (int(gx - lab.size[0] / 2), int(yg + 34)))
    elif tv < DURATION - LOOP_FADE_V:
        lab = plain_text(f"hit: {V_GHOST * 3.6:.0f} km/h", 30, (255, 220, 200, 255), True, 3)
        fx.alpha_composite(lab, (int(gx - lab.size[0] / 2) + 10, Y_GROUND + 26 - 70))
    hd = plain_text("no air", 28, (200, 205, 230, 255), True, 3)
    fx.alpha_composite(hd, (int(gx - hd.size[0] / 2), Y_TOP - 92))
    # real drops
    for name, dmm, vt in DROPS:
        cx = COL_X[name]
        z, v = drop_state(vt, ts)
        you = name == "5 mm"
        col = C_YOU if you else C_DROP
        hdr = plain_text(name + (" · you" if you else ""), 28, col + (255,), True, 3)
        fx.alpha_composite(hdr, (int(cx - hdr.size[0] / 2), Y_TOP - 92))
        if z > 0:
            cy = y_of(z)
            for j in range(6):                                # short motion trail
                d.ellipse([cx - 2, cy - 8 - j * 7 * (v / 9.1), cx + 2, cy - 4 - j * 7 * (v / 9.1)],
                          fill=col + (int(90 * (1 - j / 6)),))
            draw_drop(d, cx, cy, dmm, col, 1.0)
            lab = plain_text(f"{v * 3.6:.1f} km/h", 30, col + (255,), True, 3)
            fx.alpha_composite(lab, (int(cx - lab.size[0] / 2), int(cy + (5 + 3.2 * dmm) * DROP_SCALE * axis_ratio(dmm) + 14)))
        elif tv < DURATION - LOOP_FADE_V:
            lab = plain_text(f"landed ~{fmt_clock(LAND[name])}", 28, col + (255,), True, 3)
            fx.alpha_composite(lab, (int(cx - lab.size[0] / 2), Y_GROUND - 48))
    # loop: drops re-forming at the frame-0 state
    if loop_a > 0:
        for name, dmm, vt in DROPS + [("ghost", 5.0, None)]:
            z = ghost_state(ts0)[0] if vt is None else drop_state(vt, ts0)[0]
            draw_drop(d, COL_X[name], y_of(z), dmm, C_GHOST if vt is None else (C_YOU if name == "5 mm" else C_DROP), loop_a)
    glow = fx.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BILINEAR)
    im.alpha_composite(glow)
    im.alpha_composite(fx)
    # flash on the ghost impact (lightning-white, decays)
    if T_IMPACT_V <= tv < T_IMPACT_V + 0.5:
        a = 1 - (tv - T_IMPACT_V) / 0.5
        im.alpha_composite(Image.new("RGBA", (W, H), (230, 240, 255, int(110 * a))))
    # frame-1 lightning flicker inside the cloud (and mirrored before the loop)
    for t0 in (0.0, DURATION - 0.25):
        if t0 <= tv < t0 + 0.25:
            a = 1 - (tv - t0) / 0.25
            m = (LAYERS["cloud_mask"] * 140 * a).astype(np.uint8)
            im.alpha_composite(Image.merge("RGBA", [Image.new("L", (W, H), 225)] * 3 + [Image.fromarray(m)]))
    # text layer
    alpha_comp(im, LAYERS["wm"], 40, 34, 0.9)
    alpha_comp(im, LAYERS["series"], W - LAYERS["series"].size[0] - 40, 34, 1.0)
    t = LAYERS["title"]; alpha_comp(im, t, (W - t.size[0]) / 2, 80)
    hk = LAYERS["hook"]; alpha_comp(im, hk, (W - hk.size[0]) / 2, 80 + t.size[1] + 12)
    clock = box_text(f"real time {fmt_clock(ts)}   ·   x{round(_warp(tv), 1):g} speed", 30, (230, 240, 250),
                     (24, 44, 58), pad=(18, 8), bold=True, radius=20)
    alpha_comp(im, clock, (W - clock.size[0]) / 2, 80 + t.size[1] + 12 + hk.size[1] + 10)
    if tv >= 2.0:
        a = min(1.0, (tv - 2.0) / 0.4)
        eqk, lgk = ("eq1", "legend1") if tv < 17.0 else ("eq2", "legend2")
        alpha_comp(im, LAYERS[eqk], 0, EQ_Y, a)
        lg = LAYERS[lgk]; alpha_comp(im, lg, (W - lg.size[0]) / 2, LEGEND_Y, a)
    for k, t0, t1 in STICKER_T:
        if t0 <= tv < t1:
            a = min(1.0, (tv - t0) / 0.25, (t1 - tv) / 0.25 if t1 <= DURATION else 1.0)
            st = LAYERS["st_" + k]; alpha_comp(im, st, (W - st.size[0]) / 2, STICKER_Y, max(a, 0))
    return np.asarray(im.convert("RGB"))

# =====================================================================
# MUSIC  (raag Megh notes on the raga engine; every event from the sim)
# =====================================================================
MEGH = [0, 2, 5, 7, 10]               # S R M P n
def b_of(tv): return tv / BEAT_S
def synth_audio():
    song = ge.Song("raga", bpm=BPM, total_beats=TOTAL_BEATS, key="C#", seed=SEED)
    song.g = dict(song.g, scale=MEGH, lead="santoor")
    song.sections = [("intro", 0, 4, 0.6), ("fall", 4, 10, 0.9), ("drop", 11, 60, 1.0), ("payoff", 60, 74, 1.0)]
    song.silent_beats = [10]
    ev = [dict(beat=0.0, kind="big")]
    for j, b in enumerate([0.0, 0.5, 1.0, 1.5]):           # hook motif: falling n P R S
        ev.append(dict(beat=b, kind="hit", idx=[4, 3, 1, 0][j] + 5, vel=0.9))
    for j, t in enumerate(SPEED_EV):                         # ghost speed thresholds -> rising notes
        if t < T_IMPACT_V - BEAT_S:
            ev.append(dict(beat=b_of(t), kind="hit", idx=j + 2, vel=0.8))
    ev.append(dict(beat=IMPACT_BEAT - 0.001, kind="rise", len=4))
    ev.append(dict(beat=IMPACT_BEAT, kind="big"))
    for t in ALT_EV:                                         # "you" passes each 250 m mark
        ev.append(dict(beat=b_of(t), kind="tick"))
    for name, idx in [("5 mm", 0), ("2 mm", 3), ("1 mm", 5)]:   # bigger drop = lower note
        ev.append(dict(beat=b_of(LAND_V[name]), kind="big" if name != "2 mm" else "chord"))
        ev.append(dict(beat=b_of(LAND_V[name]), kind="hit", idx=idx + 5, vel=1.0))
    pb = b_of(LAND_V["1 mm"])
    for j, d in enumerate([0, 0.5, 1.0, 1.5]):               # hook motif returns at the payoff
        ev.append(dict(beat=pb + 1 + d, kind="hit", idx=[4, 3, 1, 0][j] + 5, vel=0.9))
    song.events = ev
    st = song.render().astype(np.float64)
    write_wav(TMP_AUDIO, st)
    m = ebur_measure(TMP_AUDIO)
    st = st * 10 ** ((TARGET_LUFS - m["I"]) / 20)
    st = limiter(st, TARGET_TP - 0.3)
    write_wav(TMP_AUDIO, st)
    m = ebur_measure(TMP_AUDIO)
    st = limiter(st * 10 ** ((TARGET_LUFS - m["I"]) / 20), TARGET_TP - 1.3)   # AAC adds ~0.9 dB overshoot
    write_wav(TMP_AUDIO, st)
    return [e for e in ev]

# =====================================================================
# OUTPUTS
# =====================================================================
SHEET = [("hook f1", 0.0), ("ghost 500+", 5.6), ("impact", T_IMPACT_V + 0.1), ("bun", 18.0),
         ("you landed", LAND_V["5 mm"] + 0.3), ("payoff", LAND_V["1 mm"] + 0.8)]

def contact_sheet():
    tiles = []
    for name, ts in SHEET:
        tile = Image.fromarray(render_frame(int(round(ts * FPS))))
        ImageDraw.Draw(tile).text((20, H - 70), f"{name} {ts:.2f}s", font=font(46), fill=(255, 80, 80))
        tiles.append(tile.resize((W // 2, H // 2), Image.LANCZOS))
    sheet = Image.new("RGB", (3 * W // 2, 2 * H // 2), BG_RGB)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * W // 2, (i // 3) * H // 2))
    sheet.save(OUT_SHEET, quality=88)

def write_beats(ev):
    out = dict(duration_s=round(DURATION, 3), bpm=BPM, s0_real=round(S0, 3), t_ghost=round(T_GHOST, 3),
               v_ghost_kmh=round(V_GHOST * 3.6, 1), land_real_s={k: round(v, 1) for k, v in LAND.items()},
               land_video_s={k: round(v, 3) for k, v in LAND_V.items()},
               speed_marks_video_s=[round(t, 3) for t in SPEED_EV], alt_marks_video_s=[round(t, 3) for t in ALT_EV],
               stickers=[dict(text=dict(STICKERS)[k], t0=round(a, 2), t1=round(min(b, DURATION), 2)) for k, a, b in STICKER_T],
               events=[dict(t=round(e["beat"] * BEAT_S, 4), kind=e["kind"]) for e in ev])
    json.dump(out, open(OUT_BEATS, "w"), indent=1)

def audio_report():
    m = ebur_measure(OUT_VIDEO)
    open(OUT_LEVELS, "w").write(f"I {m['I']:.2f} LUFS (target {TARGET_LUFS})\nTP {m['TP']:.2f} dBTP\nLRA {m['LRA']:.2f}\n")
    print(open(OUT_LEVELS).read())

def encode_upload():
    kbps = int(UPLOAD_TARGET_MB * 8e6 / DURATION / 1000) - UPLOAD_AUDIO_K
    plog = os.path.join(OUT_DIR, "_x264pass")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_VIDEO, "-c:v", "libx264", "-preset", "medium",
            "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-passlogfile", plog]
    subprocess.run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull], check=True)
    subprocess.run(base + ["-pass", "2", "-c:a", "copy", "-movflags", "+faststart", OUT_UPLOAD], check=True)
    for f in os.listdir(OUT_DIR):
        if f.startswith("_x264pass"): os.remove(os.path.join(OUT_DIR, f))
    print(f"upload {os.path.getsize(OUT_UPLOAD) / 1e6:.2f} MB")

def main():
    print(f"dur {DURATION:.1f}s S0={S0:.2f} ghost {T_GHOST:.2f}s {V_GHOST*3.6:.0f} km/h; land real {LAND}; video {LAND_V}")
    build_world()
    if "--sheet" in sys.argv:
        contact_sheet(); return
    ev = synth_audio()
    write_beats(ev)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", TMP_AUDIO, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT_VIDEO]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(N_FRAMES):
        p.stdin.write(render_frame(fi).tobytes())
        if fi % 200 == 0: print("frame", fi, flush=True)
    p.stdin.close(); p.wait()
    os.remove(TMP_AUDIO)
    contact_sheet()
    audio_report()
    encode_upload()
    print("DONE")

if __name__ == "__main__":
    main()
