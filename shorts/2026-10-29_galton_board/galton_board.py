"""
Coded Cosmos — Galton board (hypnotic physics #2), 29 Oct 2026
Format F2 (prediction challenge) · Look L7 (blueprint) · Music: Brazilian funk 130 BPM (genre_engine "funk")

5,000 balls fall through 12 rows of pegs. At every peg a ball goes left or right, 50/50 (ideal board).
After 12 rows it lands in bin k = number of rights, so
    P(k) = C(12, k) / 2^12        (binomial; 2^12 = 4,096 equally likely paths)
Middle bin: 924/4096 = 22.6 %.  Each edge bin: 1/4096, so 5,000 balls expect ~1.2 there.
The pile tends to a normal curve, mean 6, SD sqrt(12)/2 = 1.73 bins (de Moivre-Laplace).
Galton described the board in Natural Inheritance (1889).
The bins are counted from THIS run (seed 29), not typed in. The dashed curve is the formula.

Sound: every peg hit = a tiny click (ASMR layer, binned per 10 ms, panned by x);
ball landings = glass pings pitched by bin (max one per 16th, at the exact landing time);
an edge-bin landing = big hit + flash; 1,000 / 2,500 / 5,000 landed = chord stabs;
drop on beat 16 when the formula curve snaps on. All synthesised, no samples.
"""
import os, io, sys, json, math, wave, subprocess
from math import comb
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_music"))
from genre_engine import Song, SR, highpass

# =====================================================================
# GLOBALS
# =====================================================================
SLUG          = "galton_board"
OUT_DIR       = HERE
OUT_VIDEO     = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD    = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET     = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_BEATS     = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO     = os.path.join(OUT_DIR, "_gb_audio.wav")

W, H, FPS     = 1080, 1920, 30
GENRE         = "funk"
BPM           = 130
KEY           = "A"
BEAT          = 60.0 / BPM
TOTAL_BEATS   = 64                      # 29.54 s
DURATION      = TOTAL_BEATS * BEAT
N_FRAMES      = int(round(DURATION * FPS))
DROP_BEAT     = 16                      # formula curve snaps on + beat drops
TARGET_LUFS   = -14.0
TARGET_TP     = -1.0
LIMIT_DB      = -3.0
UPLOAD_MB     = 9.0
UPLOAD_AUDIO_K= 160

# --- physics / sim ---
N_BALLS       = 5000
N_ROWS        = 12
SEED          = 29
T_HOP         = 0.085                   # s per peg row
T_FALL        = 0.32                    # s from last row to the pile
REL_START     = -1.3                    # pre-roll so frame 1 is already busy
LAND_LAST     = 25.6                    # last ball lands here (video s)
RATE0         = 40.0                    # balls/s at the start, ramps up

# --- look L7 blueprint (ONE BG value everywhere) ---
BG_COLOR      = "#0e3a6e"
GRID_MINOR    = "#174a85"
GRID_MAJOR    = "#24609f"
INK           = "#e8f3ff"
CYAN          = "#7fd8ff"
HILITE        = "#ffd84d"
FONT_BOLD     = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_SEMI     = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_MONO     = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

CX            = W // 2
DX            = 62                      # peg spacing (px) = bin width
PEG_Y0        = 440
PEG_DY        = 40
BIN_TOP       = PEG_Y0 + PEG_DY * N_ROWS + 6
BIN_BOT       = 1205
PX_PER_BALL   = (BIN_BOT - BIN_TOP - 20) / 1250
TITLE_Y       = 182
READ_Y        = 352
COUNT_Y       = 1228
EQ_Y          = 1262
SERIES_Y      = 1418
STICKER_Y     = 1486
EQ_DPI        = 200
EQ_FONTSIZE   = 15
EQ_STROKE_W   = 4
GLOW_R        = 7

TITLE_TXT     = "WHERE DO 5,000 BALLS LAND?"
HOOK_TXT      = "guess the far-left bin"
SERIES_TXT    = "hypnotic physics #2"
STICKER_TXT   = "far left: 0, 1, 2 or 3+?"
EQ_LINE       = r"$P(k) = \binom{12}{k}\,/\,2^{12}$"
LEGEND_TXT    = "12 coin flips per ball"
EQ_FROM_S     = 2.0
CURVE_LABEL   = "dashed = the formula"


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


BG = hex_rgb(BG_COLOR)
_F = {}


def font(path, size):
    k = (path, size)
    if k not in _F:
        _F[k] = ImageFont.truetype(path, size)
    return _F[k]


# =====================================================================
# SIM (deterministic, seed 29)
# =====================================================================
rng = np.random.default_rng(SEED)
STEPS = (rng.random((N_BALLS, N_ROWS)) < 0.5).astype(np.int8)      # 1 = right
BIN = STEPS.sum(1)
FINAL = np.bincount(BIN, minlength=N_ROWS + 1)
P_K = np.array([comb(N_ROWS, k) for k in range(N_ROWS + 1)]) / 2 ** N_ROWS
T_FLIGHT = N_ROWS * T_HOP + T_FALL

# release times: N(t) = RATE0*t + c2*t^2 over [REL_START, LAND_LAST - T_FLIGHT]
_TR = LAND_LAST - T_FLIGHT - REL_START
_C2 = (N_BALLS - RATE0 * _TR) / _TR ** 2
_u = np.arange(N_BALLS) + 0.5
REL = REL_START + (-RATE0 + np.sqrt(RATE0 ** 2 + 4 * _C2 * _u)) / (2 * _C2)
LAND = REL + T_FLIGHT
CUMX = np.concatenate([np.zeros((N_BALLS, 1)), np.cumsum(2 * STEPS - 1, axis=1)], axis=1) * DX / 2  # x offset after row r
BIN_X = CX + (np.arange(N_ROWS + 1) - N_ROWS / 2) * DX


def peg_xy(r, j):
    return CX + (j - r / 2) * DX, PEG_Y0 + r * PEG_DY


def ball_pos(idx, tv):
    """positions of balls idx (in flight) at video time tv."""
    s = tv - REL[idx]
    r = np.clip((s // T_HOP).astype(int), 0, N_ROWS)
    f = np.clip(s / T_HOP - r, 0, 1)
    on_pegs = s < N_ROWS * T_HOP
    x0 = CUMX[idx, np.minimum(r, N_ROWS)]
    x1 = CUMX[idx, np.minimum(r + 1, N_ROWS)]
    x = CX + x0 + (x1 - x0) * f
    y = PEG_Y0 - 14 + r * PEG_DY + f * PEG_DY - 22 * np.sin(np.pi * f) * 0.6     # small hop off each peg
    # falling into the bin
    sf = np.clip((s - N_ROWS * T_HOP) / T_FALL, 0, 1)
    xb = CX + CUMX[idx, N_ROWS]
    yb_top = BIN_TOP
    yb_end = BIN_BOT - 10
    yb = yb_top + (yb_end - yb_top) * sf ** 1.6
    x = np.where(on_pegs, x, xb)
    y = np.where(on_pegs, y, yb)
    return x, y


def counts_at(tv):
    landed = LAND <= tv
    return np.bincount(BIN[landed], minlength=N_ROWS + 1), int(landed.sum())


def edge_events():
    e = [(float(LAND[i]), int(BIN[i])) for i in range(N_BALLS) if BIN[i] in (0, N_ROWS) and 0 <= LAND[i] < DURATION]
    return sorted(e)


EDGE_EVENTS = edge_events()
MILESTONES = [(float(LAND[m - 1]), m) for m in (1000, 2500, 5000)]


# =====================================================================
# AUDIO
# =====================================================================
def peg_click_layer(n):
    """ASMR layer: every peg hit = tiny click; hits binned per 10 ms, gain ~ sqrt(count)."""
    out = np.zeros((n, 2))
    hit_t = (REL[:, None] + np.arange(N_ROWS)[None, :] * T_HOP).ravel()
    hit_x = (CUMX[:, :N_ROWS]).ravel() / (DX * 6)
    ok = (hit_t >= 0) & (hit_t < DURATION)
    hit_t, hit_x = hit_t[ok], hit_x[ok]
    blk = (hit_t / 0.01).astype(int)
    lr = np.random.default_rng(7)
    cl = highpass(lr.standard_normal(int(0.004 * SR)), 3500) * np.exp(-np.arange(int(0.004 * SR)) / (0.0009 * SR))
    for b in np.unique(blk):
        sel = blk == b
        c = sel.sum()
        pan = float(np.clip(hit_x[sel].mean(), -0.8, 0.8))
        i = int(b * 0.01 * SR + lr.integers(0, int(0.008 * SR)))
        g = 0.05 * min(c, 40) ** 0.5
        L = min(len(cl), n - i)
        if L <= 0:
            continue
        out[i:i + L, 0] += cl[:L] * g * math.cos((pan + 1) * math.pi / 4)
        out[i:i + L, 1] += cl[:L] * g * math.sin((pan + 1) * math.pi / 4)
    return out


def build_audio():
    song = Song(GENRE, bpm=BPM, total_beats=TOTAL_BEATS, key=KEY, seed=29)
    song.sections = [("intro", 0, DROP_BEAT, 0.7), ("main", DROP_BEAT, TOTAL_BEATS - 6, 1.0),
                     ("out", TOTAL_BEATS - 6, TOTAL_BEATS, 0.8)]
    song.silent_beats = [DROP_BEAT - 1]
    evs = []
    # landing pings: at most one per 16th slot, exact landing time, pitch by bin
    used = set()
    for i in np.argsort(LAND):
        t = LAND[i]
        if not (0 <= t < DURATION):
            continue
        slot = int(t / (BEAT / 4))
        if slot in used:
            continue
        used.add(slot)
        evs.append(dict(beat=t / BEAT, kind="hit", idx=3 + int(BIN[i]), vel=0.30, dur=0.9,
                        pan=-0.7 + 1.4 * BIN[i] / N_ROWS))
    for t, k in EDGE_EVENTS:
        evs.append(dict(beat=t / BEAT, kind="big"))
    for t, m in MILESTONES:
        if t < DURATION:
            evs.append(dict(beat=t / BEAT, kind="chord"))
    evs.append(dict(beat=DROP_BEAT, kind="rise", len=4))
    evs.append(dict(beat=DROP_BEAT, kind="big"))
    for b0 in range(0, TOTAL_BEATS, 8):                     # hook motif: 4 vowel-chop stabs every 2 bars
        for off, idx in ((0, 14), (0.75, 14), (1.5, 16), (3, 13)):
            if b0 + off < TOTAL_BEATS:
                evs.append(dict(beat=b0 + off, kind="hit", idx=idx, vel=0.9, inst="vox_chop", dur=0.3, pan=0.0))
    song.events = evs
    drop_t = DROP_BEAT * BEAT
    song.brightness = lambda t: 0.3 + 0.4 * (t / drop_t) if t < drop_t else 1.0
    st = song.render().astype(np.float64)
    st += peg_click_layer(len(st))
    st -= st.mean(axis=0)
    beats = dict(bpm=BPM, drop_beat=DROP_BEAT, edge_events=EDGE_EVENTS, milestones=MILESTONES,
                 final_counts=FINAL.tolist(), n_pings=len(used))
    json.dump(beats, open(OUT_BEATS, "w"), indent=1)
    return st


def write_wav(path, st):
    pcm = (np.clip(st, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())


def ebur(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    t = r.stderr
    js = json.loads(t[t.rindex("{"):t.rindex("}") + 1])
    return float(js["input_i"]), float(js["input_tp"])


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


def master(st):
    for _ in range(3):
        write_wav(TMP_AUDIO, limiter(st, LIMIT_DB))
        I, tp = ebur(TMP_AUDIO)
        if abs(I - TARGET_LUFS) < 0.3:
            break
        st = st * 10 ** ((TARGET_LUFS - I) / 20)
    print(f"audio: {I:.2f} LUFS, TP {tp:.2f}")


# =====================================================================
# STATIC LAYERS
# =====================================================================
def mathtext_png(line, fs):
    fig = plt.figure(figsize=(W / EQ_DPI, 150 / EQ_DPI), dpi=EQ_DPI)
    fig.patch.set_facecolor(BG_COLOR); fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, line, ha="center", va="center", fontsize=fs, color=INK,
             path_effects=[pe.withStroke(linewidth=EQ_STROKE_W, foreground=BG_COLOR)])
    b = io.BytesIO()
    fig.savefig(b, dpi=EQ_DPI, transparent=True, facecolor=BG_COLOR)
    plt.close(fig)
    b.seek(0)
    return Image.open(b).convert("RGBA").resize((W, 150), Image.LANCZOS)


def arrow(d, p0, p1, col, w=2):
    d.line([p0, p1], fill=col, width=w)
    for (a, b) in ((p0, p1), (p1, p0)):
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in (-1, 1):
            d.line([b, (b[0] - 14 * math.cos(ang + s * 0.4), b[1] - 14 * math.sin(ang + s * 0.4))], fill=col, width=w)


def background():
    im = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(0, W, 30):
        d.line([(x, 0), (x, H)], fill=GRID_MAJOR if x % 150 == 0 else GRID_MINOR, width=1)
    for y in range(0, H, 30):
        d.line([(0, y), (W, y)], fill=GRID_MAJOR if y % 150 == 0 else GRID_MINOR, width=1)
    ink = INK
    # drawing frame + registration marks
    d.rectangle([40, 60, W - 40, 1560], outline=CYAN, width=2)
    for (x, y) in ((40, 60), (W - 40, 60), (40, 1560), (W - 40, 1560)):
        d.line([(x - 18, y), (x + 18, y)], fill=ink, width=2)
        d.line([(x, y - 18), (x, y + 18)], fill=ink, width=2)
    # funnel
    d.line([(CX - 120, PEG_Y0 - 70), (CX - 16, PEG_Y0 - 30)], fill=ink, width=3)
    d.line([(CX + 120, PEG_Y0 - 70), (CX + 16, PEG_Y0 - 30)], fill=ink, width=3)
    # pegs with crosshairs
    for r in range(N_ROWS):
        for j in range(r + 1):
            x, y = peg_xy(r, j)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=ink, width=2)
            d.line([(x - 9, y), (x + 9, y)], fill=CYAN, width=1)
    # bin walls
    for k in range(N_ROWS + 2):
        x = BIN_X[0] - DX / 2 + k * DX
        d.line([(x, BIN_TOP + 30), (x, BIN_BOT)], fill=ink, width=2)
    d.line([(BIN_X[0] - DX / 2, BIN_BOT), (BIN_X[-1] + DX / 2, BIN_BOT)], fill=ink, width=3)
    # dimension: 12 rows (left), 13 bins (below)
    xl = BIN_X[0] - DX / 2 - 30
    arrow(d, (xl, PEG_Y0), (xl, PEG_Y0 + (N_ROWS - 1) * PEG_DY), CYAN)
    lbl = Image.new("RGBA", (260, 50), (0, 0, 0, 0))
    ImageDraw.Draw(lbl).text((130, 25), "12 ROWS", font=font(FONT_MONO, 26), anchor="mm", fill=CYAN)
    lbl = lbl.rotate(90, expand=True)
    im.paste(lbl, (int(xl - 48), int(PEG_Y0 + 5.5 * PEG_DY - 130)), lbl)
    d.text((BIN_X[0] - DX / 2 - 6, BIN_TOP + 8), "k=0", font=font(FONT_MONO, 20), anchor="rt", fill=CYAN)
    d.text((BIN_X[-1] + DX / 2 + 6, BIN_TOP + 8), "k=12", font=font(FONT_MONO, 20), anchor="lt", fill=CYAN)
    d.text((CX, 100), "@cosmoscoded", font=font(FONT_SEMI, 30), anchor="mm", fill=INK)
    d.text((W - 60, 76), "DWG GB-12 · 5000 BALLS · SEED 29", font=font(FONT_MONO, 18), anchor="rt", fill=CYAN)
    return im


def box_text(d, cx, cy, text, fnt, fill_box, fill_txt, pad=(28, 14), outline=None):
    w = d.textlength(text, font=fnt)
    asc, desc = fnt.getmetrics()
    h = asc + desc
    d.rounded_rectangle([cx - w / 2 - pad[0], cy - h / 2 - pad[1], cx + w / 2 + pad[0], cy + h / 2 + pad[1]],
                        radius=10, fill=fill_box, outline=outline, width=3 if outline else 0)
    d.text((cx, cy), text, font=fnt, anchor="mm", fill=fill_txt)


BG_IMG = None
EQ_IMG = None
CURVE_X = np.linspace(BIN_X[0] - DX / 2, BIN_X[-1] + DX / 2, 200)
_kx = (CURVE_X - CX) / DX + N_ROWS / 2
CURVE_PDF = np.exp(-0.5 * ((_kx - 6) / math.sqrt(3)) ** 2) / (math.sqrt(2 * math.pi) * math.sqrt(3))


# =====================================================================
# FRAME
# =====================================================================
def render_frame(i):
    global BG_IMG, EQ_IMG
    if BG_IMG is None:
        BG_IMG = background()
        EQ_IMG = mathtext_png(EQ_LINE, EQ_FONTSIZE)
    tv = i / FPS
    im = BG_IMG.copy()
    d = ImageDraw.Draw(im, "RGBA")
    cnt, landed = counts_at(tv)
    # pile bars (hatched blueprint fill)
    for k in range(N_ROWS + 1):
        h = cnt[k] * PX_PER_BALL
        if h <= 0:
            continue
        h = max(h, 4)
        x0, x1 = BIN_X[k] - DX / 2 + 5, BIN_X[k] + DX / 2 - 5
        y0 = BIN_BOT - h
        d.rectangle([x0, y0, x1, BIN_BOT - 2], fill=(127, 216, 255, 70), outline=INK, width=2)
        for yy in range(int(BIN_BOT - 2), int(y0), -10):
            d.line([(x0 + 2, yy), (min(x1 - 2, x0 + 2 + (yy - y0)), max(y0 + 2, yy - (x1 - x0)))], fill=(232, 243, 255, 90), width=1)
    # glow layer: balls in flight
    fl = np.where((REL <= tv) & (LAND > tv))[0]
    gl = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(gl)
    if len(fl):
        xs, ys = ball_pos(fl, tv)
        for x, y, b in zip(xs, ys, BIN[fl]):
            c = (255, 216, 77) if b in (0, N_ROWS) else (232, 243, 255)
            gd.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c)
    flash = 0.0
    for et, k in EDGE_EVENTS + [(DROP_BEAT * BEAT, -1)]:
        dt = tv - et
        if 0 <= dt < 0.5:
            flash = max(flash, (1 - dt / 0.5) ** 2)
    glow = gl.filter(ImageFilter.GaussianBlur(GLOW_R))
    base = np.asarray(im).astype(np.float32)
    base += np.asarray(glow).astype(np.float32) * (1.3 + flash) + np.asarray(gl).astype(np.float32) * 0.85
    if flash > 0:
        base += flash * 22
    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    # formula curve (dashed) from the drop
    drop_t = DROP_BEAT * BEAT
    if tv >= drop_t and landed > 0:
        a = min(1, (tv - drop_t) / 0.25)
        ycur = BIN_BOT - landed * CURVE_PDF * PX_PER_BALL
        pts = list(zip(CURVE_X, ycur))
        for s in range(0, len(pts) - 1, 2):
            d.line([pts[s], pts[s + 1]], fill=(255, 216, 77, int(255 * a)), width=4)
        d.text((BIN_X[10], BIN_TOP + 40), CURVE_LABEL, font=font(FONT_SEMI, 26), anchor="mm",
               fill=(255, 216, 77, int(255 * a)), stroke_width=4, stroke_fill=BG_COLOR)
    # edge bins highlight
    for k in (0, N_ROWS):
        x = BIN_X[k]
        col = HILITE if tv > 3 else INK
        d.line([(x - DX / 2 + 4, BIN_BOT - 50), (x - DX / 2 + 4, BIN_BOT - 2), (x + DX / 2 - 4, BIN_BOT - 2), (x + DX / 2 - 4, BIN_BOT - 50)], fill=col, width=4)
    # counts under the bins
    fm = font(FONT_MONO, 23)
    for k in range(N_ROWS + 1):
        col = HILITE if k in (0, N_ROWS) else INK
        d.text((BIN_X[k], COUNT_Y), f"{cnt[k]}", font=fm, anchor="mm", fill=col)
    # title + hook
    box_text(d, CX, TITLE_Y, TITLE_TXT, font(FONT_BOLD, 54), INK, BG_COLOR, pad=(30, 16))
    box_text(d, CX, TITLE_Y + 88, HOOK_TXT, font(FONT_SEMI, 36), BG_COLOR, HILITE, pad=(22, 10), outline=HILITE)
    # readouts from the sim
    fr = font(FONT_MONO, 32)
    mid = 100 * cnt[6] / landed if landed else 0.0
    d.text((CX - 250, READ_Y), f"landed {landed:4d}", font=fr, anchor="mm", fill=CYAN)
    d.text((CX + 30, READ_Y), f"mid {mid:4.1f}%", font=fr, anchor="mm", fill=CYAN)
    d.text((CX + 300, READ_Y), f"edges {cnt[0]}|{cnt[-1]}", font=fr, anchor="mm", fill=HILITE)
    # equation + legend
    if tv >= EQ_FROM_S - 0.4:
        a = min(1, (tv - (EQ_FROM_S - 0.4)) / 0.4)
        eq = np.asarray(EQ_IMG).astype(np.float32)
        b = np.asarray(im).astype(np.float32)
        y0 = EQ_Y - 20
        al = eq[..., 3:4] / 255 * a
        b[y0:y0 + 150] = b[y0:y0 + 150] * (1 - al) + eq[..., :3] * al
        im = Image.fromarray(b.astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        d.text((CX, EQ_Y + 100), LEGEND_TXT, font=font(FONT_SEMI, 34), anchor="mm",
               fill=(232, 243, 255, int(255 * a)), stroke_width=4, stroke_fill=BG_COLOR)
    # series + sticker / payoff
    if landed < N_BALLS:
        d.text((CX, SERIES_Y), SERIES_TXT, font=font(FONT_SEMI, 32), anchor="mm", fill=CYAN,
               stroke_width=5, stroke_fill=BG_COLOR)
    if landed >= N_BALLS:
        txt = f"far left {FINAL[0]}. far right {FINAL[-1]}."
        box_text(d, CX, STICKER_Y - 14, txt, font(FONT_BOLD, 40), HILITE, BG_COLOR, pad=(24, 12))
        d.text((CX, STICKER_Y + 46), "the formula said 1.2 each. luck voted.", font=font(FONT_SEMI, 36), anchor="mm",
               fill=INK, stroke_width=4, stroke_fill=BG_COLOR)
    else:
        box_text(d, CX, STICKER_Y, STICKER_TXT, font(FONT_BOLD, 40), HILITE, BG_COLOR, pad=(24, 12))
    return np.asarray(im)


# =====================================================================
# OUTPUT
# =====================================================================
SHEET_TIMES = [("hook", 0.0), ("pre-drop", 6.5), ("drop/curve", DROP_BEAT * BEAT + 0.1),
               ("mid", 16.0), ("late", 23.0), ("payoff", 27.5)]


def contact_sheet():
    tiles = []
    for name, ts in SHEET_TIMES:
        im = Image.fromarray(render_frame(int(round(ts * FPS))))
        ImageDraw.Draw(im).text((20, H - 70), f"{name} t={ts:.2f}", font=font(FONT_SEMI, 44), fill=(255, 80, 80))
        tiles.append(im.resize((W // 2, H // 2), Image.LANCZOS))
    sh = Image.new("RGB", (3 * W // 2, 2 * H // 2), BG_COLOR)
    for k, t in enumerate(tiles):
        sh.paste(t, ((k % 3) * W // 2, (k // 3) * H // 2))
    sh.save(OUT_SHEET, quality=88)


def render_video():
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", TMP_AUDIO,
                          "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                          "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", OUT_VIDEO],
                         stdin=subprocess.PIPE)
    for i in range(N_FRAMES):
        p.stdin.write(render_frame(i).tobytes())
        if i % 150 == 0:
            print("frame", i, flush=True)
    p.stdin.close()
    p.wait()


def encode_upload():
    kbps = int(UPLOAD_MB * 8e6 / DURATION / 1000) - UPLOAD_AUDIO_K
    plog = os.path.join(OUT_DIR, "_x264pass")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_VIDEO, "-c:v", "libx264", "-preset", "medium",
            "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-passlogfile", plog]
    subprocess.run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull], check=True)
    p2 = ["ffmpeg", "-y", "-loglevel", "error", "-i", OUT_VIDEO, "-i", TMP_AUDIO, "-map", "0:v", "-map", "1:a",
          "-c:v", "libx264", "-preset", "medium", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-passlogfile", plog,
          "-pass", "2", "-c:a", "aac", "-b:a", f"{UPLOAD_AUDIO_K}k", "-movflags", "+faststart", OUT_UPLOAD]
    subprocess.run(p2, check=True)
    for f in os.listdir(OUT_DIR):
        if f.startswith("_x264pass"):
            os.remove(os.path.join(OUT_DIR, f))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("final counts:", FINAL.tolist(), "expected:", np.round(P_K * N_BALLS, 1).tolist())
    print("edge events:", EDGE_EVENTS, "milestones:", MILESTONES)
    if mode in ("sheet", "all"):
        contact_sheet()
    if mode == "upload":
        encode_upload()
    if mode in ("all",):
        master(build_audio())
        render_video()
        encode_upload()
        os.remove(TMP_AUDIO)
        print("done")
