"""
Coded Cosmos — Pendulum wave (hypnotic physics #1), 28 Oct 2026
Format F5 (sync / emergence) · Look L5 (neon synthwave grid) · Music: hard techno 140 BPM (genre_engine)

20 simple pendulums. Pendulum n (n = 0..19) is cut so it makes exactly 20+n full swings in 30 s:
    T_n = 30 s / (20+n),   L_n = g T_n^2 / (4 pi^2)        (small-angle simple pendulum)
Seen from above, the bobs make snakes, braids, 2/3/4/5/6 groups, and at 30 s all 20 are back in step.
The video is exactly one 30 s cycle, so it loops perfectly.

Sound: every bob reaching its right-hand turning point = one glass ping (pitch set by its row),
exact time k*T_n from the physics. When the pendulums line up, the pings land together and stack
into a chord. Phase groups of 2 and 3 and the full re-sync = "big" hits; filter opens as the cycle
heads to the re-sync. All sound synthesised in genre_engine.py, no samples.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_music"))
sys.path.insert(0, HERE)
from genre_engine import Song, GENRES, SR

# =====================================================================
# GLOBALS
# =====================================================================
SLUG          = "pendulum_wave"
OUT_DIR       = HERE
OUT_VIDEO     = os.path.join(OUT_DIR, SLUG + ".mp4")
OUT_UPLOAD    = os.path.join(OUT_DIR, SLUG + "_upload.mp4")
OUT_SHEET     = os.path.join(OUT_DIR, "preview_frames.jpg")
OUT_LEVELS    = os.path.join(OUT_DIR, "audio_levels.txt")
OUT_BEATS     = os.path.join(OUT_DIR, "beats.json")
TMP_AUDIO     = os.path.join(OUT_DIR, "_pw_audio.wav")
W, H, FPS     = 1080, 1920, 30
GENRE         = "hardtechno"
BPM           = 140
KEY           = "E"
BEAT          = 60.0 / BPM
TOTAL_BEATS   = 70                      # 70 beats @140 = 30.000 s = one full cycle
DURATION      = TOTAL_BEATS * BEAT
N_FRAMES      = int(round(DURATION * FPS))
TARGET_LUFS   = -14.0
TARGET_TP     = -1.0
LIMIT_DB      = -3.0
UPLOAD_MB     = 9.0
UPLOAD_AUDIO_K= 160

# physics
G_ACC         = 9.81
N_PEND        = 20
K0            = 20                      # pendulum n makes K0+n swings per cycle
CYCLE_S       = 30.0
T0_BEATS      = 6                       # video starts 6 beats into the cycle (snake already moving)
T0_S          = T0_BEATS * BEAT
AMP_DEG       = 12                      # swing amplitude (small angle)
STEP_TOL      = 0.30                    # rad: "in step with #1"
GROUP_TOL     = 0.22                    # rad: phase-cluster tolerance for the groups readout

# look L5: neon synthwave, ONE background value everywhere
BG_COLOR      = "#0d0221"
GRID_COLOR    = "#ff2bd6"
SUN_TOP       = "#ffb347"
SUN_BOT       = "#ff2bd6"
TEXT_COLOR    = "#f4f1ff"
TITLE_BOX     = "#21f0ff"
HOOK_BOX      = "#ff2bd6"
STICKER_BOX   = "#ffe14d"
READ_COLOR    = "#21f0ff"
EQ_COLOR      = "#f4f1ff"
FONT_BOLD     = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_SEMI     = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_MONO     = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# layout (px)
CX            = W // 2
AMP_PX        = 340
ROW_Y0        = 450
ROW_DY        = 40
HORIZON_Y     = 1445
TITLE_Y       = 180
READ_Y        = 362
EQ_Y          = ROW_Y0 + ROW_DY * (N_PEND - 1) + 50
STICKER_Y     = 1490
SERIES_Y      = 1428
TRAIL_N       = 7
TRAIL_DT      = 1 / 150
GLOW_R        = 13
EQ_DPI        = 200
EQ_FONTSIZE   = 15
EQ_STROKE_W   = 4

TITLE_TXT     = "THEY'RE NOT RANDOM"
HOOK_TXT      = "watch them line up"
SERIES_TXT    = "hypnotic physics #1"
STICKER_TXT   = "which second clicked for you?"
PAYOFF_TXT    = "at 30 s. all 20 back in step."
EQ_LINE       = r"$T_n = 2\pi\sqrt{L_n/g} = \dfrac{30\ \mathrm{s}}{20+n}$"
LEGEND_TXT    = "longer string, slower swing"
EQ_FROM_S     = 2.0

SHEET_TIMES   = [("hook", 0.0), ("4 groups", 4.93), ("3 groups", 7.43),
                 ("2 lines", 12.43), ("braid", 20.6), ("re-sync", 27.43)]


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
# PHYSICS
# =====================================================================
NS        = np.arange(N_PEND)
SWINGS    = K0 + NS
PERIODS   = CYCLE_S / SWINGS
LENGTHS   = G_ACC * PERIODS ** 2 / (4 * math.pi ** 2)
OMEGAS    = 2 * math.pi / PERIODS


def phases(ts):
    return (OMEGAS * ts) % (2 * math.pi)


def in_step(ts):
    d = np.angle(np.exp(1j * (phases(ts) - phases(ts)[0])))
    return int(np.sum(np.abs(d) < STEP_TOL))


def groups(ts):
    p = np.sort(phases(ts))
    gaps = np.diff(np.concatenate([p, [p[0] + 2 * math.pi]]))
    return int(np.sum(gaps > GROUP_TOL))


def bob_x(ts):
    return CX + AMP_PX * np.cos(OMEGAS * ts)


def hue_rgb(n):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(0.83 - 0.83 * n / (N_PEND - 1), 0.85, 1.0)
    return np.array([r, g, b]) * 255


COLS = [hue_rgb(n) for n in NS]

# events computed from the physics (video time)
def group_events():
    out = []
    for q in range(1, 7):
        for p in range(0, q + 1):
            ts = CYCLE_S * p / q
            tv = ts - T0_S
            if tv < 0:
                tv += CYCLE_S
            if 0 <= tv < DURATION and groups(ts + 1e-9) == q:
                out.append((round(tv, 4), q))
    best = {}
    for tv, q in out:
        if tv not in best or q < best[tv]:
            best[tv] = q
    return sorted(best.items())


GROUP_EVENTS = group_events()
SYNC_T = [tv for tv, q in GROUP_EVENTS if q == 1][0]


def ping_events():
    ev = []
    for n in NS:
        k = 0
        while True:
            ts = k * PERIODS[n]
            k += 1
            tv = ts - T0_S
            if tv < 0:
                continue
            if tv >= DURATION:
                break
            ev.append((tv, int(n)))
    return sorted(ev)


# =====================================================================
# AUDIO
# =====================================================================
def build_audio():
    song = Song(GENRE, bpm=BPM, total_beats=TOTAL_BEATS, key=KEY, seed=28)
    sync_b = SYNC_T / BEAT
    first_big = [tv for tv, q in GROUP_EVENTS if q == 3][0] / BEAT
    song.sections = [("intro", 0, int(first_big), 0.7), ("main", int(first_big), int(sync_b) - 1, 0.9),
                     ("riser", int(sync_b) - 1, int(sync_b), 0.9), ("drop", int(sync_b), TOTAL_BEATS, 1.0)]
    song.silent_beats = [int(round(sync_b)) - 1]
    evs = []
    for tv, n in ping_events():
        evs.append(dict(beat=tv / BEAT, kind="hit", idx=7 + int(n * 0.75), vel=0.22, pan=-0.6 + 1.2 * n / (N_PEND - 1)))
    for tv, q in GROUP_EVENTS:
        if q <= 3:
            evs.append(dict(beat=tv / BEAT, kind="big"))
        elif q <= 5:
            evs.append(dict(beat=tv / BEAT, kind="chord"))
    evs.append(dict(beat=sync_b, kind="rise", len=4))
    for b0 in range(0, TOTAL_BEATS, 8):                 # hook motif, 4 notes every 2 bars
        for off, idx in ((0, 14), (0.5, 15), (1.5, 17), (3, 15)):
            if b0 + off < TOTAL_BEATS:
                evs.append(dict(beat=b0 + off, kind="hit", idx=idx, vel=0.75, pan=0.0))
    song.events = evs

    def bright(t):
        if t < SYNC_T:
            return 0.25 + 0.75 * (t / SYNC_T) ** 1.6
        return 1.0 - 0.7 * ((t - SYNC_T) / (DURATION - SYNC_T))
    song.brightness = bright
    st = song.render().astype(np.float64)
    st[:, :] -= st.mean(axis=0)
    beats = dict(bpm=BPM, sync_t=SYNC_T, silent_beats=song.silent_beats,
                 group_events=GROUP_EVENTS, n_pings=len(ping_events()))
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
    fig.text(0.5, 0.5, line, ha="center", va="center", fontsize=fs, color=EQ_COLOR,
             path_effects=[pe.withStroke(linewidth=EQ_STROKE_W, foreground=BG_COLOR)])
    b = io.BytesIO()
    fig.savefig(b, dpi=EQ_DPI, transparent=True, facecolor=BG_COLOR)
    plt.close(fig)
    b.seek(0)
    return Image.open(b).convert("RGBA").resize((W, 150), Image.LANCZOS)


def background():
    im = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(im, "RGBA")
    # sun behind the horizon, banded
    R = 210
    top, bot = hex_rgb(SUN_TOP), hex_rgb(SUN_BOT)
    for y in range(HORIZON_Y - R, HORIZON_Y):
        f = (y - (HORIZON_Y - R)) / R
        if int(f * 14) % 2 == 1 and f > 0.45:
            continue
        hw = math.sqrt(max(0, R * R - (HORIZON_Y - y) ** 2))
        c = top * (1 - f) + bot * f
        c = BG + (c - BG) * 0.24
        d.line([(CX - hw, y), (CX + hw, y)], fill=tuple(int(v) for v in c))
    # perspective grid floor
    gc = hex_rgb(GRID_COLOR)
    for i in range(-14, 15):
        x_far = CX + i * 26
        x_near = CX + i * 170
        d.line([(x_far, HORIZON_Y), (x_near, H)], fill=tuple(int(v) for v in BG + (gc - BG) * 0.45), width=2)
    for k in range(1, 14):
        y = HORIZON_Y + (H - HORIZON_Y) * (k / 13) ** 2.2
        a = 0.25 + 0.35 * k / 13
        d.line([(0, y), (W, y)], fill=tuple(int(v) for v in BG + (gc - BG) * a), width=2)
    d.line([(0, HORIZON_Y), (W, HORIZON_Y)], fill=tuple(int(v) for v in BG + (gc - BG) * 0.9), width=3)
    # faint rails + centre ticks
    for n in NS:
        y = ROW_Y0 + n * ROW_DY
        c = BG + (COLS[n] - BG) * 0.16
        d.line([(CX - AMP_PX, y), (CX + AMP_PX, y)], fill=tuple(int(v) for v in c), width=2)
        d.line([(CX, y - 6), (CX, y + 6)], fill=tuple(int(v) for v in BG + (COLS[n] - BG) * 0.4), width=2)
    # length labels
    f = font(FONT_SEMI, 23)
    d.text((CX - AMP_PX - 16, ROW_Y0), f"{LENGTHS[0] * 100:.0f} cm · {PERIODS[0]:.2f} s", font=f, anchor="rm", fill=(200, 190, 230))
    d.text((CX - AMP_PX - 16, ROW_Y0 + (N_PEND - 1) * ROW_DY), f"{LENGTHS[-1] * 100:.0f} cm · {PERIODS[-1]:.2f} s", font=f,
           anchor="rm", fill=(200, 190, 230))
    # watermark
    d.text((CX, 100), "@cosmoscoded", font=font(FONT_SEMI, 30), anchor="mm", fill=(190, 180, 225))
    return np.asarray(im).astype(np.float32)


def box_text(d, cx, cy, text, fnt, fill_box, fill_txt, pad=(28, 14), stroke=False):
    w = d.textlength(text, font=fnt)
    asc, desc = fnt.getmetrics()
    h = asc + desc
    d.rounded_rectangle([cx - w / 2 - pad[0], cy - h / 2 - pad[1], cx + w / 2 + pad[0], cy + h / 2 + pad[1]],
                        radius=14, fill=fill_box)
    d.text((cx, cy), text, font=fnt, anchor="mm", fill=fill_txt)


BG_ARR = None
EQ_IMG = None


# =====================================================================
# FRAME
# =====================================================================
def render_frame(i):
    global BG_ARR, EQ_IMG
    if BG_ARR is None:
        BG_ARR = background()
        EQ_IMG = mathtext_png(EQ_LINE, EQ_FONTSIZE)
    tv = i / FPS
    ts = T0_S + tv
    # glow layer: trails + bobs
    gl = Image.new("RGB", (W, H), (0, 0, 0))
    gd = ImageDraw.Draw(gl)
    pts = []
    for n in NS:
        y = ROW_Y0 + n * ROW_DY
        col = COLS[n]
        for k in range(TRAIL_N, 0, -1):
            x = CX + AMP_PX * math.cos(OMEGAS[n] * (ts - k * TRAIL_DT))
            a = 0.55 * (1 - k / (TRAIL_N + 1))
            r = 13 * (1 - 0.5 * k / TRAIL_N)
            gd.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v * a) for v in col))
        x = CX + AMP_PX * math.cos(OMEGAS[n] * ts)
        pts.append((x, y))
        gd.ellipse([x - 17, y - 17, x + 17, y + 17], fill=tuple(int(v) for v in col))
    gd.line(pts, fill=(120, 110, 150), width=3)
    glow = np.asarray(gl.filter(ImageFilter.GaussianBlur(GLOW_R))).astype(np.float32)
    sharp = np.asarray(gl).astype(np.float32)
    # click flash near a group event
    flash = 0.0
    for et, q in GROUP_EVENTS:
        dt = tv - et
        if 0 <= dt < 0.6 and q <= 3:
            flash = max(flash, (1 - dt / 0.6) ** 2 * (1.0 if q == 1 else 0.6))
    frame = BG_ARR + glow * (1.6 + 1.6 * flash) + sharp * 0.9
    # white cores
    im = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    for (x, y) in pts:
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255, 230))

    # title + hook
    box_text(d, CX, TITLE_Y, TITLE_TXT, font(FONT_BOLD, 66), TITLE_BOX, BG_COLOR, pad=(32, 16))
    box_text(d, CX, TITLE_Y + 92, HOOK_TXT, font(FONT_SEMI, 38), HOOK_BOX, "#ffffff", pad=(22, 10))

    # readouts (computed from the physics)
    stp, grp = in_step(ts), groups(ts)
    fm = font(FONT_MONO, 36)
    d.text((CX - 300, READ_Y), f"t {ts:5.2f} s", font=fm, anchor="mm", fill=READ_COLOR)
    d.text((CX, READ_Y), f"in step {stp:2d}/20", font=fm, anchor="mm",
           fill=STICKER_BOX if stp == N_PEND else READ_COLOR)
    gtxt = f"groups {grp}" if grp <= 6 else "groups --"
    if grp <= 4:
        box_text(d, CX + 300, READ_Y, gtxt, fm, STICKER_BOX, BG_COLOR, pad=(14, 8))
    else:
        d.text((CX + 300, READ_Y), gtxt, font=fm, anchor="mm", fill=READ_COLOR)

    # equation + legend under the visual
    if tv >= EQ_FROM_S - 0.4:
        a = min(1, (tv - (EQ_FROM_S - 0.4)) / 0.4)
        eq = np.asarray(EQ_IMG).astype(np.float32)
        base = np.asarray(im).astype(np.float32)
        y0 = EQ_Y - 30
        al = eq[..., 3:4] / 255 * a
        base[y0:y0 + 150] = base[y0:y0 + 150] * (1 - al) + eq[..., :3] * al
        im = Image.fromarray(base.astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        d.text((CX, EQ_Y + 108), LEGEND_TXT, font=font(FONT_SEMI, 36), anchor="mm",
               fill=(230, 225, 250, int(255 * a)), stroke_width=4, stroke_fill=BG_COLOR)

    # series + sticker / payoff
    d.text((CX, SERIES_Y), SERIES_TXT, font=font(FONT_SEMI, 34), anchor="mm", fill="#21f0ff",
           stroke_width=5, stroke_fill=BG_COLOR)
    txt = PAYOFF_TXT if tv >= SYNC_T else STICKER_TXT
    box_text(d, CX, STICKER_Y, txt, font(FONT_BOLD, 40), STICKER_BOX, BG_COLOR, pad=(26, 14))
    return np.asarray(im)


# =====================================================================
# OUTPUT
# =====================================================================
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
    subprocess.run(base[:5] + ["-i", TMP_AUDIO, "-map", "0:v", "-map", "1:a"] + base[5:] + ["-pass", "2", "-c:a", "aac", "-b:a", f"{UPLOAD_AUDIO_K}k",
                           "-movflags", "+faststart", OUT_UPLOAD], check=True)
    for f in os.listdir(OUT_DIR):
        if f.startswith("_x264pass"):
            os.remove(os.path.join(OUT_DIR, f))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("lengths cm:", np.round(LENGTHS * 100, 1).tolist())
    print("group events (video s, groups):", GROUP_EVENTS, "sync at", SYNC_T)
    if mode in ("sheet", "all"):
        contact_sheet()
    if mode in ("all",):
        master(build_audio())
        render_video()
        encode_upload()
        os.remove(TMP_AUDIO)
        print("done")
