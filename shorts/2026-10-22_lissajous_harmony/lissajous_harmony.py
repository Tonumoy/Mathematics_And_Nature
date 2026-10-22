"""
Coded Cosmos — "This circle is a sound."  (Lissajous / harmony Short)
-------------------------------------------------------------------
Two notes draw one shape:  x = sin(a*theta),  y = sin(b*theta + phi).
Simple frequency ratios (chords that sound sweet) close into clean shapes.
The equal-tempered tritone (ratio sqrt(2), irrational) never closes.

Everything — picture AND music — is generated from the same equations,
so the audio is 100% original (no copyright claims possible).

Layout follows the Coded Cosmos template: 9:16, simulation centred at ~47%
of the canvas height, equations at top (matplotlib mathtext + pe.withStroke),
story text at bottom, ONE uniform BG_COLOR everywhere, text beats driven by
pre-computed schedule events with a minimum on-screen time, seamless loop.

Run:  python lissajous_harmony.py      ->  lissajous_harmony.mp4
"""

import os
import io
import math
import wave
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# =====================================================================
# GLOBAL PARAMETERS
# =====================================================================
OUT_DIR          = os.path.dirname(os.path.abspath(__file__))
OUT_VIDEO        = os.path.join(OUT_DIR, "lissajous_harmony.mp4")
TMP_AUDIO        = os.path.join(OUT_DIR, "_lissajous_audio.wav")

W, H             = 1080, 1920
FPS              = 30
BG_COLOR         = "#05060a"
BG_RGB           = (5, 6, 10)

# --- music grid ---
BPM              = 84
BEATS_PER_BAR    = 4
SR               = 44100
BASE_FREQ        = 220.0          # lower note (A3) stays fixed — upper note = BASE_FREQ * b/a
DETUNE_HZ        = 0.12           # upper note detune -> phase drift phi(t) (shape slowly "rotates"); off in unison sections
PHI0             = math.pi / 2    # starting phase: 1:1 at phi=pi/2 is a perfect circle

# --- sections: (key, a, b, bars, color, chip label, drums on?) ---
SQRT2 = math.sqrt(2.0)
SECTIONS = [
    ("unison",  1, 1,     2, "#4fd1ff", "UNISON  1 : 1",          True),
    ("octave",  1, 2,     2, "#7cf5c4", "OCTAVE  1 : 2",          True),
    ("fifth",   2, 3,     2, "#ffd166", "PERFECT FIFTH  2 : 3",   True),
    ("fourth",  3, 4,     2, "#ff9f5a", "PERFECT FOURTH  3 : 4",  True),
    ("third",   4, 5,     2, "#c79bff", "MAJOR THIRD  4 : 5",     True),
    ("tritone", 1, SQRT2, 3, "#ff3b5c", "TRITONE  1 : √2",   True),
    ("silence", 1, 1,     1, "#4fd1ff", "",                       False),
    ("resolve", 1, 1,     2, "#4fd1ff", "UNISON  1 : 1",          True),
]

# --- layout (pixels) ---
SIM_CY_FRAC      = 0.47
SIM_CY           = int(H * SIM_CY_FRAC)
SIM_SIZE         = 800                     # square simulation box
SIM_R            = 330                     # curve amplitude in px
SIM_X0           = (W - SIM_SIZE) // 2
SIM_Y0           = SIM_CY - SIM_SIZE // 2
EQ_Y_TOP         = 150                     # equations block top
CHIP_Y           = SIM_Y0 + SIM_SIZE + 25  # interval label under sim
STORY_Y          = CHIP_Y + 95             # story text centre-top
STORY_MAX_W      = 960
SIGN_Y           = 70                      # sign-off at very top, clear of Shorts UI

# --- curve drawing ---
CURVE_POINTS     = 2400
CURVE_WIDTH      = 4
GLOW_DOWNSCALE   = 4
GLOW_RADIUS      = 5
GLOW_GAIN        = 1.6
KICK_FLASH       = 0.55
KICK_FLASH_DECAY = 7.0
DRAW_ON_BEATS    = 1.0                     # how long a new shape takes to trace itself
DOT_RADIUS       = 9
DOT_LOOPS_PER_BEAT = 0.5
TRITONE_TURNS_START = 6                    # tritone keeps winding: turns of theta shown
TRITONE_TURNS_END   = 40

# --- text ---
FONT_BOLD        = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG         = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
STORY_FONT_SIZE  = 56
CHIP_FONT_SIZE   = 38
SIGN_FONT_SIZE   = 34
TEXT_STROKE      = 5
TEXT_FADE_S      = 0.25
MIN_BEAT_S       = 3.0
EQ_FONTSIZE      = 12                      # matplotlib points (house style 11-13)
EQ_DPI           = 330
EQ_COLOR         = "#ffffffdd"
EQ_STROKE_W      = 3
EQ_BLOCK_H       = 300                     # px height of equation block

# --- audio mix ---
TONE_GAIN        = 0.20
PAD_GAIN         = 0.05
SUB_GAIN         = 0.16
KICK_GAIN        = 0.55
HAT_GAIN         = 0.045
CLAP_GAIN        = 0.10
RISER_GAIN       = 0.12
CROSSFEED        = 0.15
PUMP_DEPTH       = 0.35
PUMP_DECAY_S     = 0.12
SECTION_XFADE_S  = 0.02
TARGET_LUFS      = -14

# Story beats: (trigger event, offset in beats, text). Empty text = clear screen.
STORY_BEATS = [
    ("unison",  0.0, "This circle is a sound."),
    ("unison",  4.5, "One note moves the dot sideways. One moves it up and down."),
    ("octave",  0.0, "Note 2 now runs twice as fast."),
    ("fifth",   0.0, "2 : 3 — the sweetest chord in music."),
    ("fourth",  0.0, "Simple ratio = clean shape = sounds good"),
    ("third",   0.0, "Every happy chord has a shape."),
    ("tritone", 0.0, "Now the “Devil’s interval”. Ratio: √2"),
    ("tritone", 5.0, "√2 never repeats. So the shape never closes."),
    ("silence", 0.0, ""),
    ("resolve", 0.0, "Your ears were seeing math all along."),
    ("resolve", 4.0, "Lissajous curves · Nature is Running on Code"),
]
SIGN_OFF         = "@cosmoscoded"

# =====================================================================
# DERIVED SCHEDULE
# =====================================================================
BEAT_S  = 60.0 / BPM
BAR_S   = BEAT_S * BEATS_PER_BAR
starts, t = {}, 0.0
SCHED = []
for key, a, b, bars, col, chip, drums in SECTIONS:
    SCHED.append(dict(key=key, a=a, b=b, t0=t, t1=t + bars * BAR_S,
                      color=col, chip=chip, drums=drums))
    starts[key] = t
    t += bars * BAR_S
DURATION = t
N_FRAMES = int(round(DURATION * FPS))
N_SAMP   = int(round(DURATION * SR))


def section_at(time_s):
    for s in SCHED:
        if s["t0"] <= time_s < s["t1"]:
            return s
    return SCHED[-1]


# text beat schedule with sequential guard + minimum time
TEXT_EVENTS = []
prev = -1e9
for key, off_beats, txt in STORY_BEATS:
    trig = starts[key] + off_beats * BEAT_S
    start = max(trig, prev + MIN_BEAT_S) if TEXT_EVENTS else trig
    TEXT_EVENTS.append([start, txt])
    prev = start
for i in range(len(TEXT_EVENTS)):
    TEXT_EVENTS[i].append(TEXT_EVENTS[i + 1][0] if i + 1 < len(TEXT_EVENTS) else DURATION + 1)


# =====================================================================
# AUDIO
# =====================================================================
def synth_audio():
    tt = np.arange(N_SAMP) / SR
    fL = np.zeros(N_SAMP)
    fR = np.zeros(N_SAMP)
    tone_env = np.ones(N_SAMP)
    drums_on = np.zeros(N_SAMP)
    for s in SCHED:
        i0, i1 = int(s["t0"] * SR), int(s["t1"] * SR)
        F = BASE_FREQ / s["a"]
        fL[i0:i1] = F * s["a"]
        fR[i0:i1] = F * s["b"] + (DETUNE_HZ if s["a"] != s["b"] else 0.0)
        drums_on[i0:i1] = 1.0 if s["drums"] else 0.0
        if s["key"] == "silence":
            # beat 1 = total silence (the pause at the peak), then tones swell back in
            seg = tt[i0:i1] - s["t0"]
            env = np.clip((seg - BEAT_S) / (BAR_S - BEAT_S), 0, 1) ** 2
            tone_env[i0:i1] = env

    # continuous phase -> no clicks when frequency changes
    pL = 2 * np.pi * np.cumsum(fL) / SR
    pR = 2 * np.pi * np.cumsum(fR) / SR

    # short crossfade dip at each section boundary
    for s in SCHED[1:]:
        i = int(s["t0"] * SR)
        n = int(SECTION_XFADE_S * SR)
        lo, hi = max(i - n, 0), min(i + n, N_SAMP)
        x = np.linspace(-1, 1, hi - lo)
        tone_env[lo:hi] *= 0.35 + 0.65 * np.abs(x)

    # sidechain pump from kicks
    beat_pos = (tt % BEAT_S)
    pump = 1.0 - PUMP_DEPTH * np.exp(-beat_pos / PUMP_DECAY_S) * drums_on

    toneL = np.sin(pL)
    toneR = np.sin(pR)
    # warm pad: same two notes, one octave down, soft triangle
    tri = lambda p: (2 / np.pi) * np.arcsin(np.sin(p))
    pad = 0.5 * (tri(pL / 2) + tri(pR / 2))
    sub = np.sin(pL / 2)

    # kick on every beat
    kick = np.zeros(N_SAMP)
    klen = int(0.35 * SR)
    kt = np.arange(klen) / SR
    kfreq = 48 + 110 * np.exp(-kt / 0.035)
    kwave = np.sin(2 * np.pi * np.cumsum(kfreq) / SR) * np.exp(-kt / 0.16)
    kwave[:int(0.003 * SR)] += np.linspace(0.6, 0, int(0.003 * SR))
    # hat on off-beats, clap on 2 & 4
    rng = np.random.default_rng(7)
    hlen = int(0.06 * SR)
    hat = np.diff(rng.standard_normal(hlen + 1)) * np.exp(-np.arange(hlen) / SR / 0.012)
    clen = int(0.2 * SR)
    cn = rng.standard_normal(clen)
    cn = cn - np.convolve(cn, np.ones(12) / 12, mode="same")
    clap = cn * np.exp(-np.arange(clen) / SR / 0.05)
    hats = np.zeros(N_SAMP)
    claps = np.zeros(N_SAMP)
    n_beats = int(round(DURATION / BEAT_S))
    for k in range(n_beats):
        tb = k * BEAT_S
        sec = section_at(tb + 1e-6)
        if not sec["drums"]:
            continue
        i = int(tb * SR)
        e = min(i + klen, N_SAMP)
        kick[i:e] += kwave[:e - i]
        ih = int((tb + BEAT_S / 2) * SR)
        eh = min(ih + hlen, N_SAMP)
        if ih < N_SAMP:
            hats[ih:eh] += hat[:eh - ih]
        if k % BEATS_PER_BAR in (1, 3):
            ec = min(i + clen, N_SAMP)
            claps[i:ec] += clap[:ec - i]

    # riser over the last bar of the tritone (tension before the silence)
    riser = np.zeros(N_SAMP)
    tri_sec = [s for s in SCHED if s["key"] == "tritone"][0]
    r0, r1 = int((tri_sec["t1"] - BAR_S) * SR), int(tri_sec["t1"] * SR)
    rn = rng.standard_normal(r1 - r0)
    rn = rn - np.convolve(rn, np.ones(30) / 30, mode="same")
    riser[r0:r1] = rn * np.linspace(0, 1, r1 - r0) ** 2.5

    toneL_m = TONE_GAIN * toneL * tone_env * pump
    toneR_m = TONE_GAIN * toneR * tone_env * pump
    common = (PAD_GAIN * pad * tone_env * pump
              + SUB_GAIN * sub * tone_env * pump * drums_on
              + KICK_GAIN * kick + HAT_GAIN * hats + CLAP_GAIN * claps
              + RISER_GAIN * riser)
    L = toneL_m + CROSSFEED * toneR_m + common
    R = toneR_m + CROSSFEED * toneL_m + common
    st = np.stack([L, R], axis=1)
    st /= np.max(np.abs(st)) / 0.89
    pcm = (st * 32767).astype(np.int16)
    with wave.open(TMP_AUDIO, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# =====================================================================
# STATIC TEXT LAYERS (pre-rendered, cached)
# =====================================================================
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mathtext_png(lines):
    """Render equation lines with matplotlib mathtext -> RGBA PIL image."""
    fig = plt.figure(figsize=(W / EQ_DPI, EQ_BLOCK_H / EQ_DPI), dpi=EQ_DPI)
    fig.patch.set_alpha(0)
    n = len(lines)
    for i, ln in enumerate(lines):
        y = 1 - (i + 0.5) / n
        fig.text(0.5, y, ln, ha="center", va="center", fontsize=EQ_FONTSIZE,
                 color=EQ_COLOR,
                 path_effects=[pe.withStroke(linewidth=EQ_STROKE_W, foreground=BG_COLOR)])
    buf = io.BytesIO()
    fig.savefig(buf, dpi=EQ_DPI, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").resize((W, EQ_BLOCK_H), Image.LANCZOS)


def eq_lines_for(sec):
    if sec["key"] == "tritone":
        ratio = r"$a : b = 1 : \sqrt{2}$     $220\,\mathrm{Hz} : 311\,\mathrm{Hz}$"
    else:
        a, b = sec["a"], sec["b"]
        f2 = BASE_FREQ * b / a
        ratio = rf"$a : b = {a} : {b}$     $220\,\mathrm{{Hz}} : {f2:.0f}\,\mathrm{{Hz}}$"
    return [r"$x(\theta) = \sin(a\,\theta)$",
            r"$y(\theta) = \sin(b\,\theta + \varphi)$",
            ratio]


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_layer(text, size, color, font_path, max_w):
    font = ImageFont.truetype(font_path, size)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = wrap(tmp, text, font, max_w)
    lh = int(size * 1.22)
    img = Image.new("RGBA", (W, lh * len(lines) + 2 * TEXT_STROKE + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((W // 2, TEXT_STROKE + i * lh), ln, font=font, fill=color, anchor="ma",
               stroke_width=TEXT_STROKE, stroke_fill=BG_RGB)
    return img


EQ_CACHE, CHIP_CACHE, STORY_CACHE = {}, {}, {}


def build_caches():
    for s in SCHED:
        EQ_CACHE[s["key"]] = mathtext_png(eq_lines_for(s))
        if s["chip"]:
            CHIP_CACHE[s["key"]] = text_layer(s["chip"], CHIP_FONT_SIZE, hex_rgb(s["color"]),
                                              FONT_BOLD, STORY_MAX_W)
    for st, txt, _ in TEXT_EVENTS:
        if txt and txt not in STORY_CACHE:
            STORY_CACHE[txt] = text_layer(txt, STORY_FONT_SIZE, (255, 255, 255),
                                          FONT_BOLD, STORY_MAX_W)
    global SIGN_LAYER
    SIGN_LAYER = text_layer(SIGN_OFF, SIGN_FONT_SIZE, (255, 255, 255), FONT_REG, STORY_MAX_W)


# =====================================================================
# FRAME RENDERING
# =====================================================================
def curve_points(sec, time_s, reveal):
    a, b = sec["a"], sec["b"]
    phi = PHI0 + (2 * np.pi * DETUNE_HZ * (time_s - sec["t0"]) if a != b else 0.0)
    if sec["key"] == "tritone":
        f = (time_s - sec["t0"]) / (sec["t1"] - sec["t0"])
        turns = TRITONE_TURNS_START + (TRITONE_TURNS_END - TRITONE_TURNS_START) * f ** 1.3
        n = int(CURVE_POINTS * turns / 3)
    else:
        turns = 1.0  # theta in [0, 2pi] closes every integer ratio
        n = CURVE_POINTS
    th = np.linspace(0, 2 * np.pi * turns * reveal, max(int(n * reveal), 2))
    x = np.sin(a * th)
    y = np.sin(b * th + phi)
    return x, y, th, phi


def alpha_paste(base, layer, top_y, alpha):
    if alpha <= 0:
        return
    h = layer.shape[0]
    y0, y1 = max(top_y, 0), min(top_y + h, H)
    if y1 <= y0:
        return
    lay = layer[y0 - top_y:y1 - top_y]
    a = (lay[..., 3:4] / 255.0) * alpha
    base[y0:y1] = base[y0:y1] * (1 - a) + lay[..., :3] * a


def render_frame(fi, np_layers):
    time_s = fi / FPS
    sec = section_at(time_s)
    col = np.array(hex_rgb(sec["color"]), dtype=np.float32)
    frame = np.empty((H, W, 3), dtype=np.float32)
    frame[:] = BG_RGB

    since = time_s - sec["t0"]
    # reveal / draw-on
    if sec["key"] == "unison" and sec["t0"] == 0.0:
        reveal = 1.0                                   # hook: full shape on frame 1
    elif sec["key"] == "silence":
        reveal = float(np.clip((since - BEAT_S) / (BAR_S - BEAT_S), 0, 1))
    elif sec["key"] == "tritone":
        reveal = 1.0
    else:
        reveal = float(np.clip(since / (DRAW_ON_BEATS * BEAT_S), 0, 1))
        reveal = 1 - (1 - reveal) ** 3

    # kick flash
    beat_pos = time_s % BEAT_S
    flash = KICK_FLASH * math.exp(-beat_pos * KICK_FLASH_DECAY) if sec["drums"] else 0.0
    intensity = 1.0 + flash
    if sec["key"] == "silence":
        intensity = reveal

    if reveal > 0.002:
        x, y, th, phi = curve_points(sec, time_s, reveal)
        px = SIM_SIZE / 2 + SIM_R * x
        py = SIM_SIZE / 2 - SIM_R * y
        layer = Image.new("RGB", (SIM_SIZE, SIM_SIZE), (0, 0, 0))
        d = ImageDraw.Draw(layer)
        width = CURVE_WIDTH if sec["key"] != "tritone" else 2
        d.line(list(zip(px.tolist(), py.tolist())), fill=tuple(int(c) for c in col), width=width,
               joint="curve")
        # tracer dot
        if sec["key"] != "tritone":
            dth = 2 * np.pi * ((time_s / BEAT_S) * DOT_LOOPS_PER_BEAT % 1.0) * reveal
            dx = SIM_SIZE / 2 + SIM_R * math.sin(sec["a"] * dth)
            dy = SIM_SIZE / 2 - SIM_R * math.sin(sec["b"] * dth + phi)
        else:
            dx, dy = px[-1], py[-1]
        d.ellipse([dx - DOT_RADIUS, dy - DOT_RADIUS, dx + DOT_RADIUS, dy + DOT_RADIUS],
                  fill=(255, 255, 255))
        small = layer.resize((SIM_SIZE // GLOW_DOWNSCALE,) * 2, Image.BILINEAR)
        glow = small.filter(ImageFilter.GaussianBlur(GLOW_RADIUS)).resize((SIM_SIZE,) * 2,
                                                                          Image.BILINEAR)
        sim = (np.asarray(layer, dtype=np.float32)
               + GLOW_GAIN * np.asarray(glow, dtype=np.float32)) * intensity
        frame[SIM_Y0:SIM_Y0 + SIM_SIZE, SIM_X0:SIM_X0 + SIM_SIZE] += sim

    # equations (top)
    eq = np_layers["eq"][sec["key"]]
    alpha_paste(frame, eq, EQ_Y_TOP, 1.0 if sec["key"] != "silence" else 0.35)

    # interval chip
    if sec["key"] in np_layers["chip"]:
        a_chip = 1.0 if sec["t0"] == 0.0 else float(np.clip(since / 0.3, 0, 1))
        alpha_paste(frame, np_layers["chip"][sec["key"]], CHIP_Y, a_chip)

    # story text
    for st, txt, en in TEXT_EVENTS:
        if txt and st <= time_s < en:
            a_in = 1.0 if st == 0 else np.clip((time_s - st) / TEXT_FADE_S, 0, 1)
            a_out = np.clip((en - time_s) / TEXT_FADE_S, 0, 1) if en <= DURATION else 1.0
            alpha_paste(frame, np_layers["story"][txt], STORY_Y, float(min(a_in, a_out)))

    # sign-off (always, subtle)
    alpha_paste(frame, np_layers["sign"], SIGN_Y, 0.55)

    return np.clip(frame, 0, 255).astype(np.uint8)


def main():
    print(f"Duration {DURATION:.2f}s, {N_FRAMES} frames, BPM {BPM}")
    synth_audio()
    build_caches()
    np_layers = dict(
        eq={k: np.asarray(v, dtype=np.float32) for k, v in EQ_CACHE.items()},
        chip={k: np.asarray(v, dtype=np.float32) for k, v in CHIP_CACHE.items()},
        story={k: np.asarray(v, dtype=np.float32) for k, v in STORY_CACHE.items()},
        sign=np.asarray(SIGN_LAYER, dtype=np.float32),
    )
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", TMP_AUDIO,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k",
           "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.0:LRA=11",
           "-shortest", "-movflags", "+faststart", OUT_VIDEO]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(N_FRAMES):
        proc.stdin.write(render_frame(fi, np_layers).tobytes())
        if fi % 150 == 0:
            print(f"frame {fi}/{N_FRAMES}")
    proc.stdin.close()
    proc.wait()
    os.remove(TMP_AUDIO)
    print("Saved", OUT_VIDEO)


if __name__ == "__main__":
    main()
