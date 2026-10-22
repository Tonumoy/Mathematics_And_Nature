"""
Coded Cosmos — thumbnail generator for "This circle is a sound." (Lissajous Short)
Produces (in ./assets):
  cover_9x16.png  (1080x1920, Shorts cover)
  cover_16x9.png  (1280x720, wide cover)
All drawing is from the same Lissajous equations used in the video.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ===================== GLOBAL PARAMETERS =====================
OUT_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
OUT_916          = os.path.join(OUT_DIR, "cover_9x16.png")
OUT_169          = os.path.join(OUT_DIR, "cover_16x9.png")
BG_RGB           = (5, 6, 10)
FONT_BLACK       = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_REG         = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SWEET_COLOR      = (255, 209, 102)   # perfect fifth (gold)
TENSE_COLOR      = (255, 59, 92)     # tritone (red)
ACCENT_COLOR     = (79, 209, 255)    # cyan
WHITE            = (255, 255, 255)
SWEET_RATIO      = (2, 3)
TENSE_TURNS      = 9
CURVE_W          = 7
TENSE_W          = 2
GLOW_BLUR        = 22
GLOW_GAIN        = 2.2
SUPERSAMPLE      = 2
VIGNETTE         = 0.55
STROKE           = 8


def lissajous(a, b, phi, turns, n):
    th = np.linspace(0, 2 * np.pi * turns, n)
    return np.sin(a * th), np.sin(b * th + phi)


def glow_layer(size, draw_fn):
    W, H = size
    s = SUPERSAMPLE
    base = Image.new("RGB", (W * s, H * s), (0, 0, 0))
    draw_fn(ImageDraw.Draw(base), s)
    base = base.resize((W, H), Image.LANCZOS)
    glow = base.filter(ImageFilter.GaussianBlur(GLOW_BLUR))
    wide = base.filter(ImageFilter.GaussianBlur(GLOW_BLUR * 3))
    arr = (np.asarray(base, np.float32) + GLOW_GAIN * np.asarray(glow, np.float32)
           + 1.2 * np.asarray(wide, np.float32))
    return arr


def vignette(W, H):
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H * 0.47) / (H / 2)) ** 2)
    return np.clip(1 - VIGNETTE * r ** 2, 0, 1)[..., None]


def curves_fn(cx, cy, R):
    def fn(d, s):
        # tritone (behind)
        x, y = lissajous(1, math.sqrt(2), math.pi / 2, TENSE_TURNS, 9000)
        pts = list(zip(((cx + R * 1.08 * x) * s).tolist(), ((cy - R * 1.08 * y) * s).tolist()))
        d.line(pts, fill=tuple(int(c * 0.35) for c in TENSE_COLOR), width=TENSE_W * s)
        # perfect fifth (front)
        x, y = lissajous(*SWEET_RATIO, math.pi / 2, 1, 4000)
        pts = list(zip(((cx + R * x) * s).tolist(), ((cy - R * y) * s).tolist()))
        d.line(pts, fill=SWEET_COLOR, width=CURVE_W * s, joint="curve")
        # tracer dot
        d.ellipse([(cx + R * x[700] - 14) * s, (cy - R * y[700] - 14) * s,
                   (cx + R * x[700] + 14) * s, (cy - R * y[700] + 14) * s], fill=WHITE)
    return fn


def text(d, xy, s, size, color, font=FONT_BLACK, anchor="mm"):
    f = ImageFont.truetype(font, size)
    d.text(xy, s, font=f, fill=color, anchor=anchor, stroke_width=STROKE, stroke_fill=BG_RGB)


def make(W, H, cx, cy, R, layout):
    arr = np.empty((H, W, 3), np.float32)
    arr[:] = BG_RGB
    arr += glow_layer((W, H), curves_fn(cx, cy, R))
    arr = arr * vignette(W, H) + np.array(BG_RGB) * (1 - vignette(W, H))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    layout(ImageDraw.Draw(img), W, H)
    return img


def layout_916(d, W, H):
    text(d, (W // 2, 250), "THIS SHAPE", 150, WHITE)
    text(d, (W // 2, 410), "IS A SOUND", 150, SWEET_COLOR)
    text(d, (W // 2, 1560), "2 : 3", 110, SWEET_COLOR)
    text(d, (W // 2, 1680), "the math behind every chord", 52, ACCENT_COLOR, FONT_REG)
    text(d, (W // 2, 1840), "@cosmoscoded", 40, (200, 200, 210), FONT_REG)


def layout_169(d, W, H):
    text(d, (int(W * 0.27), 255), "THIS SHAPE", 92, WHITE)
    text(d, (int(W * 0.27), 365), "IS A SOUND", 92, SWEET_COLOR)
    text(d, (int(W * 0.27), 470), "the math behind every chord", 34, ACCENT_COLOR, FONT_REG)
    text(d, (int(W * 0.27), 660), "@cosmoscoded", 28, (200, 200, 210), FONT_REG)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make(1080, 1920, 540, 980, 400, layout_916).save(OUT_916)
    make(1280, 720, 930, 360, 280, layout_169).save(OUT_169)
    print("saved")
