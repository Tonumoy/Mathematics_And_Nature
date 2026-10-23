"""
Coded Cosmos - cover generator for "Poets found this before Fibonacci" (2026-10-23).
Produces (into OUT_DIR):
  cover_9x16.jpg  (1080x1920, Shorts cover)
  cover_16x9.png  (1280x720, Community post / shelf crop)

The idea, drawn exactly as in the video:
  short sound = 1 beat  = small cyan block
  long sound  = 2 beats = double-width gold block
  ways to fill n beats: 1, 2, 3, 5, ...
  4 beats = (the 3 ways for 3 beats + a short)  +  (the 2 ways for 2 beats + a long) = 5
Every rhythm is generated, and the counts are asserted.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ===================== GLOBAL PARAMETERS =====================
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_916      = OUT_DIR + "/cover_9x16.jpg"
OUT_169      = OUT_DIR + "/cover_16x9.png"
BG_RGB       = (5, 6, 10)                       # #05060a
FONT_BLACK   = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG     = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WHITE        = (255, 255, 255)
GREY         = (200, 200, 210)
CYAN         = (79, 209, 255)                   # short = 1 beat
GOLD         = (255, 209, 102)                  # long  = 2 beats
VIOLET       = (184, 146, 255)                  # "+ short" group (matches video)
ORANGE       = (255, 150, 60)                   # "+ long" group (matches video)
HEADLINE     = ("BEFORE", "FIBONACCI")     # white, gold
MAX_BEATS    = 4                                # columns 1..4
SUPERSAMPLE  = 2
GLOW_BLUR    = 14
GLOW_GAIN    = 1.0
VIGNETTE     = 0.5
STROKE       = 10


# ===================== MATHS =====================
def ways(n):
    """All rhythms (tuples of 1=short, 2=long) filling n beats, built by the poets' rule:
    (ways for n-1) + a short, then (ways for n-2) + a long."""
    if n == 0:
        return [()]
    if n < 0:
        return []
    return [w + (1,) for w in ways(n - 1)] + [w + (2,) for w in ways(n - 2)]


COUNTS = [len(ways(n)) for n in range(1, 9)]
assert COUNTS == [1, 2, 3, 5, 8, 13, 21, 34]          # 8 beats -> 34 (pinned-comment answer)
assert len(ways(4)) == len(ways(3)) + len(ways(2)) == 5
assert all(sum(w) == n for n in range(1, 9) for w in ways(n))


# ===================== DRAWING =====================
def font(path, size):
    return ImageFont.truetype(path, size)


def text(d, xy, s, size, color, path=FONT_BLACK, anchor="mm", stroke=STROKE):
    d.text(xy, s, font=font(path, size), fill=color, anchor=anchor,
           stroke_width=stroke, stroke_fill=BG_RGB)


def vignette(W, H):
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    return np.clip(1 - VIGNETTE * r ** 2, 0, 1)[..., None]


def layout_columns(g):
    """Return list of (n, x_left, rows[(y, rhythm, group)]) and group boxes for column 4."""
    u, rh, rg, cg, x0, y0, split_gap = g["u"], g["rh"], g["rg"], g["cg"], g["x0"], g["y0"], g["split"]
    cols, x = [], x0
    for n in range(1, MAX_BEATS + 1):
        rows, y = [], y0
        ws = ways(n)
        n_short = len(ways(n - 1))
        for i, w in enumerate(ws):
            grp = None
            if n == MAX_BEATS:
                grp = "short" if i < n_short else "long"
                if i == n_short:
                    y += split_gap
            rows.append((y, w, grp))
            y += rh + rg
        cols.append((n, x, rows))
        x += n * u + cg
    return cols


def draw_rhythm(d, s, x, y, w, u, rh, dim=1.0):
    pad = u * 0.08
    for v in w:
        col = CYAN if v == 1 else GOLD
        col = tuple(int(c * dim) for c in col)
        d.rounded_rectangle([(x + pad) * s, y * s, (x + v * u - pad) * s, (y + rh) * s],
                            radius=rh * 0.28 * s, fill=col)
        x += v * u


def geometry_fn(g):
    u, rh = g["u"], g["rh"]

    def fn(d, s):
        for n, x, rows in layout_columns(g):
            if n == MAX_BEATS:   # coloured split boxes: +short group (cyan), +long group (gold)
                for grp, col in (("short", VIOLET), ("long", ORANGE)):
                    ys = [r[0] for r in rows if r[2] == grp]
                    m = u * 0.22
                    d.rounded_rectangle([(x - m) * s, (min(ys) - m) * s,
                                         (x + n * u + m) * s, (max(ys) + rh + m) * s],
                                        radius=18 * s, outline=col, width=int(4 * s))
            for y, w, grp in rows:
                draw_rhythm(d, s, x, y, w, u, rh, dim=0.82)
    return fn


def render(W, H, g, layout):
    s = SUPERSAMPLE
    base = Image.new("RGB", (W * s, H * s), (0, 0, 0))
    geometry_fn(g)(ImageDraw.Draw(base), s)
    base = base.resize((W, H), Image.LANCZOS)
    glow = base.filter(ImageFilter.GaussianBlur(GLOW_BLUR))
    wide = base.filter(ImageFilter.GaussianBlur(GLOW_BLUR * 3))
    arr = np.empty((H, W, 3), np.float32)
    arr[:] = BG_RGB
    arr += (np.asarray(base, np.float32) + GLOW_GAIN * np.asarray(glow, np.float32)
            + 0.6 * np.asarray(wide, np.float32))
    v = vignette(W, H)
    arr = arr * v + np.array(BG_RGB, np.float32) * (1 - v)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    labels(d, g)
    layout(d, W, H, g)
    return img


def labels(d, g):
    u, hs, cs = g["u"], g["hdr"], g["cnt"]
    for n, x, rows in layout_columns(g):
        cx = x + n * u / 2
        text(d, (cx, g["y0"] - g["hdr_dy"]), f"{n} beat" + ("s" if n > 1 else ""), hs, GREY,
             FONT_BOLD, stroke=4)
        text(d, (cx, g["cnt_y"]), str(len(ways(n))), cs, GOLD if n == MAX_BEATS else WHITE, stroke=8)
    # split labels beside column 4
    n, x, rows = layout_columns(g)[-1]
    for grp, col, lab in (("short", VIOLET, f"{len(ways(n - 1))}"), ("long", ORANGE, f"{len(ways(n - 2))}")):
        ys = [r[0] for r in rows if r[2] == grp]
        yc = (min(ys) + max(ys) + g["rh"]) / 2
        text(d, (x + n * u + g["side_dx"], yc), lab, g["side"], col, anchor="lm", stroke=6)


def layout_916(d, W, H, g):
    text(d, (W // 2, 70), "@cosmoscoded", 38, GREY, FONT_REG, stroke=4)
    text(d, (W // 2, 250), HEADLINE[0], 150, WHITE, stroke=11)
    text(d, (W // 2, 400), HEADLINE[1], 158, GOLD, stroke=11)
    text(d, (W // 2, 530), "short = 1 beat     long = 2 beats", 44, GREY, FONT_BOLD, stroke=5)
    # legend chips
    draw_legend(d, W // 2 - 395, 530, CYAN, 1)
    draw_legend(d, W // 2 + 40, 530, GOLD, 2)
    # 3 + 2 = 5
    y = g["cnt_y"] + 150
    f = font(FONT_BLACK, 96)
    parts = [("3", VIOLET), (" + ", WHITE), ("2", ORANGE), (" = ", WHITE), ("5", GOLD)]
    tw = sum(d.textlength(p, font=f) for p, _ in parts)
    x = W / 2 - tw / 2
    for p, c in parts:
        d.text((x, y), p, font=f, fill=c, anchor="lm", stroke_width=8, stroke_fill=BG_RGB)
        x += d.textlength(p, font=f)
    text(d, (W // 2, 1700), "Indian poets ~700 CE  ·  Fibonacci 1202", 42, WHITE, FONT_BOLD, stroke=5)


def draw_legend(d, x, y, col, v):
    pass   # legend is carried by the coloured words below; kept as a hook for tweaks


def layout_169(d, W, H, g):
    lx = int(W * 0.255)
    text(d, (lx, 230), HEADLINE[0], 96, WHITE, stroke=8)
    text(d, (lx, 330), HEADLINE[1], 96, GOLD, stroke=8)
    text(d, (lx, 430), "short = 1 beat  ·  long = 2 beats", 28, GREY, FONT_BOLD, stroke=3)
    text(d, (lx, 480), "Indian poets ~700 CE · Fibonacci 1202", 28, WHITE, FONT_BOLD, stroke=3)
    text(d, (lx, 670), "@cosmoscoded", 26, GREY, FONT_REG, stroke=3)


G916 = dict(u=70, rh=46, rg=18, cg=58, x0=50, y0=720, split=34, hdr=34, hdr_dy=48,
            cnt_y=1210, cnt=110, side=64, side_dx=34)
G169 = dict(u=42, rh=27, rg=11, cg=40, x0=680, y0=150, split=20, hdr=20, hdr_dy=32,
            cnt_y=560, cnt=70, side=40, side_dx=20)

if __name__ == "__main__":
    render(1080, 1920, G916, layout_916).save(OUT_916, quality=93)
    render(1280, 720, G169, layout_169).save(OUT_169)
    print("saved", OUT_916, OUT_169, "counts", COUNTS, "ways4", ways(4))
