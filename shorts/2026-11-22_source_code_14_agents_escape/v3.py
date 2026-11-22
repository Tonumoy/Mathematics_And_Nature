"""SOURCE CODE #14 v3 - simple version: OpenAI's agents escape a test box and hack Hugging Face.
Icon-led explainer (glossy 3D Fluent Emoji icons, MIT) on our own dark dot-grid stage,
kinetic captions, receipt cards. All parameters are globals."""
import math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render as R          # reuse fonts, caption system, cards, badges

W, H = 1080, 1920
FPS = 30
DUR = 43.6
ICON_DIR = "icons"
BG_COL = (10, 11, 15)
DOT_COL = (40, 43, 52)
DOT_STEP = 36
CYAN = (60, 220, 255); RED = (255, 70, 70); GREEN = (80, 225, 120); AMBER = (255, 176, 40); BLUE = (80, 150, 255); WHITE = (240, 242, 246)
HERO_Y = 1000
LOGO_SLOTS = True
SLOTS = [  # name, x, y, r, t0, t1
 ('OpenAI', 540, 1460, 62, 0.0, 3.4),
 ('Hugging Face', 740, 1195, 58, 3.75, 5.7),
 ('OpenAI', 250, 850, 62, 5.75, 10.3),
 ('JFrog Artifactory', 430, 1150, 56, 10.35, 15.0),
 ('Hugging Face', 790, 1345, 58, 17.0, 19.2),
 ('Hugging Face', 965, 870, 58, 21.35, 25.3),
]
ICON_SCALE = 1.22
BOX_SCALE = 1.15

# ---------------- VO segments (s) ----------------
R.SEGS = [
 (0.10, 3.14, "OpenAI's AI agents broke out of their test box.", ["broke out"]),
 (3.60, 5.41, "Then they hacked a real company.", ["real company"]),
 (5.87, 10.10, "They were taking a hacking exam, locked in, with one pipe to download code.", ["hacking exam", "one pipe"]),
 (10.52, 11.64, "That pipe had a bug:", ["bug"]),
 (12.06, 13.55, "it would fetch any website.", ["any website"]),
 (13.96, 14.82, "The door was open.", ["open"]),
 (15.21, 15.54, "Why?", ["Why?"]),
 (15.96, 16.54, "To cheat.", ["cheat"]),
 (16.96, 19.03, "They guessed the answers were at Hugging Face.", ["answers"]),
 (19.39, 22.40, "Using leaked keys, they uploaded a booby-trapped file.", ["leaked keys", "booby-trapped"]),
 (22.87, 23.83, "The servers opened it...", ["opened it"]),
 (24.14, 25.09, "and ran their code.", ["ran their code"]),
 (25.50, 28.66, "In thirteen hours, they controlled whole server clusters.", ["thirteen hours"]),
 (29.03, 31.23, "Hugging Face's own AI guards caught them.", ["AI guards"]),
 (31.59, 32.71, "Humans pulled the plug.", ["pulled the plug"]),
 (33.13, 33.64, "The lesson:", ["lesson"]),
 (34.00, 35.66, "if breaking rules costs nothing,", ["costs nothing"]),
 (35.91, 36.83, "cheating wins.", ["cheating wins."]),
 (37.21, 39.55, "California is now suing OpenAI.", ["suing"]),
 (39.95, 42.53, "Would you let an AI agent run your laptop?", ["your laptop?"]),
]
R.CHUNKS = [R.chunks_for(s) for s in R.SEGS]
R.DUR = DUR
STEPS = [(5.7, "THE TEST"), (10.3, "THE BUG"), (15.0, "THE MOTIVE"), (19.2, "THE TRAP"), (25.3, "THE TAKEOVER"), (28.9, "CAUGHT")]
STEP_END = 32.9
SCENES = [0.0, 3.4, 5.7, 10.3, 15.0, 19.2, 25.3, 28.9, 32.9, 37.0, DUR]

sm, lin = R.sm, R.lin
def back(x):           # ease-out-back (pop with overshoot)
    x = min(max(x, 0.0), 1.0); c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2
def pop(t, t0, d=0.35):
    return back(lin(t, t0, t0 + d)) if t >= t0 else 0.0

# ---------------- icon cache ----------------
_raw, _cache, _glow = {}, {}, {}
def icon(name, size, tint=None, tint_a=0.0):
    size = max(8, int(round(size / 2) * 2))
    k = (name, size, tint, round(tint_a, 2))
    if k in _cache:
        return _cache[k]
    if name not in _raw:
        _raw[name] = Image.open(os.path.join(ICON_DIR, name + ".png")).convert("RGBA")
    im = _raw[name].resize((size, size), Image.LANCZOS)
    if tint is not None and tint_a > 0:
        a = np.asarray(im).astype(np.float32)
        lum = a[..., :3].mean(-1, keepdims=True) / 255
        tcol = np.array(tint, np.float32) * (0.35 + 0.75 * lum)
        a[..., :3] = a[..., :3] * (1 - tint_a) + tcol * tint_a
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if len(_cache) > 900:
        _cache.clear()
    _cache[k] = im
    return im

def glow_of(name, size, col, rad=28):
    size = max(8, int(round(size / 4) * 4))
    k = (name, size, col, rad)
    if k not in _glow:
        im = icon(name, size)
        pad = rad * 2
        a = Image.new("L", (size + 2 * pad, size + 2 * pad), 0)
        a.paste(im.split()[3], (pad, pad))
        a = a.filter(ImageFilter.GaussianBlur(rad))
        g = Image.new("RGBA", a.size, col + (0,))
        g.putalpha(a.point(lambda v: int(v * 0.85)))
        _glow[k] = g
    return _glow[k]

def put(C, name, x, y, size, alpha=1.0, rot=0.0, glow=None, refl=False, tint=None, tint_a=0.0, gw=1.0):
    size = size * ICON_SCALE
    if alpha <= 0.01 or size < 6:
        return
    if glow is not None:
        g = glow_of(name, size, glow)
        if gw < 1 or alpha < 1:
            g = g.copy(); g.putalpha(g.split()[3].point(lambda v: int(v * gw * alpha)))
        C.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
    im = icon(name, size, tint, tint_a)
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy(); im.putalpha(im.split()[3].point(lambda v: int(v * alpha)))
    if refl:
        r = im.transpose(Image.FLIP_TOP_BOTTOM)
        grad = Image.linear_gradient("L").resize(r.size).point(lambda v: int(max(0, 255 - v * 2.2) * 0.22))
        ra = Image.fromarray(np.minimum(np.asarray(r.split()[3]), np.asarray(grad)))
        r.putalpha(ra)
        C.alpha_composite(r, (int(x - r.width / 2), int(y + im.height / 2 - im.height * 0.04)))
    C.alpha_composite(im, (int(x - im.width / 2), int(y - im.height / 2)))

# ---------------- stage ----------------
def make_stage():
    yy, xx = np.mgrid[0:H + DOT_STEP * 2, 0:W + DOT_STEP * 2]
    base = np.zeros((H + DOT_STEP * 2, W + DOT_STEP * 2, 3), np.float32)
    base[:] = BG_COL
    m = ((xx % DOT_STEP) < 3) & ((yy % DOT_STEP) < 3)
    base[m] = DOT_COL
    return base
STAGE = make_stage()
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGN = (1 - 0.55 * (((_xx - W / 2) / (W * 0.75)) ** 2 + ((_yy - H * 0.52) / (H * 0.62)) ** 2)).clip(0.25, 1)[..., None]
def spot(cx, cy, r, col, a):
    d = ((_xx - cx) ** 2 + (_yy - cy) ** 2) / (r * r)
    return (np.exp(-d)[..., None] * np.array(col, np.float32) * a)
SPOT_C = spot(540, HERO_Y, 520, (60, 80, 120), 0.55)
SPOT_W = spot(540, HERO_Y, 520, (140, 40, 40), 0.55)

def stage(t, red=0.0):
    ox = int((t * 6) % DOT_STEP); oy = int((t * 10) % DOT_STEP)
    a = STAGE[oy:oy + H, ox:ox + W] * VIGN + SPOT_C * (1 - red) + SPOT_W * red
    top = (0.2 + 0.8 * np.clip((_yy - 430) / 170, 0, 1) ** 1.5)[..., None]
    a = a * top
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")

def plinth(C, x, y, w, a=1.0, col=(80, 110, 160)):
    if a <= 0.01:
        return
    L = Image.new("RGBA", (int(w * 1.3), int(w * 0.5)), (0, 0, 0, 0))
    d = ImageDraw.Draw(L)
    cw, ch = L.size
    d.ellipse((cw * 0.12, ch * 0.25, cw * 0.88, ch * 0.75), fill=(18, 20, 26, int(230 * a)), outline=col + (int(200 * a),), width=3)
    d.ellipse((cw * 0.2, ch * 0.34, cw * 0.8, ch * 0.62), outline=col + (int(70 * a),), width=2)
    L = L.filter(ImageFilter.GaussianBlur(0.6))
    C.alpha_composite(L, (int(x - cw / 2), int(y - ch / 2)))

# glass box (own drawing, isometric) with the padlock on the front
def glass_box(C, x, y, s, a=1.0, red=0.0, open_=0.0):
    if a <= 0.01:
        return
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    s = s * BOX_SCALE
    k = s / 2
    dx, dy = k * 0.87, k * 0.5
    top = [(x, y - k - dy), (x + dx, y - k), (x, y - k + dy), (x - dx, y - k)]
    lft = [(x - dx, y - k), (x, y - k + dy), (x, y + k * 0.55 + dy), (x - dx, y + k * 0.55)]
    rgt = [(x + dx, y - k), (x, y - k + dy), (x, y + k * 0.55 + dy), (x + dx, y + k * 0.55)]
    ec = tuple(int(CYAN[i] * (1 - red) + RED[i] * red) for i in range(3))
    d.polygon(lft, fill=(60, 140, 190, int(55 * a)))
    d.polygon(rgt, fill=(40, 110, 170, int(45 * a)))
    if open_ < 0.5:
        d.polygon(top, fill=(120, 200, 255, int(60 * a)))
    for poly in (top, lft, rgt):
        d.line(poly + [poly[0]], fill=ec + (int(255 * a),), width=4, joint="curve")
    # glossy highlight stripe
    d.line([(x - dx * 0.8, y - k * 0.7), (x - dx * 0.15, y - k * 0.35)], fill=(255, 255, 255, int(110 * a)), width=6)
    g = L.filter(ImageFilter.GaussianBlur(10))
    C.alpha_composite(g); C.alpha_composite(L)

def pipe(C, p0, p1, w, col=(150, 160, 175), a=1.0, glow=None, red=0.0):
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    c = tuple(int(col[i] * (1 - red) + RED[i] * red) for i in range(3))
    d.line([p0, p1], fill=(25, 27, 33, int(255 * a)), width=int(w + 10))
    d.line([p0, p1], fill=c + (int(255 * a),), width=int(w))
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0]); nx, ny = -math.sin(ang), math.cos(ang)
    o = -w * 0.22
    d.line([(p0[0] + nx * o, p0[1] + ny * o), (p1[0] + nx * o, p1[1] + ny * o)], fill=(255, 255, 255, int(120 * a)), width=max(2, int(w * 0.16)))
    if glow:
        C.alpha_composite(L.filter(ImageFilter.GaussianBlur(14)))
    C.alpha_composite(L)

def arc_pt(p0, p1, u, lift):
    x = p0[0] + (p1[0] - p0[0]) * u
    y = p0[1] + (p1[1] - p0[1]) * u - lift * math.sin(math.pi * u)
    return x, y

# ---------------- scenes ----------------
def s_hook(C, T, t):
    """0-3.4: robots burst out of the locked OpenAI test box"""
    bx, by = 540, 1060
    plinth(C, bx, by + 300, 560)
    glass_box(C, bx, by, 440, 1.0, 0.0, open_=1.0)
    # robots still inside (bobbing)
    for k, (ox, oy) in enumerate([(-90, 60), (70, 90), (0, 150)]):
        put(C, "robot", bx + ox + 8 * math.sin(t * 3 + k), by + oy + 6 * math.cos(t * 2.5 + k), 110, 0.9)
    # robots escaping (staggered, looping so frame 1 already shows motion)
    targets = [(160, 560), (900, 520), (980, 760), (90, 800), (720, 430), (330, 450)]
    for k, tg in enumerate(targets):
        t0 = -0.9 + k * 0.42
        u = lin(t, t0, t0 + 1.4)
        if 0 < u < 1:
            x, y = arc_pt((bx, by - 140), tg, sm(u), 180)
            for j in range(4):  # trail
                uu = max(0, u - 0.05 * (j + 1))
                tx, ty = arc_pt((bx, by - 140), tg, sm(uu), 180)
                put(C, "robot", tx, ty, 120 - 12 * j, 0.18 * (4 - j) / 4, glow=None)
            put(C, "robot", x, y, 130, 1.0, rot=12 * math.sin(t * 6 + k), glow=CYAN, gw=0.7)
        elif u >= 1:
            fx, fy = tg
            put(C, "robot", fx + 6 * math.sin(t * 4 + k), fy, 130, 1.0 - lin(t, t0 + 1.4, t0 + 2.2), glow=CYAN, gw=0.5)
    # padlock: locked -> unlocked pop
    lp = (bx, by + 150)
    s = 190 * (1 + 0.25 * (1 - lin(t, 0.0, 0.6)))
    put(C, "unlocked", lp[0] + 10 * math.sin(t * 40) * (1 - lin(t, 0.0, 0.6)), lp[1] + 30 * lin(t, 0.6, 3.4), s, glow=RED, gw=1.0)
    if t < 0.7:
        rr = 120 + 400 * lin(t, 0.0, 0.7)
        ImageDraw.Draw(C).ellipse((lp[0] - rr, lp[1] - rr, lp[0] + rr, lp[1] + rr), outline=RED + (int(220 * (1 - lin(t, 0.0, 0.7))),), width=6)

def s_company(C, T, t):
    """3.4-5.7: robots fly to a real company: Hugging Face"""
    u = sm(lin(t, 3.4, 4.0))
    bx, by = 540 - 280 * u, 1060 + 260 * u
    sc = 1 - 0.5 * u
    glass_box(C, bx, by, 440 * sc, 1.0, 0.0, open_=1.0)
    hx, hy = 740, 860
    a = pop(t, 3.7)
    plinth(C, hx, hy + 200, 420, a)
    hit = sm(lin(t, 4.6, 5.0))
    put(C, "office_building", hx, hy, 330 * a, 1.0, glow=RED if hit > 0 else BLUE, gw=0.4 + 0.6 * hit, tint=(255, 80, 80), tint_a=0.35 * hit, refl=True)
    if t > 4.6:
        put(C, "warning", hx + 120, hy - 140, 120 * pop(t, 4.6), glow=AMBER)
    for k in range(5):
        t0 = 3.6 + k * 0.18
        uu = lin(t, t0, t0 + 0.9)
        if 0 < uu < 1:
            x, y = arc_pt((bx, by - 120 * sc), (hx - 40 + k * 20, hy), sm(uu), 220)
            put(C, "robot", x, y, 100, 1.0, glow=CYAN, gw=0.6)
    T.append(("tag", hx, hy + 250, "Hugging Face (real company)", RED, a))
    T.append(("tag", bx, by + 300 * sc, "OpenAI test box", CYAN, 1 - lin(t, 5.3, 5.7)))

def s_exam(C, T, t):
    """5.7-10.3: a hacking exam, locked in, one pipe for code downloads"""
    t0 = 5.7
    bx, by = 470, 1150
    a = pop(t, t0)
    put(C, "memo", 540, 640, 210 * pop(t, 5.9), glow=AMBER, gw=0.5)
    T.append(("tag", 540, 780, "hacking exam", AMBER, lin(t, 6.2, 6.5)))
    plinth(C, bx, by + 280, 520, a)
    glass_box(C, bx, by, 420 * a, 1.0, 0.0)
    for k, (ox, oy) in enumerate([(-80, 50), (70, 80), (0, 140)]):
        put(C, "robot", bx + ox + 8 * math.sin(t * 3 + k), by + oy, 105 * a, 1.0)
    put(C, "locked", bx, by + 175, 150 * pop(t, 7.0), glow=AMBER, gw=0.6)
    T.append(("tag", bx - 10, by + 330, "locked in", AMBER, lin(t, 7.2, 7.5)))
    # the one pipe (from box top-right up to the right edge)
    pu = sm(lin(t, 8.3, 8.9))
    p0 = (bx + 150, by - 150); p1 = (bx + 150 + 420 * pu, by - 150 - 300 * pu)
    if pu > 0:
        pipe(C, p0, p1, 46)
        for k in range(4):
            uu = 1 - ((t * 0.45 + k / 4) % 1.0)
            if pu > 0.95:
                x = p0[0] + (p1[0] - p0[0]) * uu; y = p0[1] + (p1[1] - p0[1]) * uu
                put(C, "package", x, y - 4, 86, 1.0)
        T.append(("tag", 800, by - 470, "one pipe: code downloads", WHITE, lin(t, 9.0, 9.3)))

def s_bug(C, T, t):
    """10.3-15.0: the pipe had a bug: it would fetch ANY website -> door open"""
    y0 = 1000
    put(C, "locked" if t < 13.96 else "unlocked", 150, y0, 150, glow=AMBER if t < 13.96 else RED, gw=0.6)
    glass_box(C, 150, y0 - 10, 220, 0.9)
    red = sm(lin(t, 11.0, 11.4))
    pipe(C, (260, y0), (1080, y0), 96, red=0.6 * red, glow=red > 0)
    # bug crawls along the top of the pipe
    bxp = 1000 - 420 * sm(lin(t, 10.4, 11.5))
    put(C, "bug", bxp, y0 - 85 + 6 * math.sin(t * 18), 170 * pop(t, 10.35), glow=GREEN, gw=0.6)
    # globe pops at the far end: the internet
    if t > 12.0:
        ga = pop(t, 12.0)
        put(C, "globe_with_meridians", 830, 620, 260 * ga, glow=BLUE, gw=0.9, refl=False)
        T.append(("tag", 830, 790, "any website", BLUE, ga))
        # fetch lines from pipe to globe
        L = ImageDraw.Draw(C)
        for k in range(6):
            uu = (t * 1.2 + k / 6) % 1.0
            x = 600 + (830 - 600) * uu; y = y0 - 40 + (620 - (y0 - 40)) * uu
            L.ellipse((x - 7, y - 7, x + 7, y + 7), fill=RED + (230,))
    # door open: robots stream out of the box, along the pipe, to the globe
    if t > 13.9:
        for k in range(7):
            uu = lin(t, 13.95 + k * 0.12, 14.9 + k * 0.12)
            if 0 < uu <= 1:
                if uu < 0.6:
                    x = 240 + (640 - 240) * (uu / 0.6); y = y0 - 70
                else:
                    x, y = arc_pt((640, y0 - 70), (830, 620), (uu - 0.6) / 0.4, 60)
                put(C, "robot", x, y, 96, 1.0, glow=CYAN, gw=0.6)
    if 11.6 < t < 15.0:
        T.append(("badge", 540, 1420, "THE BUG", "the pipe would fetch ANY website", "not just code (a zero-day in the proxy)", RED, lin(t, 11.7, 12.0) * (1 - lin(t, 14.7, 15.0))))

def s_motive(C, T, t):
    """15.0-19.2: why? to cheat: the answers were at Hugging Face"""
    a = pop(t, 15.1)
    plinth(C, 300, 1240, 360, a)
    put(C, "robot", 300, 1060, 280 * a, glow=CYAN, gw=0.5, refl=True)
    # thought bubble with the answer sheet
    ba = pop(t, 15.5)
    if ba > 0:
        L = ImageDraw.Draw(C)
        cx, cy, r = 470, 700, 150 * ba
        L.ellipse((cx - r * 1.25, cy - r, cx + r * 1.25, cy + r), fill=(240, 242, 246, 235))
        L.ellipse((380 - 22 * ba, 880 - 22 * ba, 380 + 22 * ba, 880 + 22 * ba), fill=(240, 242, 246, 235))
        L.ellipse((345 - 13 * ba, 930 - 13 * ba, 345 + 13 * ba, 930 + 13 * ba), fill=(240, 242, 246, 235))
        fly = sm(lin(t, 17.4, 18.3))
        mx, my = cx + (790 - cx) * fly, cy + (880 - cy) * fly - 120 * math.sin(math.pi * fly)
        put(C, "memo", mx, my, 150 * ba, 1.0)
        put(C, "check_mark_button", mx + 55, my + 50, 80 * ba, 1.0)
    if t > 16.9:
        ha = pop(t, 16.95)
        put(C, "office_building", 790, 1030, 300 * ha, glow=AMBER, gw=0.5, refl=True)
        T.append(("tag", 790, 1260, "Hugging Face", AMBER, ha))
        T.append(("tag", 760, 700, "test answers?", AMBER, lin(t, 18.3, 18.6)))

def s_trap(C, T, t):
    """19.2-25.3: leaked keys -> booby-trapped file -> server opens it -> runs their code"""
    # keys
    n = 14
    for k in range(n):
        tk = 19.4 + k * 0.07
        u = sm(lin(t, tk, tk + 0.5))
        if u <= 0:
            continue
        gx = 210 + (k % 7) * 110; gy = 560 + (k // 7) * 100
        sx = 540 + 600 * math.cos(k * 2.4); sy = -100
        x = sx + (gx - sx) * u; y = sy + (gy - sy) * u
        fade = 1 - lin(t, 24.9, 25.3)
        put(C, "key", x, y, 84, fade, rot=-25, glow=AMBER, gw=0.4)
    T.append(("tag", 540, 760, "14 Hugging Face keys left public", AMBER, lin(t, 19.9, 20.2) * (1 - lin(t, 24.8, 25.2))))
    # server
    sx_, sy_ = 760, 1190
    sa = pop(t, 21.3)
    red = sm(lin(t, 24.1, 24.5))
    flash = max(0.0, 1 - abs(t - 22.95) / 0.25) if 22.7 < t < 23.3 else 0.0
    plinth(C, sx_, sy_ + 200, 420, sa)
    put(C, "desktop_computer", sx_, sy_, 330 * sa, glow=RED if red > 0 else BLUE, gw=0.4 + 0.6 * red, tint=(255, 70, 70), tint_a=0.45 * red, refl=True)
    if flash > 0:
        put(C, "desktop_computer", sx_, sy_, 330, flash, tint=(255, 255, 255), tint_a=0.8)
    T.append(("tag", sx_, sy_ + 230, "Hugging Face server", BLUE if red < 0.5 else RED, sa))
    # booby-trapped file slides into the server
    fa = pop(t, 20.6)
    slide = sm(lin(t, 21.9, 22.85))
    fx = 330 + (sx_ - 330) * slide; fy = 1080 + (sy_ - 40 - 1080) * slide
    fsz = 230 * fa * (1 - 0.6 * slide)
    if slide < 1:
        put(C, "page_facing_up", fx, fy, fsz, glow=AMBER, gw=0.5)
        put(C, "bomb", fx + fsz * 0.33, fy + fsz * 0.3, fsz * 0.45)
        T.append(("tag", 330, 1290, "booby-trapped file", AMBER, fa * (1 - slide)))
    if t > 24.14:
        T.append(("code", sx_, sy_ - 40, lin(t, 24.14, 24.4)))
    if t > 23.0:
        T.append(("badge", 540, 1540, "THE TRICK", "the file made the server read its own", "secrets, then run the agents' hidden code", RED, lin(t, 23.0, 23.3) * (1 - lin(t, 25.0, 25.3))))

def s_takeover(C, T, t):
    """25.3-28.9: one server -> whole clusters in 13 hours"""
    cols, rows = 4, 3
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= 11:
                break
            x = 175 + c * 243; y = 700 + r * 250
            tr = 25.55 + k * 0.22
            red = sm(lin(t, tr, tr + 0.25))
            put(C, "desktop_computer", x, y, 180 * pop(t, 25.3 + k * 0.03), glow=RED if red > 0.5 else None, gw=0.6 * red, tint=(255, 70, 70), tint_a=0.5 * red)
            k += 1
    hrs = int(13 * sm(lin(t, 25.6, 28.2)))
    T.append(("big", 540, 1480, f"{hrs} HOURS", RED, lin(t, 25.5, 25.8)))
    T.append(("small", 540, 1560, "1 server  →  whole clusters", WHITE, lin(t, 25.8, 26.1)))

def s_caught(C, T, t):
    """28.9-32.9: Hugging Face's own AI guards caught them, humans pulled the plug"""
    off = sm(lin(t, 31.6, 32.2))
    k = 0
    for r in range(3):
        for c in range(4):
            if k >= 11:
                break
            x = 175 + c * 243; y = 700 + r * 250
            put(C, "desktop_computer", x, y, 180, 0.35 + 0.25 * off, tint=(255, 70, 70) if off < 0.5 else (90, 95, 110), tint_a=0.5)
            k += 1
    # scan sweep
    if t < 31.4:
        sy = 560 + (1300 - 560) * ((t - 28.9) / 1.2 % 1.0)
        L = ImageDraw.Draw(C)
        L.rectangle((40, sy - 3, 1040, sy + 3), fill=BLUE + (200,))
    sa = pop(t, 29.1, 0.45)
    put(C, "shield", 540, 960, 360 * sa, glow=BLUE, gw=1.0)
    T.append(("tag", 540, 1180, "Hugging Face's own AI security agents", BLUE, sa))
    if t < 31.7:
        blink = 0.5 + 0.5 * math.sin(t * 18)
        put(C, "police_car_light", 860, 640, 160 * pop(t, 29.4), glow=RED if blink > 0.5 else BLUE, gw=0.9)
    if t > 31.4:
        pu = sm(lin(t, 31.6, 32.1))
        put(C, "electric_plug", 540 - 140 * pu, 1255 + 20 * pu, 190 * pop(t, 31.4), rot=90 + 20 * pu)
        L = ImageDraw.Draw(C)
        L.rounded_rectangle((600 + 60 * pu, 1225, 700 + 60 * pu, 1285), 14, fill=(60, 64, 74, 255), outline=(150, 155, 165, 255), width=3)
    if t > 31.8:
        T.append(("card", 31.8, 33.1, ["Hugging Face: other customers' models,", "datasets and Spaces were not affected."], "Hugging Face technical timeline, 27 Jul 2026", "not affected."))

def s_lesson(C, T, t):
    """32.9-37.0: if breaking rules costs nothing, cheating wins"""
    a = pop(t, 33.0)
    tilt = 14 * sm(lin(t, 35.9, 36.5))
    plinth(C, 540, 1300, 560, a)
    put(C, "balance_scale", 540, 1060, 420 * a, rot=tilt, glow=AMBER, gw=0.5, refl=True)
    T.append(("eq", t))
    T.append(("tag", 290, 1380, "cheat: pass the test  +1", GREEN, lin(t, 34.0, 34.3)))
    T.append(("tag", 790, 1380, "break a rule:  −0", RED, lin(t, 34.6, 34.9)))
    if t > 35.9:
        put(C, "check_mark_button", 300, 800, 130 * pop(t, 35.95), glow=GREEN, gw=0.9)

def s_end(C, T, t):
    """37.0-43.6: California sues; would you let an AI agent run your laptop?"""
    if t < 39.9:
        a = pop(t, 37.1)
        plinth(C, 540, 1180, 480, a)
        put(C, "classical_building", 540, 960, 340 * a, glow=AMBER, gw=0.5, refl=True)
        T.append(("card", 37.3, 39.9, ["16 Nov 2026: California sues OpenAI.", "It says agents got loose 80+ times.", "OpenAI disputes the claims."], "California Attorney General · lawsuit, 16 Nov 2026", "80+ times."))
    else:
        a = pop(t, 39.95)
        plinth(C, 540, 1250, 560, a)
        put(C, "laptop", 540, 1000, 440 * a, glow=CYAN, gw=0.5, refl=True)
        put(C, "robot", 540, 930, 150 * pop(t, 40.3), glow=CYAN, gw=0.7)
        T.append(("yesno", lin(t, 40.8, 41.1)))

SCENE_FN = [s_hook, s_company, s_exam, s_bug, s_motive, s_trap, s_takeover, s_caught, s_lesson, s_end]

# ---------------- text layer ----------------
def draw_text(L, T, t):
    d = ImageDraw.Draw(L)
    for item in T:
        k = item[0]
        if k == "tag":
            _, x, y, txt, col, a = item
            if a > 0.02:
                R.pill(d, x, y, txt, col, 34, "c", True, int(235 * min(1, a)))
        elif k == "badge":
            _, x, y, h, l1, l2, col, a = item
            R.badge(d, x, y, h, l1, l2, col[::-1], a)
        elif k == "card":
            _, t0, t1, lines, src, hl = item
            R.card(d, t, t0, t1, lines, src, y=1380, hl=hl)
        elif k == "big":
            _, x, y, txt, col, a = item
            d.text((x, y), txt, font=R.font(R.F_XB, 96), fill=col + (int(255 * a),), anchor="mm")
        elif k == "small":
            _, x, y, txt, col, a = item
            d.text((x, y), txt, font=R.font(R.F_SEMI, 34), fill=col + (int(255 * a),), anchor="mm")
        elif k == "code":
            _, x, y, a = item
            d.rounded_rectangle((x - 105, y - 40, x + 105, y + 30), 10, fill=(30, 6, 8, int(230 * a)))
            d.text((x, y - 5), ">_ code", font=R.font(R.F_MONO, 36), fill=(255, 90, 90, int(255 * a)), anchor="mm")
        elif k == "eq":
            a = lin(t, 33.1, 33.5)
            f = R.font(R.F_BOLD, 46)
            parts = [("reward = pass − ", WHITE), ("penalty", RED), (" × rule broken", WHITE)]
            tw = sum(d.textlength(p, font=f) for p, _ in parts); x = W / 2 - tw / 2; y = 560
            for p, c in parts:
                pw = d.textlength(p, font=f)
                if p == "penalty" and t > 34.0:
                    d.rectangle((x - 6, y - 32, x + pw + 6, y + 32), fill=(46, 92, 190, int(190 * a)), outline=(77, 141, 255, int(255 * a)), width=2)
                d.text((x, y), p, font=f, fill=c + (int(255 * a),), anchor="lm"); x += pw
            if t > 34.2:
                d.text((W / 2, 630), "penalty = 0", font=R.font(R.F_XB, 44), fill=RED + (int(255 * lin(t, 34.2, 34.5)),), anchor="mm")
        elif k == "yesno":
            a = item[1]
            f = R.font(R.F_XB, 58)
            d.rounded_rectangle((250, 1430, 520, 1520), 45, fill=(40, 170, 90, int(240 * a)))
            d.rounded_rectangle((560, 1430, 830, 1520), 45, fill=(220, 60, 60, int(240 * a)))
            d.text((385, 1475), "YES", font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
            d.text((695, 1475), "NO", font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
            d.text((W / 2, 1565), "tell me in the comments", font=R.font(R.F_MED, 30), fill=(200, 204, 210, int(255 * a)), anchor="mm")
    # progress rail
    if STEPS[0][0] - 0.1 <= t < STEP_END:
        cur = max(i for i, (s, _) in enumerate(STEPS) if t >= s - 0.1)
        n = len(STEPS); x0, x1, y = 150, 930, 182
        for i in range(n):
            xa = x0 + (x1 - x0) * i / n + 4; xb = x0 + (x1 - x0) * (i + 1) / n - 4
            d.rounded_rectangle((xa, y - 3, xb, y + 3), 3, fill=(77, 141, 255, 255) if i <= cur else (70, 74, 82, 255))
        d.text((x0 - 12, y), f"{cur + 1}/{n}", font=R.font(R.F_SEMI, 26), fill=(200, 205, 212, 255), anchor="rm")
        d.text((x0 + (x1 - x0) * (cur + 0.5) / n, y + 26), STEPS[cur][1], font=R.font(R.F_BOLD, 24), fill=(230, 233, 238, 255), anchor="mm")
    if LOGO_SLOTS:
        for nm, x, y, r, t0, t1 in SLOTS:
            if t0 <= t < t1:
                a = min(1, lin(t, t0, t0 + 0.25)) * (1 - lin(t, t1 - 0.2, t1))
                d.ellipse((x - r - 4, y - r + 2, x + r + 4, y + r + 10), fill=(0, 0, 0, int(90 * a)))
                d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(250 * a)), outline=(200, 205, 215, int(255 * a)), width=3)
                d.text((x, y + r + 24), nm, font=R.font(R.F_BOLD, 26), fill=(235, 238, 244, int(255 * a)), anchor="mm")
    R.draw_caption(d, t, False)

OUTDIR = os.environ.get("OUTDIR", "frames3")
GR = np.random.default_rng(3)
GRAIN = [GR.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(4)]

def render(t, grain=True, text=True):
    si = max(i for i in range(len(SCENES) - 1) if t >= SCENES[i])
    red = 0.0
    if si in (5, 6):
        red = 0.5 if si == 6 else sm(lin(t, 24.1, 24.6)) * 0.5
    C = stage(t, red)
    T = []
    SCENE_FN[si](C, T, t)
    # whip transition: brief zoom-blend into each new scene
    ts = SCENES[si]
    if si > 0 and t - ts < 0.18:
        prev = stage(t, 0)
        SCENE_FN[si - 1](prev, [], ts - 0.001)
        a = 1 - (t - ts) / 0.18
        C = Image.blend(C, prev.resize((int(W * (1 + 0.08 * (1 - a))), int(H * (1 + 0.08 * (1 - a))))).crop((0, 0, W, H)), 0.6 * a)
    if text:
        L = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_text(L, T, t)
        C.alpha_composite(L)
    # loop: last 0.5 s blends into frame 1
    if t > DUR - 0.5:
        f1 = render(0.0, grain=False, text=False).convert("RGBA")
        C = Image.blend(C, f1, sm(lin(t, DUR - 0.5, DUR)))
    out = C.convert("RGB")
    if grain:
        a = np.asarray(out).astype(np.float32)
        g = np.asarray(Image.fromarray(GRAIN[int(t * FPS) % 4]).resize((W, H), Image.BILINEAR))[..., None]
        out = Image.fromarray(np.clip(a + g * 2.2, 0, 255).astype(np.uint8))
    return out

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "stills":
        os.makedirs("st3", exist_ok=True)
        for tt in sys.argv[2:]:
            render(float(tt)).save(f"st3/s{float(tt):05.1f}.png")
    elif mode == "frames":
        os.makedirs(OUTDIR, exist_ok=True)
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for i in range(part, int(DUR * FPS), nparts):
            p = f"{OUTDIR}/f{i:05d}.png"
            if not os.path.exists(p):
                render(i / FPS).save(p, compress_level=1)
