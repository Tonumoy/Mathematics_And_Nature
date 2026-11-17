# SOURCE CODE chain series — shared render kit (Coded Cosmos)
import numpy as np, math, os, functools
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------- global parameters ----------------
W, H = 1080, 1920
FPS = 30
SS = 2                      # supersample for vector layers
BG = (5, 8, 15)
CYAN = (80, 220, 255); AMBER = (255, 176, 60); RED = (255, 72, 72)
GREEN = (90, 232, 140); VIOLET = (178, 120, 255); WHITE = (240, 244, 250)
YELLOW = (255, 214, 60); GREY = (150, 160, 175)
SAFE_X0, SAFE_X1, SAFE_Y0, SAFE_Y1 = 40, 930, 160, 1590
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_XB = os.path.join(HERE, "fonts", "Poppins-ExtraBold.ttf")
FONT_SB = os.path.join(HERE, "fonts", "Poppins-SemiBold.ttf")
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def needs_dj(s): return any(ord(ch) > 0x2000 for ch in s)

@functools.lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)

def ease(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def ease_out(x):
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3

def ramp(t, a, b):
    if b <= a: return 1.0 if t >= b else 0.0
    return min(max((t - a) / (b - a), 0.0), 1.0)

def lerp(a, b, x): return a + (b - a) * x

def lerpc(c1, c2, x): return tuple(int(lerp(c1[i], c2[i], x)) for i in range(3))

# ---------------- canvas helpers ----------------
def new_layer(ss=SS):
    return Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))

def down(layer):
    return layer.resize((W, H), Image.LANCZOS) if layer.size != (W, H) else layer

def glow(layer, radius=18, strength=1.0):
    """layer: RGBA at 1x. returns glow RGBA (blur at quarter res)."""
    small = layer.resize((W // 4, H // 4), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(radius / 4))
    g = small.resize((W, H), Image.BILINEAR)
    if strength != 1.0:
        a = np.asarray(g).astype(np.float32)
        a[..., 3] = np.clip(a[..., 3] * strength, 0, 255)
        g = Image.fromarray(a.astype(np.uint8))
    return g

def add_light(base_rgb, layer, strength=1.0):
    """additive blend of RGBA layer onto RGB float array."""
    a = np.asarray(layer).astype(np.float32)
    al = a[..., 3:4] / 255.0 * strength
    return base_rgb + a[..., :3] * al

def over(base_rgb, layer, alpha=1.0):
    a = np.asarray(layer).astype(np.float32)
    al = a[..., 3:4] / 255.0 * alpha
    return base_rgb * (1 - al) + a[..., :3] * al

def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

# ---------------- background ----------------
@functools.lru_cache(None)
def starfield(seed=3):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    y, x = np.mgrid[0:H, 0:W]
    # subtle nebula gradient
    img[...] = np.array(BG, np.float32)
    neb = np.exp(-(((x - 760) / 520.0) ** 2 + ((y - 420) / 700.0) ** 2))
    img += neb[..., None] * np.array([18, 14, 40], np.float32)
    n = 900
    xs = rng.uniform(0, W, n); ys = rng.uniform(0, H, n)
    br = rng.power(6, n) * 230 + 25
    for xx, yy, b in zip(xs, ys, br):
        xi, yi = int(xx), int(yy)
        img[yi, xi] += b
        if b > 200 and 1 <= xi < W - 1 and 1 <= yi < H - 1:
            img[yi - 1:yi + 2, xi] += b * 0.35; img[yi, xi - 1:xi + 2] += b * 0.35
    return np.clip(img, 0, 255)


def _faded(d, alpha, fn):
    """draw fn(d2) at full opacity on a temp layer, then composite at alpha (PIL draw would overwrite)."""
    img = d._image
    tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fn(ImageDraw.Draw(tmp))
    bb = tmp.getbbox()
    if not bb: return
    crop = tmp.crop(bb)
    a = np.asarray(crop).copy(); a[..., 3] = (a[..., 3].astype(np.float32) * alpha / 255).astype(np.uint8)
    img.alpha_composite(Image.fromarray(a), (bb[0], bb[1]))

# ---------------- text ----------------
def text_size(txt, f):
    b = f.getbbox(txt); return b[2] - b[0], b[3] - b[1]

def draw_text(d, xy, txt, size, fill=WHITE, path=FONT_XB, anchor="mm", stroke=0, stroke_fill=(0, 0, 0), ss=SS):
    if len(fill) == 4 and fill[3] < 255:
        if fill[3] <= 0: return
        return _faded(d, fill[3], lambda d2: draw_text(d2, xy, txt, size, fill[:3] + (255,), path, anchor, stroke, stroke_fill, ss))
    f = font(FONT_DJ if needs_dj(txt) and path != FONT_MONO else path, int(size * ss))
    d.text((xy[0] * ss, xy[1] * ss), txt, font=f, fill=fill, anchor=anchor,
           stroke_width=int(stroke * ss), stroke_fill=stroke_fill)

def caption(d, txt, y=1330, size=64, hl=(), alpha=255, ss=SS):
    """Big caption, words in hl turn yellow. Dark band behind."""
    if alpha < 255:
        if alpha <= 0: return
        return _faded(d, alpha, lambda d2: caption(d2, txt, y, size, hl, 255, ss))
    f = font(FONT_XB, int(size * ss))
    words = txt.split(" ")
    space = f.getlength(" ")
    widths = [f.getlength(w) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    maxw = (SAFE_X1 - SAFE_X0 - 40) * ss
    lines = [[]]; lw = 0
    for w, wd in zip(words, widths):
        if lines[-1] and lw + space + wd > maxw:
            lines.append([]); lw = 0
        lines[-1].append((w, wd)); lw += (space if len(lines[-1]) > 1 else 0) + wd
    lh = size * 1.18 * ss
    cx = (SAFE_X0 + SAFE_X1) / 2 * ss
    y0 = y * ss - lh * (len(lines) - 1) / 2
    for i, ln in enumerate(lines):
        tw = sum(wd for _, wd in ln) + space * (len(ln) - 1)
        x = cx - tw / 2; yy = y0 + i * lh
        pad = 18 * ss
        d.rounded_rectangle([x - pad, yy - lh / 2 - 2 * ss, x + tw + pad, yy + lh / 2 + 2 * ss],
                            radius=16 * ss, fill=(4, 6, 12, int(alpha * 0.62)))
        for w, wd in ln:
            clean = w.strip(",.?!:").lower()
            col = YELLOW if clean in hl else WHITE
            ff = font(FONT_DJ, int(size * 0.95 * ss)) if needs_dj(w) else f
            d.text((x, yy), w, font=ff, fill=col + (alpha,), anchor="lm",
                   stroke_width=int(3 * ss), stroke_fill=(0, 0, 0, alpha))
            x += wd + space

def pill(d, xy, txt, col=CYAN, size=30, alpha=255, anchor="mm", ss=SS):
    if alpha < 255:
        if alpha <= 0: return
        return _faded(d, alpha, lambda d2: pill(d2, xy, txt, col, size, 255, anchor, ss))
    f = font(FONT_DJ if needs_dj(txt) else FONT_SB, int(size * (0.92 if needs_dj(txt) else 1) * ss))
    tw = f.getlength(txt); th = size * 1.25 * ss
    x, y = xy[0] * ss, xy[1] * ss
    if anchor == "lm": x0 = x
    elif anchor == "rm": x0 = x - tw - 36 * ss
    else: x0 = x - tw / 2 - 18 * ss
    d.rounded_rectangle([x0, y - th / 2, x0 + tw + 36 * ss, y + th / 2], radius=th / 2,
                        fill=(8, 12, 22, int(alpha * 0.85)), outline=col + (alpha,), width=int(3 * ss))
    d.text((x0 + 18 * ss + tw / 2, y), txt, font=f, fill=col + (alpha,), anchor="mm")

def chip(d, xy, txt, col=AMBER, size=40, alpha=255, ss=SS):
    """equation chip, mono font"""
    if alpha < 255:
        if alpha <= 0: return
        return _faded(d, alpha, lambda d2: chip(d2, xy, txt, col, size, 255, ss))
    f = font(FONT_MONO, int(size * ss))
    tw = f.getlength(txt); th = size * 1.5 * ss
    x, y = xy[0] * ss, xy[1] * ss
    d.rounded_rectangle([x - tw / 2 - 26 * ss, y - th / 2, x + tw / 2 + 26 * ss, y + th / 2],
                        radius=12 * ss, fill=(10, 12, 20, int(alpha * 0.9)), outline=col + (int(alpha * 0.9),), width=int(2 * ss))
    d.text((x, y), txt, font=f, fill=col + (alpha,), anchor="mm")

def ring(d, xy, r, col=CYAN, alpha=255, width=4, ss=SS):
    if alpha < 255:
        if alpha <= 0: return
        return _faded(d, alpha, lambda d2: ring(d2, xy, r, col, 255, width, ss))
    x, y = xy[0] * ss, xy[1] * ss; r *= ss
    d.ellipse([x - r, y - r, x + r, y + r], outline=col + (alpha,), width=int(width * ss))

# ---------------- chain HUD (the growing "build") ----------------
CHAIN = [  # (label, kind)
    ("ROT", "p"), ("COLD", "f"), ("ICE MELTS", "p"), ("MACHINE", "f"), ("TOXIC GAS", "p"),
    ("FREON", "f"), ("OZONE HOLE", "p"), ("MONTREAL", "f"), ("HEAT TRAP", "p"), ("ISOBUTANE", "f")]

def chain_hud(d, n_done, t, active_label=True, alpha=255, ss=SS):
    """n_done: float number of nodes revealed (0..10). Drawn in top band y~205."""
    if n_done <= 0: return
    y = 214
    x0, x1 = 92, 892
    n = len(CHAIN)
    xs = [lerp(x0, x1, i / (n - 1)) for i in range(n)]
    k = int(min(math.ceil(n_done), n))
    # links
    for i in range(1, k):
        frac = min(max(n_done - i, 0), 1)
        c = RED if CHAIN[i][1] == "p" else GREEN
        xa, xb = xs[i - 1], lerp(xs[i - 1], xs[i], frac)
        d.line([xa * ss, y * ss, xb * ss, y * ss], fill=(120, 130, 150, int(alpha * 0.8)), width=int(4 * ss))
    for i in range(k):
        frac = min(max(n_done - i, 0), 1)
        c = RED if CHAIN[i][1] == "p" else GREEN
        r = 17 * ease_out(frac * 1.5)
        last = (i == k - 1)
        if last:
            pr = 17 + 9 * (0.5 + 0.5 * math.sin(t * 6))
            d.ellipse([(xs[i] - pr) * ss, (y - pr) * ss, (xs[i] + pr) * ss, (y + pr) * ss],
                      outline=c + (int(alpha * 0.55),), width=int(3 * ss))
        d.ellipse([(xs[i] - r) * ss, (y - r) * ss, (xs[i] + r) * ss, (y + r) * ss], fill=c + (alpha,))
        mark = "!" if CHAIN[i][1] == "p" else "✓"
        if r > 8:
            f = font(FONT_XB if mark == "!" else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(18 * ss))
            d.text((xs[i] * ss, (y + 1) * ss), mark, font=f, fill=(10, 12, 20, alpha), anchor="mm")
    if active_label and k > 0:
        i = k - 1; c = RED if CHAIN[i][1] == "p" else GREEN
        tag = ("PROBLEM " if CHAIN[i][1] == "p" else "FIX ") + str(i // 2 + 1)
        xl = min(max(xs[i], 230), 760)
        f = font(FONT_SB, int(27 * ss))
        d.text((xl * ss, (y + 44) * ss), tag + " · " + CHAIN[i][0], font=f, fill=c + (alpha,), anchor="mm")

# ---------------- shaded sphere sprites (molecules) ----------------
@functools.lru_cache(None)
def ball(r, col):
    s = int(2 * r + 4)
    y, x = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = s / 2
    dx = (x - cx) / r; dy = (y - cy) / r
    rr = dx * dx + dy * dy
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    nd = np.clip(dx * L[0] + dy * L[1] + z * L[2], 0, 1)
    spec = np.clip(dx * 0.0 + dy * 0 + 1, 0, 1)
    hx, hy = -0.38, -0.42
    sp = np.exp(-(((dx - hx) ** 2 + (dy - hy) ** 2) / 0.035)) * 0.85
    c = np.array(col, np.float32)
    rgb = c * (0.22 + 0.85 * nd)[..., None] + 255 * sp[..., None]
    rim = np.clip((rr - 0.75) / 0.25, 0, 1) * (rr <= 1)
    rgb += c * 0.35 * rim[..., None]
    alpha = np.clip((1 - np.sqrt(rr)) * r, 0, 1) * 255
    out = np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")

def paste_ball(layer, x, y, r, col, ss=SS):
    b = ball(max(int(r * ss), 2), col)
    layer.alpha_composite(b, (int(x * ss - b.size[0] / 2), int(y * ss - b.size[1] / 2)))

ATOM = {"C": ((70, 74, 82), 26), "Cl": ((80, 230, 110), 30), "F": ((200, 240, 120), 22),
        "H": ((235, 238, 245), 16), "O": ((255, 70, 70), 24), "N": ((90, 120, 255), 25), "S": ((255, 220, 60), 30)}

def rot3(v, ax, ay, az=0):
    ca, sa = math.cos(ax), math.sin(ax); cb, sb = math.cos(ay), math.sin(ay); cc, sc = math.cos(az), math.sin(az)
    x, y, z = v
    y, z = y * ca - z * sa, y * sa + z * ca
    x, z = x * cb + z * sb, -x * sb + z * cb
    x, y = x * cc - y * sc, x * sc + y * cc
    return x, y, z

def molecule(layer, atoms, bonds, cx, cy, scale, ax, ay, az=0, alpha=1.0, ss=SS, labels=False, d=None):
    """atoms: list of (elem,(x,y,z)); bonds: list of (i,j). scale px per unit."""
    P = []
    for el, p in atoms:
        x, y, z = rot3(p, ax, ay, az)
        P.append((el, cx + x * scale, cy + y * scale, z))
    dd = ImageDraw.Draw(layer)
    for i, j in bonds:
        _, x1, y1, _ = P[i]; _, x2, y2, _ = P[j]
        dd.line([x1 * ss, y1 * ss, x2 * ss, y2 * ss], fill=(200, 205, 215, int(220 * alpha)), width=int(scale * 0.16 * ss))
    for el, x, y, z in sorted(P, key=lambda q: q[3]):
        col, r = ATOM[el]
        rr = r * scale / 60 * (1 + 0.12 * z)
        paste_ball(layer, x, y, rr, col)
    if labels and d is not None:
        for el, x, y, z in P:
            draw_text(d, (x, y), el, 22 * scale / 60, fill=(10, 10, 15, int(255 * alpha)), path=FONT_XB)
    return P

# ---------------- Earth renderer (numpy, 1x) ----------------
_LAND = None
def land_grid():
    global _LAND
    if _LAND is None:
        p = os.path.join(HERE, "land_0p25.npy")
        if os.path.exists(p):
            _LAND = np.load(p)
        else:
            from global_land_mask import globe
            lats = np.linspace(-89.875, 89.875, 720); lons = np.linspace(-179.875, 179.875, 1440)
            LO, LA = np.meshgrid(lons, lats)
            _LAND = globe.is_land(LA, LO).astype(np.float32)
            # Antarctica ice: everything below -78 is land
            np.save(p, _LAND)
    return _LAND

_TEX = None
def fbm(h, w, seed, octaves=6, base=4, persist=0.55):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        gh, gw = base * 2 ** o // 2 + 2, base * 2 ** o + 2
        g = rng.random((gh, gw)).astype(np.float32)
        g[:, -1] = g[:, 0]
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += np.asarray(im).astype(np.float32) / 255 * amp; tot += amp; amp *= persist
    return acc / tot
def textures():
    global _TEX
    if _TEX is None:
        p = os.path.join(HERE, "tex.npz")
        if os.path.exists(p):
            z = np.load(p); _TEX = (z["c"], z["v"])
        else:
            c = fbm(720, 1440, 7, 7, 6); v = fbm(720, 1440, 11, 6, 8)
            def per(a):
                k = 120; w = np.linspace(0, 1, k)[None, :]
                a2 = a.copy(); a2[:, -k:] = a[:, -k:] * (1 - w) + a[:, :k] * w
                return a2

            np.savez(p, c=c, v=v); _TEX = (c, v)
    return _TEX

def earth(D, view_lat, lon0, hole, t, healthy_shell=1.0, sun=(-0.40, 0.15, 0.90)):
    """returns RGBA float array (D x D). hole 0..1 strength of ozone hole."""
    land = land_grid()
    y, x = np.mgrid[0:D, 0:D].astype(np.float32)
    R = D / 2 * 0.80
    cx = cy = D / 2
    nx = (x - cx) / R; ny = (y - cy) / R
    rr = nx * nx + ny * ny
    inside = rr <= 1
    nz = np.sqrt(np.clip(1 - rr, 0, 1))
    # view coords: X right, Y up, Z toward viewer
    X, Y, Z = nx, -ny, nz
    c, s = math.cos(view_lat), math.sin(view_lat)
    # rotate about X so that point (0,0,1) maps to latitude view_lat
    Yw = Y * c + Z * s; Zw = -Y * s + Z * c
    lat = np.degrees(np.arcsin(np.clip(Yw, -1, 1)))
    lon = np.degrees(np.arctan2(X, Zw)) + lon0
    lon = (lon + 180) % 360 - 180
    li = np.clip(((lat + 90) / 180 * 720).astype(int), 0, 719)
    lj = np.clip(((lon + 180) / 360 * 1440).astype(int), 0, 1439)
    L = land[li, lj]
    # colours
    ocean = np.array([12, 42, 92], np.float32)
    ocean2 = np.array([20, 70, 130], np.float32)
    landc = np.array([58, 92, 52], np.float32)
    desert = np.array([150, 125, 80], np.float32)
    ice = np.array([196, 212, 230], np.float32)
    lt = np.abs(lat)
    dry = np.clip(1 - np.abs(lt - 24) / 10, 0, 1)
    col = ocean[None, None] * (1 - nz[..., None] * 0) + 0
    col = ocean + (ocean2 - ocean) * (nz[..., None] ** 2)
    ctex, vtex = textures()
    vv = vtex[li, lj]
    lc = landc + (desert - landc) * np.clip(dry + (vv - 0.5) * 0.9, 0, 1)[..., None]
    lc = lc * (0.75 + 0.5 * vv[..., None])
    col = col * (1 - L[..., None]) + lc * L[..., None]
    icem = np.clip((lt - 62) / 6, 0, 1) * L + np.clip((lt - 82) / 4, 0, 1)
    icem = np.clip(icem, 0, 1)
    icec = ice * (0.80 + 0.2 * nz[..., None])
    col = col * (1 - icem[..., None]) + icec * icem[..., None]
    # clouds (procedural bands)
    lj2 = np.clip(((lon + 180 + t * 1.5) % 360 / 360 * 1440).astype(int), 0, 1439)
    cl = ctex[li, lj2]
    cl = np.clip((cl - 0.52) * 3.2, 0, 0.85) * np.clip((70 - lt) / 12, 0, 1)
    col = col * (1 - cl[..., None]) + 235 * cl[..., None]
    # lighting
    Ls = np.array(sun); Ls = Ls / np.linalg.norm(Ls)
    nd = X * Ls[0] + Y * (-Ls[1]) + Z * Ls[2]
    lit = np.clip(nd * 0.95 + 0.12, 0.06, 0.95)
    col = col * lit[..., None]
    # ocean specular
    sp = np.clip(nd, 0, 1) ** 40 * (1 - L) * (1 - icem) * 90
    col += sp[..., None]
    haze = (1 - nz) ** 3
    col = col * (1 - 0.5 * haze[..., None]) + np.array([90, 160, 255], np.float32) * 0.5 * haze[..., None] * np.clip(lit[..., None] * 1.5, 0.2, 1)
    # ozone hole overlay (south pole) — wavy vortex edge
    edge = -60 + 5 * np.sin(np.radians(lon) * 3 + t * 0.6) + 2.5 * np.sin(np.radians(lon) * 7 - t * 0.9)
    hm = np.clip((edge - lat) / 3.0, 0, 1) * hole
    holec = np.array([105, 40, 210], np.float32)
    col = col * (1 - 0.55 * hm[..., None]) + holec * 0.55 * hm[..., None] * np.clip(lit[..., None] + 0.35, 0, 1)
    a = np.clip((1 - np.sqrt(rr)) * R, 0, 1) * inside
    out = np.zeros((D, D, 4), np.float32)
    out[..., :3] = col; out[..., 3] = a * 255
    # atmosphere + ozone shell glow (outside rim)
    r = np.sqrt(rr)
    shell = np.exp(-((r - 1.0) / 0.035) ** 2) + 0.5 * np.exp(-np.clip(r - 1, 0, None) / 0.06) * (r > 1)
    # shell thinner where hole (bottom part of disc ~ south)
    southness = np.clip(-(lat) / 90, 0, 1)
    rim_lat_proxy = np.clip((ny - 0.2) / 0.8, 0, 1)  # lower rim faces south when viewing from below
    sc = np.array([80, 200, 255], np.float32) * healthy_shell
    shell_col = sc * shell[..., None]
    hole_rim = rim_lat_proxy[..., None] * hole
    shell_col = shell_col * (1 - hole_rim) + np.array([190, 60, 230], np.float32) * shell[..., None] * hole_rim * 0.9
    out[..., :3] = out[..., :3] * (out[..., 3:4] / 255) + shell_col * 0.9
    out[..., 3] = np.clip(out[..., 3] + shell * 255 * 0.9, 0, 255)
    out[..., :3] = np.where(out[..., 3:4] > 0, out[..., :3] / np.maximum(out[..., 3:4] / 255, 1e-3), 0)
    return out

def paste_float(base, rgba, x0, y0, alpha=1.0):
    """composite float RGBA (premult false) onto float RGB base at top-left x0,y0 (clipped)."""
    h, w = rgba.shape[:2]
    X0, Y0 = max(x0, 0), max(y0, 0); X1, Y1 = min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return base
    sub = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = sub[..., 3:4] / 255.0 * alpha
    base[Y0:Y1, X0:X1] = base[Y0:Y1, X0:X1] * (1 - a) + sub[..., :3] * a
    return base

def vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.7)) ** 2)
    return np.clip(1.15 - 0.45 * r ** 2, 0.55, 1.0)[..., None]
VIG = None
def finish(arr):
    global VIG
    if VIG is None: VIG = vignette()
    return to_img(arr * VIG)
