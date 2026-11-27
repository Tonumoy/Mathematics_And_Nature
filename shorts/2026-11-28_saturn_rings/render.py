# SOURCE CODE #16 - Why are Saturn's rings so flat? (Coded Cosmos)
# Own render: ray-traced oblate Saturn + ring plane with real ring radii (C/B/A, Cassini division, Encke gap),
# ring shadow on the planet, planet shadow on the rings; real collisional N-body ring model (sim.py).
import sys, math, json, os, numpy as np, cairo, cv2

W, H = 1080, 1920
SS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
FPS = 30
CYAN = (0.31, 0.82, 1.0); AMBER = (1.0, 0.71, 0.28); RED = (1.0, 0.32, 0.30)
GREEN = (0.35, 1.0, 0.55); WHITE = (0.95, 0.96, 1.0); GREY = (0.62, 0.68, 0.78)
YEL = (1.0, 0.86, 0.30)
F_CAP = "Poppins"; F_MONO = "DejaVu Sans Mono"; F_MATH = "Latin Modern Roman"

# ---------------- timeline (voice clip durations) ----------------
EST = [4.4, 3.4, 2.6, 2.6, 3.0, 1.5, 3.4, 2.4, 3.6, 3.6, 3.2]
DURS = json.load(open("vo/durs.json")) if os.path.exists("vo/durs.json") else EST
GAP = [0.25, 0.3, 0.25, 0.2, 0.1, 0.25, 0.4, 0.2, 0.35, 0.3, 0.0]
L = []; t0 = 0.0
for d, g in zip(DURS, GAP):
    L.append(t0); t0 += d + g
DUR = round(t0 + 1.0, 2)
def Ls(i): return L[i - 1]
def Le(i): return L[i - 1] + DURS[i - 1]

def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
def eout(u): u = min(1, max(0, u)); return 1 - (1 - u) ** 3
def win(t, a, b, fi=0.3, fo=0.3):
    if t < a or t > b: return 0.0
    return min(1, (t - a) / fi if fi > 0 else 1, (b - t) / fo if fo > 0 else 1)
def lerp(a, b, u): return a + (b - a) * u
def mix(c1, c2, u): return tuple(lerp(a, b, u) for a, b in zip(c1, c2))

# ---------------- cairo text helpers ----------------
def text(cx, s, x, y, size, col, a=1.0, face=F_CAP, bold=True, italic=False, anchor="l"):
    cx.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cx.set_font_size(size); e = cx.text_extents(s)
    if anchor == "c": x = x - e.x_advance / 2
    elif anchor == "r": x = x - e.x_advance
    cx.move_to(x, y); cx.set_source_rgba(*col, a); cx.show_text(s)
    return e.x_advance
def text_w(cx, s, size, face=F_CAP, bold=True):
    cx.select_font_face(face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cx.set_font_size(size); return cx.text_extents(s).x_advance
def rrect(cx, x, y, w, h, r):
    cx.new_sub_path()
    cx.arc(x + w - r, y + r, r, -math.pi / 2, 0); cx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cx.arc(x + r, y + h - r, r, math.pi / 2, math.pi); cx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    cx.close_path()
def ring2d(cx, x, y, r, col, a, lw=3):
    cx.new_path(); cx.set_line_width(lw); cx.set_source_rgba(*col, a); cx.arc(x, y, r, 0, 2 * math.pi); cx.stroke()
def pill(cx, s, x, y, size, col, a, anchor="c", fill=(0.02, 0.03, 0.06)):
    w = text_w(cx, s, size, F_MONO, True)
    x0 = x - w / 2 if anchor == "c" else (x - w if anchor == "r" else x)
    cx.set_source_rgba(*fill, 0.80 * a); rrect(cx, x0 - 18, y - size - 8, w + 36, size + 24, (size + 24) / 2); cx.fill()
    cx.set_line_width(2); cx.set_source_rgba(*col, 0.9 * a); rrect(cx, x0 - 18, y - size - 8, w + 36, size + 24, (size + 24) / 2); cx.stroke()
    text(cx, s, x0, y, size, col, a, F_MONO, True)
def big_text(cx, parts, y, size, a, maxw=860, scrim=True):
    tw = sum(text_w(cx, s, size) for s, _ in parts)
    if tw > maxw: size = size * maxw / tw; tw = maxw
    x = 485 - tw / 2
    if scrim:
        cx.set_source_rgba(0.0, 0.01, 0.03, 0.55 * a); rrect(cx, x - 28, y - size - 6, tw + 56, size + 30, 16); cx.fill()
    for s, c in parts: x += text(cx, s, x, y, size, c, a)

# ---------------- camera ----------------
class Cam:
    def __init__(self, pos, target, fov_deg, up=(0, 0, 1)):
        self.pos = np.array(pos, float); f = np.array(target, float) - self.pos; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.foc = (H / 2) / math.tan(math.radians(fov_deg) / 2)   # in 1080x1920 units
    def project(self, P):
        d = P - self.pos; z = d @ self.f
        x = W / 2 + self.foc * (d @ self.r) / z; y = H / 2 - self.foc * (d @ self.u) / z
        return x, y, z
def orbit_cam(az, el, dist, fov, target=(0, 0, 0)):
    a, e = math.radians(az), math.radians(el)
    p = np.array(target) + dist * np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])
    return Cam(p, target, fov)

# ---------------- Saturn (ray traced) ----------------
SUN = np.array([-0.55, -0.62, 0.42]); SUN /= np.linalg.norm(SUN)
OBL = 0.902  # polar / equatorial radius
# ring radii in Saturn equatorial radii (60,268 km): C 74,658-92,000; B 92,000-117,580; Cassini to 122,170; A to 136,775; Encke 133,589
RR = np.linspace(1.0, 2.4, 8192)
def _ring_tables():
    rng = np.random.default_rng(3)
    op = np.zeros_like(RR); al = np.zeros_like(RR)
    def band(a, b, o, alb): m = (RR >= a) & (RR < b); op[m] = o; al[m] = alb
    band(1.239, 1.527, 0.12, 0.45); band(1.527, 1.951, 0.85, 1.0); band(1.951, 2.027, 0.08, 0.5)
    band(2.027, 2.270, 0.55, 0.85)
    fine = np.zeros_like(RR)
    for k in range(40):
        fr = rng.uniform(30, 320); ph = rng.uniform(0, 6.28); fine += rng.uniform(0.2, 1) * np.sin(fr * RR + ph)
    fine /= np.abs(fine).max()
    op = np.clip(op * (1 + 0.28 * fine), 0, 0.97)
    m = (RR > 2.213) & (RR < 2.219); op[m] = 0.02
    m = (RR > 1.24) & (RR < 1.40); op[m] *= np.linspace(0.6, 1.0, m.sum())
    op[(RR < 1.239) | (RR > 2.270)] = 0
    return op, al * (1 + 0.1 * fine)
RING_OP, RING_AL = _ring_tables()
RING_COL = np.array([1.05, 0.95, 0.80])
def ring_lookup(tab, r):
    i = np.clip(((r - 1.0) / 1.4 * 8191).astype(np.int32), 0, 8191); return tab[i]
LAT = np.linspace(-1, 1, 2048)
def _band_tab():
    rng = np.random.default_rng(5)
    base = np.array([0.86, 0.76, 0.58])
    c = np.tile(base, (2048, 1))
    s = np.zeros(2048)
    for k in range(25): s += rng.uniform(0.2, 1) * np.sin(rng.uniform(8, 70) * LAT + rng.uniform(0, 6.3))
    s /= np.abs(s).max()
    c = c * (1 + 0.10 * s[:, None])
    c[:, 2] *= 1 + 0.08 * s
    pole = np.abs(LAT) > 0.82; c[pole] = c[pole] * np.array([0.78, 0.86, 0.95])
    eq = np.abs(LAT) < 0.12; c[eq] *= 1.06
    return np.clip(c, 0, 1)
BANDS = _band_tab()

def stars_bg(sw, sh):
    rng = np.random.default_rng(11)
    img = np.zeros((sh, sw, 3), np.float32)
    n = int(2600 * (sw * sh) / (1080 * 1920))
    xs = rng.integers(0, sw, n); ys = rng.integers(0, sh, n); b = rng.uniform(0.05, 1, n) ** 3 * 255
    tint = rng.uniform(0.8, 1.0, (n, 3))
    img[ys, xs] = b[:, None] * tint
    img = cv2.GaussianBlur(img, (0, 0), 0.6 * SS) * 2.2
    g = np.linspace(0, 1, sh)[:, None]
    img += np.dstack([4 + 6 * g, 6 + 8 * g, 12 + 14 * g]).astype(np.float32)
    return img
_STARS = {}
def stars(sw, sh):
    if (sw, sh) not in _STARS: _STARS[(sw, sh)] = stars_bg(sw, sh)
    return _STARS[(sw, sh)]

def saturn(cam, sw, sh, R=1.0, rings=True, ring_gain=1.0, bg=None):
    s = sw / W
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    dx = (xs / s - W / 2) / cam.foc; dy = -(ys / s - H / 2) / cam.foc
    D = cam.f[None, None, :] + dx[..., None] * cam.r[None, None, :] + dy[..., None] * cam.u[None, None, :]
    D /= np.linalg.norm(D, axis=2, keepdims=True)
    O = cam.pos / R
    # oblate sphere: scale z
    sc = np.array([1, 1, 1 / OBL])
    Os, Ds = O * sc, D * sc
    a = (Ds ** 2).sum(2); b = 2 * (Ds @ Os); c = Os @ Os - 1
    disc = b * b - 4 * a * c
    hitp = disc > 0
    tp = np.where(hitp, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    hitp &= tp > 0
    out = (bg.copy() if bg is not None else np.zeros((sh, sw, 3), np.float32)) / 255.0
    # planet
    if hitp.any():
        P = O + D[hitp] * tp[hitp][:, None]
        N = P * np.array([1, 1, 1 / OBL ** 2]); N /= np.linalg.norm(N, axis=1, keepdims=True)
        lam = np.clip(N @ SUN, 0, 1)
        mu = np.clip(-(N * D[hitp]).sum(1), 0, 1)
        lat = np.clip(P[:, 2] / OBL, -1, 1)
        lon = np.arctan2(P[:, 1], P[:, 0])
        li = np.clip(((lat + 0.004 * np.sin(7 * lon + 25 * lat) + 1) / 2 * 2047).astype(np.int32), 0, 2047)
        col = BANDS[li]
        shade = lam ** 0.85 * (0.45 + 0.55 * mu ** 0.35)
        if rings:
            sr = -P[:, 2] / SUN[2]
            Q = P + sr[:, None] * SUN[None, :]
            rq = np.sqrt(Q[:, 0] ** 2 + Q[:, 1] ** 2)
            tr = np.where(sr > 0, 1 - 0.9 * ring_lookup(RING_OP, rq) * ring_gain, 1.0)
            shade = shade * tr
        pc = col * shade[:, None] + np.array([0.02, 0.025, 0.04]) * (1 - lam[:, None])
        rim = (1 - mu) ** 4 * lam
        pc += rim[:, None] * np.array([0.25, 0.22, 0.18])
        out[hitp] = pc
    if rings and ring_gain > 0:
        tr = -O[2] / np.where(np.abs(D[..., 2]) < 1e-9, 1e-9, D[..., 2])
        Pq = O[None, None, :] + D * tr[..., None]
        rr = np.sqrt(Pq[..., 0] ** 2 + Pq[..., 1] ** 2)
        hitr = (tr > 0) & (rr > 1.23) & (rr < 2.28) & (tr < tp)
        if hitr.any():
            Q = Pq[hitr]; r = rr[hitr]
            op = ring_lookup(RING_OP, r) * ring_gain; al = ring_lookup(RING_AL, r)
            # planet shadow on rings
            Qs = Q * sc; Ls_ = SUN * sc
            a2 = Ls_ @ Ls_; b2 = 2 * (Qs @ Ls_); c2 = (Qs ** 2).sum(1) - 1
            d2 = b2 * b2 - 4 * a2 * c2
            shad = (d2 > 0) & ((-b2 + np.sqrt(np.maximum(d2, 0))) > 0)
            lit = np.where(shad, 0.06, 1.0) * (0.55 + 0.6 * abs(SUN[2]))
            rc = RING_COL[None, :] * (al * lit)[:, None]
            out[hitr] = out[hitr] * (1 - op[:, None]) + rc * op[:, None]
    return out * 255.0

# ---------------- particles ----------------
_SIM = {}
def sim(nm):
    if nm not in _SIM:
        d = np.load(f'sim_{nm}.npz'); _SIM[nm] = (d['P'], d['hz'], d['F'])
    return _SIM[nm]
_JIT = np.random.default_rng(2).normal(0, 0.012, (6, 6000, 3)).astype(np.float32)
_SZ = np.random.default_rng(4).uniform(0.4, 1.0, 6000).astype(np.float32) ** 2
def splat(img, cam, P, col, gain=1.0, R_occ=0.0, size=1.0):
    sh, sw = img.shape[:2]; s = sw / W
    allP = (P[None] + _JIT[:, :len(P)]).reshape(-1, 3)
    wgt = np.tile(_SZ[:len(P)], 6)
    x, y, z = cam.project(allP)
    ok = z > 0.05
    if R_occ > 0:   # hide points behind the planet
        d = allP - cam.pos; dn = np.linalg.norm(d, axis=1); u = d / dn[:, None]
        tc = -(cam.pos @ u.T) if False else -(u @ cam.pos)
        closest = np.linalg.norm(cam.pos[None] + u * tc[:, None], axis=1)
        ok &= ~((closest < R_occ) & (tc > 0) & (tc < dn))
    xi = (x * s).astype(np.int32); yi = (y * s).astype(np.int32)
    ok &= (xi >= 0) & (xi < sw) & (yi >= 0) & (yi < sh)
    acc = np.zeros((sh, sw), np.float32)
    np.add.at(acc, (yi[ok], xi[ok]), wgt[ok] * 0.9)
    k = max(0.7, 1.1 * s * size)
    acc = cv2.GaussianBlur(acc, (0, 0), k) * (k * k * 6.0)
    img += np.clip(acc, 0, 3)[..., None] * np.array(col, np.float32)[None, None] * 255 * 0.55 * gain
def flashes(img, cam, F, a):
    sh, sw = img.shape[:2]; s = sw / W
    F = F[~np.isnan(F[:, 0])]
    if len(F) == 0: return
    d = F - cam.pos; dn = np.linalg.norm(d, axis=1); u_ = d / dn[:, None]; tc = -(u_ @ cam.pos)
    cl = np.linalg.norm(cam.pos[None] + u_ * tc[:, None], axis=1); F = F[~((cl < 0.72) & (tc > 0) & (tc < dn))]
    if len(F) == 0: return
    x, y, z = cam.project(F)
    acc = np.zeros((sh, sw), np.float32)
    xi = (x * s).astype(np.int32); yi = (y * s).astype(np.int32)
    ok = (z > 0) & (xi >= 0) & (xi < sw) & (yi >= 0) & (yi < sh)
    np.add.at(acc, (yi[ok], xi[ok]), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), 3 * s) * 120
    img += np.clip(acc, 0, 1.5)[..., None] * np.array(AMBER, np.float32) * 255 * a

# ---------------- camera paths ----------------
HOOK_D = float(os.environ.get('HOOK_D', '10.0'))
HOOK_EL = float(os.environ.get('HOOK_EL', '26'))
def hook_cam(t):
    # frame 1 = slightly above the ring plane, swinging to edge-on
    u = ease((t + 0.3) / 3.0)
    el = lerp(HOOK_EL, 0.0, u); az = lerp(-24, -14, ease(t / 4.0))
    return orbit_cam(az, el, HOOK_D, 40)
def end_cam(t):
    u = ease((t - Ls(10)) / max(0.1, DUR - Ls(10) - 0.6))
    return orbit_cam(lerp(-60, -24, u), lerp(38, HOOK_EL, u), lerp(15, HOOK_D, u), 40)
def disk_cam(t, el=20.0):
    return orbit_cam(-30 + 3.0 * t, el, 15.5, 36)

# ---------------- scene pieces ----------------
def sim_frame(t, a0, a1, f0, f1, power=1.0):
    u = min(1, max(0, (t - a0) / max(0.01, a1 - a0))) ** power
    return int(lerp(f0, f1, u))

def thickness_meter(cx, t, frac, a, col):
    # vertical bar = cloud thickness (sim), % of start
    x, y0, h = 880, 560, 520
    cx.set_source_rgba(0.02, 0.03, 0.06, 0.75 * a); rrect(cx, x - 34, y0 - 70, 96, h + 130, 16); cx.fill()
    text(cx, "THICK", x + 14, y0 - 30, 20, GREY, a, F_MONO, True, anchor="c")
    cx.set_source_rgba(*GREY, 0.35 * a); rrect(cx, x, y0, 28, h, 8); cx.fill()
    hh = h * min(1.3, frac) / 1.3
    cx.set_source_rgba(*col, 0.95 * a); rrect(cx, x, y0 + h - hh, 28, max(hh, 6), 8); cx.fill()
    text(cx, f"{int(round(frac * 100))}%", x + 14, y0 + h + 44, 26, col, a, F_MONO, True, anchor="c")

def dial(cx, t, e, a):
    # bounciness dial e in [0,1]
    cxp, cyp, R = 485, 1290, 110
    cx.set_source_rgba(0.02, 0.03, 0.06, 0.78 * a); rrect(cx, cxp - 230, cyp - 175, 460, 255, 26); cx.fill()
    text(cx, "BOUNCINESS  e", cxp, cyp - 128, 26, WHITE, a, F_MONO, True, anchor="c")
    a0, a1 = math.pi * 1.1, math.pi * 1.9
    cx.set_line_width(14); cx.set_source_rgba(*GREY, 0.35 * a); cx.arc(cxp, cyp, R, a0, a1); cx.stroke()
    col = mix(GREEN, RED, ease((e - 0.3) / 0.6))
    cx.set_source_rgba(*col, a); cx.arc(cxp, cyp, R, a0, a0 + (a1 - a0) * e); cx.stroke()
    ang = a0 + (a1 - a0) * e
    cx.set_line_width(6); cx.move_to(cxp, cyp); cx.line_to(cxp + (R - 10) * math.cos(ang), cyp + (R - 10) * math.sin(ang)); cx.stroke()
    text(cx, f"e = {e:.1f}", cxp, cyp + 52, 40, col, a, F_MONO, True, anchor="c")
    text(cx, "0", cxp + R * math.cos(a0) - 8, cyp + R * math.sin(a0) + 44, 22, GREY, a, F_MONO, True, anchor="c")
    text(cx, "1", cxp + R * math.cos(a1) + 8, cyp + R * math.sin(a1) + 44, 22, GREY, a, F_MONO, True, anchor="c")

def eq_card(cx, a):
    x, y, w, h = 150, 440, 670, 170
    cx.set_source_rgba(0.97, 0.94, 0.86, 0.95 * a); rrect(cx, x, y, w, h, 18); cx.fill()
    text(cx, "v′", x + 90, y + 82, 62, (0.08, 0.08, 0.12), a, F_MATH, False, True)
    cx.set_font_size(30); text(cx, "up-down", x + 70, y + 112, 22, (0.35, 0.35, 0.4), a, F_CAP, True)
    text(cx, "= − e · v", x + 175, y + 82, 62, (0.08, 0.08, 0.12), a, F_MATH, False, True)
    text(cx, "after a bump, up-down speed × e", x + w / 2, y + 148, 26, (0.15, 0.35, 0.6), a, F_CAP, True, anchor="c")

def bump_inset(cx, t, a, e=0.1):
    # two ice chunks collide: vertical arrows shrink by e, sideways (orbit) arrows stay
    x0, y0 = 190, 1230
    cx.set_source_rgba(0.02, 0.03, 0.06, 0.78 * a); rrect(cx, x0 - 120, y0 - 170, 330, 300, 22); cx.fill()
    u = (t % 1.6) / 1.6
    sep = 70 * abs(1 - 2 * min(u, 0.5) * 2) if u < 0.5 else 70 * (u - 0.5) * 2
    vz = 1.0 if u < 0.5 else e
    for sgn in (-1, 1):
        x = x0 + 45 + sgn * (25 + sep * 0.5); y = y0 - sgn * (sep * 0.6)
        g = cairo.RadialGradient(x - 8, y - 8, 2, x, y, 30)
        g.add_color_stop_rgba(0, 1, 1, 1, a); g.add_color_stop_rgba(1, 0.65, 0.72, 0.82, a)
        cx.set_source(g); cx.arc(x, y, 26, 0, 2 * math.pi); cx.fill()
        L_ = 70 * vz * (-sgn if u < 0.5 else sgn)
        cx.set_line_width(6); cx.set_source_rgba(*RED, a); cx.move_to(x, y); cx.line_to(x, y + L_); cx.stroke()
        cx.set_source_rgba(*GREEN, a); cx.move_to(x, y); cx.line_to(x + 60, y); cx.stroke()
    text(cx, "up-down", x0 - 100, y0 + 100, 22, RED, a, F_MONO, True)
    text(cx, "spin", x0 + 110, y0 + 100, 22, GREEN, a, F_MONO, True)

# ---------------- captions ----------------
def CAPS():
    return [
        (Ls(1) + 2.6, Le(1), [("only ", WHITE), ("10 m", YEL), (" thick", WHITE)]),
        (Ls(2), Le(2), [("thinner than ", WHITE), ("paper", YEL)]),
        (Ls(3), Le(3), [("gravity alone? ", WHITE), ("nope", RED)]),
        (Ls(4), Le(4), [("billions of ", WHITE), ("ice chunks", YEL)]),
        (Ls(5), Le(5), [("each bump kills ", WHITE), ("up-down", YEL), (" speed", WHITE)]),
        (Ls(6), Le(6), [("the ", WHITE), ("spin", GREEN), (" survives", WHITE)]),
        (Ls(7), Le(7), [("cloud flattens to a ", WHITE), ("sheet", YEL)]),
        (Ls(8), Le(8), [("make it ", WHITE), ("perfectly bouncy", YEL)]),
        (Ls(9), Le(9) - 1.6, [("stays ", WHITE), ("puffy", RED)]),
        (Le(9) - 1.6, Le(9), [("real ice? ", WHITE), ("flat", GREEN)]),
        (Ls(10), Le(10), [("galaxies flatten the ", WHITE), ("same way", YEL)]),
        (Ls(11), Le(11) + 0.6, [("next: why does the Moon ", WHITE), ("never turn", YEL), ("?", WHITE)]),
    ]
def captions(cx, t, y=1500):
    for a, b, parts in CAPS():
        al = win(t, a, b + 0.1, 0.15, 0.15)
        if al <= 0: continue
        size = 54
        tw = sum(text_w(cx, s, size) for s, _ in parts)
        if tw > 840: size = size * 840 / tw; tw = 840
        x = 485 - tw / 2
        cx.set_source_rgba(0.0, 0.01, 0.03, 0.62 * al); rrect(cx, x - 26, y - size - 10, tw + 52, size + 34, 14); cx.fill()
        pop = 1 + 0.06 * (1 - ease((t - a) / 0.18))
        cx.save(); cx.translate(485, y - size / 3); cx.scale(pop, pop); cx.translate(-485, -(y - size / 3))
        for s, c in parts: x += text(cx, s, x, y, size, c, al)
        cx.restore()

def hud(cx, t, readout):
    text(cx, "SOURCE CODE #16", 930, 205, 24, CYAN, 0.9 * ease((t - 0.3) / 0.4), F_MONO, True, anchor="r")
    if readout: text(cx, readout, 60, 1585, 21, GREY, 0.85, F_MONO, False)

# ---------------- frame ----------------
def frame(t):
    sw, sh = int(W * SS), int(H * SS)
    img = stars(sw, sh).copy()
    ov = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw, sh)
    cx = cairo.Context(ov); cx.scale(SS, SS)
    readout = ""
    t_x = Ls(4) - 0.1           # x-ray moment
    # ---------- A: Saturn hook + zoom (0 .. x-ray) ----------
    if t < t_x + 0.6:
        cam = hook_cam(t)
        zt = Ls(2) - 0.1
        zu = ease((t - zt) / (Ls(3) - zt + 0.4))   # zoom into the ring edge
        if zu > 0:
            fov = 40 * (1e-4) ** zu
            tgt = np.array([lerp(0, 2.15, ease(zu * 3)), 0, 0])
            cam = Cam(cam.pos, tgt, fov)
        img = saturn(cam, sw, sh, 1.0, True, 1.0, img)
        lu = ease((1.5 - (HOOK_EL * (1 - ease((t + 0.3) / 3.0)))) / 1.5) if t > 1 else 0
        if zu > 0 or lu > 0:   # edge-on ring line stays a hair at every zoom
            x, y, _ = cam.project(np.array([[0.0 if zu == 0 else 2.15, 0, 0]]))
            yy = int(y[0] * SS); thick = max(1, int(1.2 * SS))
            line = np.zeros((sh, sw), np.float32); line[max(0, yy - thick):yy + thick, :] = 1.0
            line = cv2.GaussianBlur(line, (0, 0), 1.2 * SS) + cv2.GaussianBlur(line, (0, 0), 10 * SS) * 0.6
            img += line[..., None] * np.array([235, 215, 180], np.float32) * max(lu, ease(zu * 4))
        # hook text: one layer at a time
        a1 = win(t, -1, Ls(1) + 2.4, 0.01, 0.25)
        if a1 > 0: big_text(cx, [("WIDER THAN ", WHITE), ("20 EARTHS", YEL)], 330, 74, a1)
        # 270,000 km bracket
        ab = win(t, 0.5, Ls(1) + 2.3, 0.3, 0.3)
        if ab > 0 and zu == 0:
            xa, ya, _ = cam.project(np.array([[-2.27, 0, 0], [2.27, 0, 0]]))
            yb = max(ya) + 150
            cx.set_line_width(3); cx.set_source_rgba(*AMBER, ab)
            xl, xr = max(45, xa[0]), min(1035, xa[1])
            cx.move_to(xl, yb - 18); cx.line_to(xl, yb); cx.line_to(xr, yb); cx.line_to(xr, yb - 18); cx.stroke()
            pill(cx, "270,000 km", 485, yb + 52, 30, AMBER, ab)
        # cyan ring on the ring edge at edge-on
        ar = win(t, Ls(1) + 2.3, Ls(2) + 0.2, 0.3, 0.3)
        if ar > 0:
            x, y, _ = cam.project(np.array([[2.15, 0, 0]]))
            ring2d(cx, x[0], y[0], 70, CYAN, ar)
        if zu > 0:
            zoom = 1 / (1e-4) ** zu
            pill(cx, f"ZOOM ×{zoom:,.0f}", 90, 300, 30, CYAN, win(t, zt, t_x, 0.2, 0.2), anchor="l")
            ap = win(t, Ls(2) + 0.6, Ls(3) - 0.1, 0.3, 0.3)
            if ap > 0:
                pill(cx, "100 m wide → 0.004 mm thick", 485, 1250, 28, WHITE, ap)
                pill(cx, "paper ≈ 0.1 mm", 485, 1330, 28, YEL, ap)
            readout = "ring line: still razor-thin at every zoom"
        else:
            readout = "ray-traced Saturn · real ring radii (C, B, A, Cassini gap)"
        # guess -> nope
        ag = win(t, Ls(3), t_x + 0.3, 0.25, 0.25)
        if ag > 0:
            pill(cx, "GUESS: gravity squashed it?", 485, 760, 32, AMBER, ag)
            an = ease((t - (Ls(3) + 1.2)) / 0.25) * ag
            if an > 0:
                text(cx, "NOPE", 485, 880, 84, RED, an, F_CAP, True, anchor="c")
                cx.set_line_width(10); cx.set_source_rgba(*RED, an); cx.move_to(300, 800); cx.line_to(670, 900); cx.stroke()
        # x-ray scanline
        if t > t_x:
            u = ease((t - t_x) / 0.6)
            ysc = int(sh * u)
            img[:ysc] *= 0.0
    # ---------- B: particle model (x-ray .. end of twist) ----------
    if t_x <= t < Ls(10) + 0.4:
        ax = ease((t - t_x) / 0.6)
        el = lerp(32, 12, ease((t - Ls(5)) / (Le(7) - Ls(5)))) if t < Ls(8) else 18
        cam = disk_cam(t, el)
        pimg = saturn(cam, sw, sh, 0.72, False, 0.0, np.zeros((sh, sw, 3), np.float32)) * 0.9
        PA, hzA, FA = sim('A'); PB, hzB, FB = sim('B')
        # which sim frame
        if t < Ls(5):            # x-ray look at the flat ring = last frame of A
            P = PA[-1]; f = len(PA) - 1; nm = 'A'
        elif t < Ls(8):          # rewind: puffy cloud collapses (A)
            f = sim_frame(t, Ls(5), Le(7), 0, len(PA) - 1, 0.8); P = PA[f]; nm = 'A'
        elif t < Le(9) - 1.6:    # twist: bouncy (B)
            f = sim_frame(t, Ls(8) + 0.5, Le(9) - 1.6, 0, len(PB) - 1); P = PB[f]; nm = 'B'
        else:                    # back to real ice (C continues from B)
            PC, hzC, FC = sim('C')
            f = sim_frame(t, Le(9) - 1.6, Le(9) + 0.4, 0, len(PC) - 1, 0.7); P = PC[f]; nm = 'C'
        xray = 1 - ease((t - (Ls(5) - 0.2)) / 0.5)
        col = mix((0.95, 0.90, 0.80), CYAN, xray)
        if nm == 'B': col = mix(col, (1.0, 0.55, 0.5), ease((t - Ls(8) - 0.6) / 0.8))
        splat(pimg, cam, P, col, 1.0, 0.72)
        if Ls(5) <= t < Ls(10):
            Fs = {'A': FA, 'B': FB, 'C': sim('C')[2]}[nm][f]
            flashes(pimg, cam, Fs, 0.9)
        if t < t_x + 0.6:
            ysc = int(sh * ease((t - t_x) / 0.6))
            img[:ysc] = pimg[:ysc]
            img[max(0, ysc - int(6 * SS)):ysc + int(6 * SS)] += np.array([80, 210, 255], np.float32)
        else:
            img = stars(sw, sh) * 0.6 + pimg
        a_lab = win(t, t_x + 0.5, Ls(5) - 0.1, 0.3, 0.3)
        if a_lab > 0:
            pill(cx, "X-RAY: ice chunks, sand grain → house", 485, 400, 26, CYAN, a_lab)
        if Ls(5) <= t < Ls(8):
            hz = (hzA[f]) / hzA[0]
            thickness_meter(cx, t, hz, win(t, Ls(5), Ls(8), 0.3, 0.3), mix(RED, GREEN, ease(1 - hz)))
            bump_inset(cx, t, win(t, Ls(5), Le(6) + 0.3, 0.3, 0.3))
            pill(cx, "REWIND: a puffy cloud", 485, 400, 30, AMBER, win(t, Ls(5), Ls(6), 0.3, 0.3))
            asp = win(t, Ls(6), Le(7), 0.3, 0.3)
            if asp > 0:   # green spin arrow around the planet
                x, y, _ = cam.project(np.array([[0, 0, 0.0]]))
                cx.new_path(); cx.set_line_width(5); cx.set_source_rgba(*GREEN, 0.85 * asp)
                cx.save(); cx.translate(x[0], y[0]); cx.scale(1, 0.32)
                cx.arc(0, 0, 440, 0.2, 2.6); cx.restore(); cx.stroke()
                pill(cx, "spin kept ✓", 485, 400, 30, GREEN, asp)
            if t > Le(7) - 1.0:
                pill(cx, "FLAT ✓", 485, 1300, 40, GREEN, win(t, Le(7) - 1.0, Ls(8), 0.2, 0.2))
            readout = f"{len(P):,} ice chunks · real bumps · e = 0.1"
        if Ls(8) <= t < Ls(10):
            et = Ls(8) + 0.3; eb = Le(9) - 1.6
            e = lerp(0.1, 1.0, ease((t - et) / 0.6)) if t < eb else lerp(1.0, 0.1, ease((t - eb) / 0.5))
            ad = win(t, Ls(8), Ls(10), 0.3, 0.3)
            dial(cx, t, e, ad)
            ae = win(t, Ls(9) - 0.2, Ls(9) + 2.8, 0.3, 0.3)
            if ae > 0: eq_card(cx, ae)
            hzv = {'A': hzA, 'B': hzB, 'C': sim('C')[1]}[nm][f] / hzA[0]
            thickness_meter(cx, t, hzv, ad, mix(RED, GREEN, ease(1 - hzv)))
            readout = f"same model · only e changed · e = {e:.1f}"
    # ---------- C: zoom-out to Saturn + loop ----------
    if t >= Ls(10) - 0.2:
        cam = end_cam(t)
        sat = saturn(cam, sw, sh, 1.0, True, 1.0, stars(sw, sh))
        u = ease((t - (Ls(10) - 0.2)) / 0.6)
        img = img * (1 - u) + sat * u
        ag = win(t, Ls(10) + 0.5, Le(10) + 0.2, 0.3, 0.3)
        if ag > 0:
            pill(cx, "galaxies", 300, 420, 30, CYAN, ag); pill(cx, "young solar systems", 640, 420, 30, CYAN, ag)
        an = win(t, Le(10) - 1.4, Ls(11) + 0.4, 0.3, 0.3)
        if an > 0: text(cx, "nature is running on code", 485, 1300, 40, WHITE, an, F_CAP, True, anchor="c")
        readout = "sources in the description"
    captions(cx, t)
    hud(cx, t, readout)
    # glow
    small = cv2.resize(img, (sw // 4, sh // 4), interpolation=cv2.INTER_AREA)
    bright = np.clip(small - 150, 0, 255)
    gl = cv2.GaussianBlur(bright, (0, 0), 5 * SS) * 0.9 + cv2.GaussianBlur(bright, (0, 0), 16 * SS) * 0.7
    img = img + cv2.resize(gl, (sw, sh), interpolation=cv2.INTER_LINEAR) * 0.6
    img = np.clip(img, 0, 255)
    o = np.ndarray((sh, sw, 4), np.uint8, ov.get_data()).astype(np.float32)
    al = o[..., 3:4] / 255.0
    out = img * (1 - al) + o[..., [2, 1, 0]]
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "video"
    if mode == "info":
        print(json.dumps({"DUR": DUR, "L": [round(x, 2) for x in L]}))
    elif mode == "stills":
        for tt in [float(v) for v in sys.argv[3].split(",")]:
            cv2.imwrite(f"still_{tt:05.2f}.png", cv2.cvtColor(frame(tt), cv2.COLOR_RGB2BGR))
    else:
        nf = int(DUR * FPS)
        a0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        a1 = int(sys.argv[4]) if len(sys.argv) > 4 else nf
        for i in range(a0, min(a1, nf)):
            sys.stdout.buffer.write(frame(i / FPS).tobytes())
