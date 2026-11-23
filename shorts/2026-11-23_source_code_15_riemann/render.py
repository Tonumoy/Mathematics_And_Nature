# SOURCE CODE #15 - The Riemann Hypothesis (Coded Cosmos)
# Own render: real zeta values (mpmath), real zeros, real explicit formula.
import sys, math, numpy as np, cairo, cv2

# ---------------- global parameters ----------------
W, H = 1080, 1920          # logical canvas
SS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # 2.0 -> 4K
FPS = 30
DUR = 44.0
BG = (0.016, 0.024, 0.043)
CYAN = (0.31, 0.82, 1.0)
AMBER = (1.0, 0.71, 0.28)
RED = (1.0, 0.30, 0.30)
GREEN = (0.35, 1.0, 0.55)
WHITE = (0.94, 0.96, 1.0)
GREY = (0.55, 0.62, 0.72)
YEL = (1.0, 0.85, 0.30)
F_CAP = "Poppins"
F_MONO = "DejaVu Sans Mono"
F_MATH = "Latin Modern Roman"
GLOW = 0.85

D = np.load("zeta_data.npz")
ZEROS = D["zeros"]; S = D["S"]; T = D["T"]; AZ = D["absz"]

# ---------------- maths: explicit formula ----------------
XS = np.linspace(1.5, 50, 1400)
LOGX = np.log(XS)
def psi_exact(x):
    out = np.zeros_like(x)
    for n in range(2, 51):
        m = n; p = None
        for q in range(2, n + 1):
            if m % q == 0:
                p = q; break
        while m % p == 0: m //= p
        if m == 1:   # n is a prime power p^k
            out += (x >= n) * math.log(p)
    return out
PSI = psi_exact(XS)
BASE = XS - math.log(2 * math.pi) - 0.5 * np.log(1 - XS ** -2.0)
RHO = 0.5 + 1j * ZEROS
TERMS = np.array([-2 * np.real(np.exp(r * LOGX) / r) for r in RHO])   # (600, len(XS))
CUM = np.cumsum(TERMS, axis=0)
def psi_N(N):
    if N <= 0: return BASE.copy()
    return BASE + CUM[min(N, len(ZEROS)) - 1]
def psi_moved(beta):
    # zero #1 and its mirror moved to beta, 1-beta (functional equation keeps them paired)
    g = ZEROS[0]
    out = psi_N(600) - TERMS[0]
    for b in (beta, 1 - beta):
        r = b + 1j * g
        out += -np.real(np.exp(r * LOGX) / r)   # half weight each: pair replaces the one on-line conjugate pair
    return out

# ---------------- helpers ----------------
def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
def eout(u): u = min(1, max(0, u)); return 1 - (1 - u) ** 3
def win(t, a, b, fi=0.3, fo=0.3):
    if t < a or t > b: return 0.0
    return min(1, (t - a) / fi if fi > 0 else 1, (b - t) / fo if fo > 0 else 1)
def lerp(a, b, u): return a + (b - a) * u
def mix(c1, c2, u): return tuple(lerp(a, b, u) for a, b in zip(c1, c2))

def text(cx, s, x, y, size, col, a=1.0, face=F_CAP, bold=True, italic=False, anchor="l"):
    cx.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cx.set_font_size(size)
    e = cx.text_extents(s)
    if anchor == "c": x = x - e.x_advance / 2
    elif anchor == "r": x = x - e.x_advance
    cx.move_to(x, y); cx.set_source_rgba(*col, a); cx.show_text(s)
    return e.x_advance

def text_w(cx, s, size, face=F_CAP, bold=True, italic=False):
    cx.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cx.set_font_size(size); return cx.text_extents(s).x_advance

def rrect(cx, x, y, w, h, r):
    cx.new_sub_path()
    cx.arc(x + w - r, y + r, r, -math.pi / 2, 0); cx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cx.arc(x + r, y + h - r, r, math.pi / 2, math.pi); cx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    cx.close_path()

def ring(cx, x, y, r, col, a, lw=2.5):
    cx.set_line_width(lw); cx.set_source_rgba(*col, a); cx.arc(x, y, r, 0, 2 * math.pi); cx.stroke()

def check(cx, x, y, size, col, a, lw=None):
    cx.set_line_width(lw or size * 0.16); cx.set_line_cap(cairo.LINE_CAP_ROUND); cx.set_line_join(cairo.LINE_JOIN_ROUND)
    cx.set_source_rgba(*col, a)
    cx.move_to(x, y - size * 0.35); cx.line_to(x + size * 0.32, y); cx.line_to(x + size * 0.9, y - size * 0.75); cx.stroke()
    cx.set_line_cap(cairo.LINE_CAP_BUTT)

def dot(cx, x, y, r, col, a=1.0):
    cx.set_source_rgba(*col, a); cx.arc(x, y, r, 0, 2 * math.pi); cx.fill()

def poly(cx, pts, col, a, lw):
    if len(pts) < 2: return
    cx.set_line_width(lw); cx.set_source_rgba(*col, a)
    cx.move_to(*pts[0])
    for p in pts[1:]: cx.line_to(*p)
    cx.stroke()

# ---------------- background + HUD ----------------
def background(cx, t):
    cx.set_source_rgb(*BG); cx.paint()
    g = cairo.RadialGradient(W / 2, H * 0.42, 50, W / 2, H * 0.42, 1100)
    g.add_color_stop_rgba(0, 0.05, 0.09, 0.16, 0.9); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    cx.set_source(g); cx.paint()
    cx.set_line_width(1)
    for i, step in enumerate((40, 200)):
        cx.set_source_rgba(0.45, 0.6, 0.8, 0.035 if i == 0 else 0.07)
        for x in range(0, W + 1, step): cx.move_to(x, 0); cx.line_to(x, H)
        for y in range(0, H + 1, step): cx.move_to(0, y); cx.line_to(W, y)
        cx.stroke()

def hud(cx, t, readout):
    text(cx, "SOURCE CODE #15", 930, 205, 22, CYAN, 0.85, F_MONO, True, anchor="r")
    text(cx, "ζ-LAB // REAL DATA", 930, 233, 18, GREY, 0.7, F_MONO, False, anchor="r")
    # corner brackets of the instrument frame
    cx.set_line_width(2); cx.set_source_rgba(*CYAN, 0.35)
    for (x, y, dx, dy) in ((48, 168, 1, 1), (1032, 168, -1, 1), (48, 1580, 1, -1), (1032, 1580, -1, -1)):
        cx.move_to(x, y + 40 * dy); cx.line_to(x, y); cx.line_to(x + 40 * dx, y); cx.stroke()
    if readout:
        text(cx, readout, 70, 1562, 20, GREY, 0.85, F_MONO, False)

def sticker(cx, t):
    a = ease((t - 6.0) / 0.4)
    if a <= 0: return
    w = text_w(cx, "RIEMANN HYPOTHESIS", 30)
    cx.set_source_rgba(0.86, 0.14, 0.16, 0.95 * a); rrect(cx, 60, 178, w + 32, 50, 8); cx.fill()
    text(cx, "RIEMANN HYPOTHESIS", 76, 214, 30, WHITE, a)
    text(cx, "the 167-year prime puzzle", 62, 262, 24, WHITE, 0.8 * a, F_CAP, False)

# ---------------- captions ----------------
CAPS = [
    (6.15, 7.6, [("Primes look ", WHITE), ("random", YEL)]),
    (11.0, 12.0, [("no ", WHITE), ("pattern?", YEL)]),
    (12.3, 15.8, [("1859: Riemann heard ", WHITE), ("hidden music", YEL)]),
    (16.0, 19.25, [("every zero = a ", WHITE), ("wave", YEL)]),
    (19.4, 21.3, [("add the waves…", WHITE)]),
    (21.3, 23.05, [("…the ", WHITE), ("primes", YEL), (" appear", WHITE)]),
    (23.2, 26.45, [("his bet: every zero on ", WHITE), ("one line", YEL)]),
    (26.6, 29.0, [("push one off… it ", WHITE), ("breaks", (1, 0.45, 0.45))]),
    (29.2, 35.3, [("20 Nov: a ", WHITE), ("proof", YEL), (" is released", WHITE)]),
    (35.5, 37.5, [("17 million", YEL), (" lines of code", WHITE)]),
    (37.6, 40.35, [("every step ", WHITE), ("machine-checked", YEL)]),
    (40.45, 43.6, [("now humans must understand ", WHITE), ("why", YEL)]),
]
def captions(cx, t, y=1455):
    for a, b, parts in CAPS:
        al = win(t, a, b, 0.15, 0.15)
        if al <= 0: continue
        size = 52
        tw = sum(text_w(cx, s, size) for s, _ in parts)
        if tw > 900: size = size * 900 / tw; tw = 900
        x = (W - tw) / 2
        cx.set_source_rgba(0.01, 0.02, 0.04, 0.62 * al); rrect(cx, x - 26, y - size - 10, tw + 52, size + 34, 14); cx.fill()
        pop = 1 + 0.06 * (1 - ease((t - a) / 0.18))
        cx.save(); cx.translate(W / 2, y - size / 3); cx.scale(pop, pop); cx.translate(-W / 2, -(y - size / 3))
        for s, c in parts:
            x += text(cx, s, x, y, size, c, al)
        cx.restore()

# ---------------- scene: 3D zeta landscape ----------------
SX = 9.0
_F = np.load("zeta_fine.npz"); S = _F["S"]; T = _F["T"]; AZ = _F["absz"]   # fine grid: sigma step 0.025, t step 0.033
_V = np.maximum(1.0 / np.maximum(AZ, 1e-6) - 1.0, 0) ** 1.15 * 0.62
HT = 6.5 * np.tanh(_V / 6.5)    # 1/|zeta|: every zero becomes a spike      # height field (T x S)
def project(Xw, Yw, Zw, cam):
    cxp, cyp, czp, pitch, f, ox, oy = cam
    x = Xw - cxp; y = Yw - cyp; z = Zw - czp
    cp, sp = math.cos(pitch), math.sin(pitch)
    y2 = y * cp + z * sp
    z2 = -y * sp + z * cp
    z2 = np.maximum(z2, 0.3)
    return ox + f * x / z2, oy - f * y2 / z2, z2

def landscape(cx, t, cam, alpha=1.0, zero_alpha=1.0, glow_line=1.0):
    Xg = (S[None, :] - 0.5) * SX
    Zg = T[:, None] * np.ones_like(S)[None, :]
    px, py, pz = project(Xg + 0 * Zg, HT, Zg, cam)
    near = cam[2]
    # rows (constant t)
    for i in range(0, len(T), 2):
        zd = pz[i, 0]
        if zd < 1.5: continue
        fade = max(0.0, 1 - (zd - 2) / 46) * alpha
        if fade <= 0.01: continue
        pts = list(zip(px[i], py[i]))
        hmax = min(1.0, HT[i].max() / 3.0)
        col = mix((0.20, 0.34, 0.58), mix(CYAN, WHITE, 0.4), hmax)
        poly(cx, pts, col, (0.45 + 0.5 * hmax) * fade, 1.2 + 1.2 * hmax)
    # columns (constant sigma)
    for j in range(0, len(S), 4):
        ok = pz[:, j] > 1.5
        pts = [(px[i, j], py[i, j]) for i in range(len(T)) if ok[i]]
        poly(cx, pts, (0.25, 0.42, 0.7), 0.28 * alpha, 1.1)
    # critical line on the floor
    j = int(np.argmin(abs(S - 0.5)))
    lx, ly, lz = project(np.zeros_like(T), np.zeros_like(T), T, cam)
    pts = [(lx[i], ly[i]) for i in range(len(T)) if lz[i] > 0.8]
    poly(cx, pts, AMBER, 0.95 * alpha * glow_line, 3.2)
    # zeros
    for g in ZEROS[ZEROS < 52]:
        zx, zy, zz = project(np.array([0.0]), np.array([0.0]), np.array([g]), cam)
        if zz[0] < 1.0: continue
        s = min(1.6, 14 / zz[0])
        fade = max(0.0, 1 - (zz[0] - 2) / 46) * zero_alpha
        bx, by, bz = project(np.array([0.0, 0.0]), np.array([0.0, 6.8]), np.array([g, g]), cam)
        gr = cairo.LinearGradient(bx[0], by[0], bx[1], by[1])
        gr.add_color_stop_rgba(0, *CYAN, 0.0); gr.add_color_stop_rgba(1, *WHITE, 0.9 * fade)
        cx.set_source(gr); cx.set_line_width(3 * s); cx.move_to(bx[0], by[0]); cx.line_to(bx[1], by[1]); cx.stroke()
        dot(cx, bx[1], by[1], 7 * s, WHITE, fade)
        ring(cx, bx[1], by[1], 18 * s, CYAN, 0.8 * fade, 2.4)
        dot(cx, zx[0], zy[0], 4 * s, AMBER, fade)

def cam_hook(t):
    u = t / 6.0
    return (-2.0 + 0.5 * math.sin(t * 0.4), 3.0, 4.0 + 3.6 * u, 0.14, 1000, W / 2, 1000)

# ---------------- plot frame (primes / build) ----------------
PX0, PX1, PY0, PY1 = 140, 960, 1250, 560     # x px range, y px (bottom, top)
XMAX = [50.0]
def mapx(x): return PX0 + (x - 0) / XMAX[0] * (PX1 - PX0)
def mapy(y): return PY0 + y / (XMAX[0] * 1.12) * (PY1 - PY0)
def axes(cx, a, label_y="primes counted (weighted)"):
    cx.set_line_width(2); cx.set_source_rgba(*GREY, 0.7 * a)
    cx.move_to(PX0, PY1 - 20); cx.line_to(PX0, PY0); cx.line_to(PX1 + 10, PY0); cx.stroke()
    stp = 5 if XMAX[0] < 30 else 10
    for xv in range(0, int(XMAX[0]) + 1, stp):
        text(cx, str(xv), mapx(xv), PY0 + 34, 22, GREY, 0.8 * a, F_MONO, False, anchor="c")
    text(cx, "number line →", PX1, PY0 + 70, 22, GREY, 0.7 * a, F_MONO, False, anchor="r")
    cx.save(); cx.translate(PX0 - 22, PY0); cx.rotate(-math.pi / 2)
    text(cx, label_y, 0, 0, 20, GREY, 0.6 * a, F_MONO, False)
    cx.restore()

def staircase_pts(upto=50, arr=None):
    arr = PSI if arr is None else arr
    m = XS <= upto
    xs = XS[m]; ys = arr[m]
    pts = []
    for k in range(len(xs)):
        if k > 0 and ys[k] != ys[k - 1]:
            pts.append((mapx(xs[k]), mapy(ys[k - 1])))
        pts.append((mapx(xs[k]), mapy(ys[k])))
    return pts

def curve_pts(arr, upto=50):
    m = XS <= upto
    return [(mapx(x), mapy(y)) for x, y in zip(XS[m], arr[m])]

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SPOKEN = {2: 7.76, 3: 8.52, 5: 9.16, 7: 9.68, 11: 10.16}

def scene_primes(cx, t, a):
    XMAX[0] = lerp(14.0, 50.0, ease((t - 10.4) / 1.4))
    axes(cx, a)
    # staircase draws left to right
    XMAX[0] = lerp(14.0, 50.0, ease((t - 10.4) / 1.4))
    upto = min(XMAX[0], 1.5 + 48.5 * ease((t - 6.6) / 5.0))
    cx.save(); cx.rectangle(PX0, PY1 - 40, PX1 - PX0 + 20, PY0 - PY1 + 80); cx.clip()
    poly(cx, staircase_pts(upto), AMBER, 0.95 * a, 3.4)
    for p in PRIMES:
        ts = SPOKEN.get(p, 10.4 + (p - 11) * 0.06)
        al = ease((t - ts) / 0.25) * a
        if al <= 0: continue
        x = mapx(p); yb = PY0
        big = p in SPOKEN and abs(t - ts) < 0.6
        cx.set_source_rgba(*CYAN, 0.55 * al); cx.set_line_width(2)
        cx.move_to(x, yb); cx.line_to(x, mapy(PSI[np.searchsorted(XS, p)]) ); cx.stroke()
        dot(cx, x, yb, 7, CYAN, al)
        text(cx, str(p), x, yb + 70, 30, CYAN, al, F_MONO, True, anchor="c") if p in SPOKEN else None
        if big: ring(cx, x, yb, 20 + 40 * (t - ts), CYAN, 0.8 * (1 - (t - ts) / 0.6))
    cx.restore()
    # big accumulating prime readout
    xx = 540; parts = [str(p) for p in SPOKEN if t >= SPOKEN[p]]
    if parts:
        line = ",  ".join(parts) + ("  …" if t > 10.6 else "")
        text(cx, line, 540, 430, 78, WHITE, a, F_MONO, True, anchor="c")
        lastp = max(p for p in SPOKEN if t >= SPOKEN[p])
        if t - SPOKEN[lastp] < 0.5:
            pass
    if t > 11.0:
        text(cx, "gaps: 1, 2, 2, 4, 2, 4, 2, 4, 6 …", 540, 500, 28, GREY, a * ease((t - 11.0) / 0.3), F_MONO, False, anchor="c")
    text(cx, "ψ(x): a step up at every prime", PX0 + 20, PY1 + 10, 22, AMBER, 0.85 * a, F_MONO, False)

# ---------------- scene: zeros -> waves ----------------
STRIP_X0, STRIP_X1 = 110, 230
def ty(g): return PY0 - 30 - (g / 52) * (PY0 - PY1 - 60)
def scene_waves(cx, t, a, merge=0.0):
    # mini critical strip
    cx.set_source_rgba(0.3, 0.5, 0.8, 0.12 * a); cx.rectangle(STRIP_X0, PY1, STRIP_X1 - STRIP_X0, PY0 - PY1); cx.fill()
    xm = (STRIP_X0 + STRIP_X1) / 2
    poly(cx, [(xm, PY0), (xm, PY1)], AMBER, 0.9 * a, 3)
    text(cx, "Re = ½", xm, PY0 + 34, 22, AMBER, 0.9 * a, F_MONO, True, anchor="c")
    text(cx, "zeros of ζ", xm, PY1 - 18, 22, CYAN, 0.9 * a, F_MONO, True, anchor="c")
    zs = ZEROS[ZEROS < 52]
    for k, g in enumerate(zs):
        tk = 13.3 + k * 0.5
        al = ease((t - tk) / 0.3) * a
        if al <= 0: continue
        y = ty(g)
        dot(cx, xm, y, 7, CYAN, al)
        ring(cx, xm, y, 16, CYAN, 0.6 * al, 2)
        # wave flies out to the right
        reach = ease((t - tk) / 0.9)
        xx = np.linspace(1.5, 1.5 + 48.5 * reach, 500)
        amp = 15 * (1 - merge)
        wy = y + amp * np.cos(g * np.log(xx)) * (1 - merge) + merge * (mapy(PSI[-1] * 0.5) - y) * 0
        pts = [(260 + (x - 1.5) / 48.5 * (PX1 - 260), yy) for x, yy in zip(xx, wy)]
        poly(cx, pts, CYAN, 0.75 * al * (1 - merge), 2.0)
        if k < 1:
            text(cx, f"zero #{k+1}  t = {g:.2f}", PX1, y - 24, 20, CYAN, 0.8 * al * (1 - merge), F_MONO, False, anchor="r")
    # equation chip
    ca = win(t, 12.6, 19.3, 0.4, 0.3) * a
    if ca > 0:
        cx.set_source_rgba(0.03, 0.05, 0.09, 0.95 * ca); rrect(cx, 150, 330, 780, 110, 14); cx.fill()
        cx.set_line_width(2); cx.set_source_rgba(*AMBER, 0.6 * ca); rrect(cx, 150, 330, 780, 110, 14); cx.stroke()
        x = 180
        x += text(cx, "ζ", x, 402, 44, WHITE, ca, "DejaVu Serif", False, True)
        x += text(cx, "(s) = 1 + ", x, 402, 44, WHITE, ca, F_MATH, False, True)
        x += text(cx, "1/2", x, 402, 44, WHITE, ca, F_MATH, False, False)
        x += text(cx, "s", x, 382, 26, WHITE, ca, F_MATH, False, True)
        x += text(cx, " + 1/3", x, 402, 44, WHITE, ca, F_MATH, False, False)
        x += text(cx, "s", x, 382, 26, WHITE, ca, F_MATH, False, True)
        x += text(cx, " + 1/4", x, 402, 44, WHITE, ca, F_MATH, False, False)
        x += text(cx, "s", x, 382, 26, WHITE, ca, F_MATH, False, True)
        x += text(cx, " + …", x, 402, 44, WHITE, ca, F_MATH, False, False)
        text(cx, "Riemann's zeta function, 1859", 900, 432, 20, AMBER, 0.8 * ca, F_MONO, False, anchor="r")

# ---------------- scene: build ----------------
def n_at(t):
    u = ease((t - 19.45) / 2.0)
    if u <= 0: return 0
    return int(round(math.exp(u * math.log(601)) - 1))

def scene_build(cx, t, a):
    XMAX[0] = 50.0
    axes(cx, a)
    N = n_at(t)
    locked = ease((t - 22.05) / 0.3)
    stair_col = mix(AMBER, GREEN, locked)
    poly(cx, staircase_pts(), stair_col, 0.9 * a, 3.4)
    poly(cx, curve_pts(psi_N(N)), CYAN, 0.95 * a, 2.6)
    # counter
    text(cx, "waves added", 540, 400, 26, GREY, 0.9 * a, F_MONO, False, anchor="c")
    text(cx, f"N = {N}", 540, 480, 76, CYAN, a, F_MONO, True, anchor="c")
    if locked > 0:
        text(cx, "lands on every prime", 540, 528, 30, GREEN, locked * a, F_MONO, True, anchor="c")
    text(cx, "cyan = sum of waves   amber = real primes", 540, PY0 + 112, 20, GREY, 0.75 * a, F_MONO, False, anchor="c")

# ---------------- scene: one line + twist ----------------
SX0, SX1, SY0, SY1 = 170, 910, 1030, 330     # strip view: sigma 0..1 -> x, t 0..52 -> y
def sxp(s): return SX0 + s * (SX1 - SX0)
def syp(g): return SY0 - g / 52 * (SY0 - SY1)
MX0, MX1, MY0, MY1 = 150, 960, 1330, 1090    # mini plot
def mxp(x): return MX0 + x / 50 * (MX1 - MX0)
def myp(y): return MY0 + y / 56 * (MY1 - MY0)

def beta_at(t):
    u = ease((t - 26.7) / 0.8) - ease((t - 28.85) / 0.35)
    return 0.5 + 0.35 * max(0.0, u)

def scene_line(cx, t, a):
    # strip
    cx.set_source_rgba(0.3, 0.5, 0.8, 0.10 * a); cx.rectangle(sxp(0), SY1, sxp(1) - sxp(0), SY0 - SY1); cx.fill()
    cx.set_line_width(1.5); cx.set_source_rgba(*GREY, 0.5 * a)
    for s in (0, 1): cx.move_to(sxp(s), SY0); cx.line_to(sxp(s), SY1)
    cx.stroke()
    text(cx, "0", sxp(0), SY0 + 32, 22, GREY, 0.8 * a, F_MONO, False, anchor="c")
    text(cx, "1", sxp(1), SY0 + 32, 22, GREY, 0.8 * a, F_MONO, False, anchor="c")
    pulse = 0.5 + 0.5 * math.sin(max(0, t - 25.3) * 9) if 25.3 < t < 26.4 else 0
    poly(cx, [(sxp(0.5), SY0), (sxp(0.5), SY1)], AMBER, a, 3.5 + 4 * pulse)
    text(cx, "the critical line  Re(s) = ½", sxp(0.5), SY1 - 20, 24, AMBER, a, F_MONO, True, anchor="c")
    b = beta_at(t)
    broke = (b - 0.5) / 0.35
    zs = ZEROS[ZEROS < 52]
    for k, g in enumerate(zs):
        y = syp(g)
        if k == 0 and b > 0.5001:
            col = RED
            dot(cx, sxp(b), y, 9, col, a); ring(cx, sxp(b), y, 22, col, 0.8 * a)
            dot(cx, sxp(1 - b), y, 9, col, 0.7 * a); ring(cx, sxp(1 - b), y, 22, col, 0.5 * a)
            cx.set_dash([6, 6]); poly(cx, [(sxp(1 - b), y), (sxp(b), y)], col, 0.5 * a, 1.5); cx.set_dash([])
            text(cx, "what if it sat here?", sxp(b), y - 34, 22, RED, a * broke, F_MONO, True, anchor="c")
        else:
            col = GREEN if t > 28.9 and k == 0 else CYAN
            dot(cx, sxp(0.5), y, 8, col, a); ring(cx, sxp(0.5), y, 18, col, 0.6 * a, 2)
    # checked counter
    # mini plot
    cx.set_line_width(1.5); cx.set_source_rgba(*GREY, 0.6 * a)
    cx.move_to(MX0, MY1 - 10); cx.line_to(MX0, MY0); cx.line_to(MX1, MY0); cx.stroke()
    pts = [(mxp(x), myp(y)) for x, y in zip(XS, PSI)]
    poly(cx, pts, AMBER, 0.8 * a, 2.6)
    arr = psi_moved(b) if b > 0.5001 else psi_N(600)
    ccol = RED if broke > 0.02 else (GREEN if t > 28.9 else CYAN)
    poly(cx, [(mxp(x), myp(y)) for x, y in zip(XS, arr)], ccol, 0.95 * a, 2.4)
    lbl = "primes rebuilt from the zeros"
    text(cx, lbl, MX0 + 14, MY1 - 8, 20, GREY, 0.8 * a, F_MONO, False)
    # top readout
    if broke > 0.05:
        text(cx, "ERROR ↑", MX1 - 6, MY1 - 8, 26, RED, a * broke, F_MONO, True, anchor="r")

# ---------------- scene: news ----------------
def card(cx, x, y, w, h, a, head, head_col, lines, stamp=None, sa=0.0):
    if a <= 0: return
    cx.save(); cx.translate(0, 30 * (1 - a))
    cx.set_source_rgba(0.04, 0.07, 0.12, 0.92 * a); rrect(cx, x, y, w, h, 16); cx.fill()
    cx.set_line_width(2); cx.set_source_rgba(*head_col, 0.6 * a); rrect(cx, x, y, w, h, 16); cx.stroke()
    text(cx, head, x + 28, y + 50, 26, head_col, a, F_MONO, True)
    yy = y + 104
    for s, col, size, la in lines:
        text(cx, s, x + 28, yy, size, col, a * la, F_CAP, True); yy += size + 22
    if stamp and sa > 0:
        cx.save(); cx.translate(x + w - 130, y + h / 2 + 10); cx.rotate(-0.18)
        sc = 1 + 0.5 * (1 - ease(sa / 1.0))
        cx.scale(sc, sc)
        cx.set_line_width(5); cx.set_source_rgba(*GREEN, sa * a); rrect(cx, -95, -42, 190, 84, 10); cx.stroke()
        text(cx, stamp, 0, 14, 40, GREEN, sa * a, F_MONO, True, anchor="c")
        cx.restore()
    cx.restore()

def scene_news(cx, t, a):
    a1 = ease((t - 29.25) / 0.4) * a
    ln = ease((t - 31.2) / 0.3); lc = ease((t - 33.1) / 0.3); lp = ease((t - 34.4) / 0.3)
    card(cx, 80, 330, 920, 330, a1, "20 NOV 2026 · RELEASED", CYAN,
         [("Levent Alpöge", WHITE, 50, ln), ("mathematician at Anthropic", GREY, 30, ln),
          ("+ Claude (AI)", CYAN, 44, lc)])
    # Mathlib's formal statement, typed out
    aL = win(t, 29.7, 34.45, 0.4, 0.25) * a
    if aL > 0:
        cx.set_source_rgba(0.02, 0.04, 0.07, 0.95 * aL); rrect(cx, 80, 690, 920, 520, 16); cx.fill()
        cx.set_line_width(2); cx.set_source_rgba(*GREY, 0.4 * aL); rrect(cx, 80, 690, 920, 520, 16); cx.stroke()
        text(cx, "-- the claim, as Lean's library Mathlib states it", 108, 740, 22, GREY, aL, F_MONO, False)
        LEAN = ["def RiemannHypothesis : Prop :=", "  \u2200 (s : \u2102),", "    riemannZeta s = 0 \u2192",
                "    (\u00ac\u2203 n : \u2115, s = -2 * (n + 1)) \u2192", "    s \u2260 1 \u2192", "    s.re = 1 / 2"]
        total = sum(len(l) for l in LEAN); shown = int(total * ease((t - 30.0) / 3.2)); yy = 810
        for k, l in enumerate(LEAN):
            part = l[:max(0, shown)]; shown -= len(l)
            col = AMBER if k == 5 else (CYAN if k == 0 else WHITE)
            text(cx, part, 108, yy, 30, col, aL, F_MONO, k in (0, 5)); yy += 62
        if t > 33.2:
            text(cx, "\u2190 every zero on the line Re = \u00bd", 440, 810 + 5 * 62, 22, AMBER, aL * ease((t - 33.2) / 0.3), F_MONO, False)
    # lines counter terminal
    a2 = ease((t - 34.4) / 0.4) * a
    if a2 > 0:
        cx.set_source_rgba(0.02, 0.04, 0.07, 0.95 * a2); rrect(cx, 80, 690, 920, 250, 16); cx.fill()
        cx.set_line_width(2); cx.set_source_rgba(*GREY, 0.4 * a2); rrect(cx, 80, 690, 920, 250, 16); cx.stroke()
        text(cx, "$ lake build   # proof written in Lean", 108, 740, 22, GREY, a2, F_MONO, False)
        u = ease((t - 35.55) / 1.7)
        n = int(17_000_000 * u)
        text(cx, f"{n:,}", 108, 840, 74, YEL, a2, F_MONO, True)
        text(cx, "lines of proof (approx.)", 108, 890, 24, GREY, a2, F_MONO, False)
        m = int(175_187 * ease((t - 37.9) / 1.6))
        if t > 37.6:
            text(cx, f"modules checked {m:,} / 175,187", 560, 890, 22, GREEN if m == 175187 else GREY, a2, F_MONO, False)
    a3 = ease((t - 37.95) / 0.4) * a
    card(cx, 80, 970, 920, 300, a3, "21 NOV 2026 · CERTIFIED", GREEN,
         [("Lean + Mathlib communities", WHITE, 40, 1), ("statement OK · proof OK", GREEN, 36, ease((t - 38.6) / 0.3))],
         stamp="CHECKED", sa=ease((t - 39.3) / 0.25))

def scene_open(cx, t, a):
    card(cx, 80, 330, 920, 300, a, "STILL OPEN", AMBER,
         [("a paper humans can read", WHITE, 38, 1), ("expert review · the $1M Clay prize", GREY, 30, 1)])
    ta = ease((t - 42.4) / 0.4) * a
    text(cx, "next ▸ why primes guard your bank account", 540, 1335, 26, CYAN, 0.9 * ta, F_MONO, True, anchor="c")

# ---------------- hook text ----------------
def hook(cx, t):
    a = win(t, -1.0, 6.1, 0.01, 0.3)
    if a <= 0: return
    text(cx, "167 YEARS", 540, 1170, 120, WHITE, a, F_CAP, True, anchor="c")
    s1 = ease((t - 3.1) / 0.25); flip = ease((t - 5.28) / 0.25)
    if s1 > 0:
        w = text_w(cx, "UNSOLVED", 96)
        col = mix(WHITE, RED, flip)
        text(cx, "UNSOLVED", 540, 1290, 96, col, a * s1 * (1 - 0.55 * flip), F_CAP, True, anchor="c")
        if flip > 0:
            cx.set_line_width(9); cx.set_source_rgba(*RED, a * flip)
            cx.move_to(540 - w / 2 - 10, 1258); cx.line_to(540 - w / 2 - 10 + (w + 20) * flip, 1258); cx.stroke()
    if flip > 0:
        sc = 1 + 0.35 * (1 - eout((t - 5.28) / 0.35))
        cx.save(); cx.translate(540, 1410); cx.scale(sc, sc)
        wpr = text(cx, "PROVED", -60, 0, 110, GREEN, a * flip, F_CAP, True, anchor="c")
        check(cx, -60 + wpr / 2 + 24, -14, 80, GREEN, a * flip); cx.restore()
    text(cx, "1/|ζ(s)| · real values · every spike = a zero", 540, 1500, 22, GREY, 0.8 * a, F_MONO, False, anchor="c")

# ---------------- frame ----------------
def frame(t):
    sw, sh = int(W * SS), int(H * SS)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, sw, sh)
    cx = cairo.Context(surf); cx.scale(SS, SS)
    background(cx, t)
    readout = ""
    # 3D landscape: hook and ending
    la = win(t, -1, 6.25, 0.01, 0.35)
    if la > 0:
        landscape(cx, t, cam_hook(t), la)
        readout = "zeros at t = 14.13, 21.02, 25.01 …"
    ea = ease((t - 40.4) / 0.9)
    if ea > 0:
        u = ease((t - 40.4) / 3.5)
        cam = (-2.0, lerp(6, 3.0, u), lerp(-4, 4.0, u), lerp(0.3, 0.14, u), 1000, W / 2, lerp(1060, 1000, u))
        landscape(cx, t, cam, ea * 0.9, ea)
    pa = win(t, 6.0, 12.4, 0.35, 0.4)
    if pa > 0:
        scene_primes(cx, t, pa); readout = "counting primes ≤ 50"
    wa = win(t, 12.2, 19.6, 0.4, 0.3)
    if wa > 0:
        scene_waves(cx, t, wa, merge=ease((t - 19.2) / 0.4)); readout = "each zero t → a wave cos(t·ln x)"
    ba = win(t, 19.3, 23.3, 0.3, 0.35)
    if ba > 0:
        scene_build(cx, t, ba); readout = "explicit formula (von Mangoldt, 1895)"
    sa = win(t, 23.1, 29.25, 0.35, 0.3)
    if sa > 0:
        scene_line(cx, t, sa); readout = "zero #1 moved: illustration" if beta_at(t) > 0.51 else "first 10 zeros, all on the line"
        ca = ease((t - 23.4) / 0.5) * sa
        text(cx, "12 trillion+ zeros checked by computer ✓", 540, 1385, 24, GREEN, ca, F_MONO, True, anchor="c")
    na = win(t, 29.1, 40.5, 0.35, 0.35)
    if na > 0:
        scene_news(cx, t, na); readout = "sources: Lean community statement, 21 Nov 2026"
    oa = win(t, 40.4, 44.5, 0.4, 0.01)
    if oa > 0:
        scene_open(cx, t, oa)
    hook(cx, t)
    sticker(cx, t)
    captions(cx, t)
    hud(cx, t, readout)
    # numpy + glow
    buf = np.ndarray((sh, sw, 4), np.uint8, surf.get_data())
    img = buf[:, :, [2, 1, 0]].astype(np.float32)
    small = cv2.resize(img, (sw // 4, sh // 4), interpolation=cv2.INTER_AREA)
    bright = np.clip(small - 70, 0, 255)
    gl = cv2.GaussianBlur(bright, (0, 0), 6 * SS) * 1.2 + cv2.GaussianBlur(bright, (0, 0), 18 * SS) * 0.9
    gl = cv2.resize(gl, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out = np.clip(img + GLOW * gl, 0, 255)
    # film grain (subtle)
    rng = np.random.default_rng(int(t * 1000))
    out += rng.normal(0, 2.2, (sh // 2, sw // 2, 1)).repeat(2, 0).repeat(2, 1)[:sh, :sw]
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "video"
    if mode == "stills":
        for tt in [float(v) for v in sys.argv[3].split(",")]:
            cv2.imwrite(f"still_{tt:05.2f}.png", cv2.cvtColor(frame(tt), cv2.COLOR_RGB2BGR))
    else:
        nf = int(DUR * FPS)
        a0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        a1 = int(sys.argv[4]) if len(sys.argv) > 4 else nf
        for i in range(a0, a1):
            sys.stdout.buffer.write(frame(i / FPS).tobytes())
