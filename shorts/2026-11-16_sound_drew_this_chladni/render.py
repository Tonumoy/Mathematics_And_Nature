# SC#11 Chladni plate — 3D-feel render (2x supersampled, LANCZOS down)
import numpy as np, cv2, math, sys, os, functools
from PIL import Image, ImageDraw
import lib
from lib import CYAN, AMBER, RED, GREEN, YELLOW, WHITE, GREY
import timeline as tl
import sim
from plate import hz

W, H = 1080, 1920; S2 = 2; WW, HH = W * S2, H * S2
TEX = 1400
FPS = 30
DAT = None
def data():
    global DAT
    if DAT is None:
        d = np.load(os.path.join(os.path.dirname(__file__), 'sand.npz'))
        DAT = (d['P'], d['A'], d['info'])
    return DAT

def ramp(t, a, b): return float(np.clip((t - a) / max(b - a, 1e-6), 0, 1))
def sm(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

# ---------- static textures ----------
@functools.lru_cache(None)
def metal():
    rng = np.random.default_rng(7)
    n = rng.standard_normal((TEX, TEX)).astype(np.float32)
    # circular brushing: blur along tangent via polar warp
    pol = cv2.warpPolar(n, (TEX, 1440), (TEX / 2, TEX / 2), TEX * 0.72, cv2.WARP_POLAR_LINEAR)
    pol = cv2.GaussianBlur(pol, (1, 61), 0, sigmaY=18)
    br = cv2.warpPolar(pol, (TEX, TEX), (TEX / 2, TEX / 2), TEX * 0.72, cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP)
    br = br / (br.std() + 1e-6)
    y, x = np.mgrid[0:TEX, 0:TEX].astype(np.float32) / TEX
    base = 46 + 7 * br
    img = np.stack([base * 0.95, base * 1.0, base * 1.08], -1)
    # bevel edge highlight
    e = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    img += (np.exp(-e / 0.006) * 70)[..., None] * np.array([0.8, 0.95, 1.0])
    return img.astype(np.float32), x, y

@functools.lru_cache(None)
def background():
    y, x = np.mgrid[0:HH, 0:WW].astype(np.float32)
    g = np.zeros((HH, WW, 3), np.float32) + np.array([6, 8, 14], np.float32)
    spot = np.exp(-(((x - WW * 0.5) / (WW * 0.55)) ** 2 + ((y - HH * 0.42) / (HH * 0.33)) ** 2))
    g += spot[..., None] * np.array([22, 30, 48], np.float32)
    # floor glow under the plate
    fl = np.exp(-(((x - WW * 0.5) / (WW * 0.5)) ** 2 + ((y - HH * 0.62) / (HH * 0.08)) ** 2))
    g += fl[..., None] * np.array([10, 18, 26], np.float32)
    return g

DUST = np.random.default_rng(11).uniform([-3, -3, -0.6], [3, 3, 2.6], (260, 3)).astype(np.float32)

# ---------- camera ----------
def cam(t):
    """tilt (deg from top-down), yaw (deg), dist, screen center y (1x)"""
    T = tl.T; s = tl.s
    tilt = 38; yaw = 8 * math.sin(t * 0.21) ; dist = 5.2; cy = 930
    if t < s('L02'):                  # hook: slow push-in
        hc = os.environ.get('HOOKCAM', 'A')
        if hc == 'A': tilt = 24 + 6 * sm(t / 2.0); dist = 4.9 - 0.3 * sm(t / 2.2); cy = 1000
        elif hc == 'B': tilt = 30; dist = 2.9 - 0.2 * sm(t / 2.2); yaw = 20 + 4 * t; cy = 1060
        else: tilt = 56; dist = 4.4; yaw = 30 + 5 * t; cy = 1020
    elif t < s('L04'):
        tilt = 44 - 6 * sm(ramp(t, s('L02'), s('L03'))); dist = 5.3
    elif t < s('L05'):                # x-ray: lower angle to see the surface
        k = sm(ramp(t, s('L04'), s('L04') + 0.8)); tilt = 38 + 20 * k; dist = 5.3 + 0.5 * k; cy = 960
    elif t < s('L07'):
        tilt = 34; dist = 4.9; yaw = 8 * math.sin(t * 0.21) + 10 * sm(ramp(t, s('L05'), s('L07')))
    elif t < s('L09'):
        tilt = 36; dist = 5.0; yaw = 10 + 8 * math.sin(t * 0.21); cy = 990
    else:                             # zoom out
        k = sm(ramp(t, s('L09'), s('L09') + 2.5)); k2 = sm(ramp(t, s('L10') - 0.3, tl.T))
        tilt = (36 + 14 * k) * (1 - k2) + 24 * k2; dist = (5.0 + 2.6 * k) * (1 - k2) + 4.9 * k2
        yaw = (10 + 25 * k) * (1 - k2) + 0 * k2; cy = (990 - 60 * k) * (1 - k2) + 1000 * k2
    return tilt, yaw, dist, cy

def projector(t):
    tilt, yaw, dist, cy = cam(t)
    ph = math.radians(tilt); ps = math.radians(yaw)
    pos = np.array([0, -dist * math.sin(ph), dist * math.cos(ph)])
    f = -pos / np.linalg.norm(pos); r = np.array([1.0, 0, 0]); u = np.cross(r, f)
    fpx = 2150 * S2
    cyy = cy * S2; cxx = 540 * S2
    cR, sR = math.cos(ps), math.sin(ps)
    def P(X, Y, Z):
        X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); Z = np.asarray(Z, np.float64)
        X1 = cR * X - sR * Y; Y1 = sR * X + cR * Y
        dx, dy, dz = X1 - pos[0], Y1 - pos[1], Z - pos[2]
        dep = dx * f[0] + dy * f[1] + dz * f[2]
        xs = cxx + fpx * (dx * r[0] + dy * r[1] + dz * r[2]) / dep
        ys = cyy - fpx * (dx * u[0] + dy * u[1] + dz * u[2]) / dep
        return xs, ys, dep
    return P

# ---------- per-frame field ----------
def field(f, h):
    R, _ = sim.response(f, h)
    k = int(np.argmin(np.abs(sim.FR * h - f)))
    peak = abs(sim.AK[k]) * sim.Q / ((sim.FR[k] * h) ** 2) * np.abs(sim.FK[k]).max()
    return R / (peak + 1e-12)

def settled(n, info):
    a = max(0, n - 22)
    fr = info[a:n + 1, 0]; hh = info[a:n + 1, 1]
    ch = (fr.max() / max(fr.min(), 1) - 1) * 40 + (hh.max() - hh.min()) * 8
    return float(np.clip(1 - ch, 0, 1))

def plate_texture(n, t):
    P, A, info = data()
    f, h, on = info[n]
    img, xg, yg = metal()
    img = img.copy()
    # vibration shimmer: brighter where the plate moves (antinodes), pulsing
    Af = field(f, h) * on
    Af = cv2.resize(Af, (TEX, TEX), interpolation=cv2.INTER_LINEAR)
    img += (np.clip(Af, 0, 1.5) * (10 + 9 * math.sin(t * 2 * math.pi * 6)))[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    # sand
    p = P[n].astype(np.float32); a = A[n].astype(np.float32)
    ix = np.clip(((p[:, 0] + 1) * 0.5 * (TEX - 1)).astype(np.int32), 0, TEX - 1)
    iy = np.clip(((1 - (p[:, 1] + 1) * 0.5) * (TEX - 1)).astype(np.int32), 0, TEX - 1)
    act = np.clip((a - 0.05) / 0.12, 0, 1) * on * (0.0 if t < tl.s('L02') else 1.0)
    idx = iy * TEX + ix
    calm = np.bincount(idx, weights=1 - act, minlength=TEX * TEX).reshape(TEX, TEX).astype(np.float32)
    hot = np.bincount(idx, weights=act, minlength=TEX * TEX).reshape(TEX, TEX).astype(np.float32)
    calm = cv2.GaussianBlur(calm, (0, 0), 0.9); hot = cv2.GaussianBlur(hot, (0, 0), 1.1)
    st = settled(n, info)
    # grain shadow
    sh = cv2.GaussianBlur(calm + hot, (0, 0), 2.5)
    img *= (1 - np.clip(sh * 0.18, 0, 0.5))[..., None]
    sand_c = np.array([250, 232, 196], np.float32)
    img += np.clip(calm * 2.2, 0, 4.5)[..., None] * (sand_c * (0.80 if t < tl.s('L02') else 0.62))
    img += np.clip(hot * 1.5, 0, 4.0)[..., None] * (np.array([255, 130, 100], np.float32) * 0.50)
    # order glow (green) on settled lines, chaos glow (red) on moving sand
    gl = cv2.GaussianBlur(calm, (0, 0), 7)
    img += np.clip(gl * 2.0, 0, 1.5)[..., None] * np.array(GREEN, np.float32) * 0.10 * st
    hg = cv2.GaussianBlur(hot, (0, 0), 6)
    img += np.clip(hg * 1.5, 0, 1.2)[..., None] * np.array(RED, np.float32) * 0.18
    # driver bolt at centre (amber)
    c = TEX // 2; rr = int(TEX * 0.022)
    pulse = 0.6 + 0.4 * math.sin(t * 2 * math.pi * 3) if on > 0.5 else 0.3
    cv2.circle(img, (c, c), rr, (70, 72, 78), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), rr, (255 * 0.9, 176 * 0.9, 60 * 0.9), 3, cv2.LINE_AA)
    bolt = np.zeros((TEX, TEX), np.float32); cv2.circle(bolt, (c, c), rr + 6, 1.0, -1, cv2.LINE_AA)
    bolt = cv2.GaussianBlur(bolt, (0, 0), 14)
    img += bolt[..., None] * np.array(AMBER, np.float32) * 0.55 * pulse
    return np.clip(img, 0, 255), st, f, h, on

def draw_scene(n):
    t = n / FPS
    P = projector(t)
    frame = background().copy()
    # dust motes (depth bokeh)
    xs, ys, dep = P(DUST[:, 0], DUST[:, 1], DUST[:, 2])
    dl = np.zeros((HH // 4, WW // 4), np.float32)
    for x, y, d in zip(xs, ys, dep):
        if 0 <= x < WW and 0 <= y < HH:
            dl[int(y) // 4, int(x) // 4] += 2.2 / max(d, 0.5)
    dl = cv2.GaussianBlur(dl, (0, 0), 2.2)
    frame += cv2.resize(dl, (WW, HH))[..., None] * np.array([120, 170, 220], np.float32)
    tex, st, f, h, on = plate_texture(n, t)
    th = 0.035 * (1 + (h - 1) * 0.9)
    # plate side faces (thickness grows with the dial)
    C = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    top = [P(x, y, 0)[:2] for x, y in C]; bot = [P(x, y, -th)[:2] for x, y in C]
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([top[i], top[j], bot[j], bot[i]], np.float32)
        # only faces whose bottom edge is lower on screen than top (visible)
        if (bot[i][1] + bot[j][1]) > (top[i][1] + top[j][1]) + 1:
            shade = 70 if i == 0 else 45
            cv2.fillConvexPoly(frame, quad.astype(np.int32), (shade * 0.9, shade, shade * 1.15), cv2.LINE_AA)
    # soft shadow on the floor
    sx, sy, _ = P(np.array([-1.15, 1.15, 1.15, -1.15]), np.array([-1.15, -1.15, 1.15, 1.15]) , np.full(4, -0.55))
    shm = np.zeros((HH, WW), np.float32)
    cv2.fillConvexPoly(shm, np.stack([sx, sy], 1).astype(np.int32), 1.0, cv2.LINE_AA)
    shm = cv2.GaussianBlur(cv2.resize(shm, (WW // 8, HH // 8)), (0, 0), 6)
    frame *= (1 - 0.55 * cv2.resize(shm, (WW, HH)))[..., None]
    # plate top via homography
    src = np.array([[0, TEX - 1], [TEX - 1, TEX - 1], [TEX - 1, 0], [0, 0]], np.float32)  # (-1,-1),(1,-1),(1,1),(-1,1)
    dst = np.array(top, np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, Hm, (WW, HH), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.ones((TEX, TEX), np.float32), Hm, (WW, HH), flags=cv2.INTER_LINEAR)
    frame = frame * (1 - mask[..., None]) + warped
    if on > 0.5 and (t < tl.s('L02') or t >= tl.s('L10') - 0.3):
        lay = np.zeros((HH, WW, 3), np.float32)
        for j in range(3):
            ph = ((t * 0.9 + j / 3.0) % 1.0)
            rad = 0.06 + 0.55 * ph; al = (1 - ph) ** 1.5
            th_ = np.linspace(0, 2 * np.pi, 160)
            xs, ys, _ = P(rad * np.cos(th_), rad * np.sin(th_), np.full(160, 0.01))
            cv2.polylines(lay, [np.stack([xs, ys], 1).astype(np.int32)], True, tuple(c * al for c in AMBER), 5, cv2.LINE_AA)
        frame = frame + lay * 0.8 + cv2.GaussianBlur(lay, (0, 0), 14) * 1.2
    return frame, P, st, f, h, on, top

def xray(frame, P, t, k):
    """3D wireframe of the real mode shape over the plate"""
    s4 = tl.s('L04'); e4 = tl.s('L05')
    a_in = sm(ramp(t, s4 + 0.2, s4 + 1.0)) * (1 - sm(ramp(t, e4 - 0.6, e4)))
    if a_in <= 0: return frame
    frame *= (1 - 0.55 * a_in)
    G = 41
    _, F, _ = sim.lam, sim.F, None
    M = cv2.resize(sim.F[k], (G, G), interpolation=cv2.INTER_AREA)
    M = M / np.abs(M).max()
    amp = 0.32 * math.sin(t * 2 * math.pi * 0.9) * a_in
    g = np.linspace(-1, 1, G)
    X, Y = np.meshgrid(g, -g)
    xs, ys, _ = P(X, Y, 0.06 + amp * M)
    lay = np.zeros((HH, WW, 3), np.float32)
    for i in range(G):
        for arr in (np.stack([xs[i], ys[i]], 1), np.stack([xs[:, i], ys[:, i]], 1)):
            cv2.polylines(lay, [arr.astype(np.int32)], False, (CYAN[0] * 0.75, CYAN[1] * 0.75, CYAN[2] * 0.75), 2, cv2.LINE_AA)
    # nodal points glow green
    nod = np.abs(M) < 0.07
    for x, y in zip(xs[nod], ys[nod]):
        cv2.circle(lay, (int(x), int(y)), 6, GREEN, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(cv2.resize(lay, (WW // 4, HH // 4)), (0, 0), 4)
    lay += cv2.resize(glow, (WW, HH)) * 1.6
    return frame + lay * a_in

def bloom(frame):
    sm4 = cv2.resize(frame, (WW // 8, HH // 8), interpolation=cv2.INTER_AREA)
    hi = np.clip(sm4 - 150, 0, None)
    b = cv2.GaussianBlur(hi, (0, 0), 5) + cv2.GaussianBlur(hi, (0, 0), 14)
    return frame + cv2.resize(b, (WW, HH)) * 0.55

def fmt_hz(f): return f"{int(round(f)):,} Hz"

CAPS = {
 'L02': ("One note: sand runs to the lines that never move", ("never",)),
 'L03': ("Wait for the highest note", ("highest",)),
 'L04': ("One equation. Our code solved it.", ("solved",)),
 'L05': ("Right note = a shape. Wrong note = a mess", ("shape",)),
 'L06': ("Higher note, more lines", ("more",)),
 'L07': ("One number: twice as thick", ("twice",)),
 'L08': ("Twice as thick: double the pitch", ("double",)),
 'L09': ("Violin makers tune wood with this trick", ("violin",)),
}

def hud(n, t, f, h, on, st, top, P):
    lay = Image.new("RGBA", (WW, HH), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    s = tl.s; e = tl.e
    # series tag
    lib.pill(d, (930, 205), "SOURCE CODE #11", col=CYAN, size=28, anchor="rm", ss=S2)
    if t < s('L02') - 0.15:
        a = 255
        lib.draw_text(d, (540, 340), "SOUND", 128, fill=YELLOW, stroke=6, ss=S2)
        lib.draw_text(d, (540, 480), "DREW THIS.", 112, fill=WHITE, stroke=6, ss=S2)
        lib.pill(d, (540, 1520), fmt_hz(f) + "  ·  steel plate  ·  80,000 grains", col=AMBER, size=30, ss=S2)
        # cyan ring = rim around the plate
        pts = np.array(top, np.float32)
        d.polygon([tuple(p) for p in pts], outline=CYAN + (230,), width=int(5 * S2))
    else:
        # Hz counter (driver = amber)
        ca = int(255 * ramp(t, s('L02') + 0.3, s('L02') + 0.7) * (1 - ramp(t, e('L09') + 0.1, e('L09') + 0.35)))
        if ca > 0:
            col = AMBER
            lib.draw_text(d, (540, 330), fmt_hz(f) if on > 0.2 else "silent", 104, fill=col + (ca,), stroke=5, ss=S2)
            lib.draw_text(d, (540, 418), "tone driving the plate", 30, fill=GREY + (ca,), path=lib.FONT_SB, ss=S2)
        # wait-for promise
        if s('L03') <= t < s('L07') - 0.3:
            done = f >= 3940
            lib.pill(d, (540, 495), ("REACHED  3,950 Hz ✓" if done else "WAIT FOR  3,950 Hz"),
                     col=(GREEN if done else YELLOW), size=32, alpha=int(255 * ramp(t, s('L03'), s('L03') + 0.3)), ss=S2)
        # shape counter during sweep
        if s('L02') + 1.5 <= t < s('L07'):
            seq_f = [sim.F1[k] for k in tl.SEQ]
            reached = sum(1 for x in seq_f if f >= x * 0.995)
            lib.pill(d, (150, 205), f"SHAPE {max(reached,1)}/10", col=GREEN if (np.min(np.abs(sim.F1[tl.SEQ] / f - 1)) < 0.005) else RED, size=28, anchor="lm", ss=S2)
        # equation (x-ray) once, then chip
        if s('L04') + 0.6 <= t < s('L04') + 3.6:
            lib.chip(d, (540, 600), "D·∇⁴w = ρh·ω²·w", col=AMBER, size=46, ss=S2)
            lib.draw_text(d, (540, 668), "how a flat plate bends (Kirchhoff, 1850)", 28, fill=GREY, path=lib.FONT_SB, ss=S2)
        elif s('L04') + 3.6 <= t < s('L07'):
            lib.chip(d, (930 - 150, 1600 - 0), "D∇⁴w=ρhω²w", col=AMBER, size=24, alpha=200, ss=S2) if False else None
        # thickness dial (twist)
        if s('L07') <= t < s('L09'):
            dial(d, t, h, f, st)
        # status word
        if s('L05') <= t < s('L09'):
            onres = np.min(np.abs(sim.F1[tl.SEQ] * h / f - 1)) < 0.005
            word, col = ("ORDER", GREEN) if onres else ("CHAOS", RED)
            tw = s('L07') + 2.0; tsnap = s('L08') + 0.55 * (e('L08') - s('L08')) + 1.0
            if tw <= t < tsnap: word, col = ("DIFFERENT SHAPE", RED)
            lib.pill(d, (930, 270), word, col=col, size=26, anchor="rm", ss=S2)
    # captions
    for k, (txt, hl) in CAPS.items():
        a0, a1 = s(k), e(k) + 0.25
        if a0 <= t < a1:
            al = int(255 * min(ramp(t, a0, a0 + 0.12), 1 - ramp(t, a1 - 0.15, a1)))
            lib.caption(d, txt, y=(1535 if k in ('L07', 'L08') else 1475), size=56, hl=hl, alpha=al, ss=S2)
    if t >= e('L09') + 0.3:
        al = int(255 * ramp(t, e('L09') + 0.3, e('L09') + 0.6))
        lib.draw_text(d, (540, 330), "NATURE IS RUNNING ON CODE", 50, fill=CYAN + (al,), stroke=4, ss=S2)
    if t >= s('L10'):
        al = int(255 * ramp(t, s('L10'), s('L10') + 0.3))
        lib.pill(d, (540, 430), "NEXT: can sound lift water?", col=YELLOW, size=34, alpha=al, ss=S2)
    return lay

def dial(d, t, h, f, st):
    s = tl.s
    a = int(255 * ramp(t, s('L07'), s('L07') + 0.4))
    cx, cy, R = 540, 560, 96
    snapped = t > s('L08') + 0.55 * (tl.e('L08') - s('L08')) + 1.0 and st > 0.85
    broke = h > 1.05 and not snapped
    col = GREEN if snapped else (RED if broke else CYAN)
    def fade(c): return c + (a,)
    d.arc([(cx - R) * S2, (cy - R) * S2, (cx + R) * S2, (cy + R) * S2], 200, 340, fill=fade(GREY), width=int(10 * S2))
    frac = (h - 1.0) / 1.0
    d.arc([(cx - R) * S2, (cy - R) * S2, (cx + R) * S2, (cy + R) * S2], 200, 200 + 140 * frac + 0.5, fill=fade(col), width=int(10 * S2))
    ang = math.radians(200 + 140 * frac)
    d.line([cx * S2, cy * S2, (cx + (R - 18) * math.cos(ang)) * S2, (cy + (R - 18) * math.sin(ang)) * S2], fill=fade(WHITE), width=int(6 * S2))
    lib.draw_text(d, (cx, cy + 30), f"thickness {h:.1f} mm", 32, fill=fade(col), ss=S2)
    lib.draw_text(d, (cx - R - 10, cy + 2), "1", 26, fill=fade(GREY), path=lib.FONT_SB, ss=S2)
    lib.draw_text(d, (cx + R + 10, cy + 2), "2", 26, fill=fade(GREY), path=lib.FONT_SB, ss=S2)

def frame(n):
    t = n / FPS
    fr, P, st, f, h, on, top = draw_scene(n)
    if tl.s('L04') <= t < tl.s('L05'):
        fr = xray(fr, P, t, tl.SEQ[2])
    fr = bloom(fr)
    # vignette
    if not hasattr(frame, 'V'):
        y, x = np.mgrid[0:HH, 0:WW].astype(np.float32)
        r = np.sqrt(((x - WW / 2) / (WW * 0.75)) ** 2 + ((y - HH / 2) / (HH * 0.7)) ** 2)
        frame.V = np.clip(1.12 - 0.42 * r ** 2, 0.55, 1.0)[..., None]
    fr = fr * frame.V
    img = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    img.alpha_composite(hud(n, t, f, h, on, st, top, P))
    return img.convert("RGB").resize((W, H), Image.LANCZOS)

if __name__ == '__main__':
    for n in map(int, sys.argv[1:]):
        frame(n).save(f'f_{n:04d}.png')
