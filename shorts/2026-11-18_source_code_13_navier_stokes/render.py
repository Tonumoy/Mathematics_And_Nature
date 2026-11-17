"""SOURCE CODE #12 - Navier-Stokes. Stateless per-frame renderer.
usage: python3 render.py start_frame end_frame outdir [step]"""
import sys, json, os, numpy as np, cv2
from PIL import Image, ImageDraw
from common import *
import chai, xray, tube, others, cup3d
import scipy.fft as sf

TL = json.load(open('audio/timeline.json')); DUR = TL['dur']
LS = [l['start'] for l in TL['lines']]
def L(i): return LS[i]
ring_i, drop_i = [int(v) for v in open('picks.txt').read().split()]
P, S = xray.load()
_, WV = chai.load()
SC0 = chai.R_LIQ / chai.RS
FH = 158                     # chai frame at video t=0 (ending uses 0..157 -> seamless loop)
T_XR0, T_XR1 = 3.95, 4.95    # x-ray scan
T_C = 8.40                   # zoom into the drop
T_WING, T_STORM, T_BLOOD, T_E = 17.45, 21.15, 22.90, 24.75
T_NEWS, T_END = 37.45, 44.95

EQ = {k: Image.open(f'eq/{k}.png').convert('RGBA') for k in ['p1', 'p2', 'eq', 'p3', 'pl', 'p4', 'newton', 'kelvin']}

# ---------- captions (VO key words; [..] = yellow) ----------
CAPS = [(1, "every drop follows [ONE] rule"), (3, "Newton's [push & move] law"),
        (4, "[pressure] pushes"), (5, "[stickiness] slows"), (6, "[motion] carries it"),
        (7, "we use it [every day]"), (8, "[planes]"), (9, "[weather]"), (10, "[blood] in your heart"),
        (11, "so what's worth [a million?]"), (12, "stretch it, it spins [faster]"), (13, "like a [skater] pulling arms in"),
        (14, "spin [infinitely] fast?"), (15, "nobody could [prove it]"), (16, "OpenAI: AI proved [it can]"),
        (17, "experts: [still checking]"), (18, "your chai is [safe]"), (19, "the maths? [we'll see]")]
def segs_of(s):
    out = []; parts = s.replace(']', '[').split('[')
    for k, p in enumerate(parts):
        if p: out.append((p, YEL if k % 2 else WHITE))
    return out
def caption_items(t):
    cur = None
    for k, (li, txt) in enumerate(CAPS):
        t0 = L(li) - 0.08
        t1 = L(CAPS[k + 1][0]) - 0.1 if k + 1 < len(CAPS) else DUR
        if t0 <= t < t1: cur = (txt, t - t0, t1 - t)
    if cur is None: return [], 0
    txt, age, left = cur
    a = min(1, age / 0.12, max(left, 0) / 0.12 + 0.0)
    pop = 1 + 0.08 * max(0, 1 - age / 0.15)
    return [dict(segs=segs_of(txt), xy=(540, 1488), size=int(64 * pop), stroke=7, alpha=a)], a

# ---------- helpers ----------
def tr_pos(f, i):
    k = int(np.clip(f - xray.F0, 0, P.shape[0] - 1)); return P[k, i]
def lerp(a, b, x): return a + (b - a) * x
def paste(layer, im, cx, cy, scale=1.0, alpha=1.0, tint=None):
    if scale != 1.0: im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))), Image.LANCZOS)
    if tint is not None:
        a = np.array(im).astype(np.float32); a[..., :3] = a[..., :3] / 255 * np.array(tint, np.float32)
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if alpha < 1:
        a = np.array(im); a[..., 3] = (a[..., 3] * alpha).astype(np.uint8); im = Image.fromarray(a)
    layer.alpha_composite(im, (int(cx - im.size[0] / 2), int(cy - im.size[1] / 2)))
    return im.size

def equation_layer(layer, t, y=360, hl=None, legend=None, alpha=1.0, red=0.0):
    keys = ['p1', 'p2', 'eq', 'p3', 'pl', 'p4']; sc = 0.74; gap = 10
    ws = [EQ[k].size[0] * sc for k in keys]; tot = sum(ws) + gap * (len(keys) - 1)
    x = 540 - tot / 2; centers = {}
    for k, w_ in zip(keys, ws):
        tint = None
        if hl and k in hl: tint = hl[k]
        if red > 0: tint = tuple(np.array(tint or (255, 255, 255)) * (1 - red) + np.array(RED) * red)
        paste(layer, EQ[k], x + w_ / 2, y, sc, alpha, tint); centers[k] = x + w_ / 2; x += w_ + gap
    return centers

def arrow(img, p0, p1, col, th=6):
    cv2.arrowedLine(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, th, cv2.LINE_AA, tipLength=0.3)

N = 512; CELL = 2 * np.pi / N
k = np.fft.fftfreq(N, 1 / N); kr = np.fft.rfftfreq(N, 1 / N); KX, KY = np.meshgrid(kr, k); K2 = KX ** 2 + KY ** 2; K2[0, 0] = 1
def forces_at(fi, x, y):
    """real forces on the drop from the solve: -grad p, nu lap u, velocity u"""
    wh = sf.rfft2(np.asarray(WV[fi], np.float32), workers=-1); ps = wh / K2; ps[0, 0] = 0
    uh = 1j * KY * ps; vh = -1j * KX * ps
    ir = lambda a: sf.irfft2(a, s=(N, N), workers=-1)
    ux, uy, vx, vy = ir(1j * KX * uh), ir(1j * KY * uh), ir(1j * KX * vh), ir(1j * KY * vh)
    rhs = -(ux * ux + 2 * uy * vx + vy * vy)
    ph = -sf.rfft2(rhs, workers=-1) / K2; ph[0, 0] = 0
    px, py = ir(1j * KX * ph), ir(1j * KY * ph)
    lu, lv = ir(-K2 * uh), ir(-K2 * vh); u, v = ir(uh), ir(vh)
    i, j = int(y / CELL) % N, int(x / CELL) % N
    return np.array([-px[i, j], -py[i, j]]), np.array([lu[i, j], lv[i, j]]), np.array([u[i, j], v[i, j]])

def cube(img, c, s, ang, col):
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * s
    a, b = ang, ang * 0.7 + 0.5
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    p = v @ Ry.T @ Rx.T; f = 600 / (600 + p[:, 2]); q = np.stack([c[0] + p[:, 0] * f, c[1] + p[:, 1] * f], 1)
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    for fc in sorted(faces, key=lambda fc: -p[fc, 2].mean()):
        cv2.fillConvexPoly(img, (q[fc] * 4).astype(np.int32), tuple(float(x) * 0.25 for x in col), cv2.LINE_AA, shift=2)
    for e in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        cv2.line(img, tuple((q[e[0]] * 4).astype(int)), tuple((q[e[1]] * 4).astype(int)), col, 2, cv2.LINE_AA, shift=2)

# spin accumulator for the tube
TT = np.arange(0, DUR + 0.1, 1 / 600)
def s_of(t):
    if t < L(12): return 1.0
    if t < L(13): return lerp(1, 4, ease(ramp(t, L(12), L(13) - 0.3)))
    if t < L(14): return lerp(4, 6, ease(ramp(t, L(13), L(14) - 0.2)))
    return 6 * np.exp(min((t - L(14)) / (35.0 - L(14)), 1.0) * np.log(1e6 / 6))
SV = np.array([min(s_of(t), 40) for t in TT]); TH = np.cumsum(2.0 * SV) / 600
def theta(t): return float(np.interp(t, TT, TH))

T3D_END, T3D_X = 3.72, 3.95        # 3D shot, then crossfade into the flat 2D view
T_BACK = None
P0 = dict(elev=30.0, yaw=-16.0, ppx=570.0, ppy=1080.0, dist=720.0)
P1 = dict(elev=33.0, yaw=-10.0, ppx=565.0, ppy=1070.0, dist=690.0)
PT = dict(elev=90.0, yaw=0.0, ppx=500.0, ppy=965.0, dist=600.0)
def _mix(a, b, x): return {k: a[k] + (b[k] - a[k]) * x for k in a}
def cam3d(t):
    tb = DUR - 1.45
    if t >= tb: return _mix(PT, P0, ease(ramp(t, tb, DUR)))
    if t < 2.3: return _mix(P0, P1, t / 2.3)
    return _mix(P1, PT, ease(ramp(t, 2.3, T3D_END)))
def frame3d(t, fi):
    c = cam3d(t)
    img, cam = cup3d.render(fi, t, c['elev'], c['yaw'], (c['ppx'], c['ppy']), dist=c['dist'])
    # steam above the cup
    cpt = cup3d.project(cam, np.array([[0, 0, cup3d.H_C]]))[0]
    st = chai.steam(t); st = np.roll(st, (int(cpt[1] - chai.CY), int(cpt[0] - chai.CX)), axis=(0, 1))
    img = img + st[..., None] * (1 - ramp(c['elev'], 60, 90))
    # cyan ring on the strongest milk swirl, drawn on the liquid surface
    p = tr_pos(fi, ring_i)
    X = (p[0] - np.pi) * SC0 / 10; Y = -(p[1] - np.pi) * SC0 / 10
    a = np.linspace(0, 2 * np.pi, 64); rr = 9.2 + 0.4 * np.sin(t * 6)
    pts = cup3d.project(cam, np.stack([X + rr * np.cos(a), Y + rr * np.sin(a), np.full_like(a, cup3d.H_L)], 1))
    u8 = to_u8(img); ov = u8.copy()
    cv2.polylines(ov, [(pts * 4).astype(np.int32)], True, CYAN, 3, cv2.LINE_AA, shift=2)
    return cv2.addWeighted(ov, 0.9, u8, 0.1, 0).astype(np.float32) / 255

def wire_cup(img, a):
    for r, th in [(520, 2), (chai.R_RIM, 2), (chai.R_LIQ, 3)]:
        cv2.circle(img, (chai.CX * 4, chai.CY * 4), r * 4, tuple(c / 255 * a for c in CYAN), th, cv2.LINE_AA, shift=2)
    cv2.circle(img, ((chai.CX - 330) * 4, (chai.CY + 330) * 4), 78 * 4, tuple(c / 255 * a * 0.7 for c in CYAN), 2, cv2.LINE_AA, shift=2)

def fmt_spin(s):
    if s < 10: return f"{s:.1f}"
    return f"{int(round(s)):,}"

# =================== frame ===================
def frame(t):
    items = []; pills = []; eq_draw = None
    shake = (0, 0)
    if t < T_C or t >= T_END:
        # ---------- chai / x-ray ----------
        if t >= T_END:
            fi = int(round(FH - (DUR - t) * 30)); fi = max(fi, 0)
            z = lerp(1.55, 1.0, ease(ramp(t, T_END, T_END + 1.4)))
        else:
            fi = FH + int(round(t * 30)); z = 1.0
        if t < T3D_END and t < T_END:
            base = frame3d(t, fi)
        elif t < T3D_X:
            a3 = ramp(t, T3D_END, T3D_X); base = frame3d(T3D_END, fi) * (1 - a3) + chai.render(fi, t) * a3
        elif t >= DUR - 1.45:
            a3 = ramp(t, DUR - 1.45, DUR - 1.25); base = chai.render(fi, t) * (1 - a3) + frame3d(t, fi) * a3
        else:
            base = chai.render(fi, t)
        if False:
            pass
        if T_XR0 <= t < T_C:
            ys = lerp(chai.CY - chai.R_RIM - 140, H + 60, ease(ramp(t, T_XR0, T_XR1)))
            xr = xray.render(fi, np.pi, np.pi, SC0, ox=chai.CX, oy=chai.CY, disk=(chai.CX, chai.CY, chai.R_LIQ), gridpx=48)
            wire_cup(xr, 0.9); xr = glow(xr, 5, 0.6)
            yy = np.arange(H, dtype=np.float32)[:, None, None]
            m = np.clip((ys - yy) / 6, 0, 1)
            base = base * (1 - m) + xr * m
            sl = np.exp(-((yy - ys) / 5) ** 2) * (t < T_XR1 + 0.1)
            base = base + sl * np.array(CYAN, np.float32) / 255 * 0.9 + np.exp(-((yy - ys) / 40) ** 2) * np.array(CYAN, np.float32) / 255 * 0.25 * (t < T_XR1 + 0.1)
        if z != 1.0:
            M = cv2.getRotationMatrix2D((chai.CX, chai.CY), 0, z); base = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = base
        # subject ring on the strongest milk swirl (hook)
        if T3D_X <= t < T_XR0:
            p = tr_pos(fi, ring_i)
            cx = chai.CX + (p[0] - np.pi) * SC0; cy = chai.CY + (p[1] - np.pi) * SC0
            cx = chai.CX + (cx - chai.CX) * z; cy = chai.CY + (cy - chai.CY) * z
            u8 = to_u8(img); u8 = ring(u8, (cx, cy), 92 + 4 * np.sin(t * 6), CYAN, 0.9, 3); img = u8.astype(np.float32) / 255
        if t < T_XR0 + 0.25:
            a = 1 - ramp(t, T_XR0, T_XR0 + 0.25)
            items += [dict(text="THIS SWIRL IS", xy=(540, 280), size=92, stroke=9, alpha=a),
                      dict(segs=[("WORTH ", WHITE), ("$1M", YEL)], xy=(540, 385), size=104, stroke=10, alpha=a)]
        if t >= L(2) - 0.1 and t < T_C:
            a = ramp(t, L(2) - 0.1, L(2) + 0.15)
            items += [dict(text="NAVIER–STOKES", xy=(540, 330), size=96, fill=CYAN, stroke=9, alpha=a),
                      dict(text="the rule of every fluid", xy=(540, 420), size=40, font=FM, stroke=5, alpha=a)]
        if t >= T_END:
            a = ramp(t, DUR - 2.4, DUR - 2.1) * (1 - ramp(t, DUR - 0.35, DUR - 0.05))
            items += [dict(text="nature is running on code", xy=(540, 300), size=46, fill=CYAN, stroke=6, alpha=a),
                      dict(segs=[("next: how does a wing ", WHITE), ("really", YEL), (" lift a plane?", WHITE)], xy=(540, 365), size=36, font=FM, stroke=5, alpha=a)]
            ain = ramp(t, T_END, T_END + 0.6) * (1 - ramp(t, DUR - 0.5, DUR))
            img = img  # (loop: final frames = chai frames just before frame 0)
    elif t < T_WING:
        # ---------- C: one drop, its forces (real, from the solve) ----------
        fi = FH + int(round(t * 30))
        pd = tr_pos(fi, drop_i)
        zc = ease(ramp(t, T_C, T_C + 1.3))
        cx = lerp(np.pi, pd[0], zc); cy = lerp(np.pi, pd[1], zc); sc = lerp(SC0, SC0 * 3.3, zc)
        ox = lerp(chai.CX, 540, zc); oy = lerp(chai.CY, 1010, zc)
        disk_r = lerp(chai.R_LIQ, 2400, zc)
        img = xray.render(fi, cx, cy, sc, ox=ox, oy=oy, disk=(ox + (np.pi - cx) * sc, oy + (np.pi - cy) * sc, disk_r), gridpx=int(48 * sc / SC0))
        lay = np.zeros_like(img)
        dpx = (ox + (pd[0] - cx) * sc, oy + (pd[1] - cy) * sc)
        # drop path trail (motion)
        if t >= L(6) - 0.1:
            a = ramp(t, L(6) - 0.1, L(6) + 0.3)
            pts = [tr_pos(f, drop_i) for f in range(max(fi - 70, 150), fi + 1)]
            pts = np.array([(ox + (q[0] - cx) * sc, oy + (q[1] - cy) * sc) for q in pts])
            cv2.polylines(lay, [(pts * 4).astype(np.int32)], False, (a, a, a), 3, cv2.LINE_AA, shift=2)
        cube(lay, dpx, 26 * zc + 1, t * 1.3, tuple(c / 255 for c in CYAN))
        if zc > 0.9:
            fp, fv, uu = forces_at(fi, pd[0], pd[1])
            def dirv(v, L_): n = np.linalg.norm(v) + 1e-9; return v / n * L_
            if t >= L(4) - 0.1:
                a = ramp(t, L(4) - 0.1, L(4) + 0.25); d = dirv(fp, 150)
                arrow(lay, (dpx[0] - d[0] * 1.5, dpx[1] - d[1] * 1.5), (dpx[0] - d[0] * 0.35, dpx[1] - d[1] * 0.35), tuple(c / 255 * a for c in AMBER), 9)
                pills.append(("PRESSURE", (dpx[0] - d[0] * 1.6, dpx[1] - d[1] * 1.6 - 40), AMBER, a))
            if t >= L(5) - 0.1:
                a = ramp(t, L(5) - 0.1, L(5) + 0.25); d = dirv(fv, 120)
                arrow(lay, (dpx[0] + d[0] * 0.35, dpx[1] + d[1] * 0.35), (dpx[0] + d[0] * 1.3, dpx[1] + d[1] * 1.3), tuple(c / 255 * a for c in CYAN), 7)
                pills.append(("STICKINESS", (dpx[0] + d[0] * 1.45, dpx[1] + d[1] * 1.45 + 40), CYAN, a))
            if t >= L(6) - 0.1:
                a = ramp(t, L(6) - 0.1, L(6) + 0.25); d = dirv(uu, 130)
                arrow(lay, (dpx[0] + d[0] * 0.35, dpx[1] + d[1] * 0.35), (dpx[0] + d[0] * 1.35, dpx[1] + d[1] * 1.35), (a, a, a), 7)
        img = glow(img + lay, 5, 0.55)
        if zc > 0.6 and t < L(4) - 0.1:
            pills.append(("1 DROP", (dpx[0], dpx[1] + 95), CYAN, ramp(t, T_C + 0.8, T_C + 1.2)))
        # equation
        if t < L(3) + 1.6:
            a = ramp(t, T_C + 0.1, T_C + 0.4) * (1 - ramp(t, L(3) + 1.4, L(3) + 1.6))
            eq_draw = ('newton', a)
        else:
            hl = {'p1': (255, 255, 255), 'p2': (255, 255, 255)}
            if t >= L(4) - 0.1: hl['p3'] = AMBER
            if t >= L(5) - 0.1: hl['p4'] = CYAN
            eq_draw = ('ns', ramp(t, L(3) + 1.5, L(3) + 1.8), hl)
    elif t < T_E:
        # ---------- D: real life ----------
        if t < T_STORM:
            img = others.wing(int((t - T_WING) * 30) + 5)
            if t >= L(8) - 0.1: pills.append(("PLANES", (540, 700), WHITE, ramp(t, L(8) - 0.1, L(8) + 0.2)))
            if t >= L(8) + 0.4: pills.append(("lift = a pressure push", (540, 1270), AMBER, ramp(t, L(8) + 0.4, L(8) + 0.7)))
        elif t < T_BLOOD:
            img = others.storm(int((t - T_STORM) * 30) + 10)
            pills.append(("WEATHER", (540, 380), WHITE, ramp(t, T_STORM, T_STORM + 0.25)))
        else:
            img, prof, pulse = others.blood(t - T_BLOOD + 0.3)
            img = img + prof * 0.9
            pills.append(("BLOOD", (540, 380), WHITE, ramp(t, T_BLOOD, T_BLOOD + 0.25)))
            pills.append(("fastest in the middle", (540, 1110), WHITE, ramp(t, T_BLOOD + 0.4, T_BLOOD + 0.7)))
        img = img.astype(np.float32)
        if t < T_WING + 0.25:  # flash cut in
            img = img + (1 - ramp(t, T_WING, T_WING + 0.25)) * 0.35
    else:
        # ---------- E/F/G: the million-dollar part ----------
        s = s_of(t); red = ease(ramp(t, L(14), L(14) + 1.2))
        if t >= T_NEWS: red = 1.0
        tub_a = ramp(t, T_E, T_E + 0.5) * (1 - 0.65 * ease(ramp(t, T_NEWS, T_NEWS + 0.6)))
        sv = min(s, 40)
        lay, r, Lt = tube.render(theta(t), sv, red, trail=min(0.25 * sv, 2.6), alpha=tub_a)
        yy_ = np.arange(H, dtype=np.float32)[:, None, None]
        lay = lay * np.clip((yy_ - 540) / 60, 0, 1) * np.clip((1400 - yy_) / 60, 0, 1)
        img = lay + np.array([0.018, 0.02, 0.03], np.float32)
        if L(14) + 1.2 <= t < L(15):   # blow-up flash
            fl = np.exp(-((np.arange(W, dtype=np.float32)[None, :] - tube.CXT) / (60 + 400 * ramp(t, L(14) + 1.2, L(15)))) ** 2)
            img = img + (fl[..., None] * np.array(RED, np.float32) / 255 * 0.5 * (0.6 + 0.4 * np.sin(t * 40)))[None, ...].squeeze(0) * np.clip((yy_ - 540) / 60, 0, 1) * np.clip((1400 - yy_) / 60, 0, 1)
        if L(12) - 0.1 <= t < L(14) + 0.6:   # stretch arrows
            a = ramp(t, L(12) - 0.1, L(12) + 0.2) * (1 - ramp(t, L(14) + 0.3, L(14) + 0.6))
            yt = tube.CYT - Lt / 2 * np.cos(0.32) - 30; yb = tube.CYT + Lt / 2 * np.cos(0.32) + 30
            yt = max(yt, 600); yb = min(yb, 1340)
            arrow(img, (tube.CXT, yt + 70), (tube.CXT, yt - 60), tuple(c / 255 * a for c in AMBER), 8)
            arrow(img, (tube.CXT, yb - 70), (tube.CXT, yb + 60), tuple(c / 255 * a for c in AMBER), 8)
            pills.append(("STRETCH", (tube.CXT, yt - 105), AMBER, a))
        img = glow(img, 8, 0.65)
        if t < L(12) + 0.2:
            a = ramp(t, T_E + 0.1, T_E + 0.4) * (1 - ramp(t, L(12), L(12) + 0.2))
            items += [dict(text="$1,000,000", xy=(540, 330), size=120, fill=YEL, stroke=10, alpha=a),
                      dict(text="one question about this equation", xy=(540, 430), size=38, font=FM, stroke=5, alpha=a)]
        if L(12) <= t < T_NEWS:
            col = RED if s > 6.5 else WHITE
            st = "∞ ?" if s > 9e5 else "×" + fmt_spin(s)
            items += [dict(segs=[("SPIN ", WHITE), (st, col)], xy=(760, 640), size=58, stroke=7, alpha=ramp(t, L(12), L(12) + 0.2))]
            if s < 9e5:
                rr = r if s < 40 else 150 / np.sqrt(s)
                items += [dict(text=f"width ×{1/np.sqrt(s):.3f}" if s < 1000 else f"width ×{1/np.sqrt(s):.5f}", xy=(760, 705), size=30, font=FM, stroke=4, alpha=ramp(t, L(12), L(12) + 0.2))]
            eq_draw = ('kelvin', ramp(t, L(12) + 0.1, L(12) + 0.4) * (1 - ramp(t, T_NEWS - 0.2, T_NEWS)), red)
        if L(14) + 1.5 <= t < L(14) + 2.4:   # the maths "breaks"
            k_ = int(t * 30); rs = np.random.default_rng(k_)
            shake = (int(rs.integers(-14, 15)), int(rs.integers(-14, 15)))
        if L(15) - 0.1 <= t < T_NEWS:
            pills.append(("A $1M PRIZE SINCE 2000", (540, 1270), AMBER, ramp(t, L(15) - 0.1, L(15) + 0.2)))
        if t >= T_NEWS:
            a = ramp(t, T_NEWS, T_NEWS + 0.35)
            panel = np.zeros((H, W), np.float32); cv2.rectangle(panel, (90, 560), (990, 1150), 1.0, -1)
            panel = cv2.GaussianBlur(panel, (0, 0), 6)
            img = img * (1 - 0.72 * a * panel[..., None]) + a * panel[..., None] * np.array([0.02, 0.03, 0.05], np.float32)
            items += [dict(text="SEPTEMBER 2026", xy=(540, 640), size=40, fill=CYAN, stroke=5, alpha=a),
                      dict(text="OpenAI says its AI proved", xy=(540, 735), size=52, stroke=6, alpha=a),
                      dict(segs=[("the maths ", WHITE), ("CAN break", YEL)], xy=(540, 820), size=72, stroke=8, alpha=a),
                      dict(text="(if the fluid gets an outside push)", xy=(540, 900), size=36, font=FM, fill=(200, 205, 215), stroke=4, alpha=a)]
            if t >= L(17) - 0.1:
                b = ramp(t, L(17) - 0.1, L(17) + 0.2)
                pills.append(("EXPERTS: STILL CHECKING", (540, 1010), AMBER, b))
                items += [dict(text="Clay Institute review ongoing · prize not awarded", xy=(540, 1085), size=28, font=FM, fill=(200, 205, 215), stroke=4, alpha=b)]
    # ---------- compose ----------
    u8 = to_u8(img)
    if shake != (0, 0):
        u8 = np.roll(u8, shake, axis=(0, 1)); u8[..., 0] = np.roll(u8[..., 0], 6, axis=1); u8[..., 2] = np.roll(u8[..., 2], -6, axis=1)
    cap, ca = caption_items(t)
    if cap and t >= T_XR0:
        fl = u8.astype(np.float32); bd = band(None, 1420, 1560, 0.55 * ca); u8 = (fl * (1 - bd[..., None])).astype(np.uint8)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    if eq_draw:
        if eq_draw[0] == 'newton':
            a = eq_draw[1]; paste(layer, EQ['newton'], 540, 340, 0.85, a)
            d = ImageDraw.Draw(layer); d.text((540, 425), "Newton: force = mass × acceleration", font=font(34, FM), fill=(220, 225, 235, int(255 * a)), anchor='mm', stroke_width=4, stroke_fill=(5, 6, 10, int(230 * a)))
        elif eq_draw[0] == 'ns':
            a = eq_draw[1]; cen = equation_layer(layer, t, 345, eq_draw[2], alpha=a)
            d = ImageDraw.Draw(layer)
            leg = [('p1', "change", WHITE, L(3) + 1.6), ('p2', "motion", WHITE, L(6) - 0.1), ('p3', "pressure", AMBER, L(4) - 0.1), ('p4', "stickiness", CYAN, L(5) - 0.1)]
            for kk, txt, col, t0 in leg:
                if t >= t0:
                    b = ramp(t, t0, t0 + 0.25) * a
                    d.text((cen[kk], 430), txt, font=font(30), fill=tuple(col) + (int(255 * b),), anchor='mm', stroke_width=4, stroke_fill=(5, 6, 10, int(230 * b)))
            d.text((540, 485), "for liquids and air", font=font(30, FM), fill=(200, 205, 215, int(200 * a)), anchor='mm', stroke_width=4, stroke_fill=(5, 6, 10, int(200 * a)))
        elif eq_draw[0] == 'kelvin':
            a, rd = eq_draw[1], eq_draw[2]
            paste(layer, EQ['kelvin'], 540, 330, 0.8, a, tint=tuple(np.array((255, 255, 255)) * (1 - rd) + np.array(RED) * rd))
            d = ImageDraw.Draw(layer); d.text((540, 410), "spin × width² stays the same", font=font(34, FM), fill=(220, 225, 235, int(255 * a)), anchor='mm', stroke_width=4, stroke_fill=(5, 6, 10, int(230 * a)))
    for (txt, xy, col, a) in pills:
        if a > 0.01: layer.alpha_composite(pill(txt, (xy[0], xy[1]), col, 34, a))
    if t >= T_XR0 or t < 0:
        pass
    items += cap if t >= T_XR0 else []
    items.append(dict(text="SOURCE CODE #13", xy=(540, 178), size=30, fill=CYAN, stroke=4, alpha=0.9))
    layer.alpha_composite(text_layer(items))
    return comp(u8, layer)

if __name__ == '__main__':
    a, b, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(out, exist_ok=True)
    for f in range(a, b, step):
        fr = frame(f / FPS)
        cv2.imwrite(f'{out}/f{f:05d}.png', cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
