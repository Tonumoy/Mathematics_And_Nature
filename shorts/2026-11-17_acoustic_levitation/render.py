# SC#12 "Can sound lift water?" — own 3D render: TinyLev-style acoustic levitator, ray-traced water drops,
# standing-wave x-ray, stack of drops at pressure nodes, loudness dial twist. Units: mm. Canvas 2160x3840 (2x).
import numpy as np, math, cv2, os
from PIL import Image, ImageDraw
import lib
from lib import CYAN, AMBER, RED, GREEN, YELLOW, WHITE
import timeline as X
s, e = X.s, X.e
S2 = int(os.environ.get("S2", 2)); WW, HH = 1080 * S2, 1920 * S2; FPS = 30
F = 1.30 * WW
C_SOUND = 343.0; FREQ = 40000.0; LAM = C_SOUND / FREQ * 1000.0      # 8.575 mm
NODE = LAM / 2                                                        # 4.29 mm
R = 46.0                                                              # cap radius (focus at origin)
RD = 1.6                                                             # drop radius mm
NODES7 = [j * NODE for j in (0, 1, -1, 2, -2, 3, -3)]
rng = np.random.default_rng(12)

def sm(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def ramp(t, a, b): return float(np.clip((t - a) / max(b - a, 1e-6), 0, 1))
def nrm(v): return v / np.linalg.norm(v, axis=-1, keepdims=True)

# ---------------- camera (keyframes, smoothstep between = continuous) ----------------
def keys():
    return [  # t, dist, yaw, pitch, target_y, blur
        (0.0, 26.0, 20, 0, 0.0, 2.5),
        (s('L02') + 0.2, 28.0, 30, 1, 0.0, 2.5),
        (e('L02') + 0.1, 205.0, 55, 12, 0.0, 0),
        (s('L04'), 120.0, 75, 6, 0.0, 0),
        (s('L05') + 0.3, 42.0, 88, 4, 0.0, 2),
        (s('L06') + 0.6, 50.0, 95, 4, 0.0, 2),
        (e('L06'), 92.0, 105, 5, 0.0, 1),
        (s('L07') + 0.6, 82.0, 112, 5, -2.0, 1),
        (e('L08') + 1.4, 82.0, 125, 5, 0.0, 1),
        (s('L09') + 2.0, 235.0, 200, 16, 0.0, 0),
        (e('L09') + 0.2, 150.0, 300, 10, 0.0, 2),
        (X.T, 26.0, 380, 0, 0.0, 2.5),
    ]
KEYS = keys()
def cam(t):
    for i in range(len(KEYS) - 1):
        a, b = KEYS[i], KEYS[i + 1]
        if t <= b[0]:
            u = sm((t - a[0]) / max(b[0] - a[0], 1e-6)); break
    else:
        a = b = KEYS[-1]; u = 0
    v = [a[j] + (b[j] - a[j]) * u for j in range(1, 6)]
    # distance interpolates in log space (smooth dolly)
    v[0] = math.exp(math.log(a[1]) + (math.log(b[1]) - math.log(a[1])) * u)
    dist, yaw, pitch, ty, blur = v
    yaw += 3 * math.sin(t * 0.7)
    yr, pr = math.radians(yaw), math.radians(pitch)
    tgt = np.array([0.0, ty, 0.0])
    pos = tgt + dist * np.array([math.sin(yr) * math.cos(pr), math.sin(pr), math.cos(yr) * math.cos(pr)])
    fw = nrm(tgt - pos); rt = nrm(np.cross(fw, [0, 1, 0])); up = np.cross(rt, fw)
    return pos, np.stack([rt, up, fw]), blur, dist

def project(P, pos, M):
    q = (np.asarray(P, np.float64) - pos) @ M.T
    z = np.maximum(q[..., 2], 1e-3)
    return WW / 2 + F * q[..., 0] / z, HH / 2 - F * q[..., 1] / z, q[..., 2]

# ---------------- geometry of the levitator ----------------
def transducers():
    out = []
    for th, n in ((0, 1), (15, 6), (29, 12), (43, 17)):
        for k in range(n):
            ph = 2 * math.pi * k / n + (0.3 if n == 12 else 0)
            for sgn in (1, -1):
                d = np.array([math.sin(math.radians(th)) * math.cos(ph), sgn * math.cos(math.radians(th)), math.sin(math.radians(th)) * math.sin(ph)])
                out.append(d)
    return np.array(out)
TD = transducers()                      # unit directions from focus to each transducer
def circle(c, axis, r, n=22):
    a = nrm(np.cross(axis, [0.3, 0.1, 0.9]) if abs(axis[2]) < 0.9 else np.cross(axis, [1, 0, 0]))
    b = np.cross(axis, a)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + r * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)

LIGHT = nrm(np.array([-0.55, 0.65, 0.55]))
def shade(base, n, view, spec=0.6, amb=0.22):
    dif = max(0.0, float(np.dot(n, LIGHT)))
    h = nrm(LIGHT - view); sp = max(0.0, float(np.dot(n, h))) ** 40 * spec
    rim = (1 - abs(float(np.dot(n, -view)))) ** 3 * 0.35
    c = np.array(base) * (amb + 0.85 * dif) + 255 * sp
    return c + np.array([60, 190, 255]) * rim

def shell_quads():
    qs = []
    for sgn in (1, -1):
        ths = np.radians(np.linspace(0, 52, 9)); phs = np.linspace(0, 2 * np.pi, 33)
        for i in range(8):
            for j in range(32):
                pts = []
                for (a, b) in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    pts.append([math.sin(ths[a]) * math.cos(phs[b]), sgn * math.cos(ths[a]), math.sin(ths[a]) * math.sin(phs[b])])
                pts = np.array(pts)
                qs.append((pts, sgn))
    return qs
SHELL = shell_quads()

def draw_levitator(pos, M, amber_glow):
    """painter's algorithm, uint8 canvas with AA. returns float32 RGB layer + coverage."""
    img = np.zeros((HH, WW, 3), np.uint8); cov = np.zeros((HH, WW), np.uint8)
    items = []
    view_of = lambda p: nrm(p - pos)
    # shell (outer bowl, white printed plastic), radius R+7..R+8 ; inner face visible
    for pts, sgn in SHELL:
        P3 = pts * (R + 7.5); c3 = P3.mean(0); n = -nrm(c3)                     # inner normal toward focus
        v = view_of(c3)
        if np.dot(n, v) > 0: n = -n                                              # outside seen
        items.append((np.linalg.norm(c3 - pos), 'poly', P3, shade((200, 205, 212), n, v, 0.15, 0.28)))
    # struts + base
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.5
        x, z = (R + 12) * math.cos(a), (R + 12) * math.sin(a)
        P3 = np.array([[x, -R - 6, z], [x, R + 6, z]])
        items.append((np.linalg.norm(np.array([x, 0, z]) - pos), 'rod', P3, shade((190, 195, 205), nrm(np.array([x, 0, z])), view_of(np.array([x, 0, z])), 0.5)))
    # transducers: cylinder body (aluminium) + black mesh face + silver rim
    for d in TD:
        face_c = d * R; back_c = d * (R + 7); ax = -d                            # axis points at focus
        c3 = (face_c + back_c) / 2
        v = view_of(c3)
        body = np.vstack([circle(face_c, ax, 5.0), circle(back_c, ax, 5.0)])
        items.append((np.linalg.norm(c3 - pos) + 0.01, 'hull', body, shade((150, 158, 170), nrm(c3 - np.dot(c3, d) * d + 1e-6), v, 0.9, 0.3)))
        facing = np.dot(ax, -v)                                                   # >0: face toward camera
        if facing > 0:
            glow = amber_glow * (0.6 + 0.4 * facing)
            items.append((np.linalg.norm(face_c - pos) - 0.5, 'face', (face_c, ax, glow), None))
    items.sort(key=lambda it: -it[0])
    for dist, kind, G, col in items:
        if kind == 'poly' or kind == 'hull':
            x, y, z = project(G, pos, M)
            if (z < 1).any(): continue
            pts = np.stack([x, y], 1).astype(np.float32)
            hull = cv2.convexHull(pts) if kind == 'hull' else pts
            c = tuple(int(v) for v in np.clip(col, 0, 255))
            cv2.fillConvexPoly(img, (hull * 16).astype(np.int32), c, cv2.LINE_AA, 4)
            cv2.fillConvexPoly(cov, (hull * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
        elif kind == 'rod':
            x, y, z = project(G, pos, M)
            if (z < 1).any(): continue
            w = max(2, int(F * 2.2 / z.mean()))
            c = tuple(int(v) for v in np.clip(col, 0, 255))
            cv2.line(img, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), c, w, cv2.LINE_AA)
            cv2.line(cov, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), 255, w, cv2.LINE_AA)
        else:
            fc, ax, glow = G
            for r, colr, gk in ((5.0, (205, 210, 218), 0), (4.3, (20, 21, 24), 0.18), (3.2, (24, 25, 28), 0.85), (2.6, (20, 21, 24), 0.25), (1.2, (60, 62, 66), 0.6)):
                x, y, z = project(circle(fc + ax * 0.05, ax, r, 26), pos, M)
                if (z < 1).any(): break
                cc = np.array(colr, float)
                cc = cc + np.array([255, 140, 40]) * glow * gk
                cv2.fillConvexPoly(img, (np.stack([x, y], 1) * 16).astype(np.int32), tuple(int(v) for v in np.clip(cc, 0, 255)), cv2.LINE_AA, 4)
    return img.astype(np.float32), cov.astype(np.float32) / 255

# ---------------- background: dark lab, bokeh, bench ----------------
def _bokeh():
    W2, H2 = WW * 2, HH
    bg = np.zeros((H2, W2, 3), np.float32)
    yy = np.linspace(0, 1, H2)[:, None, None]
    bg += np.array([6, 9, 16]) * (1 - yy) + np.array([14, 12, 14]) * yy
    lay = np.zeros_like(bg)
    for i in range(45):
        x, y = rng.uniform(0, W2), rng.uniform(0.15, 0.75) * H2
        r = rng.uniform(30, 120)
        col = [(255, 170, 70), (80, 200, 255), (255, 230, 200), (140, 120, 255)][rng.integers(0, 4)]
        cv2.circle(lay, (int(x), int(y)), int(r), tuple(c * rng.uniform(0.05, 0.16) for c in col), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 6)
    return bg + lay
BOKEH = None
def background(yaw):
    global BOKEH
    if BOKEH is None: BOKEH = _bokeh()
    off = int((yaw % 360) / 360 * WW * 2)
    b = np.roll(BOKEH, -off, axis=1)[:, :WW]
    return b.copy()

# ---------------- ray-traced water drop (spheroid) ----------------
def env(d, glow=0.5):
    """environment radiance for direction(s) d (...,3): softbox + the two transducer arrays (amber faces, silver rims)"""
    soft = np.exp(-((d[..., 0] + 0.55) ** 2 + (d[..., 1] - 0.62) ** 2 + (d[..., 2] - 0.55) ** 2) / 0.06)
    out = 8 + soft[..., None] * np.array([1500, 1520, 1550])
    el = np.abs(d[..., 1]); az = np.arctan2(d[..., 2], d[..., 0])
    th = np.arccos(np.clip(el, 0, 1))                                          # angle from axis
    arr = (th < np.radians(50)).astype(np.float64) * np.clip((np.radians(50) - th) / 0.05, 0, 1)
    ring = th / np.radians(14.5); rf = ring - np.round(ring)
    nz = np.maximum(np.round(ring), 1) * 6
    af = az / (2 * np.pi) * nz; af = af - np.round(af)
    dd = np.sqrt(rf ** 2 + (af * 0.9) ** 2)
    face = (dd < 0.3) * arr; rim = ((dd >= 0.3) & (dd < 0.4)) * arr; body = ((dd >= 0.4)) * arr
    out = out + face[..., None] * np.array([255, 140, 40]) * (0.5 + 1.3 * glow) + rim[..., None] * np.array([210, 215, 225]) * 0.9 \
        + body[..., None] * np.array([55, 58, 66])
    out = out + np.clip(-d[..., 2], 0, 1)[..., None] ** 3 * np.array([20, 70, 110])
    return out

GLOW = [0.5]
def drop(frame, cen, a, c, pos, M, alpha=1.0, tint=1.0):
    cx, cy, cz = project(cen, pos, M)
    if cz < 2: return
    rad = F * max(a, c) / cz * 1.15 + 3
    x0, x1 = int(max(cx - rad, 0)), int(min(cx + rad, WW)); y0, y1 = int(max(cy - rad, 0)), int(min(cy + rad, HH))
    if x1 <= x0 + 1 or y1 <= y0 + 1: return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dcam = np.stack([(xs - WW / 2) / F, -(ys - HH / 2) / F, np.ones_like(xs)], -1)
    d = nrm(dcam @ M)                                                            # world dirs
    sc = np.array([a, c, a])
    o = (pos - cen) / sc; dd = d / sc
    A = (dd * dd).sum(-1); B = 2 * (dd * o).sum(-1); Cc = (o * o).sum() - 1
    disc = B * B - 4 * A * Cc
    pix_w = cz / F                                                               # mm per pixel at drop
    edge = np.clip(disc / (4 * A) / (2 * (pix_w / min(a, c)) + 1e-9) + 0.5, 0, 1) # ~1px AA
    m = edge > 0
    if not m.any(): return
    tt = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
    P = pos + d * tt[..., None]
    n = nrm((P - cen) / (sc * sc))
    cosi = np.clip(-(d * n).sum(-1), 0, 1)
    Fr = 0.02 + 0.98 * (1 - cosi) ** 5
    refl = env(d + 2 * cosi[..., None] * n, GLOW[0])
    # refraction: inverted, magnified view of what is behind (ball lens) sampled from the frame
    rpx = F * max(a, c) / cz
    k = 1.7
    sx = np.clip(cx - (xs - cx) * k * (1 + 0.5 * (1 - cosi)), 0, WW - 1).astype(np.int32)
    sy = np.clip(cy - (ys - cy) * k * (1 + 0.5 * (1 - cosi)), 0, HH - 1).astype(np.int32)
    eta = 1 / 1.333
    k1 = 1 - eta * eta * (1 - cosi ** 2)
    tdir = nrm(eta * d + (eta * cosi - np.sqrt(np.maximum(k1, 0)))[..., None] * n)
    # exit ray of a ball lens ~ mirror the entry deviation (inverted view)
    exitd = nrm(2 * (tdir * d).sum(-1)[..., None] * tdir - d)
    refr = 0.35 * env(exitd, GLOW[0]) * np.array([0.9, 0.97, 1.0]) + 0.75 * frame[sy, sx] + np.array([6, 12, 18])
    refr *= (0.2 + 0.8 * cosi ** 0.6)[..., None]
    rimg = np.exp(-cosi / 0.08)[..., None] * np.array([120, 200, 255]) * 0.9                               # dark rim (TIR edge)
    caus = np.exp(-(((n[..., 0] - 0.45) ** 2 + (n[..., 1] + 0.45) ** 2) / 0.05)) * cosi    # caustic opposite the light
    hl = np.clip((d + 2 * cosi[..., None] * n) @ LIGHT, 0, 1) ** 400 * 2600
    col = (1 - Fr)[..., None] * refr + Fr[..., None] * refl + (caus * 260)[..., None] * np.array([1.0, 0.95, 0.85]) + hl[..., None] + rimg
    al = (edge * alpha)[..., None]
    reg = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = reg * (1 - al) + col * al * tint

# ---------------- simulation state ----------------
def stack_count(t):
    a, b = s('L06') + 0.4, e('L06') + 1.2
    return 1 + int(np.clip((t - a) / (b - a), 0, 0.999) * 7) if t >= a else 1

def dial_value(t):
    """loudness 0..1 ; 0.5 = holds"""
    a = s('L07') + 0.3; f0 = s('L08'); mid = (s('L08') + e('L08')) / 2; g = e('L08') + 0.3
    if t < a: return 0.5
    if t < f0: return 0.5 - 0.0 * ramp(t, a, f0)
    if t < mid: return 0.5 - 0.38 * sm(ramp(t, f0, f0 + 0.6))
    if t < g: return 0.12 + 0.83 * sm(ramp(t, mid - 0.2, mid + 0.6))
    return 0.95 - 0.45 * sm(ramp(t, g, g + 0.7))

def drops_state(t):
    """list of (center, a, c, alpha, tint) + burst droplets list"""
    out = []; bursts = []
    n = stack_count(t) if t < s('L07') + 0.3 else 7
    if t < s('L06'): n = 1
    f0 = s('L08'); mid = (s('L08') + e('L08')) / 2; g = e('L08') + 0.3
    for i in range(n):
        y = NODES7[i]
        a = c = RD * (0.92 if i else 1.0)
        al = 1.0
        if i and t < s('L07'):                                   # slide in from the side
            ti = s('L06') + 0.4 + (i) * (e('L06') + 1.2 - s('L06') - 0.4) / 7
            u = sm(ramp(t, ti, ti + 0.45)); x = -14 * (1 - u); al = u
        else:
            x = 0.0
        # nudge test on the main drop (L05)
        if i == 0 and s('L05') <= t < s('L06'):
            tau = t - s('L05') - 0.15
            if tau > 0: y += 2.2 * math.exp(-2.2 * tau) * math.cos(9 * tau) if tau > 0.35 else 2.2 * sm(tau / 0.35)
        wob = 0.05 * math.sin(t * 23 + i * 1.7)
        a *= 1 + wob; c *= 1 - 2 * wob
        # twist: fall (soft), flatten + burst (loud), reform (green)
        if f0 + 0.3 <= t < mid + 0.25:
            tau = (t - f0 - 0.3) * 0.055                                         # slow motion x18
            y -= 0.5 * 9810 * tau ** 2
            if y < -R + 6: al = 0
        elif mid + 0.25 <= t < g + 0.7:
            u = ramp(t, mid + 0.25, mid + 0.25 + 1.1)                          # appears again, flattens
            if u < 0.85:
                q = 1 - 0.68 * sm(u / 0.85); a = RD * q ** (-0.5); c = RD * q
                al = sm(ramp(t, mid + 0.25, mid + 0.45))
            else:
                al = 0
                tb = t - (mid + 0.25 + 0.935)
                rr = np.random.default_rng(100 + i)
                for kk in range(18):
                    ang = rr.uniform(0, 2 * np.pi); sp = rr.uniform(8, 22); vy = rr.uniform(-4, 4)
                    rb = rr.uniform(0.3, 0.65)
                    p = np.array([x + math.cos(ang) * (RD * 1.8 + sp * tb), y + vy * tb - 20 * tb * tb, math.sin(ang) * (RD * 1.8 + sp * tb)])
                    bursts.append((p, rb, max(0.0, 1 - tb / 1.4)))
        elif g + 0.7 <= t < g + 1.4:
            al = sm(ramp(t, g + 0.7, g + 1.2))
        if i and t > e('L09'): al *= 1 - sm(ramp(t, e('L09'), e('L09') + 1.2))
        if al > 0.01: out.append((np.array([x, y, 0.0]), a, c, al))
    return out, bursts

# ---------------- standing-wave x-ray slice ----------------
def wave_tex(t, phase_mode):
    """texture 600 x 2000 (x,y) spanning x [-30,30], y [-R,R]: pressure field"""
    h, w = 1400, 420
    y = np.linspace(R, -R, h)[:, None]
    k = 2 * np.pi / LAM; om = 2 * np.pi * 0.9 * t                              # shown slowed: 40 kHz -> 0.9 Hz
    front = phase_mode                                                          # 0..1 how far waves travelled
    top = np.sin(k * y + om) * (y > R - 2 * R * front)
    bot = np.sin(-k * y + om) * (y < -R + 2 * R * front)
    p = (top + bot) / 2
    amp = np.abs(np.sin(k * y)) * min(1, front * 1.4)
    col = np.zeros((h, 1, 3), np.float32)
    col += np.abs(p)[..., None] * np.array([255, 160, 70]) * 0.9
    nodes = np.exp(-(np.sin(k * y) ** 2) / 0.003) * max(0, front * 2 - 1)
    col += nodes[..., None] * np.array([90, 225, 255]) * 1.5
    a = np.clip(0.06 + 0.32 * np.abs(p) + 0.9 * nodes, 0, 1)
    xs = np.linspace(-1, 1, w)[None, :]
    fade = np.exp(-xs ** 2 / 0.25) * np.clip((1 - np.abs(xs)) * 4, 0, 1)
    rgba = np.concatenate([np.repeat(col, w, 1), (a * fade)[..., None]], -1)
    return rgba.astype(np.float32)

def xray(frame, t, pos, M, amt, front):
    if amt <= 0.01: return
    tex = wave_tex(t, front)
    h, w = tex.shape[:2]
    rt = M[0]; half = 10
    corners = [(-half, R), (half, R), (half, -R), (-half, -R)]
    P3 = np.array([[rt[0] * x, yv, rt[2] * x] for x, yv in corners])
    x, y, z = project(P3, pos, M)
    if (z < 2).any(): return
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(np.stack([x, y], 1)))
    warped = cv2.warpPerspective(tex, Hm, (WW, HH), flags=cv2.INTER_LINEAR)
    al = warped[..., 3:4] * amt
    frame[:] = frame * (1 - al * 0.75) + warped[..., :3] * al

def travel_lines(lay, t, pos, M, amt, front):
    """two amber travelling sine curves meeting, then white standing sum"""
    if amt <= 0.01: return
    k = 2 * np.pi / LAM; om = 2 * np.pi * 0.9 * t
    y = np.linspace(R - 3, -R + 3, 900); rt = M[0]
    def poly(xo, vals, col, wdt):
        P3 = np.stack([rt[0] * (xo + vals), y, rt[2] * (xo + vals)], 1)
        X_, Y_, Z_ = project(P3, pos, M)
        pts = np.stack([X_, Y_], 1)
        cv2.polylines(lay, [(pts * 16).astype(np.int32)], False, tuple(c * amt for c in col), wdt, cv2.LINE_AA, 4)
    top = np.sin(k * y + om) * (y > R - 2 * R * front); bot = np.sin(-k * y + om) * (y < -R + 2 * R * front)
    poly(-18, 2.2 * top, AMBER, 5); poly(-18, 2.2 * bot, (255, 120, 40), 5)
    if front > 0.5: poly(18, 2.6 * (top + bot) / 2 * 1.0, WHITE, 6)

# ---------------- particles (fog collecting at nodes) ----------------
NP = 700
PX = rng.uniform(-1, 1, (NP, 3)); PX[:, 1] = rng.uniform(-R + 8, R - 8, NP)
rr_ = np.sqrt(rng.uniform(0, 1, NP)) * 7; aa_ = rng.uniform(0, 2 * np.pi, NP)
PX[:, 0] = rr_ * np.cos(aa_); PX[:, 2] = rr_ * np.sin(aa_)
def particles(lay, t, pos, M, collect, amt=1.0):
    if amt < 0.02: return
    y = PX[:, 1].copy()
    near = np.round(y / NODE) * NODE
    y = y + (near - y) * collect + 0.25 * np.sin(t * 2 + PX[:, 0])
    P3 = np.stack([PX[:, 0], y, PX[:, 2]], 1)
    x_, y_, z_ = project(P3, pos, M)
    for xi, yi, zi in zip(x_, y_, z_):
        if zi < 3 or not (0 <= xi < WW and 0 <= yi < HH): continue
        r = int(np.clip(F * 0.08 / zi, 1, 5)); b = min(1.0, 40 / zi) * 0.55 * amt
        cv2.circle(lay, (int(xi), int(yi)), r, (200 * b, 225 * b, 255 * b), -1, cv2.LINE_AA)

# ---------------- post ----------------
def bloom(fr):
    small = cv2.resize(fr, (WW // 4, HH // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 170, 0, None)
    b = cv2.GaussianBlur(br, (0, 0), 10) * 0.9 + cv2.GaussianBlur(br, (0, 0), 30) * 0.7
    return fr + cv2.resize(b, (WW, HH), interpolation=cv2.INTER_LINEAR)
_V = None
def vign():
    global _V
    if _V is None:
        y, x = np.mgrid[0:HH, 0:WW].astype(np.float32)
        r = np.sqrt(((x - WW / 2) / (WW * 0.75)) ** 2 + ((y - HH / 2) / (HH * 0.7)) ** 2)
        _V = np.clip(1.12 - 0.42 * r ** 2, 0.55, 1.0)[..., None]
    return _V

def tone(fr):
    x = fr / 255.0
    x = x / (1 + 0.18 * x)                                                     # soft shoulder
    return np.clip(x * 1.12 * 255, 0, 255)

# ---------------- frame ----------------
BACK = None
def add_back(fr, ox, oy):
    y, x = np.mgrid[0:HH // 8, 0:WW // 8].astype(np.float32) * 8
    r2 = ((x - ox) ** 2 + ((y - oy) * 0.8) ** 2) / ((350.0 * S2) ** 2)
    g = np.exp(-r2)[..., None] * np.array([70, 62, 52], np.float32) + np.exp(-r2 * 6)[..., None] * np.array([60, 70, 80], np.float32)
    return fr + cv2.resize(g, (WW, HH), interpolation=cv2.INTER_LINEAR)

def scene(t):
    pos, M, blur, dist = cam(t)
    yaw = KEYS[0][2]
    fr = background(math.degrees(math.atan2(pos[0], pos[2])))
    # soft lab backlight behind the levitator centre
    ox, oy, oz = project(np.array([0, 0, 0.0]), pos, M)
    g = BACK if BACK is not None else None
    fr = add_back(fr, ox, oy)
    xamt = sm(ramp(t, s('L03') - 0.3, s('L03') + 0.4)) * (1 - sm(ramp(t, s('L07'), s('L07') + 0.6)))
    xamt = max(xamt, 0.55 * sm(ramp(t, s('L07') + 0.6, s('L07') + 1.2)) * (1 - sm(ramp(t, s('L09') + 0.5, s('L09') + 1.5))))
    glow = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * 6)) * 0.4
    loud = dial_value(t)
    glow = glow * (0.4 + 1.2 * loud)
    GLOW[0] = glow
    lev, cov = draw_levitator(pos, M, glow)
    # amber spill from the arrays above/below (seen when close)
    yy = np.linspace(-1, 1, HH)[:, None, None]
    fr = fr + np.array([255, 140, 50]) * (np.abs(yy) ** 6 * 0.22 * glow) * sm(ramp(15 - dist, -5, 5))
    if blur > 0.3:
        hs = (WW // 2, HH // 2)
        lev = cv2.resize(cv2.GaussianBlur(cv2.resize(lev, hs, interpolation=cv2.INTER_AREA), (0, 0), blur), (WW, HH))
        cov = cv2.resize(cv2.GaussianBlur(cv2.resize(cov, hs, interpolation=cv2.INTER_AREA), (0, 0), blur), (WW, HH))
    fr = fr * (1 - cov[..., None]) + lev
    fr = fr * (1 - 0.25 * xamt)
    front = ramp(t, s('L03'), e('L03') + 0.2)
    xray(fr, t, pos, M, xamt, front)
    lay = np.zeros_like(fr)
    collect = (1 - sm(ramp(t, e('L02') - 0.5, s('L03')))) * (t < s('L04')) + (t >= e('L10') - 1.5) + sm(ramp(t, s('L04'), e('L04'))) * (1 - sm(ramp(t, (s('L08') + e('L08')) / 2 + 0.2, (s('L08') + e('L08')) / 2 + 0.6))) \
        + sm(ramp(t, e('L08') + 0.8, e('L08') + 1.8)) * (t > e('L08'))
    particles(lay, t, pos, M, min(collect, 1), max(sm(ramp(t, s('L03'), s('L04'))), 0.7 * (1 - sm(ramp(t, e('L02') - 1, e('L02')))), 0.7 * sm(ramp(t, e('L10') - 1.5, e('L10')))) * (1 - sm(ramp(t, s('L09') + 1, s('L09') + 2.5)) * (t < e('L10') - 1.5)))
    hl_ = cv2.resize(cv2.GaussianBlur(cv2.resize(lay, (WW // 4, HH // 4), interpolation=cv2.INTER_AREA), (0, 0), 1.6), (WW, HH))
    fr = (fr + lay + hl_ * 0.8).astype(np.float32)
    dl, bursts = drops_state(t)
    for cen, a, c, al in sorted(dl, key=lambda q: -np.linalg.norm(q[0] - pos)):
        drop(fr, cen, a, c, pos, M, al)
    for p, rb, al in bursts:
        if al > 0.02: drop(fr, p, rb, rb, pos, M, al)
    fr = tone(bloom(fr.astype(np.float32))).astype(np.float32) * vign()
    return fr, pos, M, dl

# ---------------- overlays (text drawn on own RGBA layer, then composited) ----------------
def overlays(img, t, pos, M, dl):
    T = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(T)
    al = lambda a, b, fi=0.25, fo=0.25: int(255 * min(sm(ramp(t, a, a + fi)), 1 - sm(ramp(t, b - fo, b))))
    d.text((930 * S2, 112 * S2), 'SOURCE CODE #12', font=lib.font(lib.FONT_XB, 36 * S2), fill=YELLOW + (255,), anchor='rm', stroke_width=4 * S2, stroke_fill=(0, 0, 0, 255))
    # live instrument readout (top-left)
    loud = dial_value(t)
    led(d, loud)
    # persistent red title sticker (thel.ab67 kit) after the hook
    ps = min(al(e('L01') + 0.2, X.T + 5, 0.35), al(-5, e('L10') - 0.2, 0.01, 0.5))
    if ps: sticker(d, ps)
    # instrument: oscilloscope of pressure vs height
    po = al(s('L03') - 0.2, s('L09') + 0.4, 0.35, 0.4)
    if po: scope(d, t, po, loud)
    # HOOK
    a0 = al(-1.0, e('L01') + 0.45, 0.01, 0.3)
    if a0 > 0:
        f = lib.font(lib.FONT_XB, 112 * S2)
        for yy, parts in ((300, [('SOUND ', YELLOW), ('IS', WHITE)]), (430, [('HOLDING THIS', WHITE)])):
            tw = sum(f.getlength(p) for p, _ in parts); xx = (WW - tw) / 2
            for p, c in parts:
                d.text((xx, yy * S2), p, font=f, fill=c + (a0,), anchor='lm', stroke_width=9 * S2, stroke_fill=(0, 0, 0, a0)); xx += f.getlength(p)
    # cyan ring on the main drop (hook, end-loop)
    ra = max(al(-1.0, e('L02') - 0.4, 0.01, 0.4), al(e('L10') - 0.6, X.T + 1, 0.4, 0.01))
    if ra > 0 and dl:
        cx, cy, cz = project(dl[0][0], pos, M); r = F * dl[0][1] / cz * 1.5 / S2
        lib.ring(d, (cx / S2, cy / S2), r, CYAN, ra, 5, ss=S2)
    # pills
    p = al(s('L02') + 0.6, e('L02') + 0.6)
    if p: lib.pill(d, (540, 1300), '72 speakers · 40,000 Hz · too high to hear', AMBER, 30, p, ss=S2)
    p = al(s('L04') + 0.5, e('L05') + 0.3)
    if p and dl:
        cx, cy, cz = project(np.array([0, 0, 0.0]), pos, M)
        lib.pill(d, (cx / S2 + 250, cy / S2 - 120), 'pressure node', CYAN, 30, p, ss=S2)
    # stack counter
    p = al(s('L06') + 0.3, s('L07') + 0.2)
    if p:
        n = stack_count(t)
        d.text((150 * S2, 960 * S2), f'×{n}', font=lib.font(lib.FONT_XB, 120 * S2), fill=CYAN + (p,), anchor='mm', stroke_width=6 * S2, stroke_fill=(0, 0, 0, p))
        lib.pill(d, (150, 1060), 'drops', CYAN, 26, p, ss=S2)
        # 4.3 mm bracket between the first two nodes
        a3, b3 = project(np.array([[6, 0, 0], [6, NODE, 0]]) @ np.diag([1, 1, 1]) + 0, pos, M)[:2], None
        xa, ya, _ = project(np.array([4.5, 0, 0]), pos, M); xb, yb, _ = project(np.array([4.5, NODE, 0]), pos, M)
        d.line([(xa, ya), (xb, yb)], fill=CYAN + (p,), width=3 * S2)
        for yv in (ya, yb): d.line([(xa - 10 * S2, yv), (xa + 10 * S2, yv)], fill=CYAN + (p,), width=3 * S2)
        lib.pill(d, (xa / S2 + 95, (ya + yb) / 2 / S2), '4.3 mm', CYAN, 28, p, ss=S2)
    # equation (3 s) -> chip
    p = al(s('L06') + 1.0, s('L06') + 4.0)
    if p: paper_eq(d, 'λ / 2  =  c / 2f  =  343 / 80 000 m  ≈  4.3 mm', 'one quiet pocket every 4.3 mm', p)
    p = al(s('L06') + 4.0, s('L07') + 0.2)
    if p: paper_eq(d, 'λ / 2  ≈  4.3 mm', None, p, 34)
    # loudness dial
    p = al(s('L07') + 0.1, e('L08') + 2.2)
    if p: dial(d, t, p)
    p = al(s('L08') + 0.3, (s('L08') + e('L08')) / 2 + 1.6)
    if p: lib.pill(d, (190, 1110), 'slow motion ×18', WHITE, 26, p, ss=S2)
    p = al(s('L09') + 0.3, e('L09') - 2.0)
    if p: lib.pill(d, (540, 1250), 'used to make drug particles, container-free', CYAN, 28, p, ss=S2)
    p = al(e('L09') - 1.9, e('L09') + 0.4)
    if p:
        f = lib.font(lib.FONT_XB, 70 * S2)
        d.text((540 * S2, 1240 * S2), 'NATURE IS RUNNING', font=f, fill=WHITE + (p,), anchor='mm', stroke_width=7 * S2, stroke_fill=(0, 0, 0, p))
        d.text((540 * S2, 1330 * S2), 'ON CODE', font=f, fill=YELLOW + (p,), anchor='mm', stroke_width=7 * S2, stroke_fill=(0, 0, 0, p))
    # captions
    CAP = {'L02': ("No strings. Just sound.", ('sound',)), 'L03': ("Two waves crash together", ('crash',)),
           'L04': ("A quiet pocket", ('quiet',)), 'L05': ("Pushed in. Stays.", ('stays',)),
           'L06': ("A pocket every 4.3 mm", ('4.3',)), 'L07': ("One number: loudness", ('loudness',)),
           'L08': ("Soft: falls. Loud: bursts.", ('falls', 'bursts')),
           'L09': ("Medicine made in mid-air", ('mid-air',)), 'L10': ("Next: is your chai swirl worth $1M?", ('$1m',))}
    for k, (txt, hl) in CAP.items():
        p = al(s(k), e(k) + 0.25, 0.12, 0.2)
        if p: lib.caption(d, txt, y=1480, size=58, hl=hl, alpha=p, ss=S2)
    img.alpha_composite(T)
    return img

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
def led(d, loud):
    x0, y0, x1, y1 = 40 * S2, 82 * S2, 330 * S2, 142 * S2
    d.rounded_rectangle([x0, y0, x1, y1], radius=10 * S2, fill=(10, 6, 8, 235), outline=(120, 30, 30, 255), width=3 * S2)
    fm = lib.font(lib.FONT_MONO, 34 * S2)
    d.text((x0 + 16 * S2, (y0 + y1) / 2), '40.000kHz', font=fm, fill=(255, 70, 55, 255), anchor='lm')
    d.text((x1 - 14 * S2, (y0 + y1) / 2), f"{150 + int(round(loud * 24))}dB", font=lib.font(lib.FONT_MONO, 24 * S2), fill=(255, 150, 60, 255), anchor='rm')

def sticker(d, p):
    f = lib.font(lib.FONT_XB, 50 * S2); txt = 'CAN SOUND LIFT WATER?'
    tw = f.getlength(txt); cx = 540 * S2; y = 205 * S2
    d.rounded_rectangle([cx - tw / 2 - 26 * S2, y - 40 * S2, cx + tw / 2 + 26 * S2, y + 40 * S2], radius=18 * S2, fill=(222, 38, 44, p))
    d.text((cx, y), txt, font=f, fill=(255, 255, 255, p), anchor='mm')
    f2 = lib.font(lib.FONT_SB, 28 * S2); s2 = 'watch it when the sound gets too loud'
    w2 = f2.getlength(s2); y2 = 272 * S2
    d.rounded_rectangle([cx - w2 / 2 - 16 * S2, y2 - 22 * S2, cx + w2 / 2 + 16 * S2, y2 + 22 * S2], radius=8 * S2, fill=(250, 250, 250, p))
    d.text((cx, y2), s2, font=f2, fill=(20, 20, 24, p), anchor='mm')

def scope(d, t, p, loud):
    x0, y0, x1, y1 = 700 * S2, 340 * S2, 930 * S2, 640 * S2
    d.rounded_rectangle([x0 - 8 * S2, y0 - 8 * S2, x1 + 8 * S2, y1 + 34 * S2], radius=14 * S2, fill=(30, 33, 38, int(p * 0.95)), outline=(120, 126, 136, p), width=2 * S2)
    d.rectangle([x0, y0, x1, y1], fill=(4, 18, 12, p))
    for i in range(1, 6): d.line([(x0, y0 + (y1 - y0) * i / 6), (x1, y0 + (y1 - y0) * i / 6)], fill=(30, 90, 60, p), width=1 * S2)
    for i in range(1, 4): d.line([(x0 + (x1 - x0) * i / 4, y0), (x0 + (x1 - x0) * i / 4, y1)], fill=(30, 90, 60, p), width=1 * S2)
    k = 2 * np.pi / LAM; om = 2 * np.pi * 0.9 * t
    front = ramp(t, s('L03'), e('L03') + 0.2)
    ys = np.linspace(20, -20, 160)
    top = np.sin(k * ys + om) * (ys > R - 2 * R * front); bot = np.sin(-k * ys + om) * (ys < -R + 2 * R * front)
    g = 0.25 + 1.5 * loud if t > s('L07') else 1.0
    py = y0 + (20 - ys) / 40 * (y1 - y0)
    cols = [(x0 + (x1 - x0) * 0.2, top, (255, 170, 60)), (x0 + (x1 - x0) * 0.5, bot, (255, 120, 40)), (x0 + (x1 - x0) * 0.8, (top + bot) / 2 * g, (120, 255, 170))]
    for xc, v, c in cols:
        pts = [(xc + vv * (x1 - x0) * 0.09, yy) for vv, yy in zip(v, py)]
        d.line(pts, fill=c + (p,), width=3 * S2)
    nodes_on = ramp(t, e('L03'), s('L04') + 0.5)
    if nodes_on > 0:
        for j in range(-10, 11):
            yy = y0 + (20 - j * NODE) / 40 * (y1 - y0)
            if y0 < yy < y1: d.ellipse([x0 + (x1 - x0) * 0.8 - 4 * S2, yy - 4 * S2, x0 + (x1 - x0) * 0.8 + 4 * S2, yy + 4 * S2], fill=CYAN + (int(p * nodes_on),))
    fl = lib.font(lib.FONT_SB, 19 * S2)
    for xc, lab in ((0.2, 'wave ↓'), (0.5, 'wave ↑'), (0.8, 'sum')):
        d.text((x0 + (x1 - x0) * xc, y1 + 17 * S2), lab, font=lib.font(lib.FONT_DJ if '↓' in lab or '↑' in lab else lib.FONT_SB, 18 * S2), fill=(200, 210, 220, p), anchor='mm')
    d.text((x0 + 8 * S2, y0 + 14 * S2), 'PRESSURE', font=fl, fill=(120, 255, 170, p), anchor='lm')

def paper_eq(d, eq, plain, p, size=40):
    fe = lib.font(SERIF, size * S2); tw = fe.getlength(eq); cx = 540 * S2; y = 1290 * S2
    d.rounded_rectangle([cx - tw / 2 - 30 * S2, y - size * 0.95 * S2, cx + tw / 2 + 30 * S2, y + size * 0.95 * S2], radius=10 * S2, fill=(246, 240, 226, p))
    d.text((cx, y), eq, font=fe, fill=(28, 26, 30, p), anchor='mm')
    if plain:
        fp = lib.font(lib.FONT_SB, 30 * S2)
        d.text((cx, y + 72 * S2), plain, font=fp, fill=CYAN + (p,), anchor='mm', stroke_width=3 * S2, stroke_fill=(0, 0, 0, p))

def dial(d, t, p):
    v = dial_value(t); cx, cy, r = 540 * S2, 1230 * S2, 110 * S2
    ok = abs(v - 0.5) < 0.08
    col = GREEN if ok else RED
    d.arc([cx - r, cy - r, cx + r, cy + r], 180, 360, fill=(60, 70, 85, p), width=16 * S2)
    d.arc([cx - r, cy - r, cx + r, cy + r], 180 + 180 * 0.4, 180 + 180 * 0.6, fill=GREEN + (p,), width=16 * S2)
    ang = math.pi * (1 - v)
    d.line([(cx, cy), (cx + 0.9 * r * math.cos(ang), cy - 0.9 * r * math.sin(ang))], fill=col + (p,), width=7 * S2)
    d.ellipse([cx - 12 * S2, cy - 12 * S2, cx + 12 * S2, cy + 12 * S2], fill=col + (p,))
    lib.pill(d, (540, 1272), 'LOUDNESS', col, 26, p, ss=S2)
    lab = '✓ floats' if ok else ('✗ falls' if v < 0.5 else '✗ bursts')
    lib.pill(d, (790, 1190), lab, col, 30, p, ss=S2)

def frame(n, out4k=False):
    t = n / FPS
    fr, pos, M, dl = scene(t)
    img = Image.fromarray(fr.astype(np.uint8)).convert('RGBA')
    img = overlays(img, t, pos, M, dl).convert('RGB')
    return img if out4k else img.resize((1080, 1920), Image.LANCZOS)

if __name__ == '__main__':
    import sys, time
    for n in map(int, sys.argv[1:]):
        t0 = time.time(); frame(n).save(f'f{n:04d}.png'); print(n, round(time.time() - t0, 2))
