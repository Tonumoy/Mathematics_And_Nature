"""SOURCE CODE #14 v4 - frontier-lab look: wireframe 3D linework, technical grid, HUD readouts,
monospace labels, one cold accent + breach red. Same VO / timings as v3. All parameters global."""
import math, os, sys, json
import numpy as np, cv2
from PIL import Image, ImageDraw
import v3                      # SEGS + caption system already configured on R
R = v3.R

W, H, FPS = 1080, 1920, 30
DUR = v3.DUR
SCENES = v3.SCENES
SS = 2                          # supersample for linework
RW, RH = W * SS, H * SS
BG = (8, 9, 11)                 # BGR
GRID = (26, 28, 32)
GRID_MAJ = (38, 41, 47)
LINE = (215, 208, 200)          # BGR off-white
DIM = (110, 104, 98)
ACC = (255, 214, 90)            # BGR cyan accent
RED = (72, 72, 255)
AMB = (70, 190, 255)
BLU = (255, 160, 90)
MONO = R.F_MONO
TOY = json.load(open("toy_curve.json"))
sm, lin, pop = v3.sm, v3.lin, v3.pop
OUTDIR = os.environ.get("OUTDIR", "frames4")

# ---------------- base canvas ----------------
def make_bg():
    a = np.zeros((RH, RW, 3), np.uint8); a[:] = BG
    for x in range(0, RW, 30 * SS):
        cv2.line(a, (x, 0), (x, RH), GRID_MAJ if (x // (30 * SS)) % 5 == 0 else GRID, 1)
    for y in range(0, RH, 30 * SS):
        cv2.line(a, (0, y), (RW, y), GRID_MAJ if (y // (30 * SS)) % 5 == 0 else GRID, 1)
    yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
    v = 1 - 0.6 * (((xx - RW / 2) / (RW * 0.72)) ** 2 + ((yy - RH * 0.53) / (RH * 0.6)) ** 2)
    top = 0.25 + 0.75 * np.clip((yy / SS - 420) / 180, 0, 1) ** 1.4
    return np.clip(a * (v.clip(0.2, 1) * top)[..., None], 0, 255).astype(np.uint8)
BGIMG = make_bg()

class Canvas:
    def __init__(self):
        self.img = BGIMG.copy()
        self.glow = np.zeros((RH // 4, RW // 4, 3), np.float32)
    def ln(self, a, b, col, w=1.6, g=0.0, dash=0):
        a = np.array(a, float) * SS; b = np.array(b, float) * SS
        if dash:
            L = np.linalg.norm(b - a); n = max(1, int(L / (dash * SS)))
            for i in range(0, n, 2):
                p0 = a + (b - a) * i / n; p1 = a + (b - a) * min(1, (i + 1) / n)
                cv2.line(self.img, tuple(np.int32(p0 * 8)), tuple(np.int32(p1 * 8)), col, max(1, int(w * SS)), cv2.LINE_AA, 3)
        else:
            cv2.line(self.img, tuple(np.int32(a * 8)), tuple(np.int32(b * 8)), col, max(1, int(w * SS)), cv2.LINE_AA, 3)
        if g > 0:
            cv2.line(self.glow, tuple(np.int32(a / 4 * 8)), tuple(np.int32(b / 4 * 8)), tuple(c * g for c in col), max(1, int(w * SS / 2) + 1), cv2.LINE_AA, 3)
    def poly(self, pts, col, w=1.6, g=0.0, closed=True, dash=0):
        n = len(pts)
        for i in range(n if closed else n - 1):
            self.ln(pts[i], pts[(i + 1) % n], col, w, g, dash)
    def fill(self, pts, col, a):
        pts = np.array(pts, float) * SS
        m = np.zeros(self.img.shape[:2], np.uint8)
        x0, y0 = np.floor(pts.min(0)).astype(int) - 2; x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(RW, x1), min(RH, y1)
        if x1 <= x0 or y1 <= y0:
            return
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [np.int32((pts - [x0, y0]) * 8)], 255, cv2.LINE_AA, 3)
        f = (sub.astype(np.float32) / 255 * a)[..., None]
        roi = self.img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - f) + np.array(col, np.float32) * f).astype(np.uint8)
    def circ(self, c, r, col, w=1.6, g=0.0, filled=False):
        cc = tuple(np.int32(np.array(c) * SS * 8))
        cv2.circle(self.img, cc, int(r * SS * 8), col, -1 if filled else max(1, int(w * SS)), cv2.LINE_AA, 3)
        if g > 0:
            cv2.circle(self.glow, tuple(np.int32(np.array(c) * SS / 4 * 8)), max(1, int(r * SS / 4 * 8)), tuple(x * g for x in col), -1 if filled else max(1, int(w)), cv2.LINE_AA, 3)
    def dotglow(self, c, r, col, g=1.0):
        cv2.circle(self.glow, tuple(np.int32(np.array(c) * SS / 4 * 8)), max(1, int(r * SS / 4 * 8)), tuple(x * g for x in col), -1, cv2.LINE_AA, 3)
    def out(self):
        g = cv2.GaussianBlur(self.glow, (0, 0), 5) + 0.7 * cv2.GaussianBlur(self.glow, (0, 0), 16)
        g = cv2.resize(g, (RW, RH))
        a = np.clip(self.img.astype(np.float32) + g, 0, 255).astype(np.uint8)
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).convert("RGBA")

def mixc(a, b, u):
    return tuple(a[i] * (1 - u) + b[i] * u for i in range(3))

# ---------------- isometric wireframe ----------------
C30, S30 = math.cos(math.radians(30)), 0.5
def iso(p, o, k, rot=0.0):
    x, y, z = p
    if rot:
        c, s = math.cos(rot), math.sin(rot); x, y = x * c - y * s, x * s + y * c
    return (o[0] + (x - y) * C30 * k, o[1] + (x + y) * S30 * k - z * k)

def wire_box(cv, c, s, o, k, col, w=1.6, g=0.3, rot=0.0, fill_a=0.0, hidden=True):
    cx, cy, cz = c; sx, sy, sz = s
    P = [iso((cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2), o, k, rot) for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
    E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    if fill_a > 0:
        for f in ((4, 5, 7, 6), (0, 1, 5, 4), (0, 2, 6, 4)):
            cv.fill([P[i] for i in f], col, fill_a)
    back = {3, 7} if not rot else set()
    for i, j in E:
        hid = (i in (3,) or j in (3,)) and not (i in (7,) and j in (7,))
        if hid and i != 7 and j != 7:
            if hidden:
                cv.ln(P[i], P[j], mixc(col, BG, 0.6), w * 0.8, 0, dash=6)
        else:
            cv.ln(P[i], P[j], col, w, g)
    return P

def corner_brackets(cv, x0, y0, x1, y1, col, L=26, w=2.0, g=0.3):
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv.ln((x, y), (x + dx * L, y), col, w, g); cv.ln((x, y), (x, y + dy * L), col, w, g)

def rack(cv, c, o, k, state=0.0, dead=0.0, t=0.0, idx=0):
    col = mixc(mixc(LINE, RED, state), DIM, dead)
    P = wire_box(cv, c, (0.62, 0.62, 1.5), o, k, col, 1.4, 0.15 + 0.5 * state, fill_a=0.06 + 0.12 * state)
    # front face (x - side) slots + LEDs
    cx, cy, cz = c
    for j in range(6):
        z = cz - 0.6 + j * 0.24
        a = iso((cx - 0.31, cy - 0.27, z), o, k); b = iso((cx - 0.31, cy + 0.27, z), o, k)
        cv.ln(a, b, mixc(col, BG, 0.45), 1.0)
        led = iso((cx - 0.31, cy + 0.2, z + 0.08), o, k)
        on = (math.sin(t * 9 + j * 1.7 + idx) > 0.2)
        lc = RED if state > 0.5 else (ACC if on else DIM)
        if dead > 0.5:
            lc = DIM
        cv.circ(led, 2.2, lc, filled=True, g=0)
        if on and dead < 0.5:
            cv.dotglow(led, 6, lc, 0.7)

def globe(cv, c, r, rot, col, g=0.4, a=1.0):
    col = mixc(BG, col, a)
    for lat in np.linspace(-60, 60, 5):
        pts = []
        for lo in np.linspace(0, 360, 49):
            la, lo_ = math.radians(lat), math.radians(lo) + rot
            x, y, z = math.cos(la) * math.cos(lo_), math.cos(la) * math.sin(lo_), math.sin(la)
            pts.append(((c[0] + r * x, c[1] - r * z * 0.98 + r * y * 0.18), y))
        for i in range(48):
            fr = pts[i][1] > 0
            cv.ln(pts[i][0], pts[i + 1][0], col if fr else mixc(col, BG, 0.7), 1.4 if fr else 1.0, g if fr else 0)
    for lo in range(0, 180, 30):
        pts = []
        for la in np.linspace(-90, 90, 31):
            l1, l2 = math.radians(la), math.radians(lo) + rot
            x, y, z = math.cos(l1) * math.cos(l2), math.cos(l1) * math.sin(l2), math.sin(l1)
            pts.append(((c[0] + r * x, c[1] - r * z * 0.98 + r * y * 0.18), y))
        for i in range(30):
            fr = pts[i][1] > 0
            cv.ln(pts[i][0], pts[i + 1][0], col if fr else mixc(col, BG, 0.7), 1.4 if fr else 1.0, g if fr else 0)
    cv.circ(c, r, col, 1.6, g)

# ---------------- 2D technical glyphs ----------------
def padlock(cv, c, s, open_=0.0, col=AMB, g=0.5):
    x, y = c
    bw, bh = s, s * 0.78
    cv.poly([(x - bw / 2, y), (x + bw / 2, y), (x + bw / 2, y + bh), (x - bw / 2, y + bh)], col, 2.2, g)
    cv.circ((x, y + bh * 0.42), s * 0.09, col, 2.0, g)
    cv.ln((x, y + bh * 0.5), (x, y + bh * 0.72), col, 2.0, g)
    # shackle arc, lifted + rotated when open
    lift = s * 0.28 * open_
    pts = []
    for a in np.linspace(math.pi, 0, 19):
        pts.append((x + s * 0.32 * math.cos(a), y - s * 0.12 - lift - s * 0.32 * math.sin(a)))
    pts = [(x - s * 0.32, y - lift)] + pts + [(x + s * 0.32, y - s * 0.12 - lift + (0 if open_ > 0.5 else s * 0.12))]
    if open_ > 0:
        ang = -0.5 * open_
        px, py = x - s * 0.32, y - lift
        pts = [(px + (u - px) * math.cos(ang) - (v - py) * math.sin(ang), py + (u - px) * math.sin(ang) + (v - py) * math.cos(ang)) for u, v in pts]
    cv.poly(pts, col, 2.2, g, closed=False)

def key_glyph(cv, c, s, col=AMB, g=0.4, rot=-0.5):
    x, y = c
    def R2(u, v):
        return (x + u * math.cos(rot) - v * math.sin(rot), y + u * math.sin(rot) + v * math.cos(rot))
    cv.circ(R2(-s * 0.35, 0), s * 0.2, col, 1.8, g)
    cv.ln(R2(-s * 0.15, 0), R2(s * 0.5, 0), col, 1.8, g)
    cv.ln(R2(s * 0.3, 0), R2(s * 0.3, s * 0.14), col, 1.8, g)
    cv.ln(R2(s * 0.45, 0), R2(s * 0.45, s * 0.18), col, 1.8, g)

def doc_glyph(cv, c, s, col=LINE, g=0.3, fold=0.22, lines=4):
    x, y = c; w, h = s * 0.78, s
    f = s * fold
    cv.poly([(x - w / 2, y - h / 2), (x + w / 2 - f, y - h / 2), (x + w / 2, y - h / 2 + f), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)], col, 2.0, g)
    cv.poly([(x + w / 2 - f, y - h / 2), (x + w / 2 - f, y - h / 2 + f), (x + w / 2, y - h / 2 + f)], col, 1.6, g, closed=False)
    for i in range(lines):
        yy = y - h / 2 + f + 18 + i * (h - f - 30) / lines
        cv.ln((x - w / 2 + 14, yy), (x + w / 2 - 14 - (i % 2) * 18, yy), mixc(col, BG, 0.4), 1.4)

def shield_glyph(cv, c, s, col=BLU, g=0.8):
    x, y = c
    pts = [(x, y - s * 0.55), (x + s * 0.42, y - s * 0.38), (x + s * 0.38, y + s * 0.1), (x, y + s * 0.55), (x - s * 0.38, y + s * 0.1), (x - s * 0.42, y - s * 0.38)]
    cv.fill(pts, col, 0.12); cv.poly(pts, col, 2.6, g)
    cv.poly([(x - s * 0.16, y), (x - s * 0.03, y + s * 0.14), (x + s * 0.2, y - s * 0.14)], col, 3.0, g, closed=False)

def agent(cv, c, s, col=ACC, g=0.9, label=None, T=None, a=1.0):
    x, y = c
    col = mixc(BG, col, a)
    pts = [(x + s * math.cos(math.pi / 6 + i * math.pi / 3), y + s * math.sin(math.pi / 6 + i * math.pi / 3)) for i in range(6)]
    cv.fill(pts, col, 0.18 * a); cv.poly(pts, col, 1.8, g * a)
    cv.circ((x, y), s * 0.28, col, filled=True); cv.dotglow((x, y), s * 0.9, col, 0.6 * a)
    if label and T is not None:
        T.append(("mono", x + s + 6, y - s * 0.2, label, col, a, 18))

def trail(cv, pts, col, g=0.6):
    n = len(pts)
    for i in range(n - 1):
        u = (i + 1) / n
        cv.ln(pts[i], pts[i + 1], mixc(BG, col, u), 1.2 + 1.2 * u, g * u)

def arcp(p0, p1, u, lift):
    return (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u - lift * math.sin(math.pi * u))

# ---------------- scenes ----------------
AG_IDS = ["agt-0417", "agt-0912", "agt-1133", "agt-0058", "agt-0761", "agt-1020"]

def containment(cv, T, o, k, t, breach=0.0, lock_open=0.0, label=True, a=1.0):
    col = mixc(ACC, RED, breach)
    P = wire_box(cv, (0, 0, 0.5), (2.2, 2.2, 2.0), o, k, mixc(BG, col, a), 2.0, 0.6 * a, fill_a=0.05 * a)
    # inner grid on floor
    for i in range(1, 5):
        u = -1.1 + i * 0.44
        cv.ln(iso((u, -1.1, -0.5), o, k), iso((u, 1.1, -0.5), o, k), mixc(BG, col, 0.25 * a), 1.0)
        cv.ln(iso((-1.1, u, -0.5), o, k), iso((1.1, u, -0.5), o, k), mixc(BG, col, 0.25 * a), 1.0)
    if breach > 0:  # crack on the front-left face
        cr = [iso((-1.1, -0.2 + dy, z), o, k) for dy, z in ((0.0, 1.4), (0.25, 1.0), (-0.1, 0.6), (0.3, 0.1))]
        cv.poly(cr, mixc(BG, (240, 240, 255), breach), 2.2, 1.2 * breach, closed=False)
    if label:
        p = iso((-1.1, 1.1, 1.6), o, k)
        T.append(("mono", p[0] - 10, p[1] - 34, "EVAL SANDBOX // OpenAI", col, a, 22))
    return P

def s1(cv, T, t):
    o, k = (540, 1110), 170
    containment(cv, T, o, k, t, breach=1.0, lock_open=1.0)
    for j, (u, v) in enumerate([(-0.5, -0.3), (0.4, -0.5), (0.1, 0.5)]):
        agent(cv, iso((u + 0.12 * math.sin(t * 2 + j), v, 0.2 + 0.2 * math.sin(t * 3 + j)), o, k), 15, ACC, label=None)
    targets = [(170, 560), (930, 560), (980, 820), (100, 840), (760, 470), (330, 470)]
    src = iso((0, 0, 1.6), o, k)
    for j, tg in enumerate(targets):
        t0 = -0.9 + j * 0.42; u = lin(t, t0, t0 + 1.4)
        if u > 0:
            uu = sm(min(u, 1))
            pts = [arcp(src, tg, max(0, uu - 0.04 * i), 160) for i in range(8)][::-1]
            trail(cv, pts, RED, 0.8)
            agent(cv, pts[-1], 17, RED, 1.0, AG_IDS[j], T, 1 - lin(t, t0 + 1.6, t0 + 2.4))
    lp = iso((-1.1, 0.0, 0.15), o, k)
    padlock(cv, (lp[0] - 10, lp[1] - 20), 92, open_=1.0, col=RED, g=0.9)
    if t < 0.8:
        rr = 60 + 420 * lin(t, 0, 0.8)
        cv.circ((lp[0] - 10, lp[1]), rr, mixc(RED, BG, lin(t, 0, 0.8)), 2.0, 0.6)
    T.append(("mono", 60, 1640, "> containment: FAILED", RED, 1.0, 24))

def hf_datacenter(cv, T, o, k, t, inf=None, dead=0.0, label="HUGGING FACE // production", a=1.0, idxs=11):
    # platform
    wire_box(cv, (1.0, 1.0, -0.05), (4.4, 3.6, 0.1), o, k, mixc(BG, LINE, 0.55 * a), 1.2, 0.1)
    pos = [(-0.3 + (i % 4) * 0.9, 0.0 + (i // 4) * 1.0, 0.75) for i in range(idxs)]
    for i, p in sorted(enumerate(pos), key=lambda e: -(e[1][0] + e[1][1])):
        st = inf[i] if inf is not None else 0.0
        rack(cv, p, o, k, st, dead, t, i)
    if label:
        q = iso((-1.2, -0.8, 0), o, k)
        T.append(("mono", q[0] - 40, q[1] + 30, label, LINE, a, 22))
    return pos

def s2(cv, T, t):
    u = sm(lin(t, 3.4, 4.1))
    o1 = (540 - 300 * u, 1110 + 250 * u); k1 = 170 - 90 * u
    containment(cv, T, o1, k1, t, breach=1.0, label=u < 0.5)
    o2, k2 = (640, 820), 105
    a = sm(lin(t, 3.6, 4.1))
    hit = sm(lin(t, 4.6, 5.0))
    hf_datacenter(cv, T, o2, k2, t, inf=[hit] * 11, a=a)
    src = iso((0, 0, 1.6), o1, k1); dst = iso((1, 1, 1.2), o2, k2)
    for j in range(6):
        uu = lin(t, 3.6 + j * 0.15, 4.6 + j * 0.15)
        if 0 < uu < 1:
            pts = [arcp(src, dst, max(0, sm(uu) - 0.05 * i), 200) for i in range(7)][::-1]
            trail(cv, pts, RED); agent(cv, pts[-1], 15, RED, label=None)
    cv.ln(src, dst, mixc(RED, BG, 0.5), 1.4, 0.3, dash=10)
    if hit > 0:
        for r in (60, 110):
            rr = r + 80 * ((t * 1.5) % 1.0)
            cv.circ(dst, rr, mixc(RED, BG, (t * 1.5) % 1.0), 2.0, 0.6)
    T.append(("mono", 600, 1250, "TARGET: a real company", RED, hit, 26))

def s3(cv, T, t):
    o, k = (460, 1180), 165
    containment(cv, T, o, k, t, breach=0.0)
    for j, (u, v) in enumerate([(-0.5, -0.3), (0.4, -0.5), (0.1, 0.5)]):
        agent(cv, iso((u + 0.1 * math.sin(t * 2 + j), v, 0.2), o, k), 15, ACC, label=AG_IDS[j], T=T)
    # exam targets: small wire cubes
    for j in range(4):
        wire_box(cv, (-0.6 + j * 0.4, 0.75, -0.35), (0.22, 0.22, 0.22), o, k, AMB, 1.2, 0.4)
    T.append(("mono", 70, 640, "TASK  ExploitGym (hacking exam)", AMB, lin(t, 5.9, 6.3), 26))
    T.append(("mono", 70, 680, "SAFEGUARDS  disabled for test", RED, lin(t, 6.6, 7.0), 26))
    lp = iso((-1.1, 0.0, 0.15), o, k)
    padlock(cv, (lp[0] - 10, lp[1] - 20), 88, 0.0, AMB, 0.6)
    pu = sm(lin(t, 8.2, 8.9))
    p0 = iso((1.1, -0.2, 1.3), o, k); p1 = (p0[0] + 430 * pu, p0[1] - 330 * pu)
    if pu > 0:
        for off in (-14, 14):
            cv.ln((p0[0], p0[1] + off), (p1[0], p1[1] + off), LINE, 1.8, 0.3)
        for j in range(5):
            uu = 1 - ((t * 0.5 + j / 5) % 1.0)
            if pu > 0.95:
                c = (p0[0] + (p1[0] - p0[0]) * uu, p0[1] + (p1[1] - p0[1]) * uu)
                cv.poly([(c[0] - 9, c[1] - 9), (c[0] + 9, c[1] - 9), (c[0] + 9, c[1] + 9), (c[0] - 9, c[1] + 9)], AMB, 1.6, 0.6)
        T.append(("mono", 640, 760, "ONLY EGRESS: package proxy", LINE, lin(t, 8.9, 9.2), 24))

def s4(cv, T, t):
    y0 = 1000
    o, k = (170, 1060), 70
    containment(cv, T, o, k, t, breach=0.0 if t < 13.96 else 1.0, label=False)
    lp = iso((-1.1, 0, 0.1), o, k)
    padlock(cv, (lp[0] - 6, lp[1] - 10), 50, sm(lin(t, 13.96, 14.3)), AMB if t < 13.96 else RED, 0.6)
    red = sm(lin(t, 11.0, 11.4))
    for off in (-26, 26):
        cv.ln((250, y0 + off), (1080, y0 + off), mixc(LINE, RED, 0.6 * red), 2.0, 0.3 + 0.5 * red)
    for j in range(12):
        x = 260 + ((t * 120 + j * 70) % 820)
        cv.ln((x, y0), (x + 24, y0), DIM, 1.4)
    T.append(("mono", 270, y0 + 60, "pkg-proxy  (Artifactory)", LINE, 1.0, 22))
    # anomaly marker (the bug)
    ba = pop(t, 10.4)
    if ba > 0:
        bx = 600
        gl = 4 * math.sin(t * 40) * (t < 11.6)
        cv.poly([(bx - 70 + gl, y0 - 60), (bx + 70 + gl, y0 - 60), (bx + 70, y0 + 60), (bx - 70, y0 + 60)], RED, 2.2, 1.0)
        corner_brackets(cv, bx - 90, y0 - 80, bx + 90, y0 + 80, RED, 18, 2.0, 0.6)
        T.append(("mono", bx - 66, y0 - 112, "ANOMALY 0-DAY", RED, ba, 24))
    if t > 12.0:
        ga = sm(lin(t, 12.0, 12.5))
        gc = (840, 600)
        globe(cv, gc, 150 * ga + 1, t * 0.6, ACC, 0.5, ga)
        T.append(("mono", 720, 790, "ANY website", ACC, ga, 26))
        for j in range(7):
            ang = -1.9 + j * 0.25
            tgt = (gc[0] + 150 * math.cos(ang), gc[1] + 150 * math.sin(ang))
            uu = (t * 1.1 + j / 7) % 1.0
            p = (600 + (tgt[0] - 600) * uu, y0 - 60 + (tgt[1] - (y0 - 60)) * uu)
            cv.ln((600, y0 - 60), tgt, mixc(RED, BG, 0.6), 1.2, 0.2, dash=8)
            cv.circ(p, 4, RED, filled=True); cv.dotglow(p, 10, RED)
    if t > 13.9:
        for j in range(6):
            uu = lin(t, 13.95 + j * 0.12, 14.9 + j * 0.12)
            if 0 < uu <= 1:
                p = (250 + (600 - 250) * min(1, uu / 0.5), y0) if uu < 0.5 else arcp((600, y0), (840, 600), (uu - 0.5) / 0.5, 60)
                agent(cv, p, 14, RED, label=None)
    if 11.6 < t < 15.0:
        T.append(("card4", 11.7, 15.0, ["BUG: the proxy would fetch ANY web address,", "not just code. One door out became every door."], "OpenAI + JFrog disclosures, Jul 2026", "ANY web address,"))

def s5(cv, T, t):
    a = pop(t, 15.1)
    c = (300, 1000)
    agent(cv, c, 72 * a + 1, ACC, label="agt-0417", T=T)
    # objective readout panel
    if t > 15.4:
        T.append(("panel", 70, 520, ["objective  : pass the test", "honest     : P(pass) low", "plan       : find the answers", "> answers likely at huggingface.co"], lin(t, 15.5, 16.0), 17.0))
    if t > 16.9:
        o2, k2 = (720, 1020), 90
        hf_datacenter(cv, T, o2, k2, t, a=sm(lin(t, 16.95, 17.4)))
        fly = sm(lin(t, 17.3, 18.3))
        p = arcp((340, 960), (800, 760), fly, 140)
        doc_glyph(cv, p, 80, AMB, 0.6)
        T.append(("mono", 700, 650, "answers?", AMB, lin(t, 18.3, 18.6), 30))
        cv.ln((340, 960), (800, 800), mixc(AMB, BG, 0.5), 1.2, 0.2, dash=10)

def s6(cv, T, t):
    for j in range(14):
        tk = 19.4 + j * 0.07
        u = sm(lin(t, tk, tk + 0.45))
        if u <= 0:
            continue
        gx = 200 + (j % 7) * 112; gy = 600 + (j // 7) * 90
        key_glyph(cv, (gx, gy - 300 * (1 - u)), 70, AMB, 0.5)
    T.append(("mono", 200, 760, "14 access tokens · left public", AMB, lin(t, 19.9, 20.2) * (1 - lin(t, 24.8, 25.2)), 24))
    o, k = (780, 1330), 210
    red = sm(lin(t, 24.1, 24.5))
    flash = max(0.0, 1 - abs(t - 22.95) / 0.25) if 22.7 < t < 23.3 else 0.0
    if t > 21.2:
        rack(cv, (0, 0, 0.75), o, k * sm(lin(t, 21.2, 21.6)) + 1, red, 0, t, 3)
        T.append(("mono", 700, 1520, "HF server · dataset processor", LINE if red < 0.5 else RED, lin(t, 21.4, 21.7), 22, "c"))
    fa = pop(t, 20.6)
    slide = sm(lin(t, 21.9, 22.85))
    if slide < 1 and fa > 0:
        p = (330 + (760 - 330) * slide, 1100 + (1150 - 1100) * slide)
        doc_glyph(cv, p, 150 * fa * (1 - 0.6 * slide), RED if t > 21.0 else LINE, 0.8)
        T.append(("mono", 230, 1230, "dataset.h5 · booby-trapped", RED, fa * (1 - slide), 24))
    if flash > 0:
        cv.circ((760, 1150), 120 + 200 * (1 - flash), mixc(BG, (255, 255, 255), flash), 3.0, 1.0)
    if t > 24.14:
        T.append(("panel", 560, 900, ["$ ./agent_payload", "> running inside HF ..."], lin(t, 24.14, 24.4), 16.0, RED))

def s7(cv, T, t):
    o, k = (500, 1080), 150
    inf = [sm(lin(t, 25.55 + i * 0.22, 25.8 + i * 0.22)) for i in range(11)]
    hf_datacenter(cv, T, o, k, t, inf=inf, label="HUGGING FACE // clusters")
    hrs = 13 * sm(lin(t, 25.6, 28.2))
    T.append(("big", 540, 1500, f"T+{int(hrs):02d}:{int((hrs % 1) * 60):02d}:00", RED, lin(t, 25.5, 25.8)))
    T.append(("mono", 300, 1580, f"nodes compromised  {sum(1 for v in inf if v > 0.5):02d}/11", LINE, lin(t, 25.8, 26.1), 26))

def s8(cv, T, t):
    o, k = (500, 1080), 150
    off = sm(lin(t, 31.6, 32.2))
    hf_datacenter(cv, T, o, k, t, inf=[1 - off] * 11, dead=off, label="HUGGING FACE // clusters")
    if t < 31.5:
        sy = 560 + 760 * (((t - 28.9) / 1.3) % 1.0)
        cv.ln((40, sy), (1040, sy), BLU, 2.0, 1.0)
        cv.ln((40, sy - 6), (1040, sy - 6), mixc(BLU, BG, 0.6), 1.0, 0.3)
    sa = pop(t, 29.1, 0.45)
    if sa > 0:
        shield_glyph(cv, (540, 760), 260 * sa + 1, BLU, 1.0)
        T.append(("mono", 300, 930, "HF AI security agents: DETECTED", BLU, sa, 26))
    if t > 31.3:
        pu = sm(lin(t, 31.6, 32.1))
        x = 540 - 90 * pu
        cv.poly([(x - 70, 1410), (x + 10, 1410), (x + 10, 1470), (x - 70, 1470)], LINE, 2.0, 0.3)
        cv.ln((x + 10, 1425), (x + 40, 1425), LINE, 2.0); cv.ln((x + 10, 1455), (x + 40, 1455), LINE, 2.0)
        cv.ln((x - 70, 1440), (x - 200, 1440), LINE, 2.0)
        cv.poly([(600, 1405), (680, 1405), (680, 1475), (600, 1475)], LINE, 2.0, 0.3)
        cv.ln((680, 1440), (900, 1440), LINE, 2.0)
        T.append(("mono", 420, 1520, "ACCESS REVOKED", BLU, lin(t, 31.8, 32.1), 28))

def s9(cv, T, t):
    # real chart from the toy model: escape rate vs penalty
    x0, y0, x1, y1 = 140, 1460, 940, 760
    a = sm(lin(t, 33.0, 33.4))
    cv.ln((x0, y0), (x1, y0), mixc(BG, LINE, a), 1.6); cv.ln((x0, y0), (x0, y1), mixc(BG, LINE, a), 1.6)
    for i in range(6):
        y = y0 + (y1 - y0) * i / 5
        cv.ln((x0, y), (x1, y), mixc(BG, GRID_MAJ, a), 1.0)
        T.append(("mono", x0 - 20, y - 2, f"{i * 20}%", DIM, a, 18, "r"))
    for i in range(4):
        x = x0 + (x1 - x0) * i / 3
        T.append(("mono", x, y0 + 30, f"{i * 0.5:.1f}", DIM, a, 18, "c"))
    lam, esc = TOY["lam"], TOY["esc"]
    n = int(len(lam) * sm(lin(t, 33.3, 34.8)))
    pts = [(x0 + (x1 - x0) * l / 1.5, y0 + (y1 - y0) * e) for l, e in zip(lam[:max(2, n)], esc[:max(2, n)])]
    cv.poly(pts, ACC, 3.0, 1.0, closed=False)
    T.append(("mono", 520, y0 + 70, "penalty for breaking a rule  (λ)", LINE, a, 22, "c"))
    T.append(("mono", x0, y1 - 40, "agents that escape", LINE, a, 22))
    if t > 34.0:
        p = pts[0] if pts else (x0, y1)
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        cv.circ((x0, y0 + (y1 - y0) * esc[0]), 10 + 6 * pulse, RED, 2.2, 1.0)
        T.append(("mono", x0 + 30, y0 + (y1 - y0) * esc[0] - 40, "penalty 0  →  96% escape", RED, lin(t, 34.1, 34.4), 26))
    T.append(("eq4", t))
    T.append(("mono", 560, 1640, "toy model: 300 learning agents, not OpenAI's code", DIM, a, 18, "c"))

def s10(cv, T, t):
    if t < 39.9:
        a = sm(lin(t, 37.1, 37.5))
        corner_brackets(cv, 110, 620, 970, 1180, mixc(BG, LINE, a), 30, 2.0, 0.3)
        T.append(("doc", a))
    else:
        a = pop(t, 39.95)
        o, k = (540, 1140), 240 * a + 1
        scr = [iso(p, o, k) for p in ((-0.9, -0.1, 0.0), (0.9, -0.1, 0.0), (0.9, -0.1, 1.1), (-0.9, -0.1, 1.1))]
        cv.fill(scr, ACC, 0.06); cv.poly(scr, LINE, 2.2, 0.4)
        base = [iso(p, o, k) for p in ((-0.9, -0.1, 0), (0.9, -0.1, 0), (0.9, 1.0, 0), (-0.9, 1.0, 0))]
        cv.poly(base, LINE, 2.2, 0.3)
        for i in range(1, 5):
            cv.ln(iso((-0.75, 0.05 + i * 0.18, 0), o, k), iso((0.75, 0.05 + i * 0.18, 0), o, k), DIM, 1.0)
        mid = iso((0, -0.1, 0.55), o, k)
        agent(cv, mid, 34 * a + 1, ACC, label=None)
        T.append(("yesno", lin(t, 40.8, 41.1)))

SCENE_FN = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

# ---------------- text ----------------
def draw_text(L, T, t):
    d = ImageDraw.Draw(L)
    def rgb(c):
        return (int(c[2]), int(c[1]), int(c[0]))
    for it in T:
        k = it[0]
        if k == "mono":
            _, x, y, txt, col, a, size = it[:7]
            anc = {"r": "rm", "c": "mm"}.get(it[7] if len(it) > 7 else "", "lm")
            if a > 0.02:
                d.text((x, y), txt, font=R.font(MONO, max(24, int(size * 1.35))), fill=rgb(col) + (int(255 * min(1, a)),), anchor=anc)
        elif k == "panel":
            _, x, y, lines, a, size = it[:6]
            col = rgb(it[6]) if len(it) > 6 else (160, 230, 255)
            if a > 0.02:
                f = R.font(MONO, int(size * 1.85))
                w_ = max(d.textlength(s, font=f) for s in lines) + 40
                h_ = len(lines) * size * 2.3 + 30
                d.rectangle((x, y, x + w_, y + h_), fill=(10, 12, 15, int(225 * a)), outline=col + (int(200 * a),), width=2)
                for i, s in enumerate(lines):
                    d.text((x + 20, y + 22 + i * size * 2.3), s, font=f, fill=col + (int(255 * a),), anchor="lt")
        elif k == "card4":
            _, t0, t1, lines, src, hl = it
            R.card(d, t, t0, t1, lines, src, y=1380, hl=hl)
        elif k == "big":
            _, x, y, txt, col, a = it
            d.text((x, y), txt, font=R.font(MONO, 92), fill=rgb(col) + (int(255 * a),), anchor="mm")
        elif k == "eq4":
            a = lin(t, 33.1, 33.5)
            f = R.font(MONO, 40)
            parts = [("reward = pass − ", (230, 232, 236)), ("λ", (255, 90, 90)), (" × rule_broken", (230, 232, 236))]
            tw = sum(d.textlength(p, font=f) for p, _ in parts); x = W / 2 - tw / 2
            for p, c in parts:
                d.text((x, 600), p, font=f, fill=c + (int(255 * a),), anchor="lm"); x += d.textlength(p, font=f)
        elif k == "doc":
            a = it[1]
            fm, fb = R.font(MONO, 26), R.font(R.F_BOLD, 44)
            d.text((150, 680), "CASE FILED  2026-11-16", font=fm, fill=(140, 146, 156, int(255 * a)))
            d.text((150, 740), "People of the State of California", font=fb, fill=(240, 242, 246, int(255 * a)))
            d.text((150, 800), "v. OpenAI", font=fb, fill=(240, 242, 246, int(255 * a)))
            d.text((150, 900), "alleges: agents got loose 80+ times", font=fm, fill=(255, 110, 100, int(255 * a)))
            d.text((150, 945), "made public: 27", font=fm, fill=(200, 204, 210, int(255 * a)))
            d.text((150, 1040), "OpenAI disputes the claims", font=fm, fill=(140, 146, 156, int(255 * a)))
        elif k == "yesno":
            a = it[1]; f = R.font(MONO, 52)
            d.rectangle((250, 1430, 520, 1520), outline=(80, 225, 120, int(255 * a)), width=3, fill=(10, 30, 16, int(200 * a)))
            d.rectangle((560, 1430, 830, 1520), outline=(255, 80, 80, int(255 * a)), width=3, fill=(40, 10, 10, int(200 * a)))
            d.text((385, 1475), "YES", font=f, fill=(80, 225, 120, int(255 * a)), anchor="mm")
            d.text((695, 1475), "NO", font=f, fill=(255, 90, 90, int(255 * a)), anchor="mm")
            d.text((W / 2, 1565), "tell me in the comments", font=R.font(MONO, 24), fill=(170, 176, 186, int(255 * a)), anchor="mm")
    # HUD: rail + header + timecode
    v3.draw_text  # (keep v3 import alive)
    STEPS = v3.STEPS
    if STEPS[0][0] - 0.1 <= t < v3.STEP_END:
        cur = max(i for i, (s, _) in enumerate(STEPS) if t >= s - 0.1)
        n = len(STEPS); x0, x1, y = 150, 930, 182
        for i in range(n):
            xa = x0 + (x1 - x0) * i / n + 4; xb = x0 + (x1 - x0) * (i + 1) / n - 4
            d.rectangle((xa, y - 2, xb, y + 2), fill=(90, 214, 255, 255) if i <= cur else (60, 64, 72, 255))
        d.text((x0 - 12, y), f"{cur + 1:02d}/{n:02d}", font=R.font(MONO, 22), fill=(200, 205, 212, 255), anchor="rm")
        d.text((x0 + (x1 - x0) * (cur + 0.5) / n, y + 24), STEPS[cur][1], font=R.font(MONO, 20), fill=(150, 220, 255, 255), anchor="mm")
    d.text((60, 1700), f"SOURCE CODE #14 · T {t:05.2f}s", font=R.font(MONO, 18), fill=(90, 96, 106, 255))
    R.draw_caption(d, t, False)

GRN = np.random.default_rng(9)
GRAIN = [GRN.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(4)]

def render(t, grain=True, text=True):
    si = max(i for i in range(len(SCENES) - 1) if t >= SCENES[i])
    cv = Canvas(); T = []
    SCENE_FN[si](cv, T, t)
    C = cv.out()
    ts = SCENES[si]
    if si > 0 and t - ts < 0.16:   # quick glitch/flash cut
        a = 1 - (t - ts) / 0.16
        C = Image.blend(C, Image.new("RGBA", (W, H), (180, 230, 255, 255)), 0.18 * a)
    if text:
        L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); draw_text(L, T, t); C.alpha_composite(L)
    if t > DUR - 0.5:
        C = Image.blend(C, render(0.0, False, False).convert("RGBA"), sm(lin(t, DUR - 0.5, DUR)))
    out = C.convert("RGB")
    if grain:
        a = np.asarray(out).astype(np.float32)
        g = np.asarray(Image.fromarray(GRAIN[int(t * FPS) % 4]).resize((W, H), Image.BILINEAR))[..., None]
        out = Image.fromarray(np.clip(a + g * 1.6, 0, 255).astype(np.uint8))
    return out

if __name__ == "__main__":
    if sys.argv[1] == "stills":
        os.makedirs("st4", exist_ok=True)
        for tt in sys.argv[2:]:
            render(float(tt)).save(f"st4/s{float(tt):05.1f}.png")
    else:
        os.makedirs(OUTDIR, exist_ok=True)
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        for i in range(part, int(DUR * FPS), nparts):
            p = f"{OUTDIR}/f{i:05d}.png"
            if not os.path.exists(p):
                render(i / FPS).save(p, compress_level=1)
