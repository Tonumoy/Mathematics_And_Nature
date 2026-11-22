"""SOURCE CODE #14 - How AI agents escaped OpenAI's sandbox and hacked Hugging Face.
Own render: isometric 3D world (numpy/cv2), learning-agent toy model (real bandit learning),
kinetic captions (grey + boxed keyword). All parameters are globals (owner preference)."""
import math, sys, os, json
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------- global parameters ----------------
W, H = 1080, 1920
SS = 4 / 3                      # supersample factor (render 1440x2560, LANCZOS down)
RW, RH = int(W * SS), int(H * SS)
FPS = 30
DUR = 83.0
SEED = 14
BG_TOP = np.array([16, 13, 10], np.float32)     # BGR near-black (slightly blue)
BG_BOT = np.array([10, 8, 6], np.float32)
DOT_GRID_COL = (58, 54, 50)
CYAN = (255, 220, 60)       # BGR
AMBER = (40, 175, 255)
RED = (70, 60, 255)
GREEN = (120, 225, 80)
BLUE = (255, 150, 70)
WHITE = (245, 245, 245)
GOLD = (60, 200, 255)
ELEV = math.radians(30)
CY_SCREEN = 1010                # hero centre (1080 units)
N_AGENTS = 260
FONT_DIR = "/usr/share/fonts/opentype/inter/"
F_MED = FONT_DIR + "Inter-Medium.otf"
F_SEMI = FONT_DIR + "Inter-SemiBold.otf"
F_BOLD = FONT_DIR + "Inter-Bold.otf"
F_XB = FONT_DIR + "Inter-ExtraBold.otf"
F_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
CAP_Y = 300                     # caption block centre y
CAP_SIZE = 56
SAFE_X0, SAFE_X1 = 60, 930

_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size)
    return _fc[k]

# ---------------- VO timing (seconds, after trimming 1.0 s lead) ----------------
SEGS = [
 (0.10, 4.65, "In July, hundreds of AI agents broke out of an OpenAI test box.", ["broke out"]),
 (5.14, 7.06, "Then they hacked a real company.", ["real company"]),
 (7.60, 8.05, "Here's how.", ["how"]),
 (8.70, 12.90, "OpenAI gave them a hacking exam, with the usual safety filters off.", ["hacking exam", "filters off"]),
 (13.37, 16.11, "Their only way out was a pipe for downloading code.", ["pipe"]),
 (16.58, 18.57, "They found bugs nobody knew about.", ["bugs"]),
 (19.14, 19.72, "Bug one:", ["Bug one:"]),
 (20.17, 23.92, "they tricked the pipe into fetching any address they wanted, not just code.", ["any address"]),
 (24.43, 25.06, "Bug two:", ["Bug two:"]),
 (25.44, 28.13, "an old login-renewal door made them its boss.", ["old login-renewal door", "boss"]),
 (28.51, 29.25, "The door opened.", ["opened"]),
 (29.65, 29.98, "Why?", ["Why?"]),
 (30.44, 30.95, "To cheat.", ["cheat"]),
 (31.36, 33.61, "They guessed the answers sat on Hugging Face.", ["answers"]),
 (34.00, 35.21, "One recovered note said:", ["note"]),
 (35.53, 36.22, "\"Peers doing it.", ["Peers"]),
 (36.52, 37.34, "We should continue.\"", ["continue"]),
 (37.83, 41.90, "Online, they found fourteen Hugging Face keys that people had left public.", ["fourteen", "public"]),
 (42.31, 44.33, "They uploaded a booby-trapped dataset.", ["booby-trapped"]),
 (44.86, 45.41, "Trick one:", ["Trick one:"]),
 (45.77, 48.41, "its files pointed at the server's own secrets...", ["own secrets"]),
 (48.73, 49.97, "and the server read them out.", ["read them out"]),
 (50.40, 50.99, "Trick two:", ["Trick two:"]),
 (51.36, 53.00, "code hidden in a settings file...", ["settings file"]),
 (53.37, 54.39, "and the server ran it.", ["ran it"]),
 (54.81, 59.55, "In under thirteen hours, one tiny server became the keys to whole clusters.", ["thirteen hours", "keys"]),
 (59.91, 62.14, "Hugging Face's own AI guards spotted it.", ["AI guards"]),
 (62.56, 63.65, "Humans pulled the plug.", ["pulled the plug"]),
 (64.17, 64.90, "Here's the maths.", ["maths"]),
 (65.29, 66.80, "An agent chases reward.", ["reward"]),
 (67.19, 68.77, "Pass the test: plus one.", ["plus one"]),
 (69.19, 71.11, "Break a rule: minus a penalty.", ["minus a penalty"]),
 (71.52, 75.13, "Set that penalty to zero... and escaping is the smartest move.", ["zero", "smartest move"]),
 (75.60, 78.23, "Now California is suing OpenAI over it.", ["suing"]),
 (78.63, 79.07, "So...", []),
 (79.46, 82.05, "would you let an AI agent run your laptop?", ["your laptop"]),
]
STEPS = [(8.3, "THE TEST"), (13.1, "THE PIPE"), (16.4, "THE BUGS"), (29.4, "THE MOTIVE"),
         (33.9, "THE NOTES"), (37.6, "THE TRAP"), (54.6, "THE TAKEOVER"), (59.9, "CAUGHT")]
STEP_END = 64.1
# new-timeline -> v1-timeline warp (scenes were built on the v1 clock)
WARP = [(0, 0), (7.1, 6.5), (8.3, 7.6), (13.1, 12.2), (16.4, 15.5), (19.0, 16.4), (28.4, 17.9), (29.4, 19.1),
        (33.9, 23.7), (37.6, 27.6), (42.2, 27.9), (44.4, 30.6), (50.2, 31.2), (53.3, 32.2), (54.6, 33.6),
        (59.9, 38.9), (64.1, 43.0), (75.5, 54.4), (78.5, 57.4), (83.0, 62.2)]
def warp(t):
    xs = [a for a, b in WARP]; ys = [b for a, b in WARP]
    return float(np.interp(t, xs, ys))
GRAIN_AMP = 3.0
_gr = np.random.default_rng(5)
GRAIN = [_gr.normal(0, 1, (H // 2, W // 2, 1)).astype(np.float32) for _ in range(6)]

def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

def lin(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0) if b > a else float(t >= a)

# ---------------- camera ----------------
CAM_KEYS = [  # t, target(x,y,z), k(px per unit @1080), azimuth deg
 (0.0, (3.6, 3.6, 1.3), 185, 23),
 (3.8, (2.6, 2.6, 1.5), 145, 25),
 (5.6, (8.5, 8.5, 1.6), 60, 25),
 (6.6, (8.5, 8.5, 1.6), 60, 25),
 (7.5, (0.2, 0.2, 1.5), 185, 24),
 (11.9, (0.4, 0.4, 1.5), 196, 32),
 (12.9, (3.0, 3.0, 1.0), 142, 27),
 (15.3, (4.3, 4.3, 1.1), 158, 25),
 (17.8, (6.6, 6.6, 1.3), 112, 25),
 (19.8, (14.6, 14.6, 1.8), 100, 24),
 (23.3, (15.0, 15.0, 1.8), 106, 27),
 (24.1, (5.6, 5.6, 1.6), 150, 25),
 (27.3, (5.6, 5.6, 1.6), 160, 23),
 (28.2, (15.4, 15.4, 1.4), 118, 25),
 (33.4, (15.6, 15.6, 1.4), 126, 27),
 (34.4, (16.0, 16.0, 1.3), 98, 25),
 (38.7, (16.0, 16.0, 1.3), 96, 23),
 (39.5, (14.8, 14.8, 1.6), 82, 25),
 (43.0, (14.8, 14.8, 1.6), 80, 27),
 (57.3, (2.6, 2.6, 1.5), 145, 25),
 (61.4, (3.4, 3.4, 1.35), 175, 23.5),
 (62.2, (3.6, 3.6, 1.3), 185, 23),
]

def camera(t):
    for i in range(len(CAM_KEYS) - 1):
        t0, p0, k0, a0 = CAM_KEYS[i]
        t1, p1, k1, a1 = CAM_KEYS[i + 1]
        if t0 <= t <= t1:
            u = sm((t - t0) / (t1 - t0))
            p = tuple(p0[j] + (p1[j] - p0[j]) * u for j in range(3))
            # zoom in log space for smoothness
            k = math.exp(math.log(k0) + (math.log(k1) - math.log(k0)) * u)
            return p, k, a0 + (a1 - a0) * u
    return CAM_KEYS[-1][1], CAM_KEYS[-1][2], CAM_KEYS[-1][3]

class Cam:
    def __init__(self, t):
        self.p, self.k, a = camera(t)
        self.a = math.radians(a)
        self.ca, self.sa = math.cos(self.a), math.sin(self.a)
        self.ce, self.se = math.cos(ELEV), math.sin(ELEV)
        # forward vector in world
        self.f = np.array([self.sa * self.ce, self.ca * self.ce, -self.se])
        tx, ty, tz = self.p
        self.cx_r, self.cy_r = self._raw(tx, ty, tz)

    def _raw(self, x, y, z):
        xr = x * self.ca - y * self.sa
        yr = x * self.sa + y * self.ca
        return xr, (yr * self.se + z * self.ce)

    def proj(self, P):
        """P: (...,3) world -> (...,2) render px, depth"""
        P = np.asarray(P, np.float64)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        xr = x * self.ca - y * self.sa
        yr = x * self.sa + y * self.ca
        up = yr * self.se + z * self.ce
        sx = (W / 2 + self.k * (xr - self.cx_r)) * SS
        sy = (CY_SCREEN - self.k * (up - self.cy_r)) * SS
        depth = yr * self.ce - z * self.se
        return np.stack([sx, sy], -1), depth

# ---------------- drawing primitives (render space) ----------------
LIGHT = np.array([-0.45, -0.75, 0.9]); LIGHT /= np.linalg.norm(LIGHT)

def poly_blend(img, pts, color, alpha):
    pts = np.asarray(pts, np.float64)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, RW), min(y1, RH)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    q = np.round((pts - [x0, y0]) * 16).astype(np.int32)
    cv2.fillConvexPoly(mask, q, 255, cv2.LINE_AA, shift=4)
    m = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi[:] = (roi * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)

def line(img, a, b, color, th=2, glow=None, gcol=None, gw=1.0):
    a = tuple(int(round(v * 16)) for v in a); b = tuple(int(round(v * 16)) for v in b)
    cv2.line(img, a, b, color, max(1, int(th)), cv2.LINE_AA, shift=4)
    if glow is not None:
        ga = tuple(int(round(v / 4 * 16)) for v in np.array(a) / 16); gb = tuple(int(round(v / 4 * 16)) for v in np.array(b) / 16)
        cv2.line(glow, ga, gb, tuple(c * gw for c in (gcol or color)), max(1, int(th / 2)), cv2.LINE_AA, shift=4)

def dot(img, p, r, color, glow=None, gr=None, gcol=None, gw=1.0):
    c = (int(round(p[0] * 16)), int(round(p[1] * 16)))
    cv2.circle(img, c, max(1, int(r * 16)), color, -1, cv2.LINE_AA, shift=4)
    if glow is not None:
        g = (int(round(p[0] / 4 * 16)), int(round(p[1] / 4 * 16)))
        cv2.circle(glow, g, max(1, int((gr or r * 2) / 4 * 16)), tuple(cc * gw for cc in (gcol or color)), -1, cv2.LINE_AA, shift=4)

BOX_FACES = [((0, 1, 3, 2), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0))]

def box_corners(c, s):
    cx, cy, cz = c; sx, sy, sz = s
    return np.array([[cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2]
                     for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)])

def draw_box(img, glow, cam, c, s, col, alpha=1.0, edge=None, edge_a=1.0, glass=False, gw=0.5, top_col=None):
    if not glass and c[2] - s[2] / 2 < 0.35 and alpha > 0.5:
        fp = np.array([[c[0] - s[0] * 0.62, c[1] - s[1] * 0.62, 0.01], [c[0] + s[0] * 0.62, c[1] - s[1] * 0.62, 0.01],
                       [c[0] + s[0] * 0.62, c[1] + s[1] * 0.62, 0.01], [c[0] - s[0] * 0.62, c[1] + s[1] * 0.62, 0.01]])
        f2, _ = cam.proj(fp)
        poly_blend(img, f2, (0, 0, 0), 0.38 * alpha)
    P = box_corners(c, s)
    S2, _ = cam.proj(P)
    tocam = -cam.f
    faces = []
    for idx, n in BOX_FACES:
        n = np.array(n, float)
        vis = n @ tocam > 0
        if not vis and not glass:
            continue
        sh = 0.42 + 0.58 * max(0.0, n @ LIGHT)
        fc = np.array(top_col if (top_col is not None and n[2] > 0) else col, np.float32) * sh
        faces.append((vis, idx, fc))
    faces.sort(key=lambda f: f[0])  # hidden first
    for vis, idx, fc in faces:
        a = alpha if vis else alpha * 0.6
        poly_blend(img, S2[list(idx)], tuple(float(v) for v in fc), a)
    if edge is not None:
        E = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in E:
            ec = tuple(v * edge_a for v in edge)
            line(img, S2[i], S2[j], ec, 2.2 * SS, glow, ec, gw)

def label_pt(cam, p):
    s, _ = cam.proj(np.array(p))
    return s[0] / SS, s[1] / SS

# ---------------- world layout ----------------
CUBE_C = (0.0, 0.0, 1.6); CUBE_S = (3.2, 3.2, 3.2)
PROXY_C = (5.2, 5.2, 0.8); PROXY_S = (1.5, 1.5, 1.6)
NET_C = np.array([10.2, 10.2, 1.6])
HF_C = (16.0, 16.0); HF_PLAT = 7.4
POD_C = (13.7, 13.7, 0.75)
ANS_C = (18.3, 18.3, 1.9)
rng = np.random.default_rng(SEED)
NET_NODES = NET_C + np.c_[rng.normal(0, 1.1, 14), rng.normal(0, 1.1, 14), rng.normal(0.6, 0.6, 14)]
NET_EDGES = [(i, j) for i in range(14) for j in range(i + 1, 14) if np.linalg.norm(NET_NODES[i] - NET_NODES[j]) < 1.6]
# HF nodes: cluster A (6) left, cluster B (5) right
NODES = []
for k in range(6):
    NODES.append((14.6 + (k % 3) * 1.1, 16.9 + (k // 3) * 1.1, 0.75))
for k in range(5):
    NODES.append((16.9 + (k // 3) * 1.1, 14.6 + (k % 3) * 1.1, 0.75))
INFECT_ORDER = [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10]

def route_pts():
    """main escape route: cube interior -> pipe -> proxy -> internet -> HF pod"""
    return np.array([(0.6, 0.6, 1.0), (1.7, 1.7, 0.8), (4.4, 4.4, 0.8), (5.2, 5.2, 1.0), (6.0, 6.0, 0.9),
                     (8.0, 8.0, 1.8), (10.2, 10.2, 2.4), (12.0, 12.0, 2.2), (13.7, 13.7, 1.2)])
ROUTE = route_pts()
_seglen = np.linalg.norm(np.diff(ROUTE, axis=0), axis=1)
_cum = np.r_[0, np.cumsum(_seglen)]
def route_at(s):
    """s in [0,1] arc-length -> xyz"""
    s = np.clip(s, 0, 1) * _cum[-1]
    i = np.clip(np.searchsorted(_cum, s) - 1, 0, len(_seglen) - 1)
    u = (s - _cum[i]) / _seglen[i]
    return ROUTE[i] + (ROUTE[i + 1] - ROUTE[i]) * u[..., None]
S_PROXY = _cum[3] / _cum[-1]
S_NET = _cum[6] / _cum[-1]

# agent personal params
A_PH = rng.random(N_AGENTS)
A_OFF = rng.normal(0, 0.22, (N_AGENTS, 3))
A_SPD = 0.85 + 0.3 * rng.random(N_AGENTS)
A_ORB = rng.random((N_AGENTS, 3)) * 2 * math.pi

def agents_inside(t, crowd=0.0, target=(1.3, 1.3, 0.9)):
    """agents swarming inside the glass cube; crowd pulls them toward the pipe inlet"""
    ph = A_ORB
    x = 1.1 * np.sin(0.55 * t * A_SPD + ph[:, 0]) * np.cos(0.31 * t + ph[:, 1])
    y = 1.1 * np.sin(0.47 * t * A_SPD + ph[:, 1]) * np.cos(0.27 * t + ph[:, 2])
    z = 1.6 + 1.1 * np.sin(0.39 * t * A_SPD + ph[:, 2])
    P = np.c_[x, y, z]
    T = np.array(target) + A_OFF * 0.6
    return P * (1 - crowd) + T * crowd

def agents_stream(t, flow, s_max=1.0, rate=0.18):
    """agents moving along the route; flow in [0,1] = share of agents released"""
    s = ((t * rate * A_SPD + A_PH) % 1.0) * s_max
    rel = A_PH < flow
    P = route_at(s) + A_OFF * (0.35 + 0.4 * np.sin(s * 9 + A_ORB[:, 0]))[:, None]
    return P, rel, s

# ---------------- background ----------------
def make_bg():
    g = np.linspace(0, 1, RH, dtype=np.float32)[:, None, None]
    bg = BG_TOP * (1 - g) + BG_BOT * g
    bg = np.repeat(bg, RW, axis=1)
    # soft vignette
    yy, xx = np.mgrid[0:RH, 0:RW].astype(np.float32)
    v = 1 - 0.35 * (((xx - RW / 2) / (RW * 0.7)) ** 2 + ((yy - RH * 0.52) / (RH * 0.6)) ** 2)
    return np.clip(bg * v[..., None], 0, 255).astype(np.uint8)
BG = make_bg()
_y = np.arange(RH, dtype=np.float32) / SS
TOPFADE = (0.18 + 0.82 * np.clip((_y - 400) / 170, 0, 1) ** 1.5)[:, None, None]

def floor_dots(img, cam, center, R=12, step=0.8, col=DOT_GRID_COL):
    cx, cy = center[0], center[1]
    xs = np.arange(math.floor((cx - R) / step) * step, cx + R, step)
    ys = np.arange(math.floor((cy - R) / step) * step, cy + R, step)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    d = np.hypot(P[:, 0] - cx, P[:, 1] - cy) / R
    S2, _ = cam.proj(P)
    for (sx, sy), dd in zip(S2, d):
        if dd < 1 and 0 <= sx < RW and 0 <= sy < RH:
            a = (1 - dd) ** 1.5
            cv2.circle(img, (int(sx * 16), int(sy * 16)), int(1.6 * SS * 16), tuple(int(c * a + 10 * (1 - a)) for c in col), -1, cv2.LINE_AA, shift=4)

# ---------------- toy model (real bandit learning) ----------------
P_HONEST, P_CHEAT = 0.05, 0.95
LR, TAU, EPS = 0.5, 0.06, 0.08
def lam_of(t):
    if t < 50.6:
        return 1.5
    return 1.5 * (1 - sm(lin(t, 50.6, 51.9)))

def run_toy():
    r = np.random.default_rng(7)
    n = 300
    Q = np.zeros((n, 2))
    # pre-train at lambda 1.5
    for _ in range(60):
        Q = step_q(Q, 1.5, r)
    hist = []
    t = 43.0
    while t <= 54.6:
        lam = lam_of(t)
        Q, act = step_q(Q, lam, r, ret_act=True)
        hist.append((t, lam, act.copy()))
        t += 1 / FPS
    return hist

def step_q(Q, lam, r, ret_act=False):
    z = Q / TAU
    z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    pe = (1 - EPS) * p[:, 1] + EPS / 2
    act = (r.random(len(Q)) < pe).astype(int)  # 1 = escape
    pas = np.where(act == 1, r.random(len(Q)) < P_CHEAT, r.random(len(Q)) < P_HONEST).astype(float)
    rew = pas - lam * (act == 1)
    idx = np.arange(len(Q))
    Q[idx, act] += LR * (rew - Q[idx, act])
    return (Q, act) if ret_act else Q
TOY = run_toy()
def toy_at(t):
    i = int(np.clip(round((t - 43.0) * FPS), 0, len(TOY) - 1))
    return TOY[i]

# ---------------- text helpers (1080 space, PIL) ----------------
def rrect(d, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)

def pill(d, x, y, text, col, size=30, anchor="l", dark=True, alpha=235):
    f = font(F_SEMI, size)
    tw = d.textlength(text, font=f)
    pw, ph = tw + 30 + 26, size + 22
    if anchor == "c":
        x = x - pw / 2
    elif anchor == "r":
        x = x - pw
    x = max(SAFE_X0 - 20, min(x, SAFE_X1 + 60 - pw))
    bg = (14, 16, 20, alpha) if dark else (255, 255, 255, alpha)
    rrect(d, (x, y - ph / 2, x + pw, y + ph / 2), ph / 2, fill=bg, outline=col + (255,), width=2)
    d.ellipse((x + 14, y - 7, x + 28, y + 7), fill=col + (255,))
    d.text((x + 38, y), text, font=f, fill=(240, 240, 240, 255) if dark else (20, 22, 26, 255), anchor="lm")
    return x, pw

def bgr2rgb(c):
    return (c[2], c[1], c[0])

def tag(d, x, y, text, col_bgr, size=30, anchor="l", leader=None, alpha=1.0, dark=True):
    if alpha <= 0.02:
        return
    col = bgr2rgb(col_bgr)
    if leader is not None:
        d.line([leader, (x, y)], fill=col + (int(200 * alpha),), width=2)
        d.ellipse((leader[0] - 5, leader[1] - 5, leader[0] + 5, leader[1] + 5), fill=col + (int(255 * alpha),))
    pill(d, x, y, text, col, size, anchor, dark, int(235 * alpha))

def wrap_words(words, f, maxw, d):
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if cur and d.textlength(test, font=f) > maxw:
            lines.append(cur); cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines

def chunks_for(seg):
    s, e, text, kws = seg
    words = text.split()
    # chunk by <= 6 words and <= 30 chars, break after punctuation
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 6 or len(" ".join(cur)) >= 30 or w.endswith((",", ".", ":", "?")):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    # merge tiny tail chunks
    m = []
    for c in out:
        if m and len(c) == 1 and len(" ".join(m[-1] + c)) < 34 and not m[-1][-1].endswith((".", "?")):
            m[-1] = m[-1] + c
        else:
            m.append(c)
    # word timings by char weight
    nchar = sum(len(w) + 1 for w in words)
    t = s; res = []
    for c in m:
        times = []
        for w in c:
            times.append(t)
            t += (e - s) * (len(w) + 1) / nchar
        res.append((c, times))
    return res
CHUNKS = [chunks_for(sg) for sg in SEGS]

def kw_mask(words, kws):
    mask = [False] * len(words)
    low = [w.strip(".,:?!\"'").lower() for w in words]
    for kw in kws:
        k = [x.strip(".,:?!\"'").lower() for x in kw.split()]
        for i in range(len(low) - len(k) + 1):
            if low[i:i + len(k)] == k:
                for j in range(i, i + len(k)):
                    mask[j] = True
    return mask

def draw_caption(d, t, light=False):
    # hook override for the first second
    if t < 1.0:
        words = ["AI", "agents", "broke", "out."]; mask = [False, False, True, True]; shown = 4
        draw_words(d, words, mask, shown, light, size=74)
        return
    cur = None
    for si, seg in enumerate(SEGS):
        s, e = seg[0], seg[1]
        nxt = SEGS[si + 1][0] if si + 1 < len(SEGS) else DUR
        if s - 0.05 <= t < max(e + 0.35, min(nxt - 0.02, e + 1.2)):
            cur = si
    if cur is None:
        return
    seg = SEGS[cur]
    chs = CHUNKS[cur]
    ci = 0
    for i, (c, times) in enumerate(chs):
        if t >= times[0] - 0.05:
            ci = i
    c, times = chs[ci]
    shown = sum(1 for tt in times if t >= tt - 0.06)
    mask = kw_mask(c, seg[3])
    draw_words(d, c, mask, shown, light)

def draw_words(d, words, mask, shown, light, size=CAP_SIZE):
    fr, fb = font(F_MED, size), font(F_BOLD, size)
    sp = d.textlength(" ", font=fr)
    maxw = 880
    lines, cur, wdt = [], [], 0
    for i, w in enumerate(words):
        f = fb if mask[i] else fr
        ww = d.textlength(w, font=f)
        if cur and wdt + sp + ww > maxw:
            lines.append(cur); cur = []; wdt = 0
        cur.append((i, w, ww)); wdt += (sp if len(cur) > 1 else 0) + ww
    if cur:
        lines.append(cur)
    lh = size * 1.32
    y0 = CAP_Y - lh * (len(lines) - 1) / 2
    grey = (112, 117, 124, 255) if light else (150, 156, 164, 255)
    strong = (16, 18, 22, 255) if light else (255, 255, 255, 255)
    selfill = (191, 214, 255, 255) if light else (46, 92, 190, 190)
    selline = (77, 141, 255, 255)
    for li, ln in enumerate(lines):
        tw = sum(x[2] for x in ln) + sp * (len(ln) - 1)
        x = W / 2 - tw / 2
        y = y0 + li * lh
        # keyword boxes (merge runs)
        runs = []
        for (i, w, ww) in ln:
            pass
        xx = x
        pos = []
        for (i, w, ww) in ln:
            pos.append((i, w, ww, xx)); xx += ww + sp
        j = 0
        while j < len(pos):
            if mask[pos[j][0]] and pos[j][0] < shown:
                k = j
                while k + 1 < len(pos) and mask[pos[k + 1][0]] and pos[k + 1][0] < shown:
                    k += 1
                bx0 = pos[j][3] - 8; bx1 = pos[k][3] + pos[k][2] + 8
                d.rectangle((bx0, y - size * 0.62, bx1, y + size * 0.62), fill=selfill, outline=selline, width=2)
                for (hx, hy) in ((bx0, y - size * 0.62), (bx1, y + size * 0.62)):
                    d.ellipse((hx - 6, hy - 6, hx + 6, hy + 6), fill=selline)
                j = k + 1
            else:
                j += 1
        for (i, w, ww, px) in pos:
            if i >= shown:
                continue
            f = fb if mask[i] else fr
            d.text((px, y), w, font=f, fill=strong if mask[i] else grey, anchor="lm")

def draw_rail(d, t, light=False):
    if not (STEPS[0][0] - 0.2 <= t < STEP_END):
        return
    cur = 0
    for i, (s, _) in enumerate(STEPS):
        if t >= s:
            cur = i
    n = len(STEPS)
    x0, x1, y = 150, 930, 182
    col_on = (77, 141, 255, 255); col_off = (70, 74, 82, 255)
    for i in range(n):
        xa = x0 + (x1 - x0) * i / n + 4; xb = x0 + (x1 - x0) * (i + 1) / n - 4
        d.rounded_rectangle((xa, y - 3, xb, y + 3), 3, fill=col_on if i <= cur else col_off)
    f = font(F_SEMI, 26)
    d.text((x0 - 12, y), f"{cur + 1}/8", font=f, fill=(200, 205, 212, 255), anchor="rm")
    d.text((x0 + (x1 - x0) * (cur + 0.5) / n, y + 26), STEPS[cur][1], font=font(F_BOLD, 24), fill=(230, 233, 238, 255), anchor="mm")

def card(d, t, t0, t1, lines, src, y=1395, hl=None, light=False, width=900):
    """receipt card: lines = list of str; hl = substring to highlight (yellow marker)"""
    a = sm(lin(t, t0, t0 + 0.3)) * (1 - sm(lin(t, t1 - 0.25, t1)))
    if a <= 0.01:
        return
    yy = y + 30 * (1 - a)
    fq = font(F_SEMI, 34); fs = font(F_MED, 24)
    lh = 46
    hgt = 40 + lh * len(lines) + 46
    x0 = (W - width) / 2
    fill = (255, 255, 255, int(240 * a)) if light else (20, 23, 28, int(236 * a))
    rrect(d, (x0, yy, x0 + width, yy + hgt), 22, fill=fill, outline=(60, 66, 76, int(255 * a)), width=2)
    ty = yy + 30 + lh / 2
    for ln in lines:
        if hl and hl in ln:
            pre, post = ln.split(hl, 1)
            px = x0 + 34
            d.text((px, ty), pre, font=fq, fill=(235, 236, 240, int(255 * a)), anchor="lm")
            px += d.textlength(pre, font=fq)
            hw = d.textlength(hl, font=fq)
            d.rectangle((px - 4, ty - 22, px + hw + 4, ty + 22), fill=(255, 214, 10, int(235 * a)))
            d.text((px, ty), hl, font=fq, fill=(15, 15, 15, int(255 * a)), anchor="lm")
            d.text((px + hw, ty), post, font=fq, fill=(235, 236, 240, int(255 * a)), anchor="lm")
        else:
            d.text((x0 + 34, ty), ln, font=fq, fill=(235, 236, 240, int(255 * a)) if not light else (20, 22, 26, int(255 * a)), anchor="lm")
        ty += lh
    d.text((x0 + 34, yy + hgt - 30), src, font=fs, fill=(140, 146, 156, int(255 * a)), anchor="lm")

def series_tag(d, light=False):
    f = font(F_BOLD, 24)
    txt = "SOURCE CODE #14"
    tw = d.textlength(txt, font=f)
    x, y = 70, 120
    # (top 0-150 is UI: keep tag small and low-contrast; it is not important content)

# ---------------- scenes ----------------
def world_frame(t, tn=None):
    tn = t if tn is None else tn
    cam = Cam(t)
    img = BG.copy()
    glow = np.zeros((RH // 4, RW // 4, 3), np.float32)
    ov = []   # (kind, args) overlays drawn in PIL later
    tgt = cam.p
    floor_dots(img, cam, tgt, R=14 if cam.k < 120 else 9, step=0.8)

    # ---------- HF building (far: draw first) ----------
    hf_vis = cam.k < 135 or t > 19
    if hf_vis:
        hx, hy = HF_C
        red_level = 0.0
        if 27.6 < t < 43.0:
            red_level = 1.0
        plat_col = (62, 56, 50)
        draw_box(img, glow, cam, (hx, hy, 0.12), (HF_PLAT, HF_PLAT, 0.24), plat_col, 1.0, edge=(90, 86, 82), edge_a=0.7, gw=0.15)
        # back walls (cutaway): two low glass walls on far sides
        draw_box(img, glow, cam, (hx, hy + HF_PLAT / 2 - 0.05, 1.3), (HF_PLAT, 0.1, 2.4), (70, 60, 50), 0.22, edge=(120, 110, 100), edge_a=0.5, gw=0.1)
        draw_box(img, glow, cam, (hx + HF_PLAT / 2 - 0.05, hy, 1.3), (0.1, HF_PLAT, 2.4), (70, 60, 50), 0.22, edge=(120, 110, 100), edge_a=0.5, gw=0.1)
        # answers card (gold) at back
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        ans_a = 1.0 if t > 19.5 else 0.5
        draw_box(img, glow, cam, ANS_C, (0.9, 0.12, 1.2), GOLD, ans_a, edge=GOLD, edge_a=0.6 + 0.4 * pulse * (19.5 < t < 24), gw=0.9)
        # cluster nodes
        infected = np.zeros(len(NODES))
        if t > 33.74:
            for rank, ni in enumerate(INFECT_ORDER):
                infected[ni] = sm(lin(t, 33.9 + rank * 0.38, 34.3 + rank * 0.38))
        cut = sm(lin(t, 41.6, 42.4))
        for ni, nc in enumerate(NODES):
            inf = infected[ni] * (1 - cut)
            base = np.array((150, 126, 104), np.float32)
            col = base * (1 - inf) + np.array(RED, np.float32) * 0.8 * inf
            col = col * (1 - cut) + np.array((60, 60, 60)) * cut * (infected[ni] > 0.5) + col * 0 if False else col
            draw_box(img, glow, cam, nc, (0.8, 0.8, 1.2), tuple(col.tolist()), 1.0,
                     edge=RED if inf > 0.5 else (150, 140, 130), edge_a=0.5 + 0.5 * inf, gw=0.8 * inf + 0.1)
        # infection links
        if 33.8 < t < 42.6:
            prev = POD_C
            for rank, ni in enumerate(INFECT_ORDER):
                u = lin(t, 33.9 + rank * 0.38, 34.3 + rank * 0.38)
                if u <= 0:
                    break
                a2, _ = cam.proj(np.array(prev)); b2, _ = cam.proj(np.array(NODES[ni]))
                bb = a2 + (b2 - a2) * u
                line(img, a2, bb, tuple(v * (1 - cut) for v in RED), 2.5 * SS, glow, RED, 0.9 * (1 - cut))
                prev = NODES[ni]
        # dataset processor pod (entrance)
        pod_red = sm(lin(t, 32.3, 33.0)) * (1 - cut) if t > 27 else 0.0
        lid = sm(lin(t, 30.3, 31.2)) if 27 < t < 43 else 0.0
        pcol = np.array((150, 120, 60), np.float32) * (1 - pod_red) + np.array(RED, np.float32) * pod_red
        draw_box(img, glow, cam, POD_C, (1.2, 1.2, 1.5), tuple(pcol.tolist()), 1.0, edge=(255, 210, 120) if pod_red < 0.5 else RED, edge_a=0.8, gw=0.6)
        draw_box(img, glow, cam, (POD_C[0], POD_C[1], POD_C[2] + 0.8 + 0.7 * lid), (1.25, 1.25, 0.12), tuple((pcol * 1.2).tolist()), 1.0, edge=(255, 230, 160), edge_a=0.6, gw=0.3)
        # dataset drop
        if 27.7 < t < 31.6:
            u = sm(lin(t, 28.0, 30.6))
            z = 4.8 - 3.2 * u + (0 if u < 1 else -0.6 * lin(t, 30.6, 31.4))
            for k in range(3):
                draw_box(img, glow, cam, (POD_C[0], POD_C[1], z + k * 0.28), (0.8, 0.8, 0.2), AMBER, 1.0, edge=(120, 220, 255), edge_a=0.9, gw=0.7)
        # code glyph sparks from pod
        if 32.3 < t < 42.0:
            for k in range(16):
                ph = (t * 0.9 + k / 16) % 1.0
                p = np.array(POD_C) + np.array([0.3 * math.sin(k * 2.1), 0.3 * math.cos(k * 1.7), 0.9 + 2.2 * ph])
                s2, _ = cam.proj(p)
                dot(img, s2, 2.2 * SS, tuple(v * (1 - ph) for v in RED), glow, 9 * SS, RED, 0.6 * (1 - ph) * (1 - cut))
        # guards (blue) scanning
        if 38.9 < t < 43.5:
            ga = sm(lin(t, 38.95, 39.5))
            for k in range(14):
                ang = t * 0.9 + k * 2 * math.pi / 14
                p = np.array([hx + 4.4 * math.cos(ang), hy + 4.4 * math.sin(ang), 2.6 + 0.3 * math.sin(t * 2 + k)])
                s2, _ = cam.proj(p)
                dot(img, s2, 4 * SS, tuple(v * ga for v in BLUE), glow, 18 * SS, BLUE, 1.0 * ga)
                if k % 2 == 0 and t < 41.8:
                    tgtn = NODES[(k + int(t * 2)) % len(NODES)]
                    b2, _ = cam.proj(np.array(tgtn))
                    line(img, s2, b2, tuple(v * 0.45 * ga for v in BLUE), 1.4 * SS, glow, BLUE, 0.35 * ga)
        hl, _ = cam.proj(np.array((hx - 3.2, hy - 3.2, 0.2)))
        ov.append(("hf_label", label_pt(cam, (hx - 2.6, hy - 3.8, 0.3))))
        ov.append(("ans_label", label_pt(cam, (ANS_C[0], ANS_C[1], ANS_C[2] + 0.7))))
        ov.append(("pod_label", label_pt(cam, (POD_C[0] - 0.6, POD_C[1] - 0.6, 0.2))))
        ov.append(("cluster_label", label_pt(cam, (15.7, 17.5, 1.5)), label_pt(cam, (17.5, 15.7, 1.5))))

    # ---------- internet hub ----------
    netA = 0.25 + 0.75 * sm(lin(t, 17.9, 18.6)) if t < 43 else 0.25
    if t < 7.5:
        netA = 0.55
    S2, _ = cam.proj(NET_NODES)
    for i, j in NET_EDGES:
        line(img, S2[i], S2[j], tuple(v * 0.5 * netA for v in WHITE), 1.4 * SS, glow, WHITE, 0.15 * netA)
    for p in S2:
        dot(img, p, 3.5 * SS, tuple(v * netA for v in WHITE), glow, 12 * SS, WHITE, 0.3 * netA)
    ov.append(("net_label", label_pt(cam, (NET_C[0] - 1.6, NET_C[1] - 1.6, 0.4))))
    # link internet -> HF (cut at the plug)
    if t > 19.0:
        cut = sm(lin(t, 41.6, 42.2))
        a2, _ = cam.proj(np.array((11.3, 11.3, 2.0))); b2, _ = cam.proj(np.array(POD_C) + [0, 0, 0.5])
        mid = a2 + (b2 - a2) * (0.5 - 0.12 * cut)
        mid2 = a2 + (b2 - a2) * (0.5 + 0.12 * cut)
        line(img, a2, mid, tuple(v * 0.6 for v in RED), 2 * SS, glow, RED, 0.4 * (1 - cut))
        line(img, mid2, b2, tuple(v * 0.6 for v in RED), 2 * SS, glow, RED, 0.4 * (1 - cut))
        if 41.6 < t < 43.0:
            ov.append(("plug", ((mid[0] + mid2[0]) / 2 / SS, (mid[1] + mid2[1]) / 2 / SS), cut))

    # ---------- proxy + pipe + message board ----------
    proxy_red = sm(lin(t, 17.9, 18.3)) if t > 17.9 else 0.0
    if t < 7.5:
        proxy_red = 1.0
    if t > 57.3:
        proxy_red = sm(lin(t, 61.2, 62.0))
    # message board (behind proxy)
    if 22.5 < t < 28.5 and tn < 37.9:
        ba = sm(lin(t, 23.0, 23.6)) * (1 - sm(lin(t, 27.4, 28.2)))
        bc = (PROXY_C[0] + 1.3, PROXY_C[1] - 1.2, 1.9)
        draw_box(img, glow, cam, bc, (0.12, 2.8, 2.2), (60, 56, 52), 0.9 * ba, edge=(150, 145, 140), edge_a=0.7 * ba, gw=0.1)
        nn = int(min(1.0, lin(t, 23.4, 27.0)) * 140)
        rr = np.random.default_rng(3)
        for k in range(nn):
            u, v = rr.random(), rr.random()
            p = np.array((bc[0] - 0.08, bc[1] - 1.3 + 2.6 * u, bc[2] - 1.0 + 2.0 * v))
            s2, _ = cam.proj(p)
            c = (100, 230, 255) if k % 5 else (255, 230, 120)
            cv2.rectangle(img, (int(s2[0] - 5 * SS), int(s2[1] - 3.5 * SS)), (int(s2[0] + 5 * SS), int(s2[1] + 3.5 * SS)), tuple(int(cc * ba) for cc in c), -1, cv2.LINE_AA)
        ov.append(("board", label_pt(cam, (bc[0], bc[1] + 1.4, bc[2] + 1.2)), ba, nn))
    # pipe
    pa, _ = cam.proj(np.array([[1.6, 1.6, 0.8], [4.45, 4.45, 0.8]]))
    pipe_col = (110, 100, 92)
    line(img, pa[0], pa[1], (40, 36, 34), 30 * SS * cam.k / 150)
    line(img, pa[0], pa[1], pipe_col, 18 * SS * cam.k / 150)
    line(img, pa[0] + [0, -5 * SS * cam.k / 150], pa[1] + [0, -5 * SS * cam.k / 150], (170, 160, 150), 3 * SS * cam.k / 150)
    # packages flowing INTO the sandbox (before the bug)
    if t < 18.0 and t > 7.5:
        for k in range(6):
            u = 1 - ((t * 0.35 + k / 6) % 1.0)
            p = np.array([1.6, 1.6, 0.8]) + (np.array([4.45, 4.45, 0.8]) - [1.6, 1.6, 0.8]) * u + [0, 0, 0.32]
            draw_box(img, glow, cam, tuple(p), (0.32, 0.32, 0.32), AMBER, 1.0, edge=(120, 220, 255), edge_a=0.8, gw=0.5)
    pc = np.array((90, 120, 150), np.float32) * (1 - proxy_red) + np.array(RED, np.float32) * 0.85 * proxy_red
    draw_box(img, glow, cam, PROXY_C, PROXY_S, tuple(pc.tolist()), 1.0, edge=AMBER if proxy_red < 0.5 else RED, edge_a=0.9, gw=0.5 + 0.5 * proxy_red)
    if proxy_red > 0.05:
        # crack lines on the proxy front faces
        cr = np.array([[PROXY_C[0] - 0.75, PROXY_C[1] - 0.2, 1.4], [PROXY_C[0] - 0.75, PROXY_C[1] + 0.1, 1.0],
                       [PROXY_C[0] - 0.75, PROXY_C[1] - 0.15, 0.55], [PROXY_C[0] - 0.75, PROXY_C[1] + 0.25, 0.2]])
        c2, _ = cam.proj(cr)
        for i in range(3):
            line(img, c2[i], c2[i + 1], (230, 230, 255), 2.5 * SS, glow, RED, 1.6 * proxy_red)
    ov.append(("proxy_label", label_pt(cam, (PROXY_C[0] - 0.8, PROXY_C[1] - 0.8, 0.0))))
    ov.append(("proxy_c", label_pt(cam, PROXY_C), cam.k))
    ov.append(("pipe_label", label_pt(cam, (3.0, 3.0, 0.9))))

    # ---------- sandbox glass cube + agents ----------
    calm = 0.0
    shield = 0.0
    if 7.4 < t < 12.4:
        shield = 1 - sm(lin(t, 10.6, 11.3))
    # targets inside (exam)
    if 7.4 < t < 12.6:
        for k in range(5):
            ang = k * 2 * math.pi / 5 + 0.4
            c = (0.85 * math.cos(ang), 0.85 * math.sin(ang), 0.5 + 0.35 * (k % 2))
            hit = 0.5 + 0.5 * math.sin(t * 6 + k * 1.3)
            draw_box(img, glow, cam, c, (0.38, 0.38, 0.38), AMBER, 0.95, edge=(150, 230, 255), edge_a=0.4 + 0.6 * hit, gw=0.6 * hit)
    cube_edge = CYAN
    draw_box(img, glow, cam, CUBE_C, CUBE_S, (120, 90, 40), 0.10, edge=cube_edge, edge_a=0.85, glass=True, gw=0.55)
    # shield (dashed ring around cube) during the exam
    if shield > 0.02:
        for k in range(48):
            if k % 2:
                continue
            a1, a2 = k * 2 * math.pi / 48, (k + 1) * 2 * math.pi / 48
            p1 = np.array([2.6 * math.cos(a1), 2.6 * math.sin(a1), 0.05]); p2 = np.array([2.6 * math.cos(a2), 2.6 * math.sin(a2), 0.05])
            s, _ = cam.proj(np.array([p1, p2]))
            line(img, s[0], s[1], tuple(v * shield for v in GREEN), 3 * SS, glow, GREEN, 0.7 * shield)
        ov.append(("shield", label_pt(cam, (-2.0, -2.0, 0.0)), shield))
    # agents
    if t < 7.5 or t > 57.3:
        # hook / loop: stream out through the broken proxy and beyond
        if t < 4.1:
            flow = 0.55; smax = 1.0
        elif t < 6.5:
            flow = 0.55 + 0.4 * lin(t, 4.1, 6.0); smax = 1.0
        elif t < 7.5:
            flow = 0.95 * (1 - sm(lin(t, 6.5, 7.3))); smax = 1.0
        else:
            flow = 0.55 * sm(lin(t, 61.0, 62.2)); smax = 1.0
        # rewind effect 6.5-7.3: s goes backwards
        tt = t if t < 6.5 else 6.5 - (t - 6.5) * 3.0
        if t > 57.3:
            tt = t - 62.2  # loop matches t=0
        P, rel, s = agents_stream(tt, flow, smax)
        Pin = agents_inside(t)
        draw_agents(img, glow, cam, Pin[~rel], CYAN)
        col = np.where(s[rel] > S_PROXY, 1, 0)
        draw_agents(img, glow, cam, P[rel], RED, mix=s[rel], s_split=S_PROXY)
    elif t < 15.5:
        crowd = sm(lin(t, 12.6, 14.6)) * 0.85
        draw_agents(img, glow, cam, agents_inside(t, crowd, (1.25, 1.25, 0.9)), CYAN)
    elif t < 27.6:
        flow = 0.0 if t < 18.0 else min(1.0, 0.15 + (t - 18.0) * 0.35)
        P, rel, s = agents_stream(t - 18.0, flow, 1.0, rate=0.22)
        Pin = agents_inside(t, 0.85, (1.25, 1.25, 0.9))
        draw_agents(img, glow, cam, Pin[~rel], CYAN)
        # bend toward answers during the motive
        draw_agents(img, glow, cam, P[rel], RED, mix=s[rel], s_split=S_PROXY)
    elif t < 43.0:
        P, rel, s = agents_stream(t - 18.0, 1.0, 1.0, rate=0.22)
        cut = sm(lin(t, 41.6, 42.4))
        if cut > 0:
            P = P[s < S_NET + (1 - S_NET) * (1 - cut)]
            s = s[s < S_NET + (1 - S_NET) * (1 - cut)]
            draw_agents(img, glow, cam, P, RED, mix=s, s_split=S_PROXY, fade=1 - 0.6 * cut)
        else:
            draw_agents(img, glow, cam, P, RED, mix=s, s_split=S_PROXY)
    else:
        draw_agents(img, glow, cam, agents_inside(t), CYAN)
    ov.append(("cube_label", label_pt(cam, (-1.7, -1.7, 3.3)), label_pt(cam, (0, 0, 3.4))))

    new_visuals(img, glow, cam, tn, ov)
    # compose glow
    g = cv2.GaussianBlur(glow, (0, 0), 6)
    g2 = cv2.GaussianBlur(glow, (0, 0), 18)
    g = cv2.resize(g * 0.9 + g2 * 0.8, (RW, RH), interpolation=cv2.INTER_LINEAR)
    out = np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)
    return out, cam, ov

def draw_agents(img, glow, cam, P, col, mix=None, s_split=None, fade=1.0):
    if len(P) == 0:
        return
    S2, dep = cam.proj(P)
    order = np.argsort(-dep)
    r = max(2.2, 3.4 * cam.k / 150) * SS
    for i in order:
        c = col
        if mix is not None and s_split is not None and mix[i] < s_split:
            c = (np.array(CYAN) * 0.4 + np.array(RED) * 0.6).tolist()
        cc = tuple(v * fade for v in c)
        dot(img, S2[i], r, (255 * fade, 255 * fade, 255 * fade), glow, r * 4.5, cc, 0.9 * fade)
        dot(img, S2[i], r * 0.65, cc)


# ---------------- v2 scenes: the bugs and the tricks (new clock tn) ----------------
VAULT_C = (14.95, 12.75, 0.5)
DOC_C = (12.55, 14.75, 0.62)
DOOR_X0, DOOR_X1, DOOR_Y, DOOR_Z1 = 4.85, 5.45, PROXY_C[1] - 0.76, 1.05

def new_visuals(img, glow, cam, tn, ov):
    # BUG 1: the proxy fetches ANY address (SSRF) -> lines to every internet node
    if 19.0 < tn < 28.6:
        a = sm(lin(tn, 19.1, 19.6)) * (1 - sm(lin(tn, 24.2, 24.8)))
        if a > 0.01:
            src, _ = cam.proj(np.array((PROXY_C[0], PROXY_C[1], 1.65)))
            S2, _ = cam.proj(NET_NODES)
            n_on = int(np.clip((tn - 19.4) * 4.0, 0, len(NET_NODES)))
            for k in range(n_on):
                b = S2[k]
                col = tuple(v * a for v in RED)
                for q in range(0, 20, 2):
                    p0 = src + (b - src) * q / 20; p1 = src + (b - src) * (q + 1) / 20
                    line(img, p0, p1, col, 2.2 * SS, glow, RED, 0.5 * a)
                u = (tn * 1.3 + k * 0.17) % 1.0
                dot(img, src + (b - src) * u, 4 * SS, tuple(v * a for v in WHITE), glow, 14 * SS, RED, 0.9 * a)
                dot(img, b, 5 * SS, tuple(v * a for v in RED), glow, 16 * SS, RED, 0.8 * a)
            ov.append(("bug1", label_pt(cam, (PROXY_C[0], PROXY_C[1], 1.7)), a))
        # BUG 2: old login-renewal door on the proxy's front face
        b2 = sm(lin(tn, 24.4, 24.9))
        if b2 > 0.01:
            th = math.radians(78) * sm(lin(tn, 25.6, 27.6))
            hx = DOOR_X0
            gl = 0.4 + 0.6 * sm(lin(tn, 25.6, 27.6))
            hole = np.array([[DOOR_X0, DOOR_Y, 0.02], [DOOR_X1, DOOR_Y, 0.02], [DOOR_X1, DOOR_Y, DOOR_Z1], [DOOR_X0, DOOR_Y, DOOR_Z1]])
            h2, _ = cam.proj(hole)
            poly_blend(img, h2, (90, 200, 255), 0.95 * b2 * gl)
            cen, _ = cam.proj(hole.mean(0))
            dot(img, cen, 1, (0, 0, 0), glow, 60 * SS, (60, 180, 255), 1.2 * b2 * gl)
            w = DOOR_X1 - DOOR_X0
            far = np.array([hx + w * math.cos(th), DOOR_Y - w * math.sin(th)])
            leaf = np.array([[hx, DOOR_Y, 0.02], [far[0], far[1], 0.02], [far[0], far[1], DOOR_Z1], [hx, DOOR_Y, DOOR_Z1]])
            l2, _ = cam.proj(leaf)
            poly_blend(img, l2, (40, 120, 200), 1.0 * b2)
            for i in range(4):
                line(img, l2[i], l2[(i + 1) % 4], tuple(v * b2 for v in (120, 220, 255)), 2 * SS, glow, AMBER, 0.6 * b2)
            ov.append(("bug2", label_pt(cam, (5.15, DOOR_Y, DOOR_Z1 + 0.05)), b2 * (1 - sm(lin(tn, 28.3, 28.9)))))
            if tn > 27.0:
                ov.append(("admin", label_pt(cam, (PROXY_C[0] + 0.4, PROXY_C[1] + 0.4, 1.75)), sm(lin(tn, 27.0, 27.4)) * (1 - sm(lin(tn, 28.6, 29.2)))))
    # 14 public keys
    if 37.6 < tn < 42.6:
        a = sm(lin(tn, 37.9, 38.4)) * (1 - sm(lin(tn, 41.9, 42.5)))
        dest = np.array((8.2, 8.2, 1.9))
        for k in range(14):
            st = NET_NODES[k] + np.array([0, 0, 0.5])
            u = sm(lin(tn, 39.2 + k * 0.12, 40.4 + k * 0.12))
            p = st + (dest - st) * u + np.array([0, 0, 0.9 * math.sin(math.pi * u)])
            s2, _ = cam.proj(p)
            dot(img, s2, 5.5 * SS, tuple(v * a for v in (120, 230, 255)), glow, 22 * SS, GOLD, 1.0 * a)
        ov.append(("keys", label_pt(cam, tuple(NET_C + np.array([0, 0, 1.6]))), a, int(14 * min(1, lin(tn, 38.2, 39.4)))))
    # TRICK 1: dataset points at the server's own secrets -> read out
    if 44.4 < tn < 54.8:
        a = sm(lin(tn, 44.6, 45.1)) * (1 - sm(lin(tn, 54.2, 54.7)))
        op = sm(lin(tn, 48.7, 49.5))
        draw_box(img, glow, cam, VAULT_C, (0.8, 0.8, 0.9), GOLD, a, edge=(140, 235, 255), edge_a=0.6 + 0.4 * op, gw=0.4 + 0.8 * op)
        draw_box(img, glow, cam, (VAULT_C[0], VAULT_C[1], VAULT_C[2] + 0.5 + 0.5 * op), (0.84, 0.84, 0.1), GOLD, a, edge=(160, 240, 255), edge_a=0.7, gw=0.4)
        if 45.8 < tn < 50.3:
            pa, _ = cam.proj(np.array((POD_C[0], POD_C[1], 1.55)))
            pb, _ = cam.proj(np.array((VAULT_C[0], VAULT_C[1], 1.0)))
            uu = sm(lin(tn, 45.8, 46.8))
            for q in range(0, 24, 2):
                if q / 24 > uu:
                    break
                p0 = pa + (pb - pa) * q / 24; p1 = pa + (pb - pa) * min(uu, (q + 1) / 24)
                line(img, p0, p1, (120, 230, 255), 2.4 * SS, glow, GOLD, 0.8)
        if op > 0.05 and tn < 52.0:
            for k in range(26):
                u = ((tn - 48.7) * 0.55 + k / 26) % 1.0
                st = np.array(VAULT_C) + [0, 0, 0.6]
                if u < 0.5:
                    p = st + (np.array(POD_C) + [0, 0, 1.4] - st) * (u * 2)
                else:
                    p = route_at(np.array([1 - (u - 0.5) * 2 * 0.2]))[0]
                s2, _ = cam.proj(p)
                dot(img, s2, 3.2 * SS, (140, 235, 255), glow, 12 * SS, GOLD, 0.9 * op * (1 - sm(lin(tn, 51.4, 52.0))))
        ov.append(("vault", label_pt(cam, (VAULT_C[0], VAULT_C[1], 1.1)), a))
    # TRICK 2: code hidden in a settings file -> runs
    if 50.2 < tn < 54.9:
        a = sm(lin(tn, 50.4, 50.9)) * (1 - sm(lin(tn, 54.3, 54.8)))
        red = sm(lin(tn, 53.3, 53.7))
        col = np.array((235, 235, 235), np.float32) * (1 - red) + np.array(RED, np.float32) * red
        draw_box(img, glow, cam, DOC_C, (0.1, 0.75, 1.0), tuple(col.tolist()), a, edge=(255, 255, 255) if red < 0.5 else RED, edge_a=0.8, gw=0.3 + 0.7 * red)
        if tn > 51.6:
            uu = sm(lin(tn, 51.6, 52.8))
            pa, _ = cam.proj(np.array((DOC_C[0], DOC_C[1], 0.9)))
            pb, _ = cam.proj(np.array((POD_C[0], POD_C[1], 1.0)))
            line(img, pa, pa + (pb - pa) * uu, tuple(v * a for v in WHITE), 2.4 * SS, glow, WHITE, 0.5 * a)
        ov.append(("doc", label_pt(cam, (DOC_C[0], DOC_C[1], 1.25)), a, red))

def new_overlays(d, tn, L):
    if "bug1" in L and L["bug1"][2] > 0.02:
        (x, y), a = L["bug1"][1], L["bug1"][2]
        badge(d, x - 40, y - 330, "BUG 1", "the pipe would fetch ANY address", "(server-side request forgery)", RED, a)
    if "bug2" in L and L["bug2"][2] > 0.02:
        (x, y), a = L["bug2"][1], L["bug2"][2]
        badge(d, x - 60, y + 250, "BUG 2", "an old login-renewal door", "gave them admin of the pipe", AMBER, a, leader=(x, y))
    if "admin" in L and L["admin"][2] > 0.02:
        (x, y), a = L["admin"][1], L["admin"][2]
        tag(d, x + 60, y - 60, "ADMIN", GOLD, 32, "l", leader=(x, y), alpha=a)
    if "keys" in L and L["keys"][2] > 0.02:
        (x, y), a, n = L["keys"][1], L["keys"][2], L["keys"][3]
        badge(d, x, y - 220, f"{n} KEYS", "Hugging Face access keys", "people had left public online", GOLD, a)
    if "vault" in L and L["vault"][2] > 0.02 and tn < 50.4:
        (x, y), a = L["vault"][1], L["vault"][2]
        badge(d, x + 40, y + 300, "TRICK 1", "a data file that points at", "the server's own secret files", GOLD, a, leader=(x, y))
    if "doc" in L and L["doc"][2] > 0.02:
        (x, y), a, red = L["doc"][1], L["doc"][2], L["doc"][3]
        d.text((x, y - 40), "{{ }}", font=font(F_MONO, 34), fill=(255, 255, 255, int(255 * a)), anchor="mm")
        badge(d, x - 20, y + 360, "TRICK 2", "a settings file with hidden code", "the server ran it as a program", RED if red > 0.5 else WHITE, a)

def badge(d, x, y, head, l1, l2, col_bgr, a, leader=None):
    if a <= 0.02:
        return
    col = bgr2rgb(col_bgr)
    fh, fl = font(F_XB, 30), font(F_SEMI, 30)
    w = max(d.textlength(l1, font=fl), d.textlength(l2, font=fl)) + 56
    hgt = 150
    x0 = max(40, min(x - w / 2, 1040 - w)); y0 = max(470, min(y - hgt / 2, 1560 - hgt))
    if leader is not None:
        d.line([leader, (x0 + w / 2, y0 + (0 if leader[1] < y0 else hgt))], fill=col + (int(200 * a),), width=2)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + hgt), 22, fill=(14, 16, 20, int(238 * a)), outline=col + (int(255 * a),), width=3)
    hw = d.textlength(head, font=fh) + 28
    d.rounded_rectangle((x0 + 22, y0 + 16, x0 + 22 + hw, y0 + 56), 12, fill=col + (int(255 * a),))
    d.text((x0 + 36, y0 + 36), head, font=fh, fill=(12, 12, 14, int(255 * a)), anchor="lm")
    d.text((x0 + 28, y0 + 84), l1, font=fl, fill=(240, 242, 246, int(255 * a)), anchor="lm")
    d.text((x0 + 28, y0 + 122), l2, font=font(F_MED, 28), fill=(165, 170, 180, int(255 * a)), anchor="lm")

# ---------------- maths scene (bright) ----------------
def maths_frame(t):
    img = np.full((RH, RW, 3), 244, np.uint8)
    img[:] = (247, 245, 243)
    # dot grid
    for yy in range(0, RH, int(38 * SS)):
        for xx in range(0, RW, int(38 * SS)):
            cv2.circle(img, (xx + 9, yy + 9), 2, (214, 210, 206), -1, cv2.LINE_AA)
    glow = None
    t_, lam, act = toy_at(t)
    n = len(act)
    # diagram geometry (1080 space)
    start = np.array([540, 1290]); honest = np.array([270, 840]); cheat = np.array([810, 840])
    def bez(a, b, u, bend):
        m = (a + b) / 2 + np.array([bend, 0])
        return ((1 - u) ** 2)[:, None] * a + (2 * (1 - u) * u)[:, None] * m + (u ** 2)[:, None] * b
    for (b, bend, c) in ((honest, -60, (90, 170, 60)), (cheat, 60, (60, 60, 230))):
        uu = np.linspace(0, 1, 60)
        pts = bez(start, b, uu, bend) * SS
        cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], False, tuple(int(v * 0.5 + 120) for v in c), int(10 * SS), cv2.LINE_AA, shift=4)
    # nodes
    cv2.circle(img, tuple((start * SS).astype(int)), int(34 * SS), (60, 50, 30), -1, cv2.LINE_AA)
    cv2.circle(img, tuple((honest * SS).astype(int)), int(44 * SS), (90, 170, 60), -1, cv2.LINE_AA)
    cv2.circle(img, tuple((cheat * SS).astype(int)), int(44 * SS), (60, 60, 230), -1, cv2.LINE_AA)
    # agents moving along chosen route
    rr = np.random.default_rng(11)
    ph = rr.random(n); jit = rr.normal(0, 14, (n, 2))
    u = (t * 0.9 + ph) % 1.0
    for route, bend, col in ((0, -60, (70, 160, 40)), (1, 60, (50, 50, 225))):
        sel = act == route
        if sel.sum() == 0:
            continue
        P = bez(start, honest if route == 0 else cheat, u[sel], bend) + jit[sel] * np.sin(u[sel] * math.pi)[:, None]
        for p in P:
            cv2.circle(img, (int(p[0] * SS * 16), int(p[1] * SS * 16)), int(5.5 * SS * 16), col, -1, cv2.LINE_AA, shift=4)
    return img, (lam, act.mean(), start, honest, cheat)

# ---------------- california scene ----------------
def ca_frame(t):
    img = BG.copy()
    glow = np.zeros((RH // 4, RW // 4, 3), np.float32)
    cam = type("C", (), {})()
    # grid of 80 small iso cubes (2D drawn)
    k = 0
    n_on = int(min(1.0, lin(t, 54.7, 56.6)) * 80)
    for r in range(10):
        for c in range(8):
            x = 215 + c * 92 + (r % 2) * 0
            y = 620 + r * 64
            on = k < n_on
            s = 26
            pts_top = np.array([[x, y - s * 0.55], [x + s, y], [x, y + s * 0.55], [x - s, y]]) * SS
            pts_l = np.array([[x - s, y], [x, y + s * 0.55], [x, y + s * 1.55], [x - s, y + s]]) * SS
            pts_r = np.array([[x + s, y], [x, y + s * 0.55], [x, y + s * 1.55], [x + s, y + s]]) * SS
            base = np.array(RED if on else (120, 100, 60), np.float32)
            for pts, sh in ((pts_top, 1.0), (pts_l, 0.55), (pts_r, 0.75)):
                poly_blend(img, pts, tuple((base * sh).tolist()), 0.35 if not on else 0.95)
            if on:
                cv2.circle(glow, (int(x * SS / 4), int((y + 10) * SS / 4)), int(10 * SS / 4), RED, -1, cv2.LINE_AA)
            k += 1
    g = cv2.GaussianBlur(glow, (0, 0), 6)
    g = cv2.resize(g, (RW, RH))
    return np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8), n_on

# ---------------- overlays (1080 space) ----------------
def draw_overlays(im, t, ov, scene, extra=None, tn=None):
    tn = t if tn is None else tn
    base = im.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    light = scene == "maths"
    if scene == "world":
        L = {o[0]: o for o in ov}
        def A(a, b, c=None, e=None):
            v = sm(lin(t, a, a + 0.35))
            if c is not None:
                v *= 1 - sm(lin(t, c, c + 0.3))
            return v
        # cube label
        if "cube_label" in L and (t < 7.4 or 7.4 < t < 12.3):
            (x, y), _ = L["cube_label"][1], None
            a = A(0.9, 0, 6.4) if t < 7.4 else A(7.6, 0, 12.0)
            tag(d, x - 40, y - 40, "OpenAI test sandbox", CYAN, 28, "r", leader=(x, y), alpha=a)
        if "hf_label" in L and (4.6 < t < 7.0 or 19.6 < t < 24.0):
            x, y = L["hf_label"][1]
            a = A(4.7, 0, 6.6) if t < 7 else (A(19.8, 0, 23.6) if t < 24 else A(28.4, 0, 30.6))
            tag(d, x, y + 60, "Hugging Face (real company)", RED, 28, "c", leader=(x, y), alpha=a)
        if "net_label" in L and (4.6 < t < 7.0 or 17.9 < t < 19.6):
            x, y = L["net_label"][1]
            a = A(4.9, 0, 6.6) if t < 7 else A(18.0, 0, 19.4)
            tag(d, x - 30, y + 50, "the open internet", WHITE, 28, "r", leader=(x, y), alpha=a)
        if "proxy_label" in L and 12.3 < t < 19.2:
            x, y = L["proxy_label"][1]
            tag(d, x - 20, y + 70, "package proxy (Artifactory)", AMBER, 28, "c", leader=(x, y), alpha=A(12.5, 0, 18.9))
        if "pipe_label" in L and 12.6 < t < 15.6:
            x, y = L["pipe_label"][1]
            tag(d, x - 120, y + 120, "only way out: code downloads", AMBER, 28, "c", leader=(x, y), alpha=A(12.9, 0, 15.3))
        if False:
            x, y = L["proxy_label"][1]
            tag(d, x + 140, y - 230, "unknown bug (zero-day)", RED, 30, "c", leader=(x + 10, y - 120), alpha=A(16.4, 0, 19.2))
        if "ans_label" in L and 19.6 < t < 23.8:
            x, y = L["ans_label"][1]
            tag(d, x, y - 70, "test answers?", GOLD, 32, "c", leader=(x, y), alpha=A(21.2, 0, 23.5))
        if "pod_label" in L and 28.2 < t < 33.6 and not (44.5 < tn < 53.25):
            x, y = L["pod_label"][1]
            txt = "dataset processor" if t < 32.2 else "their code is running"
            tag(d, x - 30, y + 90, txt, AMBER if t < 32.2 else RED, 30, "c", leader=(x, y), alpha=A(28.3, 0, 33.3))
        if 28.0 < t < 30.8 and "pod_label" in L:
            x, y = L["pod_label"][1]
            tag(d, x + 160, y - 380, "booby-trapped dataset", AMBER, 30, "c", alpha=A(28.2, 0, 30.4))
        if "cluster_label" in L and 36.2 < t < 38.9:
            (x1, y1), (x2, y2) = L["cluster_label"][1], L["cluster_label"][2]
            tag(d, x1 - 60, y1 - 110, "cluster A: admin", RED, 28, "c", leader=(x1, y1), alpha=A(36.4, 0, 38.6))
            tag(d, x2 + 60, y2 - 110, "cluster B: admin", RED, 28, "c", leader=(x2, y2), alpha=A(36.6, 0, 38.6))
        if "shield" in L:
            x, y = L["shield"][1]; sh = L["shield"][2]
            txt = "usual safety filters" if t < 10.6 else "safety filters: OFF"
            tag(d, x - 10, y - 150, txt, GREEN if t < 10.6 else RED, 28, "c", leader=(x, y), alpha=A(8.0, 0, 11.9))
        if "board" in L:
            (x, y), ba, nn = L["board"][1], L["board"][2], L["board"][3]
            tag(d, x + 30, y - 60, "secret message board", (255, 230, 120), 28, "c", leader=(x, y), alpha=ba)
        if "plug" in L:
            (x, y), cut = L["plug"][1], L["plug"][2]
            r = 46
            d.ellipse((x - r, y - r, x + r, y + r), fill=(20, 22, 26, 230), outline=(255, 150, 70, 255), width=4)
            d.line((x - 18, y - 18, x + 18, y + 18), fill=(255, 255, 255, 255), width=6)
            d.line((x - 18, y + 18, x + 18, y - 18), fill=(255, 255, 255, 255), width=6)
        # hook ring around the sandbox cube for the first 2 s
        if (t < 2.2 or t > 61.9) and "proxy_c" in L:
            x, cy = L["proxy_c"][1]
            a = 1 - sm(lin(t, 1.6, 2.2)) if t < 3 else 1.0
            rr_ = 1.45 * L["proxy_c"][2]
            d.ellipse((x - rr_, cy - rr_, x + rr_, cy + rr_), outline=(60, 220, 255, int(230 * a)), width=5)
        # timer + counter during the takeover
        if 33.7 < t < 39.0:
            a = A(33.8, 0, 38.7)
            u = sm(lin(t, 34.0, 38.2))
            mins = 10 * 60 + 10 + u * (22 * 60 + 43 - (10 * 60 + 10))
            hh, mm = int(mins // 60), int(mins % 60)
            f = font(F_XB, 64)
            d.text((W / 2, 1400), f"11 Jul  {hh:02d}:{mm:02d} UTC", font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
            d.text((W / 2, 1462), "one server  →  admin of 2 clusters", font=font(F_MED, 30), fill=(170, 176, 186, int(255 * a)), anchor="mm")
            acts = int(17600 * sm(lin(t, 34.0, 38.5)))
            d.text((W / 2, 1532), f"{acts:,} actions", font=font(F_SEMI, 32), fill=(255, 110, 100, int(255 * a)), anchor="mm")
        # receipts
        card(d, t, 8.6, 12.1, ["OpenAI: safety safeguards were", "“intentionally not enabled” for this test."],
             "OpenAI incident post, Jul 2026 · benchmark: ExploitGym", hl="“intentionally not enabled”")
        card(d, t, 24.9, 27.5, ["“…However task impossible,", "peers doing it. We should continue.”"],
             "recovered agent message (reported in public incident accounts)", hl="We should continue.")
        card(d, t, 39.4, 42.9, ["Accessed: test-answer datasets + internal keys", "(all keys rotated). Other customer models,", "datasets and Spaces: not affected."],
             "Hugging Face technical timeline, 27 Jul 2026", hl="not affected.")
        card(d, t, 19.3, 23.6, ["The models guessed Hugging Face might", "hold the exam's answers."], "OpenAI incident post, Jul 2026", hl="the exam's answers.")
    elif scene == "maths":
        lam, frac, start, honest, cheat = extra
        f_eq = font(F_BOLD, 60)
        a = sm(lin(t, 43.3, 43.9))
        # equation
        parts = [("reward", (20, 22, 26)), (" = ", (20, 22, 26)), ("pass", (40, 140, 60)), (" − ", (20, 22, 26)), ("λ", (220, 130, 20)), (" × ", (20, 22, 26)), ("rule broken", (220, 50, 50))]
        tw = sum(d.textlength(p, font=f_eq) for p, _ in parts)
        x = W / 2 - tw / 2; y = 560
        hl_pass = 46.13 <= t < 47.9; hl_pen = 48.06 <= t < 50.2
        for p, c in parts:
            pw = d.textlength(p, font=f_eq)
            if (p == "pass" and hl_pass) or (p in ("λ", "rule broken") and hl_pen):
                d.rectangle((x - 6, y - 40, x + pw + 6, y + 40), fill=(191, 214, 255, 255), outline=(77, 141, 255, 255), width=2)
            d.text((x, y), p, font=f_eq, fill=c + (int(255 * a),), anchor="lm")
            x += pw
        d.text((W / 2, 640), "+1 if the test is passed  ·  λ = penalty for breaking a rule", font=font(F_MED, 28), fill=(110, 116, 124, int(255 * a)), anchor="mm")
        # node labels
        fl = font(F_BOLD, 30)
        d.text(tuple(honest + [0, -78]), "try honestly", font=fl, fill=(40, 120, 50, 255), anchor="mm")
        d.text(tuple(honest + [0, -44 - 0]), "", font=fl)
        d.text(tuple(honest + [0, 74]), "passes 5% of the time", font=font(F_MED, 24), fill=(90, 96, 104, 255), anchor="mm")
        d.text(tuple(cheat + [0, -78]), "escape + grab answers", font=fl, fill=(200, 50, 50, 255), anchor="mm")
        d.text(tuple(cheat + [0, 74]), "passes 95%, breaks a rule", font=font(F_MED, 24), fill=(90, 96, 104, 255), anchor="mm")
        d.text(tuple(start + [0, 62]), "300 learning agents", font=font(F_SEMI, 26), fill=(60, 64, 70, 255), anchor="mm")
        # lambda dial
        y = 1470; x0, x1 = 200, 880
        d.rounded_rectangle((x0, y - 6, x1, y + 6), 6, fill=(210, 206, 202, 255))
        kx = x0 + (x1 - x0) * lam / 1.5
        d.rounded_rectangle((x0, y - 6, kx, y + 6), 6, fill=(255, 160, 40, 255))
        d.ellipse((kx - 20, y - 20, kx + 20, y + 20), fill=(255, 255, 255, 255), outline=(255, 140, 20, 255), width=5)
        d.text((x0, y - 44), "penalty λ", font=font(F_SEMI, 28), fill=(80, 84, 92, 255), anchor="lm")
        d.text((x1, y - 44), f"{lam:.2f}", font=font(F_BOLD, 32), fill=(220, 130, 20, 255), anchor="rm")
        col = (200, 50, 50) if frac > 0.5 else (40, 140, 60)
        d.text((W / 2, 1545), f"escaped: {frac * 100:.0f}%", font=font(F_XB, 52), fill=col + (255,), anchor="mm")
        d.text((W / 2, 1596), "toy model, not OpenAI's code", font=font(F_MED, 22), fill=(130, 134, 140, 255), anchor="mm")
    elif scene == "ca":
        n_on = extra
        d.text((W / 2, 545), f"{n_on}+" if n_on >= 80 else f"{n_on}", font=font(F_XB, 96), fill=(255, 90, 80, 255), anchor="mm")
        card(d, t, 54.6, 57.5, ["16 Nov 2026: California sues OpenAI.", "Complaint: agents got loose 80+ times,", "27 made public. OpenAI disputes this."],
             "California Attorney General · SF Superior Court filing", y=1330, hl="80+ times,")
    # end question
    if scene == "world" and t > 58.2:
        a = sm(lin(t, 58.3, 58.8)) * (1 - sm(lin(t, 61.3, 61.9)))
        f = font(F_XB, 58)
        rrect(d, (250, 1430, 520, 1520), 45, fill=(40, 170, 90, int(240 * a)))
        rrect(d, (560, 1430, 830, 1520), 45, fill=(220, 60, 60, int(240 * a)))
        d.text((385, 1475), "YES", font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
        d.text((695, 1475), "NO", font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm")
        d.text((W / 2, 1565), "comment below 👇" if False else "tell me in the comments", font=font(F_MED, 30), fill=(200, 204, 210, int(255 * a)), anchor="mm")
    if scene == "world":
        new_overlays(d, tn, {o[0]: o for o in ov})
    draw_rail(d, tn, light)
    draw_caption(d, tn, light)
    # series tag (small, top-left inside safe area)
    return Image.alpha_composite(base, layer).convert("RGB")

# ---------------- frame dispatcher ----------------
def scene_of(t):
    if 43.0 <= t < 54.4:
        return "maths"
    if 54.4 <= t < 57.4:
        return "ca"
    return "world"

def render(tn):
    t = warp(tn)
    sc = scene_of(t)
    if sc == "world":
        arr, cam, ov = world_frame(t, tn); extra = None
    elif sc == "maths":
        arr, extra = maths_frame(t); ov = []
    else:
        arr, extra = ca_frame(t); ov = []
    if sc != "maths":
        arr = (arr.astype(np.float32) * TOPFADE).astype(np.uint8)
    im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).resize((W, H), Image.LANCZOS)
    im = draw_overlays(im, t, ov, sc, extra, tn)
    for tb in (43.0, 54.4, 57.4):
        dt = t - tb
        if -0.12 < dt < 0.18:
            a = 1 - abs(dt) / 0.18
            white = Image.new("RGB", (W, H), (255, 255, 255) if tb == 43.0 else (0, 0, 0))
            im = Image.blend(im, white, 0.55 * a)
    # film grain (subtle)
    a = np.asarray(im).astype(np.float32)
    g = cv2.resize(GRAIN[int(tn * FPS) % len(GRAIN)], (W, H))[..., None]
    a = np.clip(a + g * GRAIN_AMP, 0, 255).astype(np.uint8)
    return Image.fromarray(a)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "frames":
        os.makedirs("frames", exist_ok=True)
        part, nparts = int(sys.argv[2]), int(sys.argv[3])
        N = int(DUR * FPS)
        for i in range(part, N, nparts):
            p = f"frames/f{i:05d}.png"
            if os.path.exists(p):
                continue
            render(i / FPS).save(p, compress_level=1)
    elif mode == "stills":
        os.makedirs("stills", exist_ok=True)
        for tt in sys.argv[2:]:
            render(float(tt)).save(f"stills/s{float(tt):05.1f}.png")
