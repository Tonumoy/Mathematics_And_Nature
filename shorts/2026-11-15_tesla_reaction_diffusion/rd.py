# SOURCE CODE #10 — Tesla drawn by reaction-diffusion (Coded Cosmos). Own render: shaded relief fields, 3D dish, bloom.
import numpy as np, cv2, math, sys, os
from PIL import Image, ImageDraw
import lib as K
from lib import W, H, CYAN, AMBER, RED, GREEN, YELLOW, WHITE, ease, ramp, lerp
import tl as X
from tl import TL, s, e, T
FPS = 30
AMB = np.array(AMBER, np.float32); CYA = np.array(CYAN, np.float32)
NAVY = np.array([8, 12, 26], np.float32); IVORY = np.array([238, 230, 212], np.float32); INK = np.array([18, 26, 46], np.float32)

# ---------- data ----------
SN = np.load('snaps_v2.npy', mmap_mode='r')       # (300, 1920, 1080) uint8 v*3
NSN = len(SN) - 1
LMAP = np.load('L_v2.npy')
EM = np.load('e_mitosis.npz'); MV, MU = EM['V'], EM['U']
ED = np.load('e_dial.npz'); DV, DU = ED['V'], ED['U']
EX = {n: np.load(f'e_{n}.npz') for n in ('fish', 'palate', 'limb', 'membrane')}
def stack_at(S, x):
    x = min(max(x, 0), len(S) - 1.001); i = int(x); w = x - i
    return S[i].astype(np.float32) * (1 - w) + S[i + 1].astype(np.float32) * w

# ---------- shading ----------
LV = np.array([-0.45, -0.6, 0.66]); LV /= np.linalg.norm(LV); HV = LV + np.array([0, 0, 1.0]); HV /= np.linalg.norm(HV)
def relief(h, strength=6.0):
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) * strength; gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) * strength
    nz = 1 / np.sqrt(gx * gx + gy * gy + 1); nx, ny = -gx * nz, -gy * nz
    diff = np.clip(nx * LV[0] + ny * LV[1] + nz * LV[2], 0, 1)
    spec = np.clip(nx * HV[0] + ny * HV[1] + nz * HV[2], 0, 1) ** 36
    return diff, spec
def bloom(img, a=0.5, b=0.45):
    sm = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    return img + cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3) * a + cv2.GaussianBlur(sm, (0, 0), 14) * b, (W, H))
_BG = None
def bg():
    global _BG
    if _BG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = NAVY + (np.array([3, 4, 9], np.float32) - NAVY) * (yy / H)[..., None]
        g = g + np.exp(-(((xx - 560) / 560) ** 2 + ((yy - 760) / 700) ** 2))[..., None] * np.array([18, 16, 40], np.float32)
        _BG = g.astype(np.float32)
    return _BG

# ---------- portrait ----------
def portrait_field(x):
    """x: snapshot float index 0..399 -> v in 0..1 at 1080x1920"""
    v = stack_at(SN, x) / 255.0
    return v if v.shape == (H, W) else cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
def portrait_img(x, light=1.0, sweep=None):
    v = portrait_field(x)
    h = cv2.GaussianBlur(1 - v, (0, 0), 1.0)
    diff, spec = relief(h, 7.0)
    a = np.clip(v * 1.6, 0, 1)[..., None]
    base = IVORY * (1 - a) + INK * a
    img = base * (0.55 + 0.55 * diff[..., None]) + 120 * spec[..., None] * (1 - a)
    if sweep is not None:   # moving light band
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        band = np.exp(-((xx * 0.55 + yy * 0.84 - sweep) / 160) ** 2)
        img = img + band[..., None] * np.array([90, 70, 30], np.float32) * (1 - a)
    return img * light
def kmap_img(alpha):
    """halftone dots: dot size = how long sparks live (dark photo -> big amber dots)"""
    img = bg().copy(); lay = np.zeros((H * 2, W * 2), np.uint8)
    Ls = cv2.resize(LMAP, (54, 96), interpolation=cv2.INTER_AREA)
    for j in range(96):
        for i in range(54):
            r = (1 - Ls[j, i]) * 13 * 2
            if r > 1.5: cv2.circle(lay, (int((i + 0.5) * 20 * 2), int((j + 0.5) * 20 * 2)), int(r), 255, -1, cv2.LINE_AA)
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    top = np.clip((yy - 380) / 120, 0.12, 1)
    return img + lay[..., None] * AMB * 0.6 * alpha * top

# ---------- dish ----------
FOODTEX = cv2.GaussianBlur(np.random.default_rng(5).random((1000, 1000)).astype(np.float32), (0, 0), 18); FOODTEX = (FOODTEX - FOODTEX.min()) / (FOODTEX.max() - FOODTEX.min())
def field_tex(V, U, size=1000, food=True):
    """square texture: amber glossy sparks on cyan food haze"""
    v = cv2.resize(V / 255.0, (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    u = cv2.resize(U / 255.0, (size, size), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    diff, spec = relief(cv2.GaussianBlur(v, (0, 0), 1.2), 9.0)
    img = np.zeros((size, size, 3), np.float32) + np.array([10, 16, 30], np.float32)
    if food:
        dep = cv2.GaussianBlur(1 - u, (0, 0), size / 60); fd = np.clip(1 - dep * 9, 0, 1) ** 1.5
        img += (fd * (0.75 + 0.25 * FOODTEX[:size, :size]))[..., None] * np.array([30, 120, 160], np.float32) * 0.5
    a = np.clip(v * 1.8, 0, 1)[..., None]
    sp = AMB * (0.35 + 0.8 * diff[..., None]) + 255 * spec[..., None] * 0.9
    return img * (1 - a) + sp * a
def dish(tex, tilt, cx=540, cy=820, R=430, rot=0.0):
    """perspective-tilted circular dish with glass rim; tilt 0 = top view, 1 = strong tilt"""
    sz = tex.shape[0]
    if rot: tex = cv2.warpAffine(tex, cv2.getRotationMatrix2D((sz / 2, sz / 2), rot, 1.0), (sz, sz))
    sy = 1 - 0.42 * tilt; topw = 1 - 0.18 * tilt
    src = np.float32([[0, 0], [sz, 0], [sz, sz], [0, sz]])
    dst = np.float32([[cx - R * topw, cy - R * sy], [cx + R * topw, cy - R * sy], [cx + R, cy + R * sy], [cx - R, cy + R * sy]])
    Mx = cv2.getPerspectiveTransform(src, dst)
    yy, xx = np.mgrid[0:sz, 0:sz]; circ = ((((xx - sz / 2) ** 2 + (yy - sz / 2) ** 2) < (sz * 0.435) ** 2)).astype(np.float32)
    circ = cv2.GaussianBlur(circ, (0, 0), 2)
    wt = cv2.warpPerspective(tex * circ[..., None], Mx, (W, H)); wm = cv2.warpPerspective(circ, Mx, (W, H))
    img = bg().copy()
    # shadow under dish
    sh = np.zeros((H, W), np.float32); cv2.ellipse(sh, (cx, int(cy + R * sy * 0.15 + 30)), (int(R * 1.05), int(R * sy * 1.02)), 0, 0, 360, 1, -1)
    img *= (1 - 0.5 * cv2.GaussianBlur(sh, (0, 0), 30))[..., None]
    img = img * (1 - wm[..., None]) + wt
    # glass rim: two ellipses (top + bottom lip) with highlight
    rim = np.zeros((H, W), np.float32)
    ex, ey = int(R * (1 + topw) / 2 * 0.98), int(R * sy * 0.98)
    for dy, w_, a in ((0, 5, 1.0), (int(26 * (0.3 + tilt)), 3, 0.5)):
        cv2.ellipse(rim, (cx, cy + dy), (ex + 8, ey + 8), 0, 0, 360, a, w_, cv2.LINE_AA)
    hl = np.zeros((H, W), np.float32); cv2.ellipse(hl, (cx, cy), (ex + 8, ey + 8), 0, 200, 250, 1, 7, cv2.LINE_AA)
    img += cv2.GaussianBlur(rim, (0, 0), 1.2)[..., None] * np.array([150, 190, 215], np.float32) * 0.8
    img += cv2.GaussianBlur(hl, (0, 0), 3)[..., None] * 255
    return img

def masked_tex(name, x, col=AMB, size=900):
    D = EX[name]; V = stack_at(D['V'], x)
    U = stack_at(D['U'], x)
    t = field_tex(V, U, size, food=False)
    if 'M' in D:
        m = cv2.resize(D['M'], (size, size), interpolation=cv2.INTER_LINEAR); m = cv2.GaussianBlur(m, (0, 0), 3)
        body = np.array([40, 70, 110], np.float32)
        t = t * m[..., None] + (1 - m[..., None]) * np.array([10, 16, 30], np.float32)
        # body rim light
        edge = cv2.Canny((m * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255
        t += cv2.GaussianBlur(edge, (0, 0), 2.5)[..., None] * CYA * 1.6
    return t

def paste_center(img, tex, cx, cy, scale=1.0, alpha=1.0):
    t = cv2.resize(tex, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = t.shape[:2]; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return img
    img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - alpha) + t[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * alpha
    return img

# ---------- timing helpers ----------
GROW0 = s('L13') + 1.6; GROW1 = e('L14') + 0.3; REVEAL = GROW1
def grow_x(t):
    a = ramp(t, GROW0, GROW1); return NSN * (a ** 2.6)
def mito_x(t):
    # L03: slow single spark + first splits; L04-L05: medium; L06: fast to full
    a3, a6, a7 = s('L03'), s('L06'), s('L07')
    if t < a3: return 0
    a4 = s('L04')
    if t < a4: return lerp(0, 95, ease(ramp(t, a3, a4)))
    if t < a6: return lerp(95, 220, ease(ramp(t, a4, a6)))
    return lerp(220, 599, ease(ramp(t, a6, a7 - 0.2)))

# ---------- overlays ----------
def overlay(t):
    lay = K.new_layer(); d = ImageDraw.Draw(lay)
    for k, (txt, hl) in X.CAP.items():
        a, b = TL[k]
        if k == 'L01': continue
        al = ramp(t, a, a + 0.1) * (1 - ramp(t, b + 0.02, b + 0.14))
        if k == 'L15' and t > a: al = ramp(t, a, a + 0.15)
        if al > 0: K.caption(d, txt, y=1440, size=56, hl=hl, alpha=int(255 * al))
    # hook text (bottom, over the coat, never on the face)
    if t < s('L02') + 0.1:
        al = 1 - ramp(t, s('L02') - 0.2, s('L02') + 0.1)
        K.draw_text(d, (485, 1360), "NOBODY", 120, fill=WHITE + (int(255 * al),), stroke=6)
        K.draw_text(d, (485, 1490), "DREW HIM.", 120, fill=YELLOW + (int(255 * al),), stroke=6)
    # explainer labels
    if s('L03') + 0.6 <= t < s('L05'):
        al = ramp(t, s('L03') + 0.6, s('L03') + 0.9) * (1 - ramp(t, s('L05') - 0.3, s('L05')))
        K.pill(d, (485, 300), "SPARK: makes copies of itself", col=AMBER, size=34, alpha=int(255 * al))
    if s('L04') <= t < s('L06'):
        al = ramp(t, s('L04'), s('L04') + 0.3) * (1 - ramp(t, s('L06') - 0.3, s('L06')))
        K.pill(d, (485, 370), "FOOD: spreads 2× faster", col=CYAN, size=34, alpha=int(255 * al))
    if s('L05') <= t < s('L06'):
        al = ramp(t, s('L05') + 0.4, s('L05') + 0.7) * (1 - ramp(t, s('L06') - 0.3, s('L06')))
        K.pill(d, (485, 300), "dark ring = starved, no food left", col=WHITE, size=32, alpha=int(255 * al))
    if s('L06') + 1.5 <= t < s('L07') + 0.2:
        K.pill(d, (485, 300), "spots from nothing", col=AMBER, size=36, alpha=int(255 * ramp(t, s('L06') + 1.5, s('L06') + 1.8)))
    # Turing card + the two rules (equation once, ~3.2 s), then chip
    if s('L07') <= t < s('L08'):
        al = ramp(t, s('L07'), s('L07') + 0.3) * (1 - ramp(t, s('L08') - 0.3, s('L08')))
        K.pill(d, (485, 290), "Alan Turing · 1952 · 'The Chemical Basis of Morphogenesis'", col=WHITE, size=28, alpha=int(255 * al))
        K.chip(d, (485, 1225), "∂u/∂t = Du∇²u − uv² + f(1−u)", col=CYAN, size=34, alpha=int(255 * al))
        K.chip(d, (485, 1300), "∂v/∂t = Dv∇²v + uv² − (f+k)v", col=AMBER, size=34, alpha=int(255 * al))
    if s('L08') <= t < e('L08') + 0.5:
        al = ramp(t, s('L08'), s('L08') + 0.2) * (1 - ramp(t, e('L08') + 0.2, e('L08') + 0.5))
        K.chip(d, (230, 1560), "u food · v spark", col=CYAN, size=24, alpha=int(220 * al))
        kk = lerp(0.0649, 0.0600, ease(ramp(t, s('L08') + 1.5, e('L08') - 0.5)))
        cx, cy, R = 485, 330, 120
        ang = lerp(160, 20, ease(ramp(t, s('L08') + 1.5, e('L08') - 0.5)))
        def dial(d2):
            d2.arc([(cx - R) * 2, (cy - R) * 2, (cx + R) * 2, (cy + R) * 2], 180, 360, fill=(50, 60, 80, 255), width=22)
            d2.arc([(cx - R) * 2, (cy - R) * 2, (cx + R) * 2, (cy + R) * 2], 180, 180 + (180 - ang), fill=AMBER + (255,), width=22)
            th = math.radians(180 + (180 - ang))
            d2.line([cx * 2, cy * 2, (cx + (R - 8) * math.cos(th)) * 2, (cy + (R - 8) * math.sin(th)) * 2], fill=WHITE + (255,), width=10)
            d2.ellipse([(cx - 13) * 2, (cy - 13) * 2, (cx + 13) * 2, (cy + 13) * 2], fill=WHITE + (255,))
        K._faded(d, int(255 * al), dial)
        K.draw_text(d, (cx, cy + 50), f"spark death rate k = {kk:.4f}", 34, fill=AMBER + (int(255 * al),), stroke=4)
        K.pill(d, (cx, cy - R - 38), "SPOTS" if kk > 0.0615 else "STRIPES", col=WHITE, size=32, alpha=int(255 * al))
    for k, txt in (('L09', "angelfish stripes · Kondo & Asai 1995"), ('L10', "palate ridges · mice, Economou 2012"),
                   ('L11', "digits · mouse embryo, Raspopovic 2014"), ('L12', "Turing membrane · Tan et al. 2018, Science")):
        a, b = s(k), (s('L10') if k == 'L09' else s('L11') if k == 'L10' else s('L12') if k == 'L11' else s('L13'))
        if a <= t < b:
            al = ramp(t, a + 0.2, a + 0.4) * (1 - ramp(t, b - 0.25, b))
            K.pill(d, (485, 300), txt, col=CYAN, size=30, alpha=int(255 * al))
    if s('L12') + 0.8 <= t < s('L13'):
        al = ramp(t, s('L12') + 0.8, s('L12') + 1.1) * (1 - ramp(t, s('L13') - 0.25, s('L13')))
        K.pill(d, (485, 1270), "water passes ✓   salt blocked ✕", col=WHITE, size=30, alpha=int(255 * al))
    if s('L13') <= t < GROW0 + 2.5:
        al = ramp(t, s('L13') + 0.2, s('L13') + 0.5) * (1 - ramp(t, GROW0 + 2.0, GROW0 + 2.5))
        K.pill(d, (485, 300), "photo → dial map: big dot = dark = thick pattern", col=AMBER, size=30, alpha=int(255 * al))
    if GROW0 <= t < REVEAL + 2.5:
        al = ramp(t, GROW0, GROW0 + 0.3) * (1 - ramp(t, REVEAL + 2.0, REVEAL + 2.5))
        K.pill(d, (900, 1580), f"step {int(grow_x(t) * 20):,}", col=WHITE, size=26, alpha=int(220 * al), anchor="rm")
    if t >= s('L15') + 1.8:
        K.pill(d, (485, 1560), "TOMORROW: what does sound look like?", col=CYAN, size=32, alpha=int(255 * ramp(t, s('L15') + 1.8, s('L15') + 2.1)))
    K.draw_text(d, (925, 185), "SOURCE CODE #10", 26, fill=YELLOW, anchor="rm", stroke=3)
    return K.down(lay)

# ---------- frame ----------
def frame(t):
    if t < s('L02'):                                     # HOOK: Tesla finishing, light sweep, slow push
        img = portrait_img(lerp(0.75 * NSN, 0.9 * NSN, t / s('L02')), sweep=lerp(-200, 2400, t / s('L02')))
        sc = 1.06 - 0.06 * ease(t / s('L02')); img = zoom(img, sc, 540, 700)
    elif t < s('L03'):                                   # dive into the pattern -> dish
        a = ramp(t, s('L02'), s('L03') - 0.6)
        img = zoom(portrait_img(0.9 * NSN), 1 + 5 * ease(a) ** 1.5, 470, 560)
        if t > s('L03') - 1.4:
            b = ramp(t, s('L03') - 1.4, s('L03'))
            d_ = dish(field_tex(MV[0].astype(np.float32), MU[0].astype(np.float32)), 0.25)
            img = img * (1 - b) + d_ * b
    elif t < s('L08'):                                   # explainer dish (mitosis)
        x = mito_x(t); tilt = 0.25 + 0.35 * ease(ramp(t, s('L03'), s('L07'))) * (1 - 0.6 * ease(ramp(t, s('L07'), s('L08'))))
        V = stack_at(MV, x); U = stack_at(MU, x)
        zoomf = lerp(4.5, 1.0, ease(ramp(t, s('L03') + 1.0, s('L06'))))
        tex = field_tex(V, U)
        if zoomf > 1.01:
            c = 500; r = int(500 / zoomf); tex = cv2.resize(tex[c - r:c + r, c - r:c + r], (1000, 1000), interpolation=cv2.INTER_CUBIC)
        img = dish(tex, tilt, rot=(t - s('L03')) * 1.5)
    elif t < s('L09'):                                   # dial: spots -> stripes
        a = ramp(t, s('L08') + 1.5, e('L08') - 0.3)
        x = 499 * min(1, a * 1.25)
        img = dish(field_tex(stack_at(DV, x), stack_at(DU, x)), 0.3, rot=(t - s('L03')) * 1.5)
    elif t < s('L13'):                                   # nature + engineering vignettes
        for k, nm, nxt in (('L09', 'fish', 'L10'), ('L10', 'palate', 'L11'), ('L11', 'limb', 'L12'), ('L12', 'membrane', 'L13')):
            if s(k) <= t < s(nxt):
                a = ramp(t, s(k), s(nxt)); n = len(EX[nm]['V'])
                lim = {'fish': 30, 'palate': 45, 'limb': 16, 'membrane': n - 1}[nm]
                x = lim * min(1, (0.25 if nm == 'membrane' else 0.0) + a * 1.2)
                img = bg().copy()
                if nm == 'membrane':
                    img = membrane(t, x)
                else:
                    tex = masked_tex(nm, x)
                    img = paste_center(img, tex, 540, 820, scale=0.95 + 0.06 * a, alpha=1.0)
                fade = min(ramp(t, s(k), s(k) + 0.2), 1 - ramp(t, s(nxt) - 0.2, s(nxt)))
                img = img * (0.25 + 0.75 * fade)
                break
    elif t < GROW0:                                      # photo -> k map halftone
        img = kmap_img(ramp(t, s('L13'), s('L13') + 0.6))
    elif t < s('L15'):                                   # growth + reveal
        x = grow_x(t)
        sw = lerp(-300, 2600, ramp(t, REVEAL, REVEAL + 1.8)) if t > REVEAL else None
        img = portrait_img(x, sweep=sw)
        if t < GROW0 + 0.8: img = img * ramp(t, GROW0, GROW0 + 0.8) + kmap_img(1.0) * (1 - ramp(t, GROW0, GROW0 + 0.8))
    else:                                                # zoom out: the portrait sits inside a dish
        a = ease(ramp(t, s('L15'), s('L15') + 3.0))
        tex = cv2.resize(portrait_img(NSN)[200:1280, 0:1080], (1000, 1000))
        full = portrait_img(NSN)
        d_ = dish(tex, 0.3 * a, R=int(lerp(760, 430, a)))
        img = full * (1 - min(1, a * 2)) + d_ * min(1, a * 2)
    img = bloom(img, 0.35, 0.3)
    ov = overlay(t); o = np.asarray(ov).astype(np.float32); al = o[..., 3:4] / 255
    return np.clip(img * (1 - al) + o[..., :3] * al, 0, 255)

def zoom(img, sc, cx, cy):
    M = np.float32([[sc, 0, cx - sc * cx], [0, sc, cy - sc * cy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

DROPS = np.random.default_rng(11).uniform([0, 0], [1, 1], (90, 2))
def membrane(t, x):
    """membrane slab in perspective with Turing spots; water (cyan) passes, salt (red) bounces"""
    tex = field_tex(stack_at(EX['membrane']['V'], x), stack_at(EX['membrane']['U'], x), 900, food=False)
    sz = 900; src = np.float32([[0, 0], [sz, 0], [sz, sz], [0, sz]])
    dst = np.float32([[180, 700], [900, 700], [1040, 1000], [40, 1000]])
    Mx = cv2.getPerspectiveTransform(src, dst)
    img = bg().copy(); wm = cv2.warpPerspective(np.ones((sz, sz), np.float32), Mx, (W, H))
    img = img * (1 - wm[..., None]) + cv2.warpPerspective(tex, Mx, (W, H))
    # slab thickness edge
    cv2.rectangle(img, (40, 1000), (1040, 1030), (40, 60, 90), -1)
    u = t - s('L12')
    for i, (px, ph) in enumerate(DROPS):
        salt = i % 3 == 0
        yy = 420 + ((u * 260 + ph * 900) % 900)
        xx = 120 + px * 840
        if salt and yy > 690: yy = 690 - (yy - 690) * 0.6        # bounce back
        if not salt and 700 < yy < 1030: continue                  # inside the membrane
        col = (255, 90, 90) if salt else (90, 220, 255)
        cv2.circle(img, (int(xx), int(yy)), 9 if salt else 7, col, -1, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    import time
    for tt in [float(a) for a in sys.argv[1:]]:
        t0 = time.time(); K.finish(frame(tt)).save(f'f_{tt:05.2f}.png'); print(tt, round(time.time() - t0, 2))
