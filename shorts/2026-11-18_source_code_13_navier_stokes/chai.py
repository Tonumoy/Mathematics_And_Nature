"""Top-down cup of chai; milk moved by the real 2D Navier-Stokes solve (chai_dye.npy / chai_w.npy)."""
import numpy as np, cv2
from common import *
import scipy.fft as sf

CX, CY = 500, 965          # cup centre on canvas
R_LIQ, R_RIM = 392, 432    # px
RS = 2.5                   # liquid radius in sim units
_D = _Wv = None
N = 512

def load():
    global _D, _Wv
    if _D is None:
        _D = np.load('chai_dye.npy', mmap_mode='r'); _Wv = np.load('chai_w.npy', mmap_mode='r')
    return _D, _Wv

def crop_field(F, out_px, cx=np.pi, cy=np.pi, rs=RS):
    """crop sim box around (cx,cy) radius rs -> out_px square, cubic"""
    cell = 2 * np.pi / N
    i0 = int((cy - rs) / cell); j0 = int((cx - rs) / cell); n = int(2 * rs / cell)
    idx_i = np.arange(i0, i0 + n) % N; idx_j = np.arange(j0, j0 + n) % N
    sub = np.asarray(F, np.float32)[np.ix_(idx_i, idx_j)]
    return cv2.resize(sub, (out_px, out_px), interpolation=cv2.INTER_CUBIC)

# ---------- static background: walnut table + saucer + cup ----------
def _noise(shape, scale, seed):
    r = np.random.default_rng(seed).normal(size=(shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(r, (0, 0), 1.0), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

def _aniso(seed, sx, sy):
    r = np.random.default_rng(seed).normal(size=(H // sy + 3, W // sx + 3)).astype(np.float32)
    return cv2.resize(r, (W + 3 * sx, H + 3 * sy), interpolation=cv2.INTER_CUBIC)[:H, :W]

def make_table():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    plank = (xx // 270).astype(int)
    lo = _noise((H, W), 160, 1)
    g1 = _aniso(4, 3, 120); g2 = _aniso(5, 8, 300); g3 = _aniso(6, 1, 40)
    grain = 0.55 * g1 + 0.35 * g2 + 0.15 * g3 + 0.3 * lo
    rings = 0.5 + 0.5 * np.sin(xx * 0.035 + plank * 2.1 + 2.2 * _aniso(7, 30, 500) + 0.4 * lo)
    v = 0.5 + 0.12 * grain + 0.18 * (rings ** 3)
    base = np.array([0.33, 0.19, 0.10], np.float32); dark = np.array([0.13, 0.065, 0.035], np.float32)
    v = np.clip(v, 0, 1)
    col = dark[None, None] + (base - dark)[None, None] * (1 - v[..., None]) * 1.4
    col *= (0.92 + 0.10 * ((plank * 37) % 5) / 4)[..., None]
    gap = np.abs(((xx + 2) % 270) - 2) < 2.0
    col[gap] *= 0.3
    d = np.hypot((xx - 150) / W, (yy - 250) / H)
    col *= (1.25 - 0.85 * np.clip(d, 0, 1.2))[..., None]
    return np.clip(col, 0, 1)

def make_static():
    tab = make_table()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # saucer shadow + saucer
    rs = 520
    d = np.hypot(xx - CX - 28, yy - CY - 36)
    sh = np.clip(1 - np.exp(-np.maximum(d - rs + 40, 0) ** 2 / 3000), 0, 1)
    tab *= (0.45 + 0.55 * sh)[..., None]
    d = np.hypot(xx - CX, yy - CY)
    sauc = d < rs
    ang = np.arctan2(yy - CY, xx - CX)
    lit = 0.5 + 0.5 * np.cos(ang + 2.3)              # light from top-left
    cer = np.array([0.88, 0.86, 0.82], np.float32)
    sh2 = (0.80 + 0.16 * lit) * (1 - 0.10 * np.clip((d - 440) / 90, 0, 1))
    groove = np.exp(-((d - 440) ** 2) / 30) * 0.18
    rimhl = np.exp(-((d - rs + 10) ** 2) / 20) * 0.25 * lit
    sc = cer[None, None] * (sh2 - groove + rimhl)[..., None]
    edge = np.clip((rs - d) / 2.0, 0, 1)
    tab = tab * (1 - edge[..., None]) + sc * edge[..., None]
    # cup shadow on saucer
    d2 = np.hypot(xx - CX - 22, yy - CY - 28)
    sh = np.clip(1 - np.exp(-np.maximum(d2 - R_RIM + 30, 0) ** 2 / 1500), 0, 1)
    tab *= (0.55 + 0.45 * sh)[..., None]
    # handle (lower-left)
    hx, hy = CX - 330, CY + 330
    hd = np.hypot(xx - hx, yy - hy)
    hmask = np.clip((np.abs(hd - 78) < 24) & (np.hypot(xx - CX, yy - CY) > R_RIM - 10), 0, 1).astype(np.float32)
    hmask = cv2.GaussianBlur(hmask, (0, 0), 1.2)
    hl = 0.78 + 0.2 * np.cos(np.arctan2(yy - hy, xx - hx) + 2.3) * np.cos((hd - 78) / 24 * 1.5)
    tab = tab * (1 - hmask[..., None]) + (cer[None, None] * hl[..., None]) * hmask[..., None]
    # rim (white ceramic annulus) with lighting
    d = np.hypot(xx - CX, yy - CY)
    rimm = np.clip((R_RIM - d) / 1.5, 0, 1) * np.clip((d - R_LIQ + 6) / 1.5, 0, 1)
    t = np.clip((d - R_LIQ) / (R_RIM - R_LIQ), 0, 1)
    prof = 0.82 + 0.18 * np.maximum(np.sin(np.pi * t), 0) ** 0.5                     # rounded lip
    inner = (1 - t) ** 3 * 0.22 * (0.5 - 0.5 * np.cos(ang + 2.3))      # inner wall shadow
    spec = np.exp(-((t - 0.45) ** 2) / 0.02) * np.clip(np.cos(ang + 2.3), 0, 1) ** 6 * 0.35
    rc = cer[None, None] * (prof * (0.84 + 0.16 * lit) - inner + spec)[..., None]
    tab = tab * (1 - rimm[..., None]) + rc * rimm[..., None]
    return tab.astype(np.float32)

_static = None
def static():
    global _static
    if _static is None:
        import os
        if os.path.exists('static3d_top.npy'): _static = np.load('static3d_top.npy')   # 3D top-down render = same cup as the 3D shot
        elif os.path.exists('chai_static.npy'): _static = np.load('chai_static.npy')
        else: _static = make_static(); np.save('chai_static.npy', _static)
    return _static

_yy, _xx = np.mgrid[0:2 * R_LIQ, 0:2 * R_LIQ].astype(np.float32)
_dl = np.hypot(_xx - R_LIQ, _yy - R_LIQ)
_angl = np.arctan2(_yy - R_LIQ, _xx - R_LIQ)
_mask = np.clip((R_LIQ - _dl) / 1.5, 0, 1)

CHAI = np.array([0.36, 0.17, 0.06], np.float32)
MILKY = np.array([0.70, 0.47, 0.27], np.float32)
MILK = np.array([0.95, 0.89, 0.78], np.float32)

def liquid(dye):
    m = crop_field(cv2.GaussianBlur(np.asarray(dye, np.float32), (0, 0), 1.3), 2 * R_LIQ)
    m = np.clip((m - 0.035) / 0.965, 0, 1.2)
    a = np.clip(m * 1.15, 0, 1) ** 0.85
    col = np.where(a[..., None] < 0.5,
                   CHAI + (MILKY - CHAI) * (a[..., None] * 2),
                   MILKY + (MILK - MILKY) * ((a[..., None] - 0.5) * 2))
    h = cv2.GaussianBlur(m, (0, 0), 4.0)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny, nz = -gx * 1.6, -gy * 1.6, np.ones_like(h)
    nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2); nx /= nn; ny /= nn; nz /= nn
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    lam = nx * L[0] + ny * L[1] + nz * L[2]
    col = col * (0.72 + 0.38 * lam)[..., None]
    # window reflection, wobbled by the surface normals
    ox = _xx + nx * 60; oy = _yy + ny * 60
    win = np.exp(-(((ox - R_LIQ * 0.55) / 120) ** 2 + ((oy - R_LIQ * 0.42) / 60) ** 2))
    win *= np.clip(np.sin((ox - R_LIQ * 0.55) / 22) * 4, 0, 1) * 0.5 + 0.5        # window panes
    col = col + win[..., None] * 0.28
    # meniscus + inner shadow from rim (light from top-left -> shadow on top-left inner edge)
    edge = np.clip((R_LIQ - _dl) / 40, 0, 1)
    shad = (1 - edge) * (0.5 + 0.5 * np.cos(_angl + 2.3 - np.pi)) * 0.0 + (1 - edge) ** 2 * 0.35 * (0.5 + 0.5 * np.cos(_angl + 2.3))
    col = col * (1 - shad[..., None])
    men = np.exp(-((_dl - R_LIQ + 3) ** 2) / 6) * 0.25
    col = col + men[..., None]
    return col, _mask

_steam = None
def steam(t):
    global _steam
    if _steam is None:
        n = _noise((2400, 600), 40, 9) + 0.5 * _noise((2400, 600), 14, 10)
        _steam = np.clip(n - 0.6, 0, None) ** 1.5
    off = int(t * 55) % 1200
    s = _steam[1200 - off: 2400 - off: 1, :]
    s = cv2.resize(s, (W, 1200))
    out = np.zeros((H, W), np.float32)
    out[300:1500] = s
    yy = np.arange(H, dtype=np.float32)[:, None]; xx = np.arange(W, dtype=np.float32)[None]
    m = np.exp(-((xx - CX) / 330) ** 2) * np.clip((1350 - yy) / 500, 0, 1) * np.clip((yy - 300) / 400, 0, 1)
    return np.clip(out * m * 0.16, 0, 0.25)

def render(fi, t, zoom=1.0, steam_on=True):
    D, Wv = load()
    img = static().copy()
    col, m = liquid(D[fi])
    y0, x0 = CY - R_LIQ, CX - R_LIQ
    sub = img[y0:y0 + 2 * R_LIQ, x0:x0 + 2 * R_LIQ]
    img[y0:y0 + 2 * R_LIQ, x0:x0 + 2 * R_LIQ] = sub * (1 - m[..., None]) + col * m[..., None]
    if steam_on:
        s = steam(t); img = img + s[..., None] * np.array([1, 1, 1], np.float32)
    if zoom != 1.0:
        M = cv2.getRotationMatrix2D((CX, CY), 0, zoom)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return img
