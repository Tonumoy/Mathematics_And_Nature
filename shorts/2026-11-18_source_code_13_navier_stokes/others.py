"""Real-life panels: wing (penalised NS), storm (NS), blood (Poiseuille, exact NS solution)."""
import numpy as np, cv2
from common import *

def _n(shape, scale, seed):
    r = np.random.default_rng(seed).normal(size=(shape[0] // scale + 2, shape[1] // scale + 2)).astype(np.float32)
    return cv2.resize(r, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

# ---------------- wing ----------------
_wd = _wm = None
def wing(k):
    global _wd, _wm
    if _wd is None: _wd = np.load('wing_dye.npy', mmap_mode='r'); _wm = np.load('wing_mask.npy')
    N = _wd.shape[1]; cell = 2 * np.pi / N
    x0u, x1u = 0.9, 5.0; sc = W / (x1u - x0u)
    cy = np.pi
    xs = (x0u + np.arange(W) / sc) / cell; ys = (cy + (np.arange(H) - H / 2) / sc) / cell
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    d = cv2.remap(np.asarray(_wd[k % _wd.shape[0]], np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    m = cv2.remap(_wm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    yy = np.arange(H, dtype=np.float32)[:, None]
    sky = np.array([0.03, 0.05, 0.09], np.float32) + (yy / H)[..., None] * np.array([0.02, 0.03, 0.05], np.float32)
    img = np.broadcast_to(sky, (H, W, 3)).copy()
    sm = np.clip(d, 0, 1) ** 0.9
    img += sm[..., None] * np.array([0.85, 0.92, 1.0], np.float32) * 0.85
    # wing: brushed metal with top light
    gy = cv2.Sobel(cv2.GaussianBlur(m, (0, 0), 3), cv2.CV_32F, 0, 1, ksize=3)
    metal = 0.55 + 0.35 * np.clip(-gy * 6, -1, 1)
    mm = np.clip(m * 1.4, 0, 1)
    img = img * (1 - mm[..., None]) + (metal[..., None] * np.array([0.78, 0.82, 0.88], np.float32)) * mm[..., None]
    vg = np.clip(np.minimum(yy - 330, 1590 - yy) / 160, 0, 1)[..., None]
    return img * (0.25 + 0.75 * vg)

# ---------------- storm ----------------
_sd = None; _ocean = None
def storm(k):
    global _sd, _ocean
    if _sd is None:
        _sd = np.load('storm_dye.npy', mmap_mode='r')
        n = _n((H, W), 90, 31) * 0.6 + _n((H, W), 20, 32) * 0.4
        _ocean = np.array([0.02, 0.10, 0.22], np.float32) + n[..., None] * np.array([0.0, 0.015, 0.03], np.float32)
    N = _sd.shape[1]; cell = 2 * np.pi / N
    sc = W / 4.6; cx = cy = np.pi
    xs = (cx + (np.arange(W) - W / 2) / sc) / cell; ys = (cy - (np.arange(H) - 930) / sc) / cell   # flip y: CCW like N. hemisphere
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    d = cv2.remap(np.asarray(_sd[k % _sd.shape[0]], np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    yy0 = np.arange(H, dtype=np.float32)[:, None]; xx0 = np.arange(W, dtype=np.float32)[None]
    rad = np.hypot(xx0 - 540, yy0 - 930)
    c = cv2.GaussianBlur(np.clip(d, 0, 1), (0, 0), 1.6) * np.clip((780 - rad) / 200, 0, 1)
    h = cv2.GaussianBlur(c, (0, 0), 6)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    lam = np.clip(0.85 - 0.9 * gx - 0.9 * gy, 0.55, 1.1)
    shadow = cv2.warpAffine(c, np.float32([[1, 0, 14], [0, 1, 18]]), (W, H))
    img = _ocean * (1 - 0.5 * shadow[..., None])
    a = np.clip(c * 1.5, 0, 1) ** 0.8
    img = img * (1 - a[..., None]) + (lam[..., None] * np.array([0.93, 0.95, 0.98], np.float32)) * a[..., None]
    yy = np.arange(H, dtype=np.float32)[:, None]; xx = np.arange(W, dtype=np.float32)[None]
    limb = np.clip(1.15 - np.hypot((xx - 540) / 900, (yy - 930) / 1100) ** 2, 0.2, 1)   # globe curvature light
    return img * limb[..., None]

# ---------------- blood ----------------
def _cell_sprite(n=160):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32); r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    thick = np.clip(1 - r ** 8, 0, 1) * (0.55 + 0.45 * np.clip((r - 0.15) / 0.55, 0, 1) ** 0.7)   # biconcave
    gx = cv2.Sobel(thick, cv2.CV_32F, 1, 0, ksize=5) / 20; gy = cv2.Sobel(thick, cv2.CV_32F, 0, 1, ksize=5) / 20
    lam = np.clip(0.7 - 0.6 * gx - 0.8 * gy, 0.2, 1.3)
    col = np.stack([0.78 * lam + 0.1 * thick, 0.07 * lam + 0.03, 0.06 * lam + 0.03], -1)
    spec = np.clip(-gx - gy, 0, None) ** 2 * 0.5
    col = col + spec[..., None] * np.array([1, 0.6, 0.6], np.float32)
    alpha = np.clip((1 - r) * n / 6, 0, 1) * (0.75 + 0.25 * thick)
    return np.dstack([col, alpha]).astype(np.float32)
SPR = _cell_sprite()
rng = np.random.default_rng(8); NC = 110
CXr = rng.uniform(-0.95, 0.95, NC); Y0 = rng.uniform(0, 1.2, NC); ROT = rng.uniform(0, np.pi, NC)
TUM = rng.uniform(0.6, 1.4, NC); DEP = rng.uniform(0, 1, NC); SZ = rng.uniform(0.85, 1.15, NC)
VX, VR = 500, 330
_tis = None
def blood(t):
    global _tis
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - VX) / VR
    if _tis is None:
        _tis = np.array([0.16, 0.02, 0.04], np.float32) * (0.75 + 0.25 * _n((H, W), 40, 41)[..., None])
    inside = np.clip((1 - np.abs(dx)) * 25, 0, 1)
    plasma = np.array([0.42, 0.07, 0.06], np.float32) * (0.6 + 0.4 * (1 - dx ** 2))[..., None]
    wall = np.exp(-((np.abs(dx) - 1) * 10) ** 2)[..., None] * np.array([0.95, 0.5, 0.5], np.float32) * 0.5
    img = _tis * (1 - inside[..., None]) + plasma * inside[..., None] + wall
    # Poiseuille: u(r) = Umax (1 - r^2/R^2); heartbeat pulse 1.2 Hz
    pulse = 1.0 + 0.7 * np.maximum(np.sin(2 * np.pi * 1.2 * t), 0) ** 3
    disp = 0.42 * (1 - CXr ** 2) * (t + 0.10 * (1 - np.cos(2 * np.pi * 1.2 * t)))
    ys = ((Y0 + disp) % 1.3 - 0.15) * H
    for i in np.argsort(DEP):
        s = int(84 * SZ[i] * (0.8 + 0.3 * DEP[i]))
        sq = 0.3 + 0.7 * abs(np.cos(ROT[i] + t * TUM[i] * 2.4 * CXr[i]))     # shear: cells tumble near the wall
        sp = cv2.resize(SPR, (s, max(int(s * sq), 3)), interpolation=cv2.INTER_AREA)
        if DEP[i] < 0.35: sp = cv2.GaussianBlur(sp, (0, 0), 3.0 * (0.35 - DEP[i]) / 0.35 + 0.5)
        h, w = sp.shape[:2]; cx = int(VX + CXr[i] * (VR - 45) - w / 2); cy = int(ys[i] - h / 2)
        x0, y0 = max(cx, 0), max(cy, 0); x1, y1 = min(cx + w, W), min(cy + h, H)
        if x1 <= x0 or y1 <= y0: continue
        s_ = sp[y0 - cy:y1 - cy, x0 - cx:x1 - cx]; a_ = s_[..., 3:4] * (0.55 + 0.45 * DEP[i])
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a_) + s_[..., :3] * (0.6 + 0.5 * DEP[i]) * a_
    # exact velocity profile (parabola) drawn as a glowing curve with arrows
    prof = np.zeros((H, W, 3), np.float32); y0p = 1190
    xs = np.linspace(-1, 1, 80); pts = np.stack([VX + xs * VR, y0p + 170 * (1 - xs ** 2) * pulse], 1)
    cv2.polylines(prof, [(pts * 4).astype(np.int32)], False, (1, 1, 1), 3, cv2.LINE_AA, shift=2)
    for f in np.linspace(-0.8, 0.8, 7):
        x = int(VX + f * VR); L = int(170 * (1 - f ** 2) * pulse)
        cv2.arrowedLine(prof, (x, y0p), (x, y0p + L), (1, 1, 1), 2, cv2.LINE_AA, tipLength=min(0.3, 25 / max(L, 1)))
    return img, prof, pulse
