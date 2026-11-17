"""X-ray view of the same chai solve: vorticity glow + tracer streaks, with a free camera."""
import numpy as np, cv2
from common import *
import chai

N = 512; CELL = 2 * np.pi / N
_P = _S = None
F0 = 150

def load():
    global _P, _S
    if _P is None:
        _P = np.load('tracers.npy', mmap_mode='r'); _S = np.load('tracer_spd.npy', mmap_mode='r')
    return _P, _S

def field_view(F, cx, cy, scale, w=W, h=H, ox=None, oy=None):
    """sample periodic sim field on canvas: canvas (ox,oy) <-> sim (cx,cy), scale px/unit"""
    ox = w / 2 if ox is None else ox; oy = h / 2 if oy is None else oy
    xs = (cx + (np.arange(w, dtype=np.float32) - ox) / scale) / CELL
    ys = (cy + (np.arange(h, dtype=np.float32) - oy) / scale) / CELL
    mx, my = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    return cv2.remap(np.asarray(F, np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

def render(fi, cx, cy, scale, ox=None, oy=None, alpha_field=1.0, trail=9, disk=None, gridpx=None):
    """returns float image (H,W,3). disk=(x,y,r) canvas circle to clip to (cup view)."""
    _, Wv = chai.load(); P, S = load()
    ox = W / 2 if ox is None else ox; oy = H / 2 if oy is None else oy
    w = field_view(Wv[fi], cx, cy, scale, ox=ox, oy=oy)
    w = w - float(np.asarray(Wv[fi], np.float32).mean())
    wp = np.clip(w / 7.0, 0, 1) ** 1.1; wn = np.clip(-w / 7.0, 0, 1) ** 1.1
    img = np.zeros((H, W, 3), np.float32)
    img += np.array([0.02, 0.035, 0.07], np.float32)
    img += wp[..., None] * np.array(CYAN, np.float32) / 255 * 0.55 * alpha_field
    img += wn[..., None] * np.array(AMBER, np.float32) / 255 * 0.45 * alpha_field
    # maths grid
    if gridpx:
        g = np.zeros((H, W), np.float32)
        step = gridpx
        sx = (ox - cx * scale) % step; sy = (oy - cy * scale) % step
        for x in np.arange(sx, W, step): g[:, int(x)] = 1
        for y in np.arange(sy, H, step): g[int(y), :] = 1
        img += g[..., None] * np.array([0.10, 0.25, 0.32], np.float32) * 0.5
    # tracer streaks (flow lines)
    k = fi - F0
    if 0 <= k < P.shape[0]:
        lay = np.zeros((H, W), np.float32)
        k0 = max(0, k - trail)
        a = P[k0]; b = P[k]; sp = np.asarray(S[k], np.float32)
        ax = ((a[:, 0] - cx) * scale + ox); ay = ((a[:, 1] - cy) * scale + oy)
        bx = ((b[:, 0] - cx) * scale + ox); by = ((b[:, 1] - cy) * scale + oy)
        vis = (bx > -20) & (bx < W + 20) & (by > -20) & (by < H + 20)
        inten = np.clip(sp / 1.2, 0.25, 1.0)
        for i in np.nonzero(vis)[0][::2]:
            cv2.line(lay, (int(ax[i] * 4), int(ay[i] * 4)), (int(bx[i] * 4), int(by[i] * 4)), float(inten[i]), 1, cv2.LINE_AA, shift=2)
        lay = np.clip(lay, 0, 1)
        img += lay[..., None] * np.array([0.75, 0.95, 1.0], np.float32) * 0.8
    if disk is not None:
        yy, xx = np.ogrid[0:H, 0:W]
        d = np.hypot(xx - disk[0], yy - disk[1])
        m = np.clip((disk[2] - d) / 1.5, 0, 1)
        img = img * m[..., None]
    return img
