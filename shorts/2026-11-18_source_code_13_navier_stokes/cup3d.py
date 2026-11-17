"""Photoreal-ish 3D chai cup (numpy ray caster). World in mm: X right, Y away (screen-up at top-down), Z up.
The liquid surface uses the same Navier-Stokes dye as the flat top-down view, so the 3D shot lands
exactly on the 2D shot at elevation 90 deg."""
import numpy as np, cv2
from common import *
import chai

SS = 1.5                      # supersampling
R_IN, R_OUT = 39.2, 43.2      # mm (2D: 392 / 432 px at 10 px/mm)
Z_BASE, H_C, H_L = 3.0, 55.0, 47.0
R_S = 52.0                    # saucer
PXMM = 10.0
F0 = 6000.0                   # focal (px) at SS=1
D0 = 600.0                    # camera distance from target (mm)
ANG_H = np.deg2rad(225)       # handle direction (lower-left at top-down)
LDIR = np.array([-0.55, 0.45, 0.70]); LDIR = LDIR / np.linalg.norm(LDIR)
CER = np.array([0.96, 0.94, 0.90], np.float32)
FILL = np.array([0.45, -0.75, 0.5]); FILL = FILL / np.linalg.norm(FILL)

_tab = None
def table_tex():
    global _tab
    if _tab is None: _tab = chai.make_table().astype(np.float32)
    return _tab

def remap1d(img, mx, my, border):
    n = len(mx); w = 4096; h = (n + w - 1) // w; pad = h * w - n
    mx2 = np.pad(mx, (0, pad)).reshape(h, w).astype(np.float32); my2 = np.pad(my, (0, pad)).reshape(h, w).astype(np.float32)
    out = cv2.remap(img, mx2, my2, cv2.INTER_LINEAR, borderMode=border)
    return out.reshape(h * w, -1)[:n]

def saucer_z(r):
    z = np.where(r < 30, 3.0, 3.0 + 6.0 * np.clip((r - 30) / 18, 0, 1) ** 1.5)
    lip = np.clip((r - 48) / 4, 0, 1)
    z = z + 0.6 * np.sin(np.pi * lip) - 2.0 * lip ** 3
    return np.where(r <= R_S, z, -1.0)

def camera(elev_deg, yaw_deg=0.0, pp=(540, 1000), dist=D0):
    e = np.deg2rad(elev_deg); y = np.deg2rad(yaw_deg)
    tgt = np.array([0, 0, H_L * 0.5 + H_L * 0.5 * np.sin(e) ** 4])
    fwd_h = np.array([np.sin(y), np.cos(y), 0.0])                 # horizontal "away" dir
    pos = tgt - fwd_h * dist * np.cos(e) + np.array([0, 0, dist * np.sin(e)])
    fwd = tgt - pos; fwd /= np.linalg.norm(fwd)
    upg = fwd_h if e > np.deg2rad(89.9) else np.array([0, 0, 1.0])
    right = np.cross(fwd, upg); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up, pp

def project(cam, P):
    pos, fwd, right, up, pp = cam
    v = P - pos; z = v @ fwd
    return np.stack([pp[0] + F0 * (v @ right) / z, pp[1] - F0 * (v @ up) / z], -1)

def _cyl_hit(o, d, R, first=True):
    a = d[:, 0] ** 2 + d[:, 1] ** 2; b = 2 * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]); c = o[:, 0] ** 2 + o[:, 1] ** 2 - R * R
    disc = b * b - 4 * a * c; ok = disc > 0; s = np.sqrt(np.maximum(disc, 0))
    t = (-b - s) / (2 * a + 1e-12) if first else (-b + s) / (2 * a + 1e-12)
    return np.where(ok, t, np.inf)

def _handle_sdf(p):
    ch = np.array([(R_OUT + 9) * np.cos(ANG_H), (R_OUT + 9) * np.sin(ANG_H), 30.0])
    rad = np.array([np.cos(ANG_H), np.sin(ANG_H), 0.0]); tan = np.array([-np.sin(ANG_H), np.cos(ANG_H), 0.0])
    q = p - ch; u = q @ rad; v = q[:, 2]; w = q @ tan
    return np.hypot(np.hypot(u * 1.0, v * 0.82) - 14.0, w) - 4.6

def shadow(p):
    """soft shadow of cup (+saucer) along LDIR; 1 = lit"""
    lxy = LDIR[:2]; ln = np.linalg.norm(lxy); lu = lxy / ln
    pxy = p[:, :2]
    tc = -(pxy @ lu)                              # horizontal distance along light to closest approach
    dmin = np.abs(pxy[:, 0] * lu[1] - pxy[:, 1] * lu[0])
    zc = p[:, 2] + np.maximum(tc, 0) / ln * LDIR[2]
    occ = (tc > 0) & (zc < H_C + 2)
    s = np.ones(len(p), np.float32)
    pen = 2.0 + 0.04 * np.maximum(tc, 0)
    s = np.where(occ, np.clip((dmin - R_OUT + pen) / (2 * pen), 0, 1), s)
    occ2 = (tc > 0) & (zc < 9) & (p[:, 2] < 1)
    s2 = np.where(occ2, np.clip((dmin - R_S + 3) / 8, 0, 1), 1)
    return (s * s2).astype(np.float32)

def shade_ceramic(n, p, vdir, sh, occl=1.0):
    ndl = np.clip(n @ LDIR, 0, 1)
    h = LDIR[None] - vdir; h /= np.linalg.norm(h, axis=1, keepdims=True)
    spec = np.clip(np.sum(n * h, 1), 0, 1) ** 80 * 0.55 * sh
    fres = 0.04 + 0.5 * (1 - np.clip(-np.sum(n * vdir, 1), 0, 1)) ** 5
    fill = np.clip(n @ FILL, 0, 1) * 0.48
    amb = (0.26 + 0.12 * n[:, 2]) * occl
    col = CER[None] * (0.72 * ndl * sh + fill * occl + amb)[:, None] + (spec + fres * 0.25)[:, None]
    return col

def render(fi, t, elev, yaw=0.0, pp=(540, 1000), liquid_img=None, dist=D0):
    camd = dist
    Wr, Hr = int(W * SS), int(H * SS)
    cam = camera(elev, yaw, pp, dist); pos, fwd, right, up, _ = cam
    xs = (np.arange(Wr) + 0.5) / SS - pp[0]; ys = (np.arange(Hr) + 0.5) / SS - pp[1]
    X, Y = np.meshgrid(xs, ys)
    d = fwd[None] + (X.reshape(-1, 1) * right[None] - Y.reshape(-1, 1) * up[None]) / F0
    d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float64)
    n_ = len(d); o = np.broadcast_to(pos, (n_, 3))
    tbest = np.full(n_, np.inf); kind = np.zeros(n_, np.int8)   # 0 none,1 table,2 saucer,3 outer,4 rim,5 inner,6 liquid,7 handle
    # table
    tt = np.where(d[:, 2] < 0, -o[:, 2] / d[:, 2], np.inf); m = tt < tbest; tbest[m] = tt[m]; kind[m] = 1
    # outer wall
    to = _cyl_hit(o, d, R_OUT); zo = o[:, 2] + to * d[:, 2]
    m = (zo >= Z_BASE) & (zo <= H_C) & (to < tbest); tbest[m] = to[m]; kind[m] = 3
    # rim top
    tr = (H_C - o[:, 2]) / d[:, 2]; px = o[:, 0] + tr * d[:, 0]; py = o[:, 1] + tr * d[:, 1]; rr = np.hypot(px, py)
    m = (rr >= R_IN) & (rr <= R_OUT) & (tr < tbest) & (tr > 0); tbest[m] = tr[m]; kind[m] = 4
    # inside: liquid or inner wall
    ent = (rr < R_IN) & (tr > 0) & (tr < tbest)
    tl = (H_L - o[:, 2]) / d[:, 2]; lx = o[:, 0] + tl * d[:, 0]; ly = o[:, 1] + tl * d[:, 1]
    ti = _cyl_hit(o, d, R_IN, first=False); zi = o[:, 2] + ti * d[:, 2]
    hitl = ent & (np.hypot(lx, ly) <= R_IN)
    hiti = ent & ~hitl & (zi >= H_L) & (zi <= H_C)
    tbest[hitl] = tl[hitl]; kind[hitl] = 6; tbest[hiti] = ti[hiti]; kind[hiti] = 5
    # saucer (march, only rays near it)
    ts0 = _cyl_hit(o, d, R_S); cand = np.nonzero((ts0 < tbest) & (d[:, 2] < 0) & (kind != 3) & (kind != 4) & (kind != 5) & (kind != 6))[0]
    if len(cand):
        oc, dc = o[cand], d[cand]
        t0 = np.maximum((10.5 - oc[:, 2]) / np.minimum(dc[:, 2], -1e-6), 0); t1 = np.minimum(tbest[cand], (0 - oc[:, 2]) / np.minimum(dc[:, 2], -1e-6))
        found = np.full(len(cand), np.inf)
        tc = t0.copy(); step = 0.6
        prev_t = tc.copy()
        for _ in range(60):
            act = np.isinf(found) & (tc < t1)
            if not act.any(): break
            p = oc + tc[:, None] * dc; r = np.hypot(p[:, 0], p[:, 1]); zs = saucer_z(r)
            hit = act & (p[:, 2] <= zs)
            found[hit] = tc[hit]; prev_t = np.where(act & ~hit, tc, prev_t); tc = np.where(act & ~hit, tc + step, tc)
        ok = np.isfinite(found)
        lo, hi = prev_t[ok], found[ok]
        oo, dd = oc[ok], dc[ok]
        for _ in range(6):
            mid = (lo + hi) / 2; p = oo + mid[:, None] * dd; r = np.hypot(p[:, 0], p[:, 1])
            below = p[:, 2] <= saucer_z(r); hi = np.where(below, mid, hi); lo = np.where(below, lo, mid)
        idx = cand[ok]; better = hi < tbest[idx]
        tbest[idx[better]] = hi[better]; kind[idx[better]] = 2
    # handle (sphere trace on rays passing near it)
    ch = np.array([(R_OUT + 9) * np.cos(ANG_H), (R_OUT + 9) * np.sin(ANG_H), 30.0])
    oc_ = ch - pos; tca = d @ oc_; hdist = np.linalg.norm(oc_[None] - tca[:, None] * d, axis=1)
    hc = np.nonzero(hdist < 22)[0]
    if len(hc):
        th = np.maximum(tca[hc] - 25, 0.0); done = np.zeros(len(hc), bool)
        for _ in range(48):
            p = o[hc] + th[:, None] * d[hc]; sd = _handle_sdf(p)
            done |= sd < 0.05; th = np.where(done, th, th + sd)
        r = np.hypot(*(o[hc] + th[:, None] * d[hc])[:, :2].T)
        ok = done & (th < tbest[hc]) & (r > R_OUT - 0.5)
        tbest[hc[ok]] = th[ok]; kind[hc[ok]] = 7
    # ---------- shading ----------
    P = o + np.where(np.isfinite(tbest), tbest, 0)[:, None] * d
    col = np.zeros((n_, 3), np.float32)
    vdir = d
    # table
    m = kind == 1
    if m.any():
        tex = table_tex(); p = P[m]
        mx = (500 + p[:, 0] * PXMM).astype(np.float32); my = (965 - p[:, 1] * PXMM).astype(np.float32)
        c = remap1d(tex, mx, my, cv2.BORDER_REFLECT)
        r = np.hypot(p[:, 0], p[:, 1]); ao = 1 - 0.45 * np.exp(-np.maximum(r - R_S, 0) / 6)
        col[m] = c * (shadow(p) * 0.55 + 0.45)[:, None] * ao[:, None] * 1.05
    # saucer
    m = kind == 2
    if m.any():
        p = P[m]; r = np.hypot(p[:, 0], p[:, 1]); e = 0.05
        gz = (saucer_z(r + e) - saucer_z(r - e)) / (2 * e)
        nr = np.stack([-gz * p[:, 0] / (r + 1e-6), -gz * p[:, 1] / (r + 1e-6), np.ones_like(r)], 1)
        edge = r > R_S - 0.7
        nr[edge] = np.stack([p[edge, 0] / r[edge], p[edge, 1] / r[edge], np.zeros(edge.sum())], 1)
        nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        ao = 1 - 0.35 * np.exp(-np.maximum(r - R_OUT, 0) / 3)
        col[m] = shade_ceramic(nr, p, vdir[m], shadow(p + nr * 0.3), ao)
    # outer wall
    m = kind == 3
    if m.any():
        p = P[m]; r = np.hypot(p[:, 0], p[:, 1]); nr = np.stack([p[:, 0] / r, p[:, 1] / r, np.zeros_like(r)], 1)
        ao = 1 - 0.3 * np.exp(-(p[:, 2] - Z_BASE) / 4)
        col[m] = shade_ceramic(nr, p, vdir[m], np.ones(m.sum(), np.float32), ao)
    # rim (rounded lip normal)
    m = kind == 4
    if m.any():
        p = P[m]; r = np.hypot(p[:, 0], p[:, 1]); s = (r - (R_IN + R_OUT) / 2) / ((R_OUT - R_IN) / 2)
        nr = np.stack([p[:, 0] / r * s * 0.9, p[:, 1] / r * s * 0.9, np.sqrt(np.clip(1 - (s * 0.9) ** 2, 0.05, 1))], 1)
        col[m] = shade_ceramic(nr, p, vdir[m], np.ones(m.sum(), np.float32), 1.0)
    # inner wall
    m = kind == 5
    if m.any():
        p = P[m]; r = np.hypot(p[:, 0], p[:, 1]); nr = np.stack([-p[:, 0] / r, -p[:, 1] / r, np.zeros_like(r)], 1)
        dep = np.clip((p[:, 2] - H_L) / (H_C - H_L), 0, 1)
        lit = np.clip(nr @ LDIR, 0, 1) * np.clip(dep * 1.3, 0, 1)
        c = CER[None] * (0.25 + 0.30 * dep + 0.55 * lit)[:, None]
        tint = np.array([0.72, 0.48, 0.30], np.float32)        # tea stain/reflection near the surface
        col[m] = c * (1 - 0.35 * (1 - dep))[:, None] + tint[None] * 0.12 * (1 - dep)[:, None]
    # liquid (texture = the same NS dye shading as the flat view)
    m = kind == 6
    if m.any():
        if liquid_img is None:
            D, _ = chai.load(); liquid_img, _ = chai.liquid(D[fi])
        p = P[m]
        mx = (chai.R_LIQ + p[:, 0] * PXMM).astype(np.float32); my = (chai.R_LIQ - p[:, 1] * PXMM).astype(np.float32)
        c = remap1d(liquid_img.astype(np.float32), mx, my, cv2.BORDER_REPLICATE)
        cosv = np.clip(-vdir[m][:, 2], 0, 1)
        fres = 0.02 + 0.98 * (1 - cosv) ** 5
        refl = vdir[m] - 2 * vdir[m][:, 2:3] * np.array([[0, 0, 1.0]])   # mirror on flat surface
        win = np.clip(refl @ LDIR, 0, 1) ** 40 * 1.2
        r = np.hypot(p[:, 0], p[:, 1]); edge = np.exp(-(R_IN - r) / 2.0) * 0.25
        col[m] = c * (1 - 0.6 * fres)[:, None] + (fres * 0.55 + win)[:, None] * np.array([0.95, 0.93, 0.9]) + edge[:, None] * 0.4
    # handle
    m = kind == 7
    if m.any():
        p = P[m]; e = 0.05
        g = np.stack([(_handle_sdf(p + [e, 0, 0]) - _handle_sdf(p - [e, 0, 0])), (_handle_sdf(p + [0, e, 0]) - _handle_sdf(p - [0, e, 0])),
                      (_handle_sdf(p + [0, 0, e]) - _handle_sdf(p - [0, 0, e]))], 1)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        col[m] = shade_ceramic(g, p, vdir[m], shadow(p + g * 0.5), 1.0)
    m = kind == 0
    col[m] = 0.02
    img = col.reshape(Hr, Wr, 3)
    depth = np.where(np.isfinite(tbest), tbest, 2000).reshape(Hr, Wr).astype(np.float32)
    # depth of field (focus on the cup) in oblique views
    blur_amt = np.clip(1 - (elev - 40) / 45, 0, 1)
    if blur_amt > 0:
        fd = camd - 20
        coc = np.clip(np.abs(depth - fd) / 220, 0, 1) * blur_amt
        bl = cv2.GaussianBlur(img, (0, 0), 7 * SS)
        img = img * (1 - coc[..., None]) + bl * coc[..., None]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, cam
