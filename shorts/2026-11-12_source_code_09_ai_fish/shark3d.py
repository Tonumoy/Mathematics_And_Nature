# True 3D shark: lofted body mesh + fins, painter-sorted, lit from above, countershaded, smooth orientation.
import numpy as np, cv2
import sc9v2 as V
from sc9v2 import S, W2, H2, project
E = V.E
L = 8.0; NS, NA = 46, 22; SH = 4
SS = np.linspace(0, 1, NS); AA = np.linspace(0, 2*np.pi, NA, endpoint=False)
_s = [0, .02, .06, .14, .26, .40, .55, .70, .85, .95, 1]
_hh = [.02, .17, .33, .52, .67, .71, .60, .42, .24, .14, .11]
_hw = [.03, .23, .41, .56, .63, .61, .49, .32, .16, .085, .06]
def HH(s): return np.interp(s, _s, _hh)
def HW(s): return np.interp(s, _s, _hw)
def X(s): return L*(0.08 - 0.86*np.asarray(s, float))
def S_of_x(x): return (0.08*L - x)/(0.86*L)
# body vertices (body frame: x forward, y up, z side)
sa, ca = np.sin(AA), np.cos(AA)
BV = np.zeros((NS, NA, 3)); BN = np.zeros((NS, NA, 3))
for i, s in enumerate(SS):
    hh, hw = HH(s), HW(s)
    BV[i, :, 0] = X(s); BV[i, :, 1] = hh*sa*np.where(sa < 0, 0.82, 1.0) + 0.06*hh; BV[i, :, 2] = hw*ca
    n = np.stack([np.zeros(NA), sa*hw, ca*hh], 1); BN[i] = n/np.linalg.norm(n, axis=1, keepdims=True)
TOP = np.array([0.33, 0.44, 0.54]); BEL = np.array([0.93, 0.95, 0.95]); WATER = np.array([0.02, 0.11, 0.19])
tcs = np.clip((sa + 0.38)/0.30, 0, 1); tcs = tcs*tcs*(3 - 2*tcs)
BC = BEL[None]*(1 - tcs[:, None]) + TOP[None]*tcs[:, None]                 # colour by angle
def fin(pts): return np.array(pts, float)
def fins():
    F = []
    F.append(fin([[X(.35), HH(.35), 0], [X(.43), HH(.43) + 0.9, 0], [X(.49) - 0.15, HH(.49) + 1.55, 0], [X(.50) - 0.1, HH(.50) + 0.75, 0], [X(.55), HH(.55), 0]]))      # dorsal
    F.append(fin([[X(.77), HH(.77), 0], [X(.82) - 0.1, HH(.82) + 0.42, 0], [X(.84), HH(.84), 0]]))                                                                    # 2nd dorsal
    x1 = X(1.0)
    F.append(fin([[X(.96), HH(.96), 0], [x1 - 0.75, 1.25, 0], [x1 - 1.55, 2.25, 0], [x1 - 1.15, 1.15, 0], [x1 - 0.55, 0.12, 0],
                  [x1 - 0.95, -0.75, 0], [x1 - 1.05, -1.25, 0], [x1 - 0.5, -0.65, 0], [X(.96), -HH(.96)*0.8, 0]]))                                                    # caudal
    for sg in (1, -1):
        F.append(fin([[X(.22), -0.25*HH(.22), sg*0.9*HW(.22)], [X(.33), -0.75, sg*(HW(.3) + 1.25)], [X(.41), -1.0, sg*(HW(.3) + 2.0)],
                      [X(.38), -0.6, sg*(HW(.3) + 0.9)], [X(.34), -0.3*HH(.34), sg*0.9*HW(.34)]]))                                                                    # pectorals
        F.append(fin([[X(.62), -0.6*HH(.62), sg*0.6*HW(.62)], [X(.68), -HH(.66) - 0.4, sg*(HW(.66) + 0.3)], [X(.69), -0.6*HH(.69), sg*0.6*HW(.69)]]))                # pelvic
    return F
FINS = fins()
ORI = {}
def smooth_frame(key, q, u):
    uh = E.unit(u[None])[0]
    st = ORI.get(key)
    if st is None or np.linalg.norm(q - st['q']) > 2.5:
        up = np.array([0, 1.0, 0]); up = up - (up @ uh)*uh
        if np.linalg.norm(up) < 0.2: up = np.array([1.0, 0, 0]) - uh[0]*uh
        st = dict(f=uh, up=E.unit(up[None])[0], q=q.copy(), turn=0.0)
    else:
        f_new = E.unit((st['f'] + (uh - st['f'])*0.10)[None])[0]
        turn = float(np.cross(st['f'], f_new) @ st['up'])                      # signed yaw rate -> body curl
        up = st['up'] + 0.03*(np.array([0, 1.0, 0]) - st['up']); up = up - (up @ f_new)*f_new
        st = dict(f=f_new, up=E.unit(up[None])[0], q=q.copy(), turn=st['turn'] + (turn - st['turn'])*0.15)
    ORI[key] = st; return st
def bend(x, ph, turn):
    s = np.clip(S_of_x(x), 0, 1.3)
    return 0.42*np.sin(ph - 3.0*s)*s**1.5 + np.clip(turn*90, -1.2, 1.2)*s**2
SJ = 0.16; LOWM = sa < -0.45                      # jaw hinge position, lower-jaw angle mask
def jaw(P, low, op):
    """rotate snout points about the jaw hinge: lower jaw drops, upper lifts a little"""
    if op <= 0.001: return P
    P = P.copy(); s_ = S_of_x(P[..., 0]); w = np.clip((SJ - s_)/0.05, 0, 1)
    phi = np.where(low, -0.55, 0.16)*op*w
    px = X(SJ); py = -0.45*HH(SJ)*0.82 + 0.06*HH(SJ)
    dx, dy = P[..., 0] - px, P[..., 1] - py
    P[..., 0] = px + dx*np.cos(phi) - dy*np.sin(phi); P[..., 1] = py + dx*np.sin(phi) + dy*np.cos(phi)
    return P
MOUTH = {}
def draw_shark(img, q, u, C, cp, t, k):
    yaw, pitch, dist = cp
    st = smooth_frame(k, q, u); f, up = st['f'], st['up']; r = np.cross(f, up)
    ph = t*3.4 + k*2.1
    tr = getattr(V, 'CUR', None); tgt_open = 0.0
    if tr is not None:
        dm = np.linalg.norm(tr.Pd - (q + f*0.6), axis=1); ahead = ((tr.Pd - q) @ f) > -0.5
        dmin = dm[ahead].min() if ahead.any() else 99.0
        tgt_open = float(np.clip((4.5 - dmin)/2.5, 0, 1))
        if any(0 <= t - e_[0] < 0.25 and np.linalg.norm(e_[1] - q) < 4 for e_ in tr.flash[-6:]): tgt_open = 1.0
    op = MOUTH.get(k, 0.0); op += (tgt_open - op)*(0.22 if tgt_open > op else 0.10); MOUTH[k] = op
    def world(P, low=False):
        P = jaw(np.asarray(P, float), low, op)
        z = P[..., 2] + bend(P[..., 0], ph, st['turn'])
        return q + P[..., 0:1]*f + P[..., 1:2]*up + z[..., None]*r
    Wb = world(BV, LOWM[None, :] & np.ones((NS, 1), bool)); px, py, pk, pd = project(Wb, C, yaw, pitch, dist)
    if pd.min() < 5: return None
    fogv = float(np.clip(np.exp(-(pd.mean() - dist)/38.0), 0.15, 1.0))*float(np.clip((pd.min() - 5)/5, 0, 1))
    Ld = E.unit(np.array([[0.25, 1.0, 0.2]]))[0]
    Nw = BN[..., 1:2]*up + BN[..., 2:3]*r
    diff = Nw @ Ld; shade = 0.86 + 0.26*diff + 0.25*np.clip(diff, 0, 1)**6                   # (NS, NA)
    polys = []
    P2 = np.stack([px, py], -1)*SH
    for i in range(NS - 1):
        for j in range(NA):
            j2 = (j + 1) % NA
            d = (pd[i, j] + pd[i+1, j] + pd[i, j2] + pd[i+1, j2])*0.25
            c = (BC[j] + BC[j2])*0.5*(shade[i, j] + shade[i+1, j2])*0.5
            if LOWM[j] != LOWM[j2] and SS[i] < SJ and op > 0.03: c = c + (np.array([0.42, 0.05, 0.08]) - c)*min(1.0, op*3)
            polys.append((d, np.array([P2[i, j], P2[i+1, j], P2[i+1, j2], P2[i, j2]]), c))
    for Fp in FINS:
        Wf = world(Fp); fx, fy, fk, fd = project(Wf, C, yaw, pitch, dist)
        n = np.cross(Wf[1] - Wf[0], Wf[-1] - Wf[0]); n = n/(np.linalg.norm(n) + 1e-9)
        c = TOP*(0.80 + 0.35*abs(float(n @ Ld)))
        polys.append((fd.mean() + 0.05, np.stack([fx, fy], -1)*SH, c))
    polys.sort(key=lambda p: -p[0])
    tint = np.array([0.92, 1.0, 1.06])
    for d, pts, c in polys:
        col_ = WATER + (c*tint - WATER)*fogv
        cv2.fillPoly(img, [pts.astype(np.int32)], tuple(float(v) for v in col_), cv2.LINE_8, 2)
    # details on the camera-facing side: eye, gill slits, mouth line
    cam = V.unproject(V.CX, V.CY0, 0.0, C, yaw, pitch, dist)
    for sg in (1, -1):
        if (r*sg) @ (q - cam) > 0: continue
        def surf(s, ang):
            hh, hw = HH(s), HW(s); sa_ = np.sin(ang)
            return np.array([X(s), hh*sa_*(0.82 if sa_ < 0 else 1.0) + 0.06*hh, sg*hw*np.cos(ang)*1.01])
        dark = tuple(float(v) for v in (WATER + (np.array([0.03, 0.04, 0.06]) - WATER)*fogv))
        e = world(surf(0.075, 0.42)[None]); ex, ey, ek, ed = project(e, C, yaw, pitch, dist)
        cv2.circle(img, (int(ex[0]*SH), int(ey[0]*SH)), int(max(2.5, ek[0]*0.075)*SH), dark, -1, cv2.LINE_AA, 2)
        cv2.circle(img, (int((ex[0] - ek[0]*0.025)*SH), int((ey[0] - ek[0]*0.03)*SH)), int(max(1, ek[0]*0.02)*SH), (0.9*fogv, 0.95*fogv, 1.0*fogv), -1, cv2.LINE_AA, 2)
        for g in (0.185, 0.205, 0.225, 0.245, 0.265):
            pts = world(np.array([surf(g + 0.012*(a_ - 0.1), a_) for a_ in np.linspace(-0.25, 0.55, 5)]))
            gx, gy, gk, gd = project(pts, C, yaw, pitch, dist)
            cv2.polylines(img, [(np.stack([gx, gy], -1)*SH).astype(np.int32)], False, dark, max(1, int(gk[0]*0.022)), cv2.LINE_AA, 2)
        pts = world(np.array([surf(s_, -0.62 - 0.2*np.sin(np.pi*(s_ - 0.03)/0.10)) for s_ in np.linspace(0.03, 0.13, 7)]))
        gx, gy, gk, gd = project(pts, C, yaw, pitch, dist)
        if op < 0.15: cv2.polylines(img, [(np.stack([gx, gy], -1)*SH).astype(np.int32)], False, dark, max(1, int(gk[0]*0.03)), cv2.LINE_AA, 2)
        if op > 0.2:
            ab = -0.466                                                    # jaw edge angle (sin = -0.45)
            white = tuple(float(v) for v in (WATER + (np.array([1.0, 1.0, 0.97]) - WATER)*fogv))
            se = np.linspace(0.025, 0.135, 9)
            for low_, tip in ((False, -0.11), (True, 0.09)):
                for a_, b_ in zip(se[:-1], se[1:]):
                    A_ = surf(a_, ab if sg > 0 else ab); B_ = surf(b_, ab); T_ = (A_ + B_)/2 + np.array([0, tip*min(1, op*1.5), 0])
                    tri = world(np.array([A_, B_, T_]), low_)
                    tx_, ty_, _, _ = project(tri, C, yaw, pitch, dist)
                    cv2.fillPoly(img, [(np.stack([tx_, ty_], -1)*SH).astype(np.int32)], white, cv2.LINE_AA, 2)
    hx, hy, hk, hd = project((q + f*0.3)[None], C, yaw, pitch, dist)
    return hx[0], hy[0], hk[0], hd[0]
