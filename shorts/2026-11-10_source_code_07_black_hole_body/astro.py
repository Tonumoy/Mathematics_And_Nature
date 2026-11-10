# SC#7 astronaut relight + tidal slicing. Uses astro.npz from sprite.py (own SDF render).
import numpy as np, cv2
import sim7
D = np.load('astro.npz'); M = D['mask']; NX, NY, NZ = D['nx'], D['ny'], D['nz']; MAT = D['mat']; AO = D['ao']; YB = D['y']
SH, SW = M.shape; PPM = SH/2.0                     # sprite px per metre
YTOP = 1.95                                        # sprite row 0 = y 1.95 m
rng = np.random.default_rng(7)
_w = cv2.GaussianBlur(rng.normal(0, 1, (SH//10, SW//10)).astype(np.float32), (0, 0), 1.8)
WRINKLE = 1 + 0.025*cv2.resize(_w/_w.std(), (SW, SH))                       # fabric folds
ALB = np.zeros((SH, SW, 3), np.float32)
for m, c in ((0, (0.86, 0.86, 0.84)), (1, (0.93, 0.93, 0.93)), (2, (0.30, 0.31, 0.33)), (3, (0.45, 0.47, 0.52)), (4, (0.05, 0.04, 0.03))):
    ALB[MAT == m] = c
ALB[MAT == 0] *= WRINKLE[MAT == 0, None]
N3 = np.stack([NX, NY, NZ], -1)
def nrm(v): v = np.array(v, np.float32); return v/np.linalg.norm(v)
def shade(heat=1.0):
    """relit astronaut RGBA (float RGB, alpha). heat = disk brightness factor"""
    Lk = nrm((0.15, -0.75, 0.45)); Lf = nrm((-0.35, 0.5, 0.8))
    dk = (N3 @ Lk).clip(0); df = (N3 @ Lf).clip(0)
    fres = (1 - NZ.clip(0, 1))**2.6
    col = ALB*(0.16 + dk[..., None]*np.array([1.0, 0.62, 0.30])*1.1*heat + df[..., None]*np.array([0.62, 0.68, 0.80])*0.95)*AO[..., None]
    col += fres[..., None]*np.array([1.0, 0.68, 0.32])*0.85*heat*M[..., None]                      # ring behind: warm rim
    H = nrm(Lk + np.array([0, 0, 1.0])); spec = (N3 @ H).clip(0)**40
    shiny = np.isin(MAT, (1, 3, 4)).astype(np.float32)
    col += (spec*shiny)[..., None]*np.array([1.0, 0.8, 0.55])*1.4*heat
    vis = (MAT == 4)
    R = np.stack([2*NZ*NX, 2*NZ*NY, 2*NZ*NZ - 1], -1)                                             # reflect view ray
    band = np.exp(-((R[..., 1] + 0.35)/0.16)**2)*1.8 + np.exp(-((R[..., 1] + 0.1)/0.05)**2)*2.5   # disk + photon ring seen in the visor
    env = band[..., None]*np.array([1.0, 0.62, 0.25]) + 0.03
    col[vis] = (env[vis]*np.array([1.0, 0.80, 0.42])*0.9 + spec[vis, None]*2.0)
    return col.astype(np.float32), M.astype(np.float32)

def holo():
    """x-ray hologram of the same model (cyan fresnel edges + scan lines)"""
    fres = (1 - NZ.clip(0, 1))**1.6
    lines = 0.75 + 0.25*(np.arange(SH) % 6 < 3)[:, None]
    a = (0.12 + fres*1.1)*lines*M
    col = a[..., None]*np.array([90, 225, 255], np.float32)/255
    return col.astype(np.float32), np.clip(a*1.2, 0, 1).astype(np.float32)

def bands():
    """sprite row ranges for the N sim pieces (k=0 feet ... N-1 head); piece k centre y = 0.9 + X[k]"""
    out = []
    for k in range(sim7.N):
        yc = 0.9 + sim7.X[k]; lo = yc - sim7.L0/2 - (0.06 if k == sim7.N-1 else 0); hi = yc + sim7.L0/2 + (0.2 if k == sim7.N-1 else 0) - (0.0)
        if k == 0: lo -= 0.06
        r0 = int(np.clip((YTOP - hi)*PPM, 0, SH)); r1 = int(np.clip((YTOP - lo)*PPM, 0, SH)); out.append((r0, r1, yc))
    return out
BANDS = bands()

def place(canvas, rgb, a, cx, cy, pxm, offs, sy=1.0, lat=1.0, blur=0.0, strands=0.0, add=False, taper=0.0):
    """composite pieces onto canvas (float HxWx3, 2x px). offs: piece offsets (m, + toward head) relative to body centre cy."""
    Hc, Wc = canvas.shape[:2]; sc = pxm/PPM
    tops = []
    for k, (r0, r1, yc) in enumerate(BANDS):
        if r1 <= r0: continue
        h = max(2, int((r1 - r0)*sc*sy)); w = max(2, int(SW*sc*lat))
        pr = cv2.resize(rgb[r0:r1], (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        pa = cv2.resize(a[r0:r1], (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        if taper > 0:
            v = (np.arange(h, dtype=np.float32) + 0.5)/h; rs = 1 - taper*(1 - np.sin(np.pi*v)**0.6)
            mx = ((np.arange(w, dtype=np.float32)[None, :] - w/2)/rs[:, None] + w/2).astype(np.float32); my = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
            pr = cv2.remap(pr, mx, my, cv2.INTER_LINEAR, borderValue=0); pa = cv2.remap(pa, mx, my, cv2.INTER_LINEAR, borderValue=0)
            pa = pa*np.clip(np.sin(np.pi*v)*2.2, 0, 1)[:, None]
        if blur > 0.5:
            kz = int(blur)*2 + 1; pr = cv2.GaussianBlur(pr, (1, kz), 0); pa = cv2.GaussianBlur(pa, (1, kz), 0)
        ycen = cy - offs[k]*pxm*1.0                                     # piece centre on screen
        y0 = int(ycen - h/2); x0 = int(cx - w/2)
        ya, yb = max(0, y0), min(Hc, y0 + h); xa, xb = max(0, x0), min(Wc, x0 + w)
        if yb <= ya or xb <= xa: continue
        sr = pr[ya-y0:yb-y0, xa-x0:xb-x0]; sa = pa[ya-y0:yb-y0, xa-x0:xb-x0, None]
        if add: canvas[ya:yb, xa:xb] += sr*sa
        else: canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb]*(1 - sa) + sr*sa
        tops.append((y0, y0 + h))
    if strands > 0 and len(tops) > 1:                                    # thin stretched strands bridging the gaps
        for (a0, a1), (b0, b1) in zip(tops[:-1], tops[1:]):
            lo, hi = sorted([a1, b0]) if a1 < b0 else sorted([b1, a0])
            if hi - lo > 4:
                for dx in (-0.06, 0.0, 0.05):
                    x = int(cx + dx*pxm*lat)
                    cv2.line(canvas, (x, lo), (x, hi), (0.9*strands, 0.55*strands, 0.28*strands), max(1, int(0.012*pxm)), cv2.LINE_AA)
    return tops

def place_whole(canvas, rgb, a, cx, cy, pxm, sy=1.0, lat=1.0):
    """intact body: one stretched image, centre (y=0.9 m) at cy"""
    sc = pxm/PPM; h = int(SH*sc*sy); w = int(SW*sc*lat)
    pr = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA); pa = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    y0 = int(cy - (YTOP - 0.9)*PPM*sc*sy); x0 = int(cx - w/2); Hc, Wc = canvas.shape[:2]
    ya, yb = max(0, y0), min(Hc, y0 + h); xa, xb = max(0, x0), min(Wc, x0 + w)
    sa = pa[ya-y0:yb-y0, xa-x0:xb-x0, None]
    canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb]*(1 - sa) + pr[ya-y0:yb-y0, xa-x0:xb-x0]*sa
    return y0, y0 + h, w

_tear_noise = np.random.default_rng(3).random(4096).astype(np.float32)
def place_stretch(canvas, rgb, a, cx, cy, pxm, offs, lat0=1.0, tear=0.0, blur=0.0):
    """continuous body: sprite row (body y) -> screen row via the solver pieces' offsets (m, + toward head, about the centre).
    stretched zones neck down (volume-ish) and, if tear>0, rip where the pieces have separated."""
    X = sim7.X; Hc, Wc = canvas.shape[:2]; sc = pxm/PPM
    s0 = (offs[1]-offs[0])/(X[1]-X[0]); s1 = (offs[-1]-offs[-2])/(X[-1]-X[-2])
    Xe = np.concatenate([[X[0]-0.35], X, [X[-1]+0.5]]); Oe = np.concatenate([[offs[0]-0.35*s0], offs, [offs[-1]+0.5*s1]])
    top = int(cy - Oe[-1]*pxm) - 2; bot = int(cy - Oe[0]*pxm) + 2
    ya, yb = max(0, top), min(Hc, bot)
    if yb <= ya: return
    rows = np.arange(ya, yb, dtype=np.float32); rel = (cy - rows)/pxm
    ybody = np.interp(rel, Oe, Xe) + 0.9
    lam = np.interp(rel, 0.5*(Oe[1:]+Oe[:-1]), np.diff(Oe)/np.diff(Xe))            # local stretch ratio
    seg = np.clip(np.searchsorted(Oe, rel) - 1, 0, len(Oe)-2)
    u = (rel - Oe[seg])/(Oe[seg+1] - Oe[seg])                                          # 0..1 inside a gap between pieces
    neck = np.exp(-((u-0.5)/0.22)**2)*np.clip((lam-1.3)/3, 0, 1)
    latr = lat0/np.clip(lam, 1, 30)**0.42*(1 - 0.3*neck)
    latr = np.clip(latr, 0.12, 1.2)
    w = int(SW*sc*lat0) + 4; xa, xb = max(0, int(cx - w/2)), min(Wc, int(cx + w/2))
    xs = np.arange(xa, xb, dtype=np.float32)
    mx = (SW/2 + (xs[None, :] - cx)/(latr[:, None]*sc)).astype(np.float32)
    my = np.repeat(((YTOP - ybody)*PPM).astype(np.float32)[:, None], len(xs), 1)
    pr = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pa = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if tear > 0:
        nz = _tear_noise[(np.arange(len(xs))*7 + (seg[:, None]*131)) % 4096]
        rip = np.exp(-((u[:, None]-0.5)/(0.05 + 0.25*np.clip((lam[:, None]-1.3)/4, 0, 1)))**2)*tear*np.clip((lam[:, None]-1.35)*1.5, 0, 1)
        pa = pa*np.clip(1 - rip*(0.75 + 0.5*nz), 0, 1)
        pr = pr + (rip*(1-np.clip(1 - rip, 0, 1)))[..., None]*np.array([0.9, 0.35, 0.1], np.float32)*0.0
    if blur > 0.5:
        k = int(blur)*2+1; pr = cv2.GaussianBlur(pr, (1, k), 0); pa = cv2.GaussianBlur(pa, (1, k), 0)
    canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb]*(1 - pa[..., None]) + pr*pa[..., None]
