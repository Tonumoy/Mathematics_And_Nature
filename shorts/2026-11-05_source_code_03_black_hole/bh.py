# SOURCE CODE #3 - "WOULD IT EAT EARTH?" (Coded Cosmos, 5 Nov 2026)
# real thing: ray-traced Schwarzschild black hole (sim.py table) -> x-ray -> real light paths u''+u=3Mu^2
import numpy as np, cv2, subprocess, sys
from PIL import Image, ImageDraw
import sim, source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE

FPS, BPM = 30, 96; BT = 60/BPM; BEATS = 68; T = BEATS*BT                 # 42.5 s
S = 2; W2, H2 = K.W*S, K.H*S
RNG = np.random.default_rng(5)
SCAN0, SCAN1 = 11.25, 12.15; TROPHY = 39*BT                               # 24.375
N2, BREAK, SNAP, PHOT, BACK0, BACK1 = 46*BT, 50*BT, 54*BT, 58*BT, 62*BT, 62*BT+0.8
BC = sim.B_CRIT
CAPS = [(2.4, 5.4, [[('swap the Sun for a', WHITE)], [('black hole', YELLOW), ('. Earth sucked in?', WHITE)]]),
        (5.4, 8.6, [[('nope. ', YELLOW), ('same mass = same pull.', WHITE)], [('Earth just keeps ', WHITE), ('going round', YELLOW)]]),
        (8.6, 11.25, [[("it'd just get ", WHITE), ('cold + dark', YELLOW), ('.', WHITE)], [('so what ', WHITE), ('DOES', YELLOW), (' it eat?', WHITE)]]),
        (12.2, 15.0, [[("let's fire ", WHITE), ('light', YELLOW), (' at it.', WHITE)], [('real paths, solved', WHITE)]]),
        (15.0, 18.1, [[('gravity ', WHITE), ('bends', YELLOW), (' the light...', WHITE)], [('...and some never leave', WHITE)]]),
        (18.1, 21.2, [[('faster...', WHITE)], [('miss wide: it ', WHITE), ('escapes', YELLOW)]]),
        (21.2, TROPHY, [[('aim too close and', WHITE)], [('it ', WHITE), ('never comes back', RED)]]),
        (TROPHY, N2, [[("light's real path.", WHITE)], [('3Mu²', AMBER), (" = Einstein's extra pull", WHITE)]]),
        (N2, BREAK, [[('aim ', WHITE), ('5.30', AMBER), (': it whips', WHITE)], [('around and ', WHITE), ('escapes', YELLOW)]]),
        (BREAK, SNAP, [[('aim ', WHITE), ('5.15', RED), (': trapped.', WHITE)], [('circles, then ', WHITE), ('falls in', YELLOW)]]),
        (SNAP, PHOT, [[('5.25', GREEN), (': free again.', WHITE)], [('the edge is ', WHITE), ('5.196', YELLOW), (' exactly', WHITE)]]),
        (PHOT, BACK0, [[('that edge = the ', WHITE), ('shadow', YELLOW)], [('Sun-size hole: ', WHITE), ('15 km', YELLOW), (' wide', WHITE)]]),
        (BACK1, 41.0, [[('2019: a telescope', WHITE)], [('as big as ', WHITE), ('Earth', YELLOW)]]),
        (41.0, T-0.4, [[('photographed one.', WHITE)], [('same ', WHITE), ('shadow', YELLOW), ('. same maths.', WHITE)]])]
SLOW_TXT = 'light slowed ~25,000 x'
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
def tonemap(x): return 1-np.exp(-1.2*x)

# ================= real-life layer: camera ray-trace from the solver table =================
SC_R = 40*S; RCX, RCY = 540*S, 900*S; INC = np.radians(84); R_IN, R_OUT = 6.0, 19.0
def build_real():
    d = np.load('table.npz'); bs, P, U, pinf = d['bs'], d['P'], d['U'], d['pinf']
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    b = np.hypot(xx-RCX, yy-RCY)/SC_R; al = np.arctan2(-(yy-RCY), xx-RCX)
    bi = np.clip(np.round((b-bs[0])/(bs[1]-bs[0])).astype(np.int32), 0, len(bs)-1)
    dP = P[1]-P[0]
    p1 = np.mod(np.arctan2(-np.cos(INC), np.sin(al)*np.sin(INC)), np.pi)
    hit_r = np.full(b.shape, np.nan, np.float32); hit_p = np.zeros_like(hit_r)
    for k in range(3):                                         # first disk crossing along the backward ray
        pk = p1 + k*np.pi; j = np.clip((pk/dP).astype(np.int32), 0, len(P)-2); f = pk/dP - j
        u = U[bi, j]*(1-f) + U[bi, j+1]*f; r = 1/u
        ok = np.isnan(hit_r) & np.isfinite(r) & (r >= R_IN) & (r <= R_OUT) & (pk < P[-1])
        hit_r[ok] = r[ok]; hit_p[ok] = pk[ok]
    disk = np.isfinite(hit_r); r = np.where(disk, hit_r, 10.0); ph = hit_p
    ca, sa = np.cos(al), np.sin(al)
    Px, Py, Pz = r*np.sin(ph)*ca, r*np.sin(ph)*sa, r*np.cos(ph)               # hit point, camera coords
    th = np.arctan2(Py*np.cos(INC) - Pz*np.sin(INC), Px)                     # angle in disk plane
    N = np.array([0, np.sin(INC), np.cos(INC)])
    vz = (N[0]*Py - N[1]*Px)/r*np.sqrt(1/r)                                 # (N x P^)_z * v, v = sqrt(M/r)
    v = np.sqrt(1/r); Dop = np.sqrt(1-v**2)/(1-vz); g = Dop*np.sqrt(np.clip(1-3/r, 0.05, 1))
    em = np.clip((1-np.sqrt(R_IN/r)), 0, 1)*(R_IN/r)**2.2*g**3.5*38.0
    tcol = np.clip((R_IN/r)**0.75*g, 0, 1.6)                                # temperature-ish -> colour
    col = np.stack([np.ones_like(tcol), np.clip(0.28+0.62*tcol, 0, 1), np.clip(0.04+0.95*(tcol-0.45), 0.01, 1)], -1)
    # sky: escaped rays -> lensed star map
    pi_ = np.interp(b, bs, np.nan_to_num(pinf, nan=-1)); esc = (pi_ > 0) & ~disk & (b > BC*0.98)
    dx, dy, dz = np.sin(pi_)*ca, np.sin(pi_)*sa, np.cos(pi_)
    lon = np.arctan2(dx, -dz); lat = np.arcsin(np.clip(dy, -1, 1))
    capt = ~esc & ~disk
    return dict(disk=disk, r=r.astype(np.float32), th=th.astype(np.float32), em=(em*disk)[..., None].astype(np.float32),
                col=col.astype(np.float32), esc=esc, lon=lon.astype(np.float32), lat=lat.astype(np.float32), capt=capt)

def star_map(w=4096, h=2048):
    m = np.zeros((h, w, 3), np.float32); yy, xx = np.mgrid[0:h, 0:w]
    lat = (0.5-yy/h)*np.pi; lon = (xx/w-0.5)*2*np.pi
    band = np.exp(-((lat-0.35*np.sin(lon+0.6))/0.16)**2)                    # milky-way band
    nz = cv2.GaussianBlur(RNG.random((h//8, w//8)).astype(np.float32), (0, 0), 3); nz = cv2.resize(nz, (w, h))
    m += (band*(0.015+0.05*nz))[..., None]*np.array([0.75, 0.8, 1.0], np.float32)
    n = 26000; ys, xs = RNG.integers(0, h, n), RNG.integers(0, w, n); mag = RNG.pareto(2.2, n)*0.25+0.12
    tint = np.array([[1, .85, .7], [1, 1, 1], [.75, .85, 1]], np.float32)[RNG.integers(0, 3, n)]
    np.add.at(m, (ys, xs), (mag[:, None]*tint).astype(np.float32))
    return cv2.GaussianBlur(m, (0, 0), 0.7)*2.2

def disk_tex(w=2048, h=256):
    t = np.zeros((h, w), np.float32)
    for k, (fy, fx, a) in enumerate([(3, 4, .5), (9, 11, .3), (23, 29, .25), (61, 70, .15)]):
        nz = RNG.random((fy*2, fx*4)).astype(np.float32)
        nz = np.concatenate([nz, nz[:, :1]], 1)
        t += cv2.resize(nz, (w+w//(fx*4), h), interpolation=cv2.INTER_CUBIC)[:, :w]*a
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=6, sigmaY=0.8)
    return (0.55 + 0.9*(t-t.mean())/t.std()*0.35).clip(0.1, 2)

def real_at(t, R, STAR, TEX):
    Om = R['r']**-1.5*2.2                                                     # Kepler: inner disk spins faster
    mx = (np.mod(R['th'] - Om*t, 2*np.pi)/(2*np.pi)*TEX.shape[1]).astype(np.float32)
    my = ((R['r']-R_IN)/(R_OUT-R_IN)*(TEX.shape[0]-1)).astype(np.float32)
    tex = cv2.remap(TEX, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[..., None]
    sx = (np.mod(R['lon'] + 0.004*t + np.pi, 2*np.pi)/(2*np.pi)*STAR.shape[1]).astype(np.float32)
    sy = ((0.5 - R['lat']/np.pi)*STAR.shape[0]).astype(np.float32)
    sky = cv2.remap(STAR, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)*R['esc'][..., None]
    img = sky + R['em']*tex*R['col']
    small = cv2.resize(img, (W2//8, H2//8), interpolation=cv2.INTER_AREA)       # camera bloom
    bl = cv2.GaussianBlur(np.maximum(small-0.6, 0), (0, 0), 6) + cv2.GaussianBlur(np.maximum(small-0.3, 0), (0, 0), 18)*0.5
    img = img + cv2.resize(bl, (W2, H2), interpolation=cv2.INTER_LINEAR)*0.8
    s = 1.04 + 0.05*sm(t/SCAN1) if t < SCAN1 else 1.09 - 0.05*sm((t-BACK1)/(T-0.5-BACK1))
    Mw = np.float32([[s, 0, (1-s)*RCX + 3*S*np.sin(t*.4)], [0, s, (1-s)*RCY + 2*S*np.sin(t*.3)]])
    return cv2.warpAffine(img, Mw, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ================= x-ray layer: real light paths =================
XS = 38*S; XCX, XCY = 490*S, 880*S
def mk_rays():
    ev = [(13.0, 8.0, 8), (14.6, -4.0, 8), (16.0, 5.6, 8)]                  # first 3 slow
    t = 18.1
    while t < 22.8:
        rate, v = (6, 80) if t < 19.6 else (28, 320)
        b = RNG.uniform(-9.5, 9.5) if RNG.random() < 0.7 else np.sign(RNG.random()-.5)*RNG.uniform(4.8, 6.2)
        ev.append((t, b, v)); t += 1/rate
    ev += [(N2+0.15, 5.30, 28), (BREAK+0.15, 5.15, 28), (SNAP+0.15, 5.25, 28)]
    out = []
    for (t0, b, v) in ev:
        x, y, cap = sim.path_xy(b, n=2400)
        px = XCX + x*XS; py = XCY - y*XS
        seg = np.hypot(np.diff(x), np.diff(y)); s = np.concatenate([[0], np.cumsum(seg)])
        s0 = np.interp(-14.0, x[:400], s[:400]) if x[0] < -14 else 0           # start drawing at x = -14 M
        out.append(dict(t0=t0, b=b, v=v, px=px, py=py, s=s-s0, cap=cap))
    return out
RAYS = None
def xray_at(t):
    img = np.zeros((H2, W2, 3), np.float32); img[:] = K.NIGHT/255
    cv2.circle(img, (XCX, XCY), int(3*XS), (0.10, 0.30, 0.38), 1*S, cv2.LINE_AA)       # photon sphere (BGR-agnostic: RGB)
    glow = np.zeros_like(img); nr = nc = 0
    dial = t >= N2 - 0.5
    for i, R in enumerate(RAYS):
        if t < R['t0']: continue
        is_dial = i >= len(RAYS)-3
        if is_dial and not (R['t0'] <= t < R['t0'] + (BREAK-N2) - 0.1 or (i == len(RAYS)-1 and t < BACK1)): continue
        sh = (t - R['t0'])*R['v']; m = (R['s'] >= 0) & (R['s'] <= sh)
        if m.sum() < 2: continue
        done = sh >= R['s'][-1]
        if not is_dial: nr += 1; nc += int(R['cap'] and done)
        if is_dial: c = AMBER if i == len(RAYS)-3 else (RED if i == len(RAYS)-2 else GREEN); a = 1.0
        else:
            c = RED if (R['cap'] and done) else CYAN
            a = (0.55 if done else 1.0)*(0.18 if dial else 1)*(0.75 if R['v'] > 100 else 1)
        pts = np.stack([R['px'][m], R['py'][m]], 1).astype(np.int32)
        col = tuple(float(v)/255*a for v in c)
        cv2.polylines(img, [pts], False, col, 2*S if (is_dial or R['v'] < 50) else S, cv2.LINE_AA)
        if not done or is_dial:
            hx, hy = pts[-1]; cv2.circle(glow, (int(hx), int(hy)), 9*S, tuple(float(v)/255*1.4 for v in c), -1, cv2.LINE_AA)
            cv2.polylines(glow, [pts[-60:]], False, tuple(float(v)/255 for v in c), 6*S, cv2.LINE_AA)
    cv2.circle(img, (XCX, XCY), int(2*XS), (0, 0, 0), -1, cv2.LINE_AA)               # horizon
    cv2.circle(img, (XCX, XCY), int(2*XS), tuple(v/255 for v in CYAN), 2*S, cv2.LINE_AA)
    if 21.0 < t < BACK0:                                                         # shadow edge b = 3 sqrt3 M
        a = sm((t-21)/0.5)
        for sg in (1, -1):
            y = int(XCY - sg*BC*XS)
            for x0 in range(40*S, int(XCX-2.5*XS), 26*S): cv2.line(img, (x0, y), (x0+14*S, y), tuple(v/255*a for v in GREEN), 2*S, cv2.LINE_AA)
    small = cv2.resize(glow, (W2//4, H2//4), interpolation=cv2.INTER_AREA)
    img += cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (W2, H2))*0.9 + glow*0.25
    return img, nr, nc

# ================= text =================
STATE = {}
def text(t, im):
    d = ImageDraw.Draw(im)
    K.title_box(d, 'WOULD IT EAT EARTH?', 'swap the Sun for a black hole' if t < 11 or t > BACK0 else None, y=205)
    K.series_tag(d, 3)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 12.6 < t < TROPHY - 0.4:
        d.text((90, 1375), "solver: u'' + u = 3Mu²  (RK45)", font=K.F_MONO, fill=CYAN, anchor='ls')
        if t > 13: d.text((90, 1412), f"rays: {STATE.get('nr', 0)}   trapped: {STATE.get('nc', 0)}", font=K.F_MONO, fill=CYAN, anchor='ls')
    if 5.6 < t < 11.0: K.pill(d, (CX, 1300), 'sped up: 1 year in 10 s', AMBER)
    if 13.2 < t < 15.6: K.pill(d, (CX, 470), SLOW_TXT, AMBER)
    if 18.1 < t < 19.6: K.pill(d, (790, 470), 'x10', AMBER)
    elif 19.6 < t < 22.8: K.pill(d, (790, 470), 'x40', AMBER)
    if 21.5 < t < N2: d.text((70, int(880 - BC*38) - 14), 'edge 5.196', font=K.F_MONO, fill=GREEN, anchor='ls')
    if N2 <= t < BACK0:
        bv, c = ('5.30', AMBER) if t < BREAK else (('5.15', RED) if t < SNAP else ('5.25', GREEN))
        K.pill(d, (CX, 470), f'dial:  aim b = {bv} M', c)
    if TROPHY <= t < N2 - 0.3:
        a = sm((t-TROPHY)/0.35); y = 1290
        segs = [("u''", CYAN, K.F_EQ, 0), (' + ', WHITE, K.F_EQ, 0), ('u', CYAN, K.F_EQ, 0), (' = ', WHITE, K.F_EQ, 0), ('3Mu', AMBER, K.F_EQ, 0), ('2', AMBER, K.F_SUB, -26)]
        tw = sum(d.textlength(s_, font=f) for s_, _, f, _ in segs); x = CX - tw/2
        for s_, c, f, dy in segs:
            d.text((x, y+dy), s_, font=f, fill=tuple(int(v*a) for v in c), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s_, font=f)
        d.text((CX, 1365), 'u = 1 / distance,  M = mass', font=K.F_MONO, fill=tuple(int(v*a) for v in WHITE), anchor='mm')
    elif N2 - 0.3 <= t < BACK0:
        K.pill(d, (CX, 1360), "u'' + u = 3Mu²   edge: 3√3 M = 5.196 M", CYAN, font=K.F_CHIP)
    if BACK1 < t < BACK1 + 1.6: pass
    K.end_stamp(d, alpha=sm((t-41.2)/0.4)*(1-sm((t-T+0.45)/0.3)), y=1385)

# ================= Earth (owner 5 Nov): thrown in (the guess), then just orbits (the truth) =================
import earth as EA
EA_A, EA_B, EA_W, EA_T0 = 470*S, 235*S, 0.63, 1.25
def earth_state(t):
    rs = 1 + 0.8*(1 - sm(t/5.4)); th = EA_T0 + EA_W*t
    x = RCX + rs*EA_A*np.cos(th); y = RCY + rs*EA_B*np.sin(th)
    D = int(118*S*(1 + 0.28*np.sin(th))*(1 + 0.9*(rs-1)))
    return x, y, D, th
def add_earth(L, t, R):
    s = 1.04 + 0.05*sm(t/SCAN1)
    if 5.0 < t:                                                                # dashed orbit ellipse, cyan
        a = sm((t-5.0)/0.5)*0.55
        ov = np.zeros_like(L)
        for k in range(48):
            if k % 2: continue
            t0, t1 = 2*np.pi*k/48, 2*np.pi*(k+1)/48
            pts = np.array([[RCX + EA_A*np.cos(u), RCY + EA_B*np.sin(u)] for u in np.linspace(t0, t1, 6)], np.int32)
            cv2.polylines(ov, [pts], False, tuple(v/255*a for v in CYAN), 2*S, cv2.LINE_AA)
        L = L + ov
    x, y, D, th = earth_state(t)
    dim = 1 - 0.7*sm((t-8.6)/1.5)                                              # Sun gone: cold + dark
    lc = (1, 1, 1) if t < 6 else tuple(np.array([1, 1, 1])*(1-sm((t-6)/2)) + np.array([1.0, 0.62, 0.35])*sm((t-6)/2))
    lx, ly = (RCX - x), -(RCY - y); ln = np.hypot(lx, ly) + 1e-6
    light = (-0.6, -0.3, 0.75) if t < 6 else (lx/ln*0.8, ly/ln*0.8, 0.5)
    col, al = EA.sprite(max(D, 20), rot=30*t + 10, light=light, lcol=lc, gain=1.0*dim + 0.05)
    n = col.shape[0]; x0, y0 = int(x - n/2), int(y - n/2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0+n, W2), min(y0+n, H2)
    if xa >= xb or ya >= yb: return L
    c = col[ya-y0:yb-y0, xa-x0:xb-x0]; a = al[ya-y0:yb-y0, xa-x0:xb-x0][..., None].copy()
    a0 = a.copy()
    if np.sin(th) < 0:                                                         # behind the hole: hidden by shadow + disk
        yy, xx = np.mgrid[ya:yb, xa:xb]
        a *= (np.hypot(xx-RCX, yy-RCY) > BC*SC_R*s)[..., None]
        a *= np.clip(1 - R['em'][ya:yb, xa:xb]*0.8, 0.55, 1)
    L[ya:yb, xa:xb] = L[ya:yb, xa:xb]*(1-a) + c*(a/(a0+1e-6))
    return L

# ================= frame =================
GRAIN = [RNG.normal(0, 1, (K.H, K.W, 1)).astype(np.float32) for _ in range(6)]
vy, vx = np.mgrid[0:K.H, 0:K.W].astype(np.float32)
VIG = (1 - 0.30*(((vx-K.W/2)/K.W)**2 + ((vy-K.H/2)/K.H)**2)*2)[..., None]
def frame(t, R, STAR, TEX):
    L = real_at(t, R, STAR, TEX) if (t < SCAN1 or t > BACK0) else None
    if L is not None and BACK1 < t < BACK1 + 1.8:                              # same number, on the real thing
        a = sm((t-BACK1)/0.3)*(1-sm((t-BACK1-1.4)/0.4)); s = 1.09 - 0.05*sm((t-BACK1)/(T-0.5-BACK1))
        ov = np.zeros_like(L); cv2.circle(ov, (RCX, RCY), int(BC*SC_R*s), tuple(v/255*a for v in GREEN), 2*S, cv2.LINE_AA); L = L + ov
    if L is not None and t < SCAN0 + 0.2: L = add_earth(L, t, R)
    if L is not None and 2.4 < t < SCAN0:                                         # J2 subject ring (owner 5 Nov)
        a = sm((t-2.4)/0.3)*(1-sm((t-SCAN0+0.4)/0.4)); s = 1.04 + 0.05*sm(t/SCAN1)
        rr = int(BC*SC_R*s*1.32*(1+0.025*np.sin(t*4.0)))
        ov = np.zeros_like(L); cv2.circle(ov, (RCX, RCY), rr, tuple(v/255*a*1.9 for v in CYAN), 4*S, cv2.LINE_AA); L = L + ov
    X = None
    if SCAN0 <= t <= BACK1:
        X, nr, nc = xray_at(t); STATE['nr'], STATE['nc'] = nr, nc
    if L is not None and X is not None:
        if t < SCAN1: yl = int(sm((t-SCAN0)/(SCAN1-SCAN0))*H2)
        else: yl = int((1-sm((t-BACK0)/(BACK1-BACK0)))*H2)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(H2)-yl)/(6*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(H2)-yl)/(60*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    im = tonemap(img)
    im = cv2.resize(im, (K.W, K.H), interpolation=cv2.INTER_AREA)
    im = np.clip(im*VIG + GRAIN[int(t*FPS) % 6]*(0.012 if L is not None else 0.005), 0, 1)
    pim = Image.fromarray((im*255).astype(np.uint8)); text(t, pim); return pim

def music_events():
    return [(R['t0'], [float(np.clip(R['b']/10, -1, 1))], 1) for R in RAYS]
def load():
    global RAYS; RAYS = mk_rays(); return build_real(), star_map(), disk_tex()

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    R, STAR, TEX = load()
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')]
        for t in ts: frame(t, R, STAR, TEX).save(f'f_{t:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        if mode == 'renderw':
            import json; Wj = json.load(open('warp.json')); tmap = lambda tau: float(np.interp(tau, Wj['anch_t'], Wj['anch_s']))
        else: tmap = lambda tau: tau
        for i in range(a, b): p.stdin.write(frame(tmap(i/FPS), R, STAR, TEX).tobytes())
        p.stdin.close(); p.wait(); open(f'part_{a:05d}.done', 'w').write('ok')
