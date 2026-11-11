# SOURCE CODE #8 v2 - "WHY DON'T ATOMS COLLAPSE?" (Coded Cosmos, 11 Nov 2026)
# v2: 3D particle electron cloud (14k samples of the solver's |psi|^2), momentum streaks (the 'jiggle'),
# photon wavelets on the classical spiral, energy valley, crystal-lattice zoom-out. Timed to ElevenLabs lines.
import numpy as np, cv2, subprocess, sys, os, json
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import eigh_tridiagonal
import source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE

FPS = 30; S = 2; W, H = K.W, K.H; W2, H2 = W*S, H*S
NX, NY = 490, 760
RNG = np.random.default_rng(8)
HERE = os.path.dirname(os.path.abspath(__file__))
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F_EQ2 = ImageFont.truetype(DJ, 70); F_CHIP2 = ImageFont.truetype(DJ, 30); F_MONO2 = ImageFont.truetype(DJ, 30); F_CAPH = ImageFont.truetype(DJ, 56)
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
def lerp(a, b, x): return a + (b-a)*x
def col(c, m=1.0): return np.array(c, np.float32)/255*m

# ================= VO timeline =================
LINES = ['L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08', 'L09', 'L10', 'L11', 'L12']
EST = dict(L01=1.5, L02=2.3, L03=4.0, L04=1.6, L05=2.2, L06=2.2, L07=3.0, L08=2.3, L09=1.8, L10=2.3, L11=2.2, L12=2.8)
GAP = dict(L01=0.3, L02=0.15, L03=0.35, L04=0.3, L05=0.35, L06=0.4, L07=0.5, L08=0.4, L09=0.7, L10=0.6, L11=1.5, L12=0.0)
DUR = {}
for k in LINES:
    p = os.path.join(HERE, 'vo', k + '.mp3')
    if os.path.exists(p):
        DUR[k] = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]))
    else: DUR[k] = EST[k]
TL = {}; t = 0.1
for k in LINES: TL[k] = (t, t + DUR[k]); t += DUR[k] + GAP[k]
T_VO = TL['L12'][1]
NEXT0 = T_VO + 0.2; T = NEXT0 + 2.4                       # tease card, then loop
CRASH = TL['L03'][0] + DUR['L03']*0.62                    # on 'crash'
SCAN0, SCAN1 = TL['L04'][0] - 0.1, TL['L04'][0] + 1.3; CLOUD0 = TL['L05'][0] - 0.3
SQ0 = TL['L06'][0]; SQ1 = TL['L07'][0]; RELAX1 = TL['L08'][0]; PAY = TL['L09'][0]
EQ0, EQ1 = TL['L10'][0], TL['L10'][0] + 3.0
DIAL0 = TL['L11'][0] - 0.2; DIAL1 = TL['L11'][1] + 0.3; SNAP = DIAL1 + 0.5
ZOOM0 = TL['L12'][0] - 0.2; LOOP0 = T - 0.6
CAPS = [('L01', [[("why don't atoms", WHITE)], [('collapse', YELLOW), ('?', WHITE)]]),
        ('L02', [[('guess: it orbits', WHITE)], [('like a ', WHITE), ('planet', YELLOW)]]),
        ('L03', [[('nope. it leaks light...', WHITE)], [('crash in ', WHITE), ('0.000 000 000 016 s', YELLOW)]]),
        ('L04', [[('so what holds', WHITE)], [('atoms ', WHITE), ('open', YELLOW), ('?', WHITE)]]),
        ('L05', [[("it's not a dot.", WHITE)], [("it's a ", WHITE), ('cloud', YELLOW)]]),
        ('L06', [[('squeeze it and it', WHITE)], [('jiggles faster', YELLOW)]]),
        ('L07', [[('the real equation:', WHITE)], [('the jiggle ', WHITE), ('pushes out', YELLOW)]]),
        ('L08', [[('push out = pull in', WHITE)], [('at ', WHITE), ('0.053 nm', GREEN)]]),
        ('L09', [[("that's why atoms", WHITE)], [("don't collapse", GREEN)]]),
        ('L10', [[('size set by ', WHITE), ("Planck's", YELLOW)], [('constant ', YELLOW), ('ħ', YELLOW)]]),
        ('L11', [[('turn ', WHITE), ('ħ', AMBER), (' down...', WHITE)], [('the atom ', WHITE), ('collapses', RED)]]),
        ('L12', [[('every atom in you is', WHITE)], [('held open ', YELLOW), ('like this', WHITE)]])]

# ================= real solver: radial Schrodinger, atomic units =================
NR, RMAX = 1500, 30.0
rg = np.linspace(RMAX/NR, RMAX, NR); dr = rg[1]-rg[0]
def eig(h, k=400):
    d = h*h/dr**2 - 1/rg; e = np.full(NR-1, -0.5*h*h/dr**2)
    E, V = eigh_tridiagonal(d, e, select='i', select_range=(0, k-1)); return E, V/np.sqrt(dr)
E1, V1 = eig(1.0)
u0 = rg*np.exp(-rg/0.15); u0 /= np.sqrt(np.sum(u0**2)*dr)
C0 = V1.T @ u0 * dr; TAU_MAX = 16.0
def relax(tau):
    u = V1 @ (C0*np.exp(-(E1-E1[0])*tau)); u /= np.sqrt(np.sum(u**2)*dr); return u
def mean_r(u): return float(np.sum(u**2*rg)*dr)
HS = np.linspace(0.25, 1.0, 16); A_H = np.array([mean_r(eig(h, 2)[1][:, 0])/1.5 for h in HS])
if __name__ == '__main__':
    print('solver E1 = %.3f eV | relaxed <r> = %.3f (1.5) | a0(0.5) = %.4f (0.25)' % (E1[0]*27.211386, mean_r(relax(TAU_MAX)), np.interp(0.5, HS, A_H)))
# radial quantiles of the solver state -> particle radii (inverse CDF), so the cloud IS the solver output
def radii_from_u(u, q):
    P = np.cumsum(u**2)*dr; P /= P[-1]; return np.interp(q, P, rg)
NP = 20000
Q = RNG.uniform(0.001, 0.999, NP); DIRS = RNG.normal(size=(NP, 3)); DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)
R_UNIT = radii_from_u(V1[:, 0], Q)                          # ground state, a0 = 1 (solver)
REL_CACHE = {}
def sq_a(t): return lerp(1.0, 0.15, sm((t-SQ0-0.2)/1.4))
def radii_at(t):
    if t < SQ1: return R_UNIT*sq_a(t)
    if t < RELAX1:
        x = sm((t-SQ1)/(RELAX1-SQ1-0.3)); key = round(x, 3)
        if key not in REL_CACHE: REL_CACHE[key] = radii_from_u(relax(TAU_MAX*key**1.6), Q)
        return REL_CACHE[key]
    return R_UNIT*a_at(t)
def h_at(t):
    if t < DIAL0: return 1.0
    if t < SNAP: return lerp(1.0, 0.3, sm((t-DIAL0-0.2)/(DIAL1-DIAL0)))
    x = np.clip((t-SNAP)/0.35, 0, 1); return lerp(0.3, 1.0, x) + (0.07*np.sin((t-SNAP)*10)*np.exp(-(t-SNAP)*3) if x >= 1 else 0)
def a_at(t):
    if t < SQ1: return sq_a(t)
    if t < RELAX1: return mean_r(relax(TAU_MAX*sm((t-SQ1)/(RELAX1-SQ1-0.3))**1.6))/1.5
    h = h_at(t); return float(np.interp(h, HS, A_H)) if h < 1 else h*h

# ================= classical spiral (Larmor) in 3D, inclined orbit =================
TS = -4.0; R0 = 400.0; INC = np.radians(62)
tt = np.linspace(TS, CRASH, 8000); s_ = (tt-TS)/(CRASH-TS); rr_c = R0*np.clip(1-s_, 0, 1)**(1/3)
om = np.minimum(2.4*(np.maximum(rr_c, 1)/R0)**-1.5, 40.0)
th = np.concatenate([[0], np.cumsum(0.5*(om[1:]+om[:-1])*np.diff(tt))])
def epos3(t):
    r = np.interp(t, tt, rr_c); a = np.interp(t, tt, th)
    return np.array([r*np.cos(a), r*np.sin(a)*np.cos(INC), r*np.sin(a)*np.sin(INC)])

# ================= camera =================
FOC = 1500.0
def project(P, yaw, dist):
    c, s = np.cos(yaw), np.sin(yaw)
    x = P[..., 0]*c + P[..., 2]*s; z = -P[..., 0]*s + P[..., 2]*c; y = P[..., 1]
    k = FOC/(dist + z); return NX + x*k, NY + y*k, k, z

# ================= static layers (2x): nebula + stars =================
def make_bg():
    n = np.zeros((H2//16, W2//16, 3), np.float32)
    for sc, amp in ((3, 1.0), (6, 0.6), (12, 0.35)):
        z = cv2.GaussianBlur(RNG.normal(size=n.shape[:2]).astype(np.float32), (0, 0), sc); z /= np.abs(z).max()
        n += (np.clip(z, 0, 1)**1.6*amp)[..., None]*np.array([0.25, 0.32, 0.75], np.float32)*0.5
    z = cv2.GaussianBlur(RNG.normal(size=n.shape[:2]).astype(np.float32), (0, 0), 8); z /= np.abs(z).max()
    n += (np.clip(z, 0, 1)**2*0.5)[..., None]*np.array([0.55, 0.18, 0.6], np.float32)*0.4
    bg = cv2.resize(n, (W2 + 200*S, H2 + 200*S), interpolation=cv2.INTER_CUBIC)*0.42 + np.array([0.012, 0.02, 0.055], np.float32)
    yy, xx = np.mgrid[0:bg.shape[0], 0:bg.shape[1]].astype(np.float32)
    bg *= (1.15 - 0.6*np.clip(np.sqrt((xx/bg.shape[1]-0.47)**2 + (yy/bg.shape[0]-0.42)**2)/0.7, 0, 1))[..., None]
    for i in range(700):
        x, y = RNG.uniform(0, bg.shape[1]), RNG.uniform(0, bg.shape[0]); b = RNG.uniform(0.05, 0.5)**2*1.6
        cv2.circle(bg, (int(x), int(y)), 1 if b < 0.2 else 2, (b*0.85, b*0.92, b), -1, cv2.LINE_AA)
    return bg
BG = make_bg()
yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
RAD2 = np.sqrt((xx-NX*S)**2 + (yy-NY*S)**2); del yy, xx
def sphere_sprite(rpx, base):
    yy, xx = np.mgrid[-rpx:rpx+1, -rpx:rpx+1].astype(np.float32)/rpx
    m = (xx**2 + yy**2) <= 1; zz = np.sqrt(np.clip(1-xx**2-yy**2, 0, 1))
    L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)
    lam = np.clip(xx*L[0] + yy*L[1] + zz*L[2], 0, 1); spec = np.clip(xx*L[0] + yy*L[1] + zz*L[2], 0, 1)**30
    img = (0.15 + 0.95*lam)[..., None]*base + spec[..., None]*1.4
    return img*m[..., None]
PROTON = sphere_sprite(24*S, np.array([1.4, 0.75, 0.35], np.float32))
ELEC = sphere_sprite(17*S, np.array([0.6, 1.3, 1.7], np.float32))
def stamp(img, spr, x, y, a=1.0):
    h, w = spr.shape[:2]; x0, y0 = int(x - w//2), int(y - h//2)
    if x0 < 0 or y0 < 0 or x0+w > img.shape[1] or y0+h > img.shape[0]: return
    reg = img[y0:y0+h, x0:x0+w]; m = (spr.sum(2) > 0)[..., None]
    img[y0:y0+h, x0:x0+w] = np.where(m, reg*(1-a) + spr*a, reg)
def glow(img, k=0.6, sig=9):
    s = cv2.resize(img, (W2//8, H2//8), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(s, (0, 0), sig) + 0.45*cv2.GaussianBlur(s, (0, 0), sig*3.5)
    return img + k*cv2.resize(g, (W2, H2), interpolation=cv2.INTER_LINEAR)
# small cloud sprite for the lattice
def cloud_sprite(rpx):
    yy, xx = np.mgrid[-rpx:rpx+1, -rpx:rpx+1].astype(np.float32); r = np.sqrt(xx**2+yy**2)/(rpx/3)
    return (np.exp(-2*r)*1.2)[..., None]*col(CYAN)[None, None]
LAT = cloud_sprite(80*S)*1.5

# ================= frame =================
VEL = RNG.normal(size=(NP, 3))
def caption(d, lines, y=1490, alpha=1.0):
    lh = 70; y0 = y - (len(lines)-1)*lh/2
    for i, segs in enumerate(lines):
        fs = [F_CAPH if t_ == 'ħ' else K.F_CAP for t_, _ in segs]
        tw = sum(d.textlength(t_, font=f) for (t_, _), f in zip(segs, fs)); x = CX - tw/2
        for (t_, c), f in zip(segs, fs):
            d.text((x, y0 + i*lh), t_, font=f, fill=tuple(int(v*alpha) for v in c), anchor='lm', stroke_width=6, stroke_fill=(0, 0, 0)); x += d.textlength(t_, font=f)

def frame(t):
    yaw = 0.18*t
    # camera: dolly in during classical, out at the end (lattice)
    dist = 1500.0
    zoom_out = sm((t-ZOOM0)/2.6) if t >= ZOOM0 else 0.0
    ox = int(100*S + 40*S*np.sin(t*0.07)); oy = int(100*S + 30*S*np.cos(t*0.05))
    img = BG[oy:oy+H2, ox:ox+W2].copy()
    # ---------------- classical ----------------
    if t < CLOUD0 + 0.5:
        fade = 1 - sm((t-SCAN0)/(CLOUD0+0.5-SCAN0))
        if t < CRASH and fade > 0:
            ts_ = np.linspace(max(TS, t-4.0), t, 420)
            P = np.array([epos3(q) for q in ts_]); xs, ys, kk, zz = project(P, 0.25, dist)
            for i in range(len(ts_)-1):
                w = (i/len(ts_))**1.5; dep = np.clip(1.15 - zz[i]/900, 0.5, 1.4)
                c = (col(RED)*(0.45+1.5*w) + col(AMBER)*1.3*w**3)*dep*fade
                cv2.line(img, (int(xs[i]*S), int(ys[i]*S)), (int(xs[i+1]*S), int(ys[i+1]*S)), tuple(c.tolist()), int(3+11*w*kk[i]), cv2.LINE_AA)
            # photon wavelets: light leaking out (amber rings born at the electron every 0.22 s)
            for k in range(8):
                tb = (np.floor(t/0.22) - k)*0.22
                if tb < TS: continue
                age = t - tb; pb = epos3(tb); bx, by, _, _ = project(pb[None], 0.25, dist)
                rr = 30 + 300*age; aa = np.exp(-age*1.6)*0.9*fade
                cv2.circle(img, (int(bx[0]*S), int(by[0]*S)), int(rr*S), tuple((col(AMBER)*aa).tolist()), int((2+4*np.exp(-age*2))*S), cv2.LINE_AA)
            ex, ey, ek, ez = project(epos3(t)[None], 0.25, dist)
            stamp(img, ELEC, ex[0]*S, ey[0]*S, fade)
        stamp(img, PROTON, NX*S, NY*S, max(fade, 0.0) if t >= CLOUD0 - 0.2 else 1.0)
        if CRASH <= t < CRASH + 1.5:
            x = (t-CRASH)/1.5
            img += (np.exp(-RAD2/(70*S*(1+4*x)))*3.0*(1-x)**1.5)[..., None]*np.array([1.0, 0.45, 0.35], np.float32)
            for j, lag in enumerate((0, 0.12, 0.24)):
                xr = np.clip(x - lag, 0, 1)
                if xr > 0: cv2.circle(img, (NX*S, NY*S), int((20+640*xr)*S), tuple((col(RED)*1.8*(1-xr)).tolist()), int((10-3*j)*S*(1-xr)+2), cv2.LINE_AA)
    # ---------------- quantum cloud (particles) ----------------
    if t >= CLOUD0:
        ap = sm((t-CLOUD0)/0.7)
        a = a_at(t); R = radii_at(t)
        scale = 190.0*(1 - 0.82*zoom_out)                                  # px per bohr at z=0
        hot = sm((t-SQ0)/1.0) if t < SQ1 + 0.8 else 0.0
        if t >= SQ1: hot *= 1 - sm((t-SQ1)/1.2)
        coll = sm((1 - a)/0.75) if DIAL0 <= t < SNAP + 0.6 else 0.0
        green = (sm((t-RELAX1)/0.5)*(1-sm((t-DIAL0)/0.5)) if t < DIAL0 + 0.5 else 0) + (sm((t-SNAP)/0.3)*(1-sm((t-SNAP-1.5)/0.8)) if t >= SNAP else 0)
        cbase = col(CYAN)*(1-hot*0.7-coll*0.8) + col(AMBER)*hot*0.9 + col(RED)*coll*1.1
        # jiggle: random walk speed ~ momentum ~ 1/a (uncertainty), particles re-sampled faster when squeezed
        jig = np.clip(0.15/max(a, 0.06), 0.1, 1.5)
        nre = int(NP*min(0.5, 0.04*jig**2 + 0.01)); idx = RNG.choice(NP, nre, replace=False)
        DIRS[idx] = RNG.normal(size=(nre, 3)); DIRS[idx] /= np.linalg.norm(DIRS[idx], axis=1, keepdims=True)
        VEL[idx] = RNG.normal(size=(nre, 3))
        P = DIRS*R[:, None]*scale
        xs, ys, kk, zz = project(P, yaw, dist)
        dep = np.clip(1.25 - zz/(3*scale*max(a, 0.2)+1)*0.5, 0.35, 1.6)
        ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
        X = (xs[ok]*S).astype(int); Y = (ys[ok]*S).astype(int); B = dep[ok]
        acc = np.zeros((H2, W2), np.float32)
        np.add.at(acc, (Y, X), B); np.add.at(acc, (Y+1, X), B*0.6); np.add.at(acc, (Y, X+1), B*0.6)
        dens = 0.55 + 0.45/max(a, 0.15)**0.4
        img += (acc*0.42*dens*ap)[..., None]*cbase[None, None]
        # momentum streaks (subset)
        ns = int(2500*min(1, 0.25 + 0.6*jig))
        L = 6 + 46*jig*(1 + 2*coll)
        for i in range(ns):
            j = i*5 % NP
            if not ok[j]: continue
            x0, y0 = xs[j]*S, ys[j]*S; v = VEL[j]
            cv2.line(img, (int(x0), int(y0)), (int(x0 + v[0]*L*S*0.5), int(y0 + v[1]*L*S*0.5)), tuple((cbase*0.22*dep[j]*ap).tolist()), 1*S, cv2.LINE_AA)
        # soft volumetric core
        core = np.exp(-RAD2/(S*scale*max(a, 0.05)*1.1))*(0.55 + 0.5*hot + 0.8*coll)
        img += (core*ap)[..., None]*cbase[None, None]
        stamp(img, PROTON if a > 0.2 else sphere_sprite(10*S, np.array([1.4, 0.75, 0.35], np.float32)), NX*S, NY*S)
        # squeeze: amber clamp ring closing in
        if SQ0 - 0.3 <= t < SQ1 + 0.5:
            x = sm((t-SQ0+0.3)/1.0); rr = lerp(460, 0.15*2.2*190 + 22, x); fa = 1 - sm((t-SQ1)/0.5)
            for k in range(3):
                cv2.circle(img, (NX*S, NY*S), int((rr + k*14)*S), tuple((col(AMBER)*(1.4 - 0.4*k)*fa).tolist()), int((4-k)*S), cv2.LINE_AA)
        # balance shell (green dashed sphere outline at <r>)
        if green > 0.01:
            rr = 1.5*a*scale*FOC/dist*S
            for k in range(36):
                if k % 2: continue
                cv2.ellipse(img, (NX*S, NY*S), (int(rr), int(rr)), 0, k*10 + t*20, k*10 + 6 + t*20, tuple((col(GREEN)*1.5*green).tolist()), 3*S, cv2.LINE_AA)
        if SNAP <= t < SNAP + 1.2:                                          # snap-back shock
            x = (t-SNAP)/1.2
            cv2.circle(img, (NX*S, NY*S), int((60 + 560*x)*S), tuple((col(GREEN)*1.6*(1-x)).tolist()), int(8*S*(1-x)+2), cv2.LINE_AA)
        if coll > 0.3:
            img += (np.exp(-RAD2/(30*S))*2.5*coll*(0.8 + 0.2*np.sin(t*40)))[..., None]*col(RED)[None, None]
        # lattice zoom-out: neighbouring atoms appear
        if zoom_out > 0:
            sp = 380*(1-0.5*zoom_out)
            for ix in range(-4, 5):
                for iy in range(-6, 7):
                    if ix == 0 and iy == 0: continue
                    px, py = NX + ix*sp + (iy % 2)*sp*0.5, NY + iy*sp*0.87
                    if -100 < px < W + 100 and -100 < py < H + 100:
                        s_ = cv2.resize(LAT, None, fx=0.5 + 0.3*zoom_out, fy=0.5 + 0.3*zoom_out)
                        hh, ww = s_.shape[:2]; x0, y0 = int(px*S - ww//2), int(py*S - hh//2)
                        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W2, x0+ww), min(H2, y0+hh)
                        if xb > xa and yb > ya: img[ya:yb, xa:xb] += s_[ya-y0:yb-y0, xa-x0:xb-x0]*zoom_out*(0.8 + 0.2*np.sin(t*3 + ix + iy))
    # ---------------- x-ray scan ----------------
    if SCAN0 <= t < CLOUD0 + 0.6:
        yline = lerp(150, 1650, sm((t-SCAN0)/(SCAN1-SCAN0)))*S; gfa = 1 - sm((t-CLOUD0)/0.6)
        g = np.zeros(img.shape[:2], np.float32)
        for x in range(0, W2, 60*S): g[:, x:x+2] = 0.12
        for y in range(0, H2, 60*S): g[y:y+2, :] = 0.12
        g[int(yline):] = 0; img += g[..., None]*col(CYAN)*gfa
        if t < SCAN1: img[max(0, int(yline)-5*S):int(yline)+5*S] += col(CYAN)*1.5
    img = glow(img)
    # ---------------- energy valley ----------------
    if SQ0 <= t < ZOOM0:
        pa = sm((t-SQ0)/0.6)*(1 - sm((t-ZOOM0+0.6)/0.6))
        x0, x1, yb = 120, 870, 1345
        hcur = h_at(t); A = np.linspace(0.06, 3.0, 300)
        def mp(av, ev): return (x0 + (av/3.0)*(x1-x0))*S, (yb - 120 - np.clip(ev, -1.25, 1.1)*95)*S
        def curve(E, c, w=3, fill=False):
            Pt = np.array([mp(av, ev) for av, ev in zip(A, E)]).astype(np.int32)
            if fill:
                poly = np.vstack([Pt, [[x1*S, yb*S], [x0*S, yb*S]]]); ov = np.zeros_like(img); cv2.fillPoly(ov, [poly], tuple((col(c)*0.18*pa).tolist())); img[:] += ov
            cv2.polylines(img, [Pt], False, tuple((col(c)*1.5*pa).tolist()), w*S, cv2.LINE_AA)
        curve(hcur**2/(2*A**2), AMBER, 2); curve(-1/A, CYAN, 2)
        if t >= SQ1 - 0.4: curve(hcur**2/(2*A**2) - 1/A, GREEN if not (DIAL0 <= t < SNAP) else RED, 4, fill=True)
        a = a_at(t); bx, by = mp(min(a, 3), hcur**2/(2*a*a) - 1/a)
        cv2.circle(img, (int(bx), int(by)), 12*S, (1.8*pa, 1.8*pa, 1.8*pa), -1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), 20*S, tuple((col(CYAN)*1.5*pa).tolist()), 3*S, cv2.LINE_AA)
    img = np.clip(img, 0, None); img = img/(1+0.22*img)*1.22
    out = cv2.resize(np.clip(img*255, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LANCZOS4)
    # ---------------- text (1x) ----------------
    im = Image.fromarray(out); d = ImageDraw.Draw(im, 'RGBA')
    K.series_tag(d, 8)
    if t < TL['L02'][0] + 0.3 or t >= LOOP0: K.title_box(d, "WHY DON'T ATOMS COLLAPSE?", y=270)
    if t < CRASH:
        ex, ey, _, _ = project(epos3(t)[None], 0.25, 1500.0); ex, ey = ex[0], ey[0]
        d.ellipse([ex-36, ey-36, ex+36, ey+36], outline=CYAN+(255,), width=5)
        K.pill(d, (ex + (100 if ex < 650 else -100), ey - 56), 'electron', CYAN)
    if t < SCAN0: K.pill(d, (NX, NY + 66), 'nucleus', AMBER)
    if TL['L03'][0] <= t < CRASH: K.pill(d, (NX, 470), 'light leaks out', AMBER)
    if CLOUD0 + 0.3 <= t < SQ0: K.pill(d, (NX, NY - 260), 'electron cloud', CYAN)
    if SQ0 <= t < SQ1: K.pill(d, (NX + 230, NY - 300), 'squeeze', AMBER)
    if SCAN0 + 0.5 <= t < CLOUD0 + 0.3: d.text((CX, 560), 'E = ħ²/2ma² − ke²/a', font=F_MONO2, fill=CYAN+(230,), anchor='mm')
    if SQ0 + 0.3 <= t < ZOOM0 - 0.6:
        d.text((120, 1100), 'energy vs cloud size', font=K.F_MONO, fill=(200, 210, 230, 220), anchor='lt')
        K.pill(d, (300, 1165), 'jiggle', AMBER, font=K.F_MONO); K.pill(d, (720, 1330), 'pull', CYAN, font=K.F_MONO)
        if t >= SQ1 - 0.4: K.pill(d, (720, 1150), 'total', GREEN, font=K.F_MONO)
    if (RELAX1 <= t < EQ0 - 0.1) or (EQ1 <= t < DIAL0) or SNAP + 0.3 <= t < ZOOM0: K.pill(d, (NX, NY - 330), '0.053 nm', GREEN)
    if SQ1 <= t < RELAX1: K.pill(d, (NX, 470), 'solver running', CYAN, font=K.F_MONO)
    if EQ0 <= t < EQ1:
        fa = int(255*sm((t-EQ0)/0.3)*(1-sm((t-EQ1+0.3)/0.3)))
        d.rounded_rectangle([140, 250, 840, 410], 24, fill=(6, 10, 24, int(fa*0.85)), outline=YELLOW+(fa,), width=4)
        d.text((CX, 330), 'a₀ = ħ² / m k e²', font=F_EQ2, fill=YELLOW+(fa,), anchor='mm')
    if t >= EQ1 - 0.2 and t < ZOOM0:
        d.rounded_rectangle([60, 196, 400, 248], 14, fill=(6, 10, 24, 220), outline=YELLOW+(200,), width=2)
        d.text((230, 222), 'a₀ = ħ²/(m k e²)', font=F_CHIP2, fill=YELLOW, anchor='mm')
    if DIAL0 <= t < ZOOM0:
        h = h_at(t); cx_, cy_, R_ = CX, 380, 92
        cc = RED if h < 0.75 else (GREEN if t >= SNAP else AMBER)
        d.arc([cx_-R_, cy_-R_, cx_+R_, cy_+R_], 150, 390, fill=(70, 80, 100, 255), width=12)
        ang = 150 + 240*np.clip((h-0.25)/0.85, 0, 1)
        d.arc([cx_-R_, cy_-R_, cx_+R_, cy_+R_], 150, ang, fill=cc+(255,), width=12)
        rad = np.radians(ang); d.line([cx_, cy_, cx_+0.8*R_*np.cos(rad), cy_+0.8*R_*np.sin(rad)], fill=WHITE, width=6)
        d.text((cx_, cy_+42), 'ħ x%.2f' % h, font=F_CHIP2, fill=cc, anchor='mm')
    for k, lines in CAPS:
        a0_, b0_ = TL[k][0] - 0.05, TL[k][1] + min(GAP[k], 0.4)
        if a0_ <= t < b0_:
            al = sm((t-a0_)/0.12)
            d.rounded_rectangle([60, 1440 - (len(lines)-1)*35, 920, 1540 + (len(lines)-1)*35], 22, fill=(0, 0, 0, int(120*al)))
            caption(d, lines, y=1490, alpha=al)
    if t >= ZOOM0 + 1.0:
        K.end_stamp(d, sm((t-ZOOM0-1.0)/0.4), y=1330)
    if NEXT0 <= t < LOOP0:
        fa = sm((t-NEXT0)/0.3)
        d.rounded_rectangle([110, 1430, 870, 1550], 22, fill=(0, 0, 0, int(150*fa)), outline=YELLOW+(int(255*fa),), width=3)
        d.text((CX, 1468), 'NEXT · SOURCE CODE #9', font=K.F_PILL, fill=YELLOW+(int(255*fa),), anchor='mm')
        d.text((CX, 1515), 'why should the Sun not shine?', font=K.F_PILL, fill=WHITE+(int(255*fa),), anchor='mm')
    return np.array(im)

if __name__ == '__main__':
    mode = sys.argv[1]
    print('T = %.2f s, crash %.2f, payoff %.2f' % (T, CRASH, PAY)); json.dump(TL, open('tl8b.json', 'w'))
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')]
        fr = [cv2.resize(frame(q), (360, 640), interpolation=cv2.INTER_AREA) for q in ts]
        for f in fr: cv2.rectangle(f, (13, 53), (310, 530), (0, 255, 0), 1); cv2.rectangle(f, (316, 316), (359, 566), (255, 0, 0), 1)
        Image.fromarray(np.hstack(fr)).save(sys.argv[3])
    elif mode == 'one':
        Image.fromarray(frame(float(sys.argv[2]))).save(sys.argv[3])
    else:
        F0 = frame(0.0)
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', 'video8b.mp4'], stdin=subprocess.PIPE)
        for i in range(int(T*FPS)):
            t = i/FPS; f = frame(t)
            if t >= LOOP0: x = sm((t-LOOP0)/(T-LOOP0)); f = (f*(1-x) + F0*x).astype(np.uint8)
            p.stdin.write(f.tobytes())
            if i % 150 == 0: print(i, flush=True)
        p.stdin.close(); p.wait(); print('done')
