# SOURCE CODE #1 - "YOU'VE DRAWN ATOMS WRONG" (Coded Cosmos, 3 Nov 2026)
# real thing: hydrogen discharge lamp -> x-ray -> 1 atom -> real radial Schrodinger solver (sim.py)
import numpy as np, cv2, subprocess, sys
from PIL import Image, ImageDraw
import sim, source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE

FPS, BPM = 30, 96; BT = 60/BPM; BEATS = 71; T = BEATS*BT            # 44.375 s
S = 2; W2, H2 = K.W*S, K.H*S
RNG = np.random.default_rng(3)
# ---- timeline (s) ----
SCAN0, SCAN1 = 11.25, 12.15; DOT0 = 15.0; TROPHY = 39*BT              # 24.375
N2, BREAK, SNAP, PHOT, BACK0, BACK1 = 46*BT, 50*BT, 54*BT, 58*BT, 62*BT, 62*BT+0.8
CAPS = [(2.4, 5.2, [[('everyone draws atoms like', WHITE)], [('tiny planets', YELLOW), ('. easy, right?', WHITE)]]),
        (5.2, 8.6, [[('nope. ', YELLOW), ('that atom would crash', WHITE)], [('in ', WHITE), ('0.000 000 000 016 s', YELLOW)]]),
        (8.6, 11.25, [[("so what's really inside", WHITE)], [('this ', WHITE), ('pink glow', YELLOW), ('?', WHITE)]]),
        (12.2, 15.0, [[('1 electron. we solve its', WHITE)], [('real equation', YELLOW), ('. no planets.', WHITE)]]),
        (15.0, 18.1, [[('each dot = one place', WHITE)], [('we ', WHITE), ('caught', YELLOW), (' the electron', WHITE)]]),
        (18.1, 21.2, [[('faster...', WHITE)], [('no orbit. ', WHITE), ("it's a cloud", YELLOW)]]),
        (21.2, TROPHY, [[('no path. just the', WHITE)], [('odds', YELLOW), (" of where it'll be", WHITE)]]),
        (TROPHY, N2, [[('n can only be a', WHITE)], [('whole number', YELLOW)]]),
        (N2, BREAK, [[('n = 2', AMBER), (' -> a new shape.', WHITE)], [('same equation.', WHITE)]]),
        (BREAK, SNAP, [[('n = 2.5', RED), ('? no shape fits.', WHITE)], [('the maths ', WHITE), ('blows up', YELLOW)]]),
        (SNAP, PHOT, [[('n = 3', GREEN), (' works again.', WHITE)], [('only whole steps: ', WHITE), ('quantum', YELLOW)]]),
        (PHOT, BACK0, [[('fall from 3 to 2 and it', WHITE)], [('spits out ', WHITE), ('656 nm', YELLOW), (' light...', WHITE)]]),
        (BACK0, 41.6, [[('...the red in', WHITE)], [('this exact ', WHITE), ('pink glow', YELLOW)]]),
        (41.6, T-0.45, [[('same jump paints the', WHITE)], [('pink nebulae', YELLOW), (' in space', WHITE)]])]
SLOW_TXT = 'slowed ~140 billion x'
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)

# ================= real-life layer: hydrogen discharge tube (procedural, 2x) =================
def gauss_x(xx, x0, s): return np.exp(-0.5*((xx-x0)/s)**2)
def build_lamp():
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    # background: dark lab, soft gradient + out-of-focus lab lights (bokeh), slightly larger for drift
    bg = np.zeros((H2, W2, 3), np.float32)
    g = 0.035 + 0.03*np.exp(-((xx-W2*0.45)**2 + (yy-H2*0.42)**2)/(2*(900*S)**2))
    bg += g[..., None]*np.array([0.55, 0.62, 0.85], np.float32)
    bok = np.zeros_like(bg)
    for (x, y, r, c, a) in [(150, 700, 70, (1, .62, .25), .22), (880, 560, 95, (.35, .75, 1), .16), (960, 1050, 60, (1, .75, .4), .14),
                            (90, 1120, 110, (.4, .5, 1), .10), (760, 300, 55, (1, .9, .7), .12), (240, 420, 45, (.3, .9, .9), .10),
                            (690, 1160, 80, (.9, .35, .7), .08)]:
        cv2.circle(bok, (x*S, y*S), r*S, (c[0]*a, c[1]*a, c[2]*a), -1, cv2.LINE_AA)
    bok = cv2.GaussianBlur(bok, (0, 0), 9*S)
    bg += bok
    # table surface below y=1330 with sheen
    tab = (yy > 1330*S).astype(np.float32)
    bg = bg*(1-tab[..., None]*0.5) + tab[..., None]*np.array([0.03, 0.03, 0.04], np.float32)
    bg += (np.exp(-((yy-1332*S)/(2.5*S))**2)*0.10)[..., None]*np.array([0.9, 0.8, 0.9], np.float32)
    # ---- tube emission (pink-magenta: H-alpha red + H-beta/gamma blue) ----
    em = np.zeros((H2, W2, 3), np.float32); x0 = CX*S
    cap = ((yy > 650*S) & (yy < 1180*S)).astype(np.float32)
    ny = cv2.GaussianBlur(RNG.normal(0, 1, (H2, 1)).astype(np.float32), (0, 0), 40*S)[:, 0]
    var = (1 + 0.6*ny/np.abs(ny).max())[:, None]
    core = gauss_x(xx, x0, 3.6*S)*cap*var; halo = gauss_x(xx, x0, 10*S)*cap*var
    em += core[..., None]*np.array([3.2, 2.2, 3.0], np.float32)          # overexposed white-pink core
    em += halo[..., None]*np.array([1.25, 0.28, 0.85], np.float32)       # magenta halo
    for (yc, ry) in ((595, 52), (1235, 52)):                              # bulbs: diffuse glow
        neck = 650 if yc < 900 else 1180                                   # glow pools at the capillary neck
        b = np.exp(-(((xx-x0)/(40*S))**2 + ((yy-yc*S)/(ry*S))**2)*2.2)*(0.35 + 0.65*np.exp(-np.abs(yy-neck*S)/(45*S)))
        em += b[..., None]*np.array([0.55, 0.13, 0.44], np.float32)
        em += (np.exp(-(((xx-x0)/(9*S))**2 + ((yy-neck*S)/(14*S))**2))*1.2)[..., None]*np.array([1.4, 0.6, 1.2], np.float32)
    em = cv2.GaussianBlur(em, (0, 0), 0.8*S)
    # ---- glass + metal (lit by the tube) ----
    gl = np.zeros_like(em); w = (0.20, 0.20, 0.24)
    for sx in (-14, 14):
        cv2.line(gl, (x0+sx*S, 650*S), (x0+sx*S, 1180*S), w, 2*S, cv2.LINE_AA)
    cv2.line(gl, (x0-8*S, 660*S), (x0-8*S, 1170*S), (0.45, 0.42, 0.5), 1*S, cv2.LINE_AA)    # specular streak
    for yc in (595, 1235):
        cv2.ellipse(gl, (x0, yc*S), (56*S, 58*S), 0, 0, 360, (0.24, 0.22, 0.28), 2*S, cv2.LINE_AA)
        cv2.ellipse(gl, (x0, yc*S), (46*S, 48*S), 0, 200, 250, (0.6, 0.55, 0.65), 2*S, cv2.LINE_AA)
        cv2.ellipse(gl, (x0, yc*S), (52*S, 54*S), 0, 20, 70, (0.18, 0.16, 0.2), 1*S, cv2.LINE_AA)

    gl = cv2.GaussianBlur(gl, (0, 0), 0.7*S)
    met = np.zeros_like(em)
    for (ya, yb) in ((468, 540), (1290, 1330)):
        for k in range(36):
            u = (k-17.5)/17.5; v = 0.10 + 0.32*np.exp(-((u+0.35)/0.18)**2) + 0.07*(1-u*u)
            cv2.line(met, (x0+int(u*34*S), ya*S), (x0+int(u*34*S), yb*S), (v, v*0.96, v*1.02), 2*S, cv2.LINE_AA)
    met += (np.exp(-((xx-x0)/(30*S))**2)*0.06)[..., None]*np.array([1, .3, .7], np.float32)*(met.sum(2, keepdims=True) > 0)
    fg = em + gl + met
    # reflection on the table (physical: the glowing tube mirrored in a glossy dark surface, dim + blurred)
    ref = np.zeros_like(fg); src = fg[int(650*S):int(1330*S)][::-1]
    h = min(src.shape[0], H2-1334*S); ref[1334*S:1334*S+h] = src[:h]
    fade = np.exp(-np.clip(yy-1334*S, 0, None)/(140*S))[..., None]*(yy > 1334*S)[..., None]
    ref = cv2.GaussianBlur(ref, (0, 0), 5*S)*fade*0.22
    # bloom (computed at 1x full frame; smooth low-frequency light, then resized)
    e1 = cv2.resize(em, (K.W, K.H), interpolation=cv2.INTER_AREA)
    bloom = cv2.GaussianBlur(e1, (0, 0), 14)*0.55 + cv2.GaussianBlur(e1, (0, 0), 60)*0.9
    bloom = cv2.resize(bloom, (W2, H2), interpolation=cv2.INTER_CUBIC)
    return bg, fg + ref + bloom
def warp(img, s, dx, dy):
    M = np.float32([[s, 0, (1-s)*CX*S + dx*S], [0, s, (1-s)*960*S + dy*S]])
    return cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def lamp_cam(t):
    if t < SCAN1: s = 1 + 0.05*sm(t/SCAN1)
    else: s = 1.04 - 0.04*sm((t-BACK1)/(T-0.5-BACK1))
    return 1.18*s, 3*np.sin(t*0.4), 2*np.sin(t*0.31)
def lamp_at(t, BG, FG):
    s, dx, dy = lamp_cam(t)
    fl = 1 + 0.03*np.sin(t*31) + 0.02*np.sin(t*7.3)
    return warp(BG, 1+(s-1)*0.4, dx*0.4, dy*0.4) + warp(FG, s, dx, dy)*fl
GRAIN = [RNG.normal(0, 1, (K.H, K.W, 1)).astype(np.float32) for _ in range(6)]
def tonemap(x): return 1-np.exp(-1.15*x)

# ================= x-ray layer: the maths skeleton =================
N_MAX = 14000
SETS = {}
for (n, l) in ((1, 0), (2, 1), (3, 2)):
    pts, E, r, P = sim.sample(n, l, N_MAX, seed=n)
    rad = np.linalg.norm(pts, axis=1); SETS[n] = dict(pts=pts/np.percentile(rad, 93), E=E, r=r, P=P)
r_sh, u_sh = sim.shoot(1, -13.6057/2.5**2)
E1, E2, E3 = SETS[1]['E'], SETS[2]['E'], SETS[3]['E']
LAMBDA = 1239.84/(E3-E2)                                                    # nm, from the solver's energies
GRID = np.zeros((H2, W2, 3), np.float32)
for x in range(0, W2, 80*S): cv2.line(GRID, (x, 0), (x, H2), (0.10, 0.28, 0.36), 1, cv2.LINE_AA)
for y in range(0, H2, 80*S): cv2.line(GRID, (0, y), (W2, y), (0.10, 0.28, 0.36), 1, cv2.LINE_AA)
GRID *= 0.35
NIGHTF = (K.NIGHT/255).astype(np.float32)
ACX, ACY, RPX = CX, 840, 300                                              # atom centre / cloud radius (1x px)
def project(p, t):
    a, b = 0.32*t, 0.38
    x = p[:, 0]*np.cos(a) + p[:, 2]*np.sin(a); z = -p[:, 0]*np.sin(a) + p[:, 2]*np.cos(a)
    y = p[:, 1]*np.cos(b) - z*np.sin(b)
    return np.stack([ACX + x*RPX, ACY - y*RPX], 1)
def n_dots(t):
    if t < DOT0: return 0
    if t < 17.6: return 1 + int((t-DOT0)/0.6)
    if t < 18.1: return 5
    if t < 19.6: return 5 + int((t-18.1)*25)
    return int(min(N_MAX, 42*np.exp((t-19.6)*2.2)))
def cloud_state(t):
    """returns (positions px, colour, count, flash)"""
    if t < N2: P = project(SETS[1]['pts'], t); col = np.array(CYAN)/255; cnt = n_dots(t) if t < TROPHY else N_MAX
    elif t < BREAK:
        u = sm((t-N2)/1.0); P = (1-u)*project(SETS[1]['pts'], t) + u*project(SETS[2]['pts'], t); col = np.array(CYAN)/255; cnt = N_MAX
    elif t < SNAP:                                                        # n = 2.5: no solution -> blows outward, red
        u = (t-BREAK); P0 = project(SETS[2]['pts'], t); d = P0 - [ACX, ACY]
        k = (np.exp(u*1.2)-1)*0.6; jit = RNG.normal(0, 1, P0.shape)*k*6
        P = P0 + d*k*0.9 + jit; c = sm(u/0.4); col = (1-c)*np.array(CYAN)/255 + c*np.array(RED)/255; cnt = N_MAX
    elif t < PHOT:                                                        # n = 3 snaps in, green then cyan
        u = sm((t-SNAP)/0.5); P = project(SETS[3]['pts'], t)*1.0
        P = [ACX, ACY] + (P-[ACX, ACY])*u; c = sm((t-SNAP-0.7)/1.0)
        col = (1-c)*np.array(GREEN)/255 + c*np.array(CYAN)/255; cnt = N_MAX
    else:                                                                 # 3 -> 2: emits the photon
        u = sm((t-PHOT)/1.2); P = (1-u)*project(SETS[3]['pts'], t) + u*project(SETS[2]['pts'], t); col = np.array(CYAN)/255; cnt = N_MAX
    return np.asarray(P), col, cnt
def splat(P, cnt, col, t):
    img = np.zeros((H2, W2), np.float32)
    q = (P[:cnt]*S).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < W2) & (q[:, 1] >= 0) & (q[:, 1] < H2); q = q[ok]
    w = 1.0 if cnt < 200 else max(0.30, 4.5/np.sqrt(cnt/40))
    np.add.at(img, (q[:, 1], q[:, 0]), w)
    sig = 2.2 if cnt < 50 else 1.3
    img = cv2.GaussianBlur(img, (0, 0), sig*S)*(2*np.pi*(sig*S)**2)*0.5
    if cnt < 60:                                                          # newest dot flashes
        for j in range(max(0, cnt-1), cnt):
            age = t - (DOT0 + 0.6*j if j < 5 else 18.1 + (j-5)/25)
            if 0 <= age < 0.5: cv2.circle(img, tuple(int(v) for v in P[j]*S), int((14+40*age)*S), 1.0*(1-age/0.5), 2*S, cv2.LINE_AA)
    out = img[..., None]*col[None, None, :]
    small = cv2.resize(out, (K.W, K.H), interpolation=cv2.INTER_AREA)
    out += cv2.resize(cv2.GaussianBlur(small, (0, 0), 10), (W2, H2))*0.6
    return out
def xray_at(t):
    X = GRID + NIGHTF[None, None, :]
    if t < 13.0:                                                          # skeleton of the tube, then zoom into 1 atom
        a = 1 - sm((t-12.4)/0.6)
        sk = np.zeros_like(X); x0 = CX*S; c = tuple(np.array(CYAN)/255*0.9)
        for sx in (-14, 14): cv2.line(sk, (x0+sx*S, 650*S), (x0+sx*S, 1180*S), c, 2*S, cv2.LINE_AA)
        for yc in (595, 1235): cv2.ellipse(sk, (x0, yc*S), (56*S, 58*S), 0, 0, 360, c, 2*S, cv2.LINE_AA)
        for (ya, yb) in ((468, 540), (1290, 1330)): cv2.rectangle(sk, (x0-34*S, ya*S), (x0+34*S, yb*S), c, 2*S, cv2.LINE_AA)
        for k in range(14): cv2.circle(sk, (x0, (670+k*36)*S), 3*S, c, -1, cv2.LINE_AA)    # the atoms in the capillary
        sc, dx, dy = lamp_cam(SCAN1); X += warp(sk, sc, dx, dy)*a
        rr = int((10 + 290*sm((t-12.2)/0.8))*S)
        cv2.circle(X, (ACX*S, ACY*S), rr, tuple(np.array(CYAN)/255*0.7), 2*S, cv2.LINE_AA)
    if t >= 12.2:
        cv2.circle(X, (ACX*S, ACY*S), 6*S, (1.0, 1.0, 1.0), -1, cv2.LINE_AA)   # proton
        P, col, cnt = cloud_state(t)
        if cnt: X += splat(P, cnt, col, t)
    # radial panel (solver output): 12.6-TROPHY and during the break
    show = sm((t-12.6)/0.6)*(1-sm((t-TROPHY+0.4)/0.4)) + (sm((t-BREAK)/0.3)*(1-sm((t-SNAP)/0.3)))
    if show > 0.01:
        bx0, bx1, by0, by1 = 140, 840, 1200, 1320
        cv2.line(X, (bx0*S, by1*S), (bx1*S, by1*S), (0.3, 0.45, 0.55), 2, cv2.LINE_AA)
        if t < TROPHY:
            r, P_ = SETS[1]['r'], SETS[1]['P']; m = r < 8; rx, py = r[m]/8, P_[m]/P_[m].max()
            k = int(len(rx)*sm((t-12.6)/1.6)); col = tuple(np.array(CYAN)/255*show)
        else:
            rx = r_sh/r_sh[-1]; py = np.abs(u_sh)/np.abs(u_sh[:len(u_sh)//3]).max()*0.25
            k = int(len(rx)*sm((t-BREAK)/1.6)); col = tuple(np.array(RED)/255*show)
        pts = np.stack([(bx0 + rx[:k]*(bx1-bx0))*S, (by1 - np.clip(py[:k], 0, 1.6)*(by1-by0))*S], 1).astype(np.int32)
        if k > 1: cv2.polylines(X, [pts], False, col, 3*S, cv2.LINE_AA)
    if PHOT+0.3 < t < BACK0:                                              # 656 nm photon leaves the atom
        u = (t-PHOT-0.3)/1.8; xs = np.linspace(0, 1, 200)
        x = ACX + 60 + xs*min(1, u)*420; y = ACY - 40 - xs*min(1, u)*330 + 16*np.sin(xs*min(1, u)*420/14 - t*30)
        a = 1 - sm((t-BACK0+0.8)/0.8)
        cv2.polylines(X, [np.stack([x*S, y*S], 1).astype(np.int32)], False, (1.0*a, 0.08*a, 0.08*a), 3*S, cv2.LINE_AA)
    return X

# ================= guess layer: the planet atom everyone draws =================
def bohr(t, img):
    a = sm((t-2.5)/0.5)*(1-sm((t-8.4)/0.5))
    if a <= 0: return 0
    c0 = (0.95*a, 0.95*a, 0.95*a); cx, cy = CX*S, 880*S
    cv2.circle(img, (cx, cy), 12*S, (1.0*a, 0.75*a, 0.3*a), -1, cv2.LINE_AA)
    if t < 5.2:
        for k in range(3):
            ang = k*60; cv2.ellipse(img, (cx, cy), (270*S, 90*S), ang, 0, 360, c0, 2*S, cv2.LINE_AA)
            ph = t*2.2 + k*2.1; ex, ey = 270*np.cos(ph), 90*np.sin(ph); r_ = np.deg2rad(ang)
            p = (int(cx + (ex*np.cos(r_) - ey*np.sin(r_))*S), int(cy + (ex*np.sin(r_) + ey*np.cos(r_))*S))
            cv2.circle(img, p, 11*S, (0.6*a, 0.85*a, 1.0*a), -1, cv2.LINE_AA)
    else:                                                                 # classical: r^3 falls linearly (Larmor)
        tau = np.clip((t-5.4)/2.2, 0, 1); trail = []
        for s_ in np.linspace(max(0, tau-0.35), tau, 120):
            r = 270*(1-s_)**(1/3); ph = 2.0*np.cumsum([0])[0] + 14*(1-(1-s_)**(1/2))*np.pi*2
            trail.append((cx + r*np.cos(ph)*S, cy + r*np.sin(ph)*0.95*S))
        if tau < 1:
            cv2.polylines(img, [np.array(trail, np.int32)], False, (1.0*a, 0.28*a, 0.28*a), 3*S, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in trail[-1]), 11*S, (1.0*a, 0.3*a, 0.3*a), -1, cv2.LINE_AA)
        else:
            f = np.exp(-(t-7.6)*3); cv2.circle(img, (cx, cy), int((20+200*(1-f))*S), (1.0*f*a, 0.3*f*a, 0.3*f*a), 6*S, cv2.LINE_AA)
    return a

# ================= text layer (1x) =================
def text(t, im):
    d = ImageDraw.Draw(im)
    K.title_box(d, "YOU'VE DRAWN ATOMS WRONG", 'this pink glow knows why' if t < 11 or t > BACK0 else None)
    K.series_tag(d, 1)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 5.4 < t < 7.8: K.pill(d, (CX, 1290), SLOW_TXT, AMBER)
    if 12.6 < t < TROPHY-0.4: d.text((140, 1180), 'solver: chance to find it vs distance', font=K.F_MONO, fill=CYAN, anchor='ls')
    if 13.4 < t < TROPHY-0.4: d.text((140, 1360), f'solver: E1 = {E1:.2f} eV', font=K.F_MONO, fill=CYAN, anchor='ls')
    if 18.1 < t < 19.6: K.pill(d, (790, 520), 'x10', AMBER)
    elif 19.6 < t < 22.8: K.pill(d, (790, 520), 'x1000', AMBER)
    if BREAK+0.2 < t < SNAP: d.text((140, 1180), 'solver at n = 2.5: no fit, it diverges', font=K.F_MONO, fill=RED, anchor='ls')
    # dial (the ONE number) from the trophy to the lamp return
    if TROPHY <= t < BACK0:
        if t < N2: nv, c = '1', AMBER
        elif t < BREAK: nv, c = '2', AMBER
        elif t < SNAP: nv, c = '2.5', RED
        elif t < PHOT: nv, c = '3', GREEN
        else: nv, c = ('3' if t < PHOT+0.6 else '3 -> 2'), AMBER
        K.pill(d, (CX, 492), f'dial:  n = {nv}', c)
    # equation trophy: hold >= 3 s at the reveal hit, then a chip
    if TROPHY <= t < N2 - 0.3:
        a = sm((t-TROPHY)/0.35); y = 1255
        segs = [('E', CYAN, K.F_EQ, 0), ('n', AMBER, K.F_SUB, 22), (' = -13.6 eV / ', WHITE, K.F_EQ, 0), ('n', AMBER, K.F_EQ, 0), ('2', AMBER, K.F_SUB, -26)]
        tw = sum(d.textlength(s_, font=f) for s_, _, f, _ in segs); x = CX - tw/2
        for s_, c, f, dy in segs:
            d.text((x, y+dy), s_, font=f, fill=tuple(int(v*a) for v in c), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s_, font=f)
    elif N2 - 0.3 <= t < BACK0:
        nv = '2.5' if BREAK <= t < SNAP else ('1' if t < N2 else ('2' if t < BREAK or t > PHOT+0.6 else '3'))
        nn = float(nv); col = RED if BREAK <= t < SNAP else CYAN
        K.pill(d, (CX, 1395 if not (BREAK <= t < SNAP) else 1395), f'E = -13.6 / {nv}² = {-13.6057/nn**2:.2f} eV', col, font=K.F_CHIP)
    if PHOT+1.2 < t < BACK0: K.pill(d, (800, 560), f'{LAMBDA:.0f} nm', (255, 90, 90))
    K.end_stamp(d, alpha=sm((t-41.6)/0.4)*(1-sm((t-T+0.5)/0.3)), y=1385)

# ================= frame =================
def frame(t, BG, FG):
    L = lamp_at(t, BG, FG) if (t < SCAN1 or t > BACK0) else None
    if L is not None:
        g = bohr(t, ov := np.zeros_like(L)) if t < 9 else 0
        if g: L = L*(1-0.62*g) + ov
    X = xray_at(t) if SCAN0 <= t <= BACK1 else None
    if L is not None and X is not None:                                  # scanline wipe
        if t < SCAN1: yl = int(sm((t-SCAN0)/(SCAN1-SCAN0))*H2); top, bot = X, L
        else: yl = int((1-sm((t-BACK0)/(BACK1-BACK0)))*H2); top, bot = X, L
        img = bot.copy(); img[:yl] = top[:yl]
        band = np.exp(-((np.arange(H2)-yl)/(6*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        glow = np.exp(-((np.arange(H2)-yl)/(60*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
        img = img + band + glow
    else: img = L if L is not None else X
    im = tonemap(img)
    im = cv2.resize(im, (K.W, K.H), interpolation=cv2.INTER_AREA)            # 2x -> 1x (area = proper AA)
    vy, vx = np.mgrid[0:K.H, 0:K.W];
    im = np.clip(im*VIG + GRAIN[int(t*FPS) % 6]*0.006, 0, 1)
    pim = Image.fromarray((im*255).astype(np.uint8))
    text(t, pim); return pim
vy, vx = np.mgrid[0:K.H, 0:K.W].astype(np.float32)
VIG = (1 - 0.32*(((vx-K.W/2)/K.W)**2 + ((vy-K.H/2)/K.H)**2)*2)[..., None]

def music_events():
    ev = []; prev = 0
    for i in range(int(T*FPS)):
        t = i/FPS; c = n_dots(t) if t < TROPHY else prev
        if c > prev:
            us = list(np.clip((SETS[1]['pts'][prev:min(c, prev+3), 0]), -1, 1)); ev.append((t, us, c-prev))
        prev = c
    return ev

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    BG, FG = build_lamp()
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.5, 2, 6.5, 11.7, 16, 21, 25.5, 30, 32.5, 35, 37.5, 43]
        for t in ts: frame(t, BG, FG).save(f'f_{t:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS, BG, FG).tobytes())
        p.stdin.close(); p.wait()
