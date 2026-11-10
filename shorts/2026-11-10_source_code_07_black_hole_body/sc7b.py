# SOURCE CODE #7 v2 - "WHAT WOULD A BLACK HOLE DO TO YOUR BODY?" (Coded Cosmos, 10 Nov 2026)
# real layer: ray-traced Schwarzschild hole (SC#3 bh.py) + SDF astronaut (sprite.py) relit, stretched by sim7 solver
# x-ray: hologram body + tidal field (stretch along the fall, squeeze across) + solver pills; dial = black-hole mass
import numpy as np, cv2, subprocess, sys, json, os
from PIL import Image, ImageDraw
import source_code_kit as K, sim7, astro
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE
import bh as A

FPS = 30; S = 2; W2, H2 = K.W*S, K.H*S
BG = 'night'
CAM = dict(rcy=1120, scr=60, body_y=650, body_px=560)
A.SCAN1 = 1e6; A.BACK1 = 1e6
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)

# ---------------- timeline from the voice clips ----------------
DUR = json.load(open('vo/durs.json')) if os.path.exists('vo/durs.json') else {}
g = lambda k, d: DUR.get(k, d)
TL = {}; t = 0.0
TL['L01'] = (0.0, g('L01', 2.6)); e = TL['L01'][1]
TL['L02'] = (e+0.2, e+0.2+g('L02', 2.4)); e = TL['L02'][1]
TL['L03'] = (e+0.1, e+0.1+g('L03', 1.3)); e = TL['L03'][1]
SCAN0 = e+0.15; SCAN1 = SCAN0+0.8
TL['L04'] = (SCAN0+0.6, SCAN0+0.6+g('L04', 4.2)); e = TL['L04'][1]
TL['L05'] = (e+0.2, e+0.2+g('L05', 4.05)); e = TL['L05'][1]
BACK0 = e+0.1; BACK1 = BACK0+0.8; X0, BREAK_T = SCAN1, BACK0+0.3
TL['L06'] = (BACK0+0.5, BACK0+0.5+g('L06', 2.6)); e6 = TL['L06'][1]
SPAG_END = TL['L06'][0]+4.4
EQ0 = max(e6+0.3, SPAG_END-0.2); EQ1 = EQ0+3.0
SCAN2a, SCAN2b = EQ1-0.2, EQ1+0.6
TL['L07'] = (EQ1, EQ1+g('L07', 2.6)); e = TL['L07'][1]
TL['L08'] = (e+0.2, e+0.2+g('L08', 2.6)); e = TL['L08'][1]
D_SPIN = e+0.05; D_GREEN = D_SPIN+0.6
TL['L09'] = (D_GREEN-0.1, D_GREEN-0.1+g('L09', 4.28)); e = TL['L09'][1]
BACK2a, BACK2b = e-0.5, e+0.3
TL['L10'] = (e+0.15, e+0.15+g('L10', 5.25)); e = TL['L10'][1]
T = e+0.45
NF = int(round(T*FPS))

D = np.load('fall10.npz'); TAU, RC, OFF, RB, RS = D['tau'], D['rc'], D['off'], float(D['rb']), float(D['rs'])
IB = int(np.argmax(RC <= RB))
def fall_index(t):
    """solver frame for video time t: pre-break (0..IB) spans X0..BREAK_T, post-break (IB..end) spans BREAK_T..SPAG_END"""
    if t < X0: return 0
    if t < BREAK_T:
        u = sm((t-X0)/(BREAK_T-X0))**0.9; r = RC[0]*(RB/RC[0])**u
    else:
        u = np.clip((t-BREAK_T)/(SPAG_END-BREAK_T), 0, 1)**0.75; r = RB*(RS*1.0005/RB)**u
    return int(np.clip(np.searchsorted(-RC, -r), 0, len(RC)-1))
def fmt(v):
    if v >= 1e6: return f'{v/1e6:.0f} million'
    return f'{v:,.0f}' if v >= 10 else f'{v:.1g}'

# captions: key words of each spoken line, one at a time
def cap_at(k, frac): return TL[k][0] + frac*(TL[k][1]-TL[k][0])
CAPS = [(TL['L02'][0], cap_at('L02', 0.62), [[("you'd think it", WHITE)], [('crushes you', YELLOW)]]),
        (cap_at('L02', 0.62), TL['L03'][0], [[('nope.', YELLOW)]]),
        (TL['L03'][0], SCAN0+0.3, [[('it ', WHITE), ('STRETCHES', YELLOW), (' you', WHITE)]]),
        (TL['L04'][0], cap_at('L04', 0.55), [[('your feet are', WHITE)], [('closer', YELLOW), (' to the hole', WHITE)]]),
        (cap_at('L04', 0.55), TL['L04'][1]+0.2, [[("so they're pulled", WHITE)], [('harder', YELLOW)]]),
        (TL['L05'][0], cap_at('L05', 0.55), [[('that difference', WHITE)], [('stretches', YELLOW), (' you', WHITE)]]),
        (cap_at('L05', 0.55), BACK0+0.2, [[('and ', WHITE), ('squeezes', YELLOW), (' your sides', WHITE)]]),
        (TL['L06'][0], cap_at('L06', 0.42), [[('your body ', WHITE), ('snaps', RED)]]),
        (cap_at('L06', 0.42), EQ0-0.1, [[('feet first.', WHITE)], [('spaghetti', YELLOW)]]),
        (TL['L07'][0], TL['L07'][1]+0.1, [[('now: make the hole', WHITE)], [('bigger', YELLOW)]]),
        (TL['L08'][0], D_SPIN, [[('10 suns:', WHITE)], [('torn apart', RED), (' way out', WHITE)]]),
        (D_GREEN-0.05, TL['L09'][1]+0.1, [[('4 million suns:', WHITE)], [('feel ', WHITE), ('nothing', GREEN)]]),
        (TL['L10'][0], cap_at('L10', 0.45), [[("our galaxy's heart", WHITE)], [('has ', WHITE), ('one', YELLOW)]]),
        (cap_at('L10', 0.45), T, [[('next:', YELLOW)], [("why don't atoms collapse?", WHITE)]])]

RGB, ALPHA = astro.shade(); HRGB, HA = astro.holo()

# ---------------- real layer ----------------
def zoom_of(t):                                    # camera closes in on the hole as you fall
    if t < BACK0: return 1.0 + 0.04*sm(t/BACK0)
    if t < SPAG_END: return 1.04 + 0.30*sm((t-BACK0)/(SPAG_END-BACK0))
    if t < BACK2a: return 1.34
    return 1.34 - 0.30*sm((t-BACK2a)/(T-BACK2a))     # zoom-out at the end
def real_layer(t, R, STAR, TEX):
    L = A.real_at(t, R, STAR, TEX); z = zoom_of(t)
    if abs(z-1) > 1e-3:
        Mw = np.float32([[z, 0, (1-z)*A.RCX], [0, z, (1-z)*A.RCY]]); L = cv2.warpAffine(L, Mw, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    heat = 1.0 + 0.25*sm((t-BACK0)/4) if t > BACK0 else 1.0
    if t < SCAN1:                                  # before x-ray: intact, stretching (opening already mid-stretch)
        st = 0.13 + 0.12*sm(t/SCAN0) + 0.012*np.sin(t*2.1); off = sim7.X*(1+st); pxm = CAM['body_px']*S/sim7.L_BODY
        cy = CAM['body_y']*S + 10*S*np.sin(t*0.7); i = None
    elif t > BACK2a:                               # end: same opening look for the loop
        tt = (t - BACK2a); st = 0.13 + 0.012*np.sin(t*2.1); off = sim7.X*(1+st); pxm = CAM['body_px']*S/sim7.L_BODY
        cy = CAM['body_y']*S; i = None
    else:
        i = fall_index(t); off = OFF[i].copy(); span = off[-1]-off[0]
        pxm = min(CAM['body_px']/sim7.L_BODY, 880/span)*S; cy = max(CAM['body_y']*S, 470*S + off[-1]*pxm*1.08)
    cx = CX*S
    broken = i is not None and i >= IB
    q = np.clip((t-BREAK_T)/(SPAG_END-BREAK_T), 0, 1) if broken else 0
    astro.place_stretch(L, RGB*(heat if not broken else heat*(1+0.5*q)), ALPHA, cx, cy, pxm, off, 1.0, tear=0.6 if broken else 0.0, blur=4+12*q if broken else 0)
    if broken:                                     # hot glow on the torn strand
        glow = np.zeros_like(L); astro.place_stretch(glow, RGB*0+np.array([1.0, 0.45, 0.15], np.float32), ALPHA, cx, cy, pxm, off, 1.0, tear=1.0)
        fl = float(np.exp(-((t-BREAK_T-0.35)/0.22)**2))
        L += cv2.GaussianBlur(glow, (0, 0), 14*S)*(0.35*q + 0.9*fl)
    if t < SCAN0:                                  # cyan subject ring
        a = sm(t/0.2)*(1-sm((t-SCAN0+0.5)/0.4)); ry = int((off[-1]-off[0]+0.3)*pxm*0.56); rx = int(ry*0.44)
        ov = np.zeros_like(L); cv2.ellipse(ov, (int(cx), int(cy - 0.05*pxm)), (rx, ry), 0, 0, 360, tuple(v/255*1.9*a for v in CYAN), 4*S, cv2.LINE_AA); L += ov
    STATE.update(body=(int(cx/S), int(cy/S), int((off[-1]-off[0])*0.5*pxm/S)), i=i, broken=broken)
    return L

# ---------------- x-ray layer ----------------
GRID = np.zeros((H2, W2, 3), np.float32)
for x in range(0, W2, 60*S): cv2.line(GRID, (x, 0), (x, H2), tuple(v/255*0.06 for v in CYAN), 1)
for y in range(0, H2, 60*S): cv2.line(GRID, (0, y), (W2, y), tuple(v/255*0.06 for v in CYAN), 1)
BASE = K.NIGHT/255 + GRID
PRNG = np.random.default_rng(11); PX0 = PRNG.uniform(-1, 1, (420, 2)).astype(np.float32); PPH = PRNG.random(420).astype(np.float32)
STATE = {}
def field(img, t, cy, strength):
    """tidal field in your falling frame: a = k(-x/2... ) -> along the fall (+2k y), across (-k x). arrows + flowing dust."""
    amp = 0.25 + 0.75*strength
    for gy in range(560, 1400, 100):
        for gx in range(110, 900, 100):
            x, y = (gx - CX)/400, (cy/S - gy)/400                       # y + = up (away from hole)
            ax, ay = -x, 2*y; m = np.hypot(ax, ay)
            if m < 0.05: continue
            L = 26*amp*min(m, 1.6)
            p0 = (gx*S, gy*S); p1 = (int((gx + ax/m*L)*S), int((gy - ay/m*L)*S))
            cv2.arrowedLine(img, p0, p1, tuple(v/255*(0.35+0.4*strength) for v in AMBER), 2*S, cv2.LINE_AA, tipLength=0.35)
    k = 0.9*amp; tau = ((t*0.45 + PPH) % 1.4)                             # dust flows on hyperbolas x~e^{-kt}, y~e^{2kt}
    xs = PX0[:, 0]*0.9*np.exp(-k*tau); ys = PX0[:, 1]*0.25*np.exp(2*k*tau)
    a = np.clip(1.4 - tau, 0, 1)*np.clip(tau*4, 0, 1)
    for x, y, al in zip(xs, ys, a):
        px, py = int((CX + x*400)*S), int(cy - y*400*S)
        if 0 <= px < W2 and 0 <= py < H2: cv2.circle(img, (px, py), 3*S, tuple(float(v/255*al*1.1) for v in CYAN), -1, cv2.LINE_AA)
def xray_at(t):
    img = BASE.copy(); cy = 900*S
    if t < SCAN2a:                                                          # stretch explained (solver pre-break)
        i = fall_index(min(t, BREAK_T - 1e-3)); r = RC[i]; off = OFF[i]
        strain = (off[-1]-off[0])/sim7.L_BODY/(sim7.N-1)*sim7.N - 1
        gg = sim7.tidal(10, r)/9.81; s = float(np.clip(np.log10(max(gg, 1))/5.0, 0, 1))
        field(img, t, cy, s)
        q = sm(1 - np.log(r/RS)/np.log(RC[0]/RS)); ey = int((1950 - 420*q)*S)
        cv2.circle(img, (CX*S, ey + 2600*S), 2600*S, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (CX*S, ey + 2600*S), 2600*S, tuple(v/255*0.9 for v in AMBER), 3*S, cv2.LINE_AA)
        pxm = 380*S/sim7.L_BODY
        tint = np.array([1, 1, 1], np.float32) if strain < 0.2 else np.array(AMBER, np.float32)/np.array(CYAN, np.float32)
        hol = np.zeros_like(img); astro.place_stretch(hol, HRGB*tint, HA, CX*S, cy, pxm, off, 1.0); img += hol + cv2.GaussianBlur(hol, (0, 0), 6*S)*0.6
        ht, ft = cy - off[-1]*pxm, cy - off[0]*pxm
        for (y0, d) in ((ht - 40*S, -1), (ft + 40*S, 1)):                  # big tidal arrows on head and feet
            Lr = int((70 + 150*s)*S); cv2.arrowedLine(img, (CX*S, int(y0)), (CX*S, int(y0 + d*Lr)), tuple(v/255*1.5 for v in AMBER), 7*S, cv2.LINE_AA, tipLength=0.3)
        for sg in (-1, 1):
            x0 = CX*S + sg*int(190*S); cv2.arrowedLine(img, (x0, cy - 60*S), (x0 - sg*int((50 + 60*s)*S), cy - 60*S), tuple(v/255*1.3 for v in AMBER), 6*S, cv2.LINE_AA, tipLength=0.35)
        STATE.update(mode='fall', r=r, g=gg, ht=ht/S, ft=ft/S, ey=ey/S)
    else:                                                                   # dial twist
        spin = sm((t - D_SPIN)/(D_GREEN - D_SPIN)); big = t >= D_GREEN
        if not big:
            off = OFF[-1]; span = off[-1]-off[0]; pxm = min(380/sim7.L_BODY, 700/span)*S
            cv2.circle(img, (CX*S, 1380*S + 900*S), 900*S, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (CX*S, 1380*S + 900*S), 900*S, tuple(v/255*0.9 for v in RED), 3*S, cv2.LINE_AA)
            hol = np.zeros_like(img); astro.place_stretch(hol, RGB*np.array([1.6, 0.35, 0.25], np.float32), ALPHA, CX*S, 1000*S, pxm, off, 2.2, tear=0.6, blur=4)
            img += hol + cv2.GaussianBlur(hol, (0, 0), 10*S)*0.8
        else:
            k = sm((t - D_GREEN)/3.0); ey = int((1330 - 420*k)*S)
            cv2.rectangle(img, (0, ey), (W2, H2), (0, 0, 0), -1); cv2.line(img, (0, ey), (W2, ey), tuple(v/255*1.0 for v in GREEN), 3*S, cv2.LINE_AA)
            hol = np.zeros_like(img); astro.place_stretch(hol, HRGB*(np.array(GREEN, np.float32)/np.array(CYAN, np.float32)), HA, CX*S, 960*S, 380*S/sim7.L_BODY, sim7.X.copy(), 1.0)
            img += hol + cv2.GaussianBlur(hol, (0, 0), 6*S)*0.6
            STATE['ey'] = ey/S
        dc = (CX*S, 470*S); R0 = 120*S
        cv2.ellipse(img, dc, (R0, R0), 0, 180, 360, tuple(v/255*0.6 for v in AMBER), 4*S, cv2.LINE_AA)
        for k2 in range(11):
            a2 = np.radians(180 + 25 + 13*k2); cv2.line(img, (int(dc[0]+R0*0.85*np.cos(a2)), int(dc[1]+R0*0.85*np.sin(a2))), (int(dc[0]+R0*np.cos(a2)), int(dc[1]+R0*np.sin(a2))), tuple(v/255*0.6 for v in AMBER), 2*S, cv2.LINE_AA)
        ang = np.radians(180 + 25 + 130*spin); tip = (int(dc[0] + R0*0.92*np.cos(ang)), int(dc[1] + R0*0.92*np.sin(ang)))
        cv2.line(img, dc, tip, tuple(v/255*1.6 for v in AMBER), 6*S, cv2.LINE_AA); cv2.circle(img, dc, 10*S, tuple(v/255*1.5 for v in AMBER), -1, cv2.LINE_AA)
        STATE.update(mode='dial', spin=spin, big=big)
    return img

# ---------------- text ----------------
EQ_TXT = 'stretch = 2GM·L / r³'
def text(t, im):
    d = ImageDraw.Draw(im, 'RGBA'); K.series_tag(d, 7)
    if t < TL['L01'][1] + 0.1: K.title_box(d, 'FALL IN. WHAT HAPPENS?', y=222)
    xr = (SCAN1 <= t < BACK0) or (SCAN2b <= t < BACK2a)
    for a, b, lines in CAPS:
        if a <= t < b:
            al = sm((t-a)/0.12)*(1-sm((t-b+0.12)/0.12)) if b < T-0.05 else sm((t-a)/0.12)
            if not xr: d.rounded_rectangle([CX-440, 1420 - (35 if len(lines) == 2 else 0), CX+440, 1560 + (0 if len(lines) == 2 else -35)], 26, fill=(0, 0, 0, int(115*al)))
            K.caption(d, lines, alpha=al)
    if t < SCAN0 and 'body' in STATE:
        bx, by, bh = STATE['body']; K.pill(d, (bx + 205, by - bh + 30), 'you', CYAN)
        K.pill(d, (800, 1120 - 330), 'black hole', AMBER)
    if SCAN0 + 0.5 <= t < SCAN1 + 1.2: K.pill(d, (CX, 330), 'X-RAY: the code under it', CYAN)
    fall_pills = (SCAN1 <= t < BACK0 + 0.6) or (BACK0 <= t < EQ0)
    if fall_pills:
        i = fall_index(t) if t >= BACK0 else fall_index(min(t, BREAK_T-1e-3)); r = RC[i]; km = max(r - RS, 0)/1e3; gg = sim7.tidal(10, r)/9.81
        K.pill(d, (CX, 330 if t < BACK0 else 300), f'to the edge: {km:,.0f} km', WHITE)
        K.pill(d, (CX, 410 if t < BACK0 else 380), f'stretch: {fmt(gg)} g', RED if i >= IB else AMBER)
    if SCAN1 + 0.4 <= t < BACK0 and STATE.get('mode') == 'fall':
        hy = int(np.clip(STATE['ht'] - 120, 480, 1300)); fy = int(np.clip(STATE['ft'] + 120, 600, 1330))
        K.pill(d, (CX + 230, hy), 'head: pulled less', CYAN); K.pill(d, (CX + 230, min(fy, 1340)), 'feet: pulled MORE', AMBER)
        if t > TL['L05'][0] + 1.5: K.pill(d, (CX - 300, 820), 'squeezed', AMBER); K.pill(d, (CX + 300, 820), 'squeezed', AMBER)
        K.pill(d, (CX, 490), 'real solver · 10-sun black hole', WHITE)
    if BACK1 <= t < EQ0:
        K.pill(d, (250, 1340), 'you: not to scale', WHITE)
    if EQ0 <= t < EQ1:
        a = sm((t-EQ0)/0.25)*(1-sm((t-EQ1+0.25)/0.25)); c = lambda col: tuple(int(v*a) for v in col)
        d.rounded_rectangle([CX-425, 640, CX+425, 1110], 30, fill=(4, 8, 18, int(215*a)), outline=c(AMBER), width=3)
        d.text((CX, 740), EQ_TXT, font=K.F_EQ, fill=c(YELLOW), anchor='mm')
        for j, (s, col) in enumerate((("M = the black hole's mass", AMBER), ('r = how far you are', WHITE), ('L = how tall you are', CYAN))):
            d.text((CX, 870 + j*62), s, font=K.F_SUB, fill=c(col), anchor='mm')
    if t >= EQ1 - 0.1 and t < BACK2a:
        d.text((70, 250), EQ_TXT, font=K.F_MONO, fill=YELLOW, anchor='lm', stroke_width=2, stroke_fill=(0, 0, 0))
    if STATE.get('mode') == 'dial' and SCAN2b <= t < BACK2a:
        sp = STATE['spin']; mass = 10*(4.15e6/10)**sp
        lab = '4 million suns' if STATE['big'] else (f'{fmt(mass)} suns' if sp > 0.02 else '10 suns')
        K.pill(d, (CX, 520), f'MASS: {lab}', GREEN if STATE['big'] else (AMBER if sp > 0.02 else RED))
        if STATE['big']: K.pill(d, (CX, 610), 'stretch at the edge: 0.0001 g', GREEN); K.pill(d, (CX, int(STATE['ey']) + 40), 'the edge', GREEN) if STATE['ey'] < 1330 else None
        elif sp <= 0.02: K.pill(d, (CX, 610), 'stretch at the edge: 19 million g', RED)
    if t >= TL['L10'][0] + 0.3:
        K.end_stamp(d, sm((t-TL['L10'][0]-0.3)/0.3), y=1330)

# ---------------- frame ----------------
GRAIN = [np.random.default_rng(i).normal(0, 1, (K.H, K.W, 1)).astype(np.float32) for i in range(6)]
vy, vx = np.mgrid[0:K.H, 0:K.W].astype(np.float32)
VIG = (1 - 0.30*(((vx-K.W/2)/K.W)**2 + ((vy-K.H/2)/K.H)**2)*2)[..., None]
def scanmix(L, X, t, a, b, down=True):
    p = sm((t-a)/(b-a)); yl = int(p*H2) if down else int((1-p)*H2)
    img = L.copy(); img[:yl] = X[:yl] if down else img[:yl]
    if not down: img[:yl] = X[:yl]
    img += np.exp(-((np.arange(H2)-yl)/(6*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
    img += np.exp(-((np.arange(H2)-yl)/(60*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    return img
def frame(t, R, STAR, TEX):
    STATE.pop('mode', None)
    if t < SCAN0: img = real_layer(t, R, STAR, TEX)
    elif t < SCAN1: img = scanmix(real_layer(t, R, STAR, TEX), xray_at(t), t, SCAN0, SCAN1)
    elif t < BACK0: img = xray_at(t)
    elif t < BACK1: img = scanmix(real_layer(t, R, STAR, TEX), xray_at(t), t, BACK0, BACK1, down=False)
    elif t < SCAN2a: img = real_layer(t, R, STAR, TEX)
    elif t < SCAN2b: img = scanmix(real_layer(t, R, STAR, TEX), xray_at(t), t, SCAN2a, SCAN2b)
    elif t < BACK2a: img = xray_at(t)
    elif t < BACK2b: img = scanmix(real_layer(t, R, STAR, TEX), xray_at(t), t, BACK2a, BACK2b, down=False)
    else: img = real_layer(t, R, STAR, TEX)
    if EQ0 <= t < EQ1: img = img*(1 - 0.35*sm((t-EQ0)/0.3)*(1-sm((t-EQ1+0.3)/0.3)))
    im = A.tonemap(img); im = cv2.resize(im, (K.W, K.H), interpolation=cv2.INTER_AREA)
    im = np.clip(im*VIG + GRAIN[int(t*FPS) % 6]*0.008, 0, 1)
    pim = Image.fromarray((im*255).astype(np.uint8)); text(t, pim); return pim

def load():
    A.RCY = CAM['rcy']*S; A.SC_R = CAM['scr']*S
    return A.build_real(), A.star_map(), A.disk_tex()

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    if mode == 'times': print(json.dumps({k: [round(a, 2), round(b, 2)] for k, (a, b) in TL.items()}), 'T', round(T, 2), 'SCAN0', round(SCAN0, 2), 'BACK0', round(BACK0, 2), 'EQ0', round(EQ0, 2), 'D_GREEN', round(D_GREEN, 2)); sys.exit()
    R, STAR, TEX = load()
    if mode == 'sheet':
        for t in [float(x) for x in sys.argv[2].split(',')]: frame(t, R, STAR, TEX).save(f'f_{t:05.2f}.png')
    else:
        a, b = int(sys.argv[2]), min(int(sys.argv[3]), NF)
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS, R, STAR, TEX).tobytes())
        p.stdin.close(); p.wait(); open(f'part_{a:05d}.done', 'w').write('ok')
