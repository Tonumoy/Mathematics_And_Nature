# SOURCE CODE #6 - "WHY ISN'T THE SKY VIOLET?" (Coded Cosmos, 9 Nov 2026)
# real thing: CC BY 4.0 timelapse "Atmospheric convection over New Braunfels" (Brainandforce, Wikimedia Commons)
#             CC BY 4.0 "Sunset on Halfdome timelapse Yosemite" (G. Edward Johnson), NASA/JPL-Caltech/MSSS PIA19400 (Mars sunset)
# solvers (real, computed): sky.py single-scattering radiative transfer (beta ~ lambda^-n, Rayleigh n=4) -> CIE 1931 -> sRGB
#                           photon Monte Carlo: scatter chance per step ~ (550/lambda)^4, new direction from Rayleigh phase 1+cos^2
import numpy as np, cv2, subprocess, sys, json
from PIL import Image, ImageDraw, ImageFont
import source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE
import sky as SK

FPS = 30
W = json.load(open('words.json')); w0 = lambda k: W[k][0]; w1 = lambda k: W[k][1]
T = round(w1('ASK') + 0.55, 2)
TL = dict(T=T, SCAN0=w1('NOPE') + 0.4, SCAN1=w1('NOPE') + 1.2, MOL0=w0('ANT'), SIX0=w0('SIX'), RAMP=w0('SIX') + 0.9, EQ0=w0('EQ'), EQ1=w0('WHY') + 0.4,
          SPEC0=w0('WHY') + 0.4, SUNC=w0('EYE'), EYEC=w0('EYE') + 2.0, SWATCH=w0('EYE') + 3.3,
          DIAL0=w0('DIAL'), DIAL1=w0('DIAL') + 1.0, DIAL2=w1('DIAL') + 0.2, DIAL3=w1('WHITE') - 0.75, DIAL4=w1('WHITE') - 0.1,
          BACK0=w1('WHITE') + 0.05, HIT=w1('EQ') + 0.05, BACK1=w1('WHITE') + 0.65, MARS0=w0('MARS') - 0.25, STAMP0=w0('ASK') - 0.2)
G = lambda k: TL[k]
RNG = np.random.default_rng(6)
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; F_EQG = ImageFont.truetype(DJ, 96); F_EQS = ImageFont.truetype(DJ, 60)
F_MONOG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 33)
YY, XX = np.mgrid[0:1920, 0:1080].astype(np.float32)
VIG = (1 - 0.30*(((XX-540)/1080)**2 + ((YY-960)/1920)**2)*2)[..., None]
GRAIN = [RNG.normal(0, 1, (1920, 1080, 1)).astype(np.float32) for _ in range(6)]
BAND = (1 - 0.50*np.exp(-((YY-1500)/130)**2))[..., None]       # J4 dark band behind captions (bright sky)
NIGHT = K.NIGHT/255

# ---------------- real layers ----------------
def read_all(p, step=1):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
SKYV = np.memmap('src/sky.rgb', np.uint8, 'r').reshape(-1, 1920, 1080, 3)     # first 7.2 s @30 fps (from 5120x2880 original)
SUNV = np.memmap('src/sun.rgb', np.uint8, 'r').reshape(-1, 1920, 1080, 3)     # 11.1 s @29.97 fps (from 3840x2160)
def vid(fr, i): return fr[int(np.clip(i, 0, len(fr)-1))].astype(np.float32)/255
MARS = cv2.imread('src/mars.jpg')[:, :, ::-1]      # 1344 x 1080, sun at ~(462, 637)
MCROP = cv2.resize(MARS[190:1050, 40:900], (860, 860), interpolation=cv2.INTER_AREA).astype(np.float32)/255

def ring(im, c, ax, a, t):
    if a <= 0: return im
    pul = 1 + 0.025*np.sin(t*4.2); ov = np.zeros_like(im)
    cv2.ellipse(ov, c, (int(ax[0]*pul), int(ax[1]*pul)), 0, 0, 360, tuple(v/255*a for v in CYAN), 5, cv2.LINE_AA)
    return im + ov*1.4
def real_at(t):
    if t < G('BACK0') - 0.01:
        im = vid(SKYV, round(t*FPS*1.0))
        im = ring(im, (300, 590), (210, 160), sm((t-0.6)/0.3)*(1-sm((t-w0('NOPE')-0.2)/0.4)), t)   # J2: the blue
        return im*BAND
    if t < G('MARS0'):
        im = vid(SUNV, round((t - G('BACK0'))*29.97*1.25))
        a = sm((t - G('BACK1') - 0.2)/0.3)*(1 - sm((t - G('MARS0') + 0.5)/0.4))
        im = ring(im, (560, 650), (300, 210), a, t)
        return im*BAND
    return None

# ---------------- solver output: computed skies for the dial (n = 0..4) ----------------
_, EXPO = SK.sky_image(n=4.0)
EXPO *= 1.25
NGRID = np.linspace(0, 4, 41)
SKIES = [SK.sky_image(n=n, expo=EXPO)[0].astype(np.float32) for n in NGRID]     # 54x96 solver grid, upscaled per frame
def sky_n(n):
    i = np.clip(n/0.1, 0, 40); a = int(np.floor(i)); b = min(a+1, 40); u = i - a
    return np.clip(cv2.resize(SKIES[a]*(1-u) + SKIES[b]*u, (1080, 1920), interpolation=cv2.INTER_CUBIC), 0, 1)
def n_at(t):
    pts = [(G('DIAL0'), 4.0), (G('DIAL1'), 4.0), (G('DIAL2'), 0.0), (G('DIAL3'), 0.0), (G('DIAL4'), 4.0)]
    ts, vs = zip(*pts); return float(np.interp(t, ts, vs))

# ---------------- photon Monte Carlo (precomputed, deterministic) ----------------
LAMS = np.array([410, 450, 530, 600, 700.0])
def lam_rgb(l):
    S = np.zeros(len(SK.LAM)); S[np.argmin(np.abs(SK.LAM - l))] = 1
    c = np.clip(SK.spec_to_lin(S), 0, None); c = c/c.max(); return SK.gamma(c)
LCOL = np.array([lam_rgb(l) for l in LAMS]); LCOL[0] = [0.55, 0.30, 1.0]          # 410 nm display tint (out of sRGB gamut)
SUNP = np.array([120.0, 560.0]); DIR = np.array([np.cos(np.radians(16)), np.sin(np.radians(16))])
BOX = (60, 430, 1020, 1390)
MOLS = np.stack([RNG.uniform(BOX[0], BOX[2], 240), RNG.uniform(BOX[1], BOX[3], 240)], 1)
K0 = 0.00025                                          # scatter chance per px at 550 nm (thin regime)
def simulate():
    t0, t1 = G('SCAN0'), G('EQ1') + 0.1; n = int((t1 - t0)*FPS) + 1
    P = []; snaps = []; cnt = np.zeros(5); emi = np.zeros(5); rng = np.random.default_rng(66)
    for f in range(n):
        t = t0 + f/FPS; fast = t > G('RAMP')
        v = 1200.0 if fast else 240.0; rate = 14 if fast else 6
        for _ in range(rng.poisson(rate/FPS*FPS/6) if fast else (1 if f % 5 == 0 else 0)):
            k = rng.integers(0, 5); y = rng.uniform(-26, 26); emi[k] += 1
            P.append(dict(p=SUNP + np.array([0, y]) + DIR*60, d=DIR.copy(), k=k, s=False, age=0.0, tr=[]))
        for ph in P:
            step = v/FPS
            pr = 1 - np.exp(-K0*(550/LAMS[ph['k']])**4*step)
            if not ph['s'] and rng.random() < pr:
                while True:                               # Rayleigh phase: pdf ~ 1 + cos^2(theta)
                    th = rng.uniform(0, 2*np.pi)
                    if rng.random()*2 < 1 + np.cos(th)**2: break
                c, s_ = np.cos(th), np.sin(th); d = ph['d']; ph['d'] = np.array([c*d[0] - s_*d[1], s_*d[0] + c*d[1]])
                ph['s'] = True; cnt[ph['k']] += 1; ph['hit'] = ph['p'].copy()
            ph['tr'] = (ph['tr'] + [ph['p'].copy()])[-3:]
            ph['p'] = ph['p'] + ph['d']*step; ph['age'] += 1/FPS
        P = [ph for ph in P if BOX[0]-80 < ph['p'][0] < BOX[2]+80 and BOX[1]-80 < ph['p'][1] < BOX[3]+80]
        snaps.append(([(np.array(ph['tr'] + [ph['p']]), ph['k'], ph['s']) for ph in P], 100*cnt/np.maximum(emi, 1)))
    return snaps
SNAPS = simulate()
def snap_at(t): return SNAPS[int(np.clip(round((t - G('SCAN0'))*FPS), 0, len(SNAPS)-1))]

GRID = np.zeros((1920, 1080, 3), np.float32)
for x in range(0, 1080, 90): GRID[:, x] = 0.05
for y in range(0, 1920, 90): GRID[y, :] = 0.05
GRID *= np.array(CYAN, np.float32)/255

def photons_layer(t, dim=1.0):
    img = np.ones((1920, 1080, 3), np.float32)*NIGHT + GRID
    acc = np.zeros((1920//2, 1080//2, 3), np.float32)
    for m in MOLS: cv2.circle(acc, (int(m[0]/2*4), int(m[1]/2*4)), 5, tuple(v/255*0.55 for v in CYAN), -1, cv2.LINE_AA, shift=2)
    sp = np.zeros_like(acc)
    cv2.circle(sp, (int(SUNP[0]/2), int(SUNP[1]/2)), 30, (1.0, 0.95, 0.8), -1, cv2.LINE_AA)
    ps, cnt = snap_at(t)
    for tr, k, s in ps:
        pts = (tr/2*4).astype(np.int32)
        cv2.polylines(sp, [pts], False, tuple(float(v)*(1.25 if s else 0.9) for v in LCOL[k]), 6 if s else 4, cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(sp, (0, 0), 6)*1.4 + cv2.GaussianBlur(sp, (0, 0), 20)*0.8
    lay = cv2.resize(acc + sp + glow, (1080, 1920), interpolation=cv2.INTER_LINEAR)
    img = 1 - (1 - img)*np.exp(-lay*1.3*dim)
    return img, cnt

# ---------------- spectrum panel ----------------
GX0, GX1, GY0, GY1 = 120, 900, 600, 1150
def gx(l): return GX0 + (l - 380)/340*(GX1 - GX0)
STRIP = np.zeros((1920, 1080, 3), np.float32)
for x in range(GX0, GX1):
    l = 380 + (x - GX0)/(GX1 - GX0)*340; i = np.argmin(np.abs(SK.LAM - l))
    S = np.zeros(len(SK.LAM)); S[i] = 1; c = np.clip(SK.spec_to_lin(S), 0, None); c = SK.gamma(c/c.max())
    if l < 420: c = c*(0.35 + 0.65*(l-380)/40)
    STRIP[1170:1200, x] = c*0.9
LF = np.linspace(380, 720, 200)
R4 = (380/LF)**4                                                     # lambda^-4 alone (normalised)
SUNF = np.interp(LF, SK.LAM, SK.SUN)
SC = SUNF*(550/LF)**4; SC /= SC.max()                                 # sunlight x lambda^-4
ZB = np.interp(LF, SK.LAM, SK.CMF[:, 2]); ZB /= ZB.max()              # blue-cone-like response (CIE z-bar)
PROD = SC*ZB; PROD /= PROD.max()
SKYCOL = SK.colour_of(SK.SUN*(550/SK.LAM)**4)                          # computed colour of the scattered light
def curve(d, ys, col, width=6, frac=1.0, dash=False):
    m = max(2, int(len(LF)*frac)); pts = [(gx(LF[i]), GY1 - (GY1 - GY0)*ys[i]) for i in range(m)]
    if dash:
        for i in range(0, m - 1, 6): d.line(pts[i:i+4], fill=col, width=width)
    else: d.line(pts, fill=col, width=width, joint='curve')
def spectrum_layer(t, d):
    img = np.ones((1920, 1080, 3), np.float32)*NIGHT + GRID
    img = np.maximum(img, STRIP)
    d.line([(GX0, GY1), (GX1, GY1)], fill=WHITE, width=3); d.line([(GX0, GY0 - 20), (GX0, GY1)], fill=WHITE, width=3)
    for l in (400, 500, 600, 700): d.text((gx(l), 1232), f'{l}', font=K.F_MONO, fill=WHITE, anchor='mm')
    d.text((GX1, 1270), 'wavelength (nm)', font=K.F_MONO, fill=(170, 180, 200), anchor='rm')
    a, b, c = G('SPEC0'), G('SUNC'), G('EYEC')
    u = sm((t - b)/0.8)
    ys = R4*(1 - u) + SC*u
    curve(d, ys, CYAN, 7, frac=sm((t - a)/1.2))
    if t > a + 0.8 and t < b + 0.6:
        d.text((gx(392), GY0 - 40), 'violet wins?', font=K.F_PILL, fill=(190, 150, 255), anchor='lm')
    if t > b:
        curve(d, SUNF/SUNF.max(), WHITE, 3, dash=True, frac=sm((t - b)/0.8))
        if t < c: d.text((gx(560), GY0 + 10), 'sunlight', font=K.F_MONO, fill=WHITE, anchor='mm')
    if t > c:
        curve(d, ZB, (110, 140, 255), 4, dash=True, frac=sm((t - c)/0.6))
        if t > c + 0.6:
            v = sm((t - c - 0.6)/0.6)
            pts = [(gx(LF[i]), GY1 - (GY1 - GY0)*PROD[i]*v) for i in range(len(LF))] + [(gx(720), GY1), (gx(380), GY1)]
            d.polygon(pts, fill=tuple(int(x*255*0.55) for x in SKYCOL))
            d.text((gx(560), GY1 - (GY1 - GY0)*0.55), 'what your eyes get', font=K.F_MONO, fill=WHITE, anchor='mm')
        d.text((gx(690), GY0 + 60), 'eye: weak at violet', font=K.F_MONO, fill=(140, 160, 255), anchor='rm')
    return img
def swatch(t, d):
    if t < G('SWATCH'): return
    a = sm((t - G('SWATCH'))/0.4); c = tuple(int(v*255*a) for v in SKYCOL)
    d.ellipse([CX - 95, 330 + 130, CX + 95, 330 + 320], fill=c, outline=GREEN, width=5) if False else None

# ---------------- text layer ----------------
def eq_draw(d, y, a, nval=4):
    c = lambda col: tuple(int(v*a) for v in col)
    parts = [('I ∝ 1 / ', WHITE, F_EQG), ('λ', CYAN, F_EQG)]
    w = sum(d.textlength(s, font=f) for s, _, f in parts) + d.textlength(f'{nval:g}', font=F_EQS); x = CX - w/2
    for s, col, f in parts:
        d.text((x, y), s, font=f, fill=c(col), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s, font=f)
    d.text((x + 2, y - 40), f'{nval:g}', font=F_EQS, fill=c(AMBER), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0))
def chip_eq(d, nval, col=AMBER):
    s = f'I ∝ 1/λ^{nval:.1f}' if nval != 4 else 'I ∝ 1/λ⁴'
    K.pill(d, (CX, 455), s, col, font=F_MONOG)
CAPS = [
 (0.20, w0('GUESS'), ['it *should* be violet.']),
 (w0('GUESS'), w0('NOPE'), ['"it reflects the *ocean*"']),
 (w0('NOPE'), w1('NOPE') + 0.4, ['*nope.*', 'deserts get blue skies too']),
 (w1('NOPE') + 0.55, w0('ANT'), ['sunlight = *every colour*', 'mixed together']),
 (w0('ANT'), w0('ANT') + 3.0, ['air is full of', 'tiny *molecules*']),
 (w0('ANT') + 3.0, w0('SIX') - 0.05, ['light hits one: it *shakes*', 'and flings light sideways']),
 (w0('SIX') - 0.05, w0('SIX') + 2.5, ['blue gets flung', '*6x more* than red']),
 (w0('SIX') + 2.5, w0('EQ') - 0.05, ['so blue comes from', '*every part* of the sky']),
 (G('EQ0') + 0.3, G('EQ1'), ['shorter wave =', 'scattered *way harder*']),
 (G('SPEC0'), G('SUNC'), ['violet gets flung', '*even more*... so why not?']),
 (G('SUNC'), G('SUNC') + 2.45, ['the Sun makes *less violet*']),
 (G('SUNC') + 2.45, G('DIAL0'), ['your eyes are *bad* at violet', 'so the mix looks *sky blue*']),
 (G('DIAL0'), G('DIAL2') + 0.6, ['change one number:', 'the *4* becomes *0*']),
 (G('DIAL2') + 0.6, G('BACK0'), ['*white sky!*', 'every colour bounces the same']),
 (w0('SUN'), w0('SUN') + 2.6, ['sunset: *38x* more air']),
 (w0('SUN') + 2.6, G('MARS0'), ['most of the blue', 'bounces away *first*']),
 (G('MARS0'), w0('ASK') - 0.1, ['Mars dust flips it:', '*blue* sunsets']),
 (w0('ASK') - 0.1, T, ["what colour is the", "*Moon's* sky? comment"]),
]
def seg(s):
    return [(p, YELLOW if i % 2 else WHITE) for i, p in enumerate(s.split('*')) if p]
CAPS = [(a, b, [seg(l) for l in L]) for a, b, L in CAPS]
STATE = {}
def text(t, im):
    d = ImageDraw.Draw(im)
    real = t < G('SCAN0') or t >= G('BACK0')
    if t < G('SCAN0'): hook = 'the real code behind blue'
    elif t < G('EQ0'): hook = 'solver: light vs air molecules'
    elif t < G('DIAL0'): hook = 'solver: sunlight + your eye'
    elif t < G('BACK0'): hook = 'computed sky · what if?'
    elif t < G('MARS0'): hook = 'Yosemite · real timelapse'
    else: hook = 'Mars · NASA Curiosity rover'
    K.title_box(d, "WHY ISN'T THE SKY VIOLET?", hook, y=205)
    K.series_tag(d, 6)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, y=1500, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 0.7 < t < w0('NOPE') + 0.2: K.pill(d, (300, 790), 'this blue', CYAN, font=K.F_MONO)
    if w0('GUESS') + 0.2 < t < w0('NOPE'): K.pill(d, (CX, 1160), 'guess: reflects the ocean?', WHITE, font=K.F_PILL)
    if w0('NOPE') < t < w1('NOPE') + 0.35:
        a = sm((t - w0('NOPE'))/0.15); d.text((CX, 1110), 'NOPE', font=K.F_EQ, fill=tuple(int(v*a) for v in RED), anchor='mm', stroke_width=6, stroke_fill=(0, 0, 0))
    if t < G('SCAN0') - 0.3: d.text((930, 1590), 'video: Brainandforce · CC BY 4.0', font=ImageFont.truetype(DJ, 20), fill=(220, 220, 220), anchor='rs', stroke_width=2, stroke_fill=(0, 0, 0))
    if G('SCAN1') <= t < G('EQ0'):
        cnt = STATE.get('cnt', np.zeros(5))
        if t > G('MOL0') and t < G('SIX0'):
            K.pill(d, (650, 1010), 'air molecule (N₂ / O₂)', CYAN, font=F_MONOG)
        if t > G('SIX0'):
            K.pill(d, (CX, 1240), f'scattered: blue {cnt[1]:.0f}%  red {cnt[4]:.0f}%', WHITE, font=F_MONOG)
            K.pill(d, (CX, 1320), 'per molecule: blue 5.9x red', CYAN, font=F_MONOG)
        if t > G('RAMP'): K.pill(d, (790, 470), 'sped up x5', AMBER, font=F_MONOG)
        else: K.pill(d, (790, 470), 'slow motion', AMBER, font=F_MONOG)
    if G('EQ0') <= t < G('EQ1'):
        a = sm((t - G('EQ0'))/0.35); d.rounded_rectangle([CX-330, 730, CX+330, 1010], 30, fill=(6, 11, 24), outline=tuple(int(v*a) for v in CYAN), width=3); eq_draw(d, 830, a)
        K.pill(d, (CX, 960), 'λ = wavelength · 4 = the power', WHITE, font=F_MONOG) if t > G('EQ0') + 0.5 else None
    if t >= G('EQ1') and t < G('BACK0'):
        n = n_at(t) if t >= G('DIAL0') else 4
        col = AMBER if t < G('DIAL1') else (GREEN if (t > G('DIAL4') - 0.05 and n > 3.95) else RED)
        chip_eq(d, n if t >= G('DIAL0') else 4, col if t >= G('DIAL0') else AMBER)
    if G('SPEC0') <= t < G('DIAL0'):
        if t > G('SWATCH'):
            a = sm((t - G('SWATCH'))/0.4); c = tuple(int(v*255) for v in SKYCOL)
            d.rounded_rectangle([CX - 210, 1290, CX + 210, 1380], 20, fill=c, outline=GREEN, width=4)
            d.text((CX, 1335), 'computed: sky blue', font=K.F_PILL, fill=(10, 20, 40), anchor='mm')
    if G('DIAL0') <= t < G('BACK0'):
        n = n_at(t); x0, x1, yb = 160, 820, 1250
        X = lambda v: x0 + v/4*(x1 - x0)
        col = AMBER if t < G('DIAL1') else (GREEN if (t > G('DIAL4') - 0.05 and n > 3.95) else RED)
        d.rounded_rectangle([x0 - 20, yb - 12, x1 + 20, yb + 12], 12, fill=(8, 12, 22), outline=WHITE, width=3)
        d.ellipse([X(n) - 26, yb - 26, X(n) + 26, yb + 26], fill=col, outline=(0, 0, 0), width=3)
        for v in (0, 1, 2, 3, 4): d.text((X(v), yb + 50), f'{v}', font=K.F_MONO, fill=WHITE, anchor='mm', stroke_width=2, stroke_fill=(0, 0, 0))
        K.pill(d, (CX, 1150), f'dial: the power = {n:.1f}', col, font=F_MONOG)
        if n < 0.2: K.pill(d, (CX, 600), 'all colours equal → white', RED, font=F_MONOG)
        if t > G('DIAL4') - 0.05 and n > 3.95: K.pill(d, (CX, 600), 'power 4 → blue again', GREEN, font=F_MONOG)
    if G('BACK1') < t < G('MARS0'):
        K.pill(d, (CX, 1240), 'sun low: ~38x more air', AMBER, font=F_MONOG)
        d.text((930, 1590), 'video: G. Edward Johnson · CC BY 4.0', font=ImageFont.truetype(DJ, 20), fill=(220, 220, 220), anchor='rs', stroke_width=2, stroke_fill=(0, 0, 0))
    if G('MARS0') <= t < G('STAMP0'): K.pill(d, (CX, 470), 'sources: Bodhaine 1999 · NASA/JPL', WHITE, font=F_MONOG)
    if t >= G('MARS0'):
        d.text((CX, 1362), 'NASA/JPL-Caltech/MSSS · Gale crater · 15 Apr 2015', font=ImageFont.truetype(DJ, 22), fill=(200, 210, 230), anchor='mm')
    a = sm((t - G('STAMP0'))/0.4)*(1 - sm((t - T + 0.45)/0.3))
    if a > 0:
        K.end_stamp(d, alpha=a, y=480)
        pass

def mars_layer(t):
    img = np.ones((1920, 1080, 3), np.float32)*NIGHT
    u = sm((t - G('MARS0'))/0.5); z = 1.0 + 0.04*(t - G('MARS0'))
    sz = int(800*z); m = cv2.resize(MCROP, (sz, sz), interpolation=cv2.INTER_CUBIC)
    c = sz//2 - 400; m = m[c:c+800, c:c+800]
    img[540:1340, 90:890] = img[540:1340, 90:890]*(1-u) + m*u
    cv2.rectangle(img, (88, 538), (891, 1341), tuple(v/255*u for v in CYAN), 3, cv2.LINE_AA)
    return img

def xray_at(t):
    if t < G('EQ0'):
        img, cnt = photons_layer(t); STATE['cnt'] = cnt
        if G('MOL0') < t < G('SIX0'):                      # one molecule as an antenna: shakes + re-radiates
            m = np.array([560.0, 900.0]); a = sm((t - G('MOL0'))/0.3)*(1 - sm((t - G('SIX0') + 0.3)/0.3))
            ov = np.zeros_like(img); yy = 14*np.sin(t*2*np.pi*2.2)
            cv2.circle(ov, (int(m[0]), int(m[1] + yy)), 16, tuple(v/255*a for v in CYAN), -1, cv2.LINE_AA)
            for k in range(4):
                r = ((t*160 + k*70) % 280) + 20
                cv2.circle(ov, (int(m[0]), int(m[1])), int(r), tuple(v/255*a*(1 - r/300) for v in CYAN), 3, cv2.LINE_AA)
            cv2.ellipse(ov, (int(m[0]), int(m[1])), (70, 70), 0, 0, 360, tuple(v/255*a for v in CYAN), 4, cv2.LINE_AA)
            img = img + ov
        return img
    if t < G('SPEC0'):
        img, _ = photons_layer(t, dim=0.15); return img
    if t < G('DIAL0'):
        return None                                       # spectrum (drawn in PIL)
    sk = sky_n(n_at(t))
    return np.clip((sk*0.92 + GRID*1.5)*BAND, 0, 1)

def frame(t):
    L = real_at(t) if (t < G('SCAN1') or t >= G('BACK0')) else None
    if t >= G('MARS0'): L = mars_layer(t)
    X = None; spec = G('SPEC0') <= t < G('DIAL0')
    if G('SCAN0') <= t < G('BACK1') and not spec: X = xray_at(t)
    pim_ov = None
    if spec:
        pim = Image.new('RGB', (1080, 1920)); d = ImageDraw.Draw(pim)
        base = spectrum_layer(t, d)
        X = np.clip(base + np.asarray(pim).astype(np.float32)/255, 0, 1)
        if t < G('SPEC0') + 0.3:                         # quick dissolve from photons
            u = sm((t - G('SPEC0'))/0.3); X = photons_layer(t, 0.35)[0]*(1-u) + X*u
    if t >= G('DIAL0') and t < G('DIAL0') + 0.35 and X is not None:
        u = sm((t - G('DIAL0'))/0.35)
        pim = Image.new('RGB', (1080, 1920)); spectrum_layer(t, ImageDraw.Draw(pim))
        X = X*u + (np.ones_like(X)*NIGHT)*(1-u)
    if L is not None and X is not None:
        if t < G('SCAN1'): yl = int(sm((t - G('SCAN0'))/(G('SCAN1') - G('SCAN0')))*1920)
        else: yl = int((1 - sm((t - G('BACK0'))/(G('BACK1') - G('BACK0'))))*1920)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(1920) - yl)/6)**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(1920) - yl)/60)**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    if t >= G('MARS0') and t < G('MARS0') + 0.5:
        u = sm((t - G('MARS0'))/0.5); img = real_at(G('MARS0') - 0.02)*(1-u) + img*u
    tl = t - (T - 0.45)
    if tl > 0: u = sm(tl/0.45); img = img*(1-u) + real_at(0.0)*u              # seamless loop to frame 1
    realish = (t < G('SCAN0')) or (G('BACK1') <= t < G('MARS0'))
    img = np.clip(img*(VIG if realish else 1) + GRAIN[int(t*FPS) % 6]*(0.010 if realish else 0.003), 0, 1)
    pim = Image.fromarray((img*255).astype(np.uint8)); text(t, pim); return pim

if __name__ == '__main__':
    json.dump(TL, open('timeline.json', 'w'), indent=0)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    if mode == 'sheet':
        for u in [float(x) for x in sys.argv[2].split(',')]: frame(u).save(f'f_{u:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS).tobytes())
        p.stdin.close(); p.wait(); open(f'part_{a:05d}.done', 'w').write('ok')
