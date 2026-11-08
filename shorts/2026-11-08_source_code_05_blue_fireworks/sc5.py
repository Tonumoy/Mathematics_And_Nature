# SOURCE CODE #5 - "WHY IS BLUE SO RARE?" (Coded Cosmos, Diwali 8 Nov 2026)
# real thing: CC0 photo "Fireworks in Kuwait City" (Pixel.la, 2016) + PD painting "Radha and Krishna watching fireworks",
#             Sitaram, Kishangarh, late 18th c., National Museum New Delhi 63.810
# solvers (real, computed every frame):
#   1) heat glow: Planck's law B(lam,T) -> CIE 1931 colour matching -> sRGB   (colour of a hot spark at temperature T)
#   2) quantum glow: photon wavelength lam = h c / dE  (1239.84 nm*eV / dE)  + molecular emission bands
#      (SrCl ~636/661/675 nm + SrOH 606, BaCl 513.8/524.1, CuCl 428/435/443, Na 589.0/589.6 nm) -> sRGB
#   CuCl survival on the flame dial: logistic step at the cited ~1,200 C limit (Conkling) - a threshold, not a solved equilibrium
#   3) firework stars: ballistic flight with linear air drag  v' = -g y - k v  (analytic solution)
import numpy as np, cv2, subprocess, sys, json, os
from PIL import Image, ImageDraw, ImageFont
import source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE
import spectra as SP

FPS = 30
TL = json.load(open('timeline.json'))           # beats from the voice-over (vo_cuts.py)
T = TL['T']
G = lambda k: TL[k]
RNG = np.random.default_rng(5)
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; F_EQG = ImageFont.truetype(DJ, 84); F_CHIPG = ImageFont.truetype(DJ, 34); F_MONOG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 28)
HC = 1239.84                                       # h*c in nm*eV (NIST CODATA)

# ---------------- real layer: CC0 fireworks plate ----------------
PLATE = cv2.imread('art/2.jpg')[:, :, ::-1].copy()          # 3840 x 2575
PH, PW = PLATE.shape[:2]
CROP_CX = 1290                                                # towers + red/green bursts
def plate_at(t, zoom):
    h = PH/zoom; w = h*1080/1920
    cx = CROP_CX + 40*np.sin(t*0.21); cy = PH/2 - 60
    x0 = int(np.clip(cx - w/2, 0, PW - w)); y0 = int(np.clip(cy - h/2, 0, PH - h))
    sub = PLATE[y0:y0+int(h), x0:x0+int(w)]
    return cv2.resize(sub, (1080, 1920), interpolation=cv2.INTER_AREA).astype(np.float32)/255
YY, XX = np.mgrid[0:1920, 0:1080].astype(np.float32)
VIG = (1 - 0.30*(((XX-540)/1080)**2 + ((YY-960)/1920)**2)*2)[..., None]
GRAIN = [RNG.normal(0, 1, (1920, 1080, 1)).astype(np.float32) for _ in range(6)]
BAND = (1 - 0.45*np.exp(-((YY-1500)/120)**2))[..., None]     # J4 dark band behind captions

# ---------------- firework stars: v' = -g yhat - k v ----------------
GRAV, KDRAG = 9.81, 1.1           # m/s^2, 1/s (stars ~ 1 cm pellets: terminal speed g/k ~ 9 m/s)
def star_pos(p0, v0, t):
    e = np.exp(-KDRAG*t)
    x = p0[0] + v0[..., 0]*(1-e)/KDRAG
    y = p0[1] + v0[..., 1]*(1-e)/KDRAG - GRAV/KDRAG*(t - (1-e)/KDRAG)
    return x, y
def make_burst(t0, cx, cy, col, n=90, v=34, life=2.4, scale=9.0):
    th = RNG.uniform(0, 2*np.pi, n); ph = np.arccos(RNG.uniform(-1, 1, n))
    v0 = np.stack([v*np.sin(ph)*np.cos(th), v*np.cos(ph)], 1)*RNG.uniform(0.92, 1.0, (n, 1))
    return dict(t0=t0, p=(cx, cy), v0=v0, col=np.array(col, np.float32), life=life*RNG.uniform(0.85, 1.1, n), s=scale)
def draw_bursts(bursts, t, H=1920, W=1080):
    acc = np.zeros((H, W, 3), np.float32)
    for b in bursts:
        tau = t - b['t0']
        if tau < 0 or tau > b['life'].max() + 0.2: continue
        for k in range(7):                                    # trail = past positions
            tt = max(tau - k*0.05, 0)
            x, y = star_pos((0, 0), b['v0'], tt)
            px = b['p'][0] + x*b['s']; py = b['p'][1] - y*b['s']
            fade = np.clip(1 - tau/b['life'], 0, 1)**0.7*(1 - k/7)**1.5
            for i in range(len(px)):
                if fade[i] <= 0.01: continue
                c = tuple(float(v) for v in b['col']*fade[i]*(1.6 if k == 0 else 0.8))
                cv2.circle(acc, (int(px[i]*4), int(py[i]*4)), 10 if k == 0 else 6, c, -1, cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(acc, (0, 0), 9)*1.6 + cv2.GaussianBlur(acc, (0, 0), 30)*1.2
    return acc + glow

C_RED, C_GRN, C_NA, C_BLU = SP.mol_rgb('SrCl'), SP.mol_rgb('BaCl'), SP.mol_rgb('Na'), SP.mol_rgb('CuCl')
C_GOLD = SP.planck_rgb(2000)
def bursts_intro():
    return [make_burst(-0.6, 300, 560, C_RED*1.0 + 0.15), make_burst(0.15, 760, 470, C_GOLD),
            make_burst(0.9, 520, 760, C_GRN*0.9 + 0.1), make_burst(1.7, 820, 700, C_GOLD),
            make_burst(2.6, 260, 520, C_NA), make_burst(3.6, 700, 600, C_RED + 0.1),
            make_burst(4.6, 430, 480, C_GRN*0.9 + 0.1), make_burst(5.6, 760, 520, C_GOLD)]
def bursts_outro(t0):
    return [make_burst(t0 + 0.0, 540, 600, np.array([0.15, 0.38, 1.25]), n=110, v=36),   # display tint of CuCl blue (C_BLU is outside sRGB gamut, clips to violet)
            make_burst(t0 + 0.8, 300, 760, C_RED + 0.1), make_burst(t0 + 1.4, 800, 720, C_GRN*0.9 + 0.1)]

# ---------------- real layer: REAL Diwali footage (Pexels 34651770, free Pexels licence) ----------------
SEG_A = np.memmap('foot/seg_a.rgb', np.uint8, 'r').reshape(-1, 1920, 1080, 3)   # src 1.0-7.8 s @30 fps
SEG_B = np.memmap('foot/seg_b.rgb', np.uint8, 'r').reshape(-1, 1920, 1080, 3)   # src 9.0-15.4 s @30 fps
DROP = 140                                                    # footage sits 140 px lower: bursts clear of the title (J3)
SKYTOP = None
WB = np.array([1.05, 0.98, 0.86], np.float32)
SKY = np.median(SEG_A[0, :8].reshape(-1, 3), 0).astype(np.float32)/255                  # mild warm white balance (camera grade)
def foot_frame(seg, i):
    f = seg[int(np.clip(i, 0, len(seg) - 1))].astype(np.float32)/255
    out = np.empty_like(f)
    out[DROP:] = f[:1920 - DROP]
    out[:DROP] = SKY                                            # extend the night sky upward (UI zone), fixed colour
    fe = np.linspace(0, 1, 50)[:, None, None]                    # 50 px feather into the footage
    out[DROP:DROP + 50] = SKY*(1 - fe) + out[DROP:DROP + 50]*fe
    return np.clip(out*WB, 0, 1)
def real_at(t):
    if t < G('BACK0'): im = foot_frame(SEG_A, round(t*FPS))
    else:
        im = foot_frame(SEG_B, round((t - G('BACK0'))*FPS))
        tl = t - (T - 0.45)
        if tl > 0: u = sm(tl/0.45); im = im*(1 - u) + foot_frame(SEG_A, 0)*u          # seamless loop to frame 1
    if 0.8 < t < 5.6:                                           # J2 subject ring: the sky full of colour
        a = sm((t-0.8)/0.3)*(1-sm((t-5.2)/0.4)); pul = 1 + 0.02*np.sin(t*4.2); ov = np.zeros_like(im)
        cv2.ellipse(ov, (520, 720), (int(340*pul), int(360*pul)), 0, 0, 360, tuple(v/255*a for v in CYAN), 5, cv2.LINE_AA)
        im = im + ov*1.4
    return im*BAND

# ---------------- painting (PD) ----------------
ART = cv2.imread('art/0.jpg')[:, :, ::-1].copy()              # 1920 x 1487
def painting_at(t):
    a, b = G('P1'), G('SCAN0') + 1.0
    u = sm((t - a)/(b - a))
    h = 1487*(0.62 - 0.22*u); w = h*1080/1920                  # push in to the gold sparks (top left)
    cx = 470 + 120*u; cy = 330 + 30*u
    x0 = int(np.clip(cx - w/2, 0, 1920 - w)); y0 = int(np.clip(cy - h/2, 0, 1487 - h))
    sub = ART[y0:y0+int(h), x0:x0+int(w)]
    im = cv2.resize(sub, (1080, 1920), interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    t0 = G('P1') + 0.6
    if t0 < t < G('SCAN0') - 0.2:                               # J2 ring on the painted fireworks (all gold)
        a = sm((t-t0)/0.3)*(1-sm((t-G('SCAN0')+0.6)/0.4)); pul = 1 + 0.02*np.sin(t*4.2)
        ov = np.zeros_like(im)
        cv2.ellipse(ov, (540, 720), (int(400*pul), int(330*pul)), 0, 0, 360, tuple(v/255*a for v in CYAN), 5, cv2.LINE_AA)
        im = im + ov*1.4
    return im*BAND, (x0, y0, w, h)

# ---------------- x-ray layer ----------------
NIGHT = K.NIGHT/255
def T_at(t):                                         # the code runs: temperature ramps (slow first, then x10)
    a, b, c = G('HEAT0'), G('HEAT1'), G('HEAT2')
    if t < a: return 1500.0
    if t < b: return 1500 + 800*sm((t - a)/(b - a))                       # 1500 -> 2300 K slow: charcoal gold
    u = sm((t - b)/(c - b)); return float(np.exp(np.log(2300) + u*(np.log(100000) - np.log(2300))))
def dE_at(t):                                        # the code runs: energy step (eV) per molecule colour
    pts = [(G('Q0'), 2.0), (G('Q_RED'), 2.0), (G('Q_GRN') - 0.4, 2.0), (G('Q_GRN'), 2.36), (G('Q_BLU') - 0.4, 2.36), (G('Q_BLU'), 2.9),
           (G('Q_UV') - 0.3, 2.9), (G('Q_UV') + 0.4, 3.5), (G('Q_END') - 0.5, 3.5), (G('Q_END'), 2.9)]
    ts, vs = zip(*pts); return float(np.interp(t, ts, vs))
CU_LO, CU_HI = 1000.0, 1200.0                         # CuCl blue window, deg C (Conkling; ScienceABC/NOVA)
def flame_at(t):                                     # the ONE number on the dial: flame temperature (deg C)
    pts = [(G('DIAL0'), 1100), (G('DIAL1'), 1100), (G('DIAL2'), 1700), (G('DIAL3'), 1700), (G('DIAL4'), 1150)]
    ts, vs = zip(*pts); return float(np.interp(t, ts, vs))
def cucl_frac(Tc):                                   # fraction of CuCl surviving: smooth step at the cited 1200 C limit
    return float(1/(1 + np.exp((Tc - CU_HI)/40)))
def spectrum_strip(img, y0, y1, x0=90, x1=930):
    for x in range(x0, x1):
        lam = 380 + (x - x0)/(x1 - x0)*400
        c, _ = SP.line_rgb(lam); img[y0:y1, x] = c*0.85
def lam_x(lam, x0=90, x1=930): return x0 + (lam - 380)/400*(x1 - x0)
SPEC_STRIP = np.zeros((1920, 1080, 3), np.float32); spectrum_strip(SPEC_STRIP, 1120, 1150)

def xray_at(t):
    img = np.ones((1920, 1080, 3), np.float32)*NIGHT
    ov = Image.new('RGB', (1080, 1920)); d = ImageDraw.Draw(ov)
    heat = t < G('Q0')
    if heat:
        Tk = T_at(t); STATE['T'] = Tk
        col = SP.planck_rgb(Tk)
        # Planck curve on the visible strip (normalised)
        lam = np.linspace(380, 780, 300); B = SP.planck(Tk)[(lam - 300).astype(int)]
        B = B/B.max(); pts = [(lam_x(l), 1100 - 330*v) for l, v in zip(lam, B)]
        d.line(pts, fill=CYAN, width=5)
        # hot spark
        r = 120
        sp = np.zeros((1920, 1080, 3), np.float32); cv2.circle(sp, (540, 690), r, tuple(float(v) for v in col), -1, cv2.LINE_AA)
        sp = sp + cv2.GaussianBlur(sp, (0, 0), 40)*1.2
        img = 1 - (1 - img)*np.exp(-sp*1.2)
        img = np.maximum(img, SPEC_STRIP)
    elif t < G('DIAL0'):
        dE = dE_at(t); STATE['dE'] = dE; lam = HC/dE; STATE['lam'] = lam
        c, Y = SP.line_rgb(lam); vis = 380 <= lam <= 780
        y_hi, y_lo = 520, 520 + int(150*dE)                 # energy ladder: gap grows with dE
        img = np.maximum(img, SPEC_STRIP)
        sp = np.zeros((1920, 1080, 3), np.float32)
        ph = (t*1.4) % 1.0                                  # electron drops every 0.7 s
        ex = 300; ey = y_hi + (y_lo - y_hi)*sm(ph/0.35) if ph < 0.35 else y_lo
        if vis:
            for k in range(3):                              # outgoing photon wave packets
                s = (ph - 0.35 + k*0.33) % 1.0
                xs = np.linspace(0, 260, 120); px = ex + 70 + s*480 + xs
                py = y_lo - 40 + 18*np.sin(xs/lam*2*np.pi*8)*np.exp(-((xs - 130)/70)**2)
                cv2.polylines(sp, [np.stack([px, py], 1).astype(np.int32)*4], False, tuple(float(v)*1.4 for v in c), 10, cv2.LINE_AA, shift=2)
            cv2.circle(sp, (int(lam_x(lam)), 1135), 22, tuple(float(v)*1.5 for v in c), -1, cv2.LINE_AA)
        img = 1 - (1 - img)*np.exp(-(sp + cv2.GaussianBlur(sp, (0, 0), 14)*1.5))
        dd = ImageDraw.Draw(ov)
        dd.line([(150, y_hi), (520, y_hi)], fill=CYAN, width=6); dd.line([(150, y_lo), (520, y_lo)], fill=CYAN, width=6)
        dd.line([(560, y_hi), (560, y_lo)], fill=AMBER, width=4)
        dd.text((585, (y_hi + y_lo)/2), f'ΔE = {dE:.2f} eV', font=F_MONOG, fill=AMBER, anchor='lm')
        dd.ellipse([ex - 18, ey - 18, ex + 18, ey + 18], fill=CYAN)
        if not vis:
            dd.text((540, 1075), f'{lam:.0f} nm = UV: invisible to us', font=F_MONOG, fill=RED, anchor='mm')
    else:                                            # dial: flame temperature on the blue CuCl molecule
        Tc = flame_at(t); STATE['Tc'] = Tc; f = cucl_frac(Tc)
        blue = np.array([0.12, 0.32, 1.0])
        hot = SP.planck_rgb(Tc + 273.15)
        col = blue*f + hot*(1 - f)*0.9
        sep = (1 - f)*230                                  # bond breaks: atoms fly apart
        cu = (540 - 85 - sep, 760); cl = (540 + 85 + sep*0.8, 760 - sep*0.25)
        sp = np.zeros((480, 270, 3), np.float32)
        cv2.circle(sp, (135, 190), 75, tuple(float(v)*0.55 for v in col), -1, cv2.LINE_AA)
        sp = cv2.resize(cv2.GaussianBlur(sp, (0, 0), 17.5)*1.4, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        img = 1 - (1 - img)*np.exp(-sp)
        dd = ImageDraw.Draw(ov)
        if f > 0.15: dd.line([cu, cl], fill=CYAN, width=8)
        else: dd.line([cu, cl], fill=RED, width=3)
        dd.ellipse([cu[0]-62, cu[1]-62, cu[0]+62, cu[1]+62], fill=(10, 16, 34), outline=CYAN, width=6); dd.text(cu, 'Cu', font=K.F_SUB, fill=WHITE, anchor='mm')
        dd.ellipse([cl[0]-52, cl[1]-52, cl[0]+52, cl[1]+52], fill=(10, 16, 34), outline=CYAN, width=6); dd.text(cl, 'Cl', font=K.F_SUB, fill=WHITE, anchor='mm')
        # thermometer bar 800-1800 C with the blue window
        x0, x1, yb = 120, 920, 1120
        X = lambda T: x0 + (T - 800)/1000*(x1 - x0)
        dd.rounded_rectangle([x0, yb-14, x1, yb+14], 14, outline=WHITE, width=3)
        dd.rectangle([X(CU_LO), yb-12, X(CU_HI), yb+12], fill=(40, 90, 255))
        dd.polygon([(X(Tc)-16, yb-40), (X(Tc)+16, yb-40), (X(Tc), yb-16)], fill=AMBER)
        for T_, lab in ((1000, '1000'), (1200, '1200'), (1700, '1700 °C')): dd.text((X(T_), yb+46), lab, font=F_MONOG, fill=WHITE, anchor='mm')
        dd.text(((X(CU_LO)+X(CU_HI))/2, yb-58), 'blue window', font=F_MONOG, fill=(120, 160, 255), anchor='mm')
    ova = np.asarray(ov).astype(np.float32)/255
    img = np.clip(img + ova, 0, 1)
    for lm in (400, 500, 600, 700):
        pass
    return img

# ---------------- text layer ----------------
CAPS = []                                            # filled from timeline.json
def eq_draw(d, y, a):
    c = lambda col: tuple(int(v*a) for v in col)
    s1 = 'λ'; s2 = ' = hc / '; s3 = 'ΔE'
    w = sum(d.textlength(s, font=F_EQG) for s in (s1, s2, s3)); x = CX - w/2
    for s, col in ((s1, CYAN), (s2, WHITE), (s3, AMBER)):
        d.text((x, y), s, font=F_EQG, fill=c(col), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s, font=F_EQG)
def text(t, im):
    d = ImageDraw.Draw(im)
    hook = None
    if t < G('P1') or t >= G('BACK1'): hook = 'Diwali fireworks, decoded'
    elif t < G('SCAN0'): hook = 'Kishangarh, India · late 1700s'
    K.title_box(d, "WHERE'S THE BLUE?", hook, y=205)
    K.series_tag(d, 5)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, y=1500, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 0.6 < t < G('P1') - 0.2: K.pill(d, (CX, 1365), 'count the blue: 0', CYAN, font=K.F_MONO)
    if 0.6 < t < G('P1') - 0.2: d.text((CX, 1420), 'real Diwali footage', font=K.F_MONO, fill=(200, 200, 200), anchor='mm', stroke_width=2, stroke_fill=(0, 0, 0))
    if G('P1') + 0.3 < t < G('SCAN0'):
        K.pill(d, (CX, 1365), 'Sitaram · National Museum, New Delhi · PD', WHITE, font=K.F_MONO)
    if G('SCAN1') < t < G('Q0'):
        Tk = STATE.get('T', 1500)
        d.text((90, 470), "solver: Planck's law → CIE → colour", font=F_MONOG, fill=CYAN, anchor='ls')
        K.pill(d, (CX, 900), f'T = {Tk:,.0f} K', AMBER, font=K.F_CHIP)
        lab = 'charcoal spark' if Tk < 2600 else ('like the Sun' if Tk < 9000 else 'still not blue')
        if Tk > 5000 and Tk < 7000: lab = 'Sun: white'
        K.pill(d, (CX, 980), lab, RED if Tk > 30000 else WHITE, font=K.F_MONO)
        if t > G('HEAT1'): K.pill(d, (700, 1240), 'sped up x10', AMBER, font=K.F_MONO)
        for lm in (400, 500, 600, 700): d.text((lam_x(lm), 1190), f'{lm}', font=K.F_MONO, fill=WHITE, anchor='mm')
    if G('Q0') <= t < G('DIAL0'):
        d.text((90, 470), 'solver: λ = 1239.84 / ΔE  (nm)', font=F_MONOG, fill=CYAN, anchor='ls')
        for lm in (400, 500, 600, 700): d.text((lam_x(lm), 1190), f'{lm}', font=K.F_MONO, fill=WHITE, anchor='mm')
    if G('Q0') <= t < G('BACK0'):
        lam = STATE.get('lam', 620)
        if G('Q0') + 0.3 < t < G('EQ0'):
            K.pill(d, (CX, 1290), 'red SrCl · green BaCl · blue CuCl', WHITE, font=F_MONOG)
        if G('EQ0') <= t < G('DIAL0'):
            a = sm((t - G('EQ0'))/0.35); eq_draw(d, 1290, a)
            dE = STATE.get('dE', 2.0)
            if t > G('Q_RED') - 0.2: K.pill(d, (CX, 1395), f'ΔE = {dE:.2f} eV  →  λ = {lam:.0f} nm', CYAN, font=F_MONOG)
        if G('DIAL0') <= t < G('BACK0'):
            Tc = STATE.get('Tc', 1100)
            c = AMBER if t < G('DIAL1') else (GREEN if (t > G('DIAL3') and cucl_frac(Tc) > 0.5) else RED)
            K.pill(d, (CX, 1290), f'dial:  flame = {Tc:,.0f} °C', c, font=F_CHIPG)
            if cucl_frac(Tc) < 0.2: K.pill(d, (CX, 560), 'CuCl broken → blue gone', RED, font=F_MONOG)
            if t > G('DIAL3') and cucl_frac(Tc) > 0.5: K.pill(d, (CX, 560), 'kept cool → blue', GREEN, font=F_MONOG)
    a = sm((t - G('STAMP0'))/0.4)*(1 - sm((t - T + 0.45)/0.3))
    if a > 0:
        K.end_stamp(d, alpha=a, y=1365)
        K.pill(d, (CX, 1290), 'sources: NIST · Conkling · PBS NOVA', WHITE, font=F_MONOG) if t < T - 1.0 else None
STATE = {}

def frame(t):
    if t < G('P1') - 0.2 or t > G('BACK1'): L = real_at(t)
    elif t < G('P1') + 0.2:
        u = sm((t - G('P1') + 0.2)/0.4); L = real_at(t)*(1 - u) + painting_at(t)[0]*u
    elif t < G('SCAN1'): L = painting_at(t)[0]
    elif t < G('BACK0'): L = None
    else: L = real_at(t)
    X = xray_at(t) if G('SCAN0') <= t <= G('BACK1') else None
    if L is not None and X is not None:
        if t < G('SCAN1'): yl = int(sm((t - G('SCAN0'))/(G('SCAN1') - G('SCAN0')))*1920)
        else: yl = int((1 - sm((t - G('BACK0'))/(G('BACK1') - G('BACK0'))))*1920)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(1920) - yl)/6)**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(1920) - yl)/60)**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    real = X is None
    img = np.clip(img*(VIG if real else 1) + GRAIN[int(t*FPS) % 6]*(0.010 if real else 0.003), 0, 1)
    pim = Image.fromarray((img*255).astype(np.uint8)); text(t, pim); return pim

def load_caps():
    global CAPS
    def seg(s):                                       # "*word*" = yellow key word
        out = []
        for i, p in enumerate(s.split('*')):
            if p: out.append((p, YELLOW if i % 2 else WHITE))
        return out
    CAPS = [(a, b, [seg(l) for l in lines]) for a, b, lines in TL['caps']]
load_caps()

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    if mode == 'sheet':
        for u in [float(x) for x in sys.argv[2].split(',')]: frame(u).save(f'f_{u:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS).tobytes())
        p.stdin.close(); p.wait(); open(f'part_{a:05d}.done', 'w').write('ok')
