# SOURCE CODE #4 - "STONES THAT FLOAT?" (Coded Cosmos, build 6 Nov 2026)
# real thing: NASA Earth Observatory Landsat 8 OLI image, pumice raft near Tonga, 13 Aug 2019 (public domain)
# x-ray: pumice cross-section; solver = buoyancy ODE  z'' = g (rho_w * Vsub/V / rho_b - 1) - drag,
#        rho_b = (1 - phi) rho_rock + phi * f * rho_w   (f = fraction of pores flooded, diffusion front r_f ~ sqrt(t))
import numpy as np, cv2, subprocess, sys, glob, os
from PIL import Image, ImageDraw, ImageFont
import source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE

FPS = 30; T = 53.8; S = 2; W2, H2 = K.W*S, K.H*S
RNG = np.random.default_rng(4)
RHO_R, RHO_W = 2.5, 1.025                        # pumice glass ~2.4-2.5 g/cm3; sea water 1.025
PHI0, PHI_LO, PHI_HI = 0.72, 0.45, 0.75
# v2 timeline (owner 6 Nov: tell the Ramayana story with real art, show Ram Setu today, then the pumice x-ray)
P1, P2, RS, RAFT2 = 2.95, 8.3, 12.18, 22.36                 # painting 1, painting 2, Ram Setu (NASA ASTER), back to raft
SCAN0, SCAN1 = 24.5, 25.4
FORM0, COOL0, COOL1, DROPT = 25.6, 29.4, 30.5, 30.7
HIT = 32.0; EQ_END = 39.6
LESS, MORE = 39.83, 41.8
SOAK0, SOAK1 = 43.79, 47.6
BACK0, BACK1 = 47.9, 48.7
ENDP = 50.25; STAMP0 = 50.3
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)

_dj = [p for p in glob.glob('/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf') + glob.glob('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')]
F_GR = ImageFont.truetype(_dj[0], 64); F_GRS = ImageFont.truetype(_dj[0], 40)
F_SUBL = ImageFont.truetype('Poppins-ExtraBold.ttf', 30)

# ================= real layer: NASA plate =================
PLATE = cv2.imread('tonga_oli_2019225_lrg.jpg')[:, :, ::-1].copy()          # 7701 x 7811, uint8 RGB
RCX, RCY = 3390, 3519                                                         # raft centre (plate px)
def crop_h(t):
    if t < 30: return 3100 - 250*sm(t/SCAN1)
    return 2850 + 250*sm((t-ENDP)/(T-0.2-ENDP))                             # zoom back out -> last frame == frame 1
def plate_at(t):
    h = crop_h(t); w = h*1080/1920
    top = RCY - 0.461*h; left = RCX - 0.4537*w
    x0, y0 = int(np.floor(left)), int(np.floor(top)); x1, y1 = int(np.ceil(left + w)), int(np.ceil(top + h))
    sub = PLATE[y0:y1, x0:x1]; sx, sy = 1080/w, 1920/h
    im = cv2.resize(sub, (int(round((x1-x0)*sx)), int(round((y1-y0)*sy))), interpolation=cv2.INTER_AREA if h > 1920 else cv2.INTER_LANCZOS4)
    ox, oy = int(round((left-x0)*sx)), int(round((top-y0)*sy)); im = im[oy:oy+1920, ox:ox+1080]
    if im.shape[:2] != (1920, 1080): im = cv2.resize(im, (1080, 1920), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32)/255, (h, top, left)
YY, XX = np.mgrid[0:1920, 0:1080].astype(np.float32)
VIG = (1 - 0.32*(((XX-540)/1080)**2 + ((YY-960)/1920)**2)*2)[..., None]
GRAIN = [RNG.normal(0, 1, (1920, 1080, 1)).astype(np.float32) for _ in range(6)]
BAND = (1 - 0.42*np.exp(-((YY-1515)/115)**2))[..., None]                     # J4 soft dark band behind captions
def real_at(t):
    im, (h, top, left) = plate_at(t)
    im = np.clip((im - 0.02)*1.12, 0, 1)**0.95                               # gentle contrast, keep true colours
    if 0.9 < t < 6.6:                                                         # J2 subject ring around the raft
        a = sm((t-0.9)/0.3)*(1-sm((t-6.2)/0.4)); k = 1920/h
        cx, cy = (RCX - left)*k, (RCY - top)*k; pul = 1 + 0.02*np.sin(t*4.2)
        ov = np.zeros_like(im)
        cv2.ellipse(ov, (int(cx), int(cy)), (int(440*k*pul), int(790*k*pul)), 6, 0, 360, tuple(v/255*a for v in CYAN), 5, cv2.LINE_AA)
        im = im + ov*1.4
    return im*BAND

# ================= story layer: real art + Ram Setu today =================
def _load(p): return cv2.imread(p)[:, :, ::-1].copy()
ART1, ART2, ASTER = _load('dl/monkeys.jpg'), _load('dl/nivedita.jpg'), np.clip((_load('dl/aster.jpg').astype(np.float32)-14)*1.22, 0, 255).astype(np.uint8)
def _bg(img):
    b = cv2.resize(img, (1080, int(1080*img.shape[0]/img.shape[1])), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (1080, 1920), interpolation=cv2.INTER_LINEAR); b = cv2.GaussianBlur(b, (0, 0), 40)
    return b.astype(np.float32)/255*0.28
BG1, BG2, BG3 = _bg(ART1), _bg(ART2), _bg(ASTER)
CARD_CY = 905
def card(src, bg, t0, t1, t, zoom=(1.0, 1.0), pan=None, ch=None):
    u = sm((t-t0)/(t1-t0)); z = zoom[0] + (zoom[1]-zoom[0])*u
    h0, w0 = src.shape[:2]
    if pan is not None:                                                   # native crop window that pans (no scaling)
        x0 = int(pan[0] + (pan[1]-pan[0])*u); y0 = pan[2]; sub = src[y0:y0+ch, x0:x0+1080]; im = sub
    else:
        cw, chh = w0/z, h0/z; cx, cy = w0/2, h0/2
        sub = src[int(cy-chh/2):int(cy+chh/2), int(cx-cw/2):int(cx+cw/2)]
        H_ = int(round(1080*sub.shape[0]/sub.shape[1]))
        im = cv2.resize(sub, (1080, H_), interpolation=cv2.INTER_AREA if sub.shape[1] > 1080 else cv2.INTER_LANCZOS4)
    out = bg.copy(); y0c = CARD_CY - im.shape[0]//2
    out[y0c:y0c+im.shape[0]] = im.astype(np.float32)/255
    out[y0c-3:y0c] = 0.55; out[y0c+im.shape[0]:y0c+im.shape[0]+3] = 0.55        # thin frame lines
    return out, y0c
AST_PAN = (560, 640, 540); AST_H = 676
def aster_at(t):
    im, y0c = card(ASTER, BG3, RS, RAFT2, t, pan=AST_PAN, ch=AST_H)
    if 15.4 < t < RAFT2 - 0.1:                                          # J2 ring around the shoal chain
        u = sm((t-RS)/(RAFT2-RS)); x0 = AST_PAN[0] + (AST_PAN[1]-AST_PAN[0])*u
        a = sm((t-15.4)/0.3)*(1-sm((t-RAFT2+0.5)/0.4)); pul = 1 + 0.02*np.sin(t*4.2)
        cx, cy = 1180 - x0, 828 - AST_PAN[2] + y0c
        ov = np.zeros_like(im)
        cv2.ellipse(ov, (int(cx), int(cy)), (int(500*pul), int(85*pul)), 18, 0, 360, tuple(v/255*a for v in CYAN), 5, cv2.LINE_AA)
        im = im + ov*1.4
    return im
def scene(t):
    if t < P1: return real_at(t)
    if t < P2: return card(ART1, BG1, P1, P2, t, zoom=(1.0, 1.025))[0]
    if t < RS: return card(ART2, BG2, P2, RS, t, zoom=(1.0, 1.12))[0]
    if t < RAFT2: return aster_at(t)
    if t < 47.0: return real_at(t)
    if t < ENDP: return card(ART1, BG1, BACK0, ENDP, t, zoom=(1.025, 1.0))[0]
    return real_at(t)
CUTS = [P1, P2, RS, RAFT2, ENDP]
def scene_x(t):                                                         # 0.35 s crossfades
    for c in CUTS:
        if abs(t-c) < 0.175:
            a = (t-(c-0.175))/0.35; return scene(c-0.18)*(1-a) + scene(c+0.18)*a
    return scene(t)

# ================= the stone: pores (Delesse: area fraction of a slice = volume fraction) =================
NP = 240
ang = RNG.uniform(0, 2*np.pi, NP); rad = np.sqrt(RNG.uniform(0, 1, NP))*0.9
PC = np.stack([rad*np.cos(ang), rad*np.sin(ang)], 1)                           # unit-stone coords
PR0 = RNG.lognormal(-3.0, 0.45, NP)
def edge_r(th): return 1 + 0.07*np.sin(3*th+1) + 0.05*np.sin(5*th+2) + 0.03*np.sin(9*th+.5)
G = np.linspace(-1.2, 1.2, 420); GX, GY = np.meshgrid(G, G)
INSIDE = np.hypot(GX, GY) < edge_r(np.arctan2(GY, GX))
def union_frac(s, grow=1.0):
    m = np.zeros_like(INSIDE)
    for (x, y), r in zip(PC, PR0*s*grow): m |= (GX-x)**2 + (GY-y)**2 < r*r
    return (m & INSIDE).sum()/INSIDE.sum()
def solve_s(phi):
    lo, hi = 0.5, 12
    for _ in range(26):
        mid = (lo+hi)/2
        if union_frac(mid) < phi: lo = mid
        else: hi = mid
    return (lo+hi)/2
S_TAB = {p: solve_s(p) for p in (PHI_LO, PHI0, PHI_HI)}
PHI_KEYS = np.array(sorted(S_TAB)); S_KEYS = np.array([S_TAB[p] for p in PHI_KEYS])
BORN = RNG.permutation(NP)/NP
DIST = np.array([edge_r(np.arctan2(y, x)) - np.hypot(x, y) for x, y in PC])     # depth of each pore below the surface

def phi_at(t):                                                                # the dial (amber driver)
    if t < LESS: return PHI0
    if t < MORE: return PHI0 + (PHI_LO-PHI0)*sm((t-LESS)/0.6)
    return PHI_LO + (PHI_HI-PHI_LO)*sm((t-MORE)/0.6)
def front_at(t):                                                              # water diffusion front depth, r_f = sqrt(2 D t)
    if t < SOAK0: return -1
    return np.sqrt(2*0.004*(t-SOAK0)*12)                                      # sped-up time; chip says so
def flooded(t): return DIST < front_at(t)
def f_at(t):
    fl = flooded(t); a = (PR0**2)
    return float((a*fl).sum()/a.sum())

# ---- buoyancy ODE (stone ~ sphere radius R, unit R; depth h of centre below waterline, +down) ----
RPX = 175; WATER_Y = 930; FLOOR_H = (1150 - WATER_Y)/RPX
def vsub(h): d = np.clip(h + 1, 0, 2); return d*d*(3-d)/4
def rho_b(t): p = phi_at(t); return (1-p)*RHO_R + p*f_at(t)*RHO_W
def integrate():
    dt = 1/600; ts = np.arange(0, T+0.01, dt); H = np.zeros_like(ts)
    h, v = (640 - WATER_Y)/RPX, 0.0; g = 7.0
    for i, t in enumerate(ts):
        if t >= DROPT:
            fs = vsub(h); rb = rho_b(t)
            a = g*(1 - RHO_W*fs/rb) - (2.2*fs + 0.05)*v - 1.2*fs*v*abs(v)
            v += a*dt; h += v*dt
            if h > FLOOR_H: h, v = FLOOR_H, -0.15*v
        H[i] = h
    return ts, H
TS, HS = integrate()
def depth(t): return float(np.interp(t, TS, HS))

# ================= x-ray layer =================
YY2 = np.arange(H2, dtype=np.float32)
def xray_at(t):
    img = np.zeros((H2, W2, 3), np.float32); img[:] = K.NIGHT/255
    wy = WATER_Y*S
    img[wy:] += np.array([0.02, 0.06, 0.12], np.float32)*(1 + 0.3*np.exp(-(YY2[wy:]-wy)/(400*S)))[:, None, None]
    for k in range(9):                                                        # faint depth lines
        y = int(wy + (60 + k*55)*S)
        if y < 1300*S: cv2.line(img, (60*S, y), (W2-60*S, y), (0.06, 0.16, 0.24), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 1300*S), (W2, 1306*S), (0.10, 0.22, 0.30), -1)   # sea floor
    xs = np.arange(0, W2, 4)
    ys = wy + (5*np.sin(xs/(70*S) + t*2.1) + 3*np.sin(xs/(31*S) - t*3.3))*S
    cv2.polylines(img, [np.stack([xs, ys], 1).astype(np.int32)], False, tuple(v/255 for v in CYAN), 2*S, cv2.LINE_AA)
    # stone
    p = phi_at(t); s = float(np.interp(p, PHI_KEYS, S_KEYS))
    grow = 1.0
    if t < COOL0: grow = sm((t-FORM0)/(COOL0-FORM0-0.6))**0.5 if t > FORM0 else 0.0     # bubbles nucleate + grow
    born = BORN
    cyc = 540*S; cyy = (WATER_Y + depth(t)*RPX)*S; R = RPX*S
    hot = 1 - sm((t-COOL0)/(COOL1-COOL0))
    th = np.linspace(0, 2*np.pi, 240); er = edge_r(th)
    poly = np.stack([cyc + R*er*np.cos(th), cyy + R*er*np.sin(th)], 1).astype(np.int32)
    rock = np.array([0.16, 0.19, 0.24])*(1-hot) + np.array([1.4, 0.45, 0.08])*hot
    cv2.fillPoly(img, [poly], tuple(rock.tolist()), cv2.LINE_AA)
    fl = flooded(t)
    for i in range(NP):
        gi = grow if t < COOL0 else 1.0
        if t < COOL0 and t > FORM0: gi = sm((t-FORM0-born[i]*3.0)/1.2)
        r = PR0[i]*s*gi*R
        if r < 0.6: continue
        c = (int(cyc + PC[i, 0]*R), int(cyy + PC[i, 1]*R))
        if fl[i]: fillc, line = (0.08, 0.30, 0.55), tuple(v/255*0.6 for v in CYAN)
        else:
            fillc = tuple((np.array([0.012, 0.018, 0.035])*(1-hot) + np.array([0.9, 0.8, 0.3])*hot).tolist())
            line = tuple((np.array(CYAN)/255*(1-hot) + np.array([1.6, 1.3, 0.5])*hot).tolist())
        cv2.circle(img, c, int(r), fillc, -1, cv2.LINE_AA)
        cv2.circle(img, c, int(r), line, max(1, int(1.2*S)), cv2.LINE_AA)
    mask = np.zeros((H2, W2), np.uint8); cv2.fillPoly(mask, [poly], 1)
    cv2.polylines(img, [poly], True, tuple(v/255*1.6 for v in CYAN), 3*S, cv2.LINE_AA)
    if hot > 0:                                                               # molten glow
        sm_ = cv2.GaussianBlur(cv2.resize(mask.astype(np.float32), (W2//8, H2//8), interpolation=cv2.INTER_AREA), (0, 0), 40*S/8); gl = cv2.resize(sm_, (W2, H2))[..., None]*np.array([1.0, 0.35, 0.05])*hot*0.9
        img += gl
    # forces at equilibrium (code runs): weight down (white), push of water up (cyan)
    if HIT - 6.0 < t < EQ_END + 4.5 and t > DROPT + 1.2:
        fs = vsub(depth(t)); rb = rho_b(t); up = RHO_W*fs; dn = rb
        a = min(1, (t-DROPT-1.2)/0.4)
        cv2.arrowedLine(img, (cyc+R+50*S, int(cyy)), (cyc+R+50*S, int(cyy - up*110*S)), tuple(v/255*a*1.5 for v in CYAN), 5*S, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(img, (cyc-R-50*S, int(cyy)), (cyc-R-50*S, int(cyy + dn*110*S)), tuple(v/255*a*1.3 for v in WHITE), 5*S, cv2.LINE_AA, tipLength=0.22)
    return img, poly

# ================= text layer =================
CAPS = [(0.3, P1, [[('this is ', WHITE), ('ROCK', YELLOW), ('.', WHITE)], [('floating.', WHITE)]]),
        (P1, P2, [[('in the ', WHITE), ('Ramayana', YELLOW), (", Rama's army", WHITE)], [('must cross the sea to ', WHITE), ('Lanka', YELLOW)]]),
        (P2, RS, [[('Nala builds a ', WHITE), ('bridge', YELLOW)], [('and the stones ', WHITE), ('float', YELLOW)]]),
        (RS, 16.8, [[('today, satellites still see', WHITE)], [('Ram Setu', YELLOW)]]),
        (16.8, 19.9, [[('old records: a ', WHITE), ('storm', YELLOW)], [('broke it in 1480', WHITE)]]),
        (19.9, RAFT2, [[("now it's almost all", WHITE)], [('underwater', YELLOW)]]),
        (RAFT2, 25.5, [[('but floating stones', WHITE)], [('are ', WHITE), ('REAL', YELLOW)]]),
        (25.5, 28.85, [[('pumice', YELLOW), (': volcanic lava', WHITE)], [('full of gas', WHITE)]]),
        (28.85, HIT, [[('bubbles ', WHITE), ('freeze in', YELLOW)], [("it's ", WHITE), ('mostly air', YELLOW)]]),
        (HIT, 37.6, [[('rock, minus the ', WHITE), ('air', YELLOW)], [('= lighter than water', WHITE)]]),
        (37.6, LESS, [[('lighter than water?', WHITE)], [('it ', WHITE), ('floats', GREEN)]]),
        (LESS, 41.75, [[('less air:', WHITE)], [('it ', WHITE), ('sinks', RED)]]),
        (41.75, SOAK0, [[('more air:', WHITE)], [('floats ', GREEN), ('again', WHITE)]]),
        (SOAK0, 47.95, [[('water ', WHITE), ('creeps in', YELLOW), ('...', WHITE)], [('one day it ', WHITE), ('sinks', RED)]]),
        (47.95, STAMP0, [[('and faith...', WHITE)], [('is a ', WHITE), ('bridge', YELLOW), (' too', WHITE)]])]
def eq_draw(d, y, a):
    segs = [('ρ', CYAN, F_GR, 0), ('  =  (1 − ', WHITE, K.F_EQ, 0), ('φ', AMBER, F_GR, 0), (') ', WHITE, K.F_EQ, 0),
            ('ρ', WHITE, F_GR, 0), ('rock', WHITE, F_SUBL, 26), ('  <  ', WHITE, K.F_EQ, 0), ('ρ', CYAN, F_GR, 0), ('water', CYAN, F_SUBL, 26)]
    segs[1] = (' = (1 − ', WHITE, F_GR, 0); segs[3] = (') ', WHITE, F_GR, 0); segs[6] = (' < ', WHITE, F_GR, 0)
    tw = sum(d.textlength(s_, font=f) for s_, _, f, _ in segs); x = CX - tw/2
    for s_, c, f, dy in segs:
        d.text((x, y+dy), s_, font=f, fill=tuple(int(v*a) for v in c), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s_, font=f)
def text(t, im):
    d = ImageDraw.Draw(im)
    real = t < SCAN1 or t > BACK1
    hook = None
    if t < P1 or RAFT2 <= t < SCAN0 or t >= ENDP: hook = 'real photo from space · NASA 2019'
    elif t < RS or BACK1 < t < ENDP: hook = 'the Ramayana story'
    elif t < RAFT2: hook = 'Ram Setu today · NASA satellite'
    K.title_box(d, 'STONES THAT FLOAT?', hook, y=205)
    K.series_tag(d, 4)
    cy = 1515 if real else 1490
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, y=cy, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 0.8 < t < P1 - 0.2 or RAFT2 + 0.2 < t < SCAN0: K.pill(d, (CX, 1385), 'floating pumice · Landsat 8, 13 Aug 2019', CYAN, font=K.F_MONO)
    if P1 + 0.2 < t < P2 - 0.2 or BACK1 < t < ENDP - 0.2: K.pill(d, (CX, 1340), 'Indian miniature painting · public domain', WHITE, font=K.F_MONO)
    if P2 + 0.2 < t < RS - 0.2: K.pill(d, (CX, 1340), "'Myths of the Hindus & Buddhists', 1914", WHITE, font=K.F_MONO)
    if RS + 0.2 < t < RAFT2 - 0.2:
        K.pill(d, (CX, 1340), 'India – Sri Lanka · NASA Terra/ASTER', WHITE, font=K.F_MONO)
        if t > 19.9: K.pill(d, (CX, 500), 'ISRO 2024: 29 km, 99.98% under the sea', CYAN, font=K.F_MONO)
    if SCAN1 < t < HIT - 0.3:
        d.text((70, 420), "solver: z'' = g(ρw·Vsub / ρ·V − 1)", font=K.F_MONO, fill=CYAN, anchor='ls')
        st = STATE.get('phi_m', 0)
        d.text((70, 458), f"air in the slice: {st*100:4.1f} %", font=K.F_MONO, fill=AMBER, anchor='ls')
    if HIT <= t < EQ_END:
        a = sm((t-HIT)/0.35); eq_draw(d, 520, a)
        d.text((CX, 600), f'φ = air fraction = {phi_at(t):.2f}   ρ = {rho_b(t):.2f} g/cm³', font=K.F_MONO, fill=tuple(int(v*a) for v in WHITE), anchor='mm')
    elif EQ_END <= t < BACK0:
        K.pill(d, (CX, 560), f'ρ = (1−φ)·2.5 = {rho_b(t):.2f}  vs water 1.025', CYAN, font=K.F_MONO)
    if EQ_END <= t < SOAK0:
        p = phi_at(t); c = AMBER if t < LESS + 0.1 else (RED if t < MORE + 0.1 else GREEN)
        K.pill(d, (CX, 440), f'dial:  air φ = {p:.2f}', c, font=K.F_MONO)
    if SOAK0 <= t < BACK0:
        K.pill(d, (CX, 440), f'water in the bubbles: {f_at(t)*100:3.0f} %', CYAN, font=K.F_MONO)
        K.pill(d, (800, 690), 'sped up: real = days to years', AMBER, font=K.F_MONO)
    if 30.7 < t < 31.9: K.pill(d, (CX, 690), 'pumice ≈ 72% air', CYAN, font=K.F_MONO)
    a = sm((t-STAMP0)/0.4)*(1-sm((t-T+0.45)/0.3))
    if a > 0:
        K.end_stamp(d, alpha=a, y=1440)
        d.text((CX, 1515), 'Happy Diwali', font=K.F_CAP, fill=tuple(int(v*a) for v in YELLOW), anchor='mm', stroke_width=6, stroke_fill=(0, 0, 0))
STATE = {}
from functools import lru_cache
@lru_cache(None)
def UF(p): return union_frac(float(np.interp(p, PHI_KEYS, S_KEYS)))

def frame(t):
    L = scene_x(t) if (t < SCAN1 or t > BACK0) else None
    X = None
    if SCAN0 <= t <= BACK1:
        X2, poly = xray_at(t)
        if t > FORM0:
            p = phi_at(t); STATE['phi_m'] = UF(round(p, 3)) if t >= COOL0 else p*sm((t-FORM0)/(COOL0-FORM0))
        X = cv2.resize(X2, (1080, 1920), interpolation=cv2.INTER_AREA)
        X = 1 - np.exp(-1.25*X)
    if L is not None and X is not None:
        if t < SCAN1: yl = int(sm((t-SCAN0)/(SCAN1-SCAN0))*1920)
        else: yl = int((1-sm((t-BACK0)/(BACK1-BACK0)))*1920)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(1920)-yl)/6)**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(1920)-yl)/60)**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    img = np.clip(img*VIG + GRAIN[int(t*FPS) % 6]*(0.010 if L is not None else 0.004), 0, 1)
    pim = Image.fromarray((img*255).astype(np.uint8)); text(t, pim); return pim

def music_events():
    ev = [(FORM0 + b*3.0, [0.2*np.sin(i)], 1) for i, b in enumerate(np.sort(RNG.uniform(0, 1, 10)))]
    ev += [(DROPT + 0.5, [0.0], 1), (MORE + 0.3, [0.5], 1), (P1, [0.3], 1), (RS, [-0.3], 1), (15.6, [0.6], 1)]
    return ev

def inv(u): return u

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    if mode == 'sheet':
        for u in [float(x) for x in sys.argv[2].split(',')]: frame(inv(u)).save(f'f_{u:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(inv(i/FPS)).tobytes())
        p.stdin.close(); p.wait(); open(f'part_{a:05d}.done', 'w').write('ok')
