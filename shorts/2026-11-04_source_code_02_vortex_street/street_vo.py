# SOURCE CODE #2 v2 - voice version on REAL NOAA footage (Guadalupe Island, 8 May 2021, public domain)
# timeline is driven by the ElevenLabs narration (Daniel pH8TIDBxKcsLhKFzhwgP, eleven_v4, 1 take, 52.56 s)
import numpy as np, cv2, subprocess, sys, os, json
from PIL import Image, ImageDraw
import street as S0, sim, source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE
FPS = 30; T = 57.5
# VO chunks: (src start, src end, shift) -> video time = src + shift
CUTS = [(0.0, 12.1, 0.3), (12.1, 26.9, 1.3), (26.9, 43.5, 2.9), (43.5, 52.6, 3.7)]
SCAN0, SCAN1 = 12.4, 13.3
TROPHY, DIAL, BREAK, SNAP = 28.125, 41.1, 45.0, 47.38
BACK0 = 49.85; BACK1 = BACK0 + 0.8; J2 = 53.0
def vt(s):                                   # VO source time -> video time
    for a, b, sh in CUTS:
        if s < b: return s + sh
    return s + CUTS[-1][2]
# captions = key words of the spoken line, timed to the narration (src times from silencedetect)
CAPS = [(vt(0.0), vt(2.0), [[('these swirls are ', WHITE), ('real', YELLOW)]]),
        (vt(2.0), vt(4.8), [[('filmed from ', WHITE), ('space', YELLOW)], [('off the coast of Mexico', WHITE)]]),
        (vt(4.8), vt(6.5), [[('the ', WHITE), ('wind', YELLOW), (' drew them?', WHITE)]]),
        (vt(6.5), vt(8.7), [[('nope. ', YELLOW), ('wind blows straight', WHITE)]]),
        (vt(8.7), vt(12.1), [[('the artist: one tiny ', WHITE), ('island', YELLOW)]]),
        (vt(12.2), vt(15.6), [[('the real ', WHITE), ('equations of air', YELLOW)], [('83,000 boxes of it', WHITE)]]),
        (vt(15.6), vt(17.5), [[("air can't hug the island", WHITE)]]),
        (vt(17.5), vt(20.3), [[('peels off ', YELLOW), ('one side...', WHITE)], [('then the other', WHITE)]]),
        (vt(20.3), vt(23.5), [[('left, right, left', YELLOW)], [('forever', WHITE)]]),
        (vt(23.5), TROPHY, [[('a ', WHITE), ('vortex street', YELLOW)]]),
        (TROPHY, vt(29.2), [[('one simple ', WHITE), ('rule', YELLOW)]]),
        (vt(29.2), vt(35.2), [[('swirls per second', WHITE)], [('= 0.2 × wind ÷ size', YELLOW)]]),
        (vt(35.2), DIAL, [[('our solver: ', WHITE), ('0.19', YELLOW)]]),
        (DIAL, vt(42.0), [[('one dial', AMBER)]]),
        (vt(42.0), vt(43.4), [[('air ', WHITE), ('5x thicker', AMBER), ('...', WHITE)]]),
        (vt(43.4), SNAP, [[('the street ', WHITE), ('dies', RED)]]),
        (SNAP, BACK0, [[('normal air: ', WHITE), ('swirls are back', GREEN)]]),
        (BACK0 + 0.2, J2, [[('the island has no idea', WHITE)], [("it's making ", WHITE), ('art', YELLOW)]]),
        (J2, T - 0.45, [[('same maths makes', WHITE)], [('power lines ', YELLOW), ('hum', YELLOW)]])]

# ---------------- real footage layer (NOAA GOES visible, Guadalupe Island, 8 May 2021) ----------------
# plates: native 1:1 UHD crop (rotated 90 deg so the wind blows down), slowed 0.32x with frame blending (ffmpeg framerate)
_pl = {}
def _plate(name):
    if name not in _pl:
        cap = cv2.VideoCapture(name); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        _pl[name] = fr
    return _pl[name]
PB0 = BACK0 - 0.1
YOFF = 150
def grade(f, t):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    m, sd = g.mean(), g.std() + 1e-4                                       # steady exposure across the day-to-day frames
    g = np.clip(0.60 + (g - m)/sd*0.24, 0, 1)
    ocean = np.array([0.03, 0.07, 0.13], np.float32); cloud = np.array([0.97, 0.98, 1.0], np.float32)
    rgb = ocean + (cloud - ocean)*(g[..., None]**1.15)                      # visible channel shown ocean-navy to cloud-white
    z = 1.0 + 0.05*np.sin(np.pi*t/T)**2                                     # gentle push-in on the island (<= 1.05x)
    M = cv2.getRotationMatrix2D((542, 345), 0, z)
    return cv2.warpAffine(rgb, M, (K.W, K.H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
def real1x(t):
    if t < PB0:
        fr = _plate('plateA.mp4'); k = min(int(round(t*FPS)), len(fr)-1)
    else:
        fr = _plate('plateB.mp4'); k = min(int(round((t-PB0)*FPS)), len(fr)-1)
    return grade(fr[k], t)
FCFG = True

def text(t, im):
    d = ImageDraw.Draw(im)
    real = t < SCAN0 or t > BACK0
    K.title_box(d, 'WHO DREW THESE SWIRLS?', y=205)
    K.series_tag(d, 2)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, alpha=min(1, (t-a)/0.12, (b-t)/0.12))
    if t < SCAN0 - 0.3: d.text((90, 1225), 'NOAA satellite · Guadalupe Island · 8 May 2021', font=K.F_MONO, fill=WHITE, anchor='ls', stroke_width=2, stroke_fill=(0, 0, 0))
    if SCAN1 + 0.3 < t < TROPHY - 0.3:
        d.text((90, 1185), 'solver: lattice Boltzmann, 384 x 216', font=K.F_MONO, fill=CYAN, anchor='ls')
        d.text((90, 1225), 'solver: Re = 150', font=K.F_MONO, fill=CYAN, anchor='ls')
    if 21.5 < t < 25.5: K.pill(d, (790, 520), 'x3 faster', AMBER)
    if SCAN1 < t < DIAL:
        n = 0
        for (te, s_) in S0.EVS:
            if te > t: break
            if te < SCAN1 + 0.3: continue
            n += 1; age = t - te
            if age < 3.0:
                spf = S0.order_spf[S0.fidx(te)]
                y = sim.CX0*5 + 2.5*sim.D*5 + age*FPS*spf*sim.U0*0.85*5 - YOFF
                if y < 1150: K.pill(d, (sim.CY0*5 + s_*95, y), f'swirl {n}', CYAN)
    if TROPHY <= t < DIAL:
        a = S0.sm((t-TROPHY)/0.35); y = 1290
        segs = [('f', CYAN), (' = 0.2 · ', WHITE), ('U', AMBER), (' / ', WHITE), ('D', WHITE)]
        tw = sum(d.textlength(s_, font=K.F_EQ) for s_, _ in segs); x = CX - tw/2
        for s_, c in segs:
            d.text((x, y), s_, font=K.F_EQ, fill=tuple(int(v*a) for v in c), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s_, font=K.F_EQ)
        if t > vt(35.2) - 0.2: d.text((90, 1185), f"solver measured: f·D/U = {S0.STJ['St']:.2f}", font=K.F_MONO, fill=CYAN, anchor='ls')
    elif DIAL <= t < BACK0:
        K.pill(d, (CX, 1385), f"f = {S0.STJ['St']:.2f} · U / D", RED if BREAK <= t < SNAP else CYAN, font=K.F_CHIP)
    if DIAL <= t < BACK0:
        v, c = ('x5', AMBER) if t < BREAK else (('x5', RED) if t < SNAP else ('x1', GREEN))
        K.pill(d, (CX, 1290), f'dial: air thickness {v}', c)
        if DIAL + 0.6 < t < SNAP:
            d.text((90, 1185), 'solver at Re = 30: no shedding', font=K.F_MONO, fill=RED if t > BREAK else AMBER, anchor='ls')
            K.pill(d, (790, 590), 'x5 faster', AMBER)
    if t > BACK1: d.text((90, 1225), 'NOAA satellite · Guadalupe Island', font=K.F_MONO, fill=WHITE, anchor='ls', stroke_width=2, stroke_fill=(0, 0, 0))
    if J2 + 0.2 < t < T - 0.6: K.end_stamp(d, alpha=1.0, y=1385)

vy, vx = np.mgrid[0:K.H, 0:K.W].astype(np.float32)
VIG = (1 - 0.22*(((vx-K.W/2)/K.W)**2 + ((vy-K.H/2)/K.H)**2)*2)[..., None]
F0 = None
def frame(t):
    global F0
    L = real1x(t) if (t < SCAN1 or t > BACK0) else None
    X = None
    if SCAN0 <= t <= BACK1:
        X = cv2.resize(np.clip(S0.xray_at(t), 0, 1.3), (K.W, K.H), interpolation=cv2.INTER_AREA)
        X = np.concatenate([X[YOFF:], np.ones((YOFF, K.W, 3), np.float32)*K.NIGHT/255], 0)
        if t < SCAN1 + 0.2 or t > BACK0 - 0.2:                               # ghost of the real clouds under the skeleton (transitions)
            g = real1x(t).mean(2, keepdims=True)
            X = X + g*np.array(CYAN, np.float32)/255*0.10
    if L is not None and X is not None:
        yl = int(S0.sm((t-SCAN0)/(SCAN1-SCAN0))*K.H) if t < SCAN1 else int((1-S0.sm((t-BACK0)/(BACK1-BACK0)))*K.H)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(K.H)-yl)/6.0)**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(K.H)-yl)/60.0)**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    if t > T - 0.6:
        if F0 is None: F0 = real1x(0.0)
        a = S0.sm((t-(T-0.6))/0.6); img = img*(1-a) + F0*a
    im = np.clip(np.clip(img, 0, 1)*VIG + S0.GRAIN[int(t*FPS) % 6]*0.006, 0, 1)
    pim = Image.fromarray((im*255).astype(np.uint8)); text(t, pim); return pim

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sheet':
        for t in [float(x) for x in sys.argv[2].split(',')]: frame(t).save(f'v_{t:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', f'vpart_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS).tobytes())
        p.stdin.close(); p.wait()
