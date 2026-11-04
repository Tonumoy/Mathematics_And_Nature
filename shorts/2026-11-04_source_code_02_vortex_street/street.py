# SOURCE CODE #2 - "WHO DREW THESE SWIRLS?" (Coded Cosmos, 4 Nov 2026)
# real thing: satellite view of a cloud deck behind an island -> x-ray -> the air solver (sim.py, lattice Boltzmann)
# twist: one number (air thickness nu x5 -> Re 150 -> 30) kills the vortex street, nu x1 brings it back.
import numpy as np, cv2, subprocess, sys, json
from PIL import Image, ImageDraw
import sim, source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE

FPS, BPM = 30, 96; BT = 60/BPM; BEATS = 67; T = BEATS*BT                 # 41.875 s
S = 2; W2, H2 = K.W*S, K.H*S; PX = 5*S                                    # 2x canvas px per sim cell
UV = np.load('uv.npy', mmap_mode='r'); CL = np.load('cl.npy', mmap_mode='r')
SCH = json.load(open('sched.json')); STJ = json.load(open('st.json'))
NB, FORK = SCH['nbranch'], SCH['fork']
SCAN0, SCAN1 = 11.25, 12.15; TROPHY = 39*BT
DIAL, BREAK, SNAP, BACK0 = 46*BT, 50*BT, 54*BT, 62*BT; BACK1 = BACK0 + 0.8
J1, J2 = 58*BT, BACK1 + 0.1
CAPS = [(2.4, 5.2, [[('real clouds, seen from ', WHITE), ('space', YELLOW)], [('wind did it. easy, right?', WHITE)]]),
        (5.2, 8.4, [[('nope. ', YELLOW), ('wind blows straight.', WHITE)], [("it's that tiny ", WHITE), ('island', YELLOW)]]),
        (8.4, SCAN0, [[('so what ', WHITE), ('code', YELLOW), (' is running', WHITE)], [('under these clouds?', WHITE)]]),
        (12.2, 15.0, [[('we solve the real air maths', WHITE)], [('83,000', YELLOW), (' tiny boxes of air', WHITE)]]),
        (15.0, 18.1, [[("air can't hug the island.", WHITE)], [('it ', WHITE), ('peels off', YELLOW), (' one side...', WHITE)]]),
        (18.1, 21.2, [[('...then the other side.', WHITE)], [('left, right, left', YELLOW), ('. forever.', WHITE)]]),
        (21.2, TROPHY, [[('nobody tells it to. it is a', WHITE)], [('vortex street', YELLOW)]]),
        (TROPHY, DIAL, [[('swirls per second', WHITE)], [('= 0.2 × wind ÷ size', YELLOW)]]),
        (DIAL, BREAK, [[('dial: make the air', WHITE)], [('5x thicker', AMBER), ('. watch.', WHITE)]]),
        (BREAK, SNAP, [[('the street ', WHITE), ('dies', RED), ('.', WHITE)], [('no swirls at all', WHITE)]]),
        (SNAP, J1, [[('normal air: ', WHITE), ('swirls are back', GREEN)], [('same code. one number.', WHITE)]]),
        (J1, J2, [[('the island has no idea', WHITE)], [("it's making ", WHITE), ('art', YELLOW)]]),
        (J2, T-0.45, [[('same maths makes', WHITE)], [('wires hum', YELLOW), (' in the wind', WHITE)]])]
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
def fidx(t): return min(int(round(t*FPS)), len(UV)-1)
def branch(t): return FORK <= fidx(t) < FORK+NB
# cumulative solver steps per video frame (for texture scroll + honest speed chips)
order_spf = SCH['spf'][:FORK] + [SCH['bspf']]*NB + SCH['spf'][FORK:]
CUM = np.cumsum([0] + order_spf)

# ================= real-life layer: satellite view (2x) =================
RNG = np.random.default_rng(11)
def fnoise(h, w, beta, seed, per_axis=None):
    r = np.random.default_rng(seed); ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.fftfreq(w)[None, :]
    k = np.hypot(kx, ky); k[0, 0] = 1; a = k**(-beta/2); a[0, 0] = 0
    n = np.real(np.fft.ifft2(a*np.exp(2j*np.pi*r.random((h, w))))); return ((n-n.mean())/n.std()).astype(np.float32)
yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
IY, IX = sim.CX0*PX, sim.CY0*PX                                          # island centre on 2x canvas (row, col)
def build_static():
    lo = cv2.resize(fnoise(480, 270, 3.2, 1), (W2, H2), interpolation=cv2.INTER_CUBIC)
    oc = np.stack([0.030, 0.085, 0.150]).astype(np.float32)[None, None]*(1 + 0.10*lo[..., None])
    glint = np.exp(-(((xx-0.75*W2)/(0.6*W2))**2 + ((yy-0.25*H2)/(0.5*H2))**2))[..., None]
    oc = oc + glint*np.array([0.05, 0.07, 0.08], np.float32)
    th = np.arctan2(yy-IY, xx-IX); r = np.hypot(yy-IY, xx-IX)/PX; R = sim.r_island(th)
    hgt = np.clip(1 - r/R, 0, 1)**0.7*(1 + 0.25*cv2.resize(fnoise(256, 256, 2.4, 5), (W2, H2))[..., None][..., 0])
    gy, gx = np.gradient(cv2.GaussianBlur(hgt, (0, 0), 3))
    shade = np.clip(0.75 - 60*(gx + gy), 0.25, 1.4)[..., None]
    land = np.where(hgt[..., None] > 0.55, np.array([0.30, 0.26, 0.20]), np.array([0.16, 0.20, 0.10]))*shade
    m = sm((R - r)/0.5)[..., None]
    surf = np.exp(-((r - R)/0.35)**2)[..., None]*np.array([0.5, 0.55, 0.55])
    base = oc*(1-m) + land*m + surf*(1-m)*0.6
    det = fnoise(1024, 1024, 3.0, 9)                                       # fine cloud detail, tiled + scrolled
    return base.astype(np.float32), det
BASE, DET = build_static()
def detail(t):
    sh = CUM[fidx(t)]*sim.U0*PX                                           # scroll with the free-stream wind (px, 2x)
    rows = ((np.arange(H2) - sh)/1.6).astype(int) % 1024; cols = (np.arange(W2)/1.6).astype(int) % 1024
    return DET[rows][:, cols]
def real_at(t):
    c = cv2.resize(cv2.GaussianBlur(CL[fidx(t)].T.astype(np.float32)/255, (0, 0), 0.9), (W2, H2), interpolation=cv2.INTER_CUBIC)
    c = np.clip(c, 0, 1)*(0.93 + 0.07*detail(t))
    a = sm((c - 0.22)/0.55)
    sl = cv2.GaussianBlur(a, (0, 0), 6)
    lit = np.clip(1.0 + 1.0*(sl - np.roll(np.roll(sl, 10, 0), 10, 1)), 0.85, 1.06)  # sun from top-left
    shadow = np.roll(np.roll(sl, 26, 0), 22, 1)
    img = BASE*(1 - 0.55*shadow[..., None])
    cc = np.array([0.92, 0.94, 0.97], np.float32)*lit[..., None]*(0.78 + 0.22*a[..., None])
    img = img*(1 - a[..., None]) + cc*a[..., None]
    img = img*0.93 + np.array([0.06, 0.09, 0.13], np.float32)*0.07*4    # atmospheric haze
    z = 1.0 + 0.04*np.sin(np.pi*t/T)**2                                               # slow satellite drift/zoom (<=1.04)
    M = cv2.getRotationMatrix2D((W2/2, H2*0.4), 0, z); M[1, 2] -= 30*S*np.sin(np.pi*t/T)**2
    return cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# ================= x-ray layer: the solver skeleton =================
NIGHT = K.NIGHT/255
def vort(k):
    u, v = UV[k, 0].astype(np.float32), UV[k, 1].astype(np.float32)
    return (np.gradient(v, axis=1) - np.gradient(u, axis=0)).T            # screen orientation (x down)
def xray_at(t):
    k = fidx(t); w = vort(k); w[sim.SOLID.T] = 0
    img = np.ones((H2, W2, 3), np.float32)*NIGHT
    wu = cv2.resize(w, (432*2, 768*2), interpolation=cv2.INTER_CUBIC)
    glow = cv2.resize(np.abs(wu), (W2, H2))/0.02
    img += np.clip(glow, 0, 1)[..., None]*np.array(CYAN, np.float32)/255*0.16
    lines = np.zeros((H2, W2), np.uint8); sc = W2/wu.shape[1]
    for lev in (0.0015, 0.003, 0.006, 0.012, 0.022):
        for sgn in (1, -1):
            cs, _ = cv2.findContours(((sgn*wu) > lev).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cs = [(c_*sc).astype(np.int32) for c_ in cs if len(c_) > 6]
            cv2.polylines(lines, cs, True, 255 if sgn > 0 else 170, 3, cv2.LINE_AA)
    img += (lines.astype(np.float32)/255)[..., None]*np.array(CYAN, np.float32)/255*0.95
    # island outline + wind arrows (driver colour)
    th = np.linspace(0, 2*np.pi, 200); R = sim.r_island(th)*PX
    pts = np.stack([IX + R*np.sin(th), IY + R*np.cos(th)], 1).astype(np.int32)
    ov = np.zeros((H2, W2), np.uint8); cv2.polylines(ov, [pts], True, 255, 6, cv2.LINE_AA)
    img += (ov/255.)[..., None]*np.array(CYAN, np.float32)/255
    ar = np.zeros((H2, W2), np.uint8)
    for x in (260, 700, 1460, 1900):
        y0 = 900 + (CUM[k]*sim.U0*PX*0.1) % 60
        cv2.arrowedLine(ar, (x, int(y0)), (x, int(y0+130)), 255, 5, cv2.LINE_AA, tipLength=0.25)
    img += (ar/255.)[..., None]*np.array(AMBER, np.float32)/255*0.8
    return img

# ================= swirl events (from the solver probe, per video frame) =================
def shed_events():
    p = np.array([float(UV[k, 1, int(sim.CY0), int(sim.CX0 + 2.5*sim.D)]) for k in range(len(UV))])
    ev = []; on = True
    for k in range(1, len(p)):
        if np.sign(p[k]) != np.sign(p[k-1]) and p[k] != 0: ev.append((k/FPS, 1 if p[k] > 0 else -1))
    return ev
EVS = shed_events()

# ================= guess layer: straight wind arrows (2-8 s) =================
def guess(d, t):
    if not 2.4 < t < 7.6: return
    a = sm((t-2.4)/0.4)*(1 - sm((t-7.0)/0.6)); col = AMBER if t < 5.2 else RED
    col = tuple(int(v*a) for v in col)
    for x in (230, 490, 750):
        y0 = 620 + ((t*120) % 160)
        d.line([(x, y0), (x, y0+300)], fill=col, width=6)
        d.polygon([(x-18, y0+300), (x+18, y0+300), (x, y0+336)], fill=col)
    if t > 5.2: d.line([(170, 640), (810, 1040)], fill=col, width=8)

# ================= text layer (1x) =================
def text(t, im):
    d = ImageDraw.Draw(im)
    real = t < SCAN0 or t > BACK0
    K.title_box(d, 'WHO DREW THESE SWIRLS?', 'real clouds. one island.' if real else None)
    K.series_tag(d, 2)
    guess(d, t)
    for (a, b, lines) in CAPS:
        if a <= t < b: K.caption(d, lines, alpha=min(1, (t-a)/0.15, (b-t)/0.15))
    if 12.6 < t < TROPHY - 0.4:
        d.text((90, 1185), 'solver: lattice Boltzmann, 384 x 216', font=K.F_MONO, fill=CYAN, anchor='ls')
        d.text((90, 1225), f'solver: Re = 150', font=K.F_MONO, fill=CYAN, anchor='ls')
    if 18.1 < t < 22.5: K.pill(d, (790, 520), 'x3 faster', AMBER)
    # swirl tags at each shed event (drift downstream with the flow)
    if SCAN1 < t < DIAL:
        n = 0
        for (te, s_) in EVS:
            if te > t: break
            if te < SCAN1 + 0.3: continue
            n += 1; age = t - te
            if age < 3.0:
                spf = order_spf[fidx(te)]
                y = sim.CX0*5 + 2.5*sim.D*5 + age*FPS*spf*sim.U0*0.85*5
                x = sim.CY0*5 + s_*95
                if y < 1150: K.pill(d, (x, y), f'swirl {n}', CYAN)
    if TROPHY <= t < DIAL:                                                # equation trophy (>= 3 s, on the hit)
        a = sm((t-TROPHY)/0.35); y = 1290
        segs = [('f', CYAN), (' = 0.2 · ', WHITE), ('U', AMBER), (' / ', WHITE), ('D', WHITE)]
        tw = sum(d.textlength(s_, font=K.F_EQ) for s_, _ in segs); x = CX - tw/2
        for s_, c in segs:
            d.text((x, y), s_, font=K.F_EQ, fill=tuple(int(v*a) for v in c), anchor='lm', stroke_width=4, stroke_fill=(0, 0, 0)); x += d.textlength(s_, font=K.F_EQ)
        d.text((90, 1185), f"solver measured: f·D/U = {STJ['St']:.2f}", font=K.F_MONO, fill=CYAN, anchor='ls')
    elif DIAL <= t < BACK0:
        K.pill(d, (CX, 1385), f"f = {STJ['St']:.2f} · U / D", CYAN if not (BREAK <= t < SNAP) else RED, font=K.F_CHIP)
    if DIAL <= t < J1:                                                    # the ONE number
        if t < BREAK: v, c = 'x5', AMBER
        elif t < SNAP: v, c = 'x5', RED
        else: v, c = 'x1', GREEN
        K.pill(d, (CX, 492), f'dial: air thickness {v}', c)
        if DIAL + 0.6 < t < SNAP: d.text((90, 1185), 'solver at Re = 30: no shedding', font=K.F_MONO, fill=RED if t > BREAK else AMBER, anchor='ls')
        if DIAL + 0.6 < t < SNAP: K.pill(d, (790, 590), 'x5 faster', AMBER)
    K.end_stamp(d, alpha=sm((t-J2)/0.4)*(1-sm((t-T+0.5)/0.3)), y=1385)

# ================= frame =================
vy, vx = np.mgrid[0:K.H, 0:K.W].astype(np.float32)
VIG = (1 - 0.30*(((vx-K.W/2)/K.W)**2 + ((vy-K.H/2)/K.H)**2)*2)[..., None]
GRAIN = [np.random.default_rng(i).normal(0, 1, (K.H, K.W, 1)).astype(np.float32) for i in range(6)]
F0 = None
def frame(t):
    global F0
    L = real_at(t) if (t < SCAN1 or t > BACK0) else None
    X = xray_at(t) if SCAN0 <= t <= BACK1 else None
    if L is not None and X is not None:
        if t < SCAN1: yl = int(sm((t-SCAN0)/(SCAN1-SCAN0))*H2)
        else: yl = int((1-sm((t-BACK0)/(BACK1-BACK0)))*H2)
        img = L.copy(); img[:yl] = X[:yl]
        img += np.exp(-((np.arange(H2)-yl)/(6*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*1.6
        img += np.exp(-((np.arange(H2)-yl)/(60*S))**2)[:, None, None]*np.array(CYAN, np.float32)/255*0.25
    else: img = L if L is not None else X
    if t > T - 0.6:                                                       # seamless loop to frame 1
        if F0 is None: F0 = real_at(0.0)
        a = sm((t-(T-0.6))/0.6); img = img*(1-a) + F0*a
    im = cv2.resize(np.clip(img, 0, 1.2), (K.W, K.H), interpolation=cv2.INTER_AREA)
    im = np.clip(im*VIG + GRAIN[int(t*FPS) % 6]*0.008, 0, 1)
    pim = Image.fromarray((im*255).astype(np.uint8)); text(t, pim); return pim

def music_events():
    return [(te, [0.5*s_], 1) for te, s_ in EVS if te < TROPHY]

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sheet'
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')]
        for t in ts: frame(t).save(f'f_{t:05.1f}.png')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', f'part_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(frame(i/FPS).tobytes())
        p.stdin.close(); p.wait()
