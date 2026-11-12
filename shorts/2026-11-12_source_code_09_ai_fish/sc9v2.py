# SOURCE CODE #9 v2 - "AI FISH vs AI SHARKS" (Coded Cosmos, 12 Nov 2026)
# No narration: the simulation tells the story. Slow, fine-line render (owner's boids look), ASMR sound in asmr9.py.
import numpy as np, cv2, subprocess, sys, os, json, pickle
from PIL import Image, ImageDraw, ImageFont
import source_code_kit as K
from source_code_kit import CX, CYAN, AMBER, RED, GREEN, YELLOW, WHITE
import evo as E

FPS = 30; S = 2; W, H = K.W, K.H; W2, H2 = W*S, H*S
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('SEED', '3'))
EV = pickle.load(open(os.path.join(HERE, f'evo_s{SEED}.pkl'), 'rb'))
GENS = sorted(EV['snaps']); GMAX = GENS[-1]
W_FINAL, WQ_FINAL = EV['W'], EV['WQ']; W0, WQ0 = EV['snaps'][0]
LOG = np.array(EV['log'])                     # step, gen, nn, nbrs, catch/1k, o1..o3
def sm(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
def lerp(a, b, x): return a + (b-a)*x
def col(c, m=1.0): return np.array(c, np.float32)/255*m
FP = lambda n, s: ImageFont.truetype(os.path.join(HERE, n), s)
F_CHIP = FP('Poppins-SemiBold.ttf', 36); F_TXT = FP('Poppins-SemiBold.ttf', 50); F_SMALL = FP('Poppins-SemiBold.ttf', 34)
F_BIGGEN = FP('Poppins-ExtraBold.ttf', 92); F_MONO = FP('JetBrainsMono.ttf', 24)
FISHC = np.array([0.80, 0.95, 1.0], np.float32)

# ================= story clock (no voice) =================
T = 40.0; LOOP0 = T - 0.7
GEN1 = 3.6                      # cut to generation 1
BRAIN0, BRAIN1 = 6.6, 10.6      # random-brain inset
LAPSE0, LAPSE1 = 11.2, 21.0     # evolution time-lapse
FINAL = 21.0                    # last generation, real speed
ATT1, ATT2 = 23.4, 28.6         # shark attacks into the school
OUT0 = 33.0                     # pull back + end lines
NEXT0 = 37.2
TEXTS = [  # (t0, t1, text, colour) one short line at a time
    (GEN1 + 0.6, 6.4, 'random brains', WHITE),
    (BRAIN0 + 0.2, BRAIN1 - 0.1, 'every fish: 13 neurons', CYAN),
    (LAPSE0 + 0.1, LAPSE0 + 1.5, 'survivors copy their brain', WHITE),
    (LAPSE0 + 3.8, LAPSE1 - 0.3, 'they start to stick together', GREEN),
    (FINAL + 0.4, ATT1 - 0.1, 'they learned to school', GREEN),
    (ATT1 + 0.2, 27.8, 'too many targets', AMBER),
    (28.6, OUT0 - 0.2, 'safety in numbers', WHITE),
    (OUT0 + 0.3, NEXT0 - 0.1, 'nobody taught them this', YELLOW)]

# ================= worlds =================
def warm(w, n):
    for _ in range(n): w.step()
    return w
def school_center(P):
    D = np.sqrt(((P[:, None] - P[None])**2).sum(-1)); n = (D < 8).sum(1)
    i = n.argmax(); m = D[i] < 12
    return P[m].mean(0) if n[i] > 12 else np.median(P, 0)
TRAIL = 14
class Track:
    def __init__(s, w):
        s.w = w; s.ev0 = len(w.events); s.flash = []; s.born = np.full(E.N, -10.0); s.acc = 0.0
        s.prevP, s.prevQ = w.P.copy(), w.Q.copy(); s.Pd, s.Qd = w.P.copy(), w.Q.copy()
        s.hp = [w.P.copy() for _ in range(TRAIL)]; s.cam = school_center(w.P); s.eaten = 0; s.misses = []
    def adv(s, n, t, follow=0.035):
        s.acc += n
        while s.acc >= 1.0:
            s.prevP, s.prevQ = s.w.P.copy(), s.w.Q.copy(); s.w.step(); s.acc -= 1.0
            for e in s.w.events[s.ev0:]:
                s.flash.append((t, e[5], e[3], e[4]))
                if e[3]: s.born[e[2]] = t; s.prevP[e[2]] = s.w.P[e[2]]; s.eaten += 1
                else: s.misses.append(t)
            s.ev0 = len(s.w.events)
        a = s.acc; s.Pd = s.prevP + (s.w.P - s.prevP)*a; s.Qd = s.prevQ + (s.w.Q - s.prevQ)*a
        s.hp = s.hp[1:] + [s.Pd.copy()]
        jump = np.linalg.norm(s.hp[-1] - s.hp[-2], axis=1) > 4
        for h in s.hp[:-1]: h[jump] = s.hp[-1][jump]
        if follow > 0:
            tgt = school_center(s.Pd) if not getattr(s, 'wide', False) else np.median(s.Pd, 0)
            if getattr(s, 'members', None) is not None: tgt = s.Pd[s.members].mean(0)
            if getattr(s, 'chase', None) is not None: tgt = s.Qd[s.chase] + E.unit(s.w.U[s.chase][None])[0]*3.0
            dq = np.linalg.norm(s.Qd - tgt, axis=1); kq = int(dq.argmin())
            if dq[kq] < 24: tgt = tgt + (s.Qd[kq] - tgt)*0.22*min(1.0, (24 - dq[kq])/8)
            s.cam = s.cam + (tgt - s.cam)*follow

# ================= camera =================
FOC = 1250.0; CY0 = 880
YAW = {'h': 0.5, 'f': 0.5}
def cam_params(t):
    yaw = (YAW['h'] if t < GEN1 else 0.5 if t < FINAL else YAW['f'] - 0.045*FINAL) + 0.045*t; pitch = 0.28 + 0.06*np.sin(t*0.21)
    if t < GEN1: dist = lerp(15.0, 18.0, sm(t/GEN1))
    elif t < LAPSE0: dist = 21.0
    elif t < FINAL: dist = lerp(21.0, 40.0, sm((t-LAPSE0)/2.0))
    else: dist = lerp(16.0, 22.0, sm((t-FINAL-1.5)/3.0))
    if t >= OUT0: dist = lerp(dist, 30.0, sm((t-OUT0)/4.5))
    return yaw, pitch, dist
def project(X, C, yaw, pitch, dist):
    r = X - C; c, s = np.cos(yaw), np.sin(yaw)
    x = r[..., 0]*c + r[..., 2]*s; z = -r[..., 0]*s + r[..., 2]*c; y = r[..., 1]
    cp, sp = np.cos(pitch), np.sin(pitch); y2 = y*cp - z*sp; z2 = y*sp + z*cp
    dep = dist + z2; k = FOC/np.maximum(dep, 1.0)
    return (CX + x*k)*S, (CY0 - y2*k)*S, k*S, dep
def unproject(sx, sy, dep, C, yaw, pitch, dist):
    x = (sx - CX)*dep/FOC; y2 = (CY0 - sy)*dep/FOC; z2 = dep - dist
    cp_, sp_ = np.cos(pitch), np.sin(pitch); y = y2*cp_ + z2*sp_; z = -y2*sp_ + z2*cp_
    c, s_ = np.cos(yaw), np.sin(yaw)
    return C + np.array([x*c - z*s_, y, x*s_ + z*c])
def place_shark(tr, k, t, sx, sy, ddep, target):
    yaw, pitch, dist = cam_params(t)
    q = unproject(sx, sy, dist + ddep, tr.cam, yaw, pitch, dist)
    tr.w.Q[k] = q; tr.w.U[k] = E.unit((target - q)[None])[0]*1.15; tr.w.cd[k] = 3; tr.prevQ[k] = q; tr.Qd[k] = q

# ================= background =================
RNG = np.random.default_rng(91)
def make_bg():
    h4, w4 = H2//8, W2//8
    yy = np.linspace(0, 1, h4)[:, None, None]
    top = np.array([0.020, 0.085, 0.125], np.float32); bot = np.array([0.002, 0.008, 0.020], np.float32)
    bg = (top*(1-yy)**1.6 + bot*(1-(1-yy)**1.6))*np.ones((h4, w4, 1), np.float32)
    yv, xv = np.mgrid[0:h4, 0:w4].astype(np.float32)
    bg *= (1.1 - 0.5*np.clip(np.hypot(xv/w4 - 0.45, yv/h4 - 0.42)/0.75, 0, 1))[..., None]
    return cv2.resize(bg, (W2, H2), interpolation=cv2.INTER_CUBIC)
BG = make_bg()
NR = 10; RAYX = RNG.uniform(-0.2, 1.2, NR); RAYW = RNG.uniform(0.015, 0.05, NR); RAYA = RNG.uniform(0.02, 0.05, NR); RAYP = RNG.uniform(0, 6.28, NR)
YY4, XX4 = np.mgrid[0:H2//8, 0:W2//8].astype(np.float32); YY4 /= H2//8; XX4 /= W2//8
def rays(t, yaw):
    g = np.zeros_like(XX4)
    for i in range(NR):
        xc = (RAYX[i] - 0.22*YY4 - yaw*0.35 + 0.03*np.sin(t*0.3 + RAYP[i]) + 0.3) % 1.6 - 0.3
        g += RAYA[i]*(0.6 + 0.4*np.sin(t*0.5 + RAYP[i]))*np.exp(-(XX4 - xc)**2/RAYW[i]**2)
    g *= np.clip(1.0 - YY4*1.25, 0, 1)**1.6
    return cv2.resize(g, (W2, H2), interpolation=cv2.INTER_LINEAR)[..., None]*np.array([0.45, 0.8, 0.85], np.float32)
SNOW = RNG.uniform(-1, 1, (1400, 3)).astype(np.float32)*60; SNOWV = RNG.normal(0, 0.012, (1400, 3)).astype(np.float32) + np.array([0, -0.01, 0], np.float32)

def glow(img, k=0.32, sig=5):
    s = cv2.resize(img, (W2//6, H2//6), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(s, (0, 0), sig) + 0.6*cv2.GaussianBlur(s, (0, 0), sig*4)
    return img + k*cv2.resize(g, (W2, H2), interpolation=cv2.INTER_LINEAR)
def fogk(dep, dist): return float(np.clip(np.exp(-(dep - dist)/40.0), 0.18, 1.3))

# ================= fine-line fish =================
def draw_fish(lay, tr, C, cp, t, tint=None):
    yaw, pitch, dist = cp; P = tr.Pd; V = tr.w.V; n = len(P)
    Vh = E.unit(V); L = 0.95
    side = E.unit(np.cross(Vh, np.array([0, 1.0, 0])))
    ph = t*9 + np.arange(n)*1.7
    sg = np.linspace(0, 1, 7)
    spine = [project(P - Vh*L*s_ + side*(np.sin(ph - s_*4.0)*0.07*L*s_**1.5)[:, None], C, yaw, pitch, dist) for s_ in sg]
    tailp = project(P - Vh*L*1.14 + side*(np.sin(ph - 6)*0.16*L)[:, None], C, yaw, pitch, dist)
    hx, hy, hk, hd = spine[0]
    LGT = E.unit(np.array([[0.4, 0.8, 0.45]]))[0]; shim = 0.65 + 0.7*np.abs(side @ LGT)**2
    HP = [project(h, C, yaw, pitch, dist) for h in tr.hp[::2]]
    for i in range(n):
        if hd[i] < 4 or not (-60 < hx[i] < W2 + 60 and -60 < hy[i] < H2 + 60): continue
        f = fogk(hd[i], dist)*min(1.0, (t - tr.born[i])/0.6 + 0.05)
        c = (FISHC if tint is None else tint[i])*f*shim[i]
        # trail (thin, fading)
        for j in range(len(HP)-1):
            a = (j+1)/len(HP)
            cv2.line(lay, (int(HP[j][0][i]*4), int(HP[j][1][i]*4)), (int(HP[j+1][0][i]*4), int(HP[j+1][1][i]*4)), tuple((c*0.10*a).tolist()), 1, cv2.LINE_AA, 2)
        # body outline as two edge curves
        xs = np.array([sp[0][i] for sp in spine]); ys = np.array([sp[1][i] for sp in spine])
        dx, dy = np.gradient(xs), np.gradient(ys); nn = np.hypot(dx, dy) + 1e-6
        w_ = hk[i]*0.075*np.sin(np.pi*(0.12 + 0.82*sg))**0.7*(1 - 0.55*sg)
        Lp = np.stack([xs - dy/nn*w_, ys + dx/nn*w_], 1); Rp = np.stack([xs + dy/nn*w_, ys - dx/nn*w_], 1)
        poly = np.concatenate([Lp, Rp[::-1]])
        cv2.fillPoly(lay, [(poly*4).astype(np.int32)], tuple((c*0.38).tolist()), cv2.LINE_AA, 2)
        cv2.polylines(lay, [(poly*4).astype(np.int32)], True, tuple((c*0.9).tolist()), 1, cv2.LINE_AA, 2)
        tx, ty = tailp[0][i], tailp[1][i]; bx, by = xs[-1], ys[-1]
        ex, ey = (tx - bx), (ty - by); tw = hk[i]*0.085
        en = np.hypot(ex, ey) + 1e-6
        tri = np.array([[bx, by], [tx - ey/en*tw, ty + ex/en*tw], [tx - ex*0.25 , ty - ey*0.25], [tx + ey/en*tw, ty - ex/en*tw]])
        cv2.fillPoly(lay, [(tri*4).astype(np.int32)], tuple((c*0.45).tolist()), cv2.LINE_AA, 2)
        cv2.polylines(lay, [(np.stack([xs[:5], ys[:5] - w_[:5]*0.35], 1)*4).astype(np.int32)], False, tuple((c*0.5).tolist()), 1, cv2.LINE_AA, 2)
    return hx, hy, hk, hd

def draw_shark(img, q, u, C, cp, t, k):
    yaw, pitch, dist = cp; L = 7.5; uh = E.unit(u[None])[0]
    side = E.unit(np.cross(uh, np.array([0, 1.0, 0]))[None])[0]; up2 = np.cross(side, uh)
    ph = t*4.5 + k*2; segs = np.linspace(0, 1, 18)
    spine = np.array([q - uh*L*s_ + side*np.sin(ph - s_*3.2)*0.35*s_**1.6 for s_ in segs])
    sx, sy, sk, sd = project(spine, C, yaw, pitch, dist)
    if sd.min() < 6: return None
    f = fogk(sd[0], dist)*sm((sd.min() - 6)/6)
    dx, dy = np.gradient(sx), np.gradient(sy); nn = np.hypot(dx, dy) + 1e-6
    w_ = sk*0.62*np.sin(np.pi*(0.04 + 0.92*segs))**0.55*(1 - 0.62*segs)
    Lp = np.stack([sx - dy/nn*w_, sy + dx/nn*w_], 1); Rp = np.stack([sx + dy/nn*w_, sy - dx/nn*w_], 1)
    def P2(X): x, y, _, _ = project(X[None], C, yaw, pitch, dist); return [x[0]*4, y[0]*4]
    tb = spine[-1]; td = E.unit((spine[-1] - spine[-3])[None])[0]
    fins = [[tb, tb + td*1.9 + up2*2.0, tb + td*0.8], [tb, tb + td*1.3 - up2*1.2, tb + td*0.5],
            [spine[6], spine[8] + up2*1.8 + td*0.4, spine[9]], [spine[5], spine[8] + side*1.7 - up2*0.6, spine[8]], [spine[5], spine[8] - side*1.7 - up2*0.6, spine[8]]]
    lay = np.zeros_like(img)
    poly = (np.concatenate([Lp, Rp[::-1]])*4).astype(np.int32)
    cA = col(AMBER)*f
    for fn in fins:
        fp = np.array([P2(x) for x in fn], np.int32)
        cv2.fillPoly(lay, [fp], tuple((cA*0.16).tolist()), cv2.LINE_AA, 2); cv2.polylines(lay, [fp], True, tuple((cA*0.9).tolist()), 2*S, cv2.LINE_AA, 2)
    cv2.fillPoly(lay, [poly], tuple((cA*0.2).tolist()), cv2.LINE_AA, 2)
    cv2.polylines(lay, [poly], True, tuple((cA*1.25).tolist()), 2*S, cv2.LINE_AA, 2)
    gill = [spine[3] + side*0.0, spine[3]]
    for g_ in (2.6, 3.0, 3.4):
        a_ = P2(q - uh*g_ + up2*0.35); b_ = P2(q - uh*g_ - up2*0.25)
        cv2.line(lay, tuple(int(v) for v in a_), tuple(int(v) for v in b_), tuple((cA*0.7).tolist()), S, cv2.LINE_AA, 2)
    e_ = P2(q - uh*0.9 + up2*0.25); cv2.circle(lay, tuple(int(v) for v in e_), int(sk[0]*0.07*4), tuple((cA*1.6).tolist()), -1, cv2.LINE_AA, 2)
    img += lay
    return sx[0], sy[0], sk[0], sd[0]

# ================= brain inset =================
LAYN = [4, 6, 3]; BX0, BY0, BW, BH = 300, 1080, 380, 230
NXY = [[(BX0 + li*BW/2, BY0 + (j + 0.5)*BH/n) for j in range(n)] for li, n in enumerate(LAYN)]
def draw_brain(img, w, h_, o_, t, a):
    W1 = w[:24].reshape(4, 6); W2_ = w[30:48].reshape(6, 3)
    for li, M in ((0, W1), (1, W2_)):
        for i, (x0, y0) in enumerate(NXY[li]):
            for j, (x1, y1) in enumerate(NXY[li+1]):
                wv = M[i, j]; c = (col(CYAN) if wv > 0 else col(AMBER))*a*min(1.0, abs(wv)*0.7)
                cv2.line(img, (int(x0*S), int(y0*S)), (int(x1*S), int(y1*S)), tuple((c*0.8).tolist()), max(1, int(min(abs(wv), 1.5)*1.6*S)), cv2.LINE_AA)
                ph = (t*0.9 + 0.37*i + 0.21*j) % 1.0
                cv2.circle(img, (int(lerp(x0, x1, ph)*S), int(lerp(y0, y1, ph)*S)), int(2*S), tuple((c*1.4).tolist()), -1, cv2.LINE_AA)
    acts = [np.ones(4)*0.6, h_, o_]
    for li, lyr in enumerate(NXY):
        for j, (x, y) in enumerate(lyr):
            v = 0.4 + 0.8*float(np.clip(abs(acts[li][j]), 0, 1))
            cv2.circle(img, (int(x*S), int(y*S)), int(11*S), tuple((col(WHITE)*0.12*a*v).tolist()), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x*S), int(y*S)), int(11*S), tuple((col(WHITE)*0.9*a*v).tolist()), int(1.5*S), cv2.LINE_AA)

# ================= graph (real evolution log) =================
GX0, GX1, GY0, GY1 = 150, 830, 1030, 1250
gl = LOG[:, 1]; gr = LOG[:, 3]
kern = np.ones(5)/5; grs = np.convolve(np.pad(gr, 2, mode='edge'), kern, 'valid')
GX = gl + (LOG[:, 0] - np.interp(gl, gl, LOG[:, 0]))*0     # generation index per sample
GXc = LOG[:, 0]/LOG[-1, 0]                                   # fraction of the run
RMAX = max(grs.max(), 1)
def draw_graph(img, d, frac, a):
    n = max(2, int(len(GXc)*frac))
    pts = np.stack([GX0 + GXc[:n]*(GX1-GX0), GY1 - grs[:n]/RMAX*(GY1-GY0)], 1)
    pan = img[(GY0-70)*S:(GY1+60)*S, (GX0-40)*S:(GX1+40)*S]; pan *= (1 - 0.6*a)
    cv2.line(img, (GX0*S, GY1*S), (GX1*S, GY1*S), tuple((col(WHITE)*0.25*a).tolist()), S, cv2.LINE_AA)
    cv2.line(img, (GX0*S, GY0*S), (GX0*S, GY1*S), tuple((col(WHITE)*0.25*a).tolist()), S, cv2.LINE_AA)
    cv2.polylines(img, [(pts*S).astype(np.int32)], False, tuple((col(GREEN)*1.3*a).tolist()), int(2.5*S), cv2.LINE_AA)
    cv2.circle(img, (int(pts[-1, 0]*S), int(pts[-1, 1]*S)), int(6*S), tuple((col(GREEN)*1.8*a).tolist()), -1, cv2.LINE_AA)
    return pts[-1]

# ================= state =================
class State: pass
ST = None
def make(wp, wq, seed, warmn):
    return Track(warm(E.World(seed=seed, prey_w=wp, pred_w=wq, evolve=False), warmn))
def reset():
    global ST
    ST = State(); ST.hook = make(W_FINAL, WQ_FINAL, int(os.environ.get('HOOKSEED', '12')), 900); v = ST.hook.w.V.mean(0); YAW['h'] = float(np.arctan2(v[2], v[0])); ST.g1 = None; ST.fin = None; ST.gen = 0; ST.dir = 0; ST.focus = 0
    c = ST.hook.cam; place_shark(ST.hook, 0, 0.0, 960, 980, 3.0, c + np.array([0, 0.5, 0]))
def step_to(t):
    st = ST
    if t < GEN1:
        st.hook.adv(0.22, t, follow=0.12); return st.hook
    if t < FINAL:
        if st.g1 is None:
            st.g1 = make(W0, WQ0, SEED + 20, 30); w = st.g1.w
            st.focus = int(np.argmin(np.linalg.norm(w.P - np.median(w.P, 0), axis=1)))
            st.next_hunt = 1e9; w.Q[0] = np.median(w.P, 0) + np.array([3.0, 1.0, 0]); st.g1.prevQ[0] = w.Q[0]; st.g1.Qd[0] = w.Q[0]; st.g1.cam = w.Q[0].copy()
        tr = st.g1; w = tr.w
        if t < LAPSE0:
            if t >= st.next_hunt:                       # director: point a shark at a visible lone fish (its brain still decides)
                k = int(st.next_hunt*7) % E.K; yaw, pitch, dist = cam_params(t)
                x, y, _, dd = project(tr.Pd, tr.cam, yaw, pitch, dist)
                vis = np.where((x > 200*S) & (x < 880*S) & (y > 500*S) & (y < 1350*S) & (dd > 10))[0]
                if len(vis):
                    i = vis[int(t*997) % len(vis)]; tgt = w.P[i] + w.V[i]*3
                    q = tgt - E.unit(RNG.normal(size=(1, 3)))[0]*4.5
                    w.Q[k] = q; w.U[k] = E.unit((tgt - q)[None])[0]*1.15; w.cd[k] = 0; tr.prevQ[k] = q; tr.Qd[k] = q
                st.next_hunt += 0.85
            dq_ = w.P - w.Q[0]; i_ = int(np.linalg.norm(dq_, axis=1).argmin())          # director: lead shark is steered at the nearest fish so hunts are on camera
            w.U[0] = E.unit((0.86*E.unit(w.U[0][None])[0] + 0.14*E.unit(dq_[i_][None])[0])[None])[0]*1.1
            tr.chase = 0; tr.adv(0.32, t, follow=0.07)
        else:
            x = sm((t - LAPSE0 - 0.3)/(LAPSE1 - LAPSE0 - 1.2)); g = int(np.clip(round(x*GMAX), 0, GMAX))
            if g != st.gen:
                snap = max(s for s in GENS if s <= g); w.W[:] = EV['snaps'][snap][0]; w.WQ[:] = EV['snaps'][snap][1]; st.gen = g
            tr.chase = None; tr.adv(2.2, t, follow=0.05)
        return tr
    if st.fin is None:
        st.fin = make(W_FINAL, WQ_FINAL, SEED + 30, 1200); st.gen = GMAX; v = st.fin.w.V.mean(0); YAW['f'] = float(np.arctan2(v[2], v[0]))
    tr = st.fin
    if st.dir < 1 and t >= ATT1 - 1.0: place_shark(tr, 0, t, 230, 1300, 1.0, school_center(tr.w.P)); st.dir = 1
    if False and st.dir < 2 and t >= ATT2 - 1.0: place_shark(tr, 1, t, 1020, 640, 3.0, school_center(tr.w.P)); place_shark(tr, 2, t, 140, 1350, 6.0, school_center(tr.w.P)); st.dir = 2
    tr.adv(0.2, t, follow=0.08); return tr

def text_line(d, txt, c, y, a):
    tw = d.textlength(txt, font=F_TXT)
    d.rounded_rectangle([CX - tw/2 - 30, y - 44, CX + tw/2 + 30, y + 44], 22, fill=(0, 0, 0, int(110*a)))
    d.text((CX, y), txt, font=F_TXT, fill=tuple(int(v) for v in c) + (int(255*a),), anchor='mm', stroke_width=4, stroke_fill=(0, 0, 0, int(200*a)))

def frame(t):
    tr = step_to(t); w = tr.w; cp = cam_params(t); yaw, pitch, dist = cp; C = tr.cam
    img = BG.copy() + rays(t, yaw)
    sn = (SNOW + SNOWV*t*30 + 60) % 120 - 60
    sx, sy, sk, sd = project(C + sn, C, yaw, pitch, dist)
    ok = (sd > 3) & (sx > 0) & (sx < W2) & (sy > 0) & (sy < H2)
    for x, y, d_ in zip(sx[ok], sy[ok], sd[ok]):
        b = float(np.clip(0.35*np.exp(-d_/45), 0.02, 0.3)); cv2.circle(img, (int(x), int(y)), 1, (b*0.7, b*0.95, b), -1, cv2.LINE_AA)
    lay = np.zeros_like(img)
    tint = None
    if t >= FINAL:
        nn_ = (np.sqrt(((tr.Pd[:, None] - tr.Pd[None])**2).sum(-1)) < 6).sum(1)
        lone = np.clip(1 - nn_/6, 0, 1)[:, None]
        tint = FISHC*(1 - lone) + col(RED, 1.1)*lone
    fx, fy, fk, fd = draw_fish(lay, tr, C, cp, t, tint)
    img += lay
    sh = [draw_shark(img, tr.Qd[k], w.U[k], C, cp, t, k) for k in range(E.K)]
    # confusion: shark eye -> candidate targets (thin, flickering)
    if t >= FINAL:
        for k in range(E.K):
            if sh[k] is None or not (-100 < sh[k][0] < W2 + 100 and -100 < sh[k][1] < H2 + 100) or sh[k][3] > dist + 30: continue
            d = np.linalg.norm(tr.Pd - tr.Qd[k], axis=1); idx = np.argsort(d)[:7]
            if d[idx[0]] > 6: continue
            ca = float(np.clip((6 - d[idx[0]])/3, 0, 1)); idx = idx[d[idx] < 8]
            for m, i in enumerate(idx):
                fl = 0.5 + 0.5*np.sin(t*19 + m*2.1 + k)
                cv2.line(img, (int(sh[k][0]), int(sh[k][1])), (int(fx[i]), int(fy[i])), tuple((np.array([0.03, 0.12, 0.2]) + col(AMBER)*0.85*ca*fl).tolist()), int(1.5*S), cv2.LINE_AA)
                cv2.circle(img, (int(fx[i]), int(fy[i])), int(max(9*S, fk[i]*0.5)), tuple((np.array([0.03, 0.12, 0.2]) + col(AMBER)*1.2*ca*fl).tolist()), int(1.5*S), cv2.LINE_AA)
    # strike flashes: catch = red ripple, miss = soft white ring
    for (te, pos, hit, nc) in tr.flash:
        age = t - te
        if age < 0 or age > 1.2 or (not hit and t < FINAL): continue
        x, y, k_, d_ = project(pos[None], C, yaw, pitch, dist)
        if d_[0] < 3: continue
        a = 1 - age/1.2
        c = col(RED)*1.6 if hit else col(WHITE)*1.3
        for r_ in (0.0, 0.25):
            if age > r_: cv2.circle(img, (int(x[0]), int(y[0])), int(k_[0]*(0.6 + 3.0*(age - r_))), tuple((c*a).tolist()), S, cv2.LINE_AA)
    # brain inset (random wiring) + graph
    ba = sm((t - BRAIN0)/0.5)*(1 - sm((t - BRAIN1 + 0.5)/0.5)) if BRAIN0 <= t < BRAIN1 else 0
    if ba > 0.01:
        i = ST.focus
        vis_ = 40 < fx[i] < W2 - 40 and 330*S < fy[i] < (BY0 - 40)*S
        if vis_: cv2.line(img, (int((BX0 - 30)*S), int((BY0 + BH/2)*S)), (int(fx[i]), int(fy[i])), tuple((col(CYAN)*0.5*ba).tolist()), S, cv2.LINE_AA)
        if vis_: cv2.circle(img, (int(fx[i]), int(fy[i])), int(26*S), tuple((col(CYAN)*1.2*ba).tolist()), int(1.5*S), cv2.LINE_AA)
        draw_brain(img, w.W[i], w.hid[i], w.out[i], t, ba)
    ga = sm((t - LAPSE0 - 1.5)/0.5)*(1 - sm((t - FINAL - 1.0)/0.6)) if LAPSE0 + 1.5 <= t < FINAL + 1.6 else 0
    gp = None
    if ga > 0.01: gp = draw_graph(img, None, np.clip((t - LAPSE0 - 1.5)/(LAPSE1 - LAPSE0 - 2.2), 0.02, 1), ga)
    img = glow(img)
    img = np.clip(img, 0, None); img = img/(1 + 0.3*img)*1.3
    out = cv2.resize(np.clip(img*255, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LANCZOS4)
    if os.environ.get('NOUI'): return out
    im = Image.fromarray(out); d = ImageDraw.Draw(im, 'RGBA')
    K.series_tag(d, 9)
    if t < 2.8 or t >= LOOP0: K.title_box(d, 'AI FISH vs AI SHARKS', y=250)
    # big generation card at each scene start
    for t0, txt in ((GEN1, 'GENERATION 1'), (FINAL, 'GENERATION %d' % (GMAX + 1))):
        if t0 <= t < t0 + 1.4:
            a = sm((t - t0)/0.25)*(1 - sm((t - t0 - 1.0)/0.4))
            d.text((CX, 760), txt, font=F_BIGGEN, fill=(255, 255, 255, int(255*a)), anchor='mm', stroke_width=5, stroke_fill=(0, 0, 0, int(180*a)))
    if GEN1 + 1.2 <= t < OUT0:
        g = 1 if t < LAPSE0 else (st_gen() + 1)
        gc = GREEN if t >= FINAL else WHITE
        d.rounded_rectangle([55, 160, 440, 216], 14, fill=(2, 8, 18, 200), outline=gc + (200,), width=2)
        d.text((78, 188), 'GENERATION %d' % g, font=F_CHIP, fill=gc, anchor='lm')
        if LAPSE0 <= t < LAPSE1: d.text((420, 188), '>>', font=F_CHIP, fill=YELLOW, anchor='rm')
        d.text((CX, 1530), 'fish eaten: %d' % tr.eaten, font=F_SMALL, fill=(255, 140, 140, 230), anchor='mm', stroke_width=3, stroke_fill=(0, 0, 0)) if t < LAPSE0 or t >= FINAL else None
    if ga > 0.3:
        a = int(255*ga)
        d.text((GX0, GY0 - 30), 'friends close by (average)', font=F_SMALL, fill=(130, 240, 170, a), anchor='lm')
        d.text((GX1, GY1 + 32), 'generations', font=F_SMALL, fill=(200, 210, 225, a), anchor='rm')
    if ba > 0.3:
        a = int(255*ba)
        d.text((NXY[0][0][0] - 20, BY0 - 26), 'senses', font=F_MONO, fill=(200, 215, 235, a), anchor='mm')
        d.text((NXY[2][0][0] + 10, BY0 - 26), 'moves', font=F_MONO, fill=(200, 215, 235, a), anchor='mm')
    for t0, t1, txt, c in TEXTS:
        if t0 <= t < t1:
            a = sm((t - t0)/0.3)*(1 - sm((t - t1 + 0.3)/0.3)); text_line(d, txt, c, 1460, a)
    if t >= OUT0 + 1.0: K.end_stamp(d, sm((t - OUT0 - 1.0)/0.5), y=1350)
    if NEXT0 <= t < LOOP0:
        fa = sm((t - NEXT0)/0.3)
        d.rounded_rectangle([110, 1410, 870, 1530], 22, fill=(0, 0, 0, int(150*fa)), outline=YELLOW + (int(255*fa),), width=3)
        d.text((CX, 1448), 'NEXT · SOURCE CODE #10', font=K.F_PILL, fill=YELLOW + (int(255*fa),), anchor='mm')
        d.text((CX, 1495), 'why can nothing beat light?', font=K.F_PILL, fill=WHITE + (int(255*fa),), anchor='mm')
    return np.array(im)
def st_gen(): return ST.gen

def run_to(ts):
    reset(); out = {}; want = sorted(ts); i = 0
    for n in range(int(max(want)*FPS) + 1):
        t = n/FPS
        if i < len(want) and t + 1e-6 >= want[i]: out[want[i]] = frame(t); i += 1
        else: step_to(t)
    return [out[x] for x in ts]

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')]
        fr = [cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA) for f in run_to(ts)]
        Image.fromarray(np.hstack(fr)).save(sys.argv[3])
    elif mode == 'one':
        Image.fromarray(run_to([float(sys.argv[2])])[0]).save(sys.argv[3])
    else:
        reset(); F0 = None; ev = []
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                              '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', 'video9v2.mp4'], stdin=subprocess.PIPE)
        for i in range(int(T*FPS)):
            t = i/FPS; f = frame(t)
            if i == 0: F0 = f.copy()
            if t >= LOOP0: x = sm((t - LOOP0)/(T - LOOP0)); f = (f*(1-x) + F0*x).astype(np.uint8)
            p.stdin.write(f.tobytes())
            if i % 150 == 0: print(i, flush=True)
        p.stdin.close(); p.wait()
        # sound events for asmr9.py: catches, misses, generation ticks, shark speed near camera
        trs = [ST.hook, ST.g1, ST.fin]
        json.dump(dict(catch=[e[0] for tr_ in trs for e in tr_.flash if e[2]], miss=[e[0] for tr_ in trs for e in tr_.flash if not e[2]]), open('events9.json', 'w'))
        print('done')
