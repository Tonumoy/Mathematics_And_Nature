# SOURCE CODE #9 v3 look: bright ocean, shaded solid fish (depth-sorted), shaded sharks, bloom. Slow sim from sc9v2.
import numpy as np, cv2, sys, os
os.environ.setdefault('BURST', '0.8'); os.environ.setdefault('VPRED', '1.4'); os.environ.setdefault('SEED', '7')
import sc9v2 as V
from sc9v2 import S, W2, H2, W, H, project, col, sm, lerp
E = V.E
RNG = np.random.default_rng(93)

# ---------- brighter ocean ----------
def make_bg():
    h4, w4 = H2//8, W2//8
    yy = np.linspace(0, 1, h4)[:, None, None]
    top = np.array([0.035, 0.26, 0.38], np.float32); mid = np.array([0.012, 0.10, 0.20], np.float32); bot = np.array([0.004, 0.022, 0.07], np.float32)
    a = np.clip(yy/0.45, 0, 1)**0.9; b = np.clip((yy - 0.45)/0.55, 0, 1)
    bg = (top*(1 - a) + mid*a)*(1 - b) + bot*b
    bg = bg*np.ones((h4, w4, 1), np.float32)
    yv, xv = np.mgrid[0:h4, 0:w4].astype(np.float32)
    bg *= (1.12 - 0.45*np.clip(np.hypot(xv/w4 - 0.5, yv/h4 - 0.40)/0.8, 0, 1))[..., None]
    z = cv2.GaussianBlur(RNG.normal(size=(h4, w4)).astype(np.float32), (0, 0), 18); z /= np.abs(z).max()
    bg += (np.clip(z, 0, 1)**2*0.03)[..., None]*np.array([0.3, 0.8, 0.9], np.float32)
    return cv2.resize(bg, (W2, H2), interpolation=cv2.INTER_CUBIC)
V.BG = make_bg()
NR = 16; RX = RNG.uniform(-0.2, 1.3, NR); RW = RNG.uniform(0.012, 0.06, NR); RA = RNG.uniform(0.05, 0.13, NR); RP = RNG.uniform(0, 6.28, NR)
def rays(t, yaw):
    g = np.zeros_like(V.XX4)
    for i in range(NR):
        xc = (RX[i] - 0.25*V.YY4 - yaw*0.35 + 0.035*np.sin(t*0.35 + RP[i]) + 0.3) % 1.6 - 0.3
        g += RA[i]*(0.55 + 0.45*np.sin(t*0.6 + RP[i]))*np.exp(-(V.XX4 - xc)**2/RW[i]**2)
    g *= np.clip(1.05 - V.YY4*1.15, 0, 1)**1.4
    return cv2.resize(g, (W2, H2), interpolation=cv2.INTER_LINEAR)[..., None]*np.array([0.40, 0.85, 0.95], np.float32)
V.rays = rays
def glow(img, k=0.55, sig=5):
    s = cv2.resize(img, (W2//6, H2//6), interpolation=cv2.INTER_AREA)
    hi = np.clip(s - 0.35, 0, None)
    g = cv2.GaussianBlur(hi, (0, 0), sig) + 0.6*cv2.GaussianBlur(hi, (0, 0), sig*4)
    return img + k*cv2.resize(g, (W2, H2), interpolation=cv2.INTER_LINEAR)
V.glow = glow
def fogk(dep, dist): return float(np.clip(np.exp(-(dep - dist)/26.0), 0.22, 1.25))

# ---------- shaded fish ----------
BACK = np.array([0.10, 0.50, 0.78], np.float32)      # dorsal blue
BELLY = np.array([0.62, 0.86, 0.92], np.float32)     # silver-cyan flank
SH = 4
def draw_fish(lay, tr, C, cp, t, tint=None):
    yaw, pitch, dist = cp; P = tr.Pd; n = len(P)
    if getattr(tr, 'Vs', None) is None: tr.Vs = tr.w.V.copy()
    tr.Vs += (tr.w.V - tr.Vs)*0.2; jmp = (t - tr.born) < 0.1; tr.Vs[jmp] = tr.w.V[jmp]; Vv = tr.Vs
    Vh = E.unit(Vv); L = 1.45
    side = E.unit(np.cross(Vh, np.array([0, 1.0, 0])))
    ph = t*8 + np.arange(n)*1.7
    sg = np.linspace(0, 1, 9)
    spine = [project(P + Vh*L*(0.35 - s_) + side*(np.sin(ph - s_*4.0)*0.075*L*s_**1.4)[:, None], C, yaw, pitch, dist) for s_ in sg]
    tailp = project(P + Vh*L*(0.35 - 1.20) + side*(np.sin(ph - 5.2)*0.17*L)[:, None], C, yaw, pitch, dist)
    hx, hy, hk, hd = project(P, C, yaw, pitch, dist)
    LGT = E.unit(np.array([[0.35, 0.85, 0.4]]))[0]; shim = 0.62 + 0.6*np.abs(side @ LGT)**1.5
    HP = [project(h, C, yaw, pitch, dist) for h in tr.hp[::2]]
    prof = np.sin(np.pi*np.clip(0.06 + 0.9*sg, 0, 1))**0.75*(1 - 0.45*sg); prof[0] = 0.25
    for i in np.argsort(-hd):
        if hd[i] < 4 or not (-90 < hx[i] < W2 + 90 and -90 < hy[i] < H2 + 90): continue
        f = fogk(hd[i], dist)*min(1.0, (t - tr.born[i])/0.6 + 0.05)
        red = None if tint is None else tint[i]
        cb = BACK*f; cf = BELLY*f*shim[i]
        if red is not None and red[0] > red[2] + 0.15: cb = np.array([0.75, 0.16, 0.14], np.float32)*f; cf = np.array([1.0, 0.55, 0.5], np.float32)*f
        for j in range(len(HP) - 1):                                  # soft wake
            a = (j + 1)/len(HP)*0.07
            cv2.line(lay, (int(HP[j][0][i]*SH), int(HP[j][1][i]*SH)), (int(HP[j+1][0][i]*SH), int(HP[j+1][1][i]*SH)), tuple((cf*a).tolist()), 2, cv2.LINE_AA, 2)
        xs = np.array([sp[0][i] for sp in spine]); ys = np.array([sp[1][i] for sp in spine])
        dx, dy = np.gradient(xs), np.gradient(ys); nn = np.hypot(dx, dy) + 1e-6
        w_ = hk[i]*0.105*prof
        nx, ny = -dy/nn, dx/nn
        if ny.mean() < 0: nx, ny = -nx, -ny                           # normal pointing screen-down = belly side
        up = np.stack([xs - nx*w_, ys - ny*w_], 1); dn = np.stack([xs + nx*w_, ys + ny*w_], 1)
        tx, ty = tailp[0][i], tailp[1][i]; bx, by = xs[-1], ys[-1]
        ex, ey = tx - bx, ty - by; en = np.hypot(ex, ey) + 1e-6; tw = hk[i]*0.13
        tail = np.array([[bx, by], [tx - ey/en*tw, ty + ex/en*tw], [tx - ex*0.3, ty - ey*0.3], [tx + ey/en*tw, ty - ex/en*tw]])
        cv2.fillPoly(lay, [(tail*SH).astype(np.int32)], tuple((cb*0.9 + cf*0.15).tolist()), cv2.LINE_AA, 2)
        body = np.concatenate([up, dn[::-1]])
        cv2.fillPoly(lay, [(body*SH).astype(np.int32)], tuple(cb.tolist()), cv2.LINE_AA, 2)
        mid = np.stack([xs - nx*w_*0.15, ys - ny*w_*0.15], 1)
        flank = np.concatenate([mid, dn[::-1]])
        cv2.fillPoly(lay, [(flank*SH).astype(np.int32)], tuple(cf.tolist()), cv2.LINE_AA, 2)
        hl = np.stack([xs + nx*w_*0.25, ys + ny*w_*0.25], 1)[1:6]
        cv2.polylines(lay, [(hl*SH).astype(np.int32)], False, tuple((np.ones(3, np.float32)*1.15*f*shim[i]).tolist()), max(1, int(hk[i]*0.02)), cv2.LINE_AA, 2)
        e = (xs[1] - nx[1]*w_[1]*0.1, ys[1] - ny[1]*w_[1]*0.1)
        cv2.circle(lay, (int(e[0]*SH), int(e[1]*SH)), max(2, int(hk[i]*0.022*SH)), (0.01, 0.03, 0.06), -1, cv2.LINE_AA, 2)
    return hx, hy, hk, hd
V.draw_fish = draw_fish

# ---------- shaded shark ----------
def draw_shark(img, q, u, C, cp, t, k):
    yaw, pitch, dist = cp; L = 8.0; uh = E.unit(u[None])[0]
    side = E.unit(np.cross(uh, np.array([0, 1.0, 0]))[None])[0]; up2 = np.cross(side, uh)
    ph = t*4.0 + k*2; segs = np.linspace(0, 1, 22)
    spine = np.array([q + uh*L*(0.25 - s_) + side*np.sin(ph - s_*3.2)*0.4*s_**1.6 for s_ in segs])
    sx, sy, sk, sd = project(spine, C, yaw, pitch, dist)
    if sd.min() < 6: return None
    f = fogk(sd[0], dist)*sm((sd.min() - 6)/6)
    dx, dy = np.gradient(sx), np.gradient(sy); nn = np.hypot(dx, dy) + 1e-6
    prof = np.sin(np.pi*np.clip(0.03 + 0.93*segs, 0, 1))**0.55*(1 - 0.66*segs)
    w_ = sk*0.66*prof; nx, ny = -dy/nn, dx/nn
    if ny.mean() < 0: nx, ny = -nx, -ny
    up = np.stack([sx - nx*w_, sy - ny*w_], 1); dn = np.stack([sx + nx*w_, sy + ny*w_], 1)
    def P2(X): x, y, _, _ = project(X[None], C, yaw, pitch, dist); return [x[0]*SH, y[0]*SH]
    tb = spine[-1]; td = E.unit((spine[-1] - spine[-3])[None])[0]
    fins = [[tb, tb + td*2.1 + up2*2.3, tb + td*0.9], [tb, tb + td*1.4 - up2*1.3, tb + td*0.55],
            [spine[7], spine[10] + up2*2.0 + td*0.5, spine[11]],
            [spine[6], spine[9] + side*2.0 - up2*0.8, spine[9]], [spine[6], spine[9] - side*2.0 - up2*0.8, spine[9]]]
    A_D = np.array([0.78, 0.38, 0.06], np.float32)*f; A_L = np.array([1.0, 0.74, 0.30], np.float32)*f; A_B = np.array([1.0, 0.92, 0.70], np.float32)*f
    lay = np.zeros_like(img); mk = np.zeros(img.shape[:2], np.uint8)
    for fn in fins:
        fp = np.array([P2(x) for x in fn], np.int32)
        cv2.fillPoly(lay, [fp], tuple((A_D*0.9).tolist()), cv2.LINE_AA, 2); cv2.fillPoly(mk, [fp], 255, cv2.LINE_AA, 2)
    body = (np.concatenate([up, dn[::-1]])*SH).astype(np.int32)
    cv2.fillPoly(lay, [body], tuple(A_D.tolist()), cv2.LINE_AA, 2); cv2.fillPoly(mk, [body], 255, cv2.LINE_AA, 2)
    midl = np.stack([sx - nx*w_*0.35, sy - ny*w_*0.35], 1)
    cv2.fillPoly(lay, [(np.concatenate([midl, dn[::-1]])*SH).astype(np.int32)], tuple(A_L.tolist()), cv2.LINE_AA, 2)
    bl = np.stack([sx + nx*w_*0.45, sy + ny*w_*0.45], 1)
    cv2.fillPoly(lay, [(np.concatenate([bl, dn[::-1]])*SH).astype(np.int32)], tuple(A_B.tolist()), cv2.LINE_AA, 2)
    for g_ in (2.0, 2.35, 2.7):
        a_ = P2(q + uh*(L*0.25 - g_) + up2*0.42); b_ = P2(q + uh*(L*0.25 - g_ - 0.15) - up2*0.3)
        cv2.line(lay, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), tuple((A_D*0.55).tolist()), 2, cv2.LINE_AA, 2)
    e_ = P2(q + uh*(L*0.25 - 0.75) + up2*0.22 + side*0.0)
    cv2.circle(lay, (int(e_[0]), int(e_[1])), int(max(3, sk[0]*0.085)*SH), (0.02, 0.01, 0.0), -1, cv2.LINE_AA, 2)
    cv2.circle(lay, (int(e_[0] - sk[0]*0.03*SH), int(e_[1] - sk[0]*0.03*SH)), int(max(1, sk[0]*0.025)*SH), (1.2, 1.2, 1.2), -1, cv2.LINE_AA, 2)
    m = (mk.astype(np.float32)/255)[..., None]
    img *= (1 - m); img += lay*m
    return sx[4], sy[4], sk[0], sd[0]
V.draw_shark = draw_shark

import shark3d
V.draw_shark = shark3d.draw_shark

# ---------- voice timeline (Peter, eleven_v3) ----------
import subprocess, json
from sc9v2 import WHITE, CYAN, AMBER, GREEN, YELLOW, RED
HERE = os.path.dirname(os.path.abspath(__file__))
LINES = ['L%02d' % i for i in range(1, 11)]
EST = dict(L06=3.8, L07=3.2, L08=4.6, L09=1.6, L10=2.6)
DUR = {}
for k in LINES:
    p = os.path.join(HERE, 'vo3', k + '.mp3')
    DUR[k] = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p])) if os.path.exists(p) else EST[k]
TL = {}
def put(k, t0): TL[k] = (t0, t0 + DUR[k]); return t0 + DUR[k]
e = put('L01', 0.05); e = put('L02', e + 0.2)
V.GEN1 = e + 0.25; e = put('L03', V.GEN1 + 0.15)
e = put('L04', e + 0.3); V.BRAIN0 = TL['L04'][0] - 0.2; V.BRAIN1 = e + 0.3
e = put('L05', e + 0.3)
V.LAPSE0 = e + 0.2; e = put('L06', V.LAPSE0 + 0.3)
V.LAPSE1 = V.FINAL = max(e + 0.9, V.LAPSE0 + 5.5)
e = put('L07', V.FINAL + 0.25); V.ATT1 = e - 0.2
e = put('L08', e + 0.35); V.ATT2 = TL['L08'][0] + 3.4
V.OUT0 = e + 0.4; e = put('L09', V.OUT0 + 0.2); e = put('L10', e + 0.4)
V.NEXT0 = e + 0.2; V.T = V.NEXT0 + 2.2; V.LOOP0 = V.T - 0.7
CAP = dict(L01=('I gave 200 fish an AI brain', WHITE), L02=('then I added AI sharks', AMBER), L03=('generation 1: no idea what to do', WHITE),
           L04=('a few neurons, wired at random', CYAN), L05=("eaten? a survivor's brain is copied", WHITE), L06=('they start to stick together', GREEN),
           L07=('now they swim as one', GREEN), L08=("a shark can't lock onto one fish", AMBER), L09=('real fish do this too', WHITE), L10=('nobody taught them. evolution did.', YELLOW))
V.TEXTS = [(TL[k][0] - 0.05, TL[k][1] + 0.25, CAP[k][0], CAP[k][1]) for k in LINES]
_step = V.step_to
def step_to(t):
    st = V.ST
    if t < V.GEN1 and not getattr(st, 'hook2', False) and t >= TL['L02'][0] - 0.2:
        V.place_shark(st.hook, 1, t, 40, 1080, 2.0, V.school_center(st.hook.w.P)); st.hook2 = True
    if not hasattr(st, 'dir2'): st.dir2 = 0
    if t >= V.FINAL and getattr(st, 'fin', None) is not None:
        w = st.fin.w
        if getattr(st.fin, 'members', None) is None and t >= V.ATT1 - 1.2:
            st.fin.members = np.where(np.linalg.norm(w.P - V.school_center(w.P), axis=1) < 10)[0]
        for k_ in (1, 2):                                   # keep the other sharks out of this shot
            if np.linalg.norm(w.Q[k_] - w.P[st.fin.members].mean(0)) < 60 if getattr(st.fin, 'members', None) is not None else False:
                o_ = E.unit((w.Q[k_] - w.P[st.fin.members].mean(0))[None])[0]; w.Q[k_] = w.Q[k_] + o_*3; st.fin.prevQ[k_] = w.Q[k_]; w.U[k_] = o_*1.4
        if st.dir2 < 1 and t >= V.OUT0 - 1.2: V.place_shark(st.fin, 0, t, 1010, 700, 2.0, w.P[st.fin.members].mean(0)); st.dir2 = 1
        for k, t0 in ((0, V.ATT1 - 0.6), (0, V.OUT0 - 0.9)):       # director: staged lunge through the school
            if t0 <= t < t0 + 2.2:
                m_ = st.fin.members; c = w.P[m_].mean(0) + w.V[m_].mean(0)*2.0
                d_ = E.unit((c - w.Q[k])[None])[0] if (t < t0 + 1.6 and np.linalg.norm(c - w.Q[k]) > 1.5) else E.unit(w.U[k][None])[0]
                d_ = E.unit((0.8*E.unit(w.U[k][None])[0] + 0.2*d_)[None])[0]
                w.U[k] = d_*1.4; w.Q[k] = w.Q[k] + d_*0.2*1.0; st.fin.prevQ[k] = st.fin.prevQ[k] + d_*0.2*1.0
    V.CUR = _step(t); return V.CUR
V.step_to = step_to

if __name__ == '__main__':
    mode = sys.argv[1]
    from PIL import Image
    if mode == 'sheet':
        ts = [float(x) for x in sys.argv[2].split(',')]
        fr = [cv2.resize(f, (360, 640), interpolation=cv2.INTER_AREA) for f in V.run_to(ts)]
        Image.fromarray(np.hstack(fr)).save(sys.argv[3])
    elif mode == 'clip':                      # clip a0-a1,b0-b1 out.mp4
        import subprocess
        rngs = [tuple(float(x) for x in r.split('-')) for r in sys.argv[2].split(',')]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[3]], stdin=subprocess.PIPE)
        V.reset()
        for n in range(int(max(r[1] for r in rngs)*30)):
            t = n/30
            if any(a <= t < b for a, b in rngs): p.stdin.write(V.frame(t).tobytes())
            else: V.step_to(t)
        p.stdin.close(); p.wait(); print('clip done')
    elif mode == 'full':
        V.reset(); F0 = None; T = V.T; print('T', T, flush=True)
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', 'video9v3.mp4'], stdin=subprocess.PIPE)
        for i in range(int(T*30)):
            t = i/30; f = V.frame(t)
            if i == 0: F0 = f.copy()
            if t >= V.LOOP0: x = sm((t - V.LOOP0)/(T - V.LOOP0)); f = (f*(1-x) + F0*x).astype(np.uint8)
            p.stdin.write(f.tobytes())
            if i % 150 == 0: print(i, flush=True)
        p.stdin.close(); p.wait()
        trs = [V.ST.hook, V.ST.g1, V.ST.fin]
        json.dump(dict(catch=[e_[0] for tr_ in trs for e_ in tr_.flash if e_[2]], miss=[e_[0] for tr_ in trs for e_ in tr_.flash if not e_[2]]), open('events9.json', 'w'))
        print('done')
    elif mode == 'one':
        Image.fromarray(V.run_to([float(sys.argv[2])])[0]).save(sys.argv[3])
