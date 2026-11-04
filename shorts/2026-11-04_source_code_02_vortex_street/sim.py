# SOURCE CODE #2 solver: D2Q9 lattice-Boltzmann (BGK) flow past an island + cloud deck advected by that flow.
# Sim frame: flow +x. Screen: sim x -> screen down, sim y -> screen right (5 px per cell).
import numpy as np, cv2, json, sys
NX, NY = 384, 216
U0, D = 0.10, 28.0
RE_MAIN, RE_THICK = 150.0, 30.0
def tau_for(re): return 3*U0*D/re + 0.5
CX0, CY0 = 110.0, 108.4
c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
CF = c.astype(np.float32)
w = np.array([4/9]+[1/9]*4+[1/36]*4, np.float32); opp = [0,3,4,1,2,7,8,5,6]
def r_island(th): return D/2*(1 + 0.07*np.sin(3*th+1.0) + 0.04*np.sin(5*th+2.3))
yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float32)
th = np.arctan2(yy-CY0, xx-CX0); SOLID = np.hypot(xx-CX0, yy-CY0) < r_island(th)

def feq(rho, ux, uy):
    cu = 3*(CF[:,0,None,None]*ux + CF[:,1,None,None]*uy); u2 = 1.5*(ux*ux+uy*uy)
    return rho*w[:,None,None]*(1 + cu + 0.5*cu*cu - u2)
def init():
    ux = np.full((NY,NX), U0, np.float32); uy = np.zeros_like(ux)
    uy += 0.01*U0*np.cos(2*np.pi*(yy-CY0)/NY)*np.exp(-((xx-CX0-30)/15)**2)      # tiny kick so shedding starts
    return feq(np.ones_like(ux), ux, uy).astype(np.float32)
FIN = feq(np.ones((NY,1),np.float32), np.full((NY,1),U0,np.float32), np.zeros((NY,1),np.float32))
def step(f, tau):
    rho = f.sum(0); ux = (f*CF[:,0,None,None]).sum(0)/rho; uy = (f*CF[:,1,None,None]).sum(0)/rho
    ux[SOLID] = 0; uy[SOLID] = 0
    fo = f - (f - feq(rho, ux, uy))/tau
    fo[:, SOLID] = f[opp][:, SOLID]                                        # bounce-back
    for i in range(9): fo[i] = np.roll(np.roll(fo[i], c[i,0], 1), c[i,1], 0)
    fo[:, :, 0] = FIN[:, :, 0]; fo[[3,6,7], :, -1] = fo[[3,6,7], :, -2]   # inlet eq / outlet copy
    return fo, ux, uy

# ---- cloud deck (2x grid) ----
CW, CH = NX*2, NY*2; P = 40; TL = 2048
def periodic_noise(h, wd, beta, seed):
    r = np.random.default_rng(seed); ky = np.fft.fftfreq(h)[:,None]; kx = np.fft.fftfreq(wd)[None,:]
    k = np.hypot(kx, ky); k[0,0] = 1; amp = k**(-beta/2); amp[0,0] = 0
    n = np.real(np.fft.ifft2(amp*np.exp(2j*np.pi*r.random((h,wd))))); return (n-n.mean())/n.std()
TEX = periodic_noise(CH, TL, 3.0, 7)
TEX = np.clip(0.75 + 0.45*TEX, 0, 1).astype(np.float32)                   # mostly-covered stratocumulus deck
cy2, cx2 = np.mgrid[0:CH, 0:CW].astype(np.float32)
SOL2 = cv2.resize(SOLID.astype(np.uint8), (CW, CH), interpolation=cv2.INTER_NEAREST) > 0
def advect(cl, ux, uy, nsteps, shift):
    """semi-Lagrangian, 2 substeps; inflow from TEX scrolled by shift (c-grid cols)."""
    u2 = cv2.resize(ux, (CW, CH)); v2 = cv2.resize(uy, (CW, CH))
    for _ in range(2):
        d = nsteps/2*2
        pad = TEX[:, (np.arange(-P, 0) + int(shift)) % TL]
        big = np.hstack([pad, cl])
        mx = (cx2 - u2*d + P).astype(np.float32); my = (cy2 - v2*d).astype(np.float32)
        cl = cv2.remap(big, mx, my % CH, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cl[SOL2] = 0; shift -= U0*d
    return cl, shift

if __name__ == '__main__':
    # schedule: list of steps per frame for main run, fork frame + branch length + branch steps/frame
    cfg = json.load(open('sched.json'))
    spf = cfg['spf']; fork = cfg['fork']; nb = cfg['nbranch']; bspf = cfg['bspf']; warm = cfg['warm']
    NF = len(spf) + nb
    UV = np.lib.format.open_memmap('uv.npy', 'w+', np.float16, (NF, 2, NY, NX))
    CL = np.lib.format.open_memmap('cl.npy', 'w+', np.uint8, (NF, CH, CW))
    f = init(); tau = tau_for(RE_MAIN); probe = []
    cl = TEX[:, :CW][:, ::-1].copy(); shift = 0.0
    for s in range(warm):
        f, ux, uy = step(f, tau)
        if s % 40 == 39: cl, shift = advect(cl, ux, uy, 40, shift)
    # main run with fork
    order = list(range(fork)) + ['B']*nb + list(range(fork, len(spf)))
    k = 0; saved = None
    for item in order:
        if item == 'B':
            if saved is None: saved = (f.copy(), cl.copy(), shift); bt = tau_for(RE_THICK)
            for _ in range(bspf): f, ux, uy = step(f, bt)
            cl, shift = advect(cl, ux, uy, bspf, shift)
        else:
            if saved is not None and item == fork: f, cl, shift = saved; saved = 'done'
            for _ in range(spf[item]):
                f, ux, uy = step(f, tau); probe.append(float(uy[int(CY0), int(CX0+3*D)]))
            cl, shift = advect(cl, ux, uy, spf[item], shift)
        UV[k, 0] = ux; UV[k, 1] = uy; CL[k] = (np.clip(cl, 0, 1)*255).astype(np.uint8); k += 1
        if k % 100 == 0: print('frame', k, flush=True)
    p = np.array(probe); p -= p.mean(); zc = np.where((p[:-1] < 0) & (p[1:] >= 0))[0]
    per = np.diff(zc[len(zc)//3:]).mean() if len(zc) > 4 else 0
    st = D/(per*U0) if per else 0
    json.dump({'period_steps': float(per), 'St': float(st), 're': RE_MAIN}, open('st.json', 'w')); print('St', st, 'period', per)
