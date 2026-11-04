# re-run only the cloud deck over the stored solver velocities (new cellular stratocumulus texture)
import numpy as np, json, sim
def cells(h, w, seed):
    n1 = sim.periodic_noise(h, w, 2.0, seed); n2 = sim.periodic_noise(h, w, 3.0, seed+1)
    return np.clip(0.88 - 0.6*np.exp(-(n1/0.35)**2) + 0.08*n2, 0, 1).astype(np.float32)
sim.TEX = cells(sim.CH, sim.TL, 21)
# lee clearing: air sinking off the island's peak warms and dries -> cloud evaporates just downstream of the island
yy, xx = np.mgrid[0:sim.CH, 0:sim.CW].astype(np.float32)/2
r = np.hypot(xx-sim.CX0, yy-sim.CY0); LEE = (np.exp(-((r-sim.D/2)/4.0)**2)*(xx > sim.CX0-4)).astype(np.float32)
_adv = sim.advect
def adv(cl, u, v, n, sh):
    cl, sh = _adv(cl, u, v, n, sh); return cl*(1-np.clip(0.02*n, 0, 0.9)*LEE), sh
sim.advect = adv
cfg = json.load(open('sched.json')); order = cfg['spf'][:cfg['fork']] + [cfg['bspf']]*cfg['nbranch'] + cfg['spf'][cfg['fork']:]
UV = np.load('uv.npy', mmap_mode='r'); CL = np.lib.format.open_memmap('cl.npy', 'w+', np.uint8, (len(order), sim.CH, sim.CW))
cl = sim.TEX[:, :sim.CW].copy(); sh = 0.0; u0 = UV[0, 0].astype(np.float32); v0 = UV[0, 1].astype(np.float32)
for r in range(4):
    for j in range(61): cl, sh = sim.advect(cl, UV[j, 0].astype(np.float32), UV[j, 1].astype(np.float32), order[j], sh)
for k, n in enumerate(order):
    if k == cfg['fork'] + cfg['nbranch']: pass
    cl, sh = sim.advect(cl, UV[k, 0].astype(np.float32), UV[k, 1].astype(np.float32), n, sh)
    CL[k] = (np.clip(cl, 0, 1)*255).astype(np.uint8)
CL.flush(); print('ok', CL[::200].mean(axis=(1, 2)))
