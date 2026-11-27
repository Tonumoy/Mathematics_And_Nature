# SOURCE CODE #16 - why are Saturn's rings flat?
# Real collisional ring model: N ice particles orbit a point mass (GM=1, Kepler) and collide
# inelastically (DSMC-style pairing in 3D cells). After each bump the relative speed along the
# contact normal is reversed and scaled by e (coefficient of restitution): v_n' = -e v_n.
# Momentum (so angular momentum, locally) is conserved; random up-down energy is lost (1-e^2).
import numpy as np, sys
def run(e, N=6000, frames=420, sub=6, dtf=0.06, seed=7, rate=0.5, init=None):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(1.0**2, 2.0**2, N))
    th = rng.uniform(0, 2*np.pi, N)
    inc = np.abs(rng.normal(0, 0.28, N)); node = rng.uniform(0, 2*np.pi, N)
    # circular orbit in an inclined plane
    p = np.stack([r*np.cos(th), r*np.sin(th), 0*r], 1)
    vc = 1/np.sqrt(r)
    v = np.stack([-vc*np.sin(th), vc*np.cos(th), 0*r], 1)
    def rotx(a, X):  # rotate about node line
        c, s = np.cos(a), np.sin(a)
        k = np.stack([np.cos(node), np.sin(node), 0*node], 1)
        kx = np.cross(k, X); kd = (k*X).sum(1, keepdims=True)
        return X*c[:, None] + kx*s[:, None] + k*kd*(1-c[:, None])
    p = rotx(inc, p); v = rotx(inc, v)
    v += rng.normal(0, 0.02, v.shape) * vc[:, None]
    if init is not None: p, v = init[0].copy(), init[1].copy()
    dt = dtf/sub; cell = 0.09
    P = np.zeros((frames, N, 3), np.float32); hz = []; flashes = []
    acc = lambda X: -X/(np.linalg.norm(X, axis=1, keepdims=True)**2+0.25)**1.5
    a = acc(p); ncol = 0
    for f in range(frames):
        fl = []
        for s in range(sub):
            v += 0.5*dt*a; p += dt*v; a = acc(p); v += 0.5*dt*a
            # collisions
            ci = np.floor(p/cell).astype(np.int64) + 1000
            key = (ci[:, 0]*4000 + ci[:, 1])*4000 + ci[:, 2]
            order = np.lexsort((rng.random(N), key)); ks = key[order]
            same = ks[:-1] == ks[1:]
            start = np.r_[True, ks[1:] != ks[:-1]]
            idx = np.arange(N); grp0 = np.maximum.accumulate(np.where(start, idx, 0)); rank = idx - grp0
            m = same & (rank[:-1] % 2 == 0)
            i = order[:-1][m]; j = order[1:][m]
            if len(i) == 0: continue
            vr = v[i] - v[j]; sp = np.linalg.norm(vr, axis=1)
            hit = rng.random(len(i)) < np.clip(rate*sp*dt/cell*8, 0, 1)
            i, j, vr = i[hit], j[hit], vr[hit]
            n = rng.normal(size=(len(i), 3)); n /= np.linalg.norm(n, axis=1, keepdims=True)
            vn = (vr*n).sum(1); n[vn > 0] *= -1; vn = -np.abs(vn)
            dv = -(1+e)/2*vn[:, None]*n
            v[i] += dv; v[j] -= dv; ncol += len(i)
            if s == 0: fl.append(p[i[:40]].copy())
        P[f] = p; hz.append(float(np.median(np.abs(p[:, 2]))))
        flashes.append(np.concatenate(fl) if fl else np.zeros((0, 3)))
    return P, np.array(hz), flashes, ncol, (p.copy(), v.copy())
if __name__ == "__main__":
    def save(nm, P, hz, fl):
        F = np.zeros((len(fl), 40, 3), np.float32) + np.nan
        for k, x in enumerate(fl): F[k, :len(x)] = x[:40]
        np.savez_compressed(f'sim_{nm}.npz', P=P, hz=hz, F=F)
    P, hz, fl, nc, stA = run(0.1); save('A', P, hz, fl); print('A', hz[0], hz[-1])
    P, hz, fl, nc, stC = run(1.0, init=stA, frames=300, rate=float(sys.argv[1]) if len(sys.argv) > 1 else 3.0); save('B', P, hz, fl); print('B(flat->bouncy)', [round(hz[k], 3) for k in (0, 50, 100, 200, 299)])
    P, hz, fl, nc, _ = run(0.1, init=stC, frames=200, rate=1.5); save('C', P, hz, fl); print('C(back)', [round(hz[k], 3) for k in (0, 50, 100, 199)])
