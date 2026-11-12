# SC#9 evolution engine: 3D prey with tiny neural brains vs AI sharks (co-evolving). No flocking rule coded.
# prey brain: 4 scalar senses -> 6 hidden (tanh) -> 3 muscles (flee shark, go to friends, match friends)
# shark brain: 3 senses -> 4 hidden -> 2 muscles (chase nearest, chase loner)
# confusion: strike success = 1 / (prey within R_CONF of the target)   (Olson et al. 2013 J R Soc Interface)
import numpy as np, sys, pickle, time
N, K, R = 200, 3, 55.0
R_SEE, R_PRED, R_CONF, R_STRIKE = 10.0, 25.0, 6.0, 2.0
import os
BURST = float(os.environ.get('BURST', '0.5'))
VMIN, VMAX, VPRED, CD = 0.4, 1.0, float(os.environ.get('VPRED', '1.5')), 12
R_STARTLE = float(os.environ.get('R_STARTLE', '5.5'))                             # startle zone: the escape muscle is stronger and fish can burst (C-start)
PI, PH, PO = 4, 6, 3
QI, QH, QO = 3, 4, 2
def n_prey_w(): return PI*PH + PH + PH*PO + PO
def n_pred_w(): return QI*QH + QH + QH*QO + QO
def mlp(W, x, I, Hh, O):                     # W (n, nw), x (n, I)
    a = 0; W1 = W[:, a:a+I*Hh].reshape(-1, I, Hh); a += I*Hh
    b1 = W[:, a:a+Hh]; a += Hh
    W2 = W[:, a:a+Hh*O].reshape(-1, Hh, O); a += Hh*O
    b2 = W[:, a:a+O]
    h = np.tanh(np.einsum('ni,nih->nh', x, W1) + b1)
    return np.tanh(np.einsum('nh,nho->no', h, W2) + b2), h
def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v/np.maximum(n, 1e-9)
def rand_ball(rng, n, r):
    v = unit(rng.normal(size=(n, 3))); return v*(r*rng.uniform(0, 1, (n, 1))**(1/3))

class World:
    def __init__(s, seed=0, prey_w=None, pred_w=None, evolve=True, confusion=True):
        s.rng = np.random.default_rng(seed)
        s.P = rand_ball(s.rng, N, R*0.9); s.V = unit(s.rng.normal(size=(N, 3)))*0.7
        s.Q = rand_ball(s.rng, K, R*0.6); s.U = unit(s.rng.normal(size=(K, 3)))*0.8
        s.W = prey_w.copy() if prey_w is not None else s.rng.normal(0, 1.0, (N, n_prey_w()))
        s.WQ = pred_w.copy() if pred_w is not None else s.rng.normal(0, 1.0, (K, n_pred_w()))
        s.age = np.zeros(N); s.cd = np.zeros(K, int); s.catch = np.zeros(K); s.evolve = evolve; s.conf = confusion
        s.deaths = 0; s.t = 0; s.events = []; s.mask_friends = False; s.out = np.zeros((N, PO)); s.hid = np.zeros((N, PH))
    def step(s):
        rng = s.rng; P, V, Q = s.P, s.V, s.Q
        d = P[:, None] - P[None]; D = np.sqrt((d**2).sum(-1)); np.fill_diagonal(D, 1e9)
        M = D < R_SEE; n = M.sum(1)
        cen = (M[..., None]*P[None]).sum(1)/np.maximum(n, 1)[:, None]
        avv = (M[..., None]*V[None]).sum(1)/np.maximum(n, 1)[:, None]
        tc = np.where(n[:, None] > 0, cen - P, 0.0); al = np.where(n[:, None] > 0, avv, 0.0)
        dq = P[:, None] - Q[None]; DQ = np.sqrt((dq**2).sum(-1)); j = DQ.argmin(1); dp = DQ[np.arange(N), j]
        away = unit(dq[np.arange(N), j]); cp = np.clip(1 - dp/R_PRED, 0, 1)
        x = np.stack([cp, n/10.0, np.linalg.norm(tc, axis=1)/R_SEE, np.linalg.norm(V, axis=1)], 1)
        if s.mask_friends: x[:, 1] = 0; x[:, 2] = 0
        o, h = mlp(s.W, x, PI, PH, PO); s.out, s.hid = o, h
        if s.mask_friends: tc = np.zeros_like(tc); al = np.zeros_like(al)   # friend sensor cut: blind to friends
        near = np.clip(1 - dp/R_STARTLE, 0, 1)[:, None]; s.near = near
        acc = 0.12*(o[:, :1]*away*(cp[:, None] > 0)*(1 + 4*near**2) + o[:, 1:2]*unit(tc) + o[:, 2:3]*unit(al)) + 0.03*rng.normal(size=(N, 3))
        Mc = D < 1.6; rep = (Mc[..., None]*d/np.maximum(D, 1e-6)[..., None]*(1.6 - np.minimum(D, 1.6))[..., None]).sum(1)
        acc += 0.08*rep                                     # bodies cannot overlap (physics, not brain)
        r = np.linalg.norm(P, axis=1, keepdims=True); acc -= 0.05*(P/r)*np.clip((r - R)/4, 0, 3)
        V = V + acc; sp = np.linalg.norm(V, axis=1, keepdims=True); V = V/sp*np.clip(sp, VMIN, VMAX*(1 + BURST*near*np.clip(o[:, :1], 0, 1)))
        s.P = P + V; s.V = V; s.age += 1
        # sharks
        DQt = DQ.T; vis = DQt < 35
        near = DQt.argmin(1); dn = DQt[np.arange(K), near]
        crowd = n[near]
        iso_score = np.where(vis, -n[None, :] - 0.02*DQt, -1e9); iso = iso_score.argmax(1)
        xq = np.stack([np.clip(dn/35, 0, 1), crowd/10.0, np.clip(DQt[np.arange(K), iso]/35, 0, 1)], 1)
        oq, _ = mlp(s.WQ, xq, QI, QH, QO)
        to_n = unit(s.P[near] - Q); to_i = unit(s.P[iso] - Q)
        has = vis.any(1)[:, None]
        accq = np.where(has, 0.16*(oq[:, :1]*to_n + oq[:, 1:2]*to_i), -0.02*unit(Q)) + 0.02*rng.normal(size=(K, 3))
        rq = np.linalg.norm(Q, axis=1, keepdims=True); accq -= 0.06*(Q/rq)*np.clip((rq - R)/4, 0, 3)
        U = s.U + accq; sq = np.linalg.norm(U, axis=1, keepdims=True); s.U = U/sq*np.clip(sq, 0.5, VPRED); s.Q = Q + s.U
        s.cd = np.maximum(s.cd - 1, 0)
        for k in range(K):
            dd = np.linalg.norm(s.P - s.Q[k], axis=1); tgt = dd.argmin()
            if s.cd[k] == 0 and dd[tgt] < R_STRIKE:
                s.cd[k] = CD
                nc = (np.linalg.norm(s.P - s.P[tgt], axis=1) < R_CONF).sum() if s.conf else 1
                hit = rng.uniform() < 1.0/nc
                s.events.append((s.t, k, int(tgt), bool(hit), int(nc), s.P[tgt].copy()))
                if hit:
                    s.catch[k] += 1; s.deaths += 1; s.respawn(tgt)
        s.t += 1
    def respawn(s, i):
        rng = s.rng
        s.P[i] = rand_ball(rng, 1, R*0.95)[0]; s.V[i] = unit(rng.normal(size=3))*0.7
        if s.evolve:
            cand = rng.choice(N, 3, replace=False); par = cand[s.age[cand].argmax()]
            s.W[i] = s.W[par] + rng.normal(0, 0.12, n_prey_w())*(rng.uniform(size=n_prey_w()) < 0.3)
        s.age[i] = 0
    def evolve_sharks(s):
        b, w = s.catch.argmax(), s.catch.argmin()
        if b != w: s.WQ[w] = s.WQ[b] + s.rng.normal(0, 0.12, n_pred_w())
        s.catch[:] = 0
    def stats(s):
        D = np.sqrt(((s.P[:, None]-s.P[None])**2).sum(-1)); np.fill_diagonal(D, 1e9)
        return float(np.median(D.min(1))), float((D < R_SEE).sum(1).mean())

if __name__ == '__main__':
    steps = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    w = World(seed); snaps = {0: (w.W.copy(), w.WQ.copy())}; log = []; t0 = time.time(); g_last = 0; dl = 0
    for i in range(steps):
        w.step()
        if i % 600 == 599: w.evolve_sharks()
        g = w.deaths//N
        if g > g_last: snaps[g] = (w.W.copy(), w.WQ.copy()); g_last = g
        if i % 2000 == 1999:
            nn, nb = w.stats(); rate = (w.deaths - dl)/2000*1000; dl = w.deaths
            o = w.out.mean(0); log.append((i+1, g, nn, nb, rate, *o))
            print('step %d gen %d nn %.2f nbrs %.1f catch/1k %.1f | flee %.2f friends %.2f match %.2f | %.0fs' % (i+1, g, nn, nb, rate, *o, time.time()-t0), flush=True)
    pickle.dump(dict(snaps=snaps, log=log, W=w.W, WQ=w.WQ), open(f'evo_s{seed}.pkl', 'wb'))
