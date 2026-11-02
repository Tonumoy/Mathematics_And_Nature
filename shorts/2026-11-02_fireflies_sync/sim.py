# Pulse-coupled integrate-and-fire fireflies (Mirollo & Strogatz 1990, SIAM J Appl Math 50:1645).
# State x = f(phi) = ln(1+(e^b-1)phi)/b (concave down). x reaches 1 -> flash, reset to 0.
# Every firefly that SEES a flash (within view radius) jumps x += eps/deg (capped at 1 -> flashes too).
# Extra realism: each firefly's own clock differs a little (period T_i, +-SPREAD), so with eps=0 they drift apart.
# Exact event-driven integration, no time step. No leader: same rule for all.
import numpy as np
B = 3.0
def f(p): return np.log1p((np.exp(B)-1)*p)/B
def finv(x): return np.expm1(B*x)/(np.exp(B)-1)
def run(Tn, K, eps_of_t, t_end, seed=1):
    """returns ev_t (m,), ev_phi (m,n) phases right AFTER each event, flashes list of index arrays"""
    n = len(Tn); rng = np.random.default_rng(seed)
    phi = rng.random(n); t = 0.0; deg = max(1.0, K.sum(1).mean())
    ev_t, ev_phi, fl = [0.0], [phi.copy()], [np.array([], int)]
    while t < t_end:
        dt = ((1-phi)*Tn).min(); t += dt; phi = phi + dt/Tn
        e = eps_of_t(t)/deg
        fired = np.zeros(n, bool); new = phi >= 1-1e-9
        while new.any():
            fired |= new
            x = f(np.clip(phi, 0, 1)) + e*K[new].sum(0)
            hit = (x >= 1) & ~fired
            phi = np.where(fired, 0.0, finv(np.minimum(x, 1)))
            new = hit
        ev_t.append(t); ev_phi.append(phi.copy()); fl.append(np.where(fired)[0])
    return np.array(ev_t), np.array(ev_phi), fl
def phase_at(ev_t, ev_phi, Tn, t):
    k = np.searchsorted(ev_t, t, 'right')-1
    return np.clip(ev_phi[k] + (t-ev_t[k])/Tn, 0, 1)
def order(ph): return float(np.abs(np.exp(2j*np.pi*ph).mean()))
