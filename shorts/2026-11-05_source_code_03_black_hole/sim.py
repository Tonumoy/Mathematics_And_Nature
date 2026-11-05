# SOURCE CODE #3 solver: light paths around a Schwarzschild black hole (G = c = 1, M = 1)
# photon orbit equation  u'' + u = 3 M u^2,  u = 1/r,  phi = angle swept.  RK45 (scipy), rtol 1e-9.
import numpy as np
from scipy.integrate import solve_ivp
M = 1.0
B_CRIT = 3*np.sqrt(3)*M                     # 5.196: aim closer than this and light never comes back
R_H, R_PS = 2*M, 3*M                        # horizon, photon sphere
U0 = 1/60.0                                 # start 60 M away

def ray(b, phimax=6*np.pi, n=1600):
    """trace one ray with impact parameter b>0 coming from far away. returns phi, u, captured"""
    def f(p, y): return [y[1], 3*M*y[0]**2 - y[0]]
    def hit(p, y): return y[0] - 1/R_H
    hit.terminal = True
    def out(p, y): return y[0] - U0*0.999
    out.terminal = True; out.direction = -1
    up0 = np.sqrt(max(1/b**2 - U0**2 + 2*M*U0**3, 0))
    p0 = np.arcsin(min(b*U0, 1))
    s = solve_ivp(f, (p0, phimax), [U0, up0], events=(hit, out), rtol=1e-9, atol=1e-12, dense_output=True, max_step=0.01)
    ph = np.linspace(p0, s.t[-1], n); u = s.sol(ph)[0]
    return ph, np.clip(u, 1e-6, 1/R_H), len(s.t_events[0]) > 0

def path_xy(b, n=1600):
    """ray coming from the left at height b (M units). returns x, y arrays (M units, BH at 0) + captured"""
    ph, u, cap = ray(abs(b), n=n)
    th = np.pi - ph; r = 1/u
    return r*np.cos(th), np.sign(b)*r*np.sin(th), cap

def table(bmax=40.0, nb=3000, nphi=2400, phimax=3*np.pi):
    """U[b_i, phi_j] for the camera render; NaN after capture or escape. also phi_inf (escape angle)"""
    bs = np.linspace(0.02, bmax, nb); P = np.linspace(0, phimax, nphi)
    U = np.full((nb, nphi), np.nan, np.float32); pinf = np.full(nb, np.nan, np.float32)
    for i, b in enumerate(bs):
        ph, u, cap = ray(b, phimax=phimax, n=4000)
        m = (P >= ph[0]) & (P <= ph[-1]); U[i, m] = np.interp(P[m], ph, u)
        U[i, P < ph[0]] = U0
        if not cap: pinf[i] = ph[-1] + (ph[0])          # symmetric tail ~ total sweep
    return bs, P, U, pinf

if __name__ == '__main__':
    for b in (9, 6, 5.3, 5.2, 5.19, 5.0):
        ph, u, cap = ray(b); print(b, 'captured' if cap else 'escapes', 'sweep %.2f rad' % (ph[-1]-ph[0]), 'r_min %.3f' % (1/u.max()))
    # weak-field check vs Einstein 4M/b
    ph, u, cap = ray(200.0, phimax=4); print('deflection b=200:', ph[-1]+ph[0]-np.pi, 'GR 4M/b =', 4/200)
    bs, P, U, pinf = table(); np.savez_compressed('table.npz', bs=bs, P=P, U=U, pinf=pinf); print('table ok')
