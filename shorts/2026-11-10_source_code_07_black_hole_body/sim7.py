# SOURCE CODE #7 solver: falling feet-first into a Schwarzschild black hole.
# Each body piece follows a radial geodesic in its own proper time: d2r/dtau2 = -GM/r^2 (exact for radial
# Schwarzschild geodesics). Before the break the body is a spring chain in static balance with the tidal
# field 2GM/r^3 (MTW eq. 31.4 radial tidal term); it snaps when a segment stretches > BREAK_STRAIN.
import numpy as np
from scipy.integrate import solve_ivp
G, C, MSUN = 6.674e-11, 2.998e8, 1.989e30
L_BODY, N = 1.8, 10                       # 1.8 m person, 10 pieces
L0 = L_BODY/(N-1)
OMEGA2 = 6.8e4                            # spring k/m (s^-2): body holds to ~300 g of stretch (sim choice)
BREAK_STRAIN = 0.30
X = (np.arange(N) - (N-1)/2)*L0           # piece offsets, + = toward head (away from hole)

def rs(m): return 2*G*m*MSUN/C**2
def tidal(m, r, L=L_BODY): return 2*G*m*MSUN*L/r**3          # stretch accel across L (m/s^2)
def strain_max(m, r):
    g = 2*G*m*MSUN/r**3; return g*X[X > 0].sum()/(OMEGA2*L0)

def fall(m, r0_rs=100, n=2000):
    """centre of body falls from rest at r0. returns tau, r_centre, piece offsets (N) per tau, broke_r"""
    GM = G*m*MSUN; R0 = r0_rs*rs(m); RH = rs(m)
    # centre: radial geodesic from rest
    ev = lambda t, y: y[0] - RH; ev.terminal = True
    s = solve_ivp(lambda t, y: [y[1], -GM/y[0]**2], (0, 1e9), [R0, 0], events=ev, rtol=1e-10, atol=1e-6, dense_output=True)
    tau = np.linspace(0, s.t[-1], n); rc = s.sol(tau)[0]
    st = np.array([strain_max(m, r) for r in rc]); ib = np.argmax(st > BREAK_STRAIN) if (st > BREAK_STRAIN).any() else None
    off = np.zeros((n, N))
    for i in range(n):                      # intact: static stretched chain
        g = 2*GM/rc[i]**3; T = np.array([g*X[X > X[j]].sum()*0 + g*np.sum(X[j+1:]) for j in range(N-1)])  # tension/m per segment
        segs = L0*(1 + np.clip(T/(OMEGA2*L0), 0, BREAK_STRAIN))
        p = np.concatenate([[0], np.cumsum(segs)]); off[i] = p - p.mean()
    if ib is not None:                      # broken: every piece on its own geodesic from the break state
        t0 = tau[ib]; r_p0 = rc[ib] + off[ib]; v0 = s.sol(t0)[1]
        for k in range(N):
            sk = solve_ivp(lambda t, y: [y[1], -GM/y[0]**2], (t0, tau[-1]), [r_p0[k], v0], rtol=1e-11, atol=1e-6, dense_output=True)
            off[ib:, k] = sk.sol(tau[ib:])[0] - rc[ib:]
    return tau, rc, off, (rc[ib] if ib is not None else None)

if __name__ == '__main__':
    for m in (10, 4.15e6):
        R = rs(m); a_h = tidal(m, R)
        print(f'M={m:g} Msun  r_s={R/1e3:,.1f} km  stretch@edge(1.8m)={a_h:.3g} m/s2 = {a_h/9.81:.3g} g  @10rs={tidal(m,10*R)/9.81:.3g} g')
    tau, rc, off, rb = fall(10)
    print('10 Msun: fall time %.4f s, break at %.0f km (%.1f r_s), body length at edge %.1f m' % (tau[-1], rb/1e3, rb/rs(10), off[-1, -1]-off[-1, 0]))
    tau2, rc2, off2, rb2 = fall(4.15e6)
    print('Sgr A*: fall time %.0f s (%.1f h), broke=%s, length at edge %.4f m' % (tau2[-1], tau2[-1]/3600, rb2, off2[-1, -1]-off2[-1, 0]))
    np.savez('fall10.npz', tau=tau, rc=rc, off=off, rb=rb, rs=rs(10))
