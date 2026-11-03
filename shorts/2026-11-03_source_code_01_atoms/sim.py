# Hydrogen atom: real radial Schrodinger solver (finite differences, atomic units)
#   -1/2 u'' + [l(l+1)/(2 r^2) - 1/r] u = E u ,  psi = u(r)/r * Y_lm
import numpy as np
from scipy.linalg import eigh_tridiagonal
HARTREE_EV = 27.211386
def radial(l, rmax=80.0, N=6000):
    r = np.linspace(rmax/N, rmax, N); h = r[1]-r[0]
    d = 1/h**2 + l*(l+1)/(2*r**2) - 1/r
    e = -0.5/h**2*np.ones(N-1)
    E, U = eigh_tridiagonal(d, e, select='i', select_range=(0, 3))
    U = U/np.sqrt(h)                            # normalise: sum u^2 h = 1
    return r, E*HARTREE_EV, U                   # E[k] -> n = l+1+k
def shoot(l, E_ev, rmax=40.0, N=4000):
    """Numerov outward integration at a NON-allowed energy -> u(r) blows up"""
    r = np.linspace(rmax/N, rmax, N); h = r[1]-r[0]; E = E_ev/HARTREE_EV
    k2 = 2*(E + 1/r) - l*(l+1)/r**2
    u = np.zeros(N); u[0] = r[0]**(l+1); u[1] = r[1]**(l+1)
    c = h*h/12
    for i in range(1, N-1):
        u[i+1] = (2*u[i]*(1-5*c*k2[i]) - u[i-1]*(1+c*k2[i-1]))/(1+c*k2[i+1])
    return r, u
ANG = {(0, 0): lambda c: np.ones_like(c), (1, 0): lambda c: c**2, (2, 0): lambda c: (3*c**2-1)**2}
def sample(n, l, M, seed=1):
    """M electron 'measurements': r from solver |u|^2, angles from |Y_l0|^2 (rejection)"""
    rng = np.random.default_rng(seed)
    r, E, U = radial(l); u = U[:, n-l-1]
    P = u**2; cdf = np.cumsum(P); cdf /= cdf[-1]
    rr = np.interp(rng.random(M), cdf, r)
    f = ANG[(l, 0)]; cos = np.empty(0)
    while len(cos) < M:
        c = rng.uniform(-1, 1, 4*M); keep = rng.random(4*M) < f(c)/f(np.array([1.0]))[0]
        cos = np.concatenate([cos, c[keep]])
    cos = cos[:M]; sin = np.sqrt(1-cos**2); ph = rng.uniform(0, 2*np.pi, M)
    pts = np.stack([rr*sin*np.cos(ph), rr*cos, rr*sin*np.sin(ph)], 1)   # y = z-axis (vertical)
    return pts, E[n-l-1], r, P
if __name__ == '__main__':
    for l in (0, 1, 2):
        r, E, U = radial(l); print('l', l, 'E(eV)', np.round(E, 3), 'exact', [round(-13.6057/(l+1+k)**2, 3) for k in range(4)])
    r, u = shoot(0, -13.6057/6.25); print('shoot n=2.5 |u| end', abs(u[-1]))
