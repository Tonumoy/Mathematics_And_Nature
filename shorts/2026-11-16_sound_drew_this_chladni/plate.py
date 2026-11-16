# Free square Kirchhoff plate: Ritz method, Legendre basis on [-1,1]^2 (real solver, no closed-form fakes)
import numpy as np, scipy.linalg as sl, functools
from numpy.polynomial import legendre as L

NU = 0.3; NDEG = 26           # basis degree per axis
E, RHO, SIDE = 200e9, 7850.0, 0.24   # steel plate 24 cm square

def _basis(xq, n):
    P = np.zeros((3, n + 1, len(xq)))
    for a in range(n + 1):
        c = np.zeros(a + 1); c[a] = np.sqrt((2 * a + 1) / 2.0)   # orthonormal Legendre
        P[0, a] = L.legval(xq, c); P[1, a] = L.legval(xq, L.legder(c, 1)); P[2, a] = L.legval(xq, L.legder(c, 2))
    return P

@functools.lru_cache(None)
def solve(n=NDEG):
    xq, wq = L.leggauss(n + 8)
    P = _basis(xq, n)
    A = lambda p, q: (P[p] * wq) @ P[q].T
    A00, A11, A22, A20, A02 = A(0, 0), A(1, 1), A(2, 2), A(2, 0), A(0, 2)
    K = (np.kron(A22, A00) + np.kron(A00, A22) + NU * (np.kron(A20, A02) + np.kron(A02, A20))
         + 2 * (1 - NU) * np.kron(A11, A11))
    M = np.kron(A00, A00)
    K = 0.5 * (K + K.T)
    lam, V = sl.eigh(K, M)
    return lam, V

def hz(lam, h):
    """frequency in Hz for thickness h (m). domain [-1,1] -> physical side SIDE."""
    D = E * h ** 3 / (12 * (1 - NU ** 2))
    om = np.sqrt(np.maximum(lam, 0)) * (2.0 / SIDE) ** 2 * np.sqrt(D / (RHO * h))
    return om / (2 * np.pi)

@functools.lru_cache(None)
def grid_modes(G=384, n=NDEG):
    lam, V = solve(n)
    x = np.linspace(-1, 1, G)
    P = _basis(x, n)[0]                      # (n+1, G)
    C = V.reshape(n + 1, n + 1, -1)          # coeffs [a,b,k]
    # field[k, y, x] = sum_ab C[a,b,k] P_a(x) P_b(y)
    F = np.einsum('abk,ax,by->kyx', C, P, P, optimize=True)
    c0 = _basis(np.array([0.0]), n)[0][:, 0]
    at0 = np.einsum('abk,a,b->k', C, c0, c0)
    return lam, F.astype(np.float32), at0

if __name__ == '__main__':
    lam, V = solve()
    print('first 10 lam*4 (expect ~0,0,0,13.47,19.6,24.3,34.8,34.8,61.1..)', np.round(np.sqrt(np.maximum(lam[:10],0)) * 4, 2))
    f = hz(lam, 1e-3); lam2, F, at0 = grid_modes()
    sel = [k for k in range(3, 120) if abs(at0[k]) > 0.05 * np.abs(at0[3:120]).max()]
    print('center-driven modes Hz:', [(k, round(f[k])) for k in sel][:40])
