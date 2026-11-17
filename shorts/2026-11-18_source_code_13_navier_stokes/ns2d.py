"""Real 2D incompressible Navier-Stokes solver (vorticity form, pseudo-spectral).
  dw/dt + (u.grad)w = nu lap w + F,   u = curl^-1 w,   periodic box [0,2pi)^2
  passive dye c:  dc/dt + (u.grad)c = kappa lap c
RK3 (Williamson low-storage style) with exact integrating factor for diffusion, 2/3 dealiasing.
"""
import numpy as np
import scipy.fft as sf
def rfft2(a): return sf.rfft2(a, workers=-1)
def irfft2(a, s=None): return sf.irfft2(a, s=s, workers=-1)

class NS2D:
    def __init__(self, N=512, nu=2e-4, kappa=5e-5, ndye=1):
        self.N = N; self.nu = nu; self.kappa = kappa
        k = np.fft.fftfreq(N, 1.0 / N)
        kr = np.fft.rfftfreq(N, 1.0 / N)
        self.KX, self.KY = np.meshgrid(kr, k)          # shape (N, N//2+1)
        self.K2 = self.KX**2 + self.KY**2
        self.K2i = np.where(self.K2 == 0, 0, 1.0 / np.where(self.K2 == 0, 1, self.K2))
        kmax = N / 3.0
        self.dealias = (np.abs(self.KX) < kmax) & (np.abs(self.KY) < kmax)
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, x)
        self.wh = np.zeros_like(self.K2, dtype=complex)
        self.ch = [np.zeros_like(self.wh) for _ in range(ndye)]
        self.force = None  # real-space forcing of vorticity (optional)
        self.t = 0.0

    def set_w(self, w): self.wh = rfft2(w) * self.dealias
    def set_dye(self, i, c): self.ch[i] = rfft2(c) * self.dealias
    def w(self): return irfft2(self.wh, s=(self.N, self.N))
    def dye(self, i=0): return irfft2(self.ch[i], s=(self.N, self.N))

    def vel(self, wh=None):
        wh = self.wh if wh is None else wh
        psih = wh * self.K2i
        u = irfft2(1j * self.KY * psih, s=(self.N, self.N))
        v = irfft2(-1j * self.KX * psih, s=(self.N, self.N))
        return u, v

    def _adv(self, fh, u, v):
        fx = irfft2(1j * self.KX * fh, s=(self.N, self.N))
        fy = irfft2(1j * self.KY * fh, s=(self.N, self.N))
        return -rfft2(u * fx + v * fy) * self.dealias

    def rhs(self, wh, chs):
        u, v = self.vel(wh)
        rw = self._adv(wh, u, v)
        if self.force is not None: rw = rw + self.force
        rc = [self._adv(c, u, v) for c in chs]
        return rw, rc, u, v

    def cfl_dt(self, cfl=0.5):
        u, v = self.vel()
        vmax = max(np.abs(u).max(), np.abs(v).max(), 1e-6)
        return cfl * (2 * np.pi / self.N) / vmax

    def step(self, dt):
        # Lawson RK4 with exact integrating factor for diffusion (stable for advection)
        Ew = np.exp(-self.nu * self.K2 * dt * 0.5); Ew2 = Ew * Ew
        Ec = np.exp(-self.kappa * self.K2 * dt * 0.5); Ec2 = Ec * Ec
        w0, c0 = self.wh, self.ch
        k1w, k1c, _, _ = self.rhs(w0, c0)
        k2w, k2c, _, _ = self.rhs(Ew * (w0 + 0.5 * dt * k1w), [Ec * (c + 0.5 * dt * k) for c, k in zip(c0, k1c)])
        k3w, k3c, _, _ = self.rhs(Ew * w0 + 0.5 * dt * k2w, [Ec * c + 0.5 * dt * k for c, k in zip(c0, k2c)])
        k4w, k4c, _, _ = self.rhs(Ew2 * w0 + dt * Ew * k3w, [Ec2 * c + dt * Ec * k for c, k in zip(c0, k3c)])
        self.wh = Ew2 * w0 + dt / 6 * (Ew2 * k1w + 2 * Ew * (k2w + k3w) + k4w)
        self.ch = [Ec2 * c + dt / 6 * (Ec2 * a + 2 * Ec * (b + d) + e) for c, a, b, d, e in zip(c0, k1c, k2c, k3c, k4c)]
        self.t += dt

    def advance(self, T, cfl=0.5, dtmax=0.02):
        t_end = self.t + T
        while self.t < t_end - 1e-12:
            dt = min(self.cfl_dt(cfl), dtmax, t_end - self.t)
            self.step(dt)

def vortex(X, Y, x0, y0, r, s):
    """Gaussian vortex on the periodic box (sum over nearest images)."""
    w = np.zeros_like(X)
    L = 2 * np.pi
    for ox in (-L, 0, L):
        for oy in (-L, 0, L):
            w += s * np.exp(-((X - x0 + ox)**2 + (Y - y0 + oy)**2) / r**2)
    return w
