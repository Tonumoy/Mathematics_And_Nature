# SC#6 solver: why the sky is blue (and not violet) - real single-scattering radiative transfer
#   beta(lam) = (tau550 / H) * (550/lam)^n      Rayleigh: n = 4 (Bodhaine et al. 1999: tau(550) ~ 0.097 at sea level)
#   L(lam, view) = sum_s  E_sun(lam) * beta(lam) * rho(h) * P(theta) * exp(-(tau_view + tau_sun)) ds     P = 3/(16 pi) (1 + cos^2)
#   spectrum -> CIE 1931 2deg (Wyman-Sloan-Shirley 2013 fit) -> XYZ -> linear sRGB -> gamma
#   sunlight: Planck 5778 K (top of atmosphere approximation)
import numpy as np
LAM = np.arange(380, 721, 10.0)
H, TAU550, RE, RTOP = 8000.0, 0.097, 6371e3, 6371e3 + 80e3
def planck(lam_nm, T):
    l = lam_nm*1e-9; return 1/(l**5*(np.exp(1.4388e-2/(l*T)) - 1))
SUN = planck(LAM, 5778.0); SUN /= SUN.max()
def _g(x, mu, s1, s2): s = np.where(x < mu, s1, s2); return np.exp(-0.5*((x - mu)/s)**2)
def cmf(l):
    x = 1.056*_g(l, 599.8, 37.9, 31.0) + 0.362*_g(l, 442.0, 16.0, 26.7) - 0.065*_g(l, 501.1, 20.4, 26.2)
    y = 0.821*_g(l, 568.8, 46.9, 40.5) + 0.286*_g(l, 530.9, 16.3, 31.1)
    z = 1.217*_g(l, 437.0, 11.8, 36.0) + 0.681*_g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)
CMF = cmf(LAM)                                   # (nl, 3)
M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
def spec_to_lin(S):                              # S (..., nl) -> linear sRGB (..., 3)
    return (S @ CMF) @ M.T
def gamma(c): c = np.clip(c, 0, None); return np.where(c <= 0.0031308, 12.92*c, 1.055*np.power(c, 1/2.4) - 0.055)
def beta(n): return TAU550/H*(550.0/LAM)**n

def _path_to_top(h0, mu):                        # distance from height h0 along direction with cos(zenith)=mu to RTOP
    r = RE + h0; return -r*mu + np.sqrt(np.maximum((r*mu)**2 - r*r + RTOP**2, 0))
def sky_spectrum(el_deg, az_deg, sun_el_deg, n=4.0, ns=24, nsun=10):
    """el, az: arrays (view elevation/azimuth from sun); returns spectral radiance (..., nl)"""
    el = np.radians(np.maximum(el_deg, 0.3)); az = np.radians(az_deg); se = np.radians(sun_el_deg)
    v = np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], -1)
    s = np.array([np.cos(se), 0.0, np.sin(se)])
    cth = v @ s; P = 3/(16*np.pi)*(1 + cth**2)
    b = beta(n)                                 # (nl,)
    L = _path_to_top(0.0, v[..., 2]); ds = L/ns
    out = np.zeros(el.shape + (len(LAM),)); tv = np.zeros(el.shape)
    for i in range(ns):
        d = (i + 0.5)*ds
        p = v*d[..., None] + np.array([0, 0, RE])
        r = np.linalg.norm(p, axis=-1); h = r - RE; rho = np.exp(-h/H)
        tv_mid = tv + rho*ds/2; tv += rho*ds
        mu_s = (p @ s)/r                         # sun direction cos zenith at sample
        Ls = _path_to_top(h, mu_s); dss = Ls/nsun; ts = np.zeros_like(h)
        for j in range(nsun):
            q = p + s*((j + 0.5)*dss)[..., None]; hq = np.linalg.norm(q, axis=-1) - RE
            ts += np.exp(-np.maximum(hq, 0)/H)*dss
        # blocked by Earth: if the sun ray from p hits the ground
        blocked = (mu_s < 0) & (r*r*(1 - mu_s**2) < RE*RE)
        att = np.exp(-(tv_mid + ts)[..., None]*b)
        out += np.where(blocked[..., None], 0, SUN*b*rho[..., None]*att)*ds[..., None]
    return out*P[..., None]

def sky_image(sun_el=35.0, n=4.0, W=54, Hh=96, el_lo=0.5, el_hi=70.0, az_c=150.0, az_w=40.0, expo=None):
    el = np.linspace(el_hi, el_lo, Hh)[:, None]*np.ones((1, W))
    az = (az_c + np.linspace(-az_w/2, az_w/2, W))[None, :]*np.ones((Hh, 1))
    S = sky_spectrum(el, az, sun_el, n)
    lin = spec_to_lin(S)
    if expo is None: expo = 0.80/np.percentile(lin[..., 1], 90)
    return gamma(lin*expo), expo

def colour_of(S, expo=1.0):                       # spectrum -> display colour normalised to max channel
    c = spec_to_lin(S); c = np.clip(c, 0, None); c = c/c.max()
    return gamma(c*expo)

if __name__ == '__main__':
    print('beta ratio 450/700:', round((700/450)**4, 2), ' 400/700:', round((700/400)**4, 2))
    print('tau 400/550/700:', [round(TAU550*(550/l)**4, 3) for l in (400, 550, 700)])
    im, e = sky_image(); print('zenith rgb', np.round(im[0, 27], 3), 'horizon', np.round(im[-1, 27], 3))
    im0, _ = sky_image(n=0.0, expo=e); print('n=0 zenith', np.round(im0[0, 27], 3), 'horizon', np.round(im0[-1, 27], 3))
    scat = SUN*(550/LAM)**4; print('scattered peak nm', LAM[np.argmax(scat)], 'eye-weighted peak nm', LAM[np.argmax(scat*CMF[:, 1])])
    print('sky colour', np.round(colour_of(scat), 3))
