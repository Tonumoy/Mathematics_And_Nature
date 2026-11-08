# real colour maths: spectrum -> CIE 1931 XYZ -> sRGB (colour-science)
import numpy as np, colour
SH = colour.SpectralShape(300, 830, 1)
CMFS = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(SH)
WL = np.arange(300, 831, 1.0)
def planck(T):  # spectral radiance shape, W sr^-1 m^-3 (relative)
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23; l = WL*1e-9
    return 1/(l**5*(np.exp(h*c/(l*k*T)) - 1))
def bands(rngs, w=None):  # emission bands (nm ranges, from Kosanke tables via Wikipedia 'Pyrotechnic colorant')
    s = np.zeros_like(WL)
    for i, (a, b) in enumerate(rngs):
        s += (w[i] if w else 1)*np.exp(-0.5*((WL-(a+b)/2)/((b-a)/2.355+0.6))**2)
    return s
EMIT = {  # molecule: bands nm
    'SrCl': ([(633, 639), (658, 664), (672, 678), (605, 611)], [1.0, 0.8, 0.5, 0.4]),   # SrCl + SrOH (Tarantik & Klapoetke 2009)
    'BaCl': ([(511.8, 515.8), (522.1, 526.1)], [1.0, 0.9]),
    'Na':   ([(588.8, 589.2), (589.4, 589.8)], [1.0, 0.5]),
    'CuCl': ([(426, 430), (433, 437), (441, 445), (420, 452)], [1.0, 0.9, 0.8, 0.4]),
}
def xyz(spec):
    sd = colour.SpectralDistribution(dict(zip(WL, spec)))
    return colour.sd_to_XYZ(sd, CMFS, shape=SH, method='Integration') if False else np.array([np.sum(spec*CMFS.values[:, i]) for i in range(3)])
def srgb(spec, sat_boost=1.0):
    X = xyz(spec); X = X/X[1]
    rgb = colour.XYZ_to_sRGB(X*0.5)
    rgb = np.clip(rgb, 0, None); rgb = rgb/max(rgb.max(), 1e-9)
    return np.clip(rgb, 0, 1)
def planck_rgb(T): return srgb(planck(T))
def mol_rgb(m): b, w = EMIT[m]; return srgb(bands(b, w))
def line_rgb(lam_nm):  # colour of one wavelength; 0 outside vision
    if lam_nm < 380 or lam_nm > 780: return np.zeros(3), 0.0
    s = np.exp(-0.5*((WL-lam_nm)/2)**2); X = xyz(s); Y = X[1]
    return srgb(s), float(Y/ np.max(CMFS.values[:, 1])/2/np.sqrt(2*np.pi)*1.0)
if __name__ == '__main__':
    for m in EMIT: print(m, np.round(mol_rgb(m)*255).astype(int))
    for T in (2000, 5800, 1e5): print(T, np.round(planck_rgb(T)*255).astype(int))
    for l in (620, 525, 428, 354): print(l, line_rgb(l))
