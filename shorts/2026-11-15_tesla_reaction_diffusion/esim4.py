import numpy as np
from PIL import Image, ImageDraw
from esim_lib import run, N
def mk(fn):
    im = Image.new('L', (N, N), 0); fn(ImageDraw.Draw(im)); return (np.asarray(im) > 127).astype(np.float32)
palate = mk(lambda d: (d.ellipse([45, 25, 255, 330], fill=255), d.rectangle([45, 200, 255, 300], fill=255)))
def pal_init(u, v, rng):
    yy, xx = np.mgrid[0:N, 0:N]
    for y0 in range(60, 230, 22):
        m = np.abs(yy - (y0 + 0.0022 * (xx - 150) ** 2)) < 3.0; v[m] = 0.3; u[m] = 0.5
V, U, _, _ = run(N, 0.029, 0.057, 1500, 15, palate, pal_init, seed=4); np.savez_compressed('e_palate.npz', V=V, U=U, M=palate)
limb = mk(lambda d: (d.ellipse([45, 30, 255, 230], fill=255), d.rectangle([105, 170, 195, 300], fill=255)))
def limb_init(u, v, rng):
    yy, xx = np.mgrid[0:N, 0:N]; ang = np.arctan2(yy - 235, xx - 150); r = np.hypot(xx - 150, yy - 235)
    for a in np.linspace(-2.55, -0.59, 5):
        m = (np.abs(ang - a) < 0.06) & (r > 70) & (r < 190); v[m] = 0.3; u[m] = 0.5
V, U, _, _ = run(N, 0.029, 0.057, 900, 9, limb, limb_init, seed=5); np.savez_compressed('e_limb.npz', V=V, U=U, M=limb)
def blocks(u, v, rng):
    for _ in range(700):
        y, x = rng.integers(4, N-4), rng.integers(4, N-4); u[y-2:y+3, x-2:x+3] = 0.5; v[y-2:y+3, x-2:x+3] = 0.25
V, U, _, _ = run(N, 0.0367, 0.0649, 4000, 40, None, blocks, seed=6); np.savez_compressed('e_membrane.npz', V=V, U=U)
print('ok')
