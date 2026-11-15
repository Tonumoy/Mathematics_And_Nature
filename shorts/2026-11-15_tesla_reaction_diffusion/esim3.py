import numpy as np, gs, math
from PIL import Image, ImageDraw
from esim_lib import run, N
def blocks(n):
    def init(u, v, rng):
        for _ in range(n):
            y, x = rng.integers(4, N-4), rng.integers(4, N-4); u[y-2:y+3, x-2:x+3] = 0.5; v[y-2:y+3, x-2:x+3] = 0.25
    return init
def mk(fn):
    im = Image.new('L', (N, N), 0); fn(ImageDraw.Draw(im)); return (np.asarray(im) > 127).astype(np.float32)
fish = mk(lambda d: (d.ellipse([30, 95, 250, 205], fill=255), d.polygon([(235, 150), (290, 100), (290, 200)], fill=255)))
palate = mk(lambda d: (d.ellipse([40, 30, 260, 340], fill=255), d.ellipse([85, 110, 215, 360], fill=0)))
limb = mk(lambda d: (d.ellipse([45, 30, 255, 230], fill=255), d.rectangle([105, 170, 195, 300], fill=255)))
# fish/palate: stripes param; palate seeded with transverse bands so ridges run across the arch
V, U, _, _ = run(N, 0.029, 0.057, 4000, 40, fish, blocks(120), seed=3); np.savez_compressed('e_fish.npz', V=V, U=U, M=fish)
def pal_init(u, v, rng):
    for r in range(70, 160, 18):
        yy, xx = np.mgrid[0:N, 0:N]; d = np.hypot(xx - 150, (yy - 185) * 0.72)
        m = (np.abs(d - r) < 2.5); v[m] = 0.3; u[m] = 0.5
V, U, _, _ = run(N, 0.029, 0.057, 3000, 30, palate, pal_init, seed=4); np.savez_compressed('e_palate.npz', V=V, U=U, M=palate)
def limb_init(u, v, rng):
    yy, xx = np.mgrid[0:N, 0:N]; ang = np.arctan2(yy - 230, xx - 150); r = np.hypot(xx - 150, yy - 230)
    for a in np.linspace(-2.45, -0.69, 5):
        m = (np.abs(ang - a) < 0.035) & (r > 60) & (r < 185); v[m] = 0.3; u[m] = 0.5
V, U, _, _ = run(N, 0.026, 0.059, 3000, 30, limb, limb_init, seed=5); np.savez_compressed('e_limb.npz', V=V, U=U, M=limb)
V, U, _, _ = run(N, 0.0367, 0.0649, 6000, 60, None, blocks(60), seed=6); np.savez_compressed('e_membrane.npz', V=V, U=U)
print('ok')
