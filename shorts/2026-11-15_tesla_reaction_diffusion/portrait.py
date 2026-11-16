# image -> Gray-Scott with spatially varying kill rate k (dark: pattern lives, light: dies)
import numpy as np, sys
from PIL import Image, ImageOps, ImageFilter
import gs
def load(path, n=360, m=640):
    im = Image.open(path).convert('L')
    im = ImageOps.fit(im, (m, n)[::-1] if False else (n, m), Image.LANCZOS) if False else ImageOps.fit(im, (n, m), Image.LANCZOS)
    im = ImageOps.autocontrast(im, cutoff=1).filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(im, np.float32) / 255     # shape (m, n) = (rows, cols)
def maps(L, f0=0.0545, k_dark=0.062, k_light=0.0645):
    k = k_dark + (k_light - k_dark) * L
    f = np.full_like(L, f0)
    return f.astype(np.float32), k.astype(np.float32)
def run(L, steps, snaps=(), f0=0.0545, k_dark=0.062, k_light=0.0645, seed=2):
    rng = np.random.default_rng(seed); H, W = L.shape
    f, k = maps(L, f0, k_dark, k_light)
    u = np.ones((H, W), np.float32); v = np.zeros((H, W), np.float32)
    for _ in range(140):
        y, x = rng.integers(4, H - 4), rng.integers(4, W - 4); u[y-3:y+3, x-3:x+3] = 0.5; v[y-3:y+3, x-3:x+3] = 0.25
    out = {}
    for i in range(1, steps + 1):
        gs.step(u, v, f, k)
        if i in snaps: out[i] = v.copy()
    return out, u, v
if __name__ == '__main__':
    L = load(sys.argv[1]); out, u, v = run(L, 8000, snaps=(1000, 3000, 8000))
    Image.fromarray(np.hstack([(out[s] * 3 * 255).clip(0, 255).astype(np.uint8) for s in sorted(out)] + [(L * 255).astype(np.uint8)])).save('ptest.png')
