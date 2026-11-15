# explainer sims (all Gray-Scott), saved as uint8 stacks of v and u
import numpy as np, gs
def run(N, f, k, steps, every, mask=None, init=None, fk_path=None, seed=0):
    rng = np.random.default_rng(seed)
    u = np.ones((N, N), np.float32); v = np.zeros((N, N), np.float32)
    if init is not None: init(u, v, rng)
    V, U = [], []
    for i in range(1, steps + 1):
        if fk_path is not None: f, k = fk_path(i / steps)
        gs.step(u, v, f, k)
        if mask is not None: v *= mask; u[mask == 0] = 1
        if i % every == 0:
            V.append((v * 255 * 2.5).clip(0, 255).astype(np.uint8)); U.append((u * 255).clip(0, 255).astype(np.uint8))
    return np.stack(V), np.stack(U), u, v
N = 300
y, x = np.mgrid[0:N, 0:N]; dish = ((x - N / 2) ** 2 + (y - N / 2) ** 2 < (N * 0.46) ** 2).astype(np.float32)
def one_seed(u, v, rng):
    c = N // 2; u[c-4:c+4, c-4:c+4] = 0.5; v[c-4:c+4, c-4:c+4] = 0.25
# 1) mitosis: one spark copies itself until the dish is full
V1, U1, u, v = run(N, 0.0367, 0.0649, 12000, 20, dish, one_seed)
np.savez_compressed('e_mitosis.npz', V=V1, U=U1)
# 2) dial: spots -> stripes (continue from the full dish)
def cont(uu, vv, rng): uu[:] = u; vv[:] = v
V2, U2, _, _ = run(N, 0, 0, 5000, 10, dish, cont, fk_path=lambda a: (0.0367 + (0.029 - 0.0367) * min(1, a * 1.6), 0.0649 + (0.057 - 0.0649) * min(1, a * 1.6)))
np.savez_compressed('e_dial.npz', V=V2, U=U2)
# 3) masks: fish, palate arch, limb paddle, membrane
def rand_seed(u, v, rng):
    m = rng.random(u.shape) < 0.004; v[m] = 0.5; u[m] = 0.5
from PIL import Image, ImageDraw
def mk(draw_fn):
    im = Image.new('L', (N, N), 0); draw_fn(ImageDraw.Draw(im)); return (np.asarray(im) > 127).astype(np.float32)
fish = mk(lambda d: (d.ellipse([30, 95, 250, 205], fill=255), d.polygon([(235, 150), (290, 100), (290, 200)], fill=255)))
palate = mk(lambda d: (d.ellipse([40, 40, 260, 330], fill=255), d.ellipse([90, 120, 210, 330], fill=0)))
limb = mk(lambda d: (d.ellipse([50, 40, 250, 230], fill=255), d.rectangle([110, 180, 190, 300], fill=255)))
for name, msk in (('fish', fish), ('palate', palate), ('limb', limb), ('membrane', np.ones((N, N), np.float32))):
    f, k = (0.029, 0.057) if name != 'membrane' else (0.0367, 0.0649)
    Vm, Um, _, _ = run(N, f, k, 4000, 40, msk, rand_seed, seed=3)
    np.savez_compressed(f'e_{name}.npz', V=Vm, U=Um, M=msk)
print('ok')
