# final portrait v2 (owner: thinner, compact, fills the whole screen like the reel): full 1080x1920 grid, D 0.35,
# f 0.060 (dark) -> 0.030 (light), k 0.0600 -> 0.0630; snapshots every 20 steps -> memmap uint8
import numpy as np
import gs, portrait as P
L = P.load('tesla.jpg', 1080, 1920)
k = (0.0600 + 0.0030 * L).astype(np.float32); f = (0.0600 - 0.0300 * L).astype(np.float32)
rng = np.random.default_rng(7); u = np.ones(L.shape, np.float32); v = np.zeros(L.shape, np.float32)
m = rng.random(L.shape) < 0.008
ys, xs = np.nonzero(m)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        yy = np.clip(ys + dy, 0, L.shape[0] - 1); xx = np.clip(xs + dx, 0, L.shape[1] - 1); u[yy, xx] = 0.5; v[yy, xx] = 0.25
NS, EV = 6000, 20
out = np.lib.format.open_memmap('snaps_v2.npy', mode='w+', dtype=np.uint8, shape=(NS // EV,) + L.shape)
for i in range(1, NS + 1):
    gs.step(u, v, f, k, Du=0.35, Dv=0.175)
    if i % EV == 0: out[i // EV - 1] = (v * 3 * 255).clip(0, 255).astype(np.uint8)
out.flush(); np.save('L_v2.npy', L); print('ok')
