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
