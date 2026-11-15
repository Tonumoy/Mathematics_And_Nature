# Gray-Scott reaction-diffusion (Pearson 1993 form). u = food, v = spark. Du = 2 Dv.
import numpy as np
def lap(a):
    return (np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1))*0.2 + \
           (np.roll(np.roll(a,1,0),1,1)+np.roll(np.roll(a,1,0),-1,1)+np.roll(np.roll(a,-1,0),1,1)+np.roll(np.roll(a,-1,0),-1,1))*0.05 - a
def step(u, v, f, k, Du=1.0, Dv=0.5, dt=1.0, mask=None):
    uvv = u*v*v
    u += dt*(Du*lap(u) - uvv + f*(1-u))
    v += dt*(Dv*lap(v) + uvv - (f+k)*v)
    if mask is not None: v *= mask
    np.clip(u, 0, 1, out=u); np.clip(v, 0, 1, out=v)
    return u, v
def seed(n, m, pts, r=3, rng=None):
    u = np.ones((n, m), np.float32); v = np.zeros((n, m), np.float32)
    for y, x in pts:
        u[y-r:y+r, x-r:x+r] = 0.5; v[y-r:y+r, x-r:x+r] = 0.25
    if rng is not None: v += rng.uniform(0, 0.01, v.shape).astype(np.float32)
    return u, v
