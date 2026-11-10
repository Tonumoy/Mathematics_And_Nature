# SC#7 astronaut: signed-distance-field model, orthographic sphere-trace once -> normal / material / AO maps.
# Frames relight these maps (disk key light, rim, visor reflection), so every pixel is our own render.
import numpy as np
SW, SH = 800, 1290; XW = 1.24; Y0, Y1 = -0.05, 1.95; YAW = np.radians(18)
def smin(a, b, k=0.035):
    h = np.clip(0.5 + 0.5*(b-a)/k, 0, 1); return b*(1-h) + a*h - k*h*(1-h)
def sph(p, c, r): return np.linalg.norm(p - np.array(c), axis=-1) - r
def cap(p, a, b, r):
    a, b = np.array(a), np.array(b); pa, ba = p - a, b - a
    h = np.clip((pa @ ba)/(ba @ ba), 0, 1); return np.linalg.norm(pa - h[..., None]*ba, axis=-1) - r
def rbox(p, c, hs, r):
    q = np.abs(p - np.array(c)) - np.array(hs) + r
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r
def tor(p, c, R, r):
    q = p - np.array(c); return np.hypot(np.hypot(q[..., 0], q[..., 2]) - R, q[..., 1]) - r
PARTS = []                     # (name, fn, material)
def add(n, f, m): PARTS.append((n, f, m))
add('helmet', lambda p: sph(p, (0, 1.63, 0.0), 0.175), 1)
add('neck', lambda p: tor(p, (0, 1.47, 0), 0.125, 0.032), 3)
add('torso', lambda p: rbox(p, (0, 1.19, 0), (0.205, 0.26, 0.13), 0.07), 0)
add('pack', lambda p: rbox(p, (0, 1.22, -0.18), (0.20, 0.25, 0.08), 0.03), 0)
add('chest', lambda p: rbox(p, (0, 1.13, 0.145), (0.105, 0.06, 0.03), 0.012), 3)
add('belt', lambda p: rbox(p, (0, 0.91, 0), (0.205, 0.035, 0.14), 0.02), 3)
add('hips', lambda p: rbox(p, (0, 0.82, 0), (0.19, 0.10, 0.125), 0.06), 0)
for s in (-1, 1):
    add('uarm', lambda p, s=s: cap(p, (s*0.27, 1.38, 0), (s*0.355, 1.09, 0.04), 0.078), 0)
    add('larm', lambda p, s=s: cap(p, (s*0.355, 1.09, 0.04), (s*0.40, 0.84, 0.10), 0.068), 0)
    add('glove', lambda p, s=s: sph(p, (s*0.405, 0.77, 0.11), 0.07), 2)
    add('thigh', lambda p, s=s: cap(p, (s*0.10, 0.80, 0), (s*0.12, 0.43, 0.035), 0.095), 0)
    add('shin', lambda p, s=s: cap(p, (s*0.12, 0.43, 0.035), (s*0.125, 0.13, 0.0), 0.083), 0)
    add('boot', lambda p, s=s: rbox(p, (s*0.125, 0.06, 0.045), (0.085, 0.065, 0.14), 0.04), 2)
    add('knee', lambda p, s=s: sph(p, (s*0.12, 0.43, 0.06), 0.07), 3)
def sdf(p, want_mat=False):
    ds = np.stack([f(p) for _, f, _ in PARTS], -1)
    d = ds[..., 0]
    for i in range(1, ds.shape[-1]): d = smin(d, ds[..., i])
    if want_mat: return d, np.array([m for *_, m in PARTS])[np.argmin(ds, -1)]
    return d
def rot(p):                    # yaw the model
    c, s = np.cos(YAW), np.sin(YAW); x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([c*x + s*z, y, -s*x + c*z], -1)
def render():
    xs = (np.arange(SW) + 0.5)/SW*XW - XW/2; ys = Y1 - (np.arange(SH) + 0.5)/SH*(Y1 - Y0)
    X, Y = np.meshgrid(xs, ys); n = X.size
    o = np.stack([X.ravel(), Y.ravel(), np.full(n, 0.6)], -1); dirv = np.array([0, 0, -1.0])
    t = np.zeros(n); hit = np.zeros(n, bool); alive = np.ones(n, bool)
    for _ in range(90):
        idx = np.where(alive)[0]
        if len(idx) == 0: break
        p = rot(o[idx] + t[idx, None]*dirv); d = sdf(p)
        h = d < 4e-4; hit[idx[h]] = True; alive[idx[h]] = False
        t[idx] += np.maximum(d, 2e-4)*0.95
        far = t[idx] > 1.3; alive[idx[far]] = False
    P = o + t[:, None]*dirv; idx = np.where(hit)[0]; Pr = rot(P[idx])
    e = 1.5e-3; N = np.zeros((len(idx), 3))
    for k in range(3):
        dv = np.zeros(3); dv[k] = e; N[:, k] = sdf(Pr + dv) - sdf(Pr - dv)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    c, s = np.cos(-YAW), np.sin(-YAW); N = np.stack([c*N[:, 0] + s*N[:, 2], N[:, 1], -s*N[:, 0] + c*N[:, 2]], -1)   # back to view space
    _, mat = sdf(Pr, True)
    Nw = rot(N)
    ao = np.ones(len(idx))
    for k, dd in enumerate((0.02, 0.05, 0.09, 0.14)): ao -= (dd - sdf(Pr + Nw*dd)).clip(0)*(1.6/(k+1))
    # visor = front face of helmet
    vis = (mat == 1) & (Pr[:, 2] > 0.05) & ((Pr[:, 0]/0.135)**2 + ((Pr[:, 1] - 1.632)/0.078)**2 < 1)
    mat = np.where(vis, 4, mat)
    out = dict(mask=np.zeros(n, np.float32), nx=np.zeros(n, np.float32), ny=np.zeros(n, np.float32), nz=np.zeros(n, np.float32),
               mat=np.zeros(n, np.int8), ao=np.zeros(n, np.float32), y=np.zeros(n, np.float32))
    out['mask'][idx] = 1; out['nx'][idx], out['ny'][idx], out['nz'][idx] = N.T; out['mat'][idx] = mat; out['ao'][idx] = ao.clip(0.25, 1)
    out['y'][idx] = Pr[:, 1]; out['x'] = np.zeros(n, np.float32); out['x'][idx] = Pr[:, 0]
    np.savez_compressed('astro.npz', **{k: v.reshape(SH, SW) for k, v in out.items()})
    print('hit px', hit.sum())
if __name__ == '__main__': render()
