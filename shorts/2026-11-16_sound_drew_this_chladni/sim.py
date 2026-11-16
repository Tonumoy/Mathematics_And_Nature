# sand on a center-driven free steel plate: driven response of the real modes + grain hopping
import numpy as np, cv2, sys
from plate import grid_modes, hz
import timeline as tl
FPS = 30; NG = 80000; SUB = 6; G = 384; Q = 28.0
lam, F, at0 = grid_modes(G)
F1 = hz(lam, 1e-3)
KS = [k for k in range(3, 140) if abs(at0[k]) > 0.02 * np.abs(at0[3:140]).max()]
FK = F[KS]; AK = at0[KS]; FR = F1[KS]

def response(f, h_mm):
    fk = FR * h_mm
    den = fk ** 2 - f ** 2 + 1j * f * fk / Q
    c = AK / den
    R = np.tensordot(c, FK, axes=(0, 0))
    return np.abs(R).astype(np.float32), float(np.abs(c).max() * np.abs(FK).max())

def ramp(t, a, b): return float(np.clip((t - a) / max(b - a, 1e-6), 0, 1))
def sm(x): return x * x * (3 - 2 * x)
def glide(t, t0, f0, f1, d=0.45):
    x = sm(ramp(t, t0, t0 + d)); return f0 * (f1 / f0) ** x

S, E = tl.s, tl.e
sweep_t0 = S('L05'); sweep_t1 = S('L07')
SW = tl.SEQ[3:]                     # 711 ... 3950
step = (sweep_t1 - 1.6 - sweep_t0) / (len(SW) - 1)

def drive(t):
    """returns (f_Hz, h_mm, on, reset)"""
    f = F1
    if t < S('L02'): return f[tl.HOOK], 1.0, 1.0
    if t < min(S('L03'), S('L02') + 3.8): return f[5], 1.0, ramp(t, S('L02') + 0.5, S('L02') + 0.8)
    if t < S('L04'): return glide(t, min(S('L03'), S('L02') + 3.8), f[5], f[10]), 1.0, 1.0
    if t < S('L05'): return glide(t, S('L04') - 0.2, f[10], f[16]), 1.0, 1.0
    if t < sweep_t1:
        i = min(int((t - sweep_t0) / step), len(SW) - 1)
        prev = f[16] if i == 0 else f[SW[i - 1]]
        return glide(t, sweep_t0 + i * step, prev, f[SW[i]], 0.35), 1.0, 1.0
    if t < S('L08'):   # back to the twist mode, then thicken
        fT = glide(t, S('L07'), f[tl.HOOK], f[tl.TWIST], 0.5)
        h = 1.0 + sm(ramp(t, S('L07') + 2.0, S('L08'))) * 1.0
        return fT, h, 1.0
    fT = f[tl.TWIST]
    t_up = S('L08') + 0.55 * (E('L08') - S('L08'))
    if t < S('L09') + 1.2:
        return glide(t, t_up, fT, 2 * fT, 1.0), 2.0, 1.0
    # loop: thickness back to 1 mm, tone to the hook note -> the hook pattern re-forms
    hb = 2.0 - sm(ramp(t, S('L09') + 1.2, S('L09') + 2.4))
    return glide(t, S('L09') + 1.2, 2 * fT, f[tl.HOOK], 1.2), hb, 1.0

def hdial(t):
    return drive(t)[1]

def run(T=None, seed=1, out='sand.npz'):
    T = T or tl.T
    rng = np.random.default_rng(seed)
    nf = int(round(T * FPS))
    P = rng.uniform(-0.97, 0.97, (NG, 2)).astype(np.float32)
    # hook: pre-settle on the hook mode
    def stepper(P, A, gain, dt):
        # A: amplitude field normalised 0..~1.5
        gy, gx = np.gradient(A); gx = gx.astype(np.float32); gy = gy.astype(np.float32)
        ix = ((P[:, 0] + 1) * 0.5 * (G - 1)).astype(np.float32); iy = ((P[:, 1] + 1) * 0.5 * (G - 1)).astype(np.float32)
        mp = lambda M: cv2.remap(M, ix.reshape(-1, 250), iy.reshape(-1, 250), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        a = mp(A); ax = mp(gx); ay = mp(gy)
        hop = 0.040 * gain * np.clip(a, 0, 1.6) ** 0.7
        P[:, 0] += hop * rng.standard_normal(NG).astype(np.float32) - 0.05 * gain * ax * dt * (G / 2)
        P[:, 1] += hop * rng.standard_normal(NG).astype(np.float32) - 0.05 * gain * ay * dt * (G / 2)
        np.clip(P, -0.985, 0.985, out=P)
        return a
    def field(f, h):
        R, _ = response(f, h)
        ref = np.abs(F[min(KS, key=lambda k: abs(F1[k] * h - f))]).max()  # amplitude of nearest pure mode
        Rn = R / (np.abs(AK).max() / (F1[KS].min() ** 2 / Q)) # unused scale
        return R
    # reference: resonance amplitude scale per mode -> normalise by peak of the strongest pure response
    def norm_field(f, h):
        R, _ = response(f, h)
        # resonant peak height for the nearest mode at this f
        k = int(np.argmin(np.abs(FR * h - f)))
        peak = abs(AK[k]) * Q / ((FR[k] * h) ** 2) * np.abs(FK[k]).max()
        return R / (peak + 1e-12)
    Ph = P.copy()
    A = norm_field(F1[tl.HOOK], 1.0)
    for _ in range(450): stepper(Ph, A, 1.0, 0.004)
    frames = np.zeros((nf, NG, 2), np.float16); amps = np.zeros((nf, NG), np.float16)
    info = np.zeros((nf, 3), np.float32)
    for n in range(nf):
        t = n / FPS
        f, h, on = drive(t)
        if t < tl.s('L02'):
            Pc = Ph
        else:
            Pc = P
        A = norm_field(f, h)
        for _ in range(SUB): a = stepper(Pc, A, on, 0.004)
        frames[n] = Pc; amps[n] = a * on; info[n] = (f, h, on)
        if n % 150 == 0: print('sim', n, nf, round(f), h, flush=True)
    np.savez_compressed(out, P=frames, A=amps, info=info)

if __name__ == '__main__':
    run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
