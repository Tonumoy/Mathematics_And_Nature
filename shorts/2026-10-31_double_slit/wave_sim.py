# Split-step Fourier solver for a 2D electron wave packet hitting a double / single slit.
# hbar = m = 1, grid units. Saves frames of psi (complex64) for the video.
import numpy as np, sys, time

# ---------------- globals ----------------
N        = 448          # grid points per side
DT       = 0.5          # time step
LAMBDA   = 6.0          # de Broglie wavelength in grid units (sim scale, not real scale)
K0       = 2*np.pi/LAMBDA
Y0       = 70.0         # packet start row
SIG_X    = 90.0         # wide packet across x -> coherent over both slits
SIG_Y    = 14.0
BAR_Y    = 150          # barrier row
BAR_T    = 6            # barrier thickness
SLIT_W   = 6            # slit width   (real: 62 nm)
SLIT_D   = 26           # centre-to-centre (real: 272 nm)  -> d/w = 4.4 like Bach 2013
V_WALL   = 1e4
STEPS    = 760
SAVE_EVERY = 4
ABS_W    = 40           # absorbing border width

def run(n_slits):
    x = np.arange(N) - N/2
    X, Y = np.meshgrid(x, np.arange(N))
    psi = np.exp(-((X)**2)/(2*SIG_X**2) - ((Y-Y0)**2)/(2*SIG_Y**2)) * np.exp(1j*K0*Y)
    psi /= np.sqrt((np.abs(psi)**2).sum())
    wall = (Y >= BAR_Y) & (Y < BAR_Y+BAR_T)
    c1 = -SLIT_D/2; c2 = SLIT_D/2
    open1 = np.abs(X - c1) <= SLIT_W/2
    open2 = np.abs(X - c2) <= SLIT_W/2
    hole = open1 | open2 if n_slits == 2 else open1
    V = np.where(wall & ~hole, V_WALL, 0.0)
    # absorbing mask (left/right/bottom/top)
    d = np.minimum.reduce([X + N/2, N/2 - 1 - X, Y.astype(float), N - 1 - Y])
    absorb = np.where(d < ABS_W, np.clip(np.cos(np.pi/2*np.clip(ABS_W-d,0,ABS_W)/ABS_W),0,1)**0.25, 1.0)
    kx = 2*np.pi*np.fft.fftfreq(N); KX, KY = np.meshgrid(kx, kx)
    Tk = np.exp(-1j*(KX**2+KY**2)/2*DT).astype(np.complex64)
    keep = (V == 0).astype(np.float32)   # hard wall: psi = 0 inside the barrier
    psi = psi.astype(np.complex64); absorb = absorb.astype(np.float32)
    frames = []
    for s in range(STEPS):
        psi = np.fft.ifft2(Tk*np.fft.fft2(psi))
        psi = psi*keep*absorb
        if s % SAVE_EVERY == 0:
            frames.append(psi.copy())
    return np.array(frames, dtype=np.complex64), V > 0

if __name__ == "__main__":
    t = time.time()
    f2, w2 = run(2); f1, w1 = run(1)
    np.savez("waves.npz", f2=f2[::1].astype(np.complex64), f1=f1.astype(np.complex64), w2=w2, w1=w1)
    print("frames", f2.shape, "secs", round(time.time()-t, 1))
