"""3D vortex tube being stretched (Kelvin: circulation conserved -> omega * r^2 = const)."""
import numpy as np, cv2
from common import *

rng = np.random.default_rng(21)
NP = 2000
TH0 = rng.uniform(0, 2 * np.pi, NP)
Z0 = rng.uniform(-0.5, 0.5, NP)
RH = np.sqrt(rng.uniform(0.02, 1, NP))
SPD = 1.0 / (0.35 + 0.65 * RH ** 2) * 0.6 + 0.4   # inner core a bit faster (visual only, same scaling)
CXT, CYT = 520, 970
R0, L0 = 190.0, 600.0
FOC = 1400.0

def render(theta, s, red=0.0, tilt=0.32, yaw=0.0, alpha=1.0, trail=0.35):
    """theta: accumulated spin angle (rad) for s=1 particles; s: stretch factor"""
    r = max(R0 / np.sqrt(s), 1.6); L = L0 * min(s, 6.0) ** 0.9
    lay = np.zeros((H, W, 3), np.float32)
    col_c = np.array(CYAN, np.float32) / 255; col_w = np.array([1, 1, 1], np.float32); col_r = np.array(RED, np.float32) / 255
    base = col_c * (1 - red) + col_r * red
    nseg = 5
    ct, st = np.cos(tilt), np.sin(tilt)
    pts = []
    for k in range(nseg + 1):
        th = TH0 + theta * SPD - trail * (1 - k / nseg)
        x = r * RH * np.cos(th + yaw); zd = r * RH * np.sin(th + yaw); y = Z0 * L
        # tilt about x axis (see the tube a little from above)
        y2 = y * ct - zd * st; z2 = y * st + zd * ct
        p = FOC / (FOC + z2)
        pts.append((CXT + x * p, CYT + y2 * p, z2))
    depth = (pts[-1][2] - pts[-1][2].min()) / (np.ptp(pts[-1][2]) + 1e-6)
    br = (0.5 + 0.9 * (1 - depth)) * alpha
    for i in range(NP):
        c = (base * (0.6 + 0.4 * (1 - RH[i])) + col_w * 0.25) * br[i]
        poly = np.array([[pts[k][0][i] * 4, pts[k][1][i] * 4] for k in range(nseg + 1)], np.int32)
        cv2.polylines(lay, [poly], False, c.tolist(), 1, cv2.LINE_AA, shift=2)
    # soft core glow
    g = np.zeros((H, W), np.float32)
    ytop = CYT - L / 2 * ct; ybot = CYT + L / 2 * ct
    cv2.line(g, (CXT, int(ytop)), (CXT, int(ybot)), 1.0, max(int(r * 0.5), 2), cv2.LINE_AA)
    g = cv2.GaussianBlur(g, (0, 0), max(r * 0.5, 4))
    lay += g[..., None] * base * 0.35 * alpha
    return lay, r, L
