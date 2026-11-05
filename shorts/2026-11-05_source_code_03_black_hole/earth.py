# Earth sprite: real continents (global-land-mask, NOAA GLOBE data), procedural clouds, Lambert light + atmosphere
import numpy as np, cv2
RNG = np.random.default_rng(11)
LAND = np.load('land.npy').astype(np.float32)                    # 1024 x 2048, lat 90..-90, lon -180..180
H, W = LAND.shape
def _noise(fy, fx, s):
    n = RNG.random((fy, fx)).astype(np.float32); n = np.concatenate([n, n[:, :1]], 1)
    return cv2.GaussianBlur(cv2.resize(n, (W + W//fx, H), interpolation=cv2.INTER_CUBIC)[:, :W], (0, 0), s)
latg = np.linspace(90, -90, H)[:, None]*np.ones((1, W))
n1 = _noise(16, 32, 3) + 0.5*_noise(64, 128, 1.5)
landc = np.stack([0.20+0.25*n1, 0.32+0.12*n1, 0.12+0.06*n1], -1)                                # green-brown
desert = (np.abs(np.abs(latg)-23) < 12)[..., None]*np.clip(n1-0.55, 0, 1)[..., None]*2
landc = landc*(1-desert) + np.array([0.62, 0.50, 0.32])*desert
oceanc = np.stack([0.015+0*n1, 0.07+0.02*n1, 0.22+0.05*n1], -1)
L3 = cv2.GaussianBlur(LAND, (0, 0), 0.8)[..., None]
ALB = oceanc*(1-L3) + landc*L3
ice = cv2.GaussianBlur(((np.abs(latg) > 62)*LAND + (np.abs(latg) > 78)*(1-LAND)*0.8).astype(np.float32), (0, 0), 4)[..., None]; ALB = ALB*(1-ice) + 0.85*ice
cl = _noise(24, 48, 4) + 0.6*_noise(96, 192, 2) + 0.3*_noise(256, 512, 1)
cl = cv2.GaussianBlur(cl, (0, 0), sigmaX=3, sigmaY=1.2)
CLOUD = np.clip((cl - np.percentile(cl, 55))*2.2, 0, 1).astype(np.float32)
ALB = ALB.astype(np.float32)

def sprite(D, rot, light=(-0.6, -0.3, 0.75), lcol=(1, 1, 1), gain=1.0):
    """D px diameter. returns rgb (D+pad), alpha"""
    pad = int(D*0.12); N = D + 2*pad; c = N/2
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    x = (xx-c)/(D/2); y = -(yy-c)/(D/2); r2 = x*x + y*y; inside = r2 <= 1
    z = np.sqrt(np.clip(1-r2, 0, 1))
    tilt = np.radians(23.4); yt = y*np.cos(tilt) - z*np.sin(tilt); zt = y*np.sin(tilt) + z*np.cos(tilt)
    lat = np.degrees(np.arcsin(np.clip(yt, -1, 1))); lon = np.degrees(np.arctan2(x, zt)) + rot
    mx = (np.mod(lon+180, 360)/360*W).astype(np.float32); my = ((90-lat)/180*(H-1)).astype(np.float32)
    alb = cv2.remap(ALB, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cld = cv2.remap(CLOUD, (np.mod(mx + rot*3, W)).astype(np.float32), my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)[..., None]
    lv = np.array(light, np.float32); lv /= np.linalg.norm(lv)
    ndl = np.clip(x*lv[0] + y*lv[1] + z*lv[2], 0, 1)[..., None]
    spec = (np.clip(x*lv[0]+y*lv[1]+(z+1)*lv[2]/np.sqrt(2)*0+z*lv[2], 0, 1)**30)[..., None]*0.25
    col = (alb*(1-cld) + 0.95*cld)*(ndl*1.25 + 0.02) + spec*(1-cld)
    rim = np.clip(1 - z, 0, 1)[..., None]**2.5*np.array([0.25, 0.5, 1.0])*(0.3 + 0.9*ndl)        # atmosphere
    col = (col + rim)*np.array(lcol, np.float32)*gain
    # soft limb + outer halo
    rr = np.sqrt(r2); a = np.clip((1-rr)*D/2 + 0.5, 0, 1)
    halo = np.clip(1 - (rr-1)/0.10, 0, 1)*(rr > 1)*0.6
    col = col*a[..., None] + halo[..., None]*np.array([0.2, 0.45, 1.0])*gain*np.array(lcol)*0.6
    alpha = np.clip(a + halo*0.5, 0, 1)
    return col.astype(np.float32), alpha.astype(np.float32)
