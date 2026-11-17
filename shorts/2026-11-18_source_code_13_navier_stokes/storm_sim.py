"""Cyclone from above: same NS solver, one big low-pressure vortex winding cloud bands."""
import numpy as np, time
from ns2d import NS2D, vortex
N=384
s=NS2D(N=N,nu=3e-4,kappa=6e-5)
X,Y=s.X,s.Y; c=np.pi; rng=np.random.default_rng(5)
R=np.hypot(X-c,Y-c); TH=np.arctan2(Y-c,X-c)
w=vortex(X,Y,c,c,0.9,7.0)-vortex(X,Y,c,c,2.2,0.9)
for i in range(30):
    a=rng.uniform(0,2*np.pi); r=rng.uniform(0.5,2.8)
    w+=vortex(X,Y,c+r*np.cos(a),c+r*np.sin(a),rng.uniform(0.08,0.2),rng.choice([-1,1])*rng.uniform(2,5))
s.set_w(w)
def noise(sig):
    n=np.fft.irfft2(np.fft.rfft2(rng.normal(size=(N,N)))*np.exp(-s.K2/sig),s=(N,N)); return n/n.std()
n1=noise(300); n2=noise(60)
arms=0.5+0.5*np.cos(2*TH+3.2*R)
cl=np.clip(0.55*arms+0.35*n1+0.25*n2-0.15,0,1)*np.clip((R-0.25)/0.35,0,1)*np.clip((3.0-R)/0.6,0,1)
cl+=0.9*np.exp(-((R-0.45)**2)/0.02)   # eyewall
s.set_dye(0,np.clip(cl,0,1))
t0=time.time(); s.advance(0.6); print('dev',time.time()-t0,flush=True)
F=110; DT=0.006
D=np.zeros((F,N,N),np.float16)
for f in range(F): D[f]=s.dye(); s.advance(DT)
np.save('storm_dye.npy',D); print('done',time.time()-t0)
