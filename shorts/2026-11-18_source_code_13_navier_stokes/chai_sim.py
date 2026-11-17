import numpy as np, time
from ns2d import NS2D, vortex
N=512
s=NS2D(N=N,nu=1.2e-4,kappa=1.5e-5)
X,Y=s.X,s.Y; rng=np.random.default_rng(11); c=np.pi
w=vortex(X,Y,c,c,1.3,5.0)                     # spoon stir (big swirl)
for i in range(22):                            # turbulent eddies from pouring
    a=rng.uniform(0,2*np.pi); r=rng.uniform(0.25,1.9)
    w+=vortex(X,Y,c+r*np.cos(a),c+r*np.sin(a),rng.uniform(0.1,0.28),rng.choice([-1,1])*rng.uniform(3,8))
s.set_w(w)
R=np.hypot(X-c,Y-c); TH=np.arctan2(Y-c,X-c)
milk=np.exp(-((X-c-0.7)**2+(Y-c+0.3)**2)/0.35)                       # pour blob
milk+=0.8*np.exp(-((R-1.5)**2)/0.05)*(0.5+0.5*np.cos(TH-0.8))          # crescent from first pour
milk+=0.5*np.exp(-((X-c+0.9)**2+(Y-c-0.6)**2)/0.12)
n=np.fft.irfft2(np.fft.rfft2(rng.normal(size=(N,N)))*np.exp(-s.K2/400),s=(N,N)); n/=n.std()
milk=np.clip(milk*(1+0.25*n),0,1.2)
s.set_dye(0,milk)
t0=time.time()
s.advance(1.6)       # develop
print('dev',time.time()-t0)
F=700; DT=0.0062
D=np.zeros((F,N,N),np.float16); Wv=np.zeros((F,N,N),np.float16)
for f in range(F):
    D[f]=s.dye(); Wv[f]=s.w(); s.advance(DT)
    if f%100==0: print(f,time.time()-t0,flush=True)
np.save('chai_dye.npy',D); np.save('chai_w.npy',Wv)
u,v=s.vel(); print('umax',np.abs(u).max())
