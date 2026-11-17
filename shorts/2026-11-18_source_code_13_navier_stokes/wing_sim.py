"""Flow past a wing (NACA 0012, 14 deg) with the same Navier-Stokes solver + Brinkman penalisation."""
import numpy as np, time, cv2
from ns2d import NS2D, irfft2, rfft2
N=384; U=1.0
s=NS2D(N=N,nu=1.6e-3,kappa=2e-4)
X,Y=s.X,s.Y
# NACA 0012 airfoil
c=1.5; x0=1.4; y0=np.pi; aoa=np.deg2rad(14)
xr=(X-x0)*np.cos(aoa)-(Y-y0)*np.sin(-aoa); yr=(X-x0)*np.sin(-aoa)+(Y-y0)*np.cos(aoa)
xc=np.clip(xr/c,0,1)
yt=5*0.12*c*(0.2969*np.sqrt(xc)-0.1260*xc-0.3516*xc**2+0.2843*xc**3-0.1015*xc**4)
chi=((xr>=0)&(xr<=c)&(np.abs(yr)<=yt)).astype(float)
chi=cv2.GaussianBlur(chi,(0,0),1.0); np.save('wing_mask.npy',chi.astype(np.float32))
eta=0.005
sponge=np.clip(1-np.abs(((X+0.6)%(2*np.pi))-0.6)/0.6,0,1)**2   # strip near x=0 (wraps)
streak=(0.5+0.5*np.cos(Y*14))**8
def rhs(wh,chs):
    u,v=s.vel(wh); ut=u+U
    w=irfft2(wh,s=(N,N))
    rw=s._adv(wh,ut,v)
    Fx=-(chi/eta)*ut; Fy=-(chi/eta)*v
    Fxh=rfft2(Fx); Fyh=rfft2(Fy)
    rw=rw+(1j*s.KX*Fyh-1j*s.KY*Fxh)*s.dealias
    rw=rw+rfft2(-8.0*sponge*w)*s.dealias
    rc=[]
    for ch in chs:
        cc=irfft2(ch,s=(N,N))
        r=s._adv(ch,ut,v)+rfft2(12.0*sponge*(streak-cc)-6.0*chi*cc)*s.dealias
        rc.append(r)
    return rw,rc,u,v
s.rhs=rhs
s.set_dye(0,streak*sponge)
# tiny asymmetric kick to trigger shedding
s.set_w(0.05*np.exp(-((X-3.5)**2+(Y-3.3)**2)/0.1))
t0=time.time()
def adv(T):
    tend=s.t+T
    while s.t<tend-1e-9:
        u,v=s.vel(); vm=max(np.abs(u+U).max(),np.abs(v).max())
        dt=min(0.5*(2*np.pi/N)/vm,0.004,tend-s.t); s.step(dt)
adv(7.0); print('dev',time.time()-t0,flush=True)
F=120; DT=0.025
D=np.zeros((F,N,N),np.float16)
for f in range(F):
    D[f]=s.dye(); adv(DT)
np.save('wing_dye.npy',D); print('done',time.time()-t0)
