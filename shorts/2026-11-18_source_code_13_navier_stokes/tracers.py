import numpy as np, scipy.fft as sf, cv2, time
N=512; cell=2*np.pi/N
Wv=np.load('chai_w.npy',mmap_mode='r')
k=np.fft.fftfreq(N,1/N); kr=np.fft.rfftfreq(N,1/N); KX,KY=np.meshgrid(kr,k); K2=KX**2+KY**2; K2[0,0]=1
def vel(w):
    wh=sf.rfft2(np.asarray(w,np.float32),workers=-1); ps=wh/K2; ps[0,0]=0
    return sf.irfft2(1j*KY*ps,s=(N,N),workers=-1).astype(np.float32), sf.irfft2(-1j*KX*ps,s=(N,N),workers=-1).astype(np.float32)
def samp(F,x,y):
    return cv2.remap(F,(x/cell).astype(np.float32).reshape(1,-1),(y/cell).astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP).ravel()
rng=np.random.default_rng(3); M=6000
r=2.45*np.sqrt(rng.uniform(0,1,M)); a=rng.uniform(0,2*np.pi,M)
x=np.pi+r*np.cos(a); y=np.pi+r*np.sin(a)
F0,F1=150,700; DT=0.0062
P=np.zeros((F1-F0,M,2),np.float32); Spd=np.zeros((F1-F0,M),np.float16)
t0=time.time()
for f in range(F0,F1):
    u,v=vel(Wv[f]); P[f-F0,:,0]=x; P[f-F0,:,1]=y
    uu=samp(u,x,y); vv=samp(v,x,y); Spd[f-F0]=np.hypot(uu,vv)
    xm=x+0.5*DT*uu; ym=y+0.5*DT*vv
    x=x+DT*samp(u,xm,ym); y=y+DT*samp(v,xm,ym)
np.save('tracers.npy',P); np.save('tracer_spd.npy',Spd); print(time.time()-t0)
