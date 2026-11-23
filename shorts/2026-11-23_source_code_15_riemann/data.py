import mpmath as mp, numpy as np, time
mp.mp.dps=15
t0=time.time()
Z=[float(mp.zetazero(n).imag) for n in range(1,601)]
print('zeros',time.time()-t0, Z[:5], Z[-1])
S=np.linspace(-0.3,1.3,81); T=np.linspace(0.5,52,521)
L=np.zeros((len(T),len(S)))
for i,t in enumerate(T):
    for j,s in enumerate(S):
        L[i,j]=float(abs(mp.zeta(mp.mpc(s,t))))
print('grid',time.time()-t0)
np.savez('zeta_data.npz',zeros=np.array(Z),S=S,T=T,absz=L)
