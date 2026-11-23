import mpmath as mp, numpy as np, sys
mp.mp.dps=12
part=int(sys.argv[1])
S=np.linspace(-0.1,1.1,49); T=np.linspace(6,54,1441)
Ts=T[part::2]
L=np.array([[float(abs(mp.zeta(mp.mpc(s,t)))) for s in S] for t in Ts])
np.save(f'fine_{part}.npy',L)
