import numpy as np
from scipy.integrate import solve_ivp
m=np.array([3.,4.,5.])
def f(t,y):
    r=y[:6].reshape(3,2); a=np.zeros((3,2))
    for i in range(3):
        for j in range(3):
            if i!=j:
                d=r[j]-r[i]; a[i]+=m[j]*d/np.linalg.norm(d)**3
    return np.concatenate([y[6:],a.ravel()])
def run(r0,T=70):
    y0=np.concatenate([np.array(r0).ravel(),np.zeros(6)])
    s=solve_ivp(f,(0,T),y0,method='DOP853',rtol=1e-13,atol=1e-13,dense_output=True)
    return s
r0=[[1,3],[-2,-1],[1,-1]]
def outcome(s):
    r=s.y[:6,-1].reshape(3,2); d=[np.linalg.norm(r[1]-r[2]),np.linalg.norm(r[0]-r[2]),np.linalg.norm(r[0]-r[1])]
    return np.argmax([min(d[k] for k in range(3) if k!=i) for i in range(3)]), d
if __name__=='__main__':
    s=run(r0); print('base',outcome(s), s.t.size)
    for eps in [1e-6,1e-5,1e-4,1e-3]:
        r=np.array(r0,float); r[0,0]+=eps
        print(eps,outcome(run(r)))
