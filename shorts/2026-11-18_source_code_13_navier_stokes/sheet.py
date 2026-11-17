import sys,numpy as np,cv2,time
import render
ts=[float(x) for x in sys.argv[1].split(',')]; out=sys.argv[2]
t0=time.time(); ims=[]
for t in ts:
    fr=render.frame(t); ims.append(cv2.resize(cv2.cvtColor(fr,cv2.COLOR_RGB2BGR),(270,480),interpolation=cv2.INTER_AREA))
    cv2.imwrite(f'prev_{t:.1f}.png',cv2.cvtColor(fr,cv2.COLOR_RGB2BGR))
cols=5; rows=[ims[i:i+cols] for i in range(0,len(ims),cols)]
rows=[r+[np.zeros_like(ims[0])]*(cols-len(r)) for r in rows]
cv2.imwrite(out,np.vstack([np.hstack(r) for r in rows])); print('s/frame',(time.time()-t0)/len(ts))
