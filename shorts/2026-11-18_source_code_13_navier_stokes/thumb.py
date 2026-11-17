import numpy as np, cv2
import render, chai, xray
from common import *
fi=158
base=chai.render(fi,0.0)
xr=xray.render(fi,np.pi,np.pi,render.SC0,ox=chai.CX,oy=chai.CY,disk=(chai.CX,chai.CY,chai.R_LIQ),gridpx=48); render.wire_cup(xr,0.9); xr=glow(xr,5,0.7)
yy,xx=np.mgrid[0:H,0:W]
split=(xx>chai.CX+ (yy-chai.CY)*0.25).astype(np.float32)[...,None]
img=base*(1-split)+xr*split
u8=to_u8(img)
# split line
p0=(int(chai.CX+(0-chai.CY)*0.25),0); p1=(int(chai.CX+(H-chai.CY)*0.25),H)
ov=u8.copy(); cv2.line(ov,p0,p1,CYAN,4,cv2.LINE_AA); u8=cv2.addWeighted(ov,1,u8,0,0)
p=render.tr_pos(fi,render.ring_i); cx=chai.CX+(p[0]-np.pi)*render.SC0; cy=chai.CY+(p[1]-np.pi)*render.SC0
u8=ring(u8,(cx,cy),95,CYAN,1.0,5)
lay=text_layer([dict(text="THIS SWIRL IS",xy=(540,285),size=100,stroke=10),
                dict(segs=[("WORTH ",WHITE),("$1M",YEL)],xy=(540,400),size=124,stroke=12),
                dict(text="SOURCE CODE #13",xy=(540,178),size=32,fill=CYAN,stroke=4)])
lay.alpha_composite(pill("NAVIER–STOKES · DID AI CRACK IT?",(540,1530),CYAN,40))
out=comp(u8,lay)
cv2.imwrite('out/cover.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
cv2.imwrite('out/cover_small.jpg',cv2.resize(cv2.cvtColor(out,cv2.COLOR_RGB2BGR),(360,640),interpolation=cv2.INTER_AREA))
