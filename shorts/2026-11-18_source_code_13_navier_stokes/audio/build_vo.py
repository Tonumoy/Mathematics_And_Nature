import subprocess, numpy as np, json, soundfile as sf
a=subprocess.run(['ffmpeg','-v','quiet','-i','vo.mp3','-ac','1','-ar','48000','-f','f32le','-'],capture_output=True).stdout
x=np.frombuffer(a,np.float32).copy(); SR=48000
# speech segments (start,end,text) from silencedetect
segs=[(0.02,3.35,"This swirl in your chai hides a million-dollar question."),
(3.94,6.19,"Every drop in it follows one rule."),(6.67,7.77,"Navier-Stokes."),
(8.25,11.74,"It's Newton's push-and-move law, written for liquids and air."),
(12.22,13.15,"Pressure pushes."),(13.43,14.86,"Stickiness slows."),(15.23,16.69,"Motion carries it along."),
(17.18,18.76,"We already use it every day."),(19.20,19.76,"Planes."),(20.06,20.44,"Weather."),(20.82,21.73,"Blood in your heart."),
(22.21,23.77,"So what's worth a million?"),(24.26,26.65,"Stretch a swirl, and it spins faster."),(27.02,28.78,"Like a skater pulling her arms in."),
(29.21,32.11,"Could it spin infinitely fast, and break the maths?"),(32.54,33.59,"Nobody could prove it."),
(34.05,39.31,"In September, OpenAI said its AI proved it can, with an outside push."),(39.76,41.10,"Experts are still checking."),
(41.49,42.48,"Your chai is safe."),(42.89,43.44,"The maths?"),(43.81,44.56,"We'll see.")]
# gap after each segment (s): default = original pause
extra={0:0.15,2:0.1,6:0.2,7:0.25,8:0.7,9:0.85,10:0.55,11:0.2,13:0.15,15:0.7,17:0.45,20:1.4}
out=[];t=0.0;tl=[]
pad=0.06
for i,(s,e,txt) in enumerate(segs):
    a0=max(0,int((s-pad)*SR)); a1=int((e+pad)*SR)
    clip=x[a0:a1].copy(); n=len(clip); f=int(0.01*SR)
    clip[:f]*=np.linspace(0,1,f); clip[-f:]*=np.linspace(1,0,f)
    tl.append({"i":i,"start":round(t+pad,3),"end":round(t+len(clip)/SR-pad,3),"text":txt})
    out.append(clip); t+=len(clip)/SR
    if i+1<len(segs): g=(segs[i+1][0]-e)-2*pad+extra.get(i,0.0)
    else: g=extra.get(i,0)
    g=max(g,0.05); out.append(np.zeros(int(g*SR),np.float32)); t+=g
y=np.concatenate(out); sf.write('vo_timed.wav',y,SR,subtype='FLOAT')
json.dump({"dur":round(len(y)/SR,3),"lines":tl},open('timeline.json','w'),indent=0)
print(len(y)/SR); [print(l['i'],l['start'],l['end'],l['text'][:40]) for l in tl]
