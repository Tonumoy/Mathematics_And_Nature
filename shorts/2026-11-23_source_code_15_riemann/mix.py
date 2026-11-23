import subprocess, numpy as np, sys
SR=48000; DUR=44.0
MUSIC = sys.argv[1] if len(sys.argv)>1 else 'music_b'
OUT = sys.argv[2] if len(sys.argv)>2 else 'mix.wav'
def load(f):
    raw=subprocess.run(['ffmpeg','-v','error','-i',f,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).reshape(-1,2).astype(np.float64)
N=int(SR*DUR); out=np.zeros((N,2))
vo=load('vo_a.mp3'); mu=load(MUSIC+'.mp3')
def place(buf,x,t,g):
    i=int(t*SR); j=min(N,i+len(x));
    if i<0: x=x[-i:]; i=0; j=min(N,len(x))
    buf[i:j]+=x[:j-i]*g
music=np.zeros((N,2)); place(music,mu,0.0,1.0)
# fade out music tail 42.5-44
fo=np.ones(N); a=int(42.6*SR); fo[a:]=np.linspace(1,0,N-a); music*=fo[:,None]
voice=np.zeros((N,2)); place(voice,vo,0.0,1.0)
# duck: envelope of voice
env=np.abs(voice).max(1); k=int(0.25*SR)
env=np.convolve(env,np.ones(k)/k,'same'); env=np.clip(env/ (env.max()*0.15),0,1)
duck=10**(-8*env/20)
sfx=np.zeros((N,2))
place(sfx,load('sfx_impact.mp3'),5.28-0.662,0.9)
p=load('sfx_ping.mp3')
for i,t in enumerate([7.76,8.52,9.16,9.68,10.16]): place(sfx,p,t,0.35+0.05*i)
place(sfx,p,22.05,0.5); place(sfx,p,38.6,0.4)
place(sfx,load('sfx_riser.mp3'),19.35,0.45)
place(sfx,load('sfx_glitch.mp3'),28.32-0.233,0.55)
place(sfx,load('sfx_ticks.mp3'),35.5,0.35)
place(sfx,load('sfx_stamp.mp3'),39.3-0.226,0.7)
mixd=voice*1.0+music*0.55*duck[:,None]+sfx
mixd/=np.abs(mixd).max()*1.05
raw=(mixd.astype(np.float32)).tobytes()
subprocess.run(['ffmpeg','-y','-v','error','-f','f32le','-ar',str(SR),'-ac','2','-i','-','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000',OUT],input=raw)
