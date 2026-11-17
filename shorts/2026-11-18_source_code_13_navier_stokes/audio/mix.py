import numpy as np, soundfile as sf, subprocess, sys
mus_file=sys.argv[1]; out=sys.argv[2]; dur=float(sys.argv[3]) if len(sys.argv)>3 else None
SR=48000
vo,_=sf.read('vo_timed.wav'); vo=vo.astype(np.float32)
a=subprocess.run(['ffmpeg','-v','quiet','-i',mus_file,'-ac','2','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout
mu=np.frombuffer(a,np.float32).reshape(-1,2).copy()
D=len(vo) if dur is None else int(dur*SR)
if len(mu)<D: mu=np.pad(mu,((0,D-len(mu)),(0,0)))
mu=mu[:D]; vo=np.pad(vo,(0,max(0,D-len(vo))))[:D]
# voice envelope -> duck music ~8 dB under voice
env=np.sqrt(np.convolve(vo**2,np.ones(2400)/2400,'same')); act=(env>0.01).astype(np.float32)
k=int(0.25*SR); act=np.convolve(act,np.ones(k)/k,'same'); act=np.clip(act*1.5,0,1)
g=10**(-8*act/20)
# even out music level (slow RMS normalise towards target)
rm=np.sqrt(np.convolve(mu.mean(1)**2,np.ones(SR*2)/(SR*2),'same'))+1e-4
lev=np.clip(0.06/rm,0.5,2.0)
m=mu*(lev*g)[:,None]*0.9
f=int(0.02*SR); m[:f]*=np.linspace(0,1,f)[:,None]; m[-int(0.4*SR):]*=np.linspace(1,0,int(0.4*SR))[:,None]
mix=m+vo[:,None]*1.0
sf.write('tmp_mix.wav',mix,SR,subtype='FLOAT')
subprocess.run(['ffmpeg','-y','-v','quiet','-i','tmp_mix.wav','-af','highpass=f=30,loudnorm=I=-14:TP=-1.5:LRA=11','-ar','48000','-c:a','pcm_s16le',out])
print('ok')
