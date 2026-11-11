import numpy as np, wave, json, subprocess, score_engine as SE, sc8b as V
SR = 48000; BPM = 96; BT = 60/BPM; BEATS = int(np.ceil(V.T/BT)) + 1
B = lambda t: int(round(t/BT))
st = json.load(open('style8.json'))                     # score style already picked + logged for SC#8
S = dict(A_END=B(V.SCAN0)-4, DROP=B(V.SCAN1), D0=B(V.DIAL0), D_END=B(V.SNAP), SHUT0=B(V.DIAL0), SHUT1=B(V.SNAP),
         SPOT0=B(V.EQ0), OPEN1=B(V.SNAP), FADE0=B(V.ZOOM0+1.0), SCAN_T=V.SCAN0)
ev = [(V.TL[k][0], [float(np.clip(V.TL[k][0]/V.T*2-1, -1, 1))], 1) for k in V.LINES] + [(V.CRASH, [0.9], 1), (V.PAY, [1.0], 1), (V.SNAP, [1.0], 1)]
SE.compose(ev, BEATS, BPM, S, 'score8b.wav', st)
n = int((V.T + 0.3)*SR); vo = np.zeros(n)
for k in V.LINES:
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', f'vo/{k}.mp3', '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']), np.int16)/32767.
    i = int(V.TL[k][0]*SR); vo[i:i+len(x)] += x[:n-i]
w = wave.open('score8b.wav'); sr0 = w.getframerate(); m = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
idx = np.clip((np.arange(n)*sr0/SR).astype(int), 0, len(m)-1); m = m[idx]
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25*SR))/int(0.25*SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15*SR))/int(0.15*SR), 'same')
duck = 10**(-6.0*env/20)
m = m/np.abs(m).max()*0.42*duck[:, None]
v = vo/np.abs(vo).max()*0.95
mix = np.clip(m + v[:, None], -1, 1)
w = wave.open('mix8b.wav', 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
print('mix ok', V.T)
