import numpy as np, wave, json, subprocess, score_engine as SE, sc7b as V
SR = 48000; BT = 0.6; BPM = 100; BEATS = int(np.ceil(V.T/BT)) + 1
B = lambda t: int(round(t/BT))
st = SE.pick_style(seed='2026-11-10 SC7 v2 black hole body', log_path='MUSIC_LOG.md', topic_number=0.7)
S = dict(A_END=B(V.SCAN0)-4, DROP=B(V.SCAN1), D0=B(V.TL['L08'][0]), D_END=B(V.D_GREEN), SHUT0=B(V.TL['L08'][0]), SHUT1=B(V.TL['L08'][0]),
         SPOT0=B(V.EQ0), OPEN1=B(V.D_GREEN), FADE0=B(V.BACK2a), SCAN_T=V.SCAN0)
ev = [(a, [float(np.clip(a/V.T*2-1, -1, 1))], 1) for a, b, _ in V.CAPS] + [(V.BREAK_T, [0.9], 1), (V.D_GREEN, [1.0], 1)]
SE.compose(ev, BEATS, BPM, S, 'score7b.wav', st); json.dump(st, open('style7b.json', 'w')); print(json.dumps(st))
# voice track
n = int((V.T + 0.5)*SR); vo = np.zeros(n)
for k, (a, b) in V.TL.items():
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', f'vo/{k}.wav', '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']), np.int16)/32767.
    i = int(a*SR); vo[i:i+len(x)] += x[:n-i]
w = wave.open('score7b.wav'); sr0 = w.getframerate(); m = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
idx = np.clip((np.arange(n)*sr0/SR).astype(int), 0, len(m)-1); m = m[idx]
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25*SR))/int(0.25*SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15*SR))/int(0.15*SR), 'same')
duck = 10**(-6.5*env/20)
m = m/np.abs(m).max()*0.55*duck[:, None]
v = vo/np.abs(vo).max()*0.95
y = m + v[:, None]; y /= np.abs(y).max()*1.02
with wave.open('mix7b.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((y*32767).astype(np.int16).tobytes())
mm = m.mean(1); print('music RMS/2s', [round(20*np.log10(np.sqrt(np.mean(mm[i:i+2*SR]**2))+1e-9), 1) for i in range(0, len(mm), 2*SR)])
