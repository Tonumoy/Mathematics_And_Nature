import numpy as np, json, wave, subprocess, music_cinematic as M, source_code_kit as K, bh as A
A.RAYS = A.mk_rays(); Wj = json.load(open('warp.json')); T = Wj['T']
tau = lambda s: float(np.interp(s, Wj['anch_s'], Wj['anch_t'])); BT = A.BT; b = lambda s: int(round(tau(s)/BT))
ev = [(tau(t), u, c) for (t, u, c) in A.music_events()]
S = dict(A_END=b(18.1), DROP=b(A.TROPHY), D0=b(A.BREAK), D_END=b(A.SNAP), SHUT0=b(A.BREAK), SHUT1=b(A.BREAK), SPOT0=b(A.N2), OPEN1=b(A.SNAP), FADE0=int(T/BT)-4)
BEATS = int(np.ceil(T/BT)); print(S, BEATS)
M.compose(ev, BEATS, A.BPM, S, 'score_w.wav')
w = wave.open('score_w.wav'); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
k = K.motif_and_swell(M, T, tau(A.SCAN0)+0.9)
n = int(T*44100); x = np.pad(x, ((0, max(0, n-len(x))), (0, 0)))[:n]; k = np.pad(k, ((0, max(0, n-len(k))), (0, 0)))[:n]
mus = x + k*0.9; mus /= np.abs(mus).max()*1.05
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'vo.mp3', '-ar', '44100', '-ac', '1', 'vo.wav'])
w = wave.open('vo.wav'); v = np.frombuffer(w.readframes(w.getnframes()), np.int16)/32767.; w.close()
V = np.zeros(n)
for (a_, e_), t0 in zip(Wj['chunks'], Wj['tau']):
    seg = v[int(a_*44100):int(e_*44100)]; i0 = int(t0*44100); m = min(len(seg), n-i0); V[i0:i0+m] += seg[:m]
env = np.convolve(np.abs(V) > 0.02, np.ones(int(0.25*44100))/(0.25*44100), 'same'); env = np.clip(env*4, 0, 1)
env = np.convolve(env, np.ones(int(0.15*44100))/(0.15*44100), 'same')
duck = 10**(-6.5*env/20)
vo_g = 10**(4/20)                                                   # voice sits above the score
y = mus*duck[:, None]*0.75 + (V*vo_g)[:, None]
y /= np.abs(y).max()*1.05
with wave.open('mix_vo.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes((y*32767).astype(np.int16).tobytes())
hit = tau(A.TROPHY); print('hit', round(hit, 2), 'voice near hit', float(np.abs(V[int((hit-0.3)*44100):int((hit+0.6)*44100)]).max()))
m = y.mean(1); print([round(20*np.log10(np.sqrt(np.mean(m[i:i+88200]**2))+1e-9), 1) for i in range(0, len(m), 88200)])
