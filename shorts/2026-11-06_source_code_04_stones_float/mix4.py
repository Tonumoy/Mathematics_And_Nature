# SC#4 music (score_engine v3: new score) + Daniel VO (ducked) -> mix.wav
import numpy as np, wave, json, sys
import score_engine as SE, sc4 as A
BPM = 84; BT = 60/BPM; T = A.T; BEATS = int(np.ceil(T/BT)); b = lambda s: int(round(s/BT))
style = SE.pick_style(seed='2026-11-07 stones_float', log_path='MUSIC_LOG.md', topic_number=0.59)
print('STYLE', json.dumps(style))
S = dict(A_END=b(A.SCAN0), DROP=b(A.HIT), D0=b(A.LESS), D_END=b(A.MORE), SHUT0=b(A.LESS), SHUT1=b(A.LESS), SPOT0=b(A.EQ_END),
         OPEN1=b(A.MORE), FADE0=BEATS-5, SCAN_T=A.SCAN0+0.45)
SE.compose(A.music_events(), BEATS, BPM, S, 'score.wav', style)
json.dump(dict(style=style, bpm=BPM, S=S), open('style.json', 'w'))
w = wave.open('score.wav'); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
n = int(T*44100); x = np.pad(x, ((0, max(0, n-len(x))), (0, 0)))[:n]
w = wave.open('vo2.wav'); v = np.frombuffer(w.readframes(w.getnframes()), np.int16)/32767.; w.close()
V = np.zeros(n); cut = int(31.9*44100); V[:cut] = v[:cut]; sh = int(32.4*44100); m = min(len(v)-cut, n-sh); V[sh:sh+m] = v[cut:cut+m]
env = np.convolve(np.abs(V) > 0.02, np.ones(int(0.25*44100))/(0.25*44100), 'same'); env = np.clip(env*4, 0, 1)
env = np.convolve(env, np.ones(int(0.15*44100))/(0.15*44100), 'same')
duck = 10**(-6.5*env/20)
fade = np.ones(n); fo = int(0.35*44100); fade[-fo:] = np.linspace(1, 0, fo); fade[:int(0.02*44100)] = np.linspace(0, 1, int(0.02*44100))
mus = x/np.abs(x).max(); mus[int(50.3*44100):] *= np.linspace(1, 0.7, n-int(50.3*44100))[:, None]
y = (mus*duck[:, None]*0.75 + (V*10**(4/20))[:, None])*fade[:, None]
y /= np.abs(y).max()*1.05
with wave.open('mix.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes((y*32767).astype(np.int16).tobytes())
hit = A.HIT; print('voice near hit', round(float(np.abs(V[int((hit-0.3)*44100):int((hit+0.4)*44100)]).max()), 3))
mm = y.mean(1); print('RMS/2s', [round(20*np.log10(np.sqrt(np.mean(mm[i:i+88200]**2))+1e-9), 1) for i in range(0, len(mm), 88200)])
mus_m = (mus*duck[:, None]).mean(1); print('music RMS/2s', [round(20*np.log10(np.sqrt(np.mean(mus_m[i:i+88200]**2))+1e-9), 1) for i in range(0, len(mus_m), 88200)])
