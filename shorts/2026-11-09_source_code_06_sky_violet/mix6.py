# SC#6 music (score_engine v3: NEW score) + Daniel VO (ducked ~6 dB, no voice on the reveal hit) -> mix.wav
import numpy as np, wave, json, subprocess
import score_engine as SE
TL = json.load(open('timeline.json')); T = TL['T']; G = lambda k: TL[k]
BPM = 80; BT = 60/BPM; BEATS = int(np.ceil(T/BT)); b = lambda s: int(round(s/BT))
style = SE.pick_style(seed='2026-11-09 sky_violet', log_path='MUSIC_LOG.md', topic_number=4.0)
print('STYLE', json.dumps(style))
S = dict(A_END=b(G('SCAN0')), DROP=b(G('HIT')), D0=b(G('DIAL0')), D_END=b(G('DIAL4')), SHUT0=b(G('DIAL2')), SHUT1=b(G('DIAL3')),
         SPOT0=b(G('DIAL4')), OPEN1=b(G('BACK0')), FADE0=BEATS-5, SCAN_T=G('SCAN0')+0.45)
ev = [(t, [0.3*np.sin(i)], 1) for i, t in enumerate(TL.get('events', []))]
SE.compose(ev, BEATS, BPM, S, 'score.wav', style)
json.dump(dict(style=style, bpm=BPM, S=S), open('style.json', 'w'))
def rd(p):
    w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16); ch = w.getnchannels(); w.close()
    return x.reshape(-1, ch)/32767.
x = rd('score.wav'); n = int(T*44100); x = np.pad(x, ((0, max(0, n-len(x))), (0, 0)))[:n]
v = rd('vo.wav')[:, 0]; V = np.zeros(n); V[:min(n, len(v))] = v[:n]
env = np.convolve(np.abs(V) > 0.02, np.ones(int(0.25*44100))/(0.25*44100), 'same'); env = np.clip(env*4, 0, 1)
env = np.convolve(env, np.ones(int(0.15*44100))/(0.15*44100), 'same'); duck = 10**(-6.5*env/20)
fade = np.ones(n); fo = int(0.25*44100); fade[-fo:] = np.linspace(1, 0.6, fo); fade[:int(0.02*44100)] = np.linspace(0, 1, int(0.02*44100))
mus = x/np.abs(x).max()
y = (mus*duck[:, None]*0.75 + (V*10**(4/20))[:, None])*fade[:, None]
y /= np.abs(y).max()*1.05
with wave.open('mix.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes((y*32767).astype(np.int16).tobytes())
h = G('HIT'); print('voice near hit', round(float(np.abs(V[int((h-0.3)*44100):int((h+0.5)*44100)]).max()), 3))
mm = (mus*duck[:, None]).mean(1); print('music RMS/2s', [round(20*np.log10(np.sqrt(np.mean(mm[i:i+88200]**2))+1e-9), 1) for i in range(0, len(mm), 88200)])
c = SE.chroma_print(mus); print('chroma ok', [round(float(v),2) for v in c])
