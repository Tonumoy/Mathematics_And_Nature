import numpy as np, wave, json, subprocess, os
import score_engine as SE, timeline as X
from scipy.signal import butter, sosfilt
SR = 48000; T = X.T
st = SE.pick_style(seed='2026-11-14 fix chain fridge', log_path='MUSIC_LOG.md', topic_number=1.0e5)
json.dump(st, open('style.json', 'w')); print(st)
BPM = 92; BT = 60/BPM; BEATS = int(np.ceil(T/BT)) + 1; B = lambda t: int(round(t/BT))
s = X.s; e = X.e
S = dict(A_END=B(s('L05')) - 4, DROP=B(s('L09')), D0=B(s('L06')), D_END=B(s('L07')), SHUT0=B(s('L06')), SHUT1=B(s('L07')),
         SPOT0=B(s('L09')), OPEN1=B(s('L11')), FADE0=B(s('L15')), SCAN_T=s('L05') + 0.17*(e('L05')-s('L05')))
ev = [(X.TL[k][0], [float(np.clip(X.TL[k][0]/T*2-1, -1, 1))], 1) for k in X.LINES]
SE.compose(ev, BEATS, BPM, S, 'score.wav', st)
n = int((T + 0.3)*SR); vo = np.zeros(n)
for k in X.LINES:
    p = f'vo/{k}.mp3'
    if not os.path.exists(p): continue
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', p, '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']), np.int16)/32767.
    x = x/np.abs(x).max()*0.9; i = int(X.TL[k][0]*SR); vo[i:i+len(x)] += x[:n-i]
w = wave.open('score.wav'); sr0 = w.getframerate(); m = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
idx = np.clip((np.arange(n)*sr0/SR).astype(int), 0, len(m)-1); m = m[idx]
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25*SR))/int(0.25*SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15*SR))/int(0.15*SR), 'same')
m = m/np.abs(m).max()*0.42*(10**(-6.0*env/20))[:, None]
# end fade
fo = int(1.2*SR); m[-fo:] *= np.linspace(1, 0, fo)[:, None]
mix = m + vo[:, None]
mix = sosfilt(butter(2, 30, 'highpass', fs=SR, output='sos'), mix, axis=0)
mix = np.clip(mix, -1, 1)
w = wave.open('mix.wav', 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
print('mix ok', T, 'vo lines:', sum(os.path.exists(f'vo/{k}.mp3') for k in X.LINES))
