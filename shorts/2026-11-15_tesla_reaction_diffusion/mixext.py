import numpy as np, wave, json, subprocess, os
import music_lofi as M, tl as X
from scipy.signal import butter, sosfilt
SR = 48000; T = X.T; s, e = X.s, X.e
marks = dict(ARP=s('L01'), GROOVE=s('L03'), LIFT=s('L08'), QUIET0=s('L13'), QUIET1=e('L14') + 0.3, REVEAL=e('L14') + 0.4, END=T)
import sys
src = sys.argv[1]; out = sys.argv[2]
subprocess.check_call(['ffmpeg','-y','-loglevel','error','-i',src,'-ar','48000','-ac','2','score_ext.wav'])
n = int(T * SR); vo = np.zeros(n)
for k in X.CAP:
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', f'vo/{k}.mp3', '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']), np.int16) / 32767.
    x = x / np.abs(x).max() * 0.9; i = int(X.TL[k][0] * SR); vo[i:i + len(x)] += x[:n - i]
w = wave.open('score_ext.wav'); m = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2) / 32767.; w.close()
m = np.pad(m, ((0, max(0, n - len(m))), (0, 0)))[:n]
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25 * SR)) / int(0.25 * SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15 * SR)) / int(0.15 * SR), 'same')
m = m / (np.abs(m).max() + 1e-9) * 0.45 * (10 ** (-8.0 * env / 20))[:, None]
fo = int(1.5 * SR); m[-fo:] *= np.linspace(1, 0, fo)[:, None]
mix = m + vo[:, None]
mix = sosfilt(butter(2, 30, 'highpass', fs=SR, output='sos'), mix, axis=0); mix = np.clip(mix, -1, 1)
w = wave.open(out, 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
print('mix ok', T)
