# VO (ElevenLabs Daniel, timed by timeline) + ElevenLabs Music, ducked 8 dB, HP 30 Hz -> mix_<tag>.wav ; then loudnorm in ffmpeg
import numpy as np, wave, subprocess, sys
import timeline as X
from scipy.signal import butter, sosfilt
SR = 48000; T = X.T; mus = sys.argv[1]; tag = sys.argv[2]
rd = lambda p, ch: np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', p, '-f', 's16le', '-ac', str(ch), '-ar', str(SR), '-']), np.int16) / 32767.
n = int(T * SR); vo = np.zeros(n)
for k in X.LINES:
    x = rd(f'vo/{k}.mp3', 1); x = x / np.abs(x).max() * 0.9; i = int(X.TL[k][0] * SR); vo[i:i + len(x)] += x[:n - i]
m = rd(mus, 2).reshape(-1, 2)[:n]
if len(m) < n: m = np.vstack([m, np.zeros((n - len(m), 2))])
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25 * SR)) / int(0.25 * SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15 * SR)) / int(0.15 * SR), 'same')
m = m / np.abs(m).max() * 0.55 * (10 ** (-8.0 * env / 20))[:, None]
fo = int(1.0 * SR); m[-fo:] *= np.linspace(1, 0, fo)[:, None]
import os
fx = rd('sfx.wav', 2).reshape(-1, 2)[:n] * 0.55 if os.path.exists('sfx.wav') else 0
if not np.isscalar(fx) and len(fx) < n: fx = np.vstack([fx, np.zeros((n - len(fx), 2))])
mix = np.clip(sosfilt(butter(2, 30, 'highpass', fs=SR, output='sos'), m + fx + vo[:, None], axis=0), -1, 1)
w = wave.open(f'mix_{tag}.wav', 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
print('ok', tag, T)
