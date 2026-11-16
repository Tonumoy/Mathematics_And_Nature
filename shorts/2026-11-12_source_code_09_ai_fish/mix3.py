import numpy as np, wave, json, subprocess, os
os.environ['SEED'] = '3'
import score_engine as SE, sc9v3 as X, sc9v2 as V
from scipy.signal import butter, sosfilt
SR = 48000; T = V.T; rng = np.random.default_rng(5)
st = json.load(open('style9.json')); BPM = 92; BT = 60/BPM; BEATS = int(np.ceil(T/BT)) + 1; B = lambda t: int(round(t/BT))
S = dict(A_END=B(V.BRAIN0)-4, DROP=B(V.BRAIN1), D0=B(V.ATT1-1), D_END=B(V.OUT0), SHUT0=B(V.ATT1-1), SHUT1=B(V.OUT0),
         SPOT0=B(V.FINAL), OPEN1=B(V.OUT0), FADE0=B(V.NEXT0), SCAN_T=V.BRAIN0)
ev = [(X.TL[k][0], [float(np.clip(X.TL[k][0]/T*2-1, -1, 1))], 1) for k in X.LINES] + [(V.GEN1, [0.8], 1), (V.FINAL, [1.0], 1), (V.OUT0, [1.0], 1)]
SE.compose(ev, BEATS, BPM, S, 'score3.wav', st)
n = int((T + 0.3)*SR); vo = np.zeros(n)
for k in X.LINES:
    x = np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', f'vo3/{k}.mp3', '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']), np.int16)/32767.
    x = x/np.abs(x).max()*0.9; i = int(X.TL[k][0]*SR); vo[i:i+len(x)] += x[:n-i]
w = wave.open('score3.wav'); sr0 = w.getframerate(); m = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
idx = np.clip((np.arange(n)*sr0/SR).astype(int), 0, len(m)-1); m = m[idx]
env = np.convolve((np.abs(vo) > 0.01).astype(float), np.ones(int(0.25*SR))/int(0.25*SR), 'same').clip(0, 1)
env = np.convolve(env, np.ones(int(0.15*SR))/int(0.15*SR), 'same')
m = m/np.abs(m).max()*0.40*(10**(-7.0*env/20))[:, None]
# quiet simulation sounds: soft thump + bubbles on every catch, sparse ambient bubbles
sfx = np.zeros((n, 2))
def put(sig, t, pan, g):
    i = int(t*SR); k = min(len(sig), n - i)
    if k > 0 and i >= 0: sfx[i:i+k, 0] += sig[:k]*g*np.sqrt(0.5*(1-pan)); sfx[i:i+k, 1] += sig[:k]*g*np.sqrt(0.5*(1+pan))
def bubble():
    f0 = rng.uniform(600, 1500); d = rng.uniform(0.03, 0.07); k = int(d*SR); t = np.arange(k)/SR
    return np.sin(2*np.pi*np.cumsum(f0*(1 + 1.4*t/d))/SR)*np.exp(-t/(d*0.35))*np.minimum(1, t/0.002)
def thump():
    k = int(0.35*SR); t = np.arange(k)/SR; return np.sin(2*np.pi*np.cumsum(55 + 95*np.exp(-t/0.05))/SR)*np.exp(-t/0.09)
for tc in json.load(open('events9.json'))['catch']:
    if tc >= T - 0.8: continue
    p = rng.uniform(-0.4, 0.4); put(thump(), tc, p, 0.30)
    for j in range(6): put(bubble(), tc + 0.02 + j*rng.uniform(0.02, 0.05), p, 0.07)
t = 0.3
while t < T: put(bubble(), t, rng.uniform(-0.8, 0.8), rng.uniform(0.015, 0.04)); t += rng.exponential(0.8)
mix = np.clip(m + sfx + vo[:, None], -1, 1)
w = wave.open('mix3.wav', 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
print('mix ok', T)
