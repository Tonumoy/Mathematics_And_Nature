# score (dark cinematic, fixed motif) + narration, ducked -6.5 dB under voice; reveal hit is voice-free
import numpy as np, subprocess, wave, music_cinematic as M, source_code_kit as K
import street as S0, street_vo as V
BT = 0.625; NB = 92; sr = 44100; NT = V.T
ev = [(te, [0.5*s_], 1) for te, s_ in S0.EVS if V.SCAN1 < te < V.TROPHY]
S = dict(A_END=35, DROP=45, D0=66, D_END=76, SHUT0=66, SHUT1=66, SPOT0=62, OPEN1=76, FADE0=86)
M.compose(ev, NB, 96, S, 'score_vo.wav')
with wave.open('score_vo.wav') as w: mus = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.
k = K.motif_and_swell(M, NT, V.SCAN0 + 0.9); n = min(len(mus), len(k), int(NT*sr)); mus = mus[:n] + k[:n]*0.9
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'vo.mp3', '-ac', '1', '-ar', str(sr), '-f', 'f32le', 'vo.raw'])
vo = np.fromfile('vo.raw', np.float32); voice = np.zeros(n)
for (s0, s1, sh) in V.CUTS:
    seg = vo[int(s0*sr):int(s1*sr)].copy(); f = int(0.02*sr); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
    i = int((s0+sh)*sr); L = min(len(seg), n-i); voice[i:i+L] += seg[:L]
env = np.sqrt(np.maximum(M.lp(voice**2, 8), 0)); on = (env > 0.01).astype(float)
on = np.convolve(on, np.ones(int(0.25*sr))/int(0.25*sr), 'same').clip(0, 1); on = M.lp(on, 4)
tt = np.arange(n)/sr
g = np.interp(tt, [0, V.TROPHY-0.3, V.TROPHY, V.TROPHY+1.8, V.TROPHY+2.4, NT], [0, 0, 3.0, 3.0, 0, 0])   # reveal hit loudest
mus = mus*(10**(g/20))[:, None]
mix = mus*(10**(-6.5*on/20))[:, None]/np.abs(mus).max()*0.55 + (voice/np.abs(voice).max()*0.75)[:, None]
fl = int(0.6*sr); mix[-fl:] *= np.linspace(1, 0.4, fl)[:, None]
mix /= np.abs(mix).max()*1.02
with wave.open('mix_vo.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((mix*32767).astype(np.int16).tobytes())
m = mix.mean(1); print('ok', n/sr, [round(float(20*np.log10(np.sqrt(np.mean(m[i:i+2*sr]**2))+1e-9)), 1) for i in range(0, n, 2*sr)])
