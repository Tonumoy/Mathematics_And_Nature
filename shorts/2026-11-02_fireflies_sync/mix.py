# place ElevenLabs narration segments, duck music -6 dB under voice, write out/mix.wav
import numpy as np, wave, subprocess, re, sys
from fireflies_v2 import VO_PLACE
SR = 44100
def readwav(p):
    w = wave.open(p); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)/32768
    return a.reshape(-1, w.getnchannels()), w.getframerate()
vo, sr = readwav("vo.wav"); vo = vo[:, 0]; assert sr == SR
log = subprocess.run(["ffmpeg", "-i", "vo.wav", "-af", "silencedetect=n=-38dB:d=0.3", "-f", "null", "-"], capture_output=True, text=True).stderr
st = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", log)]; en = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", log)]
segs = list(zip([0.0] + en, st + [len(vo)/SR])); assert len(segs) == len(VO_PLACE), (len(segs), len(VO_PLACE))
mus, sr2 = readwav(sys.argv[1] if len(sys.argv) > 1 else "out/music_raw.wav"); assert sr2 == SR
L = len(mus); track = np.zeros(L, np.float32); act = np.zeros(L, np.float32)
for (a, b), t in zip(segs, VO_PLACE):
    seg = vo[int(max(0, a-0.04)*SR):int((b+0.08)*SR)].copy()
    fi = int(0.01*SR); seg[:fi] *= np.linspace(0, 1, fi); seg[-fi*4:] *= np.linspace(1, 0, fi*4)
    i = int(t*SR); n = min(len(seg), L-i); track[i:i+n] += seg[:n]; act[i:i+n] = 1
from scipy.signal import lfilter
k = np.ones(int(0.25*SR))/int(0.25*SR); env = np.clip(np.convolve(act, k, 'same')*1.6, 0, 1)
duck = 10**(-6/20*env)
vo_rms = np.sqrt((track[act > 0]**2).mean()); mu_rms = np.sqrt((mus**2).mean())
g_vo = 0.25/vo_rms*mu_rms/0.25*2.2                             # voice ~7 dB over the music bed
out = mus*duck[:, None] + (track*g_vo)[:, None]
out /= np.abs(out).max()*1.05
with wave.open("out/mix.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes())
print("segments", len(segs), "vo/music gain", round(float(g_vo), 2))
