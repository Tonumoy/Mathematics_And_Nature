# ElevenLabs SFX placed sample-accurately on the picture's events -> sfx.wav (stereo 48k)
import numpy as np, subprocess, wave
import timeline as X
s, e = X.s, X.e; SR = 48000; n = int(X.T * SR)
rd = lambda p: np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', p, '-ac', '1', '-ar', str(SR), '-f', 's16le', '-']), np.int16) / 32767.
def pitch(x, semi):
    r = 2 ** (semi / 12); idx = np.arange(0, len(x) - 1, r); return np.interp(idx, np.arange(len(x)), x)
ping, burst, whoosh, fall = [rd(f'sfx_{k}.mp3') for k in ('ping', 'burst', 'whoosh', 'fall')]
for a in (ping, burst, whoosh, fall): a /= np.abs(a).max() + 1e-9
out = np.zeros(n)
def put(x, t, g):
    i = int(t * SR); x = x[:max(0, n - i)]; out[i:i + len(x)] += x * g
put(ping, 0.02, 0.55)                                   # hook: the drop
put(whoosh, e('L01') + 0.35, 0.45)                      # reveal pull-back
put(pitch(ping, -7), s('L05') + 0.15, 0.4)              # nudge
put(pitch(ping, -5), s('L05') + 0.5, 0.3)               # spring back
a, b = s('L06') + 0.4, e('L06') + 1.2
for i, st in enumerate([0, 2, 3, 5, 7, 8, 10]):         # stack: one rising note per new drop
    put(pitch(ping, st), a + i * (b - a) / 7, 0.5)
f0 = s('L08'); mid = (s('L08') + e('L08')) / 2; g = e('L08') + 0.3
put(fall, f0 + 0.3, 0.5)                                # soft -> falls
put(burst, mid + 0.25 + 0.93, 0.9)                      # loud -> bursts
put(pitch(ping, 12), g + 0.75, 0.45)                    # green: floats again
put(ping, X.T - 0.25, 0.35)                             # loop into frame 1
w = wave.open('sfx.wav', 'w'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
st = np.stack([out, out], 1); w.writeframes((np.clip(st, -1, 1) * 32767).astype(np.int16).tobytes()); w.close(); print('sfx ok')
