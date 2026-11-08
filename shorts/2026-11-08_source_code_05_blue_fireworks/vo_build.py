# tighten pauses (keep 0.17 s), insert 0.85 s hit-pause after "colour." (orig 36.56), atempo 1.1 -> vo.wav ; map() orig->new time
import numpy as np, wave, json, subprocess
sr = 44100; KEEP = 0.17; TEMPO = 1.1; HITPAUSE = (36.6, 0.85)
w = wave.open('vo_raw.wav'); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)/32767.; w.close()
segs = json.load(open('segs.json'))
pieces = []; tab = []; cur = 0.0
for a, b in segs:
    a0 = max(0, a - (0.0 if abs(a-36.62)<0.01 else 0.04)); b0 = min(len(x)/sr, b + (0.0 if abs(b-36.58)<0.01 else 0.06))
    tab.append((a0, b0, cur)); pieces.append(x[int(a0*sr):int(b0*sr)]); cur += b0 - a0
    gap = KEEP + (HITPAUSE[1] if abs(b - HITPAUSE[0]) < 0.05 else 0)
    pieces.append(np.zeros(int(gap*sr))); cur += gap
y = np.concatenate(pieces)
w = wave.open('vo_t2.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(y, -1, 1)*32767).astype(np.int16).tobytes()); w.close()
subprocess.run(f'ffmpeg -y -loglevel error -i vo_t2.wav -af atempo={TEMPO} -ar 44100 -ac 1 vo.wav', shell=True)
json.dump(tab, open('vo_map.json', 'w'))
def mp(t):
    for a0, b0, c in tab:
        if t <= b0: return (c + max(0, t - a0))/TEMPO
    return cur/TEMPO
W = {k: (mp(a), mp(b)) for k, a, b in json.load(open('words.json'))}
json.dump(W, open('words_new.json', 'w'), indent=0)
print('dur', cur/TEMPO); print({k: round(v[0], 2) for k, v in W.items()})
