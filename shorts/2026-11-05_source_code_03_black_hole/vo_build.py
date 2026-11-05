# VO pipeline SC#3: vo.mp3 -> chunks (silencedetect) -> anchors (source t -> video tau) -> warp.json
import subprocess, re, json, numpy as np, bh
SRC = [0.3, 5.4, 8.6, 12.2, 15.0, 18.1, 21.2, bh.TROPHY+0.6, bh.N2, bh.BREAK, bh.SNAP, bh.PHOT, bh.BACK1]   # source start of each VO line
NL = len(SRC)
out = subprocess.run(['ffmpeg', '-i', 'vo.mp3', '-af', 'silencedetect=n=-38dB:d=0.28', '-f', 'null', '-'], capture_output=True, text=True).stderr
st = [float(x) for x in re.findall(r'silence_start: ([\d.]+)', out)]; en = [float(x) for x in re.findall(r'silence_end: ([\d.]+)', out)]
dur = float(re.findall(r'Duration: (\d+):(\d+):([\d.]+)', out)[0][2]) + 60*float(re.findall(r'Duration: (\d+):(\d+):([\d.]+)', out)[0][1])
STARTS = [0.0, 4.51, 9.17, 13.91, 17.08, 20.89, 23.89, 26.82, 30.13, 33.91, 37.11, 39.53, 44.65]   # mapped by ear-order from silencedetect (lines have internal '...' pauses)
SIL = dict(zip(en, [e-s_ for s_, e in zip(st, en)]))
chunks = [(STARTS[k], (STARTS[k+1] - SIL.get(STARTS[k+1], 0.4)) if k+1 < NL else dur) for k in range(NL)]
gaps = []
print('gaps', len(gaps), 'chunks', [round(b-a, 2) for a, b in chunks])
# place: tau_i >= previous end + 0.25, and visuals may run at most 1/0.8 faster than source
tau = []; prev_end = 0
for i, (a, b) in enumerate(chunks):
    t = SRC[i] if i == 0 else max(prev_end + 0.25, tau[-1] + (SRC[i]-SRC[i-1])*0.8)
    if i == 7:                                                    # reveal hit voice-free: hit >= 0.4 s after line 6, line 7 starts 0.7 s after hit
        t_hit = max(prev_end + 0.4, tau[-1] + (bh.TROPHY-SRC[6])*0.8); t = t_hit + 0.7
    tau.append(t); prev_end = t + (b-a)
T_end = prev_end + 1.6
anch_s = [0.0] + SRC[:7] + [bh.TROPHY] + SRC[7:] + [bh.T]; anch_t = [0.0] + tau[:7] + [tau[7]-0.7] + tau[7:] + [max(T_end, tau[-1] + 2.5)]
# extra anchors so special beats keep their place relative to lines
json.dump(dict(chunks=chunks, tau=tau, anch_s=anch_s, anch_t=anch_t, T=anch_t[-1]), open('warp.json', 'w'), indent=1)
print('T new', round(anch_t[-1], 2), 'tau', [round(x, 2) for x in tau])
