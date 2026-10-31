# Coded Cosmos — dark cinematic score engine v2 (reusable). Serious, deep, smooth.
# v2 (owner 31 Oct 19:45: "more professional and smooth, no abrupt or out-of-tone sounds"):
#  - every pitched sound follows the current chord (chord_at), incl. electron notes, motif, braams
#  - no harsh noise clicks, no Shepard sweeps, no hard cuts; soft attacks, long releases
#  - chords re-phased so the reveal lands on the home chord; bus glue compressor + soft clip
import numpy as np, wave as wavmod
from scipy.signal import butter, lfilter, fftconvolve

SR  = 44100
RNG = np.random.default_rng(11)

# ---------------- helpers ----------------
def hz(n): return 440*2**((n-69)/12)
def lp(x, fc, o=2): b, a = butter(o, min(fc, SR/2-100)/(SR/2)); return lfilter(b, a, x)
def hp(x, fc, o=2): b, a = butter(o, fc/(SR/2), 'high'); return lfilter(b, a, x)
def bp(x, lo, hi, o=2): b, a = butter(o, [lo/(SR/2), hi/(SR/2)], 'band'); return lfilter(b, a, x)
def place(buf, sig, t, g=1.0):
    i = int(t*SR); j = min(buf.shape[0], i+len(sig))
    if 0 <= i < buf.shape[0] and j > i: buf[i:j] += sig[:j-i]*g
def env(n, a, r):
    """attack a s, then a smooth release over the last r s (no clicks at either end)"""
    t = np.arange(n)/SR; up = np.sin(np.pi/2*np.minimum(1, t/max(a, 1e-4)))**2
    down = np.sin(np.pi/2*np.clip((n/SR - t)/max(r, 1e-4), 0, 1))**2
    return up*down
def saw(f, dur, det=(0,)):
    t = np.arange(int(dur*SR))/SR
    return sum(2*((t*f*2**(d/1200)+RNG.random()) % 1)-1 for d in det)/len(det)
def sweep_lp(x, f0, f1, curve=2.0):
    """low-pass whose cutoff glides f0 -> f1 (state carried, so it is click-free)"""
    y = np.zeros_like(x); zi = np.zeros(2); seg = int(0.01*SR); n = len(x)
    for i in range(0, n, seg):
        fc = f0 + (f1-f0)*(i/n)**curve
        b, a = butter(2, min(fc, SR/2-200)/(SR/2)); y[i:i+seg], zi = lfilter(b, a, x[i:i+seg], zi=zi)
    return y
def reverb_ir(sec=3.0, damp=2.6):
    n = int(sec*SR); t = np.arange(n)/SR; irs = []
    for _ in range(2):
        ir = lp(RNG.normal(0, 1, n), 5000)*np.exp(-t*damp)
        ir[:int(0.03*SR)] *= np.linspace(0, 1, int(0.03*SR))
        irs.append(ir/np.sqrt((ir**2).sum()))
    return irs
IR = reverb_ir()
def verb(mono, wet=0.35):
    out = np.zeros((len(mono), 2))
    for c in range(2):
        out[:, c] = mono*(1-wet) + fftconvolve(mono, IR[c])[:len(mono)]*wet*2.0
    return out

# ---------------- instruments (all soft-edged) ----------------
def drone(f, dur):
    t = np.arange(int(dur*SR))/SR; lfo = 0.5+0.5*np.sin(2*np.pi*0.06*t)
    x = 0.7*np.sin(2*np.pi*f*t) + 0.2*np.sin(2*np.pi*2*f*t+0.3) + 0.2*lp(saw(f, dur, (-6, 6)), 200)*(0.6+0.4*lfo)
    return x*env(len(x), 1.5, 1.0)
def strings(notes, dur, cutoff=1400):
    x = sum(saw(hz(m), dur, (-8, -3, 3, 8)) for m in notes)/len(notes)
    t = np.arange(len(x))/SR
    x = lp(lp(x*(1+0.003*np.sin(2*np.pi*5.0*t)), cutoff), cutoff*1.4)
    return x*env(len(x), 1.1, 1.4)
def ostinato_note(f, dur):
    x = lp(lp(saw(f, dur, (-4, 4)), 650), 900)
    return x*env(len(x), 0.008, dur*0.7)*np.exp(-np.arange(len(x))/SR*4)
def cello(f, dur):
    x = lp(saw(f, dur, (-6, 0, 5)), 1500)
    t = np.arange(len(x))/SR
    return x*(1+0.005*np.sin(2*np.pi*5*t))*env(len(x), 0.22, 0.5)
def bell(f, dur=2.6):                                    # soft sine bell, gentle attack
    t = np.arange(int(dur*SR))/SR
    x = np.sin(2*np.pi*f*t) + 0.25*np.sin(2*np.pi*2*f*t)*np.exp(-t*3) + 0.08*np.sin(2*np.pi*3*f*t)*np.exp(-t*6)
    return x*np.exp(-t*1.6)*env(len(t), 0.012, 0.3)
def grain():                                             # soft detector 'rain' grain (no harsh click)
    n = int(0.03*SR)
    return bp(RNG.normal(0, 1, n), 1200, 3500)*env(n, 0.004, 0.024)
def shaker():
    n = int(0.06*SR)
    return bp(RNG.normal(0, 1, n), 3000, 9000)*env(n, 0.006, 0.05)
def kick_deep():
    t = np.arange(int(0.6*SR))/SR; f = 40+60*np.exp(-t*20)
    x = np.tanh(1.2*np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*5.5))
    return lp(x, 2500)*env(len(t), 0.002, 0.2)
def snare_soft():
    t = np.arange(int(0.5*SR))/SR
    x = 0.6*np.sin(2*np.pi*170*t)*np.exp(-t*16) + 0.5*bp(RNG.normal(0, 1, len(t)), 700, 3500)*np.exp(-t*8)
    return x*env(len(t), 0.003, 0.25)
def taiko():
    t = np.arange(int(2.0*SR))/SR; f = 50+30*np.exp(-t*8)
    x = np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*2.2) + 0.15*lp(RNG.normal(0, 1, len(t)), 250)*np.exp(-t*5)
    return x*env(len(t), 0.004, 0.6)
def braam(chord, dur=3.6):                               # low brass swell on the CURRENT chord
    root, fifth = chord[0], chord[1]
    notes = (root-24, root-12, fifth-12, root)
    x = sum(saw(hz(m), dur, (-10, -3, 4, 11)) for m in notes)/len(notes)
    y = sweep_lp(x, 250, 1600, 0.5); y = sweep_lp(y[::-1], 300, 1800, 3)[::-1]     # opens then closes
    return np.tanh(1.2*y)*env(len(y), 0.12, 2.0)
def swell(dur):                                          # smooth riser: filtered noise + rising strings
    n = int(dur*SR); t = np.arange(n)/SR
    x = sweep_lp(RNG.normal(0, 1, n), 300, 5000, 2.5)*(t/dur)**2.2
    return x*env(n, 0.5, 0.15)

# ---------------- harmony ----------------
PROG_MAIN = [(50, 57, 62, 65), (46, 53, 58, 62), (43, 50, 55, 58), (45, 52, 57, 61)]  # Dm Bb Gm A
PROG_DARK = [(50, 57, 62, 65), (43, 50, 55, 58), (46, 53, 58, 62), (45, 52, 57, 61)]  # Dm Gm Bb A
MOTIF_IDX = [(0, 2, 1.0), (1, 3, 1.0), (2, 1, 1.5), (3.5, 2, 2.5)]   # (beat, chord-tone index, length)

def make_harmony(S):
    def chord_at(b):
        idx = ((int(b) - S['DROP']) // 8) % 4          # re-phased: the reveal lands on chord i
        prog = PROG_DARK if S['D0'] <= b < S['D_END'] else PROG_MAIN
        return prog[idx]
    starts = sorted({0} | {S['DROP'] + 8*k for k in range(-10, 20) if 0 < S['DROP'] + 8*k < 400}
                    | {S['D0'], S['D_END']})
    return chord_at, starts
def chord_tone(ch, u):
    """electron note: the chord tone nearest to where the dot landed (u in -1..1), upper octave"""
    tones = sorted({ch[0]+12, ch[1]+12, ch[3]+12, ch[2]+12, ch[0]+24})
    return tones[int(np.clip((u*0.5+0.5)*len(tones), 0, len(tones)-1))]

def compose(events, beats, bpm, sections, out_path):
    """events: [(t, [u...], n_new)], sections: dict of beat markers (see double_slit_v2)."""
    bt = 60/bpm; L = int(beats*bt*SR) + 3*SR; S = sections
    chord_at, starts = make_harmony(S)
    dry, wetbus, drums, tex = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
    place(dry, drone(hz(38), beats*bt+1.5)*0.26, 0)
    # strings: one chord per 2 bars, overlapping releases
    starts = [s for s in starts if s < beats]
    for k, b0 in enumerate(starts):
        b1 = starts[k+1] if k+1 < len(starts) else beats
        dark = S['D0'] <= b0 < S['D_END']
        cut = 700 if b0 < S['DROP'] else (900 if dark else 1600)
        place(wetbus, strings(chord_at(b0), (b1-b0)*bt+1.2, cut), b0*bt, 0.22 if b0 < S['DROP'] else 0.32)
    # electrons: single hits = soft chord-tone bell; crowds = quiet detector rain
    nd = 0
    for (t, us, new) in events:
        b = t/bt
        single_ok = b < S['A_END'] + 0.2
        if not single_ok and S['D0'] <= b < S['D_END'] and nd < 5 and new <= 1:
            single_ok = True; nd += 1
        if new <= 3 and single_ok:
            for u in us: place(wetbus, bell(hz(chord_tone(chord_at(b), u))), t, 0.20)
        else:
            k = int(min(4, 1+np.log2(new)))
            for j in range(k): place(tex, grain(), t + RNG.random()/30, 0.035)
    for bi in range(beats):
        t = bi*bt
        build = S['A_END'] <= bi < S['DROP']
        full = (S['DROP'] <= bi < S['SHUT0']) or (S['OPEN1'] <= bi < S['FADE0'])
        dark = S['D0'] <= bi < S['D_END']
        prog = (bi-S['A_END'])/(S['DROP']-S['A_END']) if build else 1.0
        if build or dark or full:                       # soft clock pulse (shaker)
            sub = 2 if (build and prog < 0.5) or dark else 4
            for s in range(sub): place(drums, shaker(), t+s*bt/sub, 0.035+0.04*prog)
        if build and bi % 2 == 0: place(drums, kick_deep(), t, 0.45+0.35*prog)
        if full or dark:
            if bi % 2 == 0: place(drums, kick_deep(), t, 0.85)
            if bi % 4 == 2: place(drums, snare_soft(), t, 0.30 if not dark else 0.2)
            root = chord_at(bi)[0]
            pat = [0, 0, 12, 0, 0, 12, 7, 0]
            for s in range(2):
                place(dry, ostinato_note(hz(root - 12 + pat[(bi*2+s) % 8]), bt/2), t+s*bt/2, 0.32 if not dark else 0.22)
    # cello motif built from the chord under it
    for b0 in (S['DROP'], S['DROP']+8, S['OPEN1'], S['OPEN1']+8):
        for (o, ti, ln) in MOTIF_IDX:
            ch = chord_at(b0+o)
            place(wetbus, cello(hz(ch[ti]-12), ln*bt+0.3), (b0+o)*bt, 0.30)
    # smooth swell into the reveal (no Shepard, no hard cut)
    sw = (S['DROP']-S['A_END']-2)*bt
    place(wetbus, swell(sw), (S['DROP']*bt) - sw, 0.07)
    # braams + taiko on the big moments, always on the current chord
    for (bb, g) in ((0, 0.16), (S['DROP'], 0.55), (S['SHUT1'], 0.45), (S['SPOT0'], 0.32), (S['OPEN1'], 0.55)):
        place(dry, braam(chord_at(bb)), bb*bt, g); place(drums, taiko(), bb*bt, 0.6 if bb else 0.45)
    # gentle 1-beat dip before the reveal (cosine, not a cut)
    i0, i1 = int((S['DROP']-1)*bt*SR), int(S['DROP']*bt*SR)
    dip = 1-0.6*np.sin(np.linspace(0, np.pi, i1-i0))**2
    for bus in (dry, drums, tex): bus[i0:i1] *= dip
    # sidechain sustained layers to the kick
    duck = np.ones(L); kt = np.arange(int(bt*SR))/SR; shape = 1-0.35*np.exp(-kt*8)
    for bi in range(0, beats, 2):
        if (S['DROP'] <= bi < S['SHUT0']) or (S['D0'] <= bi < S['D_END']) or (S['OPEN1'] <= bi < S['FADE0']):
            i = int(bi*bt*SR); duck[i:i+len(shape)] = np.minimum(duck[i:i+len(shape)], shape[:L-i])
    st = verb(wetbus*duck, 0.42) + verb(dry*duck, 0.15) + verb(tex, 0.3) + np.stack([drums, drums], 1)*0.9
    st = st[:int(beats*bt*SR)]
    # bus glue: slow RMS compressor (ratio 2.5) + soft clip
    mono = st.mean(1); e = np.sqrt(lp(mono**2, 6)+1e-9)
    thr = np.percentile(e, 70); gain = np.where(e > thr, (e/thr)**(1/2.5-1), 1.0)
    st *= lp(gain, 3)[:, None]
    st = np.stack([hp(st[:, c], 30) for c in range(2)], 1)
    st /= np.abs(st).max(); st = np.tanh(st*1.1)/np.tanh(1.1)
    fi, fo = int(0.01*SR), int(0.4*SR)
    st[:fi] *= np.linspace(0, 1, fi)[:, None]; st[-fo:] *= np.cos(np.linspace(0, np.pi/2, fo))[:, None]**2
    st *= 0.92
    with wavmod.open(out_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st*32767).astype(np.int16).tobytes())
