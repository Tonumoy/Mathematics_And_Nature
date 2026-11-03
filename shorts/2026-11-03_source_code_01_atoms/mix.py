import numpy as np, music_cinematic as M, source_code_kit as K, atoms as A, wave
B = A.BT
S = dict(A_END=29, DROP=39, D0=50, D_END=54, SHUT0=50, SHUT1=50, SPOT0=46, OPEN1=54, FADE0=66)
M.compose(A.music_events(), A.BEATS, A.BPM, S, 'score.wav')
w = wave.open('score.wav'); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.; w.close()
k = K.motif_and_swell(M, A.T, A.SCAN0+0.9)
n = min(len(x), len(k)); y = x[:n] + k[:n]*0.9
y /= np.abs(y).max()*1.05
with wave.open('mix.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100); w.writeframes((y*32767).astype(np.int16).tobytes())
# RMS per 2 s
m = y.mean(1); print([round(20*np.log10(np.sqrt(np.mean(m[i:i+88200]**2))+1e-9), 1) for i in range(0, len(m), 88200)])
