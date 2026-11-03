# Voice version (ElevenLabs Daniel pH8TIDBxKcsLhKFzhwgP): time-warp the 44.4 s timeline onto the narration.
# VO lines placed with 2 inserted gaps: +0.9 s before line 5 (x-ray), +2.3 s total before line 9 (silent reveal hit).
import numpy as np, subprocess, sys, wave
import atoms as A
BT = A.BT
A.SLOW_TXT = 'slowed ~260 billion x'           # spiral 8.43-12.64 s = 4.2 s / 1.6e-11 s
O = [0, 2.4, 5.2, 8.6, A.SCAN0, A.SCAN1, 12.2, 15.0, 18.1, 21.2, A.TROPHY, A.N2, A.BREAK, A.SNAP, A.PHOT, A.BACK0, 41.6, A.T]
N = [0, 4.1, 8.05, 14.55, 16.8, 17.7, 17.94, 21.52, 24.04, 27.0, 48*BT, 54*BT, 59*BT, 65*BT, 45.3, 50.38, 53.0, 92*BT]
NT = 92*BT                                                     # 57.5 s
def warp(tau): return float(np.interp(tau, N, O))
def unwarp(t): return float(np.interp(t, O, N))
# spoken line chunks in vo.mp3 (silence-split, -38 dB) and where they go
CUTS = [(0.0, 17.0, 0.0), (17.0, 29.0, 0.9), (29.0, 54.0, 2.3)]   # (src start, src end, shift)

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'render':
        BG, FG = A.build_lamp(); a, b = int(sys.argv[2]), int(sys.argv[3])
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', '30', '-i', '-',
                              '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', f'vpart_{a:05d}.mp4'], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(A.frame(warp(i/30), BG, FG).tobytes())
        p.stdin.close(); p.wait()
    elif mode == 'sheet':
        BG, FG = A.build_lamp()
        for tau in [float(x) for x in sys.argv[2].split(',')]: A.frame(warp(tau), BG, FG).save(f'v_{tau:05.1f}.png')
    elif mode == 'audio':
        import music_cinematic as M, source_code_kit as K
        ev = [(unwarp(t), us, n) for (t, us, n) in A.music_events()]
        S = dict(A_END=38, DROP=48, D0=59, D_END=65, SHUT0=59, SHUT1=59, SPOT0=54, OPEN1=65, FADE0=81)
        M.compose(ev, 92, A.BPM, S, 'score_vo.wav')
        sr = 44100
        with wave.open('score_vo.wav') as w: mus = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)/32767.
        k = K.motif_and_swell(M, NT, 17.7); n = min(len(mus), len(k)); mus = mus[:n] + k[:n]*0.9
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'vo.mp3', '-ac', '1', '-ar', str(sr), '-f', 'f32le', 'vo.raw'])
        vo = np.fromfile('vo.raw', np.float32); voice = np.zeros(n)
        for (s0, s1, sh) in CUTS:
            seg = vo[int(s0*sr):int(s1*sr)].copy(); f = int(0.02*sr); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
            i = int((s0+sh)*sr); voice[i:i+len(seg)] += seg[:max(0, n-i)]
        env = np.sqrt(np.maximum(M.lp(voice**2, 8), 0)); on = (env > 0.01).astype(float)
        on = np.convolve(on, np.ones(int(0.25*sr))/int(0.25*sr), 'same').clip(0, 1); on = M.lp(on, 4)
        duck = 10**(-6.5*on/20)
        mix = mus*duck[:, None]/np.abs(mus).max()*0.55 + (voice/np.abs(voice).max()*0.75)[:, None]
        mix /= np.abs(mix).max()*1.02
        with wave.open('mix_vo.wav', 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes((mix*32767).astype(np.int16).tobytes())
        print('ok', n/sr)
