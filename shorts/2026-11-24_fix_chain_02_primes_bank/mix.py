import subprocess, numpy as np, sys
sys_argv = sys.argv[:]; sys.argv = ['render.py', '4', 'info']
import render as R
MUSIC, OUT = sys_argv[1], sys_argv[2]
SR = 48000; DUR = R.DUR; N = int(SR * DUR)
def load(f):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
def place(buf, x, t, g):
    i = int(t * SR)
    if i < 0: x = x[-i:]; i = 0
    j = min(N, i + len(x)); buf[i:j] += x[:j - i] * g
voice = np.zeros((N, 2))
for k in range(10): place(voice, load(f'vo/T{k+1:02d}.wav'), R.L[k], 1.0)
mu = load(MUSIC + '.mp3')
music = np.zeros((N, 2))
cut = int(56.5 * SR); xf = int(1.5 * SR)
a = mu[:cut].copy(); a[-xf:] *= np.linspace(1, 0, xf)[:, None]
place(music, a, 0.0, 1.0)
b = mu[:int((DUR - 55.0 + 1) * SR)].copy(); b[:xf] *= np.linspace(0, 1, xf)[:, None]
place(music, b, 55.0, 1.0)
fo = np.ones(N); s0 = int((DUR - 2.0) * SR); fo[s0:] = np.linspace(1, 0.0, N - s0); music *= fo[:, None]
env = np.abs(voice).max(1); k = int(0.25 * SR)
env = np.convolve(env, np.ones(k) / k, 'same'); env = np.clip(env / (env.max() * 0.15), 0, 1)
duck = 10 ** (-9 * env / 20)
sfx = np.zeros((N, 2)); S = {n: load(f'sfx/{n}.mp3') for n in ('crack', 'scroll', 'stamp', 'counter', 'hum', 'click', 'whoosh', 'boom')}
for i in range(2, 11): place(sfx, S['whoosh'], R.Ls(i) - 0.3, 0.25)
place(sfx, S['scroll'], R.Ls(2), 0.6)
place(sfx, S['stamp'], R.Ls(3) + 3.6, 0.9)
place(sfx, S['click'], R.Ls(5) + 2.0, 0.9)
place(sfx, S['counter'], R.Ls(6) + 3.2, 0.45)
place(sfx, S['counter'], R.Ls(7) + 0.2, 0.35)
place(sfx, S['hum'], R.Ls(8), 0.6)
place(sfx, S['crack'], R.Ls(8) + 3.0, 0.9)
place(sfx, S['boom'], R.Ls(8) + 7.6, 0.6)
place(sfx, S['boom'], R.Ls(9) + 0.0, 0.35)
m = voice * 1.0 + music * 0.6 * duck[:, None] + sfx
m /= np.abs(m).max() * 1.05
tmp = OUT + '.tmp.wav'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-', '-af', 'highpass=f=30', tmp], input=m.astype(np.float32).tobytes())
import json
J = subprocess.run(['ffmpeg', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
d = json.loads(J[J.rindex('{'):J.rindex('}') + 1])
af = f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true"
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-af', af, '-ar', '48000', OUT])
print('ok', DUR)
