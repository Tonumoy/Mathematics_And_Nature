# render frames [a,b) in K workers at 4K (S2=2): python3 render_part.py a b K tag
import subprocess, sys, os, numpy as np
from multiprocessing import Process
a, b, K, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
def work(i, s0, s1):
    import render
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '2160x3840', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'fast', '-x264-params', 'rc-lookahead=10', '-threads', '2', '-crf', '17', '-pix_fmt', 'yuv420p', f'part_{tag}{i}.mp4'], stdin=subprocess.PIPE)
    for n in range(s0, s1):
        p.stdin.write(np.asarray(render.frame(n, True)).tobytes())
        if n % 50 == 0: print('r', tag, i, n, flush=True)
    p.stdin.close(); p.wait()
if __name__ == '__main__':
    cuts = np.linspace(a, b, K + 1).astype(int)
    ps = [Process(target=work, args=(i, cuts[i], cuts[i + 1])) for i in range(K)]
    [p.start() for p in ps]; [p.join() for p in ps]
    print('done', tag, a, b, flush=True)
