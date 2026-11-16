# full render: 2 workers -> per-chunk mp4 -> concat (+ optional audio mux)
import subprocess, sys, os, numpy as np
from multiprocessing import Process
import timeline as tl
N = int(round(tl.T * 30))
def work(i, a, b):
    import render
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', f'chunk{i}.mp4'], stdin=subprocess.PIPE)
    for n in range(a, b):
        p.stdin.write(np.asarray(render.frame(n)).tobytes())
        if n % 100 == 0: print('r', i, n, flush=True)
    p.stdin.close(); p.wait()
if __name__ == '__main__':
    K = 2; cuts = np.linspace(0, N, K + 1).astype(int)
    ps = [Process(target=work, args=(i, cuts[i], cuts[i + 1])) for i in range(K)]
    [p.start() for p in ps]; [p.join() for p in ps]
    open('chunks.txt', 'w').write(''.join(f"file 'chunk{i}.mp4'\n" for i in range(K)))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'chunks.txt', '-c', 'copy', 'video_noaudio.mp4'], check=True)
    print('done', N)
