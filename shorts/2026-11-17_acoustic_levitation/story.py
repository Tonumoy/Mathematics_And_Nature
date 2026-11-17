# SC#12 Instagram story: clean render (no HUD) of the stack -> loudness twist -> burst, NEW REEL pill, reel's music + sfx
import sys, subprocess, numpy as np
from PIL import Image, ImageDraw
import render as R, lib
from lib import YELLOW, WHITE, CYAN
T0, DUR = 17.2, 12.0
S = R.S2
def frame(n):
    t = T0 + n / 30
    fr, pos, M, dl = R.scene(t)
    img = Image.fromarray(fr.astype(np.uint8)).convert('RGBA')
    L = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    a = int(255 * min(1, n / 12))
    lib.pill(d, (540, 300), 'NEW REEL  ▶', (255, 80, 80), 34, a, ss=S)
    f = lib.font(lib.FONT_XB, 78 * S)
    d.text((540 * S, 410 * S), 'Can sound', font=f, fill=WHITE + (a,), anchor='mm', stroke_width=7 * S, stroke_fill=(0, 0, 0, a))
    d.text((540 * S, 500 * S), 'lift water?', font=f, fill=YELLOW + (a,), anchor='mm', stroke_width=7 * S, stroke_fill=(0, 0, 0, a))
    b = int(255 * min(1, max(0, (n - 60) / 15)))
    if b: lib.caption(d, '72 speakers · watch it burst', y=1500, size=44, hl=('burst',), alpha=b, ss=S)
    img.alpha_composite(L)
    return img.convert('RGB').resize((1080, 1920), Image.LANCZOS)
if __name__ == '__main__':
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1080x1920', '-r', '30', '-i', '-',
                          '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', 'story_v.mp4'], stdin=subprocess.PIPE)
    for n in range(int(DUR * 30)):
        p.stdin.write(np.asarray(frame(n)).tobytes())
        if n % 60 == 0: print('s', n, flush=True)
    p.stdin.close(); p.wait()
    # audio: music C + sfx, same time window, no voice
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', str(T0), '-t', str(DUR), '-i', 'musC.mp3', '-ss', str(T0), '-t', str(DUR), '-i', 'sfx.wav',
                    '-filter_complex', '[0:a]volume=0.8[m];[1:a]volume=0.6[s];[m][s]amix=inputs=2:normalize=0,afade=t=in:d=0.3,afade=t=out:st=%.1f:d=1.0,loudnorm=I=-14:TP=-1.5' % (DUR - 1),
                    '-ar', '48000', 'story_a.m4a'], check=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', 'story_v.mp4', '-i', 'story_a.m4a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', 'SC12_story.mp4'], check=True)
    print('story done', flush=True)
