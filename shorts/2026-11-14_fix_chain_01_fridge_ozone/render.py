# render every scene to its own mp4 (2 workers), then concat
import sys, os, subprocess, math
from multiprocessing import Pool
import timeline as TLM
from timeline import TL, s, e
from PIL import ImageDraw, Image
import scenes as SC
from lib import finish, font, FONT_XB, W, H, FPS, YELLOW

TAG = "FIX CHAIN #1"

def frame_fn(name, a, b):
    T = b - a
    rel = lambda k, f=0.0: s(k) - a + f * (e(k) - s(k))
    if name == 'hook': return lambda t: SC.scene_hook(t, T)
    if name == 'bacteria': return lambda t: SC.scene_bacteria(t, T, rel('L03'), rel('L04'))
    if name == 'machine': return lambda t: SC.scene_machine(t, T, rel('L05', 0.17), toxic_t=rel('L06'))
    if name == 'freon': return lambda t: SC.scene_freon(t, T, rel('L07', 0.6))
    if name == 'rise': return lambda t: SC.scene_rise(t, T, rel('L08', 0.5))
    if name == 'cycle': return lambda t: SC.scene_cycle(t, T)
    if name == 'y1985': return lambda t: SC.scene_1985(t, T)
    if name == 'montreal': return lambda t: SC.scene_montreal(t, T)
    if name == 'heat': return lambda t: SC.scene_heat(t, T, rel('L12', 0.5))
    if name == 'iso': return lambda t: SC.scene_iso(t, T, rel('L13', 0.62))
    if name == 'end': return lambda t: SC.scene_end(t, T, rel('L15'))

def tag(im):
    d = ImageDraw.Draw(im)
    d.text((925, 168), TAG, font=font(FONT_XB, 26), fill=YELLOW, anchor="rm", stroke_width=3, stroke_fill=(0, 0, 0))
    return im

def render_scene(args):
    i, (name, a, b) = args
    n0 = int(round(a * FPS)); n1 = int(round(b * FPS))
    out = f'seg/{i:02d}_{name}.mp4'; tmp = out + '.tmp.mp4'
    fn = frame_fn(name, a, b)
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', tmp], stdin=subprocess.PIPE)
    for n in range(n0, n1):
        t = n / FPS - a
        im = tag(finish(fn(t)))
        p.stdin.write(im.tobytes())
    p.stdin.close(); rc = p.wait()
    if rc == 0: os.replace(tmp, out)
    return out, n1 - n0

if __name__ == '__main__':
    os.makedirs('seg', exist_ok=True)
    only = sys.argv[1:]
    jobs = [(i, sc) for i, sc in enumerate(TLM.SCENES) if not only or sc[0] in only]
    # longest first for balance
    jobs.sort(key=lambda j: -(j[1][2] - j[1][1]))
    with Pool(2) as pool:
        for out, n in pool.imap_unordered(render_scene, jobs):
            print(out, n, flush=True)
    files = sorted(f for f in os.listdir('seg') if f.endswith('.mp4') and '.tmp' not in f)
    open('seg/list.txt', 'w').write(''.join(f"file '{f}'\n" for f in files))
    subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'seg/list.txt', '-c', 'copy', 'video_noaudio.mp4'])
    print('concat ok')
