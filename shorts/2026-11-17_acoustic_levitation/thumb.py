# SC#12 cover: rebuilt from the Short's own render (hook camera + full 7-drop stack) + pro edit
import numpy as np, math
from PIL import Image, ImageDraw
import render as R, lib, timeline as X
from lib import CYAN, YELLOW, WHITE
c0 = R.cam(0.0); orig = R.cam
R.cam = lambda t: (c0[0] * 1.25, c0[1], 1.5, c0[3])
t = X.s('L07') + 0.2
fr, pos, M, dl = R.scene(t)
S = R.S2; HH, WW = R.HH, R.WW
y = np.mgrid[0:HH, 0:WW][0].astype(np.float32) / S
sc = np.clip(1 - (y - 140) / 560, 0, 1) ** 1.5 * 0.6
fr = fr * (1 - sc[..., None])
img = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).convert('RGBA')
T = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(T)
f = lib.font(lib.FONT_XB, 150 * S)
def line(yy, parts):
    tw = sum(f.getlength(p) for p, _ in parts); xx = (WW - tw) / 2
    for p, c in parts:
        d.text((xx, yy * S), p, font=f, fill=c + (255,), anchor='lm', stroke_width=10 * S, stroke_fill=(0, 0, 0, 255)); xx += f.getlength(p)
line(270, [('SOUND', YELLOW)]); line(420, [('LIFTS WATER', WHITE)])
cx, cy, cz = R.project(dl[0][0], pos, M); r = R.F * dl[0][1] / cz * 1.6 / S
lib.ring(d, (cx / S, cy / S), r, CYAN, 255, 6, ss=S)
lib.pill(d, (540, 1560), '72 speakers · 40,000 Hz · real physics', CYAN, 34, ss=S)
d.text((930 * S, 112 * S), 'SOURCE CODE #12', font=lib.font(lib.FONT_XB, 36 * S), fill=YELLOW + (255,), anchor='rm', stroke_width=4 * S, stroke_fill=(0, 0, 0, 255))
img.alpha_composite(T)
out = img.convert('RGB').resize((1080, 1920), Image.LANCZOS)
out.save('thumbnail.jpg', quality=93); out.resize((360, 640), Image.LANCZOS).save('thumb_phone.png'); print('ok')
