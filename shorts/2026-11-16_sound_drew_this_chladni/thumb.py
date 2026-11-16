# SC#11 cover: rebuilt from the Short's own render (hook frame) + pro edit
import os; os.environ['HOOKCAM'] = 'A'
import numpy as np, cv2, math
from PIL import Image, ImageDraw
import render as R, lib
from lib import CYAN, AMBER, YELLOW, WHITE
n = 30
fr, P, st, f, h, on, top = R.draw_scene(n)
WW, HH = R.WW, R.HH
# sound rings from the centre bolt (motion element, amber = driver)
cx, cy, _ = P(0, 0, 0.0)
lay = np.zeros_like(fr)
for k, rad in enumerate([0.12, 0.22, 0.34]):
    xs, ys, _ = P(rad * np.cos(np.linspace(0, 2 * np.pi, 200)), rad * np.sin(np.linspace(0, 2 * np.pi, 200)), np.full(200, 0.01))
    cv2.polylines(lay, [np.stack([xs, ys], 1).astype(np.int32)], True, tuple(c * (1 - 0.25 * k) for c in AMBER), 5, cv2.LINE_AA)
fr = fr + lay + cv2.GaussianBlur(lay, (0, 0), 18) * 1.5
fr = R.bloom(fr * 1.08)
y = np.mgrid[0:HH, 0:WW][0].astype(np.float32) / R.S2
sc = np.clip(1 - (y - 120) / 520, 0, 1) ** 1.5 * 0.55
fr = fr * (1 - sc[..., None])
img = Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).convert('RGBA')
T = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(T)
S = R.S2
f = lib.font(lib.FONT_XB, 160 * S)
def line(yy, parts):
    tw = sum(f.getlength(p) for p, _ in parts); xx = (WW - tw) / 2
    for p, c in parts:
        d.text((xx, yy * S), p, font=f, fill=c + (255,), anchor='lm', stroke_width=10 * S, stroke_fill=(0, 0, 0, 255)); xx += f.getlength(p)
line(250, [('SOUND', YELLOW)])
line(410, [('DREW THIS', WHITE)])
lib.pill(d, (540, 1520), '3,950 Hz · real plate equation, solved', CYAN, 36, ss=S)
d.text(((930) * S, 120 * S), 'SOURCE CODE #11', font=lib.font(lib.FONT_XB, 36 * S), fill=YELLOW + (255,), anchor='rm', stroke_width=4 * S, stroke_fill=(0, 0, 0, 255))
d.polygon([tuple(p) for p in np.array(top, np.float32)], outline=CYAN + (240,), width=6 * S)
img.alpha_composite(T)
out = img.convert('RGB').resize((1080, 1920), Image.LANCZOS)
out.save('thumbnail.jpg', quality=93)
out.resize((360, 640), Image.LANCZOS).save('thumb_phone.png')
print('ok')
