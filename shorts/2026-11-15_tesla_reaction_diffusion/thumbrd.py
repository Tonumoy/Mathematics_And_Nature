import numpy as np
from PIL import Image, ImageDraw
import rd, lib as K
from lib import W, H, CYAN, YELLOW, WHITE
img = rd.portrait_img(rd.NSN, sweep=900)
img = rd.bloom(img, 0.3, 0.25)
y = np.mgrid[0:H, 0:W][0].astype(np.float32)
sc = np.clip((y - 1150) / 500, 0, 1) ** 1.2 * 0.82
img = img * (1 - sc[..., None])
L = K.new_layer(); d = ImageDraw.Draw(L)
K.ring(d, (520, 560), 250, CYAN, 255, 7)
f = K.font(K.FONT_XB, 124 * 2)
def line(yy, parts):
    tw = sum(f.getlength(p) for p, _ in parts); xx = (W * 2 - tw) / 2
    for p, col in parts:
        d.text((xx, yy * 2), p, font=f, fill=col + (255,), anchor="lm", stroke_width=20, stroke_fill=(0, 0, 0, 255)); xx += f.getlength(p)
line(1360, [("NOBODY ", WHITE), ("DREW", YELLOW)])
line(1520, [("TESLA", WHITE)])
K.pill(d, (540, 1650), "grown by 2 chemicals · 2 rules", CYAN, 44)
d.text(((W - 40) * 2, 90 * 2), "SOURCE CODE #10", font=K.font(K.FONT_XB, 34 * 2), fill=YELLOW + (255,), anchor="rm", stroke_width=8, stroke_fill=(0, 0, 0, 255))
a = np.asarray(K.down(L)).astype(np.float32); al = a[..., 3:4] / 255
img = img * (1 - al) + a[..., :3] * al
im = K.finish(img); im.save('thumbnail.jpg', quality=93); im.resize((360, 640), Image.LANCZOS).save('thumb_phone.png'); print('ok')
