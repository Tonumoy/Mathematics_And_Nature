# FIX CHAIN #1 cover from the video's own render (owner 14 Nov: frame from the video, professionally edited)
from lib import *
from scenes import draw_fridge, earth_frame
from PIL import ImageDraw, ImageFilter
import numpy as np, math
EARTH_D, EARTH_CY = 1080, 990
FR_CX, FR_BY, FR_H = 540, 1860, 390
b, pole, R = earth_frame(1.2, EARTH_D, EARTH_CY, 1.0, 26)
# extra violet bloom over the hole
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
bl = np.exp(-(((x - pole[0]) / 210) ** 2 + ((y - pole[1]) / 150) ** 2))
b = b + bl[..., None] * np.array([70, 20, 120], np.float32) * 0.8
L = new_layer(); d = ImageDraw.Draw(L)
# gas trail: particles along a curve fridge -> hole
rng = np.random.default_rng(3)
x0, y0 = FR_CX, FR_BY - FR_H + 40
x1, y1 = pole
for k in range(900):
    s = rng.random() ** 0.8
    xc = lerp(x0, x1, s) + 70 * math.sin(s * 7.5) * (1 - s) ** 0.5
    yc = lerp(y0, y1, s)
    xc += rng.normal(0, 16 * (1 - s) + 5); yc += rng.normal(0, 8)
    r = rng.uniform(2, 6) * (1 - 0.5 * s)
    a = int(255 * rng.uniform(0.5, 1))
    d.ellipse([(xc - r) * SS, (yc - r) * SS, (xc + r) * SS, (yc + r) * SS], fill=(130, 255, 120, a))
FL = new_layer(); fd = ImageDraw.Draw(FL)
fx0, fy0, fx1, fy1 = draw_fridge(fd, FR_CX, FR_BY, FR_H, 2, 255)
fw = fx1 - fx0
# shading: darker right side + top highlight
for k in range(40):
    a = int(90 * k / 40)
    fd.line([(fx1 - 40 + k) * SS, (fy0 + 30) * SS, (fx1 - 40 + k) * SS, (fy1 - 30) * SS], fill=(20, 40, 50, a), width=SS)
# x-ray back half: cyan coils + compressor
xm = FR_CX + 6
fd.rectangle([xm * SS, (fy0 + 20) * SS, (fx1 - 14) * SS, (fy1 - 20) * SS], fill=(8, 30, 50, 235))
for k in range(9):
    yy = fy0 + 50 + k * 26
    fd.line([(xm + 14) * SS, yy * SS, (fx1 - 30) * SS, yy * SS], fill=CYAN + (255,), width=3 * SS)
    xe = (fx1 - 30) if k % 2 == 0 else (xm + 14)
    fd.arc([(xe - 13) * SS, yy * SS, (xe + 13) * SS, (yy + 26) * SS], -90 if k % 2 == 0 else 90, 90 if k % 2 == 0 else 270, fill=CYAN + (255,), width=3 * SS)
fd.rounded_rectangle([(xm + 20) * SS, (fy1 - 95) * SS, (fx1 - 30) * SS, (fy1 - 35) * SS], radius=14 * SS, outline=CYAN + (255,), width=3 * SS)
fd.rounded_rectangle([xm * SS, (fy0 + 20) * SS, (fx1 - 14) * SS, (fy1 - 20) * SS], radius=10 * SS, outline=CYAN + (255,), width=4 * SS)
fd.line([xm * SS, (fy0 - 10) * SS, xm * SS, (fy1 + 10) * SS], fill=CYAN + (255,), width=3 * SS)
# cold light from open top
L.alpha_composite(FL)
ring(d, pole, R * 0.37, CYAN, 255, 5)
L1 = down(L)
b = add_light(b, glow(L1, 26), 0.9)
b = over(b, L1)
# title
T = new_layer(); td = ImageDraw.Draw(T)
f = font(FONT_XB, 150 * SS)
def line(y, parts):
    tw = sum(f.getlength(p) for p, _ in parts); xx = (W * SS - tw) / 2
    for p, c in parts:
        td.text((xx, y * SS), p, font=f, fill=c + (255,), anchor="lm", stroke_width=10 * SS, stroke_fill=(0, 0, 0, 255))
        xx += f.getlength(p)
line(215, [("YOUR ", WHITE), ("FRIDGE", YELLOW)])
line(380, [("DID THIS", WHITE)])
pill(td, (540, 505), "1 chlorine atom wrecks 100,000+ ozone", CYAN, 34)
td.text(((W - 40) * SS, 90 * SS), "FIX CHAIN #1", font=font(FONT_XB, 34 * SS), fill=YELLOW + (255,), anchor="rm", stroke_width=4 * SS, stroke_fill=(0, 0, 0, 255))
T1 = down(T)
# soft dark scrim behind title for contrast
sc = np.clip(1 - (y - 120) / 520, 0, 1) ** 1.5 * 0.55
b = b * (1 - sc[..., None])
b = over(b, T1)
im = finish(b)
im.save('thumbnail.jpg', quality=93)
im.resize((360, 640), Image.LANCZOS).save('thumb_phone.png')
print('ok')
