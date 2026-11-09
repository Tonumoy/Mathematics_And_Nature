# SC#6 designed cover 1080x1920: real sky (CC BY 4.0 Brainandforce) | right half opened as x-ray (solver photons + computed sky)
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import source_code_kit as K, sky as SK
fr = np.memmap('src/sky.rgb', np.uint8, 'r').reshape(-1, 1920, 1080, 3)
real = fr[60].astype(np.float32)/255
real = np.clip((real - 0.5)*1.12 + 0.5, 0, 1)                              # camera-like contrast
_, e = SK.sky_image(n=4.0); skyc = cv2.resize(SK.sky_image(n=4.0, expo=e*1.25)[0].astype(np.float32), (1080, 1920), interpolation=cv2.INTER_CUBIC)
night = np.ones_like(real)*K.NIGHT/255
xr = night*0.85 + np.clip(skyc, 0, 1)*0.18
rng = np.random.default_rng(9)
acc = np.zeros((1920, 1080, 3), np.float32)
for x in range(0, 1080, 90): acc[:, x] += 0.06*np.array(K.CYAN)/255
for y in range(0, 1920, 90): acc[y, :] += 0.06*np.array(K.CYAN)/255
cols = {410: (0.6, 0.3, 1.0), 450: (0.25, 0.45, 1.0), 530: (0.2, 1.0, 0.3), 700: (1.0, 0.15, 0.1)}
sun = np.array([1000.0, 520.0])
for i in range(120):                                                       # light rays from the sun, short waves bounce
    lam = rng.choice([410, 450, 450, 530, 700]); p = sun + rng.normal(0, 18, 2); a_ = rng.uniform(0.5, 1.4); d = np.array([-np.cos(a_), np.sin(a_)])
    hit = rng.uniform(80, 1100) if lam < 500 else 2000
    L = rng.uniform(300, 1300); q = p + d*min(hit, L)
    cv2.line(acc, tuple((p*4).astype(int)), tuple((q*4).astype(int)), cols[lam], 3, cv2.LINE_AA, shift=2)
    if hit < L:
        th = rng.uniform(0, 2*np.pi); r = q + np.array([np.cos(th), np.sin(th)])*rng.uniform(60, 260)
        cv2.line(acc, tuple((q*4).astype(int)), tuple((r*4).astype(int)), cols[lam], 4, cv2.LINE_AA, shift=2)
        cv2.circle(acc, tuple((q*4).astype(int)), 20, tuple(v/255 for v in K.CYAN), -1, cv2.LINE_AA, shift=2)
cv2.circle(acc, (1000, 520), 46, (1, 0.95, 0.8), -1, cv2.LINE_AA)
acc = acc + cv2.GaussianBlur(acc, (0, 0), 8)*1.3 + cv2.GaussianBlur(acc, (0, 0), 30)*0.6
xr = 1 - (1 - xr)*np.exp(-acc*1.4)
cut = 560 + (np.arange(1920) - 960)*0.06                                    # slightly slanted x-ray seam
m = (np.arange(1080)[None, :] > cut[:, None]).astype(np.float32)[..., None]
m = cv2.GaussianBlur(m, (0, 0), 1.5)[..., None] if m.ndim == 2 else cv2.GaussianBlur(m[..., 0], (0, 0), 1.5)[..., None]
img = real*(1 - m) + xr*m
seam = np.exp(-((np.arange(1080)[None, :] - cut[:, None])/5)**2)[..., None]*np.array(K.CYAN)/255*1.6
img = np.clip(img + seam + np.exp(-((np.arange(1080)[None, :] - cut[:, None])/40)**2)[..., None]*np.array(K.CYAN)/255*0.3, 0, 1)
YY, XX = np.mgrid[0:1920, 0:1080]
img *= (1 - 0.28*(((XX-540)/1080)**2 + ((YY-960)/1920)**2)*2)[..., None]
im = Image.fromarray((img*255).astype(np.uint8)); d = ImageDraw.Draw(im)
FT = ImageFont.truetype('Poppins-ExtraBold.ttf', 112); FS = ImageFont.truetype('Poppins-ExtraBold.ttf', 50); FN = ImageFont.truetype('Poppins-ExtraBold.ttf', 64)
def box(lines, y):
    for s in lines:
        w = d.textlength(s, font=FT); d.rounded_rectangle([485 - w/2 - 34, y, 485 + w/2 + 34, y + 140], 26, fill=K.YELLOW)
        d.text((485, y + 72), s, font=FT, fill=(10, 10, 14), anchor='mm'); y += 156
    return y
y = box(['WHY NOT', 'VIOLET?'], 230)
d.text((485, y + 40), 'why the sky is really blue', font=FS, fill=(255, 255, 255), anchor='mm', stroke_width=5, stroke_fill=(0, 0, 0))
d.text((485, 1380), 'violet scatters 9.4x more', font=FN, fill=K.CYAN, anchor='mm', stroke_width=6, stroke_fill=(0, 0, 0))
d.text((485, 1460), 'than red. so why not violet?', font=FS, fill=(255, 255, 255), anchor='mm', stroke_width=5, stroke_fill=(0, 0, 0))
d.text((925, 185), 'SOURCE CODE #6', font=ImageFont.truetype('Poppins-ExtraBold.ttf', 34), fill=K.YELLOW, anchor='rm', stroke_width=4, stroke_fill=(0, 0, 0))
im.save('thumbnail.jpg', quality=93)
im.resize((360, 640), Image.LANCZOS).save('thumb_small.png')
