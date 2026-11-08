# re-draw the x-ray half of the AI cover so both halves are the same size and centred on the divider
import numpy as np, cv2
XX0 = None
from PIL import Image, ImageDraw, ImageFont, ImageFilter
im = np.asarray(Image.open('cover_raw.png').convert('RGB')).copy()
H, W = im.shape[:2]
CX, CY, R = 581, 836, 378
# 2) fill the right half-disc (radius R) with night blue + soft inner glow
YY, XX = np.mgrid[0:H, 0:W]
rr = np.hypot(XX - CX, YY - CY)
half = (XX >= CX) & (rr <= R + 1)
night = np.array([6, 11, 24], np.float32)
glow = np.exp(-(rr/R)**2*2.2)[..., None]*np.array([10, 40, 70], np.float32)
fill = np.clip(night + glow, 0, 255)
im = im.astype(np.float32); im[half] = fill[half]
# 1) remove the stray horizontal scanline right of the old arc (y ~ 800) by inpainting
a = im.astype(int); cy = (a[..., 2] > 150) & (a[..., 1] > 140) & (a[..., 0] < 140)
mask = np.zeros((H, W), np.uint8)
mask[760:985, 870:] = 255                                     # old scanline stub + old label + old spectrum bar

mask[half] = 0
im = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), mask, 9, cv2.INPAINT_NS)
pil = Image.fromarray(im.astype(np.uint8)); d = ImageDraw.Draw(pil)
CYAN, AMBER, WHITE = (90, 225, 255), (255, 184, 64), (235, 245, 255)
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
fL = ImageFont.truetype(DJ, 46); fS = ImageFont.truetype('Poppins-SemiBold.ttf', 30); fM = ImageFont.truetype(DJ, 30)
# glow layer for line art
gl = Image.new('RGB', (W, H)); g = ImageDraw.Draw(gl)
def both(fn):
    fn(d); fn(g)
both(lambda dd: dd.arc([CX - R, CY - R, CX + R, CY + R], -90, 90, fill=CYAN, width=6))
both(lambda dd: dd.line([(CX, CY - R - 40), (CX, CY + R + 40)], fill=CYAN, width=7))          # the scanline (divider)
# energy ladder: chord lengths follow the circle
def chord(y): return CX + np.sqrt(max(R*R - (y - CY)**2, 0)) - 24
y_hi, y_lo = CY - 150, CY + 70
for y, lab in ((y_hi, 'upper level'), (y_lo, 'lower level')):
    both(lambda dd, y=y: dd.line([(CX + 30, y), (chord(y), y)], fill=CYAN, width=5))
    d.text((chord(y) - 6, y - 12), lab, font=fS, fill=WHITE, anchor='rs')
ex = CX + 70
both(lambda dd: dd.line([(ex, y_hi + 16), (ex, y_lo - 30)], fill=CYAN, width=5))
both(lambda dd: dd.polygon([(ex - 16, y_lo - 34), (ex + 16, y_lo - 34), (ex, y_lo - 6)], fill=CYAN))
both(lambda dd: dd.ellipse([ex - 15, y_hi - 15, ex + 15, y_hi + 15], fill=CYAN))
d.text((ex + 22, (y_hi + y_lo)/2 - 52), 'ΔE', font=fL, fill=AMBER, anchor='lm')
# blue photon wave
BLUE = (70, 120, 255)
xs = np.linspace(ex + 40, CX + 300, 200) + 0; ys = (y_hi + y_lo)/2 + 22*np.sin((xs - xs[0])/16)
both(lambda dd: dd.line(list(zip(xs, ys)), fill=BLUE, width=6))
both(lambda dd: dd.polygon([(CX + 300, (y_hi+y_lo)/2 - 16), (CX + 300, (y_hi+y_lo)/2 + 16), (CX + 326, (y_hi+y_lo)/2)], fill=BLUE))
# equation + spectrum bar inside the half
d.text((CX + 165, CY + 150), 'λ = hc / ΔE', font=fL, fill=CYAN, anchor='mm')
bx0, bx1, by = CX + 40, CX + 290, CY + 212
for x in range(bx0, bx1):
    lam = 400 + (x - bx0)/(bx1 - bx0)*300
    import colorsys
    h = max(0, min(0.75, (700 - lam)/300*0.75)); r, gg, b = colorsys.hsv_to_rgb(h, 1, 1)
    d.line([(x, by - 14), (x, by + 14)], fill=(int(r*255), int(gg*255), int(b*255)))
d.rectangle([bx0, by - 14, bx1, by + 14], outline=WHITE, width=2)
mx = bx0 + (428 - 400)/300*(bx1 - bx0)
d.polygon([(mx - 12, by - 30), (mx + 12, by - 30), (mx, by - 14)], fill=WHITE)
d.text((bx0, by + 44), '428 nm', font=fM, fill=WHITE, anchor='lm')
gl = gl.filter(ImageFilter.GaussianBlur(10))
out = Image.fromarray(np.clip(np.asarray(pil).astype(int) + (np.asarray(gl).astype(int)*0.9), 0, 255).astype(np.uint8))
out.save('cover_fixed.png')
out.resize((1080, 1920), Image.LANCZOS).save('thumbnail.jpg', quality=93)
out.resize((360, 640), Image.LANCZOS).save('/tmp/claude-0/cover_phone.png')
out.resize((576, 1024), Image.LANCZOS).save('/tmp/claude-0/cover_fixed_s.jpg', quality=88)
