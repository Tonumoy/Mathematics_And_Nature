import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FPS = 30
BG = (5, 6, 10)
CYAN = (60, 225, 255); AMBER = (255, 178, 40); RED = (255, 70, 60); GREEN = (80, 235, 140)
YEL = (255, 222, 60); WHITE = (245, 245, 245)
FB = '/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf'
FM = '/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf'
_fc = {}
def font(sz, f=FB):
    k = (sz, f)
    if k not in _fc: _fc[k] = ImageFont.truetype(f, sz)
    return _fc[k]

def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def ramp(t, a, b): return float(np.clip((t - a) / max(b - a, 1e-6), 0, 1))

def glow(img, sigma=12, gain=1.0):
    """img float32 HxWx3 0..1 -> add bloom"""
    b = cv2.GaussianBlur(img, (0, 0), sigma)
    b2 = cv2.GaussianBlur(img, (0, 0), sigma * 3)
    return img + gain * (0.6 * b + 0.4 * b2)

def to_u8(img): return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

def text_layer(items):
    """items: list of dict(text, xy, size, fill, anchor, stroke, segs) -> RGBA PIL image.
    segs: list of (text, color) for mixed colour lines."""
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for it in items:
        f = font(it.get('size', 60), it.get('font', FB)); a = it.get('alpha', 1.0)
        sw = it.get('stroke', 6); anchor = it.get('anchor', 'mm')
        segs = it.get('segs') or [(it['text'], it.get('fill', WHITE))]
        full = ''.join(s for s, _ in segs)
        x0, y0 = it['xy']
        tw = d.textlength(full, font=f)
        if anchor[0] == 'm': x = x0 - tw / 2
        elif anchor[0] == 'r': x = x0 - tw
        else: x = x0
        for s, col in segs:
            d.text((x, y0), s, font=f, fill=tuple(col) + (int(255 * a),), anchor='l' + anchor[1],
                   stroke_width=sw, stroke_fill=(5, 6, 10, int(230 * a)))
            x += d.textlength(s, font=f)
    return im

def band(im, y0, y1, alpha=0.55):
    """dark gradient band behind captions (PIL RGBA)"""
    a = np.zeros((H, W), np.float32)
    yy = np.arange(H)[:, None]
    m = np.clip(np.minimum(yy - y0, y1 - yy) / 40.0, 0, 1) * alpha
    a[:] = m
    return a

def pill(text, xy, col, size=34, alpha=1.0):
    """label pill RGBA layer"""
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f = font(size); tw = d.textlength(text, font=f); x, y = xy
    pad = 18; h = size + 22
    d.rounded_rectangle([x - tw / 2 - pad, y - h / 2, x + tw / 2 + pad, y + h / 2], radius=h // 2,
                        fill=(8, 10, 16, int(200 * alpha)), outline=tuple(col) + (int(255 * alpha),), width=3)
    d.text((x, y), text, font=f, fill=tuple(col) + (int(255 * alpha),), anchor='mm')
    return im

def comp(base_u8, layer):
    """alpha composite PIL RGBA layer over uint8 RGB array"""
    b = Image.fromarray(base_u8).convert('RGBA'); b.alpha_composite(layer); return np.array(b.convert('RGB'))

def ring(img_u8, c, r, col=CYAN, a=1.0, th=3):
    ov = img_u8.copy(); cv2.circle(ov, (int(c[0]), int(c[1])), int(r), col, th, cv2.LINE_AA)
    return cv2.addWeighted(ov, a, img_u8, 1 - a, 0)
