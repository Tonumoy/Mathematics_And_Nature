# SOURCE CODE series kit (locked 3-23 Nov 2026): fixed text kit + fixed sound kit.
# Reuse every Short: import, never restyle. Colours: thing cyan, driver amber, chaos red, order green.
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
HERE = os.path.dirname(os.path.abspath(__file__))
def _f(name, size): return ImageFont.truetype(os.path.join(HERE, name), size)
W, H = 1080, 1920
CX = 490                                     # safe-box text centre
CYAN, AMBER, RED, GREEN = (90, 225, 255), (255, 184, 64), (255, 72, 72), (80, 235, 140)
YELLOW, WHITE = (255, 214, 0), (255, 255, 255)
NIGHT = np.array([6, 11, 24], np.float32)    # x-ray background (near-black night blue)
F_TITLE = _f('Poppins-ExtraBold.ttf', 48); F_CAP = _f('Poppins-ExtraBold.ttf', 54)
F_HOOK = _f('Poppins-SemiBold.ttf', 34); F_PILL = _f('Poppins-SemiBold.ttf', 32)
F_TAG = _f('Poppins-ExtraBold.ttf', 26); F_EQ = _f('Poppins-ExtraBold.ttf', 76); F_SUB = _f('Poppins-ExtraBold.ttf', 46)
F_CHIP = _f('Poppins-SemiBold.ttf', 34); F_MONO = _f('JetBrainsMono.ttf', 28)

def title_box(d, text, hook=None, y=300):
    tw = d.textlength(text, font=F_TITLE); x0 = CX - tw/2 - 28
    d.rounded_rectangle([x0, y, CX + tw/2 + 28, y + 78], 16, fill=YELLOW)
    d.text((CX, y + 39), text, font=F_TITLE, fill=(10, 10, 14), anchor='mm')
    if hook: d.text((CX, y + 120), hook, font=F_HOOK, fill=WHITE, anchor='mm', stroke_width=3, stroke_fill=(0, 0, 0))
def series_tag(d, n):
    d.text((925, 182), f'SOURCE CODE #{n}', font=F_TAG, fill=YELLOW, anchor='rm', stroke_width=3, stroke_fill=(0, 0, 0))
def caption(d, lines, y=1490, alpha=1.0):
    """lines: list (max 2) of [(text, colour), ...] segments; key words YELLOW"""
    if alpha <= 0: return
    lh = 70; y0 = y - (len(lines)-1)*lh/2
    for i, segs in enumerate(lines):
        tw = sum(d.textlength(t, font=F_CAP) for t, _ in segs); x = CX - tw/2
        for t, c in segs:
            col = tuple(int(v*alpha) for v in c)
            d.text((x, y0 + i*lh), t, font=F_CAP, fill=col, anchor='lm', stroke_width=6, stroke_fill=(0, 0, 0))
            x += d.textlength(t, font=F_CAP)
def pill(d, xy, text, col, font=F_PILL, anchor='mm'):
    x, y = xy; tw = d.textlength(text, font=font); hh = font.size*0.75
    if anchor == 'mm': x0 = x - tw/2 - 20
    else: x0 = x
    d.rounded_rectangle([x0, y - hh, x0 + tw + 40, y + hh], int(hh), fill=(8, 12, 22), outline=col, width=3)
    d.text((x0 + 20 + tw/2, y), text, font=font, fill=col, anchor='mm')
def end_stamp(d, alpha=1.0, y=1340):
    if alpha <= 0: return
    c = tuple(int(v*alpha) for v in YELLOW)
    d.text((CX, y), 'nature is running on code', font=F_HOOK, fill=c, anchor='mm', stroke_width=3, stroke_fill=(0, 0, 0))

# ---------------- sound kit ----------------
def motif_and_swell(compose_mod, total_s, scan_t, sr=44100):
    """fixed 4-note cello motif at 0 s (D3 F3 Bb2 A2) + in-key scan swell at scan_t. returns stereo array"""
    m = compose_mod; L = int(total_s*sr) + sr; out = np.zeros(L)
    t = 0.05
    for note, dur in ((50, 0.55), (53, 0.55), (46, 0.8), (45, 1.6)):
        m.place(out, m.cello(m.hz(note), dur + 0.35)*0.9, t, 0.32); t += dur
    sw = m.strings((50, 57, 62, 65), 2.2, 1800)
    n = len(sw); e = np.ones(n); a = int(0.9*sr); e[:a] = np.linspace(0, 1, a)**2
    m.place(out, sw*e, scan_t - 0.9, 0.30)
    m.place(out, m.swell(1.0), scan_t - 0.9, 0.05)
    st = m.verb(out, 0.4)
    return st[:int(total_s*sr)]
