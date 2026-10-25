"""Code-drawn covers: 1080x1920 + 1280x720 from the mandala at t=0 (no captions)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import chakra_mandala as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HEAD = [("50 PETALS.", (255, 255, 255)), ("50 LETTERS.", cm.hex_rgb(cm.C_HERO))]
SUB = "6 lotuses + the crown, counted"

arr = cm.sim_image(0.0)
img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
glow = img.resize((cm.SIM_W // 4, cm.SIM_H // 4)).filter(ImageFilter.GaussianBlur(5)).resize((cm.SIM_W, cm.SIM_H))
sim = Image.fromarray(np.clip(arr + 1.3 * np.asarray(glow, np.float32) + np.array(cm.BG_RGB, np.float32), 0, 255).astype(np.uint8))

def vertical():
    c = Image.new("RGB", (1080, 1920), cm.BG_RGB)
    c.paste(sim, (60, 640))
    d = ImageDraw.Draw(c)
    for i, (t, col) in enumerate(HEAD):
        d.text((540, 260 + i * 150), t, font=cm.font(140), fill=col, anchor="ma", stroke_width=8, stroke_fill=cm.BG_RGB)
    d.text((540, 1570), SUB, font=cm.font(56, False), fill=(225, 230, 240), anchor="ma")
    d.text((540, 1660), "@cosmoscoded", font=cm.font(40, False), fill=(170, 175, 190), anchor="ma")
    c.save(os.path.join(OUT, "cover_9x16.jpg"), quality=93)

def wide():
    c = Image.new("RGB", (1280, 720), cm.BG_RGB)
    s = sim.resize((int(960 * 0.7), int(900 * 0.7)), Image.LANCZOS)
    c.paste(s, (1280 - s.size[0] - 10, (720 - s.size[1]) // 2))
    d = ImageDraw.Draw(c)
    for i, (t, col) in enumerate(HEAD):
        d.text((40, 220 + i * 105), t, font=cm.font(86), fill=col, stroke_width=6, stroke_fill=cm.BG_RGB)
    d.text((44, 450), SUB, font=cm.font(44, False), fill=(225, 230, 240))
    d.text((44, 640), "@cosmoscoded", font=cm.font(32, False), fill=(170, 175, 190))
    c.save(os.path.join(OUT, "cover_16x9.jpg"), quality=93)

vertical(); wide(); print("covers ok")
