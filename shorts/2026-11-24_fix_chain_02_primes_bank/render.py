# FIX CHAIN #2 v2 - "2 numbers guard your money" (Coded Cosmos)
# Hand-built pixel-art world (270x480 logical pixels, nearest-neighbour upscale), every pixel drawn by this code.
import sys, os, math, json, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

LW, LH = 270, 480
UP = int(sys.argv[1]) if len(sys.argv) > 1 else 4          # 4 -> 1080x1920, 8 -> 2160x3840
FPS = 30
DURS = json.load(open(os.path.join(os.path.dirname(__file__) or ".", "vo/durs.json")))
GAP = [0.25, 0.2, 0.2, 0.2, 0.2, 0.15, 0.25, 0.3, 0.25, 0.0]
L = []; t0 = 0.0
for d, g in zip(DURS, GAP): L.append(t0); t0 += d + g
DUR = round(t0 + 1.6, 2)
def Ls(i): return L[i - 1]
def Le(i): return L[i - 1] + DURS[i - 1]

# palette (hand-picked, 14 colours)
P = dict(
    night=(14, 16, 33), night2=(24, 26, 54), navy=(36, 40, 82), slate=(70, 78, 120),
    sand=(232, 196, 140), parch=(244, 226, 186), brown=(120, 76, 48), dbrown=(66, 40, 30),
    gold=(255, 196, 64), amber=(255, 140, 40), red=(232, 64, 64), dred=(130, 28, 40),
    green=(96, 220, 120), teal=(64, 200, 220), purple=(150, 90, 230), white=(244, 244, 236),
    grey=(150, 156, 180), black=(6, 6, 12), skin=(222, 164, 120))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_fc = {}
def font(sz):
    if sz not in _fc: _fc[sz] = ImageFont.truetype(FONT, sz)
    return _fc[sz]

def ease(u): u = min(1, max(0, u)); return u * u * (3 - 2 * u)
def eout(u): u = min(1, max(0, u)); return 1 - (1 - u) ** 3
def win(t, a, b, fi=0.25, fo=0.25):
    if t < a or t > b: return 0.0
    return min(1, (t - a) / fi if fi > 0 else 1, (b - t) / fo if fo > 0 else 1)
def lerp(a, b, u): return a + (b - a) * u
def mixc(c1, c2, u): return tuple(int(lerp(a, b, u)) for a, b in zip(c1, c2))
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0

class Px:
    """low-res pixel canvas with PIL drawing (no anti-aliasing)"""
    def __init__(s):
        s.im = Image.new("RGB", (LW, LH), P["night"]); s.d = ImageDraw.Draw(s.im); s.d.fontmode = "1"
        s.glow = []      # (x, y, r, colour, strength) light sources for bloom
    def rect(s, x, y, w, h, c, o=None):
        if w < 1 or h < 1: return
        s.d.rectangle([int(x), int(y), int(x + w - 1), int(y + h - 1)], fill=c, outline=o)
    def circ(s, x, y, r, c, o=None):
        s.d.ellipse([int(x - r), int(y - r), int(x + r), int(y + r)], fill=c, outline=o)
    def line(s, a, b, c, w=1): s.d.line([tuple(map(int, a)), tuple(map(int, b))], fill=c, width=w)
    def poly(s, pts, c, o=None): s.d.polygon([tuple(map(int, p)) for p in pts], fill=c, outline=o)
    def text(s, txt, x, y, sz, c, anchor="l", shadow=True):
        f = font(sz); w = s.d.textlength(txt, font=f)
        if anchor == "c": x -= w / 2
        elif anchor == "r": x -= w
        if shadow: s.d.text((int(x) + 1, int(y) + 1), txt, font=f, fill=P["black"])
        s.d.text((int(x), int(y)), txt, font=f, fill=c)
        return w
    def light(s, x, y, r, c, k=1.0): s.glow.append((x, y, r, c, k))

def pp(px, xy, c):
    x, y = int(xy[0]), int(xy[1])
    if 0 <= x < LW and 0 <= y < LH: px.im.putpixel((x, y), c)

def dither_grad(px, y0, y1, c0, c1):
    a = np.array(px.im)
    ys = np.arange(y0, y1)
    u = ((ys - y0) / max(1, (y1 - y0 - 1)))[:, None]
    th = BAYER[(ys % 4)[:, None], (np.arange(LW) % 4)[None, :]]
    steps = 6
    q = np.floor(u * steps + th) / steps
    q = np.clip(q, 0, 1)[..., None]
    a[y0:y1] = (np.array(c0) * (1 - q) + np.array(c1) * q).astype(np.uint8)
    px.im = Image.fromarray(a); px.d = ImageDraw.Draw(px.im); px.d.fontmode = "1"

RNG = np.random.default_rng(3)
STARS = [(RNG.integers(0, LW), RNG.integers(0, 260), RNG.random()) for _ in range(70)]
def sky(px, t, top=P["night"], bot=P["navy"], stars=True):
    dither_grad(px, 0, LH, top, bot)
    if stars:
        for (x, y, ph) in STARS:
            if math.sin(t * 2.5 + ph * 20) > -0.3:
                pp(px, (int(x), int(y)), P["white"] if ph > 0.8 else P["slate"])

def hills(px, t, y=360, c1=P["night2"], c2=(18, 20, 40)):
    for i, (yy, c, f, amp) in enumerate(((y - 30, c1, 0.035, 14), (y, c2, 0.05, 10))):
        pts = [(0, LH)] + [(x, yy - amp * math.sin(x * f + i * 2 + t * 0.05) - amp * 0.5 * math.sin(x * f * 2.3)) for x in range(0, LW + 6, 6)] + [(LW, LH)]
        px.poly(pts, c)

# ---------------- reusable pixel props ----------------
def padlock(px, cx, cy, s=1.0, open_=0.0, body=P["gold"], crack=0.0, glow=True):
    """chunky pixel padlock, centre of body at (cx, cy); s=scale"""
    bw, bh = int(44 * s), int(36 * s)
    x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
    # shackle
    lift = int(14 * s * open_)
    sw = int(30 * s); th = max(2, int(6 * s))
    sx0 = int(cx - sw / 2); top = y0 - int(26 * s) - lift
    px.d.rounded_rectangle([sx0, top, sx0 + sw, y0 + 4 - (lift if open_ > 0 else 0)], radius=int(14 * s), outline=P["grey"], width=th)
    px.d.rounded_rectangle([sx0 + 1, top + 1, sx0 + sw - 1, y0 + 3 - (lift if open_ > 0 else 0)], radius=int(13 * s), outline=P["white"], width=1)
    # body with shading bands
    dark = mixc(body, P["black"], 0.45); light_ = mixc(body, P["white"], 0.45)
    px.d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=int(5 * s), fill=body, outline=P["black"])
    px.rect(x0 + 2, y0 + bh - int(8 * s), bw - 3, int(6 * s), dark)
    px.rect(x0 + 3, y0 + 2, bw - 6, max(1, int(3 * s)), light_)
    for k in range(3):
        px.rect(x0 + 3, y0 + int((10 + 7 * k) * s), bw - 6, 1, mixc(body, P["black"], 0.2))
    # keyhole
    px.circ(cx, cy - 2 * s, max(2, int(4 * s)), P["black"])
    px.poly([(cx - 2 * s, cy), (cx + 2 * s, cy), (cx + 3 * s, cy + 9 * s), (cx - 3 * s, cy + 9 * s)], P["black"])
    if crack > 0:
        pts = [(x0 + 2, y0 + 8 * s), (x0 + 12 * s, y0 + 14 * s), (x0 + 18 * s, y0 + 8 * s), (cx + 2, y0 + 20 * s), (cx + 10 * s, y0 + 12 * s), (x0 + bw - 2, y0 + 22 * s)]
        n = max(2, int(len(pts) * crack + 0.99))
        for a, b in zip(pts[:n - 1], pts[1:n]): px.line(a, b, P["red"], max(1, int(2 * s)))
    if glow: px.light(cx, cy, 40 * s, body, 0.6)

def phone(px, cx, cy, w=96, h=176):
    x0, y0 = cx - w // 2, cy - h // 2
    px.d.rounded_rectangle([x0 - 3, y0 - 3, x0 + w + 3, y0 + h + 3], radius=12, fill=P["black"])
    px.d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10, fill=(20, 30, 60), outline=P["slate"])
    px.rect(cx - 12, y0 + 5, 24, 4, P["black"])
    return x0, y0

def hand(px, cx, cy):
    # simple pixel hand holding the phone from below-left
    px.d.rounded_rectangle([cx - 62, cy + 60, cx - 30, cy + 140], radius=10, fill=P["skin"], outline=P["brown"])
    for k in range(3):
        px.d.rounded_rectangle([cx + 46, cy - 10 + k * 18, cx + 62, cy + 4 + k * 18], radius=6, fill=P["skin"], outline=P["brown"])
    px.d.rounded_rectangle([cx - 50, cy + 110, cx + 30, cy + 170], radius=14, fill=P["skin"], outline=P["brown"])

def scroll(px, cx, cy, w, h, c=P["parch"]):
    px.rect(cx - w // 2, cy - h // 2, w, h, c, P["brown"])
    for x in (cx - w // 2 - 6, cx + w // 2 - 2):
        px.rect(x, cy - h // 2 - 6, 8, h + 12, P["brown"], P["dbrown"])
        px.rect(x + 2, cy - h // 2 - 4, 2, h + 8, mixc(P["brown"], P["white"], 0.3))
    # paper texture specks
    rr = np.random.default_rng(5)
    for _ in range(int(w * h / 60)):
        pp(px, (int(cx - w // 2 + 2 + rr.integers(0, max(1, w - 4))), int(cy - h // 2 + 2 + rr.integers(0, max(1, h - 4)))), mixc(c, P["brown"], 0.25))

def keysprite(px, x, y, s=1.0, c=P["gold"]):
    if s < 0.08: return
    px.circ(x, y, 7 * s, c, P["black"]); px.circ(x, y, 3 * s, P["night"])
    px.rect(x + 6 * s, y - 2 * s, 22 * s, 4 * s, c, P["black"])
    px.rect(x + 20 * s, y + 2 * s, 3 * s, 5 * s, c); px.rect(x + 25 * s, y + 2 * s, 3 * s, 7 * s, c)
    px.light(x + 10 * s, y, 18 * s, c, 0.5)

def person(px, x, y, c=P["teal"], hood=False, step=0):
    # 20x40 pixel figure, x,y = feet centre
    a = 3 if step == 0 else -3
    px.poly([(x - 2, y - 14), (x - 6 + a, y), (x - 2 + a, y)], P["dbrown"]); px.poly([(x + 2, y - 14), (x + 2 - a, y), (x + 6 - a, y)], P["dbrown"])
    px.d.rounded_rectangle([x - 8, y - 32, x + 8, y - 12], radius=4, fill=c, outline=P["black"])
    px.rect(x - 7, y - 30, 3, 16, mixc(c, P["white"], 0.3))
    if hood:
        px.poly([(x - 10, y - 30), (x, y - 50), (x + 10, y - 30)], c, P["black"])
        px.rect(x - 4, y - 38, 8, 5, P["black"]); pp(px, (x - 2, y - 36), P["red"]); pp(px, (x + 2, y - 36), P["red"])
    else:
        px.circ(x, y - 39, 7, P["skin"], P["black"]); px.rect(x - 7, y - 46, 14, 4, P["dbrown"])

def fridge(px, cx, top, s=1.0, t=0.0, a=1.0):
    """pixel 'chandelier' quantum computer: gold plates + coax cables"""
    plates = [(46, 0), (38, 34), (32, 64), (26, 92), (20, 118)]
    for i, (r, dy) in enumerate(plates):
        r = int(r * s); y = int(top + dy * s)
        px.d.ellipse([cx - r, y - 5, cx + r, y + 5], fill=P["gold"], outline=P["dbrown"])
        px.d.ellipse([cx - r + 3, y - 3, cx + r - 3, y + 1], fill=mixc(P["gold"], P["white"], 0.4))
        if i < len(plates) - 1:
            r2 = int(plates[i + 1][0] * s); y2 = int(top + plates[i + 1][1] * s)
            for k in range(-3, 4):
                xa = cx + k * r // 4; xb = cx + k * r2 // 4
                px.line((xa, y + 4), (xb, y2 - 4), mixc(P["amber"], P["dbrown"], 0.3 + 0.1 * (k % 2)))
    # coiled cables
    for k in range(3):
        xs = cx - 10 + k * 10
        for yy in range(int(top + 8), int(top + 112 * s), 3):
            pp(px, (int(xs + 3 * math.sin(yy * 0.4 + k + t * 4)), yy), P["teal"])
    chip_y = int(top + 128 * s)
    px.rect(cx - 10, chip_y, 20, 10, P["purple"], P["black"])
    for i in range(4):
        pp(px, (cx - 7 + i * 5, chip_y + 4), P["white"] if math.sin(t * 9 + i) > 0 else P["purple"])
    px.light(cx, chip_y + 5, 40, P["purple"], 0.9)

# ---------------- HUD ----------------
STEPS = None
def steps():
    return [("SECRET", "!", Ls(2)), ("SHIFT +3", "ok", Ls(2) + 3.0), ("COUNTED", "!", Ls(3) + 2.2), ("LONG KEY", "ok", Ls(4)),
            ("STOLEN", "!", Ls(4) + 3.2), ("PADLOCK", "ok", Ls(5) + 2.0), ("QUANTUM", "!", Ls(8) + 1.0), ("NEW LOCK", "ok", Ls(9) + 1.0)]
def chain_hud(px, t):
    if t < Ls(2) - 0.2: return
    st = steps(); x0 = 14; dx = 33; y = 46
    for i, (lab, k, ts) in enumerate(st):
        if t < ts: continue
        x = x0 + i * dx
        col = P["red"] if k == "!" else P["green"]
        pop = 1 + int(3 * (1 - eout((t - ts) / 0.25)))
        px.rect(x - 5 - pop, y - 5 - pop, 11 + 2 * pop, 11 + 2 * pop, col, P["black"])
        if k == "!": px.rect(x - 1, y - 3, 2, 4, P["white"]); px.rect(x - 1, y + 2, 2, 2, P["white"])
        else: px.line((x - 3, y), (x - 1, y + 2), P["white"]); px.line((x - 1, y + 2), (x + 3, y - 3), P["white"])
        if i > 0 and t > ts: px.rect(x - dx + 7, y, dx - 14, 1, P["grey"])

def label_box(px, t, a, b, l1, l2=None, col=P["gold"]):
    """bit_wonder-style boxed label at top-centre"""
    if not (a <= t <= b): return
    w = max(px.d.textlength(l1, font=font(10)), px.d.textlength(l2 or "", font=font(8))) + 14
    h = 18 if not l2 else 28
    x = LW / 2 - w / 2; y = 62
    px.rect(x, y, w, h, P["black"], col); px.rect(x + 1, y + 1, w - 2, h - 2, None, mixc(col, P["black"], 0.5))
    px.text(l1, LW / 2, y + 3, 10, col, "c")
    if l2: px.text(l2, LW / 2, y + 15, 8, P["white"], "c")

def series_tag(px): px.text("FIX CHAIN #2", 232, 26, 8, P["teal"], "r", shadow=False)

CAPS = None
def caps():
    return [
        (Ls(1) + 0.0, Le(1), "2 SECRET NUMBERS", None),
        (Ls(2), Le(2), "A  ->  D", "SHIFT EVERY LETTER BY 3"),
        (Ls(3), Le(3), "COUNT THE LETTERS", "THE SECRET FALLS OUT"),
        (Ls(4), Le(4), "WHO DELIVERS THE KEY?", None),
        (Ls(5), Le(5), "ANYONE CAN LOCK IT", "ONLY YOU CAN OPEN IT"),
        (Ls(6), Le(6), "MULTIPLY = EASY", "GO BACKWARDS = HARD"),
        (Ls(7), Le(7), "617 DIGITS", "NEVER CRACKED"),
        (Ls(8), Le(8), "QUANTUM COMPUTER", "COULD CRACK IT"),
        (Ls(9), Le(9), "NEW LOCKS (2024)", "NO KNOWN SHORTCUT"),
        (Ls(10), Le(10), "EVERY FIX =", "A NEW PROBLEM"),
    ]
def captions(px, t):
    for a, b, l1, l2 in caps():
        if not (a <= t <= b + 0.1): continue
        y = 366
        w = px.d.textlength(l1, font=font(13)) + 16
        if l2: w = max(w, px.d.textlength(l2, font=font(10)) + 16)
        h = 22 if not l2 else 36
        px.rect(121 - w / 2, y - 3, w, h, P["black"])
        px.text(l1, 121, y, 13, P["gold"], "c")
        if l2: px.text(l2, 121, y + 17, 10, P["white"], "c")

# ---------------- scenes ----------------
def sc_hook(px, t, a0=0.0):
    sky(px, t, P["night"], (40, 30, 70))
    # menacing quantum chandelier in the sky (danger)
    fridge(px, 226, 150 + 3 * math.sin(t), 0.42, t)
    px.text("COMING", 226, 222, 8, P["purple"], "c")
    hills(px, t)
    zoom = 1.0
    cx, cy = 112, 250
    x0, y0 = phone(px, cx, cy, 116, 200)
    # bank app UI
    px.rect(x0 + 6, y0 + 16, 84, 14, P["navy"]); px.text("MY BANK", x0 + 10, y0 + 18, 9, P["white"])
    px.text("BALANCE", x0 + 10, y0 + 36, 7, P["grey"], shadow=False)
    bal = 25000 + int(400 * math.sin(t * 2))
    px.text(f"Rs {bal:,}", x0 + 10, y0 + 46, 10, P["green"])
    padlock(px, cx, y0 + 132, 1.25, 0.0)
    # two prime orbs orbit the lock
    for k, (lab, c) in enumerate((("p", P["amber"]), ("q", P["teal"]))):
        ang = t * 2.2 + k * math.pi
        ox = cx + 64 * math.cos(ang); oy = y0 + 128 + 16 * math.sin(ang)
        if math.sin(ang) < 0 or True:
            px.circ(ox, oy, 9, c, P["black"]); px.text(lab, ox, oy - 7, 11, P["black"], "c", shadow=False)
            px.light(ox, oy, 22, c, 0.8)
    # hook text
    px.text("2 NUMBERS", 121, 84, 24, P["gold"], "c")
    px.text("GUARD YOUR MONEY", 121, 112, 14, P["white"], "c")

ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def caesar(s, k=3): return "".join(ALPH[(ALPH.index(c) + k) % 26] if c in ALPH else c for c in s)
MSG = "ATTACK AT DAWN"
def sc_caesar(px, t):
    ts = Ls(2)
    sky(px, t, (30, 18, 24), (70, 40, 34), stars=False)
    # torch light + stone wall
    for yy in range(140, 340, 14):
        off = 0 if (yy // 14) % 2 else 10
        for xx in range(-off, LW, 22): px.rect(xx, yy, 21, 13, (58, 36, 32), (40, 24, 22))
    for tx in (30, 212):
        fl = 3 * math.sin(t * 13 + tx)
        px.rect(tx - 2, 200, 4, 18, P["brown"]); px.poly([(tx - 5, 200), (tx, 186 - fl), (tx + 5, 200)], P["amber"]); px.light(tx, 194, 60, P["amber"], 0.9)
    u = ease((t - ts) / 0.8)
    w = int(lerp(20, 210, u))
    scroll(px, 121, 236, w, 120)
    label_box(px, t, ts + 0.3, Le(2), "ROME, ~50 BC", "CAESAR'S SECRET CODE", P["amber"])
    if u > 0.9:
        # plain message, then each letter hops +3
        hop = (t - ts - 2.2) / 2.4
        for i, ch in enumerate(MSG):
            x = 121 - len(MSG) * 7 + i * 14 + 4
            if ch == " ": continue
            k = int(np.clip(hop * 3 - i * 0.05, 0, 3))
            c = ALPH[(ALPH.index(ch) + k) % 26]
            col = P["dbrown"] if k == 0 else (P["dred"] if k < 3 else P["red"])
            px.text(c, x, 212 - (3 if 0 < (hop * 3 - i * 0.05) % 1 < 0.4 and k < 3 else 0), 13, col, "c", shadow=False)
        # alphabet strip A B C D with jumping marker
        sy = 256
        for i, ch in enumerate("ABCDEFG"):
            x = 70 + i * 17
            px.rect(x - 7, sy, 14, 15, mixc(P["parch"], P["brown"], 0.15), P["brown"])
            px.text(ch, x, sy + 1, 11, P["red"] if ch == "D" and hop > 0.8 else P["dbrown"], "c", shadow=False)
        mk = 70 + 17 * min(3, max(0, hop * 3.2))
        px.poly([(mk - 4, sy - 7), (mk + 4, sy - 7), (mk, sy - 2)], P["red"])
        if hop > 0.8: px.text("+3", 121, 278, 11, P["red"], "c", shadow=False)

ENG = {'A': 8.2, 'B': 1.5, 'C': 2.8, 'D': 4.3, 'E': 12.7, 'F': 2.2, 'G': 2.0, 'H': 6.1, 'I': 7.0, 'J': 0.15, 'K': 0.77, 'L': 4.0, 'M': 2.4,
       'N': 6.7, 'O': 7.5, 'P': 1.9, 'Q': 0.095, 'R': 6.0, 'S': 6.3, 'T': 9.1, 'U': 2.8, 'V': 0.98, 'W': 2.4, 'X': 0.15, 'Y': 2.0, 'Z': 0.074}
CIPH = {caesar(k): v for k, v in ENG.items()}
def sc_count(px, t):
    ts = Ls(3)
    sky(px, t, (30, 18, 24), (52, 30, 30), stars=False)
    px.rect(0, 330, LW, 150, (60, 38, 28)); px.rect(0, 330, LW, 3, (90, 58, 40))     # wooden desk
    scroll(px, 121, 120, 200, 46)
    enc = caesar(MSG)
    torn = ease((t - ts - 3.6) / 0.4)
    if torn < 0.5: px.text(enc, 121, 110, 12, P["dred"], "c", shadow=False)
    else: px.text(MSG, 121, 110, 12, P["red"], "c", shadow=False)
    # falling letters into tally bars
    g = ease((t - ts - 0.2) / 2.2)
    base = 324; bw = 8
    for i, ch in enumerate(ALPH):
        x = 14 + i * 9
        h = int(CIPH[ch] * 8.5 * g)
        isH = ch == "H"
        flash = isH and t > ts + 2.2
        col = P["red"] if flash else P["teal"]
        px.rect(x, base - h, bw - 1, h, col, P["black"])
        if h > 3: px.rect(x + 1, base - h + 1, 2, h - 2, mixc(col, P["white"], 0.4))
        px.text(ch, x + 3, base + 3, 7, P["red"] if flash else P["sand"], "c", shadow=False)
    # letters raining
    rr = np.random.default_rng(int(t * 10))
    if g < 1:
        for k in range(10):
            x = rr.integers(10, 250); y = 150 + (t * 120 + k * 37) % 170
            px.text(ALPH[rr.integers(0, 26)], x, y, 9, P["parch"], "c")
    if t > ts + 2.2:
        hx = 14 + 7 * 9 + 3
        px.text("MOST COMMON: H", hx + 8, base - 80, 9, P["red"])
        px.text("(E shifted by 3)", hx + 8, base - 68, 8, P["white"])
    st = (t - ts - 3.6)
    if st > 0:
        sc = 1 + 1.5 * max(0, 0.18 - st) / 0.18
        y = 172; w = int(118 * sc); h = int(28 * sc)
        px.d.rectangle([121 - w // 2, y - h // 2, 121 + w // 2, y + h // 2], outline=P["red"], width=3)
        px.text("CRACKED", 121, y - 8 * sc, int(16 * sc), P["red"], "c", shadow=False)
    label_box(px, t, ts + 0.2, Le(3), "THE TRICK: COUNT", "AL-KINDI, 800s AD", P["red"])

def sc_key(px, t):
    ts = Ls(4)
    sky(px, t, P["night"], P["navy"])
    hills(px, t, 330)
    # road
    px.rect(0, 300, LW, 16, (52, 50, 64));
    for x in range(int(-t * 30) % 20 - 20, LW, 20): px.rect(x, 307, 10, 2, P["sand"])
    for (hx, lab, c) in ((34, "YOU", P["teal"]), (208, "BANK", P["gold"])):
        px.poly([(hx - 24, 260), (hx, 236), (hx + 24, 260)], P["dred"], P["black"])
        px.rect(hx - 20, 260, 40, 40, P["sand"], P["black"]); px.rect(hx - 5, 280, 10, 20, P["brown"])
        px.rect(hx - 15, 266, 8, 8, P["gold"]); px.light(hx - 11, 270, 20, P["gold"], 0.5)
        px.text(lab, hx, 318, 9, c, "c")
    # big glowing key appears first
    kp = ease((t - ts) / 0.5)
    walk = ease((t - ts - 1.6) / 2.0)
    grab = ease((t - ts - 3.2) / 0.4)
    mx = lerp(50, 150, walk); my = 300
    step = int(t * 8) % 2
    person(px, int(mx), my, P["teal"], step=step)
    kx, ky = mx + 8, my - 24
    if grab > 0:
        kx = lerp(kx, 178, grab); ky = lerp(ky, 262, grab)
    keysprite(px, kx, ky, 0.6 if walk > 0 else lerp(1.6, 0.6, walk) , P["gold"])
    if walk <= 0:
        keysprite(px, 100, 170, 1.8 * kp, P["gold"])
        px.text("A LONG SECRET KEY", 121, 205, 10, P["gold"], "c")
    tx = lerp(LW + 20, 172, ease((t - ts - 2.8) / 0.4))
    person(px, int(tx), 300, P["dred"], hood=True)
    if grab > 0:
        px.text("THIEF!", tx, 226 - 3 * math.sin(t * 20), 12, P["red"], "c")
    label_box(px, t, ts + 0.2, Le(4), "NEW PROBLEM", "THE KEY MUST TRAVEL", P["red"])

def sc_padlock(px, t):
    ts = Ls(5)
    sky(px, t, P["night"], (30, 50, 60))
    hills(px, t, 350, (20, 40, 46), (14, 28, 34))
    close = ease((t - ts - 2.0) / 0.25)
    padlock(px, 121, 214, 2.2, 1 - close)
    if 0 < t - ts - 2.0 < 0.5:
        r = int(40 + 200 * (t - ts - 2.0)); px.d.ellipse([121 - r, 214 - r, 121 + r, 214 + r], outline=P["green"], width=2)
    # many hands/arrows pressing down = anyone
    if t < ts + 2.3:
        for k in range(5):
            x = 40 + k * 40; y = 90 + 6 * math.sin(t * 6 + k)
            px.poly([(x - 5, y), (x + 5, y), (x, y + 10)], P["teal"]); px.text("ANYONE", x, y - 12, 7, P["teal"], "c", shadow=False) if k == 2 else None
    ko = ease((t - ts - 3.8) / 0.5)
    if ko > 0:
        keysprite(px, lerp(260, 150, ko), 300, 1.4, P["gold"])
        px.text("ONLY YOU", 121, 326, 11, P["gold"], "c")
    label_box(px, t, ts + 0.2, Le(5), "1977: THE PADLOCK", "RIVEST · SHAMIR · ADLEMAN", P["green"])

def sc_primes(px, t):
    ts = Ls(6)
    sky(px, t, P["night"], P["navy"])
    m = ease((t - ts - 0.6) / 0.6)
    for k, (num, c, sx) in enumerate((("61", P["amber"], -1), ("53", P["teal"], 1))):
        x = 121 + sx * lerp(70, 26, m); y = 160
        if m < 1:
            px.circ(x, y, 27, c, P["black"]); px.circ(x - 8, y - 9, 6, mixc(c, P["white"], 0.5)); px.text(num, x, y - 11, 19, P["black"], "c", shadow=False); px.light(x, y, 50, c, 0.8)
            px.text("PRIME", x, y + 32, 8, c, "c")
    if m >= 1:
        px.d.rounded_rectangle([61, 138, 181, 182], radius=6, fill=P["gold"], outline=P["black"])
        px.text("3233", 121, 145, 24, P["black"], "c", shadow=False); px.light(121, 160, 60, P["gold"], 0.9)
        px.text("61 x 53  .  INSTANT", 121, 184, 9, P["green"], "c")
    # backwards: counter of tries
    b = (t - ts - 3.2)
    if b > 0:
        px.text("3233 = ? x ?", 121, 214, 15, P["red"], "c")
        tries = int(min(10 ** (1 + b * 2.2), 9.99e15))
        px.rect(31, 238, 180, 50, P["black"], P["red"])
        px.text("GUESSES SO FAR", 40, 243, 8, P["grey"], shadow=False)
        px.text(f"{tries:,}", 202, 260, 16, P["red"], "r", shadow=False)
        px.text("a real lock needs far more", 121, 296, 9, P["white"], "c")
    label_box(px, t, ts + 0.2, Le(6), "THE TRICK: PRIME NUMBERS", None, P["gold"])

RDIG = "".join(str(d) for d in np.random.default_rng(617).integers(0, 10, 640))
def sc_617(px, t):
    ts = Ls(7)
    sky(px, t, P["night"], P["navy"])
    u = ease((t - ts) / 3.0)
    n = int(617 * u)
    # receipt ribbon unrolling down the screen
    px.rect(56, 80, 130, int(270 * u) + 10, P["parch"], P["brown"])
    for r in range(min(n // 15 + 1, 24)):
        row = RDIG[r * 15:(r + 1) * 15][:max(0, min(15, n - r * 15))]
        px.text(row, 121, 86 + r * 11, 9, P["dbrown"], "c", shadow=False)
    px.rect(46, 72, 150, 10, P["slate"], P["black"])
    px.rect(70, 340, 102, 22, P["black"], P["green"])
    px.text(f"{n} DIGITS", 121, 345, 11, P["green"], "c", shadow=False)
    label_box(px, t, ts + 0.2, Le(7), "TODAY'S LOCK NUMBER", "illustration · RSA-2048", P["green"])

def sc_quantum(px, t):
    ts = Ls(8)
    sky(px, t, (20, 10, 34), (60, 30, 80))
    d = ease((t - ts) / 1.4)
    fridge(px, 150, lerp(-160, 60, d), 1.25, t)
    cr = ease((t - ts - 3.0) / 1.0)
    padlock(px, 60, 300, 1.0, 0.0, mixc(P["gold"], P["red"], cr * 0.6), cr)
    if cr > 0:
        px.poly([(146, 228), (154, 228), (72, 290), (52, 290)], mixc(P["purple"], P["night"], 0.3))
        px.light(60, 296, 50, P["purple"], cr)
    w = ease((t - ts - 4.6) / 0.4)
    if w > 0:
        px.rect(150, 300, 108, 40, P["black"], P["red"])
        px.text("UNDER", 204, 304, 8, P["grey"], "c", shadow=False)
        px.text("1 WEEK*", 204, 316, 14, P["red"], "c", shadow=False)
    y = ease((t - ts - 7.6) / 0.3)
    if y > 0:
        px.text("*ESTIMATE, GOOGLE 2025", 204, 344, 7, P["white"], "c")
    if y > 0:
        px.rect(40, 70, 162, 18, P["black"], P["purple"]); px.text("DOESN'T EXIST... YET", 121, 73, 10, P["purple"], "c")
    else:
        label_box(px, t, ts + 0.2, Le(8), "NEW PROBLEM", "A QUANTUM COMPUTER", P["purple"])

def sc_newlock(px, t):
    ts = Ls(9)
    sky(px, t, P["night"], (16, 50, 40))
    # dot maze that forms a padlock
    g = ease((t - ts) / 1.6)
    rr = np.random.default_rng(9)
    for i in range(-11, 12):
        for j in range(-9, 13):
            x = 121 + i * 9 + (j % 2) * 4; y = 210 + j * 9
            inside = (abs(i) <= 5 and 0 <= j <= 7) or (3 <= abs(i) <= 4 and -6 <= j < 0) or (abs(i) <= 4 and j == -6)
            if rr.random() > g * 1.3 and not inside: continue
            c = P["green"] if inside else P["slate"]
            if inside and g < 1 and rr.random() > g: continue
            px.rect(x, y, 3 if inside else 2, 3 if inside else 2, c)
            if inside: px.light(x, y, 6, P["green"], 0.15)
    px.light(121, 240, 80, P["green"], 0.5 * g)
    # a purple beam enters the maze and gets lost
    b = (t - ts - 3.0)
    if b > 0:
        pts = [(0, 140)]; x, y = 20, 140
        rr2 = np.random.default_rng(11)
        for k in range(int(min(40, b * 18))):
            x += rr2.integers(-1, 2) * 9; y += rr2.integers(-1, 2) * 9
            x = min(max(x, 20), 250); y = min(max(y, 140), 330); pts.append((x, y))
        for a1, a2 in zip(pts[:-1], pts[1:]): px.line(a1, a2, P["purple"], 2)
        px.circ(x, y, 3, P["white"]); px.light(x, y, 20, P["purple"], 0.8)
        if b > 1.5: px.text("?", x, y - 14, 12, P["purple"], "c")
    label_box(px, t, ts + 0.2, Le(9), "THE FIX (2024)", "A PUZZLE WITH NO KNOWN SHORTCUT", P["green"])

def sc_end(px, t):
    ts = Ls(10)
    sky(px, t, P["night"], P["navy"])
    items = [("SCROLL", P["parch"]), ("COUNT", P["red"]), ("KEY", P["gold"]), ("PADLOCK", P["gold"]), ("QUANTUM", P["purple"]), ("NEW LOCK", P["green"])]
    for k, (lab, c) in enumerate(items):
        a = ease((t - ts - k * 0.18) / 0.25)
        if a <= 0: continue
        x = 30 + k * 36; y = 180
        px.rect(x - 13, y - 13, 26, 26, P["black"], c)
        if lab == "PADLOCK": padlock(px, x, y + 4, 0.35, glow=False)
        elif lab == "KEY": keysprite(px, x - 7, y, 0.4)
        elif lab == "SCROLL": scroll(px, x, y, 14, 12)
        elif lab == "QUANTUM": px.rect(x - 6, y - 4, 12, 8, P["purple"], P["black"])
        elif lab == "COUNT":
            for i, h in enumerate((4, 10, 6, 14)): px.rect(x - 8 + i * 4, y + 8 - h, 3, h, P["red"])
        else:
            for i in range(3):
                for j in range(3): px.rect(x - 6 + i * 5, y - 6 + j * 5, 2, 2, P["green"])
        if k < len(items) - 1: px.rect(x + 14, y, 8, 1, P["grey"])
    q = ease((t - ts - 1.4) / 0.3)
    if q > 0:
        r = 16 + 2 * math.sin(t * 6)
        px.circ(121, 262, r, P["black"], P["red"]); px.text("?", 121, 250, 20, P["red"], "c", shadow=False)
        px.text("NEXT LINK?", 121, 288, 10, P["red"], "c")
    px.text("WILL YOUR BANK", 121, 98, 14, P["white"], "c")
    px.text("CHANGE LOCKS IN TIME?", 121, 116, 12, P["gold"], "c")

SCENES = [sc_hook, sc_caesar, sc_count, sc_key, sc_padlock, sc_primes, sc_617, sc_quantum, sc_newlock, sc_end]

def frame(t):
    px = Px()
    # which line are we in?
    idx = 0
    for i in range(10):
        if t >= L[i] - 0.05: idx = i
    SCENES[idx](px, t)
    chain_hud(px, t)
    captions(px, t)
    series_tag(px)
    img = np.array(px.im)
    # transition: quick vertical wipe in the first 0.18 s of each line (not line 1)
    if idx > 0 and t - L[idx] < 0.18:
        k = int(LH * (1 - (t - L[idx]) / 0.18))
        img[:k] = (img[:k] * 0.25).astype(np.uint8)
    big = cv2.resize(img, (LW * UP, LH * UP), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    # bloom from light sources + bright pixels
    gl = np.zeros((LH, LW, 3), np.float32)
    yy, xx = np.mgrid[0:LH, 0:LW]
    for (x, y, r, c, k) in px.glow:
        if r < 1: continue
        gl += (np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (r * 0.5) ** 2)) * k)[..., None] * np.array(c, np.float32)[None, None] * 0.35
    br = np.clip(img.astype(np.float32) - 150, 0, 255)
    gl += cv2.GaussianBlur(br, (0, 0), 3) * 0.6
    gl = cv2.resize(gl, (LW * UP, LH * UP), interpolation=cv2.INTER_LINEAR)
    out = big + gl
    # subtle CRT/film feel: vignette + scanline every logical pixel row (very light)
    vy = np.linspace(-1, 1, LH * UP)[:, None]; vx = np.linspace(-1, 1, LW * UP)[None, :]
    vig = 1 - 0.28 * (vx ** 2 + vy ** 2)
    out *= vig[..., None]
    sl = np.ones(LH * UP, np.float32); sl[UP - 1::UP] = 0.9
    out *= sl[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "video"
    if mode == "info": print(json.dumps({"DUR": DUR, "L": [round(x, 2) for x in L]}))
    elif mode == "stills":
        for tt in [float(v) for v in sys.argv[3].split(",")]:
            cv2.imwrite(f"still_{tt:05.2f}.png", cv2.cvtColor(frame(tt), cv2.COLOR_RGB2BGR))
    else:
        nf = int(DUR * FPS); a0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; a1 = int(sys.argv[4]) if len(sys.argv) > 4 else nf
        for i in range(a0, min(a1, nf)): sys.stdout.buffer.write(frame(i / FPS).tobytes())
