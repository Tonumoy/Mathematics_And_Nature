# SOURCE CODE chain #1 — "How did your fridge punch a hole in the sky?"
from lib import *
from PIL import ImageDraw
import numpy as np, math

RNG = np.random.default_rng(42)
POLE_Y_FRAC = 0.309  # south pole screen offset (view_lat -72) as fraction of R

def blank():
    return starfield().copy()

def grid_bg(alpha=0.10, col=(40, 80, 120)):
    b = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    g = np.zeros((H, W), np.float32)
    g[::60, :] = 1; g[:, ::60] = 1
    b += g[..., None] * np.array(col, np.float32) * alpha
    return b

# ------------------------------------------------------------- fridge art
def draw_fridge(d, cx, by, h, era, a=255, xray=0.0, ss=SS):
    """cx centre x, by bottom y, h height (px, 1x)."""
    w = h * (0.52 if era != 3 else 0.62)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, by - h, by
    S = lambda *p: [v * ss for v in p]
    solid = int(a * (1 - xray))
    line = CYAN + (int(a * max(xray, 0.0)),)
    if era == 0:      # wooden icebox
        body = (122, 78, 44); dark = (82, 50, 28)
        if solid: d.rounded_rectangle(S(x0, y0, x1, y1), radius=10 * ss, fill=body + (solid,))
        for yy in (y0 + h * 0.06, y0 + h * 0.48):
            if solid:
                d.rounded_rectangle(S(x0 + w * 0.08, yy, x1 - w * 0.08, yy + h * 0.40), radius=6 * ss, outline=dark + (solid,), width=int(5 * ss))
                d.rectangle(S(x1 - w * 0.2, yy + h * 0.17, x1 - w * 0.12, yy + h * 0.23), fill=(212, 170, 80, solid))
        for k in range(1, 9):
            if solid: d.line(S(x0 + 6, y0 + k * h / 9, x1 - 6, y0 + k * h / 9), fill=dark + (int(solid * 0.25),), width=int(2 * ss))
    elif era == 1:    # 1920s monitor-top
        cab_top = y0 + h * 0.30
        if solid:
            d.rounded_rectangle(S(x0, cab_top, x1, y1 - h * 0.08), radius=12 * ss, fill=(222, 222, 214, solid))
            for lx in (x0 + w * 0.12, x1 - w * 0.12):
                d.rectangle(S(lx - 6, y1 - h * 0.08, lx + 6, y1), fill=(60, 60, 64, solid))
            d.rounded_rectangle(S(x0 + w * 0.06, cab_top + h * 0.04, x1 - w * 0.06, y1 - h * 0.12), radius=8 * ss, outline=(170, 170, 160, solid), width=int(4 * ss))
            d.rectangle(S(x1 - w * 0.2, cab_top + h * 0.2, x1 - w * 0.13, cab_top + h * 0.3), fill=(190, 190, 195, solid))
            # cylindrical coil "monitor" on top
            d.rounded_rectangle(S(cx - w * 0.36, y0 + h * 0.06, cx + w * 0.36, cab_top), radius=30 * ss, fill=(200, 200, 196, solid))
            for k in range(6):
                yy = y0 + h * 0.08 + k * h * 0.035
                d.line(S(cx - w * 0.34, yy, cx + w * 0.34, yy), fill=(140, 140, 136, solid), width=int(4 * ss))
    elif era == 2:    # rounded mid-century Freon fridge
        if solid:
            d.rounded_rectangle(S(x0, y0, x1, y1), radius=w * 0.22 * ss, fill=(206, 232, 222, solid))
            d.line(S(x0 + 8, y0 + h * 0.3, x1 - 8, y0 + h * 0.3), fill=(150, 175, 168, solid), width=int(4 * ss))
            d.rounded_rectangle(S(x1 - w * 0.2, y0 + h * 0.36, x1 - w * 0.1, y0 + h * 0.52), radius=6 * ss, fill=(230, 230, 236, solid), outline=(120, 120, 130, solid), width=int(2 * ss))
            d.rounded_rectangle(S(x0 + w * 0.06, y0 + h * 0.03, x0 + w * 0.2, y1 - h * 0.05), radius=8 * ss, fill=(240, 252, 248, int(solid * 0.6)))
    else:             # modern stainless
        if solid:
            d.rounded_rectangle(S(x0, y0, x1, y1), radius=14 * ss, fill=(150, 156, 166, solid))
            d.line(S(cx, y0 + 6, cx, y1 - 6), fill=(90, 96, 106, solid), width=int(4 * ss))
            for hx in (cx - w * 0.08, cx + w * 0.08):
                d.rounded_rectangle(S(hx - 5, y0 + h * 0.25, hx + 5, y0 + h * 0.6), radius=5 * ss, fill=(215, 220, 228, solid))
            d.rounded_rectangle(S(x0 + w * 0.05, y0 + h * 0.03, x0 + w * 0.15, y1 - h * 0.04), radius=6 * ss, fill=(200, 205, 214, int(solid * 0.5)))
    if xray > 0:
        d.rounded_rectangle(S(x0, y0, x1, y1), radius=14 * ss, outline=line, width=int(3 * ss))
    return x0, y0, x1, y1

# ------------------------------------------------------------- A / H / I / L earth
def earth_frame(t, D=1150, cy=880, hole=1.0, lon=30, shell=1.0):
    b = blank()
    e = earth(D, math.radians(-72), lon, hole, t, healthy_shell=shell)
    b = paste_float(b, e, (W - D) // 2, cy - D // 2)
    pole = (540, cy + POLE_Y_FRAC * D / 2 * 0.8)
    return b, pole, D / 2 * 0.8

def scene_hook(t, T):
    D = int(lerp(1120, 1260, ease(t / T)))
    b, pole, R = earth_frame(t, D, 860, 0.85 + 0.15 * math.sin(t * 3), 28 + 5 * t)
    L = new_layer(); d = ImageDraw.Draw(L)
    rr = R * 0.36
    ring(d, pole, rr + 6 * math.sin(t * 5), CYAN, 235, 4)
    pill(d, (pole[0], pole[1] - rr - 40), "the ozone hole", VIOLET, 30)
    caption(d, "your fridge did this", y=1430, size=82, hl=("fridge",))
    return over(b, down(L))

# ------------------------------------------------------------- B bacteria
class Bugs:
    def __init__(self):
        rng = np.random.default_rng(5)
        gens = [[(540.0, 760.0, rng.uniform(0, math.pi))]]
        for g in range(10):
            nxt = []
            sp = 26 * 0.86 ** g + 9
            for (x, y, a) in gens[-1]:
                for s in (-1, 1):
                    jx = rng.normal(0, sp * 0.8); jy = rng.normal(0, sp * 0.8)
                    nxt.append((x + s * math.cos(a) * sp + jx, y + s * math.sin(a) * sp + jy, a + rng.normal(0, 0.9)))
            gens.append(nxt)
        self.gens = gens
BUGS = None

def lens_mask(cx, cy, r):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dd = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip((r - dd) / 3, 0, 1)
LENS = None
FOOD = None

def scene_bacteria(t, T, t2, t3):
    """t2: start of 'bacteria double' line, t3: start of 'fix cold' line."""
    global BUGS, LENS, FOOD
    if BUGS is None:
        BUGS = Bugs(); LENS = lens_mask(540, 780, 430)
        f = fbm(H, W, 21, 6, 6)
        f2 = fbm(H, W, 22, 6, 24)
        FOOD = (np.array([150, 120, 80], np.float32) * (0.55 + 0.6 * f[..., None]) - np.array([70, 70, 60], np.float32) * np.clip((f2[..., None] - 0.55) * 4, 0, 1))
    cold = ease(ramp(t, t3 + 0.2, t3 + 1.8))
    b = grid_bg(0.06)
    food = FOOD * (1 - 0.6 * cold) + np.array([60, 110, 150], np.float32) * 0.6 * cold * (0.5 + FOOD[..., :1] / 255)
    zoom_in = ease(ramp(t, 0, 1.2))
    b = b * (1 - LENS[..., None]) + food * LENS[..., None] * (0.4 + 0.6 * zoom_in)
    L = new_layer(); d = ImageDraw.Draw(L)
    # growth: generation index driven by time in "bacteria" line
    gt = max(0.0, t - t2)
    g_f = min(gt / 0.48, 10.0)          # 10 doublings shown visually, then fill
    if cold > 0: g_f = min(g_f, (t3 - t2) / 0.48)
    g = int(g_f); frac = g_f - g
    def bug(x, y, a, ln, col, al):
        dx, dy = math.cos(a) * ln / 2, math.sin(a) * ln / 2
        d.line([(x - dx) * SS, (y - dy) * SS, (x + dx) * SS, (y + dy) * SS], fill=col + (al,), width=int(11 * SS))
        for sx in (-1, 1):
            d.ellipse([(x + sx * dx - 5.5) * SS, (y + sx * dy - 5.5) * SS, (x + sx * dx + 5.5) * SS, (y + sx * dy + 5.5) * SS], fill=col + (al,))
    if t >= t2 - 0.6:
        gg = BUGS.gens[min(g, 10)]
        nxt = BUGS.gens[min(g + 1, 10)]
        col = lerpc((140, 230, 90), (150, 220, 255), cold)
        wig = 1 - cold
        for i, (x, y, a) in enumerate(gg):
            if (x - 540) ** 2 + (y - 780) ** 2 > 415 ** 2: continue
            if frac > 0.6 and g < 10:
                for c in (nxt[2 * i], nxt[2 * i + 1]):
                    k = ease((frac - 0.6) / 0.4)
                    bug(lerp(x, c[0], k) + wig * 2 * math.sin(t * 9 + i), lerp(y, c[1], k), c[2], 22, col, 235)
            else:
                ln = 22 + 14 * ease(frac / 0.6) * (g < 10)
                bug(x + wig * 2 * math.sin(t * 9 + i), y + wig * 2 * math.cos(t * 7 + i), a + wig * 0.1 * math.sin(t * 3 + i), ln, col, 235)
    # lens rim
    ring(d, (540, 780), 430, (180, 190, 210), 200, 6)
    # counter
    if t >= t2:
        hours = 10.0 * ease(ramp(t, t2 + 0.3, t3 - 0.4))
        n = 2 ** (hours * 3)
        draw_text(d, (540, 1270), f"{int(round(n)):,}", 72, fill=(RED if cold == 0 else CYAN) + (255,), stroke=3)
        draw_text(d, (540, 1330), f"bacteria after {hours:4.1f} h", 30, fill=WHITE + (230,), path=FONT_SB)
        chip(d, (540, 300), "N = 2^(t / 20 min)", AMBER, 36, int(255 * ramp(t, t2, t2 + 0.5)))
    if cold > 0:
        # frost crystals from lens rim
        rng = np.random.default_rng(9)
        for k in range(46):
            ang = k / 46 * 2 * math.pi + rng.normal(0, 0.05)
            ln = 140 * cold * rng.uniform(0.5, 1.2)
            x0, y0 = 540 + 430 * math.cos(ang), 780 + 430 * math.sin(ang)
            x1, y1 = 540 + (430 - ln) * math.cos(ang), 780 + (430 - ln) * math.sin(ang)
            d.line([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=(200, 240, 255, 200), width=int(3 * SS))
            for j in (0.4, 0.7):
                xm, ym = lerp(x0, x1, j), lerp(y0, y1, j)
                for s in (-1, 1):
                    a2 = ang + math.pi + s * 0.6
                    d.line([xm * SS, ym * SS, (xm + 30 * cold * math.cos(a2)) * SS, (ym + 30 * cold * math.sin(a2)) * SS], fill=(200, 240, 255, 170), width=int(2 * SS))
        pill(d, (540, 1180), f"{int(lerp(37, 4, cold))} °C  ·  growth almost stops", CYAN, 32, int(255 * cold))
    # captions
    if t < t2:
        caption(d, "it started with rotting food", y=1450, size=64, hl=("rotting",))
    elif t < t3:
        caption(d, "1 becomes 1 billion in 10 hours", y=1450, size=62, hl=("billion",))
    else:
        caption(d, "fix: cold", y=1450, size=78, hl=("cold",))
    nodes = ramp(t, t2 - 0.8, t2 - 0.2) + ramp(t, t3 + 0.4, t3 + 1.0)
    chain_hud(d, nodes, t)
    L = down(L)
    b = add_light(b, glow(L, 16), 0.5)
    return over(b, L)

# ------------------------------------------------------------- C/D machine + toxic
FR_CX, FR_BY, FR_H = 540, 1290, 900

def loop_path():
    """refrigerant loop inside era-1 fridge (closed polyline)."""
    x0, x1 = FR_CX - 210, FR_CX + 210
    y_top, y_bot = FR_BY - FR_H + 150, FR_BY - 90
    pts = []
    # evaporator serpentine near top (inside)
    yy = y_top + 60
    for k in range(4):
        xa, xb = (x0 + 40, x1 - 40) if k % 2 == 0 else (x1 - 40, x0 + 40)
        pts += [(xa, yy), (xb, yy)]; yy += 42
    pts += [(x0 + 40, yy), (x0 + 40, y_bot - 40)]            # down left side to compressor
    pts += [(FR_CX - 60, y_bot - 40), (FR_CX + 60, y_bot - 40)]  # compressor
    pts += [(x1 + 70, y_bot - 40)]                           # out the back (condenser)
    yy = y_bot - 40
    for k in range(6):
        xa, xb = (x1 + 70, x1 + 150) if k % 2 == 0 else (x1 + 150, x1 + 70)
        yy -= 70; pts += [(xa, yy), (xb, yy)]
    pts += [(x1 + 70, yy - 40), (x1 - 40, yy - 40), (x1 - 40, y_top + 60)]   # back in through valve to start
    pts += [(x0 + 40, y_top + 60)]
    return pts
PATH = None

def path_sample(pts, s):
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    tot = sum(seg); s = (s % 1.0) * tot
    for i, l in enumerate(seg):
        if s <= l:
            k = s / l if l else 0
            return pts[i][0] + (pts[i + 1][0] - pts[i][0]) * k, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * k, i, len(seg)
        s -= l
    return pts[-1][0], pts[-1][1], len(seg) - 1, len(seg)

def scene_machine(t, T, t_x, toxic_t=None):
    """t_x: when x-ray starts. toxic_t: start of toxic leak (scene D)."""
    global PATH
    if PATH is None: PATH = loop_path()
    b = grid_bg(0.07)
    L = new_layer(); d = ImageDraw.Draw(L)
    xr = ease(ramp(t, t_x, t_x + 0.8))
    era = 0 if t < t_x else 1
    if era == 0:
        # icebox with melting ice
        x0, y0, x1, y1 = draw_fridge(d, FR_CX, FR_BY, FR_H * 0.85, 0, 255)
        melt = ramp(t, 0.2, t_x - 0.2)
        ih = 160 * (1 - 0.75 * melt)
        d.rounded_rectangle([(FR_CX - 120) * SS, (y0 + 70 + (160 - ih)) * SS, (FR_CX + 120) * SS, (y0 + 230) * SS], radius=14 * SS, fill=(200, 236, 255, 235), outline=(255, 255, 255, 255), width=int(3 * SS))
        pill(d, (FR_CX, y0 - 40), "block of lake ice", CYAN, 30)
        # drips + puddle
        for k in range(5):
            ph = (t * 1.3 + k * 0.21) % 1.0
            yy = lerp(y1 + 4, y1 + 60, ph)
            d.ellipse([(FR_CX - 160 + k * 80 - 6) * SS, (yy - 9) * SS, (FR_CX - 160 + k * 80 + 6) * SS, (yy + 9) * SS], fill=(120, 200, 255, int(220 * (1 - ph))))
        pw = 140 + 260 * melt
        d.ellipse([(FR_CX - pw) * SS, (y1 + 40) * SS, (FR_CX + pw) * SS, (y1 + 80) * SS], fill=(70, 150, 230, 150))
        caption(d, "but ice melts", y=1450, size=78, hl=("melts",))
    else:
        x0, y0, x1, y1 = draw_fridge(d, FR_CX, FR_BY, FR_H, 1, 255, xray=xr)
        # loop pipe
        al = int(255 * xr)
        pts = [(p[0] * SS, p[1] * SS) for p in PATH]
        d.line(pts, fill=(70, 90, 110, al), width=int(12 * SS), joint="curve")
        # refrigerant particles: colour by segment state
        n = 70
        for k in range(n):
            x, y, i, ns = path_sample(PATH, k / n + t * 0.09)
            f = i / ns
            if f < 0.45: c = CYAN
            elif f < 0.52: c = AMBER
            elif f < 0.82: c = lerpc(AMBER, RED, (f - 0.52) / 0.3)
            else: c = lerpc(RED, CYAN, (f - 0.82) / 0.18)
            r = 7
            d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], fill=c + (al,))
        # compressor box
        cy = FR_BY - 130
        pul = 1 + 0.05 * math.sin(t * 20)
        d.rounded_rectangle([(FR_CX - 80 * pul) * SS, (cy - 30) * SS, (FR_CX + 80 * pul) * SS, (cy + 50) * SS], radius=16 * SS, fill=(60, 40, 10, al), outline=AMBER + (al,), width=int(4 * SS))
        # heat in from food (into evaporator) and heat out from condenser
        for k in range(5):
            ph = (t * 0.8 + k * 0.2) % 1
            xx = FR_CX - 160 + k * 80; yy = lerp(y0 + 470, y0 + 300, ph)
            d.line([xx * SS, yy * SS, xx * SS, (yy - 30) * SS], fill=(255, 120, 80, int(al * (1 - ph))), width=int(5 * SS))
        for k in range(6):
            ph = (t * 0.9 + k * 0.17) % 1
            yy = FR_BY - 180 - k * 75
            xx = lerp(x1 + 170, x1 + 260, ph)
            for s in range(3):
                d.arc([(xx + s * 14 - 10) * SS, (yy - 22) * SS, (xx + s * 14 + 10) * SS, (yy + 22) * SS], -60, 60, fill=RED + (int(al * (1 - ph)),), width=int(4 * SS))
        if toxic_t is None or t < toxic_t:
            pill(d, (FR_CX, y0 + 120), "gas boils here → steals heat", CYAN, 28, al)
            pill(d, (FR_CX, FR_BY - 40), "compressor", AMBER, 28, al)
            pill(d, (x1 + 40, y0 + 230), "heat out", RED, 28, al, anchor="mm")
            chip(d, (540, 300), "heat taken = mass × L (latent heat)", AMBER, 30, int(255 * ramp(t, t_x + 1, t_x + 1.6)))
            caption(d, "a gas boils & carries heat out", y=1450, size=62, hl=("boils",))
    nodes = 2 + ramp(t, 0.3, 0.9) + ramp(t, t_x + 1.0, t_x + 1.6)
    if toxic_t is not None and t >= toxic_t:
        tt = t - toxic_t
        # crack + leak
        cx_, cy_ = FR_CX - 170, FR_BY - 330
        fl = 0.5 + 0.5 * math.sin(tt * 18)
        d.ellipse([(cx_ - 26) * SS, (cy_ - 26) * SS, (cx_ + 26) * SS, (cy_ + 26) * SS], outline=RED + (int(255 * fl),), width=int(5 * SS))
        rng = np.random.default_rng(3)
        CL = new_layer(); dc = ImageDraw.Draw(CL)
        for k in range(160):
            st = rng.uniform(0, 3.0)
            age = tt - st * 0.6
            if age <= 0: continue
            ang = rng.uniform(-math.pi, math.pi); sp = rng.uniform(60, 190)
            px = cx_ + math.cos(ang) * sp * age ** 0.7 + 40 * math.sin(age * 2 + k)
            py = cy_ + math.sin(ang) * sp * age ** 0.7 * 0.7 - 30 * age
            r = 10 + 26 * min(age, 1.5)
            puff = Image.new("RGBA", (int(2 * r * SS) + 2, int(2 * r * SS) + 2), (0, 0, 0, 0))
            ImageDraw.Draw(puff).ellipse([0, 0, 2 * r * SS, 2 * r * SS], fill=(175, 255, 95, int(120 * max(0, 1 - age / 4))))
            CL.alpha_composite(puff, (max(int((px - r) * SS), 0), max(int((py - r) * SS), 0)))
        L.alpha_composite(CL)
        pill(d, (300, 520), "NH₃  ammonia", (150, 240, 90), 30, int(255 * ramp(tt, 0.5, 1)))
        pill(d, (760, 600), "SO₂", (150, 240, 90), 30, int(255 * ramp(tt, 0.9, 1.4)))
        draw_text(d, (540, 330), "TOXIC", 92, fill=RED + (int(255 * ramp(tt, 0.2, 0.5) * (0.75 + 0.25 * fl)),), stroke=4)
        caption(d, "leaks killed families in their sleep", y=1450, size=60, hl=("killed",))
        nodes = 4 + ramp(tt, 0.2, 0.8)
    chain_hud(d, nodes, t)
    L = down(L)
    b = add_light(b, glow(L, 20), 0.6)
    return over(b, L)

# ------------------------------------------------------------- E freon
CFC = [("C", (0, 0, 0)), ("Cl", (1, 1, 1)), ("Cl", (-1, -1, 1)), ("F", (1, -1, -1)), ("F", (-1, 1, -1))]
CFC_B = [(0, 1), (0, 2), (0, 3), (0, 4)]

def scene_freon(t, T, t_candle):
    b = grid_bg(0.06)
    L = new_layer(); d = ImageDraw.Draw(L)
    k = ease_out(ramp(t, 0, 0.8))
    draw_text(d, (540, 360), "1928", 120, fill=AMBER + (int(255 * k),), stroke=4)
    draw_text(d, (540, 445), "Thomas Midgley invents Freon (CFC-12)", 32, fill=WHITE + (int(230 * k),), path=FONT_SB)
    sc = lerp(60, 130, ease_out(ramp(t, 0.2, 1.4))) * (1 - 0.35 * ease(ramp(t, t_candle - 0.5, t_candle + 0.3)))
    my = lerp(780, 700, ease(ramp(t, t_candle - 0.5, t_candle + 0.3)))
    P = molecule(L, CFC, CFC_B, 540, my, sc, 0.5 + t * 0.6, t * 0.9)
    lab = {"Cl": "chlorine", "F": "fluorine", "C": "carbon"}
    if t < t_candle - 0.3:
        seen = set()
        for el, x, y, z in P:
            if el in seen: continue
            seen.add(el)
            pill(d, (x, y - sc * 0.65), lab[el], ATOM[el][0] if el != "C" else (200, 200, 210), 26, int(255 * ramp(t, 1.0, 1.4)))
    hide = 1 - ramp(t, t_candle - 0.6, t_candle - 0.2)
    pill(d, (300, 1110), "not toxic ✓", GREEN, 34, int(255 * ramp(t, 2.2, 2.6) * hide))
    pill(d, (780, 1110), "won't burn ✓", GREEN, 34, int(255 * ramp(t, 2.8, 3.2) * hide))
    if t >= t_candle - 0.3:
        tc = t - t_candle
        cx, cy = 740, 1000
        d.rounded_rectangle([(cx - 26) * SS, cy * SS, (cx + 26) * SS, (cy + 170) * SS], radius=8 * SS, fill=(235, 225, 200, 255))
        d.line([cx * SS, cy * SS, cx * SS, (cy - 18) * SS], fill=(40, 40, 40, 255), width=int(4 * SS))
        out = ease(ramp(tc, 1.4, 2.2))
        fl = (1 - out) * (1 + 0.08 * math.sin(t * 25))
        if fl > 0.02:
            for r, c in ((40, (255, 140, 40)), (28, (255, 200, 80)), (14, (255, 245, 210))):
                rr = r * fl
                d.ellipse([(cx - rr * 0.6) * SS, (cy - 20 - rr * 2.2) * SS, (cx + rr * 0.6) * SS, (cy - 14) * SS], fill=c + (230,))
        # breath of freon gas from left
        for j in range(40):
            ph = (tc * 0.9 - j * 0.03)
            if ph < 0 or ph > 1.6: continue
            x = lerp(260, cx + 40, min(ph, 1.2)); y = cy - 40 + 26 * math.sin(j * 1.7 + ph * 6)
            d.ellipse([(x - 6) * SS, (y - 6) * SS, (x + 6) * SS, (y + 6) * SS], fill=GREEN + (int(160 * max(0, 1 - ph / 1.6)),))
        if out > 0:
            for j in range(10):
                yy = cy - 30 - j * 26 * out; xx = cx + 18 * math.sin(j * 0.9 + t * 2)
                d.ellipse([(xx - 8) * SS, (yy - 8) * SS, (xx + 8) * SS, (yy + 8) * SS], fill=(180, 180, 190, int(90 * out * (1 - j / 10))))
        caption(d, "he breathed it in & blew out a candle", y=1450, size=58, hl=("breathed",))
    else:
        caption(d, "fix: Freon. safe… right?", y=1450, size=66, hl=("freon.",))
    chain_hud(d, 5 + ramp(t, 0.4, 1.0), t)
    L = down(L)
    b = add_light(b, glow(L, 18), 0.45)
    return over(b, L)

# ------------------------------------------------------------- F/J atmosphere column
def km_y(km): return 1460 - km * 26.5

@functools.lru_cache(None)
def sky_bg(warm=0):
    y = np.arange(H, dtype=np.float32)[:, None, None]
    top = np.array([4, 6, 16], np.float32); mid = np.array([14, 30, 70], np.float32); low = np.array([40, 90, 150], np.float32)
    k = np.clip((y - 200) / 1300, 0, 1)
    col = np.where(k < 0.6, top + (mid - top) * (k / 0.6), mid + (low - mid) * ((k - 0.6) / 0.4))
    img = np.repeat(col, W, axis=1)
    img = np.maximum(img * 0.6, starfield() * (1 - np.clip((y - 300) / 500, 0, 1)) * 0.8 + img * 0.4)
    if warm:
        img = img + np.array([60, 10, -20], np.float32) * np.clip((y - 900) / 600, 0, 1)
    # ground
    g = (np.arange(H)[:, None] >= km_y(0)).astype(np.float32)[..., None]
    gx = np.arange(W)[None, :]
    sky_line = km_y(0) - (np.abs(np.sin(gx * 0.05)) * 40 + (np.sin(gx * 0.013) > 0.3) * 60) * (np.abs(gx - 540) > 120)
    bld = (np.arange(H)[:, None] >= sky_line)[..., None].astype(np.float32)
    img = img * (1 - bld) + np.array([10, 14, 24], np.float32) * bld
    img = img * (1 - g) + np.array([16, 22, 30], np.float32) * g
    return img

O3_PTS = None
def o3_points():
    global O3_PTS
    if O3_PTS is None:
        rng = np.random.default_rng(8)
        n = 700
        km = np.clip(rng.normal(24, 5.5, n), 14, 36)
        x = rng.uniform(150, 1040, n)
        O3_PTS = (x, km, rng.uniform(0, 6.28, n))
    return O3_PTS

def column_axes(d, a=255):
    for km in (0, 10, 20, 30, 40):
        y = km_y(km)
        d.line([70 * SS, y * SS, 95 * SS, y * SS], fill=(160, 170, 190, a), width=int(3 * SS))
        draw_text(d, (110, y), f"{km} km", 24, fill=(170, 180, 200, a), path=FONT_SB, anchor="lm")
    d.line([82 * SS, km_y(40) * SS, 82 * SS, km_y(0) * SS], fill=(160, 170, 190, int(a * 0.7)), width=int(2 * SS))

def scene_rise(t, T, t_uv):
    b = sky_bg(0).copy()
    L = new_layer(); d = ImageDraw.Draw(L)
    column_axes(d)
    xs, kms, ph = o3_points()
    eaten = ease(ramp(t, t_uv + 0.5, T + 4))
    for i in range(0, len(xs)):
        if i / len(xs) < eaten * 0.4: continue
        x = xs[i] + 6 * math.sin(t + ph[i]); y = km_y(kms[i])
        d.ellipse([(x - 4) * SS, (y - 4) * SS, (x + 4) * SS, (y + 4) * SS], fill=(90, 180, 255, 170))
    pill(d, (880, km_y(36.5)), "ozone layer", CYAN, 28)
    draw_fridge(d, 540, km_y(0) + 2, 150, 2, 255)
    # CFC parcels rising (years compress)
    rng = np.random.default_rng(12)
    N = 60
    hits = []
    for k in range(N):
        st = rng.uniform(-1.5, t_uv + 1.0)
        age = t - st
        if age < 0: continue
        km = min(age * 5.5, 22 + rng.uniform(-3, 8))
        x = 540 + rng.normal(0, 30) + 260 * math.sin(age * 0.8 + k) * min(age / 3, 1)
        y = km_y(km)
        top = km >= 18
        if top and t > t_uv:
            hits.append((x, y, k))
        col = GREEN
        d.ellipse([(x - 7) * SS, (y - 7) * SS, (x + 7) * SS, (y + 7) * SS], fill=col + (235,))
    year = int(lerp(1930, 1975, ease(ramp(t, 0.3, t_uv))))
    draw_text(d, (860, 350), str(year), 64, fill=AMBER + (255,), stroke=3)
    pill(d, (540, km_y(6)), "CFC drifts up for years", GREEN, 30, int(255 * ramp(t, 0.8, 1.2)))
    if t > t_uv:
        tu = t - t_uv
        for j in range(7):
            ph2 = (tu * 0.8 + j / 7) % 1
            x = 220 + j * 120; y0 = lerp(140, km_y(26), ph2)
            pts = [((x + 10 * math.sin((y0 - s * 6) * 0.15)) * SS, (y0 - s * 6) * SS) for s in range(12)]
            d.line(pts, fill=VIOLET + (220,), width=int(4 * SS))
        pill(d, (300, km_y(40)), "UV sunlight", VIOLET, 28)
        for (x, y, k) in hits[:14]:
            pk = (tu * 1.5 + k * 0.13) % 1
            r = 12 + 30 * pk
            d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS], outline=(255, 255, 255, int(220 * (1 - pk))), width=int(3 * SS))
            cx = x + 40 * pk; cy = y - 30 * pk
            d.ellipse([(cx - 9) * SS, (cy - 9) * SS, (cx + 9) * SS, (cy + 9) * SS], fill=(120, 255, 140, 255))
        caption(d, "sunlight rips a chlorine atom off", y=1530, size=56, hl=("chlorine",))
    else:
        caption(d, "problem: Freon is too stable", y=1530, size=58, hl=("stable",))
    chain_hud(d, 6 + ramp(t, t_uv, t_uv + 0.6), t)
    L = down(L)
    b = add_light(b, glow(L, 14), 0.6)
    return over(b, L)

# ------------------------------------------------------------- G catalytic cycle (x-ray)
O3_MOL = [("O", (0, 0, 0)), ("O", (1.1, 0.7, 0)), ("O", (-1.1, 0.7, 0))]

def scene_cycle(t, T):
    b = grid_bg(0.10, (60, 40, 120))
    L = new_layer(); d = ImageDraw.Draw(L)
    rng = np.random.default_rng(4)
    targets = [(rng.uniform(170, 910), rng.uniform(420, 1180)) for _ in range(40)]
    cyc = 0.62   # seconds per destroyed ozone (visual)
    n_done = int(t / cyc); f = (t % cyc) / cyc
    # chlorine path: from target i to target i+1
    for i, (x, y) in enumerate(targets):
        if i < n_done:   # destroyed: show O2 drifting
            age = t - (i + 1) * cyc
            for s in (-1, 1):
                ox = x + s * 30 + s * 40 * age; oy = y - 20 * age
                al = int(200 * max(0, 1 - age / 2.5))
                if al > 0:
                    paste_ball(L, ox - 10, oy, 13, (200, 205, 215)); paste_ball(L, ox + 10, oy, 13, (200, 205, 215))
        elif i > n_done:
            molecule(L, O3_MOL, [(0, 1), (0, 2)], x, y, 30, 0.3 * math.sin(t + i), t * 0.7 + i)
    cx0, cy0 = targets[max(n_done - 1, 0)] if n_done > 0 else (540, 1250)
    cx1, cy1 = targets[min(n_done, len(targets) - 1)]
    k = ease(f)
    clx, cly = lerp(cx0, cx1, k), lerp(cy0, cy1, k)
    if n_done < len(targets):
        molecule(L, O3_MOL, [(0, 1), (0, 2)], cx1, cy1, 24 * (1 + 0.3 * k), 0.2, t)
    # Cl with trail
    paste_ball(L, clx, cly, 34, (80, 240, 110))
    ring(d, (clx, cly), 50 + 6 * math.sin(t * 10), CYAN, 230, 4)
    pill(d, (clx, cly - 80), "Cl", GREEN, 30)
    # counter (log-accelerating to 100,000)
    cnt = int(min(100000, 10 ** (ramp(t, 0.3, T - 0.6) * 5)))
    draw_text(d, (540, 1290), f"{cnt:,}", 88, fill=RED + (255,), stroke=4)
    draw_text(d, (540, 1360), "ozone molecules wrecked by ONE chlorine", 30, fill=WHITE + (230,), path=FONT_SB)
    chip(d, (540, 300), "Cl + O₃ → ClO + O₂", AMBER, 36, 255)
    chip(d, (540, 372), "ClO + O → Cl + O₂", AMBER, 36, int(255 * ramp(t, 0.6, 1.0)))
    caption(d, "Cl comes back every time", y=1490, size=56, hl=("back",))
    L = down(L)
    b = add_light(b, glow(L, 16), 0.55)
    return over(b, L)

# ------------------------------------------------------------- H earth 1985
def scene_1985(t, T):
    k = ease(ramp(t, 0, T * 0.8))
    b, pole, R = earth_frame(t + 3, 1150, 860, k, 40 + 5 * t)
    L = new_layer(); d = ImageDraw.Draw(L)
    yr = int(lerp(1979, 1985, k))
    draw_text(d, (540, 370), str(yr), 110, fill=VIOLET + (255,), stroke=4)
    ring(d, pole, R * 0.36 * (0.4 + 0.6 * k), CYAN, int(230 * k), 4)
    pill(d, (540, 1320), "Farman, Gardiner & Shanklin · Nature, 1985", WHITE, 26, int(255 * ramp(t, 0.8, 1.2)))
    caption(d, "a hole over Antarctica", y=1450, size=72, hl=("hole",))
    chain_hud(d, 7, t)
    return over(b, down(L))

# ------------------------------------------------------------- I Montreal
def scene_montreal(t, T):
    b, pole, R = earth_frame(t + 6, 860, 890, 1.0 - 0.3 * ramp(t, 2, T), 50 + 4 * t)
    L = new_layer(); d = ImageDraw.Draw(L)
    n = int(198 * ease(ramp(t, 0.3, 0.5 * T)))
    for i in range(198):
        a = -math.pi / 2 + i / 198 * 2 * math.pi
        x, y = 540 + 420 * math.cos(a), 890 + 420 * math.sin(a)
        c = GREEN if i < n else (60, 70, 90)
        d.ellipse([(x - 6) * SS, (y - 6) * SS, (x + 6) * SS, (y + 6) * SS], fill=c + (255,))
    draw_text(d, (540, 345), f"{n} / 198", 84, fill=GREEN + (255,), stroke=4)
    draw_text(d, (540, 415), "countries sign the Montreal Protocol · 1987", 28, fill=WHITE + (230,), path=FONT_SB)
    tb = 0.68 * T
    if t > tb:
        s = ease_out(ramp(t, tb, tb + 0.4))
        sz = lerp(140, 82, s)
        d.rounded_rectangle([(540 - 270) * SS, (1360 - 60) * SS, (540 + 270) * SS, (1360 + 60) * SS], radius=18 * SS, outline=RED + (255,), width=int(7 * SS))
        draw_text(d, (540, 1362), "FREON BANNED", sz * 0.62, fill=RED + (255,))
    caption(d, "fix: every country on Earth signs", y=1490, size=60, hl=("every",))
    chain_hud(d, 7 + ramp(t, 0.5, 1.1), t)
    L = down(L)
    b = add_light(b, glow(L, 14), 0.5)
    return over(b, L)

# ------------------------------------------------------------- J heat trap
def scene_heat(t, T, t_bar):
    b = sky_bg(1).copy()
    warm = ease(ramp(t, 1.0, T))
    b = b + np.array([50, 8, 0], np.float32) * warm * np.clip((np.arange(H)[:, None, None] - 900) / 600, 0, 1)
    L = new_layer(); d = ImageDraw.Draw(L)
    column_axes(d)
    xs, kms, ph = o3_points()
    for i in range(0, len(xs), 2):
        x = xs[i]; y = km_y(kms[i])
        d.ellipse([(x - 4) * SS, (y - 4) * SS, (x + 4) * SS, (y + 4) * SS], fill=(90, 180, 255, 150))
    pill(d, (860, km_y(36.5)), "ozone safe ✓", GREEN, 28)
    draw_fridge(d, 540, km_y(0) + 2, 150, 2, 255)
    rng = np.random.default_rng(31)
    hfc = [(rng.uniform(160, 1000), rng.uniform(1, 11)) for _ in range(36)]
    for (x, km) in hfc:
        y = km_y(km) + 6 * math.sin(t * 2 + x)
        d.ellipse([(x - 8) * SS, (y - 8) * SS, (x + 8) * SS, (y + 8) * SS], fill=AMBER + (240,))
    pill(d, (330, km_y(12.5)), "HFC-134a", AMBER, 28)
    # IR rays: up from ground, bounce off HFC back down
    for j in range(12):
        ph2 = (t * 0.7 + j / 12) % 1
        x = 150 + j * 75
        up = ph2 < 0.5
        y0 = km_y(0) - 10
        yb = km_y(6 + (j % 4) * 1.5)
        y = lerp(y0, yb, ph2 * 2) if up else lerp(yb, y0, (ph2 - 0.5) * 2)
        pts = [((x + 9 * math.sin((y + s * 8) * 0.2)) * SS, (y + s * 8 * (1 if up else -1)) * SS) for s in range(10)]
        d.line(pts, fill=RED + (220,), width=int(4 * SS))
    # bar chart
    if t > t_bar:
        k = ease_out(ramp(t, t_bar, t_bar + 1.0))
        bx, by = 720, 1130
        d.rounded_rectangle([(bx - 200) * SS, (by - 560) * SS, (bx + 180) * SS, (by + 70) * SS], radius=16 * SS, fill=(6, 8, 16, 210))
        draw_text(d, (bx - 10, by - 520), "heat trapped vs CO₂", 26, fill=WHITE + (240,), path=FONT_SB)
        d.rectangle([(bx - 130) * SS, (by - 3) * SS, (bx - 50) * SS, by * SS], fill=(200, 200, 210, 255))
        h = 420 * k
        d.rectangle([(bx + 20) * SS, (by - h) * SS, (bx + 110) * SS, by * SS], fill=RED + (255,))
        draw_text(d, (bx - 90, by + 35), "CO₂ = 1", 24, fill=WHITE + (230,), path=FONT_SB)
        draw_text(d, (bx + 65, by + 35), "HFC", 24, fill=WHITE + (230,), path=FONT_SB)
        draw_text(d, (bx + 65, by - h - 34), f"{int(1430 * k):,}×", 40, fill=RED + (255,), stroke=2)
    if t < t_bar:
        caption(d, "new gas HFC: spares the ozone", y=1530, size=56, hl=("hfc:",))
    else:
        caption(d, "but it traps 1,430× more heat than CO2", y=1530, size=54, hl=("1,430×",))
    chain_hud(d, 8 + ramp(t, 0.5, 1.1), t)
    L = down(L)
    b = add_light(b, glow(L, 14), 0.5)
    return over(b, L)

# ------------------------------------------------------------- K isobutane + dial (one-number twist)
ISO = [("C", (0, 0, 0)), ("C", (1.3, 0.8, 0)), ("C", (-1.3, 0.8, 0)), ("C", (0, -1.5, 0.3)),
       ("H", (0.3, 0.2, 1.0)),
       ("H", (2.1, 0.2, 0.4)), ("H", (1.4, 1.8, 0.3)), ("H", (1.5, 0.8, -1.0)),
       ("H", (-2.1, 0.2, 0.4)), ("H", (-1.4, 1.8, 0.3)), ("H", (-1.5, 0.8, -1.0)),
       ("H", (0.9, -2.0, 0.6)), ("H", (-0.9, -2.0, 0.6)), ("H", (0, -1.8, -0.7))]
ISO_B = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7), (2, 8), (2, 9), (2, 10), (3, 11), (3, 12), (3, 13)]

def scene_iso(t, T, t_dial):
    b = grid_bg(0.06, (40, 120, 80))
    L = new_layer(); d = ImageDraw.Draw(L)
    draw_text(d, (540, 350), "Kigali · 2016", 64, fill=GREEN + (int(255 * ramp(t, 0, 0.5)),), stroke=3)
    molecule(L, ISO, ISO_B, 300, 680, 80, 0.4 + t * 0.5, t * 0.8)
    pill(d, (300, 930), "isobutane (R-600a)", GREEN, 28)
    draw_fridge(d, 790, 980, 500, 3, 255)
    pill(d, (790, 1025), "most new fridges", WHITE, 26)
    # dial
    k = ease(ramp(t, t_dial, t_dial + 1.6))
    val = 1430 * (1 - k) + 3 * k
    cx, cy, r = 540, 1270, 200
    d.arc([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], 180, 360, fill=(60, 70, 90, 255), width=int(18 * SS))
    frac = math.log10(max(val, 1)) / math.log10(1430)
    a = math.pi + frac * math.pi
    col = lerpc(GREEN, RED, frac)
    d.arc([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], 180, 180 + frac * 180, fill=col + (255,), width=int(18 * SS))
    d.line([cx * SS, cy * SS, (cx + (r - 30) * math.cos(a)) * SS, (cy + (r - 30) * math.sin(a)) * SS], fill=WHITE + (255,), width=int(6 * SS))
    draw_text(d, (cx, cy - 40), f"{int(round(val)):,}", 64, fill=col + (255,), stroke=3)
    draw_text(d, (cx, cy + 28), "heat score (GWP)", 26, fill=WHITE + (220,), path=FONT_SB)
    if t < t_dial:
        caption(d, "fix: Kigali 2016 → isobutane", y=1500, size=60, hl=("isobutane",))
    else:
        caption(d, "its heat score? just 3", y=1500, size=66, hl=("3",))
    chain_hud(d, 9 + ramp(t, 0.4, 1.0), t)
    L = down(L)
    b = add_light(b, glow(L, 14), 0.5)
    return over(b, L)

# ------------------------------------------------------------- L ending
def scene_end(t, T, t_next):
    k = ease(ramp(t, 0.2, t_next - 1.0))
    b, pole, R = earth_frame(t + 9, 1000, 820, 1 - k, 60 + 5 * t, shell=1.0)
    L = new_layer(); d = ImageDraw.Draw(L)
    yr = int(lerp(2000, 2066, k))
    draw_text(d, (540, 360), str(yr), 100, fill=(CYAN if k > 0.98 else VIOLET) + (255,), stroke=4)
    if k > 0.98:
        pill(d, (540, 440), "ozone back to 1980 levels (forecast)", CYAN, 26)
    if t < t_next * 0.45:
        caption(d, "the hole heals by ~2066", y=1430, size=64, hl=("2066",))
    elif t < t_next:
        caption(d, "every fix made a new problem", y=1430, size=62, hl=("new",))
    else:
        caption(d, "next chain: how did we beat the dark?", y=1430, size=60, hl=("dark?",))
    chain_hud(d, 10, t, active_label=False)
    return over(b, down(L))
