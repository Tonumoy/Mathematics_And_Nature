# Coded Cosmos — "WHO'S THE LEADER?" v2 (owner 2 Nov: real fireflies, arrival intro, slow pacing, ElevenLabs narration)
# Format F5 sync · look L6 -> photoreal night mangrove plate (ElevenLabs seedream-5-pro, our own generation) + simulated fireflies
# Real solver: pulse-coupled integrate-and-fire (Mirollo & Strogatz 1990), event-driven, sim.py. Music: music_cinematic (D minor).
import numpy as np, subprocess, json, os, sys, wave
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter
import sim, music_cinematic as mc

# ================= globals =================
W, H = 1080, 1920
FPS, BPM = 30, 75
BEAT_F = FPS*60/BPM                      # 24 frames per beat (0.8 s)
BEATS = 70
NF = int(BEATS*BEAT_F)                   # 1680 frames = 56 s
OUT_DIR = "out"
N, R_VIEW, SPREAD = 300, 120, 0.02
E_ON, E_LOW, SEED = 0.07, 0.01, 1
SLACK = 35.0
BG = (6, 10, 22)
WHITE, YELLOW = (255, 250, 240), (255, 214, 64)
FIRE = np.array([190, 255, 80], np.float32)/255          # ~560 nm yellow-green
CYAN, AMBER, RED, GREEN = (110, 225, 255), (255, 170, 60), (255, 80, 80), (90, 240, 140)
SAFE = (40, 160, 930, 1590); CAP_CX, CAP_MAXW = 490, 820
PLATE_Y, PLATE_S = 330, 0.8
FP_SYM = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FP_XB, FP_B = "fonts/Poppins-ExtraBold.ttf", "fonts/Poppins-Bold.ttf"
TITLE, HOOK, SERIES, ASK = "WHO'S THE LEADER?", "300 fireflies. no leader.", "weird nature #1", "would YOU lead or follow?"
# narration: (segment index from vo.wav silence split, place at display second)
VO_PLACE = [0.6, 2.4, 4.8, 6.0, 7.4, 11.0, 12.8, 16.0, 17.6, 19.2, 21.0, 24.0, 25.4, 27.2, 28.2,
            32.0, 34.0, 35.0, 37.8, 40.6, 42.6, 44.4, 48.0, 52.6]
CAPTIONS = [  # seconds
    (0.6, 3.9, "one firefly."),
    (4.8, 10.5, "then another... *and another.*"),
    (11.0, 12.7, "each one blinks *on its own.*"),
    (12.8, 15.8, "no pattern. *no plan.*"),
    (16.0, 17.5, "*but watch.*"),
    (17.6, 20.9, "no leader.\n*nobody keeping time.*"),
    (21.0, 23.9, "just *one tiny rule:*"),
    (24.0, 29.6, "see a flash?\n*hurry yours a little.*"),
    (32.0, 37.5, "300 fireflies. *one rhythm.*"),
    (37.8, 42.3, "nudge *OFF*...\nit falls apart."),
    (42.6, 47.6, "nudge *ON*...\nin step again."),
    (48.0, 52.5, "first used for your heart's\n*pacemaker cells.*"),
    (52.6, 56.0, "nature is *running on code.*"),
]

# ================= world plate =================
def make_world():
    im = Image.open("tree.png").convert("RGB")
    im = im.resize((int(im.width*PLATE_S), int(im.height*PLATE_S)), Image.LANCZOS)
    a = np.array(im, np.float32); x0 = (a.shape[1]-W)//2; a = a[:, x0:x0+W]; ph = a.shape[0]
    world = np.zeros((H, W, 3), np.float32)
    world[PLATE_Y:PLATE_Y+ph] = a
    top = gaussian_filter(a[:40].mean(0), (40, 0))                               # sky above: extend + darken upward
    for y in range(PLATE_Y): world[y] = top*(0.55 + 0.45*(y/PLATE_Y))
    fl = a[::-1][:H-PLATE_Y-ph]                                                  # water below: mirrored, darker
    k = np.linspace(0.85, 0.45, len(fl))[:, None, None]
    world[PLATE_Y+ph:] = fl*k
    m = np.clip((48 - world[..., 2])/22, 0, 1)                                   # tree = dark vs blue sky
    yy = np.arange(H)[:, None]
    crown = ((yy > PLATE_Y+110) & (yy < PLATE_Y+640)).astype(np.float32)
    return world, gaussian_filter(m*crown, 1.5)
WORLD, TREE = make_world()
WL = PLATE_Y + 745                                                               # waterline (world y)

def perches(n, rng):
    ys, xs = np.where(TREE > 0.92)
    keep = (ys < PLATE_Y + 600) & (xs > 70) & (xs < W-70)
    ys, xs = ys[keep], xs[keep]
    idx = rng.choice(len(xs), 4*n, replace=False)
    pts = []
    for i in idx:                                                                 # spread out: min distance 14 px
        p = np.array([xs[i], ys[i]], float)
        if all(np.hypot(*(p-q)) > 14 for q in pts): pts.append(p)
        if len(pts) == n: break
    return np.array(pts)
rng = np.random.default_rng(7)
P = perches(N, rng)
K = (np.linalg.norm(P[:, None]-P[None], axis=2) < R_VIEW) & ~np.eye(N, dtype=bool)
Tn = 1 + SPREAD*np.random.default_rng(3).standard_normal(N)

# ================= simulation + timeline =================
def big(t, fl, t0): return [x for x, i in zip(t, fl) if x > t0 and len(i) >= 0.9*N]
t1, _, f1 = sim.run(Tn, K, lambda s: E_ON, 120, seed=SEED)
BF = big(t1, f1, 0); S0 = BF[9] + 0.5; S1 = S0 + SLACK
def eps_of(s): return E_LOW if S0 <= s < S1 else E_ON
EV_T, EV_PH, FL = sim.run(Tn, K, eps_of, S1 + 90, seed=SEED)
RS = big(EV_T, FL, S1)
KNOTS = [(0, 0.0), (20, 10.0), (38, BF[0])] + [(38+k, BF[k]) for k in range(1, 10)] + [(53, S1), (59, RS[0])] + [(59+k, RS[k]) for k in range(1, 12)]
KB = np.array([k[0] for k in KNOTS], float); KT = np.array([k[1] for k in KNOTS])
def sim_t(b): return float(np.interp(b, KB, KT))
def speed(b): return (sim_t(b+0.05) - sim_t(b-0.05))/0.1
LAST = np.full((len(EV_T), N), -9.0, np.float32); cur = np.full(N, -9.0, np.float32)
for k, (tt, idx) in enumerate(zip(EV_T, FL)):
    cur[idx] = tt; LAST[k] = cur
def state(t):
    k = np.searchsorted(EV_T, t, 'right')-1
    return sim.phase_at(EV_T, EV_PH, Tn, t), LAST[k]

# ================= arrivals (display seconds) =================
C0 = np.array([540., 760.])                                                       # crown centre
first = int(np.argmin(np.hypot(P[:, 0]-700, P[:, 1]-700)))                       # the first firefly: upper right of crown
order = [first] + [int(i) for i in rng.permutation([j for j in range(N) if j != first])]
LAND = np.zeros(N); DUR = np.zeros(N); START = np.zeros((N, 2)); CTRL = np.zeros((N, 2))
for r, i in enumerate(order):
    if r == 0: LAND[i], DUR[i] = 3.3, 3.4
    elif r == 1: LAND[i], DUR[i] = 5.4, 2.2
    elif r == 2: LAND[i], DUR[i] = 6.4, 2.0
    else: LAND[i], DUR[i] = 7.0 + 6.0*((r-3)/(N-3))**0.6, rng.uniform(1.5, 2.3)
    ang = rng.uniform(0, 2*np.pi) if r else -0.35
    dist = rng.uniform(750, 1000) if r else 105
    START[i] = P[i] + dist*np.array([np.cos(ang), np.sin(ang)*0.8])
    mid = (START[i] + P[i])/2; nrm = np.array([-np.sin(ang), np.cos(ang)])
    CTRL[i] = mid + nrm*rng.uniform(-260, 260)*(1 if r else 0.3)
HEAD = rng.uniform(0, 2*np.pi, N)                                                 # perched heading
def fly_pos(i, s):
    """world position + heading at display second s, or None if not yet in the air"""
    s0 = LAND[i]-DUR[i]
    if s < s0: return None
    if s >= LAND[i]: return P[i], HEAD[i], False
    u = (s-s0)/DUR[i]; u = 1-(1-u)**2.2
    p = (1-u)**2*START[i] + 2*u*(1-u)*CTRL[i] + u*u*P[i]
    d = 2*(1-u)*(CTRL[i]-START[i]) + 2*u*(P[i]-CTRL[i])
    p = p + 6*np.array([np.sin(s*7+i), np.cos(s*5+i)])
    return p, np.arctan2(d[1], d[0]), True

# ================= camera =================
def ease(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)
def camera(s):
    u = ease((s-3.6)/8.4)
    z = 3.2 + (1.12-3.2)*u**0.8
    c = P[first]*(1-u) + np.array([W/2, H/2])*u + np.array([8*np.sin(s/5), 5*np.sin(s/7)])
    return z, c
def to_screen(p, z, c): return (p - c)*z + np.array([W/2, H/2])

# ================= text helpers =================
F_TITLE = ImageFont.truetype(FP_XB, 66); F_HOOK = ImageFont.truetype(FP_B, 38)
CAP_FONTS = {z: ImageFont.truetype(FP_XB, z) for z in (58, 54, 50, 46)}
F_READ = ImageFont.truetype(FP_SYM, 30); F_WM = ImageFont.truetype(FP_B, 28); F_TAG = ImageFont.truetype(FP_B, 26)
F_PILL = ImageFont.truetype(FP_XB, 44); F_EQ = ImageFont.truetype(FP_SYM, 88); F_LEG = ImageFont.truetype(FP_SYM, 28); F_CHIP = ImageFont.truetype(FP_SYM, 30); F_ST = ImageFont.truetype(FP_B, 32)
def box_text(d, cx, y, text, font, fg, bgc, pad=(28, 10)):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x0 = cx - (r-l)/2 - pad[0]; x1 = cx + (r-l)/2 + pad[0]
    d.rounded_rectangle((x0, y, x1, y + (b-t) + 2*pad[1]), 14, fill=bgc)
    d.text((cx - (r-l)/2 - l, y + pad[1] - t), text, font=font, fill=fg)
    return y + (b-t) + 2*pad[1]
def rich_line(d, cx, y, line, font, alpha, rise):
    segs, hi = [], False
    for part in line.split("*"):
        if part: segs.append((part, hi))
        hi = not hi
    widths = [d.textlength(s, font=font) for s, _ in segs]; x = cx - sum(widths)/2
    for (s, h), w in zip(segs, widths):
        d.text((x, y - rise), s, font=font, fill=(YELLOW if h else WHITE) + (alpha,), stroke_width=7, stroke_fill=BG + (alpha,))
        x += w
def caption(d, s):
    for (s0, s1, txt) in CAPTIONS:
        if s0 <= s < s1:
            t_in = (s-s0)/0.4; t_out = (s1-s)/0.3
            a = int(255*min(1, t_in, t_out)); rise = int(14*(1-min(1, t_in)))
            lines = txt.split("\n")
            for size in (58, 54, 50, 46):
                font = CAP_FONTS[size]
                if max(d.textlength(ln.replace("*", ""), font=font) for ln in lines) <= CAP_MAXW: break
            y0 = 1440 + (0 if len(lines) == 2 else 36)
            for i, ln in enumerate(lines): rich_line(d, CAP_CX, y0 + i*76, ln, font, a, rise)
def lerp(c0, c1, u): return tuple(int(a + (b-a)*u) for a, b in zip(c0, c1))

# ================= firefly drawing (screen space) =================
def ell(cx, cy, a, b, ang, n=14):
    t = np.linspace(0, 2*np.pi, n, endpoint=False); ca, sa = np.cos(ang), np.sin(ang)
    x, y = a*np.cos(t), b*np.sin(t)
    return list(zip(cx + x*ca - y*sa, cy + x*sa + y*ca))
def draw_bug(d, x, y, ang, z, br, flying, f, i):
    L = 8.5*z; ca, sa = np.cos(ang), np.sin(ang)
    hx, hy = x + ca*L*0.75, y + sa*L*0.75                       # head end
    tx, ty = x - ca*L*0.7, y - sa*L*0.7                         # lantern end
    if flying:                                                  # wing blur, flicker
        fl = 0.5 + 0.5*np.sin(f*2.7 + i)
        for side in (-1, 1):
            wa = ang + side*(1.2 + 0.5*fl)
            d.polygon(ell(x + np.cos(wa)*L*0.6, y + np.sin(wa)*L*0.6, L*0.75, L*0.32, wa), fill=(170, 185, 200, int(60 + 50*fl)))
    if not flying:                                              # 6 legs gripping the leaf
        for k in (-0.5, 0, 0.5):
            for side in (-1, 1):
                bx, by = x + ca*L*k*0.8, y + sa*L*k*0.8
                la = ang + side*(1.9 - 0.4*k)
                d.line((bx, by, bx + np.cos(la)*L*0.55, by + np.sin(la)*L*0.55), fill=(20, 16, 12, 230), width=max(1, int(z*0.6)))
    for side in (-1, 1):                                        # antennae
        aa = ang + side*0.45
        d.line((hx, hy, hx + np.cos(aa)*L*0.7, hy + np.sin(aa)*L*0.7), fill=(30, 24, 18, 220), width=max(1, int(z*0.5)))
    lit = int(40*min(1, br))
    d.polygon(ell(x, y, L, L*0.36, ang), fill=(22+lit, 18+lit, 14+lit//2, 255))                 # body (elytra)
    d.polygon(ell(hx, hy, L*0.33, L*0.3, ang), fill=(150+lit, 80+lit//2, 35, 255))               # orange pronotum
    g = min(1.0, 0.25 + br)
    d.polygon(ell(tx, ty, L*0.38, L*0.3, ang), fill=(int(120+120*g), int(160+95*g), int(40+60*g), int(150+105*g)))   # lantern
    return tx, ty

# ================= render =================
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.5*(((xx-W/2)/W)**2*1.4 + ((yy-H*0.48)/H)**2*1.6)*2).clip(0.35, 1)[..., None]
GRAIN = [np.random.default_rng(k).normal(0, 1.2, (H, W, 1)).astype(np.float32) for k in range(6)]
LOW = 4; HL, WLW = H//LOW, W//LOW
BOKEH = [(130, 1500, 3, 44), (930, 560, 41, 32), (80, 600, 120, 26)]
RIPPLE = list(range(0, N, 25))                                  # a few fireflies whose flashes we draw as 'seen' rings

def frame(f):
    s = f/FPS; b = s*BPM/60; t = sim_t(b); sp = max(0.5, speed(b))
    ph, last = state(t)
    tau = 0.15*sp
    age = t - last; br = np.where((age >= 0) & (age < 6*tau), np.exp(-age/tau), 0).astype(np.float32)
    r = sim.order(ph)
    z, c = camera(s)
    # positions (world) for all fireflies at this second
    pos = np.full((N, 2), np.nan); head = np.zeros(N); fly = np.zeros(N, bool)
    for i in range(N):
        q = fly_pos(i, s)
        if q is not None: pos[i], head[i], fly[i] = q
    vis = ~np.isnan(pos[:, 0])
    glow = br*0.85 + np.where(fly, 0.18, 0.0) + 0.45*(np.arange(N) == first)*(s < 4.2)                                   # lanterns glow a little while flying
    # ---- world light (low res) ----
    lm = np.zeros((HL, WLW), np.float32)
    on = vis & (glow > 0.01)
    gx = np.clip(pos[on, 0]/LOW, 0, WLW-1).astype(int); gy = np.clip(pos[on, 1]/LOW, 0, HL-1).astype(int)
    np.add.at(lm, (gy, gx), glow[on])
    wide = gaussian_filter(lm, 12)*300; near = gaussian_filter(lm, 3)*30
    upf = lambda a_: np.array(Image.fromarray(a_.astype(np.float32)).resize((W, H), Image.BILINEAR))[..., None]
    wide, near = upf(wide), upf(near)
    g_leaf = 1-np.exp(-wide*0.7); g_air = 1-np.exp(-wide*0.2); g_near = 1-np.exp(-near*0.8)
    arr = WORLD + TREE[..., None]*(g_leaf*FIRE*60 + g_near*FIRE*90) + g_air*FIRE*40
    rows = np.clip(2*WL - np.arange(WL, H), 0, H-1)
    ref = (g_air[rows]*0.9 + g_near[rows]*0.6)[:, :, 0]
    shift = (6*np.sin(np.arange(WL, H)/8 + s*3)).astype(int)
    ref = np.stack([np.roll(ref[i], shift[i]) for i in range(H-WL)])[..., None]
    arr[WL:] += ref*FIRE*110*np.linspace(0.8, 0.2, H-WL)[:, None, None]
    # ---- camera (crop + scale) ----
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    cw, chh = W/z, H/z; x0, y0 = c[0]-cw/2, c[1]-chh/2
    img = img.transform((W, H), Image.EXTENT, (x0, y0, x0+cw, y0+chh), Image.BICUBIC)
    if z > 1.3: img = img.filter(ImageFilter.GaussianBlur(float((z-1.3)*1.6)))       # shallow depth of field in the macro opening
    # ---- fireflies (screen space) ----
    sp_pos = to_screen(pos, z, c)
    bug = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(bug)
    tails = []
    for i in np.where(vis)[0]:
        x, y = sp_pos[i]
        if -40 < x < W+40 and -40 < y < H+40:
            tails.append((i,) + draw_bug(d, x, y, head[i], z, glow[i], fly[i], f, i))
    img = img.convert("RGBA"); img.alpha_composite(bug)
    a = np.array(img.convert("RGB"), np.float32)
    # lantern bloom in screen space
    bm = np.zeros((HL, WLW), np.float32)
    for (i, tx, ty) in tails:
        if 0 <= tx < W and 0 <= ty < H: bm[int(ty)//LOW, int(tx)//LOW] += glow[i]
    bl1 = upf(gaussian_filter(bm, 1.2*max(1, z*0.8))*14); bl2 = upf(gaussian_filter(bm, 4*z)*60)
    a += (1-np.exp(-bl1*0.9))*FIRE*210 + (1-np.exp(-bl2*0.35))*FIRE*70
    for (bx, by, i, rr) in BOKEH:                                 # near-camera fireflies (same sim), out of focus
        if s < 14: continue
        g = glow[i]*0.55 + 0.04; y0b, y1b, x0b, x1b = max(0, by-2*rr), min(H, by+2*rr), max(0, bx-2*rr), min(W, bx+2*rr)
        d2 = ((yy[y0b:y1b, x0b:x1b]-by)**2 + (xx[y0b:y1b, x0b:x1b]-bx)**2)/rr**2
        a[y0b:y1b, x0b:x1b] += (np.clip(1.2-d2, 0, 1)**0.6*g)[..., None]*FIRE*230*min(1, (s-14)/2)
    a = a*VIG + GRAIN[f % 6]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # 'seen' rings: a flash reaches its neighbours (only while the rule is explained)
    if 21.0 <= s < 30.0:
        for i in RIPPLE:
            if br[i] > 0.05:
                rr = R_VIEW*z*(1-br[i]**0.5); x, y = sp_pos[i]
                d.ellipse((x-rr, y-rr, x+rr, y+rr), outline=CYAN + (int(160*br[i]),), width=3)
    # x-ray (21 s): every firefly's clock x as a ring, amber arc = how full it is
    if 21.0 <= s < 31.0:
        ys = (s-21.0)/0.8*H; fade = 1-ease((s-30.0)/1.0)
        for i in range(N):
            if not vis[i] or fly[i]: continue
            x, y = sp_pos[i]
            if y > ys: continue
            rr = 9
            d.ellipse((x-rr, y-rr, x+rr, y+rr), outline=CYAN + (int(150*fade),), width=2)
            d.arc((x-rr, y-rr, x+rr, y+rr), -90, -90 + 360*ph[i], fill=AMBER + (int(235*fade),), width=3)
        if s < 21.8:
            d.rectangle((0, ys-3, W, ys+3), fill=CYAN + (230,)); d.rectangle((0, ys-20, W, ys-3), fill=CYAN + (50,))
    # equation card (the one rule), 23.4-30 s, then a chip that stays and updates live
    eps_now = eps_of(t)
    if 23.4 <= s < 30.0:
        a_ = int(255*min(ease((s-23.4)/0.5), 1-ease((s-29.6)/0.4)))
        EQ = "x → x + ε"; tw = d.textlength(EQ, font=F_EQ); x0e = 485 - tw/2; ye = 400
        hw = max(tw, d.textlength("x = a firefly's inner clock (flash at 1)", font=F_LEG))/2 + 34
        d.rounded_rectangle((485-hw, ye-14, 485+hw, ye+200), 24, fill=(6, 12, 30, int(a_*0.85)), outline=AMBER + (a_//2,), width=2)
        xs = x0e
        for ch in EQ:
            col = CYAN if ch == "x" else (AMBER if ch == "ε" else WHITE)
            d.text((xs, ye), ch, font=F_EQ, fill=col + (a_,)); xs += d.textlength(ch, font=F_EQ)
        for k, (txt, col) in enumerate((("x = a firefly's inner clock (flash at 1)", CYAN), (f"ε = the tiny nudge ({eps_now:.2f})", AMBER))):
            d.text((485 - d.textlength(txt, font=F_LEG)/2, ye + 112 + k*40), txt, font=F_LEG, fill=col + (a_,))
    if s >= 30.0:
        txt = f"see a flash:  x → x + ε   (ε = {eps_now:.2f})"; tw = d.textlength(txt, font=F_CHIP)
        cc = RED if eps_now < E_ON else AMBER
        d.rounded_rectangle((60, 1290, 60+tw+28, 1336), 12, fill=(6, 12, 30, 200), outline=cc + (150,), width=2)
        d.text((74, 1294), txt, font=F_CHIP, fill=cc)
    # readouts
    if s >= 16:
        rc = lerp(RED, GREEN, np.clip((r-0.3)/0.6, 0, 1))
        d.text((60, 1350), f"in step: {int(round(r*100))}%", font=F_READ, fill=rc)
        tag = "real simulation" + (" · sped up" if speed(b) > 1.3 else "")
        d.text((910-d.textlength(tag, font=F_TAG), 1352), tag, font=F_TAG, fill=(170, 180, 210))
    if s >= 30.4:
        off = eps_of(t) < E_ON; back = b >= 53
        txt = "nudge: OFF" if off else "nudge: ON"; col = RED if off else (GREEN if back else AMBER)
        tw = d.textlength(txt, font=F_PILL); big = (37.6 <= s < 41.0) or (42.4 <= s < 45.6)
        if big:
            d.rounded_rectangle((485-tw/2-28, 420, 485+tw/2+28, 492), 36, fill=(8, 10, 24, 215), outline=col + (255,), width=3)
            d.text((485-tw/2, 424), txt, font=F_PILL, fill=col + (255,))
        else:
            pass
    # top
    d.text((60, 176), "@cosmoscoded", font=F_WM, fill=(205, 210, 240))
    d.text((910-d.textlength(SERIES, font=F_WM), 176), SERIES, font=F_WM, fill=(150, 170, 210))
    ta = int(255*ease((s-0.3)/0.6))
    y = box_text(d, 485, 214, TITLE, F_TITLE, (10, 10, 24, ta), YELLOW + (ta,))
    if b < 59: box_text(d, 485, y+10, HOOK, F_HOOK, WHITE + (ta,), (24, 40, 60, int(ta*0.9)))
    else: box_text(d, 485, y+10, ASK, F_HOOK, (10, 10, 24), CYAN)
    caption(d, s)
    img.alpha_composite(ov)
    return img.convert("RGB"), r

def render():
    os.makedirs(OUT_DIR, exist_ok=True)
    TEST = [int(x) for x in os.environ.get('TEST', '').split(',') if x]
    if TEST:
        for f in TEST: frame(f)[0].save(f"{OUT_DIR}/t{f:04d}.png")
        return None
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", f"{OUT_DIR}/video.mp4"], stdin=subprocess.PIPE)
    for f in range(NF):
        im, r = frame(f); ff.stdin.write(np.array(im).tobytes())
        if f % 120 == 0: print("frame", f, flush=True)
    ff.stdin.close(); ff.wait()
    mev = []
    for f in range(NF):
        ta_, tb_ = sim_t(f/BEAT_F), sim_t((f+1)/BEAT_F)
        k0, k1 = np.searchsorted(EV_T, ta_, 'right'), np.searchsorted(EV_T, tb_, 'right')
        idx = np.concatenate([FL[k] for k in range(k0, k1)]) if k1 > k0 else []
        landed = [i for i in idx if LAND[i] <= f/FPS + 0.5]
        if len(landed): mev.append((f/FPS, [float(np.clip(P[landed, 0].mean()/540-1, -1, 1))], int(len(landed))))
    json.dump({"events": mev, "BF0": BF[0], "S0": S0, "S1": S1, "RS0": RS[0]}, open(f"{OUT_DIR}/events.json", "w"))
    return mev

SECTIONS = dict(A_END=20, DROP=38, SHUT0=47, SHUT1=53, OPEN1=53, FADE0=70, SPOT0=20, D0=47, D_END=53, BELL_ALL=False)
if __name__ == "__main__":
    print("perches", len(P), "BF0", BF[0], "S0", S0, "S1", S1, "RS0", RS[0])
    ev = render() if "skiprender" not in sys.argv[1:] else json.load(open(f"{OUT_DIR}/events.json"))["events"]
    if ev is None: sys.exit()
    mc.compose(ev, BEATS, BPM, SECTIONS, f"{OUT_DIR}/music_raw.wav"); print("done")
