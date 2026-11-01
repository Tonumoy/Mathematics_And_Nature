# Coded Cosmos — "WHO GETS KICKED OUT?" Pythagorean three-body problem (Burrau 1913)
# Format F8 Open Problems Lab (+F2 prediction) · Look L3 deep night sky · dark cinematic D minor 120 BPM
# Real solver: Newtonian 3-body, scipy DOP853 rtol=atol=1e-13 (sim.py). Copy run: star 3 moved 1e-4 in x.
import numpy as np, subprocess, json, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sim import run, r0, m
import music_cinematic as mc

# ================= globals =================
W, H        = 1080, 1920
FPS, BPM    = 30, 120
BEAT_F      = FPS*60/BPM                 # 15 frames per beat
BEATS       = 80
NF          = int(BEATS*BEAT_F)          # 1200 frames = 40 s
SEED        = 3
OUT_DIR     = "out"
EPS         = 1e-4                       # nudge of star 3 (x) in the copy
T_END       = 78.0
SUB         = 6                          # trail sub-samples per frame
TRAIL_T     = 5.0                        # trail length in sim time units

BG          = (5, 6, 16)                 # one BG value for the whole Short
WHITE       = (255, 250, 240)
YELLOW      = (255, 214, 64)
CYAN        = (110, 225, 255)
STAR_COL    = [(255, 120, 70), (255, 205, 90), (150, 190, 255)]   # mass 3, 4, 5
NAMES       = ["3", "4", "5"]

SAFE        = (40, 160, 930, 1590)       # Shorts safe zone (see STYLE 31 Oct 18:09)
CAP_CX      = 490
CAP_MAXW    = 800
CX, CY      = 485, 860                   # centre of mass on screen
R_PIX       = 440                        # radius (px) the camera fits the dance into
R_MIN, R_MAX = 3.2, 14.0

# beat -> sim time (piecewise linear). 0-3: rewind from t=38 (frame-1 motion), 3-14 hold at rest
KNOTS = [(0, 38.0), (3, 0.0), (14, 0.0), (40, 40.0), (50, 47.0), (56, 54.5), (62, 60.5), (68, 64.0), (80, 75.0)]
EJECT = {"orig": (0, 60.08), "copy": (1, 54.20)}   # (star index, last close pass) measured in chk.py

FP_XB, FP_B, FP_SB, FP_M = "fonts/Poppins-ExtraBold.ttf", "fonts/Poppins-Bold.ttf", "fonts/Poppins-SemiBold.ttf", "fonts/JetBrainsMono.ttf"
TITLE  = "WHO GETS KICKED OUT?"
HOOK   = "3 stars. only 2 stay."
SERIES = "weird physics #4"
ASK    = "which star would YOU bet on?"

CAPTIONS = [
    (0,  6,  "okay so 3 stars.\nthey weigh *3, 4 and 5*."),
    (6,  12, "they start *still*, on a\n3-4-5 triangle. cute."),
    (12, 18, "plus a *copy*: small star\nmoved *1/10,000*."),
    (18, 24, "same start, same ending?\n*easy, right?*"),
    (24, 30, "looks random. they just\n*fling* each other around."),
    (30, 36, "there's no *neat* formula.\n*Bruns* showed why, 1887."),
    (36, 42, "so the computer does it in\nabout *10,000 tiny steps*."),
    (42, 49, "now watch the *copy*.\n(the ring ones)"),
    (49, 56, "it's *splitting off!*\nsame start... almost."),
    (56, 62, "copy: star *4*\ngets *kicked out*."),
    (62, 68, "original: star *3*\ngets kicked out."),
    (68, 74, "a *1/10,000* nudge.\ntotally different star out."),
    (74, 80, "weather forecasters:\n*yeah. we know.*"),
]

rng = np.random.default_rng(SEED)
def sim_t(b):
    for (b0, t0), (b1, t1) in zip(KNOTS, KNOTS[1:]):
        if b <= b1: return t0 + (t1-t0)*np.clip((b-b0)/(b1-b0), 0, 1)
    return KNOTS[-1][1]
def ease(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)

# ================= physics =================
A = run(r0, T_END)
rc = np.array(r0, float); rc[0, 0] += EPS
Bc = run(rc, T_END)
N_STEPS = A.t.size - 1
def pos(sol, t): return sol.sol(np.atleast_1d(t))[:6].reshape(3, 2, -1)

# ================= text helpers (from double_slit_v2) =================
F_TITLE = ImageFont.truetype(FP_XB, 66)
F_HOOK  = ImageFont.truetype(FP_B, 38)
CAP_FONTS = {z: ImageFont.truetype(FP_XB, z) for z in (54, 50, 46, 42)}
F_READ  = ImageFont.truetype(FP_SB, 32)
F_WM    = ImageFont.truetype(FP_SB, 28)
F_LBL   = ImageFont.truetype(FP_SB, 28)
F_TAG   = ImageFont.truetype(FP_SB, 26)
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
    widths = [d.textlength(s, font=font) for s, _ in segs]
    x = cx - sum(widths)/2
    for (s, h), w in zip(segs, widths):
        d.text((x, y - rise), s, font=font, fill=(YELLOW if h else WHITE) + (alpha,), stroke_width=7, stroke_fill=BG + (alpha,))
        x += w
def caption(d, b):
    for (b0, b1, txt) in CAPTIONS:
        if b0 <= b < b1:
            t_in = (b-b0)/0.6; t_out = (b1-b)/0.35
            a = int(255*min(1, t_in, t_out)); rise = int(14*(1-min(1, t_in)))
            lines = txt.split("\n")
            for size in (54, 50, 46, 42):
                font = CAP_FONTS[size]
                if max(d.textlength(ln.replace("*", ""), font=font) for ln in lines) <= CAP_MAXW: break
            y0 = 1438 + (0 if len(lines) == 2 else 34)
            for i, ln in enumerate(lines): rich_line(d, CAP_CX, y0 + i*72, ln, font, a, rise)
def label(d, x, y, text, col, tip=None, alpha=255):
    tw = d.textlength(text, font=F_LBL); h = 40
    x0 = int(np.clip(x, SAFE[0], SAFE[2]-tw-28)); y = int(np.clip(y, SAFE[1]+250, 1262))
    if tip is not None:
        sx = x0 + (tw+28)/2; sy = y + (h if tip[1] > y + h else 0)
        d.line((sx, sy, tip[0], tip[1]), fill=col+(int(alpha*0.7),), width=2)
    d.rounded_rectangle((x0, y, x0+tw+28, y+h), 12, fill=(8, 10, 24, int(200*alpha/255)), outline=col+(int(alpha*0.7),), width=2)
    d.text((x0+14, y+3), text, font=F_LBL, fill=WHITE+(alpha,))

# ================= world: deep night sky =================
_yy = np.arange(H, dtype=np.float32)[:, None, None]
VMASK = (np.clip((_yy-370)/50, 0, 1)*np.clip((1330-_yy)/50, 0, 1)).astype(np.float32)
def make_sky():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    def noise(n):
        z = rng.normal(0, 1, (n, n*9//16+1)).astype(np.float32)
        return np.array(Image.fromarray(z).resize((W, H), Image.BICUBIC))
    neb = sum(noise(s)/(i+1) for i, s in enumerate((6, 12, 24, 48)))
    neb = np.clip((neb - 0.2)/1.6, 0, 1)**2
    band = np.exp(-((xx*0.55 + yy*0.4 - 1200)/520)**2)          # diagonal milky band
    img += (neb*band)[..., None]*np.array([38, 22, 70], np.float32)
    img += (noise(10).clip(0, None)*band*0.4)[..., None]*np.array([0, 30, 40], np.float32)
    n = 2600; sx, sy = rng.integers(0, W, n), rng.integers(0, H, n); br = rng.random(n)**3
    for x, y, b in zip(sx, sy, br):
        img[y, x] += 120*b + 25
        if b > 0.5: img[max(0, y-1):y+2, max(0, x-1):x+2] += 40*b
    vig = 1 - 0.55*(((xx-W/2)/W)**2 + ((yy-H/2)/H)**2)*2
    return np.clip(img*vig[..., None], 0, 255)
def sprite(rad, col):
    s = int(rad*6); yy, xx = np.mgrid[-s:s+1, -s:s+1]
    r2 = (xx**2 + yy**2)/rad**2
    core = np.exp(-r2*2.2)[..., None]*np.array([255, 255, 255])
    halo = (np.exp(-r2*0.35)*0.55 + np.exp(-np.sqrt(r2)*0.8)*0.35)[..., None]*np.array(col)
    return np.clip(core + halo, 0, 255).astype(np.float32)
SPR = [sprite(13*(mm/4)**(1/3), c) for mm, c in zip(m, STAR_COL)]
def add_sprite(arr, sp, x, y, g=1.0):
    s = sp.shape[0]//2; x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x-s), max(0, y-s), min(W, x+s+1), min(H, y+s+1)
    if x0 >= x1 or y0 >= y1: return
    arr[y0:y1, x0:x1] += sp[y0-(y-s):y1-(y-s), x0-(x-s):x1-(x-s)]*g

# ================= render =================
def render():
    os.makedirs(OUT_DIR, exist_ok=True)
    sky = make_sky()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                           "-pix_fmt", "yuv420p", os.path.join(OUT_DIR, "video.mp4")], stdin=subprocess.PIPE)
    web_t = np.linspace(0, 38, 3000); web = pos(A, web_t)
    R_cam = R_MIN; cam_c = np.zeros(2); events = []; last_d = None; prev_t = None
    # closest-pass events for the music (local minima of pair distance in the original run)
    tt = np.linspace(0, T_END, 40000); P = pos(A, tt)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(pairs):
        dd = np.linalg.norm(P[i]-P[j], axis=0)
        mins = np.where((dd[1:-1] < dd[:-2]) & (dd[1:-1] < dd[2:]) & (dd[1:-1] < 0.6))[0] + 1
        for q in mins: events.append((float(tt[q]), k, float(dd[q])))
    ev_t = []
    TEST = [int(x) for x in os.environ.get('TEST', '').split(',') if x]
    for f in (TEST or range(NF)):
        b = f/BEAT_F; t = sim_t(b)
        tsub = np.linspace(max(0, t-TRAIL_T), t, 90) if b >= 3 else np.array([t])
        pa = pos(A, tsub); pb = pos(Bc, tsub)
        cur_a, cur_b = pa[:, :, -1], pb[:, :, -1]
        # camera: fit the bodies still in the dance, smooth, capped
        keep_a = [i for i in range(3) if not (i == EJECT['orig'][0] and t > EJECT['orig'][1])]
        keep_b = [i for i in range(3) if not (i == EJECT['copy'][0] and t > EJECT['copy'][1])]
        kept = np.concatenate([cur_a[keep_a], cur_b[keep_b]])
        tgt = kept.mean(0) if t > 52 else np.zeros(2)                # follow the surviving pairs
        cam_c += (tgt - cam_c)*0.06
        need = np.abs(kept - cam_c).max()*1.15
        R_cam += (np.clip(need, R_MIN, R_MAX) - R_cam)*0.06
        sc = R_PIX/R_cam
        to_s = lambda p: (CX + (p[0]-cam_c[0])*sc, CY - (p[1]-cam_c[1])*sc)
        arr = sky.copy()
        tw = 1 + 0.06*np.sin(f*0.07)                                 # slow twinkle of the band
        arr *= tw
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
        # rewind web (frame 1: full tangle of the original's future)
        wa = 1 - ease((b-1)/6)
        if wa > 0:
            for i in range(3):
                pts = [to_s(p) for p in web[i].T[::3]]
                d.line(pts, fill=STAR_COL[i] + (int(95*wa),), width=3)
        # trails: original = coloured, copy = thin cyan
        split = ease((b-40)/8)
        for i in range(3):
            pts = [to_s(p) for p in pa[i].T]
            nseg = len(pts)-1
            for c in range(0, nseg, 10):
                al = (c+10)/nseg
                d.line(pts[c:c+11], fill=STAR_COL[i] + (int(220*al**1.3),), width=int(3+5*al))
            if split > 0:
                pts = [to_s(p) for p in pb[i].T]
                for c in range(0, nseg, 10):
                    al = (c+10)/nseg
                    d.line(pts[c:c+11], fill=CYAN + (int(210*split*al**1.3),), width=3)
        glow = layer.filter(ImageFilter.GaussianBlur(6))
        L = np.array(layer, np.float32); G = np.array(glow, np.float32)
        for Lr in (G, L):
            arr += Lr[..., :3]*(Lr[..., 3:4]/255)*(1.4 if Lr is G else 1.0)
        # stars
        for i in range(3):
            x, y = to_s(cur_a[i]); add_sprite(arr, SPR[i], x, y, 1.0)
        arr = sky*(1-VMASK) + arr*VMASK                               # dance lives in the safe box
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        ring_a = int(255*max(ease((b-12)/1.5)*(1-ease((b-18)/2)), split))
        for i in range(3):
            x, y = to_s(cur_b[i]); rr = 9*(m[i]/4)**(1/3) + 9
            if ring_a > 0: d.ellipse((x-rr, y-rr, x+rr, y+rr), outline=CYAN + (ring_a,), width=3)
        # off-screen ejectee arrows
        tag_y = []
        for key, cur, col0, who in (("orig", cur_a, None, ""), ("copy", cur_b, CYAN, " (copy)")):
                i, te = EJECT[key]; p = cur[i]; col = col0 or STAR_COL[i]
                x, y = to_s(p)
                if t > te and not (SAFE[0] < x < SAFE[2] and 440 < y < 1290):
                    ang = np.arctan2(y-CY, x-CX); ex = CX + 380*np.cos(ang); ey = np.clip(CY + 400*np.sin(ang), 470, 1260)
                    d.polygon([(ex+18*np.cos(ang), ey+18*np.sin(ang)), (ex+12*np.cos(ang+2.4), ey+12*np.sin(ang+2.4)),
                               (ex+12*np.cos(ang-2.4), ey+12*np.sin(ang-2.4))], fill=col + (230,))
                    dist = np.linalg.norm(p - kept.mean(0))
                    tx = f"{NAMES[i]}{who} · {dist:.0f} away"
                    ty = ey+(-50 if ey > CY else 26)
                    for py in tag_y:
                        if abs(ty-py) < 40: ty = py + 44
                    tag_y.append(ty)
                    d.rounded_rectangle((np.clip(ex-70, SAFE[0], SAFE[2]-300), ty-2, np.clip(ex-70, SAFE[0], SAFE[2]-300)+d.textlength(tx, font=F_TAG)+20, ty+34), 10, fill=(8, 10, 24, 190))
                    d.text((np.clip(ex-70, SAFE[0], SAFE[2]-300)+10, ty), tx, font=F_TAG, fill=col + (240,))
        # labels
        if 1 <= b < 12:
            al = int(255*min(1, (b-1)/1))
            for i in range(3):
                x, y = to_s(cur_a[i]); off = [(-170, -90), (40, 50), (40, -100)][i]
                label(d, x+off[0], y+off[1], f"star {NAMES[i]}", STAR_COL[i], tip=(x, y), alpha=al)
        if 12 <= b < 18:
            x, y = to_s(cur_b[0]); label(d, x-230, y-120, "the copy (rings)", CYAN, tip=(x-14, y-14))
        if 49 <= b < 56:
            x, y = to_s(cur_b[1]); label(d, x+40, y-90, "copy", CYAN, tip=(x, y))
        if b >= 56:
            kb = [i for i in range(3) if i != EJECT['copy'][0]]; cxy = cur_b[kb].mean(0); x, y = to_s(cxy)
            label(d, x-60, y+70, "copy: 3 + 5 stay", CYAN, tip=(x, y+20))
            ka = [i for i in range(3) if i != EJECT['orig'][0]]
            if b >= 62: x, y = to_s(cur_a[ka].mean(0)); label(d, x-60, y+70, "original: 4 + 5 stay", STAR_COL[1], tip=(x, y+20))
        # readouts (physics-driven)
        diff = float(np.linalg.norm(cur_a-cur_b, axis=1).max()) if b >= 3 else EPS
        d.text((60, 1336), f"copy vs original: {diff:.4f}" if diff < 10 else f"copy vs original: {diff:.1f}", font=F_READ, fill=CYAN)
        out = []
        for key, nm in (("copy", "copy"), ("orig", "orig")):
            k, te = EJECT[key]
            if t >= te + 1.0: out.append(f"{NAMES[k]} ({nm})")
        txt = "kicked out: " + (", ".join(out) if out else "?")
        d.text((60, 1380), txt, font=F_READ, fill=YELLOW if out else WHITE)
        tag = "real gravity sim · sped up"
        d.text((910-d.textlength(tag, font=F_TAG), 1340), tag, font=F_TAG, fill=(170, 175, 210))
        # top: watermark, series, title, hook / ask
        d.text((60, 176), "@cosmoscoded", font=F_WM, fill=(205, 210, 240))
        d.text((910-d.textlength(SERIES, font=F_WM), 176), SERIES, font=F_WM, fill=(150, 160, 220))
        y = box_text(d, 485, 214, TITLE, F_TITLE, (10, 10, 24), YELLOW)
        if b < 56: box_text(d, 485, y+10, HOOK, F_HOOK, WHITE, (30, 34, 80))
        elif b >= 62: box_text(d, 485, y+10, ASK, F_HOOK, (10, 10, 24), CYAN)
        caption(d, b)
        img.alpha_composite(ov)
        if TEST: img.convert('RGB').save(f'{OUT_DIR}/t{f:04d}.png'); continue
        ff.stdin.write(np.array(img.convert("RGB")).tobytes())
        if f % 200 == 0: print("frame", f, flush=True)
    ff.stdin.close(); ff.wait()
    # music events: each close pass -> a chord-tone bell; u from which pair + how close
    mev = []
    for (te, k, dd) in events:
        bb = None
        for (b0, t0), (b1, t1) in zip(KNOTS[2:], KNOTS[3:]):
            if t0 <= te <= t1: bb = b0 + (te-t0)/(t1-t0)*(b1-b0); break
        if bb is not None and bb < BEATS-2:
            mev.append((bb*60/BPM, [float(np.clip(-1 + k*0.8 + (0.6-dd), -1, 1))], 1))
    mev.sort()
    json.dump({"events": mev, "steps": int(N_STEPS)}, open(os.path.join(OUT_DIR, "events.json"), "w"))
    return mev

SECTIONS = dict(A_END=12, DROP=56, SHUT0=62, SHUT1=49, OPEN1=62, FADE0=78, SPOT0=42, D0=500, D_END=500, BELL_ALL=True)
if __name__ == "__main__":
    import sys
    ev = render() if "skiprender" not in sys.argv[1:] else json.load(open(f"{OUT_DIR}/events.json"))["events"]
    print("events", len(ev), "steps", N_STEPS)
    if os.environ.get("TEST"): sys.exit()
    mc.compose(ev, BEATS, BPM, SECTIONS, os.path.join(OUT_DIR, "music_raw.wav"))
    print("done")
