# Coded Cosmos — "1 electron at a time" (weird physics #3) · https://youtube.com/shorts/dlPTLv6AAKo
# Full-screen, Shorts-safe layout · every part labelled · captions for a curious 12-year-old · dark cinematic score
# Wave: real 2D split-step Schrodinger sim (wave_sim.py). Dots: ideal far-field pattern using the
# published slit sizes of Bach, Pope, Liou, Batelaan, New J. Phys. 15, 033018 (2013).
import numpy as np, subprocess, json, os, wave as wavmod
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.signal import butter, lfilter
from wave_sim import run as wave_run, BAR_Y, SLIT_W, SLIT_D

# ================= globals =================
W, H        = 1080, 1920
FPS, BPM    = 30, 150
BEAT_F      = FPS*60/BPM                 # 12 frames per beat
BEATS       = 100
NF          = int(BEATS*BEAT_F)          # 1200 frames = 40 s
SR          = 44100
SEED        = 7
OUT_DIR     = "out"
TRAIL       = 0.86

BG          = (12, 9, 7)                 # one BG value for the whole Short
AMBER       = (255, 178, 52)
AMBER_DIM   = (150, 98, 34)
WHITE       = (255, 250, 240)
YELLOW      = (255, 214, 64)
CYAN        = (120, 220, 255)
RED         = (255, 86, 60)

# Shorts safe zone (1080x1920): top UI ~0-150, right action buttons x>=950 for y 950-1700,
# channel name + description y>=1600. Important stuff lives inside SAFE.
SAFE        = (40, 160, 930, 1590)       # x0, y0, x1, y1
CAP_CX      = 490                        # caption centre (left of the button column)
CAP_MAXW    = 800

# scene (full bleed, top to bottom: gun -> 2 gaps -> wave -> screen)
WAVE_Y0, WAVE_Y1 = 380, 1110             # wave fills full width 0..1080
DET_BOX     = (130, 1122, 950, 1346)      # electron screen (dots), centred on the gaps
CROP_Y0     = 125                        # grid rows shown from 125 (730 px tall)
SIM_A, SIM_B = 40, 176                   # sim frames: packet at gaps -> packet at the bottom

# real experiment (published values)
E_EV, W_NM, D_NM, N_ELECTRONS = 600, 62.0, 272.0, 6235
U_MAX       = 1.5/W_NM

# beat map
B_A_END, B_DROP, B_B_END = 10, 30, 40
B_SHUT0, B_SHUT1, B_D0   = 50, 52, 54
B_D_FULL, B_D_END        = 72, 80
B_OPEN0, B_OPEN1         = 80, 82
B_E_FULL, B_FADE0        = 94, 97
B_SPOT0, B_SPOT1         = 70, 95

FP_XB  = "fonts/Poppins-ExtraBold.ttf"
FP_B   = "fonts/Poppins-Bold.ttf"
FP_SB  = "fonts/Poppins-SemiBold.ttf"
FP_M   = "fonts/JetBrainsMono.ttf"

TITLE  = "1 ELECTRON AT A TIME"
HOOK   = "guess where they land"
SERIES = "weird physics #3"
ASK    = "what if we WATCH them? comment"

# captions: (beat_from, beat_to, text). *word* = highlighted. \n = line break. Max 2 lines.
CAPTIONS = [
    (0,  10, "okay so we shoot electrons\n*one. at. a. time.*"),
    (10, 20, "each one hits the screen\nas *1 dot*. tiny paintball."),
    (20, 30, "2 gaps, so *2 stripes*.\neasy, right?"),
    (30, 40, "nope. *MANY stripes.*\nlike waves made them."),
    (40, 50, "each one travels like a *wave*\nthrough *both* gaps. then: 1 dot."),
    (50, 56, "now let's *block one gap*"),
    (56, 70, "1 gap open:\nstripes gone. just a *blob*."),
    (70, 82, "weird bit: open *both* gaps\nand this spot gets *FEWER* hits"),
    (82, 100, "more doors, fewer visitors.\nphysics, are you *okay?*"),
]

rng = np.random.default_rng(SEED)
beat_of = lambda f: f/BEAT_F
def ease(x): x = np.clip(x, 0, 1); return x*x*(3-2*x)

# ================= physics: where electrons land =================
u_grid = np.linspace(-U_MAX, U_MAX, 400001)
ENV = np.sinc(W_NM*u_grid)**2
P2, P1 = ENV*np.cos(np.pi*D_NM*u_grid)**2, ENV
def sampler(p):
    c = np.cumsum(p); c /= c[-1]
    return lambda n: np.interp(rng.random(n), c, u_grid)
u2a, u1, u2e = sampler(P2)(N_ELECTRONS), sampler(P1)(N_ELECTRONS), sampler(P2)(N_ELECTRONS)
DET_W, DET_H = DET_BOX[2]-DET_BOX[0], DET_BOX[3]-DET_BOX[1]
to_px = lambda u: (u/U_MAX*0.5+0.5)*DET_W
SPOT_PX, SPOT_HALF = to_px(1/(2*D_NM)), 4     # first dark stripe of the 2-gap pattern

def frac(b, b0, b1, p): return float(np.clip((b-b0)/(b1-b0), 0, 1))**p
def hits_until(f):
    b = beat_of(f)
    if b < B_A_END: return '2a', min(5, int(b//2)+1)
    if b < B_SHUT1: return '2a', max(5, int(round(N_ELECTRONS*frac(b, B_A_END, B_B_END, 5))))
    if b < B_D0:    return '2a', N_ELECTRONS
    if b < B_D_END: return '1', int(round(N_ELECTRONS*frac(b, B_D0, B_D_FULL, 3)))
    return '2e', int(round(N_ELECTRONS*frac(b, B_OPEN1, B_E_FULL, 3)))
def gaps_open(b): return 1 if B_SHUT1 <= b < B_OPEN1 else 2

# ================= wave frames =================
print("wave sim...", flush=True)
f2, _ = wave_run(2); f1, _ = wave_run(1)
NORM = {'2': (np.abs(f2[:20]).max(), np.abs(f2[60:, BAR_Y+20:, :]).max()),
        '1': (np.abs(f1[:20]).max(), np.abs(f1[60:, BAR_Y+20:, :]).max())}
ROWS = np.arange(f2.shape[1])[:, None]
def phosphor(v):
    r = np.clip(v*1.6, 0, 1); g = np.clip((v-0.22)*1.35, 0, 1); b = np.clip((v-0.72)*2.6, 0, 1)
    return np.dstack([r, g*0.82, b*0.75])
def wave_img(run, k):
    fr = f2 if run == '2' else f1; inc, tr = NORM[run]
    p = fr[k]; a = np.abs(p)
    g = np.where(ROWS < BAR_Y, a/inc*0.6, a/tr)
    v = np.clip(g, 0, 1)**0.7*(0.8+0.2*np.cos(np.angle(p)))
    return phosphor(v)[CROP_Y0:]
def cycle_index(f):
    period = 2*BEAT_F if beat_of(f) < B_A_END else BEAT_F
    return int(SIM_A + ((f % period)/period)*(SIM_B-SIM_A))
def wave_run_for(b): return '1' if B_SHUT1 <= b < B_OPEN1 else '2'

# ================= static background =================
SC = W/448                                       # grid px -> screen px
BAR_SY = WAVE_Y0 + (BAR_Y-CROP_Y0)*SC            # barrier y on screen
def make_base():
    arr = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
    arr += rng.normal(0, 2.2, (H, W, 1))
    yy = np.arange(H)[:, None, None]; xx = np.arange(W)[None, :, None]
    arr += 14*np.exp(-((yy-760)/650)**2)*np.exp(-((xx-540)/700)**2)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = DET_BOX                     # screen glass
    d.rounded_rectangle((x0-10, y0-6, x1+10, y1+8), 16, fill=(5, 4, 3), outline=(70, 52, 34), width=3)
    return img
def vignette_mask():
    y = np.linspace(0, 1, WAVE_Y1-WAVE_Y0)[:, None]; x = np.linspace(-1, 1, W)[None, :]
    m = np.clip(y/0.06, 0, 1)*np.clip((1-y)/0.10, 0, 1)*(1-0.45*x**4)
    return m[..., None].astype(np.float32)
VIG = vignette_mask()

# ================= text =================
F_TITLE = ImageFont.truetype(FP_XB, 68)
F_HOOK  = ImageFont.truetype(FP_B, 38)
F_CAP   = ImageFont.truetype(FP_XB, 54)
CAP_FONTS = {z: ImageFont.truetype(FP_XB, z) for z in (54, 50, 46, 42)}
F_READ  = ImageFont.truetype(FP_SB, 34)
F_SMALL = ImageFont.truetype(FP_M, 22); F_SMALL.set_variation_by_axes([600])
F_WM    = ImageFont.truetype(FP_SB, 28)
F_CHIP  = ImageFont.truetype(FP_SB, 30)

def box_text(d, cx, y, text, font, fg, bgc, pad=(28, 10)):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    x0 = cx - (r-l)/2 - pad[0]; x1 = cx + (r-l)/2 + pad[0]
    d.rounded_rectangle((x0, y, x1, y + (b-t) + 2*pad[1]), 14, fill=bgc)
    d.text((cx - (r-l)/2 - l, y + pad[1] - t), text, font=font, fill=fg)
    return y + (b-t) + 2*pad[1]
def rich_line(d, cx, y, line, font, alpha, rise):
    """draw one caption line; *x* segments highlighted yellow; thick dark outline"""
    segs, hi = [], False
    for part in line.split("*"):
        if part: segs.append((part, hi))
        hi = not hi
    widths = [d.textlength(s, font=font) for s, _ in segs]
    x = cx - sum(widths)/2
    for (s, h), w in zip(segs, widths):
        col = (YELLOW if h else WHITE) + (alpha,)
        d.text((x, y - rise), s, font=font, fill=col, stroke_width=7, stroke_fill=BG + (alpha,))
        x += w
F_LBL   = ImageFont.truetype(FP_SB, 28)
LBL_FG, LBL_BG, LBL_LINE = (255, 240, 215), (14, 10, 8, 200), (255, 220, 170)
def label(d, x, y, text, tip=None, alpha=255, anchor="l"):
    """pill label at (x,y); optional leader line to tip=(tx,ty)"""
    tw = d.textlength(text, font=F_LBL); h = 40
    x0 = x if anchor == "l" else (x - tw - 28 if anchor == "r" else x - tw/2 - 14)
    if tip is not None:
        sx = x0 + tw + 28 if tip[0] > x0 + tw + 28 else (x0 if tip[0] < x0 else x0 + (tw+28)/2)
        sy = y + h/2
        d.line((sx, sy, tip[0], tip[1]), fill=LBL_LINE+(int(alpha*0.85),), width=3)
        d.ellipse((tip[0]-5, tip[1]-5, tip[0]+5, tip[1]+5), fill=LBL_LINE+(alpha,))
    d.rounded_rectangle((x0, y, x0+tw+28, y+h), 12, fill=LBL_BG[:3]+(int(LBL_BG[3]*alpha/255),), outline=(255, 190, 90, int(alpha*0.6)), width=2)
    d.text((x0+14, y+3), text, font=F_LBL, fill=LBL_FG+(alpha,))

def caption(d, b):
    for (b0, b1, txt) in CAPTIONS:
        if b0 <= b < b1:
            t_in = (b-b0)/0.6; t_out = (b1-b)/0.35
            a = int(255*min(1, t_in, t_out)); rise = int(14*(1-min(1, t_in)))
            lines = txt.split("\n")
            font = F_CAP
            for size in (54, 50, 46, 42):
                font = CAP_FONTS[size]
                if max(d.textlength(ln.replace("*", ""), font=font) for ln in lines) <= CAP_MAXW: break
            y0 = 1438 + (0 if len(lines) == 2 else 34)
            for i, ln in enumerate(lines):
                rich_line(d, CAP_CX, y0 + i*72, ln, font, a, rise)

# ================= render =================
def render():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_base()
    acc = np.zeros((DET_H, DET_W), np.float32)
    det_y = rng.random(N_ELECTRONS*3)*(DET_H-14)+7
    yy, xx = np.mgrid[-4:5, -4:5]; splat = np.exp(-(xx**2+yy**2)/3.0).astype(np.float32)
    landed = {'2a': 0, '1': 0, '2e': 0}; yoff = {'2a': 0, '1': N_ELECTRONS, '2e': 2*N_ELECTRONS}
    srcs = {'2a': u2a, '1': u1, '2e': u2e}
    events, flash = [], []
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(OUT_DIR, "video_noaudio.mp4")],
                          stdin=subprocess.PIPE)
    trail, prev = None, '2'
    for f0 in range(-int(2*BEAT_F), 0):          # pre-warm so frame 1 is already full
        wv = wave_img('2', cycle_index(f0)); trail = wv if trail is None else np.maximum(wv, trail*TRAIL)
    wh = WAVE_Y1-WAVE_Y0
    for f in range(NF):
        b = beat_of(f)
        if B_SHUT1 <= b < B_D0:            acc *= 0.88   # screen wiped between experiments
        elif B_OPEN0 <= b < B_OPEN1 + 1:   acc *= 0.90
        elif b >= B_FADE0:                 acc *= 0.86
        run, n = hits_until(f)
        new = n - landed[run]
        if new > 0:
            us = srcs[run][landed[run]:n]
            for j, u in enumerate(us):
                x = int(to_px(u)); y = int(det_y[yoff[run] + landed[run] + j])
                if 4 <= x < DET_W-5: acc[y-4:y+5, x-4:x+5] += splat
            if b < B_A_END + 0.5:
                for j, u in enumerate(us): flash.append((to_px(u), det_y[yoff[run]+landed[run]+j], f))
            events.append((f/FPS, [float(u/U_MAX) for u in us[:3]], int(new)))
            landed[run] = n
        img = base.copy(); d = ImageDraw.Draw(img, "RGBA")
        # ---- wave, full bleed
        r = wave_run_for(b)
        wv = wave_img(r, cycle_index(f))
        trail = wv if r != prev else np.maximum(wv, trail*TRAIL); prev = r
        wim = Image.fromarray((np.clip(trail, 0, 1)*255).astype(np.uint8)).resize((W, wh), Image.BICUBIC)
        glow = np.array(wim.filter(ImageFilter.GaussianBlur(12)), np.float32)
        warr = np.clip(np.array(wim, np.float32) + 0.7*glow, 0, 255)*VIG
        region = np.array(img.crop((0, WAVE_Y0, W, WAVE_Y1)), np.float32)
        img.paste(Image.fromarray(np.clip(region + warr, 0, 255).astype(np.uint8)), (0, WAVE_Y0))
        # ---- the wall with 2 gaps
        cx = W/2
        d.rectangle((0, BAR_SY, W, BAR_SY+6*SC), fill=(170, 150, 128, 255))
        for c in (-SLIT_D/2, SLIT_D/2):
            d.rectangle((cx+(c-SLIT_W/2)*SC, BAR_SY-1, cx+(c+SLIT_W/2)*SC, BAR_SY+6*SC+1), fill=(4, 3, 2, 255))
        # electron gun (glowing nozzle at the top)
        d.rounded_rectangle((cx-46, WAVE_Y0-14, cx+46, WAVE_Y0+16), 8, fill=(60, 48, 38, 255), outline=(150, 120, 90, 255), width=2)
        pulse = 0.5+0.5*np.cos(2*np.pi*(f % BEAT_F)/BEAT_F)
        d.ellipse((cx-10, WAVE_Y0+8, cx+10, WAVE_Y0+28), fill=(255, 210, 120, int(120+130*pulse)))
        # shutter over the right gap
        if B_SHUT0 <= b < B_OPEN1:
            s = ease((b-B_SHUT0)/(B_SHUT1-B_SHUT0)) if b < B_SHUT1 else (1.0 if b < B_OPEN0 else 1-ease((b-B_OPEN0)/(B_OPEN1-B_OPEN0)))
            sx = W - s*(W-(cx+(SLIT_D/2-SLIT_W/2-2)*SC))
            d.rectangle((sx, BAR_SY-16, W, BAR_SY+6*SC+16), fill=(200, 66, 42, 255), outline=(255, 150, 100, 255), width=3)
            if s > 0.6: d.text((sx+22, BAR_SY-12), "BLOCKER", font=F_SMALL, fill=WHITE)
        # ---- electron screen
        disp = 1-np.exp(-acc*0.9)
        dimg = Image.fromarray((np.clip(np.dstack([disp, disp*0.69, disp*0.18]), 0, 1)*255).astype(np.uint8))
        dg = dimg.filter(ImageFilter.GaussianBlur(6))
        dimg = Image.fromarray(np.clip(np.array(dimg, np.int16)+np.array(dg, np.int16), 0, 255).astype(np.uint8))
        img.paste(dimg, DET_BOX[:2])
        for (fx, fy, f0) in flash:
            age = f - f0
            if 0 <= age < 12:
                rr = 12+age*3; a = int(255*(1-age/12))
                d.ellipse((DET_BOX[0]+fx-rr, DET_BOX[1]+fy-rr, DET_BOX[0]+fx+rr, DET_BOX[1]+fy+rr), outline=(255, 240, 200, a), width=4)
        flash = [q for q in flash if f - q[2] < 12]
        disc = "sim, sped up · lab sizes: Bach 2013"
        d.text((DET_BOX[2]-8-d.textlength(disc, font=F_SMALL), DET_BOX[3]-30), disc, font=F_SMALL, fill=AMBER_DIM)
        # ---- labels for every part of the experiment
        la = 255 if b < B_DROP else 205
        gx = W/2
        label(d, gx+150, WAVE_Y0-22, "electron gun", tip=(gx+48, WAVE_Y0), alpha=la)
        label(d, 50, BAR_SY-54, "wall", tip=(110, BAR_SY+3), alpha=la)
        label(d, gx-120, BAR_SY+40, "2 tiny gaps (slits)", tip=(gx+(-SLIT_D/2)*SC, BAR_SY+8), alpha=la, anchor="r")
        if wave_run_for(b) == '1':
            label(d, gx+(SLIT_D/2+14)*SC, BAR_SY+40, "1 gap open", alpha=la)
        label(d, 50, 760, "electron wave", alpha=la)
        label(d, DET_BOX[0]+10, DET_BOX[1]+10, "detector screen", alpha=la)
        if B_SPOT0 <= b < B_SPOT1:                  # the spot that goes dark
            mx = DET_BOX[0]+SPOT_PX
            d.rectangle((mx-SPOT_HALF-3, DET_BOX[1]-2, mx+SPOT_HALF+3, DET_BOX[3]+2), outline=CYAN+(255,), width=3)
            d.polygon([(mx, DET_BOX[1]-6), (mx-16, DET_BOX[1]-30), (mx+16, DET_BOX[1]-30)], fill=CYAN+(255,))
            if b < B_OPEN0:
                cnt = landed['1']
                n1 = int((np.abs(to_px(u1[:cnt])-SPOT_PX) < SPOT_HALF).sum())
                n2 = int((np.abs(to_px(u2a[:cnt])-SPOT_PX) < SPOT_HALF).sum())
                chip = f"this spot:  1 gap {n1}   ·   2 gaps {n2}"
                tw = d.textlength(chip, font=F_CHIP)
                d.rounded_rectangle((CAP_CX-tw/2-18, 1040, CAP_CX+tw/2+18, 1088), 12, fill=(6, 22, 30, 235))
                d.text((CAP_CX-tw/2, 1044), chip, font=F_CHIP, fill=CYAN)
        # ---- readouts (inside the safe zone)
        total = landed[run] if not (B_B_END <= b < B_D0) else N_ELECTRONS
        d.text((DET_BOX[0], 1356), f"electrons fired: {total:,}", font=F_READ, fill=AMBER)
        go = gaps_open(b)
        txt = f"gaps open: {go}"
        d.text((DET_BOX[2]-25-d.textlength(txt, font=F_READ), 1356), txt, font=F_READ, fill=(RED if go == 1 else AMBER))
        # ---- top: title + hook / series / ask
        d.text((60, 176), "@cosmoscoded", font=F_WM, fill=(235, 215, 185))
        d.text((900-d.textlength(SERIES, font=F_WM), 176), SERIES, font=F_WM, fill=AMBER_DIM)
        y = box_text(d, 485, 214, TITLE, F_TITLE, (24, 14, 6), AMBER)
        if b < B_DROP:
            box_text(d, 485, y+10, HOOK, F_HOOK, WHITE, (70, 46, 24))
        elif b >= B_OPEN1:
            box_text(d, 485, y+10, ASK, F_HOOK, (24, 14, 6), YELLOW)
        caption(d, b)
        arr = np.array(img); arr[::3] = (arr[::3]*0.93).astype(np.uint8)   # soft CRT scanlines
        ff.stdin.write(arr.tobytes())
        if f % 200 == 0: print("frame", f, flush=True)
    ff.stdin.close(); ff.wait()
    json.dump({"events": events}, open(os.path.join(OUT_DIR, "events.json"), "w"))
    return events

# ================= music: dark cinematic score (shorts/_music/music_cinematic.py) =================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_music"))
import music_cinematic as mc

def score(events):
    S = dict(A_END=B_A_END, DROP=B_DROP, SHUT0=B_SHUT0, SHUT1=B_SHUT1, D0=B_D0, D_END=B_D_END,
             OPEN1=B_OPEN1, FADE0=B_FADE0, SPOT0=B_SPOT0)
    raw = os.path.join(OUT_DIR, "music_raw.wav")
    mc.compose([tuple(e) for e in events], BEATS, BPM, S, raw)
    return raw

def mux(raw):
    """loudness -14 LUFS, true peak < -1 dBTP, then mux with the video"""
    tmp = os.path.join(OUT_DIR, "music.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af",
                    "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,highpass=f=30,alimiter=limit=0.7:attack=3:release=80:level=disabled",
                    tmp], check=True)
    out = os.path.join(OUT_DIR, "double_slit.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(OUT_DIR, "video_noaudio.mp4"), "-i", tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    return out

if __name__ == "__main__":
    ev = render()
    print("video ->", mux(score(ev)))
