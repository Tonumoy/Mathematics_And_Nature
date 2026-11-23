# Own-render cover for SOURCE CODE #15 (2160x3840)
import sys
sys.argv = ["render.py", "2"]
import render as R
import cairo, numpy as np, cv2, math

def cover(out, t=4.2):
    SS = R.SS; sw, sh = int(R.W * SS), int(R.H * SS)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, sw, sh)
    cx = cairo.Context(surf); cx.scale(SS, SS)
    R.background(cx, t)
    cam = (-2.0, 3.0, 5.6, 0.14, 1000, R.W / 2, 1380)
    R.landscape(cx, t, cam, 1.0)
    # dark band behind title for legibility
    g = cairo.LinearGradient(0, 250, 0, 900)
    g.add_color_stop_rgba(0, 0, 0, 0, 0.0); g.add_color_stop_rgba(0.3, 0.0, 0.01, 0.03, 0.55)
    g.add_color_stop_rgba(1, 0, 0, 0, 0.0)
    cx.set_source(g); cx.rectangle(0, 250, R.W, 650); cx.fill()
    # red sticker
    w = R.text_w(cx, "RIEMANN HYPOTHESIS", 40)
    cx.set_source_rgba(0.86, 0.14, 0.16, 0.97); R.rrect(cx, 540 - w / 2 - 22, 300, w + 44, 66, 10); cx.fill()
    R.text(cx, "RIEMANN HYPOTHESIS", 540, 348, 40, R.WHITE, 1, anchor="c")
    R.text(cx, "167 YEARS.", 540, 520, 150, R.WHITE, 1, anchor="c")
    wpr = R.text(cx, "PROVED", 480, 690, 160, R.GREEN, 1, anchor="c")
    R.check(cx, 480 + wpr / 2 + 26, 668, 110, R.GREEN, 1)
    # fact pill
    s = "every step machine-checked · 21 Nov 2026"
    w2 = R.text_w(cx, s, 34, R.F_MONO, True)
    cx.set_source_rgba(0.03, 0.06, 0.1, 0.9); R.rrect(cx, 540 - w2 / 2 - 24, 1450, w2 + 48, 64, 32); cx.fill()
    cx.set_line_width(2.5); cx.set_source_rgba(*R.CYAN, 0.9); R.rrect(cx, 540 - w2 / 2 - 24, 1450, w2 + 48, 64, 32); cx.stroke()
    R.text(cx, s, 540, 1494, 34, R.CYAN, 1, R.F_MONO, True, anchor="c")
    R.text(cx, "SOURCE CODE #15", 540, 1570, 26, R.GREY, 0.9, R.F_MONO, True, anchor="c")
    buf = np.ndarray((sh, sw, 4), np.uint8, surf.get_data())
    img = buf[:, :, [2, 1, 0]].astype(np.float32)
    small = cv2.resize(img, (sw // 4, sh // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 70, 0, 255)
    gl = cv2.GaussianBlur(br, (0, 0), 12) * 1.2 + cv2.GaussianBlur(br, (0, 0), 36) * 0.9
    out_img = np.clip(img + 0.85 * cv2.resize(gl, (sw, sh)), 0, 255).astype(np.uint8)
    cv2.imwrite(out, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])

cover("SC15_cover_A_4k.jpg")
