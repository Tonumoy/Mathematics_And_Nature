# Cover B: the build reveal (600 zeros rebuild the prime staircase)
import sys
sys.argv = ["render.py", "2"]
import render as R
import cairo, numpy as np, cv2

t = 22.7
SS = R.SS; sw, sh = int(R.W * SS), int(R.H * SS)
surf = cairo.ImageSurface(cairo.FORMAT_RGB24, sw, sh)
cx = cairo.Context(surf); cx.scale(SS, SS)
R.background(cx, t)
cx.save(); cx.translate(0, 110)
R.scene_build(cx, t, 1.0)
cx.restore()
w = R.text_w(cx, "RIEMANN HYPOTHESIS", 40)
cx.set_source_rgba(0.86, 0.14, 0.16, 0.97); R.rrect(cx, 540 - w / 2 - 22, 190, w + 44, 66, 10); cx.fill()
R.text(cx, "RIEMANN HYPOTHESIS", 540, 238, 40, R.WHITE, 1, anchor="c")
R.text(cx, "THE PRIMES", 540, 1640, 118, R.WHITE, 1, anchor="c")
R.text(cx, "HAVE A CODE", 540, 1770, 118, R.YEL, 1, anchor="c")
s = "proof machine-checked · Nov 2026"
w2 = R.text_w(cx, s, 30, R.F_MONO, True)
cx.set_source_rgba(0.03, 0.06, 0.1, 0.9); R.rrect(cx, 540 - w2 / 2 - 24, 300, w2 + 48, 58, 29); cx.fill()
cx.set_line_width(2.5); cx.set_source_rgba(*R.GREEN, 0.9); R.rrect(cx, 540 - w2 / 2 - 24, 300, w2 + 48, 58, 29); cx.stroke()
R.text(cx, s, 540, 340, 30, R.GREEN, 1, R.F_MONO, True, anchor="c")
buf = np.ndarray((sh, sw, 4), np.uint8, surf.get_data())
img = buf[:, :, [2, 1, 0]].astype(np.float32)
small = cv2.resize(img, (sw // 4, sh // 4), interpolation=cv2.INTER_AREA)
br = np.clip(small - 70, 0, 255)
gl = cv2.GaussianBlur(br, (0, 0), 12) * 1.2 + cv2.GaussianBlur(br, (0, 0), 36) * 0.9
out = np.clip(img + 0.85 * cv2.resize(gl, (sw, sh)), 0, 255).astype(np.uint8)
cv2.imwrite("SC15_cover_B_4k.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
