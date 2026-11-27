# cover rebuilt from the Short's own render (Saturn + particle edge), R7
import sys; sys.argv = ['render.py', '2', 'info']
import render as R, numpy as np, cairo, cv2, math
SS = 2; sw, sh = 1080 * SS, 1920 * SS
cam = R.orbit_cam(-22, 14, 8.2, 40)
img = R.saturn(cam, sw, sh, 1.0, True, 1.0, R.stars(sw, sh))
# bloom
small = cv2.resize(img, (sw // 4, sh // 4), interpolation=cv2.INTER_AREA)
gl = cv2.GaussianBlur(np.clip(small - 150, 0, 255), (0, 0), 20); img = np.clip(img + cv2.resize(gl, (sw, sh)) * 0.5, 0, 255)
ov = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw, sh); cx = cairo.Context(ov); cx.scale(SS, SS)
# dark scrim top for title
g = cairo.LinearGradient(0, 120, 0, 760); g.add_color_stop_rgba(0, 0, 0.01, 0.03, 0.85); g.add_color_stop_rgba(1, 0, 0, 0, 0)
cx.set_source(g); cx.rectangle(0, 120, 1080, 640); cx.fill()
R.text(cx, "WHY SO", 485, 430, 150, R.WHITE, 1, R.F_CAP, True, anchor="c")
R.text(cx, "FLAT?", 485, 600, 190, R.YEL, 1, R.F_CAP, True, anchor="c")
x, y, _ = cam.project(np.array([[-1.3, -1.75, 0.0]]))
R.ring2d(cx, x[0], y[0], 80, R.CYAN, 1, 7)
R.pill(cx, "270,000 km wide · 10 m thick", 485, 1450, 38, R.CYAN, 1)
R.text(cx, "SOURCE CODE #16", 930, 205, 30, R.CYAN, 0.95, R.F_MONO, True, anchor="r")
o = np.ndarray((sh, sw, 4), np.uint8, ov.get_data()).astype(np.float32); al = o[..., 3:4] / 255
out = np.clip(img * (1 - al) + o[..., [2, 1, 0]], 0, 255).astype(np.uint8)
out = cv2.resize(out, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('thumbnail.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite('thumb_360.png', cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (360, 640), interpolation=cv2.INTER_AREA))
