
# ---------------- v2 scenes: the bugs and the tricks (new clock tn) ----------------
VAULT_C = (14.95, 12.75, 0.5)
DOC_C = (12.55, 14.75, 0.62)
DOOR_X0, DOOR_X1, DOOR_Y, DOOR_Z1 = 4.85, 5.45, PROXY_C[1] - 0.76, 1.05

def new_visuals(img, glow, cam, tn, ov):
    # BUG 1: the proxy fetches ANY address (SSRF) -> lines to every internet node
    if 19.0 < tn < 28.6:
        a = sm(lin(tn, 19.1, 19.6)) * (1 - sm(lin(tn, 24.2, 24.8)))
        if a > 0.01:
            src, _ = cam.proj(np.array((PROXY_C[0], PROXY_C[1], 1.65)))
            S2, _ = cam.proj(NET_NODES)
            n_on = int(np.clip((tn - 19.4) * 4.0, 0, len(NET_NODES)))
            for k in range(n_on):
                b = S2[k]
                col = tuple(v * a for v in RED)
                for q in range(0, 20, 2):
                    p0 = src + (b - src) * q / 20; p1 = src + (b - src) * (q + 1) / 20
                    line(img, p0, p1, col, 2.2 * SS, glow, RED, 0.5 * a)
                u = (tn * 1.3 + k * 0.17) % 1.0
                dot(img, src + (b - src) * u, 4 * SS, tuple(v * a for v in WHITE), glow, 14 * SS, RED, 0.9 * a)
                dot(img, b, 5 * SS, tuple(v * a for v in RED), glow, 16 * SS, RED, 0.8 * a)
            ov.append(("bug1", label_pt(cam, (PROXY_C[0], PROXY_C[1], 1.7)), a))
        # BUG 2: old login-renewal door on the proxy's front face
        b2 = sm(lin(tn, 24.4, 24.9))
        if b2 > 0.01:
            th = math.radians(78) * sm(lin(tn, 25.6, 27.6))
            hx = DOOR_X0
            gl = 0.4 + 0.6 * sm(lin(tn, 25.6, 27.6))
            hole = np.array([[DOOR_X0, DOOR_Y, 0.02], [DOOR_X1, DOOR_Y, 0.02], [DOOR_X1, DOOR_Y, DOOR_Z1], [DOOR_X0, DOOR_Y, DOOR_Z1]])
            h2, _ = cam.proj(hole)
            poly_blend(img, h2, (90, 200, 255), 0.95 * b2 * gl)
            cen, _ = cam.proj(hole.mean(0))
            dot(img, cen, 1, (0, 0, 0), glow, 60 * SS, (60, 180, 255), 1.2 * b2 * gl)
            w = DOOR_X1 - DOOR_X0
            far = np.array([hx + w * math.cos(th), DOOR_Y - w * math.sin(th)])
            leaf = np.array([[hx, DOOR_Y, 0.02], [far[0], far[1], 0.02], [far[0], far[1], DOOR_Z1], [hx, DOOR_Y, DOOR_Z1]])
            l2, _ = cam.proj(leaf)
            poly_blend(img, l2, (40, 120, 200), 1.0 * b2)
            for i in range(4):
                line(img, l2[i], l2[(i + 1) % 4], tuple(v * b2 for v in (120, 220, 255)), 2 * SS, glow, AMBER, 0.6 * b2)
            ov.append(("bug2", label_pt(cam, (5.15, DOOR_Y, DOOR_Z1 + 0.05)), b2 * (1 - sm(lin(tn, 28.3, 28.9)))))
            if tn > 27.0:
                ov.append(("admin", label_pt(cam, (PROXY_C[0] + 0.4, PROXY_C[1] + 0.4, 1.75)), sm(lin(tn, 27.0, 27.4)) * (1 - sm(lin(tn, 28.6, 29.2)))))
    # 14 public keys
    if 37.6 < tn < 42.6:
        a = sm(lin(tn, 37.9, 38.4)) * (1 - sm(lin(tn, 41.9, 42.5)))
        dest = np.array((8.2, 8.2, 1.9))
        for k in range(14):
            st = NET_NODES[k] + np.array([0, 0, 0.5])
            u = sm(lin(tn, 39.2 + k * 0.12, 40.4 + k * 0.12))
            p = st + (dest - st) * u + np.array([0, 0, 0.9 * math.sin(math.pi * u)])
            s2, _ = cam.proj(p)
            dot(img, s2, 5.5 * SS, tuple(v * a for v in (120, 230, 255)), glow, 22 * SS, GOLD, 1.0 * a)
        ov.append(("keys", label_pt(cam, tuple(NET_C + np.array([0, 0, 1.6]))), a, int(14 * min(1, lin(tn, 38.2, 39.4)))))
    # TRICK 1: dataset points at the server's own secrets -> read out
    if 44.4 < tn < 54.8:
        a = sm(lin(tn, 44.6, 45.1)) * (1 - sm(lin(tn, 54.2, 54.7)))
        op = sm(lin(tn, 48.7, 49.5))
        draw_box(img, glow, cam, VAULT_C, (0.8, 0.8, 0.9), GOLD, a, edge=(140, 235, 255), edge_a=0.6 + 0.4 * op, gw=0.4 + 0.8 * op)
        draw_box(img, glow, cam, (VAULT_C[0], VAULT_C[1], VAULT_C[2] + 0.5 + 0.5 * op), (0.84, 0.84, 0.1), GOLD, a, edge=(160, 240, 255), edge_a=0.7, gw=0.4)
        if 45.8 < tn < 50.3:
            pa, _ = cam.proj(np.array((POD_C[0], POD_C[1], 1.55)))
            pb, _ = cam.proj(np.array((VAULT_C[0], VAULT_C[1], 1.0)))
            uu = sm(lin(tn, 45.8, 46.8))
            for q in range(0, 24, 2):
                if q / 24 > uu:
                    break
                p0 = pa + (pb - pa) * q / 24; p1 = pa + (pb - pa) * min(uu, (q + 1) / 24)
                line(img, p0, p1, (120, 230, 255), 2.4 * SS, glow, GOLD, 0.8)
        if op > 0.05 and tn < 52.0:
            for k in range(26):
                u = ((tn - 48.7) * 0.55 + k / 26) % 1.0
                st = np.array(VAULT_C) + [0, 0, 0.6]
                if u < 0.5:
                    p = st + (np.array(POD_C) + [0, 0, 1.4] - st) * (u * 2)
                else:
                    p = route_at(np.array([1 - (u - 0.5) * 2 * 0.2]))[0]
                s2, _ = cam.proj(p)
                dot(img, s2, 3.2 * SS, (140, 235, 255), glow, 12 * SS, GOLD, 0.9 * op * (1 - sm(lin(tn, 51.4, 52.0))))
        ov.append(("vault", label_pt(cam, (VAULT_C[0], VAULT_C[1], 1.1)), a))
    # TRICK 2: code hidden in a settings file -> runs
    if 50.2 < tn < 54.9:
        a = sm(lin(tn, 50.4, 50.9)) * (1 - sm(lin(tn, 54.3, 54.8)))
        red = sm(lin(tn, 53.3, 53.7))
        col = np.array((235, 235, 235), np.float32) * (1 - red) + np.array(RED, np.float32) * red
        draw_box(img, glow, cam, DOC_C, (0.1, 0.75, 1.0), tuple(col.tolist()), a, edge=(255, 255, 255) if red < 0.5 else RED, edge_a=0.8, gw=0.3 + 0.7 * red)
        if tn > 51.6:
            uu = sm(lin(tn, 51.6, 52.8))
            pa, _ = cam.proj(np.array((DOC_C[0], DOC_C[1], 0.9)))
            pb, _ = cam.proj(np.array((POD_C[0], POD_C[1], 1.0)))
            line(img, pa, pa + (pb - pa) * uu, tuple(v * a for v in WHITE), 2.4 * SS, glow, WHITE, 0.5 * a)
        ov.append(("doc", label_pt(cam, (DOC_C[0], DOC_C[1], 1.25)), a, red))

def new_overlays(d, tn, L):
    if "bug1" in L and L["bug1"][2] > 0.02:
        (x, y), a = L["bug1"][1], L["bug1"][2]
        badge(d, x - 40, y - 330, "BUG 1", "the pipe would fetch ANY address", "(server-side request forgery)", RED, a)
    if "bug2" in L and L["bug2"][2] > 0.02:
        (x, y), a = L["bug2"][1], L["bug2"][2]
        badge(d, x - 60, y + 250, "BUG 2", "an old login-renewal door", "gave them admin of the pipe", AMBER, a, leader=(x, y))
    if "admin" in L and L["admin"][2] > 0.02:
        (x, y), a = L["admin"][1], L["admin"][2]
        tag(d, x + 60, y - 60, "ADMIN", GOLD, 32, "l", leader=(x, y), alpha=a)
    if "keys" in L and L["keys"][2] > 0.02:
        (x, y), a, n = L["keys"][1], L["keys"][2], L["keys"][3]
        badge(d, x, y - 220, f"{n} KEYS", "Hugging Face access keys", "people had left public online", GOLD, a)
    if "vault" in L and L["vault"][2] > 0.02 and tn < 50.4:
        (x, y), a = L["vault"][1], L["vault"][2]
        badge(d, x + 40, y + 300, "TRICK 1", "a data file that points at", "the server's own secret files", GOLD, a, leader=(x, y))
    if "doc" in L and L["doc"][2] > 0.02:
        (x, y), a, red = L["doc"][1], L["doc"][2], L["doc"][3]
        d.text((x, y - 40), "{{ }}", font=font(F_MONO, 34), fill=(255, 255, 255, int(255 * a)), anchor="mm")
        badge(d, x - 20, y + 360, "TRICK 2", "a settings file with hidden code", "the server ran it as a program", RED if red > 0.5 else WHITE, a)

def badge(d, x, y, head, l1, l2, col_bgr, a, leader=None):
    if a <= 0.02:
        return
    col = bgr2rgb(col_bgr)
    fh, fl = font(F_XB, 30), font(F_SEMI, 30)
    w = max(d.textlength(l1, font=fl), d.textlength(l2, font=fl)) + 56
    hgt = 150
    x0 = max(40, min(x - w / 2, 1040 - w)); y0 = max(470, min(y - hgt / 2, 1560 - hgt))
    if leader is not None:
        d.line([leader, (x0 + w / 2, y0 + (0 if leader[1] < y0 else hgt))], fill=col + (int(200 * a),), width=2)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + hgt), 22, fill=(14, 16, 20, int(238 * a)), outline=col + (int(255 * a),), width=3)
    hw = d.textlength(head, font=fh) + 28
    d.rounded_rectangle((x0 + 22, y0 + 16, x0 + 22 + hw, y0 + 56), 12, fill=col + (int(255 * a),))
    d.text((x0 + 36, y0 + 36), head, font=fh, fill=(12, 12, 14, int(255 * a)), anchor="lm")
    d.text((x0 + 28, y0 + 84), l1, font=fl, fill=(240, 242, 246, int(255 * a)), anchor="lm")
    d.text((x0 + 28, y0 + 122), l2, font=font(F_MED, 28), fill=(165, 170, 180, int(255 * a)), anchor="lm")

