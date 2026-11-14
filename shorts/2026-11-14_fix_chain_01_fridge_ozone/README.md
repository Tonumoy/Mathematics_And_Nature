# FIX CHAIN #1 — How did your fridge punch a hole in the sky?
Short: https://youtube.com/shorts/UP-CaTbEycQ (14 Nov 2026)

A new series: one chain of problem → fix → new problem → better fix. The chain HUD at the top grows like a build.

rot → cold → ice melts → compressor fridge → toxic gas (NH₃, SO₂) → Freon (1928) → ozone hole (1985) → Montreal Protocol (1987) → HFC heat trap (GWP 1,430) → isobutane (GWP 3)

- `lib.py` — render kit: supersampled PIL layers, glow, captions/pills/chips, chain HUD, shaded-sphere molecules, numpy Earth renderer (land mask + fBm clouds + ozone-hole overlay).
- `scenes.py` — one function per scene: Earth hook, bacteria doubling N = 2^(t/20 min), refrigeration loop x-ray, toxic leak, CFC-12 molecule + candle, CFC rise + UV, Cl + O₃ catalytic cycle, 1985 hole, Montreal 198/198, HFC infrared trap + bars, isobutane GWP dial, healing to 2066.
- `timeline.py` — voice lines → scene windows (measured from `vo/Lxx.mp3`).
- `render.py` — renders each scene to its own mp4 with 2 workers, then concat.
- `mix.py` — score from `_music/score_engine.py` + ElevenLabs narration, ducked; loudness normalised to −14 LUFS afterwards.
- `thumb.py` — the cover, built from the same Earth, trail and fridge render as the video.
- `FACTS.md`, `SEO.md` — sources, title, description, tags.

Run: `python check6.py 0.5 10 17 25 40 70` (frame check) · `python render.py` · `python mix.py` (needs `vo/`, `fonts/Poppins-ExtraBold.ttf`, `fonts/Poppins-SemiBold.ttf`, `pip install global-land-mask`).
