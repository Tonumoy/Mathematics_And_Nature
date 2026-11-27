# SOURCE CODE #16 · Why Are Saturn's Rings So Flat? 🪐

YouTube Short: https://youtube.com/shorts/sKoKW8IjkXI (public 28 Nov 2026, 7:30 PM IST)

Saturn's rings are ~270,000 km wide but only ~10 m thick. This Short shows why: billions of ice chunks collide
inelastically. Each bump reverses and shrinks the relative speed along the contact (v' = -e·v), which drains
up-and-down motion while orbital spin (angular momentum) survives, so a puffy cloud flattens into a sheet.
Set e = 1 (perfectly bouncy) and the ring stays puffy.

| file | what it does |
|---|---|
| `sim.py` | collisional N-body ring model: 6,000 particles, Kepler orbits, DSMC-style pair collisions with restitution e (runs A e=0.1, B e=1, C back to e=0.1) -> `sim_*.npz` |
| `render.py` | ray-traced Saturn (real C/B/A ring radii, Cassini Division, shadows) + particle renders, captions, dial; timed to `durs.json` |
| `go.sh` | 2-worker 4K render: `bash go.sh 2 18 out.mp4` |
| `mix.py` | voice + ducked music + SFX, two-pass loudnorm to -14 LUFS |
| `thumb.py` | cover from the Short's own render |
| `FACTS.md`, `SEO.md` | sources and publishing text |

Run: `python3 sim.py && bash go.sh 1 20 preview.mp4`. Needs numpy, pycairo, opencv-python, ffmpeg, Poppins font.
Voice: ElevenLabs (Daniel), music + SFX: ElevenLabs (not included).
