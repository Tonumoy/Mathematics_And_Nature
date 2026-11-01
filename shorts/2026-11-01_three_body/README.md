# 3 stars. Who gets kicked out? — the Pythagorean three-body problem

YouTube Short: https://youtube.com/shorts/Kq_MgFi6x4A (Coded Cosmos · weird physics #4)

Three point masses (3, 4, 5) start at rest on the corners of a 3-4-5 right triangle, each opposite the side of its own length (Burrau 1913). Pure Newtonian gravity, G = 1.

- **Original run:** star 3 is ejected (last close pass t ≈ 60); stars 4 and 5 form a binary — the result first found by Szebehely & Peters, *Astronomical Journal* 72, 876 (1967).
- **Copy run:** star 3 moved by 1e-4 in x → star 4 is ejected instead (t ≈ 54). Runs differ by > 0.01 at t ≈ 34 and > 1 at t ≈ 45: chaos.
- Why no formula: Bruns (1887) — no extra algebraic first integrals; Poincaré (1889) — non-integrable/chaotic; Sundman (1912) — a convergent series exists but converges far too slowly to use.

## Files
| file | what |
|---|---|
| `sim.py` | 3-body equations + `solve_ivp(DOP853, rtol=atol=1e-13)`; energy drift ~1e-8 |
| `three_body.py` | renders the 1080×1920, 30 fps, 40 s Short (PIL compositor, procedural night sky, live "copy vs original" readout) and composes the score |
| `../_music/music_cinematic.py` | dark cinematic score engine (each close pass between two stars rings a chord-tone bell; `BELL_ALL` option) |

## Run
```bash
pip install numpy scipy pillow
cp ../_music/music_cinematic.py .   # the script imports it from its own folder
python three_body.py          # -> out/video.mp4 + out/music_raw.wav (needs ffmpeg + Poppins/JetBrains Mono fonts in ./fonts)
TEST=0,600,1150 python three_body.py   # preview single frames as PNG
```
Then loudness-normalise to -14 LUFS and mux (ffmpeg `loudnorm`).
