# Pendulum wave — hypnotic physics #1

20 simple pendulums, cut so that pendulum *n* makes exactly **20 + n** full swings in 30 seconds.
Seen from above they draw snakes and braids, split into 6, 5, 4, 3 and 2 groups, and at 30 s all 20 are back in step. The video is one full cycle, so it loops forever.

| | |
|---|---|
| Period | T_n = 30 s / (20 + n) |
| Length | L_n = g T_n² / 4π² → 55.9 cm (n = 0) down to 14.7 cm (n = 19) |
| Groups | at t = 30/q s the phases fall into q groups (7.5 s → 4, 10 s → 3, 15 s → 2) |

Small-angle simple pendulums, no air drag.

## Sound
Every bob reaching its right-hand turning point (t = k·T_n) plays one glass ping, pitched by its row, at the exact sample. When the pendulums line up, the pings stack into a chord. Group moments add chord hits, and the filter opens as the cycle heads to the re-sync. The beat (hard techno, 140 BPM) is synthesised by `_music/genre_engine.py`. No samples.

## Run
```
pip install numpy scipy matplotlib pillow
python pendulum_wave.py          # renders pendulum_wave.mp4 + upload copy (needs ffmpeg)
python pendulum_wave.py sheet    # 6-frame preview only
```
Needs `shorts/_music/genre_engine.py` from this repo (the shared music engine; this Short adds its hard techno preset).

## Sources
- J. A. Flaten & K. A. Parendo, "Pendulum waves: A lesson in aliasing", *Am. J. Phys.* 69, 778 (2001).
- Harvard Natural Sciences Lecture Demonstrations, "Pendulum Waves".
