# POV: you're a raindrop (Weird Physics #2)

YouTube Short: Coded Cosmos (@cosmoscoded), 27 Oct 2026.

Four drops leave a cloud 2 km above Kolkata rooftops at the same moment.

| drop | top speed | lands after |
|---|---|---|
| no air (ghost) | 198 m/s = 713 km/h | 20.2 s |
| 5 mm | 9.09 m/s = 32.7 km/h | 3 min 41 s |
| 2 mm | 6.49 m/s = 23.4 km/h | 5 min 9 s |
| 1 mm | 4.03 m/s = 14.5 km/h | 8 min 17 s |

## The maths
- Quadratic drag: m dv/dt = mg − ½ρC_dAv². With v_t the terminal speed this solves exactly:
  v(t) = v_t tanh(gt/v_t),  z(t) = H − (v_t²/g) ln cosh(gt/v_t).
- No air: z = H − ½gt², impact speed √(2gH).
- Terminal speeds are the measured values of Gunn & Kinzer (1949), *J. Meteorology* 6, 243–248.
- Drop shape: axis ratio b/a = 1.03 − 0.062·d (d in mm), Pruppacher & Beard (1970), *QJRMS* 96, 247–256. Big drops are flattened buns, not tears.
- Time is sped up x2.4 while the ghost falls, then x14. The on-screen clock always shows real time.
- Terminal speeds are sea-level values held for the whole fall; air is thinner aloft, so real drops land a little sooner (times are "about").

## The music
Original and code-generated (`_music/genre_engine.py`, raga mode): tanpura drone, tabla pulse and a santoor lead on the notes of raag Megh (S R M P n), the monsoon raag. Every sound event comes from the sim: each 100 km/h the ghost passes is a rising note, the ghost's impact is the drop, each 250 m "you" fall is a tabla tick, and every landing is a note (bigger drop = lower note). No samples.

## Run
```
pip install numpy scipy pillow matplotlib   # plus ffmpeg on PATH
MUSIC_DIR=../../_music python monsoon_raindrop.py          # full render
python monsoon_raindrop.py --sheet                          # 6-frame preview only
```
Outputs: `monsoon_raindrop.mp4`, `monsoon_raindrop_upload.mp4`, `preview_frames.jpg`, `beats.json`, `audio_levels.txt`.
