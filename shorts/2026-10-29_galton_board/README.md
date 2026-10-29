# Galton board: where do 5,000 balls land? (hypnotic physics #2)

Coded Cosmos Short, 29 Oct 2026 · format: prediction challenge · look: blueprint · music: Brazilian-funk-style 130 BPM, all synthesised

5,000 balls fall through 12 rows of pegs. Each peg sends a ball left or right, 50/50 (ideal board).
The bin a ball lands in is its number of "rights", so

    P(k) = C(12, k) / 2^12        (4,096 equally likely paths)

| | value |
|---|---|
| middle bin (k = 6) | 924 / 4096 = 22.6 % (~1,128 of 5,000) |
| each edge bin | 1 / 4096 (~1.2 of 5,000) |
| normal limit | mean 6, SD √12 / 2 ≈ 1.73 bins (de Moivre–Laplace) |
| this run (seed 29) | 0, 8, 85, 241, 585, 1005, 1097, 1035, 582, 283, 61, 14, 4 |

The bars are counted from the simulation, not typed in. The dashed curve is the formula.
Galton described the board in *Natural Inheritance* (1889).

## Sound
Every sound is generated in Python from the sim, no samples:
- each peg hit = a tiny click (ASMR layer), panned by the ball's x position
- each landing = a glass ping pitched by its bin (max one per 16th note, at the exact landing time)
- an edge-bin landing = a big hit; 1,000 / 2,500 / 5,000 landed = chord stabs
- beat: `funk` preset in `_music/genre_engine.py` (16-step kick/tom pattern, clap on 2 and 4, 808, vowel-chop hook)

## Run
```
python galton_board.py sheet   # 6-frame preview
python galton_board.py all     # audio + video + <10 MB upload copy
```
Needs numpy, scipy, matplotlib, Pillow, ffmpeg, and `../_music/genre_engine.py`.
