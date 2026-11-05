# SOURCE CODE #3 — Would a black hole eat Earth? (5 Nov 2026)

Coded Cosmos Short. Swap the Sun for a same-mass black hole: Earth keeps its orbit. What the hole does eat is light that comes too close.

**Solver (`sim.py`)** — photon orbits in Schwarzschild spacetime (G = c = M = 1):

    u'' + u = 3 M u²,   u = 1/r

integrated with SciPy RK45 (rtol 1e-9). Checks: capture edge b = 3√3 M = 5.196 (5.20 escapes, 5.19 is captured); weak-field bend matches 4M/b + 15πM²/4b² (b = 100: 0.04122 vs 0.04118).

**Opening shot (`bh.py`, `build_real`)** — every pixel is a backward ray from the same solver (lookup table `U[b, φ]`): lensed star map for escaping rays, black for captured rays, thin disk (r = 6–19 M, inclination 84°) with Doppler + gravitational shift. The disk is illustrative.

**X-ray layer** — real ray paths drawn from `sim.path_xy`, red once captured. Dial: aim b = 5.30 (escapes) → 5.15 (trapped) → 5.25 (escapes).

**Music** — `mix3.py` + `music_cinematic.py`: every sound is synthesized from formulas, no samples. Each ray launch triggers a note.

Run: `python sim.py` (builds table.npz) → `python bh.py sheet 0.5,20` or `python bh.py render 0 1275` → `python mix3.py`.

Sources: EHT Collaboration 2019, ApJL 875 L1/L6; Misner, Thorne & Wheeler, *Gravitation* ch. 25; Luminet 1979, A&A 75, 228.

**v2 (owner 5 Nov):** ElevenLabs narration (Daniel, eleven_v4, one take) with the picture re-timed to the voice (`vo_build.py` → `warp.json`, `bh.py renderw`), music re-composed on the warped beat grid and ducked 6.5 dB under the voice (`mix_vo.py`). Earth (`earth.py`: real continents from the `global-land-mask` package, procedural clouds, atmosphere) spirals in during the viewer's guess, then just orbits (sped up: 1 year in 10 s) and goes cold and dark. Cyan ring marks the black hole.
