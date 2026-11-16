# SOURCE CODE #11 · Sound Drew This (Chladni plate / cymatics)

YouTube Short: https://youtube.com/shorts/UqvRBOa85fE · Instagram Reel @tonumoy_mukherjee (16 Nov 2026, 7:30 PM IST)

Sand on a vibrating steel plate (24 cm × 24 cm × 1 mm), driven at its centre. Every pattern comes from a real solver:

- `plate.py` solves the free-edge Kirchhoff plate, D∇⁴w = ρh·ω²·w, with the Ritz method (27×27 Legendre basis, Gauss quadrature, generalised eigenproblem). First frequency parameters 13.47, 19.60, 24.27, 34.80 match Leissa (1969).
- `sim.py` drives the plate at a chosen frequency (sum of real modes, Q = 28) and lets 80,000 grains hop with a step set by the local vibration amplitude, so they settle on the nodal lines.
- `render.py` draws the plate in perspective at 2× (homography-warped texture, x-ray mode mesh, thickness dial, HUD), downscaled with LANCZOS. `render_all.py` renders the full video in two workers.
- `timeline.py` times every scene to the measured voice clips. `mix.py` ducks the music 8 dB under the voice.
- `thumb.py` rebuilds the cover from the Short's own render.

Twist: frequency grows linearly with thickness (ω ∝ h), so a twice-as-thick plate needs double the pitch for the same shape.

Run: `python3 plate.py` (check values) → `python3 sim.py` → `HOOKCAM=A python3 render_all.py`. Needs numpy, scipy, opencv-python, Pillow, ffmpeg, and Poppins fonts in `fonts/`.

Narration: AI voice (ElevenLabs). Music: ElevenLabs Music. Sources in FACTS.md.
