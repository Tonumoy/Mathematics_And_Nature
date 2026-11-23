# SOURCE CODE #15 · The Riemann Hypothesis (Coded Cosmos, 23 Nov 2026)

Every frame is drawn by this code from real mathematics:
- `data.py` / `fine.py`: |ζ(σ+it)| on the critical strip and the first 600 non-trivial zeros (mpmath).
- `render.py`: 1/|ζ| landscape (every spike is a zero), the prime staircase ψ(x), and its rebuild from the zeros via von Mangoldt's explicit formula
  ψ(x) = x − Σρ x^ρ/ρ − log 2π − ½ log(1 − x⁻²). The "zero pushed off the line" is an illustration (the zero and its mirror move to β, 1−β).
- `mix.py`: voice + music + SFX mix (−14 LUFS). `cover.py`, `cover_b.py`: thumbnails. `story.py`: Instagram story.

Run: `python data.py && python fine.py 0 && python fine.py 1` then `./go.sh 2 17 out.mp4` (2 = 4K).
Narration and music: ElevenLabs. News facts: Lean community statement (21 Nov 2026).
