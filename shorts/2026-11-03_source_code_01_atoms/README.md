# SOURCE CODE #1 — You've drawn atoms wrong (3 Nov 2026)
Coded Cosmos Short. Real thing: a hydrogen discharge lamp. X-ray: one hydrogen atom, solved.

- `sim.py` — real radial Schrödinger solver for hydrogen (finite differences, 6,000 points, atomic units). Gives E1 = −13.61 eV, E2 = −3.40, E3 = −1.51 (exact −13.6/n²). `shoot()` = Numerov integration at a non-allowed energy (n = 2.5): the solution blows up. `sample()` draws electron "measurements" from |ψ|² (radial part from the solver, angular part |Y_l0|²).
- `atoms.py` — renderer (procedural lamp at 2x + LANCZOS/area downscale, x-ray layer, cloud splats, captions).
- `source_code_kit.py` — the locked SOURCE CODE series kit (text kit, colours, 4-note cello motif D3 F3 Bb2 A2 + x-ray swell).
- `mix.py` — score via `_music/music_cinematic.py` + kit motif; every sound is a formula, no samples.

Facts: H-alpha 656.28 nm (air, NIST ASD); solver 3→2 gap 1.89 eV → 656 nm. Classical planet atom collapse ≈ 1.6 × 10⁻¹¹ s (Larmor radiation; Griffiths).
Run: `python atoms.py render 0 1331` then `python mix.py`.
- `atoms_vo.py` — the posted version: time-warps the 44 s timeline onto the ElevenLabs narration (voice "Daniel - Calm, Warm and Professional"), music ducked 6.5 dB under the voice, reveal hit kept voice-free. 57.5 s, -14 LUFS.
Live: https://youtube.com/shorts/oVWZ-HN0eKI
