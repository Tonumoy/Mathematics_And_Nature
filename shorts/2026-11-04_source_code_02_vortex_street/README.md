# SOURCE CODE #2 — Who drew these swirls? (von Kármán vortex street)

A cloud deck flows past an island. Behind it the air sheds swirls left, right, left: a von Kármán vortex street, visible from space.

- `sim.py` — D2Q9 lattice-Boltzmann (BGK) solver, 384 × 216 cells, island D = 28 cells, U = 0.1, Re = 150; measures the Strouhal number from a wake probe (St = 0.19).
  The twist branch forks the run and sets Re = 30 (air 5× more viscous): shedding stops.
- `readvect.py` — passive cloud deck (cellular stratocumulus texture) advected semi-Lagrangian by the solved wind, plus a simple lee-clearing term.
- `street.py` — renders the satellite view (2× supersampled), the x-ray vorticity contours, text (SOURCE CODE kit) and the timeline.
- `mix2.py` + `music_cinematic.py` + `source_code_kit.py` — dark cinematic score from the shedding events, fixed series motif.

Run: `python sim.py` (needs sched.json) → `python readvect.py` → `python street.py video 0 1256` → `python mix2.py`.

Honest notes: the twist branch is shown at 5× solver speed; returning to Re 150 resumes the main run from the fork state. The satellite view is rendered from the simulation, not footage.
Sources: Williamson (1996) Annu. Rev. Fluid Mech. 28; Strouhal (1878); NASA Earth Observatory island vortex streets.

## v2 (voice + real footage)
- `street_vo.py` — 57.5 s timeline driven by the ElevenLabs narration (`vo.mp3`, Daniel, eleven_v4); real-layer plates from NOAA's Guadalupe Island clip (rotated 90°, 0.575× UHD Lanczos, slowed 0.32× with frame blending); x-ray = `street.py` solver layer.
- `mix_vo.py` — score + narration, music ducked 6.5 dB under the voice, reveal hit voice-free.
- Footage credit: NOAA/NESDIS (public domain).
