# SOURCE CODE #5 · Where's the blue? (Diwali, 8 Nov 2026)

YouTube Short by **Coded Cosmos** (@cosmoscoded): why blue fireworks are so rare.

## The code (real, computed every frame)
| file | what it computes |
|---|---|
| `spectra.py` | Planck's law B(λ,T) → CIE 1931 2° colour matching → sRGB; molecular emission bands (SrCl, BaCl, CuCl, Na D) → sRGB; colour of one wavelength |
| `sc5.py` | the Short: real footage layer, Planck temperature ramp (2,000 K gold → 5,800 K white → 100,000 K pale sky-blue), quantum ladder `λ = 1239.84 nm·eV / ΔE`, CuCl flame dial (blue window ≈ 1,000–1,200 °C) |
| `mix5.py` | original score from `score_engine.py` (F♯ phrygian, i–III–VII–iv, analog pad, clock pulse, horn lead, shimmer) + ducked narration |
| `vo_build.py` | pause tightening + 1.1× tempo of the ElevenLabs narration, word-time map |
| `timeline.json` | every beat, synced to the narration words |

## Key numbers
- 2,000 K → xy (0.527, 0.413) gold · 5,800 K → (0.326, 0.335) white · 100,000 K → (0.243, 0.238) pale blue-white. Heat alone never makes deep blue.
- 2.0 eV → 620 nm red · 2.36 eV → 525 nm green · 2.9 eV → 428 nm blue · 3.5 eV → 354 nm UV (invisible).
- Blue emitter CuCl falls apart above ~1,200 °C; firework flames often run at 1,700–2,000 °C.

## Sources
NIST ASD (Na D lines) · Conkling & Mocella, *Chemistry of Pyrotechnics* · PBS NOVA "The science of fireworks" · Tarantik & Klapötke, *Propellants Explos. Pyrotech.* · CIE 1931 via `colour-science`.

## Images
- Opening/closing footage: "Vibrant Nighttime Fireworks Display in Sky" by Ankit Rainloure, Pexels (free Pexels licence; not redistributed here)
- "Radha and Krishna watching fireworks in the night sky", Sitaram, Kishangarh, late 18th c., National Museum New Delhi 63.810 (public domain, Wikimedia Commons)
- Cover: AI image (gpt-image-2 via ElevenLabs), stylised art.

Narration: AI voice (ElevenLabs, Daniel). Music generated from code, no samples.
