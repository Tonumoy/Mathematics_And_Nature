# SOURCE CODE #6 · Why is the sky blue, not violet? (9 Nov 2026)

YouTube Short by **Coded Cosmos** (@cosmoscoded): https://youtube.com/shorts/zDPeYO3_Fa4

## The code (real, computed)
| file | what it computes |
|---|---|
| `sky.py` | single-scattering radiative transfer through a spherical atmosphere: β(λ) = (τ₅₅₀/H)(550/λ)ⁿ, τ₅₅₀ = 0.097, H = 8 km, Rayleigh phase 3/16π(1+cos²θ), sunlight = Planck 5778 K → CIE 1931 (Wyman–Sloan–Shirley fit) → sRGB. `n = 4` is Rayleigh; the dial in the Short sweeps n 4 → 0 (white sky) → 4 |
| `sc6.py` | the Short: real CC BY sky timelapse, photon Monte Carlo (scatter chance per step ∝ (550/λ)⁴, new direction from the 1+cos²θ phase function), spectrum × eye panel, computed-sky dial, real Yosemite sunset, NASA Curiosity Mars sunset |
| `mix6.py` | original score from `score_engine.py` (G♯ aeolian, i–v–VI–iv, 80 BPM) + ducked narration |
| `vo_build.py` | narration (local Kokoro TTS), sentence-level pauses, beat times |
| `cover.py` | the designed cover (real plate + x-ray half) |

## Key numbers
- (700/450)⁴ = 5.9 → blue scatters 5.9x more than red; (700/400)⁴ = 9.4 for violet
- τ(400/550/700 nm) = 0.347 / 0.097 / 0.037
- scattered sunlight peaks at the violet end, but weighted by the eye's blue cone it peaks near 440 nm → sky blue
- horizon air mass ≈ 38

## Sources
Bodhaine et al. 1999 (J. Atmos. Oceanic Technol. 16:1854) · Kasten & Young 1989 · NASA/JPL-Caltech/MSSS PIA19400 · footage: Brainandforce (CC BY 4.0), G. Edward Johnson (CC BY 4.0), Wikimedia Commons
