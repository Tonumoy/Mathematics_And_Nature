# 1 electron at a time... so why stripes? (weird physics #3)

Coded Cosmos Short, 31 Oct 2026 · [watch on YouTube](https://youtube.com/shorts/dlPTLv6AAKo) · 40 s · full-screen, Shorts-safe layout · dark cinematic score, all synthesised

Electrons are fired at two tiny gaps, one at a time. Each one lands on the screen as a single dot.
After 6,235 of them the dots form **stripes**, not two bands. Block one gap and the stripes vanish.
Open it again and some spots get **fewer** hits than with one gap open.

## The physics

| | value |
|---|---|
| electron energy | 600 eV |
| de Broglie wavelength | λ = h / √(2 m E) = 1.226 nm / √600 ≈ 50 pm |
| slit width / centre spacing | 62 nm / 272 nm |
| electrons in the build-up | 6,235 |

Sizes and electron count are from Bach, Pope, Liou & Batelaan, *Controlled double-slit electron diffraction*,
New J. Phys. **15**, 033018 (2013) ([arXiv:1210.6243](https://arxiv.org/abs/1210.6243)).

- **Top panel:** a real 2D time-dependent Schrödinger simulation (split-step Fourier, `wave_sim.py`): a wave packet hits a wall with two slits (or one, when the blocker slides in). Grid units, not to scale, but with the same width-to-spacing ratio as the experiment (d/w ≈ 4.4).
- **Screen:** each dot is drawn at random from the ideal far-field pattern

      P(θ) ∝ cos²(π d sinθ / λ) · sinc²(w sinθ / λ)        (two gaps)
      P(θ) ∝ sinc²(w sinθ / λ)                             (one gap)

  with `sinc(x) = sin(πx)/(πx)` (numpy's normalised sinc).
- **"This spot goes dark":** at the first dark stripe, for the same 6,235 electrons, ~4 land with two gaps open and ~170 with one gap open.
- Feynman called this experiment the one that holds "the only mystery" of quantum mechanics (*Feynman Lectures on Physics*, Vol III, ch. 1).

## Sound
Original score made in Python from the sim (`../_music/music_cinematic.py`), no samples:
sub drone, low strings, staccato bass ostinato, soft clock pulse, brass swells on the reveals and a 4-note cello motif.
Every pitched note follows the chord playing at that moment. The first single electrons each play a soft bell
on a chord tone chosen by where the dot landed; later crowds of hits become a quiet detector "rain".
Mixed to -14 LUFS, true peak below -1 dBTP.

## Run
```bash
pip install -r ../requirements.txt
# fonts: download Poppins (ExtraBold, Bold, SemiBold) and JetBrains Mono from Google Fonts into ./fonts/
python double_slit.py            # renders out/video_noaudio.mp4, scores the music, muxes out/double_slit.mp4
```
All parameters are globals at the top of `double_slit.py` and `wave_sim.py`.
