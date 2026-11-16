# SOURCE CODE #10 · Nobody Drew Tesla (reaction-diffusion)
YouTube Short: https://youtube.com/shorts/v_LVdL73nU8

A portrait of Nikola Tesla grown by the Gray-Scott reaction-diffusion model (Turing patterns). A photo only sets the local feed/kill rates (dark → thick maze, light → tiny dots); every pixel of the portrait is the simulated field.

- `gs.py` Gray-Scott step (9-point Laplacian)
- `psim2.py` full-resolution 1080×1920 portrait simulation (Du 0.35, Dv 0.175, f = 0.060 − 0.030·L, k = 0.060 + 0.003·L, 6,000 steps)
- `esim*.py` explainer sims (self-replicating spots, spots→stripes dial, fish/palate/digit/membrane vignettes)
- `rd.py` renderer (relief shading, 3D dish, bloom), `rend.py` 2-worker encode, `tl.py` timeline from voice clips, `mixext.py` voice + music mix, `thumbrd.py` cover
- `FACTS.md` sources, `SEO.md` title/description/tags

Narration: ElevenLabs (Daniel). Music: ElevenLabs Music.
