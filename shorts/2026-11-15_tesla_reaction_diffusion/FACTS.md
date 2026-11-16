# FACTS · SOURCE CODE #10 · Tesla drawn by reaction-diffusion (15 Nov 2026)
| On screen / spoken | Value | Source |
|---|---|---|
| Model | Gray-Scott: ∂u/∂t = Du∇²u − uv² + f(1−u); ∂v/∂t = Dv∇²v + uv² − (f+k)v; Du = 2·Dv | Pearson 1993, Science 261:189 "Complex patterns in a simple system"; gs.py |
| "spark copies itself / eats food / food spreads faster (2×)" | v autocatalytic (uv²), consumes u; Du=1.0, Dv=0.5 | gs.py |
| "starves the space around it" | u depleted around each v spot (dark halo in dish = blurred, amplified 1−u field) | rd.py field_tex |
| one spark copies itself | f=0.0367, k=0.0649 (self-replicating spots), 1 seed, 36,000 steps | esim2.py |
| dial: how fast sparks die (k) → stripes | f fixed 0.0367, k 0.0649→0.0600 | fix run (rd session) |
| shells | Meinhardt, The Algorithmic Beauty of Sea Shells (Springer 1995) | |
| Turing 1952 | A. M. Turing, "The Chemical Basis of Morphogenesis", Phil. Trans. R. Soc. B 237:37-72 (1952) | |
| fish stripes (sim seeded with bands) | Kondo & Asai 1995, Nature 376:765 (angelfish Pomacanthus) | |
| palate ridges | Economou et al. 2012, Nature Genetics 44:348 (mouse rugae, FGF/SHH) | |
| digits | Raspopovic et al. 2014, Science 345:566 (Bmp-Sox9-Wnt Turing network, mouse) | |
| water filters | Tan et al. 2018, Science 360:518 "Polyamide membranes with nanoscale Turing structures for water purification" | |
| Tesla portrait | Gray-Scott on a full 1080×1920 grid, Du 0.35 / Dv 0.175, f = 0.060 − 0.030·brightness, k = 0.060 + 0.003·brightness (dark → thick maze, light → small dots), 0.8% random seeds, 6,000 steps; photo supplied by the owner (Napoleon Sarony c.1890, public domain) used only as the k map; every pixel of the portrait is the simulated v field; the dot-map scene (55-57 s) shows the photo's brightness as dots | psim.py |
| Simplified | vignettes (fish/palate/fingers/membrane) are Gray-Scott in simple shapes, illustrating the published mechanisms, not simulations of those exact systems; palate/finger sims start from seeded lines | esim3/4.py |
