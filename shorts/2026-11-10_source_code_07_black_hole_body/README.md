# SOURCE CODE #7 — What happens if you fall into a black hole?

YouTube Short: https://youtube.com/shorts/ilFplYer3aI (Coded Cosmos, 10 Nov 2026)

Not crushed, **stretched**: your feet are closer to the hole than your head, so they're pulled harder. Every pixel is rendered by this code.

| file | what it does |
|---|---|
| `sim7.py` | solver: radial geodesics in Schwarzschild (d²r/dτ² = −GM/r²) for 10 body pieces; spring-chain body in static balance with the tidal field 2GM/r³, snaps above 30 % strain; prints the edge numbers (10 M☉: 19 million g; Sgr A*: 0.0001 g) |
| `sprite.py` | 3D astronaut as a signed-distance field, sphere-traced once → normal / material / AO maps |
| `astro.py` | relights the astronaut each frame (disk key light, rim, visor reflection) and stretches it row by row from the solver's piece positions (taffy remap with necking and tears) |
| `sc7b.py` | the Short: ray-traced black hole (SC#3 `bh.py` + `sim.py` photon table), x-ray tidal field, mass dial, captions timed to the voice clips |
| `mix7b.py` | new score from `_music/score_engine.py` + voice track, music ducked under the voice |
| `SEO.md` | title, description, sources, tags, pinned comment |

Run: `python sim.py` (SC#3 folder, makes `table.npz`) → `python sim7.py` → `python sprite.py` → `python sc7b.py render 0 1224`.

Sources: Misner, Thorne & Wheeler, *Gravitation* §31.2 · EHT Collaboration 2022, ApJL 930 L12 · GRAVITY Collaboration 2019, A&A 625 L10.
Narration: AI voice (ElevenLabs). Music generated from code.
