# SOURCE CODE #13 — Did AI Solve Navier-Stokes? The $1M Swirl in Your Chai

A cup of chai, a swirl of milk, and the million-dollar question behind it. Everything on screen is drawn by this code.

## What the code does
| Scene | Code | The maths |
|---|---|---|
| 3D chai cup, camera tilts to top-down | `cup3d.py` | numpy ray caster: cylinders, a saucer height-field, a torus handle, soft shadows, depth of field. The liquid surface is textured with the same simulation as the flat view, so the 3D shot lands exactly on it. |
| Milk swirling in the chai | `ns2d.py`, `chai_sim.py` | 2D incompressible Navier-Stokes in vorticity form, `∂ω/∂t + (u·∇)ω = ν∇²ω`, pseudo-spectral 512², 2/3 de-aliasing, Lawson RK4 (exact integrating factor for viscosity). Milk = a passive dye `∂c/∂t + (u·∇)c = κ∇²c`. |
| X-ray flow field + tracers | `xray.py`, `tracers.py` | vorticity glow + 6,000 tracer particles advected with RK2 through the solved velocity |
| One drop and its forces | `render.py` (`forces_at`) | pressure from the Poisson equation `∇²p = -(u_x² + 2u_y v_x + v_y²)`, viscous force `ν∇²u`, velocity `u`, all read from the solve at the drop's position |
| Wing | `wing_sim.py` | same solver + Brinkman penalisation of a NACA 0012 at 14°, smoke streaklines |
| Storm | `storm_sim.py` | same solver, one big vortex winding cloud bands |
| Blood | `others.py` | Poiseuille flow `u = U(1 − r²/R²)`, an exact Navier-Stokes solution, pulsing at 1.2 Hz |
| Stretched swirl | `tube.py` | Kelvin: circulation is conserved, so `ω r² = const` — thinner means faster |

## The equation
`ρ(∂u/∂t + (u·∇)u) = −∇p + μ∇²u` — change in motion + motion carrying itself = pressure push + stickiness. Newton's F = ma, for fluids.

## The $1M question (Clay Millennium Prize)
Can a smooth 3D flow spin up to infinite speed in a finite time ("blow up")? On 8 Sep 2026 OpenAI said an internal AI system produced a proof that it can when the fluid gets a smooth outside force. Mathematicians are still checking it and the Clay Institute has not awarded the prize. This Short says exactly that and no more.

## Run it
```bash
python3 chai_sim.py && python3 tracers.py && python3 wing_sim.py && python3 storm_sim.py
python3 audio/build_vo.py            # needs the ElevenLabs narration take as audio/vo.mp3
python3 audio/mix.py audio/musB.mp3 audio/mixB.wav
python3 render.py 0 1508 frames      # stateless per frame: split across cores with start/step
ffmpeg -framerate 30 -i frames/f%05d.png -i audio/mixB.wav -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac out.mp4
```
Narration: AI voice (ElevenLabs). Music: ElevenLabs Music (original).
