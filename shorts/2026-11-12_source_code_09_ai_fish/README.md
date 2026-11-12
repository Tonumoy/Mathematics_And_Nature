# SOURCE CODE #9: AI Fish vs AI Sharks: They Evolved a School

Short: https://youtube.com/shorts/DAJ4E1GcTwI (12 Nov 2026)

200 fish, each with a 13-neuron brain (4 senses, 6 hidden, 3 muscles), and 3 sharks with small evolving brains of their own.
When a fish is eaten, a survivor's brain is copied in with small random mutations. After 9 generations the fish swim as one school.
Why it pays: predator confusion. A strike succeeds with probability 1 / (number of fish within 6 units of the target).

## Files
- `evo.py`: the 3D world and the evolution loop (`python evo.py 200000 3`).
- `evo_cont.py`: runs the evolved population 60,000 more steps with the startle burst (R_STARTLE 5.5, BURST 0.8, VPRED 1.4) and writes `evo_s7.pkl`.
- `evo_s7.pkl`: brain snapshots per generation + the log behind the on-screen graph.
- `sc9v2.py`: story clock, camera, brain inset, graph, UI. `sc9v3.py`: look (ocean, shaded fish, bloom), voice timeline, full render.
- `shark3d.py`: 3D shark mesh with hinged jaw.
- `mix3.py`: score (`_music/score_engine.py`), narration and catch sounds.

## Honest notes
- The schooling is evolved by the brains. The startle burst is a body reflex added for the final generations.
- For the camera: in generation 1 the lead shark is steered toward the nearest fish, and in the final scenes one shark is sent on a fast lunge through the school.
- Narration: AI voice (ElevenLabs). Music generated in code (C dorian, i-iv-VII-III).

## Sources
- Olson, Hintze, Dyer, Knoester, Adami (2013), "Predator confusion is sufficient to evolve swarming behaviour", J. R. Soc. Interface 10: 20130305
- Landeau & Terborgh (1986), "Oddity and the 'confusion effect' in predation", Animal Behaviour 34: 1372-1380
