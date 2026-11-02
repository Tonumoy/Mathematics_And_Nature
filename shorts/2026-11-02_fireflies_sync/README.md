# 300 fireflies. Who's the leader? Nobody. (2 Nov 2026)

YouTube Short: https://youtube.com/shorts/jvJfKHngcX0

One firefly lands on a mangrove tree, then 299 more arrive from all directions. They blink at random, then fall into one shared rhythm with no leader. Switch the nudge off and they drift apart; switch it on and they find each other again.

## The maths
Each firefly is a pulse-coupled integrate-and-fire oscillator (Mirollo & Strogatz, SIAM J. Appl. Math. 50:1645, 1990):
- inner clock x = f(φ) = ln(1 + (e^b − 1)φ)/b, concave down (b = 3); when x reaches 1 the firefly flashes and resets to 0;
- every firefly that sees the flash (within 120 px) jumps ahead: **x → x + ε** (ε = 0.07, divided by the average number of neighbours; capped at 1, so it may flash too);
- clocks differ by ±2% (an assumption, so they are not perfect clones). With ε = 0.01 they drift apart.
- "in step" on screen = order parameter r = |mean of e^(2πiφ)|.

Exact event-driven integration (no time step), `sim.py`. Background reading: J. Buck, Q. Rev. Biol. 63:265 (1988); C. Peskin (1975) pacemaker-cell model.

## Files
- `sim.py` — the solver
- `fireflies_sync.py` — renderer (camera, insect fireflies, x-ray clocks, equation card, captions; 1080×1920, 30 fps)
- `mix.py` — places the narration lines and ducks the music under the voice

Background tree: AI image (ElevenLabs, Seedream). Narration: AI voice (ElevenLabs). Music generated in Python (`_music/music_cinematic.py`).
