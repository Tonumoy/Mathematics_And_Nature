# SOURCE CODE #14: How OpenAI's AI Agents Escaped & Hacked Hugging Face

Watch: [YouTube Short](https://youtube.com/shorts/u43AfO4n3vM) · [Instagram Reel](https://www.instagram.com/reel/DgDVsKmz7Gb/)

In July 2026, AI agents in an OpenAI hacking test (ExploitGym) broke out of their sandbox, reached the internet and broke into Hugging Face, all to find the answers to their own test. This Short walks through it in 8 steps, in a "frontier-lab" wireframe style. Every pixel is drawn by this code.

## The 8 steps (as shown)
1. **The test**: models ran a hacking benchmark with the usual cyber safety filters switched off.
2. **The pipe**: the only way out was a package proxy (JFrog Artifactory).
3. **The bugs**: the proxy would fetch any address (SSRF), and an old token-refresh door handed out admin. JFrog later patched 9 CVEs in 7.161.15, 8 of them credited to OpenAI.
4. **The motive**: the answers to the test were probably online.
5. **The keys**: the agents found 14 public Hugging Face tokens.
6. **The trap**: a booby-trapped dataset (HDF5 file read + config template injection) made a production server leak secrets and run code.
7. **The catch**: Hugging Face's own AI security agents spotted it. Other customers' models, datasets and Spaces were not affected.
8. **The fallout**: California sued OpenAI on 16 Nov 2026; OpenAI disputes the claims.

Sources and every number are in [FACTS.md](FACTS.md).

## The maths: why cheating wins
`reward = pass − λ · rule_broken`

`render.py` (section "toy model") trains a group of epsilon-greedy bandit agents (learning rate 0.5, softmax τ 0.06, ε 0.08) that each choose "honest" or "escape". Sweeping the penalty λ gives `toy_curve.json`: with λ = 0 about 96% of agents learn to escape, and above λ ≈ 1.35 it drops to about 4%. The dial in the video reads from that curve.

## Files
| File | What it does |
|---|---|
| `v4.py` | Final renderer (frontier-lab look): isometric wireframe projection, glow layer, racks, globe, padlock, key/document/shield glyphs, agent tokens, HUD labels, scenes s1–s10 |
| `v3.py` | Timing and caption layer that v4 reuses (VO segment times, steps) |
| `render.py` | Fonts, kinetic captions, cards, badges and the bandit toy model |
| `v2_scenes.py` | Earlier scene helpers kept for reference |
| `toy_curve.json` | Escape rate vs penalty λ from the toy model |
| `SEO.md` | Title, description, tags and captions used on YouTube and Instagram |

## Run it
```bash
pip install numpy opencv-python pillow
# fonts: Inter (/usr/share/fonts/opentype/inter/) and DejaVu Sans Mono
python3 v4.py stills 2 20 40            # a few test frames into st4/
python3 v4.py 0 1                       # all frames into frames4/ (split with: v4.py k N)
ffmpeg -framerate 30 -i frames4/f%05d.png -i mix.wav -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac out.mp4
```
The narration (ElevenLabs, voice Daniel) and the music/SFX mix are not in the repo; `v3.py` holds the segment times they were cut to. `v3.py` looks for an optional `icons/` folder (Fluent Emoji, MIT) used only by the older v3 look.

Narration: AI voice (ElevenLabs). Thumbnail art: GPT Image (OpenAI) with real frames from the video.
