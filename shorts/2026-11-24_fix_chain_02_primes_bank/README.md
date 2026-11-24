# FIX CHAIN #2 — 2 Numbers Guard Your Money 🔐 Until Quantum Comes

**YouTube Short:** https://youtube.com/shorts/sk_XmHv46WU (24 Nov 2026, 7:30 PM IST) · also on Instagram @tonumoy_mukherjee

2,000 years of locks as one chain — every fix created a new problem:
Caesar's shift (+3) → cracked by counting letters (al-Kindi) → long secret keys → who delivers the key? → RSA (1977): a padlock anyone can close, only you can open, built from two primes (61 × 53 = 3233) → RSA-2048, 617 digits, never factored → a big enough quantum computer could break it (Gidney 2025 estimate: < 1 M noisy qubits, < 1 week; no such machine yet) → NIST 2024 lattice-based ML-KEM.

## How it is made
Every pixel is drawn by `render.py`: a hand-built pixel-art world on a 270×480 canvas (PIL, no anti-aliasing), 14-colour palette, ordered-dither skies, nearest-neighbour upscale to 4K, light-source bloom, vignette and faint scanlines. Scenes are timed to the measured voice clips (`vo_durs.json`). Real Caesar cipher, real English letter frequencies (shifted by 3), real RSA toy numbers.

```bash
pip install pillow numpy opencv-python
python render.py 4 stills 0.3,20,45     # stills at 1080x1920
python render.py 8 video | ffmpeg -f rawvideo -pix_fmt rgb24 -s 2160x3840 -r 30 -i - -c:v libx264 -crf 18 out.mp4
python mix.py music_a mix.wav            # voice + music (ducked) + SFX, 2-pass loudnorm -14 LUFS
```
Audio files (ElevenLabs voice, music and SFX) are not included.

Files: `render.py` (picture), `mix.py` (audio mix), `vo_lines.txt` (script), `FACTS.md` (sources for every claim), `SEO.md`, `thumbnail.jpg`.

Narration: AI voice (ElevenLabs). Music: ElevenLabs Music (instrumental).
