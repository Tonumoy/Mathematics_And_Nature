# SC#6 narration v2: Kokoro af_heart (top-rated natural voice), sentence by sentence, natural pauses,
# speed 1.0 (v1 at 1.17 sounded rushed/robotic), light room tone + gentle compression afterwards.
import numpy as np, soundfile as sf, json, re
from kokoro_onnx import Kokoro
k = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin')
VOICE = 'af_heart'; SR = 24000
CH = [  # key, paragraph, pause after paragraph (s)  -- v3: no fillers/ellipses (TTS reads them flat), plain 12-year-old explanation
 ('HOOK', "Why isn't the sky violet?", 0.7),
 ('GUESS', "Most people think it's reflecting the ocean.", 0.35),
 ('NOPE', "Nope. Deserts have blue skies too.", 0.65),
 ('SCAN', "So what's really going on? Sunlight is every colour, mixed together.", 0.45),
 ('ANT', "Air is full of tiny molecules. When light hits one, it shakes, and flings that light off sideways.", 0.5),
 ('SIX', "Blue gets flung about six times more than red. So blue comes at you from every part of the sky.", 0.5),
 ('EQ', "It all comes down to one number. The power of four.", 1.1),
 ('WHY', "But wait. Violet gets flung even more. So why isn't the sky violet?", 0.45),
 ('EYE', "Two reasons. The Sun makes less violet light, and your eyes are bad at seeing it. So the mix looks sky blue.", 0.7),
 ('DIAL', "Now watch. Change that four to a zero.", 0.9),
 ('WHITE', "White sky! Every colour bounces the same. Clouds are white for a similar reason.", 0.6),
 ('SUN', "At sunset, light travels through thirty-eight times more air, so most of the blue bounces away before it reaches you.", 0.5),
 ('MARS', "And on Mars? Fine dust flips it. Sunsets there are blue.", 0.6),
 ('ASK', "So, what colour is the sky on the Moon?", 0.6),
]
SPEED = {'HOOK': 1.0, 'ASK': 1.0, 'EQ': 1.0}
def trim(a):
    nz = np.where(np.abs(a) > 0.008)[0]; return a[max(0, nz[0]-240):nz[-1]+1200]
out = [np.zeros(int(0.3*SR))]; t = 0.3; W = {}
for key, para, gap in CH:
    sents = re.split(r'(?<=[.?!])\s+', para); t0 = t
    for i, s in enumerate(sents):
        a, _ = k.create(s, voice=VOICE, speed=SPEED.get(key, 1.07), lang='en-us'); a = trim(a)
        out.append(a); t += len(a)/SR
        if i < len(sents) - 1:
            g = 0.26 if len(s) > 14 else 0.18; out.append(np.zeros(int(g*SR))); t += g
    W[key] = (round(t0, 3), round(t, 3)); out.append(np.zeros(int(gap*SR))); t += gap
y = np.concatenate(out); y = y/np.abs(y).max()*0.9
sf.write('vo_raw.wav', y, SR); json.dump(W, open('words.json', 'w'), indent=0)
print('dur', round(t, 2)); print(W)
