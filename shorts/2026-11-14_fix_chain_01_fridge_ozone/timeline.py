# timeline: voice lines -> scene windows (measured if vo/Lxx.mp3 exists, else estimated)
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
LINES = {
 'L01': "Your fridge once punched a hole in the sky.",
 'L02': "And it all started with rotting food.",
 'L03': "Bacteria can double every twenty minutes. In ten hours, one becomes a billion.",
 'L04': "Fix: cold. Chill them, and they almost stop.",
 'L05': "But ice melts. So we built a machine. A gas boils inside the pipes, and carries the heat outside.",
 'L06': "Problem: that gas was ammonia, or sulfur dioxide. Leaks killed families in their sleep.",
 'L07': "Fix: 1928. Thomas Midgley makes Freon. Not toxic. Won't burn. He breathed it in, and blew out a candle.",
 'L08': "Problem: Freon is too stable. It drifts up for years, until sunlight rips a chlorine atom off.",
 'L09': "And one chlorine atom can wreck over a hundred thousand ozone molecules.",
 'L10': "1985. Scientists find a hole over Antarctica.",
 'L11': "Fix: 1987. Every country on Earth signs the Montreal Protocol. Freon gets banned.",
 'L12': "Problem: the new gas, HFC, spares the ozone. But it traps over fourteen hundred times more heat than CO2.",
 'L13': "Fix: Kigali, 2016. Most new fridges now run on isobutane. Its heat score? Just three.",
 'L14': "The hole should heal by about 2066. Every fix made a new problem. And science fixed that too.",
 'L15': "Next chain: how did we beat the dark?",
}
GAP = {'L01': 0.15, 'L04': 0.35, 'L06': 0.3, 'L08': 0.2, 'L09': 0.3, 'L11': 0.3, 'L13': 0.35, 'L14': 0.3}

def dur(k):
    p = os.path.join(HERE, 'vo', k + '.mp3')
    if os.path.exists(p):
        return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]))
    w = len(LINES[k].split())
    return w / 2.65 + 0.25 + 0.2 * LINES[k].count(".")

TL = {}; t = 0.0
for k in LINES:
    d = dur(k); TL[k] = (t, t + d); t += d + GAP.get(k, 0.2)
T = TL['L15'][1] + 1.0
s = lambda k: TL[k][0]; e = lambda k: TL[k][1]
SCENES = [  # (name, start, end, kwargs-relative builder)
 ('hook',    0.0,       s('L02')),
 ('bacteria', s('L02'), s('L05')),
 ('machine', s('L05'),  s('L07')),
 ('freon',   s('L07'),  s('L08')),
 ('rise',    s('L08'),  s('L09')),
 ('cycle',   s('L09'),  s('L10')),
 ('y1985',   s('L10'),  s('L11')),
 ('montreal', s('L11'), s('L12')),
 ('heat',    s('L12'),  s('L13')),
 ('iso',     s('L13'),  s('L14')),
 ('end',     s('L14'),  T),
]
if __name__ == '__main__':
    for k, (a, b) in TL.items(): print(k, round(a, 2), round(b, 2))
    print('T', round(T, 2))
