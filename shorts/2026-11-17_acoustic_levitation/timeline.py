# SC#12 voice lines -> times (measured if vo/Lxx.mp3 exists, else estimated)
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
LINES = {
 'L01': "Sound is holding this water.",
 'L02': "No strings. Just speakers, singing too high to hear.",
 'L03': "Waves from above and below crash together.",
 'L04': "Where they cancel, the pressure stops shaking. A quiet pocket.",
 'L05': "The drop gets pushed into it, and stays.",
 'L06': "There's a pocket every four millimetres. Wait for the stack.",
 'L07': "Now one number: loudness.",
 'L08': "Too soft, it falls. Too loud, it flattens and bursts.",
 'L09': "Chemists use this to make medicine, with nothing touching it. Nature is running on code.",
 'L10': "Next: is the swirl in your chai worth a million dollars?",
}
GAP = {'L01': 0.5, 'L02': 0.3, 'L03': 0.3, 'L04': 0.3, 'L05': 0.4, 'L06': 2.0, 'L07': 0.4, 'L08': 1.3, 'L09': 0.4}

def dur(k):
    p = os.path.join(HERE, 'vo', k + '.mp3')
    if os.path.exists(p):
        return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]))
    w = len(LINES[k].split())
    return w / 2.65 + 0.25 + 0.2 * LINES[k].count(".")

TL = {}; t = 0.0
for k in LINES:
    d = dur(k); TL[k] = (t, t + d); t += d + GAP.get(k, 0.25)
T = round(TL['L10'][1] + 1.0, 2)
s = lambda k: TL[k][0]; e = lambda k: TL[k][1]

if __name__ == '__main__':
    for k, (a, b) in TL.items(): print(k, round(a, 2), round(b, 2))
    print('T', T)
