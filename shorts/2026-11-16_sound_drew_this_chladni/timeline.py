# voice lines -> times (measured if vo/Lxx.mp3 exists, else estimated)
import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
LINES = {
 'L01': "Sound drew this.",
 'L02': "Sand on a steel plate. One note, and it runs to the lines that never move.",
 'L03': "Wait for the highest note.",
 'L04': "Underneath: one equation for a bending plate. Our code solved it.",
 'L05': "Each note it likes is a shape. Every other note? A mess.",
 'L06': "Higher note, more lines.",
 'L07': "Now one number. Make the plate twice as thick.",
 'L08': "Same note, it breaks. Twice as thick needs double the pitch.",
 'L09': "Violin makers tune wood with this trick. Nature is running on code.",
 'L10': "Next: can sound lift water?",
}
GAP = {'L01': 0.6, 'L02': 0.3, 'L03': 0.4, 'L04': 0.4, 'L05': 0.2, 'L06': 2.0, 'L07': 0.8, 'L08': 0.7, 'L09': 0.3}

def dur(k):
    p = os.path.join(HERE, 'vo', k + '.mp3')
    if os.path.exists(p):
        return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]))
    w = len(LINES[k].split())
    return w / 2.65 + 0.25 + 0.2 * LINES[k].count(".")

TL = {}; t = 0.0
for k in LINES:
    d = dur(k); TL[k] = (t, t + d); t += d + GAP.get(k, 0.25)
T = round(TL['L10'][1] + 1.2, 2)
s = lambda k: TL[k][0]; e = lambda k: TL[k][1]

# resonances of the 1 mm plate (center-driven), from plate.py
SEQ = [5, 10, 16, 21, 30, 38, 51, 63, 73, 93]   # mode ids in order shown
HOOK = 93
TWIST = 51

if __name__ == '__main__':
    for k, (a, b) in TL.items(): print(k, round(a, 2), round(b, 2))
    print('T', T)
