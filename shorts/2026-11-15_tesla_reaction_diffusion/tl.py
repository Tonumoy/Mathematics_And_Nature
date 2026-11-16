import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = {
 'L01': ("This is Nikola Tesla. Nobody drew him.", ('nobody',)),
 'L02': ("2 chemicals did. Same trick paints fish, shells, fingers", ('chemicals',)),
 'L03': ("Chemical 1, the SPARK, copies itself", ('spark',)),
 'L04': ("To copy, it eats FOOD. Food spreads faster", ('food', 'faster')),
 'L05': ("Every spark grows... and starves the space around it", ('starves',)),
 'L06': ("Grow here, starve there → spots from nothing", ('spots',)),
 'L07': ("Alan Turing wrote this down in 1952", ('1952',)),
 'L08': ("One dial: how fast sparks die → stripes", ('stripes',)),
 'L09': ("Nature uses it: stripes on a fish", ('fish',)),
 'L10': ("Ridges on the roof of a mouth", ('mouth',)),
 'L11': ("Where fingers form in an embryo", ('fingers',)),
 'L12': ("Engineers grew it into water filters", ('water', 'filters')),
 'L13': ("Dark → sparks grow thick. Light → tiny dots", ('thick', 'dots')),
 'L14': ("No pencil. 1 food, 1 spark, 2 rules.", ('2', 'rules')),
 'L15': ("Nature is running on code.", ('code',)),
}
GAP = {'L01': 0.35, 'L02': 0.4, 'L05': 0.3, 'L06': 0.5, 'L07': 0.4, 'L08': 0.6, 'L11': 0.4, 'L12': 0.6, 'L13': 1.2, 'L14': 3.2}
def dur(k):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', os.path.join(HERE, 'vo', k + '.mp3')]))
TL = {}; t = 0.25
for k in CAP:
    d = dur(k); TL[k] = (t, t + d); t += d + GAP.get(k, 0.25)
T = TL['L15'][1] + 1.2
s = lambda k: TL[k][0]; e = lambda k: TL[k][1]
if __name__ == '__main__':
    for k, (a, b) in TL.items(): print(k, round(a, 2), round(b, 2))
    print('T', round(T, 2))
