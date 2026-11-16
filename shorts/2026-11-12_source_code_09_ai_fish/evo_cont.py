import os
os.environ['BURST'] = '0.8'; os.environ['VPRED'] = '1.4'
import evo as E, pickle, numpy as np, time
e = pickle.load(open('evo_s3.pkl', 'rb')); g0 = max(e['snaps']); s0 = e['log'][-1][0]
w = E.World(33, prey_w=e['W'], pred_w=e['WQ']); snaps = dict(e['snaps']); log = list(e['log']); gl = 0; dl = 0; t0 = time.time()
for i in range(60000):
    w.step()
    if i % 600 == 599: w.evolve_sharks()
    g = w.deaths//E.N
    if g > gl: snaps[g0 + g] = (w.W.copy(), w.WQ.copy()); gl = g
    if i % 2000 == 1999:
        nn, nb = w.stats(); rate = (w.deaths - dl)/2; dl = w.deaths; o = w.out.mean(0)
        log.append((s0 + i + 1, g0 + g, nn, nb, rate, *o))
        print('step %d gen %d nn %.2f nbrs %.1f catch/1k %.1f | flee %.2f friends %.2f match %.2f | %.0fs' % (i+1, g0+g, nn, nb, rate, *o, time.time()-t0), flush=True)
pickle.dump(dict(snaps=snaps, log=log, W=w.W, WQ=w.WQ), open('evo_s7.pkl', 'wb'))
