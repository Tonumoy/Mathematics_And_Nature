import render as R, timeline as TLM
from lib import finish
from PIL import Image
import sys
ts=[float(x) for x in sys.argv[1:]]
ims=[]
for tg in ts:
    for i,(n,a,b) in enumerate(TLM.SCENES):
        if a<=tg<b:
            im=R.tag(finish(R.frame_fn(n,a,b)(tg-a))); ims.append(im); break
sh=Image.new('RGB',(len(ims)*360,640))
for i,im in enumerate(ims): sh.paste(im.resize((360,640)),(i*360,0))
sh.save('check.png')
