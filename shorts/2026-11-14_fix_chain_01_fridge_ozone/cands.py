import render as R, timeline as TLM
from lib import finish
from PIL import Image
import scenes as SC
a=R.tag(finish(SC.scene_hook(0.3,3)))
b=R.tag(finish(SC.scene_cycle(2.0,5)))
c=R.tag(finish(SC.scene_machine(8.6,13,1.0,toxic_t=7.0)))
sh=Image.new('RGB',(1080,640))
for i,im in enumerate((a,b,c)): sh.paste(im.resize((360,640)),(i*360,0))
sh.save('cands.png')
