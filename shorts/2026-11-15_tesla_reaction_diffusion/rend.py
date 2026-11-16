import sys, subprocess
import rd, lib as K
w, nw = int(sys.argv[1]), int(sys.argv[2])
NF = int(round(rd.T * rd.FPS)); a = NF * w // nw; b = NF * (w + 1) // nw
p = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1080x1920','-r','30','-i','-','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p',f'part{w}.mp4'],stdin=subprocess.PIPE)
for n in range(a, b): p.stdin.write(K.finish(rd.frame(n / rd.FPS)).tobytes())
p.stdin.close(); p.wait(); open(f'part{w}.done','w').write(str(b-a))
