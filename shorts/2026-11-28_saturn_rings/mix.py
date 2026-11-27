# mix.py <music.mp3> <out.wav>: VO at L times + music ducked ~8 dB under voice + SFX, hp 30 Hz, 2-pass loudnorm -14 LUFS / -1.5 dBTP
import sys, json, subprocess, re
L = json.loads(subprocess.check_output(['python3','render.py','1','info']))
DUR, T = L['DUR'], L['L']
mus, out = sys.argv[1], sys.argv[2]
sfx = [('sfx_whoosh.mp3', T[3]-0.6, 0.55), ('sfx_ice.mp3', T[4], 0.45), ('sfx_bounce.mp3', T[7]+0.3, 0.35)]
ins = ['-i', mus] + sum([['-i', f'vo/L{i+1:02d}.wav'] for i in range(11)], []) + sum([['-i', s[0]] for s in sfx], [])
fc = []; vl = []
for i in range(11):
    d = int(T[i]*1000); fc.append(f'[{i+1}:a]adelay={d}|{d},aformat=channel_layouts=stereo[v{i}]'); vl.append(f'[v{i}]')
fc.append(''.join(vl) + 'amix=inputs=11:normalize=0,asplit=2[vo][sc]')
for k, (f, t, g) in enumerate(sfx):
    d = int(t*1000); fc.append(f'[{12+k}:a]adelay={d}|{d},volume={g},aformat=channel_layouts=stereo[s{k}]')
fc.append(f'[0:a]atrim=0:{DUR},afade=t=out:st={DUR-1.2}:d=1.2,volume=0.9,aformat=channel_layouts=stereo[m]')
fc.append('[m][sc]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=350:makeup=1[md]')
fc.append('[md][vo][s0][s1][s2]amix=inputs=5:normalize=0,highpass=f=30[mx]')
base = ['ffmpeg','-y','-v','error'] + ins + ['-filter_complex', ';'.join(fc)]
r = subprocess.run(base + ['-map','[mx]','-af','loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json','-f','null','-'], capture_output=True, text=True)
# second pass
r = subprocess.run(['ffmpeg','-y']+ins+['-filter_complex',';'.join(fc)+';[mx]loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json[o]','-map','[o]','-f','null','-'],capture_output=True,text=True)
j = json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr)[-1])
ln = f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true"
subprocess.run(base[:4]+ins+['-filter_complex',';'.join(fc)+f';[mx]{ln},aresample=48000[o]','-map','[o]','-t',str(DUR),out], check=True)
print('ok', out)
