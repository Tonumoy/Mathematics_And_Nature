# 13 s Instagram story teaser for SOURCE CODE #15, built from the final 1080 Short
import subprocess
SRC = "SC15_riemann_1080.mp4"
# clips: hook (0.0-5.9) + build (19.4-23.0) + certified card (38.0-40.4)
segs = [(0.0, 5.9), (19.4, 23.0), (38.2, 40.6)]
parts = []
for i, (a, b) in enumerate(segs):
    p = f"st_{i}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(a), "-to", str(b), "-i", SRC, "-an",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", p], check=True)
    parts.append(p)
open("st_list.txt", "w").write("".join(f"file {p}\n" for p in parts))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", "st_list.txt", "-c", "copy", "st_v.mp4"], check=True)
# overlay: top "NEW REEL" pill and bottom CTA (kept out of story UI zones: top 250 px, bottom 340 px)
ov = ("drawbox=x=640:y=262:w=340:h=70:color=0xDC2428@0.95:t=fill,"
      "drawtext=fontfile=/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf:text='NEW REEL':fontcolor=white:fontsize=42:x=640+(340-text_w)/2:y=274,"
      "drawbox=x=60:y=1385:w=960:h=110:color=0x050a12@1.0:t=fill:enable='gte(t,9.5)',"
      "drawtext=fontfile=/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf:text='Watch how the primes appear  >':fontcolor=0x4FD1FF:fontsize=46:x=(w-text_w)/2:y=1410:enable='gte(t,9.5)'")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "st_v.mp4", "-ss", "16.6", "-i", "music_b.mp3",
                "-filter_complex", f"[0:v]{ov}[v];[1:a]afade=t=in:d=0.3,afade=t=out:st=11.6:d=1.5,volume=1.4[a]",
                "-map", "[v]", "-map", "[a]", "-shortest", "-c:v", "libx264", "-crf", "19", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k", "SC15_IG_story.mp4"], check=True)
