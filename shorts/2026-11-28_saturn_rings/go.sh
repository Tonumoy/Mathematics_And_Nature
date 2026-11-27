#!/bin/bash
# usage: go.sh SS crf out   (2 workers)
SS=$1; W=$(python3 -c "print(int(1080*$SS))"); H=$(python3 -c "print(int(1920*$SS))")
NF=$(python3 -c "import json,subprocess;d=json.loads(subprocess.check_output(['python3','render.py','1','info']));print(int(d['DUR']*30))")
M=$((NF/2))
python3 render.py $SS video 0 $M | ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s ${W}x${H} -r 30 -i - -c:v libx264 -preset veryfast -crf $2 -pix_fmt yuv420p partA.mp4 &
python3 render.py $SS video $M $NF | ffmpeg -y -v error -f rawvideo -pix_fmt rgb24 -s ${W}x${H} -r 30 -i - -c:v libx264 -preset veryfast -crf $2 -pix_fmt yuv420p partB.mp4 &
wait
printf "file partA.mp4\nfile partB.mp4\n" > parts.txt
ffmpeg -y -v error -f concat -safe 0 -i parts.txt -c copy $3
touch $3.done
