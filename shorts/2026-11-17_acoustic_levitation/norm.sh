#!/bin/bash
# two-pass loudnorm -14 LUFS, TP -1.5 -> $1.m4a
J=$(ffmpeg -hide_banner -i mix_$1.wav -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g(){ echo "$J" | grep "\"$1\"" | sed 's/.*: "\(.*\)".*/\1/'; }
ffmpeg -loglevel error -y -i mix_$1.wav -af "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aresample=48000" -c:a aac -b:a 256k mix_$1.m4a
ffmpeg -hide_banner -i mix_$1.m4a -af ebur128=peak=true -f null - 2>&1 | grep -E "I:|Peak:" | tail -2
