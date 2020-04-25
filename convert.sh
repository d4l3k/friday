#!/bin/bash
for var in "$@"
do
  rm $var*.png
  ffmpeg -i "$var" -r 4 "$var%05d.png"
done
