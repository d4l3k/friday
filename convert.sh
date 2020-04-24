#!/bin/bash
for var in "$@"
do
  ffmpeg -i "$var" "$var%05d.png"
done
