#!/bin/bash

set -ex


#poetry run python convert.py
python convert.py

scp *.tflite alarm@192.168.86.73:inference/
