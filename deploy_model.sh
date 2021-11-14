#!/bin/bash

set -ex


python convert.py

rsync -rav model.py infer.py friday_net_quant_jit.pt main.py camera.py alarm@192.168.86.73:inference/
