#!/bin/bash

set -ex

rsync -rav alarm@192.168.86.73:inference/data/ data/
