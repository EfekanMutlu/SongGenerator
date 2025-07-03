#!/bin/bash
set -e

cd "$(dirname "$0")"

apt-get update && \
apt-get install -y \
python3-pip wget ffmpeg git zip unzip x264 libx264-dev

python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

python3.10 setup.py