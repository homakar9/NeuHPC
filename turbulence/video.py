#!/usr/bin/python3

import os
import re
import glob
import shutil

image_dir = 'output'
temp_dir = 'tmp'

frame_map = {
    1: 20,
    2: 20,
    3: 20,
    4: 20,
    5: 20,
    6: 4,
    7: 20,
}

paths = sorted(glob.glob(f"{image_dir}/*.png"))

os.makedirs(temp_dir, exist_ok = True)

IMAGE_NUMBER = 1
for old_path in paths:
    dirname = os.path.dirname(old_path)
    basename = os.path.basename(old_path)
    stem, ext = os.path.splitext(basename)
    primary_stem = stem.split('-')[0]
    key = int(primary_stem)
    num_frames = frame_map[key]
    for _ in range(num_frames):
        new_path = f"{temp_dir}/{IMAGE_NUMBER:05d}.png"
        shutil.copy(old_path, new_path)
        IMAGE_NUMBER += 1
        print(f"Copied {old_path} to {new_path}")


os.system(f"ffmpeg -f image2 -r 10 -i {temp_dir}/%05d.png -vcodec mpeg4 -y -qscale:v 1 turbulence.mp4")
shutil.rmtree(temp_dir)

