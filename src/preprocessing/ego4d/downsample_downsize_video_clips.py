# modified from EgoVLP https://github.com/showlab/EgoVLP/blob/main/utils/video_resize.py

import os
import time
import sys
import subprocess
from multiprocessing import Pool, Value

image_size = 224
fps = 5

original_clips = 'datasets/Ego4D/video_clips/clips'
output_dir = f'datasets/Ego4D/video_clips/clips_downsampled_{fps}fps_downsized_{image_size}x{image_size}'

os.makedirs(output_dir, exist_ok=True)

def videos_resize(videoinfos):
    global count

    videoidx, videoname = videoinfos

    if os.path.exists(os.path.join(output_dir, videoname)):
        print(f'{videoname} already exists.')
        return

    inname = original_clips + '/' + videoname
    outname = output_dir + '/' + videoname

    # cmd = "ffmpeg -y -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -c:a copy {}".format(inname, outname)
    cmd = f"ffmpeg -loglevel info -y -i {inname} -filter:v scale={image_size}:{image_size},fps={fps} -c:a copy {outname}"
    subprocess.call(cmd, shell=True)

    return


if __name__ == "__main__":
    
    file_list = []
    mp4_list = [item for item in os.listdir(original_clips) if item.endswith('.mp4')] # load mp4 files
    
    for idx, video in enumerate(mp4_list):
        file_list.append([idx, video])
    
    print(file_list)
    print(len(file_list))

    pool = Pool(8)
    pool.map(videos_resize, tuple(file_list))