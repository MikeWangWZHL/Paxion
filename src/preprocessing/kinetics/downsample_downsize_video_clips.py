# modified from EgoVLP https://github.com/showlab/EgoVLP/blob/main/utils/video_resize.py
# Downsamples, downsizes, and converts to mp4

import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

image_size = 224
fps = 5

# original_clips = ''
# output_dir = f'<path to output root dir>/clips_downsampled_{fps}fps_downsized_{image_size}x{image_size}'

original_clips = 'datasets/Temporal/video_clips/kinetics400/clips'
output_dir = f'datasets/Temporal/video_clips/kinetics400/clips_downsampled_{fps}fps_downsized_{image_size}x{image_size}'

def resize_video(videoname, suppress_stdout=False, suppress_stderr=False):
    if os.path.exists(os.path.join(output_dir, videoname)):
        print(f'{videoname} already exists.')
        return

    inname = original_clips + '/' + videoname
    outname = output_dir + '/' + f'{videoname.split(".")[0]}.mp4'

    cmd = f"ffmpeg -loglevel info -y -i {inname} -filter:v scale={image_size}:{image_size},fps={fps} -c:a copy {outname}"

    kwargs = {}
    if suppress_stdout:
        kwargs['stdout'] = subprocess.DEVNULL
    if suppress_stderr:
        kwargs['stderr'] = subprocess.DEVNULL

    subprocess.run(cmd, shell=True, **kwargs)

    return

if __name__ == "__main__":
    suppress_stdout = True
    suppress_stderr = True
    num_proc = 10

    os.makedirs(output_dir, exist_ok=True)
    mp4_list = [item for item in os.listdir(original_clips) if item.endswith('.mp4')] # load original mp4 files
    print('Total files to consider:', len(mp4_list))

    resizer = partial(resize_video, suppress_stdout=suppress_stdout, suppress_stderr=suppress_stderr)
    for _ in tqdm(Pool(num_proc).imap_unordered(resizer, mp4_list), total=len(mp4_list)):
        pass