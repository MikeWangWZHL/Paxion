# modified from EgoVLP https://github.com/showlab/EgoVLP/blob/main/utils/video_resize.py
# Downsamples, downsizes, and converts to mp4

import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from glob import glob

def resize_video(input_output_path, suppress_stdout=False, suppress_stderr=False):
    input_path, output_path = input_output_path

    if os.path.exists(output_path):
        print(f'{output_path} already exists.')
        return

    cmd = f"ffmpeg -loglevel info -y -i {input_path} -filter:v scale={image_size}:{image_size},fps={fps} -c:a copy {output_path}"

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

    image_size = 224
    fps = 5

    original_clips = 'datasets/NextQA/video_clips/NExTVideo'
    output_dir = f'datasets/NextQA/video_clips/NExTVideo_downsampled_{fps}fps_downsized_{image_size}x{image_size}'

    os.makedirs(output_dir, exist_ok=True)

    input_output_paths = []

    input_dirs = glob(os.path.join(original_clips, "*"))
    for d in input_dirs:
        input_paths = glob(os.path.join(d, "*.mp4"))
        input_dir_name = os.path.basename(d)
        for ip in input_paths:
            video_name = os.path.basename(ip)
            os.makedirs(os.path.join(output_dir, input_dir_name), exist_ok=True)
            op = os.path.join(output_dir, input_dir_name, video_name)
            input_output_paths.append((ip,op))
    
    # mp4_list = [item for item in os.listdir(original_clips) if item.endswith('.mp4')] # load original mp4 files
    # print('Total files to consider:', len(mp4_list))

    print('Total files to consider:', len(input_output_paths))


    resizer = partial(resize_video, suppress_stdout=suppress_stdout, suppress_stderr=suppress_stderr)
    for _ in tqdm(Pool(num_proc).imap_unordered(resizer, input_output_paths), total=len(input_output_paths)):
        pass