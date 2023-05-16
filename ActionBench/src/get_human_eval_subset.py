import json
import sys
import os
sys.path.append(os.path.realpath(os.path.join(__file__, '../../../src')))
from processors import *
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip as VideoFileClip_io
from moviepy.editor import VideoFileClip as VideoFileClip_editor
from moviepy.video.fx.all import time_mirror
import random
from tqdm import tqdm
import shutil
# set random seed
random.seed(42)
SUBSET_SIZE = 50

# ego4d
def get_actionbench_ego4d_human_eval_subset():
    eval_ann_path = "../ego4d/egoclip_subset_action_antonyms_object_shuffled_train_val_test_split/test.jsonl"
    vis_root = "/shared/nas/data/m1/wangz3/Shared_Datasets/VL/Ego4D/Ego4d_data/v1/clips_downsampled_5fps_downsized_224x224"
    output_human_eval_dir = "../ego4d/human_eval"

    # load text processor
    txt_processor = VLDynamicModelBlipPretrainEgo4dTextProcessor()
    
    # load jsonl
    with open(eval_ann_path, "r") as f:
        eval_ann = [json.loads(line) for line in f]
    # shuffle eval_ann
    random.shuffle(eval_ann)

    print("doing action antonym and object shuffle task...")
    # Action Antonym and Object Shuffle task
    ann_subset_AA_and_OS = []
    AA_and_OS_gt = {}
    for i, ann_instance in enumerate(eval_ann):
        clip_uid = ann_instance['clip_uid']
        if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
            clip_relative_start = ann_instance['clip_relative_start']
            clip_relative_end = ann_instance['clip_relative_end']
            
            sample_id = len(ann_subset_AA_and_OS)
            output_subdir = os.path.join(output_human_eval_dir, "action_antonym_and_object_shuffle", f'{sample_id}')
            os.makedirs(output_subdir, exist_ok=True)

            original_text = txt_processor(ann_instance['clip_text'])
            action_antonym_text = txt_processor(ann_instance['action_antonym_clip_text'])
            object_shuffle_text = txt_processor(ann_instance['object_shuffled_clip_text'])
            
            # store video
            input_video_path = os.path.join(vis_root, f'{clip_uid}.mp4')
            output_video_path = os.path.join(output_subdir, f'original_video.mp4')
            get_video_subclip(input_video_path, clip_relative_start, clip_relative_end, output_video_path)
            
            # create text candidates and gt
            aa_candidates = [original_text, action_antonym_text]
            random.shuffle(aa_candidates)
            os_candidates = [original_text, object_shuffle_text]
            random.shuffle(os_candidates)
            aa_gt = 0 if aa_candidates[0] == original_text else 1
            os_gt = 0 if os_candidates[0] == original_text else 1
            task_instance = {
                "action_antonym": {
                    "candidates": {
                        '0': aa_candidates[0],
                        '1': aa_candidates[1]
                    },
                    "answer": None,
                },
                "object_shuffle": {
                    "candidates": {
                        '0': os_candidates[0],
                        '1': os_candidates[1]
                    },
                    "answer": None,
                },
            }
            AA_and_OS_gt[sample_id] = {
                "action_antonym": aa_gt,
                "object_shuffle": os_gt,
            }
            with open(os.path.join(output_subdir, "task_instance.json"), "w") as f:
                json.dump(task_instance, f, indent=4)

            ann_subset_AA_and_OS.append(ann_instance)
        
        if len(ann_subset_AA_and_OS) == SUBSET_SIZE:
            break

    # store AA_and_OS_gt
    with open(os.path.join(output_human_eval_dir, "AA_and_OS_gt.json"), "w") as f:
        json.dump(AA_and_OS_gt, f, indent=4)
    # store ann_subset_AA_and_OS
    with open(os.path.join(output_human_eval_dir, "ann_subset_AA_and_OS.json"), "w") as f:
        json.dump(ann_subset_AA_and_OS, f, indent=4)



    print("doing video reversal task...")
    # Video Reversal task
    ann_subset_VR = []
    VR_gt = {}
    eval_ann = eval_ann[i+1:] # take samples not overlap with AA_and_OS
    for ann_instance in eval_ann:
        clip_uid = ann_instance['clip_uid']
        if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
            clip_relative_start = ann_instance['clip_relative_start']
            clip_relative_end = ann_instance['clip_relative_end']
            if abs(clip_relative_end - clip_relative_start) < 1:
                # avoid FFMPEG error on very short clips (we use videos with 5fps)
                continue

            sample_id = len(ann_subset_VR)
            output_subdir = os.path.join(output_human_eval_dir, "video_reversal", f'{sample_id}')
            os.makedirs(output_subdir, exist_ok=True)

            original_text = txt_processor(ann_instance['clip_text'])

            # get videos
            assign_video_name = [0,1]
            random.shuffle(assign_video_name)
            
            try:
                # original video
                input_video_path = os.path.join(vis_root, f'{clip_uid}.mp4')
                output_video_path = os.path.join(output_subdir, f'{assign_video_name[0]}.mp4')
                get_video_subclip(input_video_path, clip_relative_start, clip_relative_end, output_video_path)

                # reversed video
                original_video_path = output_video_path
                reversed_video_output_path = os.path.join(output_subdir, f'{assign_video_name[1]}.mp4')
                reverse_video(original_video_path, reversed_video_output_path)
            except:
                continue

            # create exam json
            task_instance = {
                "original_text": original_text,
                "original_video_idx": None
            }
            with open(os.path.join(output_subdir, "task_instance.json"), "w") as f:
                json.dump(task_instance, f, indent=4)
            ann_subset_VR.append(ann_instance)
            VR_gt[sample_id] = assign_video_name[0]
            
        
        if len(ann_subset_VR) == SUBSET_SIZE:
            break
    # store VR_gt
    with open(os.path.join(output_human_eval_dir, "VR_gt.json"), "w") as f:
        json.dump(VR_gt, f, indent=4)
    # store ann_subset_VR
    with open(os.path.join(output_human_eval_dir, "ann_subset_VR.json"), "w") as f:
        json.dump(ann_subset_VR, f, indent=4)


def get_actionbench_ssv2_human_eval_subset():
    eval_ann_path = "../ssv2/shuffled_object_and_action_antonyms/validation.json"
    vis_root = "/shared/nas/data/m1/blume5/sp23/ssv2/videos/clips_downsampled_5fps_downsized_224x224"
    output_human_eval_dir = "../ssv2/human_eval"
    
    # load json
    with open(eval_ann_path, "r") as f:
        eval_ann = json.load(f)
    print(eval_ann[0])
    
    # text processor
    txt_processor = MinimumTextProcessor()

    # shuffle eval_ann
    random.shuffle(eval_ann)

    print("doing action antonym and object shuffle task...")
    # Action Antonym and Object Shuffle task
    ann_subset_AA_and_OS = []
    AA_and_OS_gt = {}
    for i, ann_instance in enumerate(eval_ann):
        clip_uid = ann_instance['id']
        if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
            input_video_path = os.path.join(vis_root, f'{clip_uid}.mp4')
            sample_id = len(ann_subset_AA_and_OS)
            output_subdir = os.path.join(output_human_eval_dir, "action_antonym_and_object_shuffle", f'{sample_id}')
            os.makedirs(output_subdir, exist_ok=True)
            
            original_text = txt_processor(ann_instance['label'])
            action_antonym_text = txt_processor(ann_instance['label_action_antonym_clip_text'])
            object_shuffle_text = txt_processor(ann_instance['label_object_shuffled_clip_text'])
            
            # copy video
            output_video_path = os.path.join(output_subdir, f'original_video.mp4')
            shutil.copy(input_video_path, output_video_path)

            # create text candidates and gt
            aa_candidates = [original_text, action_antonym_text]
            random.shuffle(aa_candidates)
            os_candidates = [original_text, object_shuffle_text]
            random.shuffle(os_candidates)
            aa_gt = 0 if aa_candidates[0] == original_text else 1
            os_gt = 0 if os_candidates[0] == original_text else 1
            task_instance = {
                "action_antonym": {
                    "candidates": {
                        '0': aa_candidates[0],
                        '1': aa_candidates[1]
                    },
                    "answer": None,
                },
                "object_shuffle": {
                    "candidates": {
                        '0': os_candidates[0],
                        '1': os_candidates[1]
                    },
                    "answer": None,
                },
            }
            AA_and_OS_gt[sample_id] = {
                "action_antonym": aa_gt,
                "object_shuffle": os_gt,
            }
            with open(os.path.join(output_subdir, "task_instance.json"), "w") as f:
                json.dump(task_instance, f, indent=4)

            ann_subset_AA_and_OS.append(ann_instance)

        if len(ann_subset_AA_and_OS) == SUBSET_SIZE:
            break
    
    # store AA_and_OS_gt
    with open(os.path.join(output_human_eval_dir, "AA_and_OS_gt.json"), "w") as f:
        json.dump(AA_and_OS_gt, f, indent=4)
    # store ann_subset_AA_and_OS
    with open(os.path.join(output_human_eval_dir, "ann_subset_AA_and_OS.json"), "w") as f:
        json.dump(ann_subset_AA_and_OS, f, indent=4)



    print("doing video reversal task...")
    # Video Reversal task
    ann_subset_VR = []
    VR_gt = {}
    eval_ann = eval_ann[i+1:] # take samples not overlap with AA_and_OS
    for ann_instance in eval_ann:
        clip_uid = ann_instance['id']
        if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
            sample_id = len(ann_subset_VR)
            output_subdir = os.path.join(output_human_eval_dir, "video_reversal", f'{sample_id}')
            os.makedirs(output_subdir, exist_ok=True)

            sample_id = len(ann_subset_VR)
            output_subdir = os.path.join(output_human_eval_dir, "video_reversal", f'{sample_id}')
            os.makedirs(output_subdir, exist_ok=True)

            original_text = txt_processor(ann_instance['label'])

            # get videos
            assign_video_name = [0,1]
            random.shuffle(assign_video_name)

            try:
                # original video
                input_video_path = os.path.join(vis_root, f'{clip_uid}.mp4')
                output_video_path = os.path.join(output_subdir, f'{assign_video_name[0]}.mp4')
                video = VideoFileClip_editor(input_video_path)
                video.write_videofile(output_video_path, codec='libx264', audio=False) # simply load and write the original video

                # reversed video
                original_video_path = output_video_path
                reversed_video_output_path = os.path.join(output_subdir, f'{assign_video_name[1]}.mp4')
                reverse_video(original_video_path, reversed_video_output_path)
            except:
                continue

            # create exam json
            task_instance = {
                "original_text": original_text,
                "original_video_idx": None
            }
            with open(os.path.join(output_subdir, "task_instance.json"), "w") as f:
                json.dump(task_instance, f, indent=4)
            ann_subset_VR.append(ann_instance)
            VR_gt[sample_id] = assign_video_name[0]
        
        if len(ann_subset_VR) == SUBSET_SIZE:
            break

    # store VR_gt
    with open(os.path.join(output_human_eval_dir, "VR_gt.json"), "w") as f:
        json.dump(VR_gt, f, indent=4)
    # store ann_subset_VR
    with open(os.path.join(output_human_eval_dir, "ann_subset_VR.json"), "w") as f:
        json.dump(ann_subset_VR, f, indent=4)

def reverse_video(input_video_path, output_video_path):
    # Load the video
    clip = VideoFileClip_editor(input_video_path)

    # Reverse the video
    reversed_clip = time_mirror(clip)

    # Write the result to a file
    reversed_clip.write_videofile(output_video_path, codec='libx264', audio=False)


def get_video_subclip(input_video_path, start_time, end_time, output_video_path):
    with VideoFileClip_io(input_video_path) as video:
        new = video.subclip(start_time, end_time)
        new.write_videofile(output_video_path, codec='libx264',  audio=False)

if __name__ == "__main__":

    ## Ego4D
    # get_actionbench_ego4d_human_eval_subset()

    ## SSV2
    get_actionbench_ssv2_human_eval_subset()


