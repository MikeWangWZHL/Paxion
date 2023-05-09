## Instruction for Downloading Videos
- **SSv2 videos**: refer to [acdybench_ssv2.md](./acdybench_ssv2.md).
- **Kinetics400 videos**: download the subset of Kinetic400 that are required in Temporal-Kinetics
    - Install `yt-dlp` following the instructions [here](https://github.com/yt-dlp/yt-dlp.git)
    - Download the required videos using our provided script:
        ```
            cd datasets/Temporal/ann
            bash download_kinetic_videos_yt_dlp.sh
        ```
    - Put the downloaded videos into `datasets/Temporal/video_clips/kinetics400/clips`
    - Run preprocessing script (at the root dir of this repo):
        ```
            python src/preprocessing/kinetics/downsample_downsize_video_clips.py
        ```
    - The resulting preprocessed video clips are stored at `datasets/Temporal/video_clips/kinetics400/clips_downsampled_5fps_downsized_224x224`

## Annotation Details
- paper: https://arxiv.org/abs/2301.02074
- Temporal-kinetics size: 1309 | 32 action texts
- Temporal-ssv2 size: 864 | 18 action texts
- ann_path: `datasets/Temporal/ann/val-v1.0-2.4k.csv`
- format:
    ```
        ,index,video_id,text,dataset
        4153,2561,169724,Approaching [something] with your camera,SSv2
        ...
        188,2281,cartwheeling/RUNwB3-Qxqg_000007_000017,cartwheeling,kinetics
        ...
    ````

## Video directory
- Temporal-kinetics: `datasets/Temporal/video_clips/kinetics400/clips_downsampled_5fps_downsized_224x224`
- Temporal-ssv2: `datasets/ssv2/video_clips/clips_downsampled_5fps_downsized_224x224`