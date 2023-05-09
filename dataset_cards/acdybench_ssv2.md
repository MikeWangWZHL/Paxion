# Action Dynamic Benchmark (AcDyBench) on SSv2

## Instruction for Downloading Videos
- Download the videos from [here](https://developer.qualcomm.com/software/ai-datasets/something-something)
- Put the downloaded `.webm` files into `datasets/ssv2/video_clips/clips`
- Run preprocessing script (at the root dir of this repo):
    ```
        python src/preprocessing/ssv2/downsample_downsize_video_clips.py
    ```
- The resulting preprocessed video clips are stored at `datasets/ssv2/video_clips/clips_downsampled_5fps_downsized_224x224`

## Annotation Details

### Action Antonym Task & Video Reversal Task & Object Shuffle Task
- train: 162,475
- validation: 23,807
- ann_path: `AcDyBench/ssv2/shuffled_object_and_action_antonyms`
- format:
    ```
        {
            "label": "Spinning cube that quickly stops spinning",
            "template": "Spinning something that quickly stops spinning",
            "placeholders": [
                "cube"
            ],
            "template_action_antonym_clip_text": "Spinning something that quickly starts spinning",
            "label_action_antonym_clip_text": "Spinning cube that quickly starts spinning",
            "id": 74225,
            "label_object_shuffled_clip_text": "spinning feeding lid that quickly stops spinning"
        }
    ```
