# Action Dynamic Benchmark (ActionBench) on Ego4d

## Instruction for Downloading Videos
- Set up Ego4d CLI following [here](https://ego4d-data.org/docs/start-here/)
- Download the Moment clips using the following command:
    ```
        ego4d \
            --output_directory="Ego4d" \
            --datasets clips annotations \
            --benchmarks "EM" \
            --metadata
    ```
- Put the downloaded `clips/` folder into `datasets/Ego4D/video_clips/` as `datasets/Ego4D/video_clips/clips`
- Run preprocessing on the video clips (at the root dir of this repo):
    ```
        python src/preprocessing/ego4d/downsample_downsize_video_clips.py
    ```
- The processed video clips will be stored at `datasets/Ego4D/video_clips/clips_downsampled_5fps_downsized_224x224`


## Annotation Details

### Annotation for Action Antonym Task & Video Reversal Task
- train size: 274,946
- val size: 34,368
- test size: 34,369

- ann_path: `ActionBench/ego4d/egoclip_subset_action_antonyms_train_val_test_split/{split}.jsonl`.  The original annotation is based on a subset of [EgoClip](https://github.com/showlab/EgoVLP). 
- format:
    ```
    {
        'video_uid': '002d2729-df71-438d-8396-5895b349e8fd', 
        'video_dur': 3571.4333333333334, 
        'narration_source': 'narration_pass_1', 
        'narration_ind': 229, 
        'narration_time': 592.6903, 
        'clip_start': 592.3519665973915, 
        'clip_end': 593.0286286452686, 
        'clip_text': '#C C picks up the knife from the chopping board with her right hand.', 
        'action_antonym_clip_text': '#C C drops down the knife from the chopping board with her right hand.', 
        'tag_verb': '[17, 93]', 
        'tag_noun': '[321, 268, 573, 105]', 
        'Unnamed: 10': nan, 
        'clip_uid': '116ec16b-0d76-4e71-b02c-72cb37ebd5c5', 
        'narration_relative_time': 0.6902999999999793, 
        'clip_relative_start': 0.351966597391538, 
        'clip_relative_end': 1.0286286452685545, 
        'clip_fps': 30.0}
    ```

### Annotation for Object Shuffle 
A subset from above by filtering out clips with no object in the clip text.
- val size: 31974
- test size: 31925
- ann_path: `ActionBench/ego4d/egoclip_subset_action_antonyms_object_shuffled_train_val_test_split/{split}.jsonl`
- format: additional fields:
    ```
    {
        ...
        'object_shuffled_clip_text':'#C C picks up the banana from the chopping board with her right hand.',
    }
    ```

