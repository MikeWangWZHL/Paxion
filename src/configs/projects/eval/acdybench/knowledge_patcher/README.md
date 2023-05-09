For constructing eval configs for knowledge patcher on AcdyBench, we show two examples in this folder on SSv2 action_antonym and Ego4d action_antonym, with KP-Perceiver based on InternVideo. 

- To evaluate on other method such as KP-Transformer: replace the "model" section with the corresponding "model" sections in `configs/train/acdybench/<dataset>/<method>.yaml`. And set the "model.pretrained" field to the corresponding trained checkpoint path.
- To evaluate on other tasks such as reversed_video: replace the "dataset" section with the corresponding "dataset" sections in `configs/eval/acdybench/backbone/*/*_<task>.yaml`.
- Set the "run.output_dir" according to the custom setting.
