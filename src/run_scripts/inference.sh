#!/bin/bash
export TOKENIZERS_PARALLELISM=false

### usage examples for inferencing different tasks ###
# CONFIG is taken from .yaml from configs/projects/eval

## == inference acdybench == ##
INFERENCE_TYPE="physical_knowledge_bench"
CONFIG="configs/projects/eval/acdybench/backbone/internvideo/ssv2/acdybench_ssv2_internvideo_backbone__action_antonym.yaml"

## == inference nextqa == ##
INFERENCE_TYPE="downstream_task_next_qa"
CONFIG="configs/projects/eval/downstream_task/nextqa/backbone_zero-shot.yaml"

## == inference ssv2-label == ##
INFERENCE_TYPE="downstream_task_retrieval_v2t_ssv2_label"
CONFIG="configs/projects/eval/downstream_task/ssv2_label/backbone_zero-shot.yaml"

## == inference ssv2-template == ##
INFERENCE_TYPE="downstream_task_retrieval_v2t_ssv2_template"
CONFIG="configs/projects/eval/downstream_task/ssv2_template/backbone_zero-shot.yaml"

## == inference temporal-ssv2 == ##
INFERENCE_TYPE="downstream_task_retrieval_v2t_temporal_ssv2"
CONFIG="configs/projects/eval/downstream_task/temporal_ssv2/backbone_zero-shot.yaml"



# run inference
CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.run \
    --nproc_per_node=${N_GPU} \
    --master_port=${PORT} \
    inference.py --cfg-path $CONFIG --inference_type $INFERENCE_TYPE