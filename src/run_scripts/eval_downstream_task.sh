DEVICES=0 # support one GPU only for downstream evaluation (fast)
N_GPU=1
PORT=29501

# takes in a .yaml config file from configs/projects/eval/downstream_task, e.g.,
CONFIG="configs/projects/eval/downstream_task/ssv2_template/backbone_zero-shot.yaml"
CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.run \
    --nproc_per_node=${N_GPU} \
    --master_port=${PORT} \
    evaluate.py --cfg-path ${CONFIG}