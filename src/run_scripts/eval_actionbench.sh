DEVICES=0,1 # comma-separated list of GPU IDs
N_GPU=2 # number of GPUs to use for training
PORT=29501

# takes in a .yaml config file from configs/projects/eval/actionbench, e.g.,
CONFIG="configs/projects/eval/actionbench/backbone/internvideo/ssv2/actionbench_ssv2_internvideo_backbone__action_antonym.yaml"
CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.run \
    --nproc_per_node=${N_GPU} \
    --master_port=${PORT} \
    evaluate.py --cfg-path ${CONFIG}