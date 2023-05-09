DEVICES=0,1 # comma-separated list of GPU IDs
N_GPU=2 # number of GPUs to use for training
PORT=29501

# takes in a .yaml config file from configs/projects/train, e.g.,
CONFIG="configs/projects/train/acdybench/ssv2/KP-Perceiver-VTC-DVDM.yaml"
CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.run \
    --nproc_per_node=${N_GPU} \
    --master_port=${PORT} \
    train.py --cfg-path ${CONFIG}